"""YOLO + SAM Box Prompt — speed test & visualization on real TIF tiles."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import rasterio
import torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

FILE_PATH = r"C:\Users\asus\Desktop\Esri_QC\water-digital-twin-platform\backend\data\uploads\6e5ec825907b454f86271ace07c1f411_2019062105_mosaic_group1.tif"
OUT_DIR = "debug_tiles"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("YOLO + SAM Box Prompt — Pipeline Test & Visualization")
print("=" * 70)

# --- Load models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading YOLO...", end=" ", flush=True)
t0 = time.time()
yolo = YOLO("weights/best.pt")
print(f"{time.time()-t0:.1f}s")

print("Loading SAM...", end=" ", flush=True)
t0 = time.time()
sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)
print(f"{time.time()-t0:.1f}s")

# --- Get tiles from TIF ---
from app.services.tif_service import TifService

with rasterio.open(FILE_PATH) as src:
    print(f"TIF: {src.width}x{src.height}")

tiles = list(TifService.slice_tif_generator(FILE_PATH))
non_blank = [t for t in tiles if t["tile_data"].max() >= 10 and t["tile_data"].std() >= 5]

print(f"Total: {len(tiles)}  |  Active: {len(non_blank)}")
print(f"Testing on first 10 active tiles for visualization\n")

# --- Per-tile: YOLO → Box Prompt SAM ---
viz_tiles = non_blank[:10]
tile_data = []
total_yolo_time = 0
total_sam_enc_time = 0
total_sam_dec_time = 0
total_trees = 0

for idx, tile_info in enumerate(viz_tiles):
    tile_rgb = tile_info["tile_data"].copy()
    valid_mask = tile_info["valid_mask"]
    h, w = tile_rgb.shape[:2]

    # --- YOLO ---
    t0 = time.time()
    results = yolo(tile_rgb, verbose=False, conf=0.35)
    boxes = results[0].boxes
    if boxes is None:
        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
    else:
        boxes_xyxy = boxes.xyxy.cpu().numpy()
    yolo_time = time.time() - t0

    # --- SAM encoder ---
    t0 = time.time()
    predictor.set_image(tile_rgb)
    enc_time = time.time() - t0

    # --- SAM decoder per box ---
    dec_time = 0
    masks_list = []
    for box in boxes_xyxy:
        t0 = time.time()
        mask, score, _ = predictor.predict(
            box=box[np.newaxis, :], multimask_output=False)
        dec_time += time.time() - t0

        m = mask[0]
        if np.sum(np.logical_and(m, valid_mask)) / max(np.sum(m), 1) >= 0.8:
            masks_list.append((m, box, float(score[0])))

    total_yolo_time += yolo_time
    total_sam_enc_time += enc_time
    total_sam_dec_time += dec_time
    total_trees += len(masks_list)

    # --- Build visualization ---
    overlay = tile_rgb.copy()
    # Draw YOLO boxes (green)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw SAM masks (random color per tree)
    for m, box, score in masks_list:
        color = np.random.randint(100, 255, 3).tolist()
        overlay[m > 0] = (overlay[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cy, cx = int(np.mean(np.where(m)[0])), int(np.mean(np.where(m)[1]))
        cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), -1)

    out_path = os.path.join(OUT_DIR, f"tile_{idx:02d}_yolo{len(boxes_xyxy)}_sam{len(masks_list)}.png")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"  tile {idx:02d} | YOLO:{len(boxes_xyxy):>3} boxes ({yolo_time*1000:.0f}ms) "
          f"| SAM enc:{enc_time*1000:.0f}ms dec:{dec_time*1000:.0f}ms "
          f"| kept:{len(masks_list):>3} trees | saved {os.path.basename(out_path)}")

    tile_data.append({
        "yolo_boxes": len(boxes_xyxy),
        "sam_trees": len(masks_list),
        "yolo_ms": yolo_time * 1000,
        "enc_ms": enc_time * 1000,
        "dec_ms": dec_time * 1000,
    })

# --- Summary ---
print()
print("=" * 70)
print("SUMMARY (10 tiles)")
print("=" * 70)
print(f"  YOLO avg:    {total_yolo_time/10*1000:.0f}ms/tile")
print(f"  SAM encoder avg: {total_sam_enc_time/10*1000:.0f}ms/tile")
print(f"  SAM decoder avg: {total_sam_dec_time/10*1000:.0f}ms/tile ({total_trees} trees)")
print(f"  Total/tile:  {(total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000:.0f}ms")
print(f"  Trees found: {total_trees}")
print()

# Compare with pps=16 auto-mode baseline
print("COMPARISON (from earlier baseline test, same 100-tile sample):")
print(f"  {'':<25} {'Time/tile':>10} {'Trees/10tiles':>15}")
print(f"  {'-'*50}")
print(f"  {'auto-mode pps=16':<25} {'2410ms':>10} {'~137':>15}")
print(f"  {'auto-mode pps=12':<25} {'1355ms':>10} {'~115':>15}")
print(f"  {'YOLO+Box (this test)':<25} "
      f"{(total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000:.0f}ms/tile"
      f"{'':>5} {total_trees}{'':>10}")
print()

speedup_vs_16 = 2410 / ((total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000)
speedup_vs_12 = 1355 / ((total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000)
print(f"  Speedup vs auto-pps16: {speedup_vs_16:.1f}x")
print(f"  Speedup vs auto-pps12: {speedup_vs_12:.1f}x")
print(f"\n  Visual outputs saved to: {OUT_DIR}/tile_*.png")
