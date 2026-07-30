"""YOLO + SAM - Center Orchard Local Prompt Segmentation Test"""
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

# ============================================================
# Config
# ============================================================
FILE_PATH = "C:/Users/asus/Desktop/Esri_QC/water-digital-twin-platform/backend/data/uploads/6e5ec825907b454f86271ace07c1f411_2019062105_mosaic_group1.tif"
OUT_DIR = "debug_tiles"
YOLO_CONF = 0.35
SCAN_CONF = 0.05               # Very low confidence for quick orchard scanning (2048 tiles)
TILE_SIZE = 2048
TILE_OVERLAP = 256
MAX_VIZ_TILES = 20             # Max tiles to generate visualization for

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("YOLO + SAM - Center Orchard Local Prompt Segmentation Test")
print("=" * 70)

# ============================================================
# 1. Load models
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading YOLO...", end=" ", flush=True)
t_model_start = time.time()
yolo = YOLO("weights/best.pt")
t_yolo_load = time.time() - t_model_start
print(f"{t_yolo_load:.1f}s")

print("Loading MobileSAM...", end=" ", flush=True)
t0 = time.time()
sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)
t_sam_load = time.time() - t0
print(f"{t_sam_load:.1f}s")

# ============================================================
# 2. Read TIF and generate tiles
# ============================================================
print()
print("Reading TIF and generating tiles...")
from app.services.tif_service import TifService

with rasterio.open(FILE_PATH) as src:
    print(f"  TIF size: {src.width} x {src.height}  |  CRS: {src.crs}")

tiles = list(TifService.slice_tif_generator(FILE_PATH, window_size=TILE_SIZE, overlap=TILE_OVERLAP))
total_tiles = len(tiles)
print(f"  Total tiles: {total_tiles}")

# ============================================================
# 3. Phase 1: Quick scan to find orchard region
# ============================================================
print()
print(f"Phase 1: Quick scan to locate orchard region (conf={SCAN_CONF})...")
t_scan_start = time.time()

# Only scan non-blank tiles
non_blank = []
for i, t in enumerate(tiles):
    if t["tile_data"].max() >= 10 and t["tile_data"].std() >= 5:
        non_blank.append((i, t))

print(f"  Non-blank tiles: {len(non_blank)}/{total_tiles}")

tiles_with_trees = []
for i, (orig_idx, tile_info) in enumerate(non_blank):
    if i % 100 == 0:
        print(f"  Scanning: {i}/{len(non_blank)}...")
    tile_rgb = tile_info["tile_data"]
    results = yolo(tile_rgb, verbose=False, conf=SCAN_CONF)
    boxes = results[0].boxes
    if boxes is not None and len(boxes.xyxy) > 0:
        n_boxes = len(boxes.xyxy)
        tiles_with_trees.append((orig_idx, tile_info, n_boxes))

t_scan = time.time() - t_scan_start
print(f"  Scan complete in {t_scan:.1f}s")
print(f"  Tiles with trees: {len(tiles_with_trees)}")

if len(tiles_with_trees) == 0:
    print("\nWARNING: No trees found even at low confidence!")
    print("Falling back to processing first 20 non-blank tiles from center region...")
    mid = len(non_blank) // 2
    viz_items = [(orig_idx, tile_info, 0) for orig_idx, tile_info in non_blank[mid:mid+MAX_VIZ_TILES]]
else:
    # Sort by number of detections (descending) — the "orchard center" = highest tree density
    tiles_with_trees.sort(key=lambda x: -x[2])
    print(f"\n  Top 10 orchard-dense tiles (by YOLO box count):")
    for rank, (orig_idx, _, n) in enumerate(tiles_with_trees[:10]):
        print(f"    #{rank+1}: tile {orig_idx} — {n} boxes")

    # Take the top tiles as the "center orchard region"
    viz_items = tiles_with_trees[:min(len(tiles_with_trees), MAX_VIZ_TILES * 3)]

print(f"\nSelected {len(viz_items)} tiles for detailed test.\n")

# ============================================================
# 4. Phase 2: YOLO + SAM detailed test on orchard-center tiles
# ============================================================
print(f"Phase 2: YOLO + SAM local prompt segmentation (conf={YOLO_CONF})...")
print("-" * 70)

total_yolo_ms = 0.0
total_sam_enc_ms = 0.0
total_sam_dec_ms = 0.0
total_yolo_boxes = 0
total_sam_trees = 0
tile_results = []

for idx, (orig_idx, tile_info, scan_boxes) in enumerate(viz_items):
    tile_rgb = tile_info["tile_data"].copy()
    valid_mask = tile_info["valid_mask"]

    # ===== YOLO Detection =====
    t0 = time.time()
    results = yolo(tile_rgb, verbose=False, conf=YOLO_CONF)
    boxes = results[0].boxes
    if boxes is None:
        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
    else:
        boxes_xyxy = boxes.xyxy.cpu().numpy()
    yolo_ms = (time.time() - t0) * 1000

    # ===== SAM Encoder =====
    t0 = time.time()
    predictor.set_image(tile_rgb)
    enc_ms = (time.time() - t0) * 1000

    # ===== SAM Decoder (per-box local refinement) =====
    dec_ms = 0.0
    masks_list = []
    for box in boxes_xyxy:
        t0 = time.time()
        mask, score, _ = predictor.predict(
            box=box[np.newaxis, :], multimask_output=False)
        dec_ms += (time.time() - t0) * 1000

        m = mask[0]
        intersection = np.logical_and(m, valid_mask)
        if np.sum(intersection) / max(np.sum(m), 1) >= 0.8:
            masks_list.append((m, box, float(score[0])))

    total_yolo_ms += yolo_ms
    total_sam_enc_ms += enc_ms
    total_sam_dec_ms += dec_ms
    total_yolo_boxes += len(boxes_xyxy)
    total_sam_trees += len(masks_list)

    tile_results.append({
        "idx": idx,
        "orig_idx": orig_idx,
        "yolo_boxes": len(boxes_xyxy),
        "sam_trees": len(masks_list),
        "yolo_ms": yolo_ms,
        "enc_ms": enc_ms,
        "dec_ms": dec_ms,
        "total_ms": yolo_ms + enc_ms + dec_ms,
    })

    # ===== Visualization (first MAX_VIZ_TILES tiles only) =====
    if idx < MAX_VIZ_TILES:
        # Image 1: Raw
        raw_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"center_tile_{idx:02d}_raw.png"), raw_bgr)

        # Image 2: YOLO boxes (green)
        yolo_view = tile_rgb.copy()
        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(yolo_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"center_tile_{idx:02d}_yolo.png"),
                    cv2.cvtColor(yolo_view, cv2.COLOR_RGB2BGR))

        # Image 3: SAM segmentation (colored masks + blue boxes + white centroids)
        sam_view = tile_rgb.copy()
        for m, box, score in masks_list:
            color = np.random.randint(100, 255, 3).tolist()
            sam_view[m > 0] = (sam_view[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(sam_view, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cy, cx = int(np.mean(np.where(m)[0])), int(np.mean(np.where(m)[1]))
            cv2.circle(sam_view, (cx, cy), 3, (255, 255, 255), -1)
        cv2.imwrite(os.path.join(OUT_DIR, f"center_tile_{idx:02d}_sam.png"),
                    cv2.cvtColor(sam_view, cv2.COLOR_RGB2BGR))

        # Image 4: Side-by-side comparison (left: YOLO, right: SAM)
        h, w = tile_rgb.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = yolo_view
        comparison[:, w:] = sam_view
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)
        cv2.imwrite(os.path.join(OUT_DIR, f"center_tile_{idx:02d}_comparison.png"),
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    # Progress (print every 5 tiles)
    if idx % 5 == 0 or idx == len(viz_items) - 1:
        print(f"  [{idx+1}/{len(viz_items)}] tile#{orig_idx} "
              f"YOLO:{len(boxes_xyxy):>3}box ({yolo_ms:>6.0f}ms) | "
              f"SAM enc:{enc_ms:>6.0f}ms dec:{dec_ms:>6.0f}ms | "
              f"kept:{len(masks_list):>3}trees")

# ============================================================
# 5. Summary report
# ============================================================
n = len(viz_items)
print()
print("=" * 70)
print("TEST COMPLETE - Timing Summary")
print("=" * 70)
print(f"  Tiles processed:          {n}")
print(f"  Model loading:")
print(f"    YOLO load time:         {t_yolo_load*1000:.0f}ms")
print(f"    SAM load time:          {t_sam_load*1000:.0f}ms")
print()
print(f"  Phase 1 - Orchard scan:")
print(f"    Scan time:              {t_scan*1000:.0f}ms ({t_scan:.1f}s)")
print(f"    Scanned tiles:          {len(non_blank)} (non-blank)")
print()
print(f"  YOLO Detection Time (Phase 2):")
print(f"    Total:                  {total_yolo_ms:.0f}ms")
print(f"    Avg per tile:           {total_yolo_ms/n:.1f}ms")
print(f"    Total boxes detected:   {total_yolo_boxes}")
print(f"    Avg boxes per tile:     {total_yolo_boxes/n:.1f}")
print()
print(f"  SAM Local Refinement Time (Phase 2):")
print(f"    Encoder total:          {total_sam_enc_ms:.0f}ms")
print(f"    Encoder avg/tile:       {total_sam_enc_ms/n:.1f}ms")
print(f"    Decoder total:          {total_sam_dec_ms:.0f}ms")
print(f"    Decoder avg/tile:       {total_sam_dec_ms/n:.1f}ms")
print(f"    Decoder avg/box:        {total_sam_dec_ms/max(total_yolo_boxes,1):.1f}ms")
print(f"    SAM total:              {total_sam_enc_ms + total_sam_dec_ms:.0f}ms")
print()
total_pipeline_ms = total_yolo_ms + total_sam_enc_ms + total_sam_dec_ms
print(f"  Total Pipeline Time (excl. model loading):")
print(f"    Phase 1 (scan):         {t_scan*1000:.0f}ms")
print(f"    Phase 2 (YOLO+SAM):     {total_pipeline_ms:.0f}ms")
print(f"    Phase 1+2 combined:     {t_scan*1000 + total_pipeline_ms:.0f}ms")
print(f"    Avg/tile (Phase 2):     {total_pipeline_ms/n:.1f}ms")
print()
print(f"  Final tree count:         {total_sam_trees}")
if total_yolo_boxes > 0:
    print(f"  Filter rate:              {(1 - total_sam_trees/max(total_yolo_boxes,1))*100:.1f}%")
print()
print(f"  Visualization images saved to: {OUT_DIR}/center_tile_*")
print(f"    4 images per tile: _raw.png | _yolo.png | _sam.png | _comparison.png")
print("=" * 70)

# ============================================================
# 6. Tile timing distribution
# ============================================================
tile_results.sort(key=lambda r: r["total_ms"])
fastest = tile_results[0]
slowest = tile_results[-1]
median = tile_results[len(tile_results) // 2]

print()
print("Tile Timing Distribution (Phase 2):")
print(f"  Fastest: tile#{fastest['orig_idx']} - {fastest['total_ms']:.0f}ms "
      f"(YOLO:{fastest['yolo_ms']:.0f} SAM enc:{fastest['enc_ms']:.0f} dec:{fastest['dec_ms']:.0f})")
print(f"  Median:  tile#{median['orig_idx']} - {median['total_ms']:.0f}ms "
      f"(YOLO:{median['yolo_ms']:.0f} SAM enc:{median['enc_ms']:.0f} dec:{median['dec_ms']:.0f})")
print(f"  Slowest: tile#{slowest['orig_idx']} - {slowest['total_ms']:.0f}ms "
      f"(YOLO:{slowest['yolo_ms']:.0f} SAM enc:{slowest['enc_ms']:.0f} dec:{slowest['dec_ms']:.0f})")

print()
print("Done.")
