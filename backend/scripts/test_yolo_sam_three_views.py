"""YOLO + SAM Box Prompt — 三图输出测试：原图 / YOLO检测图 / SAM分割图"""
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
print("YOLO + SAM — 三图输出测试（原图 | YOLO | SAM）")
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

print(f"Total: {len(tiles)}  |  Active (中心果园区): {len(non_blank)}")

# 快速扫描：用较低置信度在所有活跃瓦片中查找有果树的瓦片
print("\n快速扫描活跃瓦片（conf=0.15）查找果树...")
tiles_with_trees = []
for i, tile_info in enumerate(non_blank):
    if i % 50 == 0:
        print(f"  扫描进度: {i}/{len(non_blank)}")
    tile_rgb = tile_info["tile_data"]
    results = yolo(tile_rgb, verbose=False, conf=0.15)
    boxes = results[0].boxes
    if boxes is not None and len(boxes.xyxy) > 0:
        tiles_with_trees.append((i, tile_info, len(boxes.xyxy)))

print(f"\n找到 {len(tiles_with_trees)} 个有果树的瓦片 (conf=0.15)")
if len(tiles_with_trees) == 0:
    print("警告: 未找到任何果树! 尝试更低阈值...")
    for i, tile_info in enumerate(non_blank):
        if i % 100 == 0:
            print(f"  扫描进度: {i}/{len(non_blank)}")
        tile_rgb = tile_info["tile_data"]
        results = yolo(tile_rgb, verbose=False, conf=0.05)
        boxes = results[0].boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            tiles_with_trees.append((i, tile_info, len(boxes.xyxy)))
    print(f"conf=0.05: 找到 {len(tiles_with_trees)} 个瓦片")

# 取前10个有果树的瓦片进行可视化
if len(tiles_with_trees) == 0:
    print("\n未找到任何有检测结果的瓦片，使用原脚本的瓦片区域")
    # Fall back to original approach but sample from multiple areas
    viz_data = []
    step = max(1, len(non_blank) // 20)
    for i in range(0, len(non_blank), step):
        viz_data.append((i, non_blank[i]))
        if len(viz_data) >= 10:
            break
    viz_tiles_info = viz_data
else:
    viz_tiles_info = tiles_with_trees[:10]
    for idx, (orig_idx, _, n) in enumerate(viz_tiles_info):
        print(f"  瓦片 #{idx} (原始索引 {orig_idx}): {n} 个检测框")

print(f"\n开始详细测试 {len(viz_tiles_info)} 个瓦片\n")
total_yolo_time = 0
total_sam_enc_time = 0
total_sam_dec_time = 0
total_trees = 0

for idx, item in enumerate(viz_tiles_info):
    if len(item) == 3:
        orig_idx, tile_info, _ = item
    else:
        orig_idx, tile_info = item
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

    # ============================================================
    # 输出三张图: 原图 / YOLO图 / SAM图
    # ============================================================

    # --- 图1: 原图 (raw) ---
    raw_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    raw_path = os.path.join(OUT_DIR, f"tile_{idx:02d}_raw.png")
    cv2.imwrite(raw_path, raw_bgr)

    # --- 图2: YOLO检测图 (绿色框) ---
    yolo_view = tile_rgb.copy()
    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(yolo_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
    yolo_bgr = cv2.cvtColor(yolo_view, cv2.COLOR_RGB2BGR)
    yolo_path = os.path.join(OUT_DIR, f"tile_{idx:02d}_yolo.png")
    cv2.imwrite(yolo_path, yolo_bgr)

    # --- 图3: SAM分割图 (彩色mask叠加在原图上) ---
    sam_view = tile_rgb.copy()
    for m, box, score in masks_list:
        color = np.random.randint(100, 255, 3).tolist()
        sam_view[m > 0] = (sam_view[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(sam_view, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cy, cx = int(np.mean(np.where(m)[0])), int(np.mean(np.where(m)[1]))
        cv2.circle(sam_view, (cx, cy), 3, (255, 255, 255), -1)
    sam_bgr = cv2.cvtColor(sam_view, cv2.COLOR_RGB2BGR)
    sam_path = os.path.join(OUT_DIR, f"tile_{idx:02d}_sam.png")
    cv2.imwrite(sam_path, sam_bgr)

    print(f"  tile {idx:02d} | YOLO:{len(boxes_xyxy):>3} boxes ({yolo_time*1000:.0f}ms) "
          f"| SAM enc:{enc_time*1000:.0f}ms dec:{dec_time*1000:.0f}ms "
          f"| kept:{len(masks_list):>3} trees")
    print(f"         | [原图] {os.path.basename(raw_path)}")
    print(f"         | [YOLO] {os.path.basename(yolo_path)}")
    print(f"         | [SAM]  {os.path.basename(sam_path)}")

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

# Compare with auto-mode baseline
print("PERFORMANCE COMPARISON:")
print(f"  {'':<30} {'Time/tile':>10} {'Trees/10tiles':>15}")
print(f"  {'-'*55}")
print(f"  {'auto-mode pps=16':<30} {'2410ms':>10} {'~137':>15}")
print(f"  {'auto-mode pps=12':<30} {'1355ms':>10} {'~115':>15}")
print(f"  {'YOLO+Box (this test)':<30} "
      f"{(total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000:.0f}ms/tile"
      f"{'':>5} {total_trees}{'':>10}")
print()

speedup_vs_16 = 2410 / ((total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000)
speedup_vs_12 = 1355 / ((total_yolo_time+total_sam_enc_time+total_sam_dec_time)/10*1000)
print(f"  Speedup vs auto-pps16: {speedup_vs_16:.1f}x")
print(f"  Speedup vs auto-pps12: {speedup_vs_12:.1f}x")
print(f"\n  Output directory: {OUT_DIR}/")
print(f"  Files generated:")
print(f"    tile_XX_raw.png  — 原图")
print(f"    tile_XX_yolo.png — YOLO检测框")
print(f"    tile_XX_sam.png  — SAM分割掩码")
