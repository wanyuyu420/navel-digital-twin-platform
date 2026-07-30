"""Multi-config comparison: find best speed/accuracy tradeoff for MobileSAM."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import rasterio
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

FILE_PATH = r"C:\Users\asus\Desktop\Esri_QC\water-digital-twin-platform\backend\data\uploads\6e5ec825907b454f86271ace07c1f411_2019062105_mosaic_group1.tif"
SAMPLE_TILES = 100

print("=" * 70)
print("MobileSAM Config Sweep — Find Best Speed/Accuracy Tradeoff")
print("=" * 70)

from app.services.tif_service import TifService

with rasterio.open(FILE_PATH) as src:
    print(f"TIF: {src.width}x{src.height}  ({src.width*src.height/1e6:.1f}MP)")
    print(f"CRS: {src.crs}")

tiles = list(TifService.slice_tif_generator(FILE_PATH))
non_blank = [t for t in tiles if t["tile_data"].max() >= 10]
test_tiles = non_blank[:SAMPLE_TILES]
print(f"Test tiles: {len(test_tiles)} (from {len(tiles)} total, {len(non_blank)} active)")
print()

def test_config(points_per_side, label=""):
    """Load model with given pps, run on test_tiles, return (time, tree_count)."""
    checkpoint = os.path.join("weights", "mobile_sam.pt")
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join("..", "weights", "mobile_sam.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    mg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.45,
        min_mask_region_area=30,
    )

    count = 0
    t0 = time.time()
    for tile in test_tiles:
        masks = mg.generate(tile["tile_data"])
        valid = tile["valid_mask"]
        for ann in masks:
            m = ann["segmentation"]
            if np.sum(np.logical_and(m, valid)) / np.sum(m) >= 0.8:
                count += 1
    elapsed = time.time() - t0

    del mg, sam
    torch.cuda.empty_cache()
    return elapsed, count

# --- Baslines ---
configs = [
    (8,  "pps=8"),
    (10, "pps=10"),
    (12, "pps=12"),
    (16, "pps=16 (current)"),
]

results = []
for pps, label in configs:
    print(f"  [{label}] running...", end=" ", flush=True)
    t, c = test_config(pps)
    results.append((pps, label, t, c))
    print(f"{t:.1f}s  |  {c} trees  |  {SAMPLE_TILES/t:.2f} tiles/s  |  {c/t:.1f} trees/s")

# --- Summary table ---
print()
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

baseline_t, baseline_c = results[-1][2], results[-1][3]  # pps=16 as baseline
print(f"  {'Config':<22} {'Time':>7} {'Trees':>7} {'Speedup':>8} {'Recall':>8} {'Score*':>8}")
print(f"  {'-'*60}")

best_score = 0
best_config = None

for pps, label, t, c in results:
    speedup = baseline_t / t
    recall = c / baseline_c * 100
    # Score: geometric mean of speedup and recall (balanced metric)
    score = (speedup * recall) ** 0.5
    marker = " << BEST" if score > best_score else ""
    if score > best_score:
        best_score = score
        best_config = (pps, label, t, c, speedup, recall)
    print(f"  {label:<22} {t:>6.1f}s {c:>7} {speedup:>7.2f}x {recall:>7.1f}% {score:>7.1f}{marker}")

# Per-tree cost
print()
print(f"  {'Config':<22} {'ms/tree':>8} {'Full-TIF est':>14}")
print(f"  {'-'*44}")
non_blank_total = len(non_blank)
for pps, label, t, c in results:
    ms_per_tree = t / c * 1000 if c > 0 else float('inf')
    full_est = t / SAMPLE_TILES * non_blank_total / 60
    print(f"  {label:<22} {ms_per_tree:>7.1f}ms {full_est:>12.0f} min")

print()
print(f"  * Score = sqrt(speedup × recall) — balanced metric")
print(f"  Best: {best_config[1]} — {best_config[4]:.1f}x faster, {best_config[5]:.1f}% recall")
