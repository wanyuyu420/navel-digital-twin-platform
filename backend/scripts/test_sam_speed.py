"""Quick test script for SAM inference with dual-core threading on a single TIF."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import rasterio
from pyproj import Transformer
from app.services.sam_service import SamInferenceService
from app.services.tif_service import TifService

FILE_PATH = r"C:\Users\asus\Desktop\Esri_QC\water-digital-twin-platform\backend\data\uploads\6e5ec825907b454f86271ace07c1f411_2019062105_mosaic_group1.tif"

print("=" * 60)
print("MobileSAM dual-core inference test")
print("=" * 60)

# Open TIF to get metadata
with rasterio.open(FILE_PATH) as src:
    tif_w, tif_h = src.width, src.height
    tif_crs = str(src.crs)
    print(f"TIF size: {tif_w} x {tif_h} pixels")
    print(f"TIF CRS:  {tif_crs}")

# Load SAM once
t0 = time.time()
mask_generator = SamInferenceService.get_instance()
print(f"Model load time: {time.time() - t0:.1f}s")

# Generate tile list
t0 = time.time()
tiles = list(TifService.slice_tif_generator(FILE_PATH))
total_tiles = len(tiles)
print(f"Total tiles: {total_tiles}")
print(f"Tile list generation time: {time.time() - t0:.1f}s")

# Pre-count blank tiles for reference
blank_count = sum(1 for t in tiles if t["tile_data"].max() < 10)
print(f"Blank tiles (skip): {blank_count}")
print(f"Active tiles: {total_tiles - blank_count}")

transformer = Transformer.from_crs(tif_crs, "EPSG:4326", always_xy=True)

# --- DUAL-CORE INFERENCE ---
inference_lock = threading.Lock()
tile_count = [0]

def process_single_tile(tile_info):
    tile_rgb = tile_info["tile_data"]
    valid_mask = tile_info["valid_mask"]
    window_x = tile_info["window_x"]
    window_y = tile_info["window_y"]
    transform = tile_info["transform"]

    if tile_rgb.max() < 10:
        return []

    with inference_lock:
        local_trees = SamInferenceService.infer_tile(
            tile_rgb, valid_mask, mask_generator)

    results = []
    for tree in local_trees:
        local_cx, local_cy = tree["local_centroid"]
        global_px = window_x + local_cx
        global_py = window_y + local_cy
        geo_x, geo_y = rasterio.transform.xy(
            transform, global_py, global_px, offset="center")
        lng, lat = transformer.transform(geo_x, geo_y)
        results.append({
            "lng": round(lng, 8),
            "lat": round(lat, 8),
            "area_pixels": tree["area_pixels"],
        })
    return results

print("\nRunning dual-core inference (max_workers=2)...")
print("-" * 40)
t_start = time.time()
all_trees = []

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_single_tile, tile) for tile in tiles]
    for future in as_completed(futures):
        try:
            trees = future.result()
            all_trees.extend(trees)
        except Exception as e:
            print(f"[Error] tile failed: {e}")
        tile_count[0] += 1
        if tile_count[0] % 50 == 0 or tile_count[0] == total_tiles:
            elapsed = time.time() - t_start
            rate = tile_count[0] / elapsed
            print(f"  [{tile_count[0]}/{total_tiles}] tiles | "
                  f"{elapsed:.0f}s elapsed | {rate:.2f} tiles/s | "
                  f"{len(all_trees)} trees found")

t_total = time.time() - t_start
print("-" * 40)
print(f"DONE in {t_total:.1f}s ({t_total/60:.1f} min)")
print(f"Trees detected: {len(all_trees)}")
print(f"Throughput: {tile_count[0]/t_total:.2f} tiles/s")
print(f"Active tile time: {t_total/(total_tiles - blank_count):.2f}s per active tile")
