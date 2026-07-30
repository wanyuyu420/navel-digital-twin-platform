"""Analyze optimal upload chunk size for 10-second processing budget."""
import sys, os, time, numpy as np
sys.path.insert(0, '.')
from app.services.tif_service import TifService
from ultralytics import YOLO

FILE = 'data/uploads/2019081929_transparent_mosaic_group1.tif'
model = YOLO('weights/best.pt')

tiles = list(TifService.slice_tif_generator(FILE))
xs = sorted(set(t['window_x'] for t in tiles))
ys = sorted(set(t['window_y'] for t in tiles))
cols, rows = len(xs), len(ys)
print(f'Tile grid: {rows} rows x {cols} cols = {len(tiles)} tiles')
print(f'Tile = 512px (65.4m), stride = 448px')
print()

# Build tree-count grid: index by (row, col)
grid = np.zeros((rows, cols), dtype=np.int32)
for i, t in enumerate(tiles):
    if t['tile_data'].max() >= 10:
        results = model(t['tile_data'], verbose=False, conf=0.15)
        boxes = results[0].boxes
        n = len(boxes.xyxy) if boxes is not None else 0
        r = t['window_y'] // 448
        c = t['window_x'] // 448
        if r < rows and c < cols:
            grid[r, c] = n

print(f'Total boxes (low conf): {grid.sum()}')
print(f'Tiles with trees: {(grid > 0).sum()}/{grid.size}')
print(f'Max boxes in a tile: {grid.max()}')
print()

# Cost model from real measurements:
def tile_cost(n):
    if n == 0: return 45
    elif n <= 10: return 45 + n * 16
    elif n <= 30: return 45 + n * 13
    else: return 45 + n * 10

# Calculate for each chunk shape: what's the MAX cost across the image?
# MAX because we want to guarantee 10s even for the densest region
TARGET_MS = 10000

print("=" * 75)
print(f"{'Chunk (tiles)':>14s} {'Chunk (px)':>14s} {'Ground (m)':>14s} {'MB':>8s} {'Max trees':>10s} {'Max time':>10s} {'Budget':>10s}")
print("-" * 75)

for h_tiles in range(1, 10):
    for w_tiles in range(1, 12):
        # Skip unreasonable aspect ratios
        if max(h_tiles, w_tiles) / min(h_tiles, w_tiles) > 3:
            continue

        # Slide window over grid, find max cost
        max_cost = 0
        max_trees = 0
        for r in range(0, rows - h_tiles + 1, max(1, h_tiles // 2)):
            for c in range(0, cols - w_tiles + 1, max(1, w_tiles // 2)):
                chunk = grid[r:r+h_tiles, c:c+w_tiles]
                trees = int(chunk.sum())
                cost = sum(tile_cost(int(n)) for n in chunk.flatten())
                if cost > max_cost:
                    max_cost = cost
                    max_trees = trees

        # Convert to physical dimensions
        px_w = w_tiles * 448 + 64  # account for overlap
        px_h = h_tiles * 448 + 64
        ground_w = px_w * 0.12773
        ground_h = px_h * 0.12773
        # File size est: uint8, 3 bands
        mb = (px_w * px_h * 3) / (1024 * 1024)

        status = "OK" if max_cost <= TARGET_MS else "TOO SLOW"
        marker = " <<<" if status == "OK" else ""

        print(f"{h_tiles}x{w_tiles} ({h_tiles*w_tiles:>3d}t)      {px_w:>5d}x{px_h:<5d}    {ground_w:>5.0f}x{ground_h:<5.0f}m   {mb:>6.1f}MB {max_trees:>10d} {max_cost:>8.0f}ms {TARGET_MS:>8d}ms{marker}")

print()
print("Key: 'OK' = even densest chunk of this shape fits in 10s budget")