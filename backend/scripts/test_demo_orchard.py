"""YOLO + SAM - Demo orchard visualization test."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import cv2, rasterio, torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

FILE_PATH = "data/uploads/2019081929_orchard_center_demo.tif"
OUT_DIR = "debug_tiles"
YOLO_CONF = 0.15
YOLO_WEIGHTS = "weights/yolov8s_tree_crown.pt"
TILE_SIZE = 512
TILE_OVERLAP = 64
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("YOLO + SAM Demo Orchard Segmentation Test")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading YOLO...", end=" ", flush=True)
t0 = time.time()
yolo = YOLO(YOLO_WEIGHTS)
print(f"{time.time()-t0:.1f}s")

print("Loading MobileSAM...", end=" ", flush=True)
t0 = time.time()
sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
sam.to(device=device); sam.eval()
predictor = SamPredictor(sam)
print(f"{time.time()-t0:.1f}s")

from app.services.tif_service import TifService
with rasterio.open(FILE_PATH) as src:
    print(f"TIF: {src.width}x{src.height}  |  GSD: {src.res[0]:.5f}m")

tiles = list(TifService.slice_tif_generator(FILE_PATH))
non_blank = [(i, t) for i, t in enumerate(tiles) if t["tile_data"].max() >= 10 and t["tile_data"].std() >= 5]
print(f"Tiles: {len(tiles)} total, {len(non_blank)} non-blank")

# Quick scan to find tiles with trees
print("\nScanning for tree tiles...")
tree_tiles = []
for i, (orig_idx, t) in enumerate(non_blank):
    results = yolo(t["tile_data"], verbose=False, conf=0.15)
    boxes = results[0].boxes
    n = len(boxes.xyxy) if boxes is not None else 0
    if n > 0:
        tree_tiles.append((orig_idx, t, n))
tree_tiles.sort(key=lambda x: -x[2])
print(f"Tiles with trees: {len(tree_tiles)}")

# Run full YOLO+SAM and save viz
print(f"\nRunning YOLO+SAM (conf={YOLO_CONF}), saving to {OUT_DIR}/...")
print("-" * 60)

total_yolo = total_enc = total_dec = 0.0
total_boxes = total_trees = 0

for idx, (orig_idx, tile_info, _) in enumerate(tree_tiles):
    tile_rgb = tile_info["tile_data"].copy()
    valid_mask = tile_info["valid_mask"]

    # YOLO
    t0 = time.time()
    results = yolo(tile_rgb, verbose=False, conf=YOLO_CONF)
    boxes = results[0].boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0,4), dtype=np.float32)
    yolo_ms = (time.time()-t0)*1000

    # SAM enc
    t0 = time.time()
    predictor.set_image(tile_rgb)
    enc_ms = (time.time()-t0)*1000

    # SAM dec
    dec_ms = 0.0
    masks_list = []
    for box in boxes_xyxy:
        t0 = time.time()
        mask, score, _ = predictor.predict(box=box[np.newaxis,:], multimask_output=False)
        dec_ms += (time.time()-t0)*1000
        m = mask[0]
        inter = np.logical_and(m, valid_mask)
        if np.sum(inter)/max(np.sum(m),1) >= 0.8:
            masks_list.append((m, box, float(score[0])))

    total_yolo += yolo_ms; total_enc += enc_ms; total_dec += dec_ms
    total_boxes += len(boxes_xyxy); total_trees += len(masks_list)

    # --- Save 4 views per tile ---
    # 1. Raw
    cv2.imwrite(f"{OUT_DIR}/demo_{idx:02d}_raw.png",
                cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))

    # 2. YOLO boxes
    yolo_v = tile_rgb.copy()
    for b in boxes_xyxy:
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(yolo_v, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(f"{OUT_DIR}/demo_{idx:02d}_yolo.png",
                cv2.cvtColor(yolo_v, cv2.COLOR_RGB2BGR))

    # 3. SAM masks
    sam_v = tile_rgb.copy()
    for m, b, _ in masks_list:
        c = np.random.randint(100,255,3).tolist()
        sam_v[m>0] = (sam_v[m>0]*0.5 + np.array(c)*0.5).astype(np.uint8)
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(sam_v, (x1,y1), (x2,y2), (255,0,0), 1)
        cy,cx = int(np.mean(np.where(m)[0])), int(np.mean(np.where(m)[1]))
        cv2.circle(sam_v, (cx,cy), 3, (255,255,255), -1)
    cv2.imwrite(f"{OUT_DIR}/demo_{idx:02d}_sam.png",
                cv2.cvtColor(sam_v, cv2.COLOR_RGB2BGR))

    # 4. Comparison (left=YOLO, right=SAM)
    h, w = tile_rgb.shape[:2]
    comp = np.zeros((h, w*2, 3), dtype=np.uint8)
    comp[:,:w] = yolo_v; comp[:,w:] = sam_v
    cv2.line(comp, (w,0), (w,h), (255,255,255), 3)
    cv2.imwrite(f"{OUT_DIR}/demo_{idx:02d}_comparison.png",
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

    print(f"  [{idx+1}/{len(tree_tiles)}] tile#{orig_idx} "
          f"YOLO:{len(boxes_xyxy):>3}box SAM:{len(masks_list):>3}tree "
          f"({yolo_ms:.0f}+{enc_ms:.0f}+{dec_ms:.0f}ms)")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Tiles processed:    {len(tree_tiles)}")
print(f"  YOLO total:         {total_yolo:.0f}ms")
print(f"  SAM enc total:      {total_enc:.0f}ms")
print(f"  SAM dec total:      {total_dec:.0f}ms")
print(f"  Total inference:    {total_yolo+total_enc+total_dec:.0f}ms ({((total_yolo+total_enc+total_dec)/1000):.1f}s)")
print(f"  YOLO boxes:         {int(total_boxes)}")
print(f"  SAM trees kept:     {int(total_trees)}")
print(f"  Output:             {OUT_DIR}/demo_*")
