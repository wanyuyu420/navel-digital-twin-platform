import os
import numpy as np
from ultralytics import YOLO


class YoloService:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("[YoloService] Loading YOLO tree-crown model...")
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(os.path.dirname(BASE_DIR), "weights", "yolov8s_tree_crown.pt")

            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(BASE_DIR, "..", "weights", "yolov8s_tree_crown.pt")

            # Fallback to best.pt if small model not found
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(os.path.dirname(BASE_DIR), "weights", "best.pt")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(BASE_DIR, "..", "weights", "best.pt")

            print(f"[YoloService] Checkpoint: {checkpoint_path}")
            cls._model = YOLO(checkpoint_path)
            cls._instance = cls._model
            print(f"[YoloService] Loaded. Classes: {cls._model.names}")

        return cls._model

    @staticmethod
    def detect_boxes(tile_rgb, model, conf: float = 0.35):
        """
        Run YOLO inference on a tile and return bounding boxes.
        Returns np.ndarray of shape (N, 4) in xyxy format, or empty array.
        """
        results = model(tile_rgb, verbose=False, conf=conf)
        boxes = results[0].boxes
        if boxes is None:
            return np.empty((0, 4), dtype=np.float32)
        return boxes.xyxy.cpu().numpy()
