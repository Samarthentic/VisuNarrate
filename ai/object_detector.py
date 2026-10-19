"""VisuNarrate - AI Object Detector Module (YOLOv8n).

Provides:
- Local on-device object detection using Ultralytics YOLOv8n.
- Guaranteed 100% offline inference from `models/yolo/yolov8n.pt`.
- Apple Silicon Metal (MPS) acceleration.
- Extraction of genuine detector confidence scores, bounding boxes, and object counts.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from PIL import Image

import torch

logger = logging.getLogger("ObjectDetector")


class ObjectDetector:
    """Manages local YOLOv8n object detection model inference."""

    def __init__(self, model_path: Optional[str] = None):
        """Initializes the object detector with a local model file.

        Args:
            model_path: Absolute or relative path to `yolov8n.pt`.
                        Defaults to `VisuNarrate/models/yolo/yolov8n.pt`.
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "models", "yolo", "yolov8n.pt")
        else:
            self.model_path = model_path

        self.device = self._detect_device()
        self.model = None
        self._is_loaded = False
        self.model_name = "YOLOv8n"

        logger.info(f"[ObjectDetector] Initialized. Target model path: '{self.model_path}' | Device: '{self.device}'")

    def _detect_device(self) -> str:
        """Detect and return the best available compute device (MPS on Apple Silicon, fallback to CPU)."""
        if torch.backends.mps.is_available():
            logger.info("[ObjectDetector] Apple Silicon Metal (MPS) detected for YOLO inference.")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("[ObjectDetector] NVIDIA CUDA GPU detected for YOLO inference.")
            return "cuda"
        else:
            logger.info("[ObjectDetector] Standard CPU backend selected for YOLO inference.")
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded into memory."""
        return self._is_loaded

    def load_model(self) -> bool:
        """Loads YOLOv8n strictly from the local file without network fallback.

        Raises:
            FileNotFoundError: If the local `.pt` file does not exist.
        """
        if self._is_loaded:
            return True

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Local YOLOv8n model file not found at: '{self.model_path}'. "
                f"Please run 'python scripts/download_yolo.py' once to download weights."
            )

        logger.info(f"[ObjectDetector] Loading local weights from '{self.model_path}'...")
        start_time = time.time()

        try:
            from ultralytics import YOLO
            # Load from explicit local file path (no automatic download allowed)
            self.model = YOLO(self.model_path)
            self._is_loaded = True
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"[ObjectDetector] YOLOv8n model loaded successfully in {duration_ms}ms on {self.device}.")
            return True
        except Exception as e:
            logger.error(f"[ObjectDetector] Failed to load YOLOv8n: {str(e)}", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Error loading offline YOLOv8n model: {str(e)}")

    def detect_objects(self, image: Image.Image, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """Runs on-device object detection on a PIL Image.

        Args:
            image: Preprocessed PIL Image in RGB format.
            conf_threshold: Minimum confidence threshold for detections (default: 0.25).

        Returns:
            Dict containing:
                - objects: List of detected objects with genuine confidence and bbox coordinates.
                - counts: Dictionary mapping detected class labels to occurrence counts.
                - device: Hardware device used (e.g. 'mps' or 'cpu').
                - model: Model identifier ('YOLOv8n').
                - latency_ms: Execution time in milliseconds.
                - local_inference: True.
                - total_detections: Total count of detected entities.
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        # Run inference (verbose=False to suppress spammy stdout logs)
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        detected_objects: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = result.names.get(cls_id, f"class_{cls_id}")
                    confidence = round(float(box.conf[0].item()), 3)

                    # Extract bounding box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].tolist()
                    bbox = {
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1),
                    }

                    detected_objects.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": bbox,
                    })

                    counts[label] = counts.get(label, 0) + 1

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[ObjectDetector] Detected {len(detected_objects)} objects in {elapsed_ms}ms on {self.device}: {counts}")

        return {
            "objects": detected_objects,
            "counts": counts,
            "total_detections": len(detected_objects),
            "device": self.device,
            "model": self.model_name,
            "latency_ms": elapsed_ms,
            "local_inference": True,
        }
