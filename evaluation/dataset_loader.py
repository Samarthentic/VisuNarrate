"""VisuNarrate - Evaluation Dataset Loader.

Loads, validates, and manages ground-truth annotated benchmark datasets:
- Image evaluation ground truth: Bounding boxes, reference captions, VQA reference Q&A pairs, scene types.
- Video evaluation ground truth: Segment intervals, activity transitions, ground-truth timestamps.
- Explicit annotation validation: Distinguishes between annotated vs. unannotated assets to prevent fabricated evaluation scores.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DatasetLoader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "data" / "evaluation_datasets"


class EvaluationDatasetLoader:
    """Manages access to ground-truth labeled evaluation datasets."""

    def __init__(self, datasets_dir: Optional[Path] = None):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DEFAULT_DATASETS_DIR
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.image_gt_path = self.datasets_dir / "image_ground_truth.json"
        self.video_gt_path = self.datasets_dir / "video_ground_truth.json"

        self._image_gt_cache: Optional[Dict[str, Any]] = None
        self._video_gt_cache: Optional[Dict[str, Any]] = None

    def load_image_ground_truth(self, force_reload: bool = False) -> Dict[str, Any]:
        """Loads and returns all ground-truth image annotations.

        Returns:
            Dict mapping image filenames to annotation dicts.
        """
        if self._image_gt_cache is not None and not force_reload:
            return self._image_gt_cache

        if not self.image_gt_path.exists():
            logger.warning(f"[DatasetLoader] Image ground-truth file not found at: {self.image_gt_path}")
            self._image_gt_cache = {}
            return self._image_gt_cache

        try:
            with open(self.image_gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._image_gt_cache = data.get("images", data)
                logger.info(f"[DatasetLoader] Loaded {len(self._image_gt_cache)} labeled image annotations.")
                return self._image_gt_cache
        except Exception as e:
            logger.error(f"[DatasetLoader] Error loading image ground-truth: {e}", exc_info=True)
            self._image_gt_cache = {}
            return self._image_gt_cache

    def get_image_annotation(self, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieves ground-truth annotation for a specific image filename."""
        all_gt = self.load_image_ground_truth()
        # Direct match or basename match
        basename = os.path.basename(filename)
        if basename in all_gt:
            return all_gt[basename]
        if filename in all_gt:
            return all_gt[filename]
        return None

    def has_image_annotation(self, filename: str) -> bool:
        """Returns True if ground-truth annotation exists for the image."""
        return self.get_image_annotation(filename) is not None

    def load_video_ground_truth(self, force_reload: bool = False) -> Dict[str, Any]:
        """Loads and returns all ground-truth video annotations.

        Returns:
            Dict mapping video filenames/ids to annotation dicts.
        """
        if self._video_gt_cache is not None and not force_reload:
            return self._video_gt_cache

        if not self.video_gt_path.exists():
            logger.warning(f"[DatasetLoader] Video ground-truth file not found at: {self.video_gt_path}")
            self._video_gt_cache = {}
            return self._video_gt_cache

        try:
            with open(self.video_gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._video_gt_cache = data.get("videos", data)
                logger.info(f"[DatasetLoader] Loaded {len(self._video_gt_cache)} labeled video annotations.")
                return self._video_gt_cache
        except Exception as e:
            logger.error(f"[DatasetLoader] Error loading video ground-truth: {e}", exc_info=True)
            self._video_gt_cache = {}
            return self._video_gt_cache

    def get_video_annotation(self, video_name_or_path: str) -> Optional[Dict[str, Any]]:
        """Retrieves ground-truth annotation for a specific video filename."""
        all_gt = self.load_video_ground_truth()
        basename = os.path.basename(video_name_or_path)
        if basename in all_gt:
            return all_gt[basename]
        if video_name_or_path in all_gt:
            return all_gt[video_name_or_path]
        return None

    def save_image_ground_truth(self, annotations: Dict[str, Any]):
        """Saves image ground-truth annotations to disk."""
        payload = {
            "version": "1.0",
            "description": "VisuNarrate Ground Truth Annotations for Image Pipeline Evaluation",
            "total_annotated": len(annotations),
            "images": annotations,
        }
        with open(self.image_gt_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._image_gt_cache = annotations
        logger.info(f"[DatasetLoader] Saved {len(annotations)} image annotations to {self.image_gt_path}")

    def save_video_ground_truth(self, annotations: Dict[str, Any]):
        """Saves video ground-truth annotations to disk."""
        payload = {
            "version": "1.0",
            "description": "VisuNarrate Ground Truth Annotations for Video Pipeline Evaluation",
            "total_annotated": len(annotations),
            "videos": annotations,
        }
        with open(self.video_gt_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._video_gt_cache = annotations
        logger.info(f"[DatasetLoader] Saved {len(annotations)} video annotations to {self.video_gt_path}")
