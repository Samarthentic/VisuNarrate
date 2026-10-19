#!/usr/bin/env python3
"""Regression snapshot harness for VisuNarrate.

Runs the full image pipeline on every image in data/evaluation_images/ and
data/sample_images/, capturing YOLO counts, BLIP caption, narration and
prediction. Saves to data/evaluation_results/<output_name>.json.

Usage:
    .venv/bin/python scripts/regression_snapshot.py snapshot_before
    .venv/bin/python scripts/regression_snapshot.py snapshot_after
"""

import os
import sys
import json
import time
import glob

# Ensure offline mode (must be set before any model imports)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.image_engine import ImageEngine
from ai.vision_model import VisionModelLoader
from ai.object_detector import ObjectDetector
from ai.vqa_model import VQAModelLoader
from services.evidence_fusion import EvidenceFusionEngine
from services.prediction_service import PredictionService


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images():
    """Collect all images from evaluation and sample directories."""
    dirs = [
        os.path.join(PROJECT_ROOT, "data", "evaluation_images"),
        os.path.join(PROJECT_ROOT, "data", "sample_images"),
    ]
    images = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(d, f))
    return images


def main():
    if len(sys.argv) < 2:
        output_name = "snapshot"
    else:
        output_name = sys.argv[1]

    output_dir = os.path.join(PROJECT_ROOT, "data", "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.json")

    print(f"[Regression Harness] Output: {output_path}")

    images = collect_images()
    print(f"[Regression Harness] Found {len(images)} images")

    # Initialize engine
    vision_loader = VisionModelLoader()
    object_detector = ObjectDetector()
    vqa_loader = VQAModelLoader()
    evidence_fusion = EvidenceFusionEngine()
    prediction_service = PredictionService()

    engine = ImageEngine(
        vision_model=vision_loader,
        object_detector=object_detector,
        vqa_model=vqa_loader,
        evidence_fusion=evidence_fusion,
        prediction_service=prediction_service,
    )

    # Warm-up on first image
    if images:
        print("[Regression Harness] Warming up models ...")
        engine.analyze_scene(images[0])
        print("[Regression Harness] Warm-up complete.")

    results = []
    for i, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        print(f"[{i+1}/{len(images)}] Analyzing: {filename}")

        t0 = time.time()
        try:
            result = engine.analyze_scene(img_path)
            latency = round(time.time() - t0, 2)

            yolo_counts = result.get("object_detection", {}).get("counts", {})
            caption = result.get("caption_evidence", {}).get("summary", "")
            narration = result.get("scene_narration", "")
            prediction = result.get("predictive_narration", {}).get("prediction", "")

            results.append({
                "index": i + 1,
                "filename": filename,
                "yolo_counts": yolo_counts,
                "caption": caption,
                "narration": narration,
                "prediction": prediction,
                "latency_s": latency,
            })

            print(f"   YOLO: {yolo_counts}")
            print(f"   Caption: {caption}")
            print(f"   Narration: {narration}")
            print(f"   Prediction: {prediction}")
            print(f"   Latency: {latency}s")
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append({
                "index": i + 1,
                "filename": filename,
                "error": str(e),
            })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Regression Harness] Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
