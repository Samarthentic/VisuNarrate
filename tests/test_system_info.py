"""VisuNarrate - System Info & Technical Telemetry Unit Tests."""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

os.environ["VISUNARRATE_SKIP_WARMUP"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"

from app import app
from core.image_engine import ImageEngine
from core.video_engine import VideoEngine


class TestSystemInfoEndpoint(unittest.TestCase):
    """Verifies that /api/system/info returns accurate, measured runtime metadata."""

    def setUp(self):
        self.client = app.test_client()

    def test_system_info_schema(self):
        """Validates the schema and completeness of /api/system/info."""
        res = self.client.get("/api/system/info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertTrue(data.get("success"))
        for section in ["runtime", "libraries", "device", "offline_flags", "memory", "models", "ready", "training"]:
            self.assertIn(section, data, f"Missing section: {section}")

        # Check offline flags
        offline = data["offline_flags"]
        self.assertEqual(offline.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(offline.get("TRANSFORMERS_OFFLINE"), "1")
        self.assertEqual(offline.get("YOLO_OFFLINE"), "1")
        self.assertTrue(offline.get("all_offline_enforced"))

        # Check libraries
        libs = data["libraries"]
        for lib in ["torch", "transformers", "ultralytics", "cv2"]:
            self.assertIn(lib, libs)
            self.assertTrue(len(libs[lib]) > 0)

        # Check memory
        mem = data["memory"]
        self.assertGreater(mem.get("total_ram_gb", 0), 0)
        self.assertGreater(mem.get("available_ram_gb", 0), 0)
        self.assertGreater(mem.get("free_disk_gb", 0), 0)

        # Check models & parameters
        models = data["models"]
        for m in ["blip_captioning", "yolo_detection", "blip_vqa"]:
            self.assertIn(m, models)
            self.assertGreater(models[m]["parameters_count"], 1_000_000)
            self.assertGreater(models[m]["size_on_disk_mb"], 1.0)
        self.assertGreater(models["total_parameters_count"], 500_000_000)

        # Check training disclosure - MUST be 0 epochs and inference only
        training = data["training"]
        self.assertEqual(training.get("epochs_run"), 0)
        self.assertEqual(training.get("fine_tuning"), "none")
        self.assertIn("inference only", training.get("statement", "").lower())


class TestTechnicalTelemetrySchema(unittest.TestCase):
    """Verifies that ImageEngine and VideoEngine produce valid technical blocks."""

    def test_image_technical_block_structure(self):
        """Verifies ImageEngine.analyze_scene includes complete technical block."""
        mock_vision = MagicMock()
        mock_vision.device = "mps"
        mock_vision.generate_scene_observations.return_value = {
            "observations": {"summary": "A person sitting at a desk."},
            "model": "Salesforce/blip-image-captioning-base",
            "device": "mps",
            "total_latency_ms": 120.0,
            "latencies_ms": {"summary": 120.0},
            "observations_count": 1,
        }

        mock_object = MagicMock()
        mock_object.detect_objects.return_value = {
            "objects": [{"label": "person", "confidence": 0.95}],
            "counts": {"person": 1},
            "total_detections": 1,
            "model": "YOLOv8n",
            "device": "mps",
            "latency_ms": 25.0,
        }

        mock_vqa = MagicMock()
        mock_vqa.answer_questions.return_value = {
            "qa_pairs": [{"question": "Where is this located?", "answer": "indoors", "confidence": 0.9}],
            "model": "Salesforce/blip-vqa-base",
            "device": "mps",
            "total_latency_ms": 80.0,
        }

        engine = ImageEngine(
            vision_model=mock_vision,
            object_detector=mock_object,
            vqa_model=mock_vqa,
        )

        dummy_img = Image.new("RGB", (200, 200), color="blue")
        result = engine.analyze_scene(dummy_img)

        self.assertIn("technical", result)
        tech = result["technical"]

        self.assertEqual(tech["device"], "mps")
        self.assertEqual(tech["detections_count"], 1)
        self.assertEqual(tech["vqa_queries_count"], 1)
        self.assertIn("timings_ms", tech)

        timings = tech["timings_ms"]
        for key in ["preprocessing_ms", "captioning_ms", "detection_ms", "vqa_ms", "fusion_ms", "narration_ms", "prediction_ms", "total_ms"]:
            self.assertIn(key, timings)
            self.assertIsInstance(timings[key], (int, float))

        # Check training disclosure inside technical block
        self.assertEqual(tech["training"]["epochs"], 0)
        self.assertEqual(tech["training"]["fine_tuning"], "none")


if __name__ == "__main__":
    unittest.main()
