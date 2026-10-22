"""VisuNarrate - Live Mode In-Memory Stream & Session Pipeline Tests."""

import os
import time
import base64
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from PIL import Image

os.environ["VISUNARRATE_SKIP_WARMUP"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"

from app import app, ANALYSIS_LOCK
from core.live_engine import (
    LiveEngine,
    LiveSession,
    LiveSessionManager,
    MAX_LIVE_SESSION_HISTORY,
    SESSION_EXPIRY_SECONDS,
)
from core.image_engine import ImageEngine


def _make_test_jpeg_base64(color=(100, 150, 200), width=320, height=240):
    """Generates an in-memory test frame as base64 JPEG without touching disk."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


class TestLiveSessionManager(unittest.TestCase):
    """Verifies that the in-memory LiveSessionManager behaves correctly."""

    def setUp(self):
        self.mgr = LiveSessionManager()

    def test_session_lifecycle_and_ring_buffer(self):
        """Tests session creation, ring buffer truncation to 60 items, and deletion."""
        session = self.mgr.get_or_create_session("test_sess_1")
        self.assertEqual(session.session_id, "test_sess_1")

        # Push 75 items into history
        for i in range(75):
            session.history.append({"idx": i})

        # History must be bounded to MAX_LIVE_SESSION_HISTORY (60)
        self.assertEqual(len(session.history), MAX_LIVE_SESSION_HISTORY)
        self.assertEqual(session.history[0]["idx"], 15)  # Oldest 15 dropped
        self.assertEqual(session.history[-1]["idx"], 74)

        # Deletion
        deleted = self.mgr.delete_session("test_sess_1")
        self.assertTrue(deleted)
        self.assertIsNone(self.mgr.get_session("test_sess_1"))

    def test_session_expiry_cleanup(self):
        """Verifies sessions inactive for > 15 minutes are evicted."""
        session = self.mgr.get_or_create_session("old_sess")
        session.last_activity = time.time() - (SESSION_EXPIRY_SECONDS + 10)

        # Trigger cleanup via new get_or_create call
        _ = self.mgr.get_or_create_session("new_sess")
        self.assertIsNone(self.mgr.get_session("old_sess"))
        self.assertIsNotNone(self.mgr.get_session("new_sess"))


class TestLiveEngineInference(unittest.TestCase):
    """Verifies LiveEngine frame decoding, visual diff gating, and redundancy caching."""

    def setUp(self):
        mock_image_engine = MagicMock(spec=ImageEngine)
        mock_vision = MagicMock()
        mock_vision.device = "mps"
        mock_image_engine.vision_model = mock_vision
        mock_image_engine.analyze_scene.return_value = {
            "scene_narration": "A person standing in an office.",
            "predictive_narration": {
                "prediction": "They will likely remain in the office.",
                "hypotheses": [{"likelihood_label": "most likely", "text": "They will stay.", "evidence": "indoor office"}],
            },
            "prediction_status": "supported_prediction",
            "object_detection": {"counts": {"person": 1}},
            "scene_type": {"scene_type": "photographic_scene"},
            "fused_scene": {"primary_subject": {"value": "Person"}, "activity": {"value": "Standing"}, "environment": {"value": "Office"}},
            "technical": {
                "device": "mps",
                "timings_ms": {"captioning_ms": 100.0, "detection_ms": 20.0, "vqa_ms": 50.0, "total_ms": 170.0},
            },
        }

        self.live_engine = LiveEngine(image_engine=mock_image_engine)

    def test_in_memory_decoding_and_dimension_constraint(self):
        """Ensures frames decode in memory and downscale if > 640px."""
        large_b64 = _make_test_jpeg_base64(width=1280, height=960)
        decoded = self.live_engine.decode_frame(large_b64)
        h, w = decoded.shape[:2]
        self.assertLessEqual(max(h, w), 640)

    def test_redundancy_state_inheritance(self):
        """Frame 1 is FULL_ANALYSIS; identical Frame 2 must be REDUNDANT and skip full analysis."""
        frame_b64 = _make_test_jpeg_base64(color=(50, 100, 150))
        sess_id = "test_red_sess"

        # Frame 1: Full analysis
        res1 = self.live_engine.process_frame(frame_data=frame_b64, session_id=sess_id)
        self.assertTrue(res1["success"])
        snap1 = res1["snapshot"]
        self.assertEqual(snap1["processing_mode"], "FULL_ANALYSIS")
        self.assertEqual(res1["stats"]["full_analyses"], 1)
        self.assertEqual(res1["stats"]["redundant_frames"], 0)

        # Frame 2: Identical frame -> REDUNDANT
        res2 = self.live_engine.process_frame(frame_data=frame_b64, session_id=sess_id)
        self.assertTrue(res2["success"])
        snap2 = res2["snapshot"]
        self.assertEqual(snap2["processing_mode"], "REDUNDANT")
        self.assertLess(snap2["visual_diff_ratio"], 0.04)
        self.assertEqual(res2["stats"]["redundant_frames"], 1)
        self.assertEqual(snap2["narration"], snap1["narration"])

        # Frame 3: Significantly changed frame -> FULL_ANALYSIS
        frame_changed = _make_test_jpeg_base64(color=(240, 20, 20))
        res3 = self.live_engine.process_frame(frame_data=frame_changed, session_id=sess_id)
        snap3 = res3["snapshot"]
        self.assertEqual(snap3["processing_mode"], "FULL_ANALYSIS")
        self.assertGreater(snap3["visual_diff_ratio"], 0.04)
        self.assertEqual(res3["stats"]["full_analyses"], 2)


class TestLiveApiEndpoints(unittest.TestCase):
    """Verifies HTTP endpoints for Live Mode."""

    def setUp(self):
        self.client = app.test_client()

    def test_live_analyze_endpoint_and_busy_lock(self):
        """Tests POST /api/live/analyze and verifies 503 when ANALYSIS_LOCK is held."""
        frame_b64 = _make_test_jpeg_base64()

        # 1. Successful request
        res = self.client.post("/api/live/analyze", json={
            "session_id": "api_test_sess",
            "frame": frame_b64,
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("snapshot", data["data"])

        # 2. Concurrency Lock: When ANALYSIS_LOCK is locked, returns 503
        with ANALYSIS_LOCK:
            res_busy = self.client.post("/api/live/analyze", json={
                "session_id": "api_test_sess",
                "frame": frame_b64,
            })
            self.assertEqual(res_busy.status_code, 503)
            busy_data = res_busy.get_json()
            self.assertFalse(busy_data["success"])
            self.assertTrue(busy_data.get("is_busy"))

    def test_live_session_summary_and_delete(self):
        """Tests generating summary and deleting a live session."""
        sess_id = "summary_test_sess"
        frame_b64 = _make_test_jpeg_base64()

        # Ingest 2 frames
        self.client.post("/api/live/analyze", json={"session_id": sess_id, "frame": frame_b64})
        self.client.post("/api/live/analyze", json={"session_id": sess_id, "frame": frame_b64})

        # Generate summary
        res_sum = self.client.post(f"/api/live/session/{sess_id}/summary")
        self.assertEqual(res_sum.status_code, 200)
        sum_data = res_sum.get_json()
        self.assertTrue(sum_data["success"])
        self.assertIn("live_story", sum_data["data"])

        # Delete session
        res_del = self.client.delete(f"/api/live/session/{sess_id}")
        self.assertEqual(res_del.status_code, 200)


if __name__ == "__main__":
    unittest.main()
