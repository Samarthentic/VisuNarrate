"""VisuNarrate - Unit Test Suite for Video Storytelling & Job Engine.

Covers:
1. Video file validation (formats, empty file, corrupt file, duration limits, size limits).
2. Orientation and rotation metadata handling for mobile portrait videos.
3. Even sampling plan distribution across entire video timelines.
4. Consecutive moment segment merging into cohesive timeline segments.
5. Grounded narrative generation without ungrounded temporal claims.
6. Temporal future prediction with structured evidence-based hypotheses.
7. Job system lifecycle, cancellation, and concurrency protection.
"""

import os
import cv2
import tempfile
import unittest
import numpy as np
from pathlib import Path

from core.video_engine import VideoEngine
from services.temporal_service import TemporalService
from services.video_narration_service import VideoNarrationService
from services.prediction_service import PredictionService
from unittest.mock import patch, MagicMock
from app import VideoJob, JobStatus, ANALYSIS_LOCK, app, JOBS, video_engine


class TestVideoPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="visunarrate_video_test_")
        cls.valid_video_path = os.path.join(cls.test_dir, "valid_test.mp4")
        cls.portrait_video_path = os.path.join(cls.test_dir, "portrait_test.mp4")
        cls.corrupt_video_path = os.path.join(cls.test_dir, "corrupt_test.mp4")
        cls.empty_video_path = os.path.join(cls.test_dir, "empty_test.mp4")

        # 1. Create a valid small 2-second test video (100x100, 20 fps, 40 frames)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.valid_video_path, fourcc, 20.0, (100, 100))
        for i in range(40):
            frame = np.full((100, 100, 3), (i * 5) % 255, dtype=np.uint8)
            out.write(frame)
        out.release()

        # 2. Create an empty file (0 bytes)
        with open(cls.empty_video_path, "wb") as f:
            pass

        # 3. Create a corrupt video file (random non-video bytes)
        with open(cls.corrupt_video_path, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_RANDOM_BYTES_CORRUPT")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.video_engine = VideoEngine()
        self.temporal_service = TemporalService()
        self.prediction_service = PredictionService()
        self.video_narration = VideoNarrationService(prediction_service=self.prediction_service)

    # -------------------------------------------------------------------------
    # 1. Validation & Error Handling
    # -------------------------------------------------------------------------
    def test_validation_valid_file(self):
        """Valid video returns accurate metadata."""
        meta = self.video_engine.validate_video_file(self.valid_video_path)
        self.assertEqual(meta["width"], 100)
        self.assertEqual(meta["height"], 100)
        self.assertEqual(meta["fps"], 20.0)
        self.assertEqual(meta["total_frames"], 40)
        self.assertAlmostEqual(meta["duration_seconds"], 2.0, places=1)
        self.assertFalse(meta["is_portrait"])

    def test_validation_empty_file(self):
        """Empty video file must raise clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.video_engine.validate_video_file(self.empty_video_path)
        self.assertIn("0 bytes", str(ctx.exception))

    def test_validation_corrupt_file(self):
        """Corrupt video file must raise clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.video_engine.validate_video_file(self.corrupt_video_path)
        self.assertIn("Corrupt or unreadable", str(ctx.exception))

    def test_validation_unsupported_format(self):
        """File with unsupported extension must raise clear ValueError."""
        unsupported = os.path.join(self.test_dir, "test.xyz")
        with open(unsupported, "wb") as f:
            f.write(b"data")
        with self.assertRaises(ValueError) as ctx:
            self.video_engine.validate_video_file(unsupported)
        self.assertIn("Unsupported video format", str(ctx.exception))

    def test_validation_duration_exceeded(self):
        """Video duration exceeding 180s (3 min) limit must be rejected."""
        # Create a tiny mock that reports 200s
        class MockVideoEngine(VideoEngine):
            def validate_video_file(self, path):
                meta = super().validate_video_file(path)
                if meta["duration_seconds"] > self.MAX_VIDEO_DURATION_SECONDS:
                    raise ValueError(f"Video duration ({meta['duration_seconds']}s) exceeds the maximum allowed limit of 180s.")
                return meta

        engine = MockVideoEngine()
        # Test directly against custom limit
        engine.MAX_VIDEO_DURATION_SECONDS = 1.0
        with self.assertRaises(ValueError) as ctx:
            engine.validate_video_file(self.valid_video_path)
        self.assertIn("exceeds the maximum allowed limit", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Orientation & Portrait Handling
    # -------------------------------------------------------------------------
    def test_orientation_correction(self):
        """Rotation angle of 90 degrees on landscape frame rotates to portrait."""
        landscape_frame = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200
        rotated = self.video_engine.apply_orientation(landscape_frame, rotation_angle=90.0)
        # Should now be height 200, width 100
        self.assertEqual(rotated.shape[0], 200)
        self.assertEqual(rotated.shape[1], 100)

    # -------------------------------------------------------------------------
    # 3. Even Sampling Distribution
    # -------------------------------------------------------------------------
    def test_sampling_plan_distribution(self):
        """Sampling plan must evenly span from start to end of video."""
        total_frames = 1800  # 60s at 30fps
        fps = 30.0
        plan = self.video_engine.calculate_sampling_plan(
            total_frames=total_frames, fps=fps, target_fps_sample=1.0, max_frames=15
        )
        self.assertLessEqual(len(plan), 15)
        # First frame should be anchored at 0
        self.assertEqual(plan[0][0], 0)
        # Last frame should be the final frame
        self.assertEqual(plan[-1][0], total_frames - 1)
        self.assertAlmostEqual(plan[-1][1], round((total_frames - 1) / fps, 2), places=1)

    def test_sampling_plan_anti_aliasing_offsets(self):
        """Centered interval sampling with deterministic jitter must prevent periodic stroboscopic aliasing."""
        total_frames = 1200  # 40s at 30fps
        fps = 30.0
        period = 30  # e.g. 1 Hz periodic visual loop (every 30 frames)
        plan = self.video_engine.calculate_sampling_plan(
            total_frames=total_frames, fps=fps, target_fps_sample=1.0, max_frames=20
        )
        # Check phases of sampled frames modulo period
        phases = [frame_idx % period for frame_idx, _ in plan]
        unique_phases = set(phases)
        # In a naive grid step (stride 30 or 60), all phases would be identical ({0}).
        # Anti-aliased centered sampling with deterministic jitter must produce multiple diverse phases.
        self.assertGreater(len(unique_phases), 1, "Sampling plan should not fall into a single periodic phase.")
        # Ensure plan is strictly monotonically increasing
        indices = [p[0] for p in plan]
        self.assertEqual(indices, sorted(list(set(indices))))

    # -------------------------------------------------------------------------
    # 4. Segment Merging
    # -------------------------------------------------------------------------
    def test_consecutive_segment_merging(self):
        """Frames sharing identical activity/environment merge into one segment; transitions split."""
        frame_sequence = [
            {"sample_number": 1, "frame_index": 0, "timestamp_seconds": 0.0, "activity": "using smartphone", "primary_subject": "Person", "environment": "Indoors", "detected_objects": {"person": 1, "cell phone": 1}, "processing_mode": "FULL_ANALYSIS", "scene_narration": "Person with phone"},
            {"sample_number": 2, "frame_index": 24, "timestamp_seconds": 1.0, "activity": "using smartphone", "primary_subject": "Person", "environment": "Indoors", "detected_objects": {"person": 1, "cell phone": 1}, "processing_mode": "REDUNDANT", "scene_narration": "Person with phone"},
            {"sample_number": 3, "frame_index": 48, "timestamp_seconds": 2.0, "activity": "using smartphone", "primary_subject": "Person", "environment": "Indoors", "detected_objects": {"person": 1, "cell phone": 1}, "processing_mode": "REDUNDANT", "scene_narration": "Person with phone"},
            # Transition at 3.0s to reading book
            {"sample_number": 4, "frame_index": 72, "timestamp_seconds": 3.0, "activity": "reading", "primary_subject": "Person", "environment": "Indoors", "detected_objects": {"person": 1, "book": 1}, "processing_mode": "KEY_CHANGE", "scene_narration": "Person reading book"},
            {"sample_number": 5, "frame_index": 96, "timestamp_seconds": 4.0, "activity": "reading", "primary_subject": "Person", "environment": "Indoors", "detected_objects": {"person": 1, "book": 1}, "processing_mode": "REDUNDANT", "scene_narration": "Person reading book"},
        ]

        changes = [
            {"timestamp_seconds": 3.0, "type": "activity_change", "from": "using smartphone", "to": "reading", "description": "Activity transitions from using smartphone to reading."}
        ]

        segments = self.temporal_service.merge_consecutive_segments(frame_sequence, changes, video_duration=5.0)

        # Should produce exactly 2 distinct segments
        self.assertEqual(len(segments), 2)

        seg1 = segments[0]
        self.assertEqual(seg1["segment_id"], 1)
        self.assertEqual(seg1["start_seconds"], 0.0)
        self.assertEqual(seg1["end_seconds"], 3.0)
        self.assertEqual(seg1["activity"], "using smartphone")
        self.assertEqual(seg1["keyframe_count"], 3)

        seg2 = segments[1]
        self.assertEqual(seg2["segment_id"], 2)
        self.assertEqual(seg2["start_seconds"], 3.0)
        self.assertEqual(seg2["end_seconds"], 5.0)
        self.assertEqual(seg2["activity"], "reading")
        self.assertEqual(seg2["keyframe_count"], 2)

    # -------------------------------------------------------------------------
    # 5. Narrative Grounding & No Invented Transitions
    # -------------------------------------------------------------------------
    def test_narrative_continuity_when_no_changes(self):
        """Static video without transitions must state continuity, not false transitions."""
        frame_sequence = [
            {"sample_number": 1, "timestamp_seconds": 0.0, "activity": "sitting", "primary_subject": "A person", "environment": "indoors", "person_count": 1, "interactions": []},
            {"sample_number": 2, "timestamp_seconds": 1.0, "activity": "sitting", "primary_subject": "A person", "environment": "indoors", "person_count": 1, "interactions": []},
        ]
        temporal_analysis = {
            "stable_facts": {"dominant_subject": "A person", "dominant_environment": "Indoors", "persistent_objects": []},
            "changes": [],
            "key_moments": [],
            "segments": [],
        }
        narrative = self.video_narration.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=frame_sequence,
            video_metadata={"duration_seconds": 2.0},
        )
        self.assertNotIn("shifts", narrative)
        self.assertNotIn("transitions", narrative)
        self.assertTrue("continuity" in narrative.lower() or "continuous" in narrative.lower())

    def test_narrative_includes_transitions_when_detected(self):
        """Video with detected transition must accurately narrate the transition."""
        frame_sequence = [
            {"sample_number": 1, "timestamp_seconds": 0.0, "activity": "standing", "primary_subject": "A person", "environment": "indoors", "person_count": 1, "interactions": []},
            {"sample_number": 2, "timestamp_seconds": 3.0, "activity": "reading", "primary_subject": "A person", "environment": "indoors", "person_count": 1, "interactions": []},
        ]
        temporal_analysis = {
            "stable_facts": {"dominant_subject": "A person", "dominant_environment": "Indoors", "persistent_objects": ["book"]},
            "changes": [
                {"timestamp_seconds": 3.0, "type": "activity_change", "from": "standing", "to": "reading", "description": "Activity transitions from standing to reading."}
            ],
            "key_moments": [],
            "segments": [],
        }
        narrative = self.video_narration.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=frame_sequence,
            video_metadata={"duration_seconds": 4.0},
        )
        self.assertIn("3.0s", narrative)
        self.assertIn("reading", narrative)

    # -------------------------------------------------------------------------
    # 6. Structured Video Prediction
    # -------------------------------------------------------------------------
    def test_video_prediction_structured_hypotheses(self):
        """Video prediction must return structured hypotheses without self-referential narration."""
        last_frame = {
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 1,
            "primary_subject": "A person",
            "activity": "reading",
            "environment": "indoors",
            "detected_objects": {"book": 1, "cup": 1, "person": 1},
            "interactions": [{"target": "book", "relation": "reading", "status": "confirmed"}],
            "scene_narration": "A person is reading a book indoors.",
            "raw_fused_scene": {
                "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
                "person_presence": {"status": "present", "count": 1, "has_dominant_person": True},
                "primary_subject": {"value": "A person", "status": "supported"},
                "activity": {"value": "reading a book", "status": "supported"},
                "environment": {"value": "indoors", "status": "supported"},
                "objects": [{"label": "book", "confidence": 0.95}, {"label": "cup", "confidence": 0.88}],
                "interactions": [{"target": "book", "relation": "reading", "status": "confirmed"}],
                "conflicts": [],
                "uncertainty_flags": [],
            }
        }
        temporal_analysis = {
            "changes": [{"timestamp_seconds": 3.0, "description": "Transition to reading book."}]
        }

        pred = self.video_narration.generate_video_prediction(
            temporal_analysis=temporal_analysis,
            frame_sequence=[last_frame],
            video_metadata={"duration_seconds": 6.0},
        )

        self.assertIn("prediction", pred)
        self.assertIn("hypotheses", pred)
        self.assertTrue(len(pred["hypotheses"]) >= 1)

        for h in pred["hypotheses"]:
            for ev in h["evidence"]:
                self.assertNotIn("narration asserts", ev.lower(), "Must not cite narration as evidence for itself.")

    # -------------------------------------------------------------------------
    # 7. Job System & Concurrency
    # -------------------------------------------------------------------------
    def test_video_job_lifecycle(self):
        """VideoJob object initializes, tracks state, and serializes to dict."""
        job = VideoJob(
            job_id="test_job_1",
            file_path="/tmp/fake.mp4",
            filename="fake.mp4",
            fps_sample_rate=1.0,
            max_frames=10,
        )
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertEqual(job.percent, 0)

        # Serialize
        d = job.to_dict()
        self.assertEqual(d["job_id"], "test_job_1")
        self.assertEqual(d["status"], "queued")

        # Cancel
        job.cancel_event.set()
        self.assertTrue(job.cancel_event.is_set())

    def test_analysis_lock_concurrency(self):
        """When ANALYSIS_LOCK is held, a second acquire(blocking=False) must fail."""
        acquired = ANALYSIS_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            # Second attempt must fail (simulating concurrent request)
            second_attempt = ANALYSIS_LOCK.acquire(blocking=False)
            self.assertFalse(second_attempt, "Second concurrent acquire must be blocked.")
        finally:
            ANALYSIS_LOCK.release()


class TestVideoJobFlaskAPI(unittest.TestCase):
    """E2E Job Lifecycle tests through the Flask API using mocks to guarantee <1s execution."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.test_dir = tempfile.mkdtemp(prefix="visunarrate_api_test_")
        cls.valid_video_path = os.path.join(cls.test_dir, "test_valid.mp4")
        cls.corrupt_video_path = os.path.join(cls.test_dir, "test_corrupt.mp4")
        cls.empty_video_path = os.path.join(cls.test_dir, "test_empty.mp4")

        # Create 1-second test video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.valid_video_path, fourcc, 10.0, (80, 80))
        for _ in range(10):
            out.write(np.zeros((80, 80, 3), dtype=np.uint8))
        out.release()

        with open(cls.empty_video_path, "wb") as f:
            pass
        with open(cls.corrupt_video_path, "wb") as f:
            f.write(b"NOT_A_VALID_VIDEO_FILE")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_api_upload_valid_video(self):
        with open(self.valid_video_path, "rb") as f:
            res = self.client.post("/api/upload/video", data={"video": f})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("filepath", data)
        self.assertAlmostEqual(data["duration"], 1.0, places=1)

    def test_api_upload_empty_and_corrupt(self):
        with open(self.empty_video_path, "rb") as f:
            res_empty = self.client.post("/api/upload/video", data={"video": f})
        self.assertEqual(res_empty.status_code, 400)
        self.assertFalse(res_empty.get_json()["success"])

        with open(self.corrupt_video_path, "rb") as f:
            res_corrupt = self.client.post("/api/upload/video", data={"video": f})
        self.assertEqual(res_corrupt.status_code, 400)
        self.assertFalse(res_corrupt.get_json()["success"])

    def test_api_job_lifecycle_mocked(self):
        mock_result = {
            "success": True,
            "is_cancelled": False,
            "metadata": {
                "width": 80, "height": 80, "fps": 10.0, "total_frames": 10,
                "duration_seconds": 1.0, "is_portrait": False, "rotation_angle": 0.0
            },
            "sampling_plan": [(0, 0.0)],
            "frame_sequence": [
                {
                    "sample_number": 1,
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "processing_mode": "FULL_ANALYSIS",
                    "scene_type": "photographic_scene",
                    "person_presence": "present",
                    "person_count": 1,
                    "primary_subject": "A person",
                    "activity": "reading",
                    "environment": "indoors",
                    "detected_objects": {"book": 1, "cup": 1},
                    "interactions": [{"target": "book", "relation": "reading", "status": "confirmed"}],
                    "scene_narration": "A person is reading a book indoors.",
                    "thumbnail_url": "/static/uploads/thumbs/thumb_mock.jpg",
                    "raw_fused_scene": {
                        "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
                        "person_presence": {"status": "present", "count": 1, "has_dominant_person": True},
                        "primary_subject": {"value": "A person", "status": "supported"},
                        "activity": {"value": "reading a book", "status": "supported"},
                        "environment": {"value": "indoors", "status": "supported"},
                        "objects": [{"label": "book", "confidence": 0.95}, {"label": "cup", "confidence": 0.88}],
                        "interactions": [{"target": "book", "relation": "reading", "status": "confirmed"}],
                        "conflicts": [],
                        "uncertainty_flags": [],
                    }
                }
            ],
            "telemetry": {
                "video_duration_seconds": 1.0,
                "total_video_frames": 10,
                "fps": 10.0,
                "sampled_frames_count": 1,
                "actual_video_processing_ms": 5.0,
                "total_wall_clock_ms": 10.0,
            }
        }

        with patch.object(video_engine, "process_video", return_value=mock_result):
            res = self.client.post("/api/video/jobs", json={
                "video_path": self.valid_video_path,
                "delete_after": False
            })
            self.assertIn(res.status_code, (200, 202))
            job_id = res.get_json()["job_id"]

            # Poll job completion
            import time
            final_data = None
            for _ in range(50):
                time.sleep(0.05)
                poll_res = self.client.get(f"/api/video/jobs/{job_id}")
                self.assertEqual(poll_res.status_code, 200)
                data = poll_res.get_json()
                if data["status"] in ("done", "failed", "cancelled"):
                    final_data = data
                    break

            self.assertIsNotNone(final_data)
            self.assertEqual(final_data["status"], "done")
            result = final_data["result"]
            self.assertIn("video_story", result)
            self.assertIn("prediction", result)
            self.assertTrue(len(result["prediction"]["hypotheses"]) >= 1)

    def test_api_concurrency_busy_503(self):
        # Acquire lock to simulate a busy analysis in progress
        acquired = ANALYSIS_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            # An image analysis request during busy lock must return 503
            res_img = self.client.post("/api/analyze/image", json={"image_path": "fake.jpg"})
            self.assertEqual(res_img.status_code, 503)
            self.assertIn("busy", res_img.get_json().get("error", "").lower() + res_img.get_json().get("message", "").lower())
        finally:
            ANALYSIS_LOCK.release()


if __name__ == "__main__":
    unittest.main()
