"""VisuNarrate - Live Camera Stream & Session Engine (Stage 6 Production).

Provides:
- In-memory decoding of live camera JPEG frames (strictly zero disk writes / no disk caching).
- Pixel-diff based temporal redundancy detection & fast state inheritance (reduces MPS thrashing).
- LiveSessionManager: ring-buffered history (max 60 frames), 15-minute inactivity auto-cleanup.
- Session summary synthesis via TemporalService and VideoNarrationService.
"""

import io
import time
import uuid
import base64
import logging
import threading
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
import psutil

from core.image_engine import ImageEngine, _sync_mps
from services.temporal_service import TemporalService
from services.video_narration_service import VideoNarrationService

logger = logging.getLogger("LiveEngine")

MAX_LIVE_SESSION_HISTORY: int = 60
SESSION_EXPIRY_SECONDS: float = 900.0  # 15 minutes of inactivity
REDUNDANCY_PIXEL_DIFF_THRESHOLD: float = 0.04  # 4% mean absolute difference in 64x64 thumbnail
MAX_LIVE_FRAME_DIMENSION: int = 640


class LiveSession:
    """Encapsulates active live camera stream state stored strictly in RAM."""

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.created_at: float = time.time()
        self.last_activity: float = time.time()
        self.history: deque = deque(maxlen=MAX_LIVE_SESSION_HISTORY)
        self.prev_small_gray: Optional[np.ndarray] = None
        self.prev_record: Optional[Dict[str, Any]] = None
        self.frame_counter: int = 0
        self.full_analyses_count: int = 0
        self.redundant_frames_count: int = 0


class LiveSessionManager:
    """Thread-safe manager for live camera streaming sessions stored strictly in RAM."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, LiveSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> LiveSession:
        with self._lock:
            self._cleanup_expired_locked()
            if not session_id:
                session_id = f"live_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            if session_id not in self._sessions:
                self._sessions[session_id] = LiveSession(session_id)
            session = self._sessions[session_id]
            session.last_activity = time.time()
            return session

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        with self._lock:
            self._cleanup_expired_locked()
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"[LiveSessionManager] Session deleted: {session_id}")
                return True
            return False

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _cleanup_expired_locked(self) -> int:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_activity) > SESSION_EXPIRY_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"[LiveSessionManager] Evicted {len(expired)} expired live sessions.")
        return len(expired)


class LiveEngine:
    """Processes in-memory live camera frames with temporal redundancy checks and session aggregation."""

    def __init__(
        self,
        image_engine: ImageEngine,
        temporal_service: Optional[TemporalService] = None,
        video_narration_service: Optional[VideoNarrationService] = None,
        session_manager: Optional[LiveSessionManager] = None,
    ):
        self.image_engine = image_engine
        self.temporal_service = temporal_service or TemporalService()
        self.video_narration_service = video_narration_service or VideoNarrationService()
        self.session_manager = session_manager or LiveSessionManager()

    def decode_frame(self, frame_data: Any) -> np.ndarray:
        """Decodes raw bytes or base64 data to a BGR numpy image in RAM.

        Raises:
            ValueError: If frame decoding fails or data is invalid.
        """
        if isinstance(frame_data, str):
            # Check for data URI prefix e.g. "data:image/jpeg;base64,..."
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]
            try:
                frame_bytes = base64.b64decode(frame_data)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 frame: {str(e)}")
        elif isinstance(frame_data, (bytes, bytearray)):
            frame_bytes = bytes(frame_data)
        else:
            raise ValueError(f"Unsupported frame data type: {type(frame_data)}")

        if not frame_bytes:
            raise ValueError("Empty frame data received.")

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("OpenCV failed to decode frame in memory.")

        # Constrain dimensions to maximum 640px for memory & speed efficiency on Apple Silicon
        h, w = frame_bgr.shape[:2]
        if max(w, h) > MAX_LIVE_FRAME_DIMENSION:
            scale = MAX_LIVE_FRAME_DIMENSION / float(max(w, h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame_bgr

    def process_frame(
        self,
        frame_data: Any,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes an incoming camera frame in RAM, applying redundancy gating and recording snapshot."""
        t_start = time.time()
        frame_bgr = self.decode_frame(frame_data)
        decode_ms = round((time.time() - t_start) * 1000, 2)

        session = self.session_manager.get_or_create_session(session_id)
        session.frame_counter += 1
        frame_idx = session.frame_counter
        timestamp_s = round(time.time() - session.created_at, 2)

        # 64x64 grayscale representation for instant pixel difference
        small_gray = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (64, 64))

        diff_ratio = 1.0
        is_redundant = False

        if session.prev_small_gray is not None and session.prev_record is not None:
            abs_diff = np.abs(small_gray.astype(np.float32) - session.prev_small_gray.astype(np.float32))
            diff_ratio = float(np.mean(abs_diff)) / 255.0
            if diff_ratio < REDUNDANCY_PIXEL_DIFF_THRESHOLD:
                is_redundant = True

        if is_redundant and session.prev_record is not None:
            # REDUNDANT FRAME: Reuse previous analyzed state, fast return
            session.redundant_frames_count += 1
            prev = session.prev_record

            elapsed_ms = round((time.time() - t_start) * 1000, 2)
            technical = {
                "device": self.image_engine.vision_model.device,
                "timings_ms": {
                    "decode_ms": decode_ms,
                    "diff_check_ms": elapsed_ms - decode_ms,
                    "total_ms": elapsed_ms,
                },
                "is_redundant_frame": True,
                "frame_diff_ratio": round(diff_ratio, 4),
                "reused_from_snapshot": prev["snapshot_id"],
                "models_used": [
                    {"name": "Visual Diff & State Cache", "architecture": "Fast In-Memory Cache", "parameters": "0M"}
                ],
                "memory_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
                "training": {
                    "epochs": 0,
                    "fine_tuning": "none",
                    "statement": "Pretrained weights, inference only. Training epochs run in this project: 0. Fine-tuning: none."
                }
            }

            snapshot_record = {
                "snapshot_id": f"snap_{frame_idx}",
                "timestamp_seconds": timestamp_s,
                "wall_time": time.strftime("%H:%M:%S"),
                "processing_mode": "REDUNDANT",
                "visual_diff_ratio": round(diff_ratio, 4),
                "narration": prev["narration"],
                "prediction": prev["prediction"],
                "prediction_status": prev.get("prediction_status", "supported_prediction"),
                "hypotheses": prev.get("hypotheses", []),
                "detected_objects": prev.get("detected_objects", {}),
                "primary_subject": prev.get("primary_subject", "Person"),
                "activity": prev.get("activity", "Static pose"),
                "environment": prev.get("environment", "Indoors"),
                "scene_type": prev.get("scene_type", "photographic_scene"),
                "technical": technical,
                "latency_ms": elapsed_ms,
            }

        else:
            # FULL ANALYSIS FRAME: Run full evidence fusion on in-memory PIL image
            session.full_analyses_count += 1
            session.prev_small_gray = small_gray

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            analysis = self.image_engine.analyze_scene(pil_image)
            elapsed_ms = round((time.time() - t_start) * 1000, 2)

            fused_scene = analysis.get("fused_scene", {})
            pred_data = analysis.get("predictive_narration", {})
            pred_text = pred_data.get("prediction", "")
            hypotheses = pred_data.get("hypotheses", [])
            obj_det = analysis.get("object_detection", {})
            scene_type_data = analysis.get("scene_type", {})

            technical = analysis.get("technical", {})
            if "timings_ms" in technical:
                technical["timings_ms"]["decode_ms"] = decode_ms
                technical["timings_ms"]["total_ms"] = elapsed_ms
            technical["is_redundant_frame"] = False
            technical["frame_diff_ratio"] = round(diff_ratio, 4)

            snapshot_record = {
                "snapshot_id": f"snap_{frame_idx}",
                "timestamp_seconds": timestamp_s,
                "wall_time": time.strftime("%H:%M:%S"),
                "processing_mode": "FULL_ANALYSIS",
                "visual_diff_ratio": round(diff_ratio, 4),
                "narration": analysis.get("scene_narration", ""),
                "prediction": pred_text,
                "prediction_status": analysis.get("prediction_status", "supported_prediction"),
                "hypotheses": hypotheses,
                "detected_objects": obj_det.get("counts", {}),
                "primary_subject": fused_scene.get("primary_subject", {}).get("value", "Person"),
                "activity": fused_scene.get("activity", {}).get("value", "Static pose"),
                "environment": fused_scene.get("environment", {}).get("value", "Indoors"),
                "scene_type": scene_type_data.get("scene_type", "photographic_scene"),
                "fused_scene": fused_scene,
                "technical": technical,
                "latency_ms": elapsed_ms,
            }
            session.prev_record = snapshot_record

        # Append to ring-buffered session history in RAM
        session.history.append(snapshot_record)

        return {
            "success": True,
            "session_id": session.session_id,
            "snapshot": snapshot_record,
            "history_length": len(session.history),
            "stats": {
                "total_frames": session.frame_counter,
                "full_analyses": session.full_analyses_count,
                "redundant_frames": session.redundant_frames_count,
                "uptime_seconds": round(time.time() - session.created_at, 1),
            },
        }

    def generate_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Synthesizes an overall coherent story and prediction from the live session history."""
        session = self.session_manager.get_session(session_id)
        if not session or len(session.history) == 0:
            raise ValueError(f"No active history found for live session '{session_id}'.")

        t_summary_start = time.time()
        # Build frame sequence compliant with TemporalService
        frame_sequence = []
        for idx, snap in enumerate(session.history, 1):
            frame_sequence.append({
                "sample_number": idx,
                "frame_index": idx,
                "timestamp_seconds": snap["timestamp_seconds"],
                "processing_mode": snap["processing_mode"],
                "visual_change_score": snap["visual_diff_ratio"],
                "semantic_change": False,
                "inherited": (snap["processing_mode"] == "REDUNDANT"),
                "source_frame_timestamp": snap["timestamp_seconds"],
                "thumbnail_url": None,
                "scene_type": snap.get("scene_type", "photographic_scene"),
                "person_presence": "present" if "person" in snap.get("detected_objects", {}) else "not_detected",
                "person_count": snap.get("detected_objects", {}).get("person", 0),
                "primary_subject": snap.get("primary_subject", "Person"),
                "activity": snap.get("activity", "Activity"),
                "environment": snap.get("environment", "Indoors"),
                "detected_objects": snap.get("detected_objects", {}),
                "total_objects_count": sum(snap.get("detected_objects", {}).values()),
                "interactions": [],
                "conflicts": [],
                "uncertainty_flags": [],
            })

        duration = max(1.0, frame_sequence[-1]["timestamp_seconds"])
        meta = {
            "duration_seconds": duration,
            "filename": "Live Camera Stream",
            "fps": 1.0,
            "total_frames": len(frame_sequence),
        }

        temporal_analysis = self.temporal_service.detect_temporal_transitions(
            frame_sequence=frame_sequence,
            video_duration=duration,
        )

        story = self.video_narration_service.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=frame_sequence,
            video_metadata=meta,
        )

        prediction = self.video_narration_service.generate_video_prediction(
            temporal_analysis=temporal_analysis,
            frame_sequence=frame_sequence,
            video_metadata=meta,
        )

        summary_ms = round((time.time() - t_summary_start) * 1000, 2)

        technical = {
            "device": self.image_engine.vision_model.device,
            "timings_ms": {
                "summary_synthesis_ms": summary_ms,
                "total_ms": summary_ms,
            },
            "keyframes_count": len(frame_sequence),
            "full_analyses_count": session.full_analyses_count,
            "redundant_frames_count": session.redundant_frames_count,
            "models_used": [
                {
                    "name": "BLIP Base Image Captioning",
                    "architecture": "BlipForConditionalGeneration",
                    "parameters": "224.0M",
                    "size_mb": 855.1,
                },
                {
                    "name": "YOLOv8n",
                    "architecture": "YOLOv8n Detection",
                    "parameters": "3.2M",
                    "size_mb": 6.3,
                },
                {
                    "name": "BLIP VQA Base",
                    "architecture": "BlipForQuestionAnswering",
                    "parameters": "361.2M",
                    "size_mb": 1468.4,
                },
            ],
            "memory_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
            "training": {
                "epochs": 0,
                "fine_tuning": "none",
                "statement": "Pretrained weights, inference only. Training epochs run in this project: 0. Fine-tuning: none."
            }
        }

        return {
            "success": True,
            "session_id": session_id,
            "live_story": story,
            "live_prediction": prediction,
            "timeline_segments": temporal_analysis.get("segments", []),
            "changes": temporal_analysis.get("changes", []),
            "snapshots_analyzed": len(frame_sequence),
            "duration_seconds": duration,
            "technical": technical,
        }

    def delete_session(self, session_id: str) -> bool:
        """Terminates session and purges state from RAM."""
        return self.session_manager.delete_session(session_id)
