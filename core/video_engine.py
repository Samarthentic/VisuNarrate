"""VisuNarrate - Video Processing Engine (Stage 5.1 Performance Optimization & Redundancy Filter).

Conceptual Architecture:
LOCAL VIDEO FILE
  ↓
MODEL WARM-UP & PERSISTENT VRAM PRE-TRACE (MPS Metal Shader Compilation)
  ↓
OPENCV STREAM METADATA & VALIDATION
  ↓
ADAPTIVE SAMPLING (~1 FPS / Max Frames Constraint)
  ↓
TEMPORAL REDUNDANCY EVALUATION (Visual Diff + Fast YOLO Validation)
  ├── REDUNDANT -> Inherit Stable State (0 ms / 25 ms vs 2700 ms Full Pipeline)
  ├── KEY_CHANGE -> Execute Full Multimodal Fusion (BLIP + YOLO + VQA)
  └── FULL_ANALYSIS -> Initial / Complex Scene Validation
  ↓
STRUCTURED TEMPORAL TIMELINE WITH ACCURATE TIMESTAMPS
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.image_engine import ImageEngine, _sync_mps

logger = logging.getLogger("VideoEngine")


class VideoEngine:
    """Core video analysis engine that samples frames, filters redundancy, and coordinates temporal perception."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM"}

    def __init__(self, image_engine: Optional[ImageEngine] = None):
        """Initializes VideoEngine reusing the shared ImageEngine perception backend."""
        self.image_engine = image_engine or ImageEngine()
        self.is_warmed_up = False
        self.initialization_ms = 0.0
        self.warmup_ms = 0.0
        logger.info("[VideoEngine] Initialized Video Processing Engine (Stage 5.1 Optimized).")

    def ensure_warmed_up(self) -> Dict[str, Any]:
        """Pre-warms all multimodal models (BLIP, YOLO, VQA) and compiles Metal shader traces.

        Guarantees that video frame processing executes exclusively in warm state without
        penalizing the first sampled frame with cold model loading latency.
        """
        if self.is_warmed_up:
            return {
                "initialization_ms": self.initialization_ms,
                "warmup_ms": self.warmup_ms,
                "is_warmed_up": True,
            }

        logger.info("[VideoEngine] Pre-warming multimodal perception engines on Apple Silicon MPS...")
        t_init_start = time.time()

        # 1. Force model weight loading
        if not self.image_engine.vision_model.is_loaded:
            self.image_engine.vision_model.load_model()
        if not self.image_engine.object_detector.is_loaded:
            self.image_engine.object_detector.load_model()
        if not self.image_engine.vqa_model.is_loaded:
            self.image_engine.vqa_model.load_model()

        _sync_mps()
        self.initialization_ms = round((time.time() - t_init_start) * 1000, 2)

        # 2. Execute dummy synthetic frame to trace MPS Metal shader graphs
        t_warm_start = time.time()
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        _ = self.image_engine.analyze_scene(dummy_img)
        _sync_mps()
        self.warmup_ms = round((time.time() - t_warm_start) * 1000, 2)

        self.is_warmed_up = True
        logger.info(
            f"[VideoEngine] Model Warm-up Complete: Init={self.initialization_ms}ms, "
            f"JIT Tracing={self.warmup_ms}ms. Video engines marked READY."
        )

        return {
            "initialization_ms": self.initialization_ms,
            "warmup_ms": self.warmup_ms,
            "is_warmed_up": True,
        }

    def inspect_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Inspects and validates a local video file, extracting hardware properties.

        Args:
            video_path: Absolute or relative path to the local video file.

        Returns:
            Dictionary containing fps, duration, frame_count, resolution, and format.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        ext = Path(video_path).suffix
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported video format '{ext}'. Supported: {list(self.SUPPORTED_EXTENSIONS)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

            if total_frames <= 0 or fps <= 0:
                count = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    count += 1
                total_frames = count
                fps = fps if fps > 0 else 24.0

            duration_seconds = round(total_frames / fps, 2) if fps > 0 else 0.0

            metadata = {
                "file_path": video_path,
                "filename": Path(video_path).name,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": duration_seconds,
                "fourcc": fourcc_str,
            }
            logger.info(
                f"[VideoEngine] Video Metadata: {metadata['filename']} "
                f"({metadata['resolution']}, {metadata['fps']} FPS, {metadata['duration_seconds']}s, {metadata['total_frames']} frames)"
            )
            return metadata

        finally:
            cap.release()

    def calculate_sampling_plan(
        self,
        total_frames: int,
        fps: float,
        target_fps_sample: float = 1.0,
        max_frames: int = 15,
        min_interval_seconds: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """Determines optimal frame indices and timestamps to sample."""
        if total_frames <= 0 or fps <= 0:
            return [(0, 0.0)]

        step_frames = max(1, int(round(fps / target_fps_sample)))
        min_step = max(1, int(round(fps * min_interval_seconds)))
        step_frames = max(step_frames, min_step)

        candidate_indices = list(range(0, total_frames, step_frames))

        if len(candidate_indices) > max_frames:
            stride = len(candidate_indices) / float(max_frames)
            candidate_indices = [candidate_indices[int(i * stride)] for i in range(max_frames)]

        if len(candidate_indices) > 1 and (total_frames - 1) not in candidate_indices:
            candidate_indices[-1] = total_frames - 1

        sampling_plan = [(idx, round(idx / fps, 2)) for idx in candidate_indices]
        logger.info(f"[VideoEngine] Sampling Plan: {len(sampling_plan)} frames selected out of {total_frames} total frames.")
        return sampling_plan

    def compute_visual_change_score(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
        """Computes lightweight normalized pixel intensity delta between two frames (0.0 to 1.0)."""
        if prev_bgr is None or curr_bgr is None:
            return 1.0
        try:
            gray_prev = cv2.cvtColor(cv2.resize(prev_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(cv2.resize(curr_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(gray_curr.astype(np.float32) - gray_prev.astype(np.float32))) / 255.0
            return float(diff)
        except Exception:
            return 1.0

    def check_semantic_equivalence(self, prev_rec: Dict[str, Any], curr_analysis: Dict[str, Any]) -> bool:
        """Evaluates whether current frame analysis matches previous validated structured state."""
        if not prev_rec:
            return False

        # Compare person counts
        prev_p_count = prev_rec.get("person_count", 0)
        curr_p_count = curr_analysis.get("person_presence", {}).get("count", 0)
        if prev_p_count != curr_p_count:
            return False

        # Compare primary activity
        prev_act = prev_rec.get("activity", "").strip().lower()
        curr_act = curr_analysis.get("fused_scene", {}).get("activity", {}).get("value", "").strip().lower()
        if prev_act != curr_act and curr_act and prev_act:
            return False

        # Compare top interaction target
        prev_inter = prev_rec.get("interactions", [{}])[0].get("target", "").strip().lower() if prev_rec.get("interactions") else ""
        curr_inter = curr_analysis.get("fused_scene", {}).get("interactions", [{}])[0].get("target", "").strip().lower() if curr_analysis.get("fused_scene", {}).get("interactions") else ""
        if prev_inter != curr_inter and (prev_inter or curr_inter):
            return False

        return True

    def process_video(
        self,
        video_path: str,
        target_fps_sample: float = 1.0,
        max_frames: int = 15,
        visual_diff_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Extracts, filters redundancy, and analyzes representative frames from a video.

        Args:
            video_path: Path to the local video file.
            target_fps_sample: Target sampling rate (frames per second).
            max_frames: Maximum number of frames to sample and analyze.
            visual_diff_threshold: Minimum pixel delta required to trigger full analysis candidate.

        Returns:
            Structured dictionary with metadata, frame sequence representations, and timing telemetry.
        """
        wall_clock_start = time.time()

        # 1. Ensure Model Pre-Warmed
        warm_status = self.ensure_warmed_up()

        # 2. Inspect Video Properties
        metadata = self.inspect_video_metadata(video_path)
        total_frames = metadata["total_frames"]
        fps = metadata["fps"]

        sampling_plan = self.calculate_sampling_plan(
            total_frames=total_frames,
            fps=fps,
            target_fps_sample=target_fps_sample,
            max_frames=max_frames,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file for frame extraction: {video_path}")

        # 3. Start Measuring Actual Video Processing Latency (Warm)
        logger.info(f"[VideoEngine] Starting Warm Video Analysis for: {metadata['filename']}")
        actual_video_start = time.time()
        _sync_mps()

        frame_records: List[Dict[str, Any]] = []
        full_analysis_latencies_ms: List[float] = []
        all_frame_latencies_ms: List[float] = []

        prev_frame_bgr: Optional[np.ndarray] = None
        prev_validated_record: Optional[Dict[str, Any]] = None

        full_count = 0
        redundant_count = 0
        key_change_count = 0

        try:
            for sample_idx, (frame_num, timestamp) in enumerate(sampling_plan, 1):
                t_frame_start = time.time()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame_bgr = cap.read()

                if not ret or frame_bgr is None:
                    logger.warning(f"[VideoEngine] Failed to read frame {frame_num} at {timestamp}s. Skipping.")
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                # Frame 1 is always FULL_ANALYSIS
                if sample_idx == 1 or prev_validated_record is None:
                    logger.info(f"[VideoEngine] Frame #{sample_idx} ({timestamp}s) -> FULL_ANALYSIS (Initial Frame)")
                    frame_analysis = self.image_engine.analyze_scene(pil_frame)
                    _sync_mps()
                    frame_dur_ms = round((time.time() - t_frame_start) * 1000, 2)
                    full_analysis_latencies_ms.append(frame_dur_ms)
                    all_frame_latencies_ms.append(frame_dur_ms)

                    rec = self._build_frame_record(
                        sample_idx=sample_idx,
                        frame_num=frame_num,
                        timestamp=timestamp,
                        processing_mode="FULL_ANALYSIS",
                        visual_change_score=1.0,
                        semantic_change=False,
                        inherited=False,
                        source_timestamp=timestamp,
                        frame_analysis=frame_analysis,
                        latency_ms=frame_dur_ms,
                    )
                    frame_records.append(rec)
                    prev_validated_record = rec
                    prev_frame_bgr = frame_bgr
                    full_count += 1
                    continue

                # Compute Visual Change Score against previous frame
                visual_diff = self.compute_visual_change_score(prev_frame_bgr, frame_bgr)

                # Heuristic 1: If visual difference is negligible, run fast YOLO verification
                if visual_diff < visual_diff_threshold:
                    # Run lightweight detector check (~25ms on MPS)
                    t_det_start = time.time()
                    detector_res = self.image_engine.object_detector.detect_objects(pil_frame)
                    _sync_mps()
                    det_dur_ms = round((time.time() - t_det_start) * 1000, 2)

                    curr_yolo_counts = detector_res.get("counts", {})
                    prev_yolo_counts = prev_validated_record.get("detected_objects", {})
                    curr_p_count = curr_yolo_counts.get("person", 0)
                    prev_p_count = prev_validated_record.get("person_count", 0)

                    # If object counts and person presence match, frame is verified REDUNDANT
                    if curr_yolo_counts == prev_yolo_counts and curr_p_count == prev_p_count:
                        total_red_lat_ms = round((time.time() - t_frame_start) * 1000, 2)
                        all_frame_latencies_ms.append(total_red_lat_ms)
                        logger.info(
                            f"[VideoEngine] Frame #{sample_idx} ({timestamp}s) -> REDUNDANT "
                            f"(Visual Diff: {visual_diff:.3f} < {visual_diff_threshold}, Object State Identical, Inherited from {prev_validated_record['timestamp_seconds']}s in {total_red_lat_ms}ms)"
                        )

                        rec = self._build_inherited_record(
                            sample_idx=sample_idx,
                            frame_num=frame_num,
                            timestamp=timestamp,
                            visual_change_score=visual_diff,
                            source_record=prev_validated_record,
                            latency_ms=total_red_lat_ms,
                        )
                        frame_records.append(rec)
                        prev_frame_bgr = frame_bgr
                        redundant_count += 1
                        continue

                # If visual difference >= threshold or YOLO detects novel objects/person shift -> Run Full Pipeline
                logger.info(f"[VideoEngine] Frame #{sample_idx} ({timestamp}s) -> Evaluating Candidate Change (Visual Diff: {visual_diff:.3f})...")
                frame_analysis = self.image_engine.analyze_scene(pil_frame)
                _sync_mps()
                frame_dur_ms = round((time.time() - t_frame_start) * 1000, 2)
                full_analysis_latencies_ms.append(frame_dur_ms)
                all_frame_latencies_ms.append(frame_dur_ms)

                is_equiv = self.check_semantic_equivalence(prev_validated_record, frame_analysis)
                if not is_equiv:
                    mode = "KEY_CHANGE"
                    sem_change = True
                    key_change_count += 1
                    logger.info(f"[VideoEngine] Frame #{sample_idx} ({timestamp}s) -> KEY_CHANGE Confirmed (State Transition Detected)")
                else:
                    mode = "FULL_ANALYSIS"
                    sem_change = False
                    full_count += 1

                rec = self._build_frame_record(
                    sample_idx=sample_idx,
                    frame_num=frame_num,
                    timestamp=timestamp,
                    processing_mode=mode,
                    visual_change_score=visual_diff,
                    semantic_change=sem_change,
                    inherited=False,
                    source_timestamp=timestamp,
                    frame_analysis=frame_analysis,
                    latency_ms=frame_dur_ms,
                )
                frame_records.append(rec)
                prev_validated_record = rec
                prev_frame_bgr = frame_bgr

        finally:
            cap.release()

        _sync_mps()
        actual_video_ms = round((time.time() - actual_video_start) * 1000, 2)
        total_wall_clock_ms = round((time.time() - wall_clock_start) * 1000, 2)
        avg_full_ms = round(sum(full_analysis_latencies_ms) / len(full_analysis_latencies_ms), 2) if full_analysis_latencies_ms else 0.0
        avg_all_ms = round(sum(all_frame_latencies_ms) / len(all_frame_latencies_ms), 2) if all_frame_latencies_ms else 0.0

        telemetry = {
            "video_duration_seconds": metadata["duration_seconds"],
            "total_video_frames": metadata["total_frames"],
            "fps": metadata["fps"],
            "sampled_frames_count": len(frame_records),
            "target_fps_sample": target_fps_sample,
            "full_analyses_count": full_count,
            "redundant_frames_count": redundant_count,
            "key_changes_count": key_change_count,
            "inherited_frames_count": redundant_count,
            "model_initialization_ms": warm_status["initialization_ms"],
            "warmup_ms": warm_status["warmup_ms"],
            "actual_video_processing_ms": actual_video_ms,
            "average_frame_latency_full_ms": avg_full_ms,
            "average_frame_latency_all_ms": avg_all_ms,
            "total_wall_clock_ms": total_wall_clock_ms,
            "min_frame_latency_ms": min(all_frame_latencies_ms) if all_frame_latencies_ms else 0.0,
            "max_frame_latency_ms": max(all_frame_latencies_ms) if all_frame_latencies_ms else 0.0,
        }

        logger.info(
            f"[VideoEngine] Video Processing Complete: {len(frame_records)} frames "
            f"(Full: {full_count}, Redundant: {redundant_count}, Key Changes: {key_change_count}) in {actual_video_ms}ms (Actual Warm)."
        )

        return {
            "success": True,
            "metadata": metadata,
            "sampling_plan": sampling_plan,
            "frame_sequence": frame_records,
            "telemetry": telemetry,
        }

    def _build_frame_record(
        self,
        sample_idx: int,
        frame_num: int,
        timestamp: float,
        processing_mode: str,
        visual_change_score: float,
        semantic_change: bool,
        inherited: bool,
        source_timestamp: float,
        frame_analysis: Dict[str, Any],
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Constructs standardized frame record dictionary from full analysis."""
        fused_scene = frame_analysis.get("fused_scene", {})
        scene_type_info = frame_analysis.get("scene_type", {})
        person_presence = frame_analysis.get("person_presence", {})
        object_detection = frame_analysis.get("object_detection", {})
        predictive_narration = frame_analysis.get("predictive_narration", {})

        return {
            "sample_number": sample_idx,
            "frame_index": frame_num,
            "timestamp_seconds": timestamp,
            "processing_mode": processing_mode,
            "visual_change_score": round(visual_change_score, 4),
            "semantic_change": semantic_change,
            "inherited": inherited,
            "source_frame_timestamp": source_timestamp,
            "scene_type": scene_type_info.get("scene_type", "photographic_scene"),
            "person_presence": person_presence.get("status", "unknown"),
            "person_count": person_presence.get("count", 0),
            "primary_subject": fused_scene.get("primary_subject", {}).get("value", "Unspecified"),
            "activity": fused_scene.get("activity", {}).get("value", "Static pose"),
            "environment": fused_scene.get("environment", {}).get("value", "General Environment"),
            "detected_objects": object_detection.get("counts", {}),
            "total_objects_count": object_detection.get("total_detections", 0),
            "interactions": fused_scene.get("interactions", []),
            "conflicts": frame_analysis.get("conflicts", []),
            "uncertainty_flags": frame_analysis.get("uncertainty_flags", []),
            "scene_narration": frame_analysis.get("scene_narration", ""),
            "predictive_narration": predictive_narration.get("prediction", ""),
            "prediction_status": frame_analysis.get("prediction_status", "unknown"),
            "prediction_type": frame_analysis.get("prediction_type", "unknown"),
            "frame_latency_ms": latency_ms,
        }

    def _build_inherited_record(
        self,
        sample_idx: int,
        frame_num: int,
        timestamp: float,
        visual_change_score: float,
        source_record: Dict[str, Any],
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Constructs standardized frame record inheriting persistent state from prior frame."""
        return {
            "sample_number": sample_idx,
            "frame_index": frame_num,
            "timestamp_seconds": timestamp,
            "processing_mode": "REDUNDANT",
            "visual_change_score": round(visual_change_score, 4),
            "semantic_change": False,
            "inherited": True,
            "source_frame_timestamp": source_record.get("timestamp_seconds", 0.0),
            "scene_type": source_record.get("scene_type", "photographic_scene"),
            "person_presence": source_record.get("person_presence", "unknown"),
            "person_count": source_record.get("person_count", 0),
            "primary_subject": source_record.get("primary_subject", "Unspecified"),
            "activity": source_record.get("activity", "Static pose"),
            "environment": source_record.get("environment", "General Environment"),
            "detected_objects": source_record.get("detected_objects", {}),
            "total_objects_count": source_record.get("total_objects_count", 0),
            "interactions": source_record.get("interactions", []),
            "conflicts": source_record.get("conflicts", []),
            "uncertainty_flags": source_record.get("uncertainty_flags", []),
            "scene_narration": source_record.get("scene_narration", ""),
            "predictive_narration": source_record.get("predictive_narration", ""),
            "prediction_status": source_record.get("prediction_status", "unknown"),
            "prediction_type": source_record.get("prediction_type", "unknown"),
            "frame_latency_ms": latency_ms,
        }
