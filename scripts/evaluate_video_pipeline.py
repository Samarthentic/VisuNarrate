"""VisuNarrate - Video Storytelling Pipeline Evaluation Harness (Stage 5.0).

Scientific benchmark script for evaluating the local video storytelling architecture:
1. Video Metadata & OpenCV Stream Inspection
2. Adaptive Frame Sampling (~1 FPS)
3. Shared ImageEngine Execution across Sampled Frames
4. Temporal State Aggregation & Transition Detection
5. Non-Repetitive Chronological Video Narration Synthesis
6. Telemetry & Timing Metrics
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VideoEvaluator")

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from core.image_engine import ImageEngine
from core.video_engine import VideoEngine
from services.temporal_service import TemporalService
from services.video_narration_service import VideoNarrationService


def evaluate_video():
    video_path = base_dir / "data" / "test_videos" / "test_transition_activity.mp4"
    if not video_path.exists():
        logger.error(f"Test video not found at: {video_path}")
        sys.exit(1)

    results_dir = base_dir / "data" / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("VISUNARRATE VIDEO STORYTELLING & REDUNDANCY BENCHMARK (STAGE 5.1 OPTIMIZED)")
    print("=" * 120)

    t_engine_init = time.time()
    image_engine = ImageEngine()
    video_engine = VideoEngine(image_engine=image_engine)
    temporal_service = TemporalService()
    video_narration_service = VideoNarrationService()
    engine_init_ms = round((time.time() - t_engine_init) * 1000, 2)
    print(f"Engine Wrapper Allocation: {engine_init_ms} ms")

    # Execute Full Video Pipeline (Includes ensure_warmed_up + actual processing)
    t_start = time.time()
    video_result = video_engine.process_video(
        video_path=str(video_path),
        target_fps_sample=1.0,
        max_frames=10,
    )

    t_temp_start = time.time()
    temporal_analysis = temporal_service.detect_temporal_transitions(video_result["frame_sequence"])
    temporal_ms = round((time.time() - t_temp_start) * 1000, 2)

    t_narr_start = time.time()
    video_narration = video_narration_service.synthesize_video_narrative(
        temporal_analysis=temporal_analysis,
        frame_sequence=video_result["frame_sequence"],
        video_metadata=video_result["metadata"],
    )
    narr_ms = round((time.time() - t_narr_start) * 1000, 2)

    total_pipeline_ms = round((time.time() - t_start) * 1000, 2)

    meta = video_result["metadata"]
    telemetry = video_result["telemetry"]
    frames = video_result["frame_sequence"]
    changes = temporal_analysis["changes"]
    key_moments = temporal_analysis["key_moments"]

    # Console Summary Output
    print("-" * 120)
    print("MODEL WARM-UP & INITIALIZATION TELEMETRY:")
    print(f"  • Model Weight Loading:      {telemetry['model_initialization_ms']} ms")
    print(f"  • Shader JIT Warm-up Trace:  {telemetry['warmup_ms']} ms")
    print(f"  • Actual Video Processing:   {telemetry['actual_video_processing_ms']} ms ({round(telemetry['actual_video_processing_ms']/1000, 2)}s)")
    print(f"  • Total Wall-Clock Latency:  {telemetry['total_wall_clock_ms']} ms")
    print("-" * 120)
    print("FRAME REDUNDANCY FILTERING & SAMPLING TELEMETRY:")
    print(f"  • Total Sampled Frames:      {len(frames)} frames")
    print(f"  • Full Multimodal Analyses:  {telemetry['full_analyses_count']} frame(s)")
    print(f"  • Redundant Frames Filtered: {telemetry['redundant_frames_count']} frame(s) ({round(telemetry['redundant_frames_count']/len(frames)*100, 1)}% compute saved)")
    print(f"  • Key Changes Triggered:     {telemetry['key_changes_count']} frame(s)")
    print(f"  • Avg Latency (Full Passes): {telemetry['average_frame_latency_full_ms']} ms / frame")
    print(f"  • Avg Latency (All Sampled): {telemetry['average_frame_latency_all_ms']} ms / frame")
    print(f"  • Temporal Aggregation Time: {temporal_ms} ms")
    print(f"  • Narrative Synthesis Time:  {narr_ms} ms")
    print("-" * 120)
    print("SAMPLED FRAME SEQUENCE DETAILS:")
    print(f"{'Sample':<7} | {'Time (s)':<9} | {'Processing Mode':<16} | {'Visual Diff':<12} | {'Activity':<18} | {'Latency'}")
    print("-" * 120)
    for f in frames:
        s_num = f"#{f['sample_number']}"
        ts = f"{f['timestamp_seconds']}s"
        mode = f["processing_mode"]
        diff = f"{f.get('visual_change_score', 0.0):.4f}"
        act = f["activity"][:16]
        lat = f"{f['frame_latency_ms']}ms"
        print(f"{s_num:<7} | {ts:<9} | {mode:<16} | {diff:<12} | {act:<18} | {lat}")
    print("-" * 120)
    print("DETECTED TEMPORAL TRANSITIONS & STATE CHANGES:")
    if changes:
        for c in changes:
            print(f"  • [{c['timestamp_seconds']}s] ({c['type']}): {c['description']}")
    else:
        print("  • No discrete state transitions detected (Continuous Scene).")
    print("-" * 120)
    print("SYNTHESIZED VIDEO NARRATIVE:")
    print(f"  \"{video_narration}\"")
    print("-" * 120)
    print("STABLE / PERSISTENT CONTEXT:")
    print(f"  • Dominant Subject:          {temporal_analysis['stable_facts']['dominant_subject']}")
    print(f"  • Dominant Environment:      {temporal_analysis['stable_facts']['dominant_environment']}")
    print(f"  • Persistent Objects:        {', '.join(temporal_analysis['stable_facts']['persistent_objects'])}")
    print("=" * 120 + "\n")

    # Export full JSON
    export_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": meta,
        "telemetry": {
            **telemetry,
            "temporal_ms": temporal_ms,
            "narration_ms": narr_ms,
            "total_pipeline_ms": total_pipeline_ms,
        },
        "temporal_analysis": temporal_analysis,
        "video_narration": video_narration,
        "frame_sequence": frames,
    }
    json_path = results_dir / "video_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Video evaluation results saved to: {json_path}")

if __name__ == "__main__":
    evaluate_video()
