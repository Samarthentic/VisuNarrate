"""VisuNarrate - Video Pipeline Quantitative Evaluator.

Evaluates the temporal video understanding and storytelling pipeline:
1. Temporal Transition Detection: Precision, Recall, F1 against ground-truth state changes.
2. Timeline Segment Activity Matching: Segment classification accuracy against annotated intervals.
3. Narrative Coherence & Semantic Alignment: BLEU & ROUGE-L against reference story.
4. Redundancy Filtering Efficiency: Filter ratio, compute saved percentage, throughput.
5. Hardware & Processing Telemetry: Per-frame latency breakdown and total wall-clock duration.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from evaluation.dataset_loader import EvaluationDatasetLoader
from evaluation.metrics import (
    evaluate_video_transitions,
    calculate_bleu,
    calculate_rouge_l,
)

logger = logging.getLogger("VideoEvaluator")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class VideoPipelineQuantitativeEvaluator:
    """Evaluates video processing output against ground-truth temporal annotations."""

    def __init__(
        self,
        results_dir: str = "data/evaluation_results",
        dataset_loader: Optional[EvaluationDatasetLoader] = None,
        video_engine=None,
    ):
        self.results_dir = Path(PROJECT_ROOT) / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_loader = dataset_loader or EvaluationDatasetLoader()
        self.video_engine = video_engine

    def evaluate_video_result(
        self,
        video_result: Dict[str, Any],
        video_filename: str = "test_transition_activity.mp4",
        tolerance_sec: float = 1.5,
    ) -> Dict[str, Any]:
        """Evaluates video pipeline output against ground-truth annotation."""
        gt = self.dataset_loader.get_video_annotation(video_filename)
        is_annotated = gt is not None

        # Extract predicted transitions
        temporal_analysis = video_result.get("temporal_analysis", {})
        changes = temporal_analysis.get("changes", [])
        pred_timestamps = [float(c.get("timestamp_seconds", 0.0)) for c in changes]

        # Extract segments
        segments = temporal_analysis.get("segments", [])
        video_narration = video_result.get("video_narration", "")
        telemetry = video_result.get("telemetry", {})
        frames = video_result.get("frame_sequence", [])

        transition_eval = None
        segment_eval = None
        narrative_eval = None

        if is_annotated:
            # 1. Evaluate temporal transitions
            gt_timestamps = [float(t) for t in gt.get("ground_truth_transition_timestamps", [])]
            transition_eval = evaluate_video_transitions(
                predicted_transitions=pred_timestamps,
                ground_truth_transitions=gt_timestamps,
                tolerance_sec=tolerance_sec,
            )

            # 2. Evaluate timeline segment activities
            gt_segments = gt.get("ground_truth_segments", [])
            segment_matches = 0
            evaluated_segments = []

            for s_idx, s in enumerate(segments):
                s_activity = s.get("activity", "").lower()
                matched_gt = None
                s_start = s.get("start_seconds", 0.0)
                s_end = s.get("end_seconds", 0.0)

                for gts in gt_segments:
                    # Check overlap
                    if max(s_start, gts["start_seconds"]) < min(s_end, gts["end_seconds"]):
                        matched_gt = gts
                        break

                is_match = False
                if matched_gt:
                    gt_act = matched_gt.get("primary_activity", "").lower()
                    # Check semantic keyword overlap
                    gt_words = set(gt_act.split())
                    pred_words = set(s_activity.split())
                    overlap = len(gt_words.intersection(pred_words))
                    is_match = overlap >= 1 or (s_activity in gt_act) or (gt_act in s_activity)
                    if is_match:
                        segment_matches += 1

                evaluated_segments.append({
                    "segment_index": s_idx,
                    "predicted_activity": s.get("activity"),
                    "matched_ground_truth_activity": matched_gt.get("primary_activity") if matched_gt else None,
                    "activity_aligned": is_match,
                })

            segment_eval = {
                "total_segments_predicted": len(segments),
                "total_ground_truth_segments": len(gt_segments),
                "aligned_segments_count": segment_matches,
                "segment_alignment_accuracy_percent": round((segment_matches / len(segments)) * 100, 1) if segments else 0.0,
                "segments": evaluated_segments,
            }

            # 3. Narrative Semantic Evaluation
            ref_story = gt.get("reference_narrative")
            if ref_story and video_narration:
                bleu_res = calculate_bleu(video_narration, [ref_story])
                rouge_res = calculate_rouge_l(video_narration, [ref_story])
                narrative_eval = {
                    "bleu_1": bleu_res["bleu_1"],
                    "bleu_2": bleu_res["bleu_2"],
                    "bleu_4": bleu_res["bleu_4"],
                    "rouge_l_f1": rouge_res["rouge_l_f1"],
                    "reference_narrative": ref_story,
                }

        # 4. Redundancy & Efficiency Telemetry
        total_frames = len(frames)
        full_passes = telemetry.get("full_analyses_count", 0)
        redundant_frames = telemetry.get("redundant_frames_count", 0)
        redundancy_rate_pct = round((redundant_frames / total_frames) * 100, 1) if total_frames > 0 else 0.0

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "video_filename": video_filename,
            "is_ground_truth_annotated": is_annotated,
            "evaluation_type": "Video Storytelling Quantitative Benchmark",
            "model_training_disclaimer": "Pretrained models only (YOLOv8n + BLIP + BLIP-VQA). Training epochs: 0. Evaluated against local temporal ground-truth annotations.",
            # 1. GROUND TRUTH EVALUATION METRICS
            "ground_truth_evaluation_metrics": {
                "temporal_transition_detection": transition_eval if is_annotated else {"status": "Not evaluated"},
                "timeline_activity_alignment": segment_eval if is_annotated else {"status": "Not evaluated"},
                "narrative_similarity": narrative_eval if is_annotated else {"status": "Not evaluated"},
            },
            # 2. REDUNDANCY & FILTERING EFFICIENCY
            "redundancy_filtering_efficiency": {
                "total_sampled_frames": total_frames,
                "full_multimodal_analyses": full_passes,
                "redundant_frames_filtered": redundant_frames,
                "compute_saved_percentage": redundancy_rate_pct,
                "key_changes_triggered": telemetry.get("key_changes_count", 0),
            },
            # 3. OPERATIONAL TELEMETRY
            "operational_telemetry": {
                "model_initialization_ms": telemetry.get("model_initialization_ms", 0.0),
                "warmup_jit_ms": telemetry.get("warmup_ms", 0.0),
                "actual_video_processing_ms": telemetry.get("actual_video_processing_ms", 0.0),
                "total_wall_clock_ms": telemetry.get("total_wall_clock_ms", 0.0),
                "avg_frame_latency_full_ms": telemetry.get("average_frame_latency_full_ms", 0.0),
                "avg_frame_latency_all_ms": telemetry.get("average_frame_latency_all_ms", 0.0),
                "temporal_aggregation_ms": telemetry.get("temporal_ms", 0.0),
                "narrative_synthesis_ms": telemetry.get("narration_ms", 0.0),
            },
            "video_narration": video_narration,
            "video_prediction": video_result.get("video_prediction", {}),
        }

        # Save to disk
        out_file = self.results_dir / "quantitative_video_evaluation.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[VideoEvaluator] Quantitative video evaluation saved to: {out_file}")

        return report
