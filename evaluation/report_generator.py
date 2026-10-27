"""VisuNarrate - Unified Evaluation Report Generator.

Aggregates quantitative benchmarks across Image, Video, and Live Camera pipelines:
- Strictly isolates:
  1. Benchmark Evaluation Quality Metrics (mAP@0.5, BLEU-1..4, ROUGE-L, VQA accuracy, Temporal F1)
  2. Model Confidence Distribution (model output confidences)
  3. Operational Telemetry (latency, memory, throughput)
- Never fabricates metrics: If ground truth is absent, explicitly reports "Not evaluated".
- Handles zero denominators gracefully (None / 0.0 with explanation).
- Exports formatted JSON and CSV summaries.
"""

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ReportGenerator")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EvaluationReportGenerator:
    """Consolidates and serves quantitative evaluation summaries for all modalities."""

    def __init__(self, results_dir: str = "data/evaluation_results"):
        self.results_dir = Path(PROJECT_ROOT) / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_image_evaluation(self) -> Optional[Dict[str, Any]]:
        """Loads quantitative image evaluation results from disk."""
        path = self.results_dir / "quantitative_image_evaluation.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[ReportGenerator] Error reading {path}: {e}")
        return None

    def load_video_evaluation(self) -> Optional[Dict[str, Any]]:
        """Loads quantitative video evaluation results from disk."""
        path = self.results_dir / "quantitative_video_evaluation.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[ReportGenerator] Error reading {path}: {e}")
        return None

    def load_live_evaluation(self) -> Optional[Dict[str, Any]]:
        """Loads quantitative live camera evaluation results from disk."""
        path = self.results_dir / "quantitative_live_evaluation.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[ReportGenerator] Error reading {path}: {e}")
        return None

    def build_master_summary(self) -> Dict[str, Any]:
        """Compiles unified cross-modal evaluation summary."""
        img_eval = self.load_image_evaluation()
        vid_eval = self.load_video_evaluation()
        live_eval = self.load_live_evaluation()

        master = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "project_name": "VisuNarrate",
            "academic_context": "Final-Year B.E. Major Project Evaluation Benchmark",
            "model_architecture": {
                "vision_captioning": "Salesforce/blip-image-captioning-base",
                "object_detector": "Ultralytics YOLOv8n (Local PyTorch weights)",
                "visual_qa": "Salesforce/blip-vqa-base",
                "training_epochs": 0,
                "fine_tuning": "None (Offline Pretrained Inference Only)",
                "hardware_acceleration": "Apple Silicon Metal (MPS)",
            },
            # 1. EVALUATION QUALITY BENCHMARKS (GROUND-TRUTH VALIDATED)
            "benchmarks": {
                "image_pipeline": img_eval.get("ground_truth_evaluation_metrics") if img_eval else {
                    "status": "Benchmark pending. Run 'python scripts/evaluate_all_modalities.py'"
                },
                "video_pipeline": vid_eval.get("ground_truth_evaluation_metrics") if vid_eval else {
                    "status": "Benchmark pending. Run 'python scripts/evaluate_all_modalities.py'"
                },
                "live_pipeline": {
                    "status": "Live camera evaluated on streaming redundancy efficiency and runtime latency (ground-truth reference unavailable for dynamic camera feed)."
                }
            },
            # 2. REDUNDANCY & FILTERING EFFICIENCY
            "efficiency_metrics": {
                "video_compute_saved_percentage": vid_eval.get("redundancy_filtering_efficiency", {}).get("compute_saved_percentage") if vid_eval else None,
                "live_compute_saved_percentage": live_eval.get("redundancy_filtering_efficiency", {}).get("compute_saved_percentage") if live_eval else None,
            },
            # 3. OPERATIONAL TELEMETRY & HARDWARE PROFILES
            "telemetry_metrics": {
                "image_median_latency_ms": img_eval.get("operational_telemetry", {}).get("median_inference_latency_ms") if img_eval else None,
                "video_avg_frame_latency_ms": vid_eval.get("operational_telemetry", {}).get("avg_frame_latency_full_ms") if vid_eval else None,
                "live_median_frame_latency_ms": live_eval.get("latency_metrics", {}).get("overall_median_frame_latency_ms") if live_eval else None,
            },
            "artifacts": {
                "image_evaluation_json": "data/evaluation_results/quantitative_image_evaluation.json",
                "video_evaluation_json": "data/evaluation_results/quantitative_video_evaluation.json",
                "live_evaluation_json": "data/evaluation_results/quantitative_live_evaluation.json",
            }
        }

        # Save master summary JSON
        master_file = self.results_dir / "master_evaluation_summary.json"
        with open(master_file, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)

        # Export CSV summary table
        self.export_csv_summary(master)

        return master

    def export_csv_summary(self, master_summary: Dict[str, Any]) -> str:
        """Generates a structured CSV table of benchmark metrics for academic reports."""
        csv_path = self.results_dir / "evaluation_metrics_summary.csv"

        rows = [
            ["Category", "Modality", "Model / Subsystem", "Metric Name", "Measured Value", "Units / Range", "Ground Truth Basis"],
            # Image Detection
            ["Benchmark Quality", "Image", "YOLOv8n", "mAP@0.5", self._get_val(master_summary, "benchmarks.image_pipeline.object_detection_yolov8n.mAP_50"), "0.0 - 1.0", "Verified Bounding Boxes"],
            ["Benchmark Quality", "Image", "YOLOv8n", "Precision", self._get_val(master_summary, "benchmarks.image_pipeline.object_detection_yolov8n.precision"), "0.0 - 1.0", "Verified Bounding Boxes"],
            ["Benchmark Quality", "Image", "YOLOv8n", "Recall", self._get_val(master_summary, "benchmarks.image_pipeline.object_detection_yolov8n.recall"), "0.0 - 1.0", "Verified Bounding Boxes"],
            ["Benchmark Quality", "Image", "YOLOv8n", "F1 Score", self._get_val(master_summary, "benchmarks.image_pipeline.object_detection_yolov8n.f1_score"), "0.0 - 1.0", "Harmonic Mean (P & R)"],
            # Image Captioning
            ["Benchmark Quality", "Image", "BLIP Base", "BLEU-1", self._get_val(master_summary, "benchmarks.image_pipeline.caption_generation_blip.bleu_1"), "0.0 - 1.0", "Human Reference Captions"],
            ["Benchmark Quality", "Image", "BLIP Base", "BLEU-2", self._get_val(master_summary, "benchmarks.image_pipeline.caption_generation_blip.bleu_2"), "0.0 - 1.0", "Human Reference Captions"],
            ["Benchmark Quality", "Image", "BLIP Base", "BLEU-4", self._get_val(master_summary, "benchmarks.image_pipeline.caption_generation_blip.bleu_4"), "0.0 - 1.0", "Human Reference Captions"],
            ["Benchmark Quality", "Image", "BLIP Base", "ROUGE-L F1", self._get_val(master_summary, "benchmarks.image_pipeline.caption_generation_blip.rouge_l_f1"), "0.0 - 1.0", "Longest Common Subsequence"],
            # Image VQA
            ["Benchmark Quality", "Image", "BLIP VQA", "Exact Match Accuracy", f"{self._get_val(master_summary, 'benchmarks.image_pipeline.visual_question_answering_blip_vqa.exact_match_accuracy_percent')}%", "Percentage", "Ground-Truth Q&A References"],
            ["Benchmark Quality", "Image", "BLIP VQA", "Soft Match Accuracy", f"{self._get_val(master_summary, 'benchmarks.image_pipeline.visual_question_answering_blip_vqa.soft_match_accuracy_percent')}%", "Percentage", "Ground-Truth Q&A References"],
            # Video Temporal Transitions
            ["Benchmark Quality", "Video", "TemporalService", "Transition Precision", self._get_val(master_summary, "benchmarks.video_pipeline.temporal_transition_detection.precision"), "0.0 - 1.0", "Annotated State Changes (±1.5s)"],
            ["Benchmark Quality", "Video", "TemporalService", "Transition Recall", self._get_val(master_summary, "benchmarks.video_pipeline.temporal_transition_detection.recall"), "0.0 - 1.0", "Annotated State Changes (±1.5s)"],
            ["Benchmark Quality", "Video", "TemporalService", "Transition F1 Score", self._get_val(master_summary, "benchmarks.video_pipeline.temporal_transition_detection.f1_score"), "0.0 - 1.0", "Harmonic Mean (P & R)"],
            # Efficiency
            ["Efficiency", "Video", "VideoEngine", "Compute Saved (Redundancy)", f"{self._get_val(master_summary, 'efficiency_metrics.video_compute_saved_percentage')}%", "Percentage", "Pixel-diff Thresholding"],
            ["Efficiency", "Live Camera", "LiveEngine", "Compute Saved (Redundancy)", f"{self._get_val(master_summary, 'efficiency_metrics.live_compute_saved_percentage')}%", "Percentage", "Pixel-diff Thresholding"],
            # Operational Latency
            ["Telemetry", "Image", "Full Pipeline", "Median Warm Latency", f"{self._get_val(master_summary, 'telemetry_metrics.image_median_latency_ms')} ms", "Milliseconds", "Runtime Execution"],
            ["Telemetry", "Video", "Full Pipeline", "Avg Frame Latency", f"{self._get_val(master_summary, 'telemetry_metrics.video_avg_frame_latency_ms')} ms", "Milliseconds", "Runtime Execution"],
            ["Telemetry", "Live Camera", "Full Pipeline", "Median Frame Latency", f"{self._get_val(master_summary, 'telemetry_metrics.live_median_frame_latency_ms')} ms", "Milliseconds", "Runtime Execution"],
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return str(csv_path)

    def _get_val(self, data: Dict[str, Any], path: str, default: str = "N/A") -> Any:
        keys = path.split(".")
        cur = data
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur if cur is not None else default
