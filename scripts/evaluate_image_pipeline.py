#!/usr/bin/env python3
"""VisuNarrate - Real-World Image Dataset Evaluation & Validation Harness (Stage 4.7).

Academic Context:
Comprehensive evaluation framework for validating the multi-model perception,
evidence fusion, storytelling, and predictive reasoning pipeline on real-world datasets.

Features:
- Dynamic Dataset Processing: Discovers and processes all images in `data/evaluation_images/`.
- Isolated Timing Hierarchy:
    1. Cold Initialization: Model loading from local disk to host RAM / GPU memory.
    2. First Inference: JIT graph compilation & Apple MPS Metal shader tracing.
    3. Repeated Warm Passes: 3 synchronized runs per image (Median, Mean, Min, Max, P95).
- Diagnostic Failure Classification:
    * scene_misclassification
    * object_detection_failure
    * activity_ambiguity
    * vqa_hallucination
    * evidence_conflict
    * narration_weakness
    * prediction_weakness
- Empirical Human-in-the-Loop Review System:
    * human_scene_correct (0=incorrect, 1=partially correct, 2=correct)
    * human_objects_correct (0=incorrect, 1=partially correct, 2=correct)
    * human_activity_correct (0=incorrect, 1=partially correct, 2=correct)
    * human_narration_quality (0=poor, 1=acceptable, 2=strong)
    * human_prediction_correct (0=implausible/incorrect, 1=plausible but uncertain, 2=strongly plausible)
    * human_hallucination_detected (true/false)
    * human_review_notes (textual observations)
- Offline Guarantee: Validated under TRANSFORMERS_OFFLINE=1 & HF_HUB_OFFLINE=1.
- Detailed JSON and CSV Export with Aggregate Metrics calculation when human labels are populated.
"""

import os
import sys
import time
import json
import csv
import statistics
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.image_engine import ImageEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ImageEvaluator")


def sync_device():
    """Synchronizes Apple MPS or CUDA GPU device queues for exact microsecond-level timing."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


class ImagePipelineEvaluator:
    """Scientific evaluation and benchmarking harness for VisuNarrate Image Pipeline."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}

    def __init__(self, images_dir: str = "data/evaluation_images", results_dir: str = "data/evaluation_results"):
        self.images_dir = Path(PROJECT_ROOT) / images_dir
        self.results_dir = Path(PROJECT_ROOT) / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[Evaluator] Initializing Core ImageEngine and Pre-loading Models...")
        sync_device()
        cold_start = time.time()
        self.engine = ImageEngine()
        
        # Explicit cold loading of all 3 models once into memory
        self.engine.vision_model.load_model()
        self.engine.object_detector.load_model()
        self.engine.vqa_model.load_model()
        sync_device()
        self.cold_initialization_ms = round((time.time() - cold_start) * 1000, 2)
        logger.info(f"[Evaluator] Cold initialization complete in {self.cold_initialization_ms} ms on {self.engine.vision_model.device.upper()}.")

        self.first_inference_ms: Optional[float] = None

    def discover_images(self) -> List[Path]:
        """Discovers all supported test images in the evaluation directory."""
        all_files = [
            f for f in self.images_dir.iterdir()
            if f.is_file() and f.suffix in self.SUPPORTED_EXTENSIONS
        ]
        all_files.sort(key=lambda x: x.name)
        return all_files

    def warmup_and_measure_first_inference(self, first_image_path: Path) -> float:
        """Executes the very first forward pass through the entire pipeline to measure JIT overhead."""
        logger.info(f"[Evaluator] Executing first inference (Metal Shader JIT & Tracing) on: {first_image_path.name}...")
        sync_device()
        t0 = time.time()
        _ = self.engine.analyze_scene(str(first_image_path))
        sync_device()
        first_inf_ms = round((time.time() - t0) * 1000, 2)
        self.first_inference_ms = first_inf_ms
        logger.info(f"[Evaluator] First inference complete in {first_inf_ms} ms (Includes JIT compilation).")
        return first_inf_ms

    def evaluate_image_repeated(self, image_path: Path, num_runs: int = 3) -> Dict[str, Any]:
        """Runs repeated warm inference passes on a single image and calculates distribution metrics."""
        logger.info(f"[Evaluator] Running {num_runs} warm passes on: {image_path.name}...")
        
        run_latencies_ms: List[float] = []
        blip_latencies_ms: List[float] = []
        yolo_latencies_ms: List[float] = []
        vqa_latencies_ms: List[float] = []
        fusion_latencies_ms: List[float] = []
        
        last_result: Optional[Dict[str, Any]] = None

        for run_idx in range(num_runs):
            sync_device()
            t_start = time.time()
            result = self.engine.analyze_scene(str(image_path))
            sync_device()
            total_duration_ms = round((time.time() - t_start) * 1000, 2)
            
            run_latencies_ms.append(total_duration_ms)
            telemetry = result.get("telemetry", {})
            blip_lat = telemetry.get("blip", {}).get("latency_ms", 0.0)
            yolo_lat = telemetry.get("yolo", {}).get("latency_ms", 0.0)
            vqa_lat = telemetry.get("vqa", {}).get("latency_ms", 0.0)
            
            blip_latencies_ms.append(blip_lat)
            yolo_latencies_ms.append(yolo_lat)
            vqa_latencies_ms.append(vqa_lat)
            fusion_latencies_ms.append(max(0.0, round(total_duration_ms - (blip_lat + yolo_lat + vqa_lat), 2)))
            
            last_result = result
            logger.info(f"  ↳ Run {run_idx + 1}/{num_runs}: {total_duration_ms} ms (BLIP: {blip_lat}ms, YOLO: {yolo_lat}ms, VQA: {vqa_lat}ms)")

        # Calculate Statistics
        median_latency = round(statistics.median(run_latencies_ms), 2)
        mean_latency = round(statistics.mean(run_latencies_ms), 2)
        min_latency = round(min(run_latencies_ms), 2)
        max_latency = round(max(run_latencies_ms), 2)
        std_dev = round(statistics.stdev(run_latencies_ms), 2) if len(run_latencies_ms) > 1 else 0.0

        fused_scene = last_result.get("fused_scene", {})
        scene_type_info = last_result.get("scene_type", {})
        person_presence = last_result.get("person_presence", {})
        object_detection = last_result.get("object_detection", {})
        vqa_evidence = last_result.get("vqa_evidence", [])
        skipped_questions = last_result.get("skipped_questions", [])
        conflicts = last_result.get("conflicts", [])
        uncertainty_flags = last_result.get("uncertainty_flags", [])
        predictive_narration = last_result.get("predictive_narration", {})
        metadata = last_result.get("metadata", {})

        yolo_counts = object_detection.get("counts", {})
        yolo_summary_str = ", ".join([f"{k}: {v}" for k, v in yolo_counts.items()]) if yolo_counts else "None"
        vqa_summary_str = " | ".join([f"{q['question']} -> {q['answer']}" for q in vqa_evidence])

        # ---------------------------------------------------------------------
        # Diagnostic Failure Classification (Observational Analysis)
        # ---------------------------------------------------------------------
        has_conflicts = len(conflicts) > 0
        has_uncertainties = len(uncertainty_flags) > 0
        is_graphic = scene_type_info.get("is_graphic", False)
        pred_status = predictive_narration.get("status", "unknown")

        diagnostic_analysis = {
            "scene_misclassification": scene_type_info.get("trust_level") == "uncertain",
            "object_detection_failure": (person_presence.get("status") == "present" and object_detection.get("total_detections", 0) == 0),
            "activity_ambiguity": any("activity" in u.get("aspect", "").lower() for u in uncertainty_flags),
            "vqa_hallucination": any("hallucinat" in str(c).lower() or "artifact" in str(c).lower() for c in conflicts),
            "evidence_conflict": has_conflicts,
            "narration_weakness": len(last_result.get("scene_narration", "").split()) < 6,
            "prediction_weakness": pred_status in {"tentative_prediction", "insufficient_evidence"} and person_presence.get("status") == "present",
        }

        record = {
            "filename": image_path.name,
            "file_path": str(image_path),
            "resolution": f"{metadata.get('original_dimensions', {}).get('width', 0)}x{metadata.get('original_dimensions', {}).get('height', 0)}",
            "processed_resolution": f"{metadata.get('processed_dimensions', {}).get('width', 0)}x{metadata.get('processed_dimensions', {}).get('height', 0)}",
            "heuristic_scene_type": scene_type_info.get("scene_type", "unknown"),
            "scene_type_trust": scene_type_info.get("trust_level", "unknown"),
            "person_presence": person_presence.get("status", "unknown"),
            "person_presence_confidence": person_presence.get("confidence_level", "unknown"),
            "person_count": person_presence.get("count", 0),
            "blip_summary": fused_scene.get("summary", ""),
            "yolo_detected_objects": yolo_summary_str,
            "yolo_detections_count": object_detection.get("total_detections", 0),
            "vqa_questions_executed_count": len(vqa_evidence),
            "vqa_questions_skipped_count": len(skipped_questions),
            "vqa_answers_summary": vqa_summary_str,
            "vqa_details": vqa_evidence,
            "skipped_questions": skipped_questions,
            "fused_scene_representation": fused_scene,
            "current_scene_narration": last_result.get("scene_narration", ""),
            "predictive_narration": predictive_narration.get("prediction", ""),
            "prediction_status": pred_status,
            "prediction_type": predictive_narration.get("prediction_type", "unknown"),
            "prediction_reasoning": predictive_narration.get("reasoning", []),
            "prediction_source_facts": predictive_narration.get("source_facts", []),
            "conflicts_count": len(conflicts),
            "conflicts": conflicts,
            "uncertainty_flags_count": len(uncertainty_flags),
            "uncertainty_flags": uncertainty_flags,
            "diagnostic_analysis": diagnostic_analysis,
            # Multi-Run Warm Metrics
            "num_runs": num_runs,
            "runs_ms": run_latencies_ms,
            "run_1_ms": run_latencies_ms[0] if len(run_latencies_ms) > 0 else 0.0,
            "run_2_ms": run_latencies_ms[1] if len(run_latencies_ms) > 1 else 0.0,
            "run_3_ms": run_latencies_ms[2] if len(run_latencies_ms) > 2 else 0.0,
            "warm_median_ms": median_latency,
            "warm_mean_ms": mean_latency,
            "warm_min_ms": min_latency,
            "warm_max_ms": max_latency,
            "warm_std_dev_ms": std_dev,
            # Component breakdown based on median run
            "blip_median_ms": round(statistics.median(blip_latencies_ms), 2),
            "yolo_median_ms": round(statistics.median(yolo_latencies_ms), 2),
            "vqa_median_ms": round(statistics.median(vqa_latencies_ms), 2),
            "fusion_median_ms": round(statistics.median(fusion_latencies_ms), 2),
            # -----------------------------------------------------------------
            # Human Review Fields (Stage 4.7 Standard Rating Scale)
            # -----------------------------------------------------------------
            "human_scene_correct": None,         # 0=incorrect, 1=partially correct, 2=correct
            "human_objects_correct": None,       # 0=incorrect, 1=partially correct, 2=correct
            "human_activity_correct": None,      # 0=incorrect, 1=partially correct, 2=correct
            "human_narration_quality": None,     # 0=poor, 1=acceptable, 2=strong
            "human_prediction_correct": None,    # 0=implausible/incorrect, 1=plausible but uncertain, 2=strongly plausible
            "human_hallucination_detected": None,# true / false
            "human_review_notes": "",            # Textual review observations
        }

        return record

    def compute_aggregate_human_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates aggregate academic accuracy metrics if manual human ratings are populated."""
        scene_scores = [r["human_scene_correct"] for r in records if r.get("human_scene_correct") is not None]
        object_scores = [r["human_objects_correct"] for r in records if r.get("human_objects_correct") is not None]
        activity_scores = [r["human_activity_correct"] for r in records if r.get("human_activity_correct") is not None]
        narration_scores = [r["human_narration_quality"] for r in records if r.get("human_narration_quality") is not None]
        pred_scores = [r["human_prediction_correct"] for r in records if r.get("human_prediction_correct") is not None]
        hallucination_flags = [r["human_hallucination_detected"] for r in records if r.get("human_hallucination_detected") is not None]

        if not (scene_scores or object_scores or activity_scores or narration_scores or pred_scores or hallucination_flags):
            return {
                "status": "Awaiting manual grading in summary.csv / evaluation_results.json",
                "message": "No manual evaluation scores available yet.",
            }

        def _calc_accuracy(scores: List[int]) -> Optional[float]:
            if not scores:
                return None
            return round((sum(scores) / (len(scores) * 2.0)) * 100, 2)

        return {
            "status": "Human Empirical Evaluation Completed",
            "sample_size": len(scene_scores),
            "scene_accuracy_percent": _calc_accuracy(scene_scores),
            "object_accuracy_percent": _calc_accuracy(object_scores),
            "activity_accuracy_percent": _calc_accuracy(activity_scores),
            "average_narration_quality_scale_0_to_2": round(sum(narration_scores) / len(narration_scores), 2) if narration_scores else None,
            "prediction_plausibility_rate_percent": round((sum(1 for s in pred_scores if s >= 1) / len(pred_scores)) * 100, 2) if pred_scores else None,
            "hallucination_rate_percent": round((sum(1 for h in hallucination_flags if h is True) / len(hallucination_flags)) * 100, 2) if hallucination_flags else None,
        }

    def run_benchmark(self, num_runs: int = 3) -> Dict[str, Any]:
        """Runs the complete benchmark with cold, first inference, and warm multi-pass measurements."""
        images = self.discover_images()
        if not images:
            logger.warning(f"[Evaluator] No test images found in {self.images_dir}.")
            return {"total_images": 0, "records": []}

        # 1. First Inference Pass (Warmup / JIT Tracing on the first image)
        self.warmup_and_measure_first_inference(images[0])

        # 2. Repeated Warm Inference on All Images
        records: List[Dict[str, Any]] = []
        for img_path in images:
            rec = self.evaluate_image_repeated(img_path, num_runs=num_runs)
            records.append(rec)

        # Batch Aggregation
        all_warm_medians = [r["warm_median_ms"] for r in records]
        all_warm_means = [r["warm_mean_ms"] for r in records]
        all_runs_flattened = [run for r in records for run in r["runs_ms"]]
        vqa_counts = [r["vqa_questions_executed_count"] for r in records]

        overall_median = round(statistics.median(all_warm_medians), 2) if all_warm_medians else 0.0
        overall_mean = round(statistics.mean(all_warm_means), 2) if all_warm_means else 0.0
        p95_warm = round(statistics.quantiles(all_runs_flattened, n=20)[18], 2) if len(all_runs_flattened) >= 5 else max(all_runs_flattened)

        # Aggregate Diagnostic Flags Count
        diag_counts = {
            "scene_misclassifications": sum(1 for r in records if r["diagnostic_analysis"]["scene_misclassification"]),
            "object_detection_failures": sum(1 for r in records if r["diagnostic_analysis"]["object_detection_failure"]),
            "activity_ambiguities": sum(1 for r in records if r["diagnostic_analysis"]["activity_ambiguity"]),
            "vqa_hallucinations": sum(1 for r in records if r["diagnostic_analysis"]["vqa_hallucination"]),
            "evidence_conflicts": sum(1 for r in records if r["diagnostic_analysis"]["evidence_conflict"]),
            "narration_weaknesses": sum(1 for r in records if r["diagnostic_analysis"]["narration_weakness"]),
            "prediction_weaknesses": sum(1 for r in records if r["diagnostic_analysis"]["prediction_weakness"]),
        }

        # Calculate Human Review Aggregate
        human_aggregate = self.compute_aggregate_human_metrics(records)

        stats = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_images": len(records),
            "num_runs_per_image": num_runs,
            "cold_initialization_ms": self.cold_initialization_ms,
            "first_inference_ms": self.first_inference_ms,
            "overall_median_warm_latency_ms": overall_median,
            "overall_average_warm_latency_ms": overall_mean,
            "overall_p95_warm_latency_ms": p95_warm,
            "min_warm_latency_ms": min(all_runs_flattened) if all_runs_flattened else 0.0,
            "max_warm_latency_ms": max(all_runs_flattened) if all_runs_flattened else 0.0,
            "average_vqa_questions_executed": round(sum(vqa_counts) / len(vqa_counts), 2) if vqa_counts else 0.0,
            "prediction_status_counts": {
                "supported_prediction": sum(1 for r in records if r["prediction_status"] == "supported_prediction"),
                "tentative_prediction": sum(1 for r in records if r["prediction_status"] == "tentative_prediction"),
                "suppressed": sum(1 for r in records if r["prediction_status"] == "suppressed"),
                "insufficient_evidence": sum(1 for r in records if r["prediction_status"] == "insufficient_evidence"),
            },
            "prediction_type_counts": {
                "interaction_continuation": sum(1 for r in records if r.get("prediction_type") == "interaction_continuation"),
                "task_continuation": sum(1 for r in records if r.get("prediction_type") == "task_continuation"),
                "group_continuation": sum(1 for r in records if r.get("prediction_type") == "group_continuation"),
                "environmental_static": sum(1 for r in records if r.get("prediction_type") == "environmental_static"),
                "insufficient_evidence": sum(1 for r in records if r.get("prediction_type") == "insufficient_evidence"),
                "suppressed_graphic": sum(1 for r in records if r.get("prediction_type") == "suppressed_graphic"),
                "ambiguous_activity": sum(1 for r in records if r.get("prediction_type") == "ambiguous_activity"),
            },
            "scene_type_counts": {
                "photographic_scene": sum(1 for r in records if r["heuristic_scene_type"] == "photographic_scene"),
                "illustration_or_graphic": sum(1 for r in records if r["heuristic_scene_type"] == "illustration_or_graphic"),
                "uncertain": sum(1 for r in records if r["heuristic_scene_type"] == "uncertain"),
            },
            "total_conflicts_detected": sum(r["conflicts_count"] for r in records),
            "total_uncertainty_flags_detected": sum(r["uncertainty_flags_count"] for r in records),
            "diagnostic_failure_summary": diag_counts,
            "human_empirical_metrics": human_aggregate,
            "records": records,
        }

        # Export Files
        self.export_results(stats)
        self.print_summary_table(stats)

        return stats

    def export_results(self, stats: Dict[str, Any]):
        """Saves evaluation records to JSON and summary CSV."""
        json_path = self.results_dir / "evaluation_results.json"
        csv_path = self.results_dir / "summary.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"[Evaluator] Detailed JSON evaluation results saved to: {json_path}")

        records = stats.get("records", [])
        if records:
            fieldnames = [
                "filename",
                "resolution",
                "heuristic_scene_type",
                "person_presence",
                "person_count",
                "yolo_detected_objects",
                "vqa_questions_executed_count",
                "vqa_questions_skipped_count",
                "current_scene_narration",
                "predictive_narration",
                "prediction_status",
                "prediction_type",
                "conflicts_count",
                "uncertainty_flags_count",
                "run_1_ms",
                "run_2_ms",
                "run_3_ms",
                "warm_median_ms",
                "warm_mean_ms",
                "warm_min_ms",
                "warm_max_ms",
                "blip_median_ms",
                "yolo_median_ms",
                "vqa_median_ms",
                "fusion_median_ms",
                # Stage 4.7 Human Review Columns
                "human_scene_correct",
                "human_objects_correct",
                "human_activity_correct",
                "human_narration_quality",
                "human_prediction_correct",
                "human_hallucination_detected",
                "human_review_notes",
            ]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in records:
                    writer.writerow(r)
            logger.info(f"[Evaluator] Evaluation summary CSV saved to: {csv_path}")

    def print_summary_table(self, stats: Dict[str, Any]):
        """Prints a comprehensive scientific ASCII summary table in the console."""
        print("\n" + "=" * 115)
        print("VISUNARRATE REAL-WORLD IMAGE EVALUATION BENCHMARK (STAGE 4.8 REFINED)")
        print("=" * 115)
        print(f"Total Evaluated Images:        {stats['total_images']} (3 Synchronized Warm Passes Per Image)")
        print(f"Cold Model Initialization:     {stats['cold_initialization_ms']} ms (Persistent VRAM Cache)")
        print(f"First Inference (JIT Trace):   {stats['first_inference_ms']} ms (MPS Metal Shader Compilation)")
        print(f"Overall Median Warm Latency:   {stats['overall_median_warm_latency_ms']} ms")
        print(f"Overall Mean Warm Latency:     {stats['overall_average_warm_latency_ms']} ms")
        print(f"Overall P95 Warm Latency:      {stats['overall_p95_warm_latency_ms']} ms [Min: {stats['min_warm_latency_ms']} ms | Max: {stats['max_warm_latency_ms']} ms]")
        print(f"Average VQA Questions Executed:{stats['average_vqa_questions_executed']}")
        print(f"Total Conflicts / Flags:       {stats['total_conflicts_detected']} conflict(s) • {stats['total_uncertainty_flags_detected']} uncertainty flag(s)")
        print("-" * 115)
        print("PREDICTION STATUS DISTRIBUTION:")
        for status, count in stats["prediction_status_counts"].items():
            print(f"  • {status:<24}: {count}")
        print("-" * 115)
        print("PREDICTION TYPE DISTRIBUTION (Stage 4.8 Taxonomy):")
        for ptype, count in stats.get("prediction_type_counts", {}).items():
            print(f"  • {ptype:<26}: {count}")
        print("-" * 115)
        print("DIAGNOSTIC FAILURE CLASSIFICATION SUMMARY (Automated Heuristic Observations):")
        for diag_name, count in stats["diagnostic_failure_summary"].items():
            print(f"  • {diag_name:<28}: {count}")
        print("-" * 115)
        print("MANUAL HUMAN REVIEW METRICS:")
        h_metrics = stats.get("human_empirical_metrics", {})
        if h_metrics.get("status") == "Human Empirical Evaluation Completed":
            print(f"  • Scene Accuracy:            {h_metrics.get('scene_accuracy_percent')}%")
            print(f"  • Object Accuracy:           {h_metrics.get('object_accuracy_percent')}%")
            print(f"  • Activity Accuracy:         {h_metrics.get('activity_accuracy_percent')}%")
            print(f"  • Mean Narration Quality:    {h_metrics.get('average_narration_quality_scale_0_to_2')} / 2.0")
            print(f"  • Prediction Plausibility:   {h_metrics.get('prediction_plausibility_rate_percent')}%")
            print(f"  • Hallucination Rate:        {h_metrics.get('hallucination_rate_percent')}%")
        else:
            print(f"  • Status: {h_metrics.get('message', 'No manual evaluation scores available yet.')}")
            print(f"    (To populate scores, edit 'data/evaluation_results/summary.csv' using the 0-2 rating rubric)")
        print("-" * 115)
        print(f"{'Filename':<24} | {'Median':<9} | {'BLIP':<7} | {'YOLO':<6} | {'VQA':<7} | {'Status':<16} | {'Type'}")
        print("-" * 115)
        for r in stats["records"]:
            fn = r["filename"][:22]
            med = f"{r['warm_median_ms']}ms"
            b = f"{r['blip_median_ms']}ms"
            y = f"{r['yolo_median_ms']}ms"
            v = f"{r['vqa_median_ms']}ms"
            ps = r["prediction_status"][:15]
            pt = r.get("prediction_type", "unknown")[:22]
            print(f"{fn:<24} | {med:<9} | {b:<7} | {y:<6} | {v:<7} | {ps:<16} | {pt}")
        print("=" * 115 + "\n")


def main():
    """Main benchmark entrypoint."""
    evaluator = ImagePipelineEvaluator()
    evaluator.run_benchmark(num_runs=3)


if __name__ == "__main__":
    main()
