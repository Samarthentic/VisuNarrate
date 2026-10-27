"""VisuNarrate - Image Pipeline Quantitative Evaluator.

Executes and measures rigorous quantitative benchmarks for the Image Analysis pipeline:
1. Spatial Object Detection (YOLOv8n): Precision, Recall, F1, mAP@0.5 against ground-truth bounding boxes.
2. Narrative Caption Generation (BLIP + Fusion): BLEU-1 through BLEU-4, ROUGE-L against human reference captions.
3. Visual Question Answering (BLIP-VQA): Exact Match & Soft Match accuracy against ground-truth Q&A references.
4. Scene & Presence Classification: Accuracy against verified ground truth.
5. Hardware & Latency Telemetry: Isolated cold start, JIT trace, and warm inference distributions.

Explicitly isolates:
- Benchmark Quality Metrics (measured against ground truth)
- Model Confidence (model-reported internal scores)
- Operational Telemetry (measured wall-clock latencies & memory)
"""

import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from evaluation.dataset_loader import EvaluationDatasetLoader
from evaluation.metrics import (
    calculate_iou,
    evaluate_detections,
    calculate_map,
    calculate_bleu,
    calculate_rouge_l,
    evaluate_vqa,
)

logger = logging.getLogger("ImageEvaluator")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ImagePipelineQuantitativeEvaluator:
    """Evaluates the multi-model image pipeline against labeled ground truth."""

    def __init__(
        self,
        images_dir: str = "data/evaluation_images",
        results_dir: str = "data/evaluation_results",
        dataset_loader: Optional[EvaluationDatasetLoader] = None,
        image_engine=None,
    ):
        self.images_dir = Path(PROJECT_ROOT) / images_dir
        self.results_dir = Path(PROJECT_ROOT) / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_loader = dataset_loader or EvaluationDatasetLoader()
        self.ground_truth = self.dataset_loader.load_image_ground_truth()

        self._engine = image_engine
        self.cold_initialization_ms: Optional[float] = None
        self.first_inference_ms: Optional[float] = None

    def _get_engine(self):
        """Lazy-loads ImageEngine to preserve memory until needed."""
        if self._engine is None:
            from core.image_engine import ImageEngine
            t0 = time.time()
            self._engine = ImageEngine()
            self._engine.vision_model.load_model()
            self._engine.object_detector.load_model()
            self._engine.vqa_model.load_model()
            self.cold_initialization_ms = round((time.time() - t0) * 1000, 2)
        return self._engine

    def evaluate_single_image(
        self,
        image_path: Path,
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluates model output for a single image against ground truth if present.

        Args:
            image_path: Path to the image file.
            analysis_result: Pre-computed ImageEngine output. If None, runs inference.

        Returns:
            Dict containing evaluated metrics and telemetry for this image.
        """
        engine = self._get_engine()
        filename = image_path.name
        gt = self.dataset_loader.get_image_annotation(filename)
        is_annotated = gt is not None

        # Execute inference if result not provided
        t_start = time.time()
        if analysis_result is None:
            analysis_result = engine.analyze_scene(str(image_path))
        inference_latency_ms = round((time.time() - t_start) * 1000, 2)

        # Extract predictions
        objects_pred = analysis_result.get("object_detection", {}).get("objects", [])
        narration_pred = analysis_result.get("scene_narration", "")
        vqa_pred_list = analysis_result.get("vqa_evidence", [])
        scene_type_pred = analysis_result.get("scene_type", {}).get("scene_type", "unknown")
        presence_pred = analysis_result.get("person_presence", {}).get("status", "unknown")
        telemetry = analysis_result.get("telemetry", {})

        # Compute Ground-Truth Evaluation if annotated
        detection_eval = None
        caption_eval = None
        vqa_eval = None
        classification_eval = None

        if is_annotated:
            # 1. Object Detection Evaluation
            gt_objects = gt.get("objects", [])
            detection_eval = evaluate_detections(
                predictions=objects_pred,
                ground_truth=gt_objects,
                iou_threshold=0.5,
            )

            # 2. Caption Generation Evaluation (BLEU & ROUGE-L)
            ref_captions = gt.get("reference_captions", [])
            if ref_captions and narration_pred:
                bleu_res = calculate_bleu(narration_pred, ref_captions)
                rouge_res = calculate_rouge_l(narration_pred, ref_captions)
                caption_eval = {
                    "bleu_1": bleu_res["bleu_1"],
                    "bleu_2": bleu_res["bleu_2"],
                    "bleu_3": bleu_res["bleu_3"],
                    "bleu_4": bleu_res["bleu_4"],
                    "brevity_penalty": bleu_res["brevity_penalty"],
                    "rouge_l_precision": rouge_res["rouge_l_precision"],
                    "rouge_l_recall": rouge_res["rouge_l_recall"],
                    "rouge_l_f1": rouge_res["rouge_l_f1"],
                    "reference_count": len(ref_captions),
                }

            # 3. VQA Evaluation
            gt_vqa = gt.get("vqa", [])
            if gt_vqa:
                vqa_items = []
                exact_matches = 0
                soft_matches = 0
                for gt_q in gt_vqa:
                    q_text = gt_q.get("question", "").lower()
                    refs = gt_q.get("reference_answers", [])
                    # Find matching executed VQA question
                    matched_ans = None
                    for v_pred in vqa_pred_list:
                        if q_text in v_pred.get("question", "").lower() or v_pred.get("question", "").lower() in q_text:
                            matched_ans = v_pred.get("answer", "")
                            break

                    if matched_ans:
                        q_eval = evaluate_vqa(matched_ans, refs)
                        if q_eval["exact_match"]:
                            exact_matches += 1
                        if q_eval["soft_match"]:
                            soft_matches += 1
                        vqa_items.append({
                            "question": gt_q.get("question"),
                            "predicted_answer": matched_ans,
                            "reference_answers": refs,
                            "exact_match": q_eval["exact_match"],
                            "soft_match": q_eval["soft_match"],
                        })
                    else:
                        vqa_items.append({
                            "question": gt_q.get("question"),
                            "predicted_answer": None,
                            "reference_answers": refs,
                            "exact_match": False,
                            "soft_match": False,
                            "status": "question_not_executed_by_policy",
                        })

                vqa_eval = {
                    "total_questions_evaluated": len(gt_vqa),
                    "exact_match_count": exact_matches,
                    "soft_match_count": soft_matches,
                    "exact_match_accuracy": round((exact_matches / len(gt_vqa)) * 100, 2) if gt_vqa else 0.0,
                    "soft_match_accuracy": round((soft_matches / len(gt_vqa)) * 100, 2) if gt_vqa else 0.0,
                    "details": vqa_items,
                }

            # 4. Scene Classification & Presence
            gt_scene_type = gt.get("scene_type")
            gt_presence = gt.get("person_presence")
            classification_eval = {
                "scene_type_correct": (scene_type_pred == gt_scene_type) if gt_scene_type else None,
                "person_presence_correct": (presence_pred == gt_presence) if gt_presence else None,
                "predicted_scene_type": scene_type_pred,
                "ground_truth_scene_type": gt_scene_type,
                "predicted_presence": presence_pred,
                "ground_truth_presence": gt_presence,
            }

        return {
            "filename": filename,
            "is_ground_truth_annotated": is_annotated,
            "resolution": f"{analysis_result.get('metadata', {}).get('original_dimensions', {}).get('width', 0)}x{analysis_result.get('metadata', {}).get('original_dimensions', {}).get('height', 0)}",
            # Clear isolation:
            "benchmark_evaluation": {
                "object_detection": detection_eval,
                "caption_metrics": caption_eval,
                "vqa_metrics": vqa_eval,
                "classification_metrics": classification_eval,
            } if is_annotated else {"status": "Not evaluated (No ground-truth annotation)"},
            "model_confidence": {
                "object_detections_count": len(objects_pred),
                "detected_objects": objects_pred,
                "average_object_confidence": round(sum(p.get("confidence", 0.0) for p in objects_pred) / len(objects_pred), 3) if objects_pred else 0.0,
                "presence_confidence": analysis_result.get("person_presence", {}).get("confidence_level", "unknown"),
                "prediction_status": analysis_result.get("predictive_narration", {}).get("status", "unknown"),
            },
            "operational_telemetry": {
                "total_inference_ms": inference_latency_ms,
                "blip_latency_ms": telemetry.get("blip", {}).get("latency_ms", 0.0),
                "yolo_latency_ms": telemetry.get("yolo", {}).get("latency_ms", 0.0),
                "vqa_latency_ms": telemetry.get("vqa", {}).get("latency_ms", 0.0),
                "vqa_questions_executed": len(vqa_pred_list),
            },
            "narration": narration_pred,
            "prediction": analysis_result.get("predictive_narration", {}).get("prediction", ""),
        }

    def run_benchmark(self, num_runs: int = 1) -> Dict[str, Any]:
        """Runs evaluation over all images in data/evaluation_images/ against ground truth."""
        engine = self._get_engine()
        all_images = sorted([
            f for f in self.images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])

        if not all_images:
            return {"total_images": 0, "status": "No evaluation images found"}

        logger.info(f"[ImageEvaluator] Running benchmark over {len(all_images)} images...")

        # 1. First Inference / Warmup measurement
        t_jit0 = time.time()
        _ = engine.analyze_scene(str(all_images[0]))
        self.first_inference_ms = round((time.time() - t_jit0) * 1000, 2)

        # 2. Benchmark execution
        image_records = []
        dataset_detection_items = []
        bleu1_list, bleu2_list, bleu4_list, rouge_f1_list = [], [], [], []
        vqa_exact_list, vqa_soft_list = [], []
        scene_type_hits, presence_hits = 0, 0
        total_annotated = 0

        for img_p in all_images:
            rec = self.evaluate_single_image(img_p)
            image_records.append(rec)

            if rec["is_ground_truth_annotated"]:
                total_annotated += 1
                b_eval = rec["benchmark_evaluation"]

                # Caption metrics aggregation
                cap = b_eval.get("caption_metrics")
                if cap:
                    bleu1_list.append(cap["bleu_1"])
                    bleu2_list.append(cap["bleu_2"])
                    bleu4_list.append(cap["bleu_4"])
                    rouge_f1_list.append(cap["rouge_l_f1"])

                # VQA metrics aggregation
                vqa = b_eval.get("vqa_metrics")
                if vqa:
                    vqa_exact_list.append(vqa["exact_match_accuracy"])
                    vqa_soft_list.append(vqa["soft_match_accuracy"])

                # Classification aggregation
                cls_m = b_eval.get("classification_metrics")
                if cls_m:
                    if cls_m.get("scene_type_correct"):
                        scene_type_hits += 1
                    if cls_m.get("person_presence_correct"):
                        presence_hits += 1

        # Compile Summary
        latencies = [r["operational_telemetry"]["total_inference_ms"] for r in image_records]
        avg_bleu1 = round(statistics.mean(bleu1_list), 4) if bleu1_list else 0.0
        avg_bleu2 = round(statistics.mean(bleu2_list), 4) if bleu2_list else 0.0
        avg_bleu4 = round(statistics.mean(bleu4_list), 4) if bleu4_list else 0.0
        avg_rouge_f1 = round(statistics.mean(rouge_f1_list), 4) if rouge_f1_list else 0.0
        avg_vqa_exact = round(statistics.mean(vqa_exact_list), 2) if vqa_exact_list else 0.0
        avg_vqa_soft = round(statistics.mean(vqa_soft_list), 2) if vqa_soft_list else 0.0

        # Detection totals
        tp_total = sum(r["benchmark_evaluation"].get("object_detection", {}).get("true_positives", 0) for r in image_records if r["is_ground_truth_annotated"])
        fp_total = sum(r["benchmark_evaluation"].get("object_detection", {}).get("false_positives", 0) for r in image_records if r["is_ground_truth_annotated"])
        fn_total = sum(r["benchmark_evaluation"].get("object_detection", {}).get("false_negatives", 0) for r in image_records if r["is_ground_truth_annotated"])
        denom_p = tp_total + fp_total
        denom_r = tp_total + fn_total
        ov_precision = round(tp_total / denom_p, 4) if denom_p > 0 else 0.0
        ov_recall = round(tp_total / denom_r, 4) if denom_r > 0 else 0.0
        ov_f1 = round((2 * ov_precision * ov_recall) / (ov_precision + ov_recall), 4) if (ov_precision + ov_recall) > 0 else 0.0

        # Mean mAP@0.5 across annotated images
        map_list = [
            r["benchmark_evaluation"].get("object_detection", {}).get("mAP_50", 0.0)
            for r in image_records
            if r["is_ground_truth_annotated"] and r["benchmark_evaluation"].get("object_detection")
        ]
        mean_map_50 = round(statistics.mean(map_list), 4) if map_list else 0.0

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_type": "Image Pipeline Ground-Truth Quantitative Benchmark",
            "model_training_disclaimer": "All models utilize official pretrained weights. Training epochs run in this project: 0. Benchmark reflects on-device zero-shot inference evaluated against verified local ground-truth annotations.",
            "dataset_info": {
                "total_images_discovered": len(all_images),
                "total_ground_truth_annotated": total_annotated,
                "coverage_percent": round((total_annotated / len(all_images)) * 100, 1),
            },
            # 1. GROUND TRUTH EVALUATION METRICS
            "ground_truth_evaluation_metrics": {
                "object_detection_yolov8n": {
                    "mAP_50": mean_map_50,
                    "precision": ov_precision,
                    "recall": ov_recall,
                    "f1_score": ov_f1,
                    "true_positives": tp_total,
                    "false_positives": fp_total,
                    "false_negatives": fn_total,
                    "iou_threshold": 0.5,
                },
                "caption_generation_blip": {
                    "bleu_1": avg_bleu1,
                    "bleu_2": avg_bleu2,
                    "bleu_4": avg_bleu4,
                    "rouge_l_f1": avg_rouge_f1,
                    "description": "Standard linguistic overlap between generated narrative and multi-reference human descriptions.",
                },
                "visual_question_answering_blip_vqa": {
                    "exact_match_accuracy_percent": avg_vqa_exact,
                    "soft_match_accuracy_percent": avg_vqa_soft,
                    "description": "Top-1 exact match and semantic soft match against accepted reference answers.",
                },
                "scene_classification_heuristics": {
                    "scene_type_accuracy_percent": round((scene_type_hits / total_annotated) * 100, 1) if total_annotated else 0.0,
                    "person_presence_accuracy_percent": round((presence_hits / total_annotated) * 100, 1) if total_annotated else 0.0,
                }
            },
            # 2. OPERATIONAL TELEMETRY & HARDWARE METRICS
            "operational_telemetry": {
                "cold_initialization_ms": self.cold_initialization_ms,
                "first_inference_compilation_ms": self.first_inference_ms,
                "median_inference_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
                "mean_inference_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
                "min_inference_latency_ms": round(min(latencies), 2) if latencies else 0.0,
                "max_inference_latency_ms": round(max(latencies), 2) if latencies else 0.0,
                "hardware_device": engine.vision_model.device.upper(),
            },
            "records": image_records,
        }

        # Save to disk
        out_file = self.results_dir / "quantitative_image_evaluation.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[ImageEvaluator] Quantitative evaluation saved to: {out_file}")

        return report
