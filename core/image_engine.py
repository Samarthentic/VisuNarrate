"""VisuNarrate - Image Processing Engine with Adaptive Evidence Fusion (Stage 3.5).

Coordinates the optimized 5-stage visual evidence architecture:
1. Ingestion & Preprocessing
2. Stage A: Parallel Evidence Extraction (BLIP Captioner + YOLOv8n Detector)
3. Stage B: Heuristic Scene-Type Classification & Person-Presence Gating
4. Stage C: Two-Tier Adaptive VQA Routing & Early-Exit Execution
5. Stage D: Multi-Source Evidence Fusion, Contradiction Tracking & Structured Graph Synthesis
6. Stage E: Grounded Narrative Synthesis (via NarrationService)
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import torch
import psutil

from ai.vision_model import VisionModelLoader
from ai.object_detector import ObjectDetector
from ai.vqa_model import VQAModelLoader
from services.evidence_fusion import EvidenceFusionEngine
from services.narration_service import NarrationService
from services.prediction_service import PredictionService

logger = logging.getLogger("ImageEngine")


def _sync_mps():
    """Synchronizes Apple MPS or CUDA GPU device queues for exact microsecond-level timing."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()



class ImageEngine:
    """Core image analysis coordinator orchestrating adaptive evidence extraction and fusion."""

    def __init__(
        self,
        vision_model: Optional[VisionModelLoader] = None,
        object_detector: Optional[ObjectDetector] = None,
        vqa_model: Optional[VQAModelLoader] = None,
        evidence_fusion: Optional[EvidenceFusionEngine] = None,
        narration_service: Optional[NarrationService] = None,
        prediction_service: Optional[PredictionService] = None,
    ):
        """Initializes ImageEngine with modular AI and reasoning dependencies."""
        self._lock = threading.Lock()
        self.vision_model = vision_model or VisionModelLoader()
        self.object_detector = object_detector or ObjectDetector()
        self.vqa_model = vqa_model or VQAModelLoader()
        self.evidence_fusion = evidence_fusion or EvidenceFusionEngine()
        self.narration_service = narration_service or NarrationService()
        self.prediction_service = prediction_service or PredictionService()

    def preprocess_image(self, input_source: Any, max_dimension: int = 1024) -> Tuple[Image.Image, Dict[str, Any]]:
        """Validates and standardizes an image for multi-model inference.

        Args:
            input_source: File path (str) or existing PIL.Image instance.
            max_dimension: Max width or height constraint.

        Returns:
            Tuple of (preprocessed PIL.Image in RGB, metadata dictionary).
        """
        if isinstance(input_source, Image.Image):
            img = input_source.copy()
            orig_width, orig_height = img.size
            orig_format = "PIL_FRAME"
        else:
            file_path = str(input_source)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image not found at: {file_path}")

            try:
                with Image.open(file_path) as raw_img:
                    raw_img.verify()
            except Exception as e:
                raise ValueError(f"Corrupted or invalid image file: {str(e)}")

            img = Image.open(file_path)
            orig_width, orig_height = img.size
            orig_format = img.format or "UNKNOWN"

        # Standardize color space to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Constrain dimensions to optimize Apple Silicon M1 memory footprint
        width, height = img.size
        if max(width, height) > max_dimension:
            scaling_factor = max_dimension / float(max(width, height))
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"[ImageEngine] Resized image from ({orig_width}, {orig_height}) to ({new_width}, {new_height})")

        metadata = {
            "original_dimensions": {"width": orig_width, "height": orig_height},
            "processed_dimensions": {"width": img.width, "height": img.height},
            "original_format": orig_format,
            "color_mode": "RGB",
        }

        return img, metadata

    def analyze_scene(self, input_source: Any) -> Dict[str, Any]:
        """Executes adaptive 5-stage evidence fusion pipeline on an image or frame.

        Args:
            input_source: Path to the image file (str) or PIL.Image instance.

        Returns:
            Structured dictionary with fused scene representation, conflicts, and telemetry.
        """
        with self._lock:
            source_name = input_source if isinstance(input_source, str) else "In-Memory PIL Frame"
            logger.info(f"[ImageEngine] Ingesting and executing evidence fusion pipeline for: {source_name}")
            _sync_mps()
            pipeline_start = time.time()

            # 1. Validation and Preprocessing
            pil_image, metadata = self.preprocess_image(input_source)
            prep_ms = round((time.time() - pipeline_start) * 1000, 2)

            # 2. Stage A: Run Preliminary Feature Extractors (BLIP Captioner + YOLOv8n Detector)
            vision_result = self.vision_model.generate_scene_observations(pil_image)
            caption_evidence = {
                "summary": vision_result["observations"].get("summary", ""),
                "observations": vision_result["observations"],
                "model": vision_result["model"],
                "device": vision_result["device"],
                "latency_ms": vision_result["total_latency_ms"],
            }

            object_result = self.object_detector.detect_objects(pil_image, conf_threshold=0.25)
            object_evidence = {
                "objects": object_result["objects"],
                "counts": object_result["counts"],
                "total_detections": object_result["total_detections"],
                "model": object_result["model"],
                "device": object_result["device"],
                "latency_ms": object_result["latency_ms"],
            }

            # 3. Stage B: Heuristic Scene Classification & Person-Presence Gating
            scene_type_info = self.evidence_fusion.classify_scene_type(
                caption_evidence=caption_evidence,
                object_evidence=object_evidence,
                metadata=metadata,
            )

            person_presence_info = self.evidence_fusion.determine_person_presence(
                yolo_evidence=object_evidence,
                caption_evidence=caption_evidence,
            )

            # 4. Stage C: Two-Tier Adaptive VQA Routing & Early-Exit Execution
            first_pass_questions, skipped_questions = self.evidence_fusion.select_first_pass_questions(
                person_presence=person_presence_info,
                scene_type_info=scene_type_info,
            )

            vqa_first_pass = self.vqa_model.answer_questions(pil_image, first_pass_questions)
            all_vqa_evidence = list(vqa_first_pass["qa_pairs"])
            vqa_total_latency_ms = vqa_first_pass["total_latency_ms"]
            all_routed_questions = list(first_pass_questions)

            # Evaluate Pass 2 Triggers (Conditional follow-up)
            pass_two_questions = self.evidence_fusion.evaluate_pass_two_triggers(
                first_pass_qa=all_vqa_evidence,
                caption_evidence=caption_evidence,
                person_presence=person_presence_info,
            )

            if pass_two_questions:
                logger.info(f"[ImageEngine] Executing Pass 2 follow-up queries: {pass_two_questions}")
                vqa_pass_two = self.vqa_model.answer_questions(pil_image, pass_two_questions)
                all_vqa_evidence.extend(vqa_pass_two["qa_pairs"])
                vqa_total_latency_ms = round(vqa_total_latency_ms + vqa_pass_two["total_latency_ms"], 2)
                all_routed_questions.extend(pass_two_questions)
            else:
                logger.info("[ImageEngine] Early Exit triggered: Core scene facts established, skipping Pass 2.")

            # 5. Stage D: Multi-Source Evidence Fusion & Contradiction Tracking
            t_fusion_start = time.time()
            fused_scene = self.evidence_fusion.fuse_evidence(
                caption_evidence=caption_evidence,
                object_evidence=object_evidence,
                vqa_evidence=all_vqa_evidence,
                person_presence=person_presence_info,
                scene_type_info=scene_type_info,
                skipped_questions=skipped_questions,
            )
            fusion_ms = round((time.time() - t_fusion_start) * 1000, 2)

            # 6. Stage E: Grounded Current Scene Narrative Synthesis
            t_narr_start = time.time()
            scene_narration = self.narration_service.synthesize_scene_narrative(
                fused_scene=fused_scene,
                metadata=metadata,
            )
            narration_ms = round((time.time() - t_narr_start) * 1000, 2)

            # 7. Stage F: Local Evidence-Aware Predictive Reasoning
            t_pred_start = time.time()
            predictive_narration = self.prediction_service.generate_predictive_narration(
                fused_scene=fused_scene
            )
            prediction_ms = round((time.time() - t_pred_start) * 1000, 2)

            _sync_mps()
            warm_inference_ms = round((time.time() - pipeline_start) * 1000, 2)

            # Structured Technical Telemetry Block
            technical = {
                "device": self.vision_model.device,
                "timings_ms": {
                    "preprocessing_ms": prep_ms,
                    "captioning_ms": caption_evidence["latency_ms"],
                    "detection_ms": object_evidence["latency_ms"],
                    "vqa_ms": vqa_total_latency_ms,
                    "fusion_ms": fusion_ms,
                    "narration_ms": narration_ms,
                    "prediction_ms": prediction_ms,
                    "total_ms": warm_inference_ms,
                },
                "detections_count": object_evidence["total_detections"],
                "vqa_queries_count": len(all_vqa_evidence),
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
                "memory_mb": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2) if psutil else 0.0,
                "training": {
                    "epochs": 0,
                    "fine_tuning": "none",
                    "statement": "Pretrained weights, inference only. Training epochs run in this project: 0. Fine-tuning: none.",
                },
            }

            return {
                "success": True,
                "scene_narration": scene_narration,
                "predictive_narration": predictive_narration,
                "prediction_status": predictive_narration.get("status", "supported_prediction"),
                "prediction_type": predictive_narration.get("prediction_type", "unknown"),
                "fused_scene": fused_scene,
                "scene_type": scene_type_info,
                "person_presence": person_presence_info,
                "caption_evidence": caption_evidence,
                "object_detection": object_evidence,
                "vqa_evidence": all_vqa_evidence,
                "conflicts": fused_scene.get("conflicts", []),
                "uncertainty_flags": fused_scene.get("uncertainty_flags", []),
                "skipped_questions": skipped_questions,
                "metadata": metadata,
                "telemetry": {
                    "blip": {
                        "device": vision_result["device"],
                        "model": vision_result["model"],
                        "latency_ms": vision_result["total_latency_ms"],
                        "query_latencies": vision_result["latencies_ms"],
                        "observations_count": vision_result["observations_count"],
                    },
                    "yolo": {
                        "device": object_result["device"],
                        "model": object_result["model"],
                        "latency_ms": object_result["latency_ms"],
                        "detections_count": object_result["total_detections"],
                    },
                    "vqa": {
                        "device": vqa_first_pass["device"],
                        "model": vqa_first_pass["model"],
                        "latency_ms": vqa_total_latency_ms,
                        "questions_executed": len(all_vqa_evidence),
                        "routed_questions": all_routed_questions,
                        "skipped_count": len(skipped_questions),
                    },
                    "fusion": {
                        "conflicts_count": len(fused_scene.get("conflicts", [])),
                        "uncertainty_count": len(fused_scene.get("uncertainty_flags", [])),
                        "scene_type": scene_type_info.get("scene_type", "photographic_scene"),
                    },
                    "warm_inference_ms": warm_inference_ms,
                    "total_latency_ms": warm_inference_ms,
                    "local_inference": True,
                },
                "technical": technical,
            }
