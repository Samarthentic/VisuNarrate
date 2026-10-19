"""VisuNarrate - AI Visual Question Answering (VQA) Module.

Provides:
- Local on-device Visual Question Answering using Salesforce/blip-vqa-base.
- Strict 100% offline inference using local_files_only=True from `models/blip_vqa/`.
- Apple Silicon Metal (MPS) hardware acceleration.
- Targeted interrogative visual query processing.
"""

import os
import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

logger = logging.getLogger("VQAModel")


class VQAModelLoader:
    """Manages the local BLIP-VQA model and executes targeted visual question queries."""

    def __init__(self, model_dir: Optional[str] = None):
        """Initializes the VQA model loader with a local weights path.

        Args:
            model_dir: Path to `models/blip_vqa`. Defaults to `VisuNarrate/models/blip_vqa`.
        """
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "models", "blip_vqa")
        else:
            self.model_dir = model_dir

        self.device = self._detect_device()
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForQuestionAnswering] = None
        self._is_loaded = False
        self.model_name = "Salesforce/blip-vqa-base"

        logger.info(f"[VQAModel] Initialized. Target path: '{self.model_dir}' | Device: '{self.device}'")

    def _detect_device(self) -> str:
        """Detect and return the optimal compute device (MPS on Apple Silicon, fallback to CPU)."""
        if torch.backends.mps.is_available():
            logger.info("[VQAModel] Apple Silicon Metal (MPS) detected for VQA.")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("[VQAModel] NVIDIA CUDA GPU detected for VQA.")
            return "cuda"
        else:
            logger.info("[VQAModel] Standard CPU backend selected for VQA.")
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Return True if VQA model and processor are loaded in memory."""
        return self._is_loaded

    def load_model(self) -> bool:
        """Loads VQA processor and model strictly from the local directory with local_files_only=True."""
        if self._is_loaded:
            return True

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Local BLIP-VQA model directory not found at: '{self.model_dir}'. "
                f"Please run 'python scripts/download_vqa.py' to download weights."
            )

        logger.info(f"[VQAModel] Loading local weights from '{self.model_dir}' (local_files_only=True)...")
        start_time = time.time()

        try:
            self.processor = BlipProcessor.from_pretrained(
                self.model_dir,
                local_files_only=True
            )
            self.model = BlipForQuestionAnswering.from_pretrained(
                self.model_dir,
                local_files_only=True
            )

            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            duration = round(time.time() - start_time, 2)
            logger.info(f"[VQAModel] BLIP-VQA model loaded successfully on '{self.device}' in {duration}s.")
            return True

        except Exception as e:
            logger.error(f"[VQAModel] Failed to load BLIP-VQA model: {str(e)}", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Error loading offline BLIP-VQA model: {str(e)}")

    def _clean_answer(self, raw_answer: str) -> str:
        """Standardizes raw answer text."""
        if not raw_answer:
            return "unknown"
        ans = raw_answer.strip()
        ans = re.sub(r"\s+", " ", ans)
        if ans:
            ans = ans[0].upper() + ans[1:]
        return ans

    def answer_question(self, image: Image.Image, question: str) -> Tuple[str, float]:
        """Answers a single visual question about the image.

        Args:
            image: Preprocessed PIL Image in RGB format.
            question: Natural language question string.

        Returns:
            Tuple of (answer string, latency in ms).
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=25,
                num_beams=3,
            )

        raw_answer = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        cleaned_answer = self._clean_answer(raw_answer)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return cleaned_answer, latency_ms

    def answer_questions(self, image: Image.Image, questions: List[str]) -> Dict[str, Any]:
        """Executes a list of targeted interrogative questions against the image.

        Args:
            image: Preprocessed PIL Image in RGB format.
            questions: List of question strings.

        Returns:
            Dict containing:
                - qa_pairs: List of {"question": q, "answer": a, "latency_ms": lat}
                - total_latency_ms: Combined execution time
                - device: Compute device used
                - model: Model name
                - local_inference: True
                - total_questions: Count of answered questions
        """
        if not self._is_loaded:
            self.load_model()

        total_start = time.time()
        qa_pairs: List[Dict[str, Any]] = []

        for q in questions:
            try:
                ans, lat_ms = self.answer_question(image, q)
                qa_pairs.append({
                    "question": q,
                    "answer": ans,
                    "latency_ms": lat_ms,
                })
            except Exception as e:
                logger.warning(f"[VQAModel] Failed to answer '{q}': {str(e)}")
                qa_pairs.append({
                    "question": q,
                    "answer": "error processing question",
                    "latency_ms": 0.0,
                })

        total_latency_ms = round((time.time() - total_start) * 1000, 2)
        logger.info(f"[VQAModel] Answered {len(qa_pairs)} questions in {total_latency_ms}ms on {self.device}.")

        return {
            "qa_pairs": qa_pairs,
            "total_questions": len(qa_pairs),
            "total_latency_ms": total_latency_ms,
            "device": self.device,
            "model": self.model_name,
            "local_inference": True,
        }
