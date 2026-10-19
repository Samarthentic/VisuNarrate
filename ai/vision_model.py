"""VisuNarrate - AI Vision Model Loader & Local Inference Module.

Provides:
- Lazy-loaded Salesforce/blip-image-captioning-base vision-language model.
- 100% offline enforcement with local_files_only=True.
- Apple Silicon Metal (MPS) hardware acceleration.
- Streamlined scene understanding extracting Global Summary, Subject, and Environmental Setting.
"""

import os
import re
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger("VisionModel")


class VisionModelLoader:
    """Manages local model weights and executes guided visual scene analysis."""

    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the model loader with local model path."""
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "models", "blip_base")
        else:
            self.model_dir = model_dir

        self.device = self._detect_device()
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self._is_loaded = False
        self.model_name = "Salesforce/blip-image-captioning-base"

        logger.info(f"[VisionModel] Initialized. Target path: '{self.model_dir}' | Compute device: '{self.device}'")

    def _detect_device(self) -> str:
        """Detect and return the best available compute device (MPS on Apple Silicon, fallback to CPU)."""
        if torch.backends.mps.is_available():
            logger.info("[VisionModel] Apple Silicon Metal (MPS) hardware acceleration detected.")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("[VisionModel] NVIDIA CUDA GPU detected.")
            return "cuda"
        else:
            logger.info("[VisionModel] Standard CPU compute backend selected.")
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Return True if model and processor are actively loaded into RAM/VRAM."""
        return self._is_loaded

    def load_model(self) -> bool:
        """Lazy-loads processor and model from the local filesystem with local_files_only=True."""
        if self._is_loaded:
            return True

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Local model weights directory not found at: '{self.model_dir}'. "
                f"Please run 'python scripts/download_model.py' to download weights."
            )

        logger.info(f"[VisionModel] Loading model weights from local path: '{self.model_dir}' (local_files_only=True)...")
        start_time = time.time()

        try:
            self.processor = BlipProcessor.from_pretrained(
                self.model_dir,
                local_files_only=True
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_dir,
                local_files_only=True
            )

            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            load_duration = round((time.time() - start_time), 2)
            logger.info(f"[VisionModel] Model successfully loaded on '{self.device}' in {load_duration}s.")
            return True

        except Exception as e:
            logger.error(f"[VisionModel] Failed to load local model: {str(e)}", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Error loading offline BLIP model: {str(e)}")

    def _clean_text(self, text: str, prompt_prefix: Optional[str] = None) -> str:
        """Cleans and standardizes raw decoded model tokens."""
        if not text:
            return ""

        cleaned = text.strip()

        # Remove prefix if present at start of decoded text
        if prompt_prefix:
            prefix_lower = prompt_prefix.strip().lower()
            if cleaned.lower().startswith(prefix_lower):
                cleaned = cleaned[len(prefix_lower):].strip()

        # Normalize common tokenizer artifacts
        cleaned = re.sub(r"\s+([',.?!:;])", r"\1", cleaned)
        cleaned = re.sub(r"([`'\"])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _run_single_query(self, image: Image.Image, prompt: Optional[str] = None) -> Tuple[str, float]:
        """Runs a single prompt query against the BLIP model on the target device."""
        start_time = time.time()

        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=30,
                min_length=4,
                num_beams=2,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
            )

        raw_text = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        cleaned_text = self._clean_text(raw_text, prompt)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return cleaned_text, latency_ms

    def generate_scene_observations(self, image: Image.Image) -> Dict[str, Any]:
        """Executes optimized visual captioning across essential semantic axes.

        Queries:
        1. Summary: Unconditional global scene caption (Primary semantic anchor).
        2. Subject: Primary focus entity / topic.
        3. Environment: Background setting / atmospheric context.

        Args:
            image: Preprocessed PIL Image in RGB format.

        Returns:
            Dict with structured observations, per-prompt timing, and aggregate telemetry.
        """
        if not self._is_loaded:
            self.load_model()

        total_start = time.time()

        query_definitions = [
            ("summary", None, "Global scene overview"),
            ("subject", "the main subject is", "Main subject identification"),
            ("environment", "the setting and background is", "Environmental setting"),
        ]

        observations: Dict[str, str] = {}
        latencies_ms: Dict[str, float] = {}
        failed_queries: List[str] = []

        for key, prompt_text, description in query_definitions:
            try:
                obs_text, lat_ms = self._run_single_query(image, prompt_text)
                observations[key] = obs_text
                latencies_ms[key] = lat_ms
            except Exception as e:
                logger.warning(f"[VisionModel] Query '{key}' failed: {str(e)}")
                observations[key] = ""
                latencies_ms[key] = 0.0
                failed_queries.append(key)

        total_latency_ms = round((time.time() - total_start) * 1000, 2)

        logger.info(
            f"[VisionModel] Optimized analysis complete ({len(observations)} observations, "
            f"total {total_latency_ms}ms on {self.device})"
        )

        return {
            "observations": observations,
            "failed_queries": failed_queries,
            "latencies_ms": latencies_ms,
            "total_latency_ms": total_latency_ms,
            "observations_count": len(observations) - len(failed_queries),
            "device": self.device,
            "model": self.model_name,
            "local_inference": True,
        }
