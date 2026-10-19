"""VisuNarrate - One-Time Model Weight Downloader.

Downloads the Salesforce/blip-image-captioning-base model, tokenizer, and processor
from Hugging Face and saves them to the local `models/blip_base` directory.

Once downloaded, the entire application will execute locally without internet access
by utilizing `local_files_only=True`.
"""

import os
import sys
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
DESTINATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "blip_base")


def download_and_save_model():
    """Downloads BLIP weights and saves them locally for offline inference."""
    logger.info(f"Initiating one-time download for: {MODEL_ID}")
    logger.info(f"Target local storage directory: {DESTINATION_DIR}")

    os.makedirs(DESTINATION_DIR, exist_ok=True)

    # 1. Download and save processor & tokenizer
    logger.info("Downloading and caching BlipProcessor...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(DESTINATION_DIR)
    logger.info(f"Processor saved to: {DESTINATION_DIR}")

    # 2. Download and save model weights
    logger.info("Downloading and caching BlipForConditionalGeneration model weights...")
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.save_pretrained(DESTINATION_DIR)
    logger.info(f"Model weights saved to: {DESTINATION_DIR}")

    logger.info("=" * 60)
    logger.info("Model download complete!")
    logger.info(f"All files stored locally in: {DESTINATION_DIR}")
    logger.info("The application can now run 100% offline.")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        download_and_save_model()
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}", exc_info=True)
        sys.exit(1)
