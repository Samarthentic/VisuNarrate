"""VisuNarrate - Direct High-Speed BLIP-VQA Weight Downloader.

Downloads the Salesforce/blip-vqa-base model files and model.safetensors
directly to `models/blip_vqa/` using requests with chunked streaming.
"""

import os
import sys
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINATION_DIR = os.path.join(BASE_DIR, "models", "blip_vqa")

FILES_TO_DOWNLOAD = [
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
]

BASE_URL = "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main"


def download_file(filename: str, dest_dir: str):
    """Downloads a single file from Hugging Face with streaming and progress logging."""
    url = f"{BASE_URL}/{filename}"
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if filename != "model.safetensors" or os.path.getsize(dest_path) > 900 * 1024 * 1024:
            logger.info(f"File '{filename}' already exists and is verified. Skipping.")
            return

    logger.info(f"Downloading '{filename}' from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 10 * 1024 * 1024 and downloaded % (50 * 1024 * 1024) == 0:
                    pct = round((downloaded / total_size) * 100, 1)
                    mb = round(downloaded / (1024 * 1024), 1)
                    tot_mb = round(total_size / (1024 * 1024), 1)
                    logger.info(f"Progress: {mb} MB / {tot_mb} MB ({pct}%)")

    size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
    logger.info(f"Successfully saved '{filename}' ({size_mb} MB).")


def download_all_vqa():
    """Downloads all required BLIP-VQA model files directly to models/blip_vqa/."""
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    logger.info(f"Target destination: {DESTINATION_DIR}")

    # Remove temporary .cache dir inside models/blip_vqa if present
    cache_dir = os.path.join(DESTINATION_DIR, ".cache")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    for filename in FILES_TO_DOWNLOAD:
        download_file(filename, DESTINATION_DIR)

    logger.info("=" * 60)
    logger.info("All BLIP-VQA model files verified and saved locally!")
    logger.info(f"Storage path: {DESTINATION_DIR}")
    logger.info("BLIP-VQA is now ready for 100% offline inference.")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        download_all_vqa()
    except Exception as e:
        logger.error(f"Failed to download BLIP-VQA files: {str(e)}", exc_info=True)
        sys.exit(1)
