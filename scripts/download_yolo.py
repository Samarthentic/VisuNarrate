"""VisuNarrate - One-Time YOLOv8n Weight Downloader.

Downloads the official Ultralytics YOLOv8n object detection model weights (approx 6.2 MB)
using requests with verified SSL certificates and stores them locally at `models/yolo/yolov8n.pt`.
"""

import os
import sys
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINATION_DIR = os.path.join(BASE_DIR, "models", "yolo")
DESTINATION_FILE = os.path.join(DESTINATION_DIR, "yolov8n.pt")
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def download_yolo():
    """Downloads YOLOv8n weights directly and saves them to models/yolo/yolov8n.pt."""
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    logger.info(f"Target destination: {DESTINATION_FILE}")

    try:
        logger.info(f"Downloading YOLOv8n weights from: {YOLO_URL}")
        response = requests.get(YOLO_URL, stream=True, timeout=60)
        response.raise_for_status()

        with open(DESTINATION_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if os.path.exists(DESTINATION_FILE):
            size_mb = round(os.path.getsize(DESTINATION_FILE) / (1024 * 1024), 2)
            logger.info("=" * 60)
            logger.info(f"YOLOv8n successfully downloaded! ({size_mb} MB)")
            logger.info(f"Saved locally to: {DESTINATION_FILE}")
            logger.info("Object detection is now ready for 100% offline inference.")
            logger.info("=" * 60)
        else:
            raise FileNotFoundError(f"Expected file at {DESTINATION_FILE} was not found.")

    except Exception as e:
        logger.error(f"Failed to download YOLOv8n: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    download_yolo()
