"""VisuNarrate - Main Flask Application Server.

Provides:
- Web dashboard serving (zero CDN dependencies, 100% offline)
- Local image ingestion & validation
- Modular API routing for Adaptive 3-Stream Evidence Fusion
- Apple Silicon Metal (MPS) performance telemetry
"""

import os

# Enforce offline execution mode for Hugging Face & set Matplotlib temp dir
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import time
import logging
from typing import Set
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image

from core.image_engine import ImageEngine
from core.video_engine import VideoEngine
from ai.vision_model import VisionModelLoader
from ai.object_detector import ObjectDetector
from ai.vqa_model import VQAModelLoader
from services.evidence_fusion import EvidenceFusionEngine
from services.prediction_service import PredictionService
from services.temporal_service import TemporalService
from services.video_narration_service import VideoNarrationService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VisuNarrateApp")

# Initialize Flask application
app = Flask(__name__)

# Application Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload for video

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Core AI & Reasoning Engines (Modular Architecture with Lazy Loading)
vision_loader = VisionModelLoader()
object_detector = ObjectDetector()
vqa_loader = VQAModelLoader()
evidence_fusion = EvidenceFusionEngine()
prediction_service = PredictionService()

image_engine = ImageEngine(
    vision_model=vision_loader,
    object_detector=object_detector,
    vqa_model=vqa_loader,
    evidence_fusion=evidence_fusion,
    prediction_service=prediction_service,
)

video_engine = VideoEngine(image_engine=image_engine)
temporal_service = TemporalService()
video_narration_service = VideoNarrationService()


def is_allowed_file(filename: str) -> bool:
    """Validate that the file extension is among allowed image formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_allowed_video_file(filename: str) -> bool:
    """Validate that the file extension is among allowed video formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route("/")
def index():
    """Serves the main VisuNarrate dashboard."""
    return render_template(
        "index.html",
        system_mode="Offline Local Execution",
        compute_device=vision_loader.device.upper(),
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns local system health, offline status, and hardware compute telemetry."""
    return jsonify({
        "status": "online",
        "offline_mode": True,
        "compute_device": vision_loader.device,
        "models": {
            "captioning": "Salesforce/blip-image-captioning-base",
            "object_detection": "Ultralytics/YOLOv8n",
            "vqa": "Salesforce/blip-vqa-base",
        },
        "modules": {
            "image": {"status": "active", "version": "1.4 (Local Predictive Reasoning)"},
            "video": {"status": "active", "version": "2.0 (Video Architecture Foundation)"},
            "live_webcam": {"status": "planned", "version": "3.0 (Phase 3)"},
        },
        "models_loaded": {
            "blip": vision_loader.is_loaded,
            "yolo": object_detector.is_loaded,
            "vqa": vqa_loader.is_loaded,
        },
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Handles secure image upload, validates integrity, and saves to static/uploads/."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file format. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}"
        }), 400

    try:
        original_name = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"upload_{timestamp}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        file.save(save_path)

        with Image.open(save_path) as img:
            width, height = img.size
            img_format = img.format

        file_size_kb = round(os.path.getsize(save_path) / 1024, 2)
        relative_url = f"/static/uploads/{unique_filename}"

        logger.info(f"[Upload] Successfully ingested image: {unique_filename} ({width}x{height}, {file_size_kb} KB)")

        return jsonify({
            "success": True,
            "filename": unique_filename,
            "url": relative_url,
            "file_path": save_path,
            "metadata": {
                "width": width,
                "height": height,
                "format": img_format,
                "size_kb": file_size_kb,
            }
        })

    except Exception as e:
        logger.error(f"[Upload] Error processing image file: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Failed to process image: {str(e)}"}), 500


@app.route("/api/upload/video", methods=["POST"])
def api_upload_video():
    """Handles local video upload, validates format, inspects OpenCV properties, and saves to static/uploads/."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided in request."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No video file selected."}), 400

    if not is_allowed_video_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid video format. Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS)).upper()}"
        }), 400

    try:
        original_name = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"video_{timestamp}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        file.save(save_path)

        # Extract real hardware video properties using VideoEngine
        metadata = video_engine.inspect_video_metadata(save_path)
        file_size_mb = round(os.path.getsize(save_path) / (1024 * 1024), 2)
        relative_url = f"/static/uploads/{unique_filename}"

        logger.info(f"[Upload] Successfully ingested video: {unique_filename} ({metadata['resolution']}, {metadata['fps']} FPS, {metadata['duration_seconds']}s, {file_size_mb} MB)")

        return jsonify({
            "success": True,
            "filename": unique_filename,
            "url": relative_url,
            "file_path": save_path,
            "metadata": {
                **metadata,
                "size_mb": file_size_mb,
            }
        })

    except Exception as e:
        logger.error(f"[Upload] Error processing video file: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Failed to process video: {str(e)}"}), 500


@app.route("/api/analyze/image", methods=["POST"])
def api_analyze_image():
    """Triggers adaptive evidence fusion inference on an uploaded image."""
    data = request.get_json() or {}
    filename = data.get("filename")

    if not filename:
        return jsonify({"success": False, "error": "Filename is required."}), 400

    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": f"File '{safe_name}' not found on server."}), 404

    try:
        # Run real local evidence fusion pipeline
        result = image_engine.analyze_scene(file_path)
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"[Analysis] Error analyzing image: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500


@app.route("/api/analyze/video", methods=["POST"])
def api_analyze_video():
    """Triggers frame sampling, ImageEngine analysis, temporal transition detection, and narrative synthesis."""
    data = request.get_json() or {}
    filename = data.get("filename")
    fps_sample_rate = float(data.get("fps_sample_rate", 1.0))
    max_frames = int(data.get("max_frames", 15))

    if not filename:
        return jsonify({"success": False, "error": "Filename is required."}), 400

    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": f"Video file '{safe_name}' not found on server."}), 404

    try:
        t_start = time.time()

        # 1. Sample Frames & Run Reused ImageEngine
        video_result = video_engine.process_video(
            video_path=file_path,
            target_fps_sample=fps_sample_rate,
            max_frames=max_frames,
        )

        # 2. Temporal Aggregation & Change Detection
        t_temp_start = time.time()
        temporal_analysis = temporal_service.detect_temporal_transitions(video_result["frame_sequence"])
        temporal_duration_ms = round((time.time() - t_temp_start) * 1000, 2)

        # 3. Global Video Narration Synthesis
        video_narration = video_narration_service.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )

        total_server_ms = round((time.time() - t_start) * 1000, 2)

        # Append temporal aggregation telemetry
        video_result["telemetry"]["temporal_aggregation_ms"] = temporal_duration_ms
        video_result["telemetry"]["total_server_ms"] = total_server_ms

        return jsonify({
            "success": True,
            "data": {
                "metadata": video_result["metadata"],
                "sampling_plan": video_result["sampling_plan"],
                "video_narration": video_narration,
                "temporal_analysis": temporal_analysis,
                "frame_sequence": video_result["frame_sequence"],
                "telemetry": video_result["telemetry"],
            }
        })

    except Exception as e:
        logger.error(f"[Analysis] Error analyzing video: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Video inference error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting VisuNarrate Server on http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)


