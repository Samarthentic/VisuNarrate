"""VisuNarrate - Main Flask Application Server.

Provides:
- Web dashboard serving (zero CDN dependencies, 100% offline)
- Local image ingestion & synchronous adaptive evidence fusion
- Asynchronous video job queue & single-worker concurrency lock
- Real-time video progress tracking, cancellation, and timeline scrubbing
- Apple Silicon Metal (MPS) performance telemetry
"""

import os

# Enforce offline execution mode for Hugging Face, Ultralytics & set Matplotlib temp dir
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Disable Ultralytics telemetry/analytics (must run after YOLO_OFFLINE is set)
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False})
except Exception:
    pass  # App must start even if Ultralytics settings are unavailable

import time
import uuid
import shutil
import queue
import logging
import platform
import threading
from typing import Set, Dict, Any, Optional
from pathlib import Path
import psutil
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image

from core.image_engine import ImageEngine
from core.video_engine import VideoEngine
from core.live_engine import LiveEngine
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
THUMB_FOLDER = os.path.join(UPLOAD_FOLDER, "thumbs")
ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150 MB max upload for video

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["THUMB_FOLDER"] = THUMB_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMB_FOLDER, exist_ok=True)

# Concurrency Control: Only ONE analysis (image or video) may execute at a time
ANALYSIS_LOCK = threading.Lock()

# Initialize Core AI & Reasoning Engines (Single shared model instance)
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
video_narration_service = VideoNarrationService(prediction_service=prediction_service)
live_engine = LiveEngine(
    image_engine=image_engine,
    temporal_service=temporal_service,
    video_narration_service=video_narration_service,
)

# Startup warm-up: pre-load all models and run one realistic scene inference
# with a photo of a person so BLIP, YOLO and VQA pipelines and Metal shaders
# are genuinely hot and ready before the first user request arrives.
if os.environ.get("VISUNARRATE_SKIP_WARMUP") != "1":
    logger.info("[Warm-up] Running realistic startup warm-up on Apple Silicon MPS...")
    _warmup_start = time.time()
    try:
        # Locate realistic person photo from evaluation images
        _candidate_warmup_paths = [
            os.path.join(BASE_DIR, "data", "evaluation_images", "portrait_smartphone.jpg"),
            os.path.join(BASE_DIR, "data", "evaluation_images", "ChatGPT Image Aug 20, 2026, 10_31_28 PM.png"),
            os.path.join(BASE_DIR, "data", "sample_images", "test_scene.jpg"),
        ]
        _warmup_img_path = None
        for _p in _candidate_warmup_paths:
            if os.path.exists(_p):
                _warmup_img_path = _p
                break

        if _warmup_img_path:
            logger.info(f"[Warm-up] Executing realistic warm-up using: {os.path.basename(_warmup_img_path)}")
            image_engine.analyze_scene(_warmup_img_path)
        else:
            _dummy_img = Image.new("RGB", (384, 384), (128, 128, 128))
            _dummy_path = os.path.join(UPLOAD_FOLDER, "_warmup_fallback.jpg")
            _dummy_img.save(_dummy_path)
            image_engine.analyze_scene(_dummy_path)
            if os.path.exists(_dummy_path):
                os.remove(_dummy_path)

        _warmup_ms = round((time.time() - _warmup_start) * 1000, 2)
        logger.info(f"[Warm-up] Complete in {_warmup_ms} ms — BLIP, YOLO, and VQA fully primed on MPS.")
    except Exception as _warmup_err:
        logger.warning(f"[Warm-up] Failed ({_warmup_err}), first request will be cold.", exc_info=True)


# =============================================================================
# Background Video Job System
# =============================================================================
class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoJob:
    """Represents an asynchronous video analysis task."""

    def __init__(
        self,
        job_id: str,
        file_path: str,
        filename: str,
        fps_sample_rate: float = 1.0,
        max_frames: int = 30,
        delete_after: bool = False,
    ):
        self.job_id = job_id
        self.file_path = file_path
        self.filename = filename
        self.fps_sample_rate = fps_sample_rate
        self.max_frames = max_frames
        self.delete_after = delete_after

        self.status = JobStatus.QUEUED
        self.stage = "queued"
        self.stage_text = "Job queued in worker..."
        self.percent = 0
        self.frames_analysed = 0
        self.total_frames = 0
        self.eta_seconds = 0.0

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.created_at = time.time()

        self.cancel_event = threading.Event()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.thumbnail_dir = os.path.join(THUMB_FOLDER, job_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes job state for JSON polling."""
        elapsed = 0.0
        if self.start_time:
            end_t = self.end_time or time.time()
            elapsed = round(end_t - self.start_time, 1)

        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "stage_text": self.stage_text,
            "percent": self.percent,
            "progress_percent": self.percent,
            "frames_analysed": self.frames_analysed,
            "frames_analyzed": self.frames_analysed,
            "total_frames": self.total_frames,
            "total_sampled_frames": self.total_frames,
            "eta_seconds": self.eta_seconds,
            "elapsed_seconds": elapsed,
            "error": self.error,
            "result": self.result,
        }


JOBS: Dict[str, VideoJob] = {}
JOB_QUEUE: queue.Queue = queue.Queue()


def run_single_video_job(job: VideoJob):
    """Executes full video analysis pipeline for a single queued job."""
    if job.cancel_event.is_set():
        job.status = JobStatus.CANCELLED
        job.stage_text = "Cancelled before processing began."
        return

    job.status = JobStatus.PROCESSING
    job.start_time = time.time()
    job.stage = "inspecting"
    job.stage_text = "Validating video properties and decoding frames..."
    job.percent = 5

    logger.info(f"[JobWorker] Starting video job {job.job_id} ({job.filename})...")

    def progress_callback(stage_name: str, current_frame: int, total_frames: int, message: str):
        if job.cancel_event.is_set():
            return

        job.stage = stage_name
        job.frames_analysed = current_frame
        job.total_frames = total_frames

        if total_frames > 0:
            pct = int((current_frame / total_frames) * 85) + 8
            job.percent = min(92, pct)

            if current_frame > 0 and job.start_time:
                elapsed = time.time() - job.start_time
                avg_time = elapsed / current_frame
                rem_frames = total_frames - current_frame
                job.eta_seconds = round(rem_frames * avg_time, 1)

        job.stage_text = message

    try:
        # 1. Execute Frame Sampling & Vision Pipeline
        video_result = video_engine.process_video(
            video_path=job.file_path,
            target_fps_sample=job.fps_sample_rate,
            max_frames=job.max_frames,
            progress_callback=progress_callback,
            cancel_flag=job.cancel_event,
            thumbnail_dir=job.thumbnail_dir,
            job_id=job.job_id,
        )

        if job.cancel_event.is_set() or video_result.get("is_cancelled", False):
            job.status = JobStatus.CANCELLED
            job.stage = "cancelled"
            job.stage_text = "Processing cancelled by user."
            job.end_time = time.time()
            logger.info(f"[JobWorker] Job {job.job_id} successfully cancelled.")
            _cleanup_job_files(job)
            return

        # 2. Temporal State Aggregation & Segment Merging
        job.stage = "temporal_aggregation"
        job.stage_text = "Detecting chronological transitions & merging segments..."
        job.percent = 92
        t_temp_start = time.time()

        temporal_analysis = temporal_service.detect_temporal_transitions(
            frame_sequence=video_result["frame_sequence"],
            video_duration=video_result["metadata"].get("duration_seconds", 0.0),
        )
        temporal_duration_ms = round((time.time() - t_temp_start) * 1000, 2)

        # 3. Overall Narrative Synthesis
        job.stage = "story_synthesis"
        job.stage_text = "Synthesizing coherent story paragraph..."
        job.percent = 95

        t_story_start = time.time()
        video_narration = video_narration_service.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )
        story_ms = round((time.time() - t_story_start) * 1000, 2)

        # 4. Evidence-Based Temporal Prediction
        job.stage = "predictive_reasoning"
        job.stage_text = "Generating future prediction hypotheses..."
        job.percent = 98

        t_pred_start = time.time()
        video_prediction = video_narration_service.generate_video_prediction(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )
        prediction_ms = round((time.time() - t_pred_start) * 1000, 2)

        total_server_ms = round((time.time() - job.start_time) * 1000, 2)
        video_result["telemetry"]["temporal_aggregation_ms"] = temporal_duration_ms
        video_result["telemetry"]["total_server_ms"] = total_server_ms

        # Populate and finalize structured technical block
        technical = video_result.get("technical", {})
        if "timings_ms" in technical:
            technical["timings_ms"]["temporal_aggregation_ms"] = temporal_duration_ms
            technical["timings_ms"]["narration_synthesis_ms"] = story_ms
            technical["timings_ms"]["prediction_synthesis_ms"] = prediction_ms
            technical["timings_ms"]["total_ms"] = total_server_ms
        technical["memory_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)

        segments = temporal_analysis.get("segments", [])
        changes = temporal_analysis.get("changes", [])
        frames = video_result.get("frame_sequence", [])

        job.result = {
            "metadata": video_result["metadata"],
            "sampling_plan": video_result["sampling_plan"],
            "video_story": video_narration,
            "video_narration": video_narration,
            "narration": video_narration,
            "prediction": video_prediction,
            "predictive_narration": video_prediction,
            "temporal_analysis": temporal_analysis,
            "timeline_segments": segments,
            "segments": segments,
            "changes": changes,
            "sampled_frames": frames,
            "frame_sequence": frames,
            "telemetry": video_result["telemetry"],
            "technical": technical,
        }

        job.status = JobStatus.DONE
        job.stage = "done"
        job.stage_text = "Analysis complete."
        job.percent = 100
        job.eta_seconds = 0.0
        job.end_time = time.time()
        logger.info(f"[JobWorker] Job {job.job_id} completed successfully in {total_server_ms}ms.")

        if job.delete_after:
            _cleanup_uploaded_video_only(job)

    except Exception as e:
        logger.error(f"[JobWorker] Job {job.job_id} encountered fatal error: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.stage = "error"
        job.error = str(e)
        job.stage_text = f"Failed: {str(e)}"
        job.end_time = time.time()
        _cleanup_job_files(job)


def _cleanup_uploaded_video_only(job: VideoJob):
    """Deletes uploaded video file after successful analysis when requested."""
    try:
        if os.path.exists(job.file_path):
            os.remove(job.file_path)
            logger.info(f"[Cleanup] Deleted video source file after analysis: {job.file_path}")
    except Exception as e:
        logger.warning(f"[Cleanup] Failed to delete video source: {e}")


def _cleanup_job_files(job: VideoJob):
    """Deletes uploaded video and thumbnail folder for a cancelled or failed job."""
    try:
        if os.path.exists(job.file_path):
            os.remove(job.file_path)
        if os.path.exists(job.thumbnail_dir):
            shutil.rmtree(job.thumbnail_dir, ignore_errors=True)
        logger.info(f"[Cleanup] Cleaned temporary files for job {job.job_id}")
    except Exception as e:
        logger.warning(f"[Cleanup] Error cleaning job files: {e}")


def worker_loop():
    """Background worker processing video analysis jobs sequentially under the shared lock."""
    logger.info("[WorkerThread] Video job processing daemon online.")
    while True:
        try:
            job = JOB_QUEUE.get()
            if job is None:
                break

            # Acquire global model execution lock
            with ANALYSIS_LOCK:
                run_single_video_job(job)

            JOB_QUEUE.task_done()
        except Exception as e:
            logger.error(f"[WorkerThread] Unexpected worker loop exception: {e}", exc_info=True)


# Start background worker daemon
WORKER_THREAD = threading.Thread(target=worker_loop, daemon=True, name="VisuNarrateVideoWorker")
WORKER_THREAD.start()


# =============================================================================
# Periodic File Auto-Cleanup Daemon (>1 hour old files)
# =============================================================================
def periodic_cleanup_loop():
    """Removes uploads and thumbnail caches older than 1 hour to preserve disk space."""
    while True:
        try:
            time.sleep(300)  # Check every 5 minutes
            now = time.time()
            cutoff = now - 3600  # 1 hour

            # Cleanup video and image uploads
            if os.path.exists(UPLOAD_FOLDER):
                for entry in os.listdir(UPLOAD_FOLDER):
                    if entry in {".gitkeep", "thumbs", "_warmup_dummy.jpg"}:
                        continue
                    p = os.path.join(UPLOAD_FOLDER, entry)
                    if os.path.isfile(p):
                        try:
                            if os.path.getmtime(p) < cutoff:
                                os.remove(p)
                                logger.info(f"[AutoCleanup] Removed expired upload file: {entry}")
                        except Exception:
                            pass

            # Cleanup thumbnails folders
            if os.path.exists(THUMB_FOLDER):
                for job_dir in os.listdir(THUMB_FOLDER):
                    p = os.path.join(THUMB_FOLDER, job_dir)
                    if os.path.isdir(p):
                        try:
                            if os.path.getmtime(p) < cutoff:
                                shutil.rmtree(p, ignore_errors=True)
                                logger.info(f"[AutoCleanup] Removed expired thumbnail folder: {job_dir}")
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"[AutoCleanup] Cleanup loop exception: {e}")


CLEANUP_THREAD = threading.Thread(target=periodic_cleanup_loop, daemon=True, name="VisuNarrateAutoCleanup")
CLEANUP_THREAD.start()


# =============================================================================
# Helper Utilities
# =============================================================================
def is_allowed_file(filename: str) -> bool:
    """Validate that the file extension is among allowed image formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_allowed_video_file(filename: str) -> bool:
    """Validate that the file extension is among allowed video formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


# =============================================================================
# HTTP API Routes
# =============================================================================
@app.route("/")
def index():
    """Serves the main VisuNarrate dashboard."""
    return render_template(
        "index.html",
        system_mode="Offline Local Execution",
        compute_device=vision_loader.device.upper(),
    )


@app.route("/api/health", methods=["GET"])
@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns local system health, offline status, and hardware compute telemetry."""
    return jsonify({
        "status": "online",
        "healthy": True,
        "offline_mode": True,
        "compute_device": vision_loader.device,
        "is_busy": ANALYSIS_LOCK.locked(),
        "models": {
            "captioning": "Salesforce/blip-image-captioning-base",
            "object_detection": "Ultralytics/YOLOv8n",
            "vqa": "Salesforce/blip-vqa-base",
        },
        "modules": {
            "image": {"status": "active", "version": "1.5 (Local Predictive Reasoning)"},
            "video": {"status": "active", "version": "2.1 (Async Job System & Production Timeline)"},
            "live": {"status": "active", "version": "3.0 (In-Memory Camera Stream)"},
        },
        "models_loaded": {
            "blip": bool(vision_loader.is_loaded),
            "yolo": bool(object_detector.is_loaded),
            "vqa": bool(vqa_loader.is_loaded),
        },
    })


@app.route("/api/system/info", methods=["GET"])
def api_system_info():
    """Returns measured runtime environment, hardware specs, model parameter counts, and offline status.

    All values are directly measured at runtime or read from code/model weights.
    VisuNarrate performs inference only with pretrained models; no training or fine-tuning is performed.
    """
    def _dir_size_mb(path: str) -> float:
        total = 0
        if os.path.isfile(path):
            return round(os.path.getsize(path) / (1024 * 1024), 2)
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if not os.path.islink(fp) and os.path.exists(fp):
                    total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 2)

    vm_mem = psutil.virtual_memory()
    disk_info = shutil.disk_usage(BASE_DIR)
    proc_mem = psutil.Process(os.getpid()).memory_info()

    # Exact parameter counts measured from PyTorch model instances or known weight architectures
    blip_params = sum(p.numel() for p in vision_loader.model.parameters()) if (vision_loader.is_loaded and vision_loader.model) else 223971644
    yolo_params = sum(p.numel() for p in object_detector.model.model.parameters()) if (object_detector.is_loaded and object_detector.model) else 3157200
    vqa_params = sum(p.numel() for p in vqa_loader.model.parameters()) if (vqa_loader.is_loaded and vqa_loader.model) else 361230140

    import torch
    import transformers
    import ultralytics
    import cv2

    system_info = {
        "success": True,
        "runtime": {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
        },
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ultralytics": ultralytics.__version__,
            "cv2": cv2.__version__,
        },
        "device": {
            "type": vision_loader.device,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "cuda_available": torch.cuda.is_available(),
        },
        "offline_flags": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
            "YOLO_OFFLINE": os.environ.get("YOLO_OFFLINE", "0"),
            "all_offline_enforced": True,
        },
        "memory": {
            "total_ram_gb": round(vm_mem.total / (1024 ** 3), 2),
            "available_ram_gb": round(vm_mem.available / (1024 ** 3), 2),
            "used_ram_percent": vm_mem.percent,
            "process_rss_mb": round(proc_mem.rss / (1024 * 1024), 2),
            "free_disk_gb": round(disk_info.free / (1024 ** 3), 2),
        },
        "models": {
            "blip_captioning": {
                "name": "BLIP Base Image Captioning",
                "source": "Salesforce/blip-image-captioning-base",
                "architecture": "BlipForConditionalGeneration",
                "parameters_count": blip_params,
                "parameters_formatted": f"{round(blip_params / 1e6, 2)}M",
                "size_on_disk_mb": _dir_size_mb(os.path.join(BASE_DIR, "models", "blip_base")),
                "input_resolution": "384x384 (internal) / resized <= 1024x1024",
                "loaded": vision_loader.is_loaded,
            },
            "yolo_detection": {
                "name": "YOLOv8 Nano",
                "source": "Ultralytics YOLOv8n",
                "architecture": "YOLOv8n Detection (80 COCO classes)",
                "parameters_count": yolo_params,
                "parameters_formatted": f"{round(yolo_params / 1e6, 2)}M",
                "size_on_disk_mb": _dir_size_mb(os.path.join(BASE_DIR, "models", "yolo", "yolov8n.pt")),
                "input_resolution": "640x640",
                "loaded": object_detector.is_loaded,
            },
            "blip_vqa": {
                "name": "BLIP VQA Base",
                "source": "Salesforce/blip-vqa-base",
                "architecture": "BlipForQuestionAnswering",
                "parameters_count": vqa_params,
                "parameters_formatted": f"{round(vqa_params / 1e6, 2)}M",
                "size_on_disk_mb": _dir_size_mb(os.path.join(BASE_DIR, "models", "blip_vqa")),
                "input_resolution": "480x480 (internal) / resized <= 1024x1024",
                "loaded": vqa_loader.is_loaded,
            },
            "total_parameters_count": blip_params + yolo_params + vqa_params,
            "total_parameters_formatted": f"{round((blip_params + yolo_params + vqa_params) / 1e6, 2)}M",
        },
        "ready": {
            "models_loaded": vision_loader.is_loaded and object_detector.is_loaded and vqa_loader.is_loaded,
            "busy": ANALYSIS_LOCK.locked(),
        },
        "training": {
            "epochs_run": 0,
            "fine_tuning": "none",
            "statement": "Pretrained weights, inference only. Training epochs run in this project: 0. Fine-tuning: none.",
        },
    }
    return jsonify(system_info)


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
    """Handles local video upload, validates format, file size, duration limits, and decodability."""
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

    save_path = None
    try:
        original_name = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"video_{timestamp}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        file.save(save_path)

        # Validate video using VideoEngine
        metadata = video_engine.validate_video_file(save_path)
        file_size_mb = metadata.get("size_mb", round(os.path.getsize(save_path) / (1024 * 1024), 2))
        relative_url = f"/static/uploads/{unique_filename}"

        logger.info(
            f"[Upload] Valid video ingested: {unique_filename} "
            f"({metadata['resolution']}, {metadata['fps']} FPS, {metadata['duration_seconds']}s, {file_size_mb} MB)"
        )

        return jsonify({
            "success": True,
            "status": "ok",
            "filename": unique_filename,
            "url": relative_url,
            "file_url": relative_url,
            "filepath": save_path,
            "file_path": save_path,
            "duration": metadata.get("duration_seconds", 0.0),
            "width": metadata.get("width", 0),
            "height": metadata.get("height", 0),
            "fps": metadata.get("fps", 0.0),
            "total_frames": metadata.get("total_frames", 0),
            "size_mb": file_size_mb,
            "metadata": {
                **metadata,
                "size_mb": file_size_mb,
            }
        })

    except ValueError as val_err:
        # Expected validation error (e.g. corrupt file, duration > 180s, size > 150MB)
        logger.warning(f"[Upload] Video validation rejected: {val_err}")
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        return jsonify({"success": False, "error": str(val_err)}), 400

    except Exception as e:
        logger.error(f"[Upload] Error processing video upload: {e}", exc_info=True)
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Failed to ingest video: {str(e)}"}), 500


@app.route("/api/analyze/image", methods=["POST"])
def api_analyze_image():
    """Triggers adaptive evidence fusion inference on an uploaded image under single-analysis lock."""
    # Enforce concurrency restriction: only ONE analysis runs at a time
    acquired = ANALYSIS_LOCK.acquire(blocking=False)
    if not acquired:
        return jsonify({
            "success": False,
            "error": "The system is currently busy analyzing another request. Please try again in a few moments."
        }), 503

    try:
        data = request.get_json() or {}
        filename = data.get("filename")
        direct_path = data.get("image_path") or data.get("file_path")

        if direct_path and os.path.exists(direct_path):
            file_path = direct_path
        elif filename:
            safe_name = secure_filename(filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            if not os.path.exists(file_path):
                return jsonify({"success": False, "error": f"File '{safe_name}' not found on server."}), 404
        else:
            return jsonify({"success": False, "error": "Filename or image_path is required."}), 400

        result = image_engine.analyze_scene(file_path)
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"[Analysis] Error analyzing image: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500
    finally:
        ANALYSIS_LOCK.release()


# =============================================================================
# Video Job Queue Endpoints
# =============================================================================
@app.route("/api/video/jobs", methods=["POST"])
def api_create_video_job():
    """Creates an asynchronous video analysis job and enqueues it for processing."""
    data = request.get_json() or {}
    filename = data.get("filename")
    video_path_arg = data.get("video_path") or data.get("file_path")
    fps_sample_rate = float(data.get("fps_sample_rate", 1.0))
    max_frames = min(int(data.get("max_frames", 30)), video_engine.MAX_ANALYZED_KEYFRAMES)
    delete_after = bool(data.get("delete_after", False))

    if not filename and not video_path_arg:
        return jsonify({"success": False, "error": "Filename or video_path is required."}), 400

    if video_path_arg and os.path.exists(video_path_arg):
        file_path = video_path_arg
        safe_name = os.path.basename(video_path_arg)
    else:
        name_to_use = filename or (os.path.basename(video_path_arg) if video_path_arg else "")
        safe_name = secure_filename(name_to_use)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": f"Video file '{safe_name}' not found on server."}), 404

    try:
        # Pre-validate before queuing
        _ = video_engine.validate_video_file(file_path)
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job = VideoJob(
        job_id=job_id,
        file_path=file_path,
        filename=safe_name,
        fps_sample_rate=fps_sample_rate,
        max_frames=max_frames,
        delete_after=delete_after,
    )

    JOBS[job_id] = job
    JOB_QUEUE.put(job)

    logger.info(f"[JobQueue] Enqueued video job {job_id} for file: {safe_name}")

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": job.status,
        "message": "Video analysis job created and enqueued.",
    }), 202


@app.route("/api/video/jobs/<job_id>", methods=["GET"])
def api_get_video_job(job_id: str):
    """Returns the live status, progress, stage, ETA, and final results for a video job."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": f"Job '{job_id}' not found."}), 404

    return jsonify({
        "success": True,
        **job.to_dict()
    })


@app.route("/api/video/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_video_job(job_id: str):
    """Gracefully cancels a running or queued video analysis job."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": f"Job '{job_id}' not found."}), 404

    if job.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}:
        return jsonify({
            "success": True,
            "message": f"Job is already {job.status}.",
            "status": job.status,
        })

    job.cancel_event.set()
    job.status = JobStatus.CANCELLED
    job.stage = "cancelled"
    job.stage_text = "Analysis cancelled by user."
    job.end_time = time.time()

    logger.info(f"[JobQueue] User requested cancellation for job {job_id}")

    return jsonify({
        "success": True,
        "message": "Job cancellation initiated.",
        "status": JobStatus.CANCELLED,
    })


# Maintain backwards-compatible synchronous endpoint with busy locking
@app.route("/api/analyze/video", methods=["POST"])
def api_analyze_video_sync():
    """Backwards-compatible synchronous video analysis endpoint."""
    acquired = ANALYSIS_LOCK.acquire(blocking=False)
    if not acquired:
        return jsonify({
            "success": False,
            "error": "The system is currently busy analyzing another request. Please try again shortly."
        }), 503

    try:
        data = request.get_json() or {}
        filename = data.get("filename")
        fps_sample_rate = float(data.get("fps_sample_rate", 1.0))
        max_frames = min(int(data.get("max_frames", 30)), video_engine.MAX_ANALYZED_KEYFRAMES)

        if not filename:
            return jsonify({"success": False, "error": "Filename is required."}), 400

        safe_name = secure_filename(filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"Video file '{safe_name}' not found."}), 404

        t_start = time.time()
        job_id = f"sync_{int(time.time())}"
        thumb_dir = os.path.join(THUMB_FOLDER, job_id)

        video_result = video_engine.process_video(
            video_path=file_path,
            target_fps_sample=fps_sample_rate,
            max_frames=max_frames,
            thumbnail_dir=thumb_dir,
            job_id=job_id,
        )

        t_temp = time.time()
        temporal_analysis = temporal_service.detect_temporal_transitions(
            frame_sequence=video_result["frame_sequence"],
            video_duration=video_result["metadata"].get("duration_seconds", 0.0),
        )
        temp_ms = round((time.time() - t_temp) * 1000, 2)

        t_story_start = time.time()
        video_narration = video_narration_service.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )
        story_ms = round((time.time() - t_story_start) * 1000, 2)

        t_pred_start = time.time()
        video_prediction = video_narration_service.generate_video_prediction(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )
        prediction_ms = round((time.time() - t_pred_start) * 1000, 2)

        total_ms = round((time.time() - t_start) * 1000, 2)
        video_result["telemetry"]["temporal_aggregation_ms"] = temp_ms
        video_result["telemetry"]["total_server_ms"] = total_ms

        technical = video_result.get("technical", {})
        if "timings_ms" in technical:
            technical["timings_ms"]["temporal_aggregation_ms"] = temp_ms
            technical["timings_ms"]["narration_synthesis_ms"] = story_ms
            technical["timings_ms"]["prediction_synthesis_ms"] = prediction_ms
            technical["timings_ms"]["total_ms"] = total_ms
        technical["memory_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)

        return jsonify({
            "success": True,
            "data": {
                "metadata": video_result["metadata"],
                "sampling_plan": video_result["sampling_plan"],
                "video_narration": video_narration,
                "predictive_narration": video_prediction,
                "temporal_analysis": temporal_analysis,
                "segments": temporal_analysis.get("segments", []),
                "frame_sequence": video_result["frame_sequence"],
                "telemetry": video_result["telemetry"],
                "technical": technical,
            }
        })
    except Exception as e:
        logger.error(f"[Analysis] Sync video analysis error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        ANALYSIS_LOCK.release()


# =============================================================================
# Live Mode Endpoints (In-Memory Live Camera Processing)
# =============================================================================
@app.route("/api/live/analyze", methods=["POST"])
def api_live_analyze():
    """Analyzes a live camera frame strictly in RAM with non-blocking busy locking."""
    acquired = ANALYSIS_LOCK.acquire(blocking=False)
    if not acquired:
        return jsonify({
            "success": False,
            "error": "The system is currently busy analyzing another request. Please try again shortly.",
            "is_busy": True,
        }), 503

    try:
        session_id = None
        frame_data = None

        if request.is_json:
            json_data = request.get_json() or {}
            session_id = json_data.get("session_id")
            frame_data = json_data.get("frame")
        elif "frame" in request.files:
            file = request.files["frame"]
            frame_data = file.read()
            session_id = request.form.get("session_id")
        elif "frame" in request.form:
            frame_data = request.form.get("frame")
            session_id = request.form.get("session_id")

        if not frame_data:
            return jsonify({
                "success": False,
                "error": "No frame data provided in request. Provide 'frame' as base64 string or file upload."
            }), 400

        result = live_engine.process_frame(frame_data=frame_data, session_id=session_id)
        return jsonify({
            "success": True,
            "data": result,
        })
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as e:
        logger.error(f"[Live] Error analyzing frame: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Live inference error: {str(e)}"}), 500
    finally:
        ANALYSIS_LOCK.release()


@app.route("/api/live/session/<session_id>/summary", methods=["POST"])
def api_live_session_summary(session_id: str):
    """Synthesizes overall story and prediction from accumulated live camera history."""
    acquired = ANALYSIS_LOCK.acquire(blocking=False)
    if not acquired:
        return jsonify({
            "success": False,
            "error": "The system is currently busy analyzing another request. Please try again shortly.",
            "is_busy": True,
        }), 503

    try:
        summary_result = live_engine.generate_session_summary(session_id)
        return jsonify({
            "success": True,
            "data": summary_result,
        })
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 404
    except Exception as e:
        logger.error(f"[Live] Error generating session summary for {session_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Summary generation error: {str(e)}"}), 500
    finally:
        ANALYSIS_LOCK.release()


@app.route("/api/live/session/<session_id>", methods=["DELETE"])
def api_live_delete_session(session_id: str):
    """Terminates live camera session and purges in-memory frame cache."""
    deleted = live_engine.delete_session(session_id)
    if deleted:
        return jsonify({"success": True, "message": f"Session '{session_id}' deleted."})
    return jsonify({"success": False, "error": f"Session '{session_id}' not found."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting VisuNarrate Server on http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)
