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
import threading
from typing import Set, Dict, Any, Optional
from pathlib import Path
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

# Startup warm-up: pre-load all models and run one dummy inference
# so the first real user request is fast (can be skipped in test environments)
if os.environ.get("VISUNARRATE_SKIP_WARMUP") != "1":
    logger.info("[Warm-up] Running startup warm-up inference on Apple Silicon MPS...")
    _warmup_start = time.time()
    try:
        _dummy_img = Image.new("RGB", (64, 64), (128, 128, 128))
        _dummy_path = os.path.join(UPLOAD_FOLDER, "_warmup_dummy.jpg")
        _dummy_img.save(_dummy_path)
        image_engine.analyze_scene(_dummy_path)
        if os.path.exists(_dummy_path):
            os.remove(_dummy_path)
        _warmup_ms = round((time.time() - _warmup_start) * 1000, 2)
        logger.info(f"[Warm-up] Complete in {_warmup_ms} ms — all models hot and ready.")
    except Exception as _warmup_err:
        logger.warning(f"[Warm-up] Failed ({_warmup_err}), first request will be cold.")


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

        video_narration = video_narration_service.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )

        # 4. Evidence-Based Temporal Prediction
        job.stage = "predictive_reasoning"
        job.stage_text = "Generating future prediction hypotheses..."
        job.percent = 98

        video_prediction = video_narration_service.generate_video_prediction(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )

        total_server_ms = round((time.time() - job.start_time) * 1000, 2)
        video_result["telemetry"]["temporal_aggregation_ms"] = temporal_duration_ms
        video_result["telemetry"]["total_server_ms"] = total_server_ms

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


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns local system health, offline status, and hardware compute telemetry."""
    return jsonify({
        "status": "online",
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
            "live_webcam": {"status": "planned", "version": "3.0 (Coming Soon)"},
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

        if not filename:
            return jsonify({"success": False, "error": "Filename is required."}), 400

        safe_name = secure_filename(filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File '{safe_name}' not found on server."}), 404

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

        video_narration = video_narration_service.synthesize_video_narrative(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )

        video_prediction = video_narration_service.generate_video_prediction(
            temporal_analysis=temporal_analysis,
            frame_sequence=video_result["frame_sequence"],
            video_metadata=video_result["metadata"],
        )

        total_ms = round((time.time() - t_start) * 1000, 2)
        video_result["telemetry"]["temporal_aggregation_ms"] = temp_ms
        video_result["telemetry"]["total_server_ms"] = total_ms

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
            }
        })
    except Exception as e:
        logger.error(f"[Analysis] Sync video analysis error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        ANALYSIS_LOCK.release()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting VisuNarrate Server on http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)
