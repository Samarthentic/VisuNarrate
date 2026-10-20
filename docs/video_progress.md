# Video Mode Progress Log

## Iteration 1: Audit

### Findings

**Backend (what exists):**
- `core/video_engine.py` (512 lines): Video processing engine. Samples frames via OpenCV, runs redundancy filter (visual diff + fast YOLO check), runs full ImageEngine pipeline on novel frames. Reuses shared ImageEngine models. Has `process_video()` method that returns structured frame records.
- `services/temporal_service.py` (229 lines): Detects activity/interaction/person-count transitions between frames. Produces stable_facts, changes, and key_moments.
- `services/video_narration_service.py` (132 lines): Synthesizes a story paragraph from temporal analysis.
- `app.py`: Synchronous endpoints `/api/upload/video` and `/api/analyze/video`.
- No job system, no progress reporting, no cancellation, no concurrency lock.

**Frontend (what exists):**
- Video tab shows a "Preview" modal blocker. Actual video DOM is in a hidden `#videoModeView` div with stub elements and no JS logic.

---

## Iteration 2: Backend Job System, Concurrency Lock & Validation

### Implementation
1. **Concurrency Lock (`ANALYSIS_LOCK`)**:
   - Added `ANALYSIS_LOCK = threading.Lock()` in `app.py`.
   - Guaranteed single-worker execution: only one analysis (image or video) runs at a time.
   - When busy, concurrent requests receive HTTP 503 Service Unavailable with friendly guidance.
2. **Asynchronous Job System**:
   - `VideoJob` class with states: `queued`, `processing`, `done`, `failed`, `cancelled`.
   - Thread-safe `JOB_QUEUE` processed by background daemon `WORKER_THREAD`.
   - `POST /api/video/jobs` creates and enqueues job, returns `job_id` with HTTP 202 immediately.
   - `GET /api/video/jobs/<job_id>` returns live status, percent, stage, frame counter, and ETA.
   - `POST /api/video/jobs/<job_id>/cancel` sets `cancel_event` for graceful termination.
3. **Video Engine Hardening**:
   - Added `validate_video_file()` checking non-zero size, extension, OpenCV container opening, frame decoding test, duration limit (<=180s), and rotation metadata.
   - Added `apply_orientation()` using `CAP_PROP_ORIENTATION_META` and `cv2.rotate` to preserve upright portrait orientation.
   - Added `save_thumbnail()` saving local JPEG thumbnails to `static/uploads/thumbs/<job_id>/thumb_*.jpg`.
   - Added `progress_callback` and `cancel_flag` support inside `process_video()`.
4. **Temporal & Narration Upgrades**:
   - `services/temporal_service.py`: Added `merge_consecutive_segments()` to group consecutive frames into distinct timeline segments.
   - `services/video_narration_service.py`: Added `synthesize_video_narrative()` with strict temporal grounding and `generate_video_prediction()` returning structured hypotheses.
   - Cleaned out any "Narration asserts" self-citations in hypothesis evidence.

### Verification
- Created `tests/test_video_pipeline.py` with 13 unit tests.
- All 34 tests in project pass (`python -m unittest discover tests`).

---

## Iteration 3: Frontend UI, Live Progress & Interactive Timeline

### Implementation
1. **Unblocked Modality Switcher (`templates/index.html` & `static/js/app.js`)**:
   - Removed "Preview" badge from `#tabVideo`.
   - Made Video Mode a first-class view switching seamlessly between `#imageModeView` and `#videoModeView`.
2. **Left Column Ingestion & Live Progress Card**:
   - Formats supported: MP4, MOV, AVI, WEBM, MKV up to 150MB, max 3 min.
   - Drag-and-drop dropzone with instant client-side size and extension validation.
   - Video preview card with HTML5 `<video id="videoPreviewPlayer">` and metadata tags.
   - Checkbox: "Delete video file from server after analysis completes".
   - Live progress card with animated progress bar, percentage badge, stage label, frame counter (`Keyframes: X / Y`), ETA countdown, and "Cancel Analysis" button.
3. **Right Column Intelligence Cards**:
   - Chronological Video Story card with status pills.
   - What may happen next card with structured hypotheses cards (likelihood badge, rationale, evidence tags).
   - Interactive Scene Timeline card with merged chronological segments, representative thumbnail, duration, activity, and objects list.
   - **Click-to-Seek**: Clicking any segment or thumbnail seeks the HTML5 video player directly to that segment's `start_seconds`.
   - Expandable technical inspection tabs: Keyframe Samples gallery, Detected Changes, and Video Diagnostics.
   - "Download Results as JSON" button exporting complete structured results.
4. **Visual Styling (`static/css/style.css`)**:
   - Added styles for video player wrapper, live progress bar, timeline segment cards, thumbnails, seek pills, hypotheses cards, and keyframe grid matching the soft-light modern UI theme.

---

## Iteration 4: E2E Integration Testing & Acceptance Verification

### Implementation
1. Created automated E2E test script: `scripts/test_video_e2e.py`.
2. Created test videos suite in `data/test_videos/`:
   - `test_transition_activity.mp4`: 6.0s transition video (texting -> reading book).
   - `test_static_desk.mp4`: 4.0s static scene video.
   - `test_portrait.mp4`: 3.0s portrait orientation video.
   - `test_corrupt.mp4`: corrupted file with invalid header.
   - `test_empty.mp4`: 0-byte file.

### Verification Results
1. **Video Upload Validation**:
   - Corrupt video (`test_corrupt.mp4`): REJECTED -> HTTP 400 (`Corrupt or unreadable video file: OpenCV could not open container.`).
   - Empty video (`test_empty.mp4`): REJECTED -> HTTP 400 (`Uploaded video file is empty (0 bytes).`).
   - Portrait video (`test_portrait.mp4`): ACCEPTED -> HTTP 200 (`is_portrait=True`, `480x720`).
   - Valid video (`test_transition_activity.mp4`): ACCEPTED -> HTTP 200 (`duration=6.0s`, `1024x768`, `24.0 FPS`).
2. **Concurrency & Lock Guard**:
   - While video job was processing under `ANALYSIS_LOCK`, an image analysis request to `POST /api/analyze/image` immediately returned `HTTP 503 Service Unavailable` with message: *"Analysis engine is currently busy processing another request."*
3. **Progress Reporting & End-to-End Pipeline**:
   - Progress bar reported percentage (8% -> 22% -> 50% -> 92% -> 95% -> 100%), stage labels, frame counter, and ETA countdown.
   - Synthesized Video Story: *"The video opens showing a person wearing a long sleeved shirt in an indoor setting, texting while holding / interacting with a cell phone. At 3.0s, the activity shifts from using smartphone to reading. At 3.0s, the interaction transitions toward book. Prominent persistent elements include cup, potted plant, chair."*
   - Generated 3 ranked hypotheses with likelihoods, rationales, and evidence tags without self-referential narration citations.
   - Produced 2 merged timeline segments with local thumbnails.
4. **Cancellation Flow**:
   - Initiated job and issued cancellation via `POST /api/video/jobs/<id>/cancel`.
   - Worker immediately detected cancel flag, released lock, and updated status to `cancelled`.
5. **Image Invariance & Zero Regression**:
   - Ran `scripts/regression_snapshot.py` to produce `data/evaluation_results/snapshot_after_video.json`.
   - Compared against `data/evaluation_results/snapshot_before_video.json`: **EXACT 100% MATCH across all 17 benchmark images. ZERO regression.**
6. **Unit Tests**:
   - Ran `python -m unittest discover tests`: All 34 tests pass in 0.069s.
7. **Disk Space**:
   - Free disk space: 23 GiB (requirement: > 10 GiB).

---

## Final Acceptance Checklist

| Requirement | Status | Evidence / Verification |
| :--- | :---: | :--- |
| **1. UI Tab Unlocked & Usable** | **PASS** | Video tab in `#tabVideo` is fully functional; no preview modal blocker; switching between Image and Video views works cleanly. |
| **2. Drag-and-Drop & File Validation** | **PASS** | Accepts MP4, MOV, AVI, WEBM, MKV; rejects >150MB, >180s, 0-byte, and corrupt containers with clear user-facing errors. |
| **3. Live Progress & ETA Reporting** | **PASS** | Background worker updates stage, percentage, frame counter, and ETA countdown via polling `GET /api/video/jobs/<id>`. |
| **4. Clean Cancellation** | **PASS** | `POST /api/video/jobs/<id>/cancel` sets event, aborts processing, and resets UI gracefully. |
| **5. Concurrency Protection (Single Worker Lock)** | **PASS** | `ANALYSIS_LOCK` ensures only one image or video analysis runs at a time; concurrent requests receive HTTP 503. |
| **6. Grounded Chronological Story** | **PASS** | Narrative synthesizes real activity and object shifts; no false transitions for static scenes; no self-referential narration citations. |
| **7. Interactive Click-to-Seek Timeline** | **PASS** | Merges identical consecutive frames into segments; clicking any segment or thumbnail seeks HTML5 `<video>` player to timestamp. |
| **8. Structured Anticipation Hypotheses** | **PASS** | Returns 1–3 ranked hypotheses with likelihood labels, evidence tags, and rationales. |
| **9. Image Pipeline Invariance (Zero Regression)** | **PASS** | Snapshot comparison on all 17 benchmark images confirmed exact 100% match before and after video implementation. |
| **10. Fully Offline Execution** | **PASS** | Zero external CDNs, zero network access at runtime; models loaded from local directory on Apple Silicon MPS. |
| **11. Memory & Disk Management** | **PASS** | Automatic cleanup of temporary files; free disk maintained at 23 GiB (> 10 GiB limit). |
| **12. Test Coverage & Documentation** | **PASS** | 34 unit tests passing; README updated with Video Mode architecture and usage guide. |
