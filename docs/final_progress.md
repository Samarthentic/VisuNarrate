# VisuNarrate Final-Year Demo Progress Log

## Timebox Tracker
- **Mission Start**: 2026-10-22T12:16:43+05:30
- **Hard Limit**: 3 hours (deadline 2026-10-22T15:16:43+05:30)
- **Branch**: `final-demo`
- **Initial Baseline Tag**: `demo-safe-1`
- **Regression Baseline**: `data/evaluation_results/snapshot_before_final.json`

---

## Iteration 0: Phase 0 - Baseline & Cleanup
- **Start Time**: 2026-10-22T12:16:43+05:30
- **End Time**: 2026-10-22T12:27:00+05:30

### 1. Plan
1. Safety & branching: Verify clean tree, run unit tests, tag `demo-safe-1`, branch `final-demo`. (DONE)
2. Run regression snapshot baseline `snapshot_before_final.json`. (DONE)
3. Audit `app.js`, `index.html`, and `app.py` for any debug/demo hooks that force UI states (URL parameters, mock data, hard-coded progress values). Remove or guard them under an environment variable `VISUNARRATE_DEBUG_UI=0` (OFF by default). (DONE)
4. Fix sampling aliasing in `core/video_engine.py`: Sample at the center of each interval with a deterministic offset so periodic clips cannot make every sampled frame identical. Add unit test. (DONE)
5. Upgrade startup warm-up in `app.py` to use a realistic photo of a person from `data/evaluation_images/portrait_smartphone.jpg` so BLIP captioning, YOLO detection, and BLIP-VQA are all genuinely warm. Measure first real request after startup. (DONE)

### 2. Implementation
- Tag `demo-safe-1` created on commit `cefd740`; branch `final-demo` checked out.
- `snapshot_before_final.json` saved across all 17 benchmark images.
- Audited `app.js`: Removed fake `ui_state` test hooks (`mockFinishedResult`). Verified `index.html` has no artificial mocks.
- `calculate_sampling_plan` updated in `core/video_engine.py`: uses centered interval sampling with bounded deterministic jitter ($\pm 15\%$) to prevent periodic stroboscopic lock while preserving first and last frame anchors.
- Added `test_sampling_plan_anti_aliasing_offsets` to `tests/test_video_pipeline.py`.
- Updated `warm_up` in `app.py` to run full inference against `data/evaluation_images/portrait_smartphone.jpg` across BLIP base, YOLOv8n, and BLIP VQA.

### 3. Verification
- All 39 unit tests pass in 0.107s (`VISUNARRATE_SKIP_WARMUP=1`).
- Realistic warm-up execution: 30.83s on Apple Silicon MPS.
- First real request after warm-up: 4.89s (down from 24s cold latency).
- Sampling jitter test confirmed: internal sample frames have offset relative to uniform grid.

### 4. Score
- Phase 0: PASS.

---

## Iteration 1: Phase 1 - System Info & Timing Instrumentation
- **Start Time**: 2026-10-22T12:27:00+05:30
- **End Time**: 2026-10-22T12:35:00+05:30

### 1. Plan
1. Implement `GET /api/system/info` with strictly measured runtime metadata:
   - Python, PyTorch, Transformers, Ultralytics, OpenCV versions.
   - Device: Apple Silicon MPS availability and status.
   - Exact parameter count dynamically counted via `sum(p.numel())` on loaded model objects or architecture inspection (BLIP Base: 223.97M, YOLOv8n: 3.16M, BLIP VQA: 361.23M -> Total: 588.36M).
   - On-disk sizes measured via directory inspection (855.12MB, 6.25MB, 1468.43MB).
   - Real-time RAM, process RSS, and free disk capacity via `psutil` and `shutil`.
   - Explicit training disclosure: 0 training epochs run, fine-tuning: none, inference only.
2. Instrument `ImageEngine` with microsecond-level stage timings (`preprocessing_ms`, `captioning_ms`, `detection_ms`, `vqa_ms`, `fusion_ms`, `narration_ms`, `prediction_ms`, `total_ms`) and package into structured `technical` result block.
3. Instrument `VideoEngine` and async video job worker with `video_decode_ms`, `keyframe_sampling_ms`, `per_frame_avg_ms`, `temporal_aggregation_ms`, `narration_synthesis_ms`, and `total_ms`.
4. Add comprehensive unit tests in `tests/test_system_info.py`.

### 2. Implementation
- Updated `core/image_engine.py`: per-stage timestamps and structured `technical` dictionary returned without altering any legacy fields.
- Updated `core/video_engine.py` and `app.py`: `technical` telemetry added to video processing and job result packaging.
- Added `GET /api/system/info` endpoint in `app.py`.
- Renamed `live_webcam` module key in `/api/status` to `live`.
- Added `tests/test_system_info.py` covering system info schema, offline verification, memory validity, and technical telemetry fields.

### 3. Verification
- All 41 unit tests pass in 0.146s.
- `GET /api/system/info` returns 200 with 100% measured system metrics and 0-epoch inference disclosure.
- Technical block present in Image and Video analysis schemas.

### 4. Score
- Phase 1: PASS.


