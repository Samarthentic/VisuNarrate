# VisuNarrate – An AI-Based Image, Video & Real-Time Scene Storytelling System

**Final-Year B.E. Major Project | Computer Science & Engineering (AI & ML)**

---

## 1. Project Overview & Problem Statement

Traditional computer vision applications are typically restricted to isolated perception tasks—such as single-label classification or static bounding-box object detection. These systems answer **"what"** is in an image, but fail to articulate **"why"** or **"how"** entities interact in context, and are incapable of forecasting **"what happens next"**.

**VisuNarrate** addresses this limitation by formulating an end-to-end multimodal storytelling framework that transforms raw visual data into:
1. **Contextual Scene Narratives**: Coherent, structured natural-language stories detailing the entities, interactions, and atmosphere within a visual scene.
2. **Predictive Reasoning**: Forward-looking estimations of the most probable subsequent activity or state transition based on affordances and dynamic cues.

---

## 2. Core Architectural Philosophy: 100% Local & Offline

A fundamental engineering requirement of VisuNarrate is **zero runtime cloud dependency**:
* **No External AI APIs**: Does not rely on OpenAI, Google Gemini, Anthropic Claude, or any third-party cloud inference services.
* **On-Device Inference**: AI models execute strictly on the user's local machine using downloaded open-source weights (`Salesforce/blip-image-captioning-base`).
* **Zero-CDN Web Architecture**: The Flask backend serves self-contained HTML5, CSS3, and JavaScript with no remote CDN script or font dependencies.
* **Apple Silicon Hardware Acceleration**: Automatically detects and leverages Apple's **Metal Performance Shaders (MPS)** on Apple Silicon (M1/M2/M3) chips, falling back gracefully to CPU.

---

## 3. The 6-Stage Modular Pipeline

VisuNarrate rejects naive single-step "image-to-text" generation in favor of an academically rigorous, multi-stage processing pipeline:

```
[ Visual Input ] 
       │
       ▼
 1. Ingestion & Validation    ──> MIME verification, corruption checks
       │
       ▼
 2. Preprocessing             ──> RGB normalization, LANCZOS aspect-ratio scaling
       │
       ▼
 3. Visual Understanding      ──> Local Vision-Language Model (BLIP) feature extraction
       │
       ▼
 4. Scene Representation      ──> Structured semantic graph (entities, actions, context)
       │
       ▼
 5. Narration Synthesis       ──> Cohesive natural language scene storytelling
       │
       ▼
 6. Predictive Reasoning      ──> Affordance analysis & likely next state forecasting
       │
       ▼
[ Dual Output Stream ]        ──> (1) Current Scene Story  (2) Predicted Next Action
```

---

## 4. Current Implementation Scope

| Module | Status | Description |
| :--- | :--- | :--- |
| **Image Engine** | **Active (Production)** | Ingestion, validation, preprocessing, evidence fusion, and dual-output storytelling (Now + Next) for static scene imagery. |
| **Video Engine** | **Active (Production)** | Asynchronous video processing, temporal redundancy filtering, segment timeline merging, synchronized HTML5 click-to-seek, chronological narrative synthesis, and temporal future anticipation. |
| **Live Mode** | **Active (Production)** | Real-time live camera feed ingestion via browser `getUserMedia`, volatile in-memory processing with zero disk writes, pixel-diff redundancy gating, rolling event log, and session story synthesis. |

---

## 5. Repository Structure

```text
VisuNarrate/
├── run_demo.sh                 # Single-command executable demo launcher with preflight checks
├── app.py                      # Main Flask application server & API routes (Port 5001)
├── requirements.txt            # Python dependencies
├── README.md                   # Academic project documentation
├── .gitignore                  # Git ignore rules for virtual environments & models
│
├── core/
│   ├── __init__.py
│   ├── image_engine.py         # Static image preprocessing & multimodal evidence fusion
│   ├── video_engine.py         # Adaptive keyframe extraction & video processing engine
│   └── live_engine.py          # Real-time in-memory live camera processing engine
│
├── ai/
│   ├── __init__.py
│   ├── vision_model.py         # Local BLIP captioner loader (MPS/CPU, local_files_only)
│   ├── object_detector.py      # Local YOLOv8n object detector loader (MPS/CPU)
│   └── vqa_model.py            # Local BLIP VQA question answering loader (MPS/CPU)
│
├── services/
│   ├── __init__.py
│   ├── evidence_fusion.py      # Cross-verification and conflict resolution matrix
│   ├── narration_service.py    # Natural language scene storytelling service
│   ├── prediction_service.py   # Affordance-based future activity prediction
│   ├── temporal_service.py     # Video temporal aggregation & transition tracker
│   └── video_narration_service.py # Video timeline and temporal narrative synthesis
│
├── templates/
│   └── index.html              # Modern, self-contained dashboard UI (Image, Video, Live)
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark-slate visual stylesheet & live telemetry
│   ├── js/
│   │   └── app.js              # Client-side UI controller (Drag & Drop, Video, Live Camera)
│   └── uploads/                # Local storage for validated ingested media (0 live frames written)
│
├── models/
│   ├── blip_base/              # Local weights for BLIP Captioning (223.97M params, 855 MB)
│   ├── yolo/                   # Local weights for YOLOv8n (3.16M params, 6.2 MB)
│   └── blip_vqa/               # Local weights for BLIP VQA (361.23M params, 1.47 GB)
│
├── data/
│   ├── evaluation_images/      # 17-image benchmark suite for regression validation
│   ├── test_videos/            # Video test clips (including real footage and rotated clips)
│   └── evaluation_results/     # Baseline and regression snapshots
│
├── docs/
│   ├── ARCHITECTURE.md         # Full system architecture, Mermaid diagrams & memory budget
│   ├── DEMO_SCRIPT.md          # 6–8 min step-by-step presentation script & Q&A guide
│   └── final_progress.md       # Real-time phase verification log and telemetry metrics
│
├── scripts/
│   ├── preflight.py            # Rigorous pre-demo verification (Python, RAM, disk, models, tests)
│   ├── regression_snapshot.py  # 17-image snapshot generation and verification tool
│   ├── test_live_stream.py     # In-memory live streaming mock verification script
│   └── evaluate_image_pipeline.py # Benchmarking and latency instrumentation harness
│
└── tests/
    ├── test_system_info.py     # System info endpoint & timing instrumentation unit tests
    ├── test_live_pipeline.py   # Live engine & live streaming unit tests
    ├── test_image_pipeline.py  # Image engine unit tests
    └── test_video_pipeline.py  # Video engine & job lifecycle unit tests
```

---

## 6. Setup & Execution Guide

### Prerequisites
* Python 3.10+ (Tested on Python 3.12.2 on macOS ARM64 / Apple M1)
* Apple Silicon (M1/M2/M3/M4) recommended for hardware MPS acceleration
* Minimum 10 GB free disk space
* Minimum 1.0 GB free/available RAM

### Quick Start (Single-Command Demo Launch)
```bash
# 1. Clone repository and navigate to folder
cd VisuNarrate

# 2. Make launcher executable and run
chmod +x run_demo.sh
./run_demo.sh
```
`run_demo.sh` automatically:
1. Enforces strict offline environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`).
2. Cleans up any stale process on port 5001.
3. Executes `scripts/preflight.py` to verify dependencies, model files, memory, and tests.
4. Starts the server on `http://127.0.0.1:5001` and opens your default browser.

### Manual Execution
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

python app.py
# Access dashboard at http://127.0.0.1:5001
```

---

## 7. Verification & Offline Guarantee

To test the application under true offline conditions:
1. Turn off your Wi-Fi or disconnect your ethernet cable.
2. Run `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python app.py`.
3. Open `http://127.0.0.1:5000` — all perception models, evidence fusion logic, and predictive reasoning pipelines function completely without network access.

---

## 8. Predictive Reasoning Architecture & Academic Scope

VisuNarrate implements an evidence-aware **Predictive Reasoning Engine** (`services/prediction_service.py`) that models plausible state transitions:

```text
CURRENT VISUAL STATE ──> OBSERVED ACTIVITY ──> ACTION AFFORDANCES ──> LIKELY NEXT ACTIVITY ──> REASONING TRACE
```

### Key Principles:
1. **Evidence-Grounded**: Operates strictly on `confirmed` and `supported` facts in the fused structured scene graph. Never predicts from unverified hallucinations.
2. **Action Affordance Modeling**: Maps observed spatial agents, held objects, and postures to physical interaction affordances (e.g. mobile device held $\rightarrow$ continuing screen interaction/viewing).
3. **Media Safety Gating**: Automatically suppresses temporal predictions on non-photographic media (`illustration_or_graphic`), preventing speculative hallucination on abstract artwork.
4. **Academic Scope**: In the Image Storytelling module, predictive narration estimates a plausible immediate continuation from a static visual snapshot. It does not claim temporal trajectory learning across frames, which will be introduced in subsequent Video and Webcam processing milestones.

## 9. Image Module Evaluation & Performance Benchmarking

To benchmark the image storytelling and predictive reasoning pipeline with hardware-synchronized timing:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python scripts/evaluate_image_pipeline.py
```

### Rigorous Timing Methodology:
The evaluation harness strictly isolates three distinct execution phases:
1. **Cold Initialization**: Time required to load all model weights (`BLIP`, `YOLOv8n`, `BLIP-VQA`) from local disk into host RAM and GPU memory.
2. **First Inference (JIT Tracing & Shader Compilation)**: The very first forward pass through the multi-model pipeline. On Apple Silicon, PyTorch MPS lazily compiles Metal shaders and initializes GPU memory pools during this run.
3. **Repeated Warm Inference**: Subsequent forward passes evaluated over 3 consecutive runs per image using hardware-level synchronization (`torch.mps.synchronize()`). Reports **Median**, Mean, Min, Max, and P95 latency.

### Verified Benchmark Performance (Apple M1 / MPS Backend):
* **Cold Initialization**: $\approx 19.3\text{ – }21.9\text{ seconds}$ (One-time process startup).
* **First Inference Pass**: $\approx 19.4\text{ – }19.7\text{ seconds}$ (One-time Metal kernel compilation).
* **Warm Image Inference (Overall Median)**: **$\approx 2.58\text{ seconds}$** *(BLIP: 1.1–2.0s, YOLO: 21–37ms, VQA: 0.5–1.5s)*.
* **Warm Image Inference (Overall Mean)**: **$\approx 2.69\text{ seconds}$**.
* **P95 Warm Latency**: **$\approx 4.32\text{ seconds}$**.

---

## 10. Real-World Image Dataset Evaluation & Human Review Protocol

### Why Real-World Evaluation is Essential:
Academic rigor requires evaluating the multimodal perception, evidence fusion, and storytelling pipeline across diverse photographic conditions rather than isolated test samples. The dataset directory `data/evaluation_images/` supports 10 distinct real-world categories (indoor, outdoor, mobile phone use, laptop use, reading, beverage interaction, social scenes, landscapes, still life, and abstract graphics).

### Human Evaluation vs. Model Telemetry:
* **Model Telemetry (Automated)**: Measures hardware execution metrics (latency in milliseconds, question gating counts, conflicts detected, and heuristic diagnostic flags).
* **Empirical Human Evaluation**: Evaluates semantic accuracy, narrative quality, and action plausibility using a standardized 3-point academic rating scale:
  * **Scene, Object & Activity Correctness**: `0` = Incorrect, `1` = Partially Correct, `2` = Correct.
  * **Narration Quality**: `0` = Poor, `1` = Acceptable, `2` = Strong.
  * **Prediction Plausibility**: `0` = Implausible/Incorrect, `1` = Plausible but Uncertain, `2` = Strongly Plausible.
  * **Hallucination Detection**: `true` / `false`.

### Academic Scope on Single-Image Prediction Accuracy:
A single static image cannot establish future ground truth with mathematical certainty. Therefore, predictive narration is evaluated on **affordance plausibility and contextual consistency** rather than claiming speculative trajectory accuracy. Controlled empirical verification is performed by human judges rating the logical validity of projected next actions.

---

## 11. Predictive Reasoning Taxonomy & Evidence Refinement (Stage 4.8)

### Explainable Prediction Taxonomy:
1. **Strong Object Interaction (`interaction_continuation` / `task_continuation`)**:
   * Grounded focal interactions with smartphones, laptops, books, beverages, food, or mobility equipment yield conservative, immediate task continuations.
   * Assigned status: `supported_prediction`.
2. **Social / Group Dynamics (`group_continuation`)**:
   * Multi-agent scenes without an individual dominant focal action yield group-level social continuity.
   * Assigned status: `tentative_prediction`.
3. **Static Environmental Scenes (`environmental_static` / `insufficient_evidence`)**:
   * Person-absent natural landscapes or still-life scenes refuse speculative forecasting.
   * Assigned status: `insufficient_evidence`.
4. **Graphic / Abstract Media Safety Gating (`suppressed_graphic`)**:
   * Synthetic graphics and artwork are gated from physical temporal forecasting.
   * Assigned status: `suppressed`.
5. **Ambiguous Activity / Posture Only (`ambiguous_activity`)**:
   * Posture-only or underspecified actions yield conservative state continuations.
   * Assigned status: `tentative_prediction`.

### 15-Image Real-World Benchmark Results (Offline / Apple M1 MPS):
* **Total Evaluated Images**: 15 (3 Synchronized Warm Passes Per Image).
* **Overall Median Warm Latency**: **`2.43 seconds`** (`2433.14 ms`).
* **Overall Mean Warm Latency**: **`2.52 seconds`** (`2522.78 ms`).
* **Overall P95 Warm Latency**: **`3.08 seconds`** (`3075.42 ms`).
* **Supported Predictions**: **9 / 15** (60.0%).
* **Tentative Predictions**: **1 / 15** (6.7% — group social scene).
* **Insufficient Evidence (Safe Refusal)**: **3 / 15** (20.0% — static landscapes/still-life).
* **Suppressed (Media Gating)**: **2 / 15** (13.3% — 2D graphics/illustrations).
* **Diagnostic Prediction Weaknesses**: Reduced from 7 to **1**.

---

## 12. Video Storytelling Architecture & Temporal Redundancy Filter (Stage 5.0 & 5.1)

### Architectural Overview:
Phase 2 reuses the frozen multimodal ImageEngine perception backend across sampled video frames, enhanced by a lightweight temporal redundancy filter and decoupled warmup:

```
VIDEO STREAM (Local File)
  ↓ [Model Pre-Warmup & Metal Shader Tracing]
OPENCV STREAM INGESTION & PROPERTY INSPECTION
  ↓
ADAPTIVE SAMPLING (~1.0 FPS / Max Frame Constraints)
  ↓
TEMPORAL REDUNDANCY EVALUATION (Visual Diff + Fast YOLO Validation)
  ├── REDUNDANT (50% Frames) -> Inherit Stable State (~45ms vs ~2700ms)
  ├── KEY_CHANGE -> Full Multimodal Inference on State Shift
  └── FULL_ANALYSIS -> Initial & Complex Scene Validation
  ↓
TEMPORAL AGGREGATION & CHANGE DETECTION (TemporalService)
  ↓ [State Transitions & Stable Facts Extraction]
CHRONOLOGICAL VIDEO NARRATIVE SYNTHESIS (VideoNarrationService)
```

### Stage 5.1 Performance Optimization & Redundancy Filter:
1. **Model Pre-Warmup & JIT Tracing**:
   * Pre-warms weights and executes an in-memory synthetic trace to compile Apple Silicon Metal shader kernels before video processing commences.
   * Eliminates the 30+ second cold initialization penalty from video frame analysis measurements.
2. **Dual-Tier Redundancy Filter**:
   * **Tier 1 (Visual Difference)**: Downscales adjacent frames to $64 \times 64$ grayscale and calculates normalized pixel delta.
   * **Tier 2 (Semantic Safety Verification)**: If pixel delta $< 0.05$, runs the lightweight detector (~25ms). If person count and object classes match, marks the frame `REDUNDANT` and inherits the previous scene representation.
   * **Semantic Priority Rule**: If any visual or object shift occurs, forces `KEY_CHANGE` / `FULL_ANALYSIS`.

### Benchmark Validation (Controlled 6-Second Transition Video):
* **Video Stream**: 144 frames (6.0s @ 24 FPS, 1024x768).
* **Sampled Frames**: 6 frames (1.0 FPS uniform sampling).
* **Model Initialization**: `20.35 s` *(Cold model weights)*.
* **Metal Shader JIT Warmup**: `20.62 s` *(Initial graph tracing)*.
* **Actual Video Processing Time**: **`11.38 seconds`** *(Reduced from 50.72s)*.
* **Full Multimodal Analyses**: `2 / 6 frames` (`0.0s` initial, `1.0s` detail check).
* **Key Changes Triggered**: `1 / 6 frames` (`3.0s` transition to reading).
* **Redundant Frames Filtered**: **`3 / 6 frames` (50.0% compute saved)** at ~45–59ms per redundant frame.
* **Detected Transitions**:
  * `[3.0s] activity_change`: *Subject transitions from using smartphone to reading.*
  * `[3.0s] interaction_change`: *Interaction shifts from cell phone to book.*
* **Synthesized Narrative**:
  > *"The video initially captures a person wearing a long sleeved shirt in an indoor environment while holding / interacting with a cell phone. As the scene progresses at 3.0s, the activity transitions from using smartphone to reading. At 3.0s, the focus shifts as the subject begins interacting with book. Key persistent elements identified across the timeline include cup, potted plant, chair, cell phone."*

### Academic Scope & Limitations:
1. **Frame Sampling vs. Optical Flow / 3D CNNs**: Stage 5.1 uses explainable pixel delta heuristics and semantic state caching rather than dense spatiotemporal attention or optical flow.
2. **Deterministic Equivalence**: Redundancy decisions strictly prioritize semantic consistency over raw pixel differences.
3. **100% Local & Offline**: Fully self-contained without external API dependencies.

---

## 14. Live Mode Architecture & In-Memory Privacy Guarantee

Live Mode transforms VisuNarrate into a real-time conversational observer using the user's browser camera:
1. **Client-Side Capture via `getUserMedia`**:
   * Uses standard WebRTC browser camera APIs directly into an HTML5 `<canvas>`.
   * Live frames are converted to JPEG buffers and transmitted via `POST /api/live/analyze`.
   * Non-overlapping request loop: frames are dispatched sequentially; the next capture is only triggered after the prior inference cycle finishes.
2. **Strict In-Memory Execution (Zero Disk Writes)**:
   * Frames are decoded directly from incoming memory streams via `cv2.imdecode`.
   * **Absolute privacy guarantee**: live camera frames are never written to disk, filesystem, or static upload folders.
3. **Pixel-Diff Redundancy Gating**:
   * Downsamples the incoming frame to $64 \times 64$ grayscale and computes mean absolute difference against the prior frame.
   * If delta $< 0.04$, neural model inference is safely bypassed. Telemetry returns in $\approx 2\text{ ms}$ with cached state, preserving CPU/GPU thermal headroom.
4. **Ring Buffer & Inactivity Garbage Collection**:
   * Bounded in-memory ring buffer holds at most 60 historical snapshot descriptors per session.
   * Inactive sessions are automatically evicted after 15 minutes to guarantee zero memory leaks on 8 GB RAM systems.
5. **Session Narrative Synthesis**:
   * Users can request a synthesized narrative (`POST /api/live/session/<id>/summary`) at any point, summarizing chronological state transitions into a coherent session story with future-action prediction.

---

## 15. Technical Analysis Panel & Plain-English Glossary

VisuNarrate prioritizes complete transparency for academic evaluators and users:
* **Measured Runtime Telemetry**:
  * Every analysis card (Image, Video, Live) features a collapsible **Technical Analysis** panel showing measured wall-clock latencies broken down per stage: Preprocessing, YOLO Detection, BLIP Captioning, BLIP VQA, Evidence Fusion, and Narration/Prediction synthesis.
  * One-click **"Copy Technical JSON"** button exports full machine-readable performance metrics.
* **Live System Specs Dashboard**:
  * Accessible via the top navigation bar (`GET /api/system/info`), presenting dynamic host hardware details, MPS accelerator status, parameter counts per model (BLIP: 223.97M, YOLOv8n: 3.16M, BLIP VQA: 361.23M -> Total: 588.36M), disk memory footprints, and free RAM/disk space.
* **Interactive Plain-English Glossary (13 Terms)**:
  * Includes searchable definitions and specific "In VisuNarrate" implementation explanations for:
    * **Multimodal**, **MPS (Metal Performance Shaders)**, **Pretrained Model**, **VQA (Visual Question Answering)**, **Object Detection**, **Bounding Box**, **IoU (Intersection over Union)**, **Confidence Score**, **Inference**, **Latency**, **Ring Buffer**, **Zero-Epoch**, and **Evidence Fusion**.

---

## 16. Scientific Rigor: Pretrained Inference & Zero Epochs

VisuNarrate explicitly discloses that:
* **The system performs on-device inference using frozen pretrained foundation models.**
* **Zero training epochs were executed locally (`epochs = 0`, no local loss curves or training runs).**
* Rather than retraining broad foundational weights, the core scientific and engineering contribution of VisuNarrate is the **Evidence Fusion Engine**, which reconciles spatial detections, cross-attentive captions, and targeted VQA queries to eliminate visual hallucination and enforce strictly grounded temporal predictions.








