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
| **Image Engine** | **Active (Phase 1)** | Ingestion, validation, preprocessing, and modular pipeline scaffolding for static scene imagery. |
| **Video Engine** | **Planned (Phase 2)** | Frame decomposition, temporal tracking, and continuous narrative buffer across video files. |
| **Live Webcam** | **Planned (Phase 3)** | Real-time camera feed ingestion, low-latency sampling, and live streaming scene descriptions. |

---

## 5. Repository Structure

```text
VisuNarrate/
├── app.py                      # Main Flask application server & API routes
├── requirements.txt            # Python dependencies
├── README.md                   # Academic project documentation
├── .gitignore                  # Git ignore rules for virtual environments & models
│
├── core/
│   ├── __init__.py
│   ├── image_engine.py         # Static image preprocessing & orchestration
│   ├── video_engine.py         # (Phase 2) Temporal video processing engine
│   └── stream_engine.py        # (Phase 3) Real-time webcam processing engine
│
├── ai/
│   ├── __init__.py
│   └── vision_model.py         # Local model loader (MPS/CPU, local_files_only)
│
├── services/
│   ├── __init__.py
│   ├── narration_service.py    # Natural language scene storytelling service
│   └── prediction_service.py   # Affordance-based future activity prediction
│
├── templates/
│   └── index.html              # Modern, self-contained dashboard UI
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark-slate academic visual stylesheet
│   ├── js/
│   │   └── app.js              # Client-side drag-and-drop & API communication
│   └── uploads/                # Local storage for validated ingested media
│
├── models/
│   └── blip_base/              # Target directory for local model weights
│
├── data/
│   └── sample_images/          # Benchmark and test imagery
│
└── tests/
    └── __init__.py             # Unit and integration test suite
```

---

## 6. Setup & Execution Guide

### Prerequisites
* Python 3.10+ (Tested on Python 3.12.2 on macOS ARM64 / Apple M1)
* Git

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd VisuNarrate
```

### Step 2: Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application Server
```bash
python app.py
```

### Step 5: Access the Local Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
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







