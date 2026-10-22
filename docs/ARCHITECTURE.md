# VisuNarrate - System Architecture & Engineering Specifications

VisuNarrate is an on-device multimodal reasoning system designed for Apple Silicon (macOS). It ingests static imagery, video files, and real-time live camera streams to generate contextual narratives and evidence-grounded future-action predictions.

---

## 1. High-Level System Architecture

All three operational pipelines (**Image Mode**, **Video Mode**, **Live Mode**) execute against shared, persistent model singletons managed behind a non-blocking mutex (`ANALYSIS_LOCK`).

```mermaid
graph TD
    subgraph ClientLayer["Client Layer (Web UI - Vanilla JS / HTML5 / CSS3)"]
        UI_Image["Image Mode (Upload / Drag & Drop)"]
        UI_Video["Video Mode (Upload / Timeline / HTML5 Player)"]
        UI_Live["Live Mode (getUserMedia / In-Memory Stream)"]
        UI_Panels["Technical Analysis Panel & Plain-English Glossary"]
    end

    subgraph ServerLayer["Application Server (Flask 3.0 / Python 3.12)"]
        API_Image["POST /api/analyze"]
        API_Video["POST /api/video/upload & /api/video/job/:id"]
        API_Live["POST /api/live/analyze & /api/live/session/:id/summary"]
        API_Sys["GET /api/system/info & /api/health"]
        Lock["ANALYSIS_LOCK (Non-Blocking Concurrency Control)"]
    end

    subgraph CoreEngines["Core Processing Engines"]
        Eng_Image["ImageEngine (Evidence Fusion & Prediction)"]
        Eng_Video["VideoEngine (Adaptive Keyframes & Temporal Aggregation)"]
        Eng_Live["LiveEngine (In-Memory Ring Buffer & Redundancy Gate)"]
    end

    subgraph ModelSingletons["Shared Model Singletons (Apple Silicon MPS Acceleration)"]
        M_YOLO["YOLOv8n (3.16M params, 6.2 MB) - Object Detection"]
        M_BLIP["BLIP Base (223.97M params, 855 MB) - Captioning"]
        M_VQA["BLIP VQA (361.23M params, 1.47 GB) - Visual Question Answering"]
    end

    subgraph KnowledgeLayer["Grounded Reasoning & Verification"]
        Fusion["Evidence Fusion Engine (Cross-Verification Matrix)"]
        Pred["Prediction Engine (Evidence-Grounded Next-Action Hypotheses)"]
        Narr["Action-Prioritized Narration Engine"]
    end

    UI_Image --> API_Image
    UI_Video --> API_Video
    UI_Live --> API_Live
    UI_Panels --> API_Sys

    API_Image --> Lock
    API_Video --> Lock
    API_Live --> Lock

    Lock --> Eng_Image
    Lock --> Eng_Video
    Lock --> Eng_Live

    Eng_Image --> ModelSingletons
    Eng_Video --> ModelSingletons
    Eng_Live --> ModelSingletons

    ModelSingletons --> Fusion
    Fusion --> Narr
    Fusion --> Pred
```

---

## 2. Multimodal Data Flow Pipeline

From raw sensory input to grounded linguistic narration and predictive hypotheses:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Camera
    participant UI as VisuNarrate Frontend
    participant Server as Flask API & Lock
    participant YOLO as YOLOv8n (MPS)
    participant BLIP as BLIP Captioner (MPS)
    participant VQA as BLIP VQA (MPS)
    participant Fusion as Evidence Fusion Engine
    participant Reasoning as Narration & Prediction

    User->>UI: Submit Input (Image, Video, or Live Frame)
    UI->>Server: HTTP POST (Multipart Image/Video or In-Memory JPEG)
    Note over Server: Acquire ANALYSIS_LOCK (Return 503 if busy)
    
    par Spatial Detection
        Server->>YOLO: Detect Objects & Bounding Boxes
        YOLO-->>Server: Detected Objects, Counts, Centroids, BBoxes
    and Candidate Semantics
        Server->>BLIP: Generate Global Image Caption
        BLIP-->>Server: Candidate Natural Caption
    end

    Server->>VQA: Targeted Queries (Postures, Hands, Contact Affordances)
    VQA-->>Server: Targeted Semantic Answers (Confidence Gated)

    Server->>Fusion: Reconcile Detections + Caption + VQA Claims
    Note over Fusion: Detect Conflicts & Filter Hallucinations
    Fusion-->>Reasoning: Verified Scene Graph & Grounded Evidence Matrix

    Reasoning->>Reasoning: Synthesize Now Story (Action-Prioritized)
    Reasoning->>Reasoning: Generate Next Hypotheses (Ranked, Citing Exact Evidence)

    Reasoning-->>Server: Structured Payload + Stage Latencies
    Note over Server: Release ANALYSIS_LOCK
    Server-->>UI: JSON Response with Technical Timings
    UI-->>User: Render Story, Predictions, and Technical Diagnostics
```

---

## 3. Apple Silicon M1 (8 GB Unified Memory) Budget

VisuNarrate is engineered to operate safely within the strict resource envelope of an 8 GB Apple Silicon M1 machine.

| Component | Allocated Footprint | Description |
|---|---|---|
| **macOS & Core Services** | ~3.8 GB – 4.5 GB | Operating system, window server, system daemons. |
| **Model Weights (Unified Memory)** | **~2.33 GB** | Loaded once during warm-up and retained:<br>• BLIP Base: 855.1 MB<br>• BLIP VQA: 1,468.4 MB<br>• YOLOv8n: 6.2 MB |
| **Python Runtime & App State** | ~250 MB – 350 MB | Flask process, NumPy/Torch runtimes, UI serving. |
| **Working Buffer & PyTorch MPS Cache** | ~400 MB – 600 MB | Dynamic tensors during forward passes (cleared post-batch). |
| **Free / Available Headroom** | **~1.0 GB – 1.8 GB** | Safety margin ensuring zero macOS swap thrashing. |

### Memory Management Safeguards
1. **Single-Instance Mutex (`ANALYSIS_LOCK`)**: Enforces that only one inference job executes concurrently. Concurrent requests immediately return HTTP 503 with `"is_busy": True`, preventing simultaneous batch allocation.
2. **In-Memory Ring Buffers**: Live Mode maintains at most 60 historical frame descriptors in RAM; older frames are automatically evicted.
3. **Session Auto-Eviction**: Live sessions inactive for >15 minutes are garbage-collected automatically.

---

## 4. Security, Offline Integrity & Privacy

- **100% Offline Runtime**: The application runs completely disconnected from the Internet.
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `YOLO_OFFLINE=1`
  - No CDNs, remote fonts, external telemetry, or analytics scripts.
- **Strict Privacy in Live Mode**: Live camera frames are streamed via WebRTC/Canvas into volatile memory (`cv2.imdecode`) and **never written to the filesystem**. Session state exists only in RAM and can be wiped instantly via the UI Reset button.
- **Deterministic & Verifiable**: No generative LLM API calls, no hidden prompt mutations, and zero simulated/faked metrics.
