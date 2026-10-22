# VisuNarrate - Final Demo Presentation Script

**Project Title:** VisuNarrate: Multimodal Scene Understanding & Future-State Temporal Prediction  
**Platform:** macOS Apple Silicon (M1, 8 GB Unified Memory)  
**Execution Environment:** Fully Offline, Local On-Device Inference  
**Presentation Target Time:** 6–8 Minutes  

---

## 1. Quick Setup & Pre-Flight (1 Minute Before Demo)

1. Open Terminal in the project root directory:
   ```bash
   cd ~/Downloads/Veridyn/VisuNarrate
   ./run_demo.sh
   ```
2. Verify in the terminal output:
   - `[PASS]` on all preflight checks (Python 3.12, Free Disk > 10GB, RAM >= 1.0GB, Local Weights Present, Offline Flags Enforced).
   - Server initializes on `http://127.0.0.1:5001`.
   - Browser opens automatically to `http://127.0.0.1:5001`.

---

## 2. Walkthrough Sequence & Click-by-Click Instructions

### Step 1: Architecture Overview & Offline Guarantee (1:00)
- **Visual:** VisuNarrate UI home screen showing the three tabs: **Image**, **Video**, **Live**.
- **Action:** Point out the top header badge: `"Offline: Active"` and click **"System Specs"** in the top navigation bar.
- **Talking Points:**
  > *"Welcome to VisuNarrate. Our system delivers multimodal visual perception, narrative synthesis, and evidence-grounded temporal prediction completely on-device without cloud dependencies or external APIs.*
  > *As shown in our live System Specs panel, our total model footprint is 588.36 Million parameters (BLIP Base: 223.97M, YOLOv8n: 3.16M, BLIP VQA: 361.23M), accelerated directly by Apple Silicon MPS unified memory.*
  > *Strict offline environment variables are enforced (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`), ensuring total privacy and air-gapped security."*

---

### Step 2: Image Mode - Evidence Fusion & Next-Action Prediction (2:00)
- **Visual:** Image Tab.
- **Action:** 
  1. Select or drag-and-drop a benchmark image (e.g., `data/test_images/1_two_people_at_desk_with_laptops_and_coffee_cups.jpg`).
  2. Click **"Analyze Image"**.
  3. Review the result cards:
     - **Story ("Now"):** *"Two people are sitting at a desk, working on laptops, with coffee cups nearby."*
     - **Prediction ("Next"):** 1–3 ranked hypotheses with evidence citations.
     - **Technical Analysis Card:** Per-stage latencies (Preprocessing, YOLO Detection, BLIP Captioning, BLIP VQA, Evidence Fusion).
- **Talking Points:**
  > *"When an image is submitted, VisuNarrate avoids single-model hallucination through a multi-stage evidence fusion pipeline:*
  > *1. YOLOv8n detects and locates bounding boxes and class distributions (people, laptops, cups).*
  > *2. BLIP generates a global candidate caption.*
  > *3. BLIP VQA resolves targeted spatial, postural, and affordance queries to verify physical relationships.*
  > *4. Our Evidence Fusion Engine reconciles spatial detections with linguistic claims. Notice our prediction panel: we generate 1 to 3 structured hypotheses, each citing exact physical evidence. We never predict ungrounded actions like 'driving away' when only cups and desks are detected."*

---

### Step 3: Video Mode - Temporal Analysis & Interactive Timeline (2:00)
- **Visual:** Video Tab.
- **Action:**
  1. Click the **Video** tab.
  2. Upload a test clip (e.g., `data/test_videos/real/short_clip.mp4` or sample clip).
  3. Observe the live progress bar reporting keyframe extraction and neural analysis stages.
  4. View the generated results:
     - **Interactive Timeline:** Click individual keyframe thumbnails to jump to that timestamp in the synchronized HTML5 video player.
     - **Session Narrative:** Synthesizes temporal shifts and activities across scenes.
     - **Temporal Prediction:** Grounded in observed visual trajectories.
- **Talking Points:**
  > *"Video understanding on an 8 GB unified memory budget presents severe compute and memory constraints. VisuNarrate solves this with an adaptive keyframe extraction engine:*
  > *It detects structural scene changes via normalized histogram variance and centered interval sampling with deterministic jitter, analyzing only pivotal keyframes.*
  > *Notice that clicking any thumbnail in the timeline immediately seeks the video player to that exact moment.*
  > *Our temporal prediction incorporates observed state transitions rather than treating frames as isolated events."*

---

### Step 4: Live Mode - In-Memory Live Camera Analysis (1:30)
- **Visual:** Live Tab.
- **Action:**
  1. Click the **Live** tab.
  2. Explain the **Privacy Guarantee Notice**: *"Live camera frames are processed in-memory only and never written to disk."*
  3. Click **"Start Live Camera"** (accept browser permissions).
  4. Select sampling interval: **3s** or **5s**.
  5. Perform an action before the camera (e.g., hold up a cup or phone, then place it down).
  6. Watch the **Now** narration and **Next** prediction update dynamically, appending snapshots to the **Rolling Log**.
  7. Click **"Generate Session Summary"** to produce a consolidated narrative of observed activities.
  8. Click **"Stop Live Camera"**.
- **Talking Points:**
  > *"In Live Mode, VisuNarrate streams frames directly from the browser's MediaStream via `getUserMedia`. Crucially, zero frames are written to disk; frames are ingested into volatile memory, decoded directly via OpenCV, and managed in a fixed-size ring buffer capped at 60 frames.*
  > *To preserve battery and thermal headroom on Apple Silicon, we employ a pixel-diff redundancy gate: if the scene is static, neural inference is bypassed, returning cached telemetry in under 3 milliseconds.*
  > *When a new action occurs, the models execute, delivering real-time scene understanding and temporal forecasts."*

---

### Step 5: Technical Analysis & Plain-English Glossary (0:30)
- **Action:**
  1. Expand the **Technical Analysis Panel** in any mode.
  2. Click **"Copy Technical JSON"** to show machine-readable instrumentation.
  3. Click **"Glossary"** in the top navigation bar. Show the search filter (type "VQA", "MPS", "IoU").
- **Talking Points:**
  > *"Transparency is core to VisuNarrate. Every result includes an exact breakdown of wall-clock latencies across each pipeline stage. Evaluators and users can open our built-in Glossary to understand technical terminology in accessible, plain-English definitions with specific explanations of how each concept is used in VisuNarrate."*

---

## 3. Anticipated Questions & Crisp Technical Answers

### Q1: "Why did you use pretrained models instead of fine-tuning or training from scratch? Why are epochs = 0?"
> **Answer:** *"VisuNarrate is an applied multimodal reasoning and evidence fusion system, not a raw weights training benchmark. Foundation models like BLIP Base and YOLOv8n already contain broad semantic representations trained across millions of images (COCO, Conceptual Captions). Training from scratch on an M1 Mac with 8 GB of RAM is infeasible and would yield severely degraded generalization. Our primary engineering contribution is the **Evidence Fusion Engine**, which reconciles spatial detections, cross-attentive captions, and targeted VQA queries to eliminate visual hallucination and enforce strictly grounded temporal predictions. We explicitly report 0 epochs to maintain absolute scientific honesty."*

### Q2: "How do you prevent hallucinations in the narrative and predictions?"
> **Answer:** *"We enforce a strict verification matrix:
> 1. Cross-Verification: A candidate activity asserted by BLIP is verified against YOLO object detections and targeted BLIP VQA queries (e.g., verifying if hands are contacting an object).
> 2. Spatial Adjacency: Predictions require verified affordances. If an object is not detected or verified via VQA, it cannot appear in a hypothesis.
> 3. Evidence Traceability: Every prediction must cite the concrete detections or temporal transitions that support it. Ungrounded speculation is pruned or down-ranked to 'less likely'."*

### Q3: "How does the system fit within the 8 GB unified memory limit of the Apple M1?"
> **Answer:** *"Four architectural decisions maintain our memory footprint under 2.2 GB active RAM:
> 1. Model Reuse & Single Loading: BLIP Base, YOLOv8n, and BLIP VQA are loaded once into unified memory during warm-up and reused across requests.
> 2. Non-blocking Mutex (`ANALYSIS_LOCK`): Only one inference request runs at any time; concurrent calls receive a fast 503 response.
> 3. Ring Buffers: Live Mode uses a bounded 60-item in-memory ring buffer with automatic eviction.
> 4. Garbage Collection & Cache Purging: PyTorch MPS cache is flushed after batch processing."*

---

## 4. Live Demo Troubleshooting Guide

| Issue | Root Cause | Instant Fix |
|---|---|---|
| **Camera not starting in Live Mode** | Browser permission blocked or Safari insecure context | 1. Ensure accessing `http://127.0.0.1:5001` (not `0.0.0.0` or local IP).<br>2. Check camera icon in URL bar and set permission to "Always Allow". |
| **Port 5001 Already in Use** | Previous Flask instance still running | Run `./run_demo.sh` (it automatically terminates stale processes) or execute `lsof -ti :5001 \| xargs kill -9`. |
| **High Latency on First Query** | Metal Performance Shaders (MPS) shader compilation | The server runs an automatic warm-up pass on startup. If restarted, allow 3–5 seconds on the very first inference call. |
| **Out of Memory Warning in Preflight** | Heavy background apps running (e.g., Xcode, Docker) | Close unused Electron/heavy applications to free up unified memory. |
