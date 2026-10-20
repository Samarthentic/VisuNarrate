/**
 * VisuNarrate - Modern Visual AI Product Interaction Controller
 * Pure Vanilla JavaScript • Zero External CDN Dependencies • Fast Local Execution
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Workspaces & Navigation
    const imageModeView = document.getElementById("imageModeView");
    const videoModeView = document.getElementById("videoModeView");
    const tabImage = document.getElementById("tabImage");
    const tabVideo = document.getElementById("tabVideo");
    const tabWebcam = document.getElementById("tabWebcam");
    const toast = document.getElementById("toast");

    // Modal Preview Overlay for Video & Live tabs
    const previewModeNotice = document.getElementById("previewModeNotice");
    const previewModalTitle = document.getElementById("previewModalTitle");
    const previewModalText = document.getElementById("previewModalText");
    const btnReturnToImage = document.getElementById("btnReturnToImage");

    // DOM Elements - Ingestion & Preview
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const btnBrowsePrompt = document.getElementById("btnBrowsePrompt");
    const uploadProgress = document.getElementById("uploadProgress");

    const previewCard = document.getElementById("previewCard");
    const previewImage = document.getElementById("previewImage");
    const bboxCanvas = document.getElementById("bboxCanvas");
    const btnToggleBoxes = document.getElementById("btnToggleBoxes");
    const metaFilename = document.getElementById("metaFilename");
    const metaResolution = document.getElementById("metaResolution");
    const metaSize = document.getElementById("metaSize");
    const btnReplace = document.getElementById("btnReplace");
    const btnAnalyze = document.getElementById("btnAnalyze");

    // Loading & Intelligence Results Flow
    const analysisLoadingState = document.getElementById("analysisLoadingState");
    const loadingStageText = document.getElementById("loadingStageText");
    const intelligenceResultsFlow = document.getElementById("intelligenceResultsFlow");

    // Primary Intelligence Cards
    const sceneNarrationText = document.getElementById("sceneNarrationText");
    const sceneStateBadge = document.getElementById("sceneStateBadge");
    const predictNarrationText = document.getElementById("predictNarrationText");
    const predictStateBadge = document.getElementById("predictStateBadge");
    const predictReasoningContainer = document.getElementById("predictReasoningContainer");
    const predictReasoningList = document.getElementById("predictReasoningList");
    const predictSourceFactsList = document.getElementById("predictSourceFactsList");

    // Technical Inspection Tabs & Panes
    const techTabs = document.querySelectorAll(".tech-tab");
    const techPanes = document.querySelectorAll(".tech-pane");

    // Technical Pane Elements - Scene Understanding
    const statusSubject = document.getElementById("statusSubject");
    const valSubject = document.getElementById("valSubject");
    const sourceSubject = document.getElementById("sourceSubject");
    const statusActivity = document.getElementById("statusActivity");
    const valActivity = document.getElementById("valActivity");
    const sourceActivity = document.getElementById("sourceActivity");
    const statusEnvironment = document.getElementById("statusEnvironment");
    const valEnvironment = document.getElementById("valEnvironment");
    const sourceEnvironment = document.getElementById("sourceEnvironment");
    const statusInteractions = document.getElementById("statusInteractions");
    const valInteractions = document.getElementById("valInteractions");
    const sourceInteractions = document.getElementById("sourceInteractions");

    // Technical Pane Elements - YOLO Objects
    const yoloStateBadge = document.getElementById("yoloStateBadge");
    const yoloEmptyState = document.getElementById("yoloEmptyState");
    const yoloResultsContainer = document.getElementById("yoloResultsContainer");
    const yoloCountsSummary = document.getElementById("yoloCountsSummary");
    const yoloObjectsList = document.getElementById("yoloObjectsList");

    // Technical Pane Elements - BLIP-VQA
    const vqaStateBadge = document.getElementById("vqaStateBadge");
    const vqaEmptyState = document.getElementById("vqaEmptyState");
    const vqaResultsContainer = document.getElementById("vqaResultsContainer");
    const vqaRoutingNotice = document.getElementById("vqaRoutingNotice");
    const vqaGrid = document.getElementById("vqaGrid");
    const skippedDrawer = document.getElementById("skippedDrawer");
    const skippedCountSpan = document.getElementById("skippedCountSpan");
    const skippedList = document.getElementById("skippedList");

    // Technical Pane Elements - Evidence & Reasoning
    const fusionStateBadge = document.getElementById("fusionStateBadge");
    const fusionEmptyState = document.getElementById("fusionEmptyState");
    const fusionResultsContainer = document.getElementById("fusionResultsContainer");
    const sceneTypeBadge = document.getElementById("sceneTypeBadge");
    const sceneTypeDetail = document.getElementById("sceneTypeDetail");
    const presenceBadge = document.getElementById("presenceBadge");
    const presenceDetail = document.getElementById("presenceDetail");
    const conflictCountBadge = document.getElementById("conflictCountBadge");
    const conflictsList = document.getElementById("conflictsList");

    // Technical Pane Elements - Diagnostics
    const diagMode = document.getElementById("diagMode");
    const diagDevice = document.getElementById("diagDevice");
    const diagStatusBadge = document.getElementById("diagStatusBadge");
    const diagBlipTelemetry = document.getElementById("diagBlipTelemetry");
    const diagYoloTelemetry = document.getElementById("diagYoloTelemetry");
    const diagVqaTelemetry = document.getElementById("diagVqaTelemetry");
    const diagFusionTelemetry = document.getElementById("diagFusionTelemetry");

    // State Variables
    let currentUploadedFile = null;
    let currentDetectionResult = null;
    let isShowingBoxes = false;
    let loadingInterval = null;

    // =========================================================================
    // 1. Toast Notification Helper
    // =========================================================================
    function showToast(message, duration = 3500) {
        if (!toast) return;
        toast.textContent = message;
        toast.classList.remove("hidden");
        setTimeout(() => {
            toast.classList.add("hidden");
        }, duration);
    }

    // =========================================================================
    // 2. Modality Switcher & Preview Modals
    // =========================================================================
    tabImage.addEventListener("click", () => {
        tabImage.classList.add("active");
        tabVideo.classList.remove("active");
        tabWebcam.classList.remove("active");
        if (imageModeView) imageModeView.classList.remove("hidden");
        if (videoModeView) videoModeView.classList.add("hidden");
        if (previewModeNotice) previewModeNotice.classList.add("hidden");
    });

    tabVideo.addEventListener("click", () => {
        tabVideo.classList.add("active");
        tabImage.classList.remove("active");
        tabWebcam.classList.remove("active");
        if (videoModeView) videoModeView.classList.remove("hidden");
        if (imageModeView) imageModeView.classList.add("hidden");
        if (previewModeNotice) previewModeNotice.classList.add("hidden");
    });

    tabWebcam.addEventListener("click", () => {
        if (previewModeNotice) {
            previewModalTitle.textContent = "Live Scene Intelligence (Preview)";
            previewModalText.textContent = "Real-time camera scene intelligence and stream-based reasoning are scheduled for subsequent release.";
            previewModeNotice.classList.remove("hidden");
        }
    });

    if (btnReturnToImage) {
        btnReturnToImage.addEventListener("click", () => {
            if (previewModeNotice) previewModeNotice.classList.add("hidden");
            tabImage.classList.add("active");
            tabVideo.classList.remove("active");
            tabWebcam.classList.remove("active");
            if (imageModeView) imageModeView.classList.remove("hidden");
            if (videoModeView) videoModeView.classList.add("hidden");
        });
    }

    // =========================================================================
    // 3. Technical Inspection Tab Switcher
    // =========================================================================
    techTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            techTabs.forEach((t) => t.classList.remove("active"));
            techPanes.forEach((p) => {
                p.classList.remove("active");
                p.classList.add("hidden");
            });

            tab.classList.add("active");
            const targetPaneId = tab.getAttribute("data-pane");
            const targetPane = document.getElementById(targetPaneId);
            if (targetPane) {
                targetPane.classList.remove("hidden");
                targetPane.classList.add("active");
            }
        });
    });

    // =========================================================================
    // 4. Ingestion, Drag & Drop, and Reset
    // =========================================================================
    if (dropzone) {
        dropzone.addEventListener("click", (e) => {
            if (e.target !== btnBrowsePrompt) {
                fileInput.click();
            }
        });
    }

    if (btnBrowsePrompt) {
        btnBrowsePrompt.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("drag-over");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            handleFileUpload(dt.files[0]);
        }
    });

    btnReplace.addEventListener("click", () => {
        fileInput.value = "";
        previewCard.classList.add("hidden");
        dropzone.classList.remove("hidden");
        dropzonePrompt.classList.remove("hidden");
        currentUploadedFile = null;
        currentDetectionResult = null;
        isShowingBoxes = false;
        clearCanvas();
        btnToggleBoxes.disabled = true;
        btnToggleBoxes.classList.remove("active");
        btnToggleBoxes.querySelector("span").textContent = "Show Bounding Boxes";

        // Reset Primary Cards
        sceneNarrationText.innerHTML = `Upload an image and click <strong>Analyze Scene</strong> to generate a grounded description of the visual scene.`;
        sceneStateBadge.textContent = "Awaiting Analysis";
        sceneStateBadge.className = "status-pill status-awaiting";

        predictNarrationText.textContent = "Predictive reasoning infers plausible immediate next actions based on physical affordances and spatial interactions.";
        predictStateBadge.textContent = "Awaiting Prediction";
        predictStateBadge.className = "status-pill status-awaiting";
        if (predictReasoningContainer) predictReasoningContainer.classList.add("hidden");

        // Reset Technical Panes
        if (fusionEmptyState) fusionEmptyState.classList.remove("hidden");
        if (fusionResultsContainer) fusionResultsContainer.classList.add("hidden");
        if (yoloEmptyState) yoloEmptyState.classList.remove("hidden");
        if (yoloResultsContainer) yoloResultsContainer.classList.add("hidden");
        if (vqaEmptyState) vqaEmptyState.classList.remove("hidden");
        if (vqaResultsContainer) vqaResultsContainer.classList.add("hidden");

        valSubject.textContent = "-";
        valActivity.textContent = "-";
        valEnvironment.textContent = "-";
        valInteractions.textContent = "None Specified";

        if (diagBlipTelemetry) diagBlipTelemetry.textContent = "Idle";
        if (diagYoloTelemetry) diagYoloTelemetry.textContent = "Idle";
        if (diagVqaTelemetry) diagVqaTelemetry.textContent = "Idle";
        if (diagFusionTelemetry) diagFusionTelemetry.textContent = "Idle";
        if (diagStatusBadge) diagStatusBadge.textContent = "Ready";
    });

    async function handleFileUpload(file) {
        const allowedTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
        if (!allowedTypes.includes(file.type)) {
            showToast("Invalid format. Please upload a PNG, JPG, JPEG, or WEBP image.");
            return;
        }

        dropzonePrompt.classList.add("hidden");
        uploadProgress.classList.remove("hidden");

        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || "Failed to upload image.");
            }

            currentUploadedFile = result;
            displayPreview(result);
            showToast("Image ready for analysis.");
        } catch (error) {
            console.error("Upload error:", error);
            showToast(`Upload Error: ${error.message}`);
            dropzonePrompt.classList.remove("hidden");
            uploadProgress.classList.add("hidden");
        }
    }

    function displayPreview(data) {
        uploadProgress.classList.add("hidden");
        dropzone.classList.add("hidden");

        previewImage.src = data.url;
        metaFilename.textContent = data.filename;
        metaResolution.textContent = `${data.metadata.width} × ${data.metadata.height} px`;
        metaSize.textContent = `${data.metadata.format} • ${data.metadata.size_kb} KB`;

        previewCard.classList.remove("hidden");
    }

    // =========================================================================
    // 5. Bounding Box Rendering
    // =========================================================================
    if (btnToggleBoxes) {
        btnToggleBoxes.addEventListener("click", () => {
            if (!currentDetectionResult || !currentDetectionResult.objects) return;
            isShowingBoxes = !isShowingBoxes;
            if (isShowingBoxes) {
                btnToggleBoxes.classList.add("active");
                btnToggleBoxes.querySelector("span").textContent = "Hide Bounding Boxes";
                bboxCanvas.classList.remove("hidden");
                drawBoundingBoxes(currentDetectionResult.objects);
            } else {
                btnToggleBoxes.classList.remove("active");
                btnToggleBoxes.querySelector("span").textContent = "Show Bounding Boxes";
                bboxCanvas.classList.add("hidden");
                clearCanvas();
            }
        });
    }

    function drawBoundingBoxes(objects) {
        if (!bboxCanvas || !previewImage) return;

        const ctx = bboxCanvas.getContext("2d");
        const rect = previewImage.getBoundingClientRect();
        bboxCanvas.width = previewImage.clientWidth;
        bboxCanvas.height = previewImage.clientHeight;

        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

        const scaleX = bboxCanvas.width / (previewImage.naturalWidth || 1);
        const scaleY = bboxCanvas.height / (previewImage.naturalHeight || 1);

        objects.forEach((obj) => {
            const bbox = obj.bbox;
            if (!bbox) return;

            const [x1, y1, x2, y2] = bbox;
            const drawX = x1 * scaleX;
            const drawY = y1 * scaleY;
            const drawW = (x2 - x1) * scaleX;
            const drawH = (y2 - y1) * scaleY;

            ctx.strokeStyle = "#06b6d4";
            ctx.lineWidth = 2;
            ctx.strokeRect(drawX, drawY, drawW, drawH);

            const label = `${obj.label} ${Math.round((obj.confidence || 0) * 100)}%`;
            ctx.font = "600 11px system-ui, sans-serif";
            const textMetrics = ctx.measureText(label);
            const textWidth = textMetrics.width;

            ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
            ctx.fillRect(drawX, Math.max(0, drawY - 18), textWidth + 8, 18);

            ctx.fillStyle = "#000000";
            ctx.fillText(label, drawX + 4, Math.max(13, drawY - 4));
        });
    }

    function clearCanvas() {
        if (!bboxCanvas) return;
        const ctx = bboxCanvas.getContext("2d");
        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
    }

    // =========================================================================
    // 6. Analyze Scene Execution & Progressive Loading
    // =========================================================================
    btnAnalyze.addEventListener("click", async () => {
        if (!currentUploadedFile) {
            showToast("Please upload an image first.");
            return;
        }

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = `
            <div class="loading-spinner" style="width: 14px; height: 14px; border-width: 2px;"></div>
            <span>Analyzing...</span>
        `;

        // Start Animated Loading Experience
        if (analysisLoadingState) analysisLoadingState.classList.remove("hidden");
        if (intelligenceResultsFlow) intelligenceResultsFlow.style.opacity = "0.3";

        const stages = [
            { text: "Understanding visual context & scene type...", step: "lStep1" },
            { text: "Detecting spatial objects & actor boundaries...", step: "lStep2" },
            { text: "Interrogating targeted scene details (BLIP-VQA)...", step: "lStep3" },
            { text: "Synthesizing grounded scene narration...", step: "lStep4" },
            { text: "Predicting immediate next activity...", step: "lStep5" }
        ];
        let stageIdx = 0;
        loadingInterval = setInterval(() => {
            stageIdx = (stageIdx + 1) % stages.length;
            if (loadingStageText) loadingStageText.textContent = stages[stageIdx].text;
            stages.forEach((s, i) => {
                const el = document.getElementById(s.step);
                if (el) {
                    if (i <= stageIdx) el.classList.add("active");
                    else el.classList.remove("active");
                }
            });
        }, 600);

        try {
            const response = await fetch("/api/analyze/image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: currentUploadedFile.filename }),
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || "Scene analysis pipeline failed.");
            }

            const data = result.data;
            const telemetry = data.telemetry;
            const fusedScene = data.fused_scene || {};
            const sceneType = data.scene_type || {};
            const personPresence = data.person_presence || {};
            const yoloData = data.object_detection || {};
            const vqaEvidence = data.vqa_evidence || [];
            const skippedQuestions = data.skipped_questions || [];
            const conflicts = data.conflicts || [];
            const uncertaintyFlags = data.uncertainty_flags || [];

            currentDetectionResult = yoloData;

            // 1. Render "What's happening" (Current Scene Narration)
            sceneNarrationText.innerHTML = escapeHtml(data.scene_narration);
            const warmMs = telemetry.warm_inference_ms || telemetry.total_latency_ms || 2400;
            const warmSec = (warmMs / 1000).toFixed(1);
            sceneStateBadge.textContent = `Analysis: ${warmSec}s`;
            sceneStateBadge.className = "status-pill status-supported";

            // 2. Render "What may happen next" (Prediction)
            renderPredictiveNarration(data.predictive_narration || {}, data.prediction_status);

            // 3. Render Technical Panes
            renderSceneUnderstanding(fusedScene);
            renderYoloResults(yoloData);
            renderVqaResults(vqaEvidence, skippedQuestions, telemetry.vqa, personPresence, sceneType);
            renderEvidenceFusion(sceneType, personPresence, conflicts);
            renderDiagnostics(telemetry, yoloData, conflicts);

            if (btnToggleBoxes && yoloData.total_detections > 0) {
                btnToggleBoxes.disabled = false;
            }

            showToast("Scene analysis complete.");

        } catch (error) {
            console.error("Analysis error:", error);
            showToast(`Analysis Error: ${error.message}`);
            sceneNarrationText.innerHTML = `<span style="color: var(--accent-rose);">Error during scene analysis: ${escapeHtml(error.message)}</span>`;
            sceneStateBadge.textContent = "Failed";
            sceneStateBadge.className = "status-pill status-suppressed";
        } finally {
            if (loadingInterval) clearInterval(loadingInterval);
            if (analysisLoadingState) analysisLoadingState.classList.add("hidden");
            if (intelligenceResultsFlow) intelligenceResultsFlow.style.opacity = "1";

            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                <span>Analyze Scene</span>
            `;
        }
    });

    // =========================================================================
    // 7. Render Primary Prediction Card
    // =========================================================================
    function renderPredictiveNarration(pred, topLevelStatus) {
        const text = (pred && pred.prediction) ? pred.prediction : "Anticipates plausible immediate next actions based on physical context.";
        const rawStatus = (pred && pred.status) || topLevelStatus || (pred && pred.prediction_status) || "supported_prediction";
        const statusStr = String(rawStatus).toLowerCase();
        const reasonings = (pred && pred.reasoning) ? pred.reasoning : (pred && pred.reasoning_trace ? pred.reasoning_trace : []);
        const facts = (pred && pred.source_facts) ? pred.source_facts : [];

        predictNarrationText.textContent = text;

        if (statusStr.includes("supported")) {
            predictStateBadge.textContent = "Supported";
            predictStateBadge.className = "status-pill status-supported";
        } else if (statusStr.includes("tentative")) {
            predictStateBadge.textContent = "Tentative";
            predictStateBadge.className = "status-pill status-tentative";
        } else if (statusStr.includes("insufficient")) {
            predictStateBadge.textContent = "Not Enough Evidence";
            predictStateBadge.className = "status-pill status-insufficient";
        } else if (statusStr.includes("suppressed")) {
            predictStateBadge.textContent = "Not Applicable";
            predictStateBadge.className = "status-pill status-suppressed";
        } else {
            predictStateBadge.textContent = "Supported";
            predictStateBadge.className = "status-pill status-supported";
        }

        if (predictReasoningContainer) {
            predictReasoningContainer.classList.remove("hidden");
            predictReasoningList.innerHTML = "";
            reasonings.forEach((r) => {
                const li = document.createElement("li");
                li.textContent = r;
                predictReasoningList.appendChild(li);
            });

            predictSourceFactsList.innerHTML = "";
            facts.forEach((f) => {
                const span = document.createElement("span");
                span.className = "fact-tag";
                span.textContent = f;
                predictSourceFactsList.appendChild(span);
            });
        }
    }

    // =========================================================================
    // 8. Render Technical Panes
    // =========================================================================
    function cleanHumanValue(rawVal, fieldType) {
        if (!rawVal || rawVal === "Unspecified" || rawVal === "None Specified") {
            return fieldType === "subject" ? "Person" : fieldType === "environment" ? "Indoors" : "None";
        }
        let s = String(rawVal).trim();
        s = s.replace(/the image depicts/gi, "").replace(/a photo of/gi, "").replace(/a picture of/gi, "").trim();
        s = s.replace(/with blurry.+/gi, "").trim();
        s = s.replace(/prominently visible/gi, "").trim();
        s = s.replace(/holding \/ interacting with a cell phone/gi, "Using a smartphone").trim();
        s = s.replace(/holding \/ drinking from cup/gi, "Drinking from a cup").trim();
        s = s.replace(/reading \/ holding book/gi, "Reading a book").trim();
        s = s.replace(/sitting in the scene/gi, "Sitting").trim();
        s = s.replace(/standing in the scene/gi, "Standing").trim();

        if (fieldType === "environment") {
            if (/indoor|room|cafe|restaurant|kitchen|office/i.test(s)) return "Indoors";
            if (/outdoor|street|park|nature|road|sky/i.test(s)) return "Outdoors";
        }
        if (fieldType === "subject") {
            if (/wearing .+/i.test(s)) s = s.replace(/wearing .+/i, "").trim();
            if (/in a .+/i.test(s)) s = s.replace(/in a .+/i, "").trim();
            if (/man/i.test(s)) return "Man";
            if (/woman/i.test(s)) return "Woman";
            if (/people|two people/i.test(s)) return "Two people";
            return "Person";
        }
        if (fieldType === "activity") {
            if (/reading|book/i.test(s)) return "Reading a book";
            if (/laptop|computer|typing/i.test(s)) return "Working on a laptop";
            if (/phone|cell|texting/i.test(s)) return "Using a smartphone";
            if (/drinking|coffee|cup|tea/i.test(s)) return "Drinking from a cup";
            if (/skateboard/i.test(s)) return "Riding a skateboard";
            if (/bicycle|bike/i.test(s)) return "Riding a bicycle";
            if (/eating|meal|dining/i.test(s)) return "Having a meal";
            if (/sitting|seated/i.test(s)) return "Sitting";
            if (/standing/i.test(s)) return "Standing";
            if (/walking/i.test(s)) return "Walking";
        }
        if (fieldType === "interaction") {
            if (/reading|book/i.test(s)) return "Reading a book";
            if (/laptop|computer/i.test(s)) return "Working on a laptop";
            if (/phone|cell/i.test(s)) return "Using a smartphone";
            if (/cup|mug|coffee|tea|bottle|drink/i.test(s)) return "Drinking from a cup";
            if (/skateboard/i.test(s)) return "Riding a skateboard";
            if (/bicycle|bike/i.test(s)) return "Riding a bicycle";
            if (/food|meal/i.test(s)) return "Having a meal";
        }
        if (s.length > 0) {
            return s.charAt(0).toUpperCase() + s.slice(1);
        }
        return "None";
    }

    function renderSceneUnderstanding(fusedScene) {
        const sub = fusedScene.primary_subject || {};
        valSubject.textContent = cleanHumanValue(sub.value, "subject");

        const act = fusedScene.activity || {};
        valActivity.textContent = cleanHumanValue(act.value, "activity");

        const env = fusedScene.environment || {};
        valEnvironment.textContent = cleanHumanValue(env.value, "environment");

        const inter = fusedScene.interactions || [];
        if (inter.length > 0) {
            const first = inter[0];
            valInteractions.textContent = cleanHumanValue(`${first.relation} ${first.target}`, "interaction");
        } else {
            valInteractions.textContent = "None";
        }
    }

    function renderYoloResults(yoloData) {
        if (!yoloData || !yoloData.counts) return;

        yoloEmptyState.classList.add("hidden");
        yoloResultsContainer.classList.remove("hidden");
        yoloStateBadge.textContent = `${yoloData.total_detections} Detected (${yoloData.latency_ms}ms)`;
        yoloStateBadge.className = "status-pill status-supported";

        yoloCountsSummary.innerHTML = "";
        Object.entries(yoloData.counts).forEach(([cls, cnt]) => {
            const badge = document.createElement("span");
            badge.className = "yolo-count-badge";
            badge.textContent = `${cnt}× ${cls}`;
            yoloCountsSummary.appendChild(badge);
        });

        yoloObjectsList.innerHTML = "";
        (yoloData.objects || []).forEach((obj) => {
            const item = document.createElement("div");
            item.className = "yolo-obj-item";
            item.innerHTML = `
                <span><strong>${escapeHtml(obj.label)}</strong></span>
                <span style="font-family: var(--font-mono); color: var(--accent-cyan);">${Math.round((obj.confidence || 0) * 100)}%</span>
            `;
            yoloObjectsList.appendChild(item);
        });
    }

    function renderVqaResults(vqaEvidence, skippedQuestions, vqaTelemetry, personPresence, sceneType) {
        vqaEmptyState.classList.add("hidden");
        vqaResultsContainer.classList.remove("hidden");
        vqaStateBadge.textContent = `${vqaEvidence.length} Answered (${vqaTelemetry.latency_ms}ms)`;
        vqaStateBadge.className = "status-pill status-supported";

        const pStatus = personPresence.status || "uncertain";
        vqaRoutingNotice.innerHTML = `Routing Mode: <strong>${pStatus === 'present' ? 'Person-Centric Interrogation' : 'Environmental Interrogation'}</strong>`;

        vqaGrid.innerHTML = "";
        vqaEvidence.forEach((q) => {
            const div = document.createElement("div");
            div.className = "vqa-item";
            div.innerHTML = `
                <span class="vqa-q">${escapeHtml(q.question)}</span>
                <span class="vqa-a">${escapeHtml(q.answer)}</span>
                <div class="vqa-meta">
                    <span>${q.provenance || 'VQA'}</span>
                    <span>${q.confidence !== undefined ? Math.round(q.confidence * 100) + '%' : 'High'}</span>
                </div>
            `;
            vqaGrid.appendChild(div);
        });

        if (skippedQuestions && skippedQuestions.length > 0) {
            skippedDrawer.classList.remove("hidden");
            skippedCountSpan.textContent = skippedQuestions.length;
            skippedList.innerHTML = "";
            skippedQuestions.forEach((sq) => {
                const li = document.createElement("li");
                li.textContent = sq;
                skippedList.appendChild(li);
            });
        } else {
            skippedDrawer.classList.add("hidden");
        }
    }

    function renderEvidenceFusion(sceneType, personPresence, conflicts) {
        fusionEmptyState.classList.add("hidden");
        fusionResultsContainer.classList.remove("hidden");
        fusionStateBadge.textContent = "Converged";
        fusionStateBadge.className = "status-pill status-supported";

        const sType = sceneType.scene_type || "photographic_scene";
        sceneTypeBadge.textContent = sType === "illustration_or_graphic" ? "Illustration / Graphic" : "Photographic Scene";

        const pStatus = personPresence.status || "uncertain";
        presenceBadge.textContent = `Presence: ${capitalize(pStatus)}`;

        conflictCountBadge.textContent = `${conflicts.length} Conflict(s)`;
        conflictsList.innerHTML = "";
        if (conflicts.length === 0) {
            conflictsList.innerHTML = `<div class="conflict-item conflict-clean"><span>✓ All model signals converge without active contradiction.</span></div>`;
        } else {
            conflicts.forEach((c) => {
                const div = document.createElement("div");
                div.className = "conflict-item";
                div.innerHTML = `<span style="color: var(--accent-rose);">${escapeHtml(c.description || "Evidence divergence resolved")}</span>`;
                conflictsList.appendChild(div);
            });
        }
    }

    function renderDiagnostics(telemetry, yoloData, conflicts) {
        const warmMs = telemetry.warm_inference_ms || telemetry.total_latency_ms || 0;
        const warmSec = (warmMs / 1000).toFixed(1);
        diagMode.textContent = "Local Pipeline";
        diagDevice.textContent = `${telemetry.blip.device.toUpperCase()} (Apple Silicon MPS)`;
        diagBlipTelemetry.textContent = `${telemetry.blip.observations_count} queries (${telemetry.blip.latency_ms}ms)`;
        diagYoloTelemetry.textContent = `${yoloData.total_detections} objects (${yoloData.latency_ms}ms)`;
        diagVqaTelemetry.textContent = `${telemetry.vqa.questions_executed} routed Q&As (${telemetry.vqa.latency_ms}ms)`;
        diagFusionTelemetry.textContent = `${warmSec}s (${warmMs}ms) • ${conflicts.length} conflict(s)`;
    }

    // =========================================================================
    // 9. Utilities
    // =========================================================================
    function capitalize(str) {
        if (!str) return "";
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    function escapeHtml(text) {
        if (!text) return "";
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // =========================================================================
    // 10. Video Mode Controller (Stage 5 End-to-End)
    // =========================================================================

    // Video DOM Elements
    const videoDropzone = document.getElementById("videoDropzone");
    const videoFileInput = document.getElementById("videoFileInput");
    const videoDropzonePrompt = document.getElementById("videoDropzonePrompt");
    const videoBrowseBtn = document.getElementById("videoBrowseBtn");
    const videoBrowseLink = document.getElementById("videoBrowseLink");
    const videoUploadProgress = document.getElementById("videoUploadProgress");
    const videoUploadProgressText = document.getElementById("videoUploadProgressText");

    const videoPreviewCard = document.getElementById("videoPreviewCard");
    const videoPreviewPlayer = document.getElementById("videoPreviewPlayer");
    const vidMetaFilename = document.getElementById("vidMetaFilename");
    const vidMetaResolution = document.getElementById("vidMetaResolution");
    const vidMetaDuration = document.getElementById("vidMetaDuration");
    const vidMetaFps = document.getElementById("vidMetaFps");
    const vidMetaSize = document.getElementById("vidMetaSize");
    const vidMetaFrames = document.getElementById("vidMetaFrames");
    const chkDeleteAfter = document.getElementById("chkDeleteAfter");
    const changeVideoBtn = document.getElementById("changeVideoBtn");
    const analyzeVideoBtn = document.getElementById("analyzeVideoBtn");

    // Video Progress Card
    const videoProgressCard = document.getElementById("videoProgressCard");
    const videoProgressBarFill = document.getElementById("videoProgressBarFill");
    const videoProgressPercent = document.getElementById("videoProgressPercent");
    const videoStageText = document.getElementById("videoStageText");
    const videoFrameCountText = document.getElementById("videoFrameCountText");
    const videoEtaText = document.getElementById("videoEtaText");
    const btnCancelVideo = document.getElementById("btnCancelVideo");

    // Video Results Elements
    const videoStoryText = document.getElementById("videoStoryText");
    const videoStoryBadge = document.getElementById("videoStoryBadge");
    const videoNarrativeText = document.getElementById("videoNarrativeText");
    const videoNarrativeBadge = document.getElementById("videoNarrativeBadge");
    const videoPredictText = document.getElementById("videoPredictText");
    const videoPredictBadge = document.getElementById("videoPredictBadge");
    const videoHypothesesContainer = document.getElementById("videoHypothesesContainer");
    const videoSegmentsBadge = document.getElementById("videoSegmentsBadge");
    const videoTimelineEmpty = document.getElementById("videoTimelineEmpty");
    const videoTimelineList = document.getElementById("videoTimelineList");
    const videoFramesBadge = document.getElementById("videoFramesBadge");
    const videoFramesEmpty = document.getElementById("videoFramesEmpty");
    const videoKeyframesGrid = document.getElementById("videoKeyframesGrid");
    const videoChangesEmpty = document.getElementById("videoChangesEmpty");
    const videoChangesList = document.getElementById("videoChangesList");

    // Video Diagnostics Elements
    const vidDiagEngine = document.getElementById("vidDiagEngine");
    const vidDiagHardware = document.getElementById("vidDiagHardware");
    const vidDiagSampleCount = document.getElementById("vidDiagSampleCount");
    const vidDiagRedundantCount = document.getElementById("vidDiagRedundantCount");
    const vidDiagAvgLatency = document.getElementById("vidDiagAvgLatency");
    const vidDiagTemporalTime = document.getElementById("vidDiagTemporalTime");
    const vidDiagTotalTime = document.getElementById("vidDiagTotalTime");
    const videoDiagBadge = document.getElementById("videoDiagBadge");
    const btnDownloadVideoJson = document.getElementById("btnDownloadVideoJson");

    // Video Technical Tabs
    const vtechTabs = document.querySelectorAll(".vtech-tab");
    const vtechPanes = document.querySelectorAll(".vtech-pane");

    // State Variables for Video
    let currentVideoData = null;
    let currentVideoJobId = null;
    let videoPollInterval = null;
    let currentVideoResult = null;

    // Browse Button
    if (videoBrowseBtn) {
        videoBrowseBtn.addEventListener("click", () => videoFileInput.click());
    } else if (videoBrowseLink) {
        videoBrowseLink.addEventListener("click", () => videoFileInput.click());
    }

    if (videoFileInput) {
        videoFileInput.addEventListener("change", (e) => {
            const file = e.target.files && e.target.files[0];
            if (file) handleVideoUpload(file);
        });
    }

    // Drag and Drop
    if (videoDropzone) {
        ["dragenter", "dragover"].forEach((eventName) => {
            videoDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                videoDropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            videoDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                videoDropzone.classList.remove("dragover");
            });
        });

        videoDropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const file = dt.files && dt.files[0];
            if (file) handleVideoUpload(file);
        });
    }

    // Handle Upload
    function handleVideoUpload(file) {
        // Size validation: max 150 MB
        const maxBytes = 150 * 1024 * 1024;
        if (file.size > maxBytes) {
            showToast("Video exceeds 150 MB limit. Please upload a smaller video clip.");
            return;
        }

        // Extension validation
        const allowedExts = [".mp4", ".mov", ".avi", ".webm", ".mkv"];
        const lowerName = file.name.toLowerCase();
        const hasValidExt = allowedExts.some(ext => lowerName.endsWith(ext));
        if (!hasValidExt) {
            showToast("Unsupported video format. Supported formats: MP4, MOV, AVI, WEBM, MKV.");
            return;
        }

        // Show upload spinner
        videoDropzonePrompt.classList.add("hidden");
        videoUploadProgress.classList.remove("hidden");
        if (videoUploadProgressText) {
            videoUploadProgressText.textContent = "Inspecting video container & metadata...";
        }

        const formData = new FormData();
        formData.append("video", file);

        fetch("/api/upload/video", {
            method: "POST",
            body: formData
        })
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.error || "Video upload failed");
            }
            return data;
        })
        .then((data) => {
            currentVideoData = data;
            // Update preview player with local Object URL so click-to-seek survives post-analysis server deletion
            try {
                videoPreviewPlayer.src = URL.createObjectURL(file);
            } catch (e) {
                videoPreviewPlayer.src = data.file_url;
            }
            videoPreviewPlayer.load();

            // Populate metadata
            if (vidMetaFilename) vidMetaFilename.textContent = data.filename || file.name;
            if (vidMetaResolution) vidMetaResolution.textContent = `${data.width}×${data.height}`;
            if (vidMetaDuration) vidMetaDuration.textContent = `${data.duration.toFixed(1)}s`;
            if (vidMetaFps) vidMetaFps.textContent = `${Math.round(data.fps)} FPS`;
            if (vidMetaSize) vidMetaSize.textContent = `${data.size_mb.toFixed(1)} MB`;
            if (vidMetaFrames) vidMetaFrames.textContent = `${data.total_frames} frames`;

            // Transition UI
            videoDropzone.classList.add("hidden");
            videoPreviewCard.classList.remove("hidden");
            analyzeVideoBtn.disabled = false;
            showToast(`Loaded video (${data.duration.toFixed(1)}s, ${data.width}×${data.height})`);
        })
        .catch((err) => {
            console.error("Video upload error:", err);
            showToast(err.message || "Failed to load video file.");
        })
        .finally(() => {
            videoUploadProgress.classList.add("hidden");
            videoDropzonePrompt.classList.remove("hidden");
            videoFileInput.value = "";
        });
    }

    // Change Video Button
    if (changeVideoBtn) {
        changeVideoBtn.addEventListener("click", () => {
            if (videoPollInterval) clearInterval(videoPollInterval);
            currentVideoData = null;
            currentVideoJobId = null;
            currentVideoResult = null;
            videoPreviewPlayer.pause();
            videoPreviewPlayer.src = "";
            videoPreviewCard.classList.add("hidden");
            videoProgressCard.classList.add("hidden");
            videoDropzone.classList.remove("hidden");
        });
    }

    // Analyze Video Button
    if (analyzeVideoBtn) {
        analyzeVideoBtn.addEventListener("click", () => {
            if (!currentVideoData) return;

            const deleteAfter = chkDeleteAfter ? chkDeleteAfter.checked : false;

            analyzeVideoBtn.disabled = true;
            if (changeVideoBtn) changeVideoBtn.disabled = true;

            // Show progress card
            videoProgressCard.classList.remove("hidden");
            videoProgressBarFill.style.width = "0%";
            videoProgressPercent.textContent = "0%";
            videoStageText.textContent = "Submitting job to analysis worker...";
            videoFrameCountText.textContent = "Keyframes: 0 / --";
            videoEtaText.textContent = "ETA: calculating...";

            fetch("/api/video/jobs", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_path: currentVideoData.filepath,
                    delete_after: deleteAfter
                })
            })
            .then(async (res) => {
                const data = await res.json();
                if (res.status === 503) {
                    throw new Error(data.message || "System is busy analyzing another scene. Please retry in a few moments.");
                }
                if (!res.ok) {
                    throw new Error(data.error || "Failed to start video analysis job.");
                }
                return data;
            })
            .then((data) => {
                currentVideoJobId = data.job_id;
                startPollingVideoJob(currentVideoJobId);
            })
            .catch((err) => {
                console.error("Job creation error:", err);
                showToast(err.message);
                videoProgressCard.classList.add("hidden");
                analyzeVideoBtn.disabled = false;
                if (changeVideoBtn) changeVideoBtn.disabled = false;
            });
        });
    }

    // Cancel Video Analysis Button
    if (btnCancelVideo) {
        btnCancelVideo.addEventListener("click", () => {
            if (!currentVideoJobId) return;

            btnCancelVideo.disabled = true;
            videoStageText.textContent = "Cancelling analysis...";

            fetch(`/api/video/jobs/${currentVideoJobId}/cancel`, {
                method: "POST"
            })
            .then(res => res.json())
            .then((data) => {
                showToast("Cancellation requested.");
            })
            .catch((err) => {
                console.error("Cancel error:", err);
            });
        });
    }

    // Polling Function
    function startPollingVideoJob(jobId) {
        if (videoPollInterval) clearInterval(videoPollInterval);

        videoPollInterval = setInterval(() => {
            fetch(`/api/video/jobs/${jobId}`)
            .then(res => res.json())
            .then((job) => {
                if (!job || !job.status) return;

                if (job.status === "queued") {
                    videoProgressBarFill.style.width = "2%";
                    videoProgressPercent.textContent = "0%";
                    videoStageText.textContent = "Queued in worker pipeline...";
                    videoFrameCountText.textContent = "Keyframes: waiting...";
                    videoEtaText.textContent = "ETA: queued";
                } else if (job.status === "processing") {
                    const pct = Math.min(99, Math.max(2, job.progress_percent || 0));
                    videoProgressBarFill.style.width = `${pct}%`;
                    videoProgressPercent.textContent = `${Math.round(pct)}%`;
                    videoStageText.textContent = job.stage || "Analyzing keyframes...";
                    videoFrameCountText.textContent = `Keyframes: ${job.frames_analyzed || 0} / ${job.total_sampled_frames || '--'}`;
                    if (job.eta_seconds !== null && job.eta_seconds !== undefined) {
                        videoEtaText.textContent = `ETA: ~${Math.ceil(job.eta_seconds)}s`;
                    } else {
                        videoEtaText.textContent = "ETA: calculating...";
                    }
                } else if (job.status === "cancelled") {
                    clearInterval(videoPollInterval);
                    videoPollInterval = null;
                    videoProgressCard.classList.add("hidden");
                    analyzeVideoBtn.disabled = false;
                    if (changeVideoBtn) changeVideoBtn.disabled = false;
                    if (btnCancelVideo) btnCancelVideo.disabled = false;
                    showToast("Video analysis was cancelled.");
                } else if (job.status === "failed") {
                    clearInterval(videoPollInterval);
                    videoPollInterval = null;
                    videoProgressCard.classList.add("hidden");
                    analyzeVideoBtn.disabled = false;
                    if (changeVideoBtn) changeVideoBtn.disabled = false;
                    if (btnCancelVideo) btnCancelVideo.disabled = false;
                    showToast(`Video analysis failed: ${job.error || "Unknown error"}`);
                } else if (job.status === "done") {
                    clearInterval(videoPollInterval);
                    videoPollInterval = null;
                    videoProgressBarFill.style.width = "100%";
                    videoProgressPercent.textContent = "100%";
                    videoStageText.textContent = "Analysis complete!";
                    videoEtaText.textContent = "Done";

                    analyzeVideoBtn.disabled = false;
                    if (changeVideoBtn) changeVideoBtn.disabled = false;
                    if (btnCancelVideo) btnCancelVideo.disabled = false;

                    setTimeout(() => {
                        videoProgressCard.classList.add("hidden");
                    }, 1200);

                    if (job.result) {
                        renderVideoResults(job.result);
                        showToast("Video analysis completed successfully!");
                    }
                }
            })
            .catch((err) => {
                console.error("Poll error:", err);
            });
        }, 500);
    }

    // Render Video Results
    function renderVideoResults(result) {
        currentVideoResult = result;

        // 1. Story Narrative
        const story = result.video_story || result.narration || "No narrative synthesized.";
        if (videoStoryText) videoStoryText.textContent = story;
        if (videoNarrativeText) videoNarrativeText.textContent = story;
        if (videoStoryBadge) {
            videoStoryBadge.textContent = "Story Ready";
            videoStoryBadge.className = "status-pill status-supported";
        }
        if (videoNarrativeBadge) {
            videoNarrativeBadge.textContent = "Story Ready";
            videoNarrativeBadge.className = "status-pill status-supported";
        }

        // 2. Prediction & Hypotheses
        const predObj = result.prediction;
        const predText = (predObj && predObj.text) || (typeof predObj === "string" ? predObj : "No prediction generated.");
        if (videoPredictText) videoPredictText.textContent = predText;
        if (videoPredictBadge) {
            const label = (predObj && predObj.likelihood_label) || "Anticipated";
            videoPredictBadge.textContent = capitalize(label);
            videoPredictBadge.className = "status-pill status-supported";
        }

        // Render Hypotheses
        if (videoHypothesesContainer) {
            videoHypothesesContainer.innerHTML = "";
            const hypotheses = (predObj && predObj.hypotheses) || [];
            if (hypotheses.length > 0) {
                hypotheses.forEach((hyp) => {
                    const card = document.createElement("div");
                    card.className = "hypothesis-card";

                    const likelihood = hyp.likelihood_label || "possible";
                    let badgeClass = "status-pill status-gated";
                    if (likelihood === "most likely") badgeClass = "status-pill status-supported";
                    if (likelihood === "less likely") badgeClass = "status-pill status-neutral";

                    // Evidence tags
                    let evidenceHtml = "";
                    if (hyp.evidence && Array.isArray(hyp.evidence)) {
                        evidenceHtml = hyp.evidence.map(ev => `<span class="evidence-tag">${escapeHtml(ev)}</span>`).join("");
                    }

                    card.innerHTML = `
                        <div class="hypothesis-card-header">
                            <span class="hypothesis-title">${escapeHtml(hyp.text)}</span>
                            <span class="${badgeClass}">${capitalize(likelihood)}</span>
                        </div>
                        <p class="hypothesis-rationale">${escapeHtml(hyp.rationale || "")}</p>
                        ${evidenceHtml ? `<div class="hypothesis-evidence-row">${evidenceHtml}</div>` : ""}
                    `;
                    videoHypothesesContainer.appendChild(card);
                });
            }
        }

        // 3. Interactive Clickable Timeline Segments
        if (videoTimelineList) {
            videoTimelineList.innerHTML = "";
            if (videoTimelineEmpty) videoTimelineEmpty.classList.add("hidden");

            const segments = result.timeline_segments || [];
            if (videoSegmentsBadge) {
                videoSegmentsBadge.textContent = `${segments.length} Segment${segments.length === 1 ? '' : 's'}`;
            }

            segments.forEach((seg, idx) => {
                const segCard = document.createElement("div");
                segCard.className = "timeline-segment-card";

                // Format objects count string
                let objectsStr = "None detected";
                if (seg.objects && Object.keys(seg.objects).length > 0) {
                    objectsStr = Object.entries(seg.objects).map(([k, v]) => `${k} (${v})`).join(", ");
                }

                // Thumbnail URL
                const thumbSrc = seg.thumbnail_url || (result.sampled_frames && result.sampled_frames[idx] && result.sampled_frames[idx].thumbnail_url) || "";

                segCard.innerHTML = `
                    <div class="segment-thumb-wrapper">
                        ${thumbSrc ? `<img class="segment-thumb" src="${thumbSrc}" alt="Segment at ${seg.start_seconds.toFixed(1)}s">` : `<div class="segment-thumb" style="display:flex;align-items:center;justify-content:center;color:#64748b;font-size:0.75rem;">Frame</div>`}
                        <div class="segment-thumb-overlay">
                            <svg class="segment-thumb-play-icon" viewBox="0 0 24 24" fill="currentColor">
                                <polygon points="5 3 19 12 5 21 5 3"></polygon>
                            </svg>
                        </div>
                    </div>
                    <div class="segment-info-block">
                        <div class="segment-meta-row">
                            <span class="segment-time-pill">
                                <svg viewBox="0 0 24 24" width="10" height="10" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                                <span>${seg.start_seconds.toFixed(1)}s – ${seg.end_seconds.toFixed(1)}s</span>
                            </span>
                            <span class="segment-duration-tag">(${seg.duration_seconds.toFixed(1)}s duration)</span>
                        </div>
                        <div class="segment-activity-title">${escapeHtml(seg.activity || "Observed activity")}</div>
                        <div class="segment-objects-line">Objects: ${escapeHtml(objectsStr)}</div>
                        ${seg.narration ? `<div class="segment-narrative-line">${escapeHtml(seg.narration)}</div>` : ""}
                    </div>
                `;

                // Click-to-seek video player
                segCard.addEventListener("click", () => {
                    if (videoPreviewPlayer) {
                        videoPreviewPlayer.currentTime = seg.start_seconds;
                        videoPreviewPlayer.play();
                        videoPreviewPlayer.scrollIntoView({ behavior: "smooth", block: "nearest" });
                    }
                });

                videoTimelineList.appendChild(segCard);
            });
        }

        // 4. Keyframe Gallery
        if (videoKeyframesGrid) {
            videoKeyframesGrid.innerHTML = "";
            const frames = result.sampled_frames || [];
            if (videoFramesBadge) videoFramesBadge.textContent = frames.length;

            if (frames.length > 0) {
                if (videoFramesEmpty) videoFramesEmpty.classList.add("hidden");

                frames.forEach((frm) => {
                    if (!frm.thumbnail_url) return;
                    const item = document.createElement("div");
                    item.className = "keyframe-item";
                    item.title = `Frame at ${frm.timestamp_seconds.toFixed(1)}s (Click to jump)`;
                    item.innerHTML = `
                        <img src="${frm.thumbnail_url}" alt="Keyframe at ${frm.timestamp_seconds.toFixed(1)}s">
                        <span class="keyframe-timestamp-tag">${frm.timestamp_seconds.toFixed(1)}s</span>
                    `;
                    item.addEventListener("click", () => {
                        if (videoPreviewPlayer) {
                            videoPreviewPlayer.currentTime = frm.timestamp_seconds;
                            videoPreviewPlayer.pause();
                            videoPreviewPlayer.scrollIntoView({ behavior: "smooth", block: "nearest" });
                        }
                    });
                    videoKeyframesGrid.appendChild(item);
                });
            }
        }

        // 5. Detected Changes
        if (videoChangesList) {
            videoChangesList.innerHTML = "";
            const changes = result.changes || [];
            if (changes.length > 0) {
                if (videoChangesEmpty) videoChangesEmpty.classList.add("hidden");
                changes.forEach((chg) => {
                    const li = document.createElement("li");
                    li.className = "video-change-item";
                    li.innerHTML = `
                        <span class="status-indicator-dot" style="margin-top: 5px;"></span>
                        <div>
                            <strong>${escapeHtml(chg.change_type || "Transition")}:</strong>
                            <span>${escapeHtml(chg.description || JSON.stringify(chg))}</span>
                        </div>
                    `;
                    videoChangesList.appendChild(li);
                });
            } else {
                if (videoChangesEmpty) {
                    videoChangesEmpty.classList.remove("hidden");
                    videoChangesEmpty.textContent = "No significant activity or object transitions detected.";
                }
            }
        }

        // 6. Diagnostics & Telemetry
        const telemetry = result.telemetry || {};
        if (vidDiagSampleCount) vidDiagSampleCount.textContent = `${telemetry.sampled_frames_count || 0} frames`;
        if (vidDiagRedundantCount) vidDiagRedundantCount.textContent = `${telemetry.redundant_frames_skipped || 0} skipped`;
        if (vidDiagAvgLatency) vidDiagAvgLatency.textContent = `${(telemetry.avg_latency_per_frame_ms || 0).toFixed(0)} ms`;
        if (vidDiagTemporalTime) vidDiagTemporalTime.textContent = `${(telemetry.temporal_aggregation_ms || 0).toFixed(0)} ms`;
        if (vidDiagTotalTime) vidDiagTotalTime.textContent = `${(telemetry.total_latency_seconds || 0).toFixed(2)}s`;
        if (videoDiagBadge) {
            videoDiagBadge.textContent = "Analysis Complete";
            videoDiagBadge.className = "diag-val status-supported";
        }

        // 7. Enable JSON Download
        if (btnDownloadVideoJson) {
            btnDownloadVideoJson.disabled = false;
        }
    }

    // Video Technical Tabs Switcher
    vtechTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            vtechTabs.forEach(t => t.classList.remove("active"));
            vtechPanes.forEach(p => {
                p.classList.remove("active");
                p.classList.add("hidden");
            });

            tab.classList.add("active");
            const targetPaneId = tab.getAttribute("data-vpane");
            const targetPane = document.getElementById(targetPaneId);
            if (targetPane) {
                targetPane.classList.remove("hidden");
                targetPane.classList.add("active");
            }
        });
    });

    // Download Video Results as JSON
    if (btnDownloadVideoJson) {
        btnDownloadVideoJson.addEventListener("click", () => {
            if (!currentVideoResult) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentVideoResult, null, 2));
            const downloadAnchor = document.createElement("a");
            const fname = (currentVideoData && currentVideoData.filename) ? currentVideoData.filename.replace(/\.[^/.]+$/, "") : "analysis";
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", `visunarrate_${fname}_results.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
            showToast("Downloaded video analysis results JSON.");
        });
    }

    // =========================================================================
    // 11. URL Parameters for Visual Testing & UI States
    // =========================================================================
    const urlParams = new URLSearchParams(window.location.search);
    const uiState = urlParams.get("ui_state");
    const modeParam = urlParams.get("mode") || urlParams.get("tab");

    if (window.location.hash === "#video" || modeParam === "video" || uiState) {
        if (tabVideo) tabVideo.click();
    }

    if (uiState === "file_selected") {
        if (videoDropzone) videoDropzone.classList.add("hidden");
        if (videoPreviewCard) videoPreviewCard.classList.remove("hidden");
        if (vidMetaFilename) vidMetaFilename.textContent = "test_transition_activity.mp4";
        if (vidMetaResolution) vidMetaResolution.textContent = "1024×768";
        if (vidMetaDuration) vidMetaDuration.textContent = "6.0s";
        if (vidMetaFps) vidMetaFps.textContent = "24 FPS";
        if (vidMetaSize) vidMetaSize.textContent = "0.8 MB";
        if (vidMetaFrames) vidMetaFrames.textContent = "144 frames";
        if (analyzeVideoBtn) analyzeVideoBtn.disabled = false;
    } else if (uiState === "processing") {
        if (videoDropzone) videoDropzone.classList.add("hidden");
        if (videoPreviewCard) videoPreviewCard.classList.remove("hidden");
        if (videoProgressCard) videoProgressCard.classList.remove("hidden");
        if (videoProgressBarFill) videoProgressBarFill.style.width = "58%";
        if (videoProgressPercent) videoProgressPercent.textContent = "58%";
        if (videoStageText) videoStageText.textContent = "Analyzing keyframe 18 of 30 (108.0s)...";
        if (videoEtaText) videoEtaText.textContent = "Estimated time remaining: ~7s";
    } else if (uiState === "finished") {
        if (videoDropzone) videoDropzone.classList.add("hidden");
        if (videoPreviewCard) videoPreviewCard.classList.remove("hidden");
        if (videoProgressCard) videoProgressCard.classList.add("hidden");
        const mockFinishedResult = {
            video_story: "The video opens showing a person wearing a long sleeved shirt in an indoor setting, texting on a cell phone. At 3.0s, the activity shifts from using smartphone to reading a book. Prominent persistent elements include cup, potted plant, chair.",
            prediction: {
                text: "Having transitioned from using the cell phone to the book, they'll likely take a sip from the cup and continue reading.",
                hypotheses: [
                    {
                        text: "Having transitioned from using the cell phone to the book, they'll likely take a sip from the cup and continue reading.",
                        evidence: ["Detected book and cup", "Reading activity confirmed", "Beverage vessel nearby", "Observed transition: Interaction shifts from cell phone to book."],
                        likelihood_label: "most likely",
                        rationale: "Reading with a nearby drink affords periodic sips during continued reading."
                    },
                    {
                        text: "They'll likely continue reading or turn to the next page.",
                        evidence: ["Book reading engagement"],
                        likelihood_label: "possible",
                        rationale: "Sustained text reading leads naturally to turning pages."
                    }
                ],
                reasoning: [
                    "Multi-object reading affordance (book + beverage vessel) evaluated.",
                    "Temporal momentum established following observed transition at 3.0s."
                ],
                source_facts: [
                    "Recent State Shift: Interaction shifts from cell phone to book."
                ]
            },
            timeline_segments: [
                {
                    segment_id: "seg_1",
                    start_seconds: 0.0,
                    end_seconds: 3.0,
                    duration_seconds: 3.0,
                    activity: "using smartphone",
                    environment: "Indoors",
                    person_count: 1,
                    dominant_objects: ["cell phone", "chair", "cup"],
                    narration: "A person is sitting indoors, using a smartphone, with a cup nearby.",
                    thumbnail_url: ""
                },
                {
                    segment_id: "seg_2",
                    start_seconds: 3.0,
                    end_seconds: 6.0,
                    duration_seconds: 3.0,
                    activity: "reading",
                    environment: "Indoors",
                    person_count: 1,
                    dominant_objects: ["book", "chair", "cup"],
                    narration: "A person is sitting indoors, reading a book, with a cup nearby.",
                    thumbnail_url: ""
                }
            ],
            sampled_frames: [
                { sample_number: 1, timestamp_seconds: 0.0, processing_mode: "FULL_ANALYSIS", activity: "using smartphone", scene_narration: "A person is using a phone." },
                { sample_number: 2, timestamp_seconds: 1.0, processing_mode: "REDUNDANT", activity: "using smartphone", scene_narration: "A person is using a phone." },
                { sample_number: 3, timestamp_seconds: 2.0, processing_mode: "REDUNDANT", activity: "using smartphone", scene_narration: "A person is using a phone." },
                { sample_number: 4, timestamp_seconds: 3.0, processing_mode: "KEY_CHANGE", activity: "reading", scene_narration: "A person is reading a book." },
                { sample_number: 5, timestamp_seconds: 4.0, processing_mode: "REDUNDANT", activity: "reading", scene_narration: "A person is reading a book." },
                { sample_number: 6, timestamp_seconds: 5.0, processing_mode: "REDUNDANT", activity: "reading", scene_narration: "A person is reading a book." }
            ],
            changes: [
                { timestamp_seconds: 3.0, type: "activity_change", description: "Activity transitions from using smartphone to reading." },
                { timestamp_seconds: 3.0, type: "interaction_change", description: "Interaction shifts from cell phone to book." }
            ],
            telemetry: {
                video_duration_seconds: 6.0,
                sampled_frames_count: 6,
                actual_video_processing_ms: 3240.5,
                average_frame_latency_all_ms: 540.0,
                total_server_ms: 3380.0
            }
        };
        renderVideoResults(mockFinishedResult);
    } else if (uiState === "error") {
        showToast("Failed to ingest video: Corrupt or unreadable video file: unable to decode frames.", 6000);
    }
});
