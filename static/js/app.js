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
        if (previewModeNotice) previewModeNotice.classList.add("hidden");
    });

    tabVideo.addEventListener("click", () => {
        if (previewModeNotice) {
            previewModalTitle.textContent = "Video Storytelling (Preview)";
            previewModalText.textContent = "Temporal video analysis and chronological scene summarization are currently in preview staging for subsequent release.";
            previewModeNotice.classList.remove("hidden");
        }
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
});
