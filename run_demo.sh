#!/usr/bin/env bash
# VisuNarrate - Final Demo Launch Script
# Fully offline, self-contained startup for Apple Silicon / macOS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=================================================================="
echo "           VISUNARRATE - MULTIMODAL DEMO LAUNCHER"
echo "=================================================================="

# 1. Enforce strict offline environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1
export MPLCONFIGDIR="/tmp/matplotlib"

PORT=5001
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"

# Locate python binary
if [ -d ".venv/bin" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# 2. Check and clean up any stale VisuNarrate instance on port 5001
EXISTING_PID=$(lsof -ti tcp:${PORT} || true)
if [ -n "${EXISTING_PID}" ]; then
    echo "[*] Stale process detected listening on port ${PORT} (PID: ${EXISTING_PID}). Cleaning up..."
    kill -15 "${EXISTING_PID}" 2>/dev/null || true
    sleep 1
    # Force kill if still holding port
    EXISTING_PID_AFTER=$(lsof -ti tcp:${PORT} || true)
    if [ -n "${EXISTING_PID_AFTER}" ]; then
        kill -9 "${EXISTING_PID_AFTER}" 2>/dev/null || true
        sleep 1
    fi
    echo "[+] Port ${PORT} cleared."
fi

# 3. Run rigorous preflight system verification
echo ""
echo "[*] Running System Preflight Checks..."
"${PYTHON}" scripts/preflight.py
PREFLIGHT_EXIT=$?

if [ ${PREFLIGHT_EXIT} -ne 0 ]; then
    echo "[-] Preflight validation failed. Cannot proceed to demo." >&2
    exit 1
fi

echo ""
echo "[*] Launching VisuNarrate Server on ${BASE_URL}..."
# Start server in background
"${PYTHON}" app.py &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "[*] Shutting down VisuNarrate (PID: ${SERVER_PID})..."
    kill -15 "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "[+] Server stopped safely."
}
trap cleanup EXIT INT TERM

# 4. Wait until the server responds to /api/health
echo "[*] Awaiting server warm-up and model readiness on ${BASE_URL}/api/health..."
MAX_WAIT_SECONDS=45
WAIT_COUNT=0
READY=0

while [ ${WAIT_COUNT} -lt ${MAX_WAIT_SECONDS} ]; do
    if curl -s -f "${BASE_URL}/api/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    # Check if process died
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[-] Server process terminated unexpectedly during startup." >&2
        exit 1
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ ${READY} -eq 0 ]; then
    echo "[-] Server timed out after ${MAX_WAIT_SECONDS} seconds." >&2
    exit 1
fi

echo ""
echo "=================================================================="
echo " VisuNarrate is running at ${BASE_URL}"
echo " Offline mode: ACTIVE | Models: LOCAL (BLIP, YOLOv8n, BLIP VQA)"
echo " Server PID: ${SERVER_PID}"
echo " Press Ctrl+C to terminate cleanly."
echo "=================================================================="

# 5. Open browser
if command -v open > /dev/null 2>&1; then
    open "${BASE_URL}"
elif command -v xdg-open > /dev/null 2>&1; then
    xdg-open "${BASE_URL}"
fi

# Keep script running to maintain child process and handle Ctrl+C
wait "${SERVER_PID}"
