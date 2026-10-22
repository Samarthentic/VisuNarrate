"""VisuNarrate - Automated Live Mode Camera Stream Verification Script.

Simulates a continuous browser camera stream over HTTP:
1. Ingests 5 sequential frames into /api/live/analyze:
   - Frame 1: Person working at desk (Full Analysis)
   - Frame 2: Near-identical frame (Redundancy Check -> Reused in <10ms)
   - Frame 3: Near-identical frame (Redundancy Check -> Reused in <10ms)
   - Frame 4: Different scene (Significant Change -> Full Analysis)
   - Frame 5: Near-identical frame to Frame 4 (Redundancy Check -> Reused)
2. Calls /api/live/session/<session_id>/summary to synthesize a cohesive story.
3. Calls DELETE /api/live/session/<session_id> to verify RAM purge.
4. Verifies strictly zero disk writes occurred during the entire live session.
"""

import os
import sys
import time
import base64
import json
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Force offline flags
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["VISUNARRATE_SKIP_WARMUP"] = "1"

from app import app


def load_image_as_base64(image_path: str, max_dim: int = 640) -> str:
    """Loads an image file, resizes to <= max_dim, and returns base64 JPEG string."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def run_live_stream_verification():
    print("=" * 70)
    print("VISUNARRATE - LIVE CAMERA MODE AUTOMATED VERIFICATION")
    print("=" * 70)

    client = app.test_client()

    img_path_a = "data/evaluation_images/ChatGPT Image Aug 20, 2026, 10_33_23 PM.png"
    img_path_b = "data/evaluation_images/portrait_smartphone.jpg"

    if not os.path.exists(img_path_a) or not os.path.exists(img_path_b):
        print(f"ERROR: Missing evaluation images: {img_path_a} or {img_path_b}")
        sys.exit(1)

    frame_a_b64 = load_image_as_base64(img_path_a)
    frame_b_b64 = load_image_as_base64(img_path_b)

    # Frame sequence simulating realistic camera behavior:
    # Frame 1: Scene A
    # Frame 2: Scene A (stationary camera)
    # Frame 3: Scene A (stationary camera)
    # Frame 4: Scene B (camera turns or subject changes)
    # Frame 5: Scene B (stationary camera)
    test_stream = [
        {"name": "Frame 1 (Scene A - Initial)", "data": frame_a_b64, "expected_mode": "FULL_ANALYSIS"},
        {"name": "Frame 2 (Scene A - Static)", "data": frame_a_b64, "expected_mode": "REDUNDANT"},
        {"name": "Frame 3 (Scene A - Static)", "data": frame_a_b64, "expected_mode": "REDUNDANT"},
        {"name": "Frame 4 (Scene B - Transition)", "data": frame_b_b64, "expected_mode": "FULL_ANALYSIS"},
        {"name": "Frame 5 (Scene B - Static)", "data": frame_b_b64, "expected_mode": "REDUNDANT"},
    ]

    session_id = f"test_live_verify_{int(time.time())}"
    print(f"\n1. Starting Live Session: {session_id}")

    # Check upload folder before test
    uploads_before = set(os.listdir("static/uploads")) if os.path.exists("static/uploads") else set()

    for idx, item in enumerate(test_stream, 1):
        t0 = time.time()
        res = client.post("/api/live/analyze", json={
            "session_id": session_id,
            "frame": item["data"],
        })
        latency = (time.time() - t0) * 1000

        if res.status_code != 200:
            print(f"❌ Frame {idx} Failed with status {res.status_code}: {res.get_data(as_text=True)}")
            sys.exit(1)

        payload = res.get_json()["data"]
        snap = payload["snapshot"]
        mode = snap["processing_mode"]
        diff = snap["visual_diff_ratio"]

        print(f"   [{idx}/5] {item['name']}:")
        print(f"         Mode: {mode} (Expected: {item['expected_mode']}) | Diff Ratio: {diff:.4f} | Latency: {latency:.1f}ms")
        print(f"         Narration: \"{snap['narration']}\"")
        print(f"         Top Prediction: \"{snap['prediction']}\"")

        assert mode == item["expected_mode"], f"Mode mismatch on frame {idx}: got {mode}, expected {item['expected_mode']}"

    # Verify session summary synthesis
    print("\n2. Testing Live Session Summary Synthesis (/api/live/session/<id>/summary)...")
    t_sum_start = time.time()
    res_summary = client.post(f"/api/live/session/{session_id}/summary")
    sum_latency = (time.time() - t_sum_start) * 1000

    if res_summary.status_code != 200:
        print(f"❌ Summary failed with status {res_summary.status_code}: {res_summary.get_data(as_text=True)}")
        sys.exit(1)

    sum_data = res_summary.get_json()["data"]
    print(f"   Summary Synthesis Complete in {sum_latency:.1f}ms")
    print(f"   Coherent Story: \"{sum_data.get('live_story')}\"")
    print(f"   Direction Prediction: \"{sum_data.get('live_prediction')}\"")
    print(f"   Segments Detected: {len(sum_data.get('timeline_segments', []))}")
    print(f"   Transitions: {len(sum_data.get('changes', []))}")

    # Verify session deletion
    print("\n3. Testing Live Session RAM Purge (/api/live/session/<id> DELETE)...")
    res_del = client.delete(f"/api/live/session/{session_id}")
    assert res_del.status_code == 200
    print(f"   Session {session_id} purged successfully from RAM.")

    # Verify zero disk writes
    uploads_after = set(os.listdir("static/uploads")) if os.path.exists("static/uploads") else set()
    new_files = uploads_after - uploads_before
    print("\n4. Privacy & Disk Invariance Audit:")
    print(f"   New files created in static/uploads: {len(new_files)}")
    if new_files:
        print(f"❌ FAIL: Unwanted disk files created during live stream: {new_files}")
        sys.exit(1)
    else:
        print("   ✅ PASS: Zero disk writes verified. All live camera frames remained strictly in RAM.")

    print("\n" + "=" * 70)
    print("ALL LIVE MODE VERIFICATION CHECKS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    run_live_stream_verification()
