#!/usr/bin/env python3
"""End-to-end integration and API verification test for VisuNarrate Video Mode."""

import os
import sys
import time

# Set offline environment variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["VISUNARRATE_SKIP_WARMUP"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, ANALYSIS_LOCK


def test_video_e2e():
    client = app.test_client()

    print("=== 1. Testing Video Upload Endpoint ===")

    # Test corrupt video
    with open("data/test_videos/test_corrupt.mp4", "rb") as f:
        res = client.post("/api/upload/video", data={"video": f})
        print("Corrupt upload status:", res.status_code)
        assert res.status_code == 400
        assert not res.get_json()["success"]

    # Test empty video
    with open("data/test_videos/test_empty.mp4", "rb") as f:
        res = client.post("/api/upload/video", data={"video": f})
        print("Empty upload status:", res.status_code)
        assert res.status_code == 400
        assert not res.get_json()["success"]

    # Test portrait video
    with open("data/test_videos/test_portrait.mp4", "rb") as f:
        res = client.post("/api/upload/video", data={"video": f})
        print("Portrait upload status:", res.status_code)
        assert res.status_code == 200
        pdata = res.get_json()
        assert pdata["metadata"]["is_portrait"] is True

    # Test valid video
    with open("data/test_videos/test_transition_activity.mp4", "rb") as f:
        res = client.post("/api/upload/video", data={"video": f})
        print("Valid upload status:", res.status_code)
        assert res.status_code == 200
        valid_data = res.get_json()
        assert valid_data["status"] == "ok"
        assert valid_data["duration"] == 6.0

    print("\n=== 2. Testing Asynchronous Job Lifecycle & Concurrency Guard ===")

    # Start video job
    res = client.post("/api/video/jobs", json={"video_path": valid_data["filepath"], "delete_after": False})
    print("Job creation response:", res.status_code, res.get_json())
    assert res.status_code in (200, 202)
    job_id = res.get_json()["job_id"]

    # Test lock / busy on concurrent image request
    img_res = client.post("/api/analyze/image", json={"image_path": "data/evaluation_images/01_person_table_drinking.jpg"})
    print("Concurrent image request response during video job:", img_res.status_code, img_res.get_json())
    assert img_res.status_code == 503
    assert "busy" in img_res.get_json().get("message", "").lower() or "busy" in img_res.get_json().get("error", "").lower()

    # Poll job until done
    max_wait = 90
    start_time = time.time()
    final_data = None
    while time.time() - start_time < max_wait:
        time.sleep(2)
        status_res = client.get(f"/api/video/jobs/{job_id}")
        data = status_res.get_json()
        elapsed = round(time.time() - start_time, 1)
        print(f"[{elapsed}s] status={data.get('status')}, pct={data.get('progress_percent')}%, stage='{data.get('stage')}', eta={data.get('eta_seconds')}")
        if data.get("status") in ("done", "failed", "cancelled"):
            final_data = data
            break

    assert final_data is not None, "Job did not complete within timeout"
    assert final_data.get("status") == "done", f"Job failed: {final_data.get('error')}"

    result = final_data.get("result", {})
    print("\n=== 3. Validating Structured Results ===")
    story = result.get("video_story")
    prediction = result.get("prediction", {})
    hypotheses = prediction.get("hypotheses", [])
    segments = result.get("timeline_segments", [])
    frames = result.get("sampled_frames", [])

    print(f"Story: {story}")
    top_pred = prediction.get("text") or prediction.get("prediction")
    print(f"Top Prediction: {top_pred}")
    print(f"Hypotheses Count: {len(hypotheses)}")
    print(f"Segments Count: {len(segments)}")
    print(f"Sampled Keyframes Count: {len(frames)}")

    assert story and len(story) > 10, "Video story missing or too short"
    assert prediction and top_pred, "Prediction missing"
    assert len(hypotheses) >= 1, "Expected at least 1 hypothesis"
    assert len(segments) >= 1, "Expected at least 1 timeline segment"
    assert len(frames) >= 1, "Expected at least 1 sampled keyframe"

    # Verify no self-referential narration in evidence
    for hyp in hypotheses:
        assert hyp.get("likelihood_label") in ("most likely", "possible", "less likely")
        for ev in hyp.get("evidence", []):
            assert "narration asserts" not in ev.lower(), f"Self-referential evidence found: {ev}"

    # Verify thumbnails were created
    for seg in segments:
        if seg.get("thumbnail_url"):
            thumb_path = seg["thumbnail_url"].lstrip("/")
            assert os.path.exists(thumb_path), f"Thumbnail file not found at {thumb_path}"

    print("\n=== 4. Testing Video Job Cancellation ===")
    res_cancel_job = client.post("/api/video/jobs", json={"video_path": valid_data["filepath"], "delete_after": False})
    cjob_id = res_cancel_job.get_json()["job_id"]
    time.sleep(0.5)

    # Cancel job
    cancel_res = client.post(f"/api/video/jobs/{cjob_id}/cancel")
    print("Cancel request response:", cancel_res.status_code, cancel_res.get_json())
    assert cancel_res.status_code == 200

    # Poll until cancelled
    cancel_status = None
    for _ in range(30):
        time.sleep(1)
        sres = client.get(f"/api/video/jobs/{cjob_id}")
        sdata = sres.get_json()
        if sdata.get("status") in ("cancelled", "done", "failed"):
            cancel_status = sdata.get("status")
            print(f"Cancelled job final status: {cancel_status}")
            break

    assert cancel_status in ("cancelled", "done"), f"Unexpected cancel status: {cancel_status}"

    print("\n🎉 ALL VIDEO E2E INTEGRATION CHECKS PASSED SUCCESSFULLY! 🎉")


if __name__ == "__main__":
    test_video_e2e()
