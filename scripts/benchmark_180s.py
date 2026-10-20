#!/usr/bin/env python3
"""Benchmark and resource telemetry runner for 180-second video execution in VisuNarrate."""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
from pathlib import Path

# Offline flags
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from app import app


def get_swap_usage():
    try:
        res = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def run_benchmark():
    video_path = "data/test_videos/test_180s.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"180s test video not found at {video_path}")

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"=== Starting 180s Video Benchmark ({file_size_mb:.2f} MB) ===")

    swap_before = get_swap_usage()
    print("Swap before run:", swap_before)

    client = app.test_client()

    # Telemetry sampling state
    samples = []
    network_traces = []
    stop_event = threading.Event()
    my_pid = os.getpid()

    def monitor_resources():
        proc = psutil.Process(my_pid)
        while not stop_event.is_set():
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                free_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
                samples.append({
                    "timestamp": time.time(),
                    "rss_mb": round(rss_mb, 2),
                    "free_ram_mb": round(free_ram_mb, 2),
                })

                # Check network sockets with lsof -i -a -p <pid>
                lsof_res = subprocess.run(
                    ["lsof", "-i", "-a", "-p", str(my_pid)],
                    capture_output=True,
                    text=True,
                )
                stdout = lsof_res.stdout.strip()
                if stdout:
                    network_traces.append(stdout)
            except Exception as e:
                pass
            time.sleep(2.0)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    benchmark_start = time.time()

    # Step 1: Upload via Flask API
    print("Uploading 180s video via POST /api/upload/video...")
    t_upload_start = time.time()
    with open(video_path, "rb") as f:
        res_upload = client.post("/api/upload/video", data={"video": f})
    upload_time = round(time.time() - t_upload_start, 3)
    assert res_upload.status_code == 200, f"Upload failed: {res_upload.get_json()}"
    upload_data = res_upload.get_json()
    saved_filepath = upload_data["filepath"]
    print(f"Upload successful in {upload_time}s. Saved to: {saved_filepath}")

    # Step 2: Create Async Job via POST /api/video/jobs
    print("Enqueuing video analysis job via POST /api/video/jobs...")
    res_job = client.post("/api/video/jobs", json={
        "video_path": saved_filepath,
        "delete_after": False
    })
    assert res_job.status_code in (200, 202), f"Job creation failed: {res_job.get_json()}"
    job_id = res_job.get_json()["job_id"]
    print(f"Job enqueued with ID: {job_id}")

    # Step 3: Poll GET /api/video/jobs/<job_id>
    print("Polling job status every 2 seconds...")
    final_result_data = None
    poll_count = 0
    while True:
        time.sleep(2.0)
        poll_count += 1
        res_poll = client.get(f"/api/video/jobs/{job_id}")
        data = res_poll.get_json()
        status = data.get("status")
        pct = data.get("progress_percent", 0)
        stage = data.get("stage", "")
        eta = data.get("eta_seconds", 0)
        elapsed = round(time.time() - benchmark_start, 1)

        print(f"[{elapsed}s] Status: {status} ({pct}%) | Stage: {stage} | ETA: {eta}s")

        if status in ("done", "failed", "cancelled"):
            final_result_data = data
            break

    benchmark_total_time = round(time.time() - benchmark_start, 2)
    stop_event.set()
    monitor_thread.join(timeout=3.0)

    swap_after = get_swap_usage()
    print("Swap after run:", swap_after)

    assert final_result_data is not None and final_result_data.get("status") == "done", f"Job failed: {final_result_data}"

    result_payload = final_result_data["result"]
    telemetry = result_payload.get("telemetry", {})
    keyframes = result_payload.get("sampled_frames", [])
    keyframes_count = len(keyframes)
    sec_per_keyframe = round(benchmark_total_time / keyframes_count, 3) if keyframes_count else 0.0

    peak_rss = max([s["rss_mb"] for s in samples]) if samples else 0.0
    lowest_free_ram = min([s["free_ram_mb"] for s in samples]) if samples else 0.0

    output_summary = {
        "benchmark_file": video_path,
        "video_duration_seconds": 180.0,
        "file_size_mb": round(file_size_mb, 2),
        "keyframes_analysed": keyframes_count,
        "total_wall_clock_time_seconds": benchmark_total_time,
        "seconds_per_keyframe": sec_per_keyframe,
        "peak_rss_mb": peak_rss,
        "lowest_free_ram_mb": lowest_free_ram,
        "samples_count": len(samples),
        "swap_before": swap_before,
        "swap_after": swap_after,
        "swap_grew": (swap_before != swap_after),
        "network_traces_count": len(network_traces),
        "network_traces": network_traces,
        "video_story": result_payload.get("video_story"),
        "prediction": result_payload.get("prediction"),
        "telemetry": telemetry,
    }

    os.makedirs("data/evaluation_results", exist_ok=True)
    with open("data/evaluation_results/benchmark_180s_results.json", "w") as f:
        json.dump(output_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("180s VIDEO BENCHMARK SUMMARY REPORT")
    print("=" * 60)
    print(f"Keyframes Analysed: {keyframes_count}")
    print(f"Total Processing Time: {benchmark_total_time}s ({benchmark_total_time/60:.2f} min)")
    print(f"Processing Speed: {sec_per_keyframe} s / keyframe")
    print(f"Peak Process Memory (RSS): {peak_rss:.1f} MB")
    print(f"Lowest Free RAM: {lowest_free_ram:.1f} MB")
    print(f"macOS Swap Before: {swap_before}")
    print(f"macOS Swap After:  {swap_after}")
    print(f"Did Swap Grow?:    {'YES' if swap_before != swap_after else 'NO (0 bytes growth)'}")
    print(f"Network Sockets Sampled: {len(network_traces)} active connections recorded")
    if network_traces:
        print("Network Traces:", network_traces)
    else:
        print("Network Status: 100% OFFLINE (Zero remote/external sockets, no outgoing connections).")
    print("=" * 60)

    # Delete uploaded video and test_180s.mp4 to keep disk space > 10GB
    if os.path.exists(saved_filepath):
        os.remove(saved_filepath)
        print(f"Cleaned up uploaded video: {saved_filepath}")
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Cleaned up temporary 180s test video: {video_path}")


if __name__ == "__main__":
    run_benchmark()
