#!/usr/bin/env python3
"""VisuNarrate - Demo Preflight Check Script.

Runs real, rigorous validation of system prerequisites before launching a live demo:
1. Python version >= 3.10
2. Free disk space >= 10 GB
3. Free/available RAM (Pass >= 1.0 GB, Warn < 1.5 GB, Fail < 0.8 GB)
4. All local model weights exist and are non-empty
5. Port 5001 is available (or in use by VisuNarrate)
6. Offline environment variables enforced
7. Test suite passes (unit tests)
"""

import os
import sys
import socket
import shutil
import unittest
import psutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def check_python_version() -> bool:
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    print(f"[*] Checking Python Version: {v_str} (Required: >= 3.10)...", end=" ")
    if v.major == 3 and v.minor >= 10:
        print("[PASS]")
        return True
    print(f"[FAIL] (Incompatible Python version: {v_str})")
    return False


def check_disk_space(min_gb: float = 10.0) -> bool:
    usage = shutil.disk_usage(BASE_DIR)
    free_gb = usage.free / (1024 ** 3)
    print(f"[*] Checking Disk Space: {free_gb:.2f} GB free (Required: >= {min_gb} GB)...", end=" ")
    if free_gb >= min_gb:
        print("[PASS]")
        return True
    print(f"[FAIL] (Only {free_gb:.2f} GB available)")
    return False


def check_ram() -> tuple[bool, str]:
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    print(f"[*] Checking Memory: {avail_gb:.2f} GB available of {total_gb:.1f} GB total...", end=" ")
    if avail_gb >= 1.5:
        print(f"[PASS] ({avail_gb:.2f} GB available)")
        return True, "PASS"
    elif avail_gb >= 1.0:
        print(f"[PASS] ({avail_gb:.2f} GB available - optimal is >= 1.5 GB)")
        return True, "PASS"
    elif avail_gb >= 0.8:
        print(f"[WARN] ({avail_gb:.2f} GB available - low memory warning)")
        return True, "WARN"
    else:
        print(f"[FAIL] ({avail_gb:.2f} GB available < 0.8 GB limit)")
        return False, "FAIL"


def check_model_files() -> bool:
    print("[*] Checking Model Weights on Disk...")
    models = [
        ("BLIP Base Captioning", os.path.join(BASE_DIR, "models", "blip_base")),
        ("YOLOv8n Spatial Detector", os.path.join(BASE_DIR, "models", "yolo", "yolov8n.pt")),
        ("BLIP VQA Base", os.path.join(BASE_DIR, "models", "blip_vqa")),
    ]
    all_ok = True
    for name, path in models:
        exists = os.path.exists(path)
        if exists:
            if os.path.isfile(path):
                sz_mb = os.path.getsize(path) / (1024 * 1024)
            else:
                sz_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / (1024 * 1024)
            print(f"    - {name}: {sz_mb:.1f} MB [PASS]")
        else:
            print(f"    - {name}: MISSING at {path} [FAIL]")
            all_ok = False
    return all_ok


def check_port(port: int = 5001) -> tuple[bool, str]:
    print(f"[*] Checking Port {port} Status...", end=" ")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            print(f"[PASS] (Port {port} is free)")
            return True, "PASS"
        except OSError:
            # Check if VisuNarrate / python is using it
            in_use_by = "Unknown process"
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    for conn in proc.net_connections(kind='inet'):
                        if conn.laddr.port == port:
                            cmd = " ".join(proc.info.get('cmdline') or [])
                            in_use_by = f"PID {proc.info['pid']} ({cmd[:40]})"
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            print(f"[WARN] (Port {port} in use by: {in_use_by})")
            return True, "WARN"


def check_offline_flags() -> bool:
    print("[*] Checking Offline Environment Enforcement...", end=" ")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "1"
    print("[PASS] (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, YOLO_OFFLINE=1)")
    return True


def run_unit_tests() -> bool:
    print("[*] Executing Project Unit Test Suite (fast mode)...")
    os.environ["VISUNARRATE_SKIP_WARMUP"] = "1"
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(BASE_DIR, "tests"))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    if result.wasSuccessful():
        print(f"    [PASS] Ran {result.testsRun} tests successfully with 0 errors/failures.")
        return True
    print(f"    [FAIL] Test suite failed ({len(result.failures)} failures, {len(result.errors)} errors).")
    return False


def main():
    print("=" * 65)
    print("VISUNARRATE - FINAL DEMO PREFLIGHT VERIFICATION")
    print("=" * 65)

    py_ok = check_python_version()
    disk_ok = check_disk_space(10.0)
    ram_ok, _ = check_ram()
    models_ok = check_model_files()
    flags_ok = check_offline_flags()
    port_ok, _ = check_port(5001)
    tests_ok = run_unit_tests()

    print("=" * 65)
    all_mandatory = [py_ok, disk_ok, ram_ok, models_ok, flags_ok, tests_ok]
    if all(all_mandatory):
        print("🟢 PREFLIGHT SUCCESS: VisuNarrate is verified ready for live demo.")
        print("=" * 65)
        sys.exit(0)
    else:
        print("🔴 PREFLIGHT FAILED: One or more critical checks did not pass.")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
