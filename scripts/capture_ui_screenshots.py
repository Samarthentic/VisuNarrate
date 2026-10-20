#!/usr/bin/env python3
"""Capture high-resolution evidence screenshots of VisuNarrate Video Mode UI states."""

import os
import time
import subprocess
from pathlib import Path

states = [
    ("ui_01_idle.png", "http://127.0.0.1:5001/?ui_state=idle#video"),
    ("ui_02_file_selected.png", "http://127.0.0.1:5001/?ui_state=file_selected#video"),
    ("ui_03_processing.png", "http://127.0.0.1:5001/?ui_state=processing#video"),
    ("ui_04_finished_timeline.png", "http://127.0.0.1:5001/?ui_state=finished#video"),
    ("ui_05_error_state.png", "http://127.0.0.1:5001/?ui_state=error#video"),
]

screenshot_dir = Path("docs/screenshots")
screenshot_dir.mkdir(parents=True, exist_ok=True)

for filename, url in states:
    print(f"Navigating Safari to {url}...")
    apple_script = f'tell application "Safari" to set URL of current tab of front window to "{url}"'
    subprocess.run(["osascript", "-e", apple_script], check=True)
    time.sleep(2.0)  # Wait for page layout, video canvas, and CSS animations

    output_path = screenshot_dir / filename
    print(f"Capturing screenshot: {output_path}...")
    subprocess.run(["screencapture", "-x", str(output_path)], check=True)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Captured {filename} ({size_kb:.1f} KB)")

print("\nAll 5 UI evidence screenshots captured successfully in docs/screenshots/!")
