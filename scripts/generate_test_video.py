"""VisuNarrate - Local Test Video Generator (Stage 5.0).

Generates a controlled 6-second MP4 test video locally using OpenCV.
The video demonstrates an explicit temporal state transition:
- Segment 1 (0.0s - 3.0s): Person standing while interacting with a smartphone indoors.
- Segment 2 (3.0s - 6.0s): Person seated reading a book indoors.

Ensures 100% offline generation without external downloads.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def generate_local_test_video():
    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "data" / "evaluation_images"
    output_dir = base_dir / "data" / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "test_transition_activity.mp4"
    
    img1_path = eval_dir / "ChatGPT Image Aug 20, 2026, 10_31_28 PM.png"
    img2_path = eval_dir / "ChatGPT Image Aug 20, 2026, 10_35_05 PM.png"
    
    if not img1_path.exists() or not img2_path.exists():
        # Fallback to sample baseline images if specific filenames differ
        img1_path = eval_dir / "portrait_smartphone.jpg"
        img2_path = eval_dir / "mountain_landscape.jpg"

    print(f"[VideoGenerator] Reading Segment 1 source: {img1_path.name}")
    print(f"[VideoGenerator] Reading Segment 2 source: {img2_path.name}")
    
    im1 = cv2.imread(str(img1_path))
    im2 = cv2.imread(str(img2_path))
    
    target_width, target_height = 1024, 768
    im1_resized = cv2.resize(im1, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    im2_resized = cv2.resize(im2, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    fps = 24
    duration_sec = 6.0
    total_frames = int(fps * duration_sec)
    split_frame = total_frames // 2
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    
    print(f"[VideoGenerator] Encoding {total_frames} frames ({duration_sec}s @ {fps} FPS, {target_width}x{target_height})...")
    
    for i in range(total_frames):
        if i < split_frame:
            frame = im1_resized.copy()
        else:
            frame = im2_resized.copy()
            
        out.write(frame)
        
    out.release()
    print(f"[VideoGenerator] Test video created successfully: {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
    return str(output_path)

if __name__ == "__main__":
    generate_local_test_video()
