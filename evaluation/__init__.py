"""VisuNarrate - Model Evaluation and Performance Metrics Package.

Provides mathematically rigorous, offline, pure-Python metrics calculations:
- Object Detection: IoU, Precision, Recall, F1, Average Precision (AP), Mean Average Precision (mAP@0.5)
- Caption Generation: BLEU (BLEU-1 to BLEU-4) with Brevity Penalty, ROUGE-L (Longest Common Subsequence)
- Visual Question Answering: Exact Match, Soft/Normalized Match
- Video Temporal Transitions: Precision, Recall, F1 with configurable tolerance window
- Zero-denominator protection and explicit telemetry vs. quality distinction.
"""

from evaluation.metrics import (
    calculate_iou,
    evaluate_detections,
    calculate_ap,
    calculate_map,
    calculate_bleu,
    calculate_rouge_l,
    evaluate_vqa,
    evaluate_video_transitions,
)

__all__ = [
    "calculate_iou",
    "evaluate_detections",
    "calculate_ap",
    "calculate_map",
    "calculate_bleu",
    "calculate_rouge_l",
    "evaluate_vqa",
    "evaluate_video_transitions",
]
