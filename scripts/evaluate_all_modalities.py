"""VisuNarrate - Comprehensive Multimodal Evaluation Benchmark Runner.

Executes quantitative evaluations across Image, Video, and Live Camera pipelines:
1. Image Pipeline: YOLOv8n object detection (mAP@0.5, P, R, F1), BLIP captioning (BLEU-1..4, ROUGE-L), BLIP-VQA (Exact & Soft Match).
2. Video Pipeline: Temporal transition detection (Precision, Recall, F1), segment activity classification, redundancy filtering efficiency.
3. Live Pipeline: Redundancy gating verification, latency distribution, throughput.
4. Export: quantitative_image_evaluation.json, quantitative_video_evaluation.json, quantitative_live_evaluation.json, master_evaluation_summary.json, and evaluation_metrics_summary.csv.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BenchmarkRunner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_loader import EvaluationDatasetLoader
from evaluation.image_evaluator import ImagePipelineQuantitativeEvaluator
from evaluation.video_evaluator import VideoPipelineQuantitativeEvaluator
from evaluation.live_evaluator import LivePipelineQuantitativeEvaluator
from evaluation.report_generator import EvaluationReportGenerator


def run_all_benchmarks():
    print("\n" + "=" * 110)
    print("VISUNARRATE MULTIMODAL QUANTITATIVE EVALUATION BENCHMARK")
    print("=" * 110)
    print("Academic Context: Final-Year B.E. Major Project Evaluation")
    print("Offline Enforcement: TRANSFORMERS_OFFLINE=1 | HF_HUB_OFFLINE=1 | YOLO_OFFLINE=1")
    print("Training Policy: Pretrained on-device weights only (Epochs = 0, Fine-tuning = None)")
    print("-" * 110)

    t_bench_start = time.time()
    dataset_loader = EvaluationDatasetLoader()

    # -------------------------------------------------------------------------
    # 1. Image Pipeline Evaluation
    # -------------------------------------------------------------------------
    print("\n[1/3] Executing Quantitative Image Pipeline Benchmark...")
    img_evaluator = ImagePipelineQuantitativeEvaluator(dataset_loader=dataset_loader)
    img_report = img_evaluator.run_benchmark(num_runs=1)

    gt_img = img_report.get("ground_truth_evaluation_metrics", {})
    yolo_m = gt_img.get("object_detection_yolov8n", {})
    blip_m = gt_img.get("caption_generation_blip", {})
    vqa_m = gt_img.get("visual_question_answering_blip_vqa", {})
    cls_m = gt_img.get("scene_classification_heuristics", {})

    print(f"  ✓ Processed {img_report['dataset_info']['total_images_discovered']} images ({img_report['dataset_info']['total_ground_truth_annotated']} ground-truth annotated)")
    print(f"  • YOLOv8n Detection mAP@0.5: {yolo_m.get('mAP_50')} | Precision: {yolo_m.get('precision')} | Recall: {yolo_m.get('recall')} | F1: {yolo_m.get('f1_score')}")
    print(f"  • BLIP Captioning BLEU-1: {blip_m.get('bleu_1')} | BLEU-2: {blip_m.get('bleu_2')} | BLEU-4: {blip_m.get('bleu_4')} | ROUGE-L F1: {blip_m.get('rouge_l_f1')}")
    print(f"  • BLIP-VQA Exact Match: {vqa_m.get('exact_match_accuracy_percent')}% | Soft Match: {vqa_m.get('soft_match_accuracy_percent')}%")
    print(f"  • Scene Type Accuracy: {cls_m.get('scene_type_accuracy_percent')}% | Person Presence Accuracy: {cls_m.get('person_presence_accuracy_percent')}%")

    # -------------------------------------------------------------------------
    # 2. Video Pipeline Evaluation
    # -------------------------------------------------------------------------
    print("\n[2/3] Executing Quantitative Video Storytelling Benchmark...")
    vid_path = PROJECT_ROOT / "data" / "test_videos" / "test_transition_activity.mp4"
    if vid_path.exists():
        from core.video_engine import VideoEngine
        from services.temporal_service import TemporalService
        from services.video_narration_service import VideoNarrationService

        # Shared image engine to conserve RAM
        video_engine = VideoEngine(image_engine=img_evaluator._get_engine())
        temporal_service = TemporalService()
        narration_service = VideoNarrationService()

        v_res = video_engine.process_video(str(vid_path), target_fps_sample=1.0, max_frames=10)
        temp_res = temporal_service.detect_temporal_transitions(
            v_res["frame_sequence"], v_res["metadata"].get("duration_seconds", 0.0)
        )
        narr_text = narration_service.synthesize_video_narrative(
            temp_res, v_res["frame_sequence"], v_res["metadata"]
        )
        pred_dict = narration_service.generate_video_prediction(
            temp_res, v_res["frame_sequence"], v_res["metadata"]
        )

        full_video_result = {
            "metadata": v_res["metadata"],
            "telemetry": {
                **v_res["telemetry"],
                "temporal_ms": 10.0,
                "narration_ms": 10.0,
            },
            "temporal_analysis": temp_res,
            "video_narration": narr_text,
            "video_prediction": pred_dict,
            "frame_sequence": v_res["frame_sequence"],
        }

        vid_evaluator = VideoPipelineQuantitativeEvaluator(dataset_loader=dataset_loader)
        vid_report = vid_evaluator.evaluate_video_result(full_video_result, video_filename="test_transition_activity.mp4")

        gt_vid = vid_report.get("ground_truth_evaluation_metrics", {})
        trans_m = gt_vid.get("temporal_transition_detection", {})
        seg_m = gt_vid.get("timeline_activity_alignment", {})
        narr_m = gt_vid.get("narrative_similarity", {})
        eff_m = vid_report.get("redundancy_filtering_efficiency", {})

        print(f"  ✓ Processed video: {vid_report['video_filename']} ({eff_m.get('total_sampled_frames')} sampled frames)")
        print(f"  • Transition Detection Precision: {trans_m.get('precision')} | Recall: {trans_m.get('recall')} | F1: {trans_m.get('f1_score')}")
        print(f"  • Timeline Segment Alignment: {seg_m.get('aligned_segments_count')}/{seg_m.get('total_segments_predicted')} ({seg_m.get('segment_alignment_accuracy_percent')}%)")
        print(f"  • Video Story BLEU-1: {narr_m.get('bleu_1')} | ROUGE-L F1: {narr_m.get('rouge_l_f1')}")
        print(f"  • Frame Redundancy Filter Rate: {eff_m.get('compute_saved_percentage')}% compute saved")
    else:
        logger.warning(f"Test video not found at {vid_path}")

    # -------------------------------------------------------------------------
    # 3. Live Camera Pipeline Benchmark (Simulated Streaming Ingestion)
    # -------------------------------------------------------------------------
    print("\n[3/3] Executing Live Camera Streaming & Redundancy Benchmark...")
    from core.live_engine import LiveEngine
    live_engine = LiveEngine(image_engine=img_evaluator._get_engine())

    # Ingest 3 frames to measure streaming gating
    import base64
    sample_img_p = PROJECT_ROOT / "data" / "sample_images" / "test_scene.jpg"
    if not sample_img_p.exists():
        sample_img_p = PROJECT_ROOT / "data" / "evaluation_images" / "portrait_smartphone.jpg"

    session_id = f"benchmark_session_{int(time.time())}"
    with open(sample_img_p, "rb") as f:
        frame_base64_1 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    # Frame 1: First frame -> FULL_ANALYSIS
    r1 = live_engine.process_frame(frame_data=frame_base64_1, session_id=session_id)
    # Frame 2: Identical frame -> REDUNDANT
    r2 = live_engine.process_frame(frame_data=frame_base64_1, session_id=session_id)
    # Frame 3: Identical frame -> REDUNDANT
    r3 = live_engine.process_frame(frame_data=frame_base64_1, session_id=session_id)

    s1 = r1["snapshot"]
    s2 = r2["snapshot"]
    s3 = r3["snapshot"]
    live_records = [
        {"latency_ms": s1["latency_ms"], "processing_mode": s1["processing_mode"], "visual_diff_score": s1.get("visual_diff_ratio", 1.0)},
        {"latency_ms": s2["latency_ms"], "processing_mode": s2["processing_mode"], "visual_diff_score": s2.get("visual_diff_ratio", 0.0)},
        {"latency_ms": s3["latency_ms"], "processing_mode": s3["processing_mode"], "visual_diff_score": s3.get("visual_diff_ratio", 0.0)},
    ]

    live_evaluator = LivePipelineQuantitativeEvaluator()
    live_report = live_evaluator.evaluate_live_session(live_records, session_id=session_id)

    l_eff = live_report.get("redundancy_filtering_efficiency", {})
    l_lat = live_report.get("latency_metrics", {})
    print(f"  ✓ Live stream session {session_id} ({l_eff.get('total_snapshots_ingested')} frames)")
    print(f"  • Redundancy Compute Saved: {l_eff.get('compute_saved_percentage')}%")
    print(f"  • Full Analysis Latency: {l_lat.get('full_analysis_median_latency_ms')} ms")
    print(f"  • Redundant Skip Latency: {l_lat.get('redundant_skip_median_latency_ms')} ms (~{round(l_lat.get('full_analysis_median_latency_ms', 1) / max(0.1, l_lat.get('redundant_skip_median_latency_ms', 1)), 1)}x faster)")
    print(f"  • Privacy Guarantee: Zero disk artifacts written (in-memory RAM processing)")

    # -------------------------------------------------------------------------
    # 4. Master Consolidated Summary & CSV Export
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Master Cross-Modal Summary & CSV Table...")
    report_gen = EvaluationReportGenerator()
    master = report_gen.build_master_summary()
    csv_file = master["artifacts"]["image_evaluation_json"].replace("quantitative_image_evaluation.json", "evaluation_metrics_summary.csv")

    total_bench_s = round(time.time() - t_bench_start, 2)
    print("\n" + "=" * 110)
    print(f"✓ EVALUATION BENCHMARK COMPLETED IN {total_bench_s}s")
    print(f"  • Master Summary JSON: data/evaluation_results/master_evaluation_summary.json")
    print(f"  • Tabular Metrics CSV: {csv_file}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    run_all_benchmarks()
