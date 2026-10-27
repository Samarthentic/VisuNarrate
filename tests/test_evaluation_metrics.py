"""Unit tests for VisuNarrate evaluation metrics engine (evaluation/metrics.py)."""

import unittest
from evaluation.metrics import (
    calculate_iou,
    calculate_ap,
    evaluate_detections,
    calculate_map,
    calculate_bleu,
    calculate_rouge_l,
    evaluate_vqa,
    evaluate_video_transitions,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Verifies correctness and zero-denominator handling of all evaluation metrics."""

    def test_iou_identical_boxes(self):
        b1 = [10.0, 10.0, 50.0, 50.0]
        b2 = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}
        self.assertAlmostEqual(calculate_iou(b1, b2), 1.0, places=4)

    def test_iou_disjoint_boxes(self):
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [20.0, 20.0, 30.0, 30.0]
        self.assertEqual(calculate_iou(b1, b2), 0.0)

    def test_iou_half_overlap(self):
        # b1: 0..10, 0..10 (area=100)
        # b2: 0..10, 5..15 (area=100)
        # intersection: 0..10, 5..10 (area=50)
        # union: 100 + 100 - 50 = 150
        # iou: 50 / 150 = 0.3333...
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [0.0, 5.0, 10.0, 15.0]
        self.assertAlmostEqual(calculate_iou(b1, b2), 50.0 / 150.0, places=4)

    def test_iou_zero_area_box(self):
        b1 = [5.0, 5.0, 5.0, 5.0]
        b2 = [0.0, 0.0, 10.0, 10.0]
        self.assertEqual(calculate_iou(b1, b2), 0.0)

    def test_detection_evaluation_perfect_match(self):
        preds = [
            {"label": "person", "confidence": 0.95, "bbox": [10, 10, 100, 100]},
            {"label": "laptop", "confidence": 0.88, "bbox": [110, 50, 200, 150]},
        ]
        gts = [
            {"label": "person", "bbox": [10, 10, 100, 100]},
            {"label": "laptop", "bbox": [110, 50, 200, 150]},
        ]
        res = evaluate_detections(preds, gts, iou_threshold=0.5)
        self.assertEqual(res["true_positives"], 2)
        self.assertEqual(res["false_positives"], 0)
        self.assertEqual(res["false_negatives"], 0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1_score"], 1.0)

    def test_detection_evaluation_zero_predictions(self):
        preds = []
        gts = [{"label": "person", "bbox": [10, 10, 100, 100]}]
        res = evaluate_detections(preds, gts, iou_threshold=0.5)
        self.assertEqual(res["true_positives"], 0)
        self.assertEqual(res["false_positives"], 0)
        self.assertEqual(res["false_negatives"], 1)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f1_score"], 0.0)

    def test_detection_evaluation_zero_ground_truth(self):
        preds = [{"label": "cup", "confidence": 0.8, "bbox": [10, 10, 50, 50]}]
        gts = []
        res = evaluate_detections(preds, gts, iou_threshold=0.5)
        self.assertEqual(res["true_positives"], 0)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)

    def test_calculate_ap_perfect(self):
        recalls = [0.2, 0.5, 0.8, 1.0]
        precisions = [1.0, 1.0, 1.0, 1.0]
        ap = calculate_ap(recalls, precisions)
        self.assertEqual(ap, 1.0)

    def test_bleu_identical_sentence(self):
        cand = "a person is sitting at a desk working on a laptop"
        refs = ["a person is sitting at a desk working on a laptop"]
        res = calculate_bleu(cand, refs)
        self.assertEqual(res["bleu_1"], 1.0)
        self.assertEqual(res["bleu_2"], 1.0)
        self.assertEqual(res["bleu_4"], 1.0)
        self.assertEqual(res["brevity_penalty"], 1.0)

    def test_bleu_disjoint_sentence(self):
        cand = "red apples grow here"
        refs = ["blue birds fly away"]
        res = calculate_bleu(cand, refs, smoothing=False)
        self.assertEqual(res["bleu_1"], 0.0)
        self.assertEqual(res["bleu_4"], 0.0)

    def test_bleu_brevity_penalty(self):
        cand = "person desk"
        refs = ["a person is sitting at a desk working on a laptop"]
        res = calculate_bleu(cand, refs)
        self.assertTrue(res["brevity_penalty"] < 1.0)
        self.assertTrue(res["bleu_1"] < 1.0)

    def test_rouge_l_identical(self):
        cand = "a person holding a smartphone indoors"
        refs = ["a person holding a smartphone indoors"]
        res = calculate_rouge_l(cand, refs)
        self.assertAlmostEqual(res["rouge_l_precision"], 1.0, places=3)
        self.assertAlmostEqual(res["rouge_l_recall"], 1.0, places=3)
        self.assertAlmostEqual(res["rouge_l_f1"], 1.0, places=3)

    def test_rouge_l_partial_overlap(self):
        cand = "person holding phone"
        refs = ["person holding blue smartphone in room"]
        res = calculate_rouge_l(cand, refs)
        self.assertTrue(0.0 < res["rouge_l_f1"] < 1.0)
        self.assertEqual(res["lcs_length"], 2)  # "person holding"

    def test_rouge_l_empty(self):
        res = calculate_rouge_l("", ["reference sentence"])
        self.assertEqual(res["rouge_l_f1"], 0.0)

    def test_vqa_exact_and_soft_match(self):
        # Exact match with article difference
        res1 = evaluate_vqa("A smartphone", ["smartphone", "cell phone"])
        self.assertTrue(res1["exact_match"])
        self.assertTrue(res1["soft_match"])

        # Soft match
        res2 = evaluate_vqa("black hoodie", ["hoodie"])
        self.assertFalse(res2["exact_match"])
        self.assertTrue(res2["soft_match"])

        # Incorrect answer
        res3 = evaluate_vqa("pizza", ["laptop", "computer"])
        self.assertFalse(res3["exact_match"])
        self.assertFalse(res3["soft_match"])

    def test_video_transitions_tolerance(self):
        # GT transition at 3.4 seconds
        # Pred transition at 3.6 seconds (difference = 0.2s <= 1.5s tolerance)
        res = evaluate_video_transitions([3.6], [3.4], tolerance_sec=1.5)
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_positives"], 0)
        self.assertEqual(res["false_negatives"], 0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1_score"], 1.0)

    def test_video_transitions_missed_and_extra(self):
        # GT transitions at 2.0 and 8.0
        # Pred transitions at 2.2 and 15.0
        res = evaluate_video_transitions([2.2, 15.0], [2.0, 8.0], tolerance_sec=1.5)
        self.assertEqual(res["true_positives"], 1)  # 2.2 matches 2.0
        self.assertEqual(res["false_positives"], 1)  # 15.0 has no match
        self.assertEqual(res["false_negatives"], 1)  # 8.0 is missed
        self.assertEqual(res["precision"], 0.5)
        self.assertEqual(res["recall"], 0.5)
        self.assertEqual(res["f1_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
