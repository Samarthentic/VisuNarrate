"""VisuNarrate - Evaluation Metrics Engine.

Provides mathematically rigorous, offline, pure-Python metrics calculations:
- Object Detection: IoU, Precision, Recall, F1, Average Precision (AP), Mean Average Precision (mAP@0.5)
- Caption Generation: BLEU (BLEU-1, BLEU-2, BLEU-3, BLEU-4) with Brevity Penalty, ROUGE-L (LCS-based)
- Visual Question Answering (VQA): Exact Match, Soft Match
- Temporal Action/Transition Detection: Precision, Recall, F1 with tolerance window
- Zero-denominator protection: Returns explicit None / 0.0 without invalid NaN or crashes.
"""

import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Union


# =============================================================================
# Helper Utilities
# =============================================================================

def _normalize_text(text: str) -> str:
    """Normalizes text for NLP metrics (lowercase, strip punctuation, remove extraneous spaces)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenizes normalized text into word tokens."""
    norm = _normalize_text(text)
    return norm.split() if norm else []


def _box_to_coords(box: Union[List[float], Tuple[float, float, float, float], Dict[str, float]]) -> Tuple[float, float, float, float]:
    """Converts box representation to (x1, y1, x2, y2) tuple."""
    if isinstance(box, dict):
        return (float(box.get("x1", 0.0)), float(box.get("y1", 0.0)),
                float(box.get("x2", 0.0)), float(box.get("y2", 0.0)))
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    raise ValueError(f"Invalid bounding box format: {box}")


# =============================================================================
# Object Detection Metrics: IoU, Precision, Recall, F1, AP, mAP
# =============================================================================

def calculate_iou(
    box1: Union[List[float], Tuple[float, float, float, float], Dict[str, float]],
    box2: Union[List[float], Tuple[float, float, float, float], Dict[str, float]],
) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes.

    Boxes are represented as (x1, y1, x2, y2) in identical coordinate space.
    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    x1_a, y1_a, x2_a, y2_a = _box_to_coords(box1)
    x1_b, y1_b, x2_b, y2_b = _box_to_coords(box2)

    # Ensure box coordinates are ordered (min, max)
    x1_a, x2_a = min(x1_a, x2_a), max(x1_a, x2_a)
    y1_a, y2_a = min(y1_a, y2_a), max(y1_a, y2_a)
    x1_b, x2_b = min(x1_b, x2_b), max(x1_b, x2_b)
    y1_b, y2_b = min(y1_b, y2_b), max(y1_b, y2_b)

    # Intersection coordinates
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

    union_area = area_a + area_b - intersection_area
    if union_area <= 0.0:
        return 0.0

    iou = intersection_area / union_area
    return max(0.0, min(1.0, float(iou)))


def calculate_ap(recalls: List[float], precisions: List[float]) -> float:
    """Calculates Average Precision (AP) using standard VOC 11-point interpolation.

    Args:
        recalls: List of recall values.
        precisions: List of precision values corresponding to recalls.

    Returns:
        float: Average precision in [0.0, 1.0].
    """
    if not recalls or not precisions or len(recalls) != len(precisions):
        return 0.0

    # 11-point interpolation: recall levels 0.0, 0.1, ..., 1.0
    recall_levels = [i / 10.0 for i in range(11)]
    interpolated_precisions = []

    for r_level in recall_levels:
        # Maximum precision for any recall r >= r_level
        prec_candidates = [p for r, p in zip(recalls, precisions) if r >= r_level]
        if prec_candidates:
            interpolated_precisions.append(max(prec_candidates))
        else:
            interpolated_precisions.append(0.0)

    ap = sum(interpolated_precisions) / 11.0
    return round(float(ap), 4)


def evaluate_detections(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates spatial object detections against labeled ground-truth boxes.

    Each prediction dict should have:
        - "bbox": [x1, y1, x2, y2] or {"x1", "y1", "x2", "y2"}
        - "label": str (class name)
        - "confidence": float

    Each ground_truth dict should have:
        - "bbox": [x1, y1, x2, y2] or {"x1", "y1", "x2", "y2"}
        - "label": str (class name)

    Returns:
        Dict containing precision, recall, f1, TP, FP, FN, per_class metrics.
    """
    # Group by class label (normalized to lowercase)
    all_classes = set(
        [p.get("label", "").lower() for p in predictions if p.get("label")] +
        [g.get("label", "").lower() for g in ground_truth if g.get("label")]
    )

    per_class_results: Dict[str, Dict[str, Any]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in sorted(all_classes):
        cls_preds = [p for p in predictions if p.get("label", "").lower() == cls]
        cls_gts = [g for g in ground_truth if g.get("label", "").lower() == cls]

        # Sort predictions by confidence descending
        cls_preds.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        matched_gt_indices = set()
        tp = 0
        fp = 0

        # For AP calculation curve
        cls_tps: List[int] = []
        cls_fps: List[int] = []

        for pred in cls_preds:
            pred_box = pred.get("bbox")
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(cls_gts):
                if gt_idx in matched_gt_indices:
                    continue
                iou = calculate_iou(pred_box, gt.get("bbox"))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
                cls_tps.append(1)
                cls_fps.append(0)
            else:
                fp += 1
                cls_tps.append(0)
                cls_fps.append(1)

        fn = len(cls_gts) - len(matched_gt_indices)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Compute per-class precision, recall, F1
        denom_p = tp + fp
        denom_r = tp + fn
        p_val = round(tp / denom_p, 4) if denom_p > 0 else 0.0
        r_val = round(tp / denom_r, 4) if denom_r > 0 else 0.0
        denom_f1 = p_val + r_val
        f1_val = round((2 * p_val * r_val) / denom_f1, 4) if denom_f1 > 0 else 0.0

        # Calculate AP curve if predictions exist
        if cls_preds and len(cls_gts) > 0:
            cum_tp = 0
            cum_fp = 0
            rec_curve = []
            prec_curve = []
            for t_flag, f_flag in zip(cls_tps, cls_fps):
                cum_tp += t_flag
                cum_fp += f_flag
                cur_rec = cum_tp / len(cls_gts)
                cur_prec = cum_tp / (cum_tp + cum_fp)
                rec_curve.append(cur_rec)
                prec_curve.append(cur_prec)
            cls_ap = calculate_ap(rec_curve, prec_curve)
        elif len(cls_gts) == 0:
            cls_ap = None  # Class not in ground truth
        else:
            cls_ap = 0.0

        per_class_results[cls] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ground_truth_count": len(cls_gts),
            "prediction_count": len(cls_preds),
            "precision": p_val,
            "recall": r_val,
            "f1_score": f1_val,
            "average_precision": cls_ap,
        }

    # Macro-average mAP across classes present in ground truth
    classes_with_gt = [c for c, res in per_class_results.items() if res["ground_truth_count"] > 0]
    ap_values = [
        per_class_results[c]["average_precision"]
        for c in classes_with_gt
        if per_class_results[c]["average_precision"] is not None
    ]
    mean_ap = round(sum(ap_values) / len(ap_values), 4) if ap_values else 0.0

    # Overall micro precision, recall, F1
    total_denom_p = total_tp + total_fp
    total_denom_r = total_tp + total_fn
    overall_p = round(total_tp / total_denom_p, 4) if total_denom_p > 0 else 0.0
    overall_r = round(total_tp / total_denom_r, 4) if total_denom_r > 0 else 0.0
    denom_f = overall_p + overall_r
    overall_f1 = round((2 * overall_p * overall_r) / denom_f, 4) if denom_f > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth),
        "precision": overall_p,
        "recall": overall_r,
        "f1_score": overall_f1,
        "mAP_50": mean_ap,
        "evaluated_classes_count": len(classes_with_gt),
        "per_class": per_class_results,
    }


def calculate_map(
    dataset_evaluations: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculates aggregate mAP and detection metrics across a full dataset of images.

    Args:
        dataset_evaluations: List of dicts, each with:
            - "predictions": list of detection dicts (bbox, label, confidence)
            - "ground_truth": list of ground-truth dicts (bbox, label)
        iou_threshold: IoU threshold for matching (default 0.5).

    Returns:
        Aggregate detection metrics dict.
    """
    all_preds: List[Dict[str, Any]] = []
    all_gts: List[Dict[str, Any]] = []

    # Assign image-level identifier to prevent cross-image box matching
    for img_idx, item in enumerate(dataset_evaluations):
        for p in item.get("predictions", []):
            pred_copy = dict(p)
            pred_copy["img_id"] = img_idx
            all_preds.append(pred_copy)
        for g in item.get("ground_truth", []):
            gt_copy = dict(g)
            gt_copy["img_id"] = img_idx
            all_gts.append(gt_copy)

    # Class labels
    all_classes = set(
        [p.get("label", "").lower() for p in all_preds if p.get("label")] +
        [g.get("label", "").lower() for g in all_gts if g.get("label")]
    )

    per_class_results: Dict[str, Dict[str, Any]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in sorted(all_classes):
        cls_preds = [p for p in all_preds if p.get("label", "").lower() == cls]
        cls_gts = [g for g in all_gts if g.get("label", "").lower() == cls]

        cls_preds.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        matched_gt_keys = set()
        tp = 0
        fp = 0
        cls_tps = []
        cls_fps = []

        for p_idx, pred in enumerate(cls_preds):
            p_img = pred.get("img_id")
            pred_box = pred.get("bbox")
            best_iou = 0.0
            best_gt_key = None

            # Only match ground-truth boxes in the same image
            for g_idx, gt in enumerate(cls_gts):
                if gt.get("img_id") != p_img:
                    continue
                gt_key = (gt.get("img_id"), g_idx)
                if gt_key in matched_gt_keys:
                    continue
                iou = calculate_iou(pred_box, gt.get("bbox"))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_key = gt_key

            if best_iou >= iou_threshold and best_gt_key is not None:
                tp += 1
                matched_gt_keys.add(best_gt_key)
                cls_tps.append(1)
                cls_fps.append(0)
            else:
                fp += 1
                cls_tps.append(0)
                cls_fps.append(1)

        fn = len(cls_gts) - len(matched_gt_keys)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        denom_p = tp + fp
        denom_r = tp + fn
        p_val = round(tp / denom_p, 4) if denom_p > 0 else 0.0
        r_val = round(tp / denom_r, 4) if denom_r > 0 else 0.0
        denom_f1 = p_val + r_val
        f1_val = round((2 * p_val * r_val) / denom_f1, 4) if denom_f1 > 0 else 0.0

        if cls_preds and len(cls_gts) > 0:
            cum_tp = 0
            cum_fp = 0
            rec_curve = []
            prec_curve = []
            for t_flag, f_flag in zip(cls_tps, cls_fps):
                cum_tp += t_flag
                cum_fp += f_flag
                rec_curve.append(cum_tp / len(cls_gts))
                prec_curve.append(cum_tp / (cum_tp + cum_fp))
            cls_ap = calculate_ap(rec_curve, prec_curve)
        elif len(cls_gts) == 0:
            cls_ap = None
        else:
            cls_ap = 0.0

        per_class_results[cls] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ground_truth_count": len(cls_gts),
            "prediction_count": len(cls_preds),
            "precision": p_val,
            "recall": r_val,
            "f1_score": f1_val,
            "average_precision": cls_ap,
        }

    classes_with_gt = [c for c, res in per_class_results.items() if res["ground_truth_count"] > 0]
    ap_values = [
        per_class_results[c]["average_precision"]
        for c in classes_with_gt
        if per_class_results[c]["average_precision"] is not None
    ]
    mean_ap = round(sum(ap_values) / len(ap_values), 4) if ap_values else 0.0

    total_denom_p = total_tp + total_fp
    total_denom_r = total_tp + total_fn
    overall_p = round(total_tp / total_denom_p, 4) if total_denom_p > 0 else 0.0
    overall_r = round(total_tp / total_denom_r, 4) if total_denom_r > 0 else 0.0
    denom_f = overall_p + overall_r
    overall_f1 = round((2 * overall_p * overall_r) / denom_f, 4) if denom_f > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "total_images_evaluated": len(dataset_evaluations),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "total_predictions": len(all_preds),
        "total_ground_truth": len(all_gts),
        "precision": overall_p,
        "recall": overall_r,
        "f1_score": overall_f1,
        "mAP_50": mean_ap,
        "evaluated_classes_count": len(classes_with_gt),
        "per_class": per_class_results,
    }


# =============================================================================
# Caption Generation Metrics: BLEU-1 to BLEU-4 and ROUGE-L
# =============================================================================

def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extracts n-grams from a list of tokens."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def calculate_bleu(
    candidate: str,
    references: List[str],
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, Any]:
    """Calculates BLEU scores (BLEU-1 through BLEU-4) with Brevity Penalty.

    Standard Papineni et al. (2002) formulation with Chen & Cherry smoothing.

    Args:
        candidate: Generated caption string.
        references: List of reference ground-truth captions.
        max_n: Maximum n-gram order (default: 4).
        smoothing: Whether to apply smoothing for missing higher-order n-grams.

    Returns:
        Dict containing bleu_1, bleu_2, bleu_3, bleu_4, brevity_penalty, length_ratio.
    """
    cand_tokens = _tokenize(candidate)
    ref_token_lists = [_tokenize(r) for r in references if _tokenize(r)]

    if not cand_tokens or not ref_token_lists:
        return {
            "bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
            "brevity_penalty": 0.0,
            "candidate_len": len(cand_tokens),
            "closest_reference_len": 0,
            "length_ratio": 0.0,
        }

    c_len = len(cand_tokens)
    # Closest reference length for brevity penalty
    ref_lens = [len(r) for r in ref_token_lists]
    closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - c_len), r_len))

    # Brevity Penalty (BP)
    if c_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / c_len)) if c_len > 0 else 0.0

    precisions: List[float] = []

    for n in range(1, max_n + 1):
        cand_ngrams = _get_ngrams(cand_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue

        cand_counts = Counter(cand_ngrams)
        total_cand_ngrams = len(cand_ngrams)

        # Max count of each n-gram across all references
        max_ref_counts: Dict[Tuple[str, ...], int] = {}
        for r_tokens in ref_token_lists:
            r_ngrams = _get_ngrams(r_tokens, n)
            r_counts = Counter(r_ngrams)
            for ng in cand_counts:
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), r_counts.get(ng, 0))

        # Clipped counts
        clipped_count = sum(min(count, max_ref_counts.get(ng, 0)) for ng, count in cand_counts.items())

        if total_cand_ngrams > 0:
            if clipped_count > 0:
                p_n = clipped_count / total_cand_ngrams
            elif smoothing:
                # Epsilon smoothing prevents log(0) from zeroing out lower valid n-grams
                p_n = 1.0 / (2.0 ** n * total_cand_ngrams)
            else:
                p_n = 0.0
        else:
            p_n = 0.0

        precisions.append(p_n)

    # Compute cumulative geometric mean for BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu_scores = {}
    for k in range(1, max_n + 1):
        sub_precisions = precisions[:k]
        if all(p > 0.0 for p in sub_precisions):
            log_sum = sum(math.log(p) for p in sub_precisions) / float(k)
            score = bp * math.exp(log_sum)
        else:
            score = 0.0
        bleu_scores[f"bleu_{k}"] = round(float(score), 4)

    return {
        "bleu_1": bleu_scores.get("bleu_1", 0.0),
        "bleu_2": bleu_scores.get("bleu_2", 0.0),
        "bleu_3": bleu_scores.get("bleu_3", 0.0),
        "bleu_4": bleu_scores.get("bleu_4", 0.0),
        "brevity_penalty": round(float(bp), 4),
        "candidate_len": c_len,
        "closest_reference_len": closest_ref_len,
        "length_ratio": round(c_len / closest_ref_len, 4) if closest_ref_len > 0 else 0.0,
    }


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """Computes the length of the Longest Common Subsequence (LCS) between two token sequences."""
    m = len(seq_a)
    n = len(seq_b)
    if m == 0 or n == 0:
        return 0

    # 2-row dynamic programming for memory efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[n]


def calculate_rouge_l(
    candidate: str,
    references: List[str],
    beta: float = 1.2,
) -> Dict[str, Any]:
    """Calculates ROUGE-L based on Longest Common Subsequence (LCS).

    Standard Lin (2004) formulation.

    Args:
        candidate: Generated caption string.
        references: List of ground-truth reference captions.
        beta: Weighting factor favoring recall over precision (default: 1.2).

    Returns:
        Dict containing rouge_l_precision, rouge_l_recall, rouge_l_f1.
    """
    cand_tokens = _tokenize(candidate)
    ref_token_lists = [_tokenize(r) for r in references if _tokenize(r)]

    if not cand_tokens or not ref_token_lists:
        return {
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_f1": 0.0,
            "lcs_length": 0,
        }

    c_len = len(cand_tokens)

    # Find maximum LCS across all references
    best_lcs = 0
    best_ref_len = 0
    for r_tokens in ref_token_lists:
        lcs = _lcs_length(cand_tokens, r_tokens)
        if lcs > best_lcs:
            best_lcs = lcs
            best_ref_len = len(r_tokens)
        elif best_lcs == 0 and len(r_tokens) > best_ref_len:
            best_ref_len = len(r_tokens)

    if best_ref_len == 0:
        best_ref_len = len(ref_token_lists[0])

    p_lcs = (best_lcs / c_len) if c_len > 0 else 0.0
    r_lcs = (best_lcs / best_ref_len) if best_ref_len > 0 else 0.0

    beta_sq = beta * beta
    denom = r_lcs + (beta_sq * p_lcs)
    if denom > 0.0:
        f_lcs = ((1.0 + beta_sq) * r_lcs * p_lcs) / denom
    else:
        f_lcs = 0.0

    return {
        "rouge_l_precision": round(float(p_lcs), 4),
        "rouge_l_recall": round(float(r_lcs), 4),
        "rouge_l_f1": round(float(f_lcs), 4),
        "lcs_length": best_lcs,
    }


# =============================================================================
# Visual Question Answering (VQA) Accuracy
# =============================================================================

def evaluate_vqa(
    predicted_answer: str,
    reference_answers: List[str],
) -> Dict[str, Any]:
    """Evaluates a VQA response against accepted reference answers.

    Computes:
    - Exact Match: strict match after lowercasing, punctuation stripping, and article removal.
    - Soft Match: substring containment or partial semantic token overlap.

    Args:
        predicted_answer: Model output string.
        reference_answers: List of valid ground-truth answers.

    Returns:
        Dict with exact_match (bool), soft_match (bool), and normalized strings.
    """
    def _clean_vqa(ans: str) -> str:
        s = _normalize_text(ans)
        # Strip common leading stop articles
        tokens = s.split()
        if tokens and tokens[0] in {"a", "an", "the"}:
            tokens = tokens[1:]
        return " ".join(tokens)

    pred_norm = _clean_vqa(predicted_answer)
    refs_norm = [_clean_vqa(r) for r in reference_answers if _clean_vqa(r)]

    if not pred_norm or not refs_norm:
        return {
            "exact_match": False,
            "soft_match": False,
            "predicted_clean": pred_norm,
            "references_clean": refs_norm,
        }

    # Exact match check
    exact = pred_norm in refs_norm

    # Soft match check: substring containment or token overlap >= 0.5
    soft = exact
    if not soft:
        pred_set = set(pred_norm.split())
        for ref in refs_norm:
            if pred_norm in ref or ref in pred_norm:
                soft = True
                break
            ref_set = set(ref.split())
            if pred_set and ref_set:
                overlap = len(pred_set.intersection(ref_set)) / len(pred_set)
                if overlap >= 0.5:
                    soft = True
                    break

    return {
        "exact_match": exact,
        "soft_match": soft,
        "predicted_clean": pred_norm,
        "references_clean": refs_norm,
    }


# =============================================================================
# Video Temporal Action & Transition Metrics
# =============================================================================

def evaluate_video_transitions(
    predicted_transitions: List[float],
    ground_truth_transitions: List[float],
    tolerance_sec: float = 1.5,
) -> Dict[str, Any]:
    """Evaluates detected temporal transition timestamps against ground truth.

    A predicted transition matches a ground-truth transition if:
    |t_pred - t_gt| <= tolerance_sec.

    Args:
        predicted_transitions: Timestamps (in seconds) of detected state changes.
        ground_truth_transitions: Ground-truth transition timestamps (seconds).
        tolerance_sec: Temporal matching window in seconds (default: 1.5s).

    Returns:
        Dict with precision, recall, f1_score, TP, FP, FN.
    """
    preds_sorted = sorted([float(t) for t in predicted_transitions])
    gts_sorted = sorted([float(t) for t in ground_truth_transitions])

    matched_gts = set()
    tp = 0
    fp = 0

    for p in preds_sorted:
        best_diff = float("inf")
        best_gt_idx = -1
        for g_idx, g in enumerate(gts_sorted):
            if g_idx in matched_gts:
                continue
            diff = abs(p - g)
            if diff <= tolerance_sec and diff < best_diff:
                best_diff = diff
                best_gt_idx = g_idx

        if best_gt_idx >= 0:
            tp += 1
            matched_gts.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gts_sorted) - len(matched_gts)

    denom_p = tp + fp
    denom_r = tp + fn
    precision = round(tp / denom_p, 4) if denom_p > 0 else 0.0
    recall = round(tp / denom_r, 4) if denom_r > 0 else 0.0
    denom_f = precision + recall
    f1 = round((2 * precision * recall) / denom_f, 4) if denom_f > 0 else 0.0

    return {
        "tolerance_sec": tolerance_sec,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_count": len(preds_sorted),
        "ground_truth_count": len(gts_sorted),
        "predicted_timestamps": preds_sorted,
        "ground_truth_timestamps": gts_sorted,
    }
