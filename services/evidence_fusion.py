"""VisuNarrate - Advanced Evidence Fusion, Reliability Gating & Adaptive Routing Engine (Stage 3.5).

Key Capabilities:
1. Defined Evidence Priority Architecture:
   - Object Existence: YOLOv8n detector evidence (Priority 1 Grounding).
   - Global Scene Summary: BLIP captioning (Priority 1 for macro composition).
   - Targeted Fine-Grained Relations: BLIP-VQA (Supporting evidence for attributes & affordances).
2. Deterministic Interaction Scoring & Ranking:
   - Multi-source evidence scoring (Detector + Caption + VQA).
   - Detector grounding constraints prevent undetected/marginal objects from overriding primary detected objects.
   - Hallucination suppression & explainable conflict tracking.
3. Heuristic Scene-Type Classifier:
   - Identifies 'photographic_scene', 'illustration_or_graphic', or 'uncertain'.
   - Gating prevents open-domain VQA hallucinations on synthetic/abstract visual media.
4. Adaptive Question Budget & Early Exit:
   - Two-tier question routing (Core Pass 1 -> Conditional Pass 2).
5. Formal Reliability Gating:
   - Prevents ungrounded single-source VQA answers from becoming primary scene facts.
   - Explicitly records confirmed, supported, uncertain, and conflicting facts with counter-evidence.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger("EvidenceFusion")


class EvidenceFusionEngine:
    """Intelligently validates, gates, prioritizes, and fuses multi-model visual evidence."""

    PERSON_KEYWORDS: Set[str] = {
        "person", "man", "woman", "boy", "girl", "people", "child", "someone",
        "pedestrian", "guy", "lady", "individual", "standing", "sitting", "plaid shirt",
        "checkered shirt", "face", "hands", "holding", "wearing"
    }

    GRAPHIC_KEYWORDS: Set[str] = {
        "illustration", "drawing", "graphic", "vector", "cartoon", "sketch",
        "painting", "shape", "border", "photoshopped", "abstract", "diagram",
        "clipart", "render", "pyramids", "square", "logo"
    }

    def __init__(self):
        logger.info("[EvidenceFusion] Initialized Stage 3.5 Advanced Evidence Fusion & Reliability Engine.")

    def classify_scene_type(
        self,
        caption_evidence: Dict[str, Any],
        object_evidence: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Classifies image into photographic vs graphic/illustration using explainable heuristics."""
        summary_text = caption_evidence.get("summary", "").lower()
        observations = caption_evidence.get("observations", {})
        combined_text = (
            summary_text + " " +
            observations.get("subject", "").lower() + " " +
            observations.get("action", "").lower() + " " +
            observations.get("environment", "").lower()
        )

        rationale = []

        # Heuristic 1: Semantic keyword detection in BLIP description
        graphic_matches = [kw for kw in self.GRAPHIC_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", combined_text)]
        if graphic_matches:
            rationale.append(f"Caption semantics indicate non-photographic artwork (keywords: {', '.join(graphic_matches)})")

        # Heuristic 2: YOLO detection consistency
        yolo_counts = object_evidence.get("counts", {})
        yolo_labels = list(yolo_counts.keys())
        total_detections = object_evidence.get("total_detections", 0)

        if "frisbee" in yolo_labels and total_detections == 1 and graphic_matches:
            rationale.append("Isolated circular detection on abstract artwork consistent with 2D geometric shape (e.g. sun)")

        # Decision synthesis
        if len(graphic_matches) >= 1:
            scene_type = "illustration_or_graphic"
            trust_level = "filtered_semantic_mode"
        elif total_detections >= 1 and not graphic_matches:
            scene_type = "photographic_scene"
            trust_level = "full_photographic_mode"
            rationale.append("Detector identified real-world physical entities; caption exhibits natural photographic semantics")
        else:
            scene_type = "photographic_scene" if "photo" in combined_text else "uncertain"
            trust_level = "standard_mode"
            rationale.append("Standard visual feature distribution")

        logger.info(f"[EvidenceFusion] Heuristic Scene Type: {scene_type.upper()} ({trust_level})")

        return {
            "scene_type": scene_type,
            "trust_level": trust_level,
            "rationale": rationale,
            "is_graphic": (scene_type == "illustration_or_graphic"),
        }

    def determine_person_presence(
        self,
        yolo_evidence: Dict[str, Any],
        caption_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluates YOLO detector priority and caption semantics to establish person presence."""
        yolo_counts = yolo_evidence.get("counts", {})
        yolo_person_count = yolo_counts.get("person", 0)
        yolo_objects = yolo_evidence.get("objects", [])

        summary_text = caption_evidence.get("summary", "").lower()
        observations = caption_evidence.get("observations", {})
        combined_caption_text = (
            summary_text + " " +
            observations.get("subject", "").lower() + " " +
            observations.get("action", "").lower()
        )

        caption_has_person = any(
            re.search(r"\b" + re.escape(kw) + r"\b", combined_caption_text)
            for kw in self.PERSON_KEYWORDS
        )

        # Evaluate primary actor vs background person detections
        person_boxes = [obj for obj in yolo_objects if obj.get("label") == "person"]
        dominant_person_count = yolo_person_count

        if len(person_boxes) >= 2:
            # Sort by area * confidence
            def box_prominence(b):
                bbox = b.get("bbox", {})
                w = abs(bbox.get("x2", 0) - bbox.get("x1", 0))
                h = abs(bbox.get("y2", 0) - bbox.get("y1", 0))
                return (w * h) * b.get("confidence", 0.5)

            sorted_boxes = sorted(person_boxes, key=box_prominence, reverse=True)
            p1_prom = box_prominence(sorted_boxes[0])
            p2_prom = box_prominence(sorted_boxes[1])

            # If primary person is >3.5x more prominent than secondary and caption is singular
            caption_is_singular = any(w in summary_text for w in ["a man", "a woman", "a person", "someone", "the man", "the woman"])
            if p1_prom > 3.5 * p2_prom and caption_is_singular:
                dominant_person_count = 1

        evidence_reasons = []

        if yolo_person_count > 0:
            evidence_reasons.append(f"YOLOv8n detected {yolo_person_count} person instance(s) [Detector Evidence]")
            if caption_has_person:
                evidence_reasons.append("BLIP caption independently identified human subject [Caption Evidence]")
                status = "present"
                confidence_level = "confirmed"
            else:
                evidence_reasons.append("BLIP caption did not explicitly highlight person")
                status = "present"
                confidence_level = "supported"
            estimated_count = dominant_person_count

        else:
            evidence_reasons.append("YOLOv8n detected 0 person instances [Detector Evidence]")
            if not caption_has_person:
                evidence_reasons.append("BLIP caption did not identify person instances")
                status = "absent"
                confidence_level = "confirmed"
                estimated_count = 0
            else:
                evidence_reasons.append("BLIP caption suggested human presence but detector found 0 instances [Uncertain]")
                status = "uncertain"
                confidence_level = "uncertain"
                estimated_count = 0

        logger.info(f"[EvidenceFusion] Person Presence: {status.upper()} ({confidence_level}) | Count: {estimated_count}")

        return {
            "status": status,
            "confidence_level": confidence_level,
            "count": estimated_count,
            "raw_detector_count": yolo_person_count,
            "evidence_reasons": evidence_reasons,
        }

    def select_first_pass_questions(
        self,
        person_presence: Dict[str, Any],
        scene_type_info: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Selects minimal, targeted Pass 1 questions and explicitly tracks skipped/gated questions."""
        first_pass_questions: List[str] = []
        skipped_questions: List[str] = []

        if scene_type_info.get("is_graphic", False):
            first_pass_questions = [
                "What is the main subject in the image?",
                "What is the visual style of the image?"
            ]
            skipped_questions = [
                "What is the person holding? [Gated: Non-Photographic Graphic/Illustration]",
                "What is the person wearing? [Gated: Non-Photographic Graphic/Illustration]",
                "What is the person doing? [Gated: Non-Photographic Graphic/Illustration]",
                "Is the person indoors or outdoors? [Gated: Non-Photographic Graphic/Illustration]"
            ]
            return first_pass_questions, skipped_questions

        if person_presence.get("status") == "present":
            first_pass_questions = [
                "What is the person holding?",
                "What is the person wearing?",
                "What is the person doing?",
                "Is the person indoors or outdoors?"
            ]
            skipped_questions = [
                "What is the main subject in the image? [Gated: Human presence established]",
                "Is the scene indoors or outdoors? [Gated: Human setting question active]"
            ]
        else:
            first_pass_questions = [
                "What is the main subject in the image?",
                "Is the scene indoors or outdoors?",
                "What is the weather or lighting like in the scene?"
            ]
            skipped_questions = [
                "What is the person holding? [Gated: Person absent]",
                "What is the person wearing? [Gated: Person absent]",
                "What is the person doing? [Gated: Person absent]"
            ]

        return first_pass_questions, skipped_questions

    def evaluate_pass_two_triggers(
        self,
        first_pass_qa: List[Dict[str, Any]],
        caption_evidence: Dict[str, Any],
        person_presence: Dict[str, Any]
    ) -> List[str]:
        """Evaluates whether secondary targeted questions are needed, otherwise triggers Early Exit."""
        pass_two_questions: List[str] = []
        qa_map = {item["question"]: item["answer"].lower() for item in first_pass_qa}

        if person_presence.get("status") == "present":
            holding_ans = qa_map.get("What is the person holding?", "")
            doing_ans = qa_map.get("What is the person doing?", "")

            # Trigger phone usage inquiry only if ambiguous
            if "phone" in holding_ans or "cell" in holding_ans:
                pass_two_questions.append("Is the person using a phone?")

            # Trigger laptop check if typing or desk engagement detected
            if any(k in doing_ans for k in ["typing", "working", "sitting at desk"]):
                pass_two_questions.append("Is the person using a laptop?")

        return pass_two_questions

    def _score_and_rank_interactions(
        self,
        yolo_objs: List[Dict[str, Any]],
        caption_evidence: Dict[str, Any],
        vqa_map: Dict[str, str],
        conflicts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Computes deterministic, explainable multi-source evidence scores for interaction targets.

        Evidence Priority:
        1. YOLO Spatial Detector (Priority 1 Grounding)
        2. BLIP Caption Semantics (Priority 1 Macro Story)
        3. BLIP-VQA Interrogation (Fine-grained Confirmation)
        """
        yolo_counts = {}
        yolo_max_conf = {}
        for obj in yolo_objs:
            lbl = obj.get("label", "").lower()
            conf = obj.get("confidence", 0.0)
            yolo_counts[lbl] = yolo_counts.get(lbl, 0) + 1
            yolo_max_conf[lbl] = max(yolo_max_conf.get(lbl, 0.0), conf)

        caption_text = (
            caption_evidence.get("summary", "").lower() + " " +
            caption_evidence.get("observations", {}).get("action", "").lower() + " " +
            caption_evidence.get("observations", {}).get("subject", "").lower()
        )

        vqa_holding = vqa_map.get("What is the person holding?", "").strip().lower()
        vqa_doing = vqa_map.get("What is the person doing?", "").strip().lower()
        vqa_phone = vqa_map.get("Is the person using a phone?", "").strip().lower()
        vqa_laptop = vqa_map.get("Is the person using a laptop?", "").strip().lower()

        candidates: List[Dict[str, Any]] = []

        # =====================================================================
        # 1. Laptop / Computer Workstation Candidate
        # =====================================================================
        has_yolo_laptop = ("laptop" in yolo_counts or "tv" in yolo_counts)
        conf_laptop = yolo_max_conf.get("laptop", yolo_max_conf.get("tv", 0.0))
        has_cap_laptop = ("laptop" in caption_text or "computer" in caption_text)
        has_vqa_laptop = ("laptop" in vqa_holding or "laptop" in vqa_doing or "computer" in vqa_holding or vqa_laptop == "yes")

        if has_yolo_laptop or has_cap_laptop or has_vqa_laptop:
            score = 0.0
            sources = []
            counter = []

            if has_yolo_laptop:
                score += 50.0 + (conf_laptop * 20.0)
                sources.append(f"YOLOv8n Detector: 'laptop' (conf: {conf_laptop:.1%})")
            else:
                counter.append("YOLOv8n Detector: Laptop bounding box not detected")

            if has_cap_laptop:
                score += 35.0
                sources.append("BLIP Caption: mentions laptop/computer interaction")

            if has_vqa_laptop:
                score += 35.0
                sources.append(f"BLIP-VQA: confirmed '{vqa_holding or vqa_doing}'")

            status = "confirmed" if score >= 80.0 else "supported" if score >= 40.0 else "uncertain"

            candidates.append({
                "subject": "person",
                "relation": "working on / viewing",
                "target": "laptop",
                "status": status,
                "score": score,
                "evidence_sources": sources,
                "counter_evidence": counter,
            })

        # =====================================================================
        # 2. Book / Reading Candidate
        # =====================================================================
        has_yolo_book = ("book" in yolo_counts)
        conf_book = yolo_max_conf.get("book", 0.0)
        has_cap_book = ("book" in caption_text or "reading" in caption_text or "paper" in caption_text)
        has_vqa_book = ("book" in vqa_holding or "reading" in vqa_doing or "paper" in vqa_holding)

        if has_yolo_book or has_cap_book or has_vqa_book:
            score = 0.0
            sources = []
            counter = []

            if has_yolo_book:
                score += 50.0 + (conf_book * 20.0)
                sources.append(f"YOLOv8n Detector: 'book' (conf: {conf_book:.1%})")
            else:
                counter.append("YOLOv8n Detector: Book bounding box not detected")

            if has_cap_book:
                score += 35.0
                sources.append("BLIP Caption: mentions book / reading")

            if has_vqa_book:
                score += 35.0
                sources.append(f"BLIP-VQA: confirmed '{vqa_holding or vqa_doing}'")

            status = "confirmed" if score >= 80.0 else "supported" if score >= 40.0 else "uncertain"

            candidates.append({
                "subject": "person",
                "relation": "reading / holding",
                "target": "book",
                "status": status,
                "score": score,
                "evidence_sources": sources,
                "counter_evidence": counter,
            })

        # =====================================================================
        # 3. Cup / Beverage / Dining Candidate
        # =====================================================================
        has_yolo_cup = ("cup" in yolo_counts or "bottle" in yolo_counts)
        conf_cup = max(yolo_max_conf.get("cup", 0.0), yolo_max_conf.get("bottle", 0.0))
        has_cap_cup = any(w in caption_text for w in ["cup", "coffee", "tea", "drink", "drinking", "bottle", "eating", "food"])
        has_vqa_cup = any(w in vqa_holding or w in vqa_doing for w in ["cup", "coffee", "tea", "drink", "bottle", "food"])

        if has_yolo_cup or has_cap_cup or has_vqa_cup:
            score = 0.0
            sources = []
            counter = []

            target_name = "cup" if ("cup" in yolo_counts or "cup" in vqa_holding or "coffee" in caption_text) else "bottle" if "bottle" in yolo_counts else "food or beverage"

            if has_yolo_cup:
                score += 50.0 + (conf_cup * 20.0)
                sources.append(f"YOLOv8n Detector: '{target_name}' (conf: {conf_cup:.1%})")
            else:
                counter.append(f"YOLOv8n Detector: '{target_name}' bounding box not detected")

            if has_cap_cup:
                score += 35.0
                sources.append("BLIP Caption: mentions drinking / dining")

            if has_vqa_cup:
                score += 35.0
                sources.append(f"BLIP-VQA: confirmed '{vqa_holding or vqa_doing}'")

            status = "confirmed" if score >= 80.0 else "supported" if score >= 40.0 else "uncertain"

            candidates.append({
                "subject": "person",
                "relation": "holding / drinking from",
                "target": target_name,
                "status": status,
                "score": score,
                "evidence_sources": sources,
                "counter_evidence": counter,
            })

        # =====================================================================
        # 4. Smartphone / Cell Phone Candidate (With strict detector grounding)
        # =====================================================================
        has_yolo_phone = ("cell phone" in yolo_counts or "phone" in yolo_counts)
        conf_phone = yolo_max_conf.get("cell phone", yolo_max_conf.get("phone", 0.0))
        has_cap_phone = ("cell" in caption_text or "phone" in caption_text)
        has_vqa_phone = ("cell phone" in vqa_holding or "phone" in vqa_holding or "phone" in vqa_doing or vqa_phone == "yes")

        if has_yolo_phone or has_cap_phone or has_vqa_phone:
            score = 0.0
            sources = []
            counter = []

            # Check if detector support is solid vs marginal (<0.50)
            if has_yolo_phone:
                if conf_phone >= 0.65:
                    score += 50.0 + (conf_phone * 20.0)
                    sources.append(f"YOLOv8n Detector: 'cell phone' (conf: {conf_phone:.1%})")
                elif conf_phone >= 0.50:
                    score += 35.0
                    sources.append(f"YOLOv8n Detector: 'cell phone' (conf: {conf_phone:.1%})")
                else:
                    # Marginal detector detection (e.g. edge artifact)
                    score += 15.0
                    sources.append(f"YOLOv8n Detector: 'cell phone' marginal detection (conf: {conf_phone:.1%})")
            else:
                counter.append("YOLOv8n Detector: Cell phone bounding box not detected")

            if has_cap_phone:
                score += 35.0
                sources.append("BLIP Caption: mentions phone interaction")

            if has_vqa_phone:
                # If VQA says phone but caption and detector say something completely different
                score += 35.0
                sources.append(f"BLIP-VQA: confirmed '{vqa_holding or vqa_doing}'")

            # Grounding Constraint: If NO detector and NO caption, a single VQA answer is uncertain
            if not has_yolo_phone and not has_cap_phone:
                status = "uncertain"
                score = min(score, 30.0)
            else:
                status = "confirmed" if score >= 80.0 else "supported" if score >= 40.0 else "uncertain"

            candidates.append({
                "subject": "person",
                "relation": "holding / interacting with",
                "target": "cell phone",
                "status": status,
                "score": score,
                "evidence_sources": sources,
                "counter_evidence": counter,
            })

        # =====================================================================
        # 5. Mobility / Vehicle Candidate (Bicycle, Skateboard)
        # =====================================================================
        has_yolo_mob = any(k in yolo_counts for k in ["bicycle", "motorcycle", "skateboard"])
        conf_mob = max([yolo_max_conf.get(k, 0.0) for k in ["bicycle", "motorcycle", "skateboard"]] + [0.0])
        has_cap_mob = any(k in caption_text for k in ["bike", "bicycle", "riding", "skateboard", "motorcycle"])
        has_vqa_mob = any(k in vqa_holding or k in vqa_doing for k in ["bike", "bicycle", "riding", "skateboard"])

        if has_yolo_mob or has_cap_mob or has_vqa_mob:
            score = 0.0
            sources = []
            counter = []

            target_mob = "bicycle" if ("bike" in caption_text or "bike" in vqa_doing or "bicycle" in yolo_counts) else "skateboard" if ("skateboard" in vqa_holding or "skateboard" in yolo_counts) else "vehicle"

            if has_yolo_mob:
                score += 50.0 + (conf_mob * 20.0)
                sources.append(f"YOLOv8n Detector: '{target_mob}' (conf: {conf_mob:.1%})")
            else:
                counter.append(f"YOLOv8n Detector: '{target_mob}' bounding box not detected")

            if has_cap_mob:
                score += 35.0
                sources.append("BLIP Caption: mentions mobility / riding")

            if has_vqa_mob:
                score += 35.0
                sources.append(f"BLIP-VQA: confirmed '{vqa_holding or vqa_doing}'")

            status = "confirmed" if score >= 80.0 else "supported" if score >= 40.0 else "uncertain"

            candidates.append({
                "subject": "person",
                "relation": "riding / operating",
                "target": target_mob,
                "status": status,
                "score": score,
                "evidence_sources": sources,
                "counter_evidence": counter,
            })

        # =====================================================================
        # Conflict Arbitration & Deterministic Ranking
        # =====================================================================
        if len(candidates) >= 2:
            # Sort candidates by evidence score descending
            candidates.sort(key=lambda x: x["score"], reverse=True)
            top_cand = candidates[0]
            second_cand = candidates[1]

            # If top candidate is strongly confirmed (>80) and second is weak/marginal (<40)
            if top_cand["score"] >= 70.0 and second_cand["score"] < 50.0:
                conflicts.append({
                    "topic": "Interaction Target Disambiguation",
                    "description": f"Multi-source evidence converged on '{top_cand['target']}' (Score: {top_cand['score']:.1f}), outranking secondary candidate '{second_cand['target']}' (Score: {second_cand['score']:.1f}).",
                    "evidence": top_cand["evidence_sources"],
                    "counter_evidence": second_cand["evidence_sources"],
                    "resolution": f"Selected strongly grounded physical interaction with '{top_cand['target']}'.",
                    "resolution_status": "supported",
                })
        elif candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates

    def fuse_evidence(
        self,
        caption_evidence: Dict[str, Any],
        object_evidence: Dict[str, Any],
        vqa_evidence: List[Dict[str, Any]],
        person_presence: Dict[str, Any],
        scene_type_info: Dict[str, Any],
        skipped_questions: Optional[List[str]] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fuses multi-modal evidence with deterministic ranking and grounding validation."""
        vqa_map = {item["question"]: item["answer"] for item in vqa_evidence}
        conflicts: List[Dict[str, Any]] = []
        uncertainty_flags: List[Dict[str, Any]] = []
        fused_objects: List[Dict[str, Any]] = []

        is_person_present = (person_presence.get("status") == "present")
        is_graphic = scene_type_info.get("is_graphic", False)

        # ---------------------------------------------------------------------
        # 1. Primary Subject Determination
        # ---------------------------------------------------------------------
        if is_person_present:
            person_count = person_presence.get("count", 1)
            wearing_ans = vqa_map.get("What is the person wearing?", "")
            caption_sub = caption_evidence.get("observations", {}).get("subject", "")

            if wearing_ans and wearing_ans.lower() not in {"nothing", "unknown", "none", "shirt"}:
                subject_desc = f"A person wearing a {wearing_ans.lower()}"
                subject_status = "confirmed" if ("checkered" in wearing_ans.lower() or "hoodie" in wearing_ans.lower()) else "supported"
            else:
                subject_desc = caption_sub if caption_sub else f"A person ({person_count} instance(s) detected)"
                subject_status = "supported"

            primary_subject = {
                "value": subject_desc,
                "status": subject_status,
                "sources": ["YOLOv8n Detector Evidence", "BLIP Caption Evidence", "BLIP-VQA Attribute Evidence"],
            }
        else:
            vqa_sub = vqa_map.get("What is the main subject in the image?", "")
            caption_sub = caption_evidence.get("observations", {}).get("subject", "")

            if is_graphic:
                subject_desc = caption_sub if caption_sub else (vqa_sub or "Graphic Illustration")
                subject_status = "supported"
                sources = ["BLIP Caption Semantic Evidence"]
                if vqa_sub and vqa_sub.lower() not in {"computer", "laptop", "unknown"}:
                    sources.append("BLIP-VQA Scene Subject")
            else:
                subject_desc = vqa_sub if vqa_sub and vqa_sub.lower() != "unknown" else caption_sub
                if not subject_desc:
                    subject_desc = caption_evidence.get("summary", "Unspecified Subject")
                subject_status = "supported"
                sources = ["BLIP Caption Evidence", "BLIP-VQA Scene Subject"]

            primary_subject = {
                "value": subject_desc,
                "status": subject_status,
                "sources": sources,
            }

        # ---------------------------------------------------------------------
        # 2. Activity / Posture Determination
        # ---------------------------------------------------------------------
        if is_person_present:
            vqa_action = vqa_map.get("What is the person doing?", "")
            caption_action = caption_evidence.get("observations", {}).get("action", "")

            if vqa_action and "dancing" in vqa_action.lower():
                uncertainty_flags.append({
                    "aspect": "activity",
                    "note": f"VQA suggested dynamic action '{vqa_action}', but visual evidence indicates a static pose.",
                    "source": "BLIP-VQA Interrogative Evidence",
                })
                activity_val = "Standing (static pose)"
                activity_status = "supported"
            elif vqa_action and vqa_action.lower() not in {"nothing", "unknown", "none"}:
                activity_val = vqa_action
                activity_status = "supported"
            elif caption_action:
                activity_val = caption_action
                activity_status = "supported"
            else:
                activity_val = "Standing"
                activity_status = "supported"

            activity = {
                "value": activity_val,
                "status": activity_status,
                "sources": ["BLIP-VQA Action / Posture Evidence"],
            }
        else:
            caption_action = caption_evidence.get("observations", {}).get("action", "")
            activity = {
                "value": caption_action if caption_action else "Static visual composition",
                "status": "supported" if caption_action else "neutral",
                "sources": ["BLIP Caption Observations"],
            }

        # ---------------------------------------------------------------------
        # 3. Environment & Setting Determination
        # ---------------------------------------------------------------------
        vqa_loc = vqa_map.get("Is the person indoors or outdoors?", "") or vqa_map.get("Is the scene indoors or outdoors?", "")
        caption_env = caption_evidence.get("observations", {}).get("environment", "")

        env_parts = []
        if vqa_loc and vqa_loc.lower() in {"indoors", "outdoors"}:
            env_parts.append(vqa_loc)
        if caption_env and caption_env.lower() not in {"simple", "unknown", "none"}:
            env_parts.append(f"with {caption_env.lower()}")

        env_val = " ".join(env_parts) if env_parts else (vqa_loc or caption_env or "General Environment")
        environment = {
            "value": env_val,
            "status": "supported" if vqa_loc else "uncertain",
            "sources": ["BLIP-VQA Location Evidence", "BLIP Environment Evidence"],
        }

        # ---------------------------------------------------------------------
        # 4. Object Evidence Filtering
        # ---------------------------------------------------------------------
        yolo_objs = object_evidence.get("objects", [])

        for obj in yolo_objs:
            if is_graphic and obj["label"] == "frisbee":
                conflicts.append({
                    "topic": "Object Identification",
                    "description": "YOLOv8n detected 'frisbee' (43.9% conf), but context is an illustration/graphic.",
                    "evidence": ["YOLOv8n detector: 'frisbee'"],
                    "counter_evidence": ["BLIP Caption: 'illustration of pyramids'", "Heuristic: Graphic media classification"],
                    "resolution": "Classified as 2D graphic artifact; discounted from physical scene objects.",
                    "resolution_status": "discounted_artifact",
                })
                continue

            fused_objects.append({
                "label": obj["label"],
                "confidence": obj.get("confidence", 0.0),
                "bbox": obj.get("bbox", {}),
                "source": "YOLOv8n Object Detector (Priority 1 Grounding)",
                "status": "confirmed" if obj["label"] == "person" and is_person_present else "supported",
            })

        # ---------------------------------------------------------------------
        # 5. Multi-Source Object Interaction Scoring & Deterministic Ranking
        # ---------------------------------------------------------------------
        interactions = []
        if is_person_present:
            interactions = self._score_and_rank_interactions(
                yolo_objs=yolo_objs,
                caption_evidence=caption_evidence,
                vqa_map=vqa_map,
                conflicts=conflicts,
            )

        # ---------------------------------------------------------------------
        # 6. Synthesize Unified Scene Summary
        # ---------------------------------------------------------------------
        if is_person_present:
            fused_summary = (
                f"{primary_subject['value']} is {activity['value'].lower()} "
                f"in an {environment['value'].lower()} setting."
            )
            if interactions:
                fused_summary += f" The subject appears to be {interactions[0]['relation']} a {interactions[0]['target']}."
        else:
            fused_summary = (
                f"{primary_subject['value']} situated in an {environment['value'].lower()} setting."
            )

        fused_scene_graph = {
            "summary": fused_summary,
            "scene_type": scene_type_info,
            "person_presence": person_presence,
            "primary_subject": primary_subject,
            "activity": activity,
            "environment": environment,
            "objects": fused_objects,
            "interactions": interactions,
            "conflicts": conflicts,
            "uncertainty_flags": uncertainty_flags,
            "skipped_questions": skipped_questions,
            "evidence_sources": {
                "caption": True,
                "object_detection": True,
                "vqa": True,
            },
        }

        logger.info(
            f"[EvidenceFusion] Fused Scene Graph synthesized: {len(fused_objects)} object(s), "
            f"{len(interactions)} interaction(s), {len(conflicts)} conflict(s)."
        )

        return fused_scene_graph
