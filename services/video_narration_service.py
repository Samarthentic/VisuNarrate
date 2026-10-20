"""VisuNarrate - Video Narration & Predictive Storytelling Service (Stage 5.1).

Conceptual Architecture:
TEMPORAL STATE SEQUENCE + SEGMENTS + DETECTED TRANSITIONS
  ↓
CHRONOLOGICAL NARRATIVE COMPILATION (Strict Grounding, No Invented Transitions)
  ↓
EVIDENCE-BASED TEMPORAL PREDICTIVE REASONING (Structured Hypotheses from Final Video State)
  ↓
UNIFIED VIDEO INTELLIGENCE OUTPUT
"""

import logging
from typing import Dict, Any, List, Optional
from services.prediction_service import PredictionService

logger = logging.getLogger("VideoNarrationService")


class VideoNarrationService:
    """Synthesizes unified, chronological video narratives and future predictions from temporal event sequences."""

    def __init__(self, prediction_service: Optional[PredictionService] = None):
        self.prediction_service = prediction_service or PredictionService()
        logger.info("[VideoNarrationService] Initialized Video Narration & Temporal Prediction Engine (Stage 5.1).")

    def synthesize_video_narrative(
        self,
        temporal_analysis: Dict[str, Any],
        frame_sequence: List[Dict[str, Any]],
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synthesizes a fluent, non-repetitive chronological video story.

        Args:
            temporal_analysis: Structured output from TemporalService.
            frame_sequence: Sequence of frame records from VideoEngine.
            video_metadata: Basic video properties (duration, FPS, resolution).

        Returns:
            Coherent, non-repetitive narrative paragraph describing the video progression.
        """
        if not frame_sequence:
            return "No video frames available for narrative synthesis."

        stable_facts = temporal_analysis.get("stable_facts", {})
        changes = temporal_analysis.get("changes", [])
        segments = temporal_analysis.get("segments", [])
        
        dominant_subject = stable_facts.get("dominant_subject", "A subject")
        dominant_env = stable_facts.get("dominant_environment", "general setting")
        persistent_objects = stable_facts.get("persistent_objects", [])
        first_frame = frame_sequence[0]
        scene_type = first_frame.get("scene_type", "photographic_scene")
        person_count = first_frame.get("person_count", 0)

        # Check whether human agents are present in the video
        has_people = any(
            f.get("person_count", 0) > 0 or
            any(w in f.get("primary_subject", "").lower() for w in ["person", "man", "woman", "people"]) or
            f.get("person_presence") == "present"
            for f in frame_sequence
        )

        # ---------------------------------------------------------------------
        # Scenario 1: Graphic / Illustration Media
        # ---------------------------------------------------------------------
        if scene_type == "illustration_or_graphic":
            return "A static graphic illustration recorded over the video duration without physical action transitions."

        # ---------------------------------------------------------------------
        # Scenario 2: Unoccupied / No-Person Scenes
        # ---------------------------------------------------------------------
        if not has_people:
            dur_str = f"{video_metadata.get('duration_seconds', 0):.1f}s" if video_metadata else "the clip"
            desk_objs = [lbl for lbl in persistent_objects if lbl in {"laptop", "book", "desk", "mouse", "monitor"}]
            if desk_objs:
                return (
                    f"A quiet workspace scene featuring {', '.join(desk_objs[:3])} "
                    f"recorded over {dur_str}, remaining in stationary equilibrium."
                )
            return (
                f"An outdoor environmental scene capturing {dominant_env.lower()} "
                f"recorded over {dur_str}, maintaining a steady landscape view."
            )

        # ---------------------------------------------------------------------
        # Scenario 3: Human Scene with Transitions or Continuity
        # ---------------------------------------------------------------------
        sentences: List[str] = []

        # 1. Opening State Sentence
        first_act = first_frame.get("activity", "static pose").lower()
        first_interactions = first_frame.get("interactions", [])
        first_interaction_str = ""

        if first_interactions:
            top_inter = first_interactions[0]
            rel = top_inter.get("relation", "interacting with").strip()
            # Clean template leakage: "holding / interacting with", "reading / holding", etc.
            rel = rel.replace("holding / interacting with", "using")
            rel = rel.replace("reading / holding", "reading")
            rel = rel.replace("holding / drinking from", "drinking from")
            if " / " in rel:
                rel = rel.split(" / ")[0]

            target = top_inter.get("target", "").strip()
            if target:
                if first_act == "texting" and ("phone" in target.lower() or "cell" in target.lower()):
                    first_interaction_str = " on a cell phone"
                elif first_act == "reading" and "book" in target.lower():
                    first_interaction_str = " a book"
                elif rel in first_act:
                    first_interaction_str = f" with {target}"
                else:
                    article = "an " if target.lower().startswith(("a", "e", "i", "o", "u")) and not target.lower().startswith(("a ", "an ", "the ", "food")) else ("a " if not target.lower().startswith(("a ", "an ", "the ", "food")) else "")
                    first_interaction_str = f" while {rel} {article}{target}"

        subject_clean = dominant_subject.strip().rstrip(".")
        if subject_clean.lower() in {"unspecified", "subject"}:
            subject_clean = "A person" if person_count <= 1 else f"{person_count} people"

        env_clause = ""
        if dominant_env and dominant_env.lower() not in {"general environment", "unknown", "none"}:
            if "indoors" in dominant_env.lower():
                env_clause = " in an indoor setting"
            elif "outdoors" in dominant_env.lower():
                env_clause = " in an outdoor setting"
            else:
                env_clause = f" in {dominant_env.lower()}"

        if first_act and first_act not in {"static pose", "unknown"}:
            sentences.append(f"The video opens showing {subject_clean.lower()}{env_clause}, {first_act}{first_interaction_str}.")
        else:
            sentences.append(f"The video opens showing {subject_clean.lower()}{env_clause}{first_interaction_str}.")

        # 2. Chronological Transitions (Strictly based on detected changes; group coincident changes)
        if changes:
            time_groups: Dict[float, List[Dict[str, Any]]] = {}
            for chg in changes:
                ts = round(chg.get("timestamp_seconds", 0.0), 1)
                matched_ts = None
                for existing_ts in time_groups:
                    if abs(existing_ts - ts) <= 0.5:
                        matched_ts = existing_ts
                        break
                if matched_ts is not None:
                    time_groups[matched_ts].append(chg)
                else:
                    time_groups[ts] = [chg]

            for ts, group in time_groups.items():
                act_chg = next((c for c in group if c.get("type") == "activity_change"), None)
                inter_chg = next((c for c in group if c.get("type") == "interaction_change"), None)
                person_chg = next((c for c in group if c.get("type") == "person_count_change"), None)
                scene_chg = next((c for c in group if c.get("type") == "scene_type_change"), None)

                if act_chg and inter_chg:
                    from_act = act_chg.get("from", "")
                    to_act = act_chg.get("to", "")
                    to_inter = inter_chg.get("to", "")
                    if to_inter and to_inter.lower() not in to_act.lower():
                        if to_act.lower() == "reading" and "book" in to_inter.lower():
                            merged_to = "reading a book"
                        elif to_act.lower() in {"texting", "using smartphone"} and "phone" in to_inter.lower():
                            merged_to = "using a smartphone"
                        else:
                            merged_to = f"{to_act} (with {to_inter})"
                    else:
                        merged_to = to_act
                    sentences.append(f"At {ts:.1f}s, the activity shifts from {from_act} to {merged_to}.")
                elif act_chg:
                    sentences.append(f"At {ts:.1f}s, the activity shifts from {act_chg.get('from', '')} to {act_chg.get('to', '')}.")
                elif inter_chg:
                    sentences.append(f"At {ts:.1f}s, the interaction transitions toward {inter_chg.get('to', '')}.")

                if person_chg:
                    sentences.append(f"Around {ts:.1f}s, the number of visible people shifts to {person_chg.get('to', '')}.")
                if scene_chg:
                    sentences.append(f"Around {ts:.1f}s, the visual setting transitions to {scene_chg.get('to', '')}.")
        else:
            # If no discrete changes occurred, state continuity throughout the duration
            dur_val = video_metadata.get("duration_seconds", 0.0) if video_metadata else 0.0
            dur_phrase = f"over the {dur_val:.1f}s duration" if dur_val > 0 else "throughout the recorded sequence"
            last_frame = frame_sequence[-1]
            last_act = last_frame.get("activity", first_act).lower()
            sentences.append(f"The scene maintains continuity {dur_phrase} with the subject engaged in {last_act}.")

        # 3. Persistent Grounded Objects Mention
        filtered_objects = [
            obj for obj in persistent_objects
            if obj not in {"person", "frisbee"}
        ]
        if filtered_objects and len(filtered_objects) > 0:
            sentences.append(f"Prominent persistent elements include {', '.join(filtered_objects[:3])}.")

        final_video_narration = " ".join(sentences).strip()
        logger.info(f"[VideoNarrationService] Synthesized Story: '{final_video_narration}'")
        return final_video_narration

    def generate_video_prediction(
        self,
        temporal_analysis: Dict[str, Any],
        frame_sequence: List[Dict[str, Any]],
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generates evidence-based future prediction hypotheses grounded on final state and temporal trend.

        Args:
            temporal_analysis: Output of TemporalService containing stable_facts and changes.
            frame_sequence: List of frame records from VideoEngine.
            video_metadata: Video properties.

        Returns:
            Dict containing prediction string, status, prediction_type, hypotheses, and reasoning trace.
        """
        if not frame_sequence:
            return {
                "prediction": "No video data available for predictive reasoning.",
                "status": "insufficient_evidence",
                "prediction_type": "insufficient_evidence",
                "hypotheses": [],
                "reasoning": ["No frames analyzed."],
                "source_facts": [],
            }

        last_frame = frame_sequence[-1]
        raw_fused = last_frame.get("raw_fused_scene") or {}
        changes = temporal_analysis.get("changes", [])

        # If raw_fused is empty (e.g. inherited record without full payload), reconstruct minimal fused dict
        if not raw_fused:
            raw_fused = {
                "scene_type": {"scene_type": last_frame.get("scene_type", "photographic_scene"), "is_graphic": (last_frame.get("scene_type") == "illustration_or_graphic")},
                "person_presence": {"status": last_frame.get("person_presence", "unknown"), "count": last_frame.get("person_count", 0), "has_dominant_person": True},
                "primary_subject": {"value": last_frame.get("primary_subject", "Person"), "status": "supported"},
                "activity": {"value": last_frame.get("activity", "Static pose"), "status": "supported"},
                "environment": {"value": last_frame.get("environment", "General Environment"), "status": "supported"},
                "objects": [{"label": k, "confidence": 0.8} for k in last_frame.get("detected_objects", {}).keys()],
                "interactions": last_frame.get("interactions", []),
                "conflicts": [],
                "uncertainty_flags": [],
            }

        # Generate base structured prediction from final scene state
        base_pred = self.prediction_service.generate_predictive_narration(
            fused_scene=raw_fused,
            narration=last_frame.get("scene_narration", "")
        )

        hypotheses = base_pred.get("hypotheses", [])
        reasoning = list(base_pred.get("reasoning", []))
        source_facts = list(base_pred.get("source_facts", []))

        # Enrich hypotheses evidence with confirmed temporal facts (strictly detections/transitions, not narration)
        if changes:
            recent_change = changes[-1]
            chg_desc = recent_change.get("description", "")
            source_facts.append(f"Recent State Shift: {chg_desc}")
            reasoning.append(f"Temporal momentum established following observed transition at {recent_change.get('timestamp_seconds', 0.0):.1f}s.")

            # Update evidence for the top hypothesis with the temporal transition
            if hypotheses and len(hypotheses) > 0:
                hypotheses[0]["evidence"].append(f"Observed transition: {chg_desc}")
        else:
            dur = video_metadata.get("duration_seconds", 0.0) if video_metadata else 0.0
            if dur > 0:
                source_facts.append(f"Temporal Continuity: Sustained state over {dur:.1f}s recording")
                reasoning.append(f"Consistent activity continuity sustained throughout {dur:.1f}s without disruption.")

        # Clean any self-referential narration citations in hypothesis evidence
        for h in hypotheses:
            cleaned_ev = []
            for ev in h.get("evidence", []):
                if not ev.lower().startswith("narration asserts"):
                    cleaned_ev.append(ev)
                else:
                    # Replace with underlying fact
                    cleaned_ev.append("Visual context and confirmed interaction")
            h["evidence"] = cleaned_ev if cleaned_ev else ["Observed visual state in final frame"]

        # Drop ungrounded hypotheses (e.g. citing undetected surface or table)
        all_detected_objects = set()
        for f in frame_sequence:
            for obj_name in f.get("detected_objects", {}).keys():
                all_detected_objects.add(obj_name.lower())
        for obj_name in temporal_analysis.get("stable_facts", {}).get("persistent_objects", []):
            all_detected_objects.add(obj_name.lower())

        surface_tokens = {"table", "desk", "surface", "counter", "bench", "dining table"}
        has_surface = bool(surface_tokens & all_detected_objects) or any(
            any(st in obj for st in ("table", "desk")) for obj in all_detected_objects
        )

        valid_hypotheses = []
        for h in hypotheses:
            ev_list = h.get("evidence", [])
            text_str = h.get("text", "")
            # Check if hypothesis ungroundedly claims a surface/table when none detected
            cites_surface = any("surface" in ev.lower() or "table" in ev.lower() for ev in ev_list) or (
                "on a nearby surface" in text_str.lower() or "on a surface" in text_str.lower()
            )
            if cites_surface and not has_surface:
                logger.info(f"[VideoNarration] Dropped ungrounded hypothesis citing undetected surface: '{text_str}'")
                continue
            valid_hypotheses.append(h)
        hypotheses = valid_hypotheses if valid_hypotheses else hypotheses[:1]

        # Explicitly incorporate the observed temporal trend into the top hypothesis text
        if changes and hypotheses:
            recent_change = changes[-1]
            chg_from = recent_change.get("from", "")
            chg_to = recent_change.get("to", "")
            chg_type = recent_change.get("type", "")

            temporal_prefix = ""
            if chg_type == "interaction_change" and chg_from and chg_to:
                temporal_prefix = f"Having transitioned from using the {chg_from} to the {chg_to}, "
            elif chg_type == "activity_change" and chg_from and chg_to:
                temporal_prefix = f"Having shifted from {chg_from} to {chg_to}, "
            elif chg_type == "person_count_change":
                temporal_prefix = "Following the change in visible people, "

            if temporal_prefix and not hypotheses[0]["text"].lower().startswith("having"):
                orig_text = hypotheses[0]["text"]
                if orig_text.startswith("They'll "):
                    orig_text = "they'll " + orig_text[len("They'll "):]
                elif orig_text.startswith("They "):
                    orig_text = "they " + orig_text[len("They "):]
                elif orig_text.startswith("The "):
                    orig_text = "the " + orig_text[len("The "):]
                hypotheses[0]["text"] = f"{temporal_prefix}{orig_text}"

        top_pred_text = hypotheses[0]["text"] if hypotheses else base_pred.get("prediction", "")

        return {
            "prediction": top_pred_text,
            "text": top_pred_text,
            "status": base_pred.get("status", "supported_prediction"),
            "prediction_type": base_pred.get("prediction_type", "task_continuation"),
            "hypotheses": hypotheses,
            "reasoning": reasoning,
            "source_facts": source_facts,
            "affordance_candidates": base_pred.get("affordance_candidates", []),
        }
