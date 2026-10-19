"""VisuNarrate - Video Narration Synthesis Service (Stage 5.0).

Conceptual Architecture:
TEMPORAL STATE SEQUENCE + DETECTED TRANSITIONS + STABLE CONTEXT
  ↓
CHRONOLOGICAL NARRATIVE COMPILATION
  ↓
COHERENT MULTI-FRAME NARRATIVE SYNTHESIS (Zero Repetition / Fluent English)

Key Principles:
1. Anti-Repetition: Synthesizes transitions across time instead of re-stating redundant frame descriptions.
2. Evidence-Grounded: Operates strictly on confirmed temporal transitions and persistent multimodal evidence.
3. Natural Chronology: Structures opening state -> key moments & shifts -> continuous context.
4. 100% Deterministic & Local NLG without external LLM dependencies.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("VideoNarrationService")


class VideoNarrationService:
    """Synthesizes unified, chronological video narratives from temporal event sequences."""

    def __init__(self):
        logger.info("[VideoNarrationService] Initialized Video Narration Synthesis Engine (Stage 5.0).")

    def synthesize_video_narrative(
        self,
        temporal_analysis: Dict[str, Any],
        frame_sequence: List[Dict[str, Any]],
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synthesizes a fluent, academic multi-frame video narrative.

        Args:
            temporal_analysis: Structured temporal output from TemporalService.
            frame_sequence: Raw sequence of frame records from VideoEngine.
            video_metadata: Basic video properties (duration, FPS, resolution).

        Returns:
            Coherent, non-repetitive narrative paragraph describing the video event progression.
        """
        if not frame_sequence:
            return "No video frames available for narrative synthesis."

        stable_facts = temporal_analysis.get("stable_facts", {})
        changes = temporal_analysis.get("changes", [])
        key_moments = temporal_analysis.get("key_moments", [])
        
        dominant_subject = stable_facts.get("dominant_subject", "A subject")
        dominant_env = stable_facts.get("dominant_environment", "general setting")
        persistent_objects = stable_facts.get("persistent_objects", [])

        sentences: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Opening State Sentence
        # ---------------------------------------------------------------------
        first_frame = frame_sequence[0]
        first_act = first_frame.get("activity", "static pose").lower()
        first_interactions = first_frame.get("interactions", [])
        first_interaction_str = ""

        if first_interactions:
            top_inter = first_interactions[0]
            rel = top_inter.get("relation", "interacting with")
            target = top_inter.get("target", "")
            if target:
                if target.lower().startswith(("a ", "an ", "the ", "food")):
                    first_interaction_str = f" while {rel} {target}"
                elif target.lower().startswith(("a", "e", "i", "o", "u")):
                    first_interaction_str = f" while {rel} an {target}"
                else:
                    first_interaction_str = f" while {rel} a {target}"

        # Clean subject description
        subject_clean = dominant_subject.strip()
        if subject_clean.endswith("."):
            subject_clean = subject_clean[:-1]

        env_clause = ""
        if dominant_env and dominant_env.lower() not in {"general environment", "unknown", "none"}:
            if "indoors" in dominant_env.lower():
                env_clause = " in an indoor environment"
            elif "outdoors" in dominant_env.lower():
                env_clause = " in an outdoor setting"
            else:
                env_clause = f" within an environment characterized by {dominant_env.lower()}"

        if first_act and first_act not in {"static pose", "unknown"}:
            sentences.append(f"The video initially captures {subject_clean.lower()}{env_clause}{first_interaction_str}.")
        else:
            sentences.append(f"The video opens showing {subject_clean.lower()}{env_clause}{first_interaction_str}.")

        # ---------------------------------------------------------------------
        # 2. Chronological Transitions & Key Moments
        # ---------------------------------------------------------------------
        if changes:
            for chg in changes:
                ts = chg.get("timestamp_seconds", 0.0)
                chg_type = chg.get("type", "")
                from_state = chg.get("from", "")
                to_state = chg.get("to", "")

                if chg_type == "activity_change":
                    sentences.append(f"As the scene progresses at {ts}s, the activity transitions from {from_state} to {to_state}.")
                elif chg_type == "interaction_change":
                    sentences.append(f"At {ts}s, the focus shifts as the subject begins interacting with {to_state}.")
                elif chg_type == "person_count_change":
                    sentences.append(f"Around {ts}s, the presence of visible people in the frame changes to {to_state}.")
        else:
            # If no discrete changes, state continuity
            last_frame = frame_sequence[-1]
            last_act = last_frame.get("activity", "the activity").lower()
            sentences.append(f"Throughout the recorded sequence, the scene maintains continuity with the subject engaged in {last_act}.")

        # ---------------------------------------------------------------------
        # 3. Persistent Objects & Environment Grounding
        # ---------------------------------------------------------------------
        filtered_objects = [
            obj for obj in persistent_objects
            if obj not in {"person", "frisbee"}
        ]
        if filtered_objects:
            sentences.append(f"Key persistent elements identified across the timeline include {', '.join(filtered_objects[:4])}.")

        # Final Narrative Assembly
        final_video_narration = " ".join(sentences).strip()
        logger.info(f"[VideoNarrationService] Synthesized Video Narrative: '{final_video_narration}'")
        return final_video_narration
