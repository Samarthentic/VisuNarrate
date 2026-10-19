"""VisuNarrate - Temporal Aggregation & Change Detection Service (Stage 5.0).

Conceptual Architecture:
FRAME-BY-FRAME STRUCTURED SCENE EVIDENCE
  ↓
ADJACENT FRAME COMPARISON & EQUIVALENCE KERNELS
  ↓
STABLE PERSISTENT EVIDENCE EXTRACTION
  ↓
DISCRETE STATE TRANSITION DETECTION (Activity, Interactions, Actor Count, Environment)
  ↓
STRUCTURED TEMPORAL TIMELINE

Key Principles:
1. Deterministic Rule-Based Aggregation: Evaluates semantic equivalence across structured frame representations.
2. Noise Filtering: Suppresses transient detector flicker to isolate genuine physical/activity transitions.
3. Timestamp-Linked Key Moments: Anchors observed changes to video timeline seconds.
4. Explains continuity vs change cleanly for Video Narration synthesis.
"""

import logging
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger("TemporalService")


class TemporalService:
    """Aggregates frame-level evidence sequences, identifies persistence, and detects state transitions."""

    def __init__(self):
        logger.info("[TemporalService] Initialized Temporal Aggregation Engine (Stage 5.0).")

    def normalize_activity_label(self, activity_str: str) -> str:
        """Normalizes activity text to a canonical semantic descriptor."""
        act = activity_str.strip().lower()
        if any(k in act for k in ["stand", "standing"]):
            return "standing"
        elif any(k in act for k in ["sit", "sitting", "seated"]):
            return "sitting"
        elif any(k in act for k in ["walk", "walking"]):
            return "walking"
        elif any(k in act for k in ["run", "running"]):
            return "running"
        elif any(k in act for k in ["read", "reading"]):
            return "reading"
        elif any(k in act for k in ["eat", "eating", "dine"]):
            return "eating"
        elif any(k in act for k in ["drink", "drinking", "sip"]):
            return "drinking"
        elif any(k in act for k in ["laptop", "typing", "computer"]):
            return "using laptop"
        elif any(k in act for k in ["phone", "cell", "texting", "call"]):
            return "using smartphone"
        elif any(k in act for k in ["ride", "riding", "bike", "bicycle"]):
            return "riding bicycle"
        return act if act else "static pose"

    def detect_temporal_transitions(self, frame_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes a sequence of structured frames to extract persistent facts and discrete changes.

        Args:
            frame_sequence: List of frame records produced by VideoEngine.

        Returns:
            Dictionary containing stable_facts, detected changes, and key moments.
        """
        if not frame_sequence:
            return {
                "total_frames_analyzed": 0,
                "stable_facts": {},
                "changes": [],
                "key_moments": [],
            }

        # ---------------------------------------------------------------------
        # 1. Identify Persistent Facts Across the Entire Video Sequence
        # ---------------------------------------------------------------------
        all_subjects = [f.get("primary_subject", "") for f in frame_sequence if f.get("primary_subject")]
        all_environments = [f.get("environment", "") for f in frame_sequence if f.get("environment")]
        all_person_counts = [f.get("person_count", 0) for f in frame_sequence]
        
        # Determine dominant persistent environment
        dominant_env = max(set(all_environments), key=all_environments.count) if all_environments else "General Environment"
        dominant_subject = max(set(all_subjects), key=all_subjects.count) if all_subjects else "Subject"

        # Track persistent object labels across frames (present in >= 40% of frames)
        object_frequency: Dict[str, int] = {}
        for f in frame_sequence:
            for label in f.get("detected_objects", {}).keys():
                object_frequency[label] = object_frequency.get(label, 0) + 1

        min_appearance_thresh = max(1, int(len(frame_sequence) * 0.4))
        persistent_objects = [
            lbl for lbl, count in object_frequency.items()
            if count >= min_appearance_thresh
        ]

        stable_facts = {
            "dominant_subject": dominant_subject,
            "dominant_environment": dominant_env,
            "persistent_objects": persistent_objects,
            "mean_person_count": round(sum(all_person_counts) / len(all_person_counts), 1) if all_person_counts else 0,
        }

        # ---------------------------------------------------------------------
        # 2. Frame-by-Frame Transition and Change Detection
        # ---------------------------------------------------------------------
        changes: List[Dict[str, Any]] = []
        key_moments: List[Dict[str, Any]] = []

        # Record Initial State (Frame 0)
        first_frame = frame_sequence[0]
        initial_activity = self.normalize_activity_label(first_frame.get("activity", "static pose"))
        first_top_interaction = first_frame.get("interactions", [{}])[0] if first_frame.get("interactions") else {}
        initial_target = first_top_interaction.get("target", "none").lower()

        key_moments.append({
            "timestamp_seconds": first_frame.get("timestamp_seconds", 0.0),
            "frame_index": first_frame.get("frame_index", 0),
            "event_type": "scene_start",
            "activity": initial_activity,
            "interaction_target": initial_target,
            "description": f"Scene opens with subject {initial_activity} in {dominant_env.lower()}.",
        })

        prev_activity = initial_activity
        prev_target = initial_target
        prev_person_count = first_frame.get("person_count", 0)
        prev_scene_type = first_frame.get("scene_type", "photographic_scene")

        for idx in range(1, len(frame_sequence)):
            curr_frame = frame_sequence[idx]
            curr_ts = curr_frame.get("timestamp_seconds", 0.0)
            curr_frame_idx = curr_frame.get("frame_index", 0)

            curr_activity = self.normalize_activity_label(curr_frame.get("activity", "static pose"))
            curr_top_interaction = curr_frame.get("interactions", [{}])[0] if curr_frame.get("interactions") else {}
            curr_target = curr_top_interaction.get("target", "none").lower()
            curr_person_count = curr_frame.get("person_count", 0)
            curr_scene_type = curr_frame.get("scene_type", "photographic_scene")

            # 2a. Detect Activity / Posture Transition
            if curr_activity != prev_activity and curr_activity != "static pose" and prev_activity != "static pose":
                change_item = {
                    "timestamp_seconds": curr_ts,
                    "frame_index": curr_frame_idx,
                    "type": "activity_change",
                    "from": prev_activity,
                    "to": curr_activity,
                    "description": f"Subject transitions from {prev_activity} to {curr_activity}.",
                }
                changes.append(change_item)
                key_moments.append({
                    "timestamp_seconds": curr_ts,
                    "frame_index": curr_frame_idx,
                    "event_type": "activity_transition",
                    "activity": curr_activity,
                    "interaction_target": curr_target,
                    "description": change_item["description"],
                })
                prev_activity = curr_activity

            # 2b. Detect Interaction Target Transition
            if curr_target != prev_target and curr_target != "none" and prev_target != "none":
                change_item = {
                    "timestamp_seconds": curr_ts,
                    "frame_index": curr_frame_idx,
                    "type": "interaction_change",
                    "from": prev_target,
                    "to": curr_target,
                    "description": f"Interaction shifts from {prev_target} to {curr_target}.",
                }
                changes.append(change_item)
                key_moments.append({
                    "timestamp_seconds": curr_ts,
                    "frame_index": curr_frame_idx,
                    "event_type": "interaction_shift",
                    "activity": curr_activity,
                    "interaction_target": curr_target,
                    "description": change_item["description"],
                })
                prev_target = curr_target

            # 2c. Detect Person Count Shift (Actor entry / departure)
            if abs(curr_person_count - prev_person_count) >= 1 and (curr_person_count != prev_person_count):
                change_item = {
                    "timestamp_seconds": curr_ts,
                    "frame_index": curr_frame_idx,
                    "type": "person_count_change",
                    "from": prev_person_count,
                    "to": curr_person_count,
                    "description": f"Actor count changes from {prev_person_count} to {curr_person_count}.",
                }
                changes.append(change_item)
                prev_person_count = curr_person_count

            # 2d. Detect Scene Type Shift
            if curr_scene_type != prev_scene_type:
                change_item = {
                    "timestamp_seconds": curr_ts,
                    "frame_index": curr_frame_idx,
                    "type": "scene_type_change",
                    "from": prev_scene_type,
                    "to": curr_scene_type,
                    "description": f"Visual scene type changes from {prev_scene_type} to {curr_scene_type}.",
                }
                changes.append(change_item)
                prev_scene_type = curr_scene_type

        full_count = sum(1 for f in frame_sequence if f.get("processing_mode") == "FULL_ANALYSIS")
        red_count = sum(1 for f in frame_sequence if f.get("processing_mode") == "REDUNDANT")
        key_count = sum(1 for f in frame_sequence if f.get("processing_mode") == "KEY_CHANGE")

        logger.info(
            f"[TemporalService] Temporal analysis complete: {len(changes)} discrete change(s) "
            f"(Full: {full_count}, Redundant: {red_count}, Key Changes: {key_count}) across {len(frame_sequence)} frames."
        )

        return {
            "total_frames_analyzed": len(frame_sequence),
            "full_analyses_count": full_count,
            "redundant_frames_count": red_count,
            "key_changes_count": key_count,
            "stable_facts": stable_facts,
            "changes_count": len(changes),
            "changes": changes,
            "key_moments_count": len(key_moments),
            "key_moments": key_moments,
        }
