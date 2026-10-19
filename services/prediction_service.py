"""VisuNarrate - Conversational Natural Predictive Reasoning Service.

Conceptual Architecture:
CURRENT VISUAL STATE -> EVIDENCE TAXONOMY -> ACTION AFFORDANCES -> CONVERSATIONAL PREDICTION -> EXPLAINABLE TRACE

Taxonomy Scenarios & Prediction Types:
A. Strong Object Interaction   -> "interaction_continuation" / "task_continuation" (supported_prediction)
B. Social / Group Scene        -> "group_continuation"                             (tentative_prediction)
C. Static Environmental Scene  -> "environmental_static" / "insufficient_evidence" (insufficient_evidence)
D. Abstract / Graphic Media    -> "suppressed_graphic"                             (suppressed)
E. Ambiguous Visual Activity   -> "ambiguous_activity"                             (tentative_prediction)

Key Principles:
1. Natural Conversational Phrasing: Free of robotic template prefixes and stiff wording.
2. Evidence-Grounded: Derives strictly from validated, fused multimodal facts.
3. Deterministic & Explainable: 100% local rule-based cognition.
4. Media Safety: Suppresses temporal projection on non-photographic media.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("PredictionService")


class PredictionService:
    """Evaluates validated scene states, models action affordances, and generates predictive narrations."""

    AFFORDANCE_RULES = {
        "cell phone": {
            "prediction": "They'll likely continue using the smartphone.",
            "prediction_type": "interaction_continuation",
            "status": "supported_prediction",
            "rationale": "Handheld mobile device affordances center on screen interaction, browsing, or pocketing.",
            "candidates": [
                "The person continues viewing or typing on the smartphone screen.",
                "The person may adjust the device or put it away."
            ]
        },
        "phone": {
            "prediction": "They'll likely continue using the smartphone.",
            "prediction_type": "interaction_continuation",
            "status": "supported_prediction",
            "rationale": "Handheld mobile device affordances center on screen interaction, browsing, or pocketing.",
            "candidates": [
                "The person continues viewing or typing on the smartphone screen.",
                "The person may adjust the device or put it away."
            ]
        },
        "laptop": {
            "prediction": "They'll likely keep working on the laptop.",
            "prediction_type": "task_continuation",
            "status": "supported_prediction",
            "rationale": "Workstation affordances involve sustained keyboard, trackpad, and display engagement.",
            "candidates": [
                "The person continues typing or viewing the laptop screen.",
                "The person may switch tasks or adjust workstation posture."
            ]
        },
        "book": {
            "prediction": "They'll likely continue reading.",
            "prediction_type": "task_continuation",
            "status": "supported_prediction",
            "rationale": "Reading material affordances involve sequential text scanning and page navigation.",
            "candidates": [
                "The person continues reading or turns to the next page.",
                "The person may close the book or mark their place."
            ]
        },
        "cup": {
            "prediction": "They'll likely take another sip or set the cup down.",
            "prediction_type": "interaction_continuation",
            "status": "supported_prediction",
            "rationale": "Beverage vessel affordances center on sipping, drinking, or resting on a nearby surface.",
            "candidates": [
                "The person takes another sip from the cup.",
                "The person places the cup down on the table."
            ]
        },
        "bottle": {
            "prediction": "They'll likely take a drink or set the bottle down.",
            "prediction_type": "interaction_continuation",
            "status": "supported_prediction",
            "rationale": "Beverage container affordances center on consumption or securing the closure.",
            "candidates": [
                "The person drinks from the bottle or secures the cap.",
                "The person places the bottle on a nearby surface."
            ]
        },
        "food or beverage": {
            "prediction": "They'll likely continue enjoying their meal.",
            "prediction_type": "interaction_continuation",
            "status": "supported_prediction",
            "rationale": "Food and drink affordances center on meal consumption and surface resting.",
            "candidates": [
                "The person continues eating or drinking from the meal composition.",
                "The person rests utensils or beverage containers on the table."
            ]
        },
        "bicycle": {
            "prediction": "They'll likely continue riding along the path.",
            "prediction_type": "task_continuation",
            "status": "supported_prediction",
            "rationale": "Locomotion and vehicle affordances involve continued directional movement.",
            "candidates": [
                "The person maintains balance and continues pedaling.",
                "The person may slow down or navigate surrounding terrain."
            ]
        },
    }

    POSTURE_AFFORDANCES = {
        "standing": {
            "prediction": "They'll likely continue standing or begin walking.",
            "prediction_type": "ambiguous_activity",
            "status": "tentative_prediction",
            "rationale": "Static standing posture supports continued observation, subtle weight shifts, or subsequent locomotion.",
            "candidates": [
                "The person may continue standing or begin walking.",
                "The person maintains stationary observation of the surroundings."
            ]
        },
        "sitting": {
            "prediction": "They'll likely remain seated and continue what they are doing.",
            "prediction_type": "task_continuation",
            "status": "tentative_prediction",
            "rationale": "Seated posture indicates stable engagement with the immediate environment.",
            "candidates": [
                "The person will likely remain seated comfortably.",
                "The person adjusts posture or interacts with nearby objects."
            ]
        },
    }

    def __init__(self):
        logger.info("[PredictionService] Initialized Conversational Predictive Reasoning Engine.")

    def extract_current_state(self, fused_scene: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts normalized current visual state from the fused scene representation."""
        person_presence = fused_scene.get("person_presence", {})
        scene_type = fused_scene.get("scene_type", {})
        primary_subject = fused_scene.get("primary_subject", {})
        activity = fused_scene.get("activity", {})
        environment = fused_scene.get("environment", {})
        interactions = fused_scene.get("interactions", [])
        objects = fused_scene.get("objects", [])
        conflicts = fused_scene.get("conflicts", [])
        uncertainty_flags = fused_scene.get("uncertainty_flags", [])

        top_interaction = interactions[0] if interactions else {}

        return {
            "scene_type": scene_type.get("scene_type", "photographic_scene"),
            "is_graphic": scene_type.get("is_graphic", False),
            "person_status": person_presence.get("status", "uncertain"),
            "person_count": person_presence.get("count", 0),
            "person_confidence": person_presence.get("confidence_level", "supported"),
            "subject_value": primary_subject.get("value", "Unspecified"),
            "subject_status": primary_subject.get("status", "supported"),
            "activity_value": activity.get("value", "Static pose"),
            "activity_status": activity.get("status", "supported"),
            "environment_value": environment.get("value", "General Environment"),
            "environment_status": environment.get("status", "supported"),
            "interactions": interactions,
            "top_interaction": top_interaction,
            "interaction_target": top_interaction.get("target", "").lower(),
            "interaction_relation": top_interaction.get("relation", ""),
            "interaction_status": top_interaction.get("status", ""),
            "interaction_evidence": top_interaction.get("evidence_sources", []),
            "interaction_counter": top_interaction.get("counter_evidence", []),
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
            "uncertainty_flags": uncertainty_flags,
            "raw_fused_scene": fused_scene,
        }

    def generate_predictive_narration(self, fused_scene: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizes a fluent, conversational predictive narration and reasoning trace.

        Args:
            fused_scene: Unified scene representation produced by EvidenceFusionEngine.

        Returns:
            Dict containing prediction, status, prediction_type, reasoning trace, and source facts.
        """
        state = self.extract_current_state(fused_scene)

        # ---------------------------------------------------------------------
        # Scenario 1: Media Safety Gating (Illustration / Graphic Media)
        # ---------------------------------------------------------------------
        if state["is_graphic"] or state["scene_type"] == "illustration_or_graphic":
            logger.info("[PredictionService] Scene identified as Graphic/Illustration. Suppressing temporal prediction.")
            return {
                "prediction": "Predictive storytelling is paused for static illustrations.",
                "status": "suppressed",
                "prediction_type": "suppressed_graphic",
                "reasoning": [
                    "Media classification identified the visual input as an illustration/graphic.",
                    "Non-photographic abstract media lacks physical temporal continuity.",
                    "Predictive reasoning is strictly reserved for physical real-world environments."
                ],
                "source_facts": [
                    f"Media Classification: {state['scene_type']}",
                    f"Main Subject: {state['subject_value']}"
                ],
                "affordance_candidates": []
            }

        # ---------------------------------------------------------------------
        # Scenario 2: Static Environmental / No-Person Scenes
        # ---------------------------------------------------------------------
        if state["person_status"] == "absent":
            logger.info("[PredictionService] Person absent. Evaluating environmental static state.")
            return {
                "prediction": "The scene will likely remain steady.",
                "status": "insufficient_evidence",
                "prediction_type": "environmental_static",
                "reasoning": [
                    "No active human agents or moving entities detected in the scene.",
                    "Natural landscape environments remain in physical equilibrium over short horizons."
                ],
                "source_facts": [
                    "Person Presence: Absent",
                    f"Main Subject: {state['subject_value']}",
                    f"Environment: {state['environment_value']}"
                ],
                "affordance_candidates": []
            }

        # ---------------------------------------------------------------------
        # Scenario 3: Uncertain Agent Presence Gating
        # ---------------------------------------------------------------------
        if state["person_status"] == "uncertain":
            logger.info("[PredictionService] Person presence uncertain. Yielding insufficient evidence.")
            return {
                "prediction": "Not enough visual certainty to anticipate immediate next actions.",
                "status": "insufficient_evidence",
                "prediction_type": "insufficient_evidence",
                "reasoning": [
                    "Ambiguous signals regarding human presence.",
                    "Predictive affordance modeling requires confident agent grounding."
                ],
                "source_facts": [
                    "Person Presence: Uncertain"
                ],
                "affordance_candidates": []
            }

        # ---------------------------------------------------------------------
        # Scenario 4: Human Agent(s) Present
        # ---------------------------------------------------------------------
        reasoning: List[str] = []
        source_facts: List[str] = []
        affordance_candidates: List[str] = []

        source_facts.append(f"Subject: {state['subject_value']} ({state['subject_status']})")
        source_facts.append(f"Activity: {state['activity_value']} ({state['activity_status']})")
        source_facts.append(f"Environment: {state['environment_value']} ({state['environment_status']})")
        source_facts.append(f"Person Count: {state['person_count']}")

        reasoning.append("Confirmed human presence through spatial object detector and visual grounding.")

        # Check for Strong Focal Object Interactions first (highest specificity)
        interactions = state["interactions"]
        valid_interaction = None

        if interactions:
            for inter in interactions:
                if inter.get("status") in {"confirmed", "supported"}:
                    target_candidate = inter.get("target", "").lower()
                    matched_rule = next((k for k in self.AFFORDANCE_RULES if k in target_candidate), None)
                    if matched_rule:
                        valid_interaction = (inter, matched_rule)
                        break

        # 4a. Focal Object Interaction Found (e.g. phone, laptop, book, food/drink, bicycle)
        if valid_interaction:
            inter_dict, rule_key = valid_interaction
            rule_info = self.AFFORDANCE_RULES[rule_key]
            target_str = inter_dict.get("target", rule_key)

            source_facts.append(f"Interaction: {inter_dict.get('relation')} {target_str} ({inter_dict.get('status')})")
            reasoning.append(f"Observed interaction with '{target_str}' confirmed across multimodal evidence sources.")
            reasoning.append(rule_info["rationale"])
            affordance_candidates.extend(rule_info["candidates"])

            return {
                "prediction": rule_info["prediction"],
                "status": rule_info["status"],
                "prediction_type": rule_info["prediction_type"],
                "reasoning": reasoning,
                "source_facts": source_facts,
                "affordance_candidates": affordance_candidates,
            }

        # 4b. Group / Social Dynamics (Multi-person scene)
        if state["person_count"] >= 2:
            reasoning.append(f"Multi-person scene ({state['person_count']} people) identified without a single isolated focal interaction.")
            return {
                "prediction": "They'll likely continue their conversation or shared activity.",
                "status": "tentative_prediction",
                "prediction_type": "group_continuation",
                "reasoning": reasoning,
                "source_facts": source_facts,
                "affordance_candidates": [
                    "The group maintains conversational engagement.",
                    "Participants shift attention or conclude their interaction."
                ]
            }

        # 4c. Posture Affordances (Standing / Seated without detected focal object)
        act_val = state["activity_value"].lower()
        if "stand" in act_val:
            posture_rule = self.POSTURE_AFFORDANCES["standing"]
            return {
                "prediction": posture_rule["prediction"],
                "status": posture_rule["status"],
                "prediction_type": posture_rule["prediction_type"],
                "reasoning": reasoning + [posture_rule["rationale"]],
                "source_facts": source_facts,
                "affordance_candidates": posture_rule["candidates"],
            }
        elif "sit" in act_val or "seated" in act_val:
            posture_rule = self.POSTURE_AFFORDANCES["sitting"]
            return {
                "prediction": posture_rule["prediction"],
                "status": posture_rule["status"],
                "prediction_type": posture_rule["prediction_type"],
                "reasoning": reasoning + [posture_rule["rationale"]],
                "source_facts": source_facts,
                "affordance_candidates": posture_rule["candidates"],
            }

        # 4d. Default Conservative Continuation
        return {
            "prediction": "They'll likely continue their current activity.",
            "status": "tentative_prediction",
            "prediction_type": "ambiguous_activity",
            "reasoning": reasoning + ["Sustained behavioral momentum in the immediate visual context."],
            "source_facts": source_facts,
            "affordance_candidates": [
                "The person sustains current action in the scene.",
                "The person concludes the current activity."
            ]
        }
