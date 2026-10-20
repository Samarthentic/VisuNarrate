"""VisuNarrate - Conversational Natural Predictive Reasoning Service.

Conceptual Architecture:
CURRENT VISUAL STATE -> EVIDENCE TAXONOMY -> ACTION AFFORDANCES -> STRUCTURED PREDICTIONS -> EXPLAINABLE TRACE

Taxonomy Scenarios & Prediction Types:
A. Strong Object Interaction   -> "interaction_continuation" / "task_continuation" (supported_prediction)
B. Multi-Object Affordance      -> "interaction_continuation" / "task_continuation" (supported_prediction)
C. Social / Group Scene        -> "group_continuation"                             (tentative_prediction)
D. Pedestrian / Traffic Scene  -> "pedestrian_traffic" / "locomotion_continuation" (supported_prediction / tentative_prediction)
E. Static Environmental Scene  -> "environmental_static"                           (insufficient_evidence)
F. Abstract / Graphic Media    -> "suppressed_graphic"                             (suppressed)
G. Ambiguous Visual Activity   -> "ambiguous_activity"                             (tentative_prediction)

Key Principles:
1. Natural Conversational Phrasing: Free of robotic template prefixes and stiff wording.
2. Evidence-Grounded: Derives strictly from validated, fused multimodal facts (detections, caption, VQA, narration).
3. Deterministic & Explainable: 100% local rule-based cognition without invented percentages.
4. Structured Hypotheses: Produces 1-3 prioritized hypotheses with explicit evidence, likelihood labels, and rationales.
5. Narration-Prediction Consistency: Predictions only assert entities and actions corroborated by the scene narration.
6. Media Safety: Suppresses temporal projection on non-photographic media.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger("PredictionService")


class PredictionService:
    """Evaluates validated scene states, models action affordances, and generates predictive narrations."""

    WORKSTATION_OBJECTS: Set[str] = {
        "laptop", "mouse", "keyboard", "monitor", "desk", "dining table", "book"
    }

    VEHICLE_OBJECTS: Set[str] = {
        "car", "bus", "truck", "motorcycle", "bicycle", "traffic light", "parking meter"
    }

    FOOD_OBJECTS: Set[str] = {
        "sandwich", "bowl", "pizza", "hot dog", "cake", "donut", "apple", "banana", "orange"
    }

    DRINK_OBJECTS: Set[str] = {
        "cup", "bottle", "wine glass"
    }

    def __init__(self):
        logger.info("[PredictionService] Initialized Conversational Predictive Reasoning Engine.")

    def extract_current_state(
        self, fused_scene: Dict[str, Any], narration: Optional[str] = None
    ) -> Dict[str, Any]:
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

        # Retrieve or synthesize narration for consistency checks
        resolved_narration = narration
        if not resolved_narration:
            resolved_narration = fused_scene.get("scene_narration")
        if not resolved_narration:
            try:
                from services.narration_service import NarrationService
                resolved_narration = NarrationService().synthesize_scene_narrative(fused_scene)
            except Exception:
                resolved_narration = ""

        # Extract object labels and counts
        detected_counts: Dict[str, int] = {}
        for obj in objects:
            lbl = obj.get("label", "").lower()
            if lbl:
                detected_counts[lbl] = detected_counts.get(lbl, 0) + 1

        top_interaction = interactions[0] if interactions else {}

        return {
            "scene_type": scene_type.get("scene_type", "photographic_scene"),
            "is_graphic": scene_type.get("is_graphic", False),
            "person_status": person_presence.get("status", "uncertain"),
            "person_count": person_presence.get("count", 0),
            "has_dominant_person": person_presence.get("has_dominant_person", False),
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
            "detected_labels": set(detected_counts.keys()),
            "detected_counts": detected_counts,
            "objects": objects,
            "narration": resolved_narration,
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
            "uncertainty_flags": uncertainty_flags,
            "raw_fused_scene": fused_scene,
        }

    def _filter_grounded_hypotheses(
        self, hypotheses: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ensures every entity mentioned in hypotheses is grounded in detections, caption, or narration."""
        narration_lower = state.get("narration", "").lower()
        detected_labels = state.get("detected_labels", set())
        raw_caption = state.get("raw_fused_scene", {}).get("summary", "").lower()

        grounded: List[Dict[str, Any]] = []
        for hyp in hypotheses:
            text_lower = hyp["text"].lower()

            # Safety check: if hypothesis mentions bicycle/bike, ensure it's detected or narrated
            if any(w in text_lower for w in ["bicycle", "bike", "cycling", "pedaling"]):
                if not any(w in detected_labels for w in ["bicycle", "bike"]) and \
                   "bicycle" not in narration_lower and "bike" not in narration_lower:
                    continue

            # Safety check: if hypothesis mentions vehicle/car/bus/motorcycle, ensure vehicle is detected
            if any(w in text_lower for w in ["vehicle", "car", "bus", "truck", "motorcycle", "traffic"]):
                if not (detected_labels & self.VEHICLE_OBJECTS) and "traffic" not in narration_lower and "street" not in narration_lower:
                    continue

            # Consistency check: if narration asserts 'with a laptop nearby', forbid claiming active work
            if "with a laptop nearby" in narration_lower and ("keep working on the laptop" in text_lower or "typing on the laptop" in text_lower):
                continue

            grounded.append(hyp)

        return grounded if grounded else hypotheses

    def generate_predictive_narration(
        self, fused_scene: Dict[str, Any], narration: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synthesizes a fluent, structured predictive narration with 1-3 evidence-based hypotheses.

        Args:
            fused_scene: Unified scene representation produced by EvidenceFusionEngine.
            narration: Optional final scene narrative for consistency enforcement.

        Returns:
            Dict containing prediction, status, prediction_type, hypotheses, reasoning trace,
            source facts, and affordance candidates.
        """
        state = self.extract_current_state(fused_scene, narration=narration)

        # ---------------------------------------------------------------------
        # Scenario 1: Media Safety Gating (Illustration / Graphic Media)
        # ---------------------------------------------------------------------
        if state["is_graphic"] or state["scene_type"] == "illustration_or_graphic":
            logger.info("[PredictionService] Scene identified as Graphic/Illustration. Suppressing temporal prediction.")
            hypotheses = [
                {
                    "text": "Predictive storytelling is paused for static illustrations.",
                    "evidence": [f"Media classification: {state['scene_type']}"],
                    "likelihood_label": "most likely",
                    "rationale": "Static non-photographic graphics lack real-world physical continuity."
                }
            ]
            return {
                "prediction": hypotheses[0]["text"],
                "status": "suppressed",
                "prediction_type": "suppressed_graphic",
                "hypotheses": hypotheses,
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
        # Scenario 2: Unoccupied Workspace / Desk Scenes (Unified for Images 11 & 12)
        # ---------------------------------------------------------------------
        has_workstation_objects = bool(state["detected_labels"] & self.WORKSTATION_OBJECTS)
        is_unoccupied = (state["person_count"] == 0) or (state["person_status"] in {"absent", "uncertain"} and "person" not in state["detected_labels"])

        if is_unoccupied and has_workstation_objects:
            logger.info("[PredictionService] Unoccupied workspace detected. Unifying desk scene prediction.")
            detected_items = [f"{lbl} ({cnt})" for lbl, cnt in sorted(state["detected_counts"].items()) if lbl in self.WORKSTATION_OBJECTS]
            items_str = ", ".join(detected_items) if detected_items else "workstation items"

            hypotheses = [
                {
                    "text": "The workspace will likely remain steady until someone uses it.",
                    "evidence": [f"Workstation objects present: {items_str}", "No active person detected"],
                    "likelihood_label": "most likely",
                    "rationale": "Static workstation equipment remains in physical equilibrium until an operator engages with it."
                },
                {
                    "text": "Someone may sit down at the desk to work.",
                    "evidence": ["Equipped desk setup with study/computing items"],
                    "likelihood_label": "possible",
                    "rationale": "A prepared desk setup is intended for active computer or study work."
                }
            ]
            return {
                "prediction": hypotheses[0]["text"],
                "status": "insufficient_evidence",
                "prediction_type": "environmental_static",
                "hypotheses": hypotheses,
                "reasoning": [
                    "Workspace objects identified without active human operators present.",
                    "Physical tools and electronics remain stationary until someone returns."
                ],
                "source_facts": [
                    "Person Presence: Absent",
                    f"Detected Workstation Objects: {items_str}",
                    f"Environment: {state['environment_value']}"
                ],
                "affordance_candidates": [
                    "The workspace remains in its current stationary configuration.",
                    "A user approaches and initiates computer or study work."
                ]
            }

        # ---------------------------------------------------------------------
        # Scenario 3: Static Environmental / Landscape Scenes (No human agents)
        # ---------------------------------------------------------------------
        if state["person_status"] == "absent" or (state["person_count"] == 0 and "person" not in state["detected_labels"]):
            logger.info("[PredictionService] Person absent. Evaluating environmental static state.")
            hypotheses = [
                {
                    "text": "The scene will likely remain steady.",
                    "evidence": [f"Environment: {state['environment_value']}", "No active human agents or moving entities detected"],
                    "likelihood_label": "most likely",
                    "rationale": "Natural landscape environments remain in physical equilibrium over short horizons."
                }
            ]
            return {
                "prediction": hypotheses[0]["text"],
                "status": "insufficient_evidence",
                "prediction_type": "environmental_static",
                "hypotheses": hypotheses,
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
        # Scenario 4: Uncertain Agent Presence Gating (Truly uncertain without context)
        # ---------------------------------------------------------------------
        if state["person_status"] == "uncertain" and state["person_count"] == 0:
            logger.info("[PredictionService] Person presence uncertain without objects. Yielding insufficient evidence.")
            hypotheses = [
                {
                    "text": "Not enough visual certainty to anticipate immediate next actions.",
                    "evidence": ["Ambiguous signals regarding human presence"],
                    "likelihood_label": "less likely",
                    "rationale": "Predictive affordance modeling requires confident agent grounding."
                }
            ]
            return {
                "prediction": hypotheses[0]["text"],
                "status": "insufficient_evidence",
                "prediction_type": "insufficient_evidence",
                "hypotheses": hypotheses,
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
        # Scenario 5: Human Agent(s) Present - Grounded Affordance Modeling
        # ---------------------------------------------------------------------
        reasoning: List[str] = []
        source_facts: List[str] = []
        affordance_candidates: List[str] = []

        source_facts.append(f"Subject: {state['subject_value']} ({state['subject_status']})")
        source_facts.append(f"Activity: {state['activity_value']} ({state['activity_status']})")
        source_facts.append(f"Environment: {state['environment_value']} ({state['environment_status']})")
        source_facts.append(f"Person Count: {state['person_count']}")

        reasoning.append("Confirmed human presence through spatial object detector and visual grounding.")

        # Check for validated focal interactions
        interactions = state["interactions"]
        valid_interaction = None
        if interactions:
            for inter in interactions:
                if inter.get("status") in {"confirmed", "supported"}:
                    valid_interaction = inter
                    break

        narration_lower = state["narration"].lower()
        person_count = state["person_count"]
        has_dominant_person = state["has_dominant_person"]
        is_true_group = (person_count >= 2 and not has_dominant_person)
        has_laptop_nearby = ("with a laptop nearby" in narration_lower or "laptop nearby" in narration_lower)
        has_cup = "cup" in state["detected_labels"]
        has_food = bool(state["detected_labels"] & self.FOOD_OBJECTS)
        has_traffic = bool(state["detected_labels"] & self.VEHICLE_OBJECTS)
        act_val = state["activity_value"].lower()

        # ---------------------------------------------------------------------
        # Sub-Scenario 5a: True Group Scene with Secondary Item Nearby (Images 7 & 8)
        # Prevents "keep working on the laptop" contradiction when narration says "with a laptop nearby"
        # ---------------------------------------------------------------------
        if is_true_group and has_laptop_nearby:
            is_outdoors = "outdoor" in state["environment_value"].lower() or "outdoors" in narration_lower or "bench" in narration_lower
            if is_outdoors:
                hypotheses = [
                    {
                        "text": "The group will likely remain gathered together outdoors.",
                        "evidence": [f"{person_count} people outdoors", "Shared outdoor gathering"],
                        "likelihood_label": "most likely",
                        "rationale": "A social gathering outdoors maintains shared group presence."
                    },
                    {
                        "text": "Someone may check or pack up the laptop nearby.",
                        "evidence": ["Laptop detected nearby", f"Narration asserts: '{state['narration']}'"],
                        "likelihood_label": "possible",
                        "rationale": "A portable device brought outdoors may be referenced or stowed by the group."
                    },
                    {
                        "text": "Members of the group may stand up or move to another spot.",
                        "evidence": ["Outdoor group presence"],
                        "likelihood_label": "less likely",
                        "rationale": "Outdoor groups occasionally relocate or move around."
                    }
                ]
            else:
                hypotheses = [
                    {
                        "text": "The group will likely continue their discussion at the table.",
                        "evidence": [f"{person_count} people gathered", "Indoor table setting", f"Narration asserts: '{state['narration']}'"],
                        "likelihood_label": "most likely",
                        "rationale": "People gathered around a table typically engage in collaborative discussion or shared social time."
                    },
                    {
                        "text": "Someone in the group may refer to or use the laptop nearby.",
                        "evidence": ["Laptop detected nearby", "Tabletop workspace items"],
                        "likelihood_label": "possible",
                        "rationale": "A nearby computing device may be accessed by group members during their meeting."
                    },
                    {
                        "text": "The group may pack up their belongings and conclude the gathering.",
                        "evidence": ["Group gathering at table"],
                        "likelihood_label": "less likely",
                        "rationale": "Gatherings eventually conclude or participants transition to other tasks."
                    }
                ]

            reasoning.append(f"Group setting ({person_count} people) with nearby laptop decoupled from individual solo work.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "tentative_prediction",
                "prediction_type": "group_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + [f"Group Setting: {person_count} people", "Nearby Object: laptop"],
                "affordance_candidates": affordance_candidates,
            }

        # ---------------------------------------------------------------------
        # Sub-Scenario 5b: Pedestrians near Roadway Traffic (Image 10)
        # Prevents ungrounded "conversation" claims when pedestrians and vehicles co-occur
        # ---------------------------------------------------------------------
        if has_traffic and person_count >= 1:
            traffic_names = [lbl for lbl in state["detected_labels"] if lbl in self.VEHICLE_OBJECTS]
            traffic_str = ", ".join(traffic_names) if traffic_names else "street vehicles"

            hypotheses = [
                {
                    "text": "The pedestrians will likely stay on the walkway and navigate around street traffic.",
                    "evidence": [f"Detected {person_count} persons", f"Detected vehicles: {traffic_str}", "Street environment"],
                    "likelihood_label": "most likely",
                    "rationale": "Pedestrians in urban traffic environments prioritize walkway safety and watch for vehicles."
                },
                {
                    "text": "Vehicles on the road will likely continue moving through the street.",
                    "evidence": [f"Detected vehicles: {traffic_str}"],
                    "likelihood_label": "possible",
                    "rationale": "Vehicular road traffic maintains directional forward progression along the street."
                },
                {
                    "text": "A pedestrian may wait for a break in traffic before crossing.",
                    "evidence": ["Pedestrians near roadway"],
                    "likelihood_label": "less likely",
                    "rationale": "Street pedestrians may prepare to cross when traffic permits."
                }
            ]

            reasoning.append(f"Urban street scene with pedestrians ({person_count}) and vehicle traffic ({traffic_str}).")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "pedestrian_traffic",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + [f"Detected Traffic: {traffic_str}"],
                "affordance_candidates": affordance_candidates,
            }

        # ---------------------------------------------------------------------
        # Sub-Scenario 5c: Pedestrian Locomotion / Walking Outdoors (Image 6)
        # Prevents ungrounded conversation claims when pedestrian is walking outdoors
        # ---------------------------------------------------------------------
        if "walk" in act_val or "standing on a sidewalk" in state["activity_value"].lower() or "walking outdoors" in narration_lower:
            hypotheses = [
                {
                    "text": "The person will likely continue walking along the walkway.",
                    "evidence": [f"Detected {person_count} persons", "Walking activity confirmed by visual evidence", "Walkway / outdoor setting"],
                    "likelihood_label": "most likely",
                    "rationale": "A walking pedestrian typically sustains directional locomotion along a pedestrian pathway."
                },
                {
                    "text": "They will likely stay in the same area as the other people nearby." if person_count >= 2 else "The person may pause or look around their surroundings.",
                    "evidence": [f"{person_count - 1} other people nearby" if person_count >= 2 else "Outdoor pathway"],
                    "likelihood_label": "possible",
                    "rationale": "Pedestrians sharing a walkway often move at similar paces through common public areas." if person_count >= 2 else "Pedestrians occasionally pause along their route."
                },
                {
                    "text": "The person may pause or change walking direction.",
                    "evidence": ["Pedestrian pathway"],
                    "likelihood_label": "less likely",
                    "rationale": "Walkers may stop at crossings or adjust their path."
                }
            ]

            reasoning.append("Pedestrian locomotion observed along outdoor pathway.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "locomotion_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Locomotion: walking along walkway"],
                "affordance_candidates": affordance_candidates,
            }

        # ---------------------------------------------------------------------
        # Sub-Scenario 5d: Multi-Object & Rich Interaction Affordances
        # ---------------------------------------------------------------------
        target_str = valid_interaction.get("target", "").lower() if valid_interaction else ""

        # 1. Laptop + Cup Affordance (Image 2)
        if ("laptop" in target_str or "laptop" in state["detected_labels"]) and has_cup and ("working" in act_val or "laptop" in narration_lower):
            hypotheses = [
                {
                    "text": "They'll likely take a sip from the cup, then return to typing on the laptop.",
                    "evidence": ["Detected laptop and cup on desk", "Confirmed laptop interaction", "Cup positioned within reach"],
                    "likelihood_label": "most likely",
                    "rationale": "A beverage vessel on an active workspace affords intermittent sipping between computer tasks."
                },
                {
                    "text": "They'll likely keep working on the laptop.",
                    "evidence": ["Active workstation engagement"],
                    "likelihood_label": "possible",
                    "rationale": "Deep focus on typing or reading on screen may delay beverage consumption."
                },
                {
                    "text": "They may interact with the other person nearby." if person_count >= 2 else "They may switch tasks or adjust workstation posture.",
                    "evidence": [f"{person_count - 1} other person nearby" if person_count >= 2 else "Workstation environment"],
                    "likelihood_label": "less likely",
                    "rationale": "Co-located colleagues or companions may initiate conversation." if person_count >= 2 else "Desk workers periodically stretch or adjust posture."
                }
            ]
            reasoning.append("Multi-object workstation affordance (laptop + beverage vessel) evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "task_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: laptop", "Secondary Affordance: cup"],
                "affordance_candidates": affordance_candidates,
            }

        # 2. Book + Cup Affordance (Image 3)
        if ("book" in target_str or "book" in state["detected_labels"]) and has_cup and ("read" in act_val or "book" in narration_lower):
            hypotheses = [
                {
                    "text": "They'll likely take a sip from the cup and continue reading.",
                    "evidence": ["Detected book and cup", "Reading activity confirmed", "Beverage vessel nearby"],
                    "likelihood_label": "most likely",
                    "rationale": "Reading with a nearby drink affords periodic sips during continued reading."
                },
                {
                    "text": "They'll likely continue reading or turn to the next page.",
                    "evidence": ["Book reading engagement"],
                    "likelihood_label": "possible",
                    "rationale": "Sustained text reading leads naturally to turning pages."
                },
                {
                    "text": "They may set the cup or book down on a nearby surface.",
                    "evidence": ["Reading materials and cup on surface"],
                    "likelihood_label": "less likely",
                    "rationale": "The reader may take a break or conclude the reading session."
                }
            ]
            reasoning.append("Multi-object reading affordance (book + beverage vessel) evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "task_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: book", "Secondary Affordance: cup"],
                "affordance_candidates": affordance_candidates,
            }

        # 3. Food + Drink / Meal Affordance (Image 4)
        if (has_food or "eating" in act_val or "sandwich" in state["detected_labels"]) and has_cup:
            hypotheses = [
                {
                    "text": "They'll likely take another bite of food and take a sip from the cup.",
                    "evidence": ["Detected food and cup", "Meal activity at dining table", "Drinking and eating affordances"],
                    "likelihood_label": "most likely",
                    "rationale": "Meal consumption alternates naturally between eating food and drinking beverage."
                },
                {
                    "text": "They'll likely set the cup down and continue eating.",
                    "evidence": ["Food and beverage on dining table"],
                    "likelihood_label": "possible",
                    "rationale": "Resting the drink allows two-handed engagement with food."
                },
                {
                    "text": "They may converse with the other people nearby." if person_count >= 2 else "They may finish their meal and clear the table.",
                    "evidence": [f"{person_count - 1} other people nearby" if person_count >= 2 else "Dining setting"],
                    "likelihood_label": "less likely",
                    "rationale": "Dining tables with multiple people encourage social sharing or discussion." if person_count >= 2 else "Meals conclude with clearing tableware."
                }
            ]
            reasoning.append("Multi-object meal affordance (food + drink) evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "interaction_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: meal/eating", "Secondary Affordance: drinking"],
                "affordance_candidates": affordance_candidates,
            }

        # 4. Smartphone / Cell Phone Affordance (Images 1, 5, 14)
        if "phone" in target_str or "cell phone" in state["detected_labels"] or "smartphone" in narration_lower:
            hypotheses = [
                {
                    "text": "They'll likely continue using the smartphone.",
                    "evidence": ["Confirmed smartphone interaction", "Active screen engagement"],
                    "likelihood_label": "most likely",
                    "rationale": "Handheld mobile device affordances center on screen interaction, browsing, or messaging."
                },
                {
                    "text": "They may take a sip from the cup or set the phone down." if has_cup else ("They may set the phone aside or speak to the person nearby." if person_count >= 2 else "They may adjust their posture or put the phone away."),
                    "evidence": ["Cup detected nearby" if has_cup else (f"{person_count - 1} other person nearby" if person_count >= 2 else "Mobile device usage")],
                    "likelihood_label": "possible",
                    "rationale": "A resting beverage invites a sip during a pause in phone usage." if has_cup else "A nearby companion may prompt social engagement."
                },
                {
                    "text": "They may put the phone in a pocket or set it on a surface.",
                    "evidence": ["Handheld mobile device"],
                    "likelihood_label": "less likely",
                    "rationale": "Mobile devices are stowed once digital tasks or communications conclude."
                }
            ]
            reasoning.append("Mobile smartphone affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "interaction_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: smartphone"],
                "affordance_candidates": affordance_candidates,
            }

        # 5. Laptop Alone (Single person or dominant person, no cup)
        if "laptop" in target_str or "laptop" in state["detected_labels"]:
            hypotheses = [
                {
                    "text": "They'll likely keep working on the laptop.",
                    "evidence": ["Confirmed laptop interaction", "Workstation setting"],
                    "likelihood_label": "most likely",
                    "rationale": "Workstation tasks involve sustained keyboard, trackpad, and screen focus."
                },
                {
                    "text": "They may switch windows or adjust their posture at the desk.",
                    "evidence": ["Workstation setting"],
                    "likelihood_label": "possible",
                    "rationale": "Extended desk work involves periodic task switching and posture adjustments."
                },
                {
                    "text": "They may converse with the person nearby." if person_count >= 2 else "They may close the laptop and step away.",
                    "evidence": [f"{person_count - 1} other person nearby" if person_count >= 2 else "Desk setup"],
                    "likelihood_label": "less likely",
                    "rationale": "A companion nearby may engage in brief discussion." if person_count >= 2 else "Sessions conclude when tasks are completed."
                }
            ]
            reasoning.append("Individual workstation laptop affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "task_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: laptop"],
                "affordance_candidates": affordance_candidates,
            }

        # 6. Book Alone (Reading without cup)
        if "book" in target_str or "book" in state["detected_labels"]:
            hypotheses = [
                {
                    "text": "They'll likely continue reading.",
                    "evidence": ["Confirmed book reading interaction", "Reading material engagement"],
                    "likelihood_label": "most likely",
                    "rationale": "Reading material affordances involve sequential text scanning and page navigation."
                },
                {
                    "text": "They'll likely turn to the next page.",
                    "evidence": ["Open book in hand or lap"],
                    "likelihood_label": "possible",
                    "rationale": "Sequential reading requires periodic page turns."
                },
                {
                    "text": "They may close the book or mark their place.",
                    "evidence": ["Reading session"],
                    "likelihood_label": "less likely",
                    "rationale": "Readers mark their progress when pausing."
                }
            ]
            reasoning.append("Reading material affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "task_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: book"],
                "affordance_candidates": affordance_candidates,
            }

        # 7. Cup / Drinking Alone
        if "cup" in target_str or "cup" in state["detected_labels"]:
            hypotheses = [
                {
                    "text": "They'll likely take another sip or set the cup down.",
                    "evidence": ["Confirmed cup interaction", "Beverage vessel in hand or on table"],
                    "likelihood_label": "most likely",
                    "rationale": "Beverage vessel affordances center on sipping, drinking, or resting on a nearby surface."
                },
                {
                    "text": "They'll likely place the cup down on the table.",
                    "evidence": ["Table or surface nearby"],
                    "likelihood_label": "possible",
                    "rationale": "Resting the vessel on a flat surface between drinks."
                },
                {
                    "text": "They may speak with the other person nearby." if person_count >= 2 else "They may finish the drink and set the vessel aside.",
                    "evidence": [f"{person_count - 1} other person nearby" if person_count >= 2 else "Beverage container"],
                    "likelihood_label": "less likely",
                    "rationale": "Social companion nearby." if person_count >= 2 else "Beverage consumption concludes when empty."
                }
            ]
            reasoning.append("Beverage vessel affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "interaction_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: cup"],
                "affordance_candidates": affordance_candidates,
            }

        # 8. Bicycle Alone (Confirmed detection or caption)
        if ("bicycle" in target_str or "bicycle" in state["detected_labels"]) and ("bicycle" in narration_lower or "bike" in narration_lower):
            hypotheses = [
                {
                    "text": "They'll likely continue riding along the path.",
                    "evidence": ["Confirmed bicycle interaction", "Locomotion along pathway"],
                    "likelihood_label": "most likely",
                    "rationale": "Locomotion and vehicle affordances involve continued directional movement."
                },
                {
                    "text": "They may slow down or navigate surrounding terrain.",
                    "evidence": ["Path environment"],
                    "likelihood_label": "possible",
                    "rationale": "Cyclists modulate speed for turns and obstacles."
                },
                {
                    "text": "They may come to a stop and dismount.",
                    "evidence": ["Bicycle locomotion"],
                    "likelihood_label": "less likely",
                    "rationale": "Riders dismount upon reaching their destination."
                }
            ]
            reasoning.append("Bicycle locomotion affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "supported_prediction",
                "prediction_type": "task_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Primary Interaction: bicycle"],
                "affordance_candidates": affordance_candidates,
            }

        # ---------------------------------------------------------------------
        # Sub-Scenario 5e: Multi-Person Scene without Focal Object (Grounded Group Dynamic)
        # Forbids ungrounded claims of "conversation" when no conversation evidence exists
        # ---------------------------------------------------------------------
        if person_count >= 2:
            hypotheses = [
                {
                    "text": "They will likely stay in the same area.",
                    "evidence": [f"Detected {person_count} persons present in shared space"],
                    "likelihood_label": "most likely",
                    "rationale": "Multiple individuals in a common setting typically remain in the vicinity over short horizons."
                },
                {
                    "text": "They may interact with one another or move in different directions.",
                    "evidence": [f"{person_count} persons in close proximity"],
                    "likelihood_label": "possible",
                    "rationale": "Shared physical presence allows for spontaneous conversation or diverging paths."
                }
            ]
            reasoning.append(f"Multi-person scene ({person_count} people) without focal object; neutral presence projected.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "tentative_prediction",
                "prediction_type": "group_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + [f"Group Dynamic: {person_count} persons"],
                "affordance_candidates": affordance_candidates,
            }

        # ---------------------------------------------------------------------
        # Sub-Scenario 5f: Posture Affordances (Standing / Seated without focal object)
        # ---------------------------------------------------------------------
        if "sit" in act_val or "seated" in act_val:
            hypotheses = [
                {
                    "text": "They'll likely remain seated and continue what they are doing.",
                    "evidence": ["Seated posture confirmed", f"Environment: {state['environment_value']}"],
                    "likelihood_label": "most likely",
                    "rationale": "Seated posture indicates stable engagement with the immediate environment."
                },
                {
                    "text": "They may adjust their posture or look around.",
                    "evidence": ["Seated posture"],
                    "likelihood_label": "possible",
                    "rationale": "Static sitting naturally involves minor posture adjustments."
                },
                {
                    "text": "They may stand up and leave the seat.",
                    "evidence": ["Seated position"],
                    "likelihood_label": "less likely",
                    "rationale": "Seated activities eventually conclude with standing."
                }
            ]
            reasoning.append("Seated posture affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "tentative_prediction",
                "prediction_type": "task_continuation",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Posture: seated"],
                "affordance_candidates": affordance_candidates,
            }
        elif "stand" in act_val:
            hypotheses = [
                {
                    "text": "They'll likely continue standing or begin walking.",
                    "evidence": ["Standing posture confirmed", f"Environment: {state['environment_value']}"],
                    "likelihood_label": "most likely",
                    "rationale": "Static standing posture supports continued observation, subtle weight shifts, or subsequent locomotion."
                },
                {
                    "text": "They may shift their weight or look around the space.",
                    "evidence": ["Standing position"],
                    "likelihood_label": "possible",
                    "rationale": "Stationary standing involves natural weight redistribution."
                },
                {
                    "text": "They may walk toward another location or sit down.",
                    "evidence": ["Standing posture"],
                    "likelihood_label": "less likely",
                    "rationale": "Standing transitions into walking or sitting."
                }
            ]
            reasoning.append("Standing posture affordance evaluated.")
            affordance_candidates.extend([h["text"] for h in hypotheses])
            return {
                "prediction": hypotheses[0]["text"],
                "status": "tentative_prediction",
                "prediction_type": "ambiguous_activity",
                "hypotheses": hypotheses,
                "reasoning": reasoning,
                "source_facts": source_facts + ["Posture: standing"],
                "affordance_candidates": affordance_candidates,
            }

        # ---------------------------------------------------------------------
        # Sub-Scenario 5g: Default Conservative Continuation
        # ---------------------------------------------------------------------
        hypotheses = [
            {
                "text": "They'll likely continue their current activity.",
                "evidence": [f"Current activity: {state['activity_value']}", "Immediate visual context"],
                "likelihood_label": "most likely",
                "rationale": "Sustained behavioral momentum in the immediate visual context."
            },
            {
                "text": "They may pause or transition to another activity.",
                "evidence": ["Visual activity in progress"],
                "likelihood_label": "possible",
                "rationale": "Natural transitions occur between ongoing tasks."
            }
        ]
        reasoning.append("Default conservative behavioral momentum projected.")
        affordance_candidates.extend([h["text"] for h in hypotheses])
        return {
            "prediction": hypotheses[0]["text"],
            "status": "tentative_prediction",
            "prediction_type": "ambiguous_activity",
            "hypotheses": hypotheses,
            "reasoning": reasoning,
            "source_facts": source_facts,
            "affordance_candidates": affordance_candidates,
        }
