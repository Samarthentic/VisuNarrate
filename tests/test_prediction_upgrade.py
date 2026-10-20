"""VisuNarrate - Unit Tests for Prediction Upgrade & Narration Consistency.

Covers:
1. Structured hypothesis schema (1-3 hypotheses, strict likelihood labels, no percentages).
2. Narration/prediction consistency for group scenes with nearby items (forbids 'keep working on laptop').
3. Unification of empty desk scenes (Images 11 & 12).
4. Grounded pedestrian and street traffic affordances (forbids ungrounded 'conversation' claims).
5. Grounded pedestrian walking locomotion (forbids ungrounded 'conversation' claims).
6. Multi-object affordance: laptop + cup.
7. Multi-object affordance: food/sandwich + cup.
8. Multi-object affordance: book + cup.
9. Graphic media temporal suppression.
10. Strict grounding: forbids unsupported vehicles or objects.
"""

import unittest
from services.evidence_fusion import EvidenceFusionEngine
from services.narration_service import NarrationService
from services.prediction_service import PredictionService


class TestPredictionUpgrade(unittest.TestCase):

    def setUp(self):
        self.fusion = EvidenceFusionEngine()
        self.narrator = NarrationService()
        self.predictor = PredictionService()

    def _assert_valid_structured_result(self, result: dict):
        """Helper to validate structured schema conformance."""
        self.assertIn("prediction", result)
        self.assertIn("status", result)
        self.assertIn("prediction_type", result)
        self.assertIn("hypotheses", result)

        hypotheses = result["hypotheses"]
        self.assertIsInstance(hypotheses, list)
        self.assertTrue(1 <= len(hypotheses) <= 3, f"Expected 1-3 hypotheses, got {len(hypotheses)}")

        # Prediction string must match top hypothesis text
        self.assertEqual(result["prediction"], hypotheses[0]["text"])

        allowed_labels = {"most likely", "possible", "less likely"}
        for h in hypotheses:
            self.assertIn("text", h)
            self.assertIn("evidence", h)
            self.assertIn("likelihood_label", h)
            self.assertIn("rationale", h)

            self.assertIn(h["likelihood_label"], allowed_labels)
            self.assertIsInstance(h["evidence"], list)
            self.assertGreater(len(h["evidence"]), 0, "Hypothesis evidence list must not be empty.")
            self.assertIsInstance(h["rationale"], str)
            self.assertGreater(len(h["rationale"]), 0)

            # Ensure no invented percentages are present
            self.assertNotIn("%", h["text"])
            self.assertNotIn("%", h["likelihood_label"])
            self.assertNotIn("%", h["rationale"])

    def test_structured_schema_conformance(self):
        """Verifies that single-user and group predictions conform to the structured hypothesis schema."""
        fused = {
            "summary": "A person sitting at a table with a laptop",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 1, "has_dominant_person": True},
            "primary_subject": {"value": "A person", "status": "supported"},
            "activity": {"value": "working on a laptop", "status": "supported"},
            "environment": {"value": "indoors", "status": "supported"},
            "objects": [{"label": "laptop", "confidence": 0.95}],
            "interactions": [{"target": "laptop", "relation": "using", "status": "confirmed"}],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused)
        self._assert_valid_structured_result(res)
        self.assertEqual(res["status"], "supported_prediction")
        self.assertEqual(res["prediction_type"], "task_continuation")

    def test_narration_prediction_consistency_group_laptop(self):
        """Images 7 & 8: Group with laptop nearby must predict group gathering, not individual work."""
        fused = {
            "summary": "7 people gathered together around a table",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 7, "has_dominant_person": False},
            "primary_subject": {"value": "A group of people", "status": "supported"},
            "activity": {"value": "sitting", "status": "supported"},
            "environment": {"value": "indoors", "status": "supported"},
            "objects": [{"label": "laptop", "confidence": 0.88}],
            "interactions": [{"target": "laptop", "relation": "near", "status": "supported"}],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        narration = "7 people are gathered together, with a laptop nearby."
        res = self.predictor.generate_predictive_narration(fused, narration=narration)

        self._assert_valid_structured_result(res)
        # Top hypothesis must NOT claim they keep working on the laptop
        self.assertNotIn("keep working on the laptop", res["prediction"].lower())
        self.assertNotIn("typing on the laptop", res["prediction"].lower())
        self.assertIn("group", res["prediction"].lower())
        self.assertIn("discussion", res["prediction"].lower())

        # Second hypothesis should mention the nearby laptop
        self.assertEqual(res["hypotheses"][1]["likelihood_label"], "possible")
        self.assertIn("laptop", res["hypotheses"][1]["text"].lower())

    def test_unify_empty_desk_scenes(self):
        """Images 11 & 12: Both empty desk scenes must yield the exact same unified prediction."""
        # Scene 11: 0 people, presence uncertain, laptop + books + cup
        scene_11 = {
            "summary": "A laptop computer sitting on top of a wooden desk",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "uncertain", "count": 0, "has_dominant_person": False},
            "primary_subject": {"value": "A laptop", "status": "supported"},
            "activity": {"value": "sitting on desk", "status": "supported"},
            "environment": {"value": "indoors", "status": "supported"},
            "objects": [
                {"label": "laptop", "confidence": 0.95},
                {"label": "book", "confidence": 0.82},
                {"label": "cup", "confidence": 0.75},
            ],
            "interactions": [],
            "conflicts": [],
            "uncertainty_flags": [],
        }

        # Scene 12: 0 people, presence absent, laptop + books + backpack
        scene_12 = {
            "summary": "A desk with a laptop, notebook, and other items on it",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "absent", "count": 0, "has_dominant_person": False},
            "primary_subject": {"value": "A desk", "status": "supported"},
            "activity": {"value": "stationary", "status": "supported"},
            "environment": {"value": "indoors", "status": "supported"},
            "objects": [
                {"label": "laptop", "confidence": 0.92},
                {"label": "book", "confidence": 0.85},
                {"label": "backpack", "confidence": 0.70},
            ],
            "interactions": [],
            "conflicts": [],
            "uncertainty_flags": [],
        }

        res_11 = self.predictor.generate_predictive_narration(scene_11)
        res_12 = self.predictor.generate_predictive_narration(scene_12)

        self._assert_valid_structured_result(res_11)
        self._assert_valid_structured_result(res_12)

        # Both must match the unified workspace prediction
        expected_pred = "The workspace will likely remain steady until someone uses it."
        self.assertEqual(res_11["prediction"], expected_pred)
        self.assertEqual(res_12["prediction"], expected_pred)

        self.assertEqual(res_11["prediction_type"], "environmental_static")
        self.assertEqual(res_12["prediction_type"], "environmental_static")

        self.assertEqual(res_11["status"], "insufficient_evidence")
        self.assertEqual(res_12["status"], "insufficient_evidence")

    def test_pedestrian_traffic_affordance(self):
        """Image 10: Pedestrians near street vehicles must predict traffic awareness, NOT conversation."""
        fused = {
            "summary": "A wet street with cars and people",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 3, "has_dominant_person": False},
            "primary_subject": {"value": "pedestrians", "status": "supported"},
            "activity": {"value": "standing", "status": "supported"},
            "environment": {"value": "street", "status": "supported"},
            "objects": [
                {"label": "person", "confidence": 0.85},
                {"label": "car", "confidence": 0.89},
                {"label": "bus", "confidence": 0.78},
                {"label": "motorcycle", "confidence": 0.70},
            ],
            "interactions": [],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused)

        self._assert_valid_structured_result(res)
        self.assertNotIn("conversation", res["prediction"].lower())
        self.assertIn("traffic", res["prediction"].lower())
        self.assertEqual(res["prediction_type"], "pedestrian_traffic")

    def test_pedestrian_walking_locomotion(self):
        """Image 6: Walking pedestrian with others nearby must predict walking, NOT conversation."""
        fused = {
            "summary": "A man in a black hoodie walking on a sidewalk",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 5, "has_dominant_person": True},
            "primary_subject": {"value": "A person", "status": "supported"},
            "activity": {"value": "walking", "status": "supported"},
            "environment": {"value": "outdoors", "status": "supported"},
            "objects": [{"label": "person", "confidence": 0.90}],
            "interactions": [],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused, narration="A person is walking outdoors, with several other people nearby.")

        self._assert_valid_structured_result(res)
        self.assertNotIn("conversation", res["prediction"].lower())
        self.assertIn("walking", res["prediction"].lower())
        self.assertEqual(res["prediction_type"], "locomotion_continuation")

    def test_multi_object_laptop_and_cup(self):
        """Image 2: Person at desk with laptop and cup must predict sipping cup then typing on laptop."""
        fused = {
            "summary": "A person sitting at a desk with a laptop and cup",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 2, "has_dominant_person": True},
            "primary_subject": {"value": "A person", "status": "supported"},
            "activity": {"value": "working on a laptop", "status": "supported"},
            "environment": {"value": "desk", "status": "supported"},
            "objects": [
                {"label": "person", "confidence": 0.92},
                {"label": "laptop", "confidence": 0.94},
                {"label": "cup", "confidence": 0.88},
            ],
            "interactions": [{"target": "laptop", "relation": "using", "status": "confirmed"}],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused)

        self._assert_valid_structured_result(res)
        self.assertIn("sip from the cup", res["prediction"].lower())
        self.assertIn("laptop", res["prediction"].lower())
        self.assertEqual(res["prediction_type"], "task_continuation")

    def test_multi_object_food_and_drink(self):
        """Image 4: Meal with sandwich and cup must predict bite of food and sip of beverage."""
        fused = {
            "summary": "A person sitting at a table eating a sandwich and drinking coffee",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 3, "has_dominant_person": True},
            "primary_subject": {"value": "A person", "status": "supported"},
            "activity": {"value": "eating", "status": "supported"},
            "environment": {"value": "dining table", "status": "supported"},
            "objects": [
                {"label": "person", "confidence": 0.90},
                {"label": "sandwich", "confidence": 0.91},
                {"label": "cup", "confidence": 0.89},
            ],
            "interactions": [{"target": "cup", "relation": "holding", "status": "confirmed"}],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused)

        self._assert_valid_structured_result(res)
        self.assertIn("bite of food", res["prediction"].lower())
        self.assertIn("sip from the cup", res["prediction"].lower())

    def test_multi_object_book_and_cup(self):
        """Image 3: Person reading with cup must predict sip from cup and continue reading."""
        fused = {
            "summary": "A person sitting in a chair reading a book with a coffee cup",
            "scene_type": {"scene_type": "photographic_scene", "is_graphic": False},
            "person_presence": {"status": "present", "count": 1, "has_dominant_person": True},
            "primary_subject": {"value": "A person", "status": "supported"},
            "activity": {"value": "reading a book", "status": "supported"},
            "environment": {"value": "indoors", "status": "supported"},
            "objects": [
                {"label": "person", "confidence": 0.92},
                {"label": "book", "confidence": 0.93},
                {"label": "cup", "confidence": 0.85},
            ],
            "interactions": [{"target": "book", "relation": "reading", "status": "confirmed"}],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused)

        self._assert_valid_structured_result(res)
        self.assertIn("sip from the cup", res["prediction"].lower())
        self.assertIn("reading", res["prediction"].lower())

    def test_graphic_media_safety_gating(self):
        """Graphic media must pause temporal predictive storytelling."""
        fused = {
            "summary": "An illustration depicting the pyramids",
            "scene_type": {"scene_type": "illustration_or_graphic", "is_graphic": True},
            "person_presence": {"status": "absent", "count": 0, "has_dominant_person": False},
            "primary_subject": {"value": "pyramids", "status": "supported"},
            "activity": {"value": "static", "status": "supported"},
            "environment": {"value": "graphic", "status": "supported"},
            "objects": [],
            "interactions": [],
            "conflicts": [],
            "uncertainty_flags": [],
        }
        res = self.predictor.generate_predictive_narration(fused)

        self._assert_valid_structured_result(res)
        self.assertEqual(res["status"], "suppressed")
        self.assertEqual(res["prediction_type"], "suppressed_graphic")
        self.assertIn("paused for static illustrations", res["prediction"])


if __name__ == "__main__":
    unittest.main()
