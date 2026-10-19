"""VisuNarrate - Regression Test Suite for Multi-Source Interaction Grounding.

Verifies:
1. Laptop Scene grounding (laptop prioritised, weak/spurious cell phone rejected/outranked).
2. Smartphone Scene grounding (cell phone prioritised).
3. Reading Scene grounding (book prioritised).
4. Cup / Drinking Scene grounding (cup prioritised).
5. State Isolation across sequential inferences (zero cross-image leakage).
"""

import unittest
from services.evidence_fusion import EvidenceFusionEngine
from services.narration_service import NarrationService
from services.prediction_service import PredictionService


class TestInteractionGrounding(unittest.TestCase):

    def setUp(self):
        self.fusion = EvidenceFusionEngine()
        self.narrator = NarrationService()
        self.predictor = PredictionService()

    def test_laptop_scene_grounding(self):
        """Laptop scene with high YOLO laptop conf and weak edge phone must prioritize laptop."""
        caption_evidence = {
            "summary": "A man sitting at a table using a laptop",
            "observations": {
                "summary": "A man sitting at a table using a laptop",
                "subject": "A laptop",
                "action": "using a laptop",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "laptop", "confidence": 0.957, "bbox": {"x1": 356, "y1": 469, "x2": 930, "y2": 770}},
                {"label": "cup", "confidence": 0.938, "bbox": {"x1": 50, "y1": 632, "x2": 252, "y2": 793}},
                {"label": "person", "confidence": 0.904, "bbox": {"x1": 114, "y1": 45, "x2": 734, "y2": 715}},
                {"label": "chair", "confidence": 0.818, "bbox": {"x1": 0, "y1": 520, "x2": 147, "y2": 738}},
                {"label": "cell phone", "confidence": 0.496, "bbox": {"x1": 905, "y1": 654, "x2": 1023, "y2": 712}},
            ],
            "counts": {"laptop": 1, "cup": 1, "person": 1, "chair": 1, "cell phone": 1},
            "total_detections": 5
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Laptop"},
            {"question": "What is the person doing?", "answer": "Using laptop"},
            {"question": "What is the person wearing?", "answer": "Black hoodie"},
            {"question": "Is the person indoors or outdoors?", "answer": "Indoors"},
        ]
        person_presence = {"status": "present", "count": 1, "confidence_level": "confirmed"}
        scene_type_info = {"scene_type": "photographic_scene", "is_graphic": False}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=person_presence,
            scene_type_info=scene_type_info
        )

        interactions = fused.get("interactions", [])
        self.assertTrue(len(interactions) > 0, "Expected at least one interaction candidate.")

        # Top interaction MUST be laptop
        top_target = interactions[0]["target"]
        self.assertIn("laptop", top_target, f"Expected laptop as top target, got: {top_target}")
        self.assertNotIn("phone", top_target, "Phone should not outrank strongly supported laptop.")
        self.assertEqual(interactions[0]["status"], "confirmed")

        # Narration verification
        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("laptop", narration.lower())
        self.assertNotIn("smartphone", narration.lower())
        self.assertNotIn("cell phone", narration.lower())

        # Prediction verification
        prediction = self.predictor.generate_predictive_narration(fused)
        self.assertIn("laptop", prediction["prediction"].lower())
        self.assertNotIn("smartphone", prediction["prediction"].lower())
        self.assertEqual(prediction["status"], "supported_prediction")

    def test_smartphone_scene_grounding(self):
        """Portrait scene with smartphone interaction must prioritize smartphone."""
        caption_evidence = {
            "summary": "A person standing indoors holding a cell phone",
            "observations": {
                "summary": "A person standing indoors holding a cell phone",
                "subject": "A person",
                "action": "holding a cell",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.92, "bbox": {"x1": 100, "y1": 50, "x2": 600, "y2": 700}},
                {"label": "cell phone", "confidence": 0.85, "bbox": {"x1": 300, "y1": 400, "x2": 380, "y2": 520}},
            ],
            "counts": {"person": 1, "cell phone": 1},
            "total_detections": 2
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Cell phone"},
            {"question": "What is the person doing?", "answer": "Looking at phone"},
            {"question": "Is the person indoors or outdoors?", "answer": "Indoors"},
        ]
        person_presence = {"status": "present", "count": 1, "confidence_level": "confirmed"}
        scene_type_info = {"scene_type": "photographic_scene", "is_graphic": False}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=person_presence,
            scene_type_info=scene_type_info
        )

        interactions = fused.get("interactions", [])
        self.assertTrue(len(interactions) > 0)
        self.assertIn("phone", interactions[0]["target"].lower())

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("smartphone", narration.lower())

        prediction = self.predictor.generate_predictive_narration(fused)
        self.assertIn("smartphone", prediction["prediction"].lower())
        self.assertEqual(prediction["status"], "supported_prediction")

    def test_reading_scene_grounding(self):
        """Reading scene with book must prioritize book."""
        caption_evidence = {
            "summary": "A person sitting indoors reading a book",
            "observations": {
                "summary": "A person sitting indoors reading a book",
                "subject": "A person",
                "action": "reading",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.91, "bbox": {"x1": 100, "y1": 50, "x2": 500, "y2": 700}},
                {"label": "book", "confidence": 0.88, "bbox": {"x1": 200, "y1": 350, "x2": 350, "y2": 450}},
            ],
            "counts": {"person": 1, "book": 1},
            "total_detections": 2
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Book"},
            {"question": "What is the person doing?", "answer": "Reading"},
            {"question": "Is the person indoors or outdoors?", "answer": "Indoors"},
        ]
        person_presence = {"status": "present", "count": 1, "confidence_level": "confirmed"}
        scene_type_info = {"scene_type": "photographic_scene", "is_graphic": False}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=person_presence,
            scene_type_info=scene_type_info
        )

        interactions = fused.get("interactions", [])
        self.assertIn("book", interactions[0]["target"].lower())

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("book", narration.lower())

        prediction = self.predictor.generate_predictive_narration(fused)
        self.assertIn("reading", prediction["prediction"].lower())
        self.assertEqual(prediction["status"], "supported_prediction")

    def test_cup_drinking_scene_grounding(self):
        """Cup drinking scene with beverage vessel must prioritize cup."""
        caption_evidence = {
            "summary": "A person sitting at a table drinking from a cup",
            "observations": {
                "summary": "A person sitting at a table drinking from a cup",
                "subject": "A person",
                "action": "drinking",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.94, "bbox": {"x1": 100, "y1": 50, "x2": 600, "y2": 700}},
                {"label": "cup", "confidence": 0.91, "bbox": {"x1": 250, "y1": 400, "x2": 320, "y2": 480}},
                {"label": "dining table", "confidence": 0.85, "bbox": {"x1": 50, "y1": 500, "x2": 700, "y2": 750}},
            ],
            "counts": {"person": 1, "cup": 1, "dining table": 1},
            "total_detections": 3
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Cup"},
            {"question": "What is the person doing?", "answer": "Drinking"},
            {"question": "Is the person indoors or outdoors?", "answer": "Indoors"},
        ]
        person_presence = {"status": "present", "count": 1, "confidence_level": "confirmed"}
        scene_type_info = {"scene_type": "photographic_scene", "is_graphic": False}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=person_presence,
            scene_type_info=scene_type_info
        )

        interactions = fused.get("interactions", [])
        self.assertIn("cup", interactions[0]["target"].lower())

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("cup", narration.lower())

        prediction = self.predictor.generate_predictive_narration(fused)
        self.assertIn("sip", prediction["prediction"].lower())
        self.assertEqual(prediction["status"], "supported_prediction")

    def test_state_isolation_across_sequential_calls(self):
        """Ensures consecutive calls for different scenes never leak interactions."""
        # 1. First run: Smartphone scene
        self.test_smartphone_scene_grounding()

        # 2. Second run: Laptop scene immediately after
        self.test_laptop_scene_grounding()

        # 3. Third run: Reading scene immediately after
        self.test_reading_scene_grounding()

        # 4. Fourth run: Laptop scene again
        self.test_laptop_scene_grounding()


if __name__ == "__main__":
    unittest.main()
