"""VisuNarrate - Test Suite for Narration Correctness Fixes.

Covers tests for:
1. PERSON COUNT: Detector's person count preserved; dominant person with others nearby narrated correctly.
2. UNSUPPORTED ACTIONS: Discards ungrounded bicycle action when no bicycle is in YOLO or caption.
3. GROUP vs OBJECT: Group scenes with single laptop describe group and object separately.
4. DESK SCENES WITHOUT PEOPLE: Laptops, books, and monitors prioritized over backpacks and decor.
5. SECOND OBJECT: High-confidence relevant second object (e.g. cup next to laptop) mentioned briefly.
6. GRAMMAR: Cleans double articles ("a the pyramids" -> "the pyramids"), missing articles ("capturing lake" -> "capturing a lake"), and repeated words ("wet wet").
7. FALSE POSITIVES: Low-confidence anomalous detections (e.g. frisbee in sky/landscape) suppressed.
"""

import unittest
from services.evidence_fusion import EvidenceFusionEngine
from services.narration_service import NarrationService


class TestNarrationCorrectness(unittest.TestCase):

    def setUp(self):
        self.fusion = EvidenceFusionEngine()
        self.narrator = NarrationService()

    # -------------------------------------------------------------------------
    # Issue 1: PERSON COUNT
    # -------------------------------------------------------------------------
    def test_issue_1_person_count_and_dominant_actor(self):
        """Do not discard person count when caption is singular. Dominant person with others nearby."""
        caption_evidence = {
            "summary": "A man in a black hoodie is standing on a sidewalk",
            "observations": {
                "summary": "A man in a black hoodie is standing on a sidewalk",
                "subject": "A man",
                "action": "walking",
                "environment": "outdoors"
            }
        }
        # 1 large foreground person, 4 small background people
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.92, "bbox": {"x1": 200, "y1": 50, "x2": 600, "y2": 700}},  # Area: 260000
                {"label": "person", "confidence": 0.70, "bbox": {"x1": 50, "y1": 200, "x2": 150, "y2": 350}},  # Area: 15000
                {"label": "person", "confidence": 0.65, "bbox": {"x1": 700, "y1": 220, "x2": 780, "y2": 340}}, # Area: 9600
                {"label": "person", "confidence": 0.60, "bbox": {"x1": 800, "y1": 210, "x2": 870, "y2": 330}}, # Area: 8400
                {"label": "person", "confidence": 0.55, "bbox": {"x1": 890, "y1": 210, "x2": 950, "y2": 320}}, # Area: 6600
            ],
            "counts": {"person": 5},
            "total_detections": 5
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Nothing"},
            {"question": "What is the person doing?", "answer": "Walking"},
            {"question": "What is the person wearing?", "answer": "Black hoodie"},
            {"question": "Is the person indoors or outdoors?", "answer": "Outdoors"},
        ]

        presence = self.fusion.determine_person_presence(object_evidence, caption_evidence)
        self.assertEqual(presence["count"], 5, "Total detected person count should be preserved.")
        self.assertTrue(presence["has_dominant_person"], "Foreground person should be detected as dominant.")

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=presence,
            scene_type_info={"scene_type": "photographic_scene", "is_graphic": False}
        )

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("nearby", narration.lower(), "Should mention nearby people.")
        self.assertTrue("other people nearby" in narration.lower() or "another person nearby" in narration.lower())

    # -------------------------------------------------------------------------
    # Issue 2: UNSUPPORTED ACTIONS
    # -------------------------------------------------------------------------
    def test_issue_2_unsupported_bicycle_action_suppression(self):
        """Street scene with motorcycle/car/bus and VQA 'riding bike' must not claim riding a bicycle."""
        caption_evidence = {
            "summary": "A wet street with cars and people",
            "observations": {
                "summary": "A wet street with cars and people",
                "subject": "people",
                "action": "walking",
                "environment": "outdoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.85, "bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 500}},
                {"label": "person", "confidence": 0.80, "bbox": {"x1": 350, "y1": 120, "x2": 500, "y2": 480}},
                {"label": "car", "confidence": 0.88, "bbox": {"x1": 600, "y1": 200, "x2": 900, "y2": 500}},
                {"label": "motorcycle", "confidence": 0.403, "bbox": {"x1": 50, "y1": 300, "x2": 150, "y2": 450}},
            ],
            "counts": {"person": 2, "car": 1, "motorcycle": 1},
            "total_detections": 4
        }
        # VQA hallucinates "Riding bike"
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Umbrella"},
            {"question": "What is the person doing?", "answer": "Riding bike"},
            {"question": "What is the person wearing?", "answer": "Red jacket"},
            {"question": "Is the person indoors or outdoors?", "answer": "Outdoors"},
        ]
        presence = {"status": "present", "count": 2, "has_dominant_person": False, "confidence_level": "confirmed"}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=presence,
            scene_type_info={"scene_type": "photographic_scene", "is_graphic": False}
        )

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertNotIn("bicycle", narration.lower(), "Bicycle must not be claimed when none detected.")
        self.assertNotIn("riding a bike", narration.lower())

    # -------------------------------------------------------------------------
    # Issue 3: GROUP vs OBJECT
    # -------------------------------------------------------------------------
    def test_issue_3_group_vs_object_separation(self):
        """In group scenes (7 people, 1 laptop), describe group and object separately."""
        caption_evidence = {
            "summary": "A group of people sitting at a table with laptops",
            "observations": {
                "summary": "A group of people sitting at a table with laptops",
                "subject": "people",
                "action": "sitting",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.85, "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 300}},
                {"label": "person", "confidence": 0.84, "bbox": {"x1": 110, "y1": 10, "x2": 200, "y2": 300}},
                {"label": "person", "confidence": 0.83, "bbox": {"x1": 210, "y1": 10, "x2": 300, "y2": 300}},
                {"label": "person", "confidence": 0.82, "bbox": {"x1": 310, "y1": 10, "x2": 400, "y2": 300}},
                {"label": "person", "confidence": 0.81, "bbox": {"x1": 410, "y1": 10, "x2": 500, "y2": 300}},
                {"label": "person", "confidence": 0.80, "bbox": {"x1": 510, "y1": 10, "x2": 600, "y2": 300}},
                {"label": "person", "confidence": 0.79, "bbox": {"x1": 610, "y1": 10, "x2": 700, "y2": 300}},
                {"label": "laptop", "confidence": 0.90, "bbox": {"x1": 200, "y1": 250, "x2": 400, "y2": 400}},
                {"label": "dining table", "confidence": 0.85, "bbox": {"x1": 0, "y1": 200, "x2": 800, "y2": 500}},
            ],
            "counts": {"person": 7, "laptop": 1, "dining table": 1},
            "total_detections": 9
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Laptop"},
            {"question": "What is the person doing?", "answer": "Working on laptop"},
            {"question": "What is the person wearing?", "answer": "Casual clothes"},
            {"question": "Is the person indoors or outdoors?", "answer": "Indoors"},
        ]
        presence = {"status": "present", "count": 7, "has_dominant_person": False, "confidence_level": "confirmed"}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=presence,
            scene_type_info={"scene_type": "photographic_scene", "is_graphic": False}
        )

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertNotIn("working on a laptop", narration.lower(), "Should not claim all 7 people are working on one laptop.")
        self.assertIn("laptop", narration.lower(), "Should mention the laptop as part of the setting.")

    # -------------------------------------------------------------------------
    # Issue 4: DESK SCENES WITHOUT PEOPLE
    # -------------------------------------------------------------------------
    def test_issue_4_desk_scene_without_people_prioritization(self):
        """Desk scenes with laptops and books must not pick backpack over laptops."""
        caption_evidence = {
            "summary": "A desk with a laptop, notebook, and other items on it",
            "observations": {
                "summary": "A desk with a laptop, notebook, and other items on it",
                "subject": "desk",
                "action": "",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "laptop", "confidence": 0.92, "bbox": {"x1": 100, "y1": 100, "x2": 350, "y2": 300}},
                {"label": "laptop", "confidence": 0.88, "bbox": {"x1": 400, "y1": 100, "x2": 650, "y2": 300}},
                {"label": "book", "confidence": 0.80, "bbox": {"x1": 50, "y1": 250, "x2": 150, "y2": 350}},
                {"label": "backpack", "confidence": 0.65, "bbox": {"x1": 700, "y1": 100, "x2": 850, "y2": 400}},
                {"label": "dining table", "confidence": 0.85, "bbox": {"x1": 0, "y1": 150, "x2": 900, "y2": 500}},
            ],
            "counts": {"laptop": 2, "book": 1, "backpack": 1, "dining table": 1},
            "total_detections": 5
        }
        # VQA ungroundedly suggests backpack as main subject
        vqa_evidence = [
            {"question": "What is the main subject in the image?", "answer": "Backpack"},
            {"question": "Is the scene indoors or outdoors?", "answer": "Indoors"},
            {"question": "What is the weather or lighting like in the scene?", "answer": "Indoor lighting"},
        ]
        presence = {"status": "absent", "count": 0, "has_dominant_person": False, "confidence_level": "confirmed"}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=presence,
            scene_type_info={"scene_type": "photographic_scene", "is_graphic": False}
        )

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("laptop", narration.lower(), "Laptops must be prioritized on desk scene.")
        self.assertNotIn("backpack", narration.lower(), "Backpack should not take precedence over laptops.")

    # -------------------------------------------------------------------------
    # Issue 5: SECOND OBJECT
    # -------------------------------------------------------------------------
    def test_issue_5_relevant_second_object_mention(self):
        """When a cup or sandwich is present next to a laptop or meal, mention it briefly."""
        caption_evidence = {
            "summary": "A man sitting at a desk with a laptop and coffee",
            "observations": {
                "summary": "A man sitting at a desk with a laptop and coffee",
                "subject": "A man",
                "action": "working on a laptop",
                "environment": "indoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "person", "confidence": 0.95, "bbox": {"x1": 100, "y1": 50, "x2": 500, "y2": 600}},
                {"label": "laptop", "confidence": 0.93, "bbox": {"x1": 250, "y1": 350, "x2": 600, "y2": 550}},
                {"label": "cup", "confidence": 0.91, "bbox": {"x1": 50, "y1": 400, "x2": 150, "y2": 500}},
                {"label": "dining table", "confidence": 0.85, "bbox": {"x1": 0, "y1": 350, "x2": 800, "y2": 700}},
            ],
            "counts": {"person": 1, "laptop": 1, "cup": 1, "dining table": 1},
            "total_detections": 4
        }
        vqa_evidence = [
            {"question": "What is the person holding?", "answer": "Laptop"},
            {"question": "What is the person doing?", "answer": "Working on laptop"},
            {"question": "What is the person wearing?", "answer": "Shirt"},
            {"question": "Is the person indoors or outdoors?", "answer": "Indoors"},
        ]
        presence = {"status": "present", "count": 1, "has_dominant_person": False, "confidence_level": "confirmed"}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=presence,
            scene_type_info={"scene_type": "photographic_scene", "is_graphic": False}
        )

        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertIn("laptop", narration.lower())
        self.assertIn("cup", narration.lower(), "Second object (cup) should be mentioned briefly.")

    # -------------------------------------------------------------------------
    # Issue 6: GRAMMAR
    # -------------------------------------------------------------------------
    def test_issue_6_grammar_cleanup(self):
        """Fix 'a the pyramids', 'capturing lake', double articles, and repeated words ('wet wet')."""
        # Test repeated words
        cleaned_rep = self.narrator._clean_grammar("A wet wet street with rain.")
        self.assertEqual(cleaned_rep, "A wet street with rain.")

        # Test double article
        cleaned_art = self.narrator._clean_grammar("A graphic illustration depicting a the pyramids.")
        self.assertEqual(cleaned_art, "A graphic illustration depicting the pyramids.")

        # Test missing article in capturing / showing
        cleaned_cap = self.narrator._clean_grammar("An open outdoor scene capturing lake.")
        self.assertEqual(cleaned_cap, "An open outdoor scene capturing a lake.")

        cleaned_show = self.narrator._clean_grammar("A quiet scene showing backpack.")
        self.assertEqual(cleaned_show, "A quiet scene showing a backpack.")

    # -------------------------------------------------------------------------
    # Issue 7: FALSE POSITIVES
    # -------------------------------------------------------------------------
    def test_issue_7_false_positive_frisbee_suppression(self):
        """Frisbee in a landscape/sky without human presence must be suppressed as a false positive."""
        caption_evidence = {
            "summary": "The pyramids of egypt, with the sun in the background",
            "observations": {
                "summary": "The pyramids of egypt, with the sun in the background",
                "subject": "pyramids",
                "action": "",
                "environment": "outdoors"
            }
        }
        object_evidence = {
            "objects": [
                {"label": "frisbee", "confidence": 0.439, "bbox": {"x1": 500, "y1": 100, "x2": 550, "y2": 150}},
            ],
            "counts": {"frisbee": 1},
            "total_detections": 1
        }
        vqa_evidence = [
            {"question": "What is the main subject in the image?", "answer": "Pyramids"},
            {"question": "Is the scene indoors or outdoors?", "answer": "Outdoors"},
            {"question": "What is the weather or lighting like in the scene?", "answer": "Sunny"},
        ]
        presence = {"status": "absent", "count": 0, "has_dominant_person": False, "confidence_level": "confirmed"}

        fused = self.fusion.fuse_evidence(
            caption_evidence=caption_evidence,
            object_evidence=object_evidence,
            vqa_evidence=vqa_evidence,
            person_presence=presence,
            scene_type_info={"scene_type": "illustration_or_graphic", "is_graphic": True}
        )

        self.assertEqual(len(fused["objects"]), 0, "Anomalous frisbee should be filtered from fused objects.")
        narration = self.narrator.synthesize_scene_narrative(fused)
        self.assertNotIn("frisbee", narration.lower())
        self.assertNotIn("a the", narration.lower())


if __name__ == "__main__":
    unittest.main()
