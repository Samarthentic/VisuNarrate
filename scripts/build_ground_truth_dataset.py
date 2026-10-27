"""VisuNarrate - Ground Truth Dataset Builder & Annotator.

Constructs ground-truth annotations for:
1. All 15 evaluation benchmark images in data/evaluation_images/
2. Test video data/test_videos/test_transition_activity.mp4

Includes:
- Reference human captions (for BLEU and ROUGE-L)
- Labeled spatial bounding boxes & classes (for Precision, Recall, F1, mAP@0.5)
- Reference VQA question-answer pairs (for Exact & Soft Match)
- Reference scene types & person presence
- Ground-truth temporal transition points & segment activities
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "evaluation_datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def build_image_ground_truth():
    """Builds ground-truth dataset for the 15 benchmark images."""
    annotations = {
        "ChatGPT Image Aug 20, 2026, 10_31_28 PM.png": {
            "image_id": "eval_01_phone_indoor",
            "filename": "ChatGPT Image Aug 20, 2026, 10_31_28 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 1,
            "primary_activity": "using smartphone",
            "reference_captions": [
                "A person wearing a black shirt sitting indoors looking at and using a smartphone.",
                "A young person holding a cell phone indoors with a cup and couch in the background.",
                "A person sitting on a chair interacting with a mobile phone in a living room.",
            ],
            "objects": [
                {"label": "person", "bbox": [435.0, 230.0, 970.0, 1080.0]},
                {"label": "cell phone", "bbox": [650.0, 520.0, 770.0, 690.0]},
                {"label": "cup", "bbox": [1010.0, 680.0, 1140.0, 830.0]},
                {"label": "chair", "bbox": [390.0, 580.0, 980.0, 1110.0]},
                {"label": "potted plant", "bbox": [1120.0, 420.0, 1380.0, 880.0]},
                {"label": "couch", "bbox": [50.0, 520.0, 480.0, 960.0]},
            ],
            "vqa": [
                {
                    "question": "What is the person holding?",
                    "reference_answers": ["phone", "cell phone", "smartphone", "mobile phone"]
                },
                {
                    "question": "Is the person indoors or outdoors?",
                    "reference_answers": ["indoors", "indoor", "in a room", "inside"]
                },
                {
                    "question": "What is the person wearing?",
                    "reference_answers": ["black shirt", "black t shirt", "t-shirt", "black top"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_33_23 PM.png": {
            "image_id": "eval_02_coworking_laptop",
            "filename": "ChatGPT Image Aug 20, 2026, 10_33_23 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 2,
            "primary_activity": "using laptop and smartphone",
            "reference_captions": [
                "Two people sitting at a table working with a laptop and smartphone indoors.",
                "Colleagues sitting together in an office or cafe with a laptop computer and coffee cup.",
                "A person in a black hoodie and another person working at a desk with technology devices.",
            ],
            "objects": [
                {"label": "person", "bbox": [340.0, 210.0, 780.0, 960.0]},
                {"label": "person", "bbox": [760.0, 250.0, 1180.0, 940.0]},
                {"label": "laptop", "bbox": [560.0, 550.0, 900.0, 820.0]},
                {"label": "cup", "bbox": [420.0, 680.0, 520.0, 810.0]},
                {"label": "cell phone", "bbox": [520.0, 640.0, 600.0, 750.0]},
                {"label": "chair", "bbox": [320.0, 540.0, 650.0, 1020.0]},
            ],
            "vqa": [
                {
                    "question": "What device is on the desk?",
                    "reference_answers": ["laptop", "computer", "notebook", "laptop computer"]
                },
                {
                    "question": "How many people are visible?",
                    "reference_answers": ["2", "two", "two people", "couple"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_35_05 PM.png": {
            "image_id": "eval_03_reading_book",
            "filename": "ChatGPT Image Aug 20, 2026, 10_35_05 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 1,
            "primary_activity": "reading book",
            "reference_captions": [
                "A person sitting comfortably in an armchair indoors reading a book.",
                "A person holding an open book and reading near a potted plant and coffee cup.",
                "A quiet indoor reading scene with an individual focused on a book.",
            ],
            "objects": [
                {"label": "person", "bbox": [410.0, 220.0, 940.0, 1050.0]},
                {"label": "book", "bbox": [580.0, 560.0, 810.0, 760.0]},
                {"label": "cup", "bbox": [980.0, 670.0, 1090.0, 810.0]},
                {"label": "chair", "bbox": [360.0, 520.0, 990.0, 1100.0]},
                {"label": "potted plant", "bbox": [1100.0, 410.0, 1370.0, 880.0]},
            ],
            "vqa": [
                {
                    "question": "What is the person reading?",
                    "reference_answers": ["book", "a book", "novel"]
                },
                {
                    "question": "What is the person sitting on?",
                    "reference_answers": ["chair", "armchair", "couch", "seat"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_35_58 PM.png": {
            "image_id": "eval_04_dining_social",
            "filename": "ChatGPT Image Aug 20, 2026, 10_35_58 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 3,
            "primary_activity": "eating / dining",
            "reference_captions": [
                "A group of friends sitting together at a dining table sharing food and drinks.",
                "Three people having a meal indoors with sandwiches and bowls on the dining table.",
                "People gathered at a table eating sandwiches and socializing in an indoor cafe.",
            ],
            "objects": [
                {"label": "person", "bbox": [200.0, 220.0, 560.0, 920.0]},
                {"label": "person", "bbox": [520.0, 180.0, 890.0, 890.0]},
                {"label": "person", "bbox": [880.0, 240.0, 1280.0, 910.0]},
                {"label": "dining table", "bbox": [180.0, 580.0, 1260.0, 1090.0]},
                {"label": "sandwich", "bbox": [520.0, 690.0, 690.0, 830.0]},
                {"label": "bowl", "bbox": [710.0, 680.0, 870.0, 820.0]},
                {"label": "cup", "bbox": [410.0, 640.0, 500.0, 770.0]},
            ],
            "vqa": [
                {
                    "question": "What are the people doing?",
                    "reference_answers": ["eating", "dining", "eating food", "having lunch", "sharing a meal"]
                },
                {
                    "question": "What food is visible on the table?",
                    "reference_answers": ["sandwich", "sandwiches", "bowl", "food"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_36_59 PM.png": {
            "image_id": "eval_05_living_room_device",
            "filename": "ChatGPT Image Aug 20, 2026, 10_36_59 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 1,
            "primary_activity": "using smartphone",
            "reference_captions": [
                "A person sitting on a living room couch using a smartphone with a laptop nearby.",
                "An indoor living room scene with a person holding a phone, sitting on a sofa.",
                "A young adult relaxing on a couch with electronic devices and plants in the room.",
            ],
            "objects": [
                {"label": "person", "bbox": [420.0, 220.0, 930.0, 1010.0]},
                {"label": "cell phone", "bbox": [630.0, 510.0, 740.0, 670.0]},
                {"label": "laptop", "bbox": [850.0, 640.0, 1140.0, 860.0]},
                {"label": "couch", "bbox": [120.0, 480.0, 1250.0, 1080.0]},
                {"label": "potted plant", "bbox": [1110.0, 390.0, 1370.0, 860.0]},
            ],
            "vqa": [
                {
                    "question": "What is the person holding?",
                    "reference_answers": ["phone", "cell phone", "smartphone"]
                },
                {
                    "question": "What piece of furniture is the person on?",
                    "reference_answers": ["couch", "sofa", "settee"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_37_57 PM.png": {
            "image_id": "eval_06_outdoor_skatepark",
            "filename": "ChatGPT Image Aug 20, 2026, 10_37_57 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 5,
            "primary_activity": "skateboarding / sports",
            "reference_captions": [
                "A group of youth skateboarding outdoors in an urban skate park setting.",
                "People riding skateboards and hanging out in an outdoor skate park under daylight.",
                "Skateboarders in hoodies active at a concrete outdoor park.",
            ],
            "objects": [
                {"label": "person", "bbox": [480.0, 260.0, 880.0, 930.0]},
                {"label": "person", "bbox": [190.0, 360.0, 420.0, 880.0]},
                {"label": "person", "bbox": [920.0, 380.0, 1160.0, 870.0]},
                {"label": "skateboard", "bbox": [560.0, 850.0, 780.0, 960.0]},
            ],
            "vqa": [
                {
                    "question": "What sport or activity is shown?",
                    "reference_answers": ["skateboarding", "skating", "skateboard"]
                },
                {
                    "question": "Is this scene indoors or outdoors?",
                    "reference_answers": ["outdoors", "outside", "skate park"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_38_57 PM.png": {
            "image_id": "eval_07_group_meeting",
            "filename": "ChatGPT Image Aug 20, 2026, 10_38_57 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 7,
            "primary_activity": "collaborating / group meeting",
            "reference_captions": [
                "A group of students or colleagues gathered indoors talking and using laptops.",
                "Multiple people sitting in a circle in a modern office space having a discussion.",
                "A collaborative group meeting with several individuals sitting together indoors.",
            ],
            "objects": [
                {"label": "person", "bbox": [410.0, 280.0, 740.0, 890.0]},
                {"label": "person", "bbox": [150.0, 340.0, 450.0, 870.0]},
                {"label": "person", "bbox": [720.0, 320.0, 1020.0, 880.0]},
                {"label": "laptop", "bbox": [580.0, 650.0, 810.0, 830.0]},
                {"label": "cell phone", "bbox": [490.0, 580.0, 570.0, 680.0]},
            ],
            "vqa": [
                {
                    "question": "Are multiple people gathered together?",
                    "reference_answers": ["yes", "yes group", "multiple people"]
                },
                {
                    "question": "What setting is this group in?",
                    "reference_answers": ["indoors", "office", "classroom", "indoor room"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_40_13 PM.png": {
            "image_id": "eval_08_outdoor_campus",
            "filename": "ChatGPT Image Aug 20, 2026, 10_40_13 PM.png",
            "dimensions": {"width": 1402, "height": 1122},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 12,
            "primary_activity": "studying outdoors",
            "reference_captions": [
                "College students gathered outdoors on a campus lawn studying and talking.",
                "A large group of young people sitting on grass outside working on laptops.",
                "Students relaxing and studying on an outdoor campus with laptops and backpacks.",
            ],
            "objects": [
                {"label": "person", "bbox": [460.0, 310.0, 830.0, 920.0]},
                {"label": "laptop", "bbox": [590.0, 640.0, 810.0, 820.0]},
                {"label": "backpack", "bbox": [280.0, 610.0, 460.0, 840.0]},
                {"label": "backpack", "bbox": [880.0, 630.0, 1040.0, 860.0]},
            ],
            "vqa": [
                {
                    "question": "Where are the people sitting?",
                    "reference_answers": ["grass", "lawn", "outside", "ground", "outdoors"]
                },
                {
                    "question": "What is the person in front using?",
                    "reference_answers": ["laptop", "computer", "notebook"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_41_18 PM.png": {
            "image_id": "eval_09_natural_lake",
            "filename": "ChatGPT Image Aug 20, 2026, 10_41_18 PM.png",
            "dimensions": {"width": 1536, "height": 1024},
            "scene_type": "photographic_scene",
            "person_presence": "absent",
            "person_count": 0,
            "primary_activity": "landscape",
            "reference_captions": [
                "A scenic natural landscape featuring a calm mountain lake and trees.",
                "A peaceful outdoor landscape with water and distant mountains under daylight.",
                "A picturesque natural view of a lake surrounded by forest and hills.",
            ],
            "objects": [],
            "vqa": [
                {
                    "question": "Are there any people in this picture?",
                    "reference_answers": ["no", "nobody", "none"]
                },
                {
                    "question": "What natural feature is prominent?",
                    "reference_answers": ["lake", "water", "mountain", "trees", "mountains"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_42_24 PM.png": {
            "image_id": "eval_10_city_street_traffic",
            "filename": "ChatGPT Image Aug 20, 2026, 10_42_24 PM.png",
            "dimensions": {"width": 1536, "height": 1024},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 3,
            "primary_activity": "riding bicycle / transit",
            "reference_captions": [
                "A cyclist in a red jacket riding a bicycle on a busy urban street with cars.",
                "City street with a cyclist, cars, and motorcycles in daylight traffic.",
                "A person riding a bike down a city street alongside cars and pedestrians.",
            ],
            "objects": [
                {"label": "person", "bbox": [580.0, 310.0, 930.0, 910.0]},
                {"label": "bicycle", "bbox": [560.0, 560.0, 960.0, 980.0]},
                {"label": "car", "bbox": [180.0, 480.0, 560.0, 780.0]},
                {"label": "car", "bbox": [980.0, 490.0, 1380.0, 810.0]},
                {"label": "motorcycle", "bbox": [910.0, 520.0, 1140.0, 860.0]},
            ],
            "vqa": [
                {
                    "question": "What is the person riding?",
                    "reference_answers": ["bicycle", "bike", "cycle"]
                },
                {
                    "question": "What color is the rider's jacket?",
                    "reference_answers": ["red", "red jacket"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_44_22 PM.png": {
            "image_id": "eval_11_desk_still_life",
            "filename": "ChatGPT Image Aug 20, 2026, 10_44_22 PM.png",
            "dimensions": {"width": 1536, "height": 1024},
            "scene_type": "photographic_scene",
            "person_presence": "absent",
            "person_count": 0,
            "primary_activity": "still life",
            "reference_captions": [
                "A clean workstation desk with an open laptop, mouse, books, and coffee cup.",
                "A modern workspace desk setup featuring a laptop computer, notebook, and accessories.",
                "Still life of a study desk with laptop, stack of books, and computer mouse.",
            ],
            "objects": [
                {"label": "laptop", "bbox": [360.0, 340.0, 940.0, 780.0]},
                {"label": "mouse", "bbox": [970.0, 580.0, 1100.0, 720.0]},
                {"label": "cup", "bbox": [210.0, 510.0, 360.0, 740.0]},
                {"label": "book", "bbox": [980.0, 410.0, 1340.0, 750.0]},
                {"label": "potted plant", "bbox": [80.0, 260.0, 320.0, 680.0]},
            ],
            "vqa": [
                {
                    "question": "Is anyone sitting at the desk?",
                    "reference_answers": ["no", "nobody", "empty", "no one"]
                },
                {
                    "question": "What computer equipment is on the desk?",
                    "reference_answers": ["laptop", "mouse", "computer", "laptop and mouse"]
                }
            ]
        },
        "ChatGPT Image Aug 20, 2026, 10_45_33 PM.png": {
            "image_id": "eval_12_study_table_items",
            "filename": "ChatGPT Image Aug 20, 2026, 10_45_33 PM.png",
            "dimensions": {"width": 1536, "height": 1024},
            "scene_type": "photographic_scene",
            "person_presence": "absent",
            "person_count": 0,
            "primary_activity": "still life",
            "reference_captions": [
                "A wooden dining or study table with a backpack, water bottle, books, and laptop.",
                "Still life arrangement of everyday study items including backpack and bottle on a table.",
                "A study desk scene with backpack, laptop, bottle, and stationery.",
            ],
            "objects": [
                {"label": "backpack", "bbox": [280.0, 310.0, 640.0, 860.0]},
                {"label": "bottle", "bbox": [670.0, 420.0, 790.0, 780.0]},
                {"label": "laptop", "bbox": [810.0, 430.0, 1260.0, 810.0]},
                {"label": "book", "bbox": [620.0, 610.0, 890.0, 850.0]},
            ],
            "vqa": [
                {
                    "question": "What is next to the laptop?",
                    "reference_answers": ["bottle", "water bottle", "book", "backpack"]
                },
                {
                    "question": "Is a person visible in the frame?",
                    "reference_answers": ["no", "no one", "nobody", "none"]
                }
            ]
        },
        "mountain_landscape.jpg": {
            "image_id": "eval_13_graphic_pyramids",
            "filename": "mountain_landscape.jpg",
            "dimensions": {"width": 800, "height": 600},
            "scene_type": "illustration_or_graphic",
            "person_presence": "absent",
            "person_count": 0,
            "primary_activity": "graphic illustration",
            "reference_captions": [
                "An artistic graphic illustration depicting pyramids and mountains in a desert landscape.",
                "A vector art illustration of the Egyptian pyramids with desert dunes.",
                "A stylized graphic artwork of pyramids under a warm sky.",
            ],
            "objects": [],
            "vqa": [
                {
                    "question": "Is this a real photograph or an illustration?",
                    "reference_answers": ["illustration", "graphic", "drawing", "artwork"]
                },
                {
                    "question": "What famous landmark is depicted in the illustration?",
                    "reference_answers": ["pyramids", "pyramid", "egyptian pyramids"]
                }
            ]
        },
        "portrait_smartphone.jpg": {
            "image_id": "eval_14_real_person_phone",
            "filename": "portrait_smartphone.jpg",
            "dimensions": {"width": 4608, "height": 3072},
            "scene_type": "photographic_scene",
            "person_presence": "present",
            "person_count": 2,
            "primary_activity": "using smartphone",
            "reference_captions": [
                "A person wearing a checkered shirt standing indoors holding and looking at a smartphone.",
                "An indoor photograph of a person in a plaid shirt interacting with a mobile phone.",
                "A young adult standing inside using a cell phone with another person in the background.",
            ],
            "objects": [
                {"label": "person", "bbox": [1280.0, 620.0, 3340.0, 2980.0]},
                {"label": "cell phone", "bbox": [1890.0, 1680.0, 2380.0, 2190.0]},
            ],
            "vqa": [
                {
                    "question": "What pattern or clothing is the main person wearing?",
                    "reference_answers": ["checkered shirt", "plaid", "plaid shirt", "checkered"]
                },
                {
                    "question": "What is the person holding?",
                    "reference_answers": ["phone", "cell phone", "smartphone"]
                }
            ]
        },
        "synthetic_graphic.jpg": {
            "image_id": "eval_15_synthetic_graphic",
            "filename": "synthetic_graphic.jpg",
            "dimensions": {"width": 640, "height": 480},
            "scene_type": "illustration_or_graphic",
            "person_presence": "absent",
            "person_count": 0,
            "primary_activity": "graphic test pattern",
            "reference_captions": [
                "A synthetic graphic illustration showing a geometric square with a blue border.",
                "A simple test image featuring a square shape with a blue outline on white background.",
                "A synthetic geometric graphic artifact.",
            ],
            "objects": [],
            "vqa": [
                {
                    "question": "Is there any human in this graphic?",
                    "reference_answers": ["no", "none", "nobody"]
                },
                {
                    "question": "What shape is shown?",
                    "reference_answers": ["square", "rectangle", "box"]
                }
            ]
        }
    }

    gt_file = DATASETS_DIR / "image_ground_truth.json"
    payload = {
        "version": "1.0",
        "description": "VisuNarrate Ground-Truth Labeled Dataset for Image Pipeline Evaluation",
        "total_images": len(annotations),
        "classes": ["person", "cell phone", "laptop", "book", "cup", "chair", "couch",
                    "potted plant", "backpack", "bottle", "dining table", "bowl", "sandwich",
                    "skateboard", "bicycle", "car", "motorcycle", "mouse"],
        "images": annotations,
    }
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✓ Image ground-truth dataset created with {len(annotations)} labeled images at: {gt_file}")


def build_video_ground_truth():
    """Builds ground-truth dataset for the test video(s)."""
    video_annotations = {
        "test_transition_activity.mp4": {
            "video_id": "eval_vid_transition_01",
            "filename": "test_transition_activity.mp4",
            "duration_seconds": 6.0,
            "fps": 24.0,
            "resolution": "1024x768",
            "ground_truth_transitions": [
                {
                    "timestamp_seconds": 3.4,
                    "from_activity": "holding / interacting with a cell phone",
                    "to_activity": "reading / holding a book",
                    "description": "Actor transitions from using smartphone to picking up and reading a book."
                }
            ],
            "ground_truth_transition_timestamps": [3.4],
            "ground_truth_segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 3.4,
                    "primary_activity": "holding / interacting with a cell phone",
                    "objects": ["person", "cell phone", "chair", "table"]
                },
                {
                    "start_seconds": 3.4,
                    "end_seconds": 6.0,
                    "primary_activity": "reading / holding a book",
                    "objects": ["person", "book", "chair", "table"]
                }
            ],
            "reference_narrative": "A person is initially sitting and interacting with a cell phone indoors. They then transition to reading a book, remaining seated in the same environment."
        }
    }

    gt_file = DATASETS_DIR / "video_ground_truth.json"
    payload = {
        "version": "1.0",
        "description": "VisuNarrate Ground-Truth Labeled Dataset for Video Pipeline Evaluation",
        "total_videos": len(video_annotations),
        "videos": video_annotations,
    }
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✓ Video ground-truth dataset created with {len(video_annotations)} labeled videos at: {gt_file}")


if __name__ == "__main__":
    build_image_ground_truth()
    build_video_ground_truth()
