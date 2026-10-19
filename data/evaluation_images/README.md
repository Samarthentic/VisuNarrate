# VisuNarrate - Image Evaluation Dataset Directory

This directory is designated for benchmark and evaluation images used by `scripts/evaluate_image_pipeline.py`.

## Purpose
To provide an academic, reproducible evaluation benchmark testing the full offline perception, fusion, storytelling, and predictive reasoning pipeline across diverse visual scenarios.

---

## Recommended Benchmark Categories

To ensure a comprehensive evaluation for the final-year B.E. major project, place real-world test images spanning the following categories:

| # | Category | Description & Evaluation Focus |
|---|:---|:---|
| 1 | **Single-Person Indoor Scene** | Tests indoor spatial grounding, human posture, and ambient context. |
| 2 | **Single-Person Outdoor Scene** | Tests natural lighting, background landscape extraction, and outdoor classification. |
| 3 | **Person Using Smartphone** | Tests handheld device interaction grounding, phone usage conflict gating, and screen-interaction affordances. |
| 4 | **Person Using Laptop** | Tests workstation spatial setup, typing affordances, and seated posture modeling. |
| 5 | **Person Reading (Book / Paper)** | Tests reading material interaction, page-turning affordances, and static posture. |
| 6 | **Person Interacting with Food / Drink** | Tests cup/bottle/utensil interaction, consumption affordances, and resting surfaces. |
| 7 | **Multiple People (Social Scene)** | Tests multi-agent instance detection, counting, and non-individual interaction synthesis. |
| 8 | **Natural Landscape (Agent Absent)** | Tests landscape composition, person-absence gating, and static equilibrium without predictive hallucination. |
| 9 | **Object-Focused Scene (Still Life)** | Tests non-agent physical entity detection and absence of phantom human properties. |
| 10 | **Ambiguous / Graphic Scene** | Tests heuristic media classification (`illustration_or_graphic`), artifact protection, and prediction suppression. |

---

## Manual Human Evaluation Rubric (Stage 4.7)

After running the evaluation script, human evaluators can grade generated outputs in `data/evaluation_results/summary.csv` using the standardized academic rating scale:

### 1. Scene, Object & Activity Correctness:
* `0` = **Incorrect** (Major factual error or hallucination).
* `1` = **Partially Correct** (Identified broad theme, but missed specific key elements).
* `2` = **Correct** (Accurately grounded in visual reality).

### 2. Narration Quality:
* `0` = **Poor** (Disjointed, grammatically broken, or misleading).
* `1` = **Acceptable** (Semantically clear, standard descriptive sentence).
* `2` = **Strong** (Coherent, natural phrasing with accurate contextual grounding).

### 3. Prediction Plausibility:
* `0` = **Implausible / Incorrect** (Speculative, physically impossible, or contradicting observed state).
* `1` = **Plausible but Uncertain** (Reasonable continuation, though alternative actions are equally likely).
* `2` = **Strongly Plausible** (Directly follows from affordances and validated actor interactions).

### 4. Hallucination Flag:
* `true` = Phantom entity, incorrect action, or non-existent object introduced into narration.
* `false` = Output strictly grounded in visible evidence.

---

## Evaluation Workflow

1. Place your `.jpg`, `.jpeg`, `.png`, or `.webp` test images in this directory (`data/evaluation_images/`).
2. Run the evaluation harness:
   ```bash
   TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_image_pipeline.py
   ```
3. Inspect detailed JSON and tabular CSV summaries generated in `data/evaluation_results/`:
   * `data/evaluation_results/summary.csv`
   * `data/evaluation_results/evaluation_results.json`
4. *(Optional)* Manually enter ratings (0/1/2) in `summary.csv` and re-run to view computed aggregate empirical metrics (Scene Accuracy, Narration Quality, Prediction Plausibility, Hallucination Rate).
