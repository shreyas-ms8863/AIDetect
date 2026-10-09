# AIDetect — Frozen Defactify External Benchmark Error Analysis

## Executive Summary

This report provides a strict post-inference factual error and agreement analysis of the **Defactify External Benchmark (v1)**.
The benchmark evaluates four frozen V5 models across **800 out-of-distribution images** (400 natural REAL photographs from MS COCO and 100 images each from SD3, Midjourney v6, DALL-E 3, and SDXL).

- **Total Images Analyzed:** 800
- **Total Model Predictions Analyzed:** 3,200
- **Model Roster:** V5-A Spatial, V5-B Frequency, V5-C Hybrid, V5-D Gated Residual (Calibrated $T=2.198387$)
- **Model Checkpoint Status:** Frozen and unaltered
- **V5 Test Set Access:** Zero access (frozen 9,000-image test set untouched)

---

## 1. Model Agreement & Ensemble Divergence

Cross-model agreement on the external benchmark demonstrates significant prediction divergence across architectures:

| Agreement Level | Image Count | Percentage of Benchmark |
| :--- | :---: | :---: |
| **All 4 Models Agree (Unanimous)** | 298 | 37.25% |
| **3 of 4 Models Agree (1 Dissenter)** | 368 | 46.0% |
| **2 vs 2 Split Decision (Maximal Disagreement)** | 134 | 16.75% |

### Pairwise Model Agreement Matrix

| Model | V5-A Spatial | V5-B Frequency | V5-C Hybrid | V5-D Gated Residual |
| :--- | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 800 (100.0%) | 377 (47.1%) | 585 (73.1%) | 554 (69.2%) |
| **V5-B Frequency** | 377 (47.1%) | 800 (100.0%) | 478 (59.8%) | 537 (67.1%) |
| **V5-C Hybrid** | 585 (73.1%) | 478 (59.8%) | 800 (100.0%) | 629 (78.6%) |
| **V5-D Gated Residual** | 554 (69.2%) | 537 (67.1%) | 629 (78.6%) | 800 (100.0%) |

### Key Disagreement Counts

- **V5-D Correct while V5-A is Wrong:** 108 images (98 REAL photos where V5-D suppressed V5-A's false alarm, 10 AI images where V5-D detected what V5-A missed).
- **V5-A Correct while V5-D is Wrong:** 138 images (126 AI images where V5-A detected what V5-D missed, 12 REAL photos where V5-A avoided V5-D's false alarm).
- **V5-C Correct while both V5-A and V5-D are Wrong:** 25 images.
- **V5-B Correct while V5-A, V5-C, and V5-D are all Wrong:** 56 images (56 natural REAL photographs where all spatial-bearing models produced false alarms, but frequency alone stayed correct).

---

## 2. REAL Image Error Analysis (The False Alarm Problem)

Evaluation on the 400 natural MS COCO photographs reveals substantial divergence in false alarm rates across architectures:

| Model | Correct REAL (TN) | False Positives (FP) | FPR | Mean P(AI) | Median P(AI) | % P(AI) >= 50% | % P(AI) >= 70% | % P(AI) >= 90% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 222/400 | 178/400 | 44.50% | 0.4486 | 0.2814 | 44.50% | 39.50% | 31.75% |
| **V5-B Frequency** | 354/400 | 46/400 | 11.50% | 0.1306 | 0.0028 | 11.50% | 9.50% | 7.50% |
| **V5-C Hybrid** | 271/400 | 129/400 | 32.25% | 0.3224 | 0.0167 | 32.25% | 27.00% | 23.75% |
| **V5-D Gated Residual** | 308/400 | 92/400 | 23.00% | 0.2521 | 0.0641 | 23.00% | 17.75% | 9.00% |

### REAL Image Error Distribution by Model Agreement

| Error Group | Count | Percentage of REAL (N=400) | Description |
| :--- | :---: | :---: | :--- |
| **All 4 Models Correct (No Error)** | 171 | 42.75% | Breakdown across 400 REAL photos |
| **V5-A Spatial Only Wrong** | 60 | 15.00% | Breakdown across 400 REAL photos |
| **V5-B Frequency Only Wrong** | 15 | 3.75% | Breakdown across 400 REAL photos |
| **V5-C Hybrid Only Wrong** | 22 | 5.50% | Breakdown across 400 REAL photos |
| **V5-D Gated Residual Only Wrong** | 4 | 1.00% | Breakdown across 400 REAL photos |
| **Both V5-A and V5-D Wrong (Total)** | 80 | 20.00% | Breakdown across 400 REAL photos |
| **Exclusively V5-A and V5-D Wrong (B & C Correct)** | 12 | 3.00% | Breakdown across 400 REAL photos |
| **Multiple-Model Errors (>=2 Models Wrong)** | 128 | 32.00% | Breakdown across 400 REAL photos |

---

## 3. AI Generator Error Analysis

### Generator × Model AI Detection Recall Matrix

| Model | Stable Diffusion 3 | Midjourney v6 | DALL-E 3 | Stable Diffusion XL |
| :--- | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 91.00% | 73.00% | 79.00% | 74.00% |
| **V5-B Frequency** | 17.00% | 34.00% | 28.00% | 9.00% |
| **V5-C Hybrid** | 62.00% | 66.00% | 65.00% | 50.00% |
| **V5-D Gated Residual** | 56.00% | 67.00% | 47.00% | 31.00% |

### Generator × Model False Negative Rate (FNR) Matrix

| Model | Stable Diffusion 3 | Midjourney v6 | DALL-E 3 | Stable Diffusion XL |
| :--- | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 9.00% | 27.00% | 21.00% | 26.00% |
| **V5-B Frequency** | 83.00% | 66.00% | 72.00% | 91.00% |
| **V5-C Hybrid** | 38.00% | 34.00% | 35.00% | 50.00% |
| **V5-D Gated Residual** | 44.00% | 33.00% | 53.00% | 69.00% |

### Generator Vulnerability Ranking per Model

| Model | Easiest Generator (Highest Recall) | Hardest Generator (Lowest Recall) |
| :--- | :--- | :--- |
| **V5-A Spatial** | SD3 (91.00%) | Midjourney v6 (73.00%) |
| **V5-B Frequency** | Midjourney v6 (34.00%) | SDXL (9.00%) |
| **V5-C Hybrid** | Midjourney v6 (66.00%) | SDXL (50.00%) |
| **V5-D Gated Residual** | Midjourney v6 (67.00%) | SDXL (31.00%) |

### Direct AI Error Trade-offs Between V5-A and V5-D

- **V5-D Fails while V5-A Succeeds:** 126 AI images total:
  - SDXL: 44 images
  - SD3: 36 images
  - DALL-E 3: 33 images
  - Midjourney v6: 13 images
- **V5-A Fails while V5-D Succeeds:** 10 AI images total:
  - Midjourney v6: 7 images
  - SD3: 1 images
  - DALL-E 3: 1 images
  - SDXL: 1 images

---

## 4. Confidence & Error Analysis

| Model | Correct Mean Conf | Correct Median Conf | Incorrect Mean Conf | Incorrect Median Conf | High-Conf Errors (>=90%) | High-Conf Errors (>=70%) | Low-Conf Correct (<60%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 94.54% | 99.83% | 90.70% | 98.21% | 184 | 229 | 17 |
| **V5-B Frequency** | 94.55% | 99.70% | 93.59% | 99.60% | 283 | 335 | 13 |
| **V5-C Hybrid** | 95.15% | 99.87% | 91.97% | 99.44% | 214 | 252 | 11 |
| **V5-D Gated Residual** | 90.53% | 95.54% | 85.61% | 91.76% | 157 | 239 | 19 |

### Severity of Erroneous Predictions

| Model | REAL FP >=90% | REAL FP 70–89.99% | REAL FP 50–69.99% | Total REAL FP | AI FN <10% | AI FN 10–29.99% | AI FN 30–49.99% | Total AI FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 127 | 31 | 20 | 178 | 57 | 14 | 12 | 83 |
| **V5-B Frequency** | 30 | 8 | 8 | 46 | 253 | 44 | 15 | 312 |
| **V5-C Hybrid** | 95 | 13 | 21 | 129 | 119 | 25 | 13 | 157 |
| **V5-D Gated Residual** | 36 | 35 | 21 | 92 | 121 | 47 | 31 | 199 |

---

## 5. Longitudinal Comparison Across Evaluation Regimes

Comparison of model F1 scores across (1) Clean V5 Test, (2) Average Controlled Robustness, and (3) Defactify External Benchmark:

| Model | Clean V5 Test F1 | Avg Controlled Robustness F1 | External Benchmark F1 | Clean → External F1 Degradation | Robustness → External F1 Delta | Clean Rank | Robustness Rank | External Rank | Rank Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 97.02% | 76.49% | 70.84% | **-26.18 pp** | -5.65 pp | #2 | #1 | #1 | **+1** |
| **V5-B Frequency** | 90.65% | 63.38% | 32.96% | **-57.69 pp** | -30.42 pp | #4 | #4 | #4 | **0** |
| **V5-C Hybrid** | 96.89% | 71.79% | 62.95% | **-33.94 pp** | -8.84 pp | #3 | #3 | #2 | **+1** |
| **V5-D Gated Residual** | 97.17% | 73.44% | 58.01% | **-39.16 pp** | -15.43 pp | #1 | #2 | #3 | **-2** |

---

## 6. Research Findings & Answers to Evaluation Questions

### 1. Does clean-test performance predict external performance?
**Finding: No.** Clean-test F1 across the four models was tightly clustered between 90.65% and 97.17% (a spread of only 6.52 percentage points, with V5-D and V5-A separated by just 0.15 pp). On the external benchmark, performance dispersed widely from 32.96% to 70.84% (a spread of 37.88 pp), and the ranking inverted. Clean-test accuracy was uninformative regarding out-of-distribution generalization.

### 2. Does V5-D's clean-test advantage transfer externally?
**Finding: No.** V5-D led on clean test (97.17% F1), but dropped to third place on the external benchmark (58.01% F1), falling 12.83 percentage points behind V5-A Spatial (70.84%) and 4.94 percentage points behind V5-C Hybrid (62.95%). Its clean-test lead did not translate into superior external generalization.

### 3. Does V5-A show more consistent generalization?
**Finding: Yes, in overall F1 and AI recall, but with high false positive rates.** V5-A suffered the smallest degradation from clean test (-26.18 pp vs -39.16 pp for V5-D and -57.69 pp for V5-B) and from controlled robustness (-5.65 pp vs -15.43 pp for V5-D). It attained the highest external F1 (70.84%) and highest AI recall (79.25%). However, V5-A exhibited a critical false alarm problem on natural photographs, misclassifying 178 of 400 REAL images as AI (44.50% FPR).

### 4. Does frequency-only remain the weakest approach?
**Finding: Yes.** V5-B collapsed under distribution shift, achieving an external F1 of only 32.96% (a degradation of -57.69 pp from clean test). V5-B detected only 22.00% of AI images overall (including only 9.00% of SDXL images and 17.00% of SD3 images). Frequency-only analysis is severely brittle when evaluated on external generator pipelines.

### 5. Does spatial-frequency fusion provide a consistent advantage?
**Finding: Mixed.** Dual-stream fusion in V5-C Hybrid (62.95% F1) improved upon V5-D Gated Residual (58.01%) and V5-B (32.96%), but trailed spatial-only V5-A (70.84%). In V5-D, learned gating successfully reduced false alarms on REAL photographs (FPR dropped from 44.50% in V5-A to 23.00% in V5-D), but at the substantial cost of failing to detect modern AI generators (FNR increased from 20.75% in V5-A to 49.75% in V5-D).

### 6. Are errors generator-dependent?
**Finding: Yes.** Generator recall varied substantially across architectures:
- **SDXL** was the most difficult generator across models (31.00% recall for V5-D, 9.00% for V5-B, 50.00% for V5-C).
- **Midjourney v6** was the easiest generator for V5-B (34.00%), V5-C (66.00%), and V5-D (67.00%), but the hardest for V5-A (73.00%).
- **Stable Diffusion 3** exhibited high recall on V5-A (91.00%), but low recall on V5-B (17.00%) and moderate recall on V5-D (56.00%).

### 7. Is the main external weakness false positives on REAL images, false negatives on AI images, or both?
**Finding: Both, with asymmetric error profiles across models:**
- **V5-A Spatial:** Primary failure mode is **False Positives on REAL photographs** (FPR = 44.50%, 178 false alarms vs 83 missed AI).
- **V5-B Frequency:** Primary failure mode is **False Negatives on AI generators** (FNR = 78.00%, 312 missed AI vs 46 false alarms).
- **V5-D Gated Residual:** Exhibits **balanced but elevated errors in both directions** (FPR = 23.00%, 92 false alarms; FNR = 49.75%, 199 missed AI).

### 8. What evidence supports domain/generalization limitations?
**Finding: Strong evidence of severe distribution shift:**
1. **High-confidence erroneous predictions:** V5-A produced 184 errors with confidence $\ge 90\%$, including assigning $\ge 90\%$ AI probability to 127 natural COCO photographs (31.75% of all REAL samples).
2. **Unanimous failure modes:** On 56 natural REAL photographs, all three spatial-bearing models (A, C, D) produced false alarms simultaneously.
3. **Ensemble divergence:** All 4 models agreed on their prediction in only 37.25% of images, showing that out-of-distribution inputs drive models into divergent feature regimes.

---

## 7. Artifact Manifest

The following output artifacts are preserved under `diagnostic_outputs/external_benchmark_v1_analysis/`:
- `model_agreement.csv`: Complete agreement breakdown and pairwise agreement matrix.
- `real_error_analysis.csv`: REAL image error groupings and per-model false positive metrics.
- `generator_model_matrix.csv`: Detailed generator breakdown and recall/FNR matrices.
- `confidence_error_analysis.csv`: Confidence distributions, error severity, and threshold buckets.
- `clean_vs_external_comparison.csv`: Longitudinal comparison table across Clean Test, Robustness, and External Benchmark.
- `A_correct_D_wrong.csv`: 138 disagreement cases where V5-A succeeded and V5-D failed.
- `D_correct_A_wrong.csv`: 108 disagreement cases where V5-D succeeded and V5-A failed.
- `C_correct_A_D_wrong.csv`: 25 disagreement cases where V5-C succeeded while both A and D failed.
- `B_correct_A_C_D_wrong.csv`: 56 disagreement cases where V5-B succeeded while all other models failed.
- `external_analysis_summary.json`: Complete hierarchical data file containing all numerical results.
- Visual figures: `model_f1_comparison.png`, `generator_recall_comparison.png`, `real_fpr_comparison.png`, `confidence_distributions.png`, `clean_vs_robustness_vs_external_f1.png`.
