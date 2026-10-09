# AIDetect — Defactify External Benchmark Evaluation Report (v1)

## 1. Benchmark Overview & Certification

- **Benchmark Name:** Defactify External Benchmark v1
- **Dataset Source:** Defactify_Image_Dataset (Validation Split) (Commit: `787334f7857fa54f29027a7f09c30e895ad486ef`)
- **Total Image Count:** 800
- **Composition:** 400 REAL photographs + 400 AI generated images
- **Generator Breakdown:**
  - Natural Photographs (MS COCO, REAL): 400
  - Stable Diffusion 3 (SD3): 100
  - Midjourney v6: 100
  - DALL-E 3: 100
  - Stable Diffusion XL (SDXL): 100
- **Leakage Audit Status:** Certified zero leakage (min perceptual distance $d_H \ge 16$, 0 SHA-256 collisions).
- **Decision Threshold:** Calibrated $P(\text{AI}) \ge 0.50 \to \text{AI}$, $< 0.50 \to \text{REAL}$
- **V5-D Calibration:** Temperature Scaling ($T = 2.198387$)
- **V5-A, V5-B, V5-C Calibration:** Raw Softmax (Uncalibrated baseline)
- **Execution Timestamp:** `2026-10-09T16:58:05Z`
- **Execution Device:** `cuda` (Completed in 26.9s)

---

## 2. Overall Model Comparison (N=800)

| Model | Calibrated | Calibration Method | Accuracy | Precision | Recall (AI) | F1 Score | Balanced Acc | REAL FPR | AI FNR |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | No | Raw Softmax (Uncalibrated) | 67.38% | 64.04% | 79.25% | 70.84% | 67.38% | 44.50% | 20.75% |
| **V5-B Frequency** | No | Raw Softmax (Uncalibrated) | 55.25% | 65.67% | 22.00% | 32.96% | 55.25% | 11.50% | 78.00% |
| **V5-C Hybrid** | No | Raw Softmax (Uncalibrated) | 64.25% | 65.32% | 60.75% | 62.95% | 64.25% | 32.25% | 39.25% |
| **V5-D Gated Residual** | Yes | Temperature Scaling (T=2.198387) | 63.62% | 68.60% | 50.25% | 58.01% | 63.62% | 23.00% | 49.75% |

---

## 3. REAL Subset Performance & Over-Confidence Distribution (N=400 Natural Photos)

| Model | Correct REAL (TN) | False Positives (FP) | False Positive Rate | Mean P(AI) | Median P(AI) | % P(AI) >= 50% | % P(AI) >= 70% | % P(AI) >= 90% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 222/400 | 178/400 | 44.50% | 0.4486 | 0.2814 | 44.50% | 39.50% | 31.75% |
| **V5-B Frequency** | 354/400 | 46/400 | 11.50% | 0.1306 | 0.0028 | 11.50% | 9.50% | 7.50% |
| **V5-C Hybrid** | 271/400 | 129/400 | 32.25% | 0.3224 | 0.0167 | 32.25% | 27.00% | 23.75% |
| **V5-D Gated Residual** | 308/400 | 92/400 | 23.00% | 0.2521 | 0.0641 | 23.00% | 17.75% | 9.00% |

---

## 4. AI Subset Performance (N=400 AI Generated Images)

| Model | Correct AI (TP) | False Negatives (FN) | AI Recall | Mean P(AI) | Median P(AI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 317/400 | 83/400 | 79.25% | 0.7832 | 0.9969 |
| **V5-B Frequency** | 88/400 | 312/400 | 22.00% | 0.2328 | 0.0122 |
| **V5-C Hybrid** | 243/400 | 157/400 | 60.75% | 0.6025 | 0.9206 |
| **V5-D Gated Residual** | 201/400 | 199/400 | 50.25% | 0.5087 | 0.5173 |

---

## 5. Generator-Specific Performance Breakdown

| Model | Generator | N | Recall (AI Detection) | False Negative Rate | Mean P(AI) | Median P(AI) | Precision (vs REAL) | F1 (vs REAL) | Bal Acc (vs REAL) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | SD3 | 100 | 91.00% | 9.00% | 0.8905 | 0.9998 | 33.83% | 49.32% | 73.25% |
| **V5-A Spatial** | Midjourney v6 | 100 | 73.00% | 27.00% | 0.7176 | 0.9852 | 29.08% | 41.60% | 64.25% |
| **V5-A Spatial** | DALL-E 3 | 100 | 79.00% | 21.00% | 0.7927 | 0.9988 | 30.74% | 44.26% | 67.25% |
| **V5-A Spatial** | SDXL | 100 | 74.00% | 26.00% | 0.7318 | 0.9854 | 29.37% | 42.05% | 64.75% |
| **V5-B Frequency** | SD3 | 100 | 17.00% | 83.00% | 0.1570 | 0.0007 | 26.98% | 20.86% | 52.75% |
| **V5-B Frequency** | Midjourney v6 | 100 | 34.00% | 66.00% | 0.3828 | 0.2619 | 42.50% | 37.78% | 61.25% |
| **V5-B Frequency** | DALL-E 3 | 100 | 28.00% | 72.00% | 0.2819 | 0.0309 | 37.84% | 32.18% | 58.25% |
| **V5-B Frequency** | SDXL | 100 | 9.00% | 91.00% | 0.1094 | 0.0005 | 16.36% | 11.61% | 48.75% |
| **V5-C Hybrid** | SD3 | 100 | 62.00% | 38.00% | 0.6215 | 0.9305 | 32.46% | 42.61% | 64.88% |
| **V5-C Hybrid** | Midjourney v6 | 100 | 66.00% | 34.00% | 0.6582 | 0.9615 | 33.85% | 44.75% | 66.88% |
| **V5-C Hybrid** | DALL-E 3 | 100 | 65.00% | 35.00% | 0.6397 | 0.9739 | 33.51% | 44.22% | 66.38% |
| **V5-C Hybrid** | SDXL | 100 | 50.00% | 50.00% | 0.4905 | 0.5163 | 27.93% | 35.84% | 58.88% |
| **V5-D Gated Residual** | SD3 | 100 | 56.00% | 44.00% | 0.5720 | 0.6886 | 37.84% | 45.16% | 66.50% |
| **V5-D Gated Residual** | Midjourney v6 | 100 | 67.00% | 33.00% | 0.6343 | 0.8429 | 42.14% | 51.74% | 72.00% |
| **V5-D Gated Residual** | DALL-E 3 | 100 | 47.00% | 53.00% | 0.4780 | 0.3730 | 33.81% | 39.33% | 62.00% |
| **V5-D Gated Residual** | SDXL | 100 | 31.00% | 69.00% | 0.3506 | 0.1515 | 25.20% | 27.80% | 54.00% |

---

## 6. Confusion Matrices Summary

| Model | Subset | Total N | TN | FP | FN | TP | Accuracy | Recall / TPR | Specificity / TNR | FPR | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | SD3 vs REAL (N=500) | 500 | 222 | 178 | 9 | 91 | 62.60% | 91.00% | 55.50% | 44.50% | 9.00% |
| **V5-A Spatial** | Midjourney v6 vs REAL (N=500) | 500 | 222 | 178 | 27 | 73 | 59.00% | 73.00% | 55.50% | 44.50% | 27.00% |
| **V5-A Spatial** | DALL-E 3 vs REAL (N=500) | 500 | 222 | 178 | 21 | 79 | 60.20% | 79.00% | 55.50% | 44.50% | 21.00% |
| **V5-A Spatial** | SDXL vs REAL (N=500) | 500 | 222 | 178 | 26 | 74 | 59.20% | 74.00% | 55.50% | 44.50% | 26.00% |
| **V5-A Spatial** | Overall Benchmark (N=800) | 800 | 222 | 178 | 83 | 317 | 67.38% | 79.25% | 55.50% | 44.50% | 20.75% |
| **V5-A Spatial** | REAL Photographs Only (N=400) | 400 | 222 | 178 | 0 | 0 | 55.50% | 0.00% | 55.50% | 44.50% | 0.00% |
| **V5-A Spatial** | AI Subset Only (N=400) | 400 | 0 | 0 | 83 | 317 | 79.25% | 79.25% | 0.00% | 0.00% | 20.75% |
| **V5-B Frequency** | SD3 vs REAL (N=500) | 500 | 354 | 46 | 83 | 17 | 74.20% | 17.00% | 88.50% | 11.50% | 83.00% |
| **V5-B Frequency** | Midjourney v6 vs REAL (N=500) | 500 | 354 | 46 | 66 | 34 | 77.60% | 34.00% | 88.50% | 11.50% | 66.00% |
| **V5-B Frequency** | DALL-E 3 vs REAL (N=500) | 500 | 354 | 46 | 72 | 28 | 76.40% | 28.00% | 88.50% | 11.50% | 72.00% |
| **V5-B Frequency** | SDXL vs REAL (N=500) | 500 | 354 | 46 | 91 | 9 | 72.60% | 9.00% | 88.50% | 11.50% | 91.00% |
| **V5-B Frequency** | Overall Benchmark (N=800) | 800 | 354 | 46 | 312 | 88 | 55.25% | 22.00% | 88.50% | 11.50% | 78.00% |
| **V5-B Frequency** | REAL Photographs Only (N=400) | 400 | 354 | 46 | 0 | 0 | 88.50% | 0.00% | 88.50% | 11.50% | 0.00% |
| **V5-B Frequency** | AI Subset Only (N=400) | 400 | 0 | 0 | 312 | 88 | 22.00% | 22.00% | 0.00% | 0.00% | 78.00% |
| **V5-C Hybrid** | SD3 vs REAL (N=500) | 500 | 271 | 129 | 38 | 62 | 66.60% | 62.00% | 67.75% | 32.25% | 38.00% |
| **V5-C Hybrid** | Midjourney v6 vs REAL (N=500) | 500 | 271 | 129 | 34 | 66 | 67.40% | 66.00% | 67.75% | 32.25% | 34.00% |
| **V5-C Hybrid** | DALL-E 3 vs REAL (N=500) | 500 | 271 | 129 | 35 | 65 | 67.20% | 65.00% | 67.75% | 32.25% | 35.00% |
| **V5-C Hybrid** | SDXL vs REAL (N=500) | 500 | 271 | 129 | 50 | 50 | 64.20% | 50.00% | 67.75% | 32.25% | 50.00% |
| **V5-C Hybrid** | Overall Benchmark (N=800) | 800 | 271 | 129 | 157 | 243 | 64.25% | 60.75% | 67.75% | 32.25% | 39.25% |
| **V5-C Hybrid** | REAL Photographs Only (N=400) | 400 | 271 | 129 | 0 | 0 | 67.75% | 0.00% | 67.75% | 32.25% | 0.00% |
| **V5-C Hybrid** | AI Subset Only (N=400) | 400 | 0 | 0 | 157 | 243 | 60.75% | 60.75% | 0.00% | 0.00% | 39.25% |
| **V5-D Gated Residual** | SD3 vs REAL (N=500) | 500 | 308 | 92 | 44 | 56 | 72.80% | 56.00% | 77.00% | 23.00% | 44.00% |
| **V5-D Gated Residual** | Midjourney v6 vs REAL (N=500) | 500 | 308 | 92 | 33 | 67 | 75.00% | 67.00% | 77.00% | 23.00% | 33.00% |
| **V5-D Gated Residual** | DALL-E 3 vs REAL (N=500) | 500 | 308 | 92 | 53 | 47 | 71.00% | 47.00% | 77.00% | 23.00% | 53.00% |
| **V5-D Gated Residual** | SDXL vs REAL (N=500) | 500 | 308 | 92 | 69 | 31 | 67.80% | 31.00% | 77.00% | 23.00% | 69.00% |
| **V5-D Gated Residual** | Overall Benchmark (N=800) | 800 | 308 | 92 | 199 | 201 | 63.62% | 50.25% | 77.00% | 23.00% | 49.75% |
| **V5-D Gated Residual** | REAL Photographs Only (N=400) | 400 | 308 | 92 | 0 | 0 | 77.00% | 0.00% | 77.00% | 23.00% | 0.00% |
| **V5-D Gated Residual** | AI Subset Only (N=400) | 400 | 0 | 0 | 199 | 201 | 50.25% | 50.25% | 0.00% | 0.00% | 49.75% |

---

## 7. Integrity & Invariants Checklist

- [x] Model weights frozen and unaltered.
- [x] Frozen 9,000-image V5 test set untouched.
- [x] Deterministic evaluation with zero data augmentation.
- [x] Temperature calibration parameter $T = 2.198387$ loaded from authoritative configuration.
- [x] Structural pre-validation confirmed 800 images across exact categories prior to inference.
- [x] Output artifacts fully persisted without subjective interpretations.
