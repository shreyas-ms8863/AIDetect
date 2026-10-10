# AIDetect: Comprehensive Empirical Strengthening & Cross-Domain Forensic Evaluation Report

**Document Identifier:** `AIDETECT-REPORT-STRENGTHENED-EVAL-V5`
**Execution Date:** October 10, 2026
**Subject:** Multi-Model Deep Evaluation Suite (V5-A Spatial, V5-B Frequency, V5-C Hybrid, V5-D Gated Residual)
**Evaluation Scope:** Clean Test ($N=9,000$), Controlled Perturbations ($12\text{ conditions} \times 4\text{ models}$), Defactify External Benchmark ($N=800$), Source Subgroups ($N=8,356$ CIFAKE vs $N=644$ Non-CIFAKE)
**Integrity Protocol:** Zero Retraining · Zero Checkpoint Modification · Frozen Test Preservation · Bit-for-Bit Deterministic Verification

---

## 1. Executive Summary & Research Program Context

The AIDetect project evaluated four forensic model variants for synthetic-image detection:
- **V5-A Spatial:** Standard ResNet-50 evaluating spatial RGB domain features ($224 \times 224$).
- **V5-B Frequency:** ResNet-50 evaluating authoritative native-resolution 2D FFT log-magnitude spectral features.
- **V5-C Hybrid:** Dual-stream ResNet-50 fusing spatial and frequency feature vectors (4096-d).
- **V5-D Gated Residual:** Triple-stream ResNet-50 fusing spatial, frequency, and 5x5 Gaussian noise residual features via a learned nonlinear gating network (6144-d) with post-hoc Temperature Scaling ($T = 2.198387$).

This report consolidates clean-test, controlled-perturbation, and external-benchmark analyses. These evaluations use fixed model checkpoints; they do not establish performance on arbitrary image sources or deployment populations.

This analysis adds or consolidates the following evidence without retraining models or changing checkpoints:
1. **Full Continuous Probability Export & Bit-for-Bit Reproduction:** Re-executed clean inference on the frozen 9,000-image test set, verifying **exact mathematical parity** across all contingency matrix cells ($TP, TN, FP, FN$) for all four models.
2. **Threshold-Free Discrimination Metrics:** Computed AUROC, AUPRC, EER, and TPR at fixed low False Positive Rates ($1\%, 5\%, 10\%$ FPR).
3. **1,000-Resample Image-Level Bootstrap:** Derived percentile 95% confidence intervals and standard errors for the listed classification and discrimination metrics.
4. **Paired Bootstrap Model Differences:** Computed paired $\Delta\text{F1}$, $\Delta\text{AUROC}$, $\Delta\text{MCC}$, and empirical two-sided $p$-values across all model pairs.
5. **Robustness Degradation & Retention:** Quantified resilience across 12 stress conditions (Resize, Gaussian Blur, Gaussian Noise, JPEG Re-encoding).
6. **External Defactify Benchmark:** Analyzed performance across 400 MS COCO photographs and 400 AI images labeled SD3, Midjourney v6, DALL-E 3, and SDXL.
7. **Source Subgroup Decomposition:** Reports CIFAKE ($N=8,356$) and a mixed non-CIFAKE in-distribution subset ($N=644$; 292 REAL, 352 AI).
8. **Forensic Audit & pHash Clarification:** Reports the historical comparison threshold ($d_H \le 3$) and the observed accepted-set minimum ($\min d_H = 10$).

---

## 2. Clean-Test Benchmark Evaluation ($N=9,000$)

### 2.1 Extended Point Estimates & Discrimination Capacity
The frozen V5 test set contains 9,000 balanced images (4,500 REAL, 4,500 AI). Operating at the decision threshold $\tau = 0.50$:

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 Score | MCC | AUROC | AUPRC | EER | TPR @ 1% FPR | TPR @ 5% FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 97.02% | 97.02% | 97.17% | 96.87% | 97.02% | 0.9404 | 0.9945 | 0.9949 | 2.98% | 92.58% | 97.91% |
| **V5-B Frequency** | 90.66% | 90.66% | 90.68% | 90.62% | 90.65% | 0.8131 | 0.9679 | 0.9672 | 9.36% | 60.00% | 84.76% |
| **V5-C Hybrid** | 96.88% | 96.88% | 96.64% | 97.13% | 96.89% | 0.9376 | 0.9956 | 0.9954 | 3.09% | 92.36% | 98.31% |
| **V5-D Gated Residual** | **97.18%** | **97.18%** | **97.30%** | **97.04%** | **97.17%** | **0.9436** | **0.9960** | **0.9960** | **2.80%** | **93.18%** | **98.22%** |

### 2.2 Image-Level Bootstrap 95% Confidence Intervals
Intervals use an ordinary image-level bootstrap with replacement ($N=9,000$, $B=1,000$, seed $=42$) and empirical 2.5th/97.5th percentiles. The implementation does not stratify resamples by class. Paired model comparisons use the same resample indices across models.

| Model | Metric | Point Estimate | 95% Bootstrap CI | Std. Error |
| :--- | :--- | :---: | :---: | :---: |
| **V5-A Spatial** | F1 Score | 0.970176 | [0.966514, 0.973912] | 0.001854 |
| | AUROC | 0.994549 | [0.993285, 0.995589] | 0.000608 |
| | MCC | 0.940449 | [0.933555, 0.947568] | 0.003643 |
| **V5-B Frequency** | F1 Score | 0.906524 | [0.900056, 0.912564] | 0.003192 |
| | AUROC | 0.967931 | [0.964617, 0.971027] | 0.001599 |
| | MCC | 0.813111 | [0.800887, 0.824863] | 0.006073 |
| **V5-C Hybrid** | F1 Score | 0.968857 | [0.965260, 0.972605] | 0.001894 |
| | AUROC | 0.995603 | [0.994672, 0.996451] | 0.000465 |
| | MCC | 0.937568 | [0.930654, 0.944895] | 0.003714 |
| **V5-D Gated Residual** | F1 Score | **0.971740** | **[0.968022, 0.975189]** | **0.001822** |
| | AUROC | **0.995962** | **[0.995055, 0.996750]** | **0.000434** |
| | MCC | **0.943559** | **[0.936010, 0.950446]** | **0.003579** |

### 2.3 Paired Bootstrap Comparisons
The table shows selected paired bootstrap differences using identical image resample indices across models. Empirical two-sided bootstrap-tail p-values are approximate and have 1,000-resample resolution. Holm correction was applied only to the six pairwise clean-test AUROC comparisons; other metric-family comparisons are exploratory and unadjusted.

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                            CLEAN-TEST PAIRED STATISTICAL DIFFERENCES                                          |
+------------------+---------+-------------------+--------------------+------------+-------------+--------------------------+
| Model Comparison | Metric  | Δ (Model 1 - 2)   | 95% Confidence Int | Std. Error | Approx. unadj. p | Interpretation |
+------------------+---------+-------------------+--------------------+------------+-------------+--------------------------+
| V5-A vs V5-D     | F1      | -0.001564         | [-0.005814, +0.002703] | 0.002194   | 0.480            | Exploratory; no significant difference |
| V5-A vs V5-D     | AUROC   | -0.001413         | [-0.002712, -0.000152] | 0.000653   | 0.024            | Nominal; not significant after Holm |
| V5-A vs V5-D     | MCC     | -0.003110         | [-0.011548, +0.005341] | 0.004366   | 0.470            | Exploratory; no significant difference |
| V5-C vs V5-D     | F1      | -0.002883         | [-0.006444, +0.000603] | 0.001763   | 0.110            | Exploratory; no significant difference |
| V5-C vs V5-D     | AUROC   | -0.000358         | [-0.001149, +0.000522] | 0.000425   | 0.430            | Exploratory; no significant difference |
| V5-A vs V5-B     | F1      | +0.063651         | [+0.056727, +0.070699] | 0.003478   | 0.001            | Exploratory; unadjusted |
| V5-D vs V5-B     | F1      | +0.065216         | [+0.059242, +0.071127] | 0.003015   | 0.001            | Exploratory; unadjusted |
+------------------+---------+-------------------+--------------------+------------+-------------+--------------------------+
```

**Key Takeaway:** The clean-test F1 difference between V5-D and V5-A is not statistically significant ($\Delta\text{F1}_{D-A} = +0.001564$, 95% CI approximately $[-0.002703, +0.005814]$, approximate unadjusted $p=0.480$). V5-D's nominal AUROC difference from V5-A is $\Delta\text{AUROC}_{D-A}=+0.001413$ (95% CI $[+0.000152,+0.002712]$, approximate unadjusted $p=0.024$), but it is not significant after Holm correction across the six pairwise clean-test AUROC comparisons. Holm correction was applied to those six AUROC comparisons only. Other paired metric comparisons shown here are exploratory and unadjusted.

---

## 3. Controlled Perturbation Robustness Analysis ($12\text{ Conditions} \times 4\text{ Models}$)

Robustness was evaluated across 4 perturbation families (48 experimental cells):
- **Spatial Resizing:** 75%, 50%, 25% bilinear downscaling.
- **Gaussian Blur:** Radius 1.0, 2.0, 3.0.
- **Additive Gaussian Noise:** $\sigma = 0.01, 0.05, 0.10$.
- **JPEG Re-encoding:** Quality factors $Q = 95, 75, 50$.

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        EXTENDED ROBUSTNESS SUMMARY MATRIX                                             |
+--------------------+------------+--------------------------+--------------------------+-------------------------------+
| Perturbation       | Parameter  | V5-A Spatial             | V5-C Hybrid              | V5-D Gated Residual           |
| Family             | Level      | Ret%  | BalAcc | MCC     | Ret%  | BalAcc | MCC     | Ret%  | BalAcc | MCC          |
+--------------------+------------+-------+--------+---------+-------+--------+---------+-------+--------+--------------+
| JPEG Re-encoding   | Q = 95     | 100.0%| 0.9704 | 0.9409  | 98.8% | 0.9578 | 0.9159  | 98.2% | 0.9548 | 0.9103       |
| JPEG Re-encoding   | Q = 75     | 99.8% | 0.9686 | 0.9371  | 98.5% | 0.9552 | 0.9107  | 98.2% | 0.9549 | 0.9104       |
| JPEG Re-encoding   | Q = 50     | 97.5% | 0.9463 | 0.8927  | 96.5% | 0.9357 | 0.8714  | 96.3% | 0.9361 | 0.8724       |
| Gaussian Noise     | σ = 0.01   | 98.7% | 0.9581 | 0.9169  | 98.8% | 0.9576 | 0.9151  | 98.7% | 0.9582 | 0.9167       |
| Gaussian Noise     | σ = 0.05   | 90.0% | 0.8657 | 0.7370  | 87.8% | 0.8564 | 0.7150  | 85.9% | 0.8173 | 0.6490       |
| Gaussian Noise     | σ = 0.10   | 69.7% | 0.5408 | 0.1485  | 75.4% | 0.7236 | 0.4478  | 74.6% | 0.6628 | 0.3645       |
| Downsampling       | Scale 75%  | 55.2% | 0.6634 | 0.3913  | 49.0% | 0.6399 | 0.3598  | 51.6% | 0.6532 | 0.3866       |
| Downsampling       | Scale 50%  | 67.8% | 0.5864 | 0.1899  | 49.6% | 0.6117 | 0.2587  | 59.1% | 0.6444 | 0.3061       |
| Downsampling       | Scale 25%  | 68.8% | 0.5127 | 0.0706  | 67.6% | 0.5930 | 0.1991  | 71.1% | 0.5807 | 0.2295       |
| Gaussian Blur      | Radius 1.0 | 61.3% | 0.6099 | 0.2204  | 28.8% | 0.5497 | 0.1505  | 32.0% | 0.5627 | 0.1833       |
| Gaussian Blur      | Radius 2.0 | 68.8% | 0.5094 | 0.0597  | 69.1% | 0.5664 | 0.1699  | 71.6% | 0.5837 | 0.2471       |
| Gaussian Blur      | Radius 3.0 | 68.4% | 0.5028 | 0.0188  | 69.2% | 0.5292 | 0.1146  | 69.8% | 0.5309 | 0.1552       |
+--------------------+------------+-------+--------+---------+-------+--------+---------+-------+--------+--------------+
```

### Critical Robustness Insights (Controlled Transformation Robustness)
1. **JPEG Re-encoding Stability:** V5-A Spatial retains the highest stability under lossy JPEG re-compression ($97.52\%$ retention at $Q=50$, MCC $= 0.8927$). Multi-stream models show moderate sensitivity to high-frequency quantization artifacts introduced by lossy compression.
2. **Controlled Additive Noise ($\sigma = 0.10$):** Under severe Gaussian noise, V5-A exhibits substantial performance degradation ($\text{MCC} = 0.1485$). In contrast, V5-C Hybrid ($\text{MCC} = 0.4478$) and V5-D Gated Residual ($\text{MCC} = 0.3645$) demonstrate greater retention under controlled noise conditions.
3. **Severe Downsampling (25% Scale):** V5-A has $\text{MCC} = 0.0706$ and V5-D has $\text{MCC} = 0.2295$ under this controlled transformation. These experiments do not measure deployment performance under real-world image degradation.

---

## 4. Defactify External Benchmark Evaluation ($N=800$)

The external benchmark comprises 800 selected Defactify images: 400 REAL MS COCO photographs and 400 AI images (100 each labeled SD3, Midjourney v6, DALL-E 3, and SDXL). The results apply to these samples and this benchmark composition.

### 4.1 Extended Metrics & Discriminative Capacity

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 Score | MCC | AUROC | AUPRC | EER | Real FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 67.38% | 67.38% | 64.04% | 79.25% | 70.84% | 0.3577 | 0.7459 | 0.7368 | 31.25% | 44.50% |
| **V5-B Frequency** | 55.25% | 55.25% | 65.67% | 22.00% | 32.96% | 0.1406 | 0.5573 | 0.5816 | 44.25% | 11.50% |
| **V5-C Hybrid** | 64.25% | 64.25% | 65.32% | 60.75% | 62.95% | 0.2857 | 0.6859 | 0.6632 | 35.75% | 32.25% |
| **V5-D Gated Residual** | 63.63% | 63.63% | 68.60% | 50.25% | 58.01% | 0.2828 | 0.7036 | 0.7122 | 36.00% | **23.00%** |

### 4.2 High-Specificity Operating Points (TPR at Fixed FPR)
For settings where false accusations carry high cost, the low-FPR part of the interpolated ROC curve is informative:

| Model | TPR @ 1% FPR | TPR @ 5% FPR | TPR @ 10% FPR |
| :--- | :---: | :---: | :---: |
| **V5-A Spatial** | 6.22% | 22.75% | **43.75%** |
| **V5-B Frequency** | 3.50% | 9.00% | 20.25% |
| **V5-C Hybrid** | 2.50% | 16.00% | 29.75% |
| **V5-D Gated Residual** | **10.25%** | **23.00%** | 33.75% |

At the interpolated 1% FPR point, V5-D has the highest reported TPR point estimate (10.25% vs 6.22% for V5-A). This is a descriptive ROC interpolation, not necessarily an attainable discrete threshold, and no paired uncertainty interval is reported for this operating point.

### 4.3 Generator Generalization & Cross-Architecture Disparity

| Model | SD3 (N=100) | Midjourney v6 (N=100) | DALL-E 3 (N=100) | SDXL (N=100) | Macro Recall | Lowest Recall | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-A Spatial** | 91.0% | 73.0% | 79.0% | 74.0% | 79.25% | 73.0% (Midjourney v6) | 0.0826 |
| **V5-B Frequency** | 17.0% | 34.0% | 28.0% | 9.0% | 22.00% | 9.0% (SDXL) | 0.1117 |
| **V5-C Hybrid** | 62.0% | 66.0% | 65.0% | 50.0% | 60.75% | 50.0% (SDXL) | 0.0737 |
| **V5-D Gated Residual** | 56.0% | 67.0% | 47.0% | 31.0% | 50.25% | 31.0% (SDXL) | 0.1522 |

Generator-level performance varies substantially across models and generator families, indicating distribution-dependent generalization. In this sample, the lowest recall is Midjourney v6 for V5-A (73%), and SDXL for V5-B (9%), V5-C (50%), and V5-D (31%). Since each generator contributes 100 samples, the four-generator mean matches the corresponding overall AI recall. These are empirical results for the selected benchmark samples; they do not identify a causal mechanism or establish broad transfer to other generators.

---

## 5. Calibration & Selective Prediction Analysis (V5-D)

### 5.1 In-Distribution vs Out-of-Distribution Calibration Drift
V5-D incorporates post-hoc Temperature Scaling ($T = 2.198387$) fitted on in-distribution validation logits:

| Metric | In-Distribution Validation | External Benchmark (Defactify) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Expected Calibration Error (ECE)** | **0.47%** | **25.11%** | Different binning definitions; not directly comparable |
| **Maximum Calibration Error (MCE)** | **8.32%** | **32.81%** | Different binning definitions; not directly comparable |
| **Negative Log-Likelihood (NLL)** | **0.0726** | **0.9976** | Higher loss on this external benchmark |
| **Brier Score** | **0.0202** | **0.2881** | Higher squared probability error on this benchmark |

The validation ECE implementation groups by confidence, while the external analysis code groups by $P(\mathrm{AI})$ and computes a confidence/accuracy error within those bins. These definitions are not identical, so the two ECE values are not presented as a direct drift estimate. Recomputing the external predictions with confidence-based bins also gives 25.11% to the shown precision; harmonized implementations should be used for a formal ECE comparison. The frozen temperature was fit from validation data, not Defactify.

### 5.2 Selective Prediction & Risk-Coverage Curve
The following risk-coverage values are descriptive for this Defactify sample. The 0.85 row is an illustrative post-hoc operating point, not an independently validated deployment threshold:

| Confidence Threshold ($\tau_{\text{conf}}$) | Retained Coverage | Abstention Rate | Selective Accuracy | Selective Error Risk | Selective F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\ge 0.50$ (Standard)** | 100.0% (800) | 0.0% (0) | 63.63% | 36.38% | 0.5801 |
| **$\ge 0.60$** | 94.62% (757) | 5.38% (43) | 64.73% | 35.27% | 0.5835 |
| **$\ge 0.70$** | 87.25% (698) | 12.75% (102) | 65.76% | 34.24% | 0.5886 |
| **$\ge 0.80$** | 79.62% (637) | 20.38% (163) | 66.88% | 33.12% | 0.6011 |
| **$\ge 0.85$ (illustrative post-hoc)** | 74.00% (592) | 26.00% (208) | 68.07% | 31.93% | 0.6119 |
| **$\ge 0.90$** | 64.12% (513) | 35.88% (287) | 69.40% | 30.60% | 0.6288 |
| **$\ge 0.95$ (High Certainty)** | 46.88% (375) | **53.12% (425)** | **72.27%** | **27.73%** | **0.6438** |

---

## 6. Source-Aware Subgroup Analysis

This descriptive analysis partitions the frozen test predictions by source label. The non-CIFAKE subset is a mixed in-distribution group (292 REAL, 352 AI), not a natural-photograph-only or source-held-out test.

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                    SOURCE SUBGROUP DISSECTION: IN-DISTRIBUTION VS REALITY                             |
+--------------------------+-----------------------+---------+--------+--------+--------+--------+--------+-------------+
| Evaluation Subgroup      | Model Architecture    | N       | Acc    | Prec   | Rec    | F1     | MCC    | AUROC       |
+--------------------------+-----------------------+---------+--------+--------+--------+--------+--------+-------------+
| CIFAKE Subset            | V5-A Spatial          | 8,356   | 98.20% | 98.03% | 98.36% | 98.19% | 0.9641 | 0.9983      |
| (32x32 CIFAKE source)    | V5-B Frequency        | 8,356   | 90.40% | 90.29% | 90.38% | 90.34% | 0.8080 | 0.9662      |
|                          | V5-C Hybrid           | 8,356   | 96.90% | 96.66% | 97.11% | 96.89% | 0.9380 | 0.9958      |
|                          | V5-D Gated Residual   | 8,356   | 97.19% | 97.34% | 96.99% | 97.16% | 0.9438 | 0.9961      |
+--------------------------+-----------------------+---------+--------+--------+--------+--------+--------+-------------+
| Non-CIFAKE In-Distribution Mix | V5-A Spatial     | 644     | 81.68% | 86.11% | 79.26% | 82.54% | 0.6357 | 0.8920      |
| (292 REAL / 352 AI)      | V5-B Frequency        | 644     | 93.94% | 95.36% | 93.47% | 94.40% | 0.8783 | 0.9879      |
|                          | V5-C Hybrid           | 644     | 96.58% | 96.35% | 97.44% | 96.89% | 0.9311 | 0.9929      |
|                          | V5-D Gated Residual   | 644     | 97.05% | 96.90% | 97.73% | 97.31% | 0.9405 | 0.9930      |
+--------------------------+-----------------------+---------+--------+--------+--------+--------+--------+-------------+
| Defactify External       | V5-A Spatial          | 800     | 67.38% | 64.04% | 79.25% | 70.84% | 0.3577 | 0.7459      |
| (MS COCO vs Modern Gen)  | V5-B Frequency        | 800     | 55.25% | 65.67% | 22.00% | 32.96% | 0.1406 | 0.5573      |
|                          | V5-C Hybrid           | 800     | 64.25% | 65.32% | 60.75% | 62.95% | 0.2857 | 0.6859      |
|                          | V5-D Gated Residual   | 800     | 63.63% | 68.60% | 50.25% | 58.01% | 0.2828 | 0.7036      |
+--------------------------+-----------------------+---------+--------+--------+--------+--------+--------+-------------+
```

### Subgroup Interpretation
1. **CIFAKE Distribution Dominance:** CIFAKE constitutes **92.8% (8,356 / 9,000)** of the frozen clean test. V5-A has F1 98.19% on this subset and 82.54% on the mixed non-CIFAKE subset. This is an observed source-group difference; the mechanism has not been isolated by a dedicated pixelation ablation.
2. **Non-CIFAKE Subgroup:** On this mixed subset ($N=644$), V5-A has F1 82.54% and MCC 0.6357; V5-D has F1 97.31% and MCC 0.9405. This is a subgroup of the same random-split test and includes source families represented in training, so it is not independent cross-source evidence.
3. **Interpretation:** The subgroup results describe performance differences by recorded source group. They do not isolate a pixelation mechanism or establish performance on natural-image sources absent from training.

---

## 7. Audit Verification & Deduplication Scope (Step 7 Reconciliation)

The 800-image Defactify manifest and historical-reference audit report the following checks:

1. **Exact Cryptographic Separation:** 0 SHA-256 collisions against all reference ledgers.
2. **Rejection Threshold vs Observed Distance:**
   - **Rejection Boundary:** $\min d_H \le 3$. Exactly one candidate (`DEF_REAL_311`, identical to PIPE background plate `pipe_orig_140556.png`) exhibited $d_H = 0$ and was deterministically replaced by `DEF_REAL_400`.
   - **Empirical Observed Minimum:** $\min d_H = 10$. The mean perceptual distance across all 800 candidates is $\mu = 15.95$. Earlier informal notes mentioning $\min d_H \ge 16$ pertained to the specific replacement candidate, whereas the benchmark population minimum is formally documented as $d_H = 10 > 3$.
3. **Post-freeze internal check:** The frozen benchmark was additionally checked pairwise: no internal exact-SHA duplicates and no internal pHash pairs at Hamming distance $\le 10$ were found. This pairwise check is separate from the historical audit script, which compares candidates with its historical reference index.
4. **Scope:** These checks support exact-byte and stated pHash-threshold separation for the checked references. They do not establish semantic independence or rule out every source/group relationship.

---

## 8. Scientific Conclusions & Scope

### 8.1 Model Comparison Summary
- **Clean test:** V5-D has the best clean-test point estimates. Its F1 difference from V5-A is not statistically significant; its nominal AUROC difference is not significant after Holm correction across the six pairwise clean-test AUROC comparisons.
- **Defactify:** V5-A has the strongest overall F1 (70.84%). V5-D has lower REAL FPR than V5-A (23.00% vs 44.50%) and a higher interpolated TPR at 1% FPR (10.25% vs 6.22%).
- **Controlled transformations:** V5-D performs better than V5-A in some tested conditions, while V5-A leads in others. Model ranking changes by evaluation regime. V5-D provides a competitive fusion architecture with advantages in selected regimes, but the evidence does not establish universal superiority.

### 8.2 Calibration and Selective Prediction
- The frozen temperature is $T = 2.1983866642849734$, fit using validation data. Validation calibration metrics have a post-selection limitation because the same validation portion was used to select between temperature and Platt scaling.
- The Defactify 0.85 confidence row is an illustrative post-hoc operating point (31.93% selective risk at 74.0% coverage), not an independently validated deployment threshold.

### 8.3 Research Artifact State
The analysis artifacts are stored under `evaluation_results/v5/strengthened_evaluation/`. At the time of this correction pass, the strengthened scripts and output directory are untracked in Git; they are present in the local repository state but not in the tracked commit history.

### 8.4 Key Scientific Limitations & Methodological Disclosures
1. **CIFAKE Representation:** CIFAKE is 8,356 of 9,000 clean-test samples. The CIFAKE-dominated random split may preserve source-specific cues between training and test data, limiting interpretation as generic cross-source detection.
2. **External Scope:** Defactify is one internally audited external benchmark. It does not establish broad transfer to other datasets or generator populations.
3. **Controlled Perturbations:** Controlled synthetic transformations (blur, noise, resize, JPEG) are not equivalent to real-world, in-the-wild image degradation.
4. **Validation Calibration Post-Selection:** In-distribution validation calibration metrics are post-selection estimates because the reported portion was also used to select the scaling method.
5. **ECE Definitions:** Validation and external ECE implementations use different bin membership definitions. The external value is descriptive; direct cross-dataset comparison requires harmonized definitions.
6. **External Calibration:** The frozen Defactify analysis reports high NLL, Brier score, and ECE; calibration degrades on this benchmark. Defactify was not used to fit the temperature.
7. **Class Prevalence:** The 50/50 external benchmark balance does not reflect deployment prevalence.
8. **Generator Variance:** Recall varies across the four tested generator groups; no conclusion is made about all current generators.
9. **Subgroup Scope:** The 644-image non-CIFAKE subgroup is not source-held-out and does not establish transfer to unseen natural-image sources.
10. **Clean Comparison:** V5-D's clean F1 difference from V5-A is not statistically significant (approximate paired-bootstrap $p=0.480$).
