# AIDetect: Final Scientific Report Corrections Log

**Document Identifier:** `AIDETECT-CORRECTIONS-LOG-V5-FINAL`
**Date:** October 10, 2026
**Subject:** Formal Corrections to `strengthened_evaluation_report.md`
**Governing Standard:** Scientific Honesty, Methodological Rigor, and Empirical Precision

---

## 1. Summary of Corrections

The initial correction entries below are historical. The final correction pass supersedes the earlier generator-hardness interpretation and adds corrections based on direct re-aggregation of the frozen per-image predictions. See Section 4 for the final values and scope.

---

## 2. Detailed Corrections Table

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        FINAL SCIENTIFIC REPORT CORRECTION LOG                                         |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 1: HOLM-BONFERRONI MULTIPLICITY ADJUSTMENT FOR CLEAN AUROC                                                |
| - Correction Made          : Qualified nominal AUROC significance under Holm-Bonferroni correction.                   |
| - Old Wording/Value        : "...the AUROC improvement of V5-D over V5-A is statistically significant (p = 0.024)."    |
| - New Wording/Value        : "V5-D showed a nominal AUROC improvement over V5-A (ΔAUROC = +0.001413, unadjusted        |
|                              p = 0.024); however, this difference was not statistically significant after             |
|                              Holm-Bonferroni correction for the six pairwise model comparisons (alpha_Holm = 0.0167)." |
| - Reason                   : Evaluating six pairwise model comparisons inflates the family-wise error rate. Under     |
|                              Holm-Bonferroni step-down testing, the 4th rank threshold is alpha = 0.0167; because     |
|                              0.024 > 0.0167, the difference represents a non-significant trend.                       |
| - Authoritative Artifact   : clean_paired_comparisons.csv & clean_metrics_extended.json                               |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 2: DEFACTIFY GENERATOR HARDNESS BREAKDOWN ACCURACY                                                         |
| - Correction Made          : Recomputed generator recalls directly from raw per-image predictions to fix indexing bug. |
| - Old Wording/Value        : Buggy draft values (V5-A SD3 46%; V5-B SD3 19%, MJv6 10%; V5-C SD3 37%; V5-D SD3 24%).     |
| - New Wording/Value        : Authoritative per-image values:                                                          |
|                              • V5-A: SD3 91%, MJv6 73% (lowest), DALL-E 3 79%, SDXL 74% (Mean = 79.25%)              |
|                              • V5-B: SD3 17%, MJv6 34%, DALL-E 3 28%, SDXL 9% (lowest) (Mean = 22.00%)               |
|                              • V5-C: SD3 62%, MJv6 66%, DALL-E 3 65%, SDXL 50% (lowest) (Mean = 60.75%)               |
|                              • V5-D: SD3 56%, MJv6 67%, DALL-E 3 47%, SDXL 31% (lowest) (Mean = 50.25%)               |
| - Reason                   : Original script applied positional boolean mask against differently ordered pivot table.  |
|                              Recomputation verifies that arithmetic mean of the 4 generator recalls equals overall    |
|                              AI recall within 1e-9 for all models.                                                    |
| - Authoritative Artifact   : external_benchmark_v1_per_image.csv & generator_analysis.csv                             |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 3: REMOVAL OF UNVERIFIED INDUCTIVE BIAS EVASION MECHANISM                                                  |
| - Correction Made          : Eliminated architectural mechanistic assertion regarding MMDiT diffusion transformers.   |
| - Old Wording/Value        : "...MMDiT producing subtle high-frequency artifacts that evade traditional convolutional  |
|                              spatial/spectral detectors."                                                             |
| - New Wording/Value        : "Generator difficulty is reported as an empirical observation across the tested candidate|
|                              pools; no causal claims regarding diffusion transformer architectures or evasion of      |
|                              convolutional inductive biases are asserted."                                            |
| - Reason                   : The benchmark measured overall black-box empirical recall; it did not isolate the         |
|                              specific causal impact of transformer attention vs convolution.                          |
| - Authoritative Artifact   : generator_analysis.csv & external_metrics_extended.json                                   |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 4: REPLACEMENT OF TABLE 2.1 TRANSCRIPTION DISCREPANCIES                                                    |
| - Correction Made          : Replaced preliminary draft values in Table 2.1 with authoritative JSON point estimates.   |
| - Old Wording/Value        : V5-A EER = 3.01%, TPR@1% = 88.42% | V5-B EER = 8.89%, TPR@1% = 61.27%                     |
|                              V5-C EER = 2.91%, TPR@1% = 92.11% | V5-D EER = 2.71%, TPR@1% = 92.89%                     |
| - New Wording/Value        : V5-A EER = 2.98%, TPR@1% = 92.58% | V5-B EER = 9.36%, TPR@1% = 60.00%                     |
|                              V5-C EER = 3.09%, TPR@1% = 92.36% | V5-D EER = 2.80%, TPR@1% = 93.18%                     |
| - Reason                   : Table 2.1 in the draft contained preliminary draft transcription numbers.                |
| - Authoritative Artifact   : clean_metrics_extended.json & v5_clean_test_predictions.csv                             |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 5: REMOVAL OF CAUSAL SHORTCUT CLAIMS ON CIFAKE                                                             |
| - Correction Made          : Replaced causal claims regarding pixelation shortcuts with empirical domain sensitivity. |
| - Old Wording/Value        : "Because CIFAR-10 images are upscaled from 32x32, spatial models exploit low-level       |
|                              interpolation and pixelation artifacts rather than semantic generative clues. This       |
|                              artificially inflates V5-A's clean-test F1 to 98.19%."                                   |
| - New Wording/Value        : "V5-A shows substantially stronger performance on the CIFAKE-dominated subset (98.19%   |
|                              F1) and a marked reduction on the non-CIFAKE subset (82.54% F1), consistent with         |
|                              sensitivity to source/domain characteristics. Without isolated ablations on pixelation   |
|                              artifacts, this is described as domain sensitivity rather than a proven interpolation-   |
|                              shortcut mechanism."                                                                     |
| - Reason                   : Asserting that the model 'exploits interpolation artifacts' asserts an unproven causal   |
|                              mechanism without dedicated ablation experiments.                                        |
| - Authoritative Artifact   : source_subgroup_analysis.csv & source_subgroup_analysis.json                             |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 6: REMOVAL OF CAUSAL SHORTCUT PREVENTION CLAIMS FOR V5-D                                                   |
| - Correction Made          : Replaced causal prevention claim with non-causal evidence of relative stability.         |
| - Old Wording/Value        : "Gated multi-stream fusion effectively prevents shortcut reliance on natural             |
|                              photography."                                                                            |
| - New Wording/Value        : "The subgroup results provide evidence that V5-D is less sensitive to the source        |
|                              characteristics dominating the CIFAKE-heavy evaluation distribution."                    |
| - Reason                   : 'Prevents' asserts complete immunity; the empirical evidence demonstrates relative       |
|                              stability, not absolute prevention.                                                      |
| - Authoritative Artifact   : source_subgroup_analysis.csv                                                             |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 7: CONTROLLED ROBUSTNESS TERMINOLOGY & PROOF CLAIMS                                                        |
| - Correction Made          : Removed proof claims and enforced 'controlled transformation robustness' terminology.    |
| - Old Wording/Value        : "...while V5-D maintains MCC = 0.2295, proving that gated fusion mitigates resolution     |
|                              degradation."                                                                            |
| - New Wording/Value        : "...while V5-D maintains higher performance under severe downsampling (MCC = 0.2295),    |
|                              indicating greater resilience under the controlled transformation. These experiments     |
|                              evaluate controlled synthetic perturbations and should not be interpreted as definitive  |
|                              proof of real-world deployment robustness."                                              |
| - Reason                   : Synthetic digital transformations must not be conflated with real-world robustness.       |
| - Authoritative Artifact   : robustness_extended.csv & robustness_extended.json                                       |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 8: REMOVAL OF HYPERBOLIC LANGUAGE ('DRASTICALLY SUPERIOR')                                                 |
| - Correction Made          : Replaced hyperbolic terminology with precise quantitative comparisons.                   |
| - Old Wording/Value        : "...V5-D is drastically superior on authentic natural photography..."                    |
| - New Wording/Value        : "...V5-D achieves substantially higher precision and F1 on the non-CIFAKE subset         |
|                              representing authentic high-resolution natural photography (97.31% vs 82.54% F1 compared |
|                              with V5-A)."                                                                             |
| - Reason                   : Scientific publications require objective quantitative reporting without exaggeration.   |
| - Authoritative Artifact   : source_subgroup_analysis.csv                                                             |
+-----------------------------------------------------------------------------------------------------------------------+
| CORRECTION 9: FORMAL INCLUSION OF SCIENTIFIC LIMITATIONS                                                              |
| - Correction Made          : Added explicit Section 8.4 documenting all 9 key methodological limitations.             |
| - Old Wording/Value        : Omitted explicit limitation checklist.                                                   |
| - New Wording/Value        : Added full 9-point methodological disclosure covering CIFAKE overrepresentation, OOD     |
|                              calibration drift, generator variance, controlled perturbation scope, post-selection     |
|                              validation ECE, and clean F1 non-significance.                                           |
| - Reason                   : Guarantees complete scientific transparency and balanced discussion.                     |
| - Authoritative Artifact   : statistical_sanity_check.md                                                              |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Initial Correction Status

- Correction 1's Holm interpretation remains applicable to the six pairwise clean-test AUROC comparisons only.
- Initial Correction 2's claim that SD3 was the lowest-recall generator for V5-A, V5-C, and V5-D was itself incorrect and is superseded below.
- The earlier publication-grade certification is withdrawn. The report remains subject to the limitations recorded in the final consistency check.

---

## 4. Final Correction Pass (October 10, 2026)

### 4.1 Generator Analysis Regenerated from Frozen Per-Image Predictions

The generator analysis was regenerated directly from `diagnostic_outputs/external_benchmark_v1_results/external_benchmark_v1_per_image.csv`, grouping AI rows by model and generator and counting `prediction == AI`. Each group contains 100 distinct AI candidates. The corrected recall percentages are:

| Model | SD3 | Midjourney v6 | DALL-E 3 | SDXL | Macro / Overall AI Recall | Lowest in This Sample |
|---|---:|---:|---:|---:|---:|---|
| V5-A Spatial | 91% | 73% | 79% | 74% | 79.25% | Midjourney v6, 73% |
| V5-B Frequency | 17% | 34% | 28% | 9% | 22.00% | SDXL, 9% |
| V5-C Hybrid | 62% | 66% | 65% | 50% | 60.75% | SDXL, 50% |
| V5-D Gated Residual | 56% | 67% | 47% | 31% | 50.25% | SDXL, 31% |

The per-model arithmetic mean equals overall AI recall within $10^{-9}$ because the benchmark has 100 AI samples per generator. The previous strengthened table was produced by applying a boolean mask from metadata order to a pivot table with a different index order. Its generator-level values and minima must not be cited.

### 4.2 Bootstrap and Paired-Comparison Wording

- Bootstrap intervals are described as ordinary image-level sampling with replacement: clean $N=9,000$, external $N=800$, $B=1,000$, seed 42, percentile 95% intervals.
- Paired model comparisons use identical resample indices across models.
- The V5-D minus V5-A clean AUROC difference is $+0.001413$, approximate unadjusted bootstrap-tail $p=0.024$, 95% CI $[+0.000152,+0.002712]$. It is not significant after Holm correction across the six pairwise clean-test AUROC comparisons.
- The V5-D minus V5-A clean F1 difference is $+0.001564$, approximate unadjusted $p=0.480$, 95% CI approximately $[-0.002703,+0.005814]$; no significant clean F1 difference is claimed.
- Holm correction is claimed only for the six pairwise clean-test AUROC comparisons. Other metric comparisons are exploratory/descriptive and unadjusted.

### 4.3 Calibration and Risk-Coverage Scope

- The frozen temperature is $T=2.1983866642849734$. Defactify was not used to fit it.
- Calibration-method selection and reported validation calibration metrics use the same validation portion; these metrics have a post-selection limitation.
- Validation ECE bins by confidence; the external ECE implementation bins by $P(\mathrm{AI})$ and computes a confidence/accuracy error. The definitions differ and are not treated as directly comparable. The external ECE value was separately recomputed with confidence bins and rounds to the same 25.11% on these predictions.
- The 0.85 confidence result (31.93% selective risk at 74.0% coverage) is an illustrative post-hoc operating point, not an independently validated threshold.

### 4.4 Leakage and Source-Domain Language

- All 800 current payload hashes match their manifest entries; no internal exact-SHA duplicate pairs or internal pHash pairs at distance $\le 10$ were found in a pairwise check of the frozen manifest.
- The historical audit script's candidate-to-reference comparison is distinct from that post-freeze internal pairwise check. One historical near-duplicate was rejected and replaced before inference.
- The clean test has 8,356 CIFAKE samples out of 9,000. The non-CIFAKE subgroup has 292 REAL and 352 AI examples and is not a natural-photo-only or source-held-out test.
- The report describes source-specific differences without asserting a pixelation shortcut or causal mechanism.

### 4.5 V5-D Interpretation

V5-D has the best clean-test point estimates, but its clean F1 difference from V5-A is not statistically significant and its nominal AUROC difference does not survive the stated Holm correction. V5-A has the strongest overall Defactify F1; V5-D has lower Defactify REAL FPR and is higher at the interpolated 1% FPR point. V5-D leads V5-A in some controlled conditions, while V5-A leads in others. No single model leads across every reported regime.
