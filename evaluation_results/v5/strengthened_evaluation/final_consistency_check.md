# AIDetect V5 Final Consistency Check

**Date:** October 10, 2026
**Scope:** Analysis/report correction only. No training, inference, checkpoint edits, threshold changes, dataset edits, raw-prediction edits, Git commits, or pushes were performed during this pass.

## Result

**PASS for the listed consistency and integrity checks.** This is an internal, artifact-based verification, not an independent third-party audit or a claim that the remaining methodological limitations are resolved.

The checks included 1,592 assertions in the main read-only audit and 150 supplemental assertions. Generator values were separately verified from the authoritative 3,200-row per-image prediction CSV before regeneration.

## Generator Analysis

The corrected `generator_analysis.csv` was generated from rows with `true_label == AI`, grouped by model and the source generator field. All 16 model/generator groups contain 100 unique candidates. Every recall matches direct raw regrouping, and the mean of each model's four generator recalls equals its overall AI recall within $10^{-9}$.

| Model | SD3 | Midjourney v6 | DALL-E 3 | SDXL | Mean / Overall Recall | Lowest Recall in Sample |
|---|---:|---:|---:|---:|---:|---|
| V5-A Spatial | 91% | 73% | 79% | 74% | 79.25% | Midjourney v6, 73% |
| V5-B Frequency | 17% | 34% | 28% | 9% | 22.00% | SDXL, 9% |
| V5-C Hybrid | 62% | 66% | 65% | 50% | 60.75% | SDXL, 50% |
| V5-D Gated Residual | 56% | 67% | 47% | 31% | 50.25% | SDXL, 31% |

The minimum-recall labels in this table are computed from the explicit recalls. This corrects inconsistencies in the supplied proposed interpretation as well as the earlier strengthened table. The table does not support causal claims about generator architecture or detector mechanisms.

## Table and Metric Checks

- Clean-test confusion matrices and accuracy, precision, recall, F1, balanced accuracy, MCC, AUROC, AUPRC, EER, and interpolated TPR at fixed FPR were recomputed from `v5_clean_test_predictions.csv`; they match the extended metrics JSON, frozen-test comparison JSON, and displayed clean table within rounding tolerance.
- The report's displayed clean bootstrap CIs match `clean_bootstrap_ci.csv` and `clean_metrics_extended.json`. The implementation is an ordinary image-level bootstrap with replacement ($N=9,000$, $B=1,000$, seed 42), not a stratified bootstrap. The reported CIs were not changed.
- The paired-comparison CSV and JSON agree. The same resample-index array is applied to all models in each replicate. For V5-D minus V5-A, clean AUROC delta is +0.001413 (approximate unadjusted bootstrap-tail $p=0.024$, 95% CI [+0.000152, +0.002712]); it is not significant after Holm correction across the six clean-test AUROC pairs. Clean F1 delta is +0.001564 (approximate $p=0.480$, 95% CI approximately [-0.002703, +0.005814]). Holm correction is not claimed for every metric family.
- Defactify overall confusion matrices and Accuracy, Precision, Recall, F1, Balanced Accuracy, MCC, AUROC, AUPRC, EER, and interpolated fixed-FPR metrics were recomputed from the per-image CSV and match the extended JSON/report within rounding tolerance. V5-A retains the highest overall external F1 (70.84%); the negative external results were retained.
- All 48 robustness rows were checked against their confusion counts by recomputing accuracy, balanced accuracy, precision, recall, F1, and MCC. The displayed robustness matrix cells for V5-A/C/D also match the extended CSV. These remain controlled synthetic transformations.
- All 12 source-subgroup rows were recomputed from clean/external per-image predictions. Counts are CIFAKE 8,356 (4,208 REAL, 4,148 AI), non-CIFAKE 644 (292 REAL, 352 AI), and Defactify 800 (400 REAL, 400 AI). The non-CIFAKE subset is mixed and not source-held-out.
- The report's risk-coverage values at all stored confidence thresholds were recomputed from V5-D Defactify probabilities. The 0.85 point is explicitly illustrative and post-hoc, not independently validated.
- Temperature $T=2.1983866642849734$ matches the calibration configuration. Defactify was not used to fit it. External NLL, Brier, ECE, and MCE were recomputed from the stored probabilities. Validation ECE bins by confidence; the external implementation bins by $P(\mathrm{AI})$. The definitions differ, so ECE/MCE values are not treated as directly comparable. The external ECE recomputed with confidence bins rounds to the same 25.11% on this sample.
- Defactify manifest and predictions contain 800 unique samples, 400 REAL and 400 AI, with 100 AI candidates for each of four generator labels. The historical audit's one rejected near-duplicate/replacement is retained in the record.

## Leakage and Integrity Checks

- All 800 payload files are present, with no extras, and every file's SHA-256 matches the frozen Defactify manifest.
- The frozen manifest has no internal exact-SHA duplicate pairs and no internal pHash pairs at Hamming distance $\le 10$ in the post-freeze pairwise check. The accepted-set minimum historical-reference pHash distance is 10; the audit rejection boundary was $d_H \le 3$.
- V5 manifest: 60,000 rows, 60,000 unique SHA-256 values, expected split/class counts, and zero exact-SHA overlap with the Defactify manifest.
- Before/after SHA-256 checks matched for the raw external prediction CSV, external manifest, V5 dataset manifest, frozen clean prediction CSV, frozen test comparison JSON, calibration JSON, and all four V5 checkpoints. The sorted 800-payload aggregate digest also matched its before-state.
- The analysis config and reports retain the 0.50 classification threshold and the existing temperature. No training or model inference command was run during this correction pass.

Protected-file hashes (unchanged):

| File | SHA-256 |
|---|---|
| `diagnostic_outputs/external_benchmark_v1_results/external_benchmark_v1_per_image.csv` | `1cedc5b4a49133da27f07e56ee02159ea1d755c87ef73abe45faa3084a0b6b35` |
| `diagnostic_outputs/external_benchmark_v1_manifest.csv` | `65f7c7d985448f1aee76800cfc1aa4127724af2a54158b91ec8495a3ba3e4d08` |
| `v5_metadata/v5_dataset_manifest.csv` | `38dab8c65593e1ab2baf8f4384bf1407e66bd62a008ff9e096139eb96eaf436d` |
| `evaluation_results/v5/strengthened_evaluation/v5_clean_test_predictions.csv` | `ae60f17182bad358b2dfeacf04d8b46d671e1fabe16e5560aaa2f42277224d7d` |
| `backend/evaluation_results/v5/v5_final_test_comparison.json` | `8ade4075a7aadacecf053adb419b246730b7f024b399587beaedad1bdc02ec9d` |
| `backend/models/v5/v5_d_calibration.json` | `bfeab46909095fe635c6372f60d05fa326a5f9e12ebfc708feca598eaf9bc236` |
| `backend/models/v5/frequency_resnet50_v5_best.pth` | `daabc67eb2dcef38c95d7d04253beb4c13fca63285419834dc0e840cb2b59ab7` |
| `backend/models/v5/gated_residual_resnet50_v5_best.pth` | `1dfe03f78483e4eddd13869760405c17d2332d6d01e5d5d13ee5de5d9a5617ac` |
| `backend/models/v5/hybrid_resnet50_v5_best.pth` | `d3383324b97d6db8f5af7aa04849a0320f9e2da0bc22317b4c2416e088b943ff` |
| `backend/models/v5/spatial_resnet50_v5_best.pth` | `2cd55cd1ae94f15e350679d5b122ce725f9c421e16e73709c8620411fa971a9d` |
| Sorted external payload name/SHA aggregate | `7754bd64b72adc450ad8f61f744c2bbea52074ba2013f8d26f3fbad37f06b185` |

## Overclaiming Scan

The final report was searched for the requested terms. The target term **“best”** appears once in “best clean-test point estimates” (a descriptive statement supported by the clean point-estimate table), and **“universal superiority”** appears once where it is explicitly negated (“the evidence does not establish universal superiority”). No affirmative occurrences remain for “proves/proof,” “causes,” “prevents,” “guarantees,” “universal,” “always/never,” “unseen-generator,” “real-world robustness,” “production-ready,” “state-of-the-art,” “novel architecture,” “significantly better,” “superior,” “shortcut,” or “stratified.” Statistical “significant” language is retained only for the qualified, explicitly described comparisons.

## Git State

`git status --short` before and after this correction pass showed the same untracked audit scripts and the untracked `evaluation_results/v5/strengthened_evaluation/` directory. No tracked file changes, commits, or pushes were made. The four correction artifacts are inside that already-untracked directory; Git therefore continues to report the directory as untracked rather than listing each contained file separately.

## Remaining Scientific Limitations

The CIFAKE-dominated random split may preserve source-specific cues between training and test. The external set is one balanced Defactify benchmark, not broad proof of generator transfer. Bootstrap inference is image-level and unstratified; paired p-values are approximate bootstrap-tail values. Calibration-method selection is post-selection, external ECE definitions differ, and the 0.85 selective point is post-hoc. These limitations remain explicit; the corrected tables do not remove them.
