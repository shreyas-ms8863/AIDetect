# AIDetect — Final Evaluation Summary

## 1. Evaluation Overview

AIDetect evaluates AI-generated image detection using four V5 model variants:

- V5-A: Spatial-only ResNet-50
- V5-B: Frequency-only ResNet-50
- V5-C: Spatial + Frequency ResNet-50
- V5-D: Spatial + Frequency + Noise Residual + Gated Fusion

The evaluation was conducted across:

1. Frozen clean V5 test set
2. Controlled robustness transformations
3. Independent external Defactify benchmark
4. Calibration analysis
5. Generator-wise external evaluation
6. Error, confidence, agreement, and domain-shift analysis

The V5 dataset contains 60,000 images with balanced REAL/AI classes and fixed train/validation/test splits. The frozen test set contains 9,000 images.

---

## 2. Clean Frozen Test Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| V5-A Spatial | 97.02% | 97.17% | 96.87% | 97.02% |
| V5-B Frequency | 90.66% | 90.68% | 90.62% | 90.65% |
| V5-C Hybrid | 96.88% | 96.64% | 97.13% | 96.89% |
| V5-D Gated Residual | 97.18% | 97.30% | 97.04% | 97.17% |

V5-D achieved the highest clean-test F1, but its improvement over the spatial-only model was only approximately 0.15 percentage points.

This indicates that the more complex fusion architecture does not provide a large clean-domain performance advantage.

---

## 3. Robustness Evaluation

The same frozen models were evaluated without retraining under:

- Resize: 75%, 50%, 25%
- Blur: radius 1, 2, 3
- Gaussian noise: 0.01, 0.05, 0.10
- JPEG compression: Q95, Q75, Q50
- Re-encoding

Average robustness F1:

| Model | Average Robustness F1 | Average Degradation from Clean |
|---|---:|---:|
| V5-A Spatial | 76.49% | 20.53 pp |
| V5-B Frequency | 63.38% | 27.28 pp |
| V5-C Hybrid | 71.79% | 25.09 pp |
| V5-D Gated Residual | 73.44% | 23.74 pp |

Across the 12 robustness conditions:

- V5-A was best in 6 conditions.
- V5-D was best in 5 conditions.
- V5-C was best in 1 condition.
- V5-B was not best in any condition.

Therefore, there is no universal robustness winner.

The spatial-only model was the most consistently robust overall, while V5-D provided competitive performance on several conditions.

These controlled transformations should not be described as equivalent to real-world robustness.

---

## 4. Independent External Benchmark

The models were evaluated on an independently audited Defactify validation subset:

- 800 total images
- 400 REAL
- 400 AI
- 100 AI images from each of four generator groups
- SD3
- Midjourney v6
- DALL-E 3
- SDXL
- No SHA-256 collision with historical evaluation assets
- One perceptual near-duplicate was detected and replaced before evaluation
- No retraining or threshold tuning was performed on the benchmark

External results:

| Model | Accuracy | Precision | Recall | F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| V5-A Spatial | 67.38% | 64.04% | 79.25% | 70.84% | 67.38% |
| V5-B Frequency | 55.25% | 65.67% | 22.00% | 32.96% | 55.25% |
| V5-C Hybrid | 64.25% | 65.32% | 60.75% | 62.95% | 64.25% |
| V5-D Gated Residual | 63.62% | 68.60% | 50.25% | 58.01% | 63.62% |

The external benchmark reveals a substantial generalization gap relative to the clean test set.

V5-A produced the strongest external F1 and AI recall despite not being the clean-test winner.

---

## 5. External REAL False-Positive Rates

| Model | REAL FPR |
|---|---:|
| V5-A Spatial | 44.50% |
| V5-B Frequency | 11.50% |
| V5-C Hybrid | 32.25% |
| V5-D Gated Residual | 23.00% |

This demonstrates an important trade-off.

V5-A detects more AI images externally but also incorrectly labels many REAL images as AI.

V5-D reduces the REAL false-positive rate compared with V5-A, but this comes at the cost of substantially lower AI recall.

---

## 6. Generator-wise AI Recall

| Model | SD3 | Midjourney v6 | DALL-E 3 | SDXL |
|---|---:|---:|---:|---:|
| V5-A Spatial | 91% | 73% | 79% | 74% |
| V5-B Frequency | 17% | 34% | 28% | 9% |
| V5-C Hybrid | 62% | 66% | 65% | 50% |
| V5-D Gated Residual | 56% | 67% | 47% | 31% |

Performance varies substantially by generator family.

This supports the conclusion that AI-image detection performance is generator-dependent and that high performance on one dataset should not automatically be interpreted as universal generator generalization.

---

## 7. Calibration

V5-D probability calibration was evaluated using a held-out calibration/evaluation split from the validation data.

Temperature scaling produced:

- Temperature T = 2.1983866643
- Raw NLL = 0.116260
- Calibrated NLL = 0.072624
- Raw Brier = 0.022778
- Calibrated Brier = 0.020232
- Raw ECE = 1.97%
- Calibrated ECE = 0.47%

Temperature scaling was selected over Platt scaling because it produced slightly better calibration metrics.

The model checkpoint itself was not modified by calibration. Calibration is a post-hoc probability transformation.

---

## 8. Model Agreement

On the external benchmark:

- All four models agreed on 37.25% of images.
- Three of four agreed on 46.00%.
- Two of four agreed on 16.75%.

Pairwise agreement:

- A/B: 47.12%
- A/C: 73.12%
- A/D: 69.25%
- B/C: 59.75%
- B/D: 67.12%
- C/D: 78.62%

The disagreement patterns demonstrate that the models do not simply produce identical predictions and that different feature representations respond differently to the external domain.

---

## 9. Clean → Robustness → External Generalization

| Model | Clean F1 | Robustness F1 | External F1 | Clean → External |
|---|---:|---:|---:|---:|
| V5-A | 97.02% | 76.49% | 70.84% | -26.18 pp |
| V5-B | 90.65% | 63.38% | 32.96% | -57.69 pp |
| V5-C | 96.89% | 71.79% | 62.95% | -33.94 pp |
| V5-D | 97.17% | 73.44% | 58.01% | -39.16 pp |

Clean-test performance was therefore not a reliable indicator of cross-dataset generalization in this comparison.

The external benchmark produced a much larger spread between models than the clean benchmark.

---

## 10. Final Model Interpretation

### V5-A — Spatial-only

Strengths:
- Best external F1
- Highest external AI recall
- Strongest average robustness
- Most consistent longitudinal ranking

Weakness:
- Highest REAL false-positive rate among the four models.

### V5-B — Frequency-only

Strengths:
- Lowest external REAL false-positive rate.

Weaknesses:
- Weakest clean performance
- Weakest robustness
- Very low external AI recall
- Poor generalization across generators

### V5-C — Spatial + Frequency

Strengths:
- Strong clean performance
- Better external performance than frequency-only
- Competitive generator-wise behavior

Weakness:
- Does not consistently outperform the simpler spatial model.

### V5-D — Gated Residual

Strengths:
- Best clean-test F1
- Competitive robustness
- Lower external REAL FPR than spatial-only
- Strong pairwise agreement with V5-C

Weakness:
- External AI recall is substantially lower than V5-A.
- Clean-domain advantage does not transfer directly to the external benchmark.

---

## 11. Main Research Findings

1. Spatial features were the strongest overall representation in this experimental setting.
2. Frequency-only detection was consistently weaker than spatial-based approaches.
3. Adding frequency information did not guarantee improved generalization.
4. The gated residual architecture achieved the best clean-test result but did not achieve the best external result.
5. External generator performance varied substantially across generator families.
6. Robustness under controlled transformations did not guarantee strong external generalization.
7. Clean-test performance alone is insufficient for evaluating practical AI-image detector generalization.
8. High-confidence errors remain a significant limitation.
9. Model disagreement provides useful evidence that different representations capture different signals.
10. The results support a cautious interpretation of AI-image detectors as probabilistic forensic tools rather than universally reliable classifiers.

---

## 12. Important Limitations

The project should not claim:

- universal AI-image detection
- perfect real-world reliability
- universal unseen-generator detection
- that controlled JPEG/resize/blur/noise tests represent all real-world transformations
- that the V5-D architecture is a fundamentally novel detection method

The Defactify benchmark was independently audited against historical project assets using SHA-256 and perceptual-hash checks. One perceptual near-duplicate was identified and replaced before evaluation. Although some generator families may overlap with generator families represented in earlier project experiments, the benchmark should therefore be described as an independently audited external benchmark rather than as definitive proof of universal unseen-generator generalization.

The 54-image camera diagnostic is exploratory and must not be presented as a population-level false-positive estimate.

---

## 13. Final Conclusion

AIDetect demonstrates that high in-dataset accuracy can coexist with substantially weaker cross-dataset performance. Among the evaluated V5 variants, the spatial-only model provided the strongest overall external generalization and robustness consistency, while the gated residual model provided the strongest clean-test performance.

The central result is therefore not that one architecture solves AI-generated image detection, but that representation choice, robustness, calibration, and especially cross-dataset evaluation materially affect observed detector performance.

This supports the use of comprehensive multi-condition evaluation rather than relying on a single clean test-set accuracy.

