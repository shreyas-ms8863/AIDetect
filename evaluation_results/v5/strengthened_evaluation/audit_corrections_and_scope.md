# AIDetect: External Benchmark Audit Documentation & Forensic Scope Clarification

**Document ID:** `AIDETECT-AUDIT-CORRECTIONS-SCOPE-V1`
**Date:** October 10, 2026
**Evaluation Scope:** Defactify External Benchmark v1 ($N=800$) & Master V5 Ledger Cross-Auditing
**Status:** Certified & Reconciled

---

## 1. Executive Summary & Purpose

During rigorous empirical verification of the AIDetect research program, a precision audit was conducted on the deduplication claims, hash distance statistics, and forensic scope definitions associated with the Defactify External Benchmark v1 ($N=800$).

This document formally records:
1. **The Correction of the Perceptual Hash Distance Claim:** Clarification between the *rejection threshold* ($\min d_H \le 3$) and the *empirical observed minimum distance* ($\min d_H = 10$).
2. **The Three-Tiered Verification Hierarchy:** Explicit distinction between exact cryptographic matching, perceptual similarity hashing, and category/source group scope.
3. **Forensic Reconciliation with Existing Audit Reports:** Formal alignment between [`diagnostic_outputs/external_benchmark_leakage_audit_v1.md`](../../../diagnostic_outputs/external_benchmark_leakage_audit_v1.md) and [`diagnostic_outputs/external_benchmark_v1_manifest.csv`](../../../diagnostic_outputs/external_benchmark_v1_manifest.csv).

---

## 2. Perceptual Hash (pHash) Distance Clarification

### 2.1 The Historical Discrepancy
In early benchmark audit notes, an informal statement suggested that all accepted candidates exhibited $\min(d_H) \ge 16$ across all historical ledgers. While the *replacement candidate* for the single rejected sample (`DEF_REAL_400`, replacing `DEF_REAL_311`) indeed had $\min(d_H) = 16$, the benchmark-wide empirical distribution across all 800 candidates spans a wider range.

### 2.2 Authoritative Empirical Distribution
An exhaustive scan of `min_ref_phash_dist` recorded in [`external_benchmark_v1_manifest.csv`](../../../diagnostic_outputs/external_benchmark_v1_manifest.csv) establishes the following exact population statistics:

| Metric | Empirical Value | Forensic Interpretation |
| :--- | :--- | :--- |
| **Active Rejection Threshold** | $d_H \le 3$ | Defined rejection boundary for near-duplicate/shared-scene contamination |
| **Observed Minimum Distance ($\min d_H$)** | **10** | Lowest Hamming distance to any reference asset in the accepted set |
| **1st Percentile ($Q_{0.01}$)** | 12.0 | Only 1% of samples have distance $\le 12$ |
| **5th Percentile ($Q_{0.05}$)** | 14.0 | 95% of benchmark samples have distance $\ge 14$ |
| **Median Distance ($Q_{0.50}$)** | 16.0 | Half of all samples have distance $\ge 16$ |
| **Mean Distance ($\mu_{d_H}$)** | 15.95 | Robust separation across the entire 800-image benchmark |
| **Observed Maximum Distance ($\max d_H$)** | 18.0 | Maximal perceptual disparity observed |
| **Candidates with $d_H < 16$** | 181 (22.6%) | Validated between $10 \le d_H \le 14$, far above the rejection threshold of 3 |

### 2.3 Methodological Conclusion
The benchmark candidate pool satisfies the strict forensic requirement:
$$\min_{i \in \text{Benchmark}, j \in \text{Historical}} d_H(p_i, p_j) = 10 > 3$$
No candidate in the 800-image frozen external benchmark violates the perceptual near-duplicate boundary. All accepted samples are rigorously separated from the historical training, validation, test, and diagnostic corpora.

---

## 3. Three-Tiered Deduplication Scope Hierarchy

To ensure scientific integrity and eliminate ambiguity in publication reporting, AIDetect enforces three distinct evaluation tiers:

```
+---------------------------------------------------------------------------------------------------+
|                              THREE-TIERED FORENSIC VERIFICATION SCOPE                             |
+--------+------------------------+-------------------------+---------------------------------------+
| Tier   | Verification Level     | Tool / Method           | Acceptance / Rejection Criterion      |
+--------+------------------------+-------------------------+---------------------------------------+
| Tier 1 | Cryptographic Identity | SHA-256 (256-bit hash)  | EXACT COLLISION REJECTION (Match = 0) |
| Tier 2 | Perceptual Resemblance | 64-bit DCT pHash        | HAMMING DISTANCE REJECTION (d_H <= 3) |
| Tier 3 | Semantic / Group Scope | Metadata & Provenance   | BALANCED STRATIFICATION (400R : 400A) |
+--------+------------------------+-------------------------+---------------------------------------+
```

### Tier 1: Cryptographic Identity (Exact Byte Hash)
- **Scope:** Computed over raw image byte streams ($N=800$).
- **Cross-Referenced Against:**
  - Master V5 Dataset (60,000 images: Train 42k, Val 9k, Test 9k)
  - `dataset_v4` (4,055 images)
  - Phase 24 PIPE (1,360 images)
  - Phase 23 MagicBrush (725 images)
  - Phase 22 PIE-Bench (700 images)
  - Diagnostic smartphone captures (54 images)
  - Legacy benchmarks (1,724 images across external_test, real_world_test, manipulation_v1, manipulation_v2)
- **Result:** **0 SHA-256 collisions.** Zero duplicate file bytes exist between the external benchmark and any previous asset.

### Tier 2: Perceptual Similarity (dct pHash)
- **Scope:** 64-bit Discrete Cosine Transform perceptual hash computed on native RGB representations.
- **Hamming Distance Criterion:** $d_H \le 3$ indicates high probability of identical underlying photographic plate or crop.
- **Audit Action:** Candidate `DEF_REAL_311` (MS COCO `000000140556.jpg`) exhibited $d_H = 0$ against Phase 24 PIPE plate `pipe_orig_140556.png`. It was immediately purged and deterministically replaced by `DEF_REAL_400`.
- **Final Result:** Certified clean with observed minimum distance $d_H = 10$.

### Tier 3: Category and Source Group Scope
- **Scope:** Stratified sampling from `Rajarshi-Roy-research/Defactify_Image_Dataset` validation split (Commit `787334f7857fa54f29027a7f09c30e895ad486ef`).
- **Balanced Composition:**
  - 400 Authentic Real photographs from MS COCO photographic collection.
  - 100 Stable Diffusion 3 (SD3) text-to-image generations.
  - 100 Midjourney v6 photorealistic text-to-image generations.
  - 100 DALL-E 3 text-to-image generations.
  - 100 Stable Diffusion XL (SDXL) text-to-image generations.
- **Result:** Balanced 1:1 Real vs AI distribution across modern diffusion models unseen during V5 development.

---

## 4. Documentation Preservation

This reconciliation document is permanently archived alongside:
- [`diagnostic_outputs/external_benchmark_leakage_audit_v1.md`](../../../diagnostic_outputs/external_benchmark_leakage_audit_v1.md)
- [`diagnostic_outputs/external_benchmark_v1_manifest.csv`](../../../diagnostic_outputs/external_benchmark_v1_manifest.csv)
- [`evaluation_results/v5/strengthened_evaluation/external_metrics_extended.json`](external_metrics_extended.json)
