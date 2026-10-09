# AIDetect External Benchmark Pre-Inference Leakage Audit Report

**Document Identifier:** `AIDETECT-AUDIT-EXTERNAL-BENCHMARK-V1`  
**Date:** 2026-10-09 16:32:42 UTC  
**Auditor:** AIDetect Core Forensic Research Team  
**Evaluation Target:** Defactify_Image_Dataset (MS COCOAI) Validation Split  
**Status:** Certified Zero-Leakage Pre-Inference Benchmark Freeze  

---

## 1. Executive Summary

In accordance with strict research governance protocols, the proposed 800-image external evaluation benchmark for the frozen **AIDetect V5** detector suite has completed its pre-inference identity, cryptographic, and perceptual deduplication audit.

- **Dataset Source:** `Rajarshi-Roy-research/Defactify_Image_Dataset`
- **Repository Commit:** `787334f7857fa54f29027a7f09c30e895ad486ef`
- **Split Ingested:** `validation` split (9,000 total images; 4,500 Real / 4,500 AI)
- **Selection Seed:** `42` (deterministic multi-class stratification)
- **Zero Model Inference:** **No model inference (V5-A, V5-B, V5-C, V5-D) was executed.** Checkpoints, thresholds, and calibration remain 100% frozen.

```
+---------------------------------------------------------------------------------------------------------------+
|                                      PRE-INFERENCE LEAKAGE AUDIT TOTALS                                       |
+----------------------------------------------------+--------------------------+-------------------------------+
| Audit Metric                                       | Metric Value             | Status / Interpretation       |
+----------------------------------------------------+--------------------------+-------------------------------+
| Total Candidates Initially Selected                | 800                      | Exact 400 Real / 400 AI target|
| Total Candidates Ingested & Screened               | 801                      | Screened through priority queue|
| Exact SHA-256 Collisions Identified                | 0                        | Immediate hard rejection      |
| Suspicious Perceptual Near-Duplicates (d_H <= 3)   | 1                        | Conservatively rejected       |
| Total Candidates Rejected & Purged                 | 1                        | Replaced from backup pool     |
| Total Deterministic Replacements Drawn             | 1                        | Re-audited and verified clean |
| Unresolved Forensic Cases                          | 0                        | Zero ambiguous entries        |
| Final Certified Benchmark Population               | 800                      | 100% Verified Clean           |
| Final REAL / AI Ratio                              | 400 REAL : 400 AI        | Perfectly Balanced (50%/50%)   |
| Ready for Benchmark Freeze                         | YES                      | Pre-inference certified       |
+----------------------------------------------------+--------------------------+-------------------------------+
```

---

## 2. Certified Composition & Generator Breakdown

The final certified benchmark contains exactly **800 images**, distributed across authentic photography and four distinct state-of-the-art generative architectures:

```
+---------------------------------------------------------------------------------------------------------------+
|                                      FINAL CERTIFIED BENCHMARK COMPOSITION                                    |
+-------------------+---------+----------------------------+-----------------------+----------------------------+
| Stratum / Class   | Samples | Generator / Model          | Architecture Family   | Source Provenance          |
+-------------------+---------+----------------------------+-----------------------+----------------------------+
| REAL (Authentic)  | 400     | None_Authentic_Photograph  | Natural Camera Scene  | MS COCO Photographic Set   |
| AI - SD3          | 100     | Stable_Diffusion_3         | MMDiT Diffusion Trans | Defactify T2I Generation   |
| AI - Midjourney   | 100     | Midjourney_v6              | Commercial Photoreal  | Defactify T2I Generation   |
| AI - DALL-E 3     | 100     | DALL-E_3                   | Autoregressive Diff   | Defactify T2I Generation   |
| AI - SDXL         | 100     | Stable_Diffusion_XL        | Dual-Encoder Latent   | Defactify T2I Generation   |
+-------------------+---------+----------------------------+-----------------------+----------------------------+
| TOTAL BENCHMARK   | 800     | 400 REAL : 400 AI          | Balanced Multi-Gen    | 100% Certified Clean       |
+-------------------+---------+----------------------------+-----------------------+----------------------------+
```

---

## 3. Auditing Reference Ledgers

The 800 candidates were cross-referenced against the complete AIDetect historical asset ledger:
1. **Master V5 Dataset:** All 60,000 images from `v5_metadata/v5_dataset_manifest.csv` (Train: 42,000; Val: 9,000; Frozen Test: 9,000).
2. **dataset_v4:** 4,055 images (Train, Val, Test).
3. **Phase 24 PIPE:** 1,360 images (680 Real, 680 SD 1.5 inpainting).
4. **Phase 23 MagicBrush:** 725 images (266 Real, 459 DALL-E 2 inpainting).
5. **Phase 22 PIE-Bench:** 700 images (authentic and inpainted).
6. **Diagnostic Real Set:** 54 independent smartphone camera photographs from `diagnostic_real/`.
7. **Legacy Benchmarks:** `backend/external_test/` (86 images), `backend/real_world_test/` (19 images), `manipulation_v1/` (1,619 images), and `dataset_manipulation_v2/` (2,078 images).

---

## 4. Rejection and Replacement Ledger

To ensure zero overlap with our historical inpainting studies (which utilized MS COCO source images), any candidate exhibiting an exact SHA-256 match or a perceptual hash Hamming distance $d_H \le 3$ with any historical asset was rejected and deterministically replaced from the remaining validation partition pool:

Total Candidates Rejected: **1**  
Total Replacements Drawn & Verified: **1**

| Rejected Candidate | Category | Global Index | Rejection Reason | Deterministic Replacement | Replacement Distance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `DEF_REAL_311` | REAL | `2603` | NEAR_DUPLICATE_SOURCE_OVERLAP ($d=0$) with PIPE: `pipe_orig_140556.png` | `DEF_REAL_400` (`global_idx=5965`) | $\min(d_H) = 16$ (Certified Clean) |

### Rejection Forensics
Candidate `DEF_REAL_311` corresponds to MS COCO image `000000140556.jpg`. While its file compression differed slightly from our local archive (hence no raw SHA-256 collision), the 64-bit DCT perceptual hash produced an exact match ($d_H = 0$) with the original background plate utilized in our Phase 24 PIPE inpainting evaluation (`pipe_orig_140556.png`). The automated audit immediately purged this candidate and drew the next deterministic candidate from the seed-42 priority queue (`global_idx=5965`), which exhibited zero proximity to any historical asset ($\min(d_H) = 16$).


---

## 5. Payload Integrity & File Storage

- **Preservation of Raw Bytes:** All 800 image payloads were extracted directly from the Parquet byte buffers without re-compression or transformation and saved to:  
  `diagnostic_outputs/external_benchmark_payloads/`
- **Naming Standard:** `DEF_<CATEGORY>_<NUM>_<SHA12>.<EXT>`
- **Cryptographic Manifest:** The final frozen manifest [`diagnostic_outputs/external_benchmark_v1_manifest.csv`](file:///c:/Users/Shreyas/OneDrive/Desktop/AIDetect/diagnostic_outputs/external_benchmark_v1_manifest.csv) records per-sample candidate ID, original filename, category, source, generator, label, ground truth integer, original dataset identifier, caption, width, height, byte size, SHA-256 hash, and 64-bit DCT pHash.

---

## 6. Pre-Inference Governance Certification

1. **Model Weights Unmodified:** No checkpoints were altered.
2. **Frozen V5 Test Set Unmodified:** The frozen 9,000-image test set remains untouched.
3. **No Inference Executed:** Neither V5-A, V5-B, V5-C, nor V5-D has been executed against this benchmark.
4. **Ready for Freeze:** The external evaluation benchmark is certified clean, balanced, and ready to be frozen upon user authorization.
