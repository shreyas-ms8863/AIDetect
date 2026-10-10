# AIDetect V5 Reproducibility Manifest

**Scope:** Frozen V5 evaluation package only. This manifest describes the
inputs and procedures needed to reproduce the recorded analysis. It does not
authorize retraining, checkpoint replacement, test-set replacement, benchmark
replacement, or threshold changes.

## Runtime provenance

The repository declares the following reference environment in `README.md` and
`requirements.txt`:

| Component | Repository-declared reference |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| OS | Windows 11 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |

The final audit runtime itself was Windows 11 with Python 3.13.15, but the
evaluation dependencies were not installed in that audit runtime; NumPy,
pandas, PyTorch, and torchvision could therefore not be queried from
the active interpreter. The values above are repository-declared reference
values, not newly measured runtime values.

## Frozen model and calibration inputs

| Model / input | Repository path | SHA-256 |
|---|---|---|
| V5-A Spatial checkpoint | `backend/models/v5/spatial_resnet50_v5_best.pth` | `2cd55cd1ae94f15e350679d5b122ce725f9c421e16e73709c8620411fa971a9d` |
| V5-B Frequency checkpoint | `backend/models/v5/frequency_resnet50_v5_best.pth` | `daabc67eb2dcef38c95d7d04253beb4c13fca63285419834dc0e840cb2b59ab7` |
| V5-C Hybrid checkpoint | `backend/models/v5/hybrid_resnet50_v5_best.pth` | `d3383324b97d6db8f5af7aa04849a0320f9e2da0bc22317b4c2416e088b943ff` |
| V5-D Gated Residual checkpoint | `backend/models/v5/gated_residual_resnet50_v5_best.pth` | `1dfe03f78483e4eddd13869760405c17d2332d6d01e5d5d13ee5de5d9a5617ac` |
| V5-D calibration JSON | `backend/models/v5/v5_d_calibration.json` | `bfeab46909095fe635c6372f60d05fa326a5f9e12ebfc708feca598eaf9bc236` |

Model identities:

- V5-A: spatial RGB ResNet-50.
- V5-B: native-resolution FFT log-magnitude frequency ResNet-50.
- V5-C: spatial/frequency hybrid ResNet-50.
- V5-D: spatial/frequency/5x5 Gaussian residual gated fusion model.

V5-D uses frozen temperature scaling with **T = 2.1983866642849734**.

## Frozen datasets, manifests, and prediction artifacts

| Input/artifact | Repository path | SHA-256 |
|---|---|---|
| V5 dataset manifest | `v5_metadata/v5_dataset_manifest.csv` | `38dab8c65593e1ab2baf8f4384bf1407e66bd62a008ff9e096139eb96eaf436d` |
| Frozen clean-test predictions | `evaluation_results/v5/strengthened_evaluation/v5_clean_test_predictions.csv` | `ae60f17182bad358b2dfeacf04d8b46d671e1fabe16e5560aaa2f42277224d7d` |
| Defactify manifest | `diagnostic_outputs/external_benchmark_v1_manifest.csv` | `65f7c7d985448f1ee76800cfc1aa4127724af2a54158b91ec8495a3ba3e4d08` |
| Defactify raw per-image predictions | `diagnostic_outputs/external_benchmark_v1_results/external_benchmark_v1_per_image.csv` | `1cedc5b4a49133da27f07e56ee02159ea1d755c87ef73abe45faa3084a0b6b35` |

The clean test is the `test` split in the V5 dataset manifest: 9,000 images,
4,500 REAL and 4,500 AI. The Defactify benchmark contains 800 images, 400
REAL and 400 AI, with 100 AI images each from SD3, Midjourney v6, DALL-E 3,
and SDXL.

Required external/local files are the four V5 checkpoints, the V5 dataset
assets referenced by the manifest, the Defactify manifest/payloads, and the
raw Defactify prediction CSV. The raw payload directory is intentionally kept
outside normal Git tracking.

## Evaluation configuration

| Setting | Frozen value |
|---|---|
| Classification threshold | 0.50 |
| Bootstrap count | 1,000 resamples |
| Bootstrap type | Ordinary image-level sampling with replacement; unstratified |
| Bootstrap seed | 42 |
| Paired bootstrap | Same resample indices across compared models |
| Clean test | Frozen V5 test split, N=9,000 |
| External benchmark | Frozen Defactify v1, N=800 |

## Preprocessing

Spatial preprocessing is RGB conversion, resize to 256 pixels, center crop to
224x224, tensor conversion, and ImageNet normalization with mean
`[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Frequency preprocessing converts the native-resolution RGB image to a tensor,
applies a 2-D FFT and centered FFT shift, computes `log1p(abs(FFT))`, performs
per-channel min-max normalization, resizes the result to 224x224 with
bilinear interpolation, and applies the same ImageNet normalization.

V5-D additionally forms a 5x5 sigma-1 Gaussian residual from the spatial
tensor and applies the frozen temperature to its logits before calculating
calibrated probabilities.

## Intentionally not stored in Git

- The four large `.pth` model checkpoints.
- The V5 source image/dataset assets referenced by the manifest.
- Raw Defactify image payloads.
- Local Hugging Face parquet cache files.
- Local virtual environments, package caches, and generated per-image clean
  prediction exports.

The scientific summary reports and aggregate CSV/JSON artifacts are intended
to remain versionable. The frozen per-image clean prediction export is
explicitly ignored as a generated local artifact; its recorded SHA-256 is
preserved above and in `strengthened_evaluation/final_consistency_check.md`.
