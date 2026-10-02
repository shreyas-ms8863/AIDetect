# AIDetect — AI-Generated Image Forensics

A multi-model deep learning system that detects AI-generated images by analyzing both **spatial pixel patterns** and **frequency-domain artifacts**. Three complementary ResNet50-based models (Spatial, Frequency, and Hybrid fusion) provide independent predictions with confidence scores, model agreement analysis, and robustness testing against common image transformations.

Built with a **FastAPI** backend serving GPU-accelerated PyTorch inference and a **React + TypeScript** frontend dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Models](#models)
- [Datasets](#datasets)
- [Experiments & Evaluation](#experiments--evaluation)
- [Results](#results)
- [Frontend](#frontend)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Checkpoints](#model-checkpoints)
- [Analysis Figures](#analysis-figures)
- [Limitations](#limitations)
- [Technical Contribution](#technical-contribution)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

AIDetect addresses the growing challenge of distinguishing real photographs from AI-generated images. The system implements three detection approaches under controlled, comparable conditions:

1. **Spatial Analysis** — Learns pixel-level texture and artifact patterns directly from RGB images.
2. **Frequency Analysis** — Extracts spectral fingerprints by applying 2D Fast Fourier Transform (FFT) before classification.
3. **Hybrid Fusion** — Combines spatial and frequency features through late fusion for a unified prediction.

The application runs all three models on every uploaded image and presents a comprehensive forensic report including individual model predictions, confidence scores, model agreement status, and robustness under JPEG compression, resize, blur, and noise transformations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Upload                               │
│                    (PNG / JPG / WebP)                             │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (GPU)                           │
│                  POST /analyze endpoint                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Image Preprocessing                         │    │
│  │  Spatial: Resize 224×224 → ToTensor → ImageNet Normalize │    │
│  │  FFT:     ToTensor → FFT2D → Shift → Log Magnitude      │    │
│  │           → Min-Max Normalize → Resize 224×224           │    │
│  │           → ImageNet Normalize                           │    │
│  └──────────┬───────────────┬───────────────┬───────────────┘    │
│             │               │               │                    │
│             ▼               ▼               ▼                    │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐      │
│  │  Spatial V3  │ │ Frequency V3 │ │    Hybrid V3        │      │
│  │  ResNet50    │ │  ResNet50    │ │ ResNet50 (spatial)   │      │
│  │  fc → 2     │ │  fc → 2     │ │ + 4-layer CNN (freq) │      │
│  │             │ │             │ │ → Concat → FC → 2    │      │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘      │
│         │                │                    │                  │
│         ▼                ▼                    ▼                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           Softmax → Predictions + Probabilities          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                          │                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │             Robustness Analysis (Hybrid V3)              │    │
│  │  Original │ JPEG Q35 │ Resize ½ │ Blur σ=2 │ Noise σ=12│    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────┬────────────────────────────────────────┘
                          │  JSON Response
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│               React + TypeScript Frontend                        │
│                                                                  │
│  • Primary prediction badge (Hybrid V3)                          │
│  • Confidence percentage + dual probability bar                  │
│  • Three-model comparison cards                                  │
│  • Model agreement indicator                                     │
│  • Robustness check dashboard (5 transformations)                │
│  • Image metadata display                                        │
└──────────────────────────────────────────────────────────────────┘
```

### API Endpoint

| Method | Endpoint    | Content-Type          | Response                                    |
|--------|-------------|-----------------------|---------------------------------------------|
| `POST` | `/analyze`  | `multipart/form-data` | JSON with predictions, probabilities, robustness |
| `GET`  | `/`         | —                     | Health check with model status              |

**Response schema** (abbreviated):

```json
{
  "filename": "photo.jpg",
  "prediction": "REAL",
  "confidence": 99.99,
  "ai_probability": 0.01,
  "real_probability": 99.99,
  "models": {
    "spatial":   { "prediction", "confidence", "ai_probability", "real_probability" },
    "frequency": { "prediction", "confidence", "ai_probability", "real_probability" },
    "hybrid":    { "prediction", "confidence", "ai_probability", "real_probability" }
  },
  "robustness": {
    "original":         { ... },
    "jpeg_compression": { ... },
    "resize":           { ... },
    "blur":             { ... },
    "noise":            { ... }
  }
}
```

---

## Models

### Currently Active: V3 (Default)

The backend defaults to **V3 models** (`AIDETECT_USE_V2=false`). V2 models remain available via environment variable.

#### Spatial V3 — `SpatialResNet50`

- **Backbone:** ResNet50 (ImageNet architecture, trained from scratch)
- **Input:** RGB image resized to 224×224, ImageNet-normalized
- **Output:** 2-class logits (Real / AI) → Softmax probabilities
- **Checkpoint:** `models/spatial_resnet50_v3.pth` (~90 MB)

#### Frequency V3 — `FrequencyResNet50`

- **Backbone:** ResNet50 (same architecture as Spatial)
- **Input:** FFT magnitude spectrum of the image:
  1. Convert to tensor at original resolution
  2. Apply 2D FFT → shift low frequencies to center
  3. Compute magnitude → log1p scaling
  4. Per-channel min-max normalization
  5. Bilinear interpolation to 224×224
  6. ImageNet normalization
- **Output:** 2-class logits → Softmax probabilities
- **Checkpoint:** `models/frequency_resnet50_v3.pth` (~90 MB)

#### Hybrid V3 — `HybridResNet50FFT`

- **Spatial branch:** ResNet50 feature extractor (all layers except final FC) → 2048-dim vector
- **Frequency branch:** Custom 4-layer CNN:
  - Conv2d(3→32) → BN → ReLU → MaxPool
  - Conv2d(32→64) → BN → ReLU → MaxPool
  - Conv2d(64→128) → BN → ReLU → MaxPool
  - Conv2d(128→256) → BN → ReLU → AdaptiveAvgPool(1×1) → 256-dim vector
- **Fusion:** Concatenation (2048 + 256 = 2304-dim)
- **Classifier:** Linear(2304→512) → ReLU → Dropout(0.3) → Linear(512→2)
- **Checkpoint:** `models/hybrid_resnet50_fft_v3.pth` (~96 MB)
- **Primary model** for the application's top-level prediction.

### Model Version History

| Version | Training Data | Training Epochs | Batch Size | LR | Optimizer |
|---------|---------------|:---:|:---:|:---:|:---:|
| **V1** | CIFAKE (20,000 images) | 3 | 16 | 1e-4 | Adam |
| **V2** | CIFAKE (20,000 images) | 3 | 16 | 1e-4 | Adam |
| **V3** | CIFAKE + GenImage + External photos (~4,165 images with augmentation) | 5 | 32 | 1e-4 | Adam |

---

## Datasets

### Training Datasets

| Dataset | Source | Usage | Notes |
|---------|--------|-------|-------|
| **CIFAKE** | `dragonintelligence/CIFAKE-image-dataset` (HuggingFace) | V1, V2: 20,000 train (10K real + 10K AI). V3: 500 real + 500 AI sampled. | 32×32 images. Real = CIFAR-10; AI = Stable Diffusion. |
| **Tiny-GenImage** | `TheKernel01/Tiny-GenImage` (HuggingFace) | V3: 800 real + 800 AI sampled from train shards. | Higher-resolution images from multiple generators. |
| **External camera photos** | `backend/external_test/REAL/` | V3: 25 used for training, 24 held out for evaluation. | Real smartphone/camera photographs. |
| **External AI images** | `backend/external_test/AI/` | V3: 18 used for training, 19 held out for evaluation. | AI-generated images from various generators. |
| **Real-world test set** | `backend/real_world_test/` | V3: 4 real + 5 AI used for training. 9 real + 10 AI for testing. | Small curated real-world validation set. |

V3 training data: **~2,099 real + ~2,066 AI = ~4,165 base images** (before augmentation). Augmentations include color jitter and horizontal flip (12× for real camera, 15× for AI images).

### Evaluation Datasets

| Dataset | Images | Usage |
|---------|:------:|-------|
| CIFAKE test split | 20,000 (10K + 10K) | Clean test accuracy for all model versions |
| 86-image external set | 86 (49 real + 37 AI) | Cross-domain generalization evaluation |
| Real-world test set | 19 (9 real + 10 AI) | Small-sample real-world validation |
| Tiny-GenImage validation | 100 (50 + 50) | Cross-dataset generalization (V2) |
| Additional 10 real + 10 AI | 20 | V3 held-out generalization check |
| Mirror selfie | 1 | Single real-photo robustness check |

> **Data leakage verification:** The V3 rigorous evaluation confirmed **zero overlap** between training and evaluation sets (`"data_leakage": "CONFIRMED_ZERO"`).

---

## Experiments & Evaluation

### 1. Clean Test Evaluation
- **What:** Standard accuracy on CIFAKE test split (20,000 images).
- **Why:** Baseline in-distribution performance.
- **Models:** V1, V2, V3 (Spatial, Frequency, Hybrid).
- **Results:** `spatial_v2_results/`, `frequency_v2_results/`, `hybrid_v2_results/`, `backend/external_results/rigorous_validation_metrics.json`.

### 2. Cross-Dataset Generalization
- **What:** Evaluation on Tiny-GenImage (100 images) and 86-image external set (49 real + 37 AI photos from diverse real-world sources).
- **Why:** Tests whether models generalize beyond CIFAKE's distribution.
- **Models:** V2 and V3.
- **Results:** `genimage_v2_results/`, `backend/external_results/`.

### 3. Real-World Validation
- **What:** Evaluation on 19 curated real-world images (9 real camera + 10 AI).
- **Why:** Tests performance on realistic user-submitted content.
- **Models:** V2.
- **Results:** `real_world_v2_results/`, `backend/app_v2_results/`.

### 4. Robustness Evaluations (V2, CIFAKE test, 20,000 images each)

| Transformation | Parameters | Results File |
|----------------|------------|-------------|
| JPEG Compression | Quality = 50 | `robustness_results/jpeg_v2_metrics.json` |
| Resize | 50% downscale + restore (112×112 intermediate) | `robustness_results/resize_v2_metrics.json` |
| Gaussian Blur | Kernel radius = 2 | `robustness_results/blur_v2_metrics.json` |
| Gaussian Noise | σ = 12 | `robustness_results/noise_v2_metrics.json` |
| JPEG Re-encoding | Quality = 90 | `robustness_results/reencoding_v2_metrics.json` |

### 5. V2 vs V3 Rigorous Comparison
- **What:** Side-by-side evaluation of all 6 models (3 V2 + 3 V3) on the same test sets.
- **Why:** Determine whether V3 genuinely improves generalization without overfitting.
- **Results:** `backend/external_results/rigorous_validation_metrics.json`, `backend/external_results/v2_vs_v3_rigorous_evaluation.csv`.

---

## Results

### Clean Test Accuracy (CIFAKE Test, 20,000 images)

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| Spatial V1 | 87.70% | 82.04% | 96.54% | 88.70% |
| Frequency V1 | 81.84% | 88.05% | 73.68% | 80.23% |
| Hybrid V1 | 83.43% | 75.52% | 98.92% | 85.65% |
| **Spatial V2** | **95.93%** | 94.39% | 97.66% | 96.00% |
| Frequency V2 | 88.75% | 91.33% | 85.63% | 88.39% |
| **Hybrid V2** | **95.72%** | 96.87% | 94.49% | 95.67% |

### V3 Clean Test Accuracy (CIFAKE subset, 200 images)

| Model | Accuracy | Precision | AI Recall | Real Recall | F1 | FPR |
|-------|:--------:|:---------:|:---------:|:-----------:|:--:|:---:|
| Spatial V3 | 91.50% | 88.07% | 96.00% | 87.00% | 91.87% | 13.0% |
| Frequency V3 | 68.00% | 87.50% | 42.00% | 94.00% | 56.76% | 6.0% |
| Hybrid V3 | 77.00% | 75.47% | 80.00% | 74.00% | 77.67% | 26.0% |

> Note: V3 models were trained on a mixed dataset (not CIFAKE-only), so reduced CIFAKE accuracy is expected; the goal was improved real-world generalization.

### Robustness Evaluation (V2, CIFAKE Test, 20,000 images)

| Condition | Spatial V2 | Frequency V2 | Hybrid V2 |
|-----------|:----------:|:------------:|:---------:|
| Clean | 95.93% | 88.75% | 95.72% |
| JPEG Q50 | 91.44% | 85.64% | 93.46% |
| Resize (½) | 95.18% | 79.96% | 94.63% |
| Blur (σ=2) | 56.27% | 51.15% | 55.61% |
| Noise (σ=12) | 85.91% | 51.27% | 90.41% |
| Re-encoding Q90 | 95.95% | 88.08% | 95.25% |

> **Key finding:** Hybrid V2 provides the most robust performance across transformations, maintaining >90% accuracy under JPEG compression, resize, and noise. All models are significantly degraded by strong Gaussian blur.

### Cross-Dataset Generalization (86-image External Set)

| Model | Accuracy | AI Recall | Real Recall | FPR |
|-------|:--------:|:---------:|:-----------:|:---:|
| Spatial V2 | 43.02% | 100.0% | 0.0% | 100.0% |
| Frequency V2 | 44.19% | 94.59% | 6.12% | 93.88% |
| Hybrid V2 | 43.02% | 100.0% | 0.0% | 100.0% |
| Spatial V3 | 51.16% | 48.65% | 53.06% | 46.94% |
| Frequency V3 | 43.02% | 35.14% | 48.98% | 51.02% |
| Hybrid V3 | 41.86% | 0.0% | 73.47% | 26.53% |

> **Key finding:** V2 models classify nearly all external images as AI-generated (100% FPR), while V3 models show more balanced but still limited accuracy. This reveals a significant **domain gap** between CIFAKE-trained models and real-world high-resolution photographs.

### Tiny-GenImage Cross-Dataset (V2, 100 images)

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| Spatial V2 | 50.00% | 50.00% | 100.0% | 66.67% |
| Frequency V2 | 58.00% | 61.76% | 42.00% | 50.00% |
| Hybrid V2 | 50.00% | 50.00% | 100.0% | 66.67% |

---

## Frontend

The frontend provides a polished forensic analysis dashboard:

- **Image upload** — Drag-and-drop zone or file picker (PNG, JPG, JPEG, WebP)
- **Analysis pipeline stepper** — 5-stage animated progress indicator:
  1. Image preprocessing
  2. Spatial V3 analysis
  3. Frequency V3 analysis
  4. Hybrid V3 fusion
  5. Robustness analysis
- **Primary prediction** — Large Hybrid V3 verdict badge with confidence percentage
- **Dual probability display** — Visual bar showing Real vs. AI probability split
- **Three-model comparison cards** — Side-by-side Spatial V3, Frequency V3, Hybrid V3 results
- **Model agreement indicator** — Dynamic check (✓ all agree / ⚠ disagree) computed from actual predictions
- **Image information card** — Filename, resolution, file type, file size (extracted client-side)
- **Robustness dashboard** — Five cards showing Hybrid V3 predictions under:
  - Original image
  - JPEG compression (quality 35)
  - Resize (50% downscale + restore)
  - Gaussian blur (radius 2)
  - Gaussian noise (σ = 12)
- **Forensic interpretation note** — Explains that outputs are probabilistic estimates
- **Responsive design** — Desktop (3-column), tablet (2-column), mobile (single-column)
- **Accessibility** — Keyboard navigation, focus states, `prefers-reduced-motion` support

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch, torchvision |
| Model Architecture | ResNet50, 2D FFT (torch.fft), Custom CNN |
| Backend | FastAPI, Python, Pillow, NumPy |
| Frontend | React 19, TypeScript 6, Vite 8.3 |
| Linting | oxlint |
| GPU | CUDA (optional, falls back to CPU) |
| Dataset Format | HuggingFace Parquet, local image directories |
| Version Control | Git |

---

## Project Structure

```
AIDetect/
├── README.md
├── requirements.txt
├── .gitignore
│
├── backend/
│   ├── main.py                          # FastAPI server, model definitions, inference
│   ├── train_spatial.py                 # V1 Spatial training
│   ├── train_spatial_v2.py              # V2 Spatial training (CIFAKE)
│   ├── train_frequency.py               # V1 Frequency training
│   ├── train_frequency_v2.py            # V2 Frequency training
│   ├── train_hybrid.py                  # V1 Hybrid training
│   ├── train_hybrid_v2.py               # V2 Hybrid training
│   ├── train_v3_robust.py               # V3 multi-dataset training (all 3 models)
│   ├── evaluate_*.py                    # Individual model evaluation scripts
│   ├── rigorous_evaluation.py           # V2 vs V3 comparative evaluation
│   ├── external_evaluate.py             # External dataset evaluation
│   ├── generate_analysis_graphs.py      # Analysis figure generation
│   ├── generate_generalization_graph.py # Generalization comparison plots
│   ├── diagnose_model.py                # Model debugging utilities
│   ├── evaluation_results/              # V1 evaluation metrics & predictions
│   ├── external_results/                # External/rigorous evaluation results
│   ├── final_analysis/                  # Master results + analysis figures
│   │   └── figures/                     # PNG figures (5 analysis plots)
│   ├── app_v2_results/                  # Application validation (V2)
│   ├── app_validation_results/          # Additional validation metrics
│   ├── external_test/                   # External test images
│   │   ├── REAL/                        # 49 real camera photographs
│   │   └── AI/                          # 37 AI-generated images
│   └── real_world_test/                 # Real-world test set
│       ├── REAL/                        # 9 real photographs
│       └── AI/                          # 10 AI images
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── App.tsx                      # Main application component
│       ├── App.css                      # Complete design system
│       ├── main.tsx                     # React entry point
│       └── index.css                    # Base styles
│
├── models/                              # Model checkpoints (.pth, git-ignored)
│   ├── spatial_resnet50_v3.pth
│   ├── frequency_resnet50_v3.pth
│   ├── hybrid_resnet50_fft_v3.pth
│   ├── *_v2.pth                         # V2 checkpoints
│   ├── *.pth                            # V1 checkpoints
│   └── *_training_metrics.json          # Training logs
│
├── spatial_results/                     # V1 Spatial evaluation
├── spatial_v2_results/                  # V2 Spatial evaluation
├── frequency_results/                   # V1 Frequency evaluation
├── frequency_v2_results/                # V2 Frequency evaluation
├── hybrid_results/                      # V1 Hybrid evaluation
├── hybrid_v2_results/                   # V2 Hybrid evaluation
├── genimage_v2_results/                 # V2 Tiny-GenImage cross-dataset
├── real_world_v2_results/               # V2 real-world evaluation
└── robustness_results/                  # V2 robustness evaluation (5 transforms)
```

---

## Installation & Setup

### Prerequisites

- **Python:** 3.12.x (reference environment tested on Python 3.12.10)
- **Node.js:** 18+ (tested with Vite 8.3 + React 19)
- **Compute:** NVIDIA GPU with CUDA recommended; CPU fallback supported
- **Trained Model Checkpoints:** Required in `models/` (see [Model Checkpoints](#model-checkpoints))

### 1. Clone the Repository

```bash
git clone https://github.com/shreyas-ms8863/AIDetect.git
cd AIDetect
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install project dependencies from root requirements.txt
pip install -r ../requirements.txt
```

> **GPU & PyTorch Note:** The tested reference environment runs **PyTorch 2.11.0 + CUDA 12.8** on an NVIDIA GeForce RTX 4050 Laptop GPU. CUDA 12.8 is not strictly mandatory for all systems. Users on CPU-only machines or different CUDA toolkits (e.g., CUDA 11.8, 12.1, 12.4) should install the corresponding PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) prior to running inference.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Place Model Checkpoints

The application expects trained weights in the root `models/` directory:

```
models/
├── spatial_resnet50_v3.pth       # ~90 MB (Default Spatial model)
├── frequency_resnet50_v3.pth     # ~90 MB (Default Frequency model)
└── hybrid_resnet50_fft_v3.pth    # ~96 MB (Default Hybrid fusion model)
```

If evaluating or testing historical V2 models, also provide:
```
models/
├── spatial_resnet50_v2.pth       # ~90 MB (V2 Spatial model)
├── frequency_resnet50_v2.pth     # ~90 MB (V2 Frequency model)
└── hybrid_resnet50_fft_v2.pth    # ~96 MB (V2 Hybrid fusion model)
```

> **Important:** Checkpoints are intentionally excluded from GitHub due to file size limits. There is currently no automated public download URL; checkpoint files must be obtained or trained locally using the scripts in `backend/`.

### 5. Start the Backend

```bash
cd backend

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Start the FastAPI server
uvicorn main:app --host 127.0.0.1 --port 8000
```

Add `--reload` if developing or debugging.

Upon startup, the server automatically loads the active **V3** models with strict tensor matching:
```
======================================================================
AIDetect V3 Backend
======================================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU

Loading V3 models...
[OK] Spatial V3 loaded with STRICT matching.
[OK] Frequency V3 loaded with STRICT matching.
[OK] Hybrid V3 loaded with STRICT matching.
======================================================================
ALL V3 MODELS LOADED SUCCESSFULLY
======================================================================
```

To switch to **V2** models instead, set `AIDETECT_USE_V2=true`:
```bash
# Windows (PowerShell)
$env:AIDETECT_USE_V2="true"
uvicorn main:app --host 127.0.0.1 --port 8000

# Windows (CMD)
set AIDETECT_USE_V2=true
uvicorn main:app --host 127.0.0.1 --port 8000

# Linux / macOS
export AIDETECT_USE_V2=true
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 6. Start the Frontend Development Server

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

### 7. Production Build (Frontend)

To build the client dashboard for production deployment:

```bash
cd frontend
npm run build
```

This runs TypeScript type checking (`tsc -b`) and Vite production bundling into `frontend/dist/`.

---

## Usage

1. Start the FastAPI backend at `http://127.0.0.1:8000`.
2. Start the Vite frontend at `http://localhost:5173`.
3. Upload an image using drag-and-drop or the file selector (supports PNG, JPG, JPEG, WebP).
4. Click **Analyze Image**.
5. Inspect the forensic dashboard:
   - **Primary verdict:** Hybrid V3 classification with confidence score.
   - **Dual probability bar:** Visual AI vs. Real likelihood split.
   - **Model agreement check:** Confirms whether all 3 models concur or flag a divergence.
   - **Multi-model breakdown:** Side-by-side Spatial, Frequency, and Hybrid outputs.
   - **Robustness assessment:** Hybrid V3 behavior across JPEG Q35, 50% Resize, Gaussian Blur, and Gaussian Noise.

---

## Model Checkpoints

Trained `.pth` checkpoint files are excluded from git tracking via `.gitignore` because of their large file sizes (~90–96 MB each, totaling ~280 MB per model generation).

Expected checkpoint files:

| File | Size | Generation | Description |
|------|:----:|:----------:|-------------|
| `spatial_resnet50_v3.pth` | ~90 MB | V3 (Active) | Spatial ResNet-50 trained on mixed datasets |
| `frequency_resnet50_v3.pth` | ~90 MB | V3 (Active) | Frequency ResNet-50 on 2D FFT spectrums |
| `hybrid_resnet50_fft_v3.pth` | ~96 MB | V3 (Active) | Dual-branch spatial + frequency fusion |
| `spatial_resnet50_v2.pth` | ~90 MB | V2 (Optional) | CIFAKE-trained spatial model |
| `frequency_resnet50_v2.pth` | ~90 MB | V2 (Optional) | CIFAKE-trained frequency model |
| `hybrid_resnet50_fft_v2.pth` | ~96 MB | V2 (Optional) | CIFAKE-trained hybrid fusion model |

> If checkpoints are missing locally, they can be re-trained using `backend/train_v3_robust.py` (for V3) or `backend/train_*_v2.py` (for V2) with access to the Hugging Face datasets.

---

## Analysis Figures

The `backend/final_analysis/figures/` directory contains five analysis plots generated from the V2 evaluation results:

| Figure | Description |
|--------|-------------|
| `figure1_clean_accuracy.png` | Clean test accuracy comparison across all V2 models |
| `figure2_robustness_accuracy.png` | Accuracy under each robustness transformation |
| `figure3_accuracy_drop.png` | Accuracy drop from clean baseline for each transformation |
| `figure4_accuracy_heatmap.png` | Heatmap of model × transformation accuracy |
| `figure5_generalization_comparison.png` | Cross-dataset generalization comparison |

These figures are generated by `backend/generate_analysis_graphs.py` and `backend/generate_generalization_graph.py` using data from `backend/final_analysis/master_results.json`.

---

## Limitations & Reproducibility Constraints

- **Model Checkpoints Not in Repository:** Trained `.pth` model checkpoints (~280 MB per version) are not tracked in GitHub due to storage limits. The application cannot perform inference without placing or training these weights first.
- **Hardware & CUDA Dependency:** GPU acceleration requires an NVIDIA GPU with compatible drivers and PyTorch build. CPU fallback is operational but will exhibit higher latency during 2D FFT extraction and inference.
- **Dataset Acquisition:** Training and evaluation datasets (CIFAKE, Tiny-GenImage) are hosted externally on Hugging Face Hub and are not packaged in the repository; they must be cached locally for retraining.
- **Experimental Metric Precision:** Exact numerical replication of floating-point test metrics depends on the exact model checkpoints, PyTorch/CUDA backend versions, and system architecture.
- **Domain Gap & Generalization:** Models trained primarily on low-resolution datasets (such as CIFAKE's 32×32 images) experience significant accuracy degradation on high-resolution smartphone and camera photographs. V2 models exhibit high false positive rates on external photos; V3 improves this balance but domain transfer remains an active research challenge.
- **Blur Sensitivity:** All architectures degrade under aggressive Gaussian blur (dropping to ~50–56% accuracy).
- **Uncalibrated Probabilities:** Output confidence percentages reflect raw softmax probabilities rather than calibrated posterior estimates.

---

## Technical Contribution

This project provides a controlled comparison of three AI-image detection approaches under identical training, evaluation, and robustness conditions:

1. **Spatial-only detection** using a standard ResNet50 on pixel data.
2. **Frequency-only detection** using ResNet50 on FFT magnitude spectra.
3. **Hybrid fusion** combining both spatial and frequency features through late concatenation.

The project evaluates all three approaches across:
- Clean in-distribution test data
- Five robustness transformations (JPEG, resize, blur, noise, re-encoding)
- Cross-dataset generalization
- Real-world photograph classification

The findings demonstrate that while hybrid fusion provides consistent advantages in robustness (particularly under JPEG compression and noise), all approaches face significant domain transfer challenges when moving from low-resolution training data to high-resolution real-world images — a critical consideration for practical deployment of AI-image detection systems.

---

## Future Work

- **Larger and more diverse training data:** Incorporate datasets with images from a wider range of generators (e.g., Midjourney, DALL-E 3, FLUX, Firefly) at native resolutions.
- **Wavelet-based frequency features:** Explore Discrete Wavelet Transform (DWT) as an alternative or complement to FFT.
- **Attention-based fusion:** Replace concatenation fusion with cross-attention or gating mechanisms.
- **Confidence calibration:** Apply temperature scaling or Platt scaling to produce well-calibrated probability estimates.
- **Adversarial robustness:** Evaluate and improve model resilience against adversarial perturbations designed to evade detection.
- **Model distillation:** Create a smaller, faster model for real-time or mobile deployment.
- **Explainability:** Add GradCAM or attention visualization to show which image regions influence predictions.

---

## Author

**Shreyas M S**

GitHub: [github.com/shreyas-ms8863](https://github.com/shreyas-ms8863)

---

## License

This project was developed as an academic/research project. This project is currently not licensed for redistribution.
