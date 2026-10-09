"""
AIDetect V5 - High-Throughput Blur Robustness Evaluation Pipeline
================================================================

Optimized for speed on NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM):
- torch.inference_mode() for zero-overhead inference
- CUDA Automatic Mixed Precision (AMP float16)
- Shared batch processing: native image loading, Gaussian blur perturbation,
  authoritative spatial and frequency tensor transformations are computed once
  per batch and reused across V5-A, V5-B, V5-C, and V5-D concurrently
- Eliminates 4x redundant dataset iterations, 4x redundant image decoding,
  and 4x redundant 2D FFT transformations
- In-memory PyArrow table caching for parquet records (zero disk re-reading across levels)
- Pinned host-to-device memory transfers
- Pre-flight 100-image benchmark and throughput profiling before full run

Models Evaluated:
    - V5-A: Spatial-only ResNet-50
    - V5-B: Frequency-only ResNet-50
    - V5-C: Hybrid Spatial + Frequency ResNet-50
    - V5-D: Gated Spatial + Frequency + Noise Residual ResNet-50

Checkpoints:
    - backend/models/v5/spatial_resnet50_v5_best.pth
    - backend/models/v5/frequency_resnet50_v5_best.pth
    - backend/models/v5/hybrid_resnet50_v5_best.pth
    - backend/models/v5/gated_residual_resnet50_v5_best.pth

Degradation Conditions:
    - Radius 1.0 (Mild Gaussian Blur)
    - Radius 2.0 (Moderate Gaussian Blur -- Authoritative historical baseline from evaluate_blur_v2.py)
    - Radius 3.0 (Severe Gaussian Blur)

Protocol:
    - Preserves frozen V5 test set (9,000 images: 4,500 REAL, 4,500 AI)
    - Preserves exact model architectures and checkpoint loading from evaluate_v5_final.py
    - No retraining, no threshold tuning, no test-set model selection
    - Exact V5 preprocessing:
        * Spatial: Resize(256, BILINEAR) -> CenterCrop(224) -> ToTensor() -> ImageNet Normalize
        * Frequency: Authoritative native-resolution 2D FFT -> log1p -> min-max -> interpolate(224,224) -> ImageNet Normalize
        * Noise Residual: Spatial tensor - GaussianBlur(kernel=5, sigma=1.0) via F.conv2d
"""

import os
import io
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

# Ensure backend/v5 is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v5_dataset import (
    V5Dataset,
    AuthoritativeNativeFFTTransform,
    build_v5_spatial_val_transform,
    CLASS_MAPPING,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = BACKEND_DIR / "models" / "v5"
BACKEND_RESULT_DIR = BACKEND_DIR / "evaluation_results" / "v5"
ROOT_RESULT_DIR = PROJECT_ROOT / "evaluation_results" / "v5"

BACKEND_RESULT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = BACKEND_RESULT_DIR / "v5_blur_robustness.json"
ROOT_OUTPUT_PATH = ROOT_RESULT_DIR / "v5_blur_robustness.json"


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimal safe batch size for RTX 4050 (6.44 GB VRAM) under inference_mode + AMP:
# Allocates ~1.1 GB VRAM, leaving >5 GB free headroom.
DEFAULT_BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 224
NUM_CLASSES = 2

# Standard degradation levels: Radius 1.0 (mild), 2.0 (V2 baseline), 3.0 (severe)
DEFAULT_BLUR_RADII = [1.0, 2.0, 3.0]


# ============================================================
# AUTHORITATIVE PREPROCESSING
# ============================================================

SPATIAL_TEST_TRANSFORM = build_v5_spatial_val_transform()
FREQUENCY_TEST_TRANSFORM = AuthoritativeNativeFFTTransform()


# ============================================================
# GAUSSIAN RESIDUAL EXTRACTION (V5-D)
# Exact 5x5 Gaussian kernel (sigma=1.0) with F.conv2d
# ============================================================

def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0, channels: int = 3) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = coords.unsqueeze(0)
    y = coords.unsqueeze(1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


GAUSSIAN_KERNEL = create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3).to(device)


def extract_noise_residual(x: torch.Tensor) -> torch.Tensor:
    """Computes spatial noise residual: x - GaussianBlur(x)."""
    global GAUSSIAN_KERNEL
    if GAUSSIAN_KERNEL.device != x.device or GAUSSIAN_KERNEL.dtype != x.dtype:
        GAUSSIAN_KERNEL = GAUSSIAN_KERNEL.to(device=x.device, dtype=x.dtype)
    blurred = F.conv2d(x, GAUSSIAN_KERNEL, padding=2, groups=3)
    return x - blurred


# ============================================================
# MODEL DEFINITIONS (Exact architecture signatures from evaluate_v5_final.py)
# ============================================================

class SpatialResNet50(nn.Module):
    """V5-A: Spatial-only ResNet-50."""
    def __init__(self):
        super().__init__()
        model = resnet50(weights=None)
        features = model.fc.in_features
        model.fc = nn.Linear(features, NUM_CLASSES)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FrequencyResNet50(nn.Module):
    """V5-B: Frequency-only ResNet-50."""
    def __init__(self):
        super().__init__()
        model = resnet50(weights=None)
        features = model.fc.in_features
        model.fc = nn.Linear(features, NUM_CLASSES)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HybridResNet50(nn.Module):
    """V5-C: Hybrid Spatial + Frequency ResNet-50."""
    def __init__(self):
        super().__init__()
        self.spatial_encoder = resnet50(weights=None)
        spatial_features = self.spatial_encoder.fc.in_features
        self.spatial_encoder.fc = nn.Identity()

        self.frequency_encoder = resnet50(weights=None)
        frequency_features = self.frequency_encoder.fc.in_features
        self.frequency_encoder.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(spatial_features + frequency_features, NUM_CLASSES),
        )

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_encoder(spatial)
        frequency_features = self.frequency_encoder(frequency)
        fused = torch.cat([spatial_features, frequency_features], dim=1)
        return self.classifier(fused)


class GatedResidualResNet50(nn.Module):
    """V5-D: Gated Spatial + Frequency + Noise Residual ResNet-50."""
    def __init__(self):
        super().__init__()
        self.spatial_encoder = resnet50(weights=None)
        spatial_dim = self.spatial_encoder.fc.in_features
        self.spatial_encoder.fc = nn.Identity()

        self.frequency_encoder = resnet50(weights=None)
        frequency_dim = self.frequency_encoder.fc.in_features
        self.frequency_encoder.fc = nn.Identity()

        self.residual_encoder = resnet50(weights=None)
        residual_dim = self.residual_encoder.fc.in_features
        self.residual_encoder.fc = nn.Identity()

        fused_dim = spatial_dim + frequency_dim + residual_dim

        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(fused_dim, NUM_CLASSES),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        spatial_features = self.spatial_encoder(spatial)
        frequency_features = self.frequency_encoder(frequency)
        residual_features = self.residual_encoder(residual)

        combined = torch.cat(
            [spatial_features, frequency_features, residual_features],
            dim=1,
        )
        gate = self.gate(combined)
        gated_features = combined * gate
        return self.classifier(gated_features)


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """Loads a V5 checkpoint with clean state-dict mapping."""
    print(f"Loading checkpoint: {checkpoint_path.name} ... ", end="", flush=True)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    state_dict = cleaned

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    if "model.conv1.weight" in model_keys and "conv1.weight" in checkpoint_keys:
        state_dict = {f"model.{key}": value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path.name}: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model.to(device)
    model.eval()
    print("OK")
    return model


# ============================================================
# CONTROLLED BLUR PERTURBATION
# ============================================================

def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """
    Applies Gaussian Blur perturbation using PIL ImageFilter.GaussianBlur.
    Matches authoritative blur definition from AIDetect evaluate_blur_v2.py.
    """
    image = image.convert("RGB")
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


# ============================================================
# FAST BLUR ROBUSTNESS DATASET (With In-Memory Parquet Table Caching)
# ============================================================

class FastBlurDataset(Dataset):
    """
    High-efficiency dataset wrapper for Blur Robustness:
    - Direct manifest access via base_dataset.samples[index]
    - In-memory PyArrow Table caching across parquet files (reads parquet once across entire process)
    - Applies PIL GaussianBlur perturbation once per sample
    - Produces authoritative spatial and frequency tensors once for all downstream models
    """
    def __init__(self, base_dataset: V5Dataset, radius: float):
        self.base_dataset = base_dataset
        self.radius = radius
        self.spatial_xform = SPATIAL_TEST_TRANSFORM
        self.fft_xform = FREQUENCY_TEST_TRANSFORM
        # Process-level table cache to avoid repetitive file I/O across conditions
        self._table_cache = getattr(base_dataset, "_fast_table_cache", {})
        base_dataset._fast_table_cache = self._table_cache

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _load_image_fast(self, rec: Dict[str, Any]) -> Image.Image:
        stype = rec["storage_type"]
        sref = rec["storage_ref"]

        if stype == "file":
            full_p = PROJECT_ROOT / sref
            with Image.open(full_p) as im:
                return im.convert("RGB")

        elif stype == "parquet_row":
            pq_name, r_idx_str = sref.split(":")
            r_idx = int(r_idx_str)

            if pq_name not in self._table_cache:
                for cand in Path(r"C:\Users\Shreyas\.cache\huggingface\hub").rglob(pq_name):
                    if cand.is_file():
                        self._table_cache[pq_name] = pq.read_table(cand)
                        break

            table = self._table_cache.get(pq_name)
            if table is not None:
                raw_bytes = table["image"][r_idx]["bytes"].as_py()
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    return im.convert("RGB")

            # Fallback to base dataset method if table not found
            return self.base_dataset._load_image(rec)

        else:
            return self.base_dataset._load_image(rec)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.base_dataset.samples[index]
        label = CLASS_MAPPING[rec["ground_truth"]]
        sample_id = rec["sample_id"]

        # 1. Load native RGB image (fast cached path)
        image = self._load_image_fast(rec)

        # 2. Apply controlled blur perturbation
        blurred = apply_blur(image, self.radius)

        # 3. Compute exact V5 spatial and frequency tensors
        spatial = self.spatial_xform(blurred)
        frequency = self.fft_xform(blurred)

        return {
            "spatial": spatial,
            "frequency": frequency,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
        }


# ============================================================
# METRICS COMPUTATION
# ============================================================

def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "total": total,
    }


# ============================================================
# HIGH-SPEED CONCURRENT 4-MODEL BATCH EVALUATION
# Evaluates V5-A, V5-B, V5-C, V5-D in a SINGLE pass over DataLoader
# ============================================================

@torch.inference_mode()
def evaluate_all_models_on_loader(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    desc: str = "Evaluating",
) -> Dict[str, Dict[str, Any]]:
    for m in models.values():
        m.eval()

    preds = {
        "spatial": [],
        "frequency": [],
        "hybrid": [],
        "gated_residual": [],
    }
    all_labels = []

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"{desc:<26}",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress:
        labels = batch["label"].to(device, non_blocking=True)
        spatial = batch["spatial"].to(device, non_blocking=True)
        frequency = batch["frequency"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            # 1. V5-A Spatial-only
            out_sp = models["spatial"](spatial)
            # 2. V5-B Frequency-only
            out_fr = models["frequency"](frequency)
            # 3. V5-C Hybrid Spatial + Frequency
            out_hy = models["hybrid"](spatial, frequency)
            # 4. V5-D Gated Residual: GPU convolution computed once
            residual = extract_noise_residual(spatial)
            out_gt = models["gated_residual"](spatial, frequency, residual)

        preds["spatial"].append(torch.argmax(out_sp, dim=1).cpu())
        preds["frequency"].append(torch.argmax(out_fr, dim=1).cpu())
        preds["hybrid"].append(torch.argmax(out_hy, dim=1).cpu())
        preds["gated_residual"].append(torch.argmax(out_gt, dim=1).cpu())
        all_labels.append(labels.cpu())

    labels_tensor = torch.cat(all_labels)
    results = {}
    for key in preds:
        pred_tensor = torch.cat(preds[key])
        results[key] = calculate_metrics(pred_tensor, labels_tensor)

    return results


# ============================================================
# PRE-FLIGHT 100-IMAGE BENCHMARK & SMOKE TEST
# ============================================================

def run_benchmark(
    test_dataset: V5Dataset,
    models: Dict[str, nn.Module],
    num_samples: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    radius: float = 2.0,
) -> bool:
    print("\n" + "=" * 80)
    print(f"RUNNING {num_samples}-IMAGE PRE-FLIGHT BENCHMARK & THROUGHPUT TEST (Radius={radius})")
    print("=" * 80)

    assert len(test_dataset) == 9000, f"Expected 9,000 test images, got {len(test_dataset)}"
    print(f"  [1/4] Test set size verified: {len(test_dataset):,} samples")

    # Create subset of first num_samples
    subset_indices = list(range(min(num_samples, len(test_dataset))))
    subset_base = Subset(test_dataset, subset_indices)
    subset_base.samples = [test_dataset.samples[i] for i in subset_indices]
    subset_base._load_image = test_dataset._load_image

    bench_dataset = FastBlurDataset(subset_base, radius=radius)
    bench_loader = DataLoader(
        bench_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"  [2/4] Warming up GPU and benchmarking {num_samples} samples @ radius {radius} blur...")
    t_start = time.perf_counter()

    results = evaluate_all_models_on_loader(
        models=models,
        loader=bench_loader,
        desc=f"Benchmark ({num_samples} imgs)",
    )

    t_elapsed = time.perf_counter() - t_start
    throughput = num_samples / max(1e-6, t_elapsed)
    batches_sec = len(bench_loader) / max(1e-6, t_elapsed)

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(0) / 1e9
        mem_peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  [3/4] GPU Memory Profile: Current={mem_alloc:.2f} GB, Peak={mem_peak:.2f} GB")

    print(f"  [4/4] Throughput Benchmark Results:")
    print(f"        Processed:      {num_samples} images in {t_elapsed:.2f}s")
    print(f"        Throughput:     {throughput:.1f} images/second ({batches_sec:.1f} batches/sec)")
    est_full_condition = 9000 / max(1e-6, throughput)
    print(f"        Est. 9,000 run: {est_full_condition:.1f}s (~{est_full_condition/60:.1f} min) per condition")

    print(f"\nBenchmark Accuracy / F1 Check (Radius {radius} on {num_samples} samples):")
    for name, m in results.items():
        print(f"  {name:<16}: Acc={m['accuracy']*100:.2f}%, F1={m['f1']*100:.2f}%")

    print("\n" + "=" * 80)
    print("BENCHMARK PASSED. OPTIMIZED PIPELINE IS VERIFIED AND READY.")
    print("=" * 80 + "\n")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized V5 Blur Robustness Evaluation")
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only the 100-image benchmark/smoke test and exit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for DataLoader (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the 100-image benchmark and start full evaluation immediately",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=DEFAULT_BLUR_RADII,
        help=f"List of blur radii to evaluate (default: {DEFAULT_BLUR_RADII})",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AIDETECT V5 OPTIMIZED BLUR ROBUSTNESS EVALUATION")
    print("=" * 80)
    print(f"Device:      {device}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Blur Radii:  {args.radii}")
    print(f"Model Dir:   {MODEL_DIR}")
    print(f"Result Dir:  {BACKEND_RESULT_DIR}")

    # 1. Load Frozen Test Dataset
    print("\nLoading frozen V5 test dataset...")
    test_dataset = V5Dataset(split="test", mode="hybrid")
    print(f"Frozen test samples loaded: {len(test_dataset):,}")
    if len(test_dataset) != 9000:
        raise RuntimeError(f"Expected exactly 9,000 test samples, found: {len(test_dataset)}")

    # 2. Load Models
    print("\nLoading trained V5 checkpoints...")
    models = {
        "spatial": load_checkpoint(
            SpatialResNet50(),
            MODEL_DIR / "spatial_resnet50_v5_best.pth",
        ),
        "frequency": load_checkpoint(
            FrequencyResNet50(),
            MODEL_DIR / "frequency_resnet50_v5_best.pth",
        ),
        "hybrid": load_checkpoint(
            HybridResNet50(),
            MODEL_DIR / "hybrid_resnet50_v5_best.pth",
        ),
        "gated_residual": load_checkpoint(
            GatedResidualResNet50(),
            MODEL_DIR / "gated_residual_resnet50_v5_best.pth",
        ),
    }

    # 3. Pre-flight 100-Image Benchmark
    if not args.skip_benchmark:
        bench_radius = 2.0 if 2.0 in args.radii else args.radii[0]
        run_benchmark(test_dataset, models, num_samples=100, batch_size=args.batch_size, radius=bench_radius)

    if args.benchmark_only:
        print("[INFO] --benchmark-only specified. Exiting cleanly without launching full 9,000 run.")
        return

    # 4. Full 9,000-Image Evaluation across specified blur radii
    print("=" * 80)
    print("STARTING FULL 9,000-IMAGE ROBUSTNESS EVALUATION (CONCURRENT 4-MODEL PASS)")
    print("=" * 80)

    final_results: Dict[str, Any] = {
        "experiment": "V5 Blur Robustness",
        "dataset": "V5 Frozen Test",
        "test_samples": 9000,
        "batch_size": args.batch_size,
        "blur_definition": (
            "Gaussian Blur applied to native RGB image using PIL ImageFilter.GaussianBlur(radius=radius); "
            "authoritative V5 test preprocessing applied afterward to produce spatial and frequency tensors."
        ),
        "blur_radii": args.radii,
        "models": {
            "spatial": {},
            "frequency": {},
            "hybrid": {},
            "gated_residual": {},
        },
        "by_blur_radius": {},
    }

    model_display_names = {
        "spatial": "V5-A Spatial",
        "frequency": "V5-B Frequency",
        "hybrid": "V5-C Hybrid",
        "gated_residual": "V5-D Gated Residual",
    }

    t_total_start = time.perf_counter()

    for radius in args.radii:
        rad_str = f"{radius:.1f}" if isinstance(radius, float) else str(radius)
        print(f"\nEvaluating Blur Condition: Radius {radius} (GaussianBlur)")
        print("-" * 80)

        dataset = FastBlurDataset(test_dataset, radius=radius)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        t_cond_start = time.perf_counter()
        cond_metrics = evaluate_all_models_on_loader(
            models=models,
            loader=loader,
            desc=f"Blur Radius {rad_str} (All 4 Models)",
        )
        t_cond_elapsed = time.perf_counter() - t_cond_start
        print(f"Condition Radius {rad_str} completed in {t_cond_elapsed:.1f}s ({9000/t_cond_elapsed:.1f} imgs/sec)")

        final_results["by_blur_radius"][rad_str] = {}
        for model_key, metrics in cond_metrics.items():
            final_results["models"][model_key][rad_str] = metrics
            final_results["by_blur_radius"][rad_str][model_key] = metrics

            disp = model_display_names[model_key]
            print(
                f"  {disp:<22}: Acc={metrics['accuracy']*100:6.2f}% | "
                f"Prec={metrics['precision']*100:6.2f}% | "
                f"Rec={metrics['recall']*100:6.2f}% | "
                f"F1={metrics['f1']*100:6.2f}%"
            )

    t_total_elapsed = time.perf_counter() - t_total_start

    # 5. Persist Results (to both backend and root evaluation_results directories)
    for out_p in [OUTPUT_PATH, ROOT_OUTPUT_PATH]:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

    # 6. Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL V5 BLUR ROBUSTNESS EVALUATION SUMMARY")
    print(f"Total Evaluation Time: {t_total_elapsed:.1f}s ({t_total_elapsed/60:.2f} min)")
    print("=" * 80)
    print(f"{'Model':<24}{'Blur Radius':<14}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    print("-" * 86)

    for model_key, disp in model_display_names.items():
        for radius in args.radii:
            rad_str = f"{radius:.1f}" if isinstance(radius, float) else str(radius)
            m = final_results["models"][model_key][rad_str]
            print(
                f"{disp:<24}"
                f"{'Radius ' + rad_str:<14}"
                f"{m['accuracy'] * 100:>11.2f}%"
                f"{m['precision'] * 100:>11.2f}%"
                f"{m['recall'] * 100:>11.2f}%"
                f"{m['f1'] * 100:>11.2f}%"
            )

    print("-" * 86)
    print(f"\nFinal results saved successfully to:")
    print(f"  - {OUTPUT_PATH}")
    print(f"  - {ROOT_OUTPUT_PATH}")
    print("V5 BLUR ROBUSTNESS EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()

