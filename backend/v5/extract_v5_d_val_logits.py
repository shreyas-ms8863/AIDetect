"""
AIDetect V5-D Validation Logit Extraction Pipeline
===================================================
Phase 1: Deterministic Logit Extraction for Post-Hoc Probability Calibration.

SAFETY INVARIANTS:
1. Frozen V5-D model weights loaded strictly in read-only evaluation mode.
2. Evaluates ONLY the 9,000-image V5 VALIDATION split (4,500 REAL, 4,500 AI).
3. ZERO access to the frozen 9,000-image V5 TEST split.
4. Uses authoritative V5 preprocessing (Multi-Expert: Spatial + Native FFT + Noise Residual).
5. Deterministic sample ordering preserved.

Outputs:
  backend/evaluation_results/v5/v5_d_val_logits.npz
Containing:
  - sample_id: array of string sample IDs
  - label: array of ground-truth integer labels (0=REAL, 1=AI)
  - z_real: array of raw model logits for class REAL
  - z_ai: array of raw model logits for class AI
  - raw_ai_probability: array of uncalibrated softmax probabilities for AI
  - raw_real_probability: array of uncalibrated softmax probabilities for REAL
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add backend/v5 to sys.path
V5_DIR = Path(__file__).resolve().parent
BACKEND_DIR = V5_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from v5_dataset import V5Dataset
from v5_models import GatedResidualResNet50


# ==============================================================================
# NOISE RESIDUAL EXTRACTION (AUTHORITATIVE V5 GAUSSIAN FILTER)
# ==============================================================================

def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0, channels: int = 3) -> torch.Tensor:
    """Fixed 5x5 Gaussian blur kernel for noise residual extraction."""
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = coords.unsqueeze(0)
    y = coords.unsqueeze(1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


# ==============================================================================
# MODEL LOADER
# ==============================================================================

def load_v5_d_checkpoint(checkpoint_path: Path, device: torch.device) -> GatedResidualResNet50:
    """Load frozen V5-D GatedResidualResNet50 checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V5-D checkpoint not found at: {checkpoint_path}")

    print(f"[CHECKPOINT] Loading V5-D from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model = GatedResidualResNet50(num_classes=2)
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    print("[CHECKPOINT] V5-D model loaded successfully and frozen (requires_grad=False).")
    return model


# ==============================================================================
# MAIN EXTRACTION ROUTINE
# ==============================================================================

def extract_validation_logits(batch_size: int = 32, device_name: str = "auto"):
    # 1. Device selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 75)
    print("AIDetect V5-D — Validation Logit Extraction")
    print("=" * 75)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA AMP enabled: True")

    # 2. Strict Dataset Validation
    print("\n[DATASET] Initializing V5Dataset with split='validation', mode='multi_expert'...")
    val_dataset = V5Dataset(
        split="validation",
        mode="multi_expert"
    )

    n_samples = len(val_dataset)
    print(f"[DATASET] Validation samples loaded: {n_samples:,}")

    # Enforce research safety invariant: exactly 9,000 validation images
    if n_samples != 9000:
        raise ValueError(
            f"Expected exactly 9,000 validation images, but dataset returned {n_samples}."
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 3. Model & Gaussian Filter Setup
    checkpoint_path = BACKEND_DIR / "models" / "v5" / "gated_residual_resnet50_v5_best.pth"
    model = load_v5_d_checkpoint(checkpoint_path, device)

    gaussian_kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3).to(device)

    # 4. Inference Loop
    sample_ids = []
    labels = []
    z_reals = []
    z_ais = []
    raw_ais = []
    raw_reals = []

    print("\n[EXTRACTION] Running frozen V5-D forward passes under torch.inference_mode()...")
    start_time = time.time()
    total_batches = len(val_loader)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader, start=1):
            batch_ids = batch["sample_id"]
            batch_labels = batch["label"]
            spatial = batch["spatial"].to(device, non_blocking=True)
            frequency = batch["frequency"].to(device, non_blocking=True)

            # Authoritative noise residual extraction
            blurred = F.conv2d(spatial, gaussian_kernel, padding=2, groups=3)
            residual = spatial - blurred

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(spatial, frequency, residual)

            probs = F.softmax(logits, dim=1)

            sample_ids.extend(batch_ids)
            labels.append(batch_labels.cpu().numpy())
            z_reals.append(logits[:, 0].float().cpu().numpy())
            z_ais.append(logits[:, 1].float().cpu().numpy())
            raw_reals.append(probs[:, 0].float().cpu().numpy())
            raw_ais.append(probs[:, 1].float().cpu().numpy())

            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                processed = min(batch_idx * batch_size, n_samples)
                speed = processed / elapsed if elapsed > 0 else 0
                print(
                    f"\rProgress: {processed:,}/{n_samples:,} samples "
                    f"[{processed / n_samples * 100:.1f}%] — "
                    f"{speed:.1f} samples/sec — "
                    f"Elapsed: {elapsed:.1f}s",
                    end="",
                    flush=True
                )

    print()
    total_elapsed = time.time() - start_time
    print(f"\n[EXTRACTION] Completed in {total_elapsed:.2f} seconds ({n_samples / total_elapsed:.1f} samples/sec).")

    # 5. Concatenate arrays
    all_sample_ids = np.array(sample_ids)
    all_labels = np.concatenate(labels, axis=0).astype(np.int64)
    all_z_real = np.concatenate(z_reals, axis=0).astype(np.float32)
    all_z_ai = np.concatenate(z_ais, axis=0).astype(np.float32)
    all_raw_real = np.concatenate(raw_reals, axis=0).astype(np.float32)
    all_raw_ai = np.concatenate(raw_ais, axis=0).astype(np.float32)

    # 6. Verify class counts and invariants
    n_real = int(np.sum(all_labels == 0))
    n_ai = int(np.sum(all_labels == 1))

    print("\n[VERIFICATION] Validation Set Audit:")
    print(f"  Total Samples: {len(all_labels):,}")
    print(f"  REAL (0):      {n_real:,}")
    print(f"  AI (1):        {n_ai:,}")

    if n_real != 4500 or n_ai != 4500:
        raise ValueError(
            f"Class balance assertion failed: expected 4,500 REAL and 4,500 AI, got {n_real} REAL and {n_ai} AI."
        )

    # 7. Save output
    output_dir = BACKEND_DIR / "evaluation_results" / "v5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "v5_d_val_logits.npz"

    np.savez_compressed(
        output_file,
        sample_id=all_sample_ids,
        label=all_labels,
        z_real=all_z_real,
        z_ai=all_z_ai,
        raw_ai_probability=all_raw_ai,
        raw_real_probability=all_raw_real
    )

    print(f"\n[OUTPUT] Saved compressed validation logits to:")
    print(f"  {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 75)
    print("Phase 1 Complete: Validation Logits Extracted Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract V5-D Validation Logits")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    extract_validation_logits(batch_size=args.batch_size, device_name=args.device)

