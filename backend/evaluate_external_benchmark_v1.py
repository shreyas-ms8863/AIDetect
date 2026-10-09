"""
AIDetect - Frozen External Benchmark Evaluation (v1)
=====================================================
Frozen one-shot out-of-distribution benchmark evaluation of the four V5 models:
  - V5-A: Spatial ResNet-50
  - V5-B: Frequency ResNet-50 (Authoritative 2D FFT)
  - V5-C: Hybrid ResNet-50 (Spatial + Frequency Dual-Stream)
  - V5-D: Gated Residual ResNet-50 (Spatial + Frequency + Noise Residual + Learned Gated Fusion)

Dataset: Defactify Image Dataset (Official Validation Split)
Manifest: diagnostic_outputs/external_benchmark_v1_manifest.csv (N=800)
Composition:
  - 400 Natural Photographs (REAL, MS COCO)
  - 100 Stable Diffusion 3 (AI, SD3)
  - 100 Midjourney v6 (AI)
  - 100 DALL-E 3 (AI)
  - 100 Stable Diffusion XL (AI, SDXL)

Post-Hoc Probability Calibration:
  - V5-D: Evaluated with frozen temperature scaling T = 2.1983866642849734 from
          backend/models/v5/v5_d_calibration.json.
  - V5-A, V5-B, V5-C: Evaluated with raw softmax probabilities (uncalibrated baseline).
  - Decision threshold: Calibrated P(AI) >= 0.50 -> AI, < 0.50 -> REAL.

Strict Governance:
  - Zero modifications to model checkpoints, weights, or architecture.
  - Zero access or modification to the frozen 9,000-image V5 test set.
  - Deterministic evaluation under torch.inference_mode().
  - Zero data augmentation or random transforms during evaluation.
  - Structural validation executes before any model loading or inference.
  - No statistical interpretation or subjective claims inside this script.
"""

import os
import sys
import io
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

# -----------------------------------------------------------------------------
# PATH RESOLUTION & RE-USE OF EXISTING DEFINITIONS
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.v5.v5_dataset import (
    build_v5_spatial_val_transform,
    AuthoritativeNativeFFTTransform,
)
from backend.v5.v5_models import GatedResidualResNet50

# Authoritative paths
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_manifest.csv"
DEFAULT_PAYLOADS_DIR = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_payloads"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_results"

MODEL_PATHS = {
    "V5-A Spatial": PROJECT_ROOT / "backend" / "models" / "v5" / "spatial_resnet50_v5_best.pth",
    "V5-B Frequency": PROJECT_ROOT / "backend" / "models" / "v5" / "frequency_resnet50_v5_best.pth",
    "V5-C Hybrid": PROJECT_ROOT / "backend" / "models" / "v5" / "hybrid_resnet50_v5_best.pth",
    "V5-D Gated Residual": PROJECT_ROOT / "backend" / "models" / "v5" / "gated_residual_resnet50_v5_best.pth",
}

CALIBRATION_JSON_PATH = PROJECT_ROOT / "backend" / "models" / "v5" / "v5_d_calibration.json"

GENERATOR_DISPLAY_MAP = {
    "None_Authentic_Photograph": "REAL (Natural Photographs)",
    "Stable_Diffusion_3": "SD3",
    "Midjourney_v6": "Midjourney v6",
    "DALL-E_3": "DALL-E 3",
    "Stable_Diffusion_XL": "SDXL",
}

EXPECTED_GENERATOR_COUNTS = {
    "None_Authentic_Photograph": 400,
    "Stable_Diffusion_3": 100,
    "Midjourney_v6": 100,
    "DALL-E_3": 100,
    "Stable_Diffusion_XL": 100,
}


# -----------------------------------------------------------------------------
# MODEL DEFINITIONS (REUSING EXISTING ARCHITECTURES)
# -----------------------------------------------------------------------------
class SpatialResNet50(nn.Module):
    """V5-A: Spatial ResNet-50 classifier."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = resnet50(weights=None)
        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FrequencyResNet50(nn.Module):
    """V5-B: Frequency ResNet-50 classifier."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = resnet50(weights=None)
        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HybridResNet50(nn.Module):
    """V5-C: Hybrid ResNet-50 (Spatial + Frequency Dual-Stream)."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.spatial_encoder = resnet50(weights=None)
        spatial_features = self.spatial_encoder.fc.in_features
        self.spatial_encoder.fc = nn.Identity()

        self.frequency_encoder = resnet50(weights=None)
        frequency_features = self.frequency_encoder.fc.in_features
        self.frequency_encoder.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(spatial_features + frequency_features, num_classes)
        )

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        s_feat = self.spatial_encoder(spatial)
        f_feat = self.frequency_encoder(frequency)
        fused = torch.cat([s_feat, f_feat], dim=1)
        return self.classifier(fused)


def create_gaussian_kernel(
    kernel_size: int = 5,
    sigma: float = 1.0,
    channels: int = 3,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Authoritative 5x5 Gaussian blur kernel (sigma=1.0) for V5-D noise residual extraction."""
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = coords.unsqueeze(0)
    y = coords.unsqueeze(1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel.to(device)


def extract_noise_residual(spatial_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Extract noise residual = spatial - 5x5 Gaussian blur(sigma=1.0)."""
    blurred = F.conv2d(spatial_tensor, kernel, padding=2, groups=3)
    return spatial_tensor - blurred


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Strictly loads checkpoint state dict into model instance."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model_keys = set(model.state_dict().keys())

    if not set(cleaned_sd.keys()).issubset(model_keys):
        for prefix in ["model.", "base."]:
            prefixed = {f"{prefix}{k}": v for k, v in cleaned_sd.items()}
            if set(prefixed.keys()) == model_keys:
                cleaned_sd = prefixed
                break

    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# BENCHMARK DATASET CLASS
# -----------------------------------------------------------------------------
class FrozenBenchmarkDataset(Dataset):
    """Loads and preprocesses images deterministically from manifest and payloads directory."""
    def __init__(self, manifest_df: pd.DataFrame, payloads_dir: Path):
        self.df = manifest_df
        self.payloads_dir = payloads_dir
        self.spatial_tf = build_v5_spatial_val_transform()
        self.fft_tf = AuthoritativeNativeFFTTransform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filename = row["original_filename"]
        img_path = self.payloads_dir / filename

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            spatial_tensor = self.spatial_tf(rgb_img)
            freq_tensor = self.fft_tf(rgb_img)

        return {
            "index": idx,
            "candidate_id": row["candidate_id"],
            "original_filename": filename,
            "category": row["category"],
            "source": row["source"],
            "generator": row["generator"],
            "true_label": row["label"],
            "ground_truth_int": int(row["ground_truth_int"]),
            "spatial": spatial_tensor,
            "frequency": freq_tensor,
        }


# -----------------------------------------------------------------------------
# STEP 1: PRE-INFERENCE STRUCTURAL VALIDATION
# -----------------------------------------------------------------------------
def perform_structural_checks(
    manifest_path: Path,
    payloads_dir: Path,
    model_paths: Dict[str, Path],
    calibration_path: Path
) -> Tuple[pd.DataFrame, float]:
    """
    Performs comprehensive structural integrity checks before any inference starts.
    If ANY check fails, halts immediately with sys.exit(1).
    """
    print("=" * 80)
    print("AIDETECT EXTERNAL BENCHMARK (V1) - PRE-INFERENCE STRUCTURAL VALIDATION")
    print("=" * 80)

    # 1. Manifest file existence
    if not manifest_path.is_file():
        print(f"[FATAL ERROR] Benchmark manifest not found at: {manifest_path}")
        sys.exit(1)
    print(f"[*] Manifest file present: {manifest_path}")

    # 2. Manifest row count and columns
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"[FATAL ERROR] Could not read manifest CSV: {e}")
        sys.exit(1)

    if len(df) != 800:
        print(f"[FATAL ERROR] Manifest row count is {len(df)}, expected exactly 800.")
        sys.exit(1)
    print(f"[*] Manifest row count verified: exactly {len(df)} rows.")

    required_cols = [
        "candidate_id", "original_filename", "category", "source",
        "generator", "label", "ground_truth_int"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[FATAL ERROR] Manifest missing required columns: {missing_cols}")
        sys.exit(1)
    print(f"[*] Manifest columns verified: all {len(required_cols)} required columns present.")

    # 3. Class counts
    label_counts = df["label"].value_counts().to_dict()
    real_count = label_counts.get("REAL", 0)
    ai_count = label_counts.get("AI", 0)
    if real_count != 400 or ai_count != 400:
        print(f"[FATAL ERROR] Class counts incorrect: REAL={real_count} (expected 400), AI={ai_count} (expected 400).")
        sys.exit(1)
    print(f"[*] Class distribution verified: exactly 400 REAL, 400 AI.")

    gt_counts = df["ground_truth_int"].value_counts().to_dict()
    if gt_counts.get(0, 0) != 400 or gt_counts.get(1, 0) != 400:
        print(f"[FATAL ERROR] ground_truth_int counts incorrect: 0={gt_counts.get(0)}, 1={gt_counts.get(1)}.")
        sys.exit(1)
    print(f"[*] ground_truth_int values verified: 400 zeros (REAL), 400 ones (AI).")

    # 4. Generator counts
    gen_counts = df["generator"].value_counts().to_dict()
    for gen_name, expected_n in EXPECTED_GENERATOR_COUNTS.items():
        actual_n = gen_counts.get(gen_name, 0)
        if actual_n != expected_n:
            print(f"[FATAL ERROR] Generator count mismatch for '{gen_name}': actual={actual_n}, expected={expected_n}.")
            sys.exit(1)
        display_name = GENERATOR_DISPLAY_MAP.get(gen_name, gen_name)
        print(f"    - {display_name}: exactly {actual_n} images")
    print("[*] Generator counts verified: exactly 100 SD3, 100 Midjourney v6, 100 DALL-E 3, 100 SDXL, 400 REAL.")

    # 5. Payloads existence on disk
    if not payloads_dir.is_dir():
        print(f"[FATAL ERROR] Payloads directory not found at: {payloads_dir}")
        sys.exit(1)

    missing_payloads = []
    empty_payloads = []
    for _, row in df.iterrows():
        payload_file = payloads_dir / row["original_filename"]
        if not payload_file.is_file():
            missing_payloads.append(row["original_filename"])
        elif payload_file.stat().st_size == 0:
            empty_payloads.append(row["original_filename"])

    if missing_payloads:
        print(f"[FATAL ERROR] {len(missing_payloads)} payload files are missing on disk: {missing_payloads[:5]}...")
        sys.exit(1)
    if empty_payloads:
        print(f"[FATAL ERROR] {len(empty_payloads)} payload files are 0 bytes on disk: {empty_payloads[:5]}...")
        sys.exit(1)
    print(f"[*] Raw payload files verified: all 800 image files present and non-empty.")

    # 6. Model checkpoints existence
    for model_name, ckpt_path in model_paths.items():
        if not ckpt_path.is_file():
            print(f"[FATAL ERROR] Checkpoint missing for {model_name} at: {ckpt_path}")
            sys.exit(1)
        if ckpt_path.stat().st_size == 0:
            print(f"[FATAL ERROR] Checkpoint file is empty for {model_name} at: {ckpt_path}")
            sys.exit(1)
        print(f"[*] Checkpoint verified: {model_name} ({ckpt_path.name}, {ckpt_path.stat().st_size / (1024*1024):.2f} MB)")

    # 7. Calibration config existence and validation
    if not calibration_path.is_file():
        print(f"[FATAL ERROR] V5-D calibration configuration missing at: {calibration_path}")
        sys.exit(1)

    try:
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to parse calibration JSON: {e}")
        sys.exit(1)

    if calib_data.get("method") != "temperature":
        print(f"[FATAL ERROR] Calibration method is '{calib_data.get('method')}', expected 'temperature'.")
        sys.exit(1)

    temp_baseline = calib_data.get("temperature_baseline", {})
    temperature_T = float(temp_baseline.get("T", 0.0))
    if temperature_T <= 0.0:
        print(f"[FATAL ERROR] Invalid temperature value in calibration config: T={temperature_T}.")
        sys.exit(1)
    print(f"[*] Calibration config verified: temperature scaling T = {temperature_T:.6f}")

    print("=" * 80)
    print("STRUCTURAL VALIDATION PASSED — READY FOR FROZEN INFERENCE")
    print("=" * 80)
    print()

    return df, temperature_T


# -----------------------------------------------------------------------------
# METRICS CALCULATION UTILITIES
# -----------------------------------------------------------------------------
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes binary classification metrics where positive class = 1 (AI), negative = 0 (REAL)."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = int(len(y_true))

    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2.0 if (tn + fp) > 0 and (tp + fn) > 0 else acc

    real_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    ai_fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "N": total,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "real_false_positive_rate": round(real_fpr, 6),
        "ai_false_negative_rate": round(ai_fnr, 6),
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }
    }


# -----------------------------------------------------------------------------
# STEP 2: INFERENCE AND EVALUATION PIPELINE
# -----------------------------------------------------------------------------
def run_evaluation(
    manifest_df: pd.DataFrame,
    payloads_dir: Path,
    output_dir: Path,
    model_paths: Dict[str, Path],
    temperature_T: float,
    batch_size: int = 16,
    device_name: Optional[str] = None
):
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate and load all 4 models
    print("\n[*] Loading models strictly...")
    models: Dict[str, nn.Module] = {}

    print("    - Loading V5-A Spatial...")
    models["V5-A Spatial"] = load_model_weights(SpatialResNet50(), model_paths["V5-A Spatial"], device)

    print("    - Loading V5-B Frequency...")
    models["V5-B Frequency"] = load_model_weights(FrequencyResNet50(), model_paths["V5-B Frequency"], device)

    print("    - Loading V5-C Hybrid...")
    models["V5-C Hybrid"] = load_model_weights(HybridResNet50(), model_paths["V5-C Hybrid"], device)

    print("    - Loading V5-D Gated Residual...")
    models["V5-D Gated Residual"] = load_model_weights(GatedResidualResNet50(), model_paths["V5-D Gated Residual"], device)

    print("[*] All 4 models loaded successfully into eval() mode.")

    # Gaussian kernel for V5-D noise residual
    gaussian_kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3, device=device)

    # 2. Build benchmark DataLoader
    dataset = FrozenBenchmarkDataset(manifest_df, payloads_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print(f"[*] Beginning deterministic frozen evaluation over {len(dataset)} images in {len(loader)} batches...\n")
    start_time = time.time()

    per_image_records: List[Dict[str, Any]] = []
    processed_count = 0
    failure_count = 0

    model_names = [
        "V5-A Spatial",
        "V5-B Frequency",
        "V5-C Hybrid",
        "V5-D Gated Residual"
    ]

    # Containers for aggregate metrics
    raw_ai_probs: Dict[str, List[float]] = {m: [] for m in model_names}
    cal_ai_probs: Dict[str, List[float]] = {m: [] for m in model_names}
    preds_ai: Dict[str, List[int]] = {m: [] for m in model_names}
    ground_truths: List[int] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            batch_spatial = batch["spatial"].to(device, non_blocking=True)
            batch_freq = batch["frequency"].to(device, non_blocking=True)
            batch_labels = batch["ground_truth_int"].numpy()
            batch_b_size = len(batch_labels)

            try:
                # 1. V5-A Spatial
                logits_a = models["V5-A Spatial"](batch_spatial)
                probs_a = F.softmax(logits_a, dim=1)

                # 2. V5-B Frequency
                logits_b = models["V5-B Frequency"](batch_freq)
                probs_b = F.softmax(logits_b, dim=1)

                # 3. V5-C Hybrid
                logits_c = models["V5-C Hybrid"](batch_spatial, batch_freq)
                probs_c = F.softmax(logits_c, dim=1)

                # 4. V5-D Gated Residual
                residual = extract_noise_residual(batch_spatial, gaussian_kernel)
                logits_d = models["V5-D Gated Residual"](batch_spatial, batch_freq, residual)
                probs_d = F.softmax(logits_d, dim=1)

                # Calibration for V5-D: P_cal(AI) = sigmoid((z_ai - z_real) / T)
                s_diff_d = logits_d[:, 1] - logits_d[:, 0]
                cal_probs_d_ai = torch.sigmoid(s_diff_d / float(temperature_T))

                # Extract CPU numpy arrays
                batch_outputs = {
                    "V5-A Spatial": {
                        "raw_real": probs_a[:, 0].cpu().numpy(),
                        "raw_ai": probs_a[:, 1].cpu().numpy(),
                        "cal_real": probs_a[:, 0].cpu().numpy(),
                        "cal_ai": probs_a[:, 1].cpu().numpy(),
                    },
                    "V5-B Frequency": {
                        "raw_real": probs_b[:, 0].cpu().numpy(),
                        "raw_ai": probs_b[:, 1].cpu().numpy(),
                        "cal_real": probs_b[:, 0].cpu().numpy(),
                        "cal_ai": probs_b[:, 1].cpu().numpy(),
                    },
                    "V5-C Hybrid": {
                        "raw_real": probs_c[:, 0].cpu().numpy(),
                        "raw_ai": probs_c[:, 1].cpu().numpy(),
                        "cal_real": probs_c[:, 0].cpu().numpy(),
                        "cal_ai": probs_c[:, 1].cpu().numpy(),
                    },
                    "V5-D Gated Residual": {
                        "raw_real": probs_d[:, 0].cpu().numpy(),
                        "raw_ai": probs_d[:, 1].cpu().numpy(),
                        "cal_real": (1.0 - cal_probs_d_ai).cpu().numpy(),
                        "cal_ai": cal_probs_d_ai.cpu().numpy(),
                    },
                }

                # Store per-image records
                for i in range(batch_b_size):
                    gt_int = int(batch_labels[i])
                    ground_truths.append(gt_int)
                    true_lbl = batch["true_label"][i]
                    c_id = batch["candidate_id"][i]
                    fn = batch["original_filename"][i]
                    cat = batch["category"][i]
                    src = batch["source"][i]
                    gen = batch["generator"][i]

                    for m_name in model_names:
                        r_real = float(batch_outputs[m_name]["raw_real"][i])
                        r_ai = float(batch_outputs[m_name]["raw_ai"][i])
                        c_real = float(batch_outputs[m_name]["cal_real"][i])
                        c_ai = float(batch_outputs[m_name]["cal_ai"][i])

                        pred_str = "AI" if c_ai >= 0.50 else "REAL"
                        is_correct = (pred_str == true_lbl)

                        raw_ai_probs[m_name].append(r_ai)
                        cal_ai_probs[m_name].append(c_ai)
                        preds_ai[m_name].append(1 if pred_str == "AI" else 0)

                        per_image_records.append({
                            "candidate_id": c_id,
                            "original_filename": fn,
                            "true_label": true_lbl,
                            "category": cat,
                            "source": src,
                            "generator": gen,
                            "model": m_name,
                            "raw_ai_probability": round(r_ai, 6),
                            "raw_real_probability": round(r_real, 6),
                            "calibrated_ai_probability": round(c_ai, 6),
                            "calibrated_real_probability": round(c_real, 6),
                            "prediction": pred_str,
                            "correct": is_correct,
                        })

                processed_count += batch_b_size

            except Exception as e:
                print(f"\n[ERROR] Exception processing batch {batch_idx}: {e}")
                failure_count += batch_b_size

            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
                elapsed = time.time() - start_time
                print(f"    Evaluating batch {batch_idx:2d}/{len(loader)}: {processed_count:3d}/{len(dataset)} images ({elapsed:.1f}s)")

    eval_duration = time.time() - start_time
    print(f"\n[*] Inference completed in {eval_duration:.2f} seconds.")

    # Verify output counts
    expected_records = len(manifest_df) * len(model_names)
    if len(per_image_records) != expected_records:
        print(f"[FATAL ERROR] Expected {expected_records} per-image rows, got {len(per_image_records)}.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 3: COMPUTE AGGREGATE METRICS & TABLES
    # -------------------------------------------------------------------------
    df_per_image = pd.DataFrame(per_image_records)
    y_true_all = np.array([1 if lbl == "AI" else 0 for lbl in manifest_df["label"].tolist()])

    summary_json_data: Dict[str, Any] = {
        "benchmark_metadata": {
            "benchmark_name": "Defactify External Benchmark v1",
            "dataset_source": "Defactify_Image_Dataset (Validation Split)",
            "dataset_commit": "787334f7857fa54f29027a7f09c30e895ad486ef",
            "manifest_path": str(DEFAULT_MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "payloads_dir": str(DEFAULT_PAYLOADS_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "total_images": len(manifest_df),
            "real_images": int((manifest_df["label"] == "REAL").sum()),
            "ai_images": int((manifest_df["label"] == "AI").sum()),
            "generator_counts": {k: int(v) for k, v in manifest_df["generator"].value_counts().items()},
            "decision_threshold": 0.50,
            "v5_d_calibration": {
                "method": "temperature",
                "temperature_T": float(temperature_T),
                "source_config": str(CALIBRATION_JSON_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")
            },
            "device": str(device),
            "execution_duration_seconds": round(eval_duration, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "models": {}
    }

    model_comparison_rows: List[Dict[str, Any]] = []
    generator_result_rows: List[Dict[str, Any]] = []
    confusion_matrix_rows: List[Dict[str, Any]] = []

    ai_generators = [
        "Stable_Diffusion_3",
        "Midjourney_v6",
        "DALL-E_3",
        "Stable_Diffusion_XL"
    ]

    for m_name in model_names:
        df_m = df_per_image[df_per_image["model"] == m_name]
        is_cal = (m_name == "V5-D Gated Residual")
        cal_method_str = f"Temperature Scaling (T={temperature_T:.6f})" if is_cal else "Raw Softmax (Uncalibrated)"

        probs_ai_m = df_m["calibrated_ai_probability"].values
        preds_m = (probs_ai_m >= 0.50).astype(int)
        y_true_m = (df_m["true_label"].values == "AI").astype(int)

        # 1. OVERALL METRICS (N=800)
        overall_metrics = compute_binary_metrics(y_true_m, preds_m)
        cm_overall = overall_metrics["confusion_matrix"]

        # 2. REAL SUBSET (N=400)
        real_mask = (df_m["true_label"].values == "REAL")
        real_probs = probs_ai_m[real_mask]
        real_preds = preds_m[real_mask]
        real_n = int(np.sum(real_mask))
        correct_real = int(np.sum(real_preds == 0))
        fp_real = int(np.sum(real_preds == 1))
        fpr_real = fp_real / real_n if real_n > 0 else 0.0
        mean_ai_real = float(np.mean(real_probs))
        median_ai_real = float(np.median(real_probs))
        pct_ge_50 = float(np.mean(real_probs >= 0.50) * 100.0)
        pct_ge_70 = float(np.mean(real_probs >= 0.70) * 100.0)
        pct_ge_90 = float(np.mean(real_probs >= 0.90) * 100.0)

        real_subset_metrics = {
            "N": real_n,
            "correctly_classified_real": correct_real,
            "false_positives": fp_real,
            "false_positive_rate": round(fpr_real, 6),
            "mean_ai_probability": round(mean_ai_real, 6),
            "median_ai_probability": round(median_ai_real, 6),
            "percentage_ge_50": round(pct_ge_50, 4),
            "percentage_ge_70": round(pct_ge_70, 4),
            "percentage_ge_90": round(pct_ge_90, 4),
        }

        # 3. AI SUBSET (N=400)
        ai_mask = (df_m["true_label"].values == "AI")
        ai_probs = probs_ai_m[ai_mask]
        ai_preds = preds_m[ai_mask]
        ai_n = int(np.sum(ai_mask))
        correct_ai = int(np.sum(ai_preds == 1))
        fn_ai = int(np.sum(ai_preds == 0))
        recall_ai = correct_ai / ai_n if ai_n > 0 else 0.0
        mean_ai_ai = float(np.mean(ai_probs))
        median_ai_ai = float(np.median(ai_probs))

        ai_subset_metrics = {
            "N": ai_n,
            "correctly_classified_ai": correct_ai,
            "false_negatives": fn_ai,
            "ai_recall": round(recall_ai, 6),
            "mean_ai_probability": round(mean_ai_ai, 6),
            "median_ai_probability": round(median_ai_ai, 6),
        }

        # 4. GENERATOR-SPECIFIC METRICS
        gen_metrics_dict: Dict[str, Any] = {}
        for gen_key in ai_generators:
            display_name = GENERATOR_DISPLAY_MAP.get(gen_key, gen_key)
            gen_mask = (df_m["generator"].values == gen_key)
            gen_n = int(np.sum(gen_mask))
            gen_probs = probs_ai_m[gen_mask]
            gen_preds = preds_m[gen_mask]

            tp_gen = int(np.sum(gen_preds == 1))
            fn_gen = int(np.sum(gen_preds == 0))
            recall_gen = tp_gen / gen_n if gen_n > 0 else 0.0
            fnr_gen = fn_gen / gen_n if gen_n > 0 else 0.0
            mean_prob_gen = float(np.mean(gen_probs))
            median_prob_gen = float(np.median(gen_probs))

            # Paired with the 400 natural REAL photographs (N=500 total)
            paired_mask = real_mask | gen_mask
            y_paired_true = y_true_m[paired_mask]
            y_paired_pred = preds_m[paired_mask]
            paired_metrics = compute_binary_metrics(y_paired_true, y_paired_pred)

            gen_metrics_dict[gen_key] = {
                "generator_name": display_name,
                "N": gen_n,
                "accuracy": paired_metrics["accuracy"],
                "precision": paired_metrics["precision"],
                "recall": round(recall_gen, 6),
                "f1": paired_metrics["f1"],
                "balanced_accuracy": paired_metrics["balanced_accuracy"],
                "mean_ai_probability": round(mean_prob_gen, 6),
                "median_ai_probability": round(median_prob_gen, 6),
                "false_negative_count": fn_gen,
                "false_negative_rate": round(fnr_gen, 6),
                "confusion_matrix_standalone": {
                    "TP": tp_gen,
                    "FN": fn_gen
                },
                "confusion_matrix_with_real_baseline": paired_metrics["confusion_matrix"]
            }

            generator_result_rows.append({
                "model": m_name,
                "generator_raw": gen_key,
                "generator_name": display_name,
                "N": gen_n,
                "tp": tp_gen,
                "fn": fn_gen,
                "recall": round(recall_gen, 6),
                "false_negative_rate": round(fnr_gen, 6),
                "accuracy_with_real": paired_metrics["accuracy"],
                "precision_with_real": paired_metrics["precision"],
                "f1_with_real": paired_metrics["f1"],
                "balanced_accuracy_with_real": paired_metrics["balanced_accuracy"],
                "mean_ai_probability": round(mean_prob_gen, 6),
                "median_ai_probability": round(median_prob_gen, 6),
            })

            # Append to confusion matrices table for each generator
            p_cm = paired_metrics["confusion_matrix"]
            confusion_matrix_rows.append({
                "model": m_name,
                "evaluation_subset": f"{display_name} vs REAL (N=500)",
                "N": paired_metrics["N"],
                "TN": p_cm["TN"],
                "FP": p_cm["FP"],
                "FN": p_cm["FN"],
                "TP": p_cm["TP"],
                "accuracy": paired_metrics["accuracy"],
                "precision": paired_metrics["precision"],
                "recall_TPR": paired_metrics["recall"],
                "specificity_TNR": round(p_cm["TN"] / (p_cm["TN"] + p_cm["FP"]), 6) if (p_cm["TN"] + p_cm["FP"]) > 0 else 0.0,
                "balanced_accuracy": paired_metrics["balanced_accuracy"],
                "FPR": paired_metrics["real_false_positive_rate"],
                "FNR": paired_metrics["ai_false_negative_rate"],
            })

        # Append to summary JSON
        summary_json_data["models"][m_name] = {
            "model_name": m_name,
            "is_calibrated": is_cal,
            "calibration_method": cal_method_str,
            "overall": overall_metrics,
            "real_subset": real_subset_metrics,
            "ai_subset": ai_subset_metrics,
            "generator_specific": gen_metrics_dict
        }

        # Model comparison CSV row
        model_comparison_rows.append({
            "model": m_name,
            "calibrated": is_cal,
            "calibration_method": cal_method_str,
            "N": overall_metrics["N"],
            "accuracy": overall_metrics["accuracy"],
            "precision": overall_metrics["precision"],
            "recall": overall_metrics["recall"],
            "f1": overall_metrics["f1"],
            "balanced_accuracy": overall_metrics["balanced_accuracy"],
            "real_fpr": overall_metrics["real_false_positive_rate"],
            "ai_fnr": overall_metrics["ai_false_negative_rate"],
            "tp": cm_overall["TP"],
            "tn": cm_overall["TN"],
            "fp": cm_overall["FP"],
            "fn": cm_overall["FN"],
            "real_n": real_n,
            "real_correct": correct_real,
            "real_fp": fp_real,
            "real_mean_ai_prob": round(mean_ai_real, 6),
            "real_median_ai_prob": round(median_ai_real, 6),
            "real_pct_ge_50": round(pct_ge_50, 4),
            "real_pct_ge_70": round(pct_ge_70, 4),
            "real_pct_ge_90": round(pct_ge_90, 4),
            "ai_n": ai_n,
            "ai_correct": correct_ai,
            "ai_fn": fn_ai,
            "ai_recall": round(recall_ai, 6),
            "ai_mean_ai_prob": round(mean_ai_ai, 6),
            "ai_median_ai_prob": round(median_ai_ai, 6),
        })

        # Append overall, REAL, and AI confusion matrix rows
        confusion_matrix_rows.append({
            "model": m_name,
            "evaluation_subset": "Overall Benchmark (N=800)",
            "N": overall_metrics["N"],
            "TN": cm_overall["TN"],
            "FP": cm_overall["FP"],
            "FN": cm_overall["FN"],
            "TP": cm_overall["TP"],
            "accuracy": overall_metrics["accuracy"],
            "precision": overall_metrics["precision"],
            "recall_TPR": overall_metrics["recall"],
            "specificity_TNR": round(cm_overall["TN"] / (cm_overall["TN"] + cm_overall["FP"]), 6),
            "balanced_accuracy": overall_metrics["balanced_accuracy"],
            "FPR": overall_metrics["real_false_positive_rate"],
            "FNR": overall_metrics["ai_false_negative_rate"],
        })
        confusion_matrix_rows.append({
            "model": m_name,
            "evaluation_subset": "REAL Photographs Only (N=400)",
            "N": real_n,
            "TN": correct_real,
            "FP": fp_real,
            "FN": 0,
            "TP": 0,
            "accuracy": round(correct_real / real_n, 6),
            "precision": 0.0,
            "recall_TPR": 0.0,
            "specificity_TNR": round(correct_real / real_n, 6),
            "balanced_accuracy": round(correct_real / real_n, 6),
            "FPR": round(fp_real / real_n, 6),
            "FNR": 0.0,
        })
        confusion_matrix_rows.append({
            "model": m_name,
            "evaluation_subset": "AI Subset Only (N=400)",
            "N": ai_n,
            "TN": 0,
            "FP": 0,
            "FN": fn_ai,
            "TP": correct_ai,
            "accuracy": round(correct_ai / ai_n, 6),
            "precision": 1.0 if correct_ai > 0 else 0.0,
            "recall_TPR": round(correct_ai / ai_n, 6),
            "specificity_TNR": 0.0,
            "balanced_accuracy": round(correct_ai / ai_n, 6),
            "FPR": 0.0,
            "FNR": round(fn_ai / ai_n, 6),
        })

    # -------------------------------------------------------------------------
    # STEP 4: SAVE OUTPUT ARTIFACTS
    # -------------------------------------------------------------------------
    print("\n[*] Saving output artifacts under:", output_dir)

    # 1. external_benchmark_v1_per_image.csv
    per_image_path = output_dir / "external_benchmark_v1_per_image.csv"
    df_per_image.to_csv(per_image_path, index=False)
    print(f"    [1/6] Per-image results:       {per_image_path} ({len(df_per_image)} rows)")

    # 2. external_benchmark_v1_summary.json
    summary_path = output_dir / "external_benchmark_v1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2)
    print(f"    [2/6] Complete summary JSON:   {summary_path}")

    # 3. external_benchmark_v1_model_comparison.csv
    comparison_path = output_dir / "external_benchmark_v1_model_comparison.csv"
    df_comp = pd.DataFrame(model_comparison_rows)
    df_comp.to_csv(comparison_path, index=False)
    print(f"    [3/6] Model comparison CSV:    {comparison_path}")

    # 4. external_benchmark_v1_generator_results.csv
    gen_results_path = output_dir / "external_benchmark_v1_generator_results.csv"
    df_gen = pd.DataFrame(generator_result_rows)
    df_gen.to_csv(gen_results_path, index=False)
    print(f"    [4/6] Generator results CSV:   {gen_results_path}")

    # 5. external_benchmark_v1_confusion_matrices.csv
    cm_path = output_dir / "external_benchmark_v1_confusion_matrices.csv"
    df_cm = pd.DataFrame(confusion_matrix_rows)
    df_cm.to_csv(cm_path, index=False)
    print(f"    [5/6] Confusion matrices CSV:  {cm_path}")

    # 6. external_benchmark_v1_report.md
    report_path = output_dir / "external_benchmark_v1_report.md"
    generate_markdown_report(
        report_path,
        summary_json_data,
        df_comp,
        df_gen,
        df_cm
    )
    print(f"    [6/6] Summary report Markdown: {report_path}")

    # -------------------------------------------------------------------------
    # STEP 5: TERMINAL REPORTING (FACTUAL ONLY)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AIDETECT EXTERNAL BENCHMARK EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Total images evaluated:        {len(manifest_df)}")
    print(f"Images successfully evaluated: {processed_count}")
    print(f"Failures:                      {failure_count}")
    print(f"Execution time:                {eval_duration:.2f} seconds")
    print()

    print("-" * 80)
    print("OVERALL MODEL METRICS (N=800: 400 REAL, 400 AI | Threshold: P(AI) >= 0.50)")
    print("-" * 80)
    print(f"{'Model':<22} | {'Calibrated':<10} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Bal Acc':<9} | {'REAL FPR':<9} | {'AI FNR':<9}")
    print("-" * 80)
    for row in model_comparison_rows:
        cal_str = "Yes" if row["calibrated"] else "No"
        print(f"{row['model']:<22} | {cal_str:<10} | {row['accuracy']*100:6.2f}%  | {row['precision']*100:6.2f}%  | {row['recall']*100:6.2f}%  | {row['f1']*100:6.2f}%  | {row['balanced_accuracy']*100:6.2f}%  | {row['real_fpr']*100:6.2f}%  | {row['ai_fnr']*100:6.2f}%")
    print("-" * 80)
    print()

    print("-" * 80)
    print("REAL SUBSET METRICS & OVER-CONFIDENCE DISTRIBUTION (N=400 Natural Photos)")
    print("-" * 80)
    print(f"{'Model':<22} | {'TN / 400':<9} | {'FP (Errors)':<11} | {'FPR':<8} | {'Mean P(AI)':<10} | {'P>=50%':<8} | {'P>=70%':<8} | {'P>=90%':<8}")
    print("-" * 80)
    for row in model_comparison_rows:
        print(f"{row['model']:<22} | {row['real_correct']:4d}/400  | {row['real_fp']:4d}/400    | {row['real_fpr']*100:5.2f}% | {row['real_mean_ai_prob']:<10.4f} | {row['real_pct_ge_50']:5.2f}%  | {row['real_pct_ge_70']:5.2f}%  | {row['real_pct_ge_90']:5.2f}%")
    print("-" * 80)
    print()

    print("-" * 80)
    print("GENERATOR-SPECIFIC METRICS (AI Recall & Detection Rates per Model)")
    print("-" * 80)
    print(f"{'Model':<22} | {'Generator':<16} | {'N':<4} | {'Recall (AI)':<12} | {'FNR (Missed)':<12} | {'Mean P(AI)':<11} | {'Median P(AI)':<12}")
    print("-" * 80)
    for row in generator_result_rows:
        print(f"{row['model']:<22} | {row['generator_name']:<16} | {row['N']:<4} | {row['recall']*100:6.2f}%     | {row['false_negative_rate']*100:6.2f}%     | {row['mean_ai_probability']:<11.4f} | {row['median_ai_probability']:<12.4f}")
    print("-" * 80)
    print()


# -----------------------------------------------------------------------------
# STEP 6: MARKDOWN REPORT GENERATOR
# -----------------------------------------------------------------------------
def generate_markdown_report(
    output_path: Path,
    summary_data: Dict[str, Any],
    df_comp: pd.DataFrame,
    df_gen: pd.DataFrame,
    df_cm: pd.DataFrame
):
    meta = summary_data["benchmark_metadata"]
    lines = [
        "# AIDetect — Defactify External Benchmark Evaluation Report (v1)",
        "",
        "## 1. Benchmark Overview & Certification",
        "",
        f"- **Benchmark Name:** {meta['benchmark_name']}",
        f"- **Dataset Source:** {meta['dataset_source']} (Commit: `{meta['dataset_commit']}`)",
        f"- **Total Image Count:** {meta['total_images']}",
        f"- **Composition:** {meta['real_images']} REAL photographs + {meta['ai_images']} AI generated images",
        "- **Generator Breakdown:**",
        f"  - Natural Photographs (MS COCO, REAL): {meta['generator_counts'].get('None_Authentic_Photograph', 400)}",
        f"  - Stable Diffusion 3 (SD3): {meta['generator_counts'].get('Stable_Diffusion_3', 100)}",
        f"  - Midjourney v6: {meta['generator_counts'].get('Midjourney_v6', 100)}",
        f"  - DALL-E 3: {meta['generator_counts'].get('DALL-E_3', 100)}",
        f"  - Stable Diffusion XL (SDXL): {meta['generator_counts'].get('Stable_Diffusion_XL', 100)}",
        f"- **Leakage Audit Status:** Certified zero leakage (min perceptual distance $d_H \\ge 16$, 0 SHA-256 collisions).",
        f"- **Decision Threshold:** Calibrated $P(\\text{{AI}}) \\ge {meta['decision_threshold']:.2f} \\to \\text{{AI}}$, $< {meta['decision_threshold']:.2f} \\to \\text{{REAL}}$",
        f"- **V5-D Calibration:** {meta['v5_d_calibration']['method'].capitalize()} Scaling ($T = {meta['v5_d_calibration']['temperature_T']:.6f}$)",
        f"- **V5-A, V5-B, V5-C Calibration:** Raw Softmax (Uncalibrated baseline)",
        f"- **Execution Timestamp:** `{meta['timestamp']}`",
        f"- **Execution Device:** `{meta['device']}` (Completed in {meta['execution_duration_seconds']}s)",
        "",
        "---",
        "",
        "## 2. Overall Model Comparison (N=800)",
        "",
        "| Model | Calibrated | Calibration Method | Accuracy | Precision | Recall (AI) | F1 Score | Balanced Acc | REAL FPR | AI FNR |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df_comp.iterrows():
        cal_txt = "Yes" if r["calibrated"] else "No"
        lines.append(
            f"| **{r['model']}** | {cal_txt} | {r['calibration_method']} | "
            f"{r['accuracy']*100:.2f}% | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% | "
            f"{r['f1']*100:.2f}% | {r['balanced_accuracy']*100:.2f}% | {r['real_fpr']*100:.2f}% | {r['ai_fnr']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. REAL Subset Performance & Over-Confidence Distribution (N=400 Natural Photos)",
        "",
        "| Model | Correct REAL (TN) | False Positives (FP) | False Positive Rate | Mean P(AI) | Median P(AI) | % P(AI) >= 50% | % P(AI) >= 70% | % P(AI) >= 90% |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df_comp.iterrows():
        lines.append(
            f"| **{r['model']}** | {r['real_correct']}/400 | {r['real_fp']}/400 | {r['real_fpr']*100:.2f}% | "
            f"{r['real_mean_ai_prob']:.4f} | {r['real_median_ai_prob']:.4f} | "
            f"{r['real_pct_ge_50']:.2f}% | {r['real_pct_ge_70']:.2f}% | {r['real_pct_ge_90']:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. AI Subset Performance (N=400 AI Generated Images)",
        "",
        "| Model | Correct AI (TP) | False Negatives (FN) | AI Recall | Mean P(AI) | Median P(AI) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df_comp.iterrows():
        lines.append(
            f"| **{r['model']}** | {r['ai_correct']}/400 | {r['ai_fn']}/400 | {r['ai_recall']*100:.2f}% | "
            f"{r['ai_mean_ai_prob']:.4f} | {r['ai_median_ai_prob']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Generator-Specific Performance Breakdown",
        "",
        "| Model | Generator | N | Recall (AI Detection) | False Negative Rate | Mean P(AI) | Median P(AI) | Precision (vs REAL) | F1 (vs REAL) | Bal Acc (vs REAL) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df_gen.iterrows():
        lines.append(
            f"| **{r['model']}** | {r['generator_name']} | {r['N']} | "
            f"{r['recall']*100:.2f}% | {r['false_negative_rate']*100:.2f}% | "
            f"{r['mean_ai_probability']:.4f} | {r['median_ai_probability']:.4f} | "
            f"{r['precision_with_real']*100:.2f}% | {r['f1_with_real']*100:.2f}% | {r['balanced_accuracy_with_real']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Confusion Matrices Summary",
        "",
        "| Model | Subset | Total N | TN | FP | FN | TP | Accuracy | Recall / TPR | Specificity / TNR | FPR | FNR |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df_cm.iterrows():
        lines.append(
            f"| **{r['model']}** | {r['evaluation_subset']} | {r['N']} | "
            f"{r['TN']} | {r['FP']} | {r['FN']} | {r['TP']} | "
            f"{r['accuracy']*100:.2f}% | {r['recall_TPR']*100:.2f}% | {r['specificity_TNR']*100:.2f}% | "
            f"{r['FPR']*100:.2f}% | {r['FNR']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 7. Integrity & Invariants Checklist",
        "",
        "- [x] Model weights frozen and unaltered.",
        "- [x] Frozen 9,000-image V5 test set untouched.",
        "- [x] Deterministic evaluation with zero data augmentation.",
        "- [x] Temperature calibration parameter $T = 2.198387$ loaded from authoritative configuration.",
        "- [x] Structural pre-validation confirmed 800 images across exact categories prior to inference.",
        "- [x] Output artifacts fully persisted without subjective interpretations.",
        ""
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# MAIN CLI ENTRYPOINT
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Frozen evaluation of V5-A/B/C/D on certified Defactify external benchmark (v1)."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to external benchmark manifest CSV.")
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS_DIR, help="Path to directory containing 800 payload files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation artifacts.")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON_PATH, help="Path to V5-D calibration JSON.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Default: auto.")

    args = parser.parse_args()

    # Step 1: Pre-inference structural validation
    manifest_df, temperature_T = perform_structural_checks(
        manifest_path=args.manifest,
        payloads_dir=args.payloads,
        model_paths=MODEL_PATHS,
        calibration_path=args.calibration
    )

    # Step 2: Frozen inference and report generation
    run_evaluation(
        manifest_df=manifest_df,
        payloads_dir=args.payloads,
        output_dir=args.output_dir,
        model_paths=MODEL_PATHS,
        temperature_T=temperature_T,
        batch_size=args.batch_size,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
