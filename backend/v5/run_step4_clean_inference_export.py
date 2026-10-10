"""
AIDetect — Step 4: Clean-Test Continuous Probability Export
===========================================================
Runs deterministic inference over the frozen 9,000-image V5 test set across all 4 models:
  - V5-A Spatial
  - V5-B Frequency
  - V5-C Hybrid
  - V5-D Gated Residual (with Temperature Scaling T=2.1983866642849734)

Verifies EXACT mathematical reproduction of authoritative TP, TN, FP, FN:
  - V5-A: TP=4359, TN=4373, FP=127, FN=141
  - V5-B: TP=4078, TN=4081, FP=419, FN=422
  - V5-C: TP=4371, TN=4348, FP=152, FN=129
  - V5-D: TP=4367, TN=4379, FP=121, FN=133

Exports continuous probabilities to:
  - evaluation_results/v5/v5_clean_test_predictions.csv
  - evaluation_results/v5/strengthened_evaluation/v5_clean_test_predictions.csv
"""

import os
import sys
import io
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

# Ensure backend paths are discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.v5.v5_models import GatedResidualResNet50

# Authoritative Paths
MANIFEST_PATH = PROJECT_ROOT / "v5_metadata" / "v5_dataset_manifest.csv"
MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "v5"
CALIBRATION_JSON = MODEL_DIR / "v5_d_calibration.json"

MODEL_PATHS = {
    "V5-A": MODEL_DIR / "spatial_resnet50_v5_best.pth",
    "V5-B": MODEL_DIR / "frequency_resnet50_v5_best.pth",
    "V5-C": MODEL_DIR / "hybrid_resnet50_v5_best.pth",
    "V5-D": MODEL_DIR / "gated_residual_resnet50_v5_best.pth"
}

OUTPUT_CSV_1 = PROJECT_ROOT / "evaluation_results" / "v5" / "v5_clean_test_predictions.csv"
OUTPUT_CSV_2 = PROJECT_ROOT / "evaluation_results" / "v5" / "strengthened_evaluation" / "v5_clean_test_predictions.csv"
OUTPUT_CSV_1.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_2.parent.mkdir(parents=True, exist_ok=True)

# Normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_MAPPING = {"REAL": 0, "AI": 1}

EXPECTED_METRICS = {
    "V5-A": {"tp": 4359, "tn": 4373, "fp": 127, "fn": 141},
    "V5-B": {"tp": 4078, "tn": 4081, "fp": 419, "fn": 422},
    "V5-C": {"tp": 4371, "tn": 4348, "fp": 152, "fn": 129},
    "V5-D": {"tp": 4367, "tn": 4379, "fp": 121, "fn": 133}
}


# -----------------------------------------------------------------------------
# PREPROCESSING TRANSFORMS (EXACT AUTHORITATIVE IMPLEMENTATION)
# -----------------------------------------------------------------------------
def build_v5_spatial_val_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class AuthoritativeNativeFFTTransform:
    def __init__(self):
        self._to_tensor = transforms.ToTensor()
        self._norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        x = self._to_tensor(image)
        fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        for c in range(mag.shape[0]):
            ch = mag[c]
            mi, ma = ch.min(), ch.max()
            mag[c] = (ch - mi) / (ma - mi + 1e-8)
        mag_224 = F.interpolate(mag.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        return self._norm(mag_224)


def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0, channels: int = 3, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = coords.unsqueeze(0)
    y = coords.unsqueeze(1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel.to(device)


def extract_noise_residual(spatial_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    blurred = F.conv2d(spatial_tensor, kernel, padding=2, groups=3)
    return spatial_tensor - blurred


# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------
class SpatialResNet50(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = resnet50(weights=None)
        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FrequencyResNet50(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = resnet50(weights=None)
        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HybridResNet50(nn.Module):
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


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
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
# HIGH-PERFORMANCE PRE-CACHED TEST DATASET
# -----------------------------------------------------------------------------
def resolve_hf_cache_dir(cli_value: Optional[str] = None) -> Path:
    """Resolve the explicit Hugging Face cache root used by the test manifest."""
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())
    elif os.environ.get("HF_CACHE_DIR"):
        candidates.append(Path(os.environ["HF_CACHE_DIR"]).expanduser())
    else:
        # Documented cross-platform defaults only; no user-specific path.
        candidates.extend([
            Path.home() / ".cache" / "huggingface" / "hub",
            Path(os.environ["LOCALAPPDATA"]) / "huggingface" / "hub"
            if os.environ.get("LOCALAPPDATA") else Path("__missing_localappdata__"),
        ])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Hugging Face cache directory was not found. Provide --hf-cache or "
        f"set HF_CACHE_DIR. Searched: {searched}"
    )


class FastV5TestDataset(Dataset):
    def __init__(self, manifest_path: Path, hf_cache_dir: Path):
        self.spatial_tf = build_v5_spatial_val_transform()
        self.fft_tf = AuthoritativeNativeFFTTransform()
        self.samples = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"].lower() == "test":
                    self.samples.append(row)

        print(f"[*] FastV5TestDataset: Loaded {len(self.samples)} test split manifest records.")

        # Pre-cache the exact parquet tables referenced by the manifest.  A
        # supplied cache root is required so unrelated machine-local caches
        # cannot be searched accidentally.
        self.parquet_tables = {}
        cached_pq_names = {'train-00000-of-00001.parquet', 'train-00001-of-00014.parquet', 'train-00000-of-00014.parquet'}
        for parquet_name in sorted(cached_pq_names):
            matches = sorted(hf_cache_dir.rglob(parquet_name))
            if not matches:
                raise FileNotFoundError(
                    f"Required parquet file {parquet_name!r} was not found under "
                    f"the explicit Hugging Face cache {hf_cache_dir}"
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Ambiguous parquet file {parquet_name!r}: found {len(matches)} "
                    f"matches under {hf_cache_dir}; provide a cache with one authoritative copy"
                )
            cand = matches[0]
            print(f"[*] Pre-loading parquet into RAM: {cand} ({cand.stat().st_size / (1024*1024):.1f} MB)...")
            t0 = time.time()
            self.parquet_tables[parquet_name] = pq.read_table(cand)
            print(f"    Loaded in {time.time() - t0:.2f} s ({len(self.parquet_tables[parquet_name])} rows).")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rec: Dict[str, Any]) -> Image.Image:
        stype = rec["storage_type"]
        sref = rec["storage_ref"]
        if stype == "file":
            full_p = PROJECT_ROOT / sref
            with Image.open(full_p) as im:
                return im.convert("RGB")
        elif stype == "parquet_row":
            pq_name, r_idx_str = sref.split(":")
            r_idx = int(r_idx_str)
            raw_bytes = self.parquet_tables[pq_name]["image"][r_idx].as_py()["bytes"]
            with Image.open(io.BytesIO(raw_bytes)) as im:
                return im.convert("RGB")
        else:
            raise ValueError(f"Unknown storage_type: {stype}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.samples[index]
        img = self._load_image(rec)
        label = CLASS_MAPPING[rec["ground_truth"]]

        return {
            "index": index,
            "sample_id": rec["sample_id"],
            "ground_truth": rec["ground_truth"],
            "label": label,
            "domain": rec["domain"],
            "source": rec["source"],
            "storage_type": rec["storage_type"],
            "spatial": self.spatial_tf(img),
            "frequency": self.fft_tf(img)
        }


# -----------------------------------------------------------------------------
# MAIN INFERENCE EXPORT
# -----------------------------------------------------------------------------
def run_step4(hf_cache_dir: Optional[str] = None):
    print("=" * 80)
    print("AIDETECT — STEP 4: CLEAN-TEST CONTINUOUS PROBABILITY EXPORT & VERIFICATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load calibration temperature T
    with open(CALIBRATION_JSON, "r", encoding="utf-8") as f:
        cal_cfg = json.load(f)
    if "temperature_baseline" in cal_cfg and "T" in cal_cfg["temperature_baseline"]:
        T = float(cal_cfg["temperature_baseline"]["T"])
    elif "temperature" in cal_cfg:
        T = float(cal_cfg["temperature"])
    else:
        T = 2.1983866642849734
    print(f"[*] Loaded V5-D Temperature Scaling parameter: T = {T:.16f}")

    # Load models
    print("[*] Initializing and loading 4 model checkpoints...")
    model_a = load_model_weights(SpatialResNet50(), MODEL_PATHS["V5-A"], device)
    model_b = load_model_weights(FrequencyResNet50(), MODEL_PATHS["V5-B"], device)
    model_c = load_model_weights(HybridResNet50(), MODEL_PATHS["V5-C"], device)
    model_d = load_model_weights(GatedResidualResNet50(), MODEL_PATHS["V5-D"], device)
    print("[*] All 4 models loaded and placed in eval mode.")

    gaussian_kernel = create_gaussian_kernel(device=device)

    # Initialize Dataset and DataLoader (batch_size=16 strictly matches authoritative evaluate_v5_final.py)
    dataset = FastV5TestDataset(MANIFEST_PATH, resolve_hf_cache_dir(hf_cache_dir))
    assert len(dataset) == 9000, f"Expected 9,000 samples, got {len(dataset)}"
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n[*] Starting unified inference over {len(dataset):,} clean test images...")
    t_start = time.time()

    all_sample_ids: List[str] = []
    all_ground_truths: List[str] = []
    all_label_ints: List[int] = []
    all_domains: List[str] = []
    all_sources: List[str] = []
    all_storage_types: List[str] = []

    # Model probability arrays
    probs_a = []
    probs_b = []
    probs_c = []
    raw_probs_d = []
    cal_probs_d = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            spatial = batch["spatial"].to(device, non_blocking=True)
            frequency = batch["frequency"].to(device, non_blocking=True)
            residual = extract_noise_residual(spatial, gaussian_kernel)

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=torch.cuda.is_available()):
                out_a = model_a(spatial)
                out_b = model_b(frequency)
                out_c = model_c(spatial, frequency)
                out_d = model_d(spatial, frequency, residual)

                # Softmax AI probabilities (class index 1)
                p_a = F.softmax(out_a, dim=1)[:, 1]
                p_b = F.softmax(out_b, dim=1)[:, 1]
                p_c = F.softmax(out_c, dim=1)[:, 1]
                p_d_raw = F.softmax(out_d, dim=1)[:, 1]
                p_d_cal = F.softmax(out_d / T, dim=1)[:, 1]

            probs_a.append(p_a.cpu().numpy())
            probs_b.append(p_b.cpu().numpy())
            probs_c.append(p_c.cpu().numpy())
            raw_probs_d.append(p_d_raw.cpu().numpy())
            cal_probs_d.append(p_d_cal.cpu().numpy())

            all_sample_ids.extend(batch["sample_id"])
            all_ground_truths.extend(batch["ground_truth"])
            all_label_ints.extend(batch["label"].numpy().tolist())
            all_domains.extend(batch["domain"])
            all_sources.extend(batch["source"])
            all_storage_types.extend(batch["storage_type"])

            if batch_idx % 100 == 0 or batch_idx == len(loader):
                processed = min(batch_idx * 16, len(dataset))
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"\r    Processed: {processed:,} / {len(dataset):,} ({processed / len(dataset) * 100:.1f}%) | Speed: {rate:.1f} img/s", end="", flush=True)

    print(f"\n[*] Inference completed in {time.time() - t_start:.2f} seconds.")

    # Flatten probability arrays
    arr_p_a = np.concatenate(probs_a)
    arr_p_b = np.concatenate(probs_b)
    arr_p_c = np.concatenate(probs_c)
    arr_p_d_raw = np.concatenate(raw_probs_d)
    arr_p_d_cal = np.concatenate(cal_probs_d)
    arr_y_true = np.array(all_label_ints)

    # Compute binary predictions (threshold = 0.50)
    arr_pred_a = (arr_p_a >= 0.50).astype(int)
    arr_pred_b = (arr_p_b >= 0.50).astype(int)
    arr_pred_c = (arr_p_c >= 0.50).astype(int)
    arr_pred_d = (arr_p_d_cal >= 0.50).astype(int) # note: raw and cal give identical argmax at 0.5

    # -------------------------------------------------------------------------
    # STRICT CONTINGENCY MATRIX PARITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRICT CONTINGENCY MATRIX PARITY VERIFICATION")
    print("=" * 80)

    computed_models = {
        "V5-A": (arr_pred_a, "V5-A Spatial"),
        "V5-B": (arr_pred_b, "V5-B Frequency"),
        "V5-C": (arr_pred_c, "V5-C Hybrid"),
        "V5-D": (arr_pred_d, "V5-D Gated Residual")
    }

    all_matched = True
    for key, (preds, disp_name) in computed_models.items():
        tp = int(np.sum((preds == 1) & (arr_y_true == 1)))
        tn = int(np.sum((preds == 0) & (arr_y_true == 0)))
        fp = int(np.sum((preds == 1) & (arr_y_true == 0)))
        fn = int(np.sum((preds == 0) & (arr_y_true == 1)))

        exp = EXPECTED_METRICS[key]
        match = (tp == exp["tp"] and tn == exp["tn"] and fp == exp["fp"] and fn == exp["fn"])
        if not match:
            all_matched = False

        status = "MATCH (VERIFIED)" if match else "MISMATCH (FAILED)"
        print(f"[{status}] {disp_name}:")
        print(f"    Computed : TP={tp}, TN={tn}, FP={fp}, FN={fn} | Total={tp+tn+fp+fn}")
        print(f"    Expected : TP={exp['tp']}, TN={exp['tn']}, FP={exp['fp']}, FN={exp['fn']}")

    if not all_matched:
        raise RuntimeError("CRITICAL ERROR: Computed contingency matrices do not strictly match authoritative V5 results!")

    print("\n[SUCCESS] All 4 models strictly and identically reproduced the frozen V5 test set results!")

    # -------------------------------------------------------------------------
    # EXPORT CONTINUOUS PREDICTIONS CSV
    # -------------------------------------------------------------------------
    df_export = pd.DataFrame({
        "sample_id": all_sample_ids,
        "ground_truth": all_ground_truths,
        "label_int": all_label_ints,
        "source": all_sources,
        "domain": all_domains,
        "storage_type": all_storage_types,
        "prob_v5_a": np.round(arr_p_a, 6),
        "pred_v5_a": arr_pred_a,
        "prob_v5_b": np.round(arr_p_b, 6),
        "pred_v5_b": arr_pred_b,
        "prob_v5_c": np.round(arr_p_c, 6),
        "pred_v5_c": arr_pred_c,
        "raw_prob_v5_d": np.round(arr_p_d_raw, 6),
        "calibrated_prob_v5_d": np.round(arr_p_d_cal, 6),
        "pred_v5_d": arr_pred_d
    })

    print(f"\n[*] Exporting predictions to:")
    print(f"    1. {OUTPUT_CSV_1}")
    df_export.to_csv(OUTPUT_CSV_1, index=False)
    print(f"    2. {OUTPUT_CSV_2}")
    df_export.to_csv(OUTPUT_CSV_2, index=False)

    print("=" * 80)
    print("STEP 4 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export deterministic V5 clean-test predictions")
    parser.add_argument(
        "--hf-cache",
        dest="hf_cache_dir",
        help="Explicit Hugging Face hub cache directory (or use HF_CACHE_DIR)",
    )
    args = parser.parse_args()
    run_step4(args.hf_cache_dir)
