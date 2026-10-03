"""
AIDetect Phase 9: External Benchmark Evaluation on Held-Out PIE-Bench
====================================================================
Evaluates the three frozen AI-manipulation detection models:
  1. Spatial:   models/manipulation_resnet50_v1.pth
  2. Frequency: models/manipulation_frequency_resnet50_v1.pth
  3. Hybrid:    models/manipulation_hybrid_resnet50_v1.pth

Dataset:
  manipulation_external_test/pie_bench/ (700 images, 512x512 PNG)
  Manifest: pie_bench_manifest.csv

Strict Protections:
  - Frozen evaluation only: NO training, fine-tuning, or threshold tuning.
  - Zero modification to benchmark images or manifest.
  - Results written strictly to manipulation_external_test/results/.
  - Truthful ground-truth reporting without inventing labels or pairs.
"""

import os
import sys
import csv
import json
import time
import hashlib
import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manipulation_v1_models import ManipulationResNet50V1
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_hybrid_v1_models import ManipulationHybridResNet50V1

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_DIR     = BASE_DIR.parent
EXT_TEST_DIR    = PROJECT_DIR / "manipulation_external_test"
PIE_DIR         = EXT_TEST_DIR / "pie_bench"
MANIFEST_PATH   = PIE_DIR / "pie_bench_manifest.csv"
RESULTS_DIR     = EXT_TEST_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR       = PROJECT_DIR / "models"
SPATIAL_CKPT    = MODEL_DIR / "manipulation_resnet50_v1.pth"
FREQ_CKPT       = MODEL_DIR / "manipulation_frequency_resnet50_v1.pth"
HYBRID_CKPT     = MODEL_DIR / "manipulation_hybrid_resnet50_v1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# PREPROCESSING PIPELINES (IDENTICAL TO TRAINING/INTERNAL EVALUATION)
# ---------------------------------------------------------------------------

class DeterministicSpatialTransform:
    """Exact deterministic spatial evaluation transform used in Phase 6."""
    def __init__(self):
        self.xform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.xform(img)

class DeterministicFrequencyTransform:
    """Exact native-resolution FFT frequency transform used in Phase 7."""
    _to_tensor  = transforms.ToTensor()
    _normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        
        # Native resolution tensor [3, H, W]
        x = self._to_tensor(img)
        
        # 2D FFT -> shift DC component to center
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # Log-scaled magnitude spectrum
        mag = torch.log1p(torch.abs(fft))
        
        # Per-channel min-max normalization
        for c in range(mag.shape[0]):
            c_min = mag[c].min()
            c_max = mag[c].max()
            mag[c] = (mag[c] - c_min) / (c_max - c_min + 1e-8)
            
        # Bilinear resize magnitude to 224x224
        mag = F.interpolate(
            mag.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        return self._normalizer(mag)

SPATIAL_XFORM = DeterministicSpatialTransform()
FREQ_XFORM    = DeterministicFrequencyTransform()

# ---------------------------------------------------------------------------
# PIE-BENCH DATASET LOADER
# ---------------------------------------------------------------------------

class PieBenchDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                abs_path = PROJECT_DIR / row["filepath"]
                self.samples.append({
                    "sample_id": row["sample_id"],
                    "config_name": row["config_name"],
                    "category": row["category"],
                    "filename": row["filename"],
                    "filepath": str(abs_path),
                    "source_prompt": row.get("source_prompt", ""),
                    "target_prompt": row.get("target_prompt", ""),
                    "edit_action": row.get("edit_action", ""),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "sha256": row["sha256"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        p = item["filepath"]
        try:
            with Image.open(p) as raw:
                img = raw.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {p}: {e}") from e
            
        x_spatial = SPATIAL_XFORM(img)
        x_freq    = FREQ_XFORM(img)
        return x_spatial, x_freq, item

def collate_pie(batch):
    s_tensors = torch.stack([b[0] for b in batch])
    f_tensors = torch.stack([b[1] for b in batch])
    metas     = [b[2] for b in batch]
    return s_tensors, f_tensors, metas

# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

def run_external_evaluation():
    print("=" * 70)
    print("AIDETECT PHASE 9: EXTERNAL VALIDATION ON HELD-OUT PIE-BENCH")
    print("=" * 70)
    print(f"Timestamp        : {datetime.datetime.now().isoformat()}")
    print(f"Device           : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model        : {torch.cuda.get_device_name(0)}")
    print(f"Benchmark Path   : {PIE_DIR}")
    print(f"Manifest Path    : {MANIFEST_PATH}")
    print(f"Results Directory: {RESULTS_DIR}")
    print()

    # 1. Benchmark Integrity & Metadata Inspection
    print("-" * 60)
    print("1. BENCHMARK INTEGRITY & METADATA INSPECTION")
    print("-" * 60)
    
    assert MANIFEST_PATH.exists(), f"Manifest missing: {MANIFEST_PATH}"
    dataset = PieBenchDataset(MANIFEST_PATH)
    n_total = len(dataset)
    print(f"Total benchmark images in manifest : {n_total}")
    
    # Verify disk files
    all_files_exist = all(Path(s["filepath"]).exists() for s in dataset.samples)
    print(f"All {n_total} image files exist on disk  : {all_files_exist}")
    assert all_files_exist, "Missing images on disk!"
    
    # Check SHA-256 integrity
    print("Verifying image SHA-256 hashes...")
    mismatches = 0
    for s in dataset.samples:
        with open(s["filepath"], "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h != s["sha256"]:
            mismatches += 1
    print(f"SHA-256 verification              : {n_total - mismatches} / {n_total} matched (mismatches={mismatches})")
    assert mismatches == 0, "Image files corrupted or altered!"

    # Ground-truth analysis
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
    print(f"Available manifest fields          : {fields}")
    has_binary_label = "label" in fields or "ground_truth" in fields
    print(f"Binary ground truth labels present : {has_binary_label}")
    
    categories = Counter(s["category"] for s in dataset.samples)
    configs    = Counter(s["config_name"] for s in dataset.samples)
    print(f"Manipulation Categories ({len(categories)}):")
    for cat, count in categories.items():
        print(f"  - {cat:<25}: {count}")
    print(f"Benchmark Configs ({len(configs)}):")
    for cfg, count in configs.items():
        print(f"  - {cfg:<30}: {count}")
    print()

    # 2. Checkpoint Verification
    print("-" * 60)
    print("2. CHECKPOINT INTEGRITY VERIFICATION")
    print("-" * 60)
    ckpts = {
        "Spatial": SPATIAL_CKPT,
        "Frequency": FREQ_CKPT,
        "Hybrid": HYBRID_CKPT,
    }
    ckpt_hashes = {}
    for name, path in ckpts.items():
        assert path.exists(), f"Missing checkpoint: {path}"
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        ckpt_hashes[name] = h
        print(f"  {name:<10} ({path.stat().st_size:,} bytes): {h}")
    print()

    # 3. Model Loading
    print("-" * 60)
    print("3. LOADING FROZEN MODELS (EVALUATION MODE)")
    print("-" * 60)
    
    # Spatial
    spatial_model = ManipulationResNet50V1(dropout_p=0.3)
    s_ckpt = torch.load(SPATIAL_CKPT, map_location=DEVICE, weights_only=False)
    s_sd = s_ckpt.get("model_state_dict", s_ckpt)
    spatial_model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in s_sd.items()})
    spatial_model.to(DEVICE).eval()
    print("  Spatial model loaded successfully.")

    # Frequency
    freq_model = ManipulationFrequencyResNet50V1(dropout_p=0.3)
    f_ckpt = torch.load(FREQ_CKPT, map_location=DEVICE, weights_only=False)
    f_sd = f_ckpt.get("model_state_dict", f_ckpt)
    freq_model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in f_sd.items()})
    freq_model.to(DEVICE).eval()
    print("  Frequency model loaded successfully.")

    # Hybrid
    hybrid_model = ManipulationHybridResNet50V1(dropout_p=0.3)
    h_ckpt = torch.load(HYBRID_CKPT, map_location=DEVICE, weights_only=False)
    h_sd = h_ckpt.get("model_state_dict", h_ckpt)
    hybrid_model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in h_sd.items()})
    hybrid_model.to(DEVICE).eval()
    print("  Hybrid model loaded successfully.")
    print()

    # 4. Inference on 700 Images
    print("-" * 60)
    print("4. RUNNING INFERENCE ACROSS ALL THREE MODELS (BATCH SIZE = 16)")
    print("-" * 60)
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_pie)
    
    results = {
        "Spatial": {"preds": [], "probs_real": [], "probs_manip": []},
        "Frequency": {"preds": [], "probs_real": [], "probs_manip": []},
        "Hybrid": {"preds": [], "probs_real": [], "probs_manip": []},
    }
    all_metas = []

    t0 = time.time()
    with torch.no_grad():
        for xs, xf, metas in loader:
            xs = xs.to(DEVICE)
            xf = xf.to(DEVICE)
            
            # Spatial stream
            s_out = spatial_model(xs)
            s_prob = torch.softmax(s_out, dim=1).cpu()
            results["Spatial"]["preds"].extend(s_out.argmax(1).cpu().tolist())
            results["Spatial"]["probs_real"].extend(s_prob[:, 0].tolist())
            results["Spatial"]["probs_manip"].extend(s_prob[:, 1].tolist())

            # Frequency stream
            f_out = freq_model(xf)
            f_prob = torch.softmax(f_out, dim=1).cpu()
            results["Frequency"]["preds"].extend(f_out.argmax(1).cpu().tolist())
            results["Frequency"]["probs_real"].extend(f_prob[:, 0].tolist())
            results["Frequency"]["probs_manip"].extend(f_prob[:, 1].tolist())

            # Hybrid stream
            h_out = hybrid_model(xs, xf)
            h_prob = torch.softmax(h_out, dim=1).cpu()
            results["Hybrid"]["preds"].extend(h_out.argmax(1).cpu().tolist())
            results["Hybrid"]["probs_real"].extend(h_prob[:, 0].tolist())
            results["Hybrid"]["probs_manip"].extend(h_prob[:, 1].tolist())

            all_metas.extend(metas)

    elapsed = time.time() - t0
    print(f"Inference complete: {n_total} images evaluated in {elapsed:.2f}s ({n_total/elapsed:.1f} img/s)")
    print()

    # 5. Analysis & Prediction Distributions
    print("-" * 60)
    print("5. MODEL PREDICTION DISTRIBUTIONS ON PIE-BENCH (700 IMAGES)")
    print("-" * 60)

    summary_data = {
        "benchmark": "PIE-Bench",
        "total_images_evaluated": n_total,
        "images_excluded": 0,
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "device": str(DEVICE),
        "models": {}
    }

    category_list = sorted(list(categories.keys()))
    category_results = {cat: {} for cat in category_list}

    for model_name in ["Spatial", "Frequency", "Hybrid"]:
        preds = results[model_name]["preds"]
        probs_r = results[model_name]["probs_real"]
        probs_m = results[model_name]["probs_manip"]

        n_real = sum(1 for p in preds if p == 0)
        n_manip = sum(1 for p in preds if p == 1)
        mean_p_real = float(np.mean(probs_r))
        mean_p_manip = float(np.mean(probs_m))

        summary_data["models"][model_name] = {
            "checkpoint_hash": ckpt_hashes[model_name],
            "num_predicted_real": n_real,
            "pct_predicted_real": round(n_real / n_total * 100, 2),
            "num_predicted_manipulated": n_manip,
            "pct_predicted_manipulated": round(n_manip / n_total * 100, 2),
            "mean_prob_real": round(mean_p_real, 4),
            "mean_prob_manipulated": round(mean_p_manip, 4),
            "decision_rule": "argmax(logits) [threshold=0.5]",
            # Note on ground truth:
            "ground_truth_status": (
                "PIE-Bench provides editing instruction tasks on 700 benchmark source images. "
                "No binary REAL vs MANIPULATED label column is present in the official manifest, "
                "and no paired edited counterparts are stored in the benchmark repository. "
                "If these 700 inputs are evaluated as unedited source images (ORIGINAL_REAL), "
                "the specificity (REAL recall) is reported below. AI_MANIPULATED recall and pair metrics "
                "are unavailable because no manipulated counterparts exist."
            ),
            "as_source_input_specificity": round(n_real / n_total * 100, 2),
            "as_source_input_fpr": round(n_manip / n_total * 100, 2),
        }

        print(f"Model: {model_name:<10}")
        print(f"  Predicted ORIGINAL_REAL  (0): {n_real:>3} / {n_total} ({n_real/n_total*100:>5.1f}%) | Mean P(Real) = {mean_p_real:.4f}")
        print(f"  Predicted AI_MANIPULATED (1): {n_manip:>3} / {n_total} ({n_manip/n_total*100:>5.1f}%) | Mean P(Manip) = {mean_p_manip:.4f}")
        print()

        # Category breakdown
        for cat in category_list:
            cat_indices = [i for i, m in enumerate(all_metas) if m["category"] == cat]
            c_preds = [preds[i] for i in cat_indices]
            c_n_real = sum(1 for p in c_preds if p == 0)
            c_n_manip = sum(1 for p in c_preds if p == 1)
            category_results[cat][model_name] = {
                "n_images": len(cat_indices),
                "predicted_real": c_n_real,
                "predicted_manipulated": c_n_manip,
                "pct_predicted_real": round(c_n_real / len(cat_indices) * 100, 1),
                "pct_predicted_manipulated": round(c_n_manip / len(cat_indices) * 100, 1),
            }

    # 6. Category-Wise Breakdown Table
    print("-" * 60)
    print("6. CATEGORY-WISE PREDICTION BREAKDOWN ACROSS MODELS")
    print("-" * 60)
    print(f"{'Category':<25} | {'N':>3} | {'Spatial (R / M)':<16} | {'Freq (R / M)':<16} | {'Hybrid (R / M)':<16}")
    print("-" * 85)
    for cat in category_list:
        n_c = categories[cat]
        s_r = category_results[cat]["Spatial"]["predicted_real"]
        s_m = category_results[cat]["Spatial"]["predicted_manipulated"]
        f_r = category_results[cat]["Frequency"]["predicted_real"]
        f_m = category_results[cat]["Frequency"]["predicted_manipulated"]
        h_r = category_results[cat]["Hybrid"]["predicted_real"]
        h_m = category_results[cat]["Hybrid"]["predicted_manipulated"]
        print(f"{cat:<25} | {n_c:>3} | {s_r:>3}R / {s_m:>3}M ({s_r/n_c*100:4.1f}%) | {f_r:>3}R / {f_m:>3}M ({f_r/n_c*100:4.1f}%) | {h_r:>3}R / {h_m:>3}M ({h_r/n_c*100:4.1f}%)")
    print("-" * 85)
    print()

    # 7. Save Artifacts
    print("-" * 60)
    print("7. SAVING EVALUATION OUTPUTS TO manipulation_external_test/results/")
    print("-" * 60)

    # 1. external_test_summary.json
    summary_path = RESULTS_DIR / "external_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  - {summary_path.name}")

    # 2. external_category_metrics.json
    cat_path = RESULTS_DIR / "external_category_metrics.json"
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(category_results, f, indent=2)
    print(f"  - {cat_path.name}")

    # 3. external_test_predictions.csv
    pred_path = RESULTS_DIR / "external_test_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "config_name", "category", "filename",
            "spatial_pred_label", "spatial_pred_class", "spatial_prob_real", "spatial_prob_manip",
            "freq_pred_label", "freq_pred_class", "freq_prob_real", "freq_prob_manip",
            "hybrid_pred_label", "hybrid_pred_class", "hybrid_prob_real", "hybrid_prob_manip",
        ])
        writer.writeheader()
        for i in range(n_total):
            m = all_metas[i]
            writer.writerow({
                "sample_id": m["sample_id"],
                "config_name": m["config_name"],
                "category": m["category"],
                "filename": m["filename"],
                "spatial_pred_label": results["Spatial"]["preds"][i],
                "spatial_pred_class": "ORIGINAL_REAL" if results["Spatial"]["preds"][i] == 0 else "AI_MANIPULATED",
                "spatial_prob_real": round(results["Spatial"]["probs_real"][i], 4),
                "spatial_prob_manip": round(results["Spatial"]["probs_manip"][i], 4),
                "freq_pred_label": results["Frequency"]["preds"][i],
                "freq_pred_class": "ORIGINAL_REAL" if results["Frequency"]["preds"][i] == 0 else "AI_MANIPULATED",
                "freq_prob_real": round(results["Frequency"]["probs_real"][i], 4),
                "freq_prob_manip": round(results["Frequency"]["probs_manip"][i], 4),
                "hybrid_pred_label": results["Hybrid"]["preds"][i],
                "hybrid_pred_class": "ORIGINAL_REAL" if results["Hybrid"]["preds"][i] == 0 else "AI_MANIPULATED",
                "hybrid_prob_real": round(results["Hybrid"]["probs_real"][i], 4),
                "hybrid_prob_manip": round(results["Hybrid"]["probs_manip"][i], 4),
            })
    print(f"  - {pred_path.name}")

    # 4. external_confusion_matrices.csv
    cm_path = RESULTS_DIR / "external_confusion_matrices.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Predicted_ORIGINAL_REAL", "Predicted_AI_MANIPULATED", "Total_Evaluated", "Note"])
        for m_name in ["Spatial", "Frequency", "Hybrid"]:
            n_r = summary_data["models"][m_name]["num_predicted_real"]
            n_m = summary_data["models"][m_name]["num_predicted_manipulated"]
            writer.writerow([m_name, n_r, n_m, n_total, "PIE-Bench manifest contains 700 benchmark input images; no paired edited counterparts."])
    print(f"  - {cm_path.name}")

    # 5. external_test_report.txt (human-readable comprehensive report)
    report_path = RESULTS_DIR / "external_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AIDETECT PHASE 9: PIE-BENCH EXTERNAL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Evaluation Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Benchmark: PIE-Bench (700 images, 512x512, PNG format)\n")
        f.write(f"Manifest: manipulation_external_test/pie_bench/pie_bench_manifest.csv\n\n")
        f.write("1. DATASET CHARACTERISTICS & GROUND TRUTH STATUS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total benchmark images: {n_total}\n")
        f.write("Hash integrity: 700 / 700 SHA-256 matches verified.\n")
        f.write("Ground truth binary labels: NOT annotated in official manifest.\n")
        f.write("Original-manipulated pairs: NOT present (single images per sample).\n")
        f.write("Task: PIE-Bench provides source images with target edit prompts/actions.\n\n")
        f.write("2. MODEL PREDICTION SUMMARY\n")
        f.write("-" * 50 + "\n")
        for m_name in ["Spatial", "Frequency", "Hybrid"]:
            m_info = summary_data["models"][m_name]
            f.write(f"Model: {m_name}\n")
            f.write(f"  Checkpoint SHA-256: {m_info['checkpoint_hash']}\n")
            f.write(f"  Predicted ORIGINAL_REAL  (0): {m_info['num_predicted_real']} / {n_total} ({m_info['pct_predicted_real']}%)\n")
            f.write(f"  Predicted AI_MANIPULATED (1): {m_info['num_predicted_manipulated']} / {n_total} ({m_info['pct_predicted_manipulated']}%)\n")
            f.write(f"  Mean P(Real): {m_info['mean_prob_real']}, Mean P(Manipulated): {m_info['mean_prob_manipulated']}\n")
            f.write(f"  Specificity (if source=Real): {m_info['as_source_input_specificity']}%\n")
            f.write(f"  FPR (if source=Real): {m_info['as_source_input_fpr']}%\n\n")
        f.write("3. CATEGORY-WISE PREDICTIONS\n")
        f.write("-" * 50 + "\n")
        for cat in category_list:
            f.write(f"Category: {cat} (n={categories[cat]})\n")
            for m_name in ["Spatial", "Frequency", "Hybrid"]:
                c_info = category_results[cat][m_name]
                f.write(f"  {m_name:<10}: {c_info['predicted_real']} Real ({c_info['pct_predicted_real']}%) | {c_info['predicted_manipulated']} Manip ({c_info['pct_predicted_manipulated']}%)\n")
            f.write("\n")
    print(f"  - {report_path.name}")
    print()

    # 8. Post-Evaluation Integrity Confirmation
    print("-" * 60)
    print("8. DATA INTEGRITY AUDIT AFTER EVALUATION")
    print("-" * 60)
    post_disk_files = list(PIE_DIR.glob("images/*.png"))
    print(f"PIE-Bench image count on disk: {len(post_disk_files)} (expected 700)")
    assert len(post_disk_files) == 700, "Image count changed!"
    
    # Check that checkpoints remain identical
    for name, path in ckpts.items():
        with open(path, "rb") as f:
            h_after = hashlib.sha256(f.read()).hexdigest()
        assert h_after == ckpt_hashes[name], f"Checkpoint {name} was modified!"
        print(f"  {name:<10} checkpoint hash unchanged: {h_after[:16]}...")
    print("All checkpoints and benchmark files confirmed 100% UNCHANGED.")
    print()
    print("=" * 70)
    print("EXTERNAL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    sys.stdout.flush()

if __name__ == "__main__":
    run_external_evaluation()

