"""
AIDetect V4 Test Evaluation Script
=====================================
Performs CLEAN test evaluation of the three V4 models on:
  dataset_v4/test/

This script is run ONCE, after training is complete and the best checkpoints
have been selected using validation data.

DO NOT use this script for model selection -- test set is held out.
DO NOT modify this file to evaluate on the validation set.
DO NOT mix V3 and V4 test results.

Outputs saved to backend/v4_results/validation_results/:
  v4_test_metrics.json         -- per-model and aggregate metrics
  v4_test_predictions.csv      -- per-image predictions
  v4_test_domain_metrics.json  -- domain-specific analysis
  v4_confusion_matrix_*.json   -- per-model confusion matrices

Label convention: 0 = REAL, 1 = AI

Usage:
  cd backend
  .\\venv\\Scripts\\python.exe evaluate_v4.py
"""

import os
import sys
import csv
import json
import time
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_dataset import (
    V4Dataset, collate_spatial, collate_hybrid,
    PREPROCESSING_DOC, load_metadata,
)
from v4_models import SpatialV4, FrequencyV4, HybridV4

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR   = PROJECT_DIR / "models"
V4_RESULTS  = BASE_DIR / "v4_results"
OUT_DIR     = V4_RESULTS / "validation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATHS = {
    "Spatial V4":   MODEL_DIR / "spatial_resnet50_v4.pth",
    "Frequency V4": MODEL_DIR / "frequency_resnet50_v4.pth",
    "Hybrid V4":    MODEL_DIR / "hybrid_resnet50_fft_v4.pth",
}

MODEL_CLASSES = {
    "Spatial V4":   SpatialV4,
    "Frequency V4": FrequencyV4,
    "Hybrid V4":    HybridV4,
}

IS_HYBRID = {
    "Spatial V4":   False,
    "Frequency V4": False,
    "Hybrid V4":    True,
}

DATASET_MODE = {
    "Spatial V4":   "spatial",
    "Frequency V4": "frequency",
    "Hybrid V4":    "hybrid",
}

COLLATE_FN = {
    "Spatial V4":   collate_spatial,
    "Frequency V4": collate_spatial,
    "Hybrid V4":    collate_hybrid,
}

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("AIDETECT V4 TEST EVALUATION")
print("=" * 70)
print("Device : {}".format(DEVICE))
if torch.cuda.is_available():
    print("GPU    : {}".format(torch.cuda.get_device_name(0)))
print("Time   : {}".format(datetime.datetime.now().isoformat()))
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# VERIFY V3 CHECKPOINTS STILL INTACT
# ---------------------------------------------------------------------------

print("Verifying V3 checkpoints are untouched...")
v3_models = [
    MODEL_DIR / "spatial_resnet50_v3.pth",
    MODEL_DIR / "frequency_resnet50_v3.pth",
    MODEL_DIR / "hybrid_resnet50_fft_v3.pth",
]
for p in v3_models:
    status = "EXISTS" if p.exists() else "MISSING!"
    print("  {}: [{}]".format(p.name, status))
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# VERIFY V4 CHECKPOINTS EXIST
# ---------------------------------------------------------------------------

print("Verifying V4 checkpoints...")
for name, path in CHECKPOINT_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(
            "V4 checkpoint not found: {}\nRun train_v4.py first.".format(path)
        )
    size_mb = path.stat().st_size / 1e6
    print("  {}: {} ({:.1f} MB)  OK".format(name, path.name, size_mb))
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# DOMAIN DEFINITIONS
# ---------------------------------------------------------------------------

DOMAINS_REAL = ["genimage", "historical", "modern"]
DOMAINS_AI   = ["genimage", "historical_style", "modern"]
ALL_DOMAINS  = list(set(DOMAINS_REAL + DOMAINS_AI))

# ---------------------------------------------------------------------------
# METRICS UTILITY
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds, label_str=""):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels, preds, average="macro", zero_division=0)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(labels, preds, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        real_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
        ai_recall   = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)
    else:
        tn = fp = fn = tp = 0
        real_recall = ai_recall = 0.0

    return {
        "n_samples":   len(labels),
        "accuracy":    round(float(acc),  4),
        "precision":   round(float(prec), 4),
        "recall":      round(float(rec),  4),
        "f1":          round(float(f1),   4),
        "real_recall": round(float(real_recall), 4),
        "ai_recall":   round(float(ai_recall),   4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "confusion_matrix": cm.tolist(),
        "label_context": label_str,
    }


def print_metrics(name, metrics):
    print("  {}:".format(name))
    print("    Accuracy  : {:.2f}%".format(metrics["accuracy"] * 100))
    print("    Precision : {:.4f}".format(metrics["precision"]))
    print("    Recall    : {:.4f}".format(metrics["recall"]))
    print("    F1        : {:.4f}".format(metrics["f1"]))
    print("    REAL recall: {:.4f}".format(metrics["real_recall"]))
    print("    AI recall  : {:.4f}".format(metrics["ai_recall"]))
    print("    TP={} TN={} FP={} FN={}".format(
        metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]))
    print("    Confusion matrix: {}".format(metrics["confusion_matrix"]))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# LOAD MODEL FROM CHECKPOINT
# ---------------------------------------------------------------------------

def load_v4_model(model_name):
    ckpt_path  = CHECKPOINT_PATHS[model_name]
    ModelClass = MODEL_CLASSES[model_name]

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = ModelClass()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch      = checkpoint.get("epoch", "?")
        val_f1     = checkpoint.get("best_val_f1", "?")
        print("  {}: loaded epoch={}, best_val_f1={}".format(model_name, epoch, val_f1))
    else:
        state_dict = checkpoint
        print("  {}: loaded (raw state dict)".format(model_name))

    cleaned = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# BUILD TEST DATASETS (no augmentation)
# ---------------------------------------------------------------------------

print("Building test datasets (NO augmentation)...")
sys.stdout.flush()

test_datasets = {
    "Spatial V4":   V4Dataset("test", mode="spatial",   augment=False),
    "Frequency V4": V4Dataset("test", mode="frequency",  augment=False),
    "Hybrid V4":    V4Dataset("test", mode="hybrid",     augment=False),
}

for name, ds in test_datasets.items():
    print("  {}: {} test images".format(name, len(ds)))
sys.stdout.flush()

# ---------------------------------------------------------------------------
# EVALUATE EACH MODEL
# ---------------------------------------------------------------------------

all_model_results = {}
all_predictions   = []

print("\nLoading V4 checkpoints...")
sys.stdout.flush()

for model_name in ["Spatial V4", "Frequency V4", "Hybrid V4"]:
    print("\n" + "-" * 60)
    print("EVALUATING: {}".format(model_name))
    print("-" * 60)
    sys.stdout.flush()

    model = load_v4_model(model_name)

    ds = test_datasets[model_name]
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=COLLATE_FN[model_name],
    )
    is_hybrid = IS_HYBRID[model_name]

    all_labels, all_preds = [], []
    all_probs_real, all_probs_ai = [], []
    all_domains = []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            if is_hybrid:
                x_s, x_f, y, domains = batch
                x_s = x_s.to(DEVICE)
                x_f = x_f.to(DEVICE)
                logits = model(x_s, x_f)
            else:
                x, y, domains = batch
                x = x.to(DEVICE)
                logits = model(x)

            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(1).cpu().tolist()
            labels_list = y.tolist()

            all_labels  += labels_list
            all_preds   += preds
            all_probs_real += probs[:, 0].tolist()
            all_probs_ai   += probs[:, 1].tolist()
            all_domains    += list(domains)

    elapsed = time.time() - t0
    print("  Inference complete ({} images, {:.1f}s)".format(len(all_labels), elapsed))
    sys.stdout.flush()

    overall_metrics = compute_metrics(all_labels, all_preds, "overall_test")
    print("\nOverall:")
    print_metrics("Overall", overall_metrics)

    domain_metrics = {}
    for domain in ALL_DOMAINS:
        idxs = [i for i, d in enumerate(all_domains) if d == domain]
        if not idxs:
            continue
        d_labels = [all_labels[i] for i in idxs]
        d_preds  = [all_preds[i]  for i in idxs]
        domain_metrics[domain] = compute_metrics(d_labels, d_preds, domain)
        print_metrics("Domain: {}".format(domain), domain_metrics[domain])

    all_model_results[model_name] = {
        "overall":     overall_metrics,
        "domains":     domain_metrics,
        "elapsed_sec": round(elapsed, 2),
    }

    for i in range(len(ds)):
        path, true_label, domain = ds.samples[i]
        all_predictions.append({
            "model":           model_name,
            "filepath":        path,
            "domain":          domain,
            "true_label":      true_label,
            "true_class":      "REAL" if true_label == 0 else "AI",
            "predicted_label": all_preds[i],
            "predicted_class": "REAL" if all_preds[i] == 0 else "AI",
            "prob_real":       round(all_probs_real[i], 4),
            "prob_ai":         round(all_probs_ai[i], 4),
            "correct":         int(all_preds[i] == true_label),
        })

# ---------------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING V4 TEST RESULTS")
print("=" * 70)
sys.stdout.flush()

# 1. v4_test_metrics.json
metrics_path = OUT_DIR / "v4_test_metrics.json"
with open(metrics_path, "w") as f:
    json.dump({
        "timestamp":    datetime.datetime.now().isoformat(),
        "dataset":      "dataset_v4/test/",
        "label_convention": {"0": "REAL", "1": "AI"},
        "note": (
            "Test evaluation performed ONCE after best checkpoint selected "
            "using validation F1. Test set was NOT used for model selection."
        ),
        "results": all_model_results,
    }, f, indent=2)
print("  Test metrics     -> {}".format(metrics_path.name))

# 2. v4_test_domain_metrics.json
domain_path = OUT_DIR / "v4_test_domain_metrics.json"
domain_summary = {}
for model_name, res in all_model_results.items():
    domain_summary[model_name] = res["domains"]
with open(domain_path, "w") as f:
    json.dump({
        "timestamp": datetime.datetime.now().isoformat(),
        "domain_definitions": {
            "REAL": DOMAINS_REAL,
            "AI":   DOMAINS_AI,
        },
        "domain_metrics": domain_summary,
    }, f, indent=2)
print("  Domain metrics   -> {}".format(domain_path.name))

# 3. Confusion matrices
for model_name, res in all_model_results.items():
    safe_name = model_name.lower().replace(" ", "_")
    cm_path   = OUT_DIR / "v4_confusion_matrix_{}.json".format(safe_name)
    cm_data   = {
        "model": model_name,
        "confusion_matrix": res["overall"]["confusion_matrix"],
        "layout": "[[TN, FP], [FN, TP]]  (REAL=0, AI=1)",
        "TP": res["overall"]["TP"],
        "TN": res["overall"]["TN"],
        "FP": res["overall"]["FP"],
        "FN": res["overall"]["FN"],
    }
    with open(cm_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print("  Confusion matrix -> {}".format(cm_path.name))

# 4. v4_test_predictions.csv
pred_path = OUT_DIR / "v4_test_predictions.csv"
fieldnames = [
    "model", "filepath", "domain",
    "true_label", "true_class",
    "predicted_label", "predicted_class",
    "prob_real", "prob_ai", "correct",
]
with open(pred_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_predictions)
print("  Predictions CSV  -> {}".format(pred_path.name))
sys.stdout.flush()

# ---------------------------------------------------------------------------
# FINAL SUMMARY TABLE
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("V4 CLEAN TEST EVALUATION -- FINAL SUMMARY")
print("=" * 70)
print("{:<15} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>5}".format(
    "Model", "Acc%", "Prec", "Rec", "F1",
    "REAL_R", "AI_R", "TP", "TN", "FP", "FN"))
print("-" * 70)

for model_name in ["Spatial V4", "Frequency V4", "Hybrid V4"]:
    m = all_model_results[model_name]["overall"]
    print("{:<15} {:>7.2f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>5} {:>5} {:>5} {:>5}".format(
        model_name, m["accuracy"] * 100,
        m["precision"], m["recall"], m["f1"],
        m["real_recall"], m["ai_recall"],
        m["TP"], m["TN"], m["FP"], m["FN"]
    ))

print("\n" + "-" * 70)
print("DOMAIN-SPECIFIC RESULTS:")
print("-" * 70)
for model_name in ["Spatial V4", "Frequency V4", "Hybrid V4"]:
    print("\n  {}:".format(model_name))
    for domain, dm in all_model_results[model_name]["domains"].items():
        print("    {:<20} n={:>3}  acc={:.1f}%  f1={:.4f}  real_r={:.3f}  ai_r={:.3f}".format(
            domain, dm["n_samples"], dm["accuracy"] * 100,
            dm["f1"], dm["real_recall"], dm["ai_recall"]
        ))

print("\n" + "=" * 70)
print("V4 TEST EVALUATION COMPLETE")
print("Results saved to: {}".format(OUT_DIR))
print("=" * 70)
print("\nNOTE: DO NOT proceed to AI-MANIPULATED detection.")
print("STOP here and wait for next instruction.")
sys.stdout.flush()
