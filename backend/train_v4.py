"""
AIDetect V4 Training Script
==============================
Trains three V4 binary classifiers (REAL vs AI-GENERATED):
  1. Spatial V4    -> models/spatial_resnet50_v4.pth
  2. Frequency V4  -> models/frequency_resnet50_v4.pth
  3. Hybrid V4     -> models/hybrid_resnet50_fft_v4.pth

Configuration:
  Optimizer:     AdamW (lr=1e-4, weight_decay=1e-3)
  LR Scheduler:  CosineAnnealingLR(T_max=15, eta_min=1e-6)
  Batch size:    16
  Epochs:        15 (max)
  Early stopping: patience=5, based on validation F1 (higher is better)
  Best checkpoint: selected by best validation F1 (NOT test set)
  Class weights:  inverse-frequency weighted (handles REAL/AI imbalance)
  AMP:           Enabled (torch.amp.autocast, CUDA only)
  Seed:          42

V4 results saved to backend/v4_results/
V3 checkpoints and code are NOT modified.

Usage:
  cd backend
  .\\venv\\Scripts\\python.exe train_v4.py
"""

import os
import sys
import json
import time
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_dataset import (
    V4Dataset, collate_spatial, collate_hybrid,
    PREPROCESSING_DOC, print_dataset_summary, load_metadata,
)
from v4_models import (
    SpatialV4, FrequencyV4, HybridV4,
    FUSED_DIM, SPATIAL_FEATURE_DIM, FREQUENCY_FEATURE_DIM,
    ARCHITECTURE_DOC,
)

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR   = PROJECT_DIR / "models"
V4_RESULTS  = BASE_DIR / "v4_results"

RESULTS_DIRS = {
    "training_logs":      V4_RESULTS / "training_logs",
    "validation_results": V4_RESULTS / "validation_results",
    "checkpoints":        V4_RESULTS / "checkpoints",
    "metrics":            V4_RESULTS / "metrics",
    "configs":            V4_RESULTS / "configs",
}

for d in [MODEL_DIR] + list(RESULTS_DIRS.values()):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------------------------

SEED         = 42
BATCH_SIZE   = 16
MAX_EPOCHS   = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-3
T_MAX        = 15    # CosineAnnealingLR period
ETA_MIN      = 1e-6  # CosineAnnealingLR minimum LR
ES_PATIENCE  = 5     # Early stopping patience (epochs)
NUM_WORKERS  = 0     # Windows compatibility

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seeds(SEED)

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("AIDETECT V4 TRAINING")
print("=" * 70)
print("PyTorch version : {}".format(torch.__version__))
print("CUDA available  : {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("GPU             : {}".format(torch.cuda.get_device_name(0)))
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print("GPU memory      : {:.2f} GB".format(total_mem / 1e9))
print("Device          : {}".format(DEVICE))
print("Seed            : {}".format(SEED))
print()

if not torch.cuda.is_available():
    print("[WARNING] CUDA is not available. Training will use CPU.")

USE_AMP = DEVICE.type == "cuda"
print("Mixed precision (AMP): {}".format(USE_AMP))
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------------------------

print_dataset_summary()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# CLASS WEIGHTS (inverse-frequency)
# ---------------------------------------------------------------------------

train_meta = load_metadata("train")
n_real  = sum(1 for r in train_meta if r["ground_truth"] == "REAL")
n_ai    = sum(1 for r in train_meta if r["ground_truth"] == "AI")
n_total = n_real + n_ai

w_real = n_total / (2 * n_real)
w_ai   = n_total / (2 * n_ai)
class_weights = torch.tensor([w_real, w_ai], dtype=torch.float32).to(DEVICE)

print("\nClass weights (inverse-frequency):")
print("  REAL (class 0): {:.4f}  (n={})".format(w_real, n_real))
print("  AI   (class 1): {:.4f}  (n={})".format(w_ai, n_ai))
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# SAVE TRAINING CONFIG
# ---------------------------------------------------------------------------

TRAINING_CONFIG = {
    "seed":           SEED,
    "batch_size":     BATCH_SIZE,
    "max_epochs":     MAX_EPOCHS,
    "learning_rate":  LR,
    "weight_decay":   WEIGHT_DECAY,
    "optimizer":      "AdamW",
    "lr_scheduler":   "CosineAnnealingLR(T_max={}, eta_min={})".format(T_MAX, ETA_MIN),
    "early_stopping": "patience={}, monitor=val_f1 (higher=better)".format(ES_PATIENCE),
    "checkpoint_selection": "best_val_f1",
    "amp":            USE_AMP,
    "num_workers":    NUM_WORKERS,
    "class_weights":  {"REAL": w_real, "AI": w_ai},
    "class_mapping":  {"0": "REAL", "1": "AI"},
    "input_size":     224,
    "preprocessing":  PREPROCESSING_DOC,
    "architectures":  ARCHITECTURE_DOC,
    "dataset": {
        "path":        "dataset_v4/",
        "train_real":  n_real,
        "train_ai":    n_ai,
        "train_total": n_total,
    },
    "device":          str(DEVICE),
    "gpu":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "pytorch_version": torch.__version__,
    "timestamp_start": datetime.datetime.now().isoformat(),
}

with open(RESULTS_DIRS["configs"] / "v4_training_config.json", "w") as f:
    json.dump(TRAINING_CONFIG, f, indent=2)
print("Saved training config -> v4_results/configs/v4_training_config.json")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# METRICS UTILITY
# ---------------------------------------------------------------------------

def compute_metrics(all_labels, all_preds):
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    real_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
    ai_recall   = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "accuracy":    float(acc),
        "precision":   float(prec),
        "recall":      float(rec),
        "f1":          float(f1),
        "real_recall": float(real_recall),
        "ai_recall":   float(ai_recall),
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train_v4_model(model_name, model, train_loader, val_loader,
                   model_save_path, is_hybrid=False):
    """
    Train a V4 model with:
      - AdamW + CosineAnnealingLR(T_max=15, eta_min=1e-6)
      - Class-weighted CrossEntropyLoss
      - AMP (if CUDA)
      - Early stopping on val F1 (patience=ES_PATIENCE)
      - Best checkpoint saved by val F1
    """
    print("\n" + "=" * 70)
    print("TRAINING: {}".format(model_name))
    print("=" * 70)
    sys.stdout.flush()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    history       = []
    best_val_f1   = -1.0
    best_epoch    = -1
    epochs_no_imp = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # TRAIN
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch in train_loader:
            if is_hybrid:
                x_s, x_f, y, _ = batch
                x_s = x_s.to(DEVICE, non_blocking=True)
                x_f = x_f.to(DEVICE, non_blocking=True)
            else:
                x, y, _ = batch
                x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x_s, x_f) if is_hybrid else model(x)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss   += loss.item() * y.size(0)
            preds         = logits.argmax(1).cpu().tolist()
            train_preds  += preds
            train_labels += y.cpu().tolist()

        scheduler.step()

        train_loss /= len(train_labels)
        train_acc   = accuracy_score(train_labels, train_preds)
        train_f1    = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                if is_hybrid:
                    x_s, x_f, y, _ = batch
                    x_s = x_s.to(DEVICE, non_blocking=True)
                    x_f = x_f.to(DEVICE, non_blocking=True)
                else:
                    x, y, _ = batch
                    x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(x_s, x_f) if is_hybrid else model(x)
                    loss   = criterion(logits, y)

                val_loss       += loss.item() * y.size(0)
                preds           = logits.argmax(1).cpu().tolist()
                val_preds       += preds
                val_labels_all  += y.cpu().tolist()

        val_loss /= len(val_labels_all)
        val_metrics = compute_metrics(val_labels_all, val_preds)
        val_f1      = val_metrics["f1"]
        val_acc     = val_metrics["accuracy"]

        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        epoch_log = {
            "epoch":           epoch,
            "train_loss":      round(train_loss, 6),
            "train_accuracy":  round(train_acc, 6),
            "train_f1":        round(train_f1, 6),
            "val_loss":        round(val_loss, 6),
            "val_accuracy":    round(val_acc, 6),
            "val_precision":   round(val_metrics["precision"], 6),
            "val_recall":      round(val_metrics["recall"], 6),
            "val_f1":          round(val_f1, 6),
            "val_real_recall": round(val_metrics["real_recall"], 6),
            "val_ai_recall":   round(val_metrics["ai_recall"], 6),
            "val_tp": val_metrics["TP"],
            "val_tn": val_metrics["TN"],
            "val_fp": val_metrics["FP"],
            "val_fn": val_metrics["FN"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "learning_rate":   current_lr,
            "elapsed_sec":     round(elapsed, 1),
        }
        history.append(epoch_log)

        print(
            "Epoch {:02d}/{} | train_loss={:.4f} train_acc={:.2f}% | "
            "val_loss={:.4f} val_acc={:.2f}% val_f1={:.4f} | "
            "lr={:.2e} | {:.1f}s".format(
                epoch, MAX_EPOCHS,
                train_loss, train_acc * 100,
                val_loss, val_acc * 100, val_f1,
                current_lr, elapsed
            )
        )
        sys.stdout.flush()

        # CHECKPOINT & EARLY STOPPING
        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            best_epoch    = epoch
            epochs_no_imp = 0

            checkpoint = {
                "model_state_dict":   model.state_dict(),
                "model_name":         model_name,
                "version":            "V4",
                "class_mapping":      {"0": "REAL", "1": "AI"},
                "input_size":         224,
                "seed":               SEED,
                "preprocessing":      PREPROCESSING_DOC,
                "augmentation":       "see PREPROCESSING_DOC spatial_train",
                "training_config":    TRAINING_CONFIG,
                "epoch":              epoch,
                "best_val_f1":        best_val_f1,
                "validation_metrics": val_metrics,
                "training_history":   history,
            }
            torch.save(checkpoint, model_save_path)
            print("  [OK] Best checkpoint saved (val_f1={:.4f}) -> {}".format(
                best_val_f1, model_save_path.name))

            ckpt_copy_path = RESULTS_DIRS["checkpoints"] / model_save_path.name
            torch.save(checkpoint, ckpt_copy_path)
        else:
            epochs_no_imp += 1
            print("  No improvement ({}/{}) [best val_f1={:.4f} at epoch {}]".format(
                epochs_no_imp, ES_PATIENCE, best_val_f1, best_epoch))
            if epochs_no_imp >= ES_PATIENCE:
                print("  Early stopping triggered at epoch {}.".format(epoch))
                break

        sys.stdout.flush()

    print("\n{} training complete.".format(model_name))
    print("  Best epoch:   {}".format(best_epoch))
    print("  Best val F1:  {:.4f}".format(best_val_f1))
    sys.stdout.flush()

    log_path = RESULTS_DIRS["training_logs"] / "{}_training_log.json".format(
        model_name.lower().replace(" ", "_"))
    with open(log_path, "w") as f:
        json.dump({"model": model_name, "history": history}, f, indent=2)
    print("  Training log -> {}".format(log_path.name))
    sys.stdout.flush()

    return {
        "model_name":   model_name,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "total_epochs": epoch,
        "history":      history,
    }


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

print("\nBuilding data loaders...")
sys.stdout.flush()

spatial_train_ds = V4Dataset("train",      mode="spatial", augment=True)
spatial_val_ds   = V4Dataset("validation", mode="spatial", augment=False)

spatial_train_loader = DataLoader(
    spatial_train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_spatial, drop_last=False,
)
spatial_val_loader = DataLoader(
    spatial_val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_spatial, drop_last=False,
)

freq_train_ds = V4Dataset("train",      mode="frequency", augment=False)
freq_val_ds   = V4Dataset("validation", mode="frequency", augment=False)

freq_train_loader = DataLoader(
    freq_train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_spatial, drop_last=False,
)
freq_val_loader = DataLoader(
    freq_val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_spatial, drop_last=False,
)

hybrid_train_ds = V4Dataset("train",      mode="hybrid", augment=True)
hybrid_val_ds   = V4Dataset("validation", mode="hybrid", augment=False)

hybrid_train_loader = DataLoader(
    hybrid_train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_hybrid, drop_last=False,
)
hybrid_val_loader = DataLoader(
    hybrid_val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_hybrid, drop_last=False,
)

print("  Spatial  train batches: {:3d}  ({} images)".format(
    len(spatial_train_loader), len(spatial_train_ds)))
print("  Spatial  val batches:   {:3d}  ({} images)".format(
    len(spatial_val_loader), len(spatial_val_ds)))
print("  Frequency train batches: {:3d}  ({} images)".format(
    len(freq_train_loader), len(freq_train_ds)))
print("  Frequency val batches:   {:3d}  ({} images)".format(
    len(freq_val_loader), len(freq_val_ds)))
print("  Hybrid   train batches: {:3d}  ({} images)".format(
    len(hybrid_train_loader), len(hybrid_train_ds)))
print("  Hybrid   val batches:   {:3d}  ({} images)".format(
    len(hybrid_val_loader), len(hybrid_val_ds)))
sys.stdout.flush()

# ---------------------------------------------------------------------------
# VERIFY V3 CHECKPOINTS EXIST (safety check before training)
# ---------------------------------------------------------------------------

v3_models = [
    MODEL_DIR / "spatial_resnet50_v3.pth",
    MODEL_DIR / "frequency_resnet50_v3.pth",
    MODEL_DIR / "hybrid_resnet50_fft_v3.pth",
]
print("\nVerifying V3 checkpoints are untouched...")
for p in v3_models:
    status = "EXISTS" if p.exists() else "MISSING!"
    print("  {}: [{}]".format(p.name, status))
    if not p.exists():
        raise FileNotFoundError("V3 checkpoint missing: {}".format(p))
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 1. TRAIN SPATIAL V4
# ---------------------------------------------------------------------------

set_seeds(SEED)
spatial_results = train_v4_model(
    model_name      = "Spatial V4",
    model           = SpatialV4(),
    train_loader    = spatial_train_loader,
    val_loader      = spatial_val_loader,
    model_save_path = MODEL_DIR / "spatial_resnet50_v4.pth",
    is_hybrid       = False,
)

# ---------------------------------------------------------------------------
# 2. TRAIN FREQUENCY V4
# ---------------------------------------------------------------------------

set_seeds(SEED)
frequency_results = train_v4_model(
    model_name      = "Frequency V4",
    model           = FrequencyV4(),
    train_loader    = freq_train_loader,
    val_loader      = freq_val_loader,
    model_save_path = MODEL_DIR / "frequency_resnet50_v4.pth",
    is_hybrid       = False,
)

# ---------------------------------------------------------------------------
# 3. TRAIN HYBRID V4
# ---------------------------------------------------------------------------

set_seeds(SEED)
hybrid_results = train_v4_model(
    model_name      = "Hybrid V4",
    model           = HybridV4(),
    train_loader    = hybrid_train_loader,
    val_loader      = hybrid_val_loader,
    model_save_path = MODEL_DIR / "hybrid_resnet50_fft_v4.pth",
    is_hybrid       = True,
)

# ---------------------------------------------------------------------------
# TRAINING SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("V4 TRAINING COMPLETE -- SUMMARY")
print("=" * 70)
sys.stdout.flush()

all_results = [spatial_results, frequency_results, hybrid_results]
summary = []

for r in all_results:
    best_ep  = r["best_epoch"]
    best_log = r["history"][best_ep - 1]
    print("\n{}:".format(r["model_name"]))
    print("  Best epoch:      {}".format(best_ep))
    print("  Total epochs:    {}".format(r["total_epochs"]))
    print("  Best val F1:     {:.4f}".format(r["best_val_f1"]))
    print("  Val accuracy:    {:.2f}%".format(best_log["val_accuracy"] * 100))
    print("  Val precision:   {:.4f}".format(best_log["val_precision"]))
    print("  Val recall:      {:.4f}".format(best_log["val_recall"]))
    print("  Val REAL recall: {:.4f}".format(best_log["val_real_recall"]))
    print("  Val AI recall:   {:.4f}".format(best_log["val_ai_recall"]))
    print("  CM: {}".format(best_log["val_confusion_matrix"]))
    summary.append({
        "model":        r["model_name"],
        "best_epoch":   best_ep,
        "total_epochs": r["total_epochs"],
        "best_val_f1":  r["best_val_f1"],
        "best_val_log": best_log,
    })
    sys.stdout.flush()

summary_path = RESULTS_DIRS["metrics"] / "v4_training_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "config":    TRAINING_CONFIG,
        "results":   summary,
        "timestamp_end": datetime.datetime.now().isoformat(),
    }, f, indent=2)
print("\nTraining summary -> {}".format(summary_path))
print("\nCheckpoints saved:")
for name, fname in [
    ("Spatial V4",   "spatial_resnet50_v4.pth"),
    ("Frequency V4", "frequency_resnet50_v4.pth"),
    ("Hybrid V4",    "hybrid_resnet50_fft_v4.pth"),
]:
    p = MODEL_DIR / fname
    size_mb = p.stat().st_size / 1e6 if p.exists() else 0
    print("  {}: models/{} ({:.1f} MB)".format(name, fname, size_mb))

print("\nNEXT STEP: Run backend/evaluate_v4.py for clean test evaluation.")
print("DO NOT proceed to AI-MANIPULATED detection yet.")
sys.stdout.flush()
