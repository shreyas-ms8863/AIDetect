"""
AIDetect Phase 25: Multi-Generator Manipulation Detector V2 Training Script
============================================================================
Trains the research model ManipulationFrequencyResNet50V2 on dataset_manipulation_v2/
using the authoritative 2D FFT preprocessing pipeline and controlled augmentations.

Configuration (Section 13):
  - Model: ManipulationFrequencyResNet50V2 (ResNet50 + Dropout(0.3) + Linear(2048, 2))
  - Batch size: 16
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-3)
  - Scheduler: CosineAnnealingLR (T_max=15, eta_min=1e-6)
  - Epochs: 15 (Early stopping patience: 5 on validation Macro F1)
  - Mixed Precision (AMP): Enabled
  - Random seed: 42
  - Checkpoint output: models/manipulation_frequency_resnet50_v2.pth
  - History output: backend/phase25_results/training_history.json
"""

import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.manipulation_v2_models import ManipulationFrequencyResNet50V2
from backend.manipulation_v2_dataset import ManipulationV2Dataset

MODELS_DIR   = BASE_DIR / "models"
RESULTS_DIR  = BASE_DIR / "backend" / "phase25_results"
DATASET_DIR  = BASE_DIR / "dataset_manipulation_v2"
METADATA_CSV = DATASET_DIR / "metadata" / "metadata.csv"
V2_CKPT_PATH = MODELS_DIR / "manipulation_frequency_resnet50_v2.pth"

SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    n = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    manip_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (manip_recall + real_recall) / 2.0

    manip_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    real_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    macro_prec = (manip_prec + real_prec) / 2.0

    manip_f1 = (2 * manip_prec * manip_recall / (manip_prec + manip_recall)) if (manip_prec + manip_recall) > 0 else 0.0
    real_f1 = (2 * real_prec * real_recall / (real_prec + real_recall)) if (real_prec + real_recall) > 0 else 0.0
    macro_f1 = (manip_f1 + real_f1) / 2.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": round(accuracy * 100, 2),
        "balanced_accuracy": round(balanced_acc * 100, 2),
        "real_recall": round(real_recall * 100, 2),
        "manipulation_recall": round(manip_recall * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "manipulation_f1": round(manip_f1 * 100, 2),
        "real_f1": round(real_f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def train_model():
    print("=" * 80)
    print("AIDETECT PHASE 25: MULTI-GENERATOR MANIPULATION DETECTOR V2 TRAINING")
    print("=" * 80)

    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset metadata
    df = pd.read_csv(METADATA_CSV)
    train_samples = df[df["split"] == "train"].to_dict("records")
    val_samples   = df[df["split"] == "val"].to_dict("records")

    print(f"Loaded {len(df)} total dataset records.")
    print(f"  Training samples  : {len(train_samples)} ({len(train_samples)//2} pairs)")
    print(f"  Validation samples: {len(val_samples)} ({len(val_samples)//2} pairs)")

    # Generator distribution in train
    train_gens = Counter(r["generator"] for r in train_samples if r["ground_truth"] == 1)
    print(f"  Train generators  : {dict(train_gens)}")

    # 2. Datasets and Loaders
    train_dataset = ManipulationV2Dataset(train_samples, is_training=True, base_dir=BASE_DIR)
    val_dataset   = ManipulationV2Dataset(val_samples, is_training=False, base_dir=BASE_DIR)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model, Criterion, Optimizer, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing ResNet-50 V2 on device: {device}")
    model = ManipulationFrequencyResNet50V2(dropout_p=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # 4. Training Loop
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        # Training phase
        model.train()
        train_loss_total = 0.0
        train_preds, train_targets = [], []

        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(batch_y.detach().cpu().numpy())

        train_loss_avg = train_loss_total / len(train_dataset)
        train_acc = (np.array(train_preds) == np.array(train_targets)).mean() * 100.0

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                val_loss_total += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.cpu().numpy())

        val_loss_avg = val_loss_total / len(val_dataset)
        val_metrics = compute_classification_metrics(val_targets, val_preds)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        t_epoch = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss_avg, 4),
            "train_accuracy": round(train_acc, 2),
            "val_loss": round(val_loss_avg, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_manipulation_recall": val_metrics["manipulation_recall"],
            "val_real_recall": val_metrics["real_recall"],
            "val_fpr": val_metrics["fpr"],
            "learning_rate": round(current_lr, 8),
            "epoch_duration_sec": round(t_epoch, 2)
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{t_epoch:.1f}s] | "
            f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss_avg:.4f}, Macro F1: {val_metrics['macro_f1']:.2f}%, "
            f"M-Recall: {val_metrics['manipulation_recall']:.2f}%, "
            f"R-Recall: {val_metrics['real_recall']:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint based on Validation Macro F1
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_macro_f1,
                "val_metrics": val_metrics,
                "architecture": "ManipulationFrequencyResNet50V2",
                "training_seed": SEED,
            }, V2_CKPT_PATH)
            print(f"  >>> Best model saved to {V2_CKPT_PATH} (Macro F1 = {best_macro_f1:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    total_training_time = time.time() - t_start
    print(f"\nTraining completed in {total_training_time:.2f}s. Best epoch: {best_epoch} (Macro F1: {best_macro_f1:.2f}%)")

    # 5. Save History & Hashes
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_epochs": len(history),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "total_training_time_sec": round(total_training_time, 2),
            "history": history
        }, f, indent=2)

    v2_sha256 = compute_file_sha256(V2_CKPT_PATH)
    with open(RESULTS_DIR / "model_hashes.json", "w", encoding="utf-8") as f:
        json.dump({
            "manipulation_frequency_resnet50_v2.pth": {
                "sha256": v2_sha256,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1,
                "target_device": str(device)
            }
        }, f, indent=2)
    print(f"V2 Model Checkpoint SHA-256: {v2_sha256}")

    return best_epoch, best_macro_f1


if __name__ == "__main__":
    train_model()

