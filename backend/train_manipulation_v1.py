"""
AIDetect Phase 6: Training Pipeline for AI-Manipulation Detector
==================================================================
Trains the dedicated binary classifier:
  0 = ORIGINAL_REAL
  1 = AI_MANIPULATED

Configuration:
  - Model: ManipulationResNet50V1 (Pretrained ResNet50 -> Dropout(0.3) -> Linear(2048, 2))
  - Batch size: 16
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-3)
  - Scheduler: CosineAnnealingLR(T_max=15, eta_min=1e-6)
  - Max epochs: 15
  - Early stopping: patience=5 on validation F1
  - Class weighting: inverse-frequency computed strictly from training split
  - AMP: Enabled on CUDA
  - Seed: 42
  - Checkpoint: models/manipulation_resnet50_v1.pth
  - Training logs: manipulation_v1/results/training_history.json
"""

import os
import sys
import json
import time
import random
import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manipulation_v1_models import ManipulationResNet50V1, ARCHITECTURE_DOC
from manipulation_v1_dataset import (
    ManipulationV1Dataset, collate_manipulation,
    PREPROCESSING_DOC, METADATA_CSV,
)

# ---------------------------------------------------------------------------
# PATHS & OUTPUT DIRS
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_DIR     = BASE_DIR.parent
MANIP_DIR       = PROJECT_DIR / "manipulation_v1"
RESULTS_DIR     = MANIP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR       = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "manipulation_resnet50_v1.pth"

# ---------------------------------------------------------------------------
# REPRODUCIBILITY (SEED = 42)
# ---------------------------------------------------------------------------

SEED = 42

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(SEED)

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------

BATCH_SIZE   = 16
MAX_EPOCHS   = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-3
T_MAX        = 15
ETA_MIN      = 1e-6
ES_PATIENCE  = 5
NUM_WORKERS  = 0

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print("=" * 70)
print("AIDETECT PHASE 6: AI-MANIPULATION DETECTOR TRAINING")
print("=" * 70)
print(f"Device           : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Model        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"PyTorch Version  : {torch.__version__}")
print(f"Seed             : {SEED}")
print(f"Mixed Precision  : {USE_AMP}")
print(f"Batch Size       : {BATCH_SIZE}")
print(f"Max Epochs       : {MAX_EPOCHS}")
print(f"Checkpoint Path  : {CHECKPOINT_PATH}")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# DATA LOADERS & CLASS WEIGHTS
# ---------------------------------------------------------------------------

train_ds = ManipulationV1Dataset("train", augment=True)
val_ds   = ManipulationV1Dataset("validation", augment=False)
test_ds  = ManipulationV1Dataset("test", augment=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_manipulation, drop_last=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    collate_fn=collate_manipulation, drop_last=False
)

# Compute class weights strictly from the training split
train_labels = [s["label"] for s in train_ds.samples]
n_train_total = len(train_labels)
n_train_real  = sum(1 for l in train_labels if l == 0)
n_train_manip = sum(1 for l in train_labels if l == 1)

w_real  = n_train_total / (2.0 * n_train_real)
w_manip = n_train_total / (2.0 * n_train_manip)
class_weights = torch.tensor([w_real, w_manip], dtype=torch.float32).to(DEVICE)

print("-" * 60)
print("TRAINING DATASET & CLASS WEIGHTS (STRICTLY FROM TRAIN SPLIT)")
print("-" * 60)
print(f"Train Images Total   : {n_train_total}")
print(f"  - ORIGINAL_REAL (0): {n_train_real}")
print(f"  - AI_MANIPULATED (1): {n_train_manip}")
print(f"Calculated Weights   : REAL={w_real:.4f}, MANIPULATED={w_manip:.4f}")
print(f"Validation Images    : {len(val_ds)}")
print(f"Test Images (Held-out): {len(test_ds)}")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# MODEL, OPTIMIZER, SCHEDULER, LOSS
# ---------------------------------------------------------------------------

model = ManipulationResNet50V1(dropout_p=0.3).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ---------------------------------------------------------------------------
# METRICS HELPER
# ---------------------------------------------------------------------------

def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, Any]:
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels, preds, average="macro", zero_division=0)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        real_rec  = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
        manip_rec = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)
    else:
        tn = fp = fn = tp = 0
        real_rec = manip_rec = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "real_recall": float(real_rec),
        "manip_recall": float(manip_rec),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "confusion_matrix": cm.tolist()
    }

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

training_history = []
best_val_f1 = -1.0
best_epoch = -1
epochs_without_improvement = 0

print("=" * 70)
print("STARTING TRAINING (MAX 15 EPOCHS, EARLY STOPPING PATIENCE=5)")
print("=" * 70)
sys.stdout.flush()

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    
    # 1. Train Step
    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []
    
    for x, y, _ in train_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * y.size(0)
        preds = logits.argmax(1).cpu().tolist()
        train_preds.extend(preds)
        train_targets.extend(y.cpu().tolist())
        
    scheduler.step()
    
    train_loss /= len(train_targets)
    train_acc = accuracy_score(train_targets, train_preds)
    train_f1  = f1_score(train_targets, train_preds, average="macro", zero_division=0)
    
    # 2. Validation Step
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
                
            val_loss += loss.item() * y.size(0)
            preds = logits.argmax(1).cpu().tolist()
            val_preds.extend(preds)
            val_targets.extend(y.cpu().tolist())
            
    val_loss /= len(val_targets)
    val_metrics = compute_metrics(val_targets, val_preds)
    val_acc   = val_metrics["accuracy"]
    val_prec  = val_metrics["precision"]
    val_rec   = val_metrics["recall"]
    val_f1    = val_metrics["f1"]
    real_rec  = val_metrics["real_recall"]
    manip_rec = val_metrics["manip_recall"]
    
    elapsed = time.time() - t0
    current_lr = scheduler.get_last_lr()[0]
    
    epoch_entry = {
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "train_accuracy": round(train_acc, 4),
        "train_f1": round(train_f1, 4),
        "val_loss": round(val_loss, 4),
        "val_accuracy": round(val_acc, 4),
        "val_precision": round(val_prec, 4),
        "val_recall": round(val_rec, 4),
        "val_f1": round(val_f1, 4),
        "real_recall": round(real_rec, 4),
        "manipulated_recall": round(manip_rec, 4),
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "learning_rate": current_lr,
        "elapsed_seconds": round(elapsed, 1)
    }
    training_history.append(epoch_entry)
    
    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
        f"Val: loss={val_loss:.4f} acc={val_acc*100:.2f}% f1={val_f1:.4f} "
        f"[REAL_rec={real_rec:.3f}, MANIP_rec={manip_rec:.3f}] | "
        f"lr={current_lr:.2e} | {elapsed:.1f}s"
    )
    sys.stdout.flush()
    
    # 3. Model Selection & Early Stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": "ManipulationResNet50V1",
            "version": "manipulation_v1",
            "class_mapping": {"0": "ORIGINAL_REAL", "1": "AI_MANIPULATED"},
            "input_size": 224,
            "seed": SEED,
            "preprocessing": PREPROCESSING_DOC,
            "architecture": ARCHITECTURE_DOC,
            "training_config": {
                "optimizer": "AdamW",
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "scheduler": f"CosineAnnealingLR(T_max={T_MAX}, eta_min={ETA_MIN})",
                "class_weights": {"ORIGINAL_REAL": w_real, "AI_MANIPULATED": w_manip},
                "early_stopping_patience": ES_PATIENCE,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
            },
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "validation_metrics": val_metrics,
            "training_history": training_history,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"  [SAVED] New best checkpoint saved (val_f1={best_val_f1:.4f}) -> {CHECKPOINT_PATH.name}")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement ({epochs_without_improvement}/{ES_PATIENCE}) [best val_f1={best_val_f1:.4f} at epoch {best_epoch}]")
        if epochs_without_improvement >= ES_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break
            
    sys.stdout.flush()

print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)
print(f"Best Epoch   : {best_epoch}")
print(f"Best Val F1  : {best_val_f1:.4f}")
print(f"Checkpoint   : {CHECKPOINT_PATH}")

# Save training history
history_path = RESULTS_DIR / "training_history.json"
with open(history_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp_completed": datetime.datetime.now().isoformat(),
        "model_name": "ManipulationResNet50V1",
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "class_weights": {"ORIGINAL_REAL": w_real, "AI_MANIPULATED": w_manip},
        "history": training_history,
    }, f, indent=2)

print(f"History saved: {history_path}")
print()
sys.stdout.flush()

