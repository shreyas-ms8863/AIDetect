# ============================================================
# AIDetect V5 - Spatial-only ResNet-50 Training
# WITH LIVE TRAINING / VALIDATION PROGRESS
# ============================================================

import os
import json
import time
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from v5_dataset import V5Dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2

NUM_EPOCHS = 15
PATIENCE = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

NUM_CLASSES = 2
NUM_WORKERS = 0

INPUT_SIZE = 224

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "v5"
RESULTS_DIR = PROJECT_ROOT / "evaluation_results" / "v5"

CHECKPOINT_PATH = (
    MODEL_DIR / "spatial_resnet50_v5_best.pth"
)

METRICS_PATH = (
    RESULTS_DIR / "spatial_v5_training_metrics.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 75)
print("AIDetect V5 - Spatial-only ResNet-50 Training")
print("=" * 75)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(
    f"PyTorch: {torch.__version__}"
)

print()


# ============================================================
# DATASET
# ============================================================

print("=" * 75)
print("Loading V5 datasets")
print("=" * 75)

train_dataset = V5Dataset(
    split="train",
    mode="spatial",
)

val_dataset = V5Dataset(
    split="validation",
    mode="spatial",
)


# ============================================================
# DATASET SANITY CHECK
# ============================================================

assert len(train_dataset) == 42000, (
    f"Expected 42000 training samples, "
    f"got {len(train_dataset)}"
)

assert len(val_dataset) == 9000, (
    f"Expected 9000 validation samples, "
    f"got {len(val_dataset)}"
)

print(
    f"Training samples:   {len(train_dataset):,}"
)

print(
    f"Validation samples: {len(val_dataset):,}"
)

print(
    "Final test samples: NOT LOADED"
)

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# BATCH SANITY CHECK
# ============================================================

sample_batch = next(iter(train_loader))

print("=" * 75)
print("Batch sanity check")
print("=" * 75)

print(
    "Spatial batch shape:",
    tuple(sample_batch["spatial"].shape)
)

print(
    "Label batch shape:",
    tuple(sample_batch["label"].shape)
)

print()


# ============================================================
# MODEL
# ============================================================

print("=" * 75)
print("Building ResNet-50")
print("=" * 75)

weights = ResNet50_Weights.IMAGENET1K_V2

model = resnet50(
    weights=weights
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(device)

print("Model: ResNet-50")
print("Pretrained weights: ImageNet1K V2")
print("Output classes: 2")
print("REAL = 0")
print("AI   = 1")
print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# SCHEDULER
# ============================================================

optimizer_steps_per_epoch = math.ceil(
    len(train_loader) / GRAD_ACCUM_STEPS
)

total_optimizer_steps = (
    optimizer_steps_per_epoch * NUM_EPOCHS
)

warmup_optimizer_steps = (
    optimizer_steps_per_epoch
)


def lr_lambda(step):

    # Warmup
    if step < warmup_optimizer_steps:

        return float(step + 1) / float(
            max(
                1,
                warmup_optimizer_steps
            )
        )

    # Cosine decay
    progress = (
        step - warmup_optimizer_steps
    ) / float(
        max(
            1,
            total_optimizer_steps
            - warmup_optimizer_steps
        )
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    return 0.5 * (
        1.0 + math.cos(
            math.pi * progress
        )
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda,
)


# ============================================================
# AMP
# ============================================================

amp_enabled = (
    device.type == "cuda"
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=amp_enabled,
)


# ============================================================
# METRICS
# ============================================================

training_history = []

best_val_f1 = -1.0
best_epoch = 0

epochs_without_improvement = 0

global_optimizer_step = 0


# ============================================================
# PROGRESS BAR
# ============================================================

def print_progress(
    current,
    total,
    images_done,
    images_total,
    start_time,
    phase="Training",
):
    """
    Print live progress without requiring tqdm.
    """

    elapsed = time.time() - start_time

    if elapsed <= 0:
        elapsed = 0.001

    # Percentage
    percent = (
        current / total
    ) * 100

    # Speed
    batches_per_sec = (
        current / elapsed
    )

    images_per_sec = (
        images_done / elapsed
    )

    # ETA
    remaining_batches = (
        total - current
    )

    if batches_per_sec > 0:

        eta_seconds = (
            remaining_batches
            / batches_per_sec
        )

    else:

        eta_seconds = 0

    eta_seconds = max(
        0,
        eta_seconds
    )

    eta_minutes = int(
        eta_seconds // 60
    )

    eta_secs = int(
        eta_seconds % 60
    )

    # Progress bar
    bar_length = 30

    filled = int(
        bar_length
        * current
        / total
    )

    filled = min(
        filled,
        bar_length
    )

    bar = (
        "█" * filled
        +
        "░" * (
            bar_length - filled
        )
    )

    # GPU memory
    gpu_info = ""

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        gpu_info = (
            f" | GPU "
            f"{allocated:.2f}/"
            f"{reserved:.2f} GB"
        )

    message = (
        f"\r{phase}: "
        f"{bar} "
        f"{percent:6.2f}% | "
        f"{current:,}/{total:,} batches | "
        f"{images_done:,}/{images_total:,} images | "
        f"{images_per_sec:.1f} img/s | "
        f"ETA {eta_minutes:02d}:{eta_secs:02d}"
        f"{gpu_info}"
    )

    sys.stdout.write(
        message
    )

    sys.stdout.flush()


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),
    }


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    all_labels = []
    all_predictions = []

    running_loss = 0.0
    sample_count = 0

    total_batches = len(loader)
    total_images = len(loader.dataset)

    start_time = time.time()

    print()

    for batch_idx, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["spatial"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        sample_count += batch_size

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        # Live validation progress
        print_progress(
            current=batch_idx,
            total=total_batches,
            images_done=sample_count,
            images_total=total_images,
            start_time=start_time,
            phase="Validation"
        )

    print()

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = (
        running_loss
        / sample_count
    )

    return metrics


# ============================================================
# START TRAINING
# ============================================================

print("=" * 75)
print("Starting training")
print("=" * 75)

print(
    f"Max epochs:        {NUM_EPOCHS}"
)

print(
    f"Batch size:        {BATCH_SIZE}"
)

print(
    f"Grad accumulation: {GRAD_ACCUM_STEPS}"
)

print(
    f"Effective batch:   "
    f"{BATCH_SIZE * GRAD_ACCUM_STEPS}"
)

print(
    f"Learning rate:     {LEARNING_RATE}"
)

print(
    f"Weight decay:      {WEIGHT_DECAY}"
)

print(
    f"AMP enabled:       {amp_enabled}"
)

print(
    f"Early stopping:    "
    f"{PATIENCE} epochs"
)

print()

print(
    f"Training batches/epoch: "
    f"{len(train_loader):,}"
)

print(
    f"Validation batches: "
    f"{len(val_loader):,}"
)

print()


# ============================================================
# EPOCH LOOP
# ============================================================

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    epoch_start = time.time()

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0

    correct = 0
    total = 0

    optimizer_steps_this_epoch = 0

    train_start = time.time()

    print()
    print("=" * 75)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for batch_idx, batch in enumerate(
        train_loader,
        start=1
    ):

        images = batch["spatial"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        # Forward
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUM_STEPS
            )

        # Backward
        scaler.scale(
            loss_for_backward
        ).backward()

        # Gradient accumulation
        should_step = (

            batch_idx
            % GRAD_ACCUM_STEPS
            == 0

            or

            batch_idx
            == len(train_loader)
        )

        if should_step:

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            scheduler.step()

            global_optimizer_step += 1

            optimizer_steps_this_epoch += 1

        # Metrics
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total += batch_size

        correct += (
            predictions == labels
        ).sum().item()

        # ----------------------------------------------------
        # LIVE PROGRESS
        # ----------------------------------------------------

        print_progress(
            current=batch_idx,
            total=len(train_loader),
            images_done=total,
            images_total=len(train_dataset),
            start_time=train_start,
            phase="Training "
        )

    print()

    # --------------------------------------------------------
    # TRAIN METRICS
    # --------------------------------------------------------

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_metrics = evaluate(
        model,
        val_loader
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    # --------------------------------------------------------
    # EPOCH RESULTS
    # --------------------------------------------------------

    epoch_result = {

        "epoch":
            epoch,

        "train": {

            "loss":
                float(train_loss),

            "accuracy":
                float(train_accuracy),
        },

        "validation":
            val_metrics,

        "learning_rate":
            float(current_lr),

        "optimizer_steps":
            optimizer_steps_this_epoch,

        "epoch_time_seconds":
            float(epoch_time),
    }

    training_history.append(
        epoch_result
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("-" * 75)

    print(
        f"Epoch {epoch} completed"
    )

    print(
        f"Train Loss:      "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:        "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"Val Accuracy:    "
        f"{val_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val Precision:   "
        f"{val_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Val Recall:      "
        f"{val_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"Val F1:          "
        f"{val_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Learning Rate:   "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time:      "
        f"{epoch_time / 60:.2f} minutes"
    )

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            f"GPU Memory:      "
            f"{allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved"
        )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    current_val_f1 = (
        val_metrics["f1"]
    )

    if current_val_f1 > best_val_f1:

        best_val_f1 = (
            current_val_f1
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint = {

            "model_name":
                "resnet50_spatial_v5",

            "architecture":
                "ResNet-50",

            "pretrained_weights":
                "IMAGENET1K_V2",

            "state_dict":
                model.state_dict(),

            "class_mapping": {

                "REAL": 0,
                "AI": 1,
            },

            "input_size":
                [3, INPUT_SIZE, INPUT_SIZE],

            "best_epoch":
                best_epoch,

            "best_val_metrics":
                val_metrics,

            "training_config": {

                "seed":
                    SEED,

                "batch_size":
                    BATCH_SIZE,

                "gradient_accumulation":
                    GRAD_ACCUM_STEPS,

                "effective_batch_size":
                    BATCH_SIZE
                    * GRAD_ACCUM_STEPS,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "max_epochs":
                    NUM_EPOCHS,

                "early_stopping_patience":
                    PATIENCE,

                "optimizer":
                    "AdamW",

                "scheduler":
                    "1 epoch warmup + cosine decay",

                "amp":
                    amp_enabled,
            },

            "dataset": {

                "train_samples":
                    len(train_dataset),

                "validation_samples":
                    len(val_dataset),

                "final_test_evaluated":
                    False,
            },
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

        print()
        print(
            "✓ NEW BEST MODEL SAVED"
        )

        print(
            CHECKPOINT_PATH
        )

    else:

        epochs_without_improvement += 1

        print()

        print(
            f"No validation F1 improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_output = {

        "experiment":
            "V5-A Spatial-only ResNet-50",

        "class_mapping": {

            "REAL": 0,
            "AI": 1,
        },

        "dataset": {

            "train_samples":
                len(train_dataset),

            "validation_samples":
                len(val_dataset),

            "final_test_samples":
                9000,

            "final_test_evaluated":
                False,
        },

        "best_epoch":
            best_epoch,

        "best_validation_f1":
            float(best_val_f1),

        "history":
            training_history,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            "=" * 75
        )

        print(
            "EARLY STOPPING TRIGGERED"
        )

        print(
            "=" * 75
        )

        break


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation F1: "
    f"{best_val_f1 * 100:.2f}%"
)

print()

print(
    "Best checkpoint:"
)

print(
    CHECKPOINT_PATH
)

print()

print(
    "Training metrics:"
)

print(
    METRICS_PATH
)

print()

print(
    "Final 9,000-image test set:"
)

print(
    "NOT LOADED / NOT EVALUATED"
)

print()

print("=" * 75)