import os
import json
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from v5_dataset import V5Dataset


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 16
GRAD_ACCUMULATION = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUMULATION

MAX_EPOCHS = 15
PATIENCE = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

NUM_CLASSES = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "v5"
RESULT_DIR = PROJECT_ROOT / "evaluation_results" / "v5"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "hybrid_resnet50_v5_best.pth"
METRICS_PATH = RESULT_DIR / "hybrid_v5_training_metrics.json"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 75)
print("AIDetect V5-C - Spatial + Frequency Hybrid ResNet-50 Training")
print("=" * 75)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"PyTorch: {torch.__version__}")


# ============================================================
# Dataset
# ============================================================

print("\n" + "=" * 75)
print("Loading V5 datasets")
print("=" * 75)

train_dataset = V5Dataset(
    split="train",
    mode="hybrid"
)

val_dataset = V5Dataset(
    split="validation",
    mode="hybrid"
)

assert len(train_dataset) == 42000, (
    f"Expected 42,000 training samples, got {len(train_dataset)}"
)

assert len(val_dataset) == 9000, (
    f"Expected 9,000 validation samples, got {len(val_dataset)}"
)

print(f"Training samples:   {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print("Final test samples: NOT LOADED")


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# Batch sanity check
# ============================================================

print("\n" + "=" * 75)
print("Batch sanity check")
print("=" * 75)

sample_batch = next(iter(train_loader))

print(
    f"Spatial batch shape:   "
    f"{tuple(sample_batch['spatial'].shape)}"
)

print(
    f"Frequency batch shape: "
    f"{tuple(sample_batch['frequency'].shape)}"
)

print(
    f"Label batch shape:     "
    f"{tuple(sample_batch['label'].shape)}"
)

assert sample_batch["spatial"].shape == (BATCH_SIZE, 3, 224, 224)
assert sample_batch["frequency"].shape == (BATCH_SIZE, 3, 224, 224)


# ============================================================
# Hybrid Model
# ============================================================

class HybridResNet50(nn.Module):

    def __init__(self):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        self.spatial_encoder = resnet50(weights=weights)

        spatial_features = self.spatial_encoder.fc.in_features

        self.spatial_encoder.fc = nn.Identity()

        # ----------------------------------------------------
        # Frequency branch
        # ----------------------------------------------------

        self.frequency_encoder = resnet50(weights=weights)

        frequency_features = self.frequency_encoder.fc.in_features

        self.frequency_encoder.fc = nn.Identity()

        # ----------------------------------------------------
        # Fusion classifier
        # ----------------------------------------------------

        fused_features = spatial_features + frequency_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(fused_features, NUM_CLASSES)
        )

    def forward(self, spatial, frequency):

        spatial_features = self.spatial_encoder(spatial)

        frequency_features = self.frequency_encoder(frequency)

        fused = torch.cat(
            [spatial_features, frequency_features],
            dim=1
        )

        output = self.classifier(fused)

        return output


print("\n" + "=" * 75)
print("Building Hybrid ResNet-50")
print("=" * 75)

model = HybridResNet50().to(device)

print("Model: Spatial + Frequency Hybrid ResNet-50")
print("Spatial branch: ResNet-50 ImageNet1K V2")
print("Frequency branch: ResNet-50 ImageNet1K V2")
print("Fusion: Feature concatenation")
print("Fusion dimension: 4096")
print("Output classes: 2")
print("REAL = 0")
print("AI   = 1")


# ============================================================
# Loss / Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Scheduler
# ============================================================

total_optimizer_steps = math.ceil(
    len(train_loader) / GRAD_ACCUMULATION
) * MAX_EPOCHS

warmup_steps = math.ceil(
    len(train_loader) / GRAD_ACCUMULATION
)

cosine_steps = max(
    1,
    total_optimizer_steps - warmup_steps
)


def get_lr(step):

    if step < warmup_steps:

        return LEARNING_RATE * (
            float(step + 1) / float(warmup_steps)
        )

    progress = (
        float(step - warmup_steps)
        / float(cosine_steps)
    )

    progress = min(max(progress, 0.0), 1.0)

    return LEARNING_RATE * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(predictions, labels):

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    tp = ((predictions == 1) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    accuracy = (
        (tp + tn) /
        max(1, tp + tn + fp + fn)
    )

    precision = (
        tp / max(1, tp + fp)
    )

    recall = (
        tp / max(1, tp + fn)
    )

    f1 = (
        2 * precision * recall /
        max(1e-12, precision + recall)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 75)
print("Starting Hybrid training")
print("=" * 75)

print(f"Max epochs:        {MAX_EPOCHS}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Grad accumulation: {GRAD_ACCUMULATION}")
print(f"Effective batch:   {EFFECTIVE_BATCH_SIZE}")
print(f"Learning rate:     {LEARNING_RATE}")
print(f"Weight decay:      {WEIGHT_DECAY}")
print(f"AMP enabled:       {torch.cuda.is_available()}")
print(f"Early stopping:    {PATIENCE} epochs")

print(f"\nTraining batches/epoch: {len(train_loader):,}")
print(f"Validation batches: {len(val_loader):,}")


history = []

best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

global_optimizer_step = 0


for epoch in range(1, MAX_EPOCHS + 1):

    print("\n" + "=" * 75)
    print(f"Epoch {epoch}/{MAX_EPOCHS}")
    print("=" * 75)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    train_predictions = []
    train_labels = []

    optimizer.zero_grad(set_to_none=True)

    epoch_start = time.time()

    total_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader, start=1):

        spatial = batch["spatial"].to(
            device,
            non_blocking=True
        )

        frequency = batch["frequency"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            outputs = model(
                spatial,
                frequency
            )

            loss = criterion(
                outputs,
                labels
            )

            loss_for_backward = (
                loss / GRAD_ACCUMULATION
            )

        scaler.scale(
            loss_for_backward
        ).backward()

        if (
            batch_idx % GRAD_ACCUMULATION == 0
            or batch_idx == total_batches
        ):

            # ------------------------------------------------
            # Learning-rate schedule
            # ------------------------------------------------

            current_lr = get_lr(
                global_optimizer_step
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            global_optimizer_step += 1

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.append(
            predictions.detach().cpu()
        )

        train_labels.append(
            labels.detach().cpu()
        )

        # ----------------------------------------------------
        # Live progress
        # ----------------------------------------------------

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == total_batches
        ):

            elapsed = time.time() - epoch_start

            images_done = batch_idx * BATCH_SIZE

            images_done = min(
                images_done,
                len(train_dataset)
            )

            images_per_sec = (
                images_done /
                max(elapsed, 1e-6)
            )

            remaining_images = (
                len(train_dataset)
                - images_done
            )

            eta_seconds = (
                remaining_images /
                max(images_per_sec, 1e-6)
            )

            progress = (
                batch_idx /
                total_batches
            )

            bar_length = 30

            filled = int(
                bar_length * progress
            )

            bar = (
                "█" * filled +
                "░" * (bar_length - filled)
            )

            if torch.cuda.is_available():

                gpu_allocated = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                gpu_reserved = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )

                gpu_text = (
                    f"GPU "
                    f"{gpu_allocated:.2f}/"
                    f"{gpu_reserved:.2f} GB"
                )

            else:

                gpu_text = "GPU N/A"

            print(
                f"\rTraining : {bar} "
                f"{progress * 100:6.2f}% | "
                f"{batch_idx:,}/{total_batches:,} batches | "
                f"{images_done:,}/{len(train_dataset):,} images | "
                f"{images_per_sec:.1f} img/s | "
                f"ETA {int(eta_seconds // 60):02d}:"
                f"{int(eta_seconds % 60):02d} | "
                f"{gpu_text}",
                end="",
                flush=True
            )

    print()

    train_loss = (
        running_loss /
        total_batches
    )

    train_metrics = calculate_metrics(
        train_predictions,
        train_labels
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_predictions = []
    val_labels = []

    validation_start = time.time()

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            val_loader,
            start=1
        ):

            spatial = batch["spatial"].to(
                device,
                non_blocking=True
            )

            frequency = batch["frequency"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                outputs = model(
                    spatial,
                    frequency
                )

                loss = criterion(
                    outputs,
                    labels
                )

            val_running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_predictions.append(
                predictions.cpu()
            )

            val_labels.append(
                labels.cpu()
            )

            # ------------------------------------------------
            # Validation progress
            # ------------------------------------------------

            if (
                batch_idx == 1
                or batch_idx % 25 == 0
                or batch_idx == len(val_loader)
            ):

                elapsed = (
                    time.time()
                    - validation_start
                )

                images_done = (
                    batch_idx * BATCH_SIZE
                )

                images_done = min(
                    images_done,
                    len(val_dataset)
                )

                images_per_sec = (
                    images_done /
                    max(elapsed, 1e-6)
                )

                remaining_images = (
                    len(val_dataset)
                    - images_done
                )

                eta_seconds = (
                    remaining_images /
                    max(images_per_sec, 1e-6)
                )

                progress = (
                    batch_idx /
                    len(val_loader)
                )

                bar_length = 30

                filled = int(
                    bar_length * progress
                )

                bar = (
                    "█" * filled +
                    "░" * (bar_length - filled)
                )

                if torch.cuda.is_available():

                    gpu_allocated = (
                        torch.cuda.memory_allocated()
                        / 1024**3
                    )

                    gpu_reserved = (
                        torch.cuda.memory_reserved()
                        / 1024**3
                    )

                    gpu_text = (
                        f"GPU "
                        f"{gpu_allocated:.2f}/"
                        f"{gpu_reserved:.2f} GB"
                    )

                else:

                    gpu_text = "GPU N/A"

                print(
                    f"\rValidation: {bar} "
                    f"{progress * 100:6.2f}% | "
                    f"{batch_idx:,}/{len(val_loader):,} batches | "
                    f"{images_done:,}/{len(val_dataset):,} images | "
                    f"{images_per_sec:.1f} img/s | "
                    f"ETA {int(eta_seconds // 60):02d}:"
                    f"{int(eta_seconds % 60):02d} | "
                    f"{gpu_text}",
                    end="",
                    flush=True
                )

    print()

    val_loss = (
        val_running_loss /
        len(val_loader)
    )

    val_metrics = calculate_metrics(
        val_predictions,
        val_labels
    )

    epoch_time = (
        time.time() - epoch_start
    )

    # --------------------------------------------------------
    # Epoch results
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print(f"Epoch {epoch} completed")

    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  "
        f"{train_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val Loss:        {val_loss:.4f}"
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

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Learning Rate:   "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time:      "
        f"{epoch_time / 60:.2f} minutes"
    )

    if torch.cuda.is_available():

        print(
            f"GPU Memory:      "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB "
            f"allocated / "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB "
            f"reserved"
        )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "learning_rate": current_lr,
        "epoch_time_minutes": epoch_time / 60
    }

    history.append(epoch_record)

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_f1": best_f1,
            "class_mapping": {
                "REAL": 0,
                "AI": 1
            },
            "architecture": (
                "Hybrid ResNet-50 "
                "Spatial + Frequency "
                "Feature Concatenation"
            ),
            "final_test_evaluated": False
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print(
            "\n✓ NEW BEST HYBRID MODEL SAVED"
        )

        print(
            BEST_MODEL_PATH
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo validation F1 improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered after "
            f"{epoch} epochs."
        )

        break


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 75)
print("V5-C HYBRID TRAINING COMPLETE")
print("=" * 75)

print(f"Best epoch: {best_epoch}")

print(
    f"Best validation F1: "
    f"{best_f1 * 100:.2f}%"
)

print("\nBest checkpoint:")
print(BEST_MODEL_PATH)

# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics_output = {
    "experiment": "V5-C Hybrid Spatial + Frequency",
    "seed": SEED,
    "train_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "final_test_loaded": False,
    "final_test_evaluated": False,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRAD_ACCUMULATION,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "best_epoch": best_epoch,
    "best_validation_f1": best_f1,
    "history": history
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

print("\nTraining metrics:")
print(METRICS_PATH)

print("\nFinal 9,000-image test set:")
print("NOT LOADED / NOT EVALUATED")

print("=" * 75)