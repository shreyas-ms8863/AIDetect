import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"models\spatial_resnet50_v3.pth"

OUTPUT_DIR = r"evaluation_results"

METRICS_PATH = os.path.join(
    OUTPUT_DIR, "spatial_v3_noise_metrics.json"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR, "spatial_v3_noise_predictions.csv"
)

CONFUSION_PATH = os.path.join(
    OUTPUT_DIR, "spatial_v3_noise_confusion_matrix.csv"
)

BATCH_SIZE = 32
NUM_WORKERS = 0

# Gaussian noise parameters
NOISE_MEAN = 0.0
NOISE_STD = 0.05

IMAGE_SIZE = 224

# Reproducibility
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("AIDetect - Spatial V3 Gaussian Noise Robustness Evaluation")
print("=" * 75)

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )

print("=" * 75)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

print("\nChecking model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{os.path.abspath(MODEL_PATH)}"
    )

print(f"[OK] {os.path.abspath(MODEL_PATH)}")


# ============================================================
# RESNET-50 ARCHITECTURE
# ============================================================

print("\nBuilding ResNet-50 model...")

model = models.resnet50(weights=None)

# Same classifier architecture used during V3 training
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 2)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading V3 checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle possible checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Remove possible "module." prefix
clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[7:]
    clean_state_dict[key] = value

model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("[OK] V3 model loaded successfully")


# ============================================================
# DATASET
# ============================================================

print("\nLoading CIFAKE official test dataset...")

DATASET_PATH = r"C:\Users\Shreyas\.cache\huggingface\datasets\dragonintelligence___cifake-image-dataset"

# Try loading from the local Hugging Face cache
try:
    dataset = load_dataset(
        "dragonintelligence/CIFAKE-image-dataset",
        split="test"
    )
except Exception:
    print(
        "\nCould not load CIFAKE directly from Hugging Face."
    )
    print(
        "Trying local dataset cache..."
    )

    dataset = load_dataset(
        DATASET_PATH,
        split="test"
    )


print(f"[OK] Test images loaded: {len(dataset):,}")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

# IMPORTANT:
# CIFAKE images are first resized to 224x224.
# Gaussian noise is added in [0,1] image space.
# ImageNet normalization is applied AFTER noise.

base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ImageNet normalization used during V3 training
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ============================================================
# DATASET WRAPPER
# ============================================================

class NoiseDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        noise_mean=0.0,
        noise_std=0.05
    ):
        self.dataset = hf_dataset
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]
        label = int(item["label"])

        # Convert to PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(
                np.array(image)
            )

        image = image.convert("RGB")

        # Resize + convert to tensor [0,1]
        image = base_transform(image)

        # ----------------------------------------------------
        # ADD GAUSSIAN PIXEL NOISE
        # ----------------------------------------------------

        noise = torch.randn_like(image) * self.noise_std
        noise = noise + self.noise_mean

        noisy_image = image + noise

        # Keep pixels inside valid [0,1] range
        noisy_image = torch.clamp(
            noisy_image,
            0.0,
            1.0
        )

        # ImageNet normalization
        noisy_image = normalize(noisy_image)

        return noisy_image, label, index


# ============================================================
# CREATE DATASET / DATALOADER
# ============================================================

noise_dataset = NoiseDataset(
    dataset,
    noise_mean=NOISE_MEAN,
    noise_std=NOISE_STD
)

loader = DataLoader(
    noise_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("NOISE ROBUSTNESS TEST")
print("=" * 75)

print(f"Dataset             : CIFAKE official test")
print(f"Total images        : {len(noise_dataset):,}")
print(f"Noise type          : Gaussian")
print(f"Noise mean          : {NOISE_MEAN}")
print(f"Noise standard dev. : {NOISE_STD}")
print(f"Image size          : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size          : {BATCH_SIZE}")
print("=" * 75)

all_labels = []
all_predictions = []
all_probabilities = []
all_indices = []

start_time = time.time()

with torch.no_grad():

    for batch_idx, (images, labels, indices) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        all_indices.extend(
            indices.numpy().tolist()
        )

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Processed "
                f"{min((batch_idx + 1) * BATCH_SIZE, len(noise_dataset)):,}"
                f"/{len(noise_dataset):,}"
            )


elapsed_time = time.time() - start_time


# ============================================================
# NUMPY ARRAYS
# ============================================================

labels = np.array(all_labels)
predictions = np.array(all_predictions)
probabilities = np.array(all_probabilities)
indices = np.array(all_indices)


# ============================================================
# BASIC METRICS
# ============================================================

total = len(labels)

correct = int(
    np.sum(labels == predictions)
)

incorrect = total - correct

accuracy = correct / total


# ============================================================
# CONFUSION MATRIX
#
# Label:
# 0 = FAKE
# 1 = REAL
#
# Rows    = Actual
# Columns = Predicted
# ============================================================

actual_fake = labels == 0
actual_real = labels == 1

pred_fake = predictions == 0
pred_real = predictions == 1

true_fake = int(
    np.sum(actual_fake & pred_fake)
)

fake_as_real = int(
    np.sum(actual_fake & pred_real)
)

real_as_fake = int(
    np.sum(actual_real & pred_fake)
)

true_real = int(
    np.sum(actual_real & pred_real)
)


# ============================================================
# CLASS METRICS
# ============================================================

# FAKE class
fake_precision = (
    true_fake / (true_fake + real_as_fake)
    if (true_fake + real_as_fake) > 0
    else 0
)

fake_recall = (
    true_fake / (true_fake + fake_as_real)
    if (true_fake + fake_as_real) > 0
    else 0
)

fake_f1 = (
    2 * fake_precision * fake_recall /
    (fake_precision + fake_recall)
    if (fake_precision + fake_recall) > 0
    else 0
)


# REAL class
real_precision = (
    true_real / (true_real + fake_as_real)
    if (true_real + fake_as_real) > 0
    else 0
)

real_recall = (
    true_real / (true_real + real_as_fake)
    if (true_real + real_as_fake) > 0
    else 0
)

real_f1 = (
    2 * real_precision * real_recall /
    (real_precision + real_recall)
    if (real_precision + real_recall) > 0
    else 0
)


# ============================================================
# MACRO METRICS
# ============================================================

precision = (
    (fake_precision + real_precision) / 2
)

recall = (
    (fake_recall + real_recall) / 2
)

f1 = (
    (fake_f1 + real_f1) / 2
)


# ============================================================
# ERROR RATES
#
# Here we explicitly define:
# FPR = Actual FAKE predicted REAL
# FNR = Actual REAL predicted FAKE
#
# This follows the project's current convention.
# ============================================================

fpr = (
    fake_as_real / (true_fake + fake_as_real)
    if (true_fake + fake_as_real) > 0
    else 0
)

fnr = (
    real_as_fake / (true_real + real_as_fake)
    if (true_real + real_as_fake) > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL NOISE ROBUSTNESS RESULTS")
print("=" * 75)

print(f"Total images       : {total:,}")
print(f"Correct            : {correct:,}")
print(f"Incorrect          : {incorrect:,}")

print(f"\nAccuracy           : {accuracy * 100:.2f}%")
print(f"Precision          : {precision * 100:.2f}%")
print(f"Recall             : {recall * 100:.2f}%")
print(f"F1-Score           : {f1 * 100:.2f}%")

print("\n--- FAKE CLASS ---")
print(f"Precision          : {fake_precision * 100:.2f}%")
print(f"Recall             : {fake_recall * 100:.2f}%")
print(f"F1-Score           : {fake_f1 * 100:.2f}%")

print("\n--- REAL CLASS ---")
print(f"Precision          : {real_precision * 100:.2f}%")
print(f"Recall             : {real_recall * 100:.2f}%")
print(f"F1-Score           : {real_f1 * 100:.2f}%")

print("\n--- CONFUSION MATRIX ---")
print(
    f"Actual FAKE -> Pred FAKE : {true_fake:,}"
)
print(
    f"Actual FAKE -> Pred REAL : {fake_as_real:,}"
)
print(
    f"Actual REAL -> Pred FAKE : {real_as_fake:,}"
)
print(
    f"Actual REAL -> Pred REAL : {true_real:,}"
)

print("\n--- ERROR RATES ---")
print(
    f"FAKE -> REAL error rate  : {fpr * 100:.2f}%"
)
print(
    f"REAL -> FAKE error rate  : {fnr * 100:.2f}%"
)

print(
    f"\nEvaluation time     : {elapsed_time / 60:.2f} minutes"
)

print("=" * 75)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {
    "experiment": "Spatial V3 Gaussian Noise Robustness",
    "model": "ResNet-50 V3",
    "dataset": "CIFAKE",
    "split": "official_test",
    "total_images": total,
    "correct": correct,
    "incorrect": incorrect,

    "noise": {
        "type": "Gaussian",
        "mean": NOISE_MEAN,
        "std": NOISE_STD,
        "clip_range": [0.0, 1.0]
    },

    "image_size": IMAGE_SIZE,

    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,

    "fake": {
        "precision": fake_precision,
        "recall": fake_recall,
        "f1_score": fake_f1
    },

    "real": {
        "precision": real_precision,
        "recall": real_recall,
        "f1_score": real_f1
    },

    "confusion_matrix": {
        "actual_fake_pred_fake": true_fake,
        "actual_fake_pred_real": fake_as_real,
        "actual_real_pred_fake": real_as_fake,
        "actual_real_pred_real": true_real
    },

    "error_rates": {
        "fake_to_real": fpr,
        "real_to_fake": fnr
    },

    "evaluation_time_seconds": elapsed_time
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_data = {
    "index": indices,
    "actual_label": labels,
    "predicted_label": predictions,
    "fake_probability": probabilities[:, 0],
    "real_probability": probabilities[:, 1],
    "correct": labels == predictions
}

prediction_df = pd.DataFrame(
    prediction_data
)

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    [
        [true_fake, fake_as_real],
        [real_as_fake, true_real]
    ],
    index=["Actual_FAKE", "Actual_REAL"],
    columns=["Predicted_FAKE", "Predicted_REAL"]
)

confusion_df.to_csv(
    CONFUSION_PATH
)


# ============================================================
# FINISHED
# ============================================================

print("\nFiles saved:")

print(
    f"[OK] {os.path.abspath(METRICS_PATH)}"
)

print(
    f"[OK] {os.path.abspath(PREDICTIONS_PATH)}"
)

print(
    f"[OK] {os.path.abspath(CONFUSION_PATH)}"
)

print("\nNoise robustness evaluation completed successfully.")
print("=" * 75)