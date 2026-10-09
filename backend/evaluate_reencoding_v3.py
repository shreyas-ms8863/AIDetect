import os
import io
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

BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 224

# JPEG quality levels
JPEG_QUALITIES = [95, 75, 50]

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
print("AIDetect - Spatial V3 JPEG Re-encoding Robustness Evaluation")
print("=" * 75)

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU             : {torch.cuda.get_device_name(0)}"
    )

print("=" * 75)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{os.path.abspath(MODEL_PATH)}"
    )

print("\nBuilding ResNet-50 model...")

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 2)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V3 checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint


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
# LOAD CIFAKE
# ============================================================

print("\nLoading CIFAKE official test dataset...")

try:

    dataset = load_dataset(
        "dragonintelligence/CIFAKE-image-dataset",
        split="test"
    )

except Exception:

    DATASET_PATH = (
        r"C:\Users\Shreyas\.cache\huggingface\datasets"
        r"\dragonintelligence___cifake-image-dataset"
    )

    dataset = load_dataset(
        DATASET_PATH,
        split="test"
    )


print(
    f"[OK] Test images loaded: {len(dataset):,}"
)


# ============================================================
# BASE TRANSFORM
# ============================================================

base_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ============================================================
# JPEG RE-ENCODING DATASET
# ============================================================

class JPEGDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        quality
    ):

        self.dataset = hf_dataset
        self.quality = quality

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        # CIFAKE column is "image"
        image = item["image"]

        label = int(item["label"])

        if not isinstance(image, Image.Image):

            image = Image.fromarray(
                np.array(image)
            )

        image = image.convert("RGB")

        # Resize before JPEG re-encoding
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # JPEG RE-ENCODING
        # ----------------------------------------------------

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=self.quality
        )

        buffer.seek(0)

        # Decode JPEG again
        image = Image.open(buffer).convert("RGB")

        # Convert to tensor
        image = transforms.ToTensor()(image)

        # ImageNet normalization
        image = normalize(image)

        return image, label, index


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_quality(quality):

    print("\n" + "=" * 75)
    print(f"JPEG QUALITY {quality}")
    print("=" * 75)

    test_dataset = JPEGDataset(
        dataset,
        quality
    )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_indices = []

    start_time = time.time()

    with torch.no_grad():

        for batch_idx, (
            images,
            labels,
            indices
        ) in enumerate(loader):

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

                processed = min(
                    (batch_idx + 1) * BATCH_SIZE,
                    len(test_dataset)
                )

                print(
                    f"Processed "
                    f"{processed:,}/{len(test_dataset):,}"
                )


    elapsed_time = time.time() - start_time

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    indices = np.array(all_indices)


    # ========================================================
    # BASIC METRICS
    # ========================================================

    total = len(labels)

    correct = int(
        np.sum(labels == predictions)
    )

    incorrect = total - correct

    accuracy = correct / total


    # ========================================================
    # CONFUSION MATRIX
    #
    # 0 = FAKE
    # 1 = REAL
    # ========================================================

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


    # ========================================================
    # FAKE METRICS
    # ========================================================

    fake_precision = (
        true_fake /
        (true_fake + real_as_fake)
        if (true_fake + real_as_fake) > 0
        else 0
    )

    fake_recall = (
        true_fake /
        (true_fake + fake_as_real)
        if (true_fake + fake_as_real) > 0
        else 0
    )

    fake_f1 = (
        2 * fake_precision * fake_recall /
        (fake_precision + fake_recall)
        if (fake_precision + fake_recall) > 0
        else 0
    )


    # ========================================================
    # REAL METRICS
    # ========================================================

    real_precision = (
        true_real /
        (true_real + fake_as_real)
        if (true_real + fake_as_real) > 0
        else 0
    )

    real_recall = (
        true_real /
        (true_real + real_as_fake)
        if (true_real + real_as_fake) > 0
        else 0
    )

    real_f1 = (
        2 * real_precision * real_recall /
        (real_precision + real_recall)
        if (real_precision + real_recall) > 0
        else 0
    )


    # ========================================================
    # MACRO METRICS
    # ========================================================

    precision = (
        fake_precision + real_precision
    ) / 2

    recall = (
        fake_recall + real_recall
    ) / 2

    f1 = (
        fake_f1 + real_f1
    ) / 2


    # ========================================================
    # RESULTS
    # ========================================================

    print("\nFINAL RESULTS")

    print(
        f"Total images       : {total:,}"
    )

    print(
        f"Correct            : {correct:,}"
    )

    print(
        f"Incorrect          : {incorrect:,}"
    )

    print(
        f"\nAccuracy           : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision          : {precision * 100:.2f}%"
    )

    print(
        f"Recall             : {recall * 100:.2f}%"
    )

    print(
        f"F1-Score           : {f1 * 100:.2f}%"
    )

    print("\n--- FAKE CLASS ---")

    print(
        f"Precision          : {fake_precision * 100:.2f}%"
    )

    print(
        f"Recall             : {fake_recall * 100:.2f}%"
    )

    print(
        f"F1-Score           : {fake_f1 * 100:.2f}%"
    )

    print("\n--- REAL CLASS ---")

    print(
        f"Precision          : {real_precision * 100:.2f}%"
    )

    print(
        f"Recall             : {real_recall * 100:.2f}%"
    )

    print(
        f"F1-Score           : {real_f1 * 100:.2f}%"
    )

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

    print(
        f"\nEvaluation time     : "
        f"{elapsed_time / 60:.2f} minutes"
    )

    print("=" * 75)


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "experiment":
            "Spatial V3 JPEG Re-encoding Robustness",

        "model":
            "ResNet-50 V3",

        "dataset":
            "CIFAKE",

        "split":
            "official_test",

        "jpeg_quality":
            quality,

        "total_images":
            total,

        "correct":
            correct,

        "incorrect":
            incorrect,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "fake": {

            "precision":
                fake_precision,

            "recall":
                fake_recall,

            "f1_score":
                fake_f1
        },

        "real": {

            "precision":
                real_precision,

            "recall":
                real_recall,

            "f1_score":
                real_f1
        },

        "confusion_matrix": {

            "actual_fake_pred_fake":
                true_fake,

            "actual_fake_pred_real":
                fake_as_real,

            "actual_real_pred_fake":
                real_as_fake,

            "actual_real_pred_real":
                true_real
        },

        "evaluation_time_seconds":
            elapsed_time
    }


    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"spatial_v3_jpeg_q{quality}_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({

        "index":
            indices,

        "actual_label":
            labels,

        "predicted_label":
            predictions,

        "fake_probability":
            probabilities[:, 0],

        "real_probability":
            probabilities[:, 1],

        "correct":
            labels == predictions
    })


    predictions_path = os.path.join(
        OUTPUT_DIR,
        f"spatial_v3_jpeg_q{quality}_predictions.csv"
    )

    prediction_df.to_csv(
        predictions_path,
        index=False
    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    confusion_df = pd.DataFrame(

        [
            [true_fake, fake_as_real],
            [real_as_fake, true_real]
        ],

        index=[
            "Actual_FAKE",
            "Actual_REAL"
        ],

        columns=[
            "Predicted_FAKE",
            "Predicted_REAL"
        ]
    )


    confusion_path = os.path.join(
        OUTPUT_DIR,
        f"spatial_v3_jpeg_q{quality}_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )


    return {

        "quality":
            quality,

        "total":
            total,

        "correct":
            correct,

        "incorrect":
            incorrect,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "fake_precision":
            fake_precision,

        "fake_recall":
            fake_recall,

        "real_precision":
            real_precision,

        "real_recall":
            real_recall,

        "true_fake":
            true_fake,

        "fake_as_real":
            fake_as_real,

        "real_as_fake":
            real_as_fake,

        "true_real":
            true_real,

        "time":
            elapsed_time
    }


# ============================================================
# RUN ALL JPEG QUALITIES
# ============================================================

all_results = []

for quality in JPEG_QUALITIES:

    result = evaluate_quality(
        quality
    )

    all_results.append(
        result
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    all_results
)

summary_df = summary_df[
    [
        "quality",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fake_recall",
        "real_recall",
        "true_fake",
        "fake_as_real",
        "real_as_fake",
        "true_real"
    ]
]


# Convert metric columns to percentage
for column in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fake_recall",
    "real_recall"
]:

    summary_df[column] = (
        summary_df[column] * 100
    )


summary_path = os.path.join(
    OUTPUT_DIR,
    "spatial_v3_jpeg_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("JPEG RE-ENCODING SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.2f}%".format,
            "precision": "{:.2f}%".format,
            "recall": "{:.2f}%".format,
            "f1": "{:.2f}%".format,
            "fake_recall": "{:.2f}%".format,
            "real_recall": "{:.2f}%".format
        }
    )
)

print("\nSummary saved:")
print(
    f"[OK] {os.path.abspath(summary_path)}"
)

print("\nJPEG re-encoding evaluation completed.")
print("=" * 75)