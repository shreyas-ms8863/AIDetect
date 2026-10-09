import os
import json
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"models\spatial_resnet50_v3.pth"

OUTPUT_METRICS = r"evaluation_results\spatial_v3_resize_metrics.json"
OUTPUT_PREDICTIONS = r"evaluation_results\spatial_v3_resize_predictions.csv"
OUTPUT_CONFUSION = r"evaluation_results\spatial_v3_resize_confusion_matrix.csv"

BATCH_SIZE = 32
NUM_WORKERS = 0

# Resize robustness condition
# Images are first resized to this smaller resolution,
# then resized back to 224x224 for the model.
RESIZE_SIZE = 112
MODEL_SIZE = 224

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 75)
print("AIDetect - Spatial V3 Resize Robustness Evaluation")
print("=" * 75)

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )

print("=" * 75)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nChecking model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{os.path.abspath(MODEL_PATH)}"
    )

print(f"[OK] {os.path.abspath(MODEL_PATH)}")


# ============================================================
# LOAD CIFAKE
# ============================================================

print("\nLoading CIFAKE dataset...")

dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")

test_dataset = dataset["test"]

print("\n" + "=" * 75)
print("TEST DATASET")
print("=" * 75)

total_images = len(test_dataset)

fake_count = sum(1 for x in test_dataset["label"] if x == 0)
real_count = sum(1 for x in test_dataset["label"] if x == 1)

print(f"Total test images : {total_images:,}")
print(f"FAKE images       : {fake_count:,}")
print(f"REAL images       : {real_count:,}")

print("=" * 75)


# ============================================================
# TRANSFORM
# ============================================================
#
# Resize robustness:
#
# Original image
#       ↓
# Resize to 112 × 112
#       ↓
# Resize back to 224 × 224
#       ↓
# ImageNet normalization
#       ↓
# V3 ResNet-50
#
# Only the test preprocessing is changed.
# The model itself is NOT retrained.
# ============================================================

resize_transform = transforms.Compose([
    transforms.Resize(
        (RESIZE_SIZE, RESIZE_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.Resize(
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET WRAPPER
# ============================================================

class CIFAKEResizeDataset(Dataset):

    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        label = int(item["label"])

        image = self.transform(image)

        return image, label, index


resize_dataset = CIFAKEResizeDataset(
    test_dataset,
    resize_transform
)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    resize_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nDataLoader:")
print(f"Batch size : {BATCH_SIZE}")
print(f"Batches    : {len(loader)}")
print(f"Workers    : {NUM_WORKERS}")


# ============================================================
# LOAD SPATIAL V3
# ============================================================

print("\n" + "=" * 75)
print("LOADING SPATIAL V3 MODEL")
print("=" * 75)

model = models.resnet50(
    weights=ResNet50_Weights.DEFAULT
)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 2)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict, strict=True)

model = model.to(device)
model.eval()

print("[OK] Spatial V3 loaded successfully")
print("[OK] Strict checkpoint matching")


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("STARTING V3 RESIZE ROBUSTNESS EVALUATION")
print("=" * 75)

print("Resize condition:")
print(f"Original → {RESIZE_SIZE}x{RESIZE_SIZE} → {MODEL_SIZE}x{MODEL_SIZE}")

print("\nIMPORTANT:")
print("The model was NOT retrained for this resize condition.")
print("The official 20,000-image CIFAKE test set is being used.")
print("=" * 75)


all_labels = []
all_predictions = []
all_probabilities = []
all_indices = []

correct = 0
processed = 0

start_time = time.time()


with torch.no_grad():

    for batch_number, (images, labels, indices) in enumerate(loader, start=1):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        processed += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy().tolist()
        )

        all_indices.extend(
            indices.numpy().tolist()
        )

        if (
            batch_number == 1
            or batch_number % 100 == 0
            or batch_number == len(loader)
        ):

            running_accuracy = (
                correct / processed
            ) * 100

            print(
                f"Batch {batch_number}/{len(loader)} | "
                f"Images {processed:,}/{total_images:,} | "
                f"Running Accuracy: {running_accuracy:.2f}%"
            )


# ============================================================
# METRICS
# ============================================================

labels = np.array(all_labels)
predictions = np.array(all_predictions)

# Class 1 = REAL
# Class 0 = FAKE

tp_real = np.sum(
    (labels == 1) & (predictions == 1)
)

fn_real = np.sum(
    (labels == 1) & (predictions == 0)
)

fp_real = np.sum(
    (labels == 0) & (predictions == 1)
)

tn_real = np.sum(
    (labels == 0) & (predictions == 0)
)


def safe_divide(a, b):
    return a / b if b != 0 else 0.0


accuracy = safe_divide(
    np.sum(labels == predictions),
    len(labels)
)

precision_real = safe_divide(
    tp_real,
    tp_real + fp_real
)

recall_real = safe_divide(
    tp_real,
    tp_real + fn_real
)

f1_real = safe_divide(
    2 * precision_real * recall_real,
    precision_real + recall_real
)


# FAKE metrics
precision_fake = safe_divide(
    tn_real,
    tn_real + fn_real
)

recall_fake = safe_divide(
    tn_real,
    tn_real + fp_real
)

f1_fake = safe_divide(
    2 * precision_fake * recall_fake,
    precision_fake + recall_fake
)


fpr = safe_divide(
    fp_real,
    fp_real + tn_real
)

fnr = safe_divide(
    fn_real,
    fn_real + tp_real
)


# ============================================================
# TIME
# ============================================================

evaluation_time = time.time() - start_time


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("SPATIAL V3 - RESIZE ROBUSTNESS RESULTS")
print("=" * 75)

print(f"Resize condition   : {RESIZE_SIZE}x{RESIZE_SIZE}")
print(f"Total images       : {total_images:,}")
print(f"Correct            : {int(np.sum(labels == predictions)):,}")
print(f"Incorrect          : {int(np.sum(labels != predictions)):,}")

print(f"Accuracy           : {accuracy * 100:.2f}%")
print(f"Precision          : {precision_real * 100:.2f}%")
print(f"Recall             : {recall_real * 100:.2f}%")
print(f"F1-Score           : {f1_real * 100:.2f}%")

print("\nClass-wise metrics")

print(
    f"FAKE Precision     : "
    f"{precision_fake * 100:.2f}%"
)

print(
    f"FAKE Recall        : "
    f"{recall_fake * 100:.2f}%"
)

print(
    f"FAKE F1            : "
    f"{f1_fake * 100:.2f}%"
)

print(
    f"REAL Precision     : "
    f"{precision_real * 100:.2f}%"
)

print(
    f"REAL Recall        : "
    f"{recall_real * 100:.2f}%"
)

print(
    f"REAL F1            : "
    f"{f1_real * 100:.2f}%"
)


print("\nConfusion Matrix")

print(
    "                     Predicted FAKE   Predicted REAL"
)

print(
    f"Actual FAKE          "
    f"{tn_real:>12,}"
    f"{fp_real:>20,}"
)

print(
    f"Actual REAL          "
    f"{fn_real:>12,}"
    f"{tp_real:>20,}"
)


print("\nError rates")

print(
    f"False Positive Rate  : "
    f"{fpr * 100:.2f}%"
)

print(
    f"False Negative Rate  : "
    f"{fnr * 100:.2f}%"
)

print(
    f"\nEvaluation time      : "
    f"{evaluation_time / 60:.2f} minutes"
)

print("=" * 75)


# ============================================================
# SAVE JSON
# ============================================================

metrics = {

    "experiment": "Spatial V3 Resize Robustness",

    "model": "spatial_resnet50_v3",

    "dataset": "CIFAKE",

    "split": "test",

    "total_images": int(total_images),

    "fake_images": int(fake_count),

    "real_images": int(real_count),

    "resize_condition": {
        "intermediate_size": RESIZE_SIZE,
        "final_model_size": MODEL_SIZE,
        "interpolation": "bilinear"
    },

    "correct": int(np.sum(labels == predictions)),

    "incorrect": int(np.sum(labels != predictions)),

    "accuracy": float(accuracy),

    "precision": float(precision_real),

    "recall": float(recall_real),

    "f1_score": float(f1_real),

    "fake": {
        "precision": float(precision_fake),
        "recall": float(recall_fake),
        "f1": float(f1_fake)
    },

    "real": {
        "precision": float(precision_real),
        "recall": float(recall_real),
        "f1": float(f1_real)
    },

    "confusion_matrix": {
        "actual_fake_predicted_fake": int(tn_real),
        "actual_fake_predicted_real": int(fp_real),
        "actual_real_predicted_fake": int(fn_real),
        "actual_real_predicted_real": int(tp_real)
    },

    "false_positive_rate": float(fpr),

    "false_negative_rate": float(fnr),

    "evaluation_time_seconds": float(evaluation_time),

    "test_set_used_for_training": False,

    "test_set_used_for_validation": False
}


os.makedirs(
    os.path.dirname(OUTPUT_METRICS),
    exist_ok=True
)

with open(
    OUTPUT_METRICS,
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

prediction_df = pd.DataFrame({

    "index": all_indices,

    "true_label": labels,

    "predicted_label": predictions,

    "probability_real": all_probabilities,

    "correct": (
        labels == predictions
    )
})


prediction_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(

    [
        [tn_real, fp_real],
        [fn_real, tp_real]
    ],

    index=[
        "Actual FAKE",
        "Actual REAL"
    ],

    columns=[
        "Predicted FAKE",
        "Predicted REAL"
    ]
)

confusion_df.to_csv(
    OUTPUT_CONFUSION
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("V3 RESIZE ROBUSTNESS EVALUATION COMPLETE")
print("=" * 75)

print("\nSaved files:")

print(
    f"[1] {os.path.abspath(OUTPUT_METRICS)}"
)

print(
    f"[2] {os.path.abspath(OUTPUT_PREDICTIONS)}"
)

print(
    f"[3] {os.path.abspath(OUTPUT_CONFUSION)}"
)

print("=" * 75)