import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_REPO = "dragonintelligence/CIFAKE-image-dataset"

MODEL_PATH = "models/frequency_resnet50.pth"

RESULT_DIR = "frequency_results"

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FFT FUNCTION
# ============================================================

def fft_magnitude(image_tensor):

    x = image_tensor.float()

    fft = torch.fft.fft2(x)

    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)

    magnitude = torch.log1p(magnitude)

    for c in range(magnitude.shape[0]):

        channel = magnitude[c]

        min_value = channel.min()
        max_value = channel.max()

        magnitude[c] = (
            (channel - min_value)
            /
            (max_value - min_value + 1e-8)
        )

    return magnitude


# ============================================================
# DATASET
# ============================================================

class FrequencyTestDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset):

        self.dataset = hf_dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        if not isinstance(image, Image.Image):

            image = Image.fromarray(image)

        image = image.convert("RGB")

        image = image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        image_tensor = torch.from_numpy(
            np.array(image)
        ).permute(2, 0, 1).float() / 255.0

        image_tensor = fft_magnitude(
            image_tensor
        )

        # CIFAKE:
        # 0 = AI
        # 1 = REAL
        #
        # Internal:
        # 0 = REAL
        # 1 = AI

        if item["label"] == 0:

            label = 1

        else:

            label = 0

        return (
            image_tensor,
            label
        )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("AIDetect - Frequency-only FFT + ResNet50 Evaluation")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading CIFAKE TEST dataset...")

dataset = load_dataset(
    DATASET_REPO,
    split="test"
)

print(
    f"Test images: {len(dataset):,}"
)


test_data = FrequencyTestDataset(
    dataset
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading Frequency-only ResNet50..."
)

model = resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []

print("\nEvaluating...\n")

with torch.no_grad():

    for images, labels in tqdm(
        test_loader,
        desc="CIFAKE TEST"
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu().tolist()

        all_predictions.extend(
            predictions
        )

        all_labels.extend(
            labels
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FREQUENCY-ONLY RESULTS")
print("=" * 60)

print("Dataset    : CIFAKE")
print("Split      : Test")
print(
    f"Images     : {len(all_labels):,}"
)

print(
    f"\nAccuracy   : {accuracy * 100:.2f}%"
)

print(
    f"Precision  : {precision * 100:.2f}%"
)

print(
    f"Recall     : {recall * 100:.2f}%"
)

print(
    f"F1-Score   : {f1 * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

results = {

    "model":
        "Frequency-only FFT + ResNet50",

    "frequency_representation":
        "2D FFT magnitude with log scaling",

    "dataset":
        "CIFAKE",

    "split":
        "test",

    "total_images":
        len(all_labels),

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1_score":
        f1,

    "confusion_matrix":
        cm.tolist()
}


with open(
    os.path.join(
        RESULT_DIR,
        "frequency_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nResults saved to:"
)

print(
    "frequency_results/"
    "frequency_metrics.json"
)

print("\nEvaluation complete.")