import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

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
# CONFIGURATION
# ============================================================

DATASET_REPO = "TheKernel01/Tiny-GenImage"

DATA_FILE = (
    "data/validation-00000-of-00004.parquet"
)

SPATIAL_MODEL = (
    "models/spatial_resnet50.pth"
)

FREQUENCY_MODEL = (
    "models/frequency_resnet50.pth"
)

HYBRID_MODEL = (
    "models/hybrid_resnet50_fft.pth"
)

RESULT_DIR = "genimage_model_results"

SAMPLES_PER_CLASS = 50

BATCH_SIZE = 8

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("AIDetect - Cross-Dataset Model Comparison")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD TINY-GENIMAGE
# ============================================================

print("\nLoading Tiny-GenImage validation shard...")

dataset = load_dataset(
    "parquet",
    data_files=(
        f"hf://datasets/"
        f"{DATASET_REPO}/"
        f"{DATA_FILE}"
    ),
    split="train"
)

print(
    f"Images available in shard: "
    f"{len(dataset):,}"
)


# ============================================================
# CHECK LABELS
#
# Tiny-GenImage:
# 0 = REAL
# 1 = AI-generated
# ============================================================

print("\nCollecting balanced samples...")

real_indices = []
ai_indices = []

for i in range(len(dataset)):

    label = dataset[i]["label"]

    if label == 0:

        real_indices.append(i)

    elif label == 1:

        ai_indices.append(i)

    if (
        len(real_indices) >= SAMPLES_PER_CLASS
        and
        len(ai_indices) >= SAMPLES_PER_CLASS
    ):
        break


selected_indices = (
    real_indices[:SAMPLES_PER_CLASS]
    +
    ai_indices[:SAMPLES_PER_CLASS]
)

selected_dataset = dataset.select(
    selected_indices
)


print(
    f"Real images selected: "
    f"{SAMPLES_PER_CLASS}"
)

print(
    f"AI images selected: "
    f"{SAMPLES_PER_CLASS}"
)

print(
    f"Total selected: "
    f"{len(selected_dataset)}"
)


# ============================================================
# TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT

spatial_transform = weights.transforms()


# ============================================================
# FFT REPRESENTATION
# ============================================================

def fft_magnitude(image_tensor):

    x = image_tensor.float()

    fft = torch.fft.fft2(x)

    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)

    magnitude = torch.log1p(
        magnitude
    )

    for c in range(
        magnitude.shape[0]
    ):

        channel = magnitude[c]

        min_value = channel.min()

        max_value = channel.max()

        magnitude[c] = (
            (channel - min_value)
            /
            (
                max_value
                -
                min_value
                +
                1e-8
            )
        )

    return magnitude


# ============================================================
# DATASET WRAPPER
# ============================================================

class GenImageDataset(Dataset):

    def __init__(self, hf_dataset):

        self.dataset = hf_dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.fromarray(
                image
            )

        image = image.convert("RGB")


        # ----------------------------------------------------
        # Spatial representation
        # ----------------------------------------------------

        spatial_image = spatial_transform(
            image
        )


        # ----------------------------------------------------
        # Frequency representation
        # ----------------------------------------------------

        frequency_image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        frequency_image = torch.from_numpy(
            np.array(
                frequency_image
            )
        ).permute(
            2,
            0,
            1
        ).float() / 255.0

        frequency_image = fft_magnitude(
            frequency_image
        )


        # ----------------------------------------------------
        # Tiny-GenImage labels
        #
        # 0 = REAL
        # 1 = AI
        # ----------------------------------------------------

        label = int(
            item["label"]
        )


        # ----------------------------------------------------
        # Generator metadata
        # ----------------------------------------------------

        generator = item.get(
            "generator",
            "unknown"
        )

        return (
            spatial_image,
            frequency_image,
            label,
            generator,
            index
        )


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridDetector(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        spatial_model = resnet50(
            weights=None
        )

        self.spatial = nn.Sequential(
            *list(
                spatial_model.children()
            )[:-1]
        )


        # ----------------------------------------------------
        # Frequency branch
        # ----------------------------------------------------

        self.frequency = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ----------------------------------------------------
        # Fusion classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                2048 + 256,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                2
            )
        )


    def forward(
        self,
        spatial_image,
        frequency_image
    ):

        spatial_features = self.spatial(
            spatial_image
        )

        spatial_features = (
            spatial_features.flatten(1)
        )


        frequency_features = self.frequency(
            frequency_image
        )

        frequency_features = (
            frequency_features.flatten(1)
        )


        fused = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )


        return self.classifier(
            fused
        )


# ============================================================
# LOAD SPATIAL MODEL
# ============================================================

print("\nLoading Spatial-only model...")

spatial_model = resnet50(
    weights=None
)

spatial_model.fc = nn.Linear(
    spatial_model.fc.in_features,
    2
)

spatial_model.load_state_dict(
    torch.load(
        SPATIAL_MODEL,
        map_location=DEVICE
    )
)

spatial_model = spatial_model.to(
    DEVICE
)

spatial_model.eval()


# ============================================================
# LOAD FREQUENCY MODEL
# ============================================================

print("Loading Frequency-only model...")

frequency_model = resnet50(
    weights=None
)

frequency_model.fc = nn.Linear(
    frequency_model.fc.in_features,
    2
)

frequency_model.load_state_dict(
    torch.load(
        FREQUENCY_MODEL,
        map_location=DEVICE
    )
)

frequency_model = frequency_model.to(
    DEVICE
)

frequency_model.eval()


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

print("Loading Hybrid model...")

hybrid_model = HybridDetector()

hybrid_model.load_state_dict(
    torch.load(
        HYBRID_MODEL,
        map_location=DEVICE
    )
)

hybrid_model = hybrid_model.to(
    DEVICE
)

hybrid_model.eval()


# ============================================================
# DATALOADER
# ============================================================

test_data = GenImageDataset(
    selected_dataset
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# PREDICTIONS
# ============================================================

spatial_predictions = []

frequency_predictions = []

hybrid_predictions = []

true_labels = []

metadata = []


print("\nRunning all three models...\n")


with torch.no_grad():

    for (
        spatial_images,
        frequency_images,
        labels,
        generators,
        indices
    ) in tqdm(
        test_loader,
        desc="Tiny-GenImage"
    ):

        spatial_images = spatial_images.to(
            DEVICE
        )

        frequency_images = frequency_images.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Spatial
        # ----------------------------------------------------

        spatial_outputs = spatial_model(
            spatial_images
        )

        spatial_preds = torch.argmax(
            spatial_outputs,
            dim=1
        ).cpu().tolist()


        # ----------------------------------------------------
        # Frequency
        # ----------------------------------------------------

        frequency_outputs = frequency_model(
            frequency_images
        )

        frequency_preds = torch.argmax(
            frequency_outputs,
            dim=1
        ).cpu().tolist()


        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        hybrid_outputs = hybrid_model(
            spatial_images,
            frequency_images
        )

        hybrid_preds = torch.argmax(
            hybrid_outputs,
            dim=1
        ).cpu().tolist()


        spatial_predictions.extend(
            spatial_preds
        )

        frequency_predictions.extend(
            frequency_preds
        )

        hybrid_predictions.extend(
            hybrid_preds
        )

        true_labels.extend(
            labels.tolist()
        )


        for generator, index in zip(
            generators,
            indices.tolist()
        ):

            metadata.append({
                "dataset_index": int(index),
                "generator": str(generator)
            })


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    name,
    labels,
    predictions
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(
        f"Accuracy   : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-Score   : "
        f"{f1 * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# CALCULATE RESULTS
# ============================================================

print("\n" + "=" * 70)

print("TINY-GENIMAGE RESULTS")

print("=" * 70)

print(
    f"Dataset: Tiny-GenImage"
)

print(
    f"Images: {len(true_labels)}"
)

print(
    "Classes: 50 Real + 50 AI"
)


spatial_results = calculate_metrics(
    "SPATIAL-ONLY",
    true_labels,
    spatial_predictions
)


frequency_results = calculate_metrics(
    "FREQUENCY-ONLY",
    true_labels,
    frequency_predictions
)


hybrid_results = calculate_metrics(
    "HYBRID",
    true_labels,
    hybrid_predictions
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


results = {

    "dataset":
        "Tiny-GenImage",

    "source":
        DATASET_REPO,

    "validation_shard":
        DATA_FILE,

    "total_images":
        len(true_labels),

    "real_images":
        SAMPLES_PER_CLASS,

    "ai_images":
        SAMPLES_PER_CLASS,

    "models": {

        "spatial_only":
            spatial_results,

        "frequency_only":
            frequency_results,

        "hybrid":
            hybrid_results
    },

    "generator_metadata":
        metadata
}


with open(
    os.path.join(
        RESULT_DIR,
        "genimage_model_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

import csv

prediction_file = os.path.join(
    RESULT_DIR,
    "genimage_model_predictions.csv"
)


with open(
    prediction_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "dataset_index",
        "generator",
        "true_label",
        "spatial_prediction",
        "frequency_prediction",
        "hybrid_prediction"
    ])


    for i in range(
        len(true_labels)
    ):

        writer.writerow([
            metadata[i]["dataset_index"],
            metadata[i]["generator"],
            true_labels[i],
            spatial_predictions[i],
            frequency_predictions[i],
            hybrid_predictions[i]
        ])


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("CROSS-DATASET EVALUATION COMPLETE")

print("=" * 70)

print(
    "\nMetrics saved to:"
)

print(
    "genimage_model_results/"
    "genimage_model_metrics.json"
)

print(
    "\nPredictions saved to:"
)

print(
    "genimage_model_results/"
    "genimage_model_predictions.csv"
)