import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
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

DATA_DIR = "backend/real_world_test"

SPATIAL_MODEL = "models/spatial_resnet50.pth"

FREQUENCY_MODEL = "models/frequency_resnet50.pth"

HYBRID_MODEL = "models/hybrid_resnet50_fft.pth"

RESULT_DIR = "real_world_model_results"

BATCH_SIZE = 4

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("AIDetect - Real-World Model Comparison")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# FIND IMAGES
# ============================================================

AI_DIR = os.path.join(
    DATA_DIR,
    "AI"
)

REAL_DIR = os.path.join(
    DATA_DIR,
    "REAL"
)


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def get_images(folder):

    files = []

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        if (
            os.path.isfile(path)
            and
            filename.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ):

            files.append(path)

    return files


ai_images = get_images(AI_DIR)

real_images = get_images(REAL_DIR)


print("\nReal-world dataset:")

print(
    f"AI images   : {len(ai_images)}"
)

print(
    f"REAL images : {len(real_images)}"
)

print(
    f"Total       : "
    f"{len(ai_images) + len(real_images)}"
)


# ============================================================
# PREPROCESSING
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
# DATASET
# ============================================================

class RealWorldDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        path = self.image_paths[index]

        image = Image.open(path).convert(
            "RGB"
        )

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
        # Label
        #
        # Internal labels:
        # 0 = REAL
        # 1 = AI
        # ----------------------------------------------------

        folder = os.path.basename(
            os.path.dirname(path)
        )

        if folder.upper() == "AI":

            label = 1

        else:

            label = 0


        return (
            spatial_image,
            frequency_image,
            label,
            os.path.basename(path)
        )


# ============================================================
# HYBRID MODEL
# Must match training architecture
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
# LOAD DATA
# ============================================================

image_paths = (
    real_images
    +
    ai_images
)

test_data = RealWorldDataset(
    image_paths
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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
# PREDICTION
# ============================================================

spatial_predictions = []

frequency_predictions = []

hybrid_predictions = []

true_labels = []

filenames = []


print("\nRunning all three models...\n")


with torch.no_grad():

    for (
        spatial_images,
        frequency_images,
        labels,
        names
    ) in tqdm(
        test_loader,
        desc="Real-world test"
    ):

        spatial_images = spatial_images.to(
            DEVICE
        )

        frequency_images = frequency_images.to(
            DEVICE
        )


        # Spatial
        spatial_outputs = spatial_model(
            spatial_images
        )

        spatial_preds = torch.argmax(
            spatial_outputs,
            dim=1
        ).cpu().tolist()


        # Frequency
        frequency_outputs = frequency_model(
            frequency_images
        )

        frequency_preds = torch.argmax(
            frequency_outputs,
            dim=1
        ).cpu().tolist()


        # Hybrid
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

        filenames.extend(
            list(names)
        )


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    model_name,
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

    print(model_name)

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
# RESULTS
# ============================================================

print("\n" + "=" * 70)

print("REAL-WORLD RESULTS")

print("=" * 70)


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
        "AIDetect real-world test",

    "total_images":
        len(true_labels),

    "ai_images":
        len(ai_images),

    "real_images":
        len(real_images),

    "models": {

        "spatial_only":
            spatial_results,

        "frequency_only":
            frequency_results,

        "hybrid":
            hybrid_results
    }
}


with open(
    os.path.join(
        RESULT_DIR,
        "real_world_model_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

import csv

prediction_file = os.path.join(
    RESULT_DIR,
    "real_world_model_predictions.csv"
)


with open(
    prediction_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "true_label",
        "spatial_prediction",
        "frequency_prediction",
        "hybrid_prediction"
    ])


    for i in range(
        len(true_labels)
    ):

        writer.writerow([
            filenames[i],
            true_labels[i],
            spatial_predictions[i],
            frequency_predictions[i],
            hybrid_predictions[i]
        ])


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("REAL-WORLD EVALUATION COMPLETE")

print("=" * 70)

print(
    "\nMetrics saved to:"
)

print(
    "real_world_model_results/"
    "real_world_model_metrics.json"
)

print(
    "\nPredictions saved to:"
)

print(
    "real_world_model_results/"
    "real_world_model_predictions.csv"
)