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

MODEL_PATH = "models/hybrid_resnet50_fft.pth"

RESULT_DIR = "hybrid_results"

BATCH_SIZE = 16

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FFT REPRESENTATION
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
# SPATIAL PREPROCESSING
# Same ImageNet normalization used during training
# ============================================================

from torchvision.models import ResNet50_Weights

weights = ResNet50_Weights.DEFAULT

spatial_transform = weights.transforms()


# ============================================================
# DATASET
# ============================================================

class HybridTestDataset(torch.utils.data.Dataset):

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


        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        spatial_image = spatial_transform(
            image
        )


        # ----------------------------------------------------
        # Frequency branch
        # ----------------------------------------------------

        frequency_image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        frequency_image = torch.from_numpy(
            np.array(frequency_image)
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
        # ----------------------------------------------------

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
            spatial_image,
            frequency_image,
            label
        )


# ============================================================
# HYBRID MODEL
# Must exactly match training architecture
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

        self.spatial_features = 2048


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

        self.frequency_features = 256


        # ----------------------------------------------------
        # Fusion classifier
        # ----------------------------------------------------

        fusion_size = (
            self.spatial_features
            +
            self.frequency_features
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_size,
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

        spatial_features = spatial_features.flatten(
            1
        )


        frequency_features = self.frequency(
            frequency_image
        )

        frequency_features = frequency_features.flatten(
            1
        )


        fused_features = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )


        output = self.classifier(
            fused_features
        )

        return output


# ============================================================
# START
# ============================================================

print("=" * 60)

print(
    "AIDetect - Hybrid Spatial + Frequency Evaluation"
)

print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD CIFAKE TEST
# ============================================================

print("\nLoading CIFAKE TEST dataset...")

dataset = load_dataset(
    DATASET_REPO,
    split="test"
)

print(
    f"Test images: {len(dataset):,}"
)


test_data = HybridTestDataset(
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

print("\nLoading Hybrid model...")

model = HybridDetector()

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

    for (
        spatial_images,
        frequency_images,
        labels
    ) in tqdm(
        test_loader,
        desc="CIFAKE TEST"
    ):

        spatial_images = spatial_images.to(
            DEVICE
        )

        frequency_images = frequency_images.to(
            DEVICE
        )


        outputs = model(
            spatial_images,
            frequency_images
        )


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

print("HYBRID RESULTS")

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
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


results = {

    "model":
        "Hybrid Spatial + Frequency",

    "spatial_branch":
        "Pretrained ResNet50",

    "frequency_branch":
        "FFT magnitude + CNN",

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
        "hybrid_metrics.json"
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
    "hybrid_results/hybrid_metrics.json"
)

print("\nEvaluation complete.")