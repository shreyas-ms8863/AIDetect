import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/hybrid_resnet50_fft_v2.pth"
OUTPUT_DIR = "hybrid_v2_results"

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

print("Loading CIFAKE test dataset...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="test"
)

print(f"Test images: {len(dataset)}")


# ============================================================
# TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT
image_transform = weights.transforms()


def frequency_transform(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_tensor = torch.from_numpy(
        np.array(image)
    ).float()

    image_tensor = (
        image_tensor.permute(2, 0, 1) / 255.0
    )

    # FFT
    fft = torch.fft.fft2(image_tensor)
    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)
    magnitude = torch.log1p(magnitude)

    # Channel-wise normalization
    for c in range(3):

        channel = magnitude[c]

        min_val = channel.min()
        max_val = channel.max()

        magnitude[c] = (
            channel - min_val
        ) / (
            max_val - min_val + 1e-8
        )

    magnitude_image = Image.fromarray(
        (
            magnitude.permute(1, 2, 0).numpy() * 255
        )
        .clip(0, 255)
        .astype("uint8")
    )

    return image_transform(magnitude_image)


# ============================================================
# DATASET CLASS
# ============================================================

class HybridTestDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        image = item["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        # Spatial input
        spatial_image = image_transform(image)

        # Frequency input
        frequency_image = frequency_transform(image)

        # CIFAKE:
        # 0 = AI
        # 1 = Real
        #
        # Internal:
        # 0 = Real
        # 1 = AI

        original_label = int(item["label"])

        if original_label == 0:
            label = 1
        else:
            label = 0

        return spatial_image, frequency_image, label


test_dataset = HybridTestDataset(dataset)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# HYBRID MODEL
# Must exactly match train_hybrid_v2.py
# ============================================================

class HybridResNetFFT(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        spatial_model = resnet50(
            weights=None
        )

        self.spatial_features = nn.Sequential(
            *list(spatial_model.children())[:-1]
        )

        self.spatial_dim = 2048

        # ----------------------------------------------------
        # Frequency branch
        # ----------------------------------------------------

        self.frequency_features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.frequency_dim = 256

        # ----------------------------------------------------
        # Fusion classifier
        # ----------------------------------------------------

        combined_dim = (
            self.spatial_dim +
            self.frequency_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_dim,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                2
            )
        )

    def forward(
        self,
        spatial_input,
        frequency_input
    ):

        spatial = self.spatial_features(
            spatial_input
        )

        spatial = torch.flatten(
            spatial,
            1
        )

        frequency = self.frequency_features(
            frequency_input
        )

        frequency = torch.flatten(
            frequency,
            1
        )

        combined = torch.cat(
            [
                spatial,
                frequency
            ],
            dim=1
        )

        return self.classifier(combined)


# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nUsing device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("\nLoading Hybrid V2 model...")

model = HybridResNetFFT()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []

print("\nEvaluating Hybrid V2...\n")

with torch.no_grad():

    for (
        spatial_images,
        frequency_images,
        labels
    ) in tqdm(
        test_loader,
        desc="Testing"
    ):

        spatial_images = spatial_images.to(
            DEVICE,
            non_blocking=True
        )

        frequency_images = frequency_images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            spatial_images,
            frequency_images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
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
print("HYBRID V2 CIFAKE TEST RESULTS")
print("=" * 60)

print("Dataset: CIFAKE")
print("Split: Test")
print(f"Total images: {len(all_labels)}")

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

metrics = {
    "model": "Hybrid V2 - ResNet50 + FFT",
    "dataset": "CIFAKE",
    "split": "test",
    "total_images": len(all_labels),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_v2_metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nMetrics saved to:")
print(metrics_path)