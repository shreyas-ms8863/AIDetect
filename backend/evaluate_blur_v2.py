import os
import json
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights

from datasets import load_dataset

from PIL import Image, ImageFilter

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

SPATIAL_MODEL_PATH = "models/spatial_resnet50_v2.pth"
FREQUENCY_MODEL_PATH = "models/frequency_resnet50_v2.pth"
HYBRID_MODEL_PATH = "models/hybrid_resnet50_fft_v2.pth"

OUTPUT_DIR = "robustness_results"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "blur_v2_metrics.json"
)

BATCH_SIZE = 16
NUM_WORKERS = 0

BLUR_RADIUS = 2.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("BLUR ROBUSTNESS TEST - V2 MODELS")
print("=" * 65)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Gaussian Blur radius: {BLUR_RADIUS}"
)


# ============================================================
# LOAD CIFAKE TEST DATASET
# ============================================================

print("\nLoading CIFAKE test dataset...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="test"
)

print(
    f"Test images: {len(dataset)}"
)


# ============================================================
# TRANSFORMS
# Match the existing V2 evaluation pipeline
# ============================================================

weights = ResNet50_Weights.DEFAULT

image_transform = weights.transforms()


# ============================================================
# FREQUENCY TRANSFORMATION
# Exact FFT preprocessing used by Hybrid V2
# ============================================================

def frequency_transform(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_tensor = torch.from_numpy(
        np.array(image)
    ).float()

    image_tensor = (
        image_tensor.permute(
            2, 0, 1
        ) / 255.0
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft = torch.fft.fft2(
        image_tensor
    )

    fft = torch.fft.fftshift(
        fft
    )

    magnitude = torch.abs(
        fft
    )

    magnitude = torch.log1p(
        magnitude
    )

    # --------------------------------------------------------
    # Channel-wise normalization
    # --------------------------------------------------------

    for c in range(3):

        channel = magnitude[c]

        min_val = channel.min()
        max_val = channel.max()

        magnitude[c] = (
            (channel - min_val)
            /
            (
                max_val - min_val + 1e-8
            )
        )

    # --------------------------------------------------------
    # Convert frequency representation
    # back to image
    # --------------------------------------------------------

    magnitude_image = Image.fromarray(
        (
            magnitude
            .permute(1, 2, 0)
            .numpy()
            * 255
        )
        .clip(
            0,
            255
        )
        .astype(
            "uint8"
        )
    )

    return image_transform(
        magnitude_image
    )


# ============================================================
# APPLY GAUSSIAN BLUR
# ============================================================

def apply_blur(image):

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=BLUR_RADIUS
        )
    )


# ============================================================
# DATASET CLASS
# ============================================================

class BlurTestDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        hf_dataset
    ):

        self.dataset = hf_dataset

    def __len__(self):

        return len(
            self.dataset
        )

    def __getitem__(
        self,
        idx
    ):

        item = self.dataset[idx]

        image = item["image"]

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.fromarray(
                image
            )

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # APPLY BLUR BEFORE MODEL PREPROCESSING
        # ----------------------------------------------------

        blurred_image = apply_blur(
            image
        )

        # ----------------------------------------------------
        # SPATIAL INPUT
        # ----------------------------------------------------

        spatial_image = image_transform(
            blurred_image
        )

        # ----------------------------------------------------
        # FREQUENCY INPUT
        # FFT is calculated from the blurred image
        # ----------------------------------------------------

        frequency_image = frequency_transform(
            blurred_image
        )

        # ----------------------------------------------------
        # LABEL CONVERSION
        #
        # CIFAKE:
        # 0 = AI / Fake
        # 1 = Real
        #
        # Internal project convention:
        # 0 = Real
        # 1 = AI
        # ----------------------------------------------------

        original_label = int(
            item["label"]
        )

        if original_label == 0:

            label = 1

        else:

            label = 0

        return (
            spatial_image,
            frequency_image,
            label
        )


# ============================================================
# CREATE DATALOADER
# ============================================================

test_dataset = BlurTestDataset(
    dataset
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# SPATIAL MODEL
# ============================================================

def create_resnet_classifier():

    model = resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


# ============================================================
# HYBRID V2 MODEL
#
# EXACT ARCHITECTURE FROM
# evaluate_hybrid_v2.py
# ============================================================

class HybridResNetFFT(
    nn.Module
):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        spatial_model = resnet50(
            weights=None
        )

        self.spatial_features = nn.Sequential(
            *list(
                spatial_model.children()
            )[:-1]
        )

        self.spatial_dim = 2048

        # ----------------------------------------------------
        # Frequency branch
        # ----------------------------------------------------

        self.frequency_features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.frequency_dim = 256

        # ----------------------------------------------------
        # Fusion classifier
        # ----------------------------------------------------

        combined_dim = (
            self.spatial_dim
            +
            self.frequency_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_dim,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.3
            ),

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

        # ----------------------------------------------------
        # Spatial features
        # ----------------------------------------------------

        spatial = self.spatial_features(
            spatial_input
        )

        spatial = torch.flatten(
            spatial,
            1
        )

        # ----------------------------------------------------
        # Frequency features
        # ----------------------------------------------------

        frequency = self.frequency_features(
            frequency_input
        )

        frequency = torch.flatten(
            frequency,
            1
        )

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        combined = torch.cat(
            [
                spatial,
                frequency
            ],
            dim=1
        )

        return self.classifier(
            combined
        )


# ============================================================
# LOAD SPATIAL V2
# ============================================================

print("\nLoading Spatial V2...")

spatial_model = create_resnet_classifier()

spatial_checkpoint = torch.load(
    SPATIAL_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(
        spatial_checkpoint,
        dict
    )
    and
    "model_state_dict"
    in spatial_checkpoint
):

    spatial_model.load_state_dict(
        spatial_checkpoint[
            "model_state_dict"
        ]
    )

else:

    spatial_model.load_state_dict(
        spatial_checkpoint
    )

spatial_model = spatial_model.to(
    DEVICE
)

spatial_model.eval()


# ============================================================
# LOAD FREQUENCY V2
# ============================================================

print("Loading Frequency V2...")

frequency_model = create_resnet_classifier()

frequency_checkpoint = torch.load(
    FREQUENCY_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(
        frequency_checkpoint,
        dict
    )
    and
    "model_state_dict"
    in frequency_checkpoint
):

    frequency_model.load_state_dict(
        frequency_checkpoint[
            "model_state_dict"
        ]
    )

else:

    frequency_model.load_state_dict(
        frequency_checkpoint
    )

frequency_model = frequency_model.to(
    DEVICE
)

frequency_model.eval()


# ============================================================
# LOAD HYBRID V2
# ============================================================

print("Loading Hybrid V2...")

hybrid_model = HybridResNetFFT()

hybrid_checkpoint = torch.load(
    HYBRID_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(
        hybrid_checkpoint,
        dict
    )
    and
    "model_state_dict"
    in hybrid_checkpoint
):

    hybrid_model.load_state_dict(
        hybrid_checkpoint[
            "model_state_dict"
        ]
    )

else:

    hybrid_model.load_state_dict(
        hybrid_checkpoint
    )

hybrid_model = hybrid_model.to(
    DEVICE
)

hybrid_model.eval()


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
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

    return {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1_score": float(
            f1
        ),

        "confusion_matrix":
            cm.tolist()
    }


# ============================================================
# EVALUATE SPATIAL
# ============================================================

def evaluate_spatial():

    predictions = []
    labels_all = []

    print("\nSpatial V2:")

    with torch.no_grad():

        for (
            spatial_images,
            frequency_images,
            labels
        ) in tqdm(
            test_loader
        ):

            spatial_images = spatial_images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = spatial_model(
                spatial_images
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    return calculate_metrics(
        labels_all,
        predictions
    )


# ============================================================
# EVALUATE FREQUENCY
# ============================================================

def evaluate_frequency():

    predictions = []
    labels_all = []

    print("\nFrequency V2:")

    with torch.no_grad():

        for (
            spatial_images,
            frequency_images,
            labels
        ) in tqdm(
            test_loader
        ):

            frequency_images = frequency_images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = frequency_model(
                frequency_images
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    return calculate_metrics(
        labels_all,
        predictions
    )


# ============================================================
# EVALUATE HYBRID
# ============================================================

def evaluate_hybrid():

    predictions = []
    labels_all = []

    print("\nHybrid V2:")

    with torch.no_grad():

        for (
            spatial_images,
            frequency_images,
            labels
        ) in tqdm(
            test_loader
        ):

            spatial_images = spatial_images.to(
                DEVICE,
                non_blocking=True
            )

            frequency_images = frequency_images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = hybrid_model(
                spatial_images,
                frequency_images
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    return calculate_metrics(
        labels_all,
        predictions
    )


# ============================================================
# RUN TEST
# ============================================================

print("\n" + "=" * 65)
print("RUNNING BLUR ROBUSTNESS TEST")
print("=" * 65)


results = {}

results["spatial_v2"] = evaluate_spatial()

results["frequency_v2"] = evaluate_frequency()

results["hybrid_v2"] = evaluate_hybrid()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("BLUR ROBUSTNESS RESULTS")
print("=" * 65)


for name, result in results.items():

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Accuracy : "
        f"{result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{result['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{result['recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score : "
        f"{result['f1_score'] * 100:.2f}%"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            result[
                "confusion_matrix"
            ]
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("BLUR ROBUSTNESS TEST COMPLETE")
print("=" * 65)

print("\nResults saved to:")
print(OUTPUT_PATH)