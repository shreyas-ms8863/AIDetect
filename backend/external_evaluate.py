import os
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "external_test"
MODEL_DIR = BASE_DIR.parent / "models"
OUTPUT_DIR = BASE_DIR / "external_results"

OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("=" * 70)
print("AIDetect V2 - External Generalization Evaluation")
print("=" * 70)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LABELS
# REAL = 0
# AI   = 1
# ============================================================

CLASS_NAMES = {
    0: "REAL",
    1: "AI",
}

# ============================================================
# SPATIAL TRANSFORM
# ============================================================

spatial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# FREQUENCY TRANSFORM
# ============================================================

def frequency_transform(image):
    """
    EXACT V2 FFT preprocessing.

    Important:
    FFT is computed at the ORIGINAL image resolution.
    The resulting frequency representation is then
    resized to 224x224.
    """

    # PIL -> Tensor [C,H,W]
    x = transforms.ToTensor()(image)

    # FFT at original image resolution
    fft = torch.fft.fft2(x)

    # Shift low frequencies to center
    fft = torch.fft.fftshift(
        fft,
        dim=(-2, -1)
    )

    # Magnitude
    magnitude = torch.abs(fft)

    # Log scaling
    magnitude = torch.log1p(magnitude)

    # Per-channel min-max normalization
    for c in range(magnitude.shape[0]):

        channel = magnitude[c]

        min_val = channel.min()
        max_val = channel.max()

        magnitude[c] = (
            (channel - min_val)
            / (max_val - min_val + 1e-8)
        )

    # Resize frequency representation
    magnitude = F.interpolate(
        magnitude.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return magnitude

# ============================================================
# SPATIAL MODEL
# ============================================================

class SpatialResNet50(models.ResNet):

    def __init__(self):
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=2
        )
# ============================================================
# FREQUENCY MODEL
# ============================================================

class FrequencyResNet50(models.ResNet):

    def __init__(self):
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=2
        )
# ============================================================
# HYBRID MODEL
# ============================================================

class HybridResNet50FFT(nn.Module):

    def __init__(self):

        super().__init__()

        # Spatial branch
        backbone = models.resnet50(weights=None)

        self.spatial_features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.spatial_dim = 2048

        # Frequency branch
        self.frequency_features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.frequency_dim = 256

        # Fusion
        fusion_dim = (
            self.spatial_dim +
            self.frequency_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
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
            [spatial, frequency],
            dim=1
        )

        return self.classifier(
            combined
        )


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model, path, name):

    print(f"\nLoading {name}")
    print(f"Checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print(f"[✓] {name} loaded successfully")

    return model


# ============================================================
# LOAD MODELS
# ============================================================

spatial_model = load_checkpoint(
    SpatialResNet50(),
    MODEL_DIR / "spatial_resnet50_v2.pth",
    "Spatial V2"
)

frequency_model = load_checkpoint(
    FrequencyResNet50(),
    MODEL_DIR / "frequency_resnet50_v2.pth",
    "Frequency V2"
)

hybrid_model = load_checkpoint(
    HybridResNet50FFT(),
    MODEL_DIR / "hybrid_resnet50_fft_v2.pth",
    "Hybrid V2"
)

# ============================================================
# COLLECT IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

samples = []

for label_name, label in [
    ("REAL", 0),
    ("AI", 1)
]:

    folder = DATA_DIR / label_name

    if not folder.exists():
        print(
            f"ERROR: Folder not found: {folder}"
        )
        raise SystemExit(1)

    files = sorted(
        [
            p for p in folder.iterdir()
            if p.suffix.lower()
            in image_extensions
        ]
    )

    print(
        f"{label_name}: {len(files)} images"
    )

    for path in files:

        samples.append({
            "path": path,
            "filename": path.name,
            "actual": label,
            "actual_name": label_name
        })


print(
    f"\nTotal external images: {len(samples)}"
)

# ============================================================
# EVALUATION STORAGE
# ============================================================

results = []

spatial_true = []
spatial_pred = []

frequency_true = []
frequency_pred = []

hybrid_true = []
hybrid_pred = []


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting evaluation...")
print("=" * 70)

with torch.inference_mode():

    for index, sample in enumerate(samples, start=1):

        path = sample["path"]
        actual = sample["actual"]

        try:

            image = Image.open(path).convert("RGB")

            # ----------------------------------------
            # SPATIAL INPUT
            # ----------------------------------------

            spatial_input = spatial_transform(
                image
            ).unsqueeze(0).to(DEVICE)

            # ----------------------------------------
            # FREQUENCY INPUT
            # ----------------------------------------

            frequency_input = frequency_transform(
                image
            ).unsqueeze(0).to(DEVICE)

            # ----------------------------------------
            # SPATIAL
            # ----------------------------------------

            spatial_logits = spatial_model(
                spatial_input
            )

            spatial_probs = F.softmax(
                spatial_logits,
                dim=1
            )

            spatial_ai_prob = (
                spatial_probs[0, 1]
                .item()
            )

            spatial_prediction = int(
                spatial_ai_prob >= 0.5
            )

            # ----------------------------------------
            # FREQUENCY
            # ----------------------------------------

            frequency_logits = frequency_model(
                frequency_input
            )

            frequency_probs = F.softmax(
                frequency_logits,
                dim=1
            )

            frequency_ai_prob = (
                frequency_probs[0, 1]
                .item()
            )

            frequency_prediction = int(
                frequency_ai_prob >= 0.5
            )

            # ----------------------------------------
            # HYBRID
            # ----------------------------------------

            hybrid_logits = hybrid_model(
                spatial_input,
                frequency_input
            )

            hybrid_probs = F.softmax(
                hybrid_logits,
                dim=1
            )

            hybrid_ai_prob = (
                hybrid_probs[0, 1]
                .item()
            )

            hybrid_prediction = int(
                hybrid_ai_prob >= 0.5
            )

            # ----------------------------------------
            # STORE
            # ----------------------------------------

            results.append({

                "filename": sample["filename"],

                "actual": sample["actual_name"],

                "spatial_prediction":
                    CLASS_NAMES[
                        spatial_prediction
                    ],

                "spatial_ai_probability":
                    round(
                        spatial_ai_prob * 100,
                        4
                    ),

                "frequency_prediction":
                    CLASS_NAMES[
                        frequency_prediction
                    ],

                "frequency_ai_probability":
                    round(
                        frequency_ai_prob * 100,
                        4
                    ),

                "hybrid_prediction":
                    CLASS_NAMES[
                        hybrid_prediction
                    ],

                "hybrid_ai_probability":
                    round(
                        hybrid_ai_prob * 100,
                        4
                    )
            })

            spatial_true.append(actual)
            spatial_pred.append(
                spatial_prediction
            )

            frequency_true.append(actual)
            frequency_pred.append(
                frequency_prediction
            )

            hybrid_true.append(actual)
            hybrid_pred.append(
                hybrid_prediction
            )

            print(
                f"[{index:02d}/{len(samples)}] "
                f"{sample['actual_name']:4s} | "
                f"{sample['filename'][:35]:35s} | "
                f"S={spatial_ai_prob*100:6.2f}% | "
                f"F={frequency_ai_prob*100:6.2f}% | "
                f"H={hybrid_ai_prob*100:6.2f}%"
            )

        except Exception as e:

            print(
                f"[ERROR] {path.name}: {e}"
            )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # REAL recall
    real_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # AI recall
    ai_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # False positive rate:
    # REAL images incorrectly classified as AI
    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    return {

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "precision": round(
            precision * 100,
            2
        ),

        "recall_ai": round(
            recall * 100,
            2
        ),

        "f1": round(
            f1 * 100,
            2
        ),

        "real_recall": round(
            real_recall * 100,
            2
        ),

        "ai_recall": round(
            ai_recall * 100,
            2
        ),

        "false_positive_rate": round(
            false_positive_rate * 100,
            2
        ),

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

        "true_positive": int(tp),

        "confusion_matrix": cm.tolist()
    }


metrics = {

    "Spatial V2": calculate_metrics(
        spatial_true,
        spatial_pred
    ),

    "Frequency V2": calculate_metrics(
        frequency_true,
        frequency_pred
    ),

    "Hybrid V2": calculate_metrics(
        hybrid_true,
        hybrid_pred
    )
}

# ============================================================
# SAVE CSV
# ============================================================

csv_path = OUTPUT_DIR / "external_predictions.csv"

if results:

    fieldnames = results[0].keys()

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_path = (
    OUTPUT_DIR /
    "external_metrics.json"
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

# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EXTERNAL GENERALIZATION RESULTS")
print("=" * 70)

print(
    f"{'Model':<18}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'AI Recall':>12}"
    f"{'Real Recall':>13}"
    f"{'F1':>10}"
    f"{'FPR':>10}"
)

print("-" * 70)

for model_name, m in metrics.items():

    print(
        f"{model_name:<18}"
        f"{m['accuracy']:>11.2f}%"
        f"{m['precision']:>11.2f}%"
        f"{m['ai_recall']:>11.2f}%"
        f"{m['real_recall']:>12.2f}%"
        f"{m['f1']:>9.2f}%"
        f"{m['false_positive_rate']:>9.2f}%"
    )

print("=" * 70)

print("\nConfusion matrices")
print("(Rows = actual REAL/AI, Columns = predicted REAL/AI)\n")

for model_name, m in metrics.items():

    print(model_name)
    print(
        np.array(
            m["confusion_matrix"]
        )
    )
    print()

print("Files saved:")
print(csv_path)
print(metrics_path)