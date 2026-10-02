import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16

OUTPUT_DIR = "genimage_v2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("V2 CROSS-DATASET EVALUATION")
print("=" * 65)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD TINY-GENIMAGE
# ============================================================

print("\nLoading Tiny-GenImage...")

from huggingface_hub import hf_hub_download

print("\nDownloading required Tiny-GenImage validation shard...")

parquet_path = hf_hub_download(
    repo_id="TheKernel01/Tiny-GenImage",
    filename="data/validation-00000-of-00004.parquet",
    repo_type="dataset"
)

dataset = load_dataset(
    "parquet",
    data_files=parquet_path,
    split="train"
)

print(f"Available images in shard: {len(dataset)}")

print(f"Available images in shard: {len(dataset)}")


# ============================================================
# SELECT SAME-SIZE BALANCED SET
# 50 REAL + 50 AI
# ============================================================

real_indices = []
ai_indices = []

for i in range(len(dataset)):

    label = int(dataset[i]["label"])

    # The previous experiment used:
    # 50 Real + 50 AI

    if label == 0 and len(real_indices) < 50:
        real_indices.append(i)

    elif label == 1 and len(ai_indices) < 50:
        ai_indices.append(i)

    if len(real_indices) == 50 and len(ai_indices) == 50:
        break


selected_indices = real_indices + ai_indices

print(f"Real images selected: {len(real_indices)}")
print(f"AI images selected:   {len(ai_indices)}")
print(f"Total selected:       {len(selected_indices)}")


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

    fft = torch.fft.fft2(image_tensor)
    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)
    magnitude = torch.log1p(magnitude)

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
# DATASET
# ============================================================

class GenImageDataset(Dataset):

    def __init__(self, hf_dataset, indices):

        self.dataset = hf_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        item = self.dataset[real_idx]

        image = item["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        spatial = image_transform(image)

        frequency = frequency_transform(image)

        original_label = int(item["label"])

        # Tiny-GenImage:
        # 0 = Real
        # 1 = AI
        #
        # Our models:
        # 0 = Real
        # 1 = AI

        label = original_label

        return (
            spatial,
            frequency,
            label,
            real_idx
        )


test_dataset = GenImageDataset(
    dataset,
    selected_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# MODEL 1: SPATIAL V2
# ============================================================

def load_spatial_model():

    model = resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    checkpoint = torch.load(
        "models/spatial_resnet50_v2.pth",
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

    return model.to(DEVICE)


# ============================================================
# MODEL 2: FREQUENCY V2
# ============================================================

def load_frequency_model():

    model = resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    checkpoint = torch.load(
        "models/frequency_resnet50_v2.pth",
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

    return model.to(DEVICE)


# ============================================================
# MODEL 3: HYBRID V2
# ============================================================

class HybridResNetFFT(nn.Module):

    def __init__(self):

        super().__init__()

        spatial_model = resnet50(
            weights=None
        )

        self.spatial_features = nn.Sequential(
            *list(spatial_model.children())[:-1]
        )

        self.spatial_dim = 2048

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

        self.classifier = nn.Sequential(

            nn.Linear(
                2048 + 256,
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


def load_hybrid_model():

    model = HybridResNetFFT()

    checkpoint = torch.load(
        "models/hybrid_resnet50_fft_v2.pth",
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

    return model.to(DEVICE)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_spatial(model):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for (
            spatial,
            frequency,
            batch_labels,
            _
        ) in tqdm(
            test_loader,
            desc="Spatial V2"
        ):

            spatial = spatial.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(spatial)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return labels, predictions


def evaluate_frequency(model):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for (
            spatial,
            frequency,
            batch_labels,
            _
        ) in tqdm(
            test_loader,
            desc="Frequency V2"
        ):

            frequency = frequency.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(frequency)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return labels, predictions


def evaluate_hybrid(model):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for (
            spatial,
            frequency,
            batch_labels,
            _
        ) in tqdm(
            test_loader,
            desc="Hybrid V2"
        ):

            spatial = spatial.to(
                DEVICE,
                non_blocking=True
            )

            frequency = frequency.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                spatial,
                frequency
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return labels, predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, predictions):

    cm = confusion_matrix(
        labels,
        predictions
    )

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "confusion_matrix": cm.tolist()
    }


# ============================================================
# RUN ALL THREE
# ============================================================

print("\nLoading Spatial V2...")
spatial_model = load_spatial_model()

print("Loading Frequency V2...")
frequency_model = load_frequency_model()

print("Loading Hybrid V2...")
hybrid_model = load_hybrid_model()


print("\n" + "=" * 65)
print("RUNNING CROSS-DATASET TEST")
print("=" * 65)


spatial_labels, spatial_preds = evaluate_spatial(
    spatial_model
)

frequency_labels, frequency_preds = evaluate_frequency(
    frequency_model
)

hybrid_labels, hybrid_preds = evaluate_hybrid(
    hybrid_model
)


# ============================================================
# RESULTS
# ============================================================

spatial_metrics = calculate_metrics(
    spatial_labels,
    spatial_preds
)

frequency_metrics = calculate_metrics(
    frequency_labels,
    frequency_preds
)

hybrid_metrics = calculate_metrics(
    hybrid_labels,
    hybrid_preds
)


results = {

    "dataset": "Tiny-GenImage",

    "split": "validation",

    "source_shard":
        "data/validation-00000-of-00004.parquet",

    "total_images": len(selected_indices),

    "class_distribution": {
        "real": len(real_indices),
        "ai": len(ai_indices)
    },

    "models": {

        "spatial_v2":
            spatial_metrics,

        "frequency_v2":
            frequency_metrics,

        "hybrid_v2":
            hybrid_metrics
    }
}


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 65)
print("V2 TINY-GENIMAGE RESULTS")
print("=" * 65)

for name, metrics in results["models"].items():

    print(f"\n{name.upper()}")

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print("Confusion Matrix:")
    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "genimage_v2_metrics.json"
)

with open(output_path, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\n" + "=" * 65)
print("CROSS-DATASET EVALUATION COMPLETE")
print("=" * 65)

print("\nResults saved to:")
print(output_path)