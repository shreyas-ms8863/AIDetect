import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from v5_dataset import V5Dataset


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 16
NUM_CLASSES = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "v5"
RESULT_DIR = PROJECT_ROOT / "evaluation_results" / "v5"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    RESULT_DIR / "v5_final_test_comparison.json"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Device
# ============================================================

print("=" * 80)
print("AIDetect V5 FINAL FROZEN TEST EVALUATION")
print("=" * 80)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(f"PyTorch: {torch.__version__}")


# ============================================================
# Gaussian residual
# ============================================================

def create_gaussian_kernel(
    kernel_size=5,
    sigma=1.0,
    channels=3
):

    radius = kernel_size // 2

    coords = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32
    )

    x = coords.unsqueeze(0)
    y = coords.unsqueeze(1)

    kernel = torch.exp(
        -(x ** 2 + y ** 2) /
        (2 * sigma ** 2)
    )

    kernel = kernel / kernel.sum()

    kernel = kernel.unsqueeze(0).unsqueeze(0)

    kernel = kernel.repeat(
        channels,
        1,
        1,
        1
    )

    return kernel


GAUSSIAN_KERNEL = create_gaussian_kernel().to(device)


def extract_noise_residual(x):

    blurred = F.conv2d(
        x,
        GAUSSIAN_KERNEL,
        padding=2,
        groups=3
    )

    return x - blurred


# ============================================================
# Model definitions
# ============================================================

class SpatialResNet50(nn.Module):

    def __init__(self):

        super().__init__()

        model = resnet50(
            weights=None
        )

        features = model.fc.in_features

        model.fc = nn.Linear(
            features,
            NUM_CLASSES
        )

        self.model = model

    def forward(self, x):

        return self.model(x)


class FrequencyResNet50(nn.Module):

    def __init__(self):

        super().__init__()

        model = resnet50(
            weights=None
        )

        features = model.fc.in_features

        model.fc = nn.Linear(
            features,
            NUM_CLASSES
        )

        self.model = model

    def forward(self, x):

        return self.model(x)


class HybridResNet50(nn.Module):

    def __init__(self):

        super().__init__()

        self.spatial_encoder = resnet50(
            weights=None
        )

        spatial_features = (
            self.spatial_encoder.fc.in_features
        )

        self.spatial_encoder.fc = nn.Identity()

        self.frequency_encoder = resnet50(
            weights=None
        )

        frequency_features = (
            self.frequency_encoder.fc.in_features
        )

        self.frequency_encoder.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(
                spatial_features +
                frequency_features,
                NUM_CLASSES
            )
        )

    def forward(
        self,
        spatial,
        frequency
    ):

        spatial_features = (
            self.spatial_encoder(spatial)
        )

        frequency_features = (
            self.frequency_encoder(frequency)
        )

        fused = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )

        return self.classifier(fused)


class GatedResidualResNet50(nn.Module):

    def __init__(self):

        super().__init__()

        self.spatial_encoder = resnet50(
            weights=None
        )

        spatial_dim = (
            self.spatial_encoder.fc.in_features
        )

        self.spatial_encoder.fc = nn.Identity()

        self.frequency_encoder = resnet50(
            weights=None
        )

        frequency_dim = (
            self.frequency_encoder.fc.in_features
        )

        self.frequency_encoder.fc = nn.Identity()

        self.residual_encoder = resnet50(
            weights=None
        )

        residual_dim = (
            self.residual_encoder.fc.in_features
        )

        self.residual_encoder.fc = nn.Identity()

        fused_dim = (
            spatial_dim +
            frequency_dim +
            residual_dim
        )

        self.gate = nn.Sequential(
            nn.Linear(
                fused_dim,
                fused_dim
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                fused_dim,
                fused_dim
            ),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(
                fused_dim,
                NUM_CLASSES
            )
        )

    def forward(
        self,
        spatial,
        frequency,
        residual
    ):

        spatial_features = (
            self.spatial_encoder(spatial)
        )

        frequency_features = (
            self.frequency_encoder(frequency)
        )

        residual_features = (
            self.residual_encoder(residual)
        )

        combined = torch.cat(
            [
                spatial_features,
                frequency_features,
                residual_features
            ],
            dim=1
        )

        gate = self.gate(combined)

        gated_features = (
            combined * gate
        )

        return self.classifier(
            gated_features
        )


# ============================================================
# Checkpoint loader
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # V5 checkpoints store weights under "state_dict".
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint["state_dict"]
        checkpoint_metadata = checkpoint

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]
        checkpoint_metadata = checkpoint

    else:

        state_dict = checkpoint
        checkpoint_metadata = {}

    # Remove DataParallel prefix if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    state_dict = cleaned_state_dict

    # --------------------------------------------------------
    # Detect whether checkpoint belongs to a wrapper model.
    # --------------------------------------------------------

    model_state_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        state_dict.keys()
    )

    # If checkpoint keys are missing the wrapper prefix,
    # add the appropriate prefix for the simple ResNet model.
    if (
        "model.conv1.weight" in model_state_keys
        and "conv1.weight" in checkpoint_keys
    ):

        state_dict = {
            f"model.{key}": value
            for key, value in state_dict.items()
        }

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing:
        print("\nWARNING - Missing keys:")
        for key in missing:
            print("  ", key)

    if unexpected:
        print("\nWARNING - Unexpected keys:")
        for key in unexpected:
            print("  ", key)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture does not match "
            "the evaluation model."
        )

    model = model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")

    return model, checkpoint_metadata


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    predictions,
    labels
):

    tp = (
        (predictions == 1) &
        (labels == 1)
    ).sum().item()

    tn = (
        (predictions == 0) &
        (labels == 0)
    ).sum().item()

    fp = (
        (predictions == 1) &
        (labels == 0)
    ).sum().item()

    fn = (
        (predictions == 0) &
        (labels == 1)
    ).sum().item()

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) /
        max(1, total)
    )

    precision = (
        tp /
        max(1, tp + fp)
    )

    recall = (
        tp /
        max(1, tp + fn)
    )

    f1 = (
        2 * precision * recall /
        max(
            1e-12,
            precision + recall
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "total": total
    }


# ============================================================
# Load frozen test dataset
# ============================================================

print("\n" + "=" * 80)
print("LOADING FROZEN V5 TEST SET")
print("=" * 80)

test_dataset = V5Dataset(
    split="test",
    mode="hybrid"
)

assert len(test_dataset) == 9000, (
    f"Expected 9,000 test samples, "
    f"got {len(test_dataset)}"
)

print(
    f"Frozen test samples: "
    f"{len(test_dataset):,}"
)

print(
    "This test set is being evaluated "
    "for the first time."
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# Evaluation functions
# ============================================================

@torch.no_grad()
def evaluate_spatial(
    model
):

    predictions = []
    labels = []

    for batch_idx, batch in enumerate(
        test_loader,
        start=1
    ):

        spatial = batch[
            "spatial"
        ].to(
            device,
            non_blocking=True
        )

        label = batch[
            "label"
        ].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            output = model(spatial)

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.append(
            pred.cpu()
        )

        labels.append(
            label.cpu()
        )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(test_loader)
        ):

            done = min(
                batch_idx * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"\rSpatial: "
                f"{done:,}/{len(test_dataset):,}",
                end="",
                flush=True
            )

    print()

    return calculate_metrics(
        torch.cat(predictions),
        torch.cat(labels)
    )


@torch.no_grad()
def evaluate_frequency(
    model
):

    predictions = []
    labels = []

    for batch_idx, batch in enumerate(
        test_loader,
        start=1
    ):

        frequency = batch[
            "frequency"
        ].to(
            device,
            non_blocking=True
        )

        label = batch[
            "label"
        ].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            output = model(
                frequency
            )

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.append(
            pred.cpu()
        )

        labels.append(
            label.cpu()
        )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(test_loader)
        ):

            done = min(
                batch_idx * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"\rFrequency: "
                f"{done:,}/{len(test_dataset):,}",
                end="",
                flush=True
            )

    print()

    return calculate_metrics(
        torch.cat(predictions),
        torch.cat(labels)
    )


@torch.no_grad()
def evaluate_hybrid(
    model
):

    predictions = []
    labels = []

    for batch_idx, batch in enumerate(
        test_loader,
        start=1
    ):

        spatial = batch[
            "spatial"
        ].to(
            device,
            non_blocking=True
        )

        frequency = batch[
            "frequency"
        ].to(
            device,
            non_blocking=True
        )

        label = batch[
            "label"
        ].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            output = model(
                spatial,
                frequency
            )

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.append(
            pred.cpu()
        )

        labels.append(
            label.cpu()
        )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(test_loader)
        ):

            done = min(
                batch_idx * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"\rHybrid: "
                f"{done:,}/{len(test_dataset):,}",
                end="",
                flush=True
            )

    print()

    return calculate_metrics(
        torch.cat(predictions),
        torch.cat(labels)
    )


@torch.no_grad()
def evaluate_gated(
    model
):

    predictions = []
    labels = []

    for batch_idx, batch in enumerate(
        test_loader,
        start=1
    ):

        spatial = batch[
            "spatial"
        ].to(
            device,
            non_blocking=True
        )

        frequency = batch[
            "frequency"
        ].to(
            device,
            non_blocking=True
        )

        label = batch[
            "label"
        ].to(device)

        residual = (
            extract_noise_residual(
                spatial
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            output = model(
                spatial,
                frequency,
                residual
            )

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.append(
            pred.cpu()
        )

        labels.append(
            label.cpu()
        )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(test_loader)
        ):

            done = min(
                batch_idx * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"\rGated Residual: "
                f"{done:,}/{len(test_dataset):,}",
                end="",
                flush=True
            )

    print()

    return calculate_metrics(
        torch.cat(predictions),
        torch.cat(labels)
    )


# ============================================================
# Paths
# ============================================================

spatial_path = (
    MODEL_DIR /
    "spatial_resnet50_v5_best.pth"
)

frequency_path = (
    MODEL_DIR /
    "frequency_resnet50_v5_best.pth"
)

hybrid_path = (
    MODEL_DIR /
    "hybrid_resnet50_v5_best.pth"
)

gated_path = (
    MODEL_DIR /
    "gated_residual_resnet50_v5_best.pth"
)


# ============================================================
# Final evaluation
# ============================================================

results = {
    "test_samples": len(test_dataset),
    "models": {}
}


# ------------------------------------------------------------
# V5-A
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("V5-A SPATIAL-ONLY")
print("=" * 80)

spatial_model = SpatialResNet50()

spatial_model, spatial_checkpoint = load_checkpoint(
    spatial_model,
    spatial_path
)

spatial_results = evaluate_spatial(
    spatial_model
)

results["models"]["V5-A Spatial"] = {
    "best_validation_f1":
        spatial_checkpoint.get(
            "best_val_f1"
        ),
    "test_metrics":
        spatial_results
}

print(
    f"Accuracy : "
    f"{spatial_results['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{spatial_results['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{spatial_results['recall'] * 100:.2f}%"
)

print(
    f"F1       : "
    f"{spatial_results['f1'] * 100:.2f}%"
)


# ------------------------------------------------------------
# V5-B
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("V5-B FREQUENCY-ONLY")
print("=" * 80)

frequency_model = FrequencyResNet50()

frequency_model, frequency_checkpoint = load_checkpoint(
    frequency_model,
    frequency_path
)

frequency_results = evaluate_frequency(
    frequency_model
)

results["models"]["V5-B Frequency"] = {
    "best_validation_f1":
        frequency_checkpoint.get(
            "best_val_f1"
        ),
    "test_metrics":
        frequency_results
}

print(
    f"Accuracy : "
    f"{frequency_results['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{frequency_results['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{frequency_results['recall'] * 100:.2f}%"
)

print(
    f"F1       : "
    f"{frequency_results['f1'] * 100:.2f}%"
)


# ------------------------------------------------------------
# V5-C
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("V5-C SPATIAL + FREQUENCY HYBRID")
print("=" * 80)

hybrid_model = HybridResNet50()

hybrid_model, hybrid_checkpoint = load_checkpoint(
    hybrid_model,
    hybrid_path
)

hybrid_results = evaluate_hybrid(
    hybrid_model
)

results["models"]["V5-C Hybrid"] = {
    "best_validation_f1":
        hybrid_checkpoint.get(
            "best_val_f1"
        ),
    "test_metrics":
        hybrid_results
}

print(
    f"Accuracy : "
    f"{hybrid_results['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{hybrid_results['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{hybrid_results['recall'] * 100:.2f}%"
)

print(
    f"F1       : "
    f"{hybrid_results['f1'] * 100:.2f}%"
)


# ------------------------------------------------------------
# V5-D
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("V5-D GATED + NOISE RESIDUAL")
print("=" * 80)

gated_model = GatedResidualResNet50()

gated_model, gated_checkpoint = load_checkpoint(
    gated_model,
    gated_path
)

gated_results = evaluate_gated(
    gated_model
)

results["models"]["V5-D Gated Residual"] = {
    "best_validation_f1":
        gated_checkpoint.get(
            "best_val_f1"
        ),
    "test_metrics":
        gated_results
}

print(
    f"Accuracy : "
    f"{gated_results['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{gated_results['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{gated_results['recall'] * 100:.2f}%"
)

print(
    f"F1       : "
    f"{gated_results['f1'] * 100:.2f}%"
)


# ============================================================
# Save final results
# ============================================================

results["methodology"] = {
    "test_split": "V5 frozen final test",
    "test_size": 9000,
    "class_mapping": {
        "REAL": 0,
        "AI": 1
    },
    "test_evaluated_after_training": True,
    "training_performed_during_evaluation": False
}

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# Final comparison
# ============================================================

print("\n" + "=" * 80)
print("FINAL V5 TEST COMPARISON")
print("=" * 80)

print(
    f"{'Model':<30}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print("-" * 80)

for name, data in results["models"].items():

    m = data["test_metrics"]

    print(
        f"{name:<30}"
        f"{m['accuracy'] * 100:>11.2f}%"
        f"{m['precision'] * 100:>11.2f}%"
        f"{m['recall'] * 100:>11.2f}%"
        f"{m['f1'] * 100:>11.2f}%"
    )

print("-" * 80)

print("\nResults saved to:")
print(OUTPUT_PATH)

print(
    "\nFINAL TEST EVALUATION COMPLETE."
)

print(
    "The frozen 9,000-image test set has now "
    "been evaluated."
)

print("=" * 80)