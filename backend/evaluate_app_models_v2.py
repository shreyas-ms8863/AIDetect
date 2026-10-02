"""
AIDetect - Correct V2 Application Validation

Evaluates:
1. Spatial V2
2. Frequency V2
3. Hybrid V2

IMPORTANT:
- Uses the architectures used to create the V2 checkpoints.
- Uses strict checkpoint loading.
- Does NOT retrain anything.
- The 19 application images are external validation data.
- CIFAKE labels:
      0 = REAL
      1 = AI
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

REAL_DIR = os.path.join(
    BASE_DIR,
    "real_world_test",
    "REAL"
)

AI_DIR = os.path.join(
    BASE_DIR,
    "real_world_test",
    "AI"
)

MODEL_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "models"
)

SPATIAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "spatial_resnet50_v2.pth"
)

FREQUENCY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "frequency_resnet50_v2.pth"
)

HYBRID_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hybrid_resnet50_fft_v2.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "app_v2_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FFT TRANSFORM
# ============================================================

class FFTTransform:

    def __call__(self, image):

        # ----------------------------------------------------
        # PIL -> Tensor
        # ----------------------------------------------------

        x = transforms.ToTensor()(image)

        # ----------------------------------------------------
        # FFT over H and W
        # ----------------------------------------------------

        fft = torch.fft.fft2(x)

        # ----------------------------------------------------
        # Shift low frequencies to center
        # ----------------------------------------------------

        fft = torch.fft.fftshift(
            fft,
            dim=(-2, -1)
        )

        # ----------------------------------------------------
        # Magnitude
        # ----------------------------------------------------

        magnitude = torch.abs(fft)

        # ----------------------------------------------------
        # Log scaling
        # ----------------------------------------------------

        magnitude = torch.log1p(
            magnitude
        )

        # ----------------------------------------------------
        # Per-channel min-max normalization
        # ----------------------------------------------------

        for c in range(
            magnitude.shape[0]
        ):

            channel = magnitude[c]

            min_val = channel.min()
            max_val = channel.max()

            magnitude[c] = (
                (channel - min_val)
                /
                (
                    max_val - min_val
                    + 1e-8
                )
            )

        # ----------------------------------------------------
        # Resize to 224 x 224
        # ----------------------------------------------------

        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return magnitude


# ============================================================
# DATASET
# ============================================================

class ApplicationDataset(Dataset):

    def __init__(self):

        self.samples = []

        # ----------------------------------------------------
        # REAL = 0
        # ----------------------------------------------------

        if os.path.exists(REAL_DIR):

            for filename in sorted(
                os.listdir(REAL_DIR)
            ):

                path = os.path.join(
                    REAL_DIR,
                    filename
                )

                if self.is_image(path):

                    self.samples.append(
                        (path, 0)
                    )

        # ----------------------------------------------------
        # AI = 1
        # ----------------------------------------------------

        if os.path.exists(AI_DIR):

            for filename in sorted(
                os.listdir(AI_DIR)
            ):

                path = os.path.join(
                    AI_DIR,
                    filename
                )

                if self.is_image(path):

                    self.samples.append(
                        (path, 1)
                    )

    @staticmethod
    def is_image(path):

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp"
        )

        return path.lower().endswith(
            extensions
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(
            path
        ).convert("RGB")

        return image, label, path


# ============================================================
# SPATIAL MODEL
# ============================================================

class SpatialResNet50(nn.Module):

    """
    Spatial V2 uses a ResNet50 classifier.

    The checkpoint contains RAW ResNet50 keys:

        conv1.weight
        bn1.weight
        layer1.*
        layer2.*
        layer3.*
        layer4.*
        fc.weight
        fc.bias

    Therefore the checkpoint is loaded directly
    into self.model.
    """

    def __init__(self):

        super().__init__()

        self.model = models.resnet50(
            weights=None
        )

        num_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            num_features,
            2
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# FREQUENCY MODEL
# ============================================================

class FrequencyResNet50(nn.Module):

    """
    Frequency V2 uses the same ResNet50
    architecture as Spatial V2.

    The input is the FFT magnitude
    representation.
    """

    def __init__(self):

        super().__init__()

        self.model = models.resnet50(
            weights=None
        )

        num_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            num_features,
            2
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridResNet50FFT(nn.Module):

    """
    Exact Hybrid V2 architecture.

    The checkpoint is expected to contain:

        spatial_features.*
        frequency_features.*
        classifier.*
    """

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # SPATIAL BRANCH
        # ----------------------------------------------------

        spatial_model = models.resnet50(
            weights=None
        )

        self.spatial_features = nn.Sequential(
            *list(
                spatial_model.children()
            )[:-1]
        )

        self.spatial_dim = 2048

        # ----------------------------------------------------
        # FREQUENCY BRANCH
        # ----------------------------------------------------

        self.frequency_features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.frequency_dim = 256

        # ----------------------------------------------------
        # FUSION CLASSIFIER
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

        # ----------------------------------------------------
        # Spatial features
        # ----------------------------------------------------

        spatial_features = (
            self.spatial_features(
                spatial_input
            )
        )

        spatial_features = (
            spatial_features.view(
                spatial_features.size(0),
                -1
            )
        )

        # ----------------------------------------------------
        # Frequency features
        # ----------------------------------------------------

        frequency_features = (
            self.frequency_features(
                frequency_input
            )
        )

        frequency_features = (
            frequency_features.view(
                frequency_features.size(0),
                -1
            )
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        combined = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        return self.classifier(
            combined
        )


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    model_name
):

    print()
    print("-" * 70)
    print(
        f"Loading {model_name}"
    )
    print("-" * 70)

    print(
        checkpoint_path
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            # Raw state dictionary
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        cleaned_state_dict[
            key
        ] = value

    print(
        f"Checkpoint keys: "
        f"{len(cleaned_state_dict)}"
    )

    # --------------------------------------------------------
    # SELECT TARGET MODEL
    #
    # Spatial/Frequency:
    #
    # checkpoint:
    #     conv1.weight
    #     bn1.weight
    #     layer1.*
    #     fc.weight
    #
    # wrapper:
    #     model.conv1.weight
    #
    # Therefore load into model.model.
    #
    # Hybrid:
    #     spatial_features.*
    #     frequency_features.*
    #     classifier.*
    #
    # Therefore load into model itself.
    # --------------------------------------------------------

    if (
        model_name == "Spatial V2"
        or
        model_name == "Frequency V2"
    ):

        target_model = model.model

    else:

        target_model = model

    # --------------------------------------------------------
    # STRICT LOAD
    # --------------------------------------------------------

    try:

        target_model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    except RuntimeError as error:

        print()
        print(
            "ERROR: CHECKPOINT DOES NOT "
            "MATCH MODEL ARCHITECTURE."
        )

        print()
        print(error)

        raise RuntimeError(
            f"\n{model_name} checkpoint "
            f"does not match the architecture."
            f"\n\nDo NOT continue with "
            f"invalid weights."
        )

    # --------------------------------------------------------
    # Device + evaluation mode
    # --------------------------------------------------------

    model.to(DEVICE)
    model.eval()

    print(
        f"[✓] {model_name} "
        f"loaded with STRICT matching."
    )

    return model


# ============================================================
# PREDICTION FROM LOGITS
# ============================================================

def prediction_from_logits(
    logits
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_class = (
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )

    real_probability = (
        probabilities[0, 0].item()
    )

    ai_probability = (
        probabilities[0, 1].item()
    )

    return (
        predicted_class,
        real_probability,
        ai_probability
    )


# ============================================================
# PRINT PREDICTION
# ============================================================

def print_prediction(
    label,
    predicted,
    real_prob,
    ai_prob,
    path
):

    actual_text = (
        "REAL"
        if label == 0
        else "AI"
    )

    predicted_text = (
        "REAL"
        if predicted == 0
        else "AI"
    )

    correct = (
        label == predicted
    )

    print(
        f"[{'✓' if correct else '✗'}] "
        f"{actual_text:4s} | "
        f"{predicted_text:4s} | "
        f"AI: {ai_prob * 100:6.2f}% | "
        f"REAL: {real_prob * 100:6.2f}% | "
        f"{os.path.basename(path)}"
    )

    return correct


# ============================================================
# CREATE RESULT ROW
# ============================================================

def make_row(
    model_name,
    label,
    predicted,
    real_prob,
    ai_prob,
    path
):

    actual_text = (
        "REAL"
        if label == 0
        else "AI"
    )

    predicted_text = (
        "REAL"
        if predicted == 0
        else "AI"
    )

    return {

        "model":
            model_name,

        "filename":
            os.path.basename(path),

        "actual":
            actual_text,

        "prediction":
            predicted_text,

        "ai_probability":
            ai_prob,

        "real_probability":
            real_prob,

        "correct":
            label == predicted
    }


# ============================================================
# SPATIAL EVALUATION
# ============================================================

def evaluate_spatial(
    dataset
):

    print()
    print("=" * 70)
    print(
        "EVALUATING SPATIAL V2"
    )
    print("=" * 70)

    model = SpatialResNet50()

    model = load_checkpoint(
        model,
        SPATIAL_MODEL_PATH,
        "Spatial V2"
    )

    y_true = []
    y_pred = []
    rows = []

    for image, label, path in dataset:

        input_tensor = (
            image_transform(
                image
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            logits = model(
                input_tensor
            )

        (
            predicted,
            real_prob,
            ai_prob
        ) = prediction_from_logits(
            logits
        )

        y_true.append(label)
        y_pred.append(predicted)

        print_prediction(
            label,
            predicted,
            real_prob,
            ai_prob,
            path
        )

        rows.append(
            make_row(
                "Spatial V2",
                label,
                predicted,
                real_prob,
                ai_prob,
                path
            )
        )

    return calculate_metrics(
        "Spatial V2",
        y_true,
        y_pred,
        rows
    )


# ============================================================
# FREQUENCY EVALUATION
# ============================================================

def evaluate_frequency(
    dataset
):

    print()
    print("=" * 70)
    print(
        "EVALUATING FREQUENCY V2"
    )
    print("=" * 70)

    model = FrequencyResNet50()

    model = load_checkpoint(
        model,
        FREQUENCY_MODEL_PATH,
        "Frequency V2"
    )

    frequency_transform = (
        FFTTransform()
    )

    y_true = []
    y_pred = []
    rows = []

    for image, label, path in dataset:

        input_tensor = (
            frequency_transform(
                image
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            logits = model(
                input_tensor
            )

        (
            predicted,
            real_prob,
            ai_prob
        ) = prediction_from_logits(
            logits
        )

        y_true.append(label)
        y_pred.append(predicted)

        print_prediction(
            label,
            predicted,
            real_prob,
            ai_prob,
            path
        )

        rows.append(
            make_row(
                "Frequency V2",
                label,
                predicted,
                real_prob,
                ai_prob,
                path
            )
        )

    return calculate_metrics(
        "Frequency V2",
        y_true,
        y_pred,
        rows
    )


# ============================================================
# HYBRID EVALUATION
# ============================================================

def evaluate_hybrid(
    dataset
):

    print()
    print("=" * 70)
    print(
        "EVALUATING HYBRID V2"
    )
    print("=" * 70)

    model = HybridResNet50FFT()

    model = load_checkpoint(
        model,
        HYBRID_MODEL_PATH,
        "Hybrid V2"
    )

    frequency_transform = (
        FFTTransform()
    )

    y_true = []
    y_pred = []
    rows = []

    for image, label, path in dataset:

        # ----------------------------------------------------
        # Spatial input
        # ----------------------------------------------------

        spatial_input = (
            image_transform(
                image
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Frequency input
        # ----------------------------------------------------

        frequency_input = (
            frequency_transform(
                image
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                spatial_input,
                frequency_input
            )

        (
            predicted,
            real_prob,
            ai_prob
        ) = prediction_from_logits(
            logits
        )

        y_true.append(label)
        y_pred.append(predicted)

        print_prediction(
            label,
            predicted,
            real_prob,
            ai_prob,
            path
        )

        rows.append(
            make_row(
                "Hybrid V2",
                label,
                predicted,
                real_prob,
                ai_prob,
                path
            )
        )

    return calculate_metrics(
        "Hybrid V2",
        y_true,
        y_pred,
        rows
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    rows
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    real_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    ai_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    metrics = {

        "model":
            model_name,

        "total_images":
            len(y_true),

        "correct":
            int(
                sum(
                    a == b
                    for a, b in zip(
                        y_true,
                        y_pred
                    )
                )
            ),

        "incorrect":
            int(
                sum(
                    a != b
                    for a, b in zip(
                        y_true,
                        y_pred
                    )
                )
            ),

        "accuracy":
            accuracy,

        "precision_ai":
            precision,

        "recall_ai":
            recall,

        "f1_ai":
            f1,

        "real_recall":
            real_recall,

        "ai_recall":
            ai_recall,

        "true_negatives":
            int(tn),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_positives":
            int(tp)
    }

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    print(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision (AI)  : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall (AI)     : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-score (AI)   : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"REAL recall     : "
        f"{real_recall * 100:.2f}%"
    )

    print(
        f"AI recall       : "
        f"{ai_recall * 100:.2f}%"
    )

    print()
    print("Confusion Matrix")
    print()
    print("                 Predicted")
    print("              REAL       AI")

    print(
        f"Actual REAL    "
        f"{tn:4d}      {fp:4d}"
    )

    print(
        f"Actual AI      "
        f"{fn:4d}      {tp:4d}"
    )

    return metrics, rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AIDetect - CORRECT V2 APPLICATION VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print(
        "Validation directories:"
    )

    print(
        "REAL:",
        REAL_DIR
    )

    print(
        "AI  :",
        AI_DIR
    )

    # ========================================================
    # CHECK MODEL FILES
    # ========================================================

    required_models = [

        SPATIAL_MODEL_PATH,

        FREQUENCY_MODEL_PATH,

        HYBRID_MODEL_PATH
    ]

    print()
    print(
        "Checking model files..."
    )

    for path in required_models:

        if not os.path.exists(path):

            print()
            print(
                "ERROR: Model not found:"
            )

            print(path)

            return

        print(
            "[✓]",
            path
        )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = ApplicationDataset()

    real_count = sum(
        1
        for _, label in dataset.samples
        if label == 0
    )

    ai_count = sum(
        1
        for _, label in dataset.samples
        if label == 1
    )

    print()
    print("=" * 70)
    print(
        "APPLICATION VALIDATION DATASET"
    )
    print("=" * 70)

    print()

    print(
        f"REAL images found : "
        f"{real_count}"
    )

    print(
        f"AI images found   : "
        f"{ai_count}"
    )

    print(
        f"Total images      : "
        f"{len(dataset)}"
    )

    if len(dataset) == 0:

        print()
        print(
            "ERROR: No images found."
        )

        return

    # ========================================================
    # EVALUATE MODELS
    # ========================================================

    spatial_metrics, spatial_rows = (
        evaluate_spatial(
            dataset
        )
    )

    frequency_metrics, frequency_rows = (
        evaluate_frequency(
            dataset
        )
    )

    hybrid_metrics, hybrid_rows = (
        evaluate_hybrid(
            dataset
        )
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    all_rows = (
        spatial_rows
        +
        frequency_rows
        +
        hybrid_rows
    )

    predictions_csv = os.path.join(
        OUTPUT_DIR,
        "application_v2_predictions.csv"
    )

    pd.DataFrame(
        all_rows
    ).to_csv(
        predictions_csv,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    metrics_list = [

        spatial_metrics,

        frequency_metrics,

        hybrid_metrics
    ]

    summary_rows = []

    for m in metrics_list:

        summary_rows.append({

            "Model":
                m["model"],

            "Accuracy (%)":
                m["accuracy"] * 100,

            "Precision AI (%)":
                m["precision_ai"] * 100,

            "Recall AI (%)":
                m["recall_ai"] * 100,

            "F1 AI (%)":
                m["f1_ai"] * 100,

            "REAL Recall (%)":
                m["real_recall"] * 100,

            "AI Recall (%)":
                m["ai_recall"] * 100
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = os.path.join(
        OUTPUT_DIR,
        "application_v2_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    # ========================================================
    # JSON
    # ========================================================

    json_output = {

        "device":
            str(DEVICE),

        "dataset": {

            "real_images":
                real_count,

            "ai_images":
                ai_count,

            "total_images":
                len(dataset)
        },

        "models":
            metrics_list
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "application_v2_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_output,
            f,
            indent=4
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL APPLICATION MODEL COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        summary_df.to_string(
            index=False,
            formatters={

                "Accuracy (%)":
                    "{:.2f}".format,

                "Precision AI (%)":
                    "{:.2f}".format,

                "Recall AI (%)":
                    "{:.2f}".format,

                "F1 AI (%)":
                    "{:.2f}".format,

                "REAL Recall (%)":
                    "{:.2f}".format,

                "AI Recall (%)":
                    "{:.2f}".format
            }
        )
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VALIDATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Results saved:"
    )

    print()
    print(
        "Predictions:"
    )

    print(
        predictions_csv
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_csv
    )

    print()
    print(
        "Metrics:"
    )

    print(
        json_path
    )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()