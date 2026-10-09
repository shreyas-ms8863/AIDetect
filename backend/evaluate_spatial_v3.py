"""
AIDetect - Spatial V3 Clean Test Evaluation

Evaluates the trained Spatial ResNet-50 V3 model on the
untouched CIFAKE official test split.

Dataset:
    CIFAKE
    Test = 20,000 images
    10,000 FAKE
    10,000 REAL

Important:
    The official test set was NOT used during V3 training
    or validation.

Output:
    evaluation_results/spatial_v3_test_metrics.json
    evaluation_results/spatial_v3_test_predictions.csv
    evaluation_results/spatial_v3_confusion_matrix.csv
"""

import os
import json
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "spatial_resnet50_v3.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "evaluation_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "spatial_v3_test_metrics.json"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "spatial_v3_test_predictions.csv"
)

CONFUSION_PATH = os.path.join(
    OUTPUT_DIR,
    "spatial_v3_confusion_matrix.csv"
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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# DATASET WRAPPER
# ============================================================

class CIFAKETestDataset(Dataset):

    def __init__(self, hf_dataset):

        self.dataset = hf_dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]
        label = int(item["label"])

        # Ensure RGB
        image = image.convert("RGB")

        image = image_transform(image)

        return image, label, index


# ============================================================
# MODEL
# ============================================================

def create_model():

    weights = ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    # Same classifier structure used during V3 training
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(2048, 2)
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    fake_class=0,
    real_class=1
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total = len(y_true)

    correct = int(np.sum(y_true == y_pred))

    accuracy = correct / total if total > 0 else 0.0

    # Confusion matrix:
    #
    #                Pred FAKE    Pred REAL
    # Actual FAKE       TN          FP
    # Actual REAL       FN          TP
    #
    # Because class 1 = REAL.

    tn = int(np.sum(
        (y_true == fake_class) &
        (y_pred == fake_class)
    ))

    fp = int(np.sum(
        (y_true == fake_class) &
        (y_pred == real_class)
    ))

    fn = int(np.sum(
        (y_true == real_class) &
        (y_pred == fake_class)
    ))

    tp = int(np.sum(
        (y_true == real_class) &
        (y_pred == real_class)
    ))

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Class-wise metrics

    fake_precision = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    fake_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fake_f1 = (
        2 * fake_precision * fake_recall /
        (fake_precision + fake_recall)
        if (fake_precision + fake_recall) > 0
        else 0.0
    )

    real_precision = precision
    real_recall = recall
    real_f1 = f1

    # False positive / negative rates
    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "total_images": total,
        "correct": correct,
        "incorrect": total - correct,

        "accuracy": accuracy,

        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,

        "real_precision": real_precision,
        "real_recall": real_recall,
        "real_f1": real_f1,

        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,

        "confusion_matrix": {
            "actual_fake_predicted_fake": tn,
            "actual_fake_predicted_real": fp,
            "actual_real_predicted_fake": fn,
            "actual_real_predicted_real": tp
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("AIDetect - Spatial V3 Clean Test Evaluation")
    print("=" * 75)

    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory      : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print("=" * 75)


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    print("\nChecking model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nV3 model not found:\n{MODEL_PATH}"
        )

    print(f"[OK] {MODEL_PATH}")


    # --------------------------------------------------------
    # LOAD CIFAKE
    # --------------------------------------------------------

    print("\nLoading CIFAKE dataset...")

    dataset = load_dataset(
        "dragonintelligence/CIFAKE-image-dataset"
    )

    test_dataset = dataset["test"]

    print("\n" + "=" * 75)
    print("TEST DATASET")
    print("=" * 75)

    print(
        f"Total test images : {len(test_dataset):,}"
    )

    # Count labels

    labels = test_dataset["label"]

    fake_count = sum(
        1 for x in labels if int(x) == 0
    )

    real_count = sum(
        1 for x in labels if int(x) == 1
    )

    print(f"FAKE images       : {fake_count:,}")
    print(f"REAL images       : {real_count:,}")

    print("=" * 75)


    # --------------------------------------------------------
    # DATASET / DATALOADER
    # --------------------------------------------------------

    test_data = CIFAKETestDataset(
        test_dataset
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nDataLoader:")
    print(f"Batch size : 32")
    print(f"Batches    : {len(test_loader)}")
    print(f"Workers    : 0")


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("LOADING SPATIAL V3 MODEL")
    print("=" * 75)

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Handle either a raw state_dict or a checkpoint dictionary

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    print("[OK] Spatial V3 loaded successfully")
    print("[OK] Strict checkpoint matching")
    print("=" * 75)


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STARTING V3 CLEAN TEST EVALUATION")
    print("=" * 75)

    print("IMPORTANT:")
    print("The official 20,000-image CIFAKE test set")
    print("was NOT used during training or validation.")
    print("=" * 75)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_images = 0
    correct_images = 0

    start_time = time.time()

    with torch.no_grad():

        for batch_idx, (
            images,
            labels,
            indices
        ) in enumerate(test_loader, start=1):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct = (
                predictions == labels
            ).sum().item()

            total_images += labels.size(0)
            correct_images += correct

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            if (
                batch_idx == 1
                or batch_idx % 100 == 0
                or batch_idx == len(test_loader)
            ):

                running_accuracy = (
                    correct_images /
                    total_images *
                    100
                )

                print(
                    f"Batch "
                    f"{batch_idx}/{len(test_loader)} | "
                    f"Images "
                    f"{total_images:,}/"
                    f"{len(test_data):,} | "
                    f"Running Accuracy: "
                    f"{running_accuracy:.2f}%"
                )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    elapsed_time = time.time() - start_time

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["model"] = "Spatial ResNet-50 V3"

    metrics["dataset"] = "CIFAKE"

    metrics["split"] = "Test"

    metrics["model_path"] = MODEL_PATH

    metrics["evaluation_time_seconds"] = elapsed_time

    metrics["evaluation_time_minutes"] = (
        elapsed_time / 60
    )

    metrics["test_set_status"] = (
        "Official CIFAKE test set - "
        "untouched during training and validation"
    )


    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SPATIAL V3 - FINAL CIFAKE TEST RESULTS")
    print("=" * 75)

    print(
        f"Total images       : "
        f"{metrics['total_images']:,}"
    )

    print(
        f"Correct            : "
        f"{metrics['correct']:,}"
    )

    print(
        f"Incorrect          : "
        f"{metrics['incorrect']:,}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision          : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall             : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score           : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print("\nClass-wise metrics")

    print(
        f"FAKE Precision     : "
        f"{metrics['fake_precision'] * 100:.2f}%"
    )

    print(
        f"FAKE Recall        : "
        f"{metrics['fake_recall'] * 100:.2f}%"
    )

    print(
        f"FAKE F1            : "
        f"{metrics['fake_f1'] * 100:.2f}%"
    )

    print(
        f"REAL Precision     : "
        f"{metrics['real_precision'] * 100:.2f}%"
    )

    print(
        f"REAL Recall        : "
        f"{metrics['real_recall'] * 100:.2f}%"
    )

    print(
        f"REAL F1            : "
        f"{metrics['real_f1'] * 100:.2f}%"
    )

    print("\nConfusion Matrix")

    cm = metrics["confusion_matrix"]

    print(
        "                     Predicted FAKE   Predicted REAL"
    )

    print(
        f"Actual FAKE          "
        f"{cm['actual_fake_predicted_fake']:>14,}   "
        f"{cm['actual_fake_predicted_real']:>14,}"
    )

    print(
        f"Actual REAL          "
        f"{cm['actual_real_predicted_fake']:>14,}   "
        f"{cm['actual_real_predicted_real']:>14,}"
    )

    print("\nError rates")

    print(
        f"False Positive Rate  : "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )

    print(
        f"False Negative Rate  : "
        f"{metrics['false_negative_rate'] * 100:.2f}%"
    )

    print(
        f"\nEvaluation time      : "
        f"{elapsed_time / 60:.2f} minutes"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # SAVE METRICS JSON
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # SAVE PREDICTIONS CSV
    # --------------------------------------------------------

    prediction_rows = []

    for i in range(len(all_labels)):

        prediction_rows.append({

            "index": i,

            "true_label": (
                "FAKE"
                if all_labels[i] == 0
                else "REAL"
            ),

            "predicted_label": (
                "FAKE"
                if all_predictions[i] == 0
                else "REAL"
            ),

            "fake_probability":
                float(all_probabilities[i][0]),

            "real_probability":
                float(all_probabilities[i][1]),

            "correct":
                bool(
                    all_labels[i] ==
                    all_predictions[i]
                )
        })


    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )


    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX CSV
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        [
            [
                cm["actual_fake_predicted_fake"],
                cm["actual_fake_predicted_real"]
            ],
            [
                cm["actual_real_predicted_fake"],
                cm["actual_real_predicted_real"]
            ]
        ],
        index=[
            "Actual FAKE",
            "Actual REAL"
        ],
        columns=[
            "Predicted FAKE",
            "Predicted REAL"
        ]
    )

    confusion_df.to_csv(
        CONFUSION_PATH
    )


    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("V3 EVALUATION COMPLETE")
    print("=" * 75)

    print("\nSaved files:")

    print(
        f"[1] {METRICS_PATH}"
    )

    print(
        f"[2] {PREDICTIONS_PATH}"
    )

    print(
        f"[3] {CONFUSION_PATH}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()