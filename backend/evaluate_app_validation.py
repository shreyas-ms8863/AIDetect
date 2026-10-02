import os
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REAL_DIR = BASE_DIR / "real_world_test" / "REAL"
AI_DIR = BASE_DIR / "real_world_test" / "AI"

OUTPUT_DIR = BASE_DIR / "app_validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "validation_predictions.csv"
OUTPUT_JSON = OUTPUT_DIR / "validation_metrics.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("AIDetect - Application Validation")
print("=" * 70)
print(f"Device: {DEVICE}")


# ============================================================
# MODEL
# ============================================================

# IMPORTANT:
# This section uses the SAME pretrained detector architecture
# used by the existing AIDetect backend.
#
# We import the model and preprocessing directly from main.py
# so the evaluation uses the same detector as the website.

from main import model

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


model = model.to(DEVICE)
model.eval()

print("AI detection model loaded successfully.")


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):
    if not folder.exists():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ]
    )


real_images = get_images(REAL_DIR)
ai_images = get_images(AI_DIR)

print()
print(f"REAL images found : {len(real_images)}")
print(f"AI images found   : {len(ai_images)}")
print(f"Total images      : {len(real_images) + len(ai_images)}")


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path):
    """
    Uses the same model/polarity as the existing AIDetect backend.

    IMPORTANT:
    The existing detector outputs REAL probability.

    Therefore:
        real_probability = sigmoid(logit)
        ai_probability   = 1 - real_probability
    """

    image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    # Existing detector returns:
    # classifier output + gates
    if isinstance(output, tuple):
        logits = output[0]
        gates = output[1]
    else:
        logits = output
        gates = None

    # Binary classifier
    #
    # Existing model interpretation:
    # sigmoid(logit) = REAL probability
    real_probability = torch.sigmoid(logits).item()

    ai_probability = 1.0 - real_probability

    if ai_probability >= 0.5:
        prediction = "AI"
        confidence = ai_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    # Extract branch weights if available
    spatial_weight = None
    frequency_weight = None
    noise_weight = None

    if gates is not None:
        try:
            gate_values = gates.squeeze().detach().cpu().numpy().tolist()

            if len(gate_values) >= 3:
                spatial_weight = float(gate_values[0]) * 100
                frequency_weight = float(gate_values[1]) * 100
                noise_weight = float(gate_values[2]) * 100

        except Exception:
            pass

    return {
        "prediction": prediction,
        "confidence": confidence * 100,
        "ai_probability": ai_probability * 100,
        "real_probability": real_probability * 100,
        "spatial_weight": spatial_weight,
        "frequency_weight": frequency_weight,
        "noise_weight": noise_weight,
    }


# ============================================================
# EVALUATION
# ============================================================

results = []

print()
print("=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)


def evaluate_folder(image_paths, true_label):
    for index, image_path in enumerate(image_paths, start=1):

        try:
            result = predict_image(image_path)

            predicted_label = result["prediction"]

            correct = predicted_label == true_label

            row = {
                "filename": image_path.name,
                "true_label": true_label,
                "prediction": predicted_label,
                "correct": correct,
                "ai_probability": round(
                    result["ai_probability"], 4
                ),
                "real_probability": round(
                    result["real_probability"], 4
                ),
                "confidence": round(
                    result["confidence"], 4
                ),
                "spatial_weight": (
                    round(result["spatial_weight"], 4)
                    if result["spatial_weight"] is not None
                    else ""
                ),
                "frequency_weight": (
                    round(result["frequency_weight"], 4)
                    if result["frequency_weight"] is not None
                    else ""
                ),
                "noise_weight": (
                    round(result["noise_weight"], 4)
                    if result["noise_weight"] is not None
                    else ""
                ),
            }

            results.append(row)

            status = "✓" if correct else "✗"

            print(
                f"[{status}] "
                f"{true_label:4} | "
                f"{predicted_label:4} | "
                f"AI: {result['ai_probability']:6.2f}% | "
                f"REAL: {result['real_probability']:6.2f}% | "
                f"{image_path.name}"
            )

        except Exception as e:
            print(
                f"[ERROR] {image_path.name}: {e}"
            )


# REAL first
evaluate_folder(real_images, "REAL")

# AI second
evaluate_folder(ai_images, "AI")


# ============================================================
# METRICS
# ============================================================

total = len(results)

correct = sum(
    1 for row in results
    if row["correct"]
)

accuracy = (
    correct / total
    if total > 0
    else 0
)


# ------------------------------------------------------------
# Confusion matrix
#
#                  PREDICTED
#                REAL       AI
#
# TRUE REAL       TN        FP
# TRUE AI         FN        TP
# ------------------------------------------------------------

true_real = sum(
    1 for row in results
    if row["true_label"] == "REAL"
)

true_ai = sum(
    1 for row in results
    if row["true_label"] == "AI"
)

TN = sum(
    1 for row in results
    if row["true_label"] == "REAL"
    and row["prediction"] == "REAL"
)

FP = sum(
    1 for row in results
    if row["true_label"] == "REAL"
    and row["prediction"] == "AI"
)

FN = sum(
    1 for row in results
    if row["true_label"] == "AI"
    and row["prediction"] == "REAL"
)

TP = sum(
    1 for row in results
    if row["true_label"] == "AI"
    and row["prediction"] == "AI"
)


# ------------------------------------------------------------
# AI-class metrics
# ------------------------------------------------------------

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ------------------------------------------------------------
# Per-class recall
# ------------------------------------------------------------

real_recall = (
    TN / (TN + FP)
    if (TN + FP) > 0
    else 0
)

ai_recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("AIDETECT APPLICATION VALIDATION RESULTS")
print("=" * 70)

print(f"Total images       : {total}")
print(f"Correct            : {correct}")
print(f"Incorrect          : {total - correct}")

print()
print(f"Accuracy           : {accuracy * 100:.2f}%")
print(f"Precision (AI)     : {precision * 100:.2f}%")
print(f"Recall (AI)        : {recall * 100:.2f}%")
print(f"F1-score (AI)      : {f1 * 100:.2f}%")

print()
print(f"REAL recall        : {real_recall * 100:.2f}%")
print(f"AI recall          : {ai_recall * 100:.2f}%")

print()
print("Confusion Matrix")
print()
print("                 Predicted")
print("              REAL       AI")
print(
    f"Actual REAL   {TN:4d}      {FP:4d}"
)
print(
    f"Actual AI     {FN:4d}      {TP:4d}"
)

print()
print(f"True Negatives  : {TN}")
print(f"False Positives : {FP}")
print(f"False Negatives : {FN}")
print(f"True Positives  : {TP}")


# ============================================================
# SAVE CSV
# ============================================================

if results:

    fieldnames = [
        "filename",
        "true_label",
        "prediction",
        "correct",
        "ai_probability",
        "real_probability",
        "confidence",
        "spatial_weight",
        "frequency_weight",
        "noise_weight",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# SAVE JSON
# ============================================================

metrics = {
    "total_images": total,
    "correct": correct,
    "incorrect": total - correct,

    "accuracy": round(accuracy * 100, 4),

    "precision_ai": round(
        precision * 100, 4
    ),

    "recall_ai": round(
        recall * 100, 4
    ),

    "f1_ai": round(
        f1 * 100, 4
    ),

    "real_recall": round(
        real_recall * 100, 4
    ),

    "ai_recall": round(
        ai_recall * 100, 4
    ),

    "confusion_matrix": {
        "true_negative": TN,
        "false_positive": FP,
        "false_negative": FN,
        "true_positive": TP,
    },

    "dataset": {
        "real_images": len(real_images),
        "ai_images": len(ai_images),
    },
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print()
print("Results saved to:")

print(f"CSV : {OUTPUT_CSV}")
print(f"JSON: {OUTPUT_JSON}")

print("=" * 70)