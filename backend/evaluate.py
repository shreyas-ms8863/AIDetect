import os
import json
import csv

import torch
import torchvision.transforms as T

from PIL import Image

from huggingface_hub import hf_hub_download
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO = "Medsa/ai-image-authenticity-detector"
MODEL_FILE = "detector_scripted.pt"

DATASET_REPO = "dragonintelligence/CIFAKE-image-dataset"

BATCH_SIZE = 64

OUTPUT_DIR = "evaluation_results"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("AIDetect Dataset Evaluation")
print("=" * 60)
print()

print("Loading AI detection model...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

model = torch.jit.load(
    model_path,
    map_location="cpu"
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREPROCESSING
# ============================================================

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# LOAD CIFAKE TEST DATASET
# ============================================================

print()
print("Loading CIFAKE test dataset...")
print()

dataset = load_dataset(
    DATASET_REPO,
    split="test"
)

total_images = len(dataset)

print(
    f"Dataset loaded: {total_images:,} images"
)

print()

print("Dataset labels:")
print("0 = FAKE / AI-GENERATED")
print("1 = REAL")
print()


# ============================================================
# STORAGE
# ============================================================

true_labels = []
predicted_labels = []
ai_probabilities = []
real_probabilities = []


# ============================================================
# MODEL EVALUATION
# ============================================================

print("Running model inference...")
print()


for start in range(
    0,
    total_images,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_images
    )

    batch = dataset[start:end]

    images = batch["image"]
    labels = batch["label"]

    tensors = []

    for image in images:

        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        tensor = transform(image)

        tensors.append(tensor)

    input_tensor = torch.stack(tensors)


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

        logit = output[0]

        # IMPORTANT:
        #
        # Diagnostic verification showed that the sigmoid
        # output behaves as REAL probability for this model.
        #
        # Therefore:
        #
        # real_probability = sigmoid(logit)
        # ai_probability   = 1 - real_probability

        real_probability_tensor = torch.sigmoid(logit)

        real_probability_tensor = (
            real_probability_tensor
            .reshape(-1)
        )

        real_probability_list = (
            real_probability_tensor
            .cpu()
            .tolist()
        )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    for dataset_label, real_probability in zip(
        labels,
        real_probability_list
    ):

        # CIFAKE labels:
        #
        # 0 = FAKE / AI-GENERATED
        # 1 = REAL
        #
        # Our evaluation labels:
        #
        # 0 = REAL
        # 1 = AI-GENERATED

        if dataset_label == 0:

            true_label = 1

        else:

            true_label = 0


        # Convert model output

        ai_probability = (
            1.0 - real_probability
        )


        # Prediction

        if ai_probability >= 0.5:

            predicted_label = 1

        else:

            predicted_label = 0


        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

        ai_probabilities.append(
            float(ai_probability)
        )

        real_probabilities.append(
            float(real_probability)
        )


    print(
        f"Processed {end:,} / {total_images:,} images"
    )


# ============================================================
# MANUAL METRIC CALCULATION
# ============================================================

true_negative = 0
false_positive = 0
false_negative = 0
true_positive = 0


for actual, predicted in zip(
    true_labels,
    predicted_labels
):

    if actual == 0 and predicted == 0:

        true_negative += 1

    elif actual == 0 and predicted == 1:

        false_positive += 1

    elif actual == 1 and predicted == 0:

        false_negative += 1

    elif actual == 1 and predicted == 1:

        true_positive += 1


correct = (
    true_negative
    + true_positive
)

incorrect = (
    false_positive
    + false_negative
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = (
    correct / total_images
)


# ============================================================
# PRECISION
# ============================================================

precision_denominator = (
    true_positive
    + false_positive
)

if precision_denominator > 0:

    precision = (
        true_positive
        / precision_denominator
    )

else:

    precision = 0.0


# ============================================================
# RECALL
# ============================================================

recall_denominator = (
    true_positive
    + false_negative
)

if recall_denominator > 0:

    recall = (
        true_positive
        / recall_denominator
    )

else:

    recall = 0.0


# ============================================================
# F1 SCORE
# ============================================================

f1_denominator = (
    precision
    + recall
)

if f1_denominator > 0:

    f1 = (
        2
        * precision
        * recall
        / f1_denominator
    )

else:

    f1 = 0.0


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print()

print(
    f"Dataset      : CIFAKE"
)

print(
    f"Split        : Test"
)

print(
    f"Total images : {total_images:,}"
)

print(
    f"Correct      : {correct:,}"
)

print(
    f"Incorrect    : {incorrect:,}"
)

print()

print(
    f"Accuracy     : {accuracy * 100:.2f}%"
)

print(
    f"Precision    : {precision * 100:.2f}%"
)

print(
    f"Recall       : {recall * 100:.2f}%"
)

print(
    f"F1-Score     : {f1 * 100:.2f}%"
)

print()

print("Confusion Matrix")
print()

print(
    "                    Predicted"
)

print(
    "                 REAL       AI"
)

print(
    f"Actual REAL    {true_negative:6d}  "
    f"{false_positive:6d}"
)

print(
    f"Actual AI      {false_negative:6d}  "
    f"{true_positive:6d}"
)

print()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "dataset": "CIFAKE",

    "split": "test",

    "total_images": total_images,

    "correct": correct,

    "incorrect": incorrect,

    "accuracy": round(
        accuracy,
        6
    ),

    "accuracy_percent": round(
        accuracy * 100,
        2
    ),

    "precision": round(
        precision,
        6
    ),

    "precision_percent": round(
        precision * 100,
        2
    ),

    "recall": round(
        recall,
        6
    ),

    "recall_percent": round(
        recall * 100,
        2
    ),

    "f1_score": round(
        f1,
        6
    ),

    "f1_score_percent": round(
        f1 * 100,
        2
    ),

    "true_negative": true_negative,

    "false_positive": false_positive,

    "false_negative": false_negative,

    "true_positive": true_positive

}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)


with open(
    cm_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "",
        "Predicted REAL",
        "Predicted AI"
    ])

    writer.writerow([
        "Actual REAL",
        true_negative,
        false_positive
    ])

    writer.writerow([
        "Actual AI",
        false_negative,
        true_positive
    ])


# ============================================================
# SAVE IMAGE-LEVEL PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)


with open(
    predictions_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "true_label",
        "predicted_label",
        "true_class",
        "predicted_class",
        "ai_probability",
        "real_probability"
    ])


    for actual, predicted, ai_prob, real_prob in zip(
        true_labels,
        predicted_labels,
        ai_probabilities,
        real_probabilities
    ):

        if actual == 1:

            true_class = "AI-GENERATED"

        else:

            true_class = "REAL"


        if predicted == 1:

            predicted_class = "AI-GENERATED"

        else:

            predicted_class = "REAL"


        writer.writerow([
            actual,
            predicted,
            true_class,
            predicted_class,
            round(ai_prob, 6),
            round(real_prob, 6)
        ])


# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print()

print(
    f"Metrics saved to: "
    f"{metrics_path}"
)

print(
    f"Confusion matrix saved to: "
    f"{cm_path}"
)

print(
    f"Predictions saved to: "
    f"{predictions_path}"
)

print()

print(
    "AIDetect dataset evaluation finished."
)