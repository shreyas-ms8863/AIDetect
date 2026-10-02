import os
import csv
import json

import torch
import torchvision.transforms as T

from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO = "Medsa/ai-image-authenticity-detector"
MODEL_FILE = "detector_scripted.pt"

TEST_FOLDER = "real_world_test"

AI_FOLDER = os.path.join(
    TEST_FOLDER,
    "AI"
)

REAL_FOLDER = os.path.join(
    TEST_FOLDER,
    "REAL"
)

OUTPUT_FOLDER = "real_world_results"


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 65)
print("AIDetect Real-World Generalization Test")
print("=" * 65)
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
# GET IMAGE FILES
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)


def get_images(folder):

    if not os.path.exists(folder):

        return []

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            valid_extensions
        ):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return sorted(files)


ai_images = get_images(
    AI_FOLDER
)

real_images = get_images(
    REAL_FOLDER
)


print()
print(
    f"AI images found   : {len(ai_images)}"
)

print(
    f"REAL images found : {len(real_images)}"
)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if len(ai_images) == 0:

    print(
        "ERROR: No images found in "
        "real_world_test/AI"
    )

    print(
        "Add confirmed AI-generated images "
        "and run the script again."
    )

    raise SystemExit


if len(real_images) == 0:

    print(
        "ERROR: No images found in "
        "real_world_test/REAL"
    )

    print(
        "Add confirmed real photographs "
        "and run the script again."
    )

    raise SystemExit


# ============================================================
# MODEL INFERENCE
# ============================================================

def analyze_image(
    image_path
):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = transform(
            image
        ).unsqueeze(0)


        with torch.no_grad():

            output = model(
                tensor
            )

            logit = output[0]

            # Verified model polarity:
            #
            # sigmoid(logit) = REAL probability

            real_probability = (
                torch.sigmoid(
                    logit
                )
                .reshape(-1)[0]
                .item()
            )


        ai_probability = (
            1.0
            - real_probability
        )


        if ai_probability >= 0.5:

            prediction = "AI-GENERATED"

        else:

            prediction = "REAL"


        return {

            "prediction": prediction,

            "ai_probability":
                ai_probability * 100,

            "real_probability":
                real_probability * 100

        }


    except Exception as error:

        print(
            f"ERROR processing {image_path}: "
            f"{error}"
        )

        return None


# ============================================================
# RUN TEST
# ============================================================

results = []


print(
    "Running real-world inference..."
)

print()


# ------------------------------------------------------------
# AI IMAGES
# ------------------------------------------------------------

for image_path in ai_images:

    result = analyze_image(
        image_path
    )

    if result is None:

        continue


    results.append({

        "filename":
            os.path.basename(
                image_path
            ),

        "ground_truth":
            "AI-GENERATED",

        "prediction":
            result["prediction"],

        "ai_probability":
            round(
                result["ai_probability"],
                2
            ),

        "real_probability":
            round(
                result["real_probability"],
                2
            )

    })


# ------------------------------------------------------------
# REAL IMAGES
# ------------------------------------------------------------

for image_path in real_images:

    result = analyze_image(
        image_path
    )

    if result is None:

        continue


    results.append({

        "filename":
            os.path.basename(
                image_path
            ),

        "ground_truth":
            "REAL",

        "prediction":
            result["prediction"],

        "ai_probability":
            round(
                result["ai_probability"],
                2
            ),

        "real_probability":
            round(
                result["real_probability"],
                2
            )

    })


# ============================================================
# CALCULATE RESULTS
# ============================================================

total = len(results)

correct = 0

false_positives = 0
false_negatives = 0

true_positives = 0
true_negatives = 0


for result in results:

    actual = result[
        "ground_truth"
    ]

    predicted = result[
        "prediction"
    ]


    if actual == predicted:

        correct += 1


    if (
        actual == "REAL"
        and
        predicted == "AI-GENERATED"
    ):

        false_positives += 1


    elif (
        actual == "AI-GENERATED"
        and
        predicted == "REAL"
    ):

        false_negatives += 1


    elif (
        actual == "AI-GENERATED"
        and
        predicted == "AI-GENERATED"
    ):

        true_positives += 1


    elif (
        actual == "REAL"
        and
        predicted == "REAL"
    ):

        true_negatives += 1


# ============================================================
# METRICS
# ============================================================

if total > 0:

    accuracy = (
        correct / total
    )

else:

    accuracy = 0


precision_denominator = (
    true_positives
    + false_positives
)

if precision_denominator > 0:

    precision = (
        true_positives
        / precision_denominator
    )

else:

    precision = 0


recall_denominator = (
    true_positives
    + false_negatives
)

if recall_denominator > 0:

    recall = (
        true_positives
        / recall_denominator
    )

else:

    recall = 0


if (
    precision
    + recall
) > 0:

    f1 = (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )
    )

else:

    f1 = 0


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 65)
print("REAL-WORLD TEST RESULTS")
print("=" * 65)

print()

print(
    f"Total images : {total}"
)

print(
    f"Correct      : {correct}"
)

print(
    f"Incorrect    : {total - correct}"
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
    f"Actual REAL    {true_negatives:6d}  "
    f"{false_positives:6d}"
)

print(
    f"Actual AI      {false_negatives:6d}  "
    f"{true_positives:6d}"
)

print()


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "real_world_predictions.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "ground_truth",
            "prediction",
            "ai_probability",
            "real_probability"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "test_type":
        "Real-world generalization",

    "total_images":
        total,

    "correct":
        correct,

    "incorrect":
        total - correct,

    "accuracy_percent":
        round(
            accuracy * 100,
            2
        ),

    "precision_percent":
        round(
            precision * 100,
            2
        ),

    "recall_percent":
        round(
            recall * 100,
            2
        ),

    "f1_score_percent":
        round(
            f1 * 100,
            2
        ),

    "true_negative":
        true_negatives,

    "false_positive":
        false_positives,

    "false_negative":
        false_negatives,

    "true_positive":
        true_positives
}


json_path = os.path.join(
    OUTPUT_FOLDER,
    "real_world_metrics.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# FINISH
# ============================================================

print("=" * 65)
print("REAL-WORLD TEST COMPLETE")
print("=" * 65)

print()

print(
    f"Predictions saved to: {csv_path}"
)

print(
    f"Metrics saved to: {json_path}"
)

print()