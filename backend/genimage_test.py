import os
import json
import csv

import torch
import torchvision.transforms as T

from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_REPO = "TheKernel01/Tiny-GenImage"

VALIDATION_FILE = (
    "data/validation-00000-of-00004.parquet"
)

SAMPLE_PER_CLASS = 50

OUTPUT_DIR = "genimage_results"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 65)
print("AIDetect Cross-Dataset Evaluation")
print("=" * 65)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading AIDetect model...")

MODEL_REPO = (
    "Medsa/ai-image-authenticity-detector"
)

MODEL_FILE = "detector_scripted.pt"


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
print()


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
# DOWNLOAD ONLY ONE VALIDATION SHARD
# ============================================================

print("Downloading ONE validation shard...")
print()

parquet_path = hf_hub_download(

    repo_id=DATASET_REPO,

    filename=VALIDATION_FILE,

    repo_type="dataset"

)

print()
print(
    "Validation shard downloaded:"
)

print(parquet_path)
print()


# ============================================================
# LOAD ONLY THAT PARQUET FILE
# ============================================================

print("Reading validation data...")

dataset = load_dataset(

    "parquet",

    data_files=parquet_path,

    split="train"

)


print(
    f"Images available in shard: "
    f"{len(dataset):,}"
)

print()


# ============================================================
# SHOW LABEL INFORMATION
# ============================================================

print("Dataset labels:")
print("0 = REAL")
print("1 = AI-GENERATED")
print()


# ============================================================
# COLLECT BALANCED SAMPLE
# ============================================================

real_indices = []

ai_indices = []


print("Searching for balanced samples...")
print()


for index in range(
    len(dataset)
):

    label = dataset[index]["label"]


    if label == 0:

        if len(real_indices) < SAMPLE_PER_CLASS:

            real_indices.append(index)


    elif label == 1:

        if len(ai_indices) < SAMPLE_PER_CLASS:

            ai_indices.append(index)


    if (
        len(real_indices) >= SAMPLE_PER_CLASS
        and
        len(ai_indices) >= SAMPLE_PER_CLASS
    ):

        break


print(
    f"REAL samples found : "
    f"{len(real_indices)}"
)

print(
    f"AI samples found   : "
    f"{len(ai_indices)}"
)

print()


# ============================================================
# CHECK BALANCE
# ============================================================

if len(real_indices) < SAMPLE_PER_CLASS:

    print(
        "ERROR: This validation shard does not contain "
        "enough REAL images."
    )

    print(
        "We will use another shard if necessary."
    )

    raise SystemExit


if len(ai_indices) < SAMPLE_PER_CLASS:

    print(
        "ERROR: This validation shard does not contain "
        "enough AI images."
    )

    print(
        "We will use another shard if necessary."
    )

    raise SystemExit


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(image):

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


    tensor = transform(
        image
    ).unsqueeze(0)


    with torch.no_grad():

        output = model(
            tensor
        )

        logit = output[0]


        # IMPORTANT
        #
        # We previously verified using
        # CIFAKE that this downloaded model
        # behaves as:
        #
        # sigmoid(logit) = REAL probability
        #

        real_probability = (

            torch.sigmoid(
                logit
            )
            .reshape(-1)[0]
            .item()

        )


    ai_probability = (

        1.0
        -
        real_probability

    )


    if ai_probability >= 0.5:

        prediction = (
            "AI-GENERATED"
        )

    else:

        prediction = "REAL"


    return (
        prediction,
        ai_probability,
        real_probability
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


selected_indices = (

    real_indices
    +
    ai_indices

)


print(
    "Running model inference..."
)

print()


for count, index in enumerate(

    selected_indices,

    start=1

):

    item = dataset[index]


    label = item["label"]

    generator = item["generator"]

    image = item["image"]


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if label == 0:

        ground_truth = "REAL"

    else:

        ground_truth = "AI-GENERATED"


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    (
        prediction,
        ai_probability,
        real_probability
    ) = predict(
        image
    )


    # --------------------------------------------------------
    # Generator name
    # --------------------------------------------------------

    try:

        generator_name = (

            dataset.features[
                "generator"
            ]
            .int2str(
                generator
            )

        )

    except Exception:

        generator_name = str(
            generator
        )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({

        "index":
            index,

        "ground_truth":
            ground_truth,

        "generator":
            generator_name,

        "prediction":
            prediction,

        "ai_probability":
            round(
                ai_probability * 100,
                2
            ),

        "real_probability":
            round(
                real_probability * 100,
                2
            )

    })


    if count % 10 == 0:

        print(
            f"Processed "
            f"{count} / "
            f"{len(selected_indices)}"
        )


# ============================================================
# CALCULATE CONFUSION MATRIX
# ============================================================

total = len(results)

correct = 0

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


for result in results:

    actual = result[
        "ground_truth"
    ]

    predicted = result[
        "prediction"
    ]


    if actual == predicted:

        correct += 1


    # AI correctly detected
    if (
        actual == "AI-GENERATED"
        and
        predicted == "AI-GENERATED"
    ):

        true_positive += 1


    # REAL correctly detected
    elif (
        actual == "REAL"
        and
        predicted == "REAL"
    ):

        true_negative += 1


    # REAL incorrectly detected as AI
    elif (
        actual == "REAL"
        and
        predicted == "AI-GENERATED"
    ):

        false_positive += 1


    # AI incorrectly detected as REAL
    elif (
        actual == "AI-GENERATED"
        and
        predicted == "REAL"
    ):

        false_negative += 1


# ============================================================
# METRICS
# ============================================================

accuracy = (

    correct
    /
    total

)


precision_denominator = (

    true_positive
    +
    false_positive

)


if precision_denominator > 0:

    precision = (

        true_positive
        /
        precision_denominator

    )

else:

    precision = 0


recall_denominator = (

    true_positive
    +
    false_negative

)


if recall_denominator > 0:

    recall = (

        true_positive
        /
        recall_denominator

    )

else:

    recall = 0


if (
    precision
    +
    recall
) > 0:

    f1 = (

        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )

    )

else:

    f1 = 0


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print("=" * 65)
print("GENIMAGE CROSS-DATASET RESULTS")
print("=" * 65)

print()

print(
    "Dataset       : Tiny-GenImage"
)

print(
    "Split         : Validation"
)

print(
    f"Images tested : {total}"
)

print(
    f"Correct       : {correct}"
)

print(
    f"Incorrect     : "
    f"{total - correct}"
)

print()

print(
    f"Accuracy      : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision     : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall        : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score      : "
    f"{f1 * 100:.2f}%"
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
    f"Actual REAL    "
    f"{true_negative:6d}  "
    f"{false_positive:6d}"
)

print(
    f"Actual AI      "
    f"{false_negative:6d}  "
    f"{true_positive:6d}"
)

print()


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(

    OUTPUT_DIR,

    "genimage_predictions.csv"

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

            "index",

            "ground_truth",

            "generator",

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

    "dataset":
        "Tiny-GenImage",

    "split":
        "validation",

    "total_images":
        total,

    "sample_per_class":
        SAMPLE_PER_CLASS,

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
        true_negative,

    "false_positive":
        false_positive,

    "false_negative":
        false_negative,

    "true_positive":
        true_positive

}


json_path = os.path.join(

    OUTPUT_DIR,

    "genimage_metrics.json"

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
# COMPLETE
# ============================================================

print("=" * 65)

print(
    "GENIMAGE TEST COMPLETE"
)

print("=" * 65)

print()

print(
    "Predictions saved to:"
)

print(
    csv_path
)

print()

print(
    "Metrics saved to:"
)

print(
    json_path
)

print()