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

SAMPLES_PER_CLASS = 500


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("AIDetect Model Polarity Diagnostic")
print("=" * 60)
print()

print("Loading model...")

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
# LOAD DATASET
# ============================================================

print()
print("Loading CIFAKE test dataset...")

dataset = load_dataset(
    DATASET_REPO,
    split="test"
)

print(
    f"Dataset loaded: {len(dataset)} images"
)

print()


# ============================================================
# COLLECT SAMPLES FROM BOTH CLASSES
# ============================================================

fake_images = []
real_images = []

print("Collecting samples...")


for index in range(len(dataset)):

    label = dataset[index]["label"]

    if label == 0:

        if len(fake_images) < SAMPLES_PER_CLASS:

            fake_images.append(
                dataset[index]["image"]
            )

    elif label == 1:

        if len(real_images) < SAMPLES_PER_CLASS:

            real_images.append(
                dataset[index]["image"]
            )


    if (
        len(fake_images) >= SAMPLES_PER_CLASS
        and
        len(real_images) >= SAMPLES_PER_CLASS
    ):

        break


print(
    f"FAKE samples collected : {len(fake_images)}"
)

print(
    f"REAL samples collected : {len(real_images)}"
)

print()


# ============================================================
# FUNCTION TO GET MODEL PROBABILITY
# ============================================================

def get_probability(image):

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
    )

    tensor = tensor.unsqueeze(
        0
    )


    with torch.no_grad():

        output = model(
            tensor
        )

        logit = output[0]

        probability = torch.sigmoid(
            logit
        ).item()


    return probability


# ============================================================
# RUN FAKE SAMPLES
# ============================================================

print("Analyzing FAKE images...")

fake_probabilities = []


for image in fake_images:

    probability = get_probability(
        image
    )

    fake_probabilities.append(
        probability
    )


# ============================================================
# RUN REAL SAMPLES
# ============================================================

print("Analyzing REAL images...")

real_probabilities = []


for image in real_images:

    probability = get_probability(
        image
    )

    real_probabilities.append(
        probability
    )


# ============================================================
# AVERAGES
# ============================================================

fake_average = (
    sum(fake_probabilities)
    / len(fake_probabilities)
)

real_average = (
    sum(real_probabilities)
    / len(real_probabilities)
)


# ============================================================
# ACCURACY — INTERPRETATION 1
#
# Model card interpretation:
#
# probability >= 0.5
#       -> FAKE / AI
#
# probability < 0.5
#       -> REAL
# ============================================================

card_correct = 0

for probability in fake_probabilities:

    if probability >= 0.5:

        card_correct += 1


for probability in real_probabilities:

    if probability < 0.5:

        card_correct += 1


card_total = (
    len(fake_probabilities)
    + len(real_probabilities)
)

card_accuracy = (
    card_correct
    / card_total
)


# ============================================================
# ACCURACY — INTERPRETATION 2
#
# Reversed interpretation:
#
# probability >= 0.5
#       -> REAL
#
# probability < 0.5
#       -> FAKE / AI
# ============================================================

reverse_correct = 0


for probability in fake_probabilities:

    if probability < 0.5:

        reverse_correct += 1


for probability in real_probabilities:

    if probability >= 0.5:

        reverse_correct += 1


reverse_accuracy = (
    reverse_correct
    / card_total
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("DIAGNOSTIC RESULTS")
print("=" * 60)

print()

print(
    f"Average probability for FAKE images : "
    f"{fake_average * 100:.2f}%"
)

print(
    f"Average probability for REAL images : "
    f"{real_average * 100:.2f}%"
)

print()

print(
    "Interpretation 1:"
)

print(
    "Probability >= 50% = AI-GENERATED"
)

print(
    f"Accuracy = {card_accuracy * 100:.2f}%"
)

print()

print(
    "Interpretation 2:"
)

print(
    "Probability >= 50% = REAL"
)

print(
    f"Accuracy = {reverse_accuracy * 100:.2f}%"
)

print()


# ============================================================
# FINAL CONCLUSION
# ============================================================

if reverse_accuracy > card_accuracy:

    print(
        "RESULT: The model output appears to represent "
        "REAL probability."
    )

    print(
        "The prediction polarity should be reversed."
    )

else:

    print(
        "RESULT: The model output appears to represent "
        "FAKE probability."
    )

    print(
        "The original polarity should be retained."
    )


print()

print("=" * 60)
print("Diagnostic complete.")
print("=" * 60)
