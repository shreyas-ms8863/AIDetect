import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_REPO = "dragonintelligence/CIFAKE-image-dataset"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "frequency_resnet50.pth"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "frequency_training_metrics.json"
)

# Keep identical to Spatial-only experiment
TRAIN_SAMPLES = 2000

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("AIDetect - Frequency-only FFT + ResNet50 Training")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD CIFAKE TRAIN
# ============================================================

print("\nLoading CIFAKE train dataset...")

dataset = load_dataset(
    DATASET_REPO,
    split="train"
)

print(
    f"Total training images available: "
    f"{len(dataset):,}"
)


# ============================================================
# BALANCED SUBSET
#
# CIFAKE:
# 0 = AI
# 1 = REAL
# ============================================================

fake_indices = []
real_indices = []

for i, item in enumerate(dataset):

    label = item["label"]

    if label == 0:
        fake_indices.append(i)

    elif label == 1:
        real_indices.append(i)

    if (
        len(fake_indices) >= TRAIN_SAMPLES // 2
        and
        len(real_indices) >= TRAIN_SAMPLES // 2
    ):
        break


selected_indices = (
    fake_indices[:TRAIN_SAMPLES // 2]
    +
    real_indices[:TRAIN_SAMPLES // 2]
)

train_dataset = dataset.select(
    selected_indices
)

print(
    f"Selected training images: "
    f"{len(train_dataset):,}"
)

print(
    f"AI images: "
    f"{len(fake_indices[:TRAIN_SAMPLES // 2]):,}"
)

print(
    f"Real images: "
    f"{len(real_indices[:TRAIN_SAMPLES // 2]):,}"
)


# ============================================================
# FREQUENCY TRANSFORMATION
# ============================================================

def fft_magnitude(image_tensor):
    """
    Convert an RGB image into an FFT magnitude representation.

    Input:
        [3, H, W]

    Output:
        [3, H, W]

    Each RGB channel is transformed independently.
    """

    # Convert image from [0,1] to floating point
    x = image_tensor.float()

    # 2D FFT for each RGB channel
    fft = torch.fft.fft2(x)

    # Move low frequencies to the center
    fft = torch.fft.fftshift(fft)

    # Magnitude spectrum
    magnitude = torch.abs(fft)

    # Log scaling makes the frequency information
    # easier for the neural network to learn.
    magnitude = torch.log1p(magnitude)

    # Normalize each channel independently
    for c in range(magnitude.shape[0]):

        channel = magnitude[c]

        min_value = channel.min()
        max_value = channel.max()

        magnitude[c] = (
            (channel - min_value)
            /
            (max_value - min_value + 1e-8)
        )

    return magnitude


# ============================================================
# DATASET WRAPPER
# ============================================================

class FrequencyDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset):

        self.dataset = hf_dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        if not isinstance(image, Image.Image):

            image = Image.fromarray(image)

        image = image.convert("RGB")

        # Resize to a fixed size before FFT
        image = image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        # Convert to tensor [C,H,W]
        image_tensor = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0

        # FFT representation
        image_tensor = fft_magnitude(
            image_tensor
        )

        # CIFAKE:
        # 0 = AI
        # 1 = REAL
        #
        # Internal:
        # 0 = REAL
        # 1 = AI

        if item["label"] == 0:

            label = 1

        else:

            label = 0

        return (
            image_tensor,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


train_data = FrequencyDataset(
    train_dataset
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{100 * correct / total:.2f}%"
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    print(
        f"\nEpoch {epoch + 1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy="
        f"{epoch_accuracy * 100:.2f}%"
    )

    history.append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "accuracy": epoch_accuracy
    })


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model":
        "Frequency-only FFT + ResNet50",

    "frequency_representation":
        "2D FFT magnitude with log scaling",

    "dataset":
        DATASET_REPO,

    "split":
        "train",

    "training_samples":
        len(train_data),

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "device":
        str(DEVICE),

    "history":
        history
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FREQUENCY TRAINING COMPLETE")
print("=" * 60)

print(
    "Model saved to:"
)

print(MODEL_PATH)

print(
    "\nMetrics saved to:"
)

print(METRICS_PATH)

print(
    "\nNext step:"
)

print(
    "Evaluate the frequency-only model "
    "on the untouched CIFAKE TEST split."
)