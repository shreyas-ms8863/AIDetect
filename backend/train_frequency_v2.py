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
# CONFIG
# ============================================================

TRAIN_SAMPLES = 20000
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "frequency_resnet50_v2.pth"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "frequency_v2_training_metrics.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("FREQUENCY V2 TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD CIFAKE TRAINING DATA
# ============================================================

print("\nLoading CIFAKE training dataset...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="train"
)

print(f"Total training images available: {len(dataset)}")


# ============================================================
# SELECT BALANCED DATASET
# 10,000 AI + 10,000 REAL
# ============================================================

fake_indices = []
real_indices = []

for i in range(len(dataset)):

    label = int(dataset[i]["label"])

    # CIFAKE:
    # 0 = Fake / AI
    # 1 = Real

    if label == 0 and len(fake_indices) < TRAIN_SAMPLES // 2:
        fake_indices.append(i)

    elif label == 1 and len(real_indices) < TRAIN_SAMPLES // 2:
        real_indices.append(i)

    if (
        len(fake_indices) == TRAIN_SAMPLES // 2
        and len(real_indices) == TRAIN_SAMPLES // 2
    ):
        break


selected_indices = fake_indices + real_indices

print(f"AI images selected: {len(fake_indices)}")
print(f"Real images selected: {len(real_indices)}")
print(f"Total selected: {len(selected_indices)}")


# ============================================================
# FFT PREPROCESSING
# ============================================================

weights = ResNet50_Weights.DEFAULT

image_transform = weights.transforms()


def frequency_transform(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_tensor = torch.from_numpy(
        __import__("numpy").array(image)
    ).float()

    image_tensor = image_tensor.permute(2, 0, 1) / 255.0

    # --------------------------------------------------------
    # FFT for each RGB channel
    # --------------------------------------------------------

    fft = torch.fft.fft2(image_tensor)

    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)

    magnitude = torch.log1p(magnitude)

    # --------------------------------------------------------
    # Channel-wise normalization
    # --------------------------------------------------------

    for c in range(3):

        channel = magnitude[c]

        min_val = channel.min()
        max_val = channel.max()

        magnitude[c] = (
            channel - min_val
        ) / (
            max_val - min_val + 1e-8
        )

    # --------------------------------------------------------
    # Convert to ImageNet-normalized tensor
    # --------------------------------------------------------

    magnitude = image_transform(
        Image.fromarray(
            (magnitude.permute(1, 2, 0).numpy() * 255)
            .clip(0, 255)
            .astype("uint8")
        )
    )

    return magnitude


# ============================================================
# DATASET CLASS
# ============================================================

class FrequencyDataset(torch.utils.data.Dataset):

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

        image = frequency_transform(image)

        original_label = int(item["label"])

        # Convert:
        # CIFAKE 0 = AI
        # CIFAKE 1 = Real
        #
        # Our labels:
        # 0 = Real
        # 1 = AI

        if original_label == 0:
            label = 1
        else:
            label = 0

        return image, label


train_dataset = FrequencyDataset(
    dataset,
    selected_indices
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print(f"Training batches per epoch: {len(train_loader)}")


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet50...")

model = resnet50(
    weights=ResNet50_Weights.DEFAULT
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
# MIXED PRECISION
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# TRAINING
# ============================================================

training_metrics = []

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

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        accuracy = correct / total

        progress.set_postfix(
            accuracy=f"{accuracy * 100:.2f}%",
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    print(
        f"\nEpoch {epoch + 1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={epoch_accuracy * 100:.2f}%"
    )

    training_metrics.append({
        "epoch": epoch + 1,
        "loss": float(epoch_loss),
        "accuracy": float(epoch_accuracy)
    })


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        training_metrics,
        f,
        indent=4
    )


print("\n" + "=" * 60)
print("FREQUENCY V2 TRAINING COMPLETE")
print("=" * 60)

print(f"\nModel saved to:")
print(MODEL_PATH)

print("\nTraining metrics saved to:")
print(METRICS_PATH)