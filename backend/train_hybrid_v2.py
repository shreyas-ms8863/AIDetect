import os
import json
import numpy as np
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
    "hybrid_resnet50_fft_v2.pth"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "hybrid_v2_training_metrics.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("HYBRID V2 TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD CIFAKE
# ============================================================

print("\nLoading CIFAKE training dataset...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="train"
)

print(f"Total training images available: {len(dataset)}")


# ============================================================
# BALANCED 20,000 IMAGE SUBSET
# 10,000 AI + 10,000 REAL
# ============================================================

fake_indices = []
real_indices = []

for i in range(len(dataset)):

    label = int(dataset[i]["label"])

    # CIFAKE:
    # 0 = AI / Fake
    # 1 = Real

    if label == 0 and len(fake_indices) < 10000:
        fake_indices.append(i)

    elif label == 1 and len(real_indices) < 10000:
        real_indices.append(i)

    if len(fake_indices) == 10000 and len(real_indices) == 10000:
        break

selected_indices = fake_indices + real_indices

print(f"AI images selected:   {len(fake_indices)}")
print(f"Real images selected: {len(real_indices)}")
print(f"Total selected:       {len(selected_indices)}")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

weights = ResNet50_Weights.DEFAULT
image_transform = weights.transforms()


# ============================================================
# FREQUENCY TRANSFORM
# Same FFT representation as Frequency V2
# ============================================================

def frequency_transform(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_tensor = torch.from_numpy(
        np.array(image)
    ).float()

    image_tensor = image_tensor.permute(2, 0, 1) / 255.0

    # FFT
    fft = torch.fft.fft2(image_tensor)
    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)

    # Log magnitude
    magnitude = torch.log1p(magnitude)

    # Channel-wise normalization
    for c in range(3):

        channel = magnitude[c]

        min_val = channel.min()
        max_val = channel.max()

        magnitude[c] = (
            channel - min_val
        ) / (
            max_val - min_val + 1e-8
        )

    # Convert frequency representation to image
    magnitude_image = Image.fromarray(
        (
            magnitude.permute(1, 2, 0).numpy() * 255
        )
        .clip(0, 255)
        .astype("uint8")
    )

    return image_transform(magnitude_image)


# ============================================================
# DATASET
# ============================================================

class HybridDataset(torch.utils.data.Dataset):

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

        image = image.convert("RGB")

        # Spatial representation
        spatial_image = image_transform(image)

        # Frequency representation
        frequency_image = frequency_transform(image)

        original_label = int(item["label"])

        # Convert:
        # CIFAKE 0 = AI
        # CIFAKE 1 = Real
        #
        # Internal:
        # 0 = Real
        # 1 = AI

        if original_label == 0:
            label = 1
        else:
            label = 0

        return spatial_image, frequency_image, label


train_dataset = HybridDataset(
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
# HYBRID MODEL
# ============================================================

class HybridResNetFFT(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        spatial_model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        self.spatial_features = nn.Sequential(
            *list(spatial_model.children())[:-1]
        )

        self.spatial_dim = 2048

        # ----------------------------------------------------
        # Frequency branch
        # ----------------------------------------------------

        self.frequency_features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.frequency_dim = 256

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        combined_dim = (
            self.spatial_dim +
            self.frequency_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_dim,
                512
            ),

            nn.ReLU(inplace=True),

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

        # Spatial features
        spatial = self.spatial_features(
            spatial_input
        )

        spatial = torch.flatten(
            spatial,
            1
        )

        # Frequency features
        frequency = self.frequency_features(
            frequency_input
        )

        frequency = torch.flatten(
            frequency,
            1
        )

        # Feature fusion
        combined = torch.cat(
            [
                spatial,
                frequency
            ],
            dim=1
        )

        output = self.classifier(
            combined
        )

        return output


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating Hybrid model...")

model = HybridResNetFFT()
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

    for (
        spatial_images,
        frequency_images,
        labels
    ) in progress:

        spatial_images = spatial_images.to(
            DEVICE,
            non_blocking=True
        )

        frequency_images = frequency_images.to(
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

            outputs = model(
                spatial_images,
                frequency_images
            )

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() *
            labels.size(0)
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
            accuracy=f"{correct / total * 100:.2f}%",
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


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("HYBRID V2 TRAINING COMPLETE")
print("=" * 60)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nTraining metrics saved to:")
print(METRICS_PATH)