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
# CONFIGURATION
# ============================================================

DATASET_REPO = "dragonintelligence/CIFAKE-image-dataset"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hybrid_resnet50_fft.pth"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "hybrid_training_metrics.json"
)

TRAIN_SAMPLES = 2000

BATCH_SIZE = 16

EPOCHS = 1

LEARNING_RATE = 1e-4

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("AIDetect - Hybrid Spatial + Frequency Training")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD CIFAKE
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
    f"{TRAIN_SAMPLES // 2:,}"
)

print(
    f"Real images: "
    f"{TRAIN_SAMPLES // 2:,}"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

weights = ResNet50_Weights.DEFAULT

spatial_transform = weights.transforms()


# ============================================================
# FFT
# ============================================================

def fft_magnitude(image_tensor):
    """
    Convert RGB image to FFT magnitude representation.

    Input:
        [3, 224, 224]

    Output:
        [3, 224, 224]
    """

    x = image_tensor.float()

    fft = torch.fft.fft2(x)

    fft = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft)

    magnitude = torch.log1p(magnitude)

    # Normalize each channel
    for c in range(magnitude.shape[0]):

        channel = magnitude[c]

        min_value = channel.min()
        max_value = channel.max()

        magnitude[c] = (
            (channel - min_value)
            /
            (
                max_value
                -
                min_value
                +
                1e-8
            )
        )

    return magnitude


# ============================================================
# DATASET
# ============================================================

class HybridDataset(torch.utils.data.Dataset):

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

        # ----------------------------------------------------
        # Spatial representation
        # ----------------------------------------------------

        spatial_image = spatial_transform(
            image
        )

        # spatial_transform already gives
        # [3,224,224]

        # ----------------------------------------------------
        # Frequency representation
        # ----------------------------------------------------

        frequency_image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        frequency_image = torch.from_numpy(
            np.array(frequency_image)
        ).permute(
            2, 0, 1
        ).float() / 255.0

        frequency_image = fft_magnitude(
            frequency_image
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

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
            spatial_image,
            frequency_image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


train_data = HybridDataset(
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
# HYBRID MODEL
# ============================================================

class HybridDetector(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # SPATIAL BRANCH
        # ----------------------------------------------------

        print(
            "\nLoading pretrained ResNet50..."
        )

        spatial_model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        # Remove final classifier
        self.spatial = nn.Sequential(
            *list(spatial_model.children())[:-1]
        )

        # ResNet50 feature size
        self.spatial_features = 2048

        # ----------------------------------------------------
        # FREQUENCY BRANCH
        # ----------------------------------------------------

        self.frequency = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.frequency_features = 256

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion_size = (
            self.spatial_features
            +
            self.frequency_features
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_size,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                2
            )
        )


    def forward(
        self,
        spatial_image,
        frequency_image
    ):

        # Spatial features
        spatial_features = self.spatial(
            spatial_image
        )

        spatial_features = spatial_features.flatten(
            1
        )

        # Frequency features
        frequency_features = self.frequency(
            frequency_image
        )

        frequency_features = frequency_features.flatten(
            1
        )

        # Feature fusion
        fused_features = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )

        # Classification
        output = self.classifier(
            fused_features
        )

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = HybridDetector()

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
# TRAINING
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

    for (
        spatial_images,
        frequency_images,
        labels
    ) in progress:

        spatial_images = spatial_images.to(
            DEVICE
        )

        frequency_images = frequency_images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(
            spatial_images,
            frequency_images
        )

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        # Statistics
        running_loss += (
            loss.item()
            *
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
        "Hybrid Spatial + Frequency",

    "spatial_branch":
        "Pretrained ResNet50",

    "frequency_branch":
        "FFT magnitude + CNN",

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
print("HYBRID TRAINING COMPLETE")
print("=" * 60)

print(
    "\nModel saved to:"
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
    "Evaluate the hybrid model "
    "on the untouched CIFAKE TEST split."
)