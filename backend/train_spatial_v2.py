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

DATASET_REPO = "dragonintelligence/CIFAKE-image-dataset"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "spatial_resnet50_v2.pth"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "spatial_v2_training_metrics.json"
)

TRAIN_SAMPLES = 20000

BATCH_SIZE = 16

EPOCHS = 3

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 65)
print("AIDetect - Spatial-only ResNet50 V2 Training")
print("=" * 65)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD CIFAKE TRAIN
# ============================================================

print("\nLoading CIFAKE TRAIN dataset...")

dataset = load_dataset(
    DATASET_REPO,
    split="train"
)

print(
    f"Total training images available: "
    f"{len(dataset):,}"
)


# ============================================================
# COLLECT BALANCED INDICES
# ============================================================

print("\nCollecting balanced training samples...")

fake_indices = []
real_indices = []

target_per_class = TRAIN_SAMPLES // 2

for i in range(len(dataset)):

    label = dataset[i]["label"]

    if label == 0:

        fake_indices.append(i)

    elif label == 1:

        real_indices.append(i)

    if (
        len(fake_indices) >= target_per_class
        and
        len(real_indices) >= target_per_class
    ):

        break


selected_indices = (
    fake_indices[:target_per_class]
    +
    real_indices[:target_per_class]
)

train_dataset = dataset.select(
    selected_indices
)


print(
    f"Selected images: "
    f"{len(train_dataset):,}"
)

print(
    f"AI images: {target_per_class:,}"
)

print(
    f"Real images: {target_per_class:,}"
)


# ============================================================
# TRANSFORM
# ============================================================

weights = ResNet50_Weights.DEFAULT

transform = weights.transforms()


# ============================================================
# DATASET WRAPPER
# ============================================================

class CIFAKEDataset(
    torch.utils.data.Dataset
):

    def __init__(self, hf_dataset):

        self.dataset = hf_dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.fromarray(
                image
            )

        image = image.convert("RGB")

        image = transform(image)


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
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# DATALOADER
# ============================================================

train_data = CIFAKEDataset(
    train_dataset
)

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
# TRAINING
# ============================================================

history = []


print("\nStarting training...\n")


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
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


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
            accuracy=
                f"{100 * correct / total:.2f}%"
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

        "epoch":
            epoch + 1,

        "loss":
            epoch_loss,

        "accuracy":
            epoch_accuracy
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
        "Spatial-only ResNet50 V2",

    "dataset":
        DATASET_REPO,

    "split":
        "train",

    "training_samples":
        len(train_data),

    "ai_samples":
        target_per_class,

    "real_samples":
        target_per_class,

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

print("\n" + "=" * 65)

print("SPATIAL V2 TRAINING COMPLETE")

print("=" * 65)

print(
    "\nModel saved to:"
)

print(MODEL_PATH)

print(
    "\nMetrics saved to:"
)

print(METRICS_PATH)