"""
AIDetect Spatial V3
===================

Transfer-learning based AI-generated image detector.

Dataset:
    dragonintelligence/CIFAKE-image-dataset

Official CIFAKE dataset:
    Train: 100,000 images
        50,000 FAKE
        50,000 REAL

    Test: 20,000 images
        10,000 FAKE
        10,000 REAL

Our V3 experiment:
    90,000 images -> training
        45,000 FAKE
        45,000 REAL

    10,000 images -> validation
         5,000 FAKE
         5,000 REAL

    20,000 official test images -> untouched

Model:
    ImageNet-pretrained ResNet-50

Classes:
    0 = FAKE / AI-generated
    1 = REAL

Hardware:
    CUDA / NVIDIA GPU when available
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Image size expected by ResNet-50
IMAGE_SIZE = 224

# DataLoader workers
# Keep this at 2 initially on Windows.
NUM_WORKERS = 0


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "evaluation_results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    MODEL_DIR /
    "spatial_resnet50_v3.pth"
)

METRICS_PATH = (
    RESULT_DIR /
    "spatial_v3_training_metrics.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Allow cuDNN to select faster algorithms.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 75)
print("AIDetect Spatial V3 - Transfer Learning")
print("=" * 75)

print(
    f"PyTorch version : {torch.__version__}"
)

print(
    f"Device          : {device}"
)


if torch.cuda.is_available():

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    gpu_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory      : "
        f"{gpu_memory:.2f} GB"
    )


print("=" * 75)


# ============================================================
# LOAD CIFAKE DATASET
# ============================================================

print()
print("Loading CIFAKE dataset...")
print()

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)


print(dataset)

print()
print(
    f"Official training images : "
    f"{len(dataset['train']):,}"
)

print(
    f"Official test images     : "
    f"{len(dataset['test']):,}"
)


# ============================================================
# IMPORTANT DATASET RULE
# ============================================================
#
# We ONLY split the official training set.
#
# The official CIFAKE test set is NEVER used for training
# or validation.
#
# This preserves a clean final test set.
# ============================================================


train_split = dataset["train"].train_test_split(
    test_size=0.10,
    seed=SEED,
    stratify_by_column="label"
)

train_data = train_split["train"]

validation_data = train_split["test"]

test_data = dataset["test"]


# ============================================================
# VERIFY SPLIT SIZES
# ============================================================

print()
print("=" * 75)
print("DATASET SPLIT")
print("=" * 75)

print(
    f"Training   : "
    f"{len(train_data):,}"
)

print(
    f"Validation : "
    f"{len(validation_data):,}"
)

print(
    f"Test       : "
    f"{len(test_data):,}"
)

print("=" * 75)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def count_labels(data):

    fake_count = 0
    real_count = 0

    for label in data["label"]:

        if label == 0:

            fake_count += 1

        elif label == 1:

            real_count += 1

    return fake_count, real_count


train_fake, train_real = count_labels(
    train_data
)

val_fake, val_real = count_labels(
    validation_data
)

test_fake, test_real = count_labels(
    test_data
)


print()
print("CLASS DISTRIBUTION")
print("-" * 75)

print(
    f"TRAIN      | "
    f"FAKE: {train_fake:,} | "
    f"REAL: {train_real:,}"
)

print(
    f"VALIDATION | "
    f"FAKE: {val_fake:,} | "
    f"REAL: {val_real:,}"
)

print(
    f"TEST       | "
    f"FAKE: {test_fake:,} | "
    f"REAL: {test_real:,}"
)

print("-" * 75)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# ImageNet normalization because we are using
# ImageNet-pretrained ResNet-50.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ------------------------------------------------------------
# Training augmentation
# ------------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ------------------------------------------------------------
# Validation transformation
# ------------------------------------------------------------

validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# CIFAKE PYTORCH DATASET
# ============================================================

class CIFAKEDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        transform=None
    ):

        self.dataset = hf_dataset

        self.transform = transform


    def __len__(self):

        return len(
            self.dataset
        )


    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]

        label = item["label"]


        # Ensure RGB input.
        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )


        if self.transform is not None:

            image = self.transform(
                image
            )


        return image, label


# ============================================================
# CREATE PYTORCH DATASETS
# ============================================================

train_dataset = CIFAKEDataset(
    train_data,
    train_transform
)

validation_dataset = CIFAKEDataset(
    validation_data,
    validation_transform
)


# ============================================================
# DATALOADERS
# ============================================================

print()
print("Creating DataLoaders...")


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


print()
print("DATALOADER INFORMATION")
print("-" * 75)

print(
    f"Batch size          : "
    f"{BATCH_SIZE}"
)

print(
    f"Training batches     : "
    f"{len(train_loader):,}"
)

print(
    f"Validation batches   : "
    f"{len(validation_loader):,}"
)

print(
    f"Workers              : "
    f"{NUM_WORKERS}"
)

print("-" * 75)


# ============================================================
# LOAD RESNET-50
# ============================================================

print()
print("=" * 75)
print("LOADING IMAGENET-PRETRAINED RESNET-50")
print("=" * 75)


weights = ResNet50_Weights.DEFAULT


model = resnet50(
    weights=weights
)


# ============================================================
# REPLACE ORIGINAL CLASSIFIER
# ============================================================

num_features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Dropout(
        p=0.30
    ),

    nn.Linear(
        num_features,
        2
    )
)


model = model.to(
    device
)


print()
print(
    "Model: ResNet-50"
)

print(
    "Pretrained weights: ImageNet"
)

print(
    "Output classes: 2"
)

print(
    "Class 0: FAKE"
)

print(
    "Class 1: REAL"
)

print("=" * 75)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


print()
print(
    f"Mixed precision: "
    f"{use_amp}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()


    total_loss = 0.0

    total_samples = 0


    # Confusion matrix components.
    #
    # Positive class here = REAL (class 1)
    #
    true_positive = 0

    true_negative = 0

    false_positive = 0

    false_negative = 0


    with torch.no_grad():

        for images, labels in loader:


            images = images.to(

                device,

                non_blocking=True
            )


            labels = labels.to(

                device,

                non_blocking=True
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(

                outputs,

                dim=1
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            total_loss += (

                loss.item()
                *
                labels.size(0)

            )


            total_samples += (
                labels.size(0)
            )


            # ------------------------------------------------
            # Confusion matrix
            #
            # 0 = FAKE
            # 1 = REAL
            # ------------------------------------------------

            true_positive += (

                (
                    (predictions == 1)
                    &
                    (labels == 1)
                )

                .sum()
                .item()
            )


            true_negative += (

                (
                    (predictions == 0)
                    &
                    (labels == 0)
                )

                .sum()
                .item()
            )


            false_positive += (

                (
                    (predictions == 1)
                    &
                    (labels == 0)
                )

                .sum()
                .item()
            )


            false_negative += (

                (
                    (predictions == 0)
                    &
                    (labels == 1)
                )

                .sum()
                .item()
            )


    # ========================================================
    # METRICS
    # ========================================================

    average_loss = (

        total_loss
        /
        total_samples
    )


    accuracy = (

        true_positive
        +
        true_negative

    ) / total_samples


    precision = (

        true_positive
        /

        (
            true_positive
            +
            false_positive
        )

        if (
            true_positive
            +
            false_positive
        ) > 0

        else 0.0
    )


    recall = (

        true_positive
        /

        (
            true_positive
            +
            false_negative
        )

        if (
            true_positive
            +
            false_negative
        ) > 0

        else 0.0
    )


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

        if (
            precision
            +
            recall
        ) > 0

        else 0.0
    )


    return {

        "loss": average_loss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "true_positive": true_positive,

        "true_negative": true_negative,

        "false_positive": false_positive,

        "false_negative": false_negative
    }


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 75)
print("STARTING V3 TRAINING")
print("=" * 75)

print()
print(
    f"Training images : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation      : "
    f"{len(validation_dataset):,}"
)

print(
    f"Epochs          : "
    f"{NUM_EPOCHS}"
)

print(
    f"Batch size      : "
    f"{BATCH_SIZE}"
)

print(
    f"Learning rate   : "
    f"{LEARNING_RATE}"
)

print()
print(
    "The official 20,000-image "
    "test set is NOT being used."
)

print("=" * 75)


training_start = time.time()


history = []


best_validation_f1 = -1.0

best_epoch = 0


# ============================================================
# EPOCH LOOP
# ============================================================

for epoch in range(
    NUM_EPOCHS
):


    epoch_start = time.time()


    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    # ========================================================
    # TRAINING BATCHES
    # ========================================================

    for batch_index, (
        images,
        labels
    ) in enumerate(
        train_loader
    ):


        images = images.to(

            device,

            non_blocking=True
        )


        labels = labels.to(

            device,

            non_blocking=True
        )


        # Clear gradients.
        optimizer.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # FORWARD + BACKWARD
        # ====================================================

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


        # ====================================================
        # TRAINING METRICS
        # ====================================================

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

            (
                predictions
                ==
                labels
            )

            .sum()
            .item()
        )


        total += (
            labels.size(0)
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            (batch_index + 1)
            % 100
            == 0
        ):

            current_accuracy = (
                correct / total
            )


            print(

                f"Epoch "
                f"{epoch + 1}/{NUM_EPOCHS} | "

                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} | "

                f"Loss: "
                f"{loss.item():.4f} | "

                f"Accuracy: "
                f"{current_accuracy * 100:.2f}%"

            )


    # ========================================================
    # TRAINING EPOCH METRICS
    # ========================================================

    train_loss = (

        running_loss
        /
        total
    )


    train_accuracy = (

        correct
        /
        total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation_metrics = evaluate(

        model,

        validation_loader
    )


    # ========================================================
    # LEARNING RATE UPDATE
    # ========================================================

    scheduler.step()


    # ========================================================
    # EPOCH TIME
    # ========================================================

    epoch_time = (

        time.time()
        -
        epoch_start
    )


    # ========================================================
    # SAVE EPOCH RESULT
    # ========================================================

    epoch_result = {

        "epoch":
        epoch + 1,

        "train_loss":
        train_loss,

        "train_accuracy":
        train_accuracy,

        "validation_loss":
        validation_metrics[
            "loss"
        ],

        "validation_accuracy":
        validation_metrics[
            "accuracy"
        ],

        "validation_precision":
        validation_metrics[
            "precision"
        ],

        "validation_recall":
        validation_metrics[
            "recall"
        ],

        "validation_f1":
        validation_metrics[
            "f1"
        ],

        "learning_rate":
        optimizer.param_groups[
            0
        ]["lr"],

        "epoch_time_seconds":
        epoch_time
    }


    history.append(
        epoch_result
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print()
    print("=" * 75)

    print(
        f"Epoch "
        f"{epoch + 1}/{NUM_EPOCHS} COMPLETE"
    )

    print("-" * 75)

    print(
        f"Train Loss       : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss  : "
        f"{validation_metrics['loss']:.4f}"
    )

    print(
        f"Validation Acc.  : "
        f"{validation_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation Prec. : "
        f"{validation_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Validation Recall: "
        f"{validation_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"Validation F1    : "
        f"{validation_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Epoch Time       : "
        f"{epoch_time / 60:.2f} minutes"
    )

    print("=" * 75)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    current_f1 = (
        validation_metrics["f1"]
    )


    if current_f1 > best_validation_f1:

        best_validation_f1 = current_f1

        best_epoch = epoch + 1


        checkpoint = {

            "model_state_dict":
            model.state_dict(),

            "architecture":
            "resnet50",

            "version":
            "V3",

            "pretrained":
            "ImageNet",

            "class_names":
            [
                "FAKE",
                "REAL"
            ],

            "image_size":
            IMAGE_SIZE,

            "best_validation_f1":
            best_validation_f1,

            "best_epoch":
            best_epoch,

            "training_images":
            len(train_dataset),

            "validation_images":
            len(validation_dataset),

            "official_test_images":
            len(test_data),

            "train_fake":
            train_fake,

            "train_real":
            train_real,

            "validation_fake":
            val_fake,

            "validation_real":
            val_real
        }


        torch.save(

            checkpoint,

            BEST_MODEL_PATH
        )


        print()
        print(
            "✓ NEW BEST MODEL SAVED"
        )

        print(
            f"Validation F1: "
            f"{best_validation_f1 * 100:.2f}%"
        )

        print(
            f"Saved to:"
        )

        print(
            BEST_MODEL_PATH
        )

        print()


# ============================================================
# TRAINING FINISHED
# ============================================================

total_training_time = (

    time.time()
    -
    training_start
)


# ============================================================
# SAVE TRAINING RESULTS
# ============================================================

results = {

    "project":
    "AIDetect",

    "model":
    "ResNet-50",

    "version":
    "V3",

    "task":
    "AI-generated image detection",

    "transfer_learning":
    True,

    "pretrained_weights":
    "ImageNet",

    "dataset":
    "CIFAKE",

    "official_training_images":
    100000,

    "training_images":
    len(train_dataset),

    "validation_images":
    len(validation_dataset),

    "official_test_images":
    len(test_data),

    "train_fake":
    train_fake,

    "train_real":
    train_real,

    "validation_fake":
    val_fake,

    "validation_real":
    val_real,

    "test_fake":
    test_fake,

    "test_real":
    test_real,

    "epochs":
    NUM_EPOCHS,

    "batch_size":
    BATCH_SIZE,

    "learning_rate":
    LEARNING_RATE,

    "weight_decay":
    WEIGHT_DECAY,

    "image_size":
    IMAGE_SIZE,

    "best_validation_f1":
    best_validation_f1,

    "best_epoch":
    best_epoch,

    "total_training_time_seconds":
    total_training_time,

    "total_training_time_hours":
    total_training_time / 3600,

    "history":
    history
}


with open(

    METRICS_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        results,

        file,

        indent=4
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print()
print("=" * 75)
print("V3 TRAINING COMPLETE")
print("=" * 75)

print()

print(
    f"Best epoch        : "
    f"{best_epoch}"
)

print(
    f"Best validation F1: "
    f"{best_validation_f1 * 100:.2f}%"
)

print()

print(
    f"Total training time: "
    f"{total_training_time / 3600:.2f} hours"
)

print()

print(
    "BEST MODEL:"
)

print(
    BEST_MODEL_PATH
)

print()

print(
    "TRAINING METRICS:"
)

print(
    METRICS_PATH
)

print()

print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "The official 20,000 CIFAKE test images "
    "were NOT used during training or validation."
)

print(
    "They are reserved for the final V3 evaluation."
)

print("=" * 75)