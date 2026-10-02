import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/spatial_resnet50_v2.pth"
OUTPUT_DIR = "spatial_v2_results"

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

print("Loading CIFAKE test dataset...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="test"
)

print(f"Test images: {len(dataset)}")


# ============================================================
# TRANSFORM
# ============================================================

weights = ResNet50_Weights.DEFAULT
transform = weights.transforms()


class CIFAKEDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        image = item["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        image = transform(image)

        # CIFAKE:
        # 0 = Fake / AI
        # 1 = Real
        #
        # Our model:
        # 0 = Real
        # 1 = AI

        original_label = int(item["label"])

        if original_label == 0:
            label = 1       # AI
        else:
            label = 0       # Real

        return image, label


test_dataset = CIFAKEDataset(dataset)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# MODEL
# ============================================================

print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = resnet50(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle either a raw state_dict or checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []

print("\nEvaluating Spatial V2...\n")

with torch.no_grad():

    for images, labels in tqdm(
        test_loader,
        desc="Testing"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL V2 CIFAKE TEST RESULTS")
print("=" * 60)

print(f"Dataset: CIFAKE")
print(f"Split: Test")
print(f"Total images: {len(all_labels)}")

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("=" * 60)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Spatial V2 - ResNet50",
    "dataset": "CIFAKE",
    "split": "test",
    "total_images": len(all_labels),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "spatial_v2_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print(f"\nMetrics saved to:")
print(metrics_path)