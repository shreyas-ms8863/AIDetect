"""
AIDetect Phase 6: Manipulation Dataset Loader & Authoritative Transforms
==========================================================================
Data loader for manipulation_v1/ utilizing canonical metadata.csv.
Guarantees:
  - Strict binary label mapping: 0 = ORIGINAL_REAL, 1 = AI_MANIPULATED
  - Deterministic evaluation preprocessing (Resize 256 -> CenterCrop 224)
  - Forensic-safe training augmentation (no aggressive artifacts erasure)
  - Group-based splitting strictly preserved from split_manifest.csv
"""

import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# DIRECTORIES & METADATA
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
MANIP_DIR    = PROJECT_DIR / "manipulation_v1"
METADATA_CSV = MANIP_DIR / "metadata" / "metadata.csv"

CLASS_TO_IDX = {"ORIGINAL_REAL": 0, "AI_MANIPULATED": 1}
IDX_TO_CLASS = {0: "ORIGINAL_REAL", 1: "AI_MANIPULATED"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------------

def get_train_transforms():
    """
    Forensics-appropriate training transformations:
      - RandomResizedCrop(224) preserves aspect-ratio and local scale
      - RandomHorizontalFlip(p=0.5)
      - Mild ColorJitter (brightness, contrast, saturation, hue)
      - ImageNet Normalization
    Note: Heavy blur and heavy compression are intentionally excluded to
    preserve localized generative manipulation boundary artifacts.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transforms():
    """
    Deterministic evaluation transformation for validation and test.
      - Resize(256)
      - CenterCrop(224)
      - ImageNet Normalization
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

TRAIN_TRANSFORMS = get_train_transforms()
EVAL_TRANSFORMS  = get_eval_transforms()

# ---------------------------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------------------------

class ManipulationV1Dataset(Dataset):
    """
    Dataset for AI-manipulated image detection loading from manipulation_v1/.
    """
    def __init__(self, split: str = "train", augment: bool = False):
        assert split in ("train", "validation", "test"), f"Invalid split: {split}"
        assert not (augment and split != "train"), "Augmentation only allowed for train split"
        
        self.split = split
        self.augment = augment
        self.transform = TRAIN_TRANSFORMS if augment else EVAL_TRANSFORMS
        
        self.samples: List[Dict[str, Any]] = []
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    lbl = int(row["label"])
                    orig_id = row["original_id"]
                    
                    if lbl == 0:
                        rel_path = f"manipulation_v1/{split}/original/orig_{orig_id}.png"
                    else:
                        cat = row["manipulation_type"]
                        manip_id = row["image_id"].replace(f"m1_{split}_manip_", "")
                        rel_path = f"manipulation_v1/{split}/manipulated/{cat}/manip_{manip_id}.png"
                        
                    abs_path = PROJECT_DIR / rel_path
                    self.samples.append({
                        "filepath": str(abs_path),
                        "image_id": row["image_id"],
                        "original_id": orig_id,
                        "label": lbl,
                        "ground_truth": row["ground_truth"],
                        "manipulation_type": row["manipulation_type"],
                        "generator_or_editor": row["generator_or_editor"],
                        "source_dataset": row["source_dataset"],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        p = sample["filepath"]
        
        try:
            with Image.open(p) as raw_img:
                img = raw_img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image at {p}: {e}") from e
            
        tensor_img = self.transform(img)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        
        return tensor_img, label_tensor, sample

def collate_manipulation(batch):
    tensors = torch.stack([item[0] for item in batch])
    labels  = torch.stack([item[1] for item in batch])
    metas   = [item[2] for item in batch]
    return tensors, labels, metas

PREPROCESSING_DOC = {
    "train": (
        "RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.333), BILINEAR) -> "
        "RandomHorizontalFlip(p=0.5) -> "
        "ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02) -> "
        "ToTensor() -> "
        "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
    ),
    "eval": (
        "Resize(256, BILINEAR) -> "
        "CenterCrop(224) -> "
        "ToTensor() -> "
        "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
    ),
    "input_size": 224,
    "label_convention": "0 = ORIGINAL_REAL, 1 = AI_MANIPULATED"
}

if __name__ == "__main__":
    for s in ["train", "validation", "test"]:
        ds = ManipulationV1Dataset(s, augment=(s == "train"))
        print(f"Split '{s:<10}': {len(ds)} images")
        x, y, m = ds[0]
        print(f"  Sample tensor : {tuple(x.shape)}, label: {y.item()} ({m['ground_truth']})")
    print("Dataset module verification passed.")

