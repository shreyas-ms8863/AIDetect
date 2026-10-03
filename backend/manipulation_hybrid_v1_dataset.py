"""
AIDetect Phase 8: Hybrid Manipulation Dataset Loader & Synchronized Preprocessing
==================================================================================
Dataset loader for manipulation_v1/ producing synchronized dual representations:
  1. Spatial Branch: RGB Image -> shared augmentation -> 224x224 -> ImageNet norm
  2. Frequency Branch: Same augmented RGB image -> native-resolution FFT -> 224x224 -> ImageNet norm

Strict Guarantee Against Augmentation Mismatch:
  The spatial and frequency branches represent the EXACT SAME underlying augmented image.
  A single shared augmentation decision (horizontal flip, color jitter) is applied to the
  native PIL image before branching. The frequency branch evaluates the native-resolution
  spectrum of that augmented image prior to any downsampling.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
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
# AUTHORITATIVE FFT FREQUENCY TRANSFORM
# ---------------------------------------------------------------------------

class AuthoritativeFrequencyTransform:
    """
    Authoritative native-resolution FFT frequency transform matching Phase 7 & V4.
    Performs FFT at native image resolution BEFORE resizing magnitude to 224x224.
    """
    _to_tensor  = transforms.ToTensor()
    _normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Native resolution tensor [3, H, W]
        x = self._to_tensor(image)

        # 2D FFT -> shift DC component to center
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Magnitude spectrum -> log1p scale
        magnitude = torch.log1p(torch.abs(fft))

        # Independent per-channel min-max normalization
        for c in range(magnitude.shape[0]):
            ch_min = magnitude[c].min()
            ch_max = magnitude[c].max()
            magnitude[c] = (magnitude[c] - ch_min) / (ch_max - ch_min + 1e-8)

        # Bilinear interpolation of frequency map to 224x224
        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # ImageNet normalization
        return self._normalizer(magnitude)

# ---------------------------------------------------------------------------
# SPATIAL TRANSFORMS
# ---------------------------------------------------------------------------

SPATIAL_TRAIN_HEAD = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        ratio=(0.75, 1.3333333),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

SPATIAL_EVAL_XFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

COLOR_JITTER = transforms.ColorJitter(
    brightness=0.1,
    contrast=0.1,
    saturation=0.1,
    hue=0.02,
)

FFT_TRANSFORM = AuthoritativeFrequencyTransform()

# ---------------------------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------------------------

class ManipulationHybridV1Dataset(Dataset):
    """
    Dataset for Two-Stream Hybrid AI-manipulation detection.
    Returns: (x_spatial, x_freq, label_tensor, sample_metadata)
    """
    def __init__(self, split: str = "train", augment: bool = False):
        assert split in ("train", "validation", "test"), f"Invalid split: {split}"
        assert not (augment and split != "train"), "Augmentation only allowed for train split"
        
        self.split = split
        self.augment = augment
        self.fft_xform = FFT_TRANSFORM
        self.spatial_eval_xform = SPATIAL_EVAL_XFORM
        self.spatial_train_head = SPATIAL_TRAIN_HEAD
        self.color_jitter = COLOR_JITTER
        
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        p = sample["filepath"]
        
        try:
            with Image.open(p) as raw_img:
                img = raw_img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image at {p}: {e}") from e
            
        if self.augment:
            # 1. Single shared augmentation decision applied to native PIL image
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = self.color_jitter(img)
            
            # 2. Branch A: Native-resolution FFT of the augmented image
            x_freq = self.fft_xform(img)
            
            # 3. Branch B: Spatial crop/resize of the same augmented image
            x_spatial = self.spatial_train_head(img)
        else:
            # Deterministic evaluation without augmentation
            x_freq = self.fft_xform(img)
            x_spatial = self.spatial_eval_xform(img)
            
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return x_spatial, x_freq, label_tensor, sample

def collate_manipulation_hybrid(batch):
    spatial_tensors = torch.stack([item[0] for item in batch])
    freq_tensors    = torch.stack([item[1] for item in batch])
    labels          = torch.stack([item[2] for item in batch])
    metas           = [item[3] for item in batch]
    return spatial_tensors, freq_tensors, labels, metas

PREPROCESSING_DOC = {
    "spatial_branch": (
        "PIL(RGB) -> shared spatial augmentation [flip+jitter] -> "
        "RandomResizedCrop(224, scale=(0.7,1.0), BILINEAR) [eval: Resize(256)->CenterCrop(224)] -> "
        "ToTensor() -> Normalize(ImageNet mean/std)"
    ),
    "frequency_branch": (
        "PIL(RGB) -> same shared spatial augmentation [flip+jitter] -> "
        "ToTensor() [native resolution] -> "
        "torch.fft.fft2() -> fftshift(dim=(-2,-1)) -> abs() -> log1p() -> "
        "per-channel min-max normalisation -> "
        "F.interpolate(size=(224,224), mode='bilinear', align_corners=False) -> "
        "Normalize(ImageNet mean/std)"
    ),
    "shared_augmentation": "Single synchronized flip & jitter decision executed on native image before branching.",
    "input_size": 224,
    "label_convention": "0 = ORIGINAL_REAL, 1 = AI_MANIPULATED"
}

if __name__ == "__main__":
    for s in ["train", "validation", "test"]:
        ds = ManipulationHybridV1Dataset(s, augment=(s == "train"))
        print(f"Split '{s:<10}': {len(ds)} images")
        xs, xf, y, m = ds[0]
        print(f"  Sample Spatial tensor: {tuple(xs.shape)}")
        print(f"  Sample Freq tensor   : {tuple(xf.shape)}")
        print(f"  Label: {y.item()} ({m['ground_truth']})")
    print("ManipulationHybridV1Dataset check passed.")

