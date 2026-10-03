"""
AIDetect Phase 7: Frequency-Domain Manipulation Dataset Loader & Preprocessing
==============================================================================
Dataset loader for manipulation_v1/ computing the authoritative native-resolution
2D FFT representation:
  1. Load PIL Image at native resolution
  2. For training: apply conservative spatial augmentations (RandomHorizontalFlip, ColorJitter)
     preserving native dimensions (no downscaling before FFT)
  3. Convert to tensor [3, H, W] at native resolution
  4. 2D FFT via torch.fft.fft2()
  5. Center zero-frequency via torch.fft.fftshift(dim=(-2, -1))
  6. Magnitude via torch.abs()
  7. Log compression via torch.log1p()
  8. Independent per-channel min-max normalization into [0, 1]
  9. Resize frequency representation to [3, 224, 224] via bilinear interpolation
 10. ImageNet normalization (mean/std)

Labels:
  0 = ORIGINAL_REAL
  1 = AI_MANIPULATED
"""

import os
import csv
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
    Authoritative native-resolution FFT frequency transform matching V4.
    Performs FFT at native resolution BEFORE resizing to 224x224.
    """
    _to_tensor  = transforms.ToTensor()
    _normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Step 1: Ensure RGB PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Step 2: PIL -> [3, H, W] float32 at native resolution
        x = self._to_tensor(image)

        # Steps 3-4: 2D FFT -> shift DC component to center
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Steps 5-6: Magnitude -> log1p compression
        magnitude = torch.log1p(torch.abs(fft))

        # Step 7: Independent per-channel min-max normalization
        for c in range(magnitude.shape[0]):
            ch_min = magnitude[c].min()
            ch_max = magnitude[c].max()
            magnitude[c] = (magnitude[c] - ch_min) / (ch_max - ch_min + 1e-8)

        # Step 8: Resize frequency map to 224x224 (after FFT, NOT before)
        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Step 9: ImageNet normalization
        return self._normalizer(magnitude)

    def __repr__(self) -> str:
        return (
            "AuthoritativeFrequencyTransform("
            "native_res_fft=True, log1p=True, "
            "per_channel_minmax=True, resize=(224,224), imagenet_norm=True)"
        )

# ---------------------------------------------------------------------------
# TRAINING SPATIAL AUGMENTATIONS (APPLIED BEFORE NATIVE FFT)
# ---------------------------------------------------------------------------

TRAIN_SPATIAL_AUG = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02,
    ),
])

FFT_TRANSFORM = AuthoritativeFrequencyTransform()

# ---------------------------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------------------------

class ManipulationFreqV1Dataset(Dataset):
    """
    Dataset for Frequency-Domain AI-manipulated image detection.
    Reads manipulation_v1/ metadata and transforms each image to its 224x224 FFT spectrum.
    """
    def __init__(self, split: str = "train", augment: bool = False):
        assert split in ("train", "validation", "test"), f"Invalid split: {split}"
        assert not (augment and split != "train"), "Augmentation only allowed for train split"
        
        self.split = split
        self.augment = augment
        self.fft_xform = FFT_TRANSFORM
        self.spatial_aug = TRAIN_SPATIAL_AUG if augment else None
        
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
            
        # If train augmentation enabled, apply mild spatial transform at native resolution
        if self.augment and self.spatial_aug is not None:
            img = self.spatial_aug(img)
            
        # Native-resolution FFT -> resize 224x224 -> ImageNet norm
        freq_tensor = self.fft_xform(img)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        
        return freq_tensor, label_tensor, sample

def collate_manipulation_freq(batch):
    tensors = torch.stack([item[0] for item in batch])
    labels  = torch.stack([item[1] for item in batch])
    metas   = [item[2] for item in batch]
    return tensors, labels, metas

PREPROCESSING_DOC = {
    "type": "authoritative_native_fft",
    "description": (
        "PIL(RGB) at native resolution -> "
        "[optional train: RandomHorizontalFlip(p=0.5) + ColorJitter(0.1,0.1,0.1,0.02)] -> "
        "ToTensor() [native resolution] -> "
        "torch.fft.fft2() -> fftshift(dim=(-2,-1)) -> abs() -> log1p() -> "
        "per-channel min-max normalisation -> "
        "F.interpolate(size=(224,224), mode='bilinear', align_corners=False) -> "
        "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
    ),
    "input_size": 224,
    "native_resolution_preserved_prior_to_fft": True,
    "label_convention": "0 = ORIGINAL_REAL, 1 = AI_MANIPULATED"
}

if __name__ == "__main__":
    for s in ["train", "validation", "test"]:
        ds = ManipulationFreqV1Dataset(s, augment=(s == "train"))
        print(f"Split '{s:<10}': {len(ds)} images")
        x, y, m = ds[0]
        print(f"  Sample FFT tensor: {tuple(x.shape)}, label: {y.item()} ({m['ground_truth']})")
    print("ManipulationFreqV1Dataset check passed.")

