"""
AIDetect Phase 25: Multi-Generator Manipulation Dataset Loader & Preprocessing
==============================================================================
Dataset loader for dataset_manipulation_v2/ computing the authoritative native-resolution
2D FFT representation:
  1. Load PIL Image at native resolution
  2. For training: apply controlled augmentations (RandomResizedCrop, HorizontalFlip,
     ColorJitter, JPEG simulation, GaussianBlur)
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
import io
import csv
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
MANIP_V2_DIR = PROJECT_DIR / "dataset_manipulation_v2"
METADATA_CSV = MANIP_V2_DIR / "metadata" / "metadata.csv"

CLASS_TO_IDX = {"ORIGINAL_REAL": 0, "AI_MANIPULATED": 1}
IDX_TO_CLASS = {0: "ORIGINAL_REAL", 1: "AI_MANIPULATED"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class AuthoritativeFrequencyTransform:
    """
    Authoritative native-resolution FFT frequency transform matching Phase 24.
    Performs FFT at native resolution BEFORE resizing to 224x224.
    """
    _to_tensor  = transforms.ToTensor()
    _normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Tensor at native resolution
        x = self._to_tensor(image)

        # 2D FFT -> shift DC component to center
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Magnitude -> log1p compression
        magnitude = torch.log1p(torch.abs(fft))

        # Independent per-channel min-max normalization
        for c in range(magnitude.shape[0]):
            ch_min = magnitude[c].min()
            ch_max = magnitude[c].max()
            magnitude[c] = (magnitude[c] - ch_min) / (ch_max - ch_min + 1e-8)

        # Resize frequency map to 224x224 (after FFT)
        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # ImageNet normalization
        return self._normalizer(magnitude)


class ControlledSpatialAugmentations:
    """
    Controlled augmentations for training (Section 14):
    - HorizontalFlip p=0.5
    - ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
    - JPEG simulation (quality=75-95, p=0.3)
    - GaussianBlur (kernel=3, sigma=0.1-1.0, p=0.2)
    Preserves native resolution so FFT is computed prior to 224x224 interpolation.
    """
    def __init__(self):
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        )
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))

    def __call__(self, img: Image.Image) -> Image.Image:
        # Random horizontal flip
        img = self.hflip(img)
        # Color jitter
        img = self.color_jitter(img)
        # JPEG simulation (p=0.3)
        if random.random() < 0.3:
            q = random.randint(75, 95)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        # Gaussian blur (p=0.2)
        if random.random() < 0.2:
            img = self.gaussian_blur(img)
        return img


class ManipulationV2Dataset(Dataset):
    """
    PyTorch Dataset for Phase 25 Multi-Generator Manipulation Dataset.
    Loads images and generates authoritative FFT frequency tensors.
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        is_training: bool = False,
        base_dir: Optional[Path] = None
    ):
        self.samples = samples
        self.is_training = is_training
        self.base_dir = base_dir or PROJECT_DIR
        self.fft_xform = AuthoritativeFrequencyTransform()
        self.spatial_aug = ControlledSpatialAugmentations() if is_training else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        meta = self.samples[idx]
        rel_path = meta.get("file_path") or meta.get("image_path")
        full_path = self.base_dir / rel_path

        img = Image.open(full_path).convert("RGB")

        if self.is_training and self.spatial_aug is not None:
            img = self.spatial_aug(img)

        freq_tensor = self.fft_xform(img)
        label = int(meta["ground_truth"])

        return freq_tensor, label, meta
