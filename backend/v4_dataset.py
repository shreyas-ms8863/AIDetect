"""
AIDetect V4 Dataset Module
===========================
Single authoritative source for ALL V4 preprocessing:
  - Spatial training transforms (with augmentation)
  - Spatial validation/test transforms (deterministic)
  - Authoritative FFT frequency transform (native-resolution FFT, then resize)
  - V4Dataset class (spatial, frequency, hybrid modes)
  - Domain-annotated loading from dataset_v4/ directory structure

Label convention (strictly enforced throughout V4):
  0 = REAL
  1 = AI

IMPORTANT: This file is the canonical preprocessing definition.
  backend/train_v4.py and backend/evaluate_v4.py import from here.
  Future V4 inference in backend/main.py must also import from here
  (or faithfully replicate these exact transforms) to avoid train/inference
  distribution shift.

FFT Pipeline (Authoritative -- matches V3 inference FFTTransform in main.py):
  1. PIL Image -> convert("RGB")
  2. torchvision.transforms.ToTensor()  -> [3, H, W] at native resolution
  3. torch.fft.fft2()                   -> complex [3, H, W]
  4. torch.fft.fftshift(..., dim=(-2,-1))
  5. torch.abs()                        -> magnitude [3, H, W]
  6. torch.log1p()                      -> log-scaled magnitude
  7. Per-channel min-max normalisation  -> [0, 1] per channel
  8. F.interpolate(..., size=(224,224), mode='bilinear', align_corners=False)
  9. torchvision.transforms.Normalize(ImageNet mean/std)
  Output: [3, 224, 224] tensor

Spatial Train Pipeline:
  RandomResizedCrop(224, scale=(0.7,1.0), ratio=(0.75,1.33))
  -> RandomHorizontalFlip(p=0.5)
  -> ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
  -> RandomApply(GaussianBlur(kernel=3, sigma=(0.1,1.0)), p=0.2)
  -> JPEG simulation (quality 75-95, p=0.3)
  -> ToTensor()
  -> Normalize(ImageNet mean/std)

Spatial Val/Test Pipeline:
  Resize(256, interpolation=BILINEAR)
  -> CenterCrop(224)
  -> ToTensor()
  -> Normalize(ImageNet mean/std)
"""

import os
import io
import csv
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
DATASET_V4   = PROJECT_DIR / "dataset_v4"
METADATA_CSV = DATASET_V4 / "metadata" / "metadata.csv"

# ---------------------------------------------------------------------------
# LABEL CONVENTION
# ---------------------------------------------------------------------------

CLASS_MAPPING = {"REAL": 0, "AI": 1}
IDX_TO_CLASS  = {0: "REAL", 1: "AI"}

# ---------------------------------------------------------------------------
# IMAGENET NORMALISATION CONSTANTS
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# JPEG SIMULATION TRANSFORM
# ---------------------------------------------------------------------------

class JPEGSimulation:
    """
    Simulate JPEG compression at a random quality level [min_q, max_q].
    Applied with probability p.
    """

    def __init__(self, min_q=75, max_q=95, p=0.3):
        self.min_q = min_q
        self.max_q = max_q
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.min_q, self.max_q)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

    def __repr__(self):
        return "JPEGSimulation(quality=[{},{}], p={})".format(
            self.min_q, self.max_q, self.p)


# ---------------------------------------------------------------------------
# SPATIAL TRANSFORMS
# ---------------------------------------------------------------------------

def build_spatial_train_transform():
    """
    Spatial training transform with controlled forensics-appropriate augmentation.
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
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.2,
        ),
        JPEGSimulation(min_q=75, max_q=95, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_spatial_val_transform():
    """
    Spatial validation/test transform -- deterministic, no augmentation.
    Uses Resize(256) -> CenterCrop(224) matching ResNet50_Weights.DEFAULT.transforms().
    This is also the transform that MUST be used during V4 inference.
    """
    return transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# AUTHORITATIVE FFT TRANSFORM
# ---------------------------------------------------------------------------

class AuthoritativeFFTTransform:
    """
    Authoritative V4 FFT frequency transform.

    This is the SINGLE canonical definition of the FFT preprocessing pipeline.
    It must be used identically during training, validation, test, and inference.
    DO NOT introduce a different FFT path in backend/main.py for V4 inference.

    Pipeline:
        1. Ensure PIL Image in RGB mode
        2. ToTensor() -> [3, H, W] float32 in [0,1] at NATIVE resolution
        3. torch.fft.fft2()              -> complex [3, H, W]
        4. torch.fft.fftshift(dim=(-2,-1)) -> center low frequencies
        5. torch.abs()                   -> magnitude spectrum [3, H, W]
        6. torch.log1p()                 -> log-scale
        7. Per-channel min-max norm      -> [0, 1] independently per channel
        8. F.interpolate(size=(224,224), mode='bilinear', align_corners=False)
        9. ImageNet Normalize            -> final [3, 224, 224] tensor

    Notes:
      - FFT is performed at NATIVE image resolution before any resize.
        This preserves high-frequency forensic artefacts.
      - Per-channel normalisation preserves relative channel differences.
      - The output is ImageNet-normalised to be compatible with the ResNet50
        backbone pretrained with ImageNet statistics.
    """

    _to_tensor  = transforms.ToTensor()
    _normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image):
        # Step 1: ensure PIL RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Step 2: PIL -> [3, H, W] float32 at native resolution
        x = self._to_tensor(image)

        # Steps 3-4: FFT -> shift low frequencies to centre
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Steps 5-6: magnitude -> log scale
        magnitude = torch.log1p(torch.abs(fft))

        # Step 7: per-channel min-max normalisation -> [0, 1]
        for c in range(magnitude.shape[0]):
            ch_min = magnitude[c].min()
            ch_max = magnitude[c].max()
            magnitude[c] = (magnitude[c] - ch_min) / (ch_max - ch_min + 1e-8)

        # Step 8: resize to 224x224 (after FFT, not before)
        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Step 9: ImageNet normalisation
        return self._normalizer(magnitude)

    def __repr__(self):
        return (
            "AuthoritativeFFTTransform("
            "native_res_fft=True, log1p=True, "
            "per_channel_minmax=True, "
            "resize=(224,224), imagenet_normalize=True)"
        )


# Singleton instances (import these in train/evaluate scripts)
FFT_TRANSFORM        = AuthoritativeFFTTransform()
SPATIAL_TRAIN_XFORM  = build_spatial_train_transform()
SPATIAL_VAL_XFORM    = build_spatial_val_transform()


# ---------------------------------------------------------------------------
# METADATA LOADER
# ---------------------------------------------------------------------------

def load_metadata(split):
    """
    Load metadata.csv and return rows for the given split.
    split: one of 'train', 'validation', 'test'
    """
    assert split in ("train", "validation", "test"), \
        "Unknown split: {!r}".format(split)
    rows = []
    with open(METADATA_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# V4 DATASET CLASS
# ---------------------------------------------------------------------------

class V4Dataset(Dataset):
    """
    V4Dataset -- loads images from dataset_v4/ using metadata.csv.

    Returns (spatial_tensor, fft_tensor, label, domain) tuples for Hybrid.
    For Spatial-only or Frequency-only models, returns (tensor, label, domain).

    Args:
        split:   'train', 'validation', or 'test'
        mode:    'spatial' | 'frequency' | 'hybrid'
        augment: If True, use training augmentation. Only for split='train'.

    Label convention: 0 = REAL, 1 = AI
    """

    def __init__(self, split, mode="hybrid", augment=False):
        assert mode in ("spatial", "frequency", "hybrid"), \
            "Unknown mode: {!r}".format(mode)
        assert not (augment and split != "train"), \
            "Augmentation should only be used with split='train'."

        self.split   = split
        self.mode    = mode
        self.augment = augment

        self.spatial_xform = (
            SPATIAL_TRAIN_XFORM if augment else SPATIAL_VAL_XFORM
        )
        self.fft_xform = FFT_TRANSFORM

        rows = load_metadata(split)
        self.samples = []
        for row in rows:
            abs_path = PROJECT_DIR / row["filepath"]
            label    = CLASS_MAPPING[row["ground_truth"]]
            domain   = row["domain"]
            self.samples.append((str(abs_path), label, domain))

        if split == "train":
            rng = random.Random(42)
            rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError("Failed to open image: {}".format(path)) from e

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.mode == "spatial":
            x_spatial = self.spatial_xform(img)
            return x_spatial, label_tensor, domain

        elif self.mode == "frequency":
            x_fft = self.fft_xform(img)
            return x_fft, label_tensor, domain

        else:  # hybrid
            if self.augment:
                img_aug   = self._apply_pil_augmentations(img)
                x_spatial = self._pil_to_spatial_tensor(img_aug)
                x_fft     = self.fft_xform(img_aug)
            else:
                x_spatial = self.spatial_xform(img)
                x_fft     = self.fft_xform(img)
            return x_spatial, x_fft, label_tensor, domain

    # PIL-level augmentation pipeline (shared between spatial+FFT for hybrid)
    _pil_aug = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.2,
        ),
        JPEGSimulation(min_q=75, max_q=95, p=0.3),
    ])

    _to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def _apply_pil_augmentations(self, img):
        return self._pil_aug(img)

    def _pil_to_spatial_tensor(self, img):
        return self._to_tensor_norm(img)


# ---------------------------------------------------------------------------
# COLLATE FUNCTIONS
# ---------------------------------------------------------------------------

def collate_spatial(batch):
    """Collate for spatial/frequency mode: (tensors, labels, domains)."""
    tensors = torch.stack([b[0] for b in batch])
    labels  = torch.stack([b[1] for b in batch])
    domains = [b[2] for b in batch]
    return tensors, labels, domains


def collate_hybrid(batch):
    """Collate for hybrid mode: (s_tensors, f_tensors, labels, domains)."""
    s_tensors = torch.stack([b[0] for b in batch])
    f_tensors = torch.stack([b[1] for b in batch])
    labels    = torch.stack([b[2] for b in batch])
    domains   = [b[3] for b in batch]
    return s_tensors, f_tensors, labels, domains


# ---------------------------------------------------------------------------
# DATASET SUMMARY UTILITY
# ---------------------------------------------------------------------------

def print_dataset_summary():
    """Print comprehensive dataset counts by split, class, and domain."""
    print("=" * 70)
    print("V4 DATASET SUMMARY")
    print("=" * 70)

    domains_real = ["genimage", "historical", "modern"]
    domains_ai   = ["genimage", "historical_style", "modern"]

    for split in ("train", "validation", "test"):
        rows      = load_metadata(split)
        real_rows = [r for r in rows if r["ground_truth"] == "REAL"]
        ai_rows   = [r for r in rows if r["ground_truth"] == "AI"]

        print("\n  {} SPLIT (Total: {})".format(split.upper(), len(rows)))
        print("    REAL: {}".format(len(real_rows)))
        for d in domains_real:
            cnt = sum(1 for r in real_rows if r["domain"] == d)
            print("      {:<20}: {}".format(d, cnt))
        print("    AI:   {}".format(len(ai_rows)))
        for d in domains_ai:
            cnt = sum(1 for r in ai_rows if r["domain"] == d)
            print("      {:<20}: {}".format(d, cnt))

    print("=" * 70)


# ---------------------------------------------------------------------------
# PREPROCESSING DOCUMENTATION (for checkpoint embedding)
# ---------------------------------------------------------------------------

PREPROCESSING_DOC = {
    "spatial_train": (
        "RandomResizedCrop(224, scale=(0.7,1.0), ratio=(0.75,1.33)) -> "
        "RandomHorizontalFlip(p=0.5) -> "
        "ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02) -> "
        "RandomApply(GaussianBlur(k=3, sigma=(0.1,1.0)), p=0.2) -> "
        "JPEGSimulation(quality=[75,95], p=0.3) -> "
        "ToTensor() -> "
        "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
    ),
    "spatial_val": (
        "Resize(256, BILINEAR) -> "
        "CenterCrop(224) -> "
        "ToTensor() -> "
        "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
    ),
    "fft": (
        "PIL->RGB -> "
        "ToTensor() [native resolution, NOT pre-resized] -> "
        "fft2() -> fftshift(dim=(-2,-1)) -> abs() -> log1p() -> "
        "per-channel min-max normalisation -> "
        "F.interpolate(size=(224,224), mode='bilinear', align_corners=False) -> "
        "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
    ),
    "hybrid_shared_aug": (
        "Same augmented PIL image is used for BOTH spatial and FFT branches "
        "during training, ensuring consistent inputs across branches."
    ),
    "label_convention": "0 = REAL, 1 = AI",
    "input_size": 224,
}


if __name__ == "__main__":
    print_dataset_summary()
    print("\nFFT Transform:", FFT_TRANSFORM)
    print("Spatial Train:", SPATIAL_TRAIN_XFORM)
    print("Spatial Val:  ", SPATIAL_VAL_XFORM)
    print("\nPREPROCESSING_DOC:")
    for k, v in PREPROCESSING_DOC.items():
        print("  {}: {}".format(k, v))
