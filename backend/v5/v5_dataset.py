"""
AIDetect V5 Dataset Module
==========================
PyTorch Dataset implementation for the 60,000-sample V5 Master Robustness Dataset.
Loads samples deterministically from v5_metadata/v5_dataset_manifest.csv across:
  - split="train" (42,000 images)
  - split="validation" (9,000 images)
  - split="test" (9,000 images)

Features:
  - Supports file-backed samples (dataset_v4) and memory/parquet-backed samples (Tiny-GenImage, CIFAKE)
  - Authoritative native-resolution FFT frequency spectrum preprocessing
  - Noise residual preprocessing
  - Deterministic label convention: 0 = REAL, 1 = AI
"""
def load_original_image(self, index):
    """
    Return the original PIL image for robustness experiments.
    """
    return self._load_image(index)

import os
import io
import csv
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_DIR / "v5_metadata" / "v5_dataset_manifest.csv"

# Label Convention
CLASS_MAPPING = {"REAL": 0, "AI": 1}
IDX_TO_CLASS = {0: "REAL", 1: "AI"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------------

class JPEGSimulation:
    def __init__(self, min_q=70, max_q=95, p=0.3):
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

def build_v5_spatial_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        JPEGSimulation(min_q=70, max_q=95, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_v5_spatial_val_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class AuthoritativeNativeFFTTransform:
    """Authoritative native-resolution 2D FFT magnitude transform."""
    def __init__(self):
        self._to_tensor = transforms.ToTensor()
        self._norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        x = self._to_tensor(image) # [3, H, W] native
        fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        # per-channel min-max
        for c in range(mag.shape[0]):
            ch = mag[c]
            mi, ma = ch.min(), ch.max()
            mag[c] = (ch - mi) / (ma - mi + 1e-8)
        mag_224 = F.interpolate(mag.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        return self._norm(mag_224)

# ---------------------------------------------------------------------------
# V5 DATASET CLASS
# ---------------------------------------------------------------------------

class V5Dataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        mode: str = "spatial", # 'spatial', 'frequency', 'hybrid', 'multi_expert'
        manifest_path: Path = MANIFEST_PATH,
        spatial_transform = None,
        fft_transform = None,
    ):
        self.split = split.lower()
        self.mode = mode.lower()
        self.spatial_xform = spatial_transform or (
            build_v5_spatial_train_transform() if self.split == "train" else build_v5_spatial_val_transform()
        )
        self.fft_xform = fft_transform or AuthoritativeNativeFFTTransform()

        # Load samples from manifest
        self.samples = []
        if not manifest_path.exists():
            raise FileNotFoundError(f"V5 manifest not found at {manifest_path}")

        # Cached parquets to avoid reloading
        self._parquet_cache = {}

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"].lower() == self.split:
                    self.samples.append(row)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rec: Dict[str, Any]) -> Image.Image:
        stype = rec["storage_type"]
        sref = rec["storage_ref"]
        if stype == "file":
            full_p = PROJECT_DIR / sref
            with Image.open(full_p) as im:
                return im.convert("RGB")
        elif stype == "parquet_row":
            pq_name, r_idx_str = sref.split(":")
            r_idx = int(r_idx_str)
            # Find parquet path
            if pq_name not in self._parquet_cache:
                for cand in Path(r"C:\Users\Shreyas\.cache\huggingface\hub").rglob(pq_name):
                    if cand.is_file():
                        self._parquet_cache[pq_name] = pq.ParquetFile(cand)
                        break

            parquet_file = self._parquet_cache.get(pq_name)

            if parquet_file is None:
                raise FileNotFoundError(
                    f"Could not locate cached parquet: {pq_name}"
                )

            # Read only the required row instead of loading the entire parquet
            # into pandas memory.
            table = parquet_file.read_row_groups(
                list(range(
                    0,
                    parquet_file.num_row_groups
                ))
            )

            row = table.slice(
                r_idx,
                1
            ).to_pylist()[0]

            raw_bytes = row["image"]["bytes"]
            with Image.open(io.BytesIO(raw_bytes)) as im:
                return im.convert("RGB")
        else:
            raise ValueError(f"Unknown storage type: {stype}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.samples[index]
        img = self._load_image(rec)
        label = CLASS_MAPPING[rec["ground_truth"]]

        item = {
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": rec["sample_id"],
            "domain": rec["domain"]
        }

        if self.mode in ("spatial", "hybrid", "multi_expert"):
            item["spatial"] = self.spatial_xform(img)
        if self.mode in ("frequency", "hybrid", "multi_expert"):
            item["frequency"] = self.fft_xform(img)

        return item

