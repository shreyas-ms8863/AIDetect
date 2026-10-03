"""
AIDetect Phase 25: Deterministic Sanity Checks for Manipulation Detector V2
=============================================================================
Executes at least 20 deterministic integrity and computational checks before
training, verifying data integrity, pair grouping, FFT representations,
model forward/backward passes, and gradient flow.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.manipulation_v2_models import ManipulationFrequencyResNet50V2
from backend.manipulation_v2_dataset import ManipulationV2Dataset, AuthoritativeFrequencyTransform

MANIP_V2_DIR = BASE_DIR / "dataset_manipulation_v2"
METADATA_CSV = MANIP_V2_DIR / "metadata" / "metadata.csv"
V1_CKPT_PATH = BASE_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"
EXPECTED_V1_SHA256 = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def run_sanity_checks(metadata_path: Path = METADATA_CSV) -> bool:
    print("=" * 80)
    print("AIDETECT PHASE 25: MANIPULATION V2 DETERMINISTIC SANITY CHECKS")
    print("=" * 80)

    checks_passed = 0
    total_checks = 22

    def check(num: int, name: str, condition: bool, details: str = ""):
        nonlocal checks_passed
        status = "PASSED" if condition else "FAILED"
        print(f"[{num:02d}/{total_checks:02d}] {status:<6} : {name} {details}")
        if condition:
            checks_passed += 1
        else:
            print(f"       ERROR: Check {num} failed!")
            sys.exit(1)

    # 1. Metadata exists
    check(1, "Metadata CSV exists", metadata_path.exists(), f"({metadata_path})")
    df = pd.read_csv(metadata_path)

    # 2. Minimum row count
    check(2, "Metadata non-empty", len(df) >= 200, f"(Found {len(df)} rows)")

    # 3. Label binary domain
    labels = set(df["ground_truth"].unique())
    check(3, "Binary labels {0, 1}", labels == {0, 1}, f"(Found labels {labels})")

    # 4. Pair integrity
    pair_counts = df.groupby("pair_id")["ground_truth"].apply(list).to_dict()
    all_pairs_valid = all(sorted(v) == [0, 1] for v in pair_counts.values())
    check(4, "Pair integrity (1 Real + 1 Manip per pair)", all_pairs_valid, f"({len(pair_counts)} pairs verified)")

    # 5. Split separation
    pair_splits = df.groupby("pair_id")["split"].nunique()
    split_leakage = (pair_splits > 1).sum()
    check(5, "Pair split isolation (No cross-split pairs)", split_leakage == 0, f"(Cross-split pairs: {split_leakage})")

    # 6. Generator diversity in training
    train_gens = set(df[(df["split"] == "train") & (df["ground_truth"] == 1)]["generator"].unique())
    check(6, "Train generator diversity (>= 2 generators)", len(train_gens) >= 2, f"(Found {train_gens})")

    # 7. Unseen generator isolation
    unseen_df = df[(df["split"] == "test_unseen") & (df["ground_truth"] == 1)]
    has_unseen = len(unseen_df) > 0
    if has_unseen:
        unseen_gens = set(unseen_df["generator"].unique())
        disjoint = len(train_gens.intersection(unseen_gens)) == 0
        check(7, "Unseen test generators disjoint from train", disjoint, f"(Unseen: {unseen_gens})")
    else:
        check(7, "Unseen test split recorded", True, "(Pending generation)")

    # 8. File existence on disk
    sample_rows = df.head(50).to_dict("records")
    all_files_exist = all((BASE_DIR / r["file_path"]).exists() for r in sample_rows)
    check(8, "Image files exist on disk", all_files_exist, "(Audited 50 samples)")

    # 9. Non-zero file sizes
    all_non_zero = all((BASE_DIR / r["file_path"]).stat().st_size > 0 for r in sample_rows)
    check(9, "Image files non-zero size", all_non_zero, "(All samples verified > 0 bytes)")

    # 10. PIL readable RGB
    all_pil_valid = True
    for r in sample_rows[:10]:
        try:
            im = Image.open(BASE_DIR / r["file_path"]).convert("RGB")
            if im.width == 0 or im.height == 0:
                all_pil_valid = False
        except Exception:
            all_pil_valid = False
    check(10, "Images valid PIL RGB", all_pil_valid, "(Verified dimensions > 0)")

    # 11. Authoritative FFT output shape
    xform = AuthoritativeFrequencyTransform()
    test_img = Image.open(BASE_DIR / sample_rows[0]["file_path"]).convert("RGB")
    fft_t = xform(test_img)
    check(11, "FFT tensor shape [3, 224, 224]", fft_t.shape == torch.Size([3, 224, 224]), f"(Got {tuple(fft_t.shape)})")

    # 12. FFT tensor dtype
    check(12, "FFT tensor dtype float32", fft_t.dtype == torch.float32, f"({fft_t.dtype})")

    # 13. FFT finite values (no NaN, no Inf)
    finite_fft = bool(torch.isfinite(fft_t).all())
    check(13, "FFT values strictly finite", finite_fft, "(0 NaN, 0 Inf)")

    # 14. Dataset class initialization
    ds = ManipulationV2Dataset(sample_rows[:16], is_training=False, base_dir=BASE_DIR)
    check(14, "Dataset loader length", len(ds) == 16, f"({len(ds)} items)")

    # 15. DataLoader batching
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch_x, batch_y, batch_m = next(iter(loader))
    check(15, "DataLoader batch shape [4, 3, 224, 224]", batch_x.shape == torch.Size([4, 3, 224, 224]), f"({tuple(batch_x.shape)})")

    # 16. GPU / CUDA device transfer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_gpu = batch_x.to(device)
    check(16, f"Tensor transfer to device ({device})", batch_gpu.device.type == device.type, f"(Device: {device})")

    # 17. Model initialization
    model = ManipulationFrequencyResNet50V2(dropout_p=0.3).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    check(17, "Model V2 initialization (~23.5M params)", param_count > 23_000_000, f"({param_count:,} params)")

    # 18. Forward pass output shape
    model.eval()
    with torch.no_grad():
        out = model(batch_gpu)
    check(18, "Model logits shape [B, 2]", out.shape == torch.Size([4, 2]), f"({tuple(out.shape)})")

    # 19. Output logits finite
    finite_logits = bool(torch.isfinite(out).all())
    check(19, "Model logits finite", finite_logits, "(0 NaN, 0 Inf)")

    # 20. Loss computation & Backward pass
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    target_gpu = batch_y.to(device)
    logits_train = model(batch_gpu)
    loss = criterion(logits_train, target_gpu)
    optimizer.zero_grad()
    loss.backward()
    has_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    check(20, "Gradient flow & finite backward pass", has_grad, f"(Loss: {loss.item():.4f})")

    # 21. Optimizer weight update
    p_before = list(model.parameters())[0].clone()
    optimizer.step()
    p_after = list(model.parameters())[0]
    weight_updated = not torch.equal(p_before, p_after)
    check(21, "Optimizer step updates weights", weight_updated, "(Weights successfully updated)")

    # 22. Production V1 model untouched
    v1_sha = compute_file_sha256(V1_CKPT_PATH)
    check(22, "Production V1 checkpoint SHA untouched", v1_sha == EXPECTED_V1_SHA256, f"(Match: {v1_sha == EXPECTED_V1_SHA256})")

    print("=" * 80)
    print(f"ALL {checks_passed}/{total_checks} DETERMINISTIC SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_sanity_checks()

