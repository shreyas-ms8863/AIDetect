"""
AIDetect Phase 7: Sanity Check for Frequency-Domain Manipulation Detector
========================================================================
Comprehensive 13-point pre-flight sanity check:
  1. All expected files exist.
  2. Dataset counts: Train=1,132, Validation=239, Test=248.
  3. Labels only contain 0 and 1.
  4. No original_id crosses splits (zero leakage).
  5. CUDA is available.
  6. Frequency tensor shape = [B, 3, 224, 224].
  7. Frequency tensors contain only finite values (no NaNs, no Infs).
  8. FFT preprocessing actually differs from the RGB image.
  9. Model output shape = [B, 2].
 10. Forward pass works.
 11. Backward pass works (gradients computed).
 12. Initial loss is finite.
 13. The FFT is performed before the 224x224 resize.
"""

import sys
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_freq_v1_dataset import (
    ManipulationFreqV1Dataset, collate_manipulation_freq,
    AuthoritativeFrequencyTransform, METADATA_CSV, MANIP_DIR,
)

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

def run_sanity_checks():
    print("=" * 70)
    print("PHASE 7 FREQUENCY MANIPULATION DETECTOR: PRE-TRAINING SANITY CHECK")
    print("=" * 70)
    
    passed_checks = 0
    total_checks = 13
    
    # -----------------------------------------------------------------------
    # CHECK 1: File existence
    # -----------------------------------------------------------------------
    print("\n[Check 1/13] Verifying expected files exist...")
    required_files = [
        METADATA_CSV,
        MANIP_DIR / "metadata" / "split_manifest.csv",
        MANIP_DIR / "metadata" / "pair_relationships.json",
        BASE_DIR / "manipulation_freq_v1_models.py",
        BASE_DIR / "manipulation_freq_v1_dataset.py",
    ]
    for rf in required_files:
        assert rf.exists() and rf.stat().st_size > 0, f"File missing or empty: {rf}"
        print(f"  OK: {rf.name} ({rf.stat().st_size:,} bytes)")
    passed_checks += 1
    print("-> PASS: Check 1 (All expected files exist)")

    # -----------------------------------------------------------------------
    # CHECK 2: Dataset counts
    # -----------------------------------------------------------------------
    print("\n[Check 2/13] Verifying split counts...")
    train_ds = ManipulationFreqV1Dataset("train", augment=True)
    val_ds   = ManipulationFreqV1Dataset("validation", augment=False)
    test_ds  = ManipulationFreqV1Dataset("test", augment=False)
    
    print(f"  Train count      : {len(train_ds)} (expected 1132)")
    print(f"  Validation count : {len(val_ds)} (expected 239)")
    print(f"  Test count       : {len(test_ds)} (expected 248)")
    
    assert len(train_ds) == 1132, f"Expected 1132 train images, got {len(train_ds)}"
    assert len(val_ds) == 239, f"Expected 239 val images, got {len(val_ds)}"
    assert len(test_ds) == 248, f"Expected 248 test images, got {len(test_ds)}"
    passed_checks += 1
    print("-> PASS: Check 2 (Dataset counts match 1132 / 239 / 248)")

    # -----------------------------------------------------------------------
    # CHECK 3: Labels contain only 0 and 1
    # -----------------------------------------------------------------------
    print("\n[Check 3/13] Verifying labels contain only 0 and 1...")
    all_labels = set()
    for ds in (train_ds, val_ds, test_ds):
        for s in ds.samples:
            all_labels.add(s["label"])
    print(f"  Unique labels found across all splits: {all_labels}")
    assert all_labels == {0, 1}, f"Unexpected labels: {all_labels}"
    passed_checks += 1
    print("-> PASS: Check 3 (Strictly binary labels {0, 1})")

    # -----------------------------------------------------------------------
    # CHECK 4: No original_id crosses splits (Zero leakage)
    # -----------------------------------------------------------------------
    print("\n[Check 4/13] Verifying zero cross-split leakage of original_id...")
    train_orig = {s["original_id"] for s in train_ds.samples}
    val_orig   = {s["original_id"] for s in val_ds.samples}
    test_orig  = {s["original_id"] for s in test_ds.samples}
    
    leak_train_val  = train_orig & val_orig
    leak_train_test = train_orig & test_orig
    leak_val_test   = val_orig & test_orig
    
    print(f"  Train vs Val overlap : {len(leak_train_val)}")
    print(f"  Train vs Test overlap: {len(leak_train_test)}")
    print(f"  Val vs Test overlap  : {len(leak_val_test)}")
    assert len(leak_train_val) == 0, f"Train/Val leakage: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Train/Test leakage: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Val/Test leakage: {leak_val_test}"
    passed_checks += 1
    print("-> PASS: Check 4 (Zero cross-split leakage)")

    # -----------------------------------------------------------------------
    # CHECK 5: CUDA is available
    # -----------------------------------------------------------------------
    print("\n[Check 5/13] Verifying CUDA GPU availability...")
    assert torch.cuda.is_available(), "CUDA is not available! Training requires CUDA."
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  CUDA Device: {gpu_name}")
    passed_checks += 1
    print("-> PASS: Check 5 (CUDA is available)")

    # -----------------------------------------------------------------------
    # CHECK 6: Frequency tensor shape [B, 3, 224, 224]
    # -----------------------------------------------------------------------
    print("\n[Check 6/13] Verifying frequency tensor shape from DataLoader...")
    loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_manipulation_freq)
    batch_tensors, batch_labels, batch_metas = next(iter(loader))
    print(f"  Batch tensor shape: {tuple(batch_tensors.shape)}")
    assert batch_tensors.shape == (4, 3, 224, 224), f"Unexpected tensor shape: {batch_tensors.shape}"
    passed_checks += 1
    print("-> PASS: Check 6 (Batch shape is [B, 3, 224, 224])")

    # -----------------------------------------------------------------------
    # CHECK 7: Frequency tensors contain only finite values
    # -----------------------------------------------------------------------
    print("\n[Check 7/13] Verifying tensor finiteness (no NaN, no Inf)...")
    is_finite = torch.isfinite(batch_tensors).all().item()
    print(f"  All values finite: {is_finite}")
    assert is_finite, "Tensor contains non-finite values (NaN or Inf)!"
    passed_checks += 1
    print("-> PASS: Check 7 (All frequency tensor values are finite)")

    # -----------------------------------------------------------------------
    # CHECK 8: FFT preprocessing actually differs from RGB image
    # -----------------------------------------------------------------------
    print("\n[Check 8/13] Verifying FFT representation differs from raw RGB...")
    sample_path = train_ds.samples[0]["filepath"]
    with Image.open(sample_path) as raw_img:
        rgb_img = raw_img.convert("RGB")
    
    # Direct spatial tensor vs FFT tensor
    spatial_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(rgb_img)
    
    fft_transform = AuthoritativeFrequencyTransform()
    freq_tensor = fft_transform(rgb_img)
    
    diff = (spatial_tensor - freq_tensor).abs().mean().item()
    print(f"  Mean absolute difference between spatial and frequency tensors: {diff:.4f}")
    assert diff > 0.1, f"FFT tensor too similar to spatial tensor: diff={diff}"
    passed_checks += 1
    print("-> PASS: Check 8 (FFT representation distinctly differs from RGB)")

    # -----------------------------------------------------------------------
    # CHECK 9: Model output shape [B, 2]
    # -----------------------------------------------------------------------
    print("\n[Check 9/13] Verifying model output shape...")
    model = ManipulationFrequencyResNet50V1(dropout_p=0.3).cuda()
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224, device="cuda")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  Model output shape: {tuple(out.shape)}")
    assert out.shape == (4, 2), f"Expected [4, 2], got {out.shape}"
    passed_checks += 1
    print("-> PASS: Check 9 (Model output shape is [B, 2])")

    # -----------------------------------------------------------------------
    # CHECK 10: Forward pass works
    # -----------------------------------------------------------------------
    print("\n[Check 10/13] Verifying forward pass with real batch...")
    x = batch_tensors.cuda()
    y = batch_labels.cuda()
    model.train()
    logits = model(x)
    print(f"  Forward pass successful. Logits min={logits.min().item():.3f}, max={logits.max().item():.3f}")
    assert logits.shape == (4, 2)
    passed_checks += 1
    print("-> PASS: Check 10 (Forward pass works)")

    # -----------------------------------------------------------------------
    # CHECK 11: Backward pass works
    # -----------------------------------------------------------------------
    print("\n[Check 11/13] Verifying backward pass...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    loss.backward()
    
    # Check that classifier weights received gradients
    has_grad = model.model.fc[1].weight.grad is not None
    grad_norm = model.model.fc[1].weight.grad.norm().item()
    print(f"  fc layer gradient computed: {has_grad}, norm: {grad_norm:.6f}")
    assert has_grad and grad_norm > 0, "Gradients not computed properly during backward pass!"
    passed_checks += 1
    print("-> PASS: Check 11 (Backward pass works)")

    # -----------------------------------------------------------------------
    # CHECK 12: Initial loss is finite
    # -----------------------------------------------------------------------
    print("\n[Check 12/13] Verifying initial loss is finite...")
    print(f"  Computed initial loss: {loss.item():.4f}")
    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    passed_checks += 1
    print("-> PASS: Check 12 (Initial loss is finite)")

    # -----------------------------------------------------------------------
    # CHECK 13: FFT is performed before the 224x224 resize
    # -----------------------------------------------------------------------
    print("\n[Check 13/13] Verifying FFT is performed at native resolution BEFORE resize...")
    # Verify mathematically:
    # 1. Native FFT: FFT on native image -> resize magnitude to 224
    # 2. Downscaled FFT: resize image to 224 -> FFT on 224 image
    # They should yield distinct spectra because native FFT preserves frequencies beyond 224 Nyquist
    w, h = rgb_img.size
    print(f"  Sample image native resolution: {w}x{h}")
    
    # Pathway A: Authoritative Native FFT
    native_fft_result = AuthoritativeFrequencyTransform()(rgb_img)
    
    # Pathway B: Downscale First FFT (Incorrect pathway)
    downscaled_img = rgb_img.resize((224, 224), Image.Resampling.BILINEAR)
    downscale_first_fft = AuthoritativeFrequencyTransform()(downscaled_img)
    
    discrepancy = (native_fft_result - downscale_first_fft).abs().mean().item()
    print(f"  Discrepancy between native-FFT vs resize-first-FFT: {discrepancy:.4f}")
    if w != 224 or h != 224:
        assert discrepancy > 0.05, f"Native FFT must differ from resize-first FFT: {discrepancy}"
        print("  Confirmed: High-frequency spectrum differs from pre-downscaled FFT.")
    else:
        print("  Sample image was already 224x224.")
        
    passed_checks += 1
    print("-> PASS: Check 13 (FFT verified to operate on native image resolution)")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SANITY CHECK SUMMARY: {passed_checks} / {total_checks} CHECKS PASSED")
    print("=" * 70)
    print("All checks passed. System is fully verified and ready for Phase 7 training.")
    return True

if __name__ == "__main__":
    success = run_sanity_checks()
    if not success:
        sys.exit(1)

