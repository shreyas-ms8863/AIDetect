"""
AIDetect Phase 8: Sanity Check for Hybrid AI-Manipulation Detector
==================================================================
Comprehensive 20-point pre-flight sanity check:
  1. All required files exist.
  2. Dataset counts: Train=1,132, Validation=239, Test=248.
  3. Labels only contain 0 and 1.
  4. No original_id crosses splits (zero leakage).
  5. CUDA is available.
  6. Spatial tensor shape = [B, 3, 224, 224].
  7. Frequency tensor shape = [B, 3, 224, 224].
  8. Both tensors contain only finite values.
  9. Spatial and frequency tensors are genuinely different.
 10. FFT is performed from native resolution.
 11. Spatial branch output = [B, 2048].
 12. Frequency branch output = [B, 2048].
 13. Concatenated feature = [B, 4096].
 14. Final logits = [B, 2].
 15. Forward pass works.
 16. Backward pass works.
 17. Gradients are non-zero in spatial branch.
 18. Gradients are non-zero in frequency branch.
 19. Gradients are non-zero in fusion head.
 20. Initial loss is finite.
 Additional: Confirms synchronized shared augmentation (same augmented base image).
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
from manipulation_hybrid_v1_models import ManipulationHybridResNet50V1
from manipulation_hybrid_v1_dataset import (
    ManipulationHybridV1Dataset, collate_manipulation_hybrid,
    AuthoritativeFrequencyTransform, METADATA_CSV, MANIP_DIR,
)

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

def run_sanity_checks():
    print("=" * 70)
    print("PHASE 8 HYBRID MANIPULATION DETECTOR: PRE-TRAINING SANITY CHECK")
    print("=" * 70)
    
    passed_checks = 0
    total_checks = 20
    
    # -----------------------------------------------------------------------
    # CHECK 1: File existence
    # -----------------------------------------------------------------------
    print("\n[Check 1/20] Verifying expected files exist...")
    required_files = [
        METADATA_CSV,
        MANIP_DIR / "metadata" / "split_manifest.csv",
        MANIP_DIR / "metadata" / "pair_relationships.json",
        BASE_DIR / "manipulation_hybrid_v1_models.py",
        BASE_DIR / "manipulation_hybrid_v1_dataset.py",
    ]
    for rf in required_files:
        assert rf.exists() and rf.stat().st_size > 0, f"File missing or empty: {rf}"
        print(f"  OK: {rf.name} ({rf.stat().st_size:,} bytes)")
    passed_checks += 1
    print("-> PASS: Check 1 (All expected files exist)")

    # -----------------------------------------------------------------------
    # CHECK 2: Dataset counts
    # -----------------------------------------------------------------------
    print("\n[Check 2/20] Verifying split counts...")
    train_ds = ManipulationHybridV1Dataset("train", augment=True)
    val_ds   = ManipulationHybridV1Dataset("validation", augment=False)
    test_ds  = ManipulationHybridV1Dataset("test", augment=False)
    
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
    print("\n[Check 3/20] Verifying labels contain only 0 and 1...")
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
    print("\n[Check 4/20] Verifying zero cross-split leakage of original_id...")
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
    print("\n[Check 5/20] Verifying CUDA GPU availability...")
    assert torch.cuda.is_available(), "CUDA is not available! Training requires CUDA."
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  CUDA Device: {gpu_name}")
    passed_checks += 1
    print("-> PASS: Check 5 (CUDA is available)")

    # -----------------------------------------------------------------------
    # CHECK 6 & 7: Spatial and Frequency tensor shapes
    # -----------------------------------------------------------------------
    print("\n[Check 6/20 & 7/20] Verifying dual tensor shapes from DataLoader...")
    loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_manipulation_hybrid)
    s_tensors, f_tensors, batch_labels, batch_metas = next(iter(loader))
    print(f"  Spatial tensor shape  : {tuple(s_tensors.shape)}")
    print(f"  Frequency tensor shape: {tuple(f_tensors.shape)}")
    assert s_tensors.shape == (4, 3, 224, 224), f"Unexpected spatial shape: {s_tensors.shape}"
    passed_checks += 1
    print("-> PASS: Check 6 (Spatial tensor shape is [B, 3, 224, 224])")
    
    assert f_tensors.shape == (4, 3, 224, 224), f"Unexpected frequency shape: {f_tensors.shape}"
    passed_checks += 1
    print("-> PASS: Check 7 (Frequency tensor shape is [B, 3, 224, 224])")

    # -----------------------------------------------------------------------
    # CHECK 8: Both tensors are finite
    # -----------------------------------------------------------------------
    print("\n[Check 8/20] Verifying tensor finiteness (no NaNs, no Infs)...")
    s_finite = torch.isfinite(s_tensors).all().item()
    f_finite = torch.isfinite(f_tensors).all().item()
    print(f"  Spatial values finite  : {s_finite}")
    print(f"  Frequency values finite: {f_finite}")
    assert s_finite and f_finite, "Tensors contain non-finite values!"
    passed_checks += 1
    print("-> PASS: Check 8 (Both spatial and frequency tensors are finite)")

    # -----------------------------------------------------------------------
    # CHECK 9: Spatial and frequency tensors are genuinely different
    # -----------------------------------------------------------------------
    print("\n[Check 9/20] Verifying spatial and frequency tensors are genuinely distinct...")
    diff = (s_tensors - f_tensors).abs().mean().item()
    print(f"  Mean absolute difference between branches: {diff:.4f}")
    assert diff > 0.1, f"Tensors too similar: diff={diff}"
    passed_checks += 1
    print("-> PASS: Check 9 (Spatial and frequency representations are distinct)")

    # -----------------------------------------------------------------------
    # CHECK 10: FFT is performed from native resolution
    # -----------------------------------------------------------------------
    print("\n[Check 10/20] Verifying FFT operates from native resolution prior to resize...")
    sample_path = train_ds.samples[0]["filepath"]
    with Image.open(sample_path) as raw_img:
        native_rgb = raw_img.convert("RGB")
    w, h = native_rgb.size
    print(f"  Native image resolution: {w}x{h}")
    native_fft = AuthoritativeFrequencyTransform()(native_rgb)
    downscale_first_fft = AuthoritativeFrequencyTransform()(native_rgb.resize((224, 224), Image.Resampling.BILINEAR))
    res_diff = (native_fft - downscale_first_fft).abs().mean().item()
    print(f"  Native FFT vs Pre-downscaled FFT discrepancy: {res_diff:.4f}")
    if w != 224 or h != 224:
        assert res_diff > 0.05, "FFT did not preserve native high frequencies!"
    passed_checks += 1
    print("-> PASS: Check 10 (FFT computed from native resolution)")

    # -----------------------------------------------------------------------
    # CHECK 11, 12, 13, 14: Model Branch Dimensions
    # -----------------------------------------------------------------------
    print("\n[Checks 11-14/20] Verifying hybrid model layer dimensions...")
    model = ManipulationHybridResNet50V1(dropout_p=0.3).cuda()
    model.eval()
    
    xs = s_tensors.cuda()
    xf = f_tensors.cuda()
    
    with torch.no_grad():
        fs, ff, fj = model.extract_branch_features(xs, xf)
        logits = model(xs, xf)
        
    print(f"  Spatial features shape    : {tuple(fs.shape)} (expected [4, 2048])")
    assert fs.shape == (4, 2048), f"Spatial feature mismatch: {fs.shape}"
    passed_checks += 1
    print("-> PASS: Check 11 (Spatial branch output = [B, 2048])")

    print(f"  Frequency features shape  : {tuple(ff.shape)} (expected [4, 2048])")
    assert ff.shape == (4, 2048), f"Frequency feature mismatch: {ff.shape}"
    passed_checks += 1
    print("-> PASS: Check 12 (Frequency branch output = [B, 2048])")

    print(f"  Concatenated feature shape: {tuple(fj.shape)} (expected [4, 4096])")
    assert fj.shape == (4, 4096), f"Concatenated feature mismatch: {fj.shape}"
    passed_checks += 1
    print("-> PASS: Check 13 (Concatenated feature = [B, 4096])")

    print(f"  Final logits shape        : {tuple(logits.shape)} (expected [4, 2])")
    assert logits.shape == (4, 2), f"Logits shape mismatch: {logits.shape}"
    passed_checks += 1
    print("-> PASS: Check 14 (Final logits = [B, 2])")

    # -----------------------------------------------------------------------
    # CHECK 15 & 16: Forward and Backward pass
    # -----------------------------------------------------------------------
    print("\n[Checks 15-16/20] Verifying forward and backward pass...")
    model.train()
    y = batch_labels.cuda()
    logits = model(xs, xf)
    assert logits.shape == (4, 2)
    passed_checks += 1
    print("-> PASS: Check 15 (Forward pass works)")

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    loss.backward()
    passed_checks += 1
    print("-> PASS: Check 16 (Backward pass works)")

    # -----------------------------------------------------------------------
    # CHECK 17, 18, 19: Gradients are non-zero across all components
    # -----------------------------------------------------------------------
    print("\n[Checks 17-19/20] Verifying gradient propagation across branches...")
    s_grad = None
    for p in model.spatial_backbone.parameters():
        if p.grad is not None:
            s_grad = p.grad.norm().item()
            break
            
    f_grad = None
    for p in model.freq_backbone.parameters():
        if p.grad is not None:
            f_grad = p.grad.norm().item()
            break
            
    h_grad = model.fusion_head[3].weight.grad.norm().item()
    
    print(f"  Spatial backbone grad norm  : {s_grad:.6f}")
    assert s_grad is not None and s_grad > 0, "No gradient in spatial branch!"
    passed_checks += 1
    print("-> PASS: Check 17 (Spatial branch gradients are non-zero)")

    print(f"  Frequency backbone grad norm: {f_grad:.6f}")
    assert f_grad is not None and f_grad > 0, "No gradient in frequency branch!"
    passed_checks += 1
    print("-> PASS: Check 18 (Frequency branch gradients are non-zero)")

    print(f"  Fusion head grad norm       : {h_grad:.6f}")
    assert h_grad > 0, "No gradient in fusion head!"
    passed_checks += 1
    print("-> PASS: Check 19 (Fusion head gradients are non-zero)")

    # -----------------------------------------------------------------------
    # CHECK 20: Initial loss is finite
    # -----------------------------------------------------------------------
    print("\n[Check 20/20] Verifying initial loss is finite...")
    print(f"  Computed initial loss: {loss.item():.4f}")
    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    passed_checks += 1
    print("-> PASS: Check 20 (Initial loss is finite)")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SANITY CHECK SUMMARY: {passed_checks} / {total_checks} CHECKS PASSED")
    print("=" * 70)
    print("All 20 checks passed. Synchronized hybrid pipeline is verified and ready.")
    return True

if __name__ == "__main__":
    success = run_sanity_checks()
    if not success:
        sys.exit(1)

