"""
AIDetect V4 Sanity Check
=========================
Verifies all V4 components before launching a long training run:

1.  V3 checkpoints exist (not modified)
2.  V4 dataset metadata and image files are accessible
3.  Label convention: 0=REAL, 1=AI
4.  Dataset counts match expected (2837 train, 607 val, 613 test)
5.  Domain counts verified
6.  Spatial transform: produces [3, 224, 224] float32 tensor
7.  FFT transform: produces [3, 224, 224] float32 tensor at correct range
8.  FFT is computed at native resolution (not 224x224 pre-resize)
9.  Training augmentation vs. val/test transform verified
10. One batch from Spatial train DataLoader — shapes, labels, domain strings
11. One batch from Frequency train DataLoader — shapes, labels
12. One batch from Hybrid train DataLoader — shapes (spatial+freq+label+domain)
13. SpatialV4 forward pass — output shape [B, 2]
14. FrequencyV4 forward pass — output shape [B, 2]
15. HybridV4 forward pass — output shape [B, 2]
16. Hybrid feature dimensions: 2048 + 256 = 2304
17. CUDA is being used
18. V4 checkpoint paths (not yet saved — just verifies directory)
19. v4_results/ directory structure created

All checks print PASS / FAIL.
"""

import sys
import os
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_dataset import (
    V4Dataset, collate_spatial, collate_hybrid,
    FFT_TRANSFORM, SPATIAL_TRAIN_XFORM, SPATIAL_VAL_XFORM,
    load_metadata, CLASS_MAPPING, IDX_TO_CLASS,
    PREPROCESSING_DOC,
)
from v4_models import (
    SpatialV4, FrequencyV4, HybridV4,
    SPATIAL_FEATURE_DIM, FREQUENCY_FEATURE_DIM, FUSED_DIM,
)

from PIL import Image

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR   = PROJECT_DIR / "models"

PASS_STR = "  [PASS]"
FAIL_STR = "  [FAIL]"

results = []

def check(name: str, ok: bool, detail: str = ""):
    status = PASS_STR if ok else FAIL_STR
    msg    = f"{status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, ok, detail))
    return ok

print("=" * 70)
print("AIDETECT V4 SANITY CHECK")
print("=" * 70)
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. CUDA check -----------------------------------------------------------
print("--- 1. Device ---")
check("CUDA is available", torch.cuda.is_available(),
      f"device={DEVICE}")
check("Device is CUDA (not CPU)", DEVICE.type == "cuda",
      f"type={DEVICE.type}")

# --- 2. V3 checkpoints intact ------------------------------------------------
print("\n--- 2. V3 Checkpoint Preservation ---")
for fname in ["spatial_resnet50_v3.pth", "frequency_resnet50_v3.pth", "hybrid_resnet50_fft_v3.pth"]:
    p = MODEL_DIR / fname
    check(f"V3 exists: {fname}", p.exists(), f"size={p.stat().st_size/1e6:.1f}MB" if p.exists() else "MISSING")

# --- 3. V4 dataset counts ----------------------------------------------------
print("\n--- 3. V4 Dataset Counts ---")
expected = {"train": 2837, "validation": 607, "test": 613}
for split, exp_count in expected.items():
    rows = load_metadata(split)
    check(f"{split} count matches expected",
          len(rows) == exp_count,
          f"got={len(rows)}, expected={exp_count}")

# --- 4. Label convention -----------------------------------------------------
print("\n--- 4. Label Convention (0=REAL, 1=AI) ---")
check("REAL → 0", CLASS_MAPPING["REAL"] == 0, f"CLASS_MAPPING['REAL']={CLASS_MAPPING['REAL']}")
check("AI   → 1", CLASS_MAPPING["AI"]   == 1, f"CLASS_MAPPING['AI']={CLASS_MAPPING['AI']}")
check("idx 0 → REAL", IDX_TO_CLASS[0] == "REAL")
check("idx 1 → AI",   IDX_TO_CLASS[1] == "AI")

# Verify from dataset
train_rows = load_metadata("train")
real_rows  = [r for r in train_rows if r["ground_truth"] == "REAL"]
ai_rows    = [r for r in train_rows if r["ground_truth"] == "AI"]
check("Train REAL count > 0", len(real_rows) > 0, f"n_real={len(real_rows)}")
check("Train AI count > 0",   len(ai_rows)   > 0, f"n_ai={len(ai_rows)}")

# --- 5. Domain counts --------------------------------------------------------
print("\n--- 5. Domain Counts (Train) ---")
# Expected per-class domain counts (REAL and AI have separate domain namespaces)
# genimage: 1050 REAL + 1050 AI = 2100 total in train
# modern:    244 REAL +  244 AI =  488 total in train
# historical: 74 REAL only
# historical_style: 175 AI only
expected_domains_real = {"genimage": 1050, "historical": 74, "modern": 244}
expected_domains_ai   = {"genimage": 1050, "historical_style": 175, "modern": 244}
for domain, exp in expected_domains_real.items():
    cnt = sum(1 for r in real_rows if r["domain"] == domain)
    check(f"Train REAL domain '{domain}'", cnt == exp, f"got={cnt}, expected={exp}")
for domain, exp in expected_domains_ai.items():
    cnt = sum(1 for r in ai_rows if r["domain"] == domain)
    check(f"Train AI domain '{domain}'", cnt == exp, f"got={cnt}, expected={exp}")

# --- 6. Spatial transform output shape ---------------------------------------
print("\n--- 6. Spatial Transform Shape & Type ---")
try:
    # Use a real image from the dataset
    sample_path, _, _ = V4Dataset("train", mode="spatial", augment=False).samples[0]
    img = Image.open(sample_path).convert("RGB")
    orig_w, orig_h = img.size
    check("Sample image loads (PIL)", True, f"original_size=({orig_w}x{orig_h})")

    t_train = SPATIAL_TRAIN_XFORM(img)
    check("Spatial train shape [3,224,224]",
          tuple(t_train.shape) == (3, 224, 224), f"shape={tuple(t_train.shape)}")
    check("Spatial train dtype float32", t_train.dtype == torch.float32, f"dtype={t_train.dtype}")

    t_val = SPATIAL_VAL_XFORM(img)
    check("Spatial val shape [3,224,224]",
          tuple(t_val.shape) == (3, 224, 224), f"shape={tuple(t_val.shape)}")
    check("Spatial val dtype float32", t_val.dtype == torch.float32, f"dtype={t_val.dtype}")

    # Train augmentation is stochastic — run twice and check shapes are consistent
    t_train2 = SPATIAL_TRAIN_XFORM(img)
    check("Spatial train augmentation shape consistent on 2nd call",
          tuple(t_train2.shape) == (3, 224, 224))
except Exception as e:
    check("Spatial transform (exception)", False, str(e))

# --- 7. FFT transform — native resolution FFT ---------------------------------
print("\n--- 7. FFT Transform — Native-Resolution FFT ---")
try:
    img = Image.open(sample_path).convert("RGB")
    orig_w, orig_h = img.size

    fft_out = FFT_TRANSFORM(img)
    check("FFT output shape [3,224,224]",
          tuple(fft_out.shape) == (3, 224, 224), f"shape={tuple(fft_out.shape)}")
    check("FFT output dtype float32",
          fft_out.dtype == torch.float32, f"dtype={fft_out.dtype}")

    # Verify output is finite (no NaN/Inf)
    check("FFT output is finite",
          torch.isfinite(fft_out).all().item(), f"has_nan={torch.isnan(fft_out).any().item()}")

    # Verify it's NOT identical to spatial output (FFT != spatial)
    is_different = not torch.allclose(fft_out, t_val, atol=0.01)
    check("FFT output differs from spatial output (sanity)",
          is_different, f"are_different={is_different}")

    # Verify the FFT was computed at native resolution by checking with a
    # manually computed FFT at native res
    import torchvision.transforms as tvt
    x_native = tvt.ToTensor()(img)         # [3, H_orig, W_orig]
    fft_native = torch.fft.fft2(x_native)
    fft_native = torch.fft.fftshift(fft_native, dim=(-2, -1))
    mag_native = torch.log1p(torch.abs(fft_native))
    for c in range(3):
        mn, mx = mag_native[c].min(), mag_native[c].max()
        mag_native[c] = (mag_native[c] - mn) / (mx - mn + 1e-8)
    mag_resized = F.interpolate(
        mag_native.unsqueeze(0), size=(224, 224),
        mode="bilinear", align_corners=False
    ).squeeze(0)
    norm = tvt.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    mag_resized = norm(mag_resized)
    check("FFT matches manually computed native-res pipeline",
          torch.allclose(fft_out, mag_resized, atol=1e-4),
          f"max_diff={float((fft_out - mag_resized).abs().max()):.6f}")

    print(f"     Original image size: {orig_w}x{orig_h}")
    print(f"     FFT tensor shape after pipeline: {tuple(fft_out.shape)}")

except Exception as e:
    check("FFT transform (exception)", False, traceback.format_exc())

# --- 8. Spatial DataLoader — one batch ----------------------------------------
print("\n--- 8. Spatial DataLoader — One Batch ---")
try:
    ds_s = V4Dataset("train", mode="spatial", augment=True)
    dl_s = DataLoader(ds_s, batch_size=4, shuffle=False, collate_fn=collate_spatial)
    batch = next(iter(dl_s))
    x, y, domains = batch
    check("Spatial batch: x shape [4,3,224,224]",
          tuple(x.shape) == (4, 3, 224, 224), f"shape={tuple(x.shape)}")
    check("Spatial batch: labels shape [4]",
          tuple(y.shape) == (4,), f"shape={tuple(y.shape)}")
    check("Spatial batch: labels dtype long",
          y.dtype == torch.long, f"dtype={y.dtype}")
    check("Spatial batch: domains is list of str",
          isinstance(domains, list) and isinstance(domains[0], str),
          f"type={type(domains[0])}")
    check("Spatial batch: labels in {0,1}",
          all(l in (0, 1) for l in y.tolist()),
          f"labels={y.tolist()}")
    print(f"     Labels: {y.tolist()}  ({['REAL' if l==0 else 'AI' for l in y.tolist()]})")
    print(f"     Domains: {domains}")
except Exception as e:
    check("Spatial DataLoader (exception)", False, traceback.format_exc())

# --- 9. Frequency DataLoader — one batch --------------------------------------
print("\n--- 9. Frequency DataLoader — One Batch ---")
try:
    ds_f = V4Dataset("train", mode="frequency", augment=False)
    dl_f = DataLoader(ds_f, batch_size=4, shuffle=False, collate_fn=collate_spatial)
    batch = next(iter(dl_f))
    x, y, domains = batch
    check("Frequency batch: x shape [4,3,224,224]",
          tuple(x.shape) == (4, 3, 224, 224), f"shape={tuple(x.shape)}")
    check("Frequency batch: x is finite",
          torch.isfinite(x).all().item())
    check("Frequency batch: labels in {0,1}",
          all(l in (0, 1) for l in y.tolist()),
          f"labels={y.tolist()}")
    print(f"     FFT tensor range: [{x.min():.3f}, {x.max():.3f}]")
except Exception as e:
    check("Frequency DataLoader (exception)", False, traceback.format_exc())

# --- 10. Hybrid DataLoader — one batch ----------------------------------------
print("\n--- 10. Hybrid DataLoader — One Batch ---")
try:
    ds_h = V4Dataset("train", mode="hybrid", augment=True)
    dl_h = DataLoader(ds_h, batch_size=4, shuffle=False, collate_fn=collate_hybrid)
    batch = next(iter(dl_h))
    x_s, x_f, y, domains = batch
    check("Hybrid batch: x_s shape [4,3,224,224]",
          tuple(x_s.shape) == (4, 3, 224, 224), f"shape={tuple(x_s.shape)}")
    check("Hybrid batch: x_f shape [4,3,224,224]",
          tuple(x_f.shape) == (4, 3, 224, 224), f"shape={tuple(x_f.shape)}")
    check("Hybrid batch: labels in {0,1}",
          all(l in (0, 1) for l in y.tolist()),
          f"labels={y.tolist()}")
    # Spatial and FFT tensors must differ (different pipelines)
    check("Hybrid: spatial and FFT inputs differ",
          not torch.allclose(x_s, x_f, atol=0.01))
    print(f"     Spatial range: [{x_s.min():.3f}, {x_s.max():.3f}]")
    print(f"     FFT range:     [{x_f.min():.3f}, {x_f.max():.3f}]")
except Exception as e:
    check("Hybrid DataLoader (exception)", False, traceback.format_exc())

# --- 11. SpatialV4 forward pass -----------------------------------------------
print("\n--- 11. SpatialV4 Forward Pass ---")
try:
    model_s = SpatialV4().to(DEVICE)
    model_s.eval()
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        out = model_s(dummy)
    check("SpatialV4 output shape [2, 2]", tuple(out.shape) == (2, 2), f"shape={tuple(out.shape)}")
    check("SpatialV4 output finite", torch.isfinite(out).all().item())
    check("SpatialV4 on CUDA", out.device.type == "cuda" if DEVICE.type == "cuda" else True,
          f"device={out.device}")
    del model_s
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
except Exception as e:
    check("SpatialV4 (exception)", False, traceback.format_exc())

# --- 12. FrequencyV4 forward pass ---------------------------------------------
print("\n--- 12. FrequencyV4 Forward Pass ---")
try:
    model_f = FrequencyV4().to(DEVICE)
    model_f.eval()
    dummy_fft = torch.randn(2, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        out = model_f(dummy_fft)
    check("FrequencyV4 output shape [2, 2]", tuple(out.shape) == (2, 2), f"shape={tuple(out.shape)}")
    check("FrequencyV4 output finite", torch.isfinite(out).all().item())
    del model_f
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
except Exception as e:
    check("FrequencyV4 (exception)", False, traceback.format_exc())

# --- 13. HybridV4 forward pass + feature dimensions --------------------------
print("\n--- 13. HybridV4 Forward Pass + Feature Dimensions ---")
try:
    model_h = HybridV4().to(DEVICE)
    model_h.eval()
    dummy_s = torch.randn(2, 3, 224, 224).to(DEVICE)
    dummy_f = torch.randn(2, 3, 224, 224).to(DEVICE)

    # Check spatial feature dim
    with torch.no_grad():
        s_feat = model_h.spatial_features(dummy_s)
        s_feat = torch.flatten(s_feat, 1)
    check("Hybrid spatial branch dim = 2048",
          s_feat.shape[1] == SPATIAL_FEATURE_DIM,
          f"got={s_feat.shape[1]}, expected={SPATIAL_FEATURE_DIM}")

    # Check frequency feature dim
    with torch.no_grad():
        f_feat = model_h.frequency_features(dummy_f)
        f_feat = torch.flatten(f_feat, 1)
    check("Hybrid frequency branch dim = 256",
          f_feat.shape[1] == FREQUENCY_FEATURE_DIM,
          f"got={f_feat.shape[1]}, expected={FREQUENCY_FEATURE_DIM}")

    # Check fused dim
    fused_actual = s_feat.shape[1] + f_feat.shape[1]
    check("Hybrid fused dim = 2048 + 256 = 2304",
          fused_actual == FUSED_DIM,
          f"got={fused_actual}, expected={FUSED_DIM}")

    # Check full forward pass
    with torch.no_grad():
        out = model_h(dummy_s, dummy_f)
    check("HybridV4 output shape [2, 2]", tuple(out.shape) == (2, 2), f"shape={tuple(out.shape)}")
    check("HybridV4 output finite", torch.isfinite(out).all().item())
    del model_h
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
except Exception as e:
    check("HybridV4 (exception)", False, traceback.format_exc())

# --- 14. Checkpoint output paths ----------------------------------------------
print("\n--- 14. Checkpoint Output Paths ---")
for fname in ["spatial_resnet50_v4.pth", "frequency_resnet50_v4.pth", "hybrid_resnet50_fft_v4.pth"]:
    p = MODEL_DIR / fname
    # These don't exist yet (not trained) — just check MODEL_DIR is writable
    check(f"MODEL_DIR writable for {fname}", MODEL_DIR.exists() and os.access(MODEL_DIR, os.W_OK))
    if p.exists():
        print(f"     NOTE: {fname} already exists ({p.stat().st_size/1e6:.1f}MB). Will be overwritten by training.")

# --- 15. v4_results/ directory structure --------------------------------------
print("\n--- 15. v4_results/ Directory Structure ---")
v4_results = BASE_DIR / "v4_results"
subdirs = ["training_logs", "validation_results", "checkpoints", "metrics", "configs"]
for sd in subdirs:
    p = v4_results / sd
    p.mkdir(parents=True, exist_ok=True)
    check(f"v4_results/{sd}/ exists", p.exists())

# --- 16. Preprocessing documentation embedded ---------------------------------
print("\n--- 16. PREPROCESSING_DOC Keys ---")
required_keys = ["spatial_train", "spatial_val", "fft", "label_convention", "input_size"]
for k in required_keys:
    check(f"PREPROCESSING_DOC has '{k}'", k in PREPROCESSING_DOC)

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)

print("\n" + "=" * 70)
print(f"SANITY CHECK COMPLETE: {n_pass}/{len(results)} passed, {n_fail} failed")
print("=" * 70)

if n_fail == 0:
    print("\n[OK] ALL CHECKS PASSED — V4 training is ready to launch.")
else:
    print("\n[X] SOME CHECKS FAILED. Fix the issues above before training.")
    print("\nFailed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL: {name} — {detail}")

print()

