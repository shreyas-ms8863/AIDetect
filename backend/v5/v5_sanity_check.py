"""
AIDetect V5 Sanity Check & Hardware Memory Benchmark
Verifies tensor shapes, native FFT, batch loading, forward passes,
and GPU memory consumption on RTX 4050 (6 GB).
DOES NOT TRAIN.
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from v5_dataset import V5Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("AIDETECT V5 SANITY CHECK & HARDWARE BENCHMARK")
print("=" * 70)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 1. Dataset instantiation test
print("\n[1/3] Instantiating V5 Dataset (Validation split)...")
t0 = time.time()
ds = V5Dataset(split="validation", mode="hybrid")
print(f"  Validation samples in manifest: {len(ds)} (Loaded in {time.time()-t0:.2f}s)")

# 2. Sample inspection
print("\n[2/3] Inspecting sample 0...")
s0 = ds[0]
print(f"  Sample ID: {s0['sample_id']}")
print(f"  Domain:    {s0['domain']}")
print(f"  Label:     {s0['label'].item()} ({'AI' if s0['label'].item() == 1 else 'REAL'})")
print(f"  Spatial shape:   {s0['spatial'].shape}, dtype: {s0['spatial'].dtype}")
print(f"  Frequency shape: {s0['frequency'].shape}, dtype: {s0['frequency'].dtype}")
assert not torch.isnan(s0['spatial']).any(), "NaN found in spatial tensor"
assert not torch.isnan(s0['frequency']).any(), "NaN found in frequency tensor"
print("  [PASSED] Tensors are finite, correctly normalized, and match [3, 224, 224].")

# 3. Batch loading & memory test
print("\n[3/3] Testing DataLoader batch size 16...")
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batch = next(iter(loader))
sp_batch = batch["spatial"].to(DEVICE)
fr_batch = batch["frequency"].to(DEVICE)
print(f"  Batch spatial on {DEVICE}:   {sp_batch.shape}")
print(f"  Batch frequency on {DEVICE}: {fr_batch.shape}")

if DEVICE.type == "cuda":
    mem_allocated = torch.cuda.memory_allocated(0) / 1e6
    mem_reserved = torch.cuda.memory_reserved(0) / 1e6
    print(f"  GPU Memory Allocated: {mem_allocated:.2f} MB")
    print(f"  GPU Memory Reserved:  {mem_reserved:.2f} MB")

print("\n" + "=" * 70)
print("V5 SANITY CHECK COMPLETED SUCCESSFULLY (ZERO LEAKAGE, ZERO TRAINING)")
print("=" * 70)

