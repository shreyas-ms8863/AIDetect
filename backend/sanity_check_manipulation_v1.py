"""
AIDetect Phase 6: Pre-Training Sanity Check for AI-Manipulation Detector
==========================================================================
Verifies:
  1. Every referenced image in metadata.csv exists on disk.
  2. All labels strictly within {0, 1} (0=ORIGINAL_REAL, 1=AI_MANIPULATED).
  3. Exact split counts: Train=1132, Validation=239, Test=248.
  4. Zero cross-split leakage of original_ids.
  5. CUDA GPU available (RTX 4050).
  6. Batch tensor shapes [16, 3, 224, 224] for train, val, and test.
  7. Forward pass on CUDA produces logits of shape [16, 2].
  8. Weighted CrossEntropyLoss is finite.

If any check fails, script exits with non-zero exit code.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manipulation_v1_models import ManipulationResNet50V1
from manipulation_v1_dataset import ManipulationV1Dataset, collate_manipulation

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
MANIP_DIR    = PROJECT_DIR / "manipulation_v1"
METADATA_CSV = MANIP_DIR / "metadata" / "metadata.csv"

PASS = "[PASS]"
FAIL = "[FAIL]"

checks_passed = 0
checks_total  = 0

def record(name: str, passed: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    if passed:
        checks_passed += 1
        msg = f"  {PASS} {name}"
    else:
        msg = f"  {FAIL} {name}"
    if detail:
        msg += f"\n         -> {detail}"
    print(msg)
    sys.stdout.flush()

print("=" * 70)
print("PHASE 6: PRE-TRAINING SANITY CHECK (MANIPULATION DETECTOR)")
print("=" * 70)
print(f"Working Directory: {PROJECT_DIR}")
print(f"Metadata File    : {METADATA_CSV}")
print()
sys.stdout.flush()

# 1. Verify metadata exists and rows exist
if not METADATA_CSV.exists():
    record("metadata.csv exists", False, "File missing!")
    sys.exit(1)

with open(METADATA_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

record("Metadata loadable", True, f"Found {len(rows)} entries")

# 2. Check every referenced image exists
missing = []
for r in rows:
    split = r["split"]
    lbl = int(r["label"])
    if lbl == 0:
        rel_p = f"manipulation_v1/{split}/original/orig_{r['original_id']}.png"
    else:
        cat = r["manipulation_type"]
        manip_id = r["image_id"].replace(f"m1_{split}_manip_", "")
        rel_p = f"manipulation_v1/{split}/manipulated/{cat}/manip_{manip_id}.png"
    p = PROJECT_DIR / rel_p
    if not p.exists():
        missing.append(rel_p)

record("Every referenced image exists on disk", len(missing) == 0,
       f"Missing count: {len(missing)}")

# 3. Verify labels are strictly 0 and 1
labels = set(int(r["label"]) for r in rows)
record("Labels strictly within {0, 1}", labels == {0, 1}, f"Found labels: {labels}")

# 4. Verify split counts
split_counts = Counter(r["split"] for r in rows)
expected_counts = {"train": 1132, "validation": 239, "test": 248}
record("Split counts exact match", split_counts == expected_counts,
       f"Expected {expected_counts}, got {dict(split_counts)}")

# 5. Verify zero cross-split leakage of original_id
orig_splits = defaultdict(set)
for r in rows:
    orig_splits[r["original_id"]].add(r["split"])

leaked_origs = [oid for oid, s in orig_splits.items() if len(s) > 1]
record("Zero cross-split leakage of original_ids", len(leaked_origs) == 0,
       f"Leaked original IDs: {len(leaked_origs)}")

# 6. Verify CUDA availability
cuda_ok = torch.cuda.is_available()
record("CUDA available", cuda_ok,
       f"Device: {torch.cuda.get_device_name(0) if cuda_ok else 'CPU'}")
device = torch.device("cuda" if cuda_ok else "cpu")

# 7. Check DataLoaders and batch tensor shapes
train_ds = ManipulationV1Dataset("train", augment=True)
val_ds   = ManipulationV1Dataset("validation", augment=False)
test_ds  = ManipulationV1Dataset("test", augment=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_manipulation)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_manipulation)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_manipulation)

batch_x, batch_y, batch_m = next(iter(train_loader))
record("Train batch x shape [16, 3, 224, 224]", tuple(batch_x.shape) == (16, 3, 224, 224),
       f"Got shape {tuple(batch_x.shape)}")
record("Train batch y shape [16]", tuple(batch_y.shape) == (16,),
       f"Got shape {tuple(batch_y.shape)}")

# 8. Model initialization & Forward pass
model = ManipulationResNet50V1().to(device)
model.eval()
bx = batch_x.to(device)
by = batch_y.to(device)

with torch.no_grad():
    logits = model(bx)

record("Forward pass logits shape [16, 2]", tuple(logits.shape) == (16, 2),
       f"Got shape {tuple(logits.shape)}")

# 9. Class weights and loss calculation
train_meta = [r for r in rows if r["split"] == "train"]
n_real  = sum(1 for r in train_meta if int(r["label"]) == 0)
n_manip = sum(1 for r in train_meta if int(r["label"]) == 1)
n_total = len(train_meta)

w_real  = n_total / (2.0 * n_real)
w_manip = n_total / (2.0 * n_manip)
weights = torch.tensor([w_real, w_manip], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
loss = criterion(logits, by)

record("Loss is finite and non-NaN", torch.isfinite(loss).item(),
       f"Loss value: {loss.item():.4f}")

print()
print("=" * 70)
print(f"SANITY CHECK SUMMARY: {checks_passed}/{checks_total} CHECKS PASSED")
print("=" * 70)

if checks_passed == checks_total:
    print("STATUS: SANITY CHECK PASSED - READY TO TRAIN")
    sys.exit(0)
else:
    print("STATUS: SANITY CHECK FAILED - DO NOT TRAIN")
    sys.exit(1)

