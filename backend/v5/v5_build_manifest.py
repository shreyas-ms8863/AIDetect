"""
AIDetect V5 Master Dataset Manifest Builder
Assembles 60,000 balanced, clean, non-leaking images for AIDetect V5:
- 30,000 REAL, 30,000 AI
- 70% TRAIN (42,000), 15% VALIDATION (9,000), 15% TEST (9,000)
- Strict cryptographic verification: 0 cross-split leakage, 0 benchmark leakage
"""

import os
import sys
import json
import csv
import hashlib
import random
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
print(f"Project root: {PROJECT_DIR}")

SEED = 42
random.seed(SEED)

def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest().lower()

# -------------------------------------------------------------
# 1. LOAD PROHIBITED EVALUATION BLACKLIST
# -------------------------------------------------------------
print("\n[1/5] Compiling Prohibited Evaluation Blacklist...")
blacklist_sha = set()

# CIFAKE official test (20,000)
p_ctest = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\test-00000-of-00001.parquet")
if p_ctest.exists():
    df = pd.read_parquet(p_ctest)
    for _, r in df.iterrows():
        blacklist_sha.add(compute_sha256(r["image"]["bytes"]))

# dataset_v4 test (613)
for p in (PROJECT_DIR / "dataset_v4" / "test").rglob("*.*"):
    if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
        blacklist_sha.add(compute_sha256(p.read_bytes()))

# external_test (86)
for p in (PROJECT_DIR / "backend" / "external_test").rglob("*.*"):
    if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
        blacklist_sha.add(compute_sha256(p.read_bytes()))

# real_world_test (19)
for p in (PROJECT_DIR / "backend" / "real_world_test").rglob("*.*"):
    if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
        blacklist_sha.add(compute_sha256(p.read_bytes()))

# Tiny-GenImage validation shard (1,750)
p_gval = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\validation-00000-of-00004.parquet")
if p_gval.exists():
    df = pd.read_parquet(p_gval)
    for _, r in df.iterrows():
        blacklist_sha.add(compute_sha256(r["image"]["bytes"]))

# Challenge specimens
dl_dir = Path(r"C:\Users\Shreyas\Downloads")
for cname in ["ChatGPT Image Sep 30, 2026, 11_23_01 PM.png", "WhatsApp Image 2026-10-01 at 11.28.52 PM.jpeg", "WhatsApp Image 2026-09-13 at 4.08.01 PM.jpeg", "scan-1786806054318-1.png"]:
    cp = dl_dir / cname
    if cp.exists():
        blacklist_sha.add(compute_sha256(cp.read_bytes()))

print(f"  -> Total Blacklisted Evaluation Hashes: {len(blacklist_sha)}")

# -------------------------------------------------------------
# 2. COLLECT AND DEDUPLICATE CANDIDATE IMAGES
# -------------------------------------------------------------
print("\n[2/5] Ingesting candidate sources (High-res prioritized)...")

pool_real = []
pool_ai = []
seen_sha = set()
duplicates_dropped = 0
blacklist_dropped = 0

# 2a. High-Res dataset_v4 (train + validation)
v4_dir = PROJECT_DIR / "dataset_v4"
for split in ["train", "validation"]:
    for f in (v4_dir / split).rglob("*.*"):
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            b = f.read_bytes()
            sha = compute_sha256(b)
            if sha in blacklist_sha:
                blacklist_dropped += 1
                continue
            if sha in seen_sha:
                duplicates_dropped += 1
                continue
            seen_sha.add(sha)
            gt = "REAL" if "real" in f.parts else "AI"
            rec = {
                "sample_id": f"v4_{f.stem}",
                "ground_truth": gt,
                "domain": f.parent.name,
                "source": f"dataset_v4_{split}",
                "resolution_type": "native_high_res",
                "sha256": sha,
                "storage_type": "file",
                "storage_ref": str(f.relative_to(PROJECT_DIR).as_posix()),
                "file_size_bytes": len(b)
            }
            if gt == "REAL": pool_real.append(rec)
            else: pool_ai.append(rec)

print(f"  After dataset_v4: REAL={len(pool_real)}, AI={len(pool_ai)}")

# 2b. Tiny-GenImage train shards (shards 0 and 1)
gen_shards = [
    Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00000-of-00014.parquet"),
    Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00001-of-00014.parquet"),
]
for s_idx, sp in enumerate(gen_shards):
    if sp.exists():
        df_g = pd.read_parquet(sp)
        for r_idx, row in df_g.iterrows():
            b = row["image"]["bytes"]
            sha = compute_sha256(b)
            if sha in blacklist_sha:
                blacklist_dropped += 1
                continue
            if sha in seen_sha:
                duplicates_dropped += 1
                continue
            seen_sha.add(sha)
            gt = "REAL" if row["label"] == 0 else "AI"
            gen_id = str(row.get("generator", "0"))
            rec = {
                "sample_id": f"genimage_s{s_idx}_{r_idx:05d}",
                "ground_truth": gt,
                "domain": f"genimage_gen{gen_id}",
                "source": f"Tiny-GenImage_train_shard_{s_idx}",
                "resolution_type": "native_high_res",
                "sha256": sha,
                "storage_type": "parquet_row",
                "storage_ref": f"{sp.name}:{r_idx}",
                "file_size_bytes": len(b)
            }
            if gt == "REAL": pool_real.append(rec)
            else: pool_ai.append(rec)

print(f"  After Tiny-GenImage: REAL={len(pool_real)}, AI={len(pool_ai)}")

# 2c. CIFAKE train split to reach quota
TARGET_TOTAL = 60000
TARGET_PER_CLASS = TARGET_TOTAL // 2 # 30,000 Real, 30,000 AI

needed_real = TARGET_PER_CLASS - len(pool_real)
needed_ai = TARGET_PER_CLASS - len(pool_ai)
print(f"  Quota needed from CIFAKE train: REAL={needed_real}, AI={needed_ai}")

cifake_train_p = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\train-00000-of-00001.parquet")
if cifake_train_p.exists():
    df_ctrain = pd.read_parquet(cifake_train_p)
    # Shuffle indices deterministically
    indices = list(range(len(df_ctrain)))
    random.seed(SEED)
    random.shuffle(indices)
    
    added_real = 0
    added_ai = 0
    for idx in indices:
        if added_real >= needed_real and added_ai >= needed_ai:
            break
        row = df_ctrain.iloc[idx]
        b = row["image"]["bytes"]
        sha = compute_sha256(b)
        if sha in blacklist_sha:
            blacklist_dropped += 1
            continue
        if sha in seen_sha:
            duplicates_dropped += 1
            continue
        gt = "AI" if row["label"] == 0 else "REAL" # CIFAKE: 0=AI, 1=REAL
        if gt == "REAL" and added_real < needed_real:
            seen_sha.add(sha)
            pool_real.append({
                "sample_id": f"cifake_train_{idx:06d}",
                "ground_truth": "REAL",
                "domain": "cifar10_real",
                "source": "CIFAKE_train",
                "resolution_type": "cifake_32x32",
                "sha256": sha,
                "storage_type": "parquet_row",
                "storage_ref": f"{cifake_train_p.name}:{idx}",
                "file_size_bytes": len(b)
            })
            added_real += 1
        elif gt == "AI" and added_ai < needed_ai:
            seen_sha.add(sha)
            pool_ai.append({
                "sample_id": f"cifake_train_{idx:06d}",
                "ground_truth": "AI",
                "domain": "sd14_ai",
                "source": "CIFAKE_train",
                "resolution_type": "cifake_32x32",
                "sha256": sha,
                "storage_type": "parquet_row",
                "storage_ref": f"{cifake_train_p.name}:{idx}",
                "file_size_bytes": len(b)
            })
            added_ai += 1

print(f"  Final candidate pool: REAL={len(pool_real)}, AI={len(pool_ai)}")
print(f"  Excluded duplicates: {duplicates_dropped}, Excluded blacklist collisions: {blacklist_dropped}")

# -------------------------------------------------------------
# 3. DETERMINISTIC 70 / 15 / 15 PARTITIONING
# -------------------------------------------------------------
print("\n[3/5] Partitioning deterministically (70% Train, 15% Val, 15% Test)...")
random.seed(SEED)
random.shuffle(pool_real)
random.shuffle(pool_ai)

n_real = len(pool_real)
n_ai = len(pool_ai)

n_real_train = int(n_real * 0.70)
n_real_val = int(n_real * 0.15)
n_real_test = n_real - n_real_train - n_real_val

n_ai_train = int(n_ai * 0.70)
n_ai_val = int(n_ai * 0.15)
n_ai_test = n_ai - n_ai_train - n_ai_val

train_records = pool_real[:n_real_train] + pool_ai[:n_ai_train]
val_records = pool_real[n_real_train:n_real_train+n_real_val] + pool_ai[n_ai_train:n_ai_train+n_ai_val]
test_records = pool_real[n_real_train+n_real_val:] + pool_ai[n_ai_train+n_ai_val:]

for r in train_records: r["split"] = "train"
for r in val_records: r["split"] = "validation"
for r in test_records: r["split"] = "test"

all_v5_records = train_records + val_records + test_records
random.seed(SEED)
random.shuffle(all_v5_records)

print(f"  TRAIN      : {len(train_records)} (REAL: {n_real_train}, AI: {n_ai_train})")
print(f"  VALIDATION : {len(val_records)} (REAL: {n_real_val}, AI: {n_ai_val})")
print(f"  FINAL TEST : {len(test_records)} (REAL: {n_real_test}, AI: {n_ai_test})")
print(f"  TOTAL V5   : {len(all_v5_records)}")

# -------------------------------------------------------------
# 4. STRICT ZERO-LEAKAGE CRYPTOGRAPHIC VERIFICATION
# -------------------------------------------------------------
print("\n[4/5] Executing Cryptographic Cross-Split Assertion Checks...")
train_hashes = {r["sha256"] for r in train_records}
val_hashes = {r["sha256"] for r in val_records}
test_hashes = {r["sha256"] for r in test_records}

leak_tv = train_hashes.intersection(val_hashes)
leak_tt = train_hashes.intersection(test_hashes)
leak_vt = val_hashes.intersection(test_hashes)
leak_black = (train_hashes | val_hashes | test_hashes).intersection(blacklist_sha)

print(f"  Train AND Validation: {len(leak_tv)} (Expected: 0)")
print(f"  Train AND Test      : {len(leak_tt)} (Expected: 0)")
print(f"  Validation AND Test : {len(leak_vt)} (Expected: 0)")
print(f"  V5 AND Blacklist    : {len(leak_black)} (Expected: 0)")

assert len(leak_tv) == 0, f"Train-Val leak: {len(leak_tv)}"
assert len(leak_tt) == 0, f"Train-Test leak: {len(leak_tt)}"
assert len(leak_vt) == 0, f"Val-Test leak: {len(leak_vt)}"
assert len(leak_black) == 0, f"Benchmark leak: {len(leak_black)}"
print("  [VERIFICATION PASSED] ZERO LEAKAGE MATHEMATICALLY CONFIRMED (0.00% LEAKAGE).")

# -------------------------------------------------------------
# 5. WRITE MANIFESTS AND METADATA REPORTS
# -------------------------------------------------------------
print("\n[5/5] Writing master manifests and metadata...")
v5_meta_dir = PROJECT_DIR / "v5_metadata"
v5_meta_dir.mkdir(parents=True, exist_ok=True)

# 5a. Dataset Manifest CSV
manifest_csv = v5_meta_dir / "v5_dataset_manifest.csv"
fieldnames = ["sample_id", "ground_truth", "domain", "source", "resolution_type", "split", "sha256", "storage_type", "storage_ref", "file_size_bytes"]
with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_v5_records)
print(f"  -> Saved manifest ({len(all_v5_records)} rows) -> v5_metadata/v5_dataset_manifest.csv")

# 5b. Split Summary Manifest CSV
split_summary = [
    {"split": "train", "total": len(train_records), "real": n_real_train, "ai": n_ai_train, "percentage": "70.0%"},
    {"split": "validation", "total": len(val_records), "real": n_real_val, "ai": n_ai_val, "percentage": "15.0%"},
    {"split": "test", "total": len(test_records), "real": n_real_test, "ai": n_ai_test, "percentage": "15.0%"},
    {"split": "TOTAL", "total": len(all_v5_records), "real": n_real, "ai": n_ai, "percentage": "100.0%"}
]
with open(v5_meta_dir / "v5_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "total", "real", "ai", "percentage"])
    writer.writeheader()
    writer.writerows(split_summary)

# 5c. Statistics JSON
stats_data = {
    "v5_dataset_name": "AIDetect_V5_Master_Robustness_Dataset",
    "seed": SEED,
    "total_images": len(all_v5_records),
    "split_counts": {
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records)
    },
    "ground_truth_counts": {
        "REAL": n_real,
        "AI": n_ai
    },
    "split_proportions": {
        "train": "70.0%",
        "validation": "15.0%",
        "test": "15.0%"
    },
    "source_breakdown": {
        "dataset_v4": sum(1 for r in all_v5_records if "dataset_v4" in r["source"]),
        "Tiny_GenImage": sum(1 for r in all_v5_records if "Tiny-GenImage" in r["source"]),
        "CIFAKE_train": sum(1 for r in all_v5_records if "CIFAKE" in r["source"])
    },
    "leakage_verification": {
        "train_val_overlap": len(leak_tv),
        "train_test_overlap": len(leak_tt),
        "val_test_overlap": len(leak_vt),
        "benchmark_overlap": len(leak_black),
        "status": "STRICT_ZERO_LEAKAGE_CONFIRMED"
    }
}
with open(v5_meta_dir / "v5_dataset_statistics.json", "w", encoding="utf-8") as f:
    json.dump(stats_data, f, indent=2)

print("\n" + "="*80)
print("AIDETECT V5 DATASET PREPARATION COMPLETED SUCCESSFULLY (ZERO TRAINING)")
print("="*80)
