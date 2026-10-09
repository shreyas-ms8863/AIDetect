"""
AIDetect V5 Dataset Preparation & Leakage Protection Engine
Profiles all historical evaluation sets to create a cryptographic blacklist,
then audits and constructs a leak-free candidate pool for V5.
"""

import os
import sys
import io
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
print(f"Project root: {PROJECT_DIR}")

def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest().lower()

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    try:
        gray = ImageOps.grayscale(image)
        resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        if hasattr(resized, "get_flattened_data"):
            pixels = list(resized.get_flattened_data())
        else:
            pixels = list(resized.tobytes())
        diff = []
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
        decimal_val = 0
        hex_str = []
        for index, value in enumerate(diff):
            if value:
                decimal_val += 2 ** (index % 4)
            if (index % 4) == 3:
                hex_str.append(hex(decimal_val)[2:])
                decimal_val = 0
        return "".join(hex_str)
    except Exception:
        return "0" * 16

# -------------------------------------------------------------
# 1. BUILD PROHIBITED EVALUATION BLACKLIST
# -------------------------------------------------------------
print("\n" + "="*70)
print("1. PROFILING EVALUATION DATASETS (PROHIBITED BLACKLIST)")
print("="*70)

blacklist_sha = set()
blacklist_details = {}

def add_blacklist(sha, source, name):
    blacklist_sha.add(sha)
    blacklist_details[sha] = {"source": source, "name": name}

# 1a. CIFAKE test split (20,000 images)
cifake_test_p = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\test-00000-of-00001.parquet")
if cifake_test_p.exists():
    print("Hashing CIFAKE test parquet (20,000 images)...")
    df_ctest = pd.read_parquet(cifake_test_p)
    for idx, row in df_ctest.iterrows():
        b = row["image"]["bytes"]
        sha = compute_sha256(b)
        add_blacklist(sha, "CIFAKE_official_test", f"cifake_test_{idx:05d}")
    print(f"  -> Added {len(df_ctest)} CIFAKE test images to blacklist.")

# 1b. dataset_v4 test split (613 images)
v4_test_dir = PROJECT_DIR / "dataset_v4" / "test"
if v4_test_dir.exists():
    v4_test_files = list(v4_test_dir.rglob("*.*"))
    print(f"Hashing dataset_v4/test ({len(v4_test_files)} images)...")
    for p in v4_test_files:
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            sha = compute_sha256(p.read_bytes())
            add_blacklist(sha, "dataset_v4_test", p.name)
    print(f"  -> Added {len(v4_test_files)} V4 test images to blacklist.")

# 1c. backend/external_test (86 images)
ext_dir = PROJECT_DIR / "backend" / "external_test"
if ext_dir.exists():
    ext_files = list(ext_dir.rglob("*.*"))
    print(f"Hashing backend/external_test ({len(ext_files)} images)...")
    for p in ext_files:
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            sha = compute_sha256(p.read_bytes())
            add_blacklist(sha, "external_test", p.name)
    print(f"  -> Added {len(ext_files)} external_test images to blacklist.")

# 1d. backend/real_world_test (19 images)
rw_dir = PROJECT_DIR / "backend" / "real_world_test"
if rw_dir.exists():
    rw_files = list(rw_dir.rglob("*.*"))
    print(f"Hashing backend/real_world_test ({len(rw_files)} images)...")
    for p in rw_files:
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            sha = compute_sha256(p.read_bytes())
            add_blacklist(sha, "real_world_test", p.name)
    print(f"  -> Added {len(rw_files)} real_world_test images to blacklist.")

# 1e. Tiny-GenImage validation shard (1,750 images)
gen_val_p = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\validation-00000-of-00004.parquet")
if gen_val_p.exists():
    print(f"Hashing Tiny-GenImage validation shard...")
    df_gval = pd.read_parquet(gen_val_p)
    for idx, row in df_gval.iterrows():
        b = row["image"]["bytes"]
        sha = compute_sha256(b)
        add_blacklist(sha, "Tiny_GenImage_val", f"gen_val_{idx:05d}")
    print(f"  -> Added {len(df_gval)} Tiny-GenImage validation images to blacklist.")

# 1f. Real-World Challenge Specimens in Downloads
dl_dir = Path(r"C:\Users\Shreyas\Downloads")
if dl_dir.exists():
    challenge_names = [
        "ChatGPT Image Sep 30, 2026, 11_23_01 PM.png",
        "WhatsApp Image 2026-10-01 at 11.28.52 PM.jpeg",
        "WhatsApp Image 2026-09-13 at 4.08.01 PM.jpeg",
        "scan-1786806054318-1.png",
    ]
    for cname in challenge_names:
        cp = dl_dir / cname
        if cp.exists():
            sha = compute_sha256(cp.read_bytes())
            add_blacklist(sha, "challenge_specimen", cname)
            print(f"  -> Added challenge specimen to blacklist: {cname}")

print(f"\nTOTAL BLACKLISTED SHA-256 HASHES: {len(blacklist_sha)}")

# -------------------------------------------------------------
# 2. AUDIT CANDIDATE TRAINING DATASETS
# -------------------------------------------------------------
print("\n" + "="*70)
print("2. AUDITING LEGITIMATE CANDIDATE POOLS AGAINST BLACKLIST")
print("="*70)

candidate_records = []
collisions_with_blacklist = []
internal_duplicates = 0
seen_candidate_sha = set()

# 2a. dataset_v4 train & validation
print("Auditing dataset_v4 (train + validation)...")
v4_dir = PROJECT_DIR / "dataset_v4"
for split in ["train", "validation"]:
    s_dir = v4_dir / split
    for f in s_dir.rglob("*.*"):
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            b = f.read_bytes()
            sha = compute_sha256(b)
            if sha in blacklist_sha:
                collisions_with_blacklist.append((f.name, blacklist_details[sha]))
                continue
            if sha in seen_candidate_sha:
                internal_duplicates += 1
                continue
            seen_candidate_sha.add(sha)
            gt = "REAL" if "real" in f.parts else "AI"
            domain = f.parent.name
            candidate_records.append({
                "source": f"dataset_v4_{split}",
                "filename": f.name,
                "ground_truth": gt,
                "domain": domain,
                "sha256": sha,
                "bytes_len": len(b),
                "is_parquet": False,
                "file_path": str(f)
            })

print(f"  -> Valid dataset_v4 candidates: {len(candidate_records)}")
print(f"  -> Collisions with blacklist: {len(collisions_with_blacklist)}")

# 2b. Tiny-GenImage train shards
print("\nAuditing Tiny-GenImage train shards...")
gen_train_shards = [
    Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00000-of-00014.parquet"),
    Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00001-of-00014.parquet"),
]

gen_candidates_added = 0
gen_overlap_with_v4 = 0
for shard_idx, sp in enumerate(gen_train_shards):
    if sp.exists():
        df_g = pd.read_parquet(sp)
        for idx, row in df_g.iterrows():
            b = row["image"]["bytes"]
            sha = compute_sha256(b)
            if sha in blacklist_sha:
                collisions_with_blacklist.append((f"gen_train_s{shard_idx}_{idx}", blacklist_details[sha]))
                continue
            if sha in seen_candidate_sha:
                gen_overlap_with_v4 += 1
                continue
            seen_candidate_sha.add(sha)
            gt = "REAL" if row["label"] == 0 else "AI"
            candidate_records.append({
                "source": f"Tiny-GenImage_train_shard_{shard_idx}",
                "filename": f"gen_train_s{shard_idx}_{idx:05d}.jpg",
                "ground_truth": gt,
                "domain": "genimage",
                "sha256": sha,
                "bytes_len": len(b),
                "is_parquet": True,
                "parquet_path": str(sp),
                "parquet_row": idx
            })
            gen_candidates_added += 1

print(f"  -> Tiny-GenImage candidates added: {gen_candidates_added}")
print(f"  -> Tiny-GenImage duplicates/overlap with V4: {gen_overlap_with_v4}")

# 2c. CIFAKE train split
print("\nAuditing CIFAKE train split (100,000 images)...")
cifake_train_p = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\train-00000-of-00001.parquet")
cifake_candidates_added = 0
cifake_overlap = 0

if cifake_train_p.exists():
    df_ctrain = pd.read_parquet(cifake_train_p)
    # We will sample balanced subset or all
    print(f"  Total rows in CIFAKE train parquet: {len(df_ctrain):,}")
    
    # Audit for blacklist collisions
    for idx, row in df_ctrain.iterrows():
        b = row["image"]["bytes"]
        sha = compute_sha256(b)
        if sha in blacklist_sha:
            collisions_with_blacklist.append((f"cifake_train_{idx}", blacklist_details[sha]))
            continue
        if sha in seen_candidate_sha:
            cifake_overlap += 1
            continue
        seen_candidate_sha.add(sha)
        gt = "AI" if row["label"] == 0 else "REAL" # CIFAKE: 0=AI, 1=REAL
        candidate_records.append({
            "source": "CIFAKE_train",
            "filename": f"cifake_train_{idx:06d}.png",
            "ground_truth": gt,
            "domain": "cifake_cifar_sd14",
            "sha256": sha,
            "bytes_len": len(b),
            "is_parquet": True,
            "parquet_path": str(cifake_train_p),
            "parquet_row": idx
        })
        cifake_candidates_added += 1

print(f"  -> CIFAKE train candidates added: {cifake_candidates_added}")
print(f"  -> CIFAKE duplicates/overlap: {cifake_overlap}")

# Summary of candidate pool
real_total = sum(1 for r in candidate_records if r["ground_truth"] == "REAL")
ai_total = sum(1 for r in candidate_records if r["ground_truth"] == "AI")

print("\n" + "="*70)
print("AUDIT SUMMARY RESULTS")
print("="*70)
print(f"Total Unique Valid Candidates Available: {len(candidate_records)}")
print(f"  - REAL: {real_total}")
print(f"  - AI  : {ai_total}")
print(f"Total Collisions with Blacklist (Exclusions): {len(collisions_with_blacklist)}")

out_dir = PROJECT_DIR / "v5_metadata"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "audit_candidate_summary.json", "w") as f:
    json.dump({
        "total_unique_candidates": len(candidate_records),
        "real_count": real_total,
        "ai_count": ai_total,
        "blacklist_size": len(blacklist_sha),
        "collisions_excluded": len(collisions_with_blacklist)
    }, f, indent=2)

print(f"Saved audit candidate summary -> v5_metadata/audit_candidate_summary.json")

