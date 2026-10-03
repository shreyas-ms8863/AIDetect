"""
AIDetect Phase 5: Manipulation Dataset Preparation & Curation Script
======================================================================
Prepares manipulation_v1/ dataset and manipulation_external_test/ benchmark:
  1. Streams curated subsets from MagicBrush (DALL-E 2 inpainting)
  2. Categorizes into 6 balanced manipulation types:
     - object_insertion
     - object_removal
     - object_replacement
     - background_replacement
     - face_modification
     - inpainting
  3. Enforces GROUP-BASED SPLITTING by original_id (Seed=42: 70% train, 15% val, 15% test)
  4. Saves original and manipulated images losslessly
  5. Computes SHA-256 and dHash for all images
  6. Preserves PIE-Bench in manipulation_external_test/
  7. Generates canonical metadata.csv and audit reports

Label convention:
  0 = ORIGINAL_REAL
  1 = AI_MANIPULATED
"""

import os
import sys
import io
import csv
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_DIR     = BASE_DIR.parent
MANIP_DIR       = PROJECT_DIR / "manipulation_v1"
EXT_TEST_DIR    = PROJECT_DIR / "manipulation_external_test"
METADATA_DIR    = MANIP_DIR / "metadata"

SEED            = 42
TARGET_PER_CAT  = {
    "object_insertion":       250,
    "object_removal":         250,
    "object_replacement":     250,
    "background_replacement": 200,
    "face_modification":      200,
    "inpainting":             150,
}
TOTAL_TARGET_PAIRS = sum(TARGET_PER_CAT.values())  # 1300 pairs = 2600 images

print("=" * 70)
print("AIDETECT PHASE 5: MANIPULATION DATASET CURATOR")
print("=" * 70)
print(f"Project Directory : {PROJECT_DIR}")
print(f"Target Directory  : {MANIP_DIR}")
print(f"External Benchmark: {EXT_TEST_DIR}")
print(f"Random Seed       : {SEED}")
print(f"Target Pairs      : {TOTAL_TARGET_PAIRS} ({TOTAL_TARGET_PAIRS * 2} images)")
for cat, target in TARGET_PER_CAT.items():
    print(f"  - {cat:<24}: {target} pairs")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# HASH UTILITIES
# ---------------------------------------------------------------------------

def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()

def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Computes standard 64-bit difference hash (16 hex chars)."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "{:016x}".format(int("".join(diff.flatten().astype(int).astype(str)), 2))

# ---------------------------------------------------------------------------
# INSTRUCTION CLASSIFICATION LOGIC
# ---------------------------------------------------------------------------

def classify_instruction(inst: str) -> str:
    inst_l = inst.lower()
    
    # 1. Background replacement
    if any(k in inst_l for k in [
        'background', 'sky', 'behind', 'landscape', 'setting', 
        'in the sea', 'on a beach', 'at a beach', 'on the grass', 
        'in the snow', 'in a forest', 'in space', 'in the park',
        'indoor', 'outdoor', 'on the water', 'desert', 'underwater'
    ]):
        return 'background_replacement'
        
    # 2. Face / appearance modification
    if any(k in inst_l for k in [
        'face', 'hair', 'smile', 'beard', 'glasses', 'sunglasses', 
        'mustache', 'bald', 'head', 'eyes', 'mouth', 'expression', 
        'hat', 'cap', 'helmet', 'teeth', 'wrinkles', 'eyebrows',
        'makeup', 'blonde', 'brunette', 'person smile', 'look happy'
    ]):
        return 'face_modification'
        
    # 3. Object removal
    if any(k in inst_l for k in [
        'remove', 'delete', 'erase', 'take away', 'take out', 
        'get rid of', 'clear the', 'cut out', 'eliminate', 
        'disappear', 'without the', 'take off the'
    ]):
        return 'object_removal'
        
    # 4. Object insertion
    if any(k in inst_l for k in [
        'add', 'put a', 'put an', 'insert', 'place a', 'place an', 
        'include', 'draw a', 'draw an', 'give him', 'give her', 
        'give the', 'set a', 'have a', 'there be a', 'lay a', 'lay an'
    ]):
        return 'object_insertion'
        
    # 5. Object replacement
    if any(k in inst_l for k in [
        'replace', 'change the', 'change that', 'change this', 
        'turn into', 'turn the', 'swap', 'convert', 'switch', 
        'transform', 'substitute', 'instead of'
    ]):
        return 'object_replacement'
        
    # 6. Inpainting / restoration / local texture
    return 'inpainting'

# ---------------------------------------------------------------------------
# STEP 1: POPULATE MANIPULATION_EXTERNAL_TEST (PIE-BENCH)
# ---------------------------------------------------------------------------

def setup_external_pie_bench():
    print("-" * 60)
    print("STEP 1: Setting up HELD-OUT PIE-Bench in manipulation_external_test/")
    print("-" * 60)
    
    pie_dir = EXT_TEST_DIR / "pie_bench"
    pie_images_dir = pie_dir / "images"
    pie_images_dir.mkdir(parents=True, exist_ok=True)
    
    pie_manifest_path = pie_dir / "pie_bench_manifest.csv"
    if pie_manifest_path.exists():
        print(f"PIE-Bench manifest already exists at {pie_manifest_path.name}. Verifying...")
        return
        
    configs = [
        ('0_random_140', 'random_edit'),
        ('1_change_object_80', 'object_replacement'),
        ('2_add_object_80', 'object_insertion'),
        ('3_delete_object_80', 'object_removal'),
        ('4_change_attribute_content_40', 'content_modification'),
        ('5_change_attribute_pose_40', 'pose_modification'),
        ('6_change_attribute_color_40', 'color_modification'),
        ('7_change_attribute_material_40', 'material_modification'),
        ('8_change_background_80', 'background_replacement'),
        ('9_change_style_80', 'style_modification'),
    ]
    
    manifest_rows = []
    total_saved = 0
    
    for cfg_name, category in configs:
        print(f"  Downloading PIE-Bench config: {cfg_name}...")
        ds = load_dataset("UB-CVML-Group/PIE_Bench_pp", cfg_name, split="V1")
        for ex in ds:
            img_id = ex["id"]
            img = ex["image"].convert("RGB")
            out_filename = f"pie_{img_id}.png"
            out_path = pie_images_dir / out_filename
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
            
            sha256 = compute_sha256(img_bytes)
            dhash = compute_dhash(img)
            
            manifest_rows.append({
                "benchmark": "PIE-Bench",
                "sample_id": img_id,
                "config_name": cfg_name,
                "category": category,
                "filename": out_filename,
                "filepath": str(out_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_prompt": ex.get("source_prompt", ""),
                "target_prompt": ex.get("target_prompt", ""),
                "edit_action": ex.get("edit_action", ""),
                "width": img.width,
                "height": img.height,
                "sha256": sha256,
                "dhash": dhash
            })
            total_saved += 1
            
    with open(pie_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "benchmark", "sample_id", "config_name", "category", "filename",
            "filepath", "source_prompt", "target_prompt", "edit_action",
            "width", "height", "sha256", "dhash"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    print(f"PIE-Bench held-out benchmark saved: {total_saved} images to {pie_dir}")
    print()
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# STEP 2: STREAM & CURATE MAGICBRUSH FOR MANIPULATION_V1
# ---------------------------------------------------------------------------

def curate_magicbrush():
    print("-" * 60)
    print("STEP 2: Streaming & Curating MagicBrush for manipulation_v1/")
    print("-" * 60)
    
    # We select shards from dev and train
    shards_to_use = [
        # dev shards (4 shards, ~528 examples total)
        "data/dev-00000-of-00004-f147d414270a90e1.parquet",
        "data/dev-00001-of-00004-8ef3de1dc8cb8a6a.parquet",
        "data/dev-00002-of-00004-54c4d7b0a9e49db5.parquet",
        "data/dev-00003-of-00004-384b81a61c93b7e3.parquet",
        # train shards (curated selection for remaining quotas)
        "data/train-00000-of-00051-9fd9f23e2b1cb397.parquet",
        "data/train-00001-of-00051-7fc041c6c75c7a1b.parquet",
        "data/train-00002-of-00051-4683be833ed06f1c.parquet",
        "data/train-00003-of-00051-dbbf01b8d4790783.parquet",
        "data/train-00004-of-00051-d373bd832c322236.parquet",
        "data/train-00005-of-00051-261c379dee1873a0.parquet",
        "data/train-00006-of-00051-1601e4998b870515.parquet",
    ]
    
    cat_counts = Counter()
    selected_pairs = []
    seen_manip_ids = set()
    
    for shard_idx, shard_name in enumerate(shards_to_use, 1):
        # Check if all quotas met
        all_met = all(cat_counts[cat] >= TARGET_PER_CAT[cat] for cat in TARGET_PER_CAT)
        if all_met:
            print(f"All category quotas met after {shard_idx - 1} shards!")
            break
            
        print(f"Fetching shard {shard_idx}/{len(shards_to_use)}: {shard_name}...")
        sys.stdout.flush()
        
        shard_path = hf_hub_download(
            repo_id="osunlp/MagicBrush",
            filename=shard_name,
            repo_type="dataset"
        )
        
        table = pq.read_table(shard_path)
        df = table.to_pandas()
        
        for _, row in df.iterrows():
            img_id = str(row["img_id"])
            turn_idx = int(row.get("turn_index", 0))
            manip_id = f"{img_id}_turn{turn_idx}"
            
            if manip_id in seen_manip_ids:
                continue
                
            inst = row["instruction"]
            cat = classify_instruction(inst)
            
            if cat_counts[cat] < TARGET_PER_CAT[cat]:
                s_bytes = row["source_img"]["bytes"]
                t_bytes = row["target_img"]["bytes"]
                
                selected_pairs.append({
                    "original_id": img_id,
                    "manip_id": manip_id,
                    "instruction": inst,
                    "category": cat,
                    "source_bytes": s_bytes,
                    "target_bytes": t_bytes,
                })
                
                seen_manip_ids.add(manip_id)
                cat_counts[cat] += 1
                
        print(f"  Progress: {sum(cat_counts.values())}/{TOTAL_TARGET_PAIRS} pairs collected")
        for c, count in cat_counts.most_common():
            print(f"    - {c:<24}: {count}/{TARGET_PER_CAT[c]}")
        sys.stdout.flush()
        
    print()
    print(f"Total curated pairs assembled: {len(selected_pairs)}")
    return selected_pairs

# ---------------------------------------------------------------------------
# STEP 3: PERFORM GROUP-BASED SPLIT & DISK WRITING
# ---------------------------------------------------------------------------

def process_and_save_dataset(pairs):
    print("-" * 60)
    print("STEP 3: Deterministic Group-Based Splitting & Disk Writing")
    print("-" * 60)
    
    # Group pairs by original_id
    grouped_by_orig = defaultdict(list)
    for p in pairs:
        grouped_by_orig[p["original_id"]].append(p)
        
    unique_orig_ids = sorted(list(grouped_by_orig.keys()))
    print(f"Unique original images: {len(unique_orig_ids)}")
    
    # Shuffle unique original IDs with SEED=42
    rng = random.Random(SEED)
    rng.shuffle(unique_orig_ids)
    
    n_total = len(unique_orig_ids)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    # Remaining for test
    
    train_orig_ids = set(unique_orig_ids[:n_train])
    val_orig_ids   = set(unique_orig_ids[n_train:n_train + n_val])
    test_orig_ids  = set(unique_orig_ids[n_train + n_val:])
    
    print(f"Group Split: Train={len(train_orig_ids)} originals, Val={len(val_orig_ids)} originals, Test={len(test_orig_ids)} originals")
    
    # Prepare directory structure
    splits = ["train", "validation", "test"]
    categories = list(TARGET_PER_CAT.keys())
    
    for split in splits:
        (MANIP_DIR / split / "original").mkdir(parents=True, exist_ok=True)
        for cat in categories:
            (MANIP_DIR / split / "manipulated" / cat).mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each pair
    metadata_rows = []
    split_manifest_rows = []
    pair_relationships = defaultdict(list)
    sha256_registry = {}
    dhash_registry = {}
    duplicate_records = []
    
    saved_orig_set = set() # Avoid saving the same original twice if multiple edits share it
    
    total_images_written = 0
    orig_idx_map = {}
    
    print("Writing images to disk with full hashing...")
    for orig_id, orig_pairs in grouped_by_orig.items():
        if orig_id in train_orig_ids:
            split = "train"
        elif orig_id in val_orig_ids:
            split = "validation"
        else:
            split = "test"
            
        split_manifest_rows.append({
            "original_id": orig_id,
            "split": split,
            "num_manipulations": len(orig_pairs)
        })
        
        # 1. Process ORIGINAL image once per unique original_id
        first_pair = orig_pairs[0]
        s_bytes = first_pair["source_bytes"]
        s_img = Image.open(io.BytesIO(s_bytes)).convert("RGB")
        
        s_sha256 = compute_sha256(s_bytes)
        s_dhash  = compute_dhash(s_img)
        
        orig_filename = f"orig_{orig_id}.png"
        orig_rel_path = f"manipulation_v1/{split}/original/{orig_filename}"
        orig_abs_path = PROJECT_DIR / orig_rel_path
        
        if not orig_abs_path.exists():
            with open(orig_abs_path, "wb") as f:
                f.write(s_bytes)
            total_images_written += 1
            
        orig_img_id = f"m1_{split}_orig_{orig_id}"
        orig_idx_map[orig_id] = orig_img_id
        
        metadata_rows.append({
            "image_id": orig_img_id,
            "original_id": orig_id,
            "split": split,
            "ground_truth": "ORIGINAL_REAL",
            "label": 0,
            "manipulation_type": "none",
            "source_dataset": "MagicBrush (MS-COCO)",
            "source_url": "https://huggingface.co/datasets/osunlp/MagicBrush",
            "generator_or_editor": "camera_original",
            "original_width": s_img.width,
            "original_height": s_img.height,
            "format": "PNG",
            "file_size_bytes": len(s_bytes),
            "sha256": s_sha256,
            "dhash": s_dhash,
        })
        
        # Check duplicate for original
        if s_sha256 in sha256_registry:
            duplicate_records.append({
                "type": "exact_sha256",
                "image_id": orig_img_id,
                "first_seen_id": sha256_registry[s_sha256],
                "hash": s_sha256
            })
        else:
            sha256_registry[s_sha256] = orig_img_id
            
        dhash_registry[s_dhash] = orig_img_id
        
        # 2. Process each MANIPULATED image of this original
        for p in orig_pairs:
            t_bytes = p["target_bytes"]
            t_img = Image.open(io.BytesIO(t_bytes)).convert("RGB")
            
            t_sha256 = compute_sha256(t_bytes)
            t_dhash  = compute_dhash(t_img)
            
            manip_cat = p["category"]
            manip_filename = f"manip_{p['manip_id']}.png"
            manip_rel_path = f"manipulation_v1/{split}/manipulated/{manip_cat}/{manip_filename}"
            manip_abs_path = PROJECT_DIR / manip_rel_path
            
            if not manip_abs_path.exists():
                with open(manip_abs_path, "wb") as f:
                    f.write(t_bytes)
                total_images_written += 1
                
            manip_img_id = f"m1_{split}_manip_{p['manip_id']}"
            pair_relationships[orig_img_id].append(manip_img_id)
            
            metadata_rows.append({
                "image_id": manip_img_id,
                "original_id": orig_id,
                "split": split,
                "ground_truth": "AI_MANIPULATED",
                "label": 1,
                "manipulation_type": manip_cat,
                "source_dataset": "MagicBrush",
                "source_url": "https://huggingface.co/datasets/osunlp/MagicBrush",
                "generator_or_editor": "DALL-E 2 Inpainting",
                "original_width": t_img.width,
                "original_height": t_img.height,
                "format": "PNG",
                "file_size_bytes": len(t_bytes),
                "sha256": t_sha256,
                "dhash": t_dhash,
            })
            
            # Check duplicate for manipulated
            if t_sha256 in sha256_registry:
                duplicate_records.append({
                    "type": "exact_sha256",
                    "image_id": manip_img_id,
                    "first_seen_id": sha256_registry[t_sha256],
                    "hash": t_sha256
                })
            else:
                sha256_registry[t_sha256] = manip_img_id
                
            dhash_registry[t_dhash] = manip_img_id
            
    print(f"Total disk writes complete: {total_images_written} files written.")
    sys.stdout.flush()
    
    # -----------------------------------------------------------------------
    # WRITE METADATA FILES
    # -----------------------------------------------------------------------
    print("Writing metadata manifests and audit reports...")
    
    # 1. metadata.csv
    meta_path = METADATA_DIR / "metadata.csv"
    fieldnames = [
        "image_id", "original_id", "split", "ground_truth", "label",
        "manipulation_type", "source_dataset", "source_url", "generator_or_editor",
        "original_width", "original_height", "format", "file_size_bytes",
        "sha256", "dhash"
    ]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
        
    # 2. split_manifest.csv
    split_path = METADATA_DIR / "split_manifest.csv"
    with open(split_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_id", "split", "num_manipulations"])
        writer.writeheader()
        writer.writerows(split_manifest_rows)
        
    # 3. pair_relationships.json
    pairs_path = METADATA_DIR / "pair_relationships.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pair_relationships, f, indent=2)
        
    # 4. duplicate_report.csv
    dup_path = METADATA_DIR / "duplicate_report.csv"
    with open(dup_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "image_id", "first_seen_id", "hash"])
        writer.writeheader()
        writer.writerows(duplicate_records)
        
    # 5. dataset_statistics.json
    stats = {
        "dataset_name": "manipulation_v1",
        "total_images": len(metadata_rows),
        "total_pairs": len(pairs),
        "unique_originals": len(unique_orig_ids),
        "class_counts": dict(Counter(r["ground_truth"] for r in metadata_rows)),
        "split_counts": dict(Counter(r["split"] for r in metadata_rows)),
        "split_by_class": {
            s: dict(Counter(r["ground_truth"] for r in metadata_rows if r["split"] == s))
            for s in splits
        },
        "manipulation_categories": dict(Counter(r["manipulation_type"] for r in metadata_rows if r["label"] == 1)),
        "generators": dict(Counter(r["generator_or_editor"] for r in metadata_rows)),
        "sources": dict(Counter(r["source_dataset"] for r in metadata_rows)),
        "exact_duplicates_found": len(duplicate_records),
    }
    stats_path = METADATA_DIR / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print("Metadata generation complete:")
    print(f"  - metadata.csv            : {len(metadata_rows)} entries")
    print(f"  - split_manifest.csv      : {len(split_manifest_rows)} originals")
    print(f"  - pair_relationships.json : {len(pair_relationships)} pairs mapped")
    print(f"  - duplicate_report.csv    : {len(duplicate_records)} duplicate warnings")
    print(f"  - dataset_statistics.json : summary saved")
    print()
    return stats

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_external_pie_bench()
    pairs = curate_magicbrush()
    stats = process_and_save_dataset(pairs)
    print("=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(json.dumps(stats, indent=2))

