"""
AIDetect V5 Dataset Discovery and Candidate Audit Tool
Analyzes every dataset source present across the AIDetect project,
evaluating total images, Real/AI distributions, generators, and roles.
"""

import os
import sys
import io
import json
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
print(f"Project root: {PROJECT_DIR}")

def analyze_source(name, get_stats_fn):
    print(f"\n{'='*70}\nAnalyzing: {name}\n{'='*70}")
    try:
        stats = get_stats_fn()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return stats
    except Exception as e:
        print(f"  ERROR analyzing {name}: {e}")
        return {"error": str(e)}

results = {}

# 1. CIFAKE train split
def stats_cifake_train():
    p = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\train-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    lbl_counts = df["label"].value_counts().to_dict()
    # 0 = AI, 1 = Real in CIFAKE
    return {
        "total": len(df),
        "real": int(lbl_counts.get(1, 0)),
        "ai": int(lbl_counts.get(0, 0)),
        "generator": "Stable Diffusion v1.4 (for AI), CIFAR-10 (for Real)",
        "resolution": "32x32",
        "current_role": "Training (V1 used 2k, V2 used 20k, V3 used 1k)"
    }
results["CIFAKE_train"] = stats_cifake_train()

# 2. CIFAKE test split
def stats_cifake_test():
    p = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    lbl_counts = df["label"].value_counts().to_dict()
    return {
        "total": len(df),
        "real": int(lbl_counts.get(1, 0)),
        "ai": int(lbl_counts.get(0, 0)),
        "generator": "Stable Diffusion v1.4 (AI), CIFAR-10 (Real)",
        "resolution": "32x32",
        "current_role": "Official Benchmark Test Split (MUST BE EXCLUDED FROM V5 TRAINING)"
    }
results["CIFAKE_test"] = stats_cifake_test()

# 3. Tiny-GenImage train shards
def stats_tiny_genimage_train():
    shard0 = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00000-of-00014.parquet")
    shard1 = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00001-of-00014.parquet")
    df = pd.concat([pd.read_parquet(shard0), pd.read_parquet(shard1)], ignore_index=True)
    lbl_counts = df["label"].value_counts().to_dict()
    gens = df["generator"].value_counts().to_dict() if "generator" in df.columns else "Unknown"
    return {
        "total": len(df),
        "real": int(lbl_counts.get(0, 0)), # 0 = Real in GenImage
        "ai": int(lbl_counts.get(1, 0)),   # 1 = AI in GenImage
        "generator": gens,
        "resolution": "Variable / High-res ImageNet",
        "current_role": "Training (V3 sampled 1.6k, V4 sampled 3k)"
    }
results["Tiny-GenImage_train"] = stats_tiny_genimage_train()

# 4. Tiny-GenImage validation shard
def stats_tiny_genimage_val():
    val_shard = Path(r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\validation-00000-of-00004.parquet")
    df = pd.read_parquet(val_shard)
    lbl_counts = df["label"].value_counts().to_dict()
    gens = df["generator"].value_counts().to_dict() if "generator" in df.columns else "Unknown"
    return {
        "total": len(df),
        "real": int(lbl_counts.get(0, 0)),
        "ai": int(lbl_counts.get(1, 0)),
        "generator": gens,
        "resolution": "Variable / High-res ImageNet",
        "current_role": "Benchmark Evaluation (100 images sampled in V2 genimage_test; MUST BE HELD OUT)"
    }
results["Tiny-GenImage_val"] = stats_tiny_genimage_val()

# 5. dataset_v4 train, val, test
def stats_dataset_v4():
    v4_dir = PROJECT_DIR / "dataset_v4"
    stats = {}
    for split in ["train", "validation", "test"]:
        s_dir = v4_dir / split
        real_files = list((s_dir / "real").rglob("*.*")) if (s_dir / "real").exists() else []
        ai_files = list((s_dir / "ai").rglob("*.*")) if (s_dir / "ai").exists() else []
        stats[f"{split}_total"] = len(real_files) + len(ai_files)
        stats[f"{split}_real"] = len(real_files)
        stats[f"{split}_ai"] = len(ai_files)
    return stats
results["dataset_v4"] = stats_dataset_v4()

# 6. external_test and real_world_test
def stats_external_tests():
    ext_dir = PROJECT_DIR / "backend" / "external_test"
    rw_dir = PROJECT_DIR / "backend" / "real_world_test"
    return {
        "external_test_total": len(list(ext_dir.rglob("*.*"))),
        "external_test_real": len(list((ext_dir / "REAL").glob("*.*"))),
        "external_test_ai": len(list((ext_dir / "AI").glob("*.*"))),
        "real_world_total": len(list(rw_dir.rglob("*.*"))),
        "real_world_real": len(list((rw_dir / "REAL").glob("*.*"))),
        "real_world_ai": len(list((rw_dir / "AI").glob("*.*"))),
        "current_role": "Historical Evaluation Benchmarks (MUST BE EXCLUDED)"
    }
results["external_and_real_world"] = stats_external_tests()

# 7. manipulation datasets
def stats_manipulation():
    m1_dir = PROJECT_DIR / "manipulation_v1"
    m2_dir = PROJECT_DIR / "dataset_manipulation_v2"
    m_ext_dir = PROJECT_DIR / "manipulation_external_test"
    return {
        "manipulation_v1_images": len(list(m1_dir.rglob("*.png"))) + len(list(m1_dir.rglob("*.jpg"))),
        "dataset_manipulation_v2_images": len(list(m2_dir.rglob("*.png"))) + len(list(m2_dir.rglob("*.jpg"))),
        "manipulation_external_test_images": len(list(m_ext_dir.rglob("*.png"))) + len(list(m_ext_dir.rglob("*.jpg"))),
        "current_role": "Localized Inpainting / Manipulation (Separate research branch from whole-image Generation)"
    }
results["manipulation"] = stats_manipulation()

print("\n" + "="*80)
print("AUDIT SUMMARY COMPLETE")
print("="*80)
with open(PROJECT_DIR / "backend" / "v5" / "v5_sources_audit.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved summary -> backend/v5/v5_sources_audit.json")

