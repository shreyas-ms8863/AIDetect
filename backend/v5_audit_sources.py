"""
V5 Source Dataset Discovery & Inventory Audit
Scans disk and caches for all available image sources.
"""
import os
import sys
import glob
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
print(f"Project root: {PROJECT_DIR}")

# 1. Check local project folders
folders_to_check = [
    PROJECT_DIR / "dataset_v4",
    PROJECT_DIR / "backend" / "external_test",
    PROJECT_DIR / "backend" / "real_world_test",
    PROJECT_DIR / "manipulation_v1",
    PROJECT_DIR / "dataset_manipulation_v2",
    PROJECT_DIR / "manipulation_external_test",
]

for folder in folders_to_check:
    if folder.exists():
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
            img_files.extend(list(folder.rglob(ext)))
        parquets = list(folder.rglob("*.parquet"))
        csvs = list(folder.rglob("*.csv"))
        print(f"Folder: {folder.name:<30} -> {len(img_files)} images, {len(parquets)} parquets, {len(csvs)} csvs")
    else:
        print(f"Folder: {folder.name:<30} -> NOT FOUND")

# 2. Check HuggingFace Cache
hf_cache = Path(r"C:\Users\Shreyas\.cache\huggingface")
if hf_cache.exists():
    print("\nScanning HuggingFace Cache:")
    for ds_dir in (hf_cache / "hub").glob("datasets--*"):
        print(f"  Hub dataset: {ds_dir.name}")
        parquets = list(ds_dir.rglob("*.parquet"))
        total_rows = 0
        p_details = []
        for p in parquets:
            try:
                df = pd.read_parquet(p)
                p_details.append((p.name, len(df), df.columns.tolist()))
                total_rows += len(df)
            except Exception as e:
                p_details.append((p.name, f"Error: {e}"))
        print(f"    Parquet files: {len(parquets)} (Total rows: {total_rows})")
        for name, count, cols in p_details[:5]:
            print(f"      - {name}: {count} rows, cols: {cols}")
        if len(p_details) > 5:
            print(f"      ... and {len(p_details) - 5} more parquets")

# 3. Check Downloads folder for specific camera images
dl_dir = Path(r"C:\Users\Shreyas\Downloads")
if dl_dir.exists():
    chatgpt_imgs = list(dl_dir.glob("ChatGPT*.png")) + list(dl_dir.glob("ChatGPT*.jpg"))
    wa_imgs = list(dl_dir.glob("WhatsApp Image*.jpeg")) + list(dl_dir.glob("WhatsApp Image*.jpg"))
    print(f"\nDownloads folder:")
    print(f"  ChatGPT images: {len(chatgpt_imgs)}")
    print(f"  WhatsApp images: {len(wa_imgs)}")

