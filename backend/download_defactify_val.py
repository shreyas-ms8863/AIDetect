"""
Download and Checksum Verification for Defactify Validation Split.
Only downloads the 2 validation parquet files (~646 MB total).
Verifies against documented Git LFS SHA-256 / ETags.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
import requests
import pyarrow.parquet as pq

BASE_URL = "https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset/resolve/main/data"
OUTPUT_DIR = Path("diagnostic_outputs/defactify_validation_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    {
        "filename": "validation-00000-of-00002.parquet",
        "url": f"{BASE_URL}/validation-00000-of-00002.parquet",
        "expected_sha256": "f759e276b9dbb6965a427ec54384f608bcef9336983d77b9d2c70affef35ab8c",
        "expected_bytes": 333255675,
    },
    {
        "filename": "validation-00001-of-00002.parquet",
        "url": f"{BASE_URL}/validation-00001-of-00002.parquet",
        "expected_sha256": "429ee1ff44d0c9e913c624532c0014cafc55ea6a9cc48a8ac8d820ba435beae5",
        "expected_bytes": 344988959,
    },
]


def download_file(file_info: dict) -> Path:
    dest_path = OUTPUT_DIR / file_info["filename"]
    if dest_path.exists() and dest_path.stat().st_size == file_info["expected_bytes"]:
        print(f"[CACHE] {file_info['filename']} already exists with matching size. Verifying hash...")
        with open(dest_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h == file_info["expected_sha256"]:
            print(f"[VERIFIED] {file_info['filename']} matches expected SHA-256: {h}")
            return dest_path
        else:
            print(f"[CORRUPT] Hash mismatch for {file_info['filename']}. Re-downloading...")

    print(f"\n[DOWNLOADING] {file_info['filename']} ({file_info['expected_bytes'] / (1024*1024):.2f} MB)...")
    t0 = time.time()
    resp = requests.get(file_info["url"], stream=True, timeout=30)
    resp.raise_for_status()

    total_downloaded = 0
    last_pct = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                total_downloaded += len(chunk)
                pct = int((total_downloaded / file_info["expected_bytes"]) * 100)
                if pct >= last_pct + 10:
                    elapsed = time.time() - t0
                    speed = (total_downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {pct}% ({total_downloaded / (1024*1024):.1f} MB, {speed:.2f} MB/s)")
                    last_pct = pct

    elapsed = time.time() - t0
    print(f"[COMPLETE] {file_info['filename']} downloaded in {elapsed:.1f}s.")

    # Verify SHA-256
    print(f"[VERIFYING] Computing SHA-256 for {file_info['filename']}...")
    hasher = hashlib.sha256()
    with open(dest_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    computed_sha256 = hasher.hexdigest()

    if computed_sha256 == file_info["expected_sha256"]:
        print(f"[VERIFIED] SHA-256 matches: {computed_sha256}")
    else:
        raise ValueError(
            f"Checksum mismatch for {file_info['filename']}! "
            f"Expected: {file_info['expected_sha256']}, Got: {computed_sha256}"
        )

    return dest_path


def main():
    print("=" * 70)
    print("Defactify Validation Split Downloader & Integrity Verifier")
    print("=" * 70)

    downloaded_paths = []
    for f_info in FILES:
        path = download_file(f_info)
        downloaded_paths.append(path)

    print("\n" + "=" * 70)
    print("Inspecting Parquet Schema & Metadata")
    print("=" * 70)

    total_rows = 0
    for p in downloaded_paths:
        table = pq.read_table(p, columns=["Caption", "Label_A", "Label_B"])
        df = table.to_pandas()
        total_rows += len(df)
        print(f"\nFile: {p.name}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {table.column_names}")
        print(f"  Label_A value counts:\n{df['Label_A'].value_counts().to_dict()}")
        print(f"  Label_B value counts:\n{df['Label_B'].value_counts().to_dict()}")

    print(f"\nTotal Validation Rows across both shards: {total_rows}")
    print("[SUCCESS] Defactify validation split downloaded and verified.")


if __name__ == "__main__":
    main()
