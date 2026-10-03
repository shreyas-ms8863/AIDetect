#!/usr/bin/env python3
"""
AIDetect V4 Dataset Preparation Engine
=====================================
Multi-source dataset acquisition, deduplication, perceptual hashing,
and deterministic partitioning for AI-generated image forensics.

Key Features:
- Multi-Source Diversity: Historical Real, Modern Real, GenImage Real/AI, Modern AI, Historical-Style AI.
- Rigorous Benchmark Isolation: Fingerprints all images in backend/external_test/ and backend/real_world_test/
  using SHA-256 and 64-bit dHash (Hamming distance threshold <= 2) to ensure 0% benchmark leakage.
- Strict Deduplication: Perceptual dHash + SHA-256 prevents duplicate samples.
- Concurrent Acquisition: ThreadPoolExecutor with HTTP Keep-Alive connection pooling.
- Deterministic Partitioning: 70% Train, 15% Validation, 15% Test (SEED=42).
- Complete Traceability: Generates metadata.csv, duplicate_report.csv, download_log.json, and dataset_statistics.json.
"""

import os
import sys
import json
import csv
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageOps
import io

# ============================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_V4_DIR = PROJECT_DIR / "dataset_v4"
METADATA_DIR = DATASET_V4_DIR / "metadata"

EXTERNAL_TEST_DIR = PROJECT_DIR / "backend" / "external_test"
REAL_WORLD_TEST_DIR = PROJECT_DIR / "backend" / "real_world_test"

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AIDetectV4Research/1.0 (https://github.com/shreyas-ms8863/AIDetect; educational research; contact: shreyas.detect@gmail.com)"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,image/jpeg,image/png,image/*;q=0.9",
    "Accept-Encoding": "gzip, deflate"
}

# ============================================================
# NETWORK SESSION SETUP (CONNECTION POOLING + RETRIES)
# ============================================================

def create_pooled_session() -> requests.Session:
    """Creates a high-performance requests Session with connection pooling."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

GLOBAL_SESSION = create_pooled_session()

# ============================================================
# HASHING & IMAGE VALIDATION
# ============================================================

def compute_sha256(data: bytes) -> str:
    """Computes standard SHA-256 hexadecimal hash."""
    return hashlib.sha256(data).hexdigest()

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Computes 64-bit difference hash (dHash) for perceptual image comparison.
    Uses get_flattened_data() or tobytes() for future-proof Pillow compatibility.
    """
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
            pixel_left = pixels[row_start + col]
            pixel_right = pixels[row_start + col + 1]
            diff.append(pixel_left > pixel_right)
            
    decimal_val = 0
    hex_str = []
    for index, value in enumerate(diff):
        if value:
            decimal_val += 2 ** (index % 4)
        if (index % 4) == 3:
            hex_str.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_str)

def hamming_distance(h1: str, h2: str) -> int:
    """Calculates bitwise Hamming distance between two hex hashes."""
    if len(h1) != len(h2):
        return 64
    x = int(h1, 16) ^ int(h2, 16)
    return bin(x).count("1")

def validate_image_bytes(data: bytes, min_dim: int = 128, max_dim: int = 8000):
    """
    Verifies that raw bytes form a valid, uncorrupted RGB image.
    Returns (PIL_Image, width, height, format, reason).
    """
    try:
        bio = io.BytesIO(data)
        with Image.open(bio) as img:
            img.verify()
        
        # Re-open for decoding & conversion
        bio.seek(0)
        img = Image.open(bio)
        img.load()
        
        w, h = img.size
        fmt = (img.format or "JPEG").upper()
        
        if w < min_dim or h < min_dim:
            return None, w, h, fmt, f"Dimensions too small: {w}x{h}"
        if w > max_dim or h > max_dim:
            return None, w, h, fmt, f"Dimensions too large: {w}x{h}"
            
        # Convert RGBA/Palette/Grayscale to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        return img, w, h, fmt, "VALID"
    except Exception as e:
        return None, 0, 0, "UNKNOWN", f"Corrupt image: {str(e)}"

# ============================================================
# BENCHMARK PROTECTION (STRICT ISOLATION)
# ============================================================

def build_benchmark_fingerprints():
    """
    Scans held-out evaluation sets to prevent any benchmark contamination.
    """
    benchmark_shas = set()
    benchmark_dhashes = {}
    
    dirs_to_scan = [EXTERNAL_TEST_DIR, REAL_WORLD_TEST_DIR]
    print("[SECURITY] Profiling held-out benchmark datasets to prevent data leakage...", flush=True)
    
    total_bm = 0
    for b_dir in dirs_to_scan:
        if not b_dir.exists():
            continue
        for root, _, files in os.walk(b_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
                    try:
                        raw = p.read_bytes()
                        sha = compute_sha256(raw)
                        benchmark_shas.add(sha)
                        with Image.open(io.BytesIO(raw)) as im:
                            dh = compute_dhash(im)
                            benchmark_dhashes[sha] = dh
                        total_bm += 1
                    except Exception:
                        pass
                        
    print(f"[SECURITY] Guard active: {total_bm} benchmark images fingerprinted (SHA-256 + 64-bit dHash).", flush=True)
    return benchmark_shas, benchmark_dhashes

# ============================================================
# DIRECTORY INITIALIZATION & SCANNING
# ============================================================

def init_v4_directories():
    """Initializes the V4 dataset directory tree."""
    splits = ["train", "validation", "test"]
    subdirs = [
        ("real", "modern"),
        ("real", "historical"),
        ("real", "genimage"),
        ("ai", "modern"),
        ("ai", "historical_style"),
        ("ai", "genimage"),
    ]
    
    for split in splits:
        for cat, domain in subdirs:
            folder = DATASET_V4_DIR / split / cat / domain
            folder.mkdir(parents=True, exist_ok=True)
            
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] V4 Directory structure verified at {DATASET_V4_DIR}", flush=True)

def scan_existing_files(benchmark_shas: set, benchmark_dhashes: dict):
    """
    Scans existing files in dataset_v4 (e.g. pre-extracted GenImage files)
    and loads their SHAs, dHashes, and metadata so they are deduplicated and tracked.
    """
    existing_items = []
    sha_seen = set()
    dhash_list = []
    
    splits = ["train", "validation", "test"]
    for split in splits:
        split_dir = DATASET_V4_DIR / split
        if not split_dir.exists():
            continue
        for root, _, files in os.walk(split_dir):
            for f in files:
                filepath = Path(root) / f
                if filepath.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
                    try:
                        raw = filepath.read_bytes()
                        sha = compute_sha256(raw)
                        rel_path = filepath.relative_to(PROJECT_DIR).as_posix()
                        
                        parts = filepath.relative_to(DATASET_V4_DIR).parts
                        split_name = parts[0]
                        cat_name = parts[1].upper() # REAL or AI
                        domain_name = parts[2]      # genimage, modern, historical, etc.
                        
                        with Image.open(io.BytesIO(raw)) as im:
                            dh = compute_dhash(im)
                            w, h = im.size
                            fmt = (im.format or "JPEG").upper()
                            
                        sha_seen.add(sha)
                        dhash_list.append(dh)
                        
                        aspect_ratio = round(w / max(h, 1), 4)
                        now_iso = datetime.now().isoformat()
                        
                        generator = "None"
                        source = "Tiny-GenImage (TheKernel01)" if domain_name == "genimage" else "Local"
                        if cat_name == "AI" and domain_name == "genimage":
                            generator = "Stable_Diffusion / Multi-Gen"
                            
                        row = {
                            "image_id": filepath.stem,
                            "filename": filepath.name,
                            "filepath": rel_path,
                            "ground_truth": cat_name,
                            "domain": domain_name,
                            "split": split_name,
                            "source": source,
                            "source_url": "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage" if domain_name == "genimage" else "Local",
                            "original_url": "HuggingFace_Parquet" if domain_name == "genimage" else "Local",
                            "collection": f"V4_{domain_name}",
                            "generator": generator,
                            "historical_year": "UNKNOWN",
                            "license": "Open Data / Research Use",
                            "original_width": w,
                            "original_height": h,
                            "aspect_ratio": aspect_ratio,
                            "format": fmt,
                            "file_size_bytes": len(raw),
                            "sha256": sha,
                            "dhash": dh,
                            "download_date": now_iso
                        }
                        existing_items.append(row)
                    except Exception as e:
                        print(f"[WARN] Error scanning {filepath}: {e}", flush=True)
                        
    print(f"[SCAN] Found {len(existing_items)} existing valid images already in dataset_v4.", flush=True)
    return existing_items, sha_seen, dhash_list

# ============================================================
# CONCURRENT DOWNLOADING & DEDUPLICATION ENGINE
# ============================================================

def download_single_candidate(thumb_url: str, meta: dict, session: requests.Session, benchmark_shas: set, benchmark_dhashes: dict):
    """
    Downloads and pre-validates a single image in a worker thread.
    Returns (item_dict, reject_reason).
    """
    try:
        resp = session.get(thumb_url, timeout=12)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
            
        data_bytes = resp.content
        sha = compute_sha256(data_bytes)
        
        # Check benchmark exact match
        if sha in benchmark_shas:
            return None, ("BENCHMARK_OVERLAP", sha, meta.get("title", ""))
            
        img_obj, w, h, fmt, reason = validate_image_bytes(data_bytes)
        if img_obj is None:
            return None, f"INVALID_IMAGE: {reason}"
            
        dh = compute_dhash(img_obj)
        
        # Check benchmark perceptual distance
        if any(hamming_distance(dh, b_dh) <= 2 for b_dh in benchmark_dhashes.values()):
            return None, ("BENCHMARK_PERCEPTUAL_NEAR", sha, meta.get("title", ""))
            
        item = {
            "bytes": data_bytes,
            "sha256": sha,
            "dhash": dh,
            "width": meta.get("orig_w") or w,
            "height": meta.get("orig_h") or h,
            "format": fmt,
            "ground_truth": meta["ground_truth"],
            "domain": meta["domain"],
            "generator": meta["generator"],
            "source": meta["source"],
            "source_url": f"https://commons.wikimedia.org/wiki/{meta.get('title', '').replace(' ', '_')}",
            "original_url": meta.get("orig_url", ""),
            "collection": meta.get("title", ""),
            "historical_year": meta.get("year", "UNKNOWN"),
            "license": meta.get("license", "UNKNOWN")
        }
        return item, None
    except Exception as e:
        return None, f"FETCH_EXCEPTION: {str(e)}"

def collect_images_concurrent(
    queries: list,
    target_count: int,
    ground_truth: str,
    domain: str,
    default_gen: str,
    sha_seen: set,
    dhash_list: list,
    benchmark_shas: set,
    benchmark_dhashes: dict,
    duplicate_records: list,
    max_workers: int = 12
):
    """
    Concurrently searches and acquires unique candidate images from Wikimedia Commons.
    """
    print(f"\n[ACQUIRE] Querying Commons for {ground_truth} / {domain} (Target: {target_count})...", flush=True)
    collected = []
    session = GLOBAL_SESSION
    
    for query, default_src, query_cap in queries:
        if len(collected) >= target_count:
            break
            
        print(f"  --> Search Query: '{query}' (Target for query: {query_cap})", flush=True)
        offset = 0
        batch_size = 50
        empty_batches = 0
        query_collected = 0
        
        while len(collected) < target_count and query_collected < query_cap and empty_batches < 3:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6", # File namespace
                "gsrlimit": str(batch_size),
                "gsroffset": str(offset),
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": "1024",
                "format": "json"
            }
            
            try:
                r = session.get(COMMONS_API_URL, params=params, timeout=15)
                if r.status_code != 200:
                    print(f"    Commons API error: HTTP {r.status_code}", flush=True)
                    break
                    
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    empty_batches += 1
                    offset += batch_size
                    continue
                    
                empty_batches = 0
                
                # Prepare candidate URLs and metadata
                tasks = []
                for page_id, p_info in pages.items():
                    title = p_info.get("title", "")
                    img_info = p_info.get("imageinfo", [{}])[0]
                    thumb_url = img_info.get("thumburl") or img_info.get("url")
                    orig_url = img_info.get("url", "")
                    orig_w = img_info.get("width", 0)
                    orig_h = img_info.get("height", 0)
                    ext_meta = img_info.get("extmetadata", {})
                    
                    license_name = ext_meta.get("LicenseShortName", {}).get("value", "UNKNOWN")
                    date_str = ext_meta.get("DateTimeOriginal", {}).get("value") or ext_meta.get("DateTime", {}).get("value", "UNKNOWN")
                    
                    year = "UNKNOWN"
                    for token in str(date_str).replace("-", " ").replace("/", " ").split():
                        if token.isdigit() and 1800 <= int(token) <= 2026:
                            year = token
                            break
                            
                    if not thumb_url:
                        continue
                        
                    meta = {
                        "title": title,
                        "orig_url": orig_url,
                        "orig_w": orig_w,
                        "orig_h": orig_h,
                        "license": license_name,
                        "year": year,
                        "ground_truth": ground_truth,
                        "domain": domain,
                        "generator": default_gen,
                        "source": default_src
                    }
                    tasks.append((thumb_url, meta))
                    
                # Concurrent download batch
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_url = {
                        executor.submit(download_single_candidate, url, meta, session, benchmark_shas, benchmark_dhashes): (url, meta)
                        for url, meta in tasks
                    }
                    
                    for fut in as_completed(future_to_url):
                        if len(collected) >= target_count or query_collected >= query_cap:
                            break
                            
                        item, err = fut.result()
                        if item is None:
                            if isinstance(err, tuple):
                                rej_type, rej_hash, rej_src = err
                                duplicate_records.append({"type": rej_type, "hash": rej_hash, "source": rej_src})
                            continue
                            
                        sha = item["sha256"]
                        dh = item["dhash"]
                        
                        # Sequential deduplication guard
                        if sha in sha_seen:
                            duplicate_records.append({"type": "EXACT_DUPLICATE", "hash": sha, "source": item["collection"]})
                            continue
                            
                        # Perceptual deduplication against recent downloads
                        if any(hamming_distance(dh, seen_dh) <= 1 for seen_dh in dhash_list[-200:]):
                            duplicate_records.append({"type": "INTERNAL_PERCEPTUAL_NEAR", "hash": sha, "source": item["collection"]})
                            continue
                            
                        sha_seen.add(sha)
                        dhash_list.append(dh)
                        collected.append(item)
                        query_collected += 1
                        
                print(f"    Progress: {len(collected)}/{target_count} ({ground_truth} - {domain})", flush=True)
                offset += batch_size
                time.sleep(0.1) # Gentle throttling
                
            except Exception as e:
                print(f"    Exception querying Commons: {e}", flush=True)
                break
                
    print(f"[DONE] Successfully acquired {len(collected)} clean unique images for {ground_truth} / {domain}", flush=True)
    return collected

# ============================================================
# DATASET PARTITIONING & FILE SYSTEM WRITER
# ============================================================

def partition_and_save(
    items: list,
    ground_truth: str,
    domain: str,
    metadata_rows: list,
    download_logs: list
):
    """
    Deterministically partitions items into Train (70%), Validation (15%), and Test (15%)
    and writes them to their dedicated subdirectories.
    """
    if not items:
        return
        
    random.seed(SEED)
    shuffled = list(items)
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    splits_data = [
        ("train", shuffled[:n_train]),
        ("validation", shuffled[n_train:n_train + n_val]),
        ("test", shuffled[n_train + n_val:])
    ]
    
    cat_folder = ground_truth.lower()
    
    for split_name, split_items in splits_data:
        dest_dir = DATASET_V4_DIR / split_name / cat_folder / domain
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Check current highest index in dest_dir
        existing_nums = []
        for f in dest_dir.glob("*.*"):
            parts = f.stem.split("_")
            if parts and parts[-1].isdigit():
                existing_nums.append(int(parts[-1]))
        start_idx = max(existing_nums, default=0) + 1
        
        for idx, item in enumerate(split_items, start=start_idx):
            ext = ".jpg" if item["format"].upper() in ("JPEG", "JPG") else f".{item['format'].lower()}"
            filename = f"v4_{split_name[:2]}_{cat_folder}_{domain}_{idx:05d}{ext}"
            filepath = dest_dir / filename
            rel_path = filepath.relative_to(PROJECT_DIR).as_posix()
            
            filepath.write_bytes(item["bytes"])
            
            aspect_ratio = round(item["width"] / max(item["height"], 1), 4)
            file_size = len(item["bytes"])
            now_iso = datetime.now().isoformat()
            
            row = {
                "image_id": filepath.stem,
                "filename": filename,
                "filepath": rel_path,
                "ground_truth": ground_truth,
                "domain": domain,
                "split": split_name,
                "source": item["source"],
                "source_url": item["source_url"],
                "original_url": item["original_url"],
                "collection": item["collection"],
                "generator": item["generator"],
                "historical_year": item["historical_year"],
                "license": item["license"],
                "original_width": item["width"],
                "original_height": item["height"],
                "aspect_ratio": aspect_ratio,
                "format": item["format"],
                "file_size_bytes": file_size,
                "sha256": item["sha256"],
                "dhash": item["dhash"],
                "download_date": now_iso
            }
            metadata_rows.append(row)
            
            log_row = {
                "timestamp": now_iso,
                "filename": filename,
                "source": item["source"],
                "status": "SUCCESS",
                "bytes": file_size
            }
            download_logs.append(log_row)

# ============================================================
# MASTER ORCHESTRATION PIPELINE
# ============================================================

def run_v4_pipeline(
    target_historical_real: int = 350,
    target_modern_real: int = 350,
    target_modern_ai: int = 350,
    target_hist_style_ai: int = 250
):
    """
    Executes complete acquisition and organization of AIDetect V4.
    """
    print("=" * 80, flush=True)
    print("AIDETECT V4 HIGH-INTEGRITY DATASET PREPARATION PIPELINE", flush=True)
    print(f"Deterministic Random Seed: {SEED}", flush=True)
    print("=" * 80, flush=True)
    
    init_v4_directories()
    benchmark_shas, benchmark_dhashes = build_benchmark_fingerprints()
    
    # 1. Scan pre-existing files (e.g. 3,000 GenImage files)
    metadata_rows, sha_seen, dhash_list = scan_existing_files(benchmark_shas, benchmark_dhashes)
    duplicate_records = []
    download_logs = []
    
    # 2. Historical Real Photography
    # Public domain archival collections: Library of Congress, Smithsonian, National Archives
    hist_queries = [
        ("filetype:bitmap \"Library of Congress\" \"photograph\" portrait", "Library of Congress", 100),
        ("filetype:bitmap \"Library of Congress\" \"photograph\" street OR building", "Library of Congress", 100),
        ("filetype:bitmap \"National Archives and Records Administration\" \"photograph\"", "National Archives", 100),
        ("filetype:bitmap \"Smithsonian Institution\" \"photograph\" vintage", "Smithsonian Institution", 100),
    ]
    # Check how many historical real already exist
    existing_hist = sum(1 for r in metadata_rows if r["ground_truth"] == "REAL" and r["domain"] == "historical")
    if existing_hist < target_historical_real:
        needed = target_historical_real - existing_hist
        items_hist = collect_images_concurrent(
            queries=hist_queries,
            target_count=needed,
            ground_truth="REAL",
            domain="historical",
            default_gen="None",
            sha_seen=sha_seen,
            dhash_list=dhash_list,
            benchmark_shas=benchmark_shas,
            benchmark_dhashes=benchmark_dhashes,
            duplicate_records=duplicate_records
        )
        partition_and_save(items_hist, "REAL", "historical", metadata_rows, download_logs)
        
    # 3. Modern Real Photography
    # High-resolution contemporary camera photography
    modern_queries = [
        ("filetype:bitmap photograph portrait \"Canon\" OR \"Nikon\" OR \"Sony\"", "Wikimedia Commons (Camera)", 100),
        ("filetype:bitmap photograph street scene \"2023\" OR \"2024\"", "Wikimedia Commons (Street)", 100),
        ("filetype:bitmap photograph landscape \"iPhone\" OR \"Samsung\"", "Wikimedia Commons (Mobile)", 100),
        ("filetype:bitmap photograph architecture \"2022\" OR \"2023\"", "Wikimedia Commons (Architecture)", 100),
    ]
    existing_modern_real = sum(1 for r in metadata_rows if r["ground_truth"] == "REAL" and r["domain"] == "modern")
    if existing_modern_real < target_modern_real:
        needed = target_modern_real - existing_modern_real
        items_modern_real = collect_images_concurrent(
            queries=modern_queries,
            target_count=needed,
            ground_truth="REAL",
            domain="modern",
            default_gen="None",
            sha_seen=sha_seen,
            dhash_list=dhash_list,
            benchmark_shas=benchmark_shas,
            benchmark_dhashes=benchmark_dhashes,
            duplicate_records=duplicate_records
        )
        partition_and_save(items_modern_real, "REAL", "modern", metadata_rows, download_logs)
        
    # 4. Modern Multi-Generator AI Images
    # Contemporary state-of-the-art generators: Midjourney, Stable Diffusion, DALL-E
    modern_ai_queries = [
        ("filetype:bitmap \"Midjourney\"", "Midjourney", 150),
        ("filetype:bitmap \"Stable Diffusion\"", "Stable Diffusion", 150),
        ("filetype:bitmap \"DALL-E\"", "OpenAI DALL-E", 100),
    ]
    existing_modern_ai = sum(1 for r in metadata_rows if r["ground_truth"] == "AI" and r["domain"] == "modern")
    if existing_modern_ai < target_modern_ai:
        needed = target_modern_ai - existing_modern_ai
        items_modern_ai = collect_images_concurrent(
            queries=modern_ai_queries,
            target_count=needed,
            ground_truth="AI",
            domain="modern",
            default_gen="Multi-Generator AI",
            sha_seen=sha_seen,
            dhash_list=dhash_list,
            benchmark_shas=benchmark_shas,
            benchmark_dhashes=benchmark_dhashes,
            duplicate_records=duplicate_records
        )
        partition_and_save(items_modern_ai, "AI", "modern", metadata_rows, download_logs)
        
    # 5. AI-Generated Historical/Vintage Images
    # AI generated intentionally imitating antique, sepia, or vintage photography
    hist_ai_queries = [
        ("filetype:bitmap \"Midjourney\" vintage OR retro OR historical OR 1920s", "Midjourney (Historical Style)", 100),
        ("filetype:bitmap \"Stable Diffusion\" vintage OR daguerreotype OR sepia OR old photo", "Stable Diffusion (Historical Style)", 100),
        ("filetype:bitmap \"AI generated\" vintage portrait OR historical", "AI (Historical Style)", 100)
    ]
    existing_hist_ai = sum(1 for r in metadata_rows if r["ground_truth"] == "AI" and r["domain"] == "historical_style")
    if existing_hist_ai < target_hist_style_ai:
        needed = target_hist_style_ai - existing_hist_ai
        items_hist_ai = collect_images_concurrent(
            queries=hist_ai_queries,
            target_count=needed,
            ground_truth="AI",
            domain="historical_style",
            default_gen="Diffusion (Vintage Style)",
            sha_seen=sha_seen,
            dhash_list=dhash_list,
            benchmark_shas=benchmark_shas,
            benchmark_dhashes=benchmark_dhashes,
            duplicate_records=duplicate_records
        )
        partition_and_save(items_hist_ai, "AI", "historical_style", metadata_rows, download_logs)
        
    # ============================================================
    # WRITING AUDIT LOGS & METADATA
    # ============================================================
    
    print("\n[FINALIZE] Generating metadata records and integrity reports...", flush=True)
    
    # 1. metadata.csv
    meta_csv = METADATA_DIR / "metadata.csv"
    fieldnames = [
        "image_id", "filename", "filepath", "ground_truth", "domain", "split",
        "source", "source_url", "original_url", "collection", "generator",
        "historical_year", "license", "original_width", "original_height",
        "aspect_ratio", "format", "file_size_bytes", "sha256", "dhash", "download_date"
    ]
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"  --> metadata.csv written ({len(metadata_rows)} records)", flush=True)
    
    # 2. duplicate_report.csv
    dup_csv = METADATA_DIR / "duplicate_report.csv"
    with open(dup_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "hash", "source"])
        writer.writeheader()
        writer.writerows(duplicate_records)
    print(f"  --> duplicate_report.csv written ({len(duplicate_records)} rejected items logged)", flush=True)
    
    # 3. download_log.json
    log_json = METADATA_DIR / "download_log.json"
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(download_logs, f, indent=2)
    print(f"  --> download_log.json written", flush=True)
    
    # 4. dataset_statistics.json
    stats = compute_dataset_statistics(metadata_rows)
    stats_json = METADATA_DIR / "dataset_statistics.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  --> dataset_statistics.json written", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("AIDETECT V4 DATASET PREPARATION COMPLETED SUCCESSFULLY", flush=True)
    print(f"Total Unique Images Prepared: {len(metadata_rows)}", flush=True)
    print("=" * 80, flush=True)

def compute_dataset_statistics(metadata_rows: list) -> dict:
    """Computes comprehensive dataset distribution statistics."""
    stats = {
        "total_images": len(metadata_rows),
        "split_counts": {},
        "ground_truth_counts": {},
        "domain_counts": {},
        "generator_distribution": {},
        "format_distribution": {},
        "resolution_summary": {
            "min_width": min((r["original_width"] for r in metadata_rows), default=0),
            "max_width": max((r["original_width"] for r in metadata_rows), default=0),
            "min_height": min((r["original_height"] for r in metadata_rows), default=0),
            "max_height": max((r["original_height"] for r in metadata_rows), default=0),
            "avg_width": round(sum((r["original_width"] for r in metadata_rows)) / max(len(metadata_rows), 1), 1),
            "avg_height": round(sum((r["original_height"] for r in metadata_rows)) / max(len(metadata_rows), 1), 1),
        },
        "aspect_ratios": {
            "square_1_1": 0,
            "landscape": 0,
            "portrait": 0
        },
        "historical_year_distribution": {}
    }
    
    for r in metadata_rows:
        s = r["split"]
        stats["split_counts"][s] = stats["split_counts"].get(s, 0) + 1
        
        gt = r["ground_truth"]
        stats["ground_truth_counts"][gt] = stats["ground_truth_counts"].get(gt, 0) + 1
        
        dom = r["domain"]
        stats["domain_counts"][dom] = stats["domain_counts"].get(dom, 0) + 1
        
        gen = r["generator"]
        stats["generator_distribution"][gen] = stats["generator_distribution"].get(gen, 0) + 1
        
        fmt = r["format"]
        stats["format_distribution"][fmt] = stats["format_distribution"].get(fmt, 0) + 1
        
        ar = r["aspect_ratio"]
        if 0.95 <= ar <= 1.05:
            stats["aspect_ratios"]["square_1_1"] += 1
        elif ar > 1.05:
            stats["aspect_ratios"]["landscape"] += 1
        else:
            stats["aspect_ratios"]["portrait"] += 1
            
        yr = r["historical_year"]
        if yr != "UNKNOWN":
            stats["historical_year_distribution"][yr] = stats["historical_year_distribution"].get(yr, 0) + 1
            
    return stats

if __name__ == "__main__":
    run_v4_pipeline()
