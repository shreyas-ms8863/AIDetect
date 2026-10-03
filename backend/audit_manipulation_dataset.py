"""
AIDetect Phase 5: Manipulation Dataset Integrity Audit
========================================================
Runs the 14-point automated audit on:
  - manipulation_v1/
  - manipulation_external_test/
Cross-checks against dataset_v4/ and external_test/ to verify zero leakage.
"""

import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_DIR     = BASE_DIR.parent
MANIP_DIR       = PROJECT_DIR / "manipulation_v1"
EXT_TEST_DIR    = PROJECT_DIR / "manipulation_external_test"
V4_DIR          = PROJECT_DIR / "dataset_v4"
OLD_EXT_TEST    = BASE_DIR / "external_test"
OLD_REAL_TEST   = BASE_DIR / "real_world_test"

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

checks = {}

def record(name, status, detail=""):
    checks[name] = (status == PASS)
    msg = f"  {status} {name}"
    if detail:
        msg += f"\n         -> {detail}"
    print(msg)
    sys.stdout.flush()

print("=" * 70)
print("PHASE 5: MANIPULATION DATASET 14-POINT INTEGRITY AUDIT")
print("=" * 70)
print(f"Dataset under audit: {MANIP_DIR}")
print(f"External Benchmark : {EXT_TEST_DIR}")
print()
sys.stdout.flush()

# Load metadata.csv
meta_path = MANIP_DIR / "metadata" / "metadata.csv"
record("metadata.csv exists", PASS if meta_path.exists() else FAIL, str(meta_path))

with open(meta_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total_rows = len(rows)
print(f"Total rows in metadata: {total_rows}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# CHECK 1 & 2: Existing files & PIL decodability
# ---------------------------------------------------------------------------
print("\nVerifying file existence and PIL decodability...")
missing_files = []
corrupted_files = []
valid_files = 0

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
        missing_files.append(rel_p)
    else:
        try:
            with Image.open(p) as img:
                img.verify()
            valid_files += 1
        except Exception:
            corrupted_files.append(rel_p)

record("Check 1: Every metadata row points to an existing file",
       PASS if len(missing_files) == 0 else FAIL,
       f"Valid: {valid_files}/{total_rows}, Missing: {len(missing_files)}")

record("Check 2: Every image is PIL-decodable",
       PASS if len(corrupted_files) == 0 else FAIL,
       f"Corrupted: {len(corrupted_files)}")

# ---------------------------------------------------------------------------
# CHECK 3 & 4: SHA-256 and dHash integrity
# ---------------------------------------------------------------------------
has_sha = all(len(r.get("sha256", "")) == 64 for r in rows)
has_dhash = all(len(r.get("dhash", "")) == 16 for r in rows)

record("Check 3: Every image has valid SHA-256",
       PASS if has_sha else FAIL)
record("Check 4: Every image has valid dHash (16 hex chars)",
       PASS if has_dhash else FAIL)

# ---------------------------------------------------------------------------
# CHECK 5: Cross-split exact duplicates
# ---------------------------------------------------------------------------
sha_by_split = defaultdict(set)
cross_split_exact = []
for r in rows:
    h = r["sha256"]
    s = r["split"]
    for other_s, other_hashes in sha_by_split.items():
        if other_s != s and h in other_hashes:
            cross_split_exact.append((r["image_id"], s, other_s))
    sha_by_split[s].add(h)

record("Check 5: No accidental cross-split exact duplicates",
       PASS if len(cross_split_exact) == 0 else FAIL,
       f"Cross-split duplicates found: {len(cross_split_exact)}")

# ---------------------------------------------------------------------------
# CHECK 6: Cross-split perceptual duplicates (unrelated pairs)
# ---------------------------------------------------------------------------
# We check if an original from one split matches an original from another split
orig_rows = [r for r in rows if r["label"] == "0"]
dhash_by_split = defaultdict(dict)
cross_split_perceptual = []

for r in orig_rows:
    dh = r["dhash"]
    s = r["split"]
    for other_s, other_dict in dhash_by_split.items():
        if other_s != s and dh in other_dict:
            cross_split_perceptual.append((r["image_id"], other_dict[dh], s, other_s))
    dhash_by_split[s][dh] = r["image_id"]

record("Check 6: No accidental cross-split perceptual duplicates (originals)",
       PASS if len(cross_split_perceptual) == 0 else FAIL,
       f"Collisions: {len(cross_split_perceptual)}")

# ---------------------------------------------------------------------------
# CHECK 7: Original/manipulated pairs remain in the same split
# ---------------------------------------------------------------------------
pairs_path = MANIP_DIR / "metadata" / "pair_relationships.json"
orig_split_map = {r["image_id"]: r["split"] for r in rows if r["label"] == "0"}
manip_split_map = {r["image_id"]: r["split"] for r in rows if r["label"] == "1"}

with open(pairs_path, encoding="utf-8") as f:
    pair_map = json.load(f)

leaked_pairs = []
for orig_img_id, manip_list in pair_map.items():
    orig_s = orig_split_map.get(orig_img_id)
    for m_id in manip_list:
        m_s = manip_split_map.get(m_id)
        if orig_s != m_s:
            leaked_pairs.append((orig_img_id, m_id, orig_s, m_s))

record("Check 7: Original/manipulated pairs remain in the exact same split",
       PASS if len(leaked_pairs) == 0 else FAIL,
       f"Leaked pairs: {len(leaked_pairs)}")

# ---------------------------------------------------------------------------
# CHECK 8: Category counts
# ---------------------------------------------------------------------------
cat_counts = Counter(r["manipulation_type"] for r in rows if r["label"] == "1")
print("\nManipulation Categories:")
for cat, cnt in cat_counts.most_common():
    print(f"    {cat:<26}: {cnt}")
record("Check 8: All 6 manipulation categories present",
       PASS if len(cat_counts) == 6 else FAIL,
       f"Categories present: {len(cat_counts)}/6")

# ---------------------------------------------------------------------------
# CHECK 9 & 10: Generator & Source counts
# ---------------------------------------------------------------------------
gen_counts = Counter(r["generator_or_editor"] for r in rows)
src_counts = Counter(r["source_dataset"] for r in rows)
print("\nGenerators:")
for g, c in gen_counts.items():
    print(f"    {g:<26}: {c}")
print("Sources:")
for s, c in src_counts.items():
    print(f"    {s:<26}: {c}")

record("Check 9: Generator counts recorded", PASS if len(gen_counts) >= 2 else FAIL)
record("Check 10: Source counts recorded", PASS if len(src_counts) >= 2 else FAIL)

# ---------------------------------------------------------------------------
# CHECK 11 & 12: Class balance & Split counts
# ---------------------------------------------------------------------------
cls_counts = Counter(r["ground_truth"] for r in rows)
split_counts = Counter(r["split"] for r in rows)
print("\nClass Counts:")
for cl, c in cls_counts.items():
    print(f"    {cl:<20}: {c}")
print("Split Counts:")
for sp, c in split_counts.items():
    print(f"    {sp:<20}: {c}")

record("Check 11: Binary class representation (ORIGINAL_REAL vs AI_MANIPULATED)",
       PASS if set(cls_counts.keys()) == {"ORIGINAL_REAL", "AI_MANIPULATED"} else FAIL)

record("Check 12: Train/Validation/Test splits present",
       PASS if set(split_counts.keys()) == {"train", "validation", "test"} else FAIL)

# ---------------------------------------------------------------------------
# CHECK 13: V4 Overlap Check
# ---------------------------------------------------------------------------
v4_meta_path = V4_DIR / "metadata" / "metadata.csv"
v4_overlap = []
if v4_meta_path.exists():
    with open(v4_meta_path, encoding="utf-8") as f:
        v4_rows = list(csv.DictReader(f))
    v4_shas = set(r["sha256"] for r in v4_rows)
    for r in rows:
        if r["sha256"] in v4_shas:
            v4_overlap.append(r["image_id"])

record("Check 13: Zero overlap with dataset_v4/",
       PASS if len(v4_overlap) == 0 else FAIL,
       f"Overlapping images: {len(v4_overlap)}")

# ---------------------------------------------------------------------------
# CHECK 14: External Benchmark Isolation Check
# ---------------------------------------------------------------------------
pie_manifest = EXT_TEST_DIR / "pie_bench" / "pie_bench_manifest.csv"
ext_overlap = []
if pie_manifest.exists():
    with open(pie_manifest, encoding="utf-8") as f:
        pie_rows = list(csv.DictReader(f))
    pie_shas = set(r["sha256"] for r in pie_rows)
    for r in rows:
        if r["sha256"] in pie_shas:
            ext_overlap.append(r["image_id"])

record("Check 14: Zero overlap with manipulation_external_test/ (PIE-Bench)",
       PASS if len(ext_overlap) == 0 else FAIL,
       f"Overlapping images: {len(ext_overlap)}")

# Also check old benchmarks
old_ext_shas = set()
for d in [OLD_EXT_TEST, OLD_REAL_TEST]:
    if d.exists():
        for p in d.rglob("*.*"):
            if p.is_file():
                old_ext_shas.add(hashlib.sha256(p.read_bytes()).hexdigest())

old_overlap = sum(1 for r in rows if r["sha256"] in old_ext_shas)
record("Zero overlap with existing external_test/ and real_world_test/",
       PASS if old_overlap == 0 else FAIL,
       f"Overlapping images: {old_overlap}")

# Summary
print()
print("=" * 70)
total_checks = len(checks)
passed_checks = sum(1 for v in checks.values() if v)
print(f"TOTAL AUDIT CHECKS: {total_checks} | PASSED: {passed_checks} | FAILED: {total_checks - passed_checks}")
if total_checks == passed_checks:
    print("MANIPULATION_V1 INTEGRITY AUDIT: 100% VERIFIED [PASS]")
else:
    print("MANIPULATION_V1 INTEGRITY AUDIT: FAILED (see above)")
print("=" * 70)
sys.stdout.flush()
