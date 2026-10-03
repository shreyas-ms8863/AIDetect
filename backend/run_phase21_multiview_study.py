"""
AIDetect - Phase 21: Multi-View Forensic Inference Robustness Study
Evaluates deterministic multi-view inference rules on frozen manipulation detector.
No training. No fine-tuning. Strict model and code freeze.
"""

import os
import sys
import io
import time
import json
import hashlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from backend.manipulation_freq_v1_dataset import AuthoritativeFrequencyTransform

MODELS_DIR = BASE_DIR / "models"
MANIP_CKPT_PATH = MODELS_DIR / "manipulation_frequency_resnet50_v1.pth"
GEN_CKPT_PATH = MODELS_DIR / "frequency_resnet50_v4.pth"
CALIB_PATH = BASE_DIR / "backend" / "phase12_results" / "strategy_e_calibration.json"
TEST_DIR = BASE_DIR / "manipulation_v1" / "test"
RESULTS_DIR = BASE_DIR / "backend" / "phase21_results"
SUSPECT_IMAGE_PATH = Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-10-02 at 1.52.03 PM.jpeg")
PHASE20_CACHE_PATH = BASE_DIR / "backend" / "phase20_results" / "phase20_raw_eval_cache.json"

EXPECTED_MANIP_SHA256 = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"
EXPECTED_GEN_SHA256   = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_CALIB_SHA256 = "30e1da6e2ca709cc6113d0d6726db01aef8a127e627869f195449da52571b1ec"
EXPECTED_SUSPECT_SHA256 = "3b1e6714c2709b5438e6c38f30ed4105f1a0184e3a5070c92b9bbf34eab5dece"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def pad_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    w, h = image.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 1e-4:
        return image
    if current_ratio < target_ratio:
        new_w = int(round(h * target_ratio))
        pad_total = new_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        canvas = Image.new("RGB", (new_w, h))
        canvas.paste(image, (pad_left, 0))
        left_strip = image.crop((0, 0, min(pad_left, w), h)).transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(left_strip, (pad_left - left_strip.size[0], 0))
        right_strip = image.crop((max(0, w - pad_right), 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(right_strip, (pad_left + w, 0))
        return canvas
    else:
        new_h = int(round(w / target_ratio))
        pad_total = new_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        canvas = Image.new("RGB", (w, new_h))
        canvas.paste(image, (0, pad_top))
        top_strip = image.crop((0, 0, w, min(pad_top, h))).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(top_strip, (0, pad_top - top_strip.size[1]))
        bottom_strip = image.crop((0, max(0, h - pad_bottom), w, h)).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(bottom_strip, (0, pad_top + h))
        return canvas


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, Any]:
    n = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    manip_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * manip_recall / (precision + manip_recall) if (precision + manip_recall) > 0 else 0.0

    real_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0.0

    macro_precision = (precision + real_precision) / 2.0
    macro_recall = (manip_recall + real_recall) / 2.0
    macro_f1 = (f1 + real_f1) / 2.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    probs_manip = [p for yt, p in zip(y_true, y_prob) if yt == 1]
    mean_p = float(np.mean(probs_manip)) if probs_manip else 0.0
    med_p = float(np.median(probs_manip)) if probs_manip else 0.0
    std_p = float(np.std(probs_manip)) if probs_manip else 0.0

    return {
        "accuracy": round(accuracy * 100, 2),
        "manipulation_recall": round(manip_recall * 100, 2),
        "real_recall": round(real_recall * 100, 2),
        "precision": round(precision * 100, 2),
        "f1": round(f1 * 100, 2),
        "macro_precision": round(macro_precision * 100, 2),
        "macro_recall": round(macro_recall * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mean_p_manip": round(mean_p * 100, 2),
        "median_p_manip": round(med_p * 100, 2),
        "std_p_manip": round(std_p * 100, 2),
    }


def main():
    print("=" * 80)
    print("AIDETECT PHASE 21: MULTI-VIEW FORENSIC INFERENCE ROBUSTNESS STUDY")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. VERIFY CHECKPOINT INTEGRITY
    manip_sha = compute_sha256(MANIP_CKPT_PATH)
    gen_sha = compute_sha256(GEN_CKPT_PATH)
    calib_sha = compute_sha256(CALIB_PATH)
    suspect_sha = compute_sha256(SUSPECT_IMAGE_PATH) if SUSPECT_IMAGE_PATH.exists() else "NOT_FOUND"

    print(f"Manipulation Checkpoint: {manip_sha} (Match: {manip_sha == EXPECTED_MANIP_SHA256})")
    print(f"Generation Checkpoint:   {gen_sha} (Match: {gen_sha == EXPECTED_GEN_SHA256})")
    print(f"Calibration JSON:        {calib_sha} (Match: {calib_sha == EXPECTED_CALIB_SHA256})")
    print(f"Suspect Image:           {suspect_sha} (Match: {suspect_sha == EXPECTED_SUSPECT_SHA256})")

    assert manip_sha == EXPECTED_MANIP_SHA256, "Manipulation SHA mismatch"
    assert gen_sha == EXPECTED_GEN_SHA256, "Generation SHA mismatch"
    assert calib_sha == EXPECTED_CALIB_SHA256, "Calibration SHA mismatch"
    assert suspect_sha == EXPECTED_SUSPECT_SHA256, "Suspect SHA mismatch"

    # Save Hashes and Environment
    with open(RESULTS_DIR / "phase21_hashes.json", "w", encoding="utf-8") as f:
        json.dump({
            "manipulation_checkpoint": {"path": str(MANIP_CKPT_PATH), "sha256": manip_sha, "expected": EXPECTED_MANIP_SHA256, "match": True},
            "generation_checkpoint": {"path": str(GEN_CKPT_PATH), "sha256": gen_sha, "expected": EXPECTED_GEN_SHA256, "match": True},
            "calibration_file": {"path": str(CALIB_PATH), "sha256": calib_sha, "expected": EXPECTED_CALIB_SHA256, "match": True},
            "suspect_image": {"path": str(SUSPECT_IMAGE_PATH), "sha256": suspect_sha, "expected": EXPECTED_SUSPECT_SHA256, "match": True},
        }, f, indent=2)

    with open(RESULTS_DIR / "phase21_environment.json", "w", encoding="utf-8") as f:
        json.dump({
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    # 2. LOAD DATASET METADATA
    test_real_dir = TEST_DIR / "original"
    test_manip_dir = TEST_DIR / "manipulated"

    real_files = sorted(list(test_real_dir.glob("*.png")) + list(test_real_dir.glob("*.jpg")))
    manip_subdirs = sorted([d for d in test_manip_dir.iterdir() if d.is_dir()])

    dataset_items = []
    for p in real_files:
        dataset_items.append({
            "sample_id": p.name,
            "category": "original",
            "ground_truth": 0,
            "file_format": p.suffix.lower().replace(".", ""),
            "path": p
        })

    for d in manip_subdirs:
        m_files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
        for p in m_files:
            dataset_items.append({
                "sample_id": p.name,
                "category": d.name,
                "ground_truth": 1,
                "file_format": p.suffix.lower().replace(".", ""),
                "path": p
            })

    print(f"\nDataset loaded: {len(dataset_items)} total images ({sum(1 for x in dataset_items if x['ground_truth']==0)} Real, {sum(1 for x in dataset_items if x['ground_truth']==1)} Manipulated)")

    # 3. GATHER 7 SINGLE-VIEW INFERENCE DATA
    # Views:
    # 1: Native
    # 2: 1:1 Reflect Pad
    # 3: 16:9 Reflect Pad
    # 4: 9:16 Reflect Pad
    # 5: 0.4586 Reflect Pad
    # 6: Native JPEG Q95
    # 7: Native JPEG Q75
    VIEW_MAP = {
        "view_1_native": "native_baseline",
        "view_2_pad_1:1": "pad_1:1_uncompressed",
        "view_3_pad_16:9": "pad_16:9_uncompressed",
        "view_4_pad_9:16": "pad_9:16_uncompressed",
        "view_5_pad_0.4586": "pad_0.4586_uncompressed",
        "view_6_jpeg_95": "native_jpeg_95",
        "view_7_jpeg_75": "native_jpeg_75",
    }

    # Load from verified Phase 20 evaluation cache
    print(f"Loading raw evaluations from {PHASE20_CACHE_PATH}...")
    with open(PHASE20_CACHE_PATH, "r", encoding="utf-8") as f:
        cache_data = json.load(f)["per_image_results"]

    # Build per-image view probabilities
    # image_views_data: dict[sample_id] -> dict[view_key] -> p_manip (0..1)
    image_views_data = {}
    sample_meta = {}

    for item in dataset_items:
        sid = item["sample_id"]
        sample_meta[sid] = item
        image_views_data[sid] = {}

    for row in cache_data:
        sid = row["sample_id"]
        cid = row["condition_id"]
        for v_key, c_key in VIEW_MAP.items():
            if cid == c_key:
                image_views_data[sid][v_key] = row["P_manipulated_raw"]
                if "width" not in sample_meta[sid]:
                    sample_meta[sid]["width"] = row["original_width"]
                    sample_meta[sid]["height"] = row["original_height"]
                    sample_meta[sid]["aspect_ratio"] = row["original_aspect_ratio"]

    print("[OK] View probabilities compiled for all 248 images across 7 views.")

    # 4. EVALUATE SINGLE VIEWS
    print("\n--- SECTION 5: SINGLE-VIEW BASELINES ---")
    single_view_results = {}
    for v_key in VIEW_MAP.keys():
        y_true = [sample_meta[sid]["ground_truth"] for sid in sample_meta]
        y_prob = [image_views_data[sid][v_key] for sid in sample_meta]
        y_pred = [1 if p >= 0.50 else 0 for p in y_prob]
        m = calculate_metrics(y_true, y_pred, y_prob)
        single_view_results[v_key] = m
        print(f"  {v_key:<20}: Acc={m['accuracy']:5.2f}% | ManipRec={m['manipulation_recall']:5.2f}% | RealRec={m['real_recall']:5.2f}% | FPR={m['fpr']:5.2f}% | F1={m['f1']:5.2f}% | MeanP={m['mean_p_manip']:5.2f}%")

    # 5. MULTI-VIEW FUSION SETS
    # SET A: Native
    # SET B: Native + 1:1
    # SET C: Native + 9:16
    # SET D: Native + 0.4586
    # SET E: Native + 16:9
    # SET F: Native + 1:1 + 9:16
    # SET G: Native + 1:1 + 9:16 + 16:9
    # SET H: Native + 1:1 + 9:16 + 0.4586 + 16:9
    # SET I: Native + JPEG95
    # SET J: Native + JPEG75
    # SET K: Native + 1:1 + 9:16 + 16:9 + JPEG95
    # SET L: Native + 1:1 + 9:16 + 0.4586 + 16:9 + JPEG75
    VIEW_SETS = {
        "SET_A_Native": ["view_1_native"],
        "SET_B_Native+1:1": ["view_1_native", "view_2_pad_1:1"],
        "SET_C_Native+9:16": ["view_1_native", "view_4_pad_9:16"],
        "SET_D_Native+0.4586": ["view_1_native", "view_5_pad_0.4586"],
        "SET_E_Native+16:9": ["view_1_native", "view_3_pad_16:9"],
        "SET_F_Native+1:1+9:16": ["view_1_native", "view_2_pad_1:1", "view_4_pad_9:16"],
        "SET_G_Native+1:1+9:16+16:9": ["view_1_native", "view_2_pad_1:1", "view_4_pad_9:16", "view_3_pad_16:9"],
        "SET_H_Native+1:1+9:16+0.4586+16:9": ["view_1_native", "view_2_pad_1:1", "view_4_pad_9:16", "view_5_pad_0.4586", "view_3_pad_16:9"],
        "SET_I_Native+JPEG95": ["view_1_native", "view_6_jpeg_95"],
        "SET_J_Native+JPEG75": ["view_1_native", "view_7_jpeg_75"],
        "SET_K_Native+1:1+9:16+16:9+JPEG95": ["view_1_native", "view_2_pad_1:1", "view_4_pad_9:16", "view_3_pad_16:9", "view_6_jpeg_95"],
        "SET_L_Native+1:1+9:16+0.4586+16:9+JPEG75": ["view_1_native", "view_2_pad_1:1", "view_4_pad_9:16", "view_5_pad_0.4586", "view_3_pad_16:9", "view_7_jpeg_75"],
    }

    print("\n--- SECTION 8: PRIMARY VIEW SETS x FUSION RULES ---")
    fusion_results = {}
    fused_probabilities = {}  # (set_key, rule) -> dict[sid] -> float

    for set_key, v_list in VIEW_SETS.items():
        fusion_results[set_key] = {}

        # 4 Fusion rules: Mean, Median, Max, Trimmed Mean
        rules = ["mean", "median", "max"]
        if len(v_list) >= 3:
            rules.append("trimmed_mean")

        for rule in rules:
            probs = []
            sid_to_p = {}
            for sid in sample_meta:
                v_vals = [image_views_data[sid][v] for v in v_list]
                if rule == "mean":
                    p_fused = float(np.mean(v_vals))
                elif rule == "median":
                    p_fused = float(np.median(v_vals))
                elif rule == "max":
                    p_fused = float(np.max(v_vals))
                elif rule == "trimmed_mean":
                    # Remove min and max, average remainder
                    s_vals = sorted(v_vals)
                    p_fused = float(np.mean(s_vals[1:-1]))
                probs.append(p_fused)
                sid_to_p[sid] = p_fused

            fused_probabilities[(set_key, rule)] = sid_to_p
            y_true = [sample_meta[sid]["ground_truth"] for sid in sample_meta]
            y_pred = [1 if p >= 0.50 else 0 for p in probs]
            m = calculate_metrics(y_true, y_pred, probs)
            fusion_results[set_key][rule] = m

            print(f"  {set_key:<42} [{rule:<12}]: ManipRec={m['manipulation_recall']:5.2f}% | RealRec={m['real_recall']:5.2f}% | FPR={m['fpr']:5.2f}% | F1={m['f1']:5.2f}% | MeanP={m['mean_p_manip']:5.2f}%")

    # 6. FALSE NEGATIVE RECOVERY ANALYSIS (SECTION 10)
    print("\n--- SECTION 10: FALSE NEGATIVE RECOVERY ANALYSIS ---")
    # Identify Native FNs
    manip_sids = [sid for sid in sample_meta if sample_meta[sid]["ground_truth"] == 1]
    native_fns = [sid for sid in manip_sids if image_views_data[sid]["view_1_native"] < 0.50]
    print(f"Native Manipulation False Negatives: {len(native_fns)} / {len(manip_sids)} ({len(native_fns)/len(manip_sids)*100:.2f}%)")

    fn_recovery_table = []
    for sid in native_fns:
        row = {
            "sample_id": sid,
            "category": sample_meta[sid]["category"],
            "native_p": round(image_views_data[sid]["view_1_native"] * 100, 2),
            "views_p": {v: round(image_views_data[sid][v] * 100, 2) for v in VIEW_MAP.keys()},
            "views_pred": {v: ("TP" if image_views_data[sid][v] >= 0.50 else "FN") for v in VIEW_MAP.keys()}
        }
        fn_recovery_table.append(row)
        print(f"  FN: {sid:<24} | Cat: {row['category']:<20} | Native P: {row['native_p']:5.2f}%")
        for v, p in row["views_p"].items():
            print(f"     {v:<22}: {p:5.2f}% ({row['views_pred'][v]})")

    # Recovery counts by view
    view_recovery_counts = {}
    for v in VIEW_MAP.keys():
        if v == "view_1_native": continue
        view_recovery_counts[v] = sum(1 for sid in native_fns if image_views_data[sid][v] >= 0.50)
    print("Recovery count by alternative view:")
    for v, c in view_recovery_counts.items():
        print(f"  {v:<22}: {c} / {len(native_fns)} ({c/len(native_fns)*100:.1f}%)")

    # 7. FALSE POSITIVE ANALYSIS ON REAL IMAGES (SECTION 11)
    print("\n--- SECTION 11: FALSE POSITIVE ANALYSIS ON REAL IMAGES ---")
    real_sids = [sid for sid in sample_meta if sample_meta[sid]["ground_truth"] == 0]
    native_fps = [sid for sid in real_sids if image_views_data[sid]["view_1_native"] >= 0.50]
    print(f"Native Real False Positives: {len(native_fps)} / {len(real_sids)} (FPR: {len(native_fps)/len(real_sids)*100:.2f}%)")

    fp_analysis = {}
    for v in VIEW_MAP.keys():
        fps = [sid for sid in real_sids if image_views_data[sid][v] >= 0.50]
        new_fps = set(fps) - set(native_fps)
        cleared_fps = set(native_fps) - set(fps)
        fp_analysis[v] = {
            "total_fp": len(fps),
            "fpr": round(len(fps) / len(real_sids) * 100, 2),
            "new_fp_count": len(new_fps),
            "cleared_fp_count": len(cleared_fps),
        }
        print(f"  {v:<22}: Total FP={len(fps):2d} (FPR={fp_analysis[v]['fpr']:5.2f}%) | New FP={len(new_fps):2d} | Cleared={len(cleared_fps):2d}")

    # FP under fusion sets
    fusion_fp_analysis = {}
    for set_key, v_list in VIEW_SETS.items():
        fusion_fp_analysis[set_key] = {}
        for rule in ["mean", "median", "max"]:
            probs = [fused_probabilities[(set_key, rule)][sid] for sid in real_sids]
            fps = sum(1 for p in probs if p >= 0.50)
            fusion_fp_analysis[set_key][rule] = {
                "total_fp": fps,
                "fpr": round(fps / len(real_sids) * 100, 2),
                "additional_fp_vs_native": fps - len(native_fps)
            }
            print(f"  Fusion {set_key:<36} [{rule:<6}]: Total FP={fps:2d} (FPR={fps/len(real_sids)*100:5.2f}%) | AddFP={fps - len(native_fps):+2d}")

    # 8. CATEGORY ANALYSIS (SECTION 12)
    print("\n--- SECTION 12: CATEGORY ANALYSIS ---")
    categories = sorted(list(set(sample_meta[sid]["category"] for sid in manip_sids)))
    category_summary = {}

    for cat in categories:
        c_sids = [sid for sid in manip_sids if sample_meta[sid]["category"] == cat]
        n_c = len(c_sids)

        # Native recall
        nat_rec = sum(1 for sid in c_sids if image_views_data[sid]["view_1_native"] >= 0.50) / n_c * 100

        # All views recall
        v_recs = {}
        for v in VIEW_MAP.keys():
            v_recs[v] = sum(1 for sid in c_sids if image_views_data[sid][v] >= 0.50) / n_c * 100

        # Best single alternative view
        alt_views = {v: r for v, r in v_recs.items() if v != "view_1_native"}
        best_alt_v = max(alt_views, key=alt_views.get)
        best_alt_rec = alt_views[best_alt_v]
        least_v = min(v_recs, key=v_recs.get)

        # Best fusion recall
        f_recs = {}
        for (sk, rule), pmap in fused_probabilities.items():
            f_recs[f"{sk}_{rule}"] = sum(1 for sid in c_sids if pmap[sid] >= 0.50) / n_c * 100
        best_f_key = max(f_recs, key=f_recs.get)
        best_f_rec = f_recs[best_f_key]

        # Recovery count in category
        cat_fns = [sid for sid in c_sids if image_views_data[sid]["view_1_native"] < 0.50]
        rec_count = sum(1 for sid in cat_fns if any(image_views_data[sid][v] >= 0.50 for v in alt_views))

        category_summary[cat] = {
            "n": n_c,
            "native_recall": round(nat_rec, 2),
            "best_alt_view": best_alt_v,
            "best_alt_recall": round(best_alt_rec, 2),
            "best_fusion_key": best_f_key,
            "best_fusion_recall": round(best_f_rec, 2),
            "recovery_count": rec_count,
            "native_fn_count": len(cat_fns),
            "most_useful_view": best_alt_v,
            "least_useful_view": least_v,
            "view_recalls": {v: round(r, 2) for v, r in v_recs.items()}
        }

        print(f"  {cat:<24} (N={n_c:2d}): Native={nat_rec:5.2f}% | BestAlt({best_alt_v})={best_alt_rec:5.2f}% | BestFusion={best_f_rec:5.2f}% | RecCount={rec_count}/{len(cat_fns)}")

    # 9. COMPRESSION-SPECIFIC RECOVERY ANALYSIS (SECTION 13)
    print("\n--- SECTION 13: COMPRESSION-SPECIFIC RECOVERY ANALYSIS ---")
    # For Q95
    q95_fns = [sid for sid in manip_sids if image_views_data[sid]["view_6_jpeg_95"] < 0.50]
    # For Q75
    q75_fns = [sid for sid in manip_sids if image_views_data[sid]["view_7_jpeg_75"] < 0.50]

    uncompressed_views = ["view_1_native", "view_2_pad_1:1", "view_3_pad_16:9", "view_4_pad_9:16", "view_5_pad_0.4586"]

    q95_rec_by_uncomp = sum(1 for sid in q95_fns if any(image_views_data[sid][v] >= 0.50 for v in uncompressed_views))
    q75_rec_by_uncomp = sum(1 for sid in q75_fns if any(image_views_data[sid][v] >= 0.50 for v in uncompressed_views))

    # Per-view breakdown for Q95 and Q75
    q95_by_view = {v: sum(1 for sid in q95_fns if image_views_data[sid][v] >= 0.50) for v in uncompressed_views}
    q75_by_view = {v: sum(1 for sid in q75_fns if image_views_data[sid][v] >= 0.50) for v in uncompressed_views}

    print(f"Q95 False Negatives: {len(q95_fns)} / {len(manip_sids)}")
    print(f"  Recovered by uncompressed views: {q95_rec_by_uncomp} ({q95_rec_by_uncomp/len(q95_fns)*100:.2f}%)")
    for v, c in q95_by_view.items():
        print(f"    {v:<22}: {c:2d} ({c/len(q95_fns)*100:.2f}%)")

    print(f"\nQ75 False Negatives: {len(q75_fns)} / {len(manip_sids)}")
    print(f"  Recovered by uncompressed views: {q75_rec_by_uncomp} ({q75_rec_by_uncomp/len(q75_fns)*100:.2f}%)")
    for v, c in q75_by_view.items():
        print(f"    {v:<22}: {c:2d} ({c/len(q75_fns)*100:.2f}%)")

    # 10. SUSPECT IMAGE EVALUATION (SECTION 14)
    print("\n--- SECTION 14: SUSPECT IMAGE ANALYSIS ---")
    suspect_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManipulationFrequencyResNet50V1(dropout_p=0.3)
    ckpt = torch.load(MANIP_CKPT_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    xform = AuthoritativeFrequencyTransform()

    with Image.open(SUSPECT_IMAGE_PATH) as raw:
        s_img = raw.convert("RGB")

    suspect_view_images = {
        "view_1_native": s_img,
        "view_2_pad_1:1": pad_to_ratio(s_img, 1.0),
        "view_3_pad_16:9": pad_to_ratio(s_img, 16.0/9.0),
        "view_4_pad_9:16": pad_to_ratio(s_img, 9.0/16.0),
        "view_5_pad_0.4586": pad_to_ratio(s_img, 0.4586),
        "view_6_jpeg_95": jpeg_roundtrip(s_img, 95),
        "view_7_jpeg_75": jpeg_roundtrip(s_img, 75),
    }

    suspect_view_probs = {}
    for v_name, v_im in suspect_view_images.items():
        t = xform(v_im).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        p_orig, p_manip = float(probs[0]), float(probs[1])
        suspect_view_probs[v_name] = p_manip
        suspect_results[v_name] = {
            "p_orig": round(p_orig * 100, 2),
            "p_manip": round(p_manip * 100, 2),
            "prediction": "AI_MANIPULATED" if p_manip >= 0.50 else "ORIGINAL_REAL"
        }
        print(f"  {v_name:<20}: P(Manip) = {p_manip*100:5.2f}% | P(Orig) = {p_orig*100:5.2f}% ({suspect_results[v_name]['prediction']})")

    # Suspect Fusion across all sets
    suspect_fusions = {}
    for sk, v_list in VIEW_SETS.items():
        v_probs = [suspect_view_probs[v] for v in v_list]
        p_mean = float(np.mean(v_probs))
        p_med = float(np.median(v_probs))
        p_max = float(np.max(v_probs))
        suspect_fusions[sk] = {
            "mean": round(p_mean * 100, 2),
            "median": round(p_med * 100, 2),
            "max": round(p_max * 100, 2),
        }
        print(f"  {sk:<38}: Mean={p_mean*100:5.2f}% | Med={p_med*100:5.2f}% | Max={p_max*100:5.2f}%")

    # 11. STATISTICAL TESTING (SECTION 16)
    print("\n--- SECTION 16: STATISTICAL TESTING ---")
    stat_comparisons = [
        ("Native vs Set E (Max)", "SET_E_Native+16:9", "max"),
        ("Native vs Set G (Max)", "SET_G_Native+1:1+9:16+16:9", "max"),
        ("Native vs Set H (Max)", "SET_H_Native+1:1+9:16+0.4586+16:9", "max"),
        ("Native vs Set B (Mean)", "SET_B_Native+1:1", "mean"),
        ("Native vs Set C (Mean)", "SET_C_Native+9:16", "mean"),
        ("Native vs Set D (Mean)", "SET_D_Native+0.4586", "mean"),
        ("Native vs Set E (Mean)", "SET_E_Native+16:9", "mean"),
    ]

    stat_results = []
    base_probs = [image_views_data[sid]["view_1_native"] for sid in manip_sids]

    for label, sk, rule in stat_comparisons:
        cmp_probs = [fused_probabilities[(sk, rule)][sid] for sid in manip_sids]
        diffs = [c - b for c, b in zip(cmp_probs, base_probs)]
        mean_shift = float(np.mean(diffs)) * 100
        med_shift = float(np.median(diffs)) * 100

        # Paired Wilcoxon
        non_zero_diffs = [d for d in diffs if abs(d) > 1e-7]
        if len(non_zero_diffs) > 0:
            stat, p_val = wilcoxon(cmp_probs, base_probs)
        else:
            stat, p_val = 0.0, 1.0

        # Effect size r = Z / sqrt(N)
        # Approximate Z from p-value or statistic
        stat_results.append({
            "comparison": label,
            "set_key": sk,
            "rule": rule,
            "n": len(manip_sids),
            "mean_shift_pct": round(mean_shift, 2),
            "median_shift_pct": round(med_shift, 2),
            "statistic": float(stat),
            "p_value_raw": float(p_val),
        })

    # Holm-Bonferroni correction
    stat_results = sorted(stat_results, key=lambda x: x["p_value_raw"])
    m_tests = len(stat_results)
    for idx, row in enumerate(stat_results):
        alpha_adj = 0.05 / (m_tests - idx)
        row["alpha_adjusted"] = alpha_adj
        row["significant_after_correction"] = bool(row["p_value_raw"] < alpha_adj)
        print(f"  {row['comparison']:<26}: MeanShift={row['mean_shift_pct']:+5.2f}% | MedShift={row['median_shift_pct']:+5.2f}% | p-val={row['p_value_raw']:.2e} | Sig={row['significant_after_correction']}")

    # 12. ABLATION ANALYSIS (SECTION 17)
    print("\n--- SECTION 17: ABLATION ANALYSIS ---")
    # Base on Set H (Native + 1:1 + 9:16 + 0.4586 + 16:9) under Max fusion
    full_h_views = ["view_1_native", "view_2_pad_1:1", "view_4_pad_9:16", "view_5_pad_0.4586", "view_3_pad_16:9"]
    ablation_runs = {
        "Full Set H (5 views)": full_h_views,
        "Without 1:1": [v for v in full_h_views if v != "view_2_pad_1:1"],
        "Without 9:16": [v for v in full_h_views if v != "view_4_pad_9:16"],
        "Without 0.4586": [v for v in full_h_views if v != "view_5_pad_0.4586"],
        "Without 16:9": [v for v in full_h_views if v != "view_3_pad_16:9"],
    }

    ablation_summary = {}
    for ab_name, ab_vlist in ablation_runs.items():
        y_true = [sample_meta[sid]["ground_truth"] for sid in sample_meta]
        y_prob = [float(np.max([image_views_data[sid][v] for v in ab_vlist])) for sid in sample_meta]
        y_pred = [1 if p >= 0.50 else 0 for p in y_prob]
        m = calculate_metrics(y_true, y_pred, y_prob)
        ablation_summary[ab_name] = m
        print(f"  {ab_name:<24}: ManipRec={m['manipulation_recall']:5.2f}% | RealRec={m['real_recall']:5.2f}% | FPR={m['fpr']:5.2f}% | F1={m['f1']:5.2f}%")

    # 13. EXPORT MASTER CSV
    csv_file = RESULTS_DIR / "phase21_multiview_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "category", "ground_truth", "ground_truth_label",
            "width", "height", "aspect_ratio", "file_format",
            "p_view_1_native", "p_view_2_pad_1:1", "p_view_3_pad_16:9",
            "p_view_4_pad_9:16", "p_view_5_pad_0.4586", "p_view_6_jpeg_95", "p_view_7_jpeg_75",
            "p_fused_set_e_max", "p_fused_set_g_max", "p_fused_set_h_max",
            "p_fused_set_e_mean", "p_fused_set_h_mean",
            "pred_native", "pred_set_e_max", "pred_set_h_max"
        ]
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sid in sample_meta:
            gt = sample_meta[sid]["ground_truth"]
            p_nat = image_views_data[sid]["view_1_native"]
            p_emax = fused_probabilities[("SET_E_Native+16:9", "max")][sid]
            p_gmax = fused_probabilities[("SET_G_Native+1:1+9:16+16:9", "max")][sid]
            p_hmax = fused_probabilities[("SET_H_Native+1:1+9:16+0.4586+16:9", "max")][sid]
            p_emean = fused_probabilities[("SET_E_Native+16:9", "mean")][sid]
            p_hmean = fused_probabilities[("SET_H_Native+1:1+9:16+0.4586+16:9", "mean")][sid]

            writer.writerow({
                "sample_id": sid,
                "category": sample_meta[sid]["category"],
                "ground_truth": gt,
                "ground_truth_label": "AI_MANIPULATED" if gt == 1 else "ORIGINAL_REAL",
                "width": sample_meta[sid]["width"],
                "height": sample_meta[sid]["height"],
                "aspect_ratio": sample_meta[sid]["aspect_ratio"],
                "file_format": sample_meta[sid]["file_format"],
                "p_view_1_native": round(p_nat * 100, 2),
                "p_view_2_pad_1:1": round(image_views_data[sid]["view_2_pad_1:1"] * 100, 2),
                "p_view_3_pad_16:9": round(image_views_data[sid]["view_3_pad_16:9"] * 100, 2),
                "p_view_4_pad_9:16": round(image_views_data[sid]["view_4_pad_9:16"] * 100, 2),
                "p_view_5_pad_0.4586": round(image_views_data[sid]["view_5_pad_0.4586"] * 100, 2),
                "p_view_6_jpeg_95": round(image_views_data[sid]["view_6_jpeg_95"] * 100, 2),
                "p_view_7_jpeg_75": round(image_views_data[sid]["view_7_jpeg_75"] * 100, 2),
                "p_fused_set_e_max": round(p_emax * 100, 2),
                "p_fused_set_g_max": round(p_gmax * 100, 2),
                "p_fused_set_h_max": round(p_hmax * 100, 2),
                "p_fused_set_e_mean": round(p_emean * 100, 2),
                "p_fused_set_h_mean": round(p_hmean * 100, 2),
                "pred_native": "AI_MANIPULATED" if p_nat >= 0.50 else "ORIGINAL_REAL",
                "pred_set_e_max": "AI_MANIPULATED" if p_emax >= 0.50 else "ORIGINAL_REAL",
                "pred_set_h_max": "AI_MANIPULATED" if p_hmax >= 0.50 else "ORIGINAL_REAL",
            })
    print(f"[OK] Master CSV saved: {csv_file}")

    # 14. EXPORT SUMMARY JSON
    summary_file = RESULTS_DIR / "phase21_summary.json"
    full_summary = {
        "metadata": {
            "total_images": len(sample_meta),
            "real_images": len(real_sids),
            "manipulated_images": len(manip_sids),
            "total_single_views": len(VIEW_MAP),
            "total_view_sets": len(VIEW_SETS),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "single_view_results": single_view_results,
        "fusion_results": fusion_results,
        "fn_recovery": {
            "native_fn_count": len(native_fns),
            "recovered_count": sum(1 for sid in native_fns if any(image_views_data[sid][v] >= 0.50 for v in VIEW_MAP if v != "view_1_native")),
            "view_recovery_counts": view_recovery_counts,
            "recovered_samples": fn_recovery_table
        },
        "fp_analysis": {
            "native_fp_count": len(native_fps),
            "single_view_fps": fp_analysis,
            "fusion_fps": fusion_fp_analysis
        },
        "category_summary": category_summary,
        "compression_recovery": {
            "q95_fn_count": len(q95_fns),
            "q95_recovered_count": q95_rec_by_uncomp,
            "q95_by_view": q95_by_view,
            "q75_fn_count": len(q75_fns),
            "q75_recovered_count": q75_rec_by_uncomp,
            "q75_by_view": q75_by_view
        },
        "suspect_results": {
            "views": suspect_results,
            "fusions": suspect_fusions
        },
        "statistical_tests": stat_results,
        "ablation_summary": ablation_summary
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"[OK] Summary JSON saved: {summary_file}")

    # 15. GENERATE ALL 10 VISUALIZATIONS
    print("\n--- GENERATING PHASE 21 VISUALIZATIONS ---")

    # 1. single_view_recall.png
    fig, ax = plt.subplots(figsize=(10, 5))
    v_labels = ["Native", "1:1 Pad", "16:9 Pad", "9:16 Pad", "0.4586 Pad", "JPEG Q95", "JPEG Q75"]
    v_keys = list(VIEW_MAP.keys())
    m_recs = [single_view_results[k]["manipulation_recall"] for k in v_keys]
    r_recs = [single_view_results[k]["real_recall"] for k in v_keys]

    x = np.arange(len(v_labels))
    w = 0.35
    ax.bar(x - w/2, m_recs, w, label="Manipulation Recall (%)", color="#dc2626")
    ax.bar(x + w/2, r_recs, w, label="Real Recall (%)", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(v_labels, fontsize=10)
    ax.set_ylabel("Metric Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Single-View Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    for i in range(len(x)):
        ax.text(x[i] - w/2, m_recs[i] + 1.5, f"{m_recs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#dc2626")
        ax.text(x[i] + w/2, r_recs[i] + 1.5, f"{r_recs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#2563eb")
    plt.tight_layout()
    p1 = RESULTS_DIR / "single_view_recall.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"[OK] Plot 1 saved: {p1}")

    # 2. fusion_comparison.png
    fig, ax = plt.subplots(figsize=(11, 5.5))
    eval_sets = ["SET_B_Native+1:1", "SET_E_Native+16:9", "SET_F_Native+1:1+9:16", "SET_G_Native+1:1+9:16+16:9", "SET_H_Native+1:1+9:16+0.4586+16:9"]
    set_clean_labels = ["Native+1:1", "Native+16:9", "Native+1:1+9:16", "Native+1:1+9:16+16:9", "Native+All Padded"]
    mean_recs = [fusion_results[s]["mean"]["manipulation_recall"] for s in eval_sets]
    median_recs = [fusion_results[s]["median"]["manipulation_recall"] for s in eval_sets]
    max_recs = [fusion_results[s]["max"]["manipulation_recall"] for s in eval_sets]

    x = np.arange(len(eval_sets))
    w = 0.25
    ax.bar(x - w, mean_recs, w, label="Mean Fusion", color="#3b82f6")
    ax.bar(x, median_recs, w, label="Median Fusion", color="#8b5cf6")
    ax.bar(x + w, max_recs, w, label="Max Fusion", color="#ef4444")
    ax.axhline(single_view_results["view_1_native"]["manipulation_recall"], color="black", linestyle="--", linewidth=1.5, label=f"Native Baseline ({single_view_results['view_1_native']['manipulation_recall']}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(set_clean_labels, fontsize=10)
    ax.set_ylabel("Manipulation Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Fusion Rule Comparison across Multi-View Sets", fontsize=12, fontweight="bold")
    ax.set_ylim(85, 102)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    for i in range(len(x)):
        ax.text(x[i] + w, max_recs[i] + 0.4, f"{max_recs[i]:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#ef4444")
    plt.tight_layout()
    p2 = RESULTS_DIR / "fusion_comparison.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[OK] Plot 2 saved: {p2}")

    # 3. native_vs_multiview.png
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_compare = ["manipulation_recall", "real_recall", "accuracy", "f1", "macro_f1"]
    m_labels = ["Manip Recall", "Real Recall", "Accuracy", "F1 Score", "Macro F1"]
    v_nat = [single_view_results["view_1_native"][m] for m in metrics_to_compare]
    v_fused = [fusion_results["SET_H_Native+1:1+9:16+0.4586+16:9"]["max"][m] for m in metrics_to_compare]

    x = np.arange(len(m_labels))
    w = 0.35
    ax.bar(x - w/2, v_nat, w, label="Native Single-View", color="#64748b")
    ax.bar(x + w/2, v_fused, w, label="Set H Max-Fusion (Multi-View)", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Native Single-View vs Multi-View Max-Fusion", fontsize=12, fontweight="bold")
    ax.set_ylim(80, 103)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    for i in range(len(x)):
        ax.text(x[i] - w/2, v_nat[i] + 0.4, f"{v_nat[i]:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(x[i] + w/2, v_fused[i] + 0.4, f"{v_fused[i]:.2f}%", ha="center", fontsize=9.5, fontweight="bold", color="#059669")
    plt.tight_layout()
    p3 = RESULTS_DIR / "native_vs_multiview.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"[OK] Plot 3 saved: {p3}")

    # 4. fpr_comparison.png
    fig, ax = plt.subplots(figsize=(10, 5))
    fpr_keys = ["view_1_native", "view_2_pad_1:1", "view_3_pad_16:9", "view_4_pad_9:16", "view_5_pad_0.4586"]
    fpr_labels = ["Native", "1:1 Pad", "16:9 Pad", "9:16 Pad", "0.4586 Pad"]
    single_fprs = [single_view_results[k]["fpr"] for k in fpr_keys]

    # Fused FPRs for same sets
    fused_fpr_sets = ["SET_A_Native", "SET_B_Native+1:1", "SET_E_Native+16:9", "SET_C_Native+9:16", "SET_D_Native+0.4586"]
    fused_fprs = [fusion_results[k]["max"]["fpr"] for k in fused_fpr_sets]

    x = np.arange(len(fpr_labels))
    w = 0.35
    ax.bar(x - w/2, single_fprs, w, label="Single-View FPR", color="#f97316")
    ax.bar(x + w/2, fused_fprs, w, label="Max-Fusion FPR", color="#ea580c")
    ax.set_xticks(x)
    ax.set_xticklabels(fpr_labels, fontsize=10)
    ax.set_ylabel("False Positive Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Real-Image False Positive Rate Across Views and Fusions", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 20)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    for i in range(len(x)):
        ax.text(x[i] - w/2, single_fprs[i] + 0.3, f"{single_fprs[i]:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(x[i] + w/2, fused_fprs[i] + 0.3, f"{fused_fprs[i]:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    p4 = RESULTS_DIR / "fpr_comparison.png"
    plt.savefig(p4, dpi=150)
    plt.close()
    print(f"[OK] Plot 4 saved: {p4}")

    # 5. false_negative_recovery.png
    fig, ax = plt.subplots(figsize=(10, 5))
    rec_v_labels = ["1:1 Pad", "16:9 Pad", "9:16 Pad", "0.4586 Pad", "JPEG Q95", "JPEG Q75", "Any Multi-View"]
    rec_counts = [
        view_recovery_counts["view_2_pad_1:1"],
        view_recovery_counts["view_3_pad_16:9"],
        view_recovery_counts["view_4_pad_9:16"],
        view_recovery_counts["view_5_pad_0.4586"],
        view_recovery_counts["view_6_jpeg_95"],
        view_recovery_counts["view_7_jpeg_75"],
        sum(1 for sid in native_fns if any(image_views_data[sid][v] >= 0.50 for v in VIEW_MAP if v != "view_1_native"))
    ]
    ax.bar(rec_v_labels, rec_counts, color=["#94a3b8", "#10b981", "#94a3b8", "#94a3b8", "#94a3b8", "#94a3b8", "#059669"])
    ax.set_ylabel("Number of FNs Recovered (out of 2)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Native False Negative Recovery by Alternative View", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for i, c in enumerate(rec_counts):
        ax.text(i, c + 0.08, f"{c} / 2 ({c/2*100:.0f}%)", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    p5 = RESULTS_DIR / "false_negative_recovery.png"
    plt.savefig(p5, dpi=150)
    plt.close()
    print(f"[OK] Plot 5 saved: {p5}")

    # 6. view_ablation.png
    fig, ax = plt.subplots(figsize=(10, 5))
    ab_labels = list(ablation_summary.keys())
    ab_recs = [ablation_summary[k]["manipulation_recall"] for k in ab_labels]
    ab_f1s = [ablation_summary[k]["f1"] for k in ab_labels]

    x = np.arange(len(ab_labels))
    w = 0.35
    ax.bar(x - w/2, ab_recs, w, label="Manipulation Recall (%)", color="#6366f1")
    ax.bar(x + w/2, ab_f1s, w, label="F1 Score (%)", color="#a855f7")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_labels, fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Leave-One-View-Out Ablation on Set H (Max-Fusion)", fontsize=12, fontweight="bold")
    ax.set_ylim(97, 100.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    for i in range(len(x)):
        ax.text(x[i] - w/2, ab_recs[i] + 0.05, f"{ab_recs[i]:.2f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + w/2, ab_f1s[i] + 0.05, f"{ab_f1s[i]:.2f}%", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p6 = RESULTS_DIR / "view_ablation.png"
    plt.savefig(p6, dpi=150)
    plt.close()
    print(f"[OK] Plot 6 saved: {p6}")

    # 7. category_view_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cat_mat = np.zeros((len(categories), len(v_labels)))
    for i, cat in enumerate(categories):
        for j, vk in enumerate(v_keys):
            cat_mat[i, j] = category_summary[cat]["view_recalls"][vk]

    im = ax.imshow(cat_mat, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(v_labels)))
    ax.set_xticklabels(v_labels, fontsize=10)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in categories], fontsize=10)
    ax.set_title("Category x View: Manipulation Recall (%)", fontsize=12, fontweight="bold")

    for i in range(len(categories)):
        for j in range(len(v_labels)):
            val = cat_mat[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if val < 50 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (%)")
    plt.tight_layout()
    p7 = RESULTS_DIR / "category_view_heatmap.png"
    plt.savefig(p7, dpi=150)
    plt.close()
    print(f"[OK] Plot 7 saved: {p7}")

    # 8. probability_shift.png
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_shifts = [
        ("Set E Max (Native+16:9)", [fused_probabilities[("SET_E_Native+16:9", "max")][sid] - image_views_data[sid]["view_1_native"] for sid in manip_sids]),
        ("Set H Max (All Padded)", [fused_probabilities[("SET_H_Native+1:1+9:16+0.4586+16:9", "max")][sid] - image_views_data[sid]["view_1_native"] for sid in manip_sids]),
        ("Set E Mean (Native+16:9)", [fused_probabilities[("SET_E_Native+16:9", "mean")][sid] - image_views_data[sid]["view_1_native"] for sid in manip_sids]),
        ("Set H Mean (All Padded)", [fused_probabilities[("SET_H_Native+1:1+9:16+0.4586+16:9", "mean")][sid] - image_views_data[sid]["view_1_native"] for sid in manip_sids]),
    ]
    shift_data = [[s * 100 for s in p[1]] for p in plot_shifts]
    shift_lbls = [p[0] for p in plot_shifts]

    bp = ax.boxplot(shift_data, patch_artist=True)
    ax.set_xticks(range(1, len(shift_lbls) + 1))
    ax.set_xticklabels(shift_lbls, fontsize=9.5)
    for box in bp["boxes"]:
        box.set_facecolor("#dbeafe")
        box.set_edgecolor("#1d4ed8")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Delta P(MANIPULATED) (%) vs Native", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Distribution of Probability Shifts under Fusion Rules", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p8 = RESULTS_DIR / "probability_shift.png"
    plt.savefig(p8, dpi=150)
    plt.close()
    print(f"[OK] Plot 8 saved: {p8}")

    # 9. suspect_multiview.png
    fig, ax = plt.subplots(figsize=(10, 5))
    s_view_keys = list(suspect_view_probs.keys())
    s_view_labels = ["Native", "1:1 Pad", "16:9 Pad", "9:16 Pad", "0.4586 Pad", "JPEG Q95", "JPEG Q75"]
    s_probs = [suspect_view_probs[k] * 100 for k in s_view_keys]

    ax.bar(s_view_labels, s_probs, color="#f43f5e", width=0.55)
    ax.axhline(50.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Decision Boundary (50%)")
    ax.set_ylabel("P(AI_MANIPULATED) (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Suspect Image (WhatsApp 587x1280) Across All 7 Views", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    for i, p in enumerate(s_probs):
        ax.text(i, p + 1.8, f"{p:.2f}%", ha="center", fontsize=9.5, fontweight="bold", color="#f43f5e")
    plt.tight_layout()
    p9 = RESULTS_DIR / "suspect_multiview.png"
    plt.savefig(p9, dpi=150)
    plt.close()
    print(f"[OK] Plot 9 saved: {p9}")

    # 10. compression_recovery.png
    fig, ax = plt.subplots(figsize=(10, 5))
    cr_categories = ["JPEG Q95 FNs", "JPEG Q75 FNs"]
    cr_total_fns = [len(q95_fns), len(q75_fns)]
    cr_recovered = [q95_rec_by_uncomp, q75_rec_by_uncomp]
    cr_unrecovered = [cr_total_fns[0] - cr_recovered[0], cr_total_fns[1] - cr_recovered[1]]

    x = np.arange(len(cr_categories))
    w = 0.45
    ax.bar(x, cr_recovered, w, label="Recovered by Uncompressed Alternative Views", color="#10b981")
    ax.bar(x, cr_unrecovered, w, bottom=cr_recovered, label="Unrecovered (Persistent FNs)", color="#f87171")
    ax.set_xticks(x)
    ax.set_xticklabels(cr_categories, fontsize=11)
    ax.set_ylabel("Number of Manipulated Images", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 21: Recovery of Compression-Induced False Negatives", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)

    for i in range(len(x)):
        ax.text(x[i], cr_recovered[i] / 2, f"{cr_recovered[i]} ({cr_recovered[i]/cr_total_fns[i]*100:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        if cr_unrecovered[i] > 0:
            ax.text(x[i], cr_recovered[i] + cr_unrecovered[i]/2, f"{cr_unrecovered[i]}", ha="center", va="center", color="black", fontweight="bold", fontsize=10)
    plt.tight_layout()
    p10 = RESULTS_DIR / "compression_recovery.png"
    plt.savefig(p10, dpi=150)
    plt.close()
    print(f"[OK] Plot 10 saved: {p10}")

    print("\n[COMPLETE] Phase 21 execution finished successfully.")


if __name__ == "__main__":
    main()
