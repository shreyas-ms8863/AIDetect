"""
AIDetect Phase 20: Compression Robustness and Aspect-Ratio x JPEG Interaction Study
===================================================================================
Comprehensive evaluation of the frozen manipulation detector across all 248
test images of manipulation_v1 under systematic JPEG compression and aspect-ratio
reflect-padding variations.

CRITICAL SAFETY:
- Research-only evaluation experiment.
- Frozen manipulation model: models/manipulation_frequency_resnet50_v1.pth
- Expected SHA-256: 8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889
- Zero modifications to production models, calibration, code, or datasets.
"""

import sys
import os
import io
import csv
import json
import time
import platform
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
RESULTS_DIR = BACKEND_DIR / "phase20_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_freq_v1_dataset import AuthoritativeFrequencyTransform

MANIP_CKPT_PATH    = PROJECT_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"
GEN_CKPT_PATH      = PROJECT_DIR / "models" / "frequency_resnet50_v4.pth"
CALIB_PATH         = BACKEND_DIR / "phase12_results" / "strategy_e_calibration.json"
TEST_DIR           = PROJECT_DIR / "manipulation_v1" / "test"
SUSPECT_IMAGE_PATH = Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-10-02 at 1.52.03 PM.jpeg")

EXPECTED_MANIP_SHA256 = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"
EXPECTED_GEN_SHA256   = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_CALIB_SHA256 = "30e1da6e2ca709cc6113d0d6726db01aef8a127e627869f195449da52571b1ec"

# Aspect Ratios for interaction
ASPECT_RATIOS = [
    ("native", None),
    ("1:1", 1.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
    ("0.4586", 0.4586),
    ("2.1805", 2.1805),
]

# JPEG Qualities
NATIVE_JPEG_QUALITIES = [None, 100, 95, 90, 85, 75, 60, 50, 40, 30]
INTERACTION_JPEG_QUALITIES = [None, 95, 90, 75, 50, 30]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def pad_to_ratio(img: Image.Image, target_ratio: float, mode: str = "reflect") -> Image.Image:
    """Pad image to target aspect ratio (W / H) using reflect padding."""
    arr = np.array(img)
    h, w, c = arr.shape
    curr_ratio = w / h
    if abs(curr_ratio - target_ratio) < 1e-4:
        return img.copy()
    if curr_ratio > target_ratio:
        new_h = int(round(w / target_ratio))
        pad_h = new_h - h
        top = pad_h // 2
        bottom = pad_h - top
        left, right = 0, 0
    else:
        new_w = int(round(h * target_ratio))
        pad_w = new_w - w
        left = pad_w // 2
        right = pad_w - left
        top, bottom = 0, 0

    pad_tuple = ((top, bottom), (left, right), (0, 0))
    if top >= h or bottom >= h or left >= w or right >= w:
        pad_mode = "edge"
    else:
        pad_mode = mode
    padded = np.pad(arr, pad_tuple, mode=pad_mode)
    return Image.fromarray(padded)


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode image to JPEG at specified quality in-memory and decode back to RGB."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return decoded.convert("RGB")


def evaluate_batch(
    model: torch.nn.Module,
    xform: AuthoritativeFrequencyTransform,
    images: List[Image.Image],
    device: torch.device,
) -> List[Tuple[float, float, str]]:
    """Batch forward pass through frequency transform and frozen ResNet-50."""
    tensors = [xform(img) for img in images]
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    results = []
    for p in probs:
        p_orig = float(p[0])
        p_manip = float(p[1])
        label = "AI_MANIPULATED" if p_manip > 0.50 else "ORIGINAL_REAL"
        results.append((p_orig, p_manip, label))
    return results


def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob_manip: List[float]) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0

    prec_manip = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_manip = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_manip = (2 * prec_manip * rec_manip) / (prec_manip + rec_manip) if (prec_manip + rec_manip) > 0 else 0.0

    prec_real = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_real = (2 * prec_real * rec_real) / (prec_real + rec_real) if (prec_real + rec_real) > 0 else 0.0

    macro_prec = (prec_manip + prec_real) / 2.0
    macro_rec = (rec_manip + rec_real) / 2.0
    macro_f1 = (f1_manip + f1_real) / 2.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    mean_p = float(np.mean(y_prob_manip))
    median_p = float(np.median(y_prob_manip))
    std_p = float(np.std(y_prob_manip))

    return {
        "total": total,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec_manip * 100, 2),
        "recall": round(rec_manip * 100, 2),
        "f1": round(f1_manip * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_rec * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "manipulation_recall": round(rec_manip * 100, 2),
        "real_recall": round(rec_real * 100, 2),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "mean_p_manip": round(mean_p * 100, 2),
        "median_p_manip": round(median_p * 100, 2),
        "std_p_manip": round(std_p * 100, 2),
    }


def main():
    print("=" * 80)
    print("AIDETECT PHASE 20: COMPRESSION ROBUSTNESS & ASPECT-RATIO x JPEG STUDY")
    print("=" * 80)

    # 1. VERIFY CHECKPOINT INTEGRITY
    manip_sha = compute_sha256(MANIP_CKPT_PATH)
    gen_sha   = compute_sha256(GEN_CKPT_PATH)
    calib_sha = compute_sha256(CALIB_PATH)

    print(f"Manipulation Model Checkpoint: {MANIP_CKPT_PATH.name}")
    print(f"  Hash:     {manip_sha}")
    print(f"  Expected: {EXPECTED_MANIP_SHA256}")
    if manip_sha != EXPECTED_MANIP_SHA256:
        print("[FATAL] Checkpoint hash mismatch! STOPPING.")
        sys.exit(1)
    print("  Status:   MATCH [OK]")

    # Save hash record
    hashes_record = {
        "manipulation_checkpoint": {"path": str(MANIP_CKPT_PATH), "sha256": manip_sha, "expected": EXPECTED_MANIP_SHA256, "match": manip_sha == EXPECTED_MANIP_SHA256},
        "generation_checkpoint": {"path": str(GEN_CKPT_PATH), "sha256": gen_sha, "expected": EXPECTED_GEN_SHA256, "match": gen_sha == EXPECTED_GEN_SHA256},
        "calibration_file": {"path": str(CALIB_PATH), "sha256": calib_sha, "expected": EXPECTED_CALIB_SHA256, "match": calib_sha == EXPECTED_CALIB_SHA256}
    }
    with open(RESULTS_DIR / "phase20_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes_record, f, indent=2)

    # Save environment record
    env_record = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "phase20_environment.json", "w", encoding="utf-8") as f:
        json.dump(env_record, f, indent=2)

    # 2. LOAD FROZEN MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading frozen manipulation model on {device}...")
    model = ManipulationFrequencyResNet50V1(dropout_p=0.3)
    ckpt = torch.load(MANIP_CKPT_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device).eval()
    print("[OK] Model loaded with strict state_dict matching.")

    xform = AuthoritativeFrequencyTransform()

    # 3. GATHER DATASET
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
            "path": p
        })

    for d in manip_subdirs:
        m_files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
        for p in m_files:
            dataset_items.append({
                "sample_id": p.name,
                "category": d.name,
                "ground_truth": 1,
                "path": p
            })

    total_images = len(dataset_items)
    n_real = sum(1 for x in dataset_items if x["ground_truth"] == 0)
    n_manip = sum(1 for x in dataset_items if x["ground_truth"] == 1)
    print(f"\nDataset loaded: {total_images} total images ({n_real} Real, {n_manip} AI-Manipulated)")

    # Preload all raw images in memory once
    print("Preloading all 248 test images into memory...")
    t0_load = time.time()
    preloaded_images = []
    for item in dataset_items:
        with Image.open(item["path"]) as raw:
            img = raw.convert("RGB")
            preloaded_images.append(img.copy())
    print(f"Preloaded in {time.time() - t0_load:.2f}s.")

    # 4. DEFINE CONDITIONS
    # Part 2: Native Aspect Ratio with qualities: [None, 100, 95, 90, 85, 75, 60, 50, 40, 30] (10 conditions)
    # Part 3: Other 5 aspect ratios with qualities: [None, 95, 90, 75, 50, 30] (5 * 6 = 30 conditions)
    # Total = 40 conditions!
    all_conditions = []
    # Native
    for q in NATIVE_JPEG_QUALITIES:
        cond_id = f"native_jpeg_{q}" if q is not None else "native_baseline"
        all_conditions.append({
            "condition_id": cond_id,
            "aspect_ratio_name": "native",
            "aspect_ratio_val": None,
            "jpeg_quality": q,
            "is_part2": True,
            "is_part3": (q in INTERACTION_JPEG_QUALITIES),
        })

    # Padded ratios
    for r_name, r_val in ASPECT_RATIOS:
        if r_name == "native":
            continue
        for q in INTERACTION_JPEG_QUALITIES:
            cond_id = f"pad_{r_name}_jpeg_{q}" if q is not None else f"pad_{r_name}_uncompressed"
            all_conditions.append({
                "condition_id": cond_id,
                "aspect_ratio_name": r_name,
                "aspect_ratio_val": r_val,
                "jpeg_quality": q,
                "is_part2": False,
                "is_part3": True,
            })

    print(f"Total conditions defined: {len(all_conditions)} ({len(all_conditions) * total_images} inferences)")

    # 5. EXECUTE EVALUATION
    per_image_results = []
    baseline_p_manip_map = {}
    batch_size = 32
    raw_cache_file = RESULTS_DIR / "phase20_raw_eval_cache.json"
    skip_eval = raw_cache_file.exists()
    total_eval_time = 754.61

    if skip_eval:
        print(f"\n[CACHE HIT] Found raw evaluations cache at {raw_cache_file}. Loading...")
        with open(raw_cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            per_image_results = cached_data["per_image_results"]
            baseline_p_manip_map = cached_data["baseline_p_manip_map"]
        print(f"[OK] Loaded {len(per_image_results)} evaluation records from cache.")

    start_time = time.time()

    print("\n--- RUNNING INFERENCE ACROSS ALL CONDITIONS ---")
    for cond_idx, cinfo in enumerate(all_conditions):
    for cond_idx, cinfo in enumerate([] if skip_eval else all_conditions):
        cond_id = cinfo["condition_id"]
        r_name = cinfo["aspect_ratio_name"]
        r_val = cinfo["aspect_ratio_val"]
        q = cinfo["jpeg_quality"]

        t0_c = time.time()
        transformed_images = []
        meta_list = []

        for item, img_raw in zip(dataset_items, preloaded_images):
            ow, oh = img_raw.size
            o_ratio = ow / oh

            # Step 1: Aspect-ratio reflect padding
            if r_val is not None:
                img_padded = pad_to_ratio(img_raw, r_val, mode="reflect")
            else:
                img_padded = img_raw

            tw, th = img_padded.size

            # Step 2: JPEG round-trip if applicable
            if q is not None:
                img_final = jpeg_roundtrip(img_padded, quality=q)
            else:
                img_final = img_padded

            transformed_images.append(img_final)
            meta_list.append({
                "item": item,
                "ow": ow, "oh": oh, "o_ratio": o_ratio,
                "tw": tw, "th": th,
            })

        # Batch forward pass
        cond_outputs = []
        for i in range(0, len(transformed_images), batch_size):
            b_imgs = transformed_images[i:i + batch_size]
            b_res = evaluate_batch(model, xform, b_imgs, device)
            cond_outputs.extend(b_res)

        # Store outputs
        for meta, (p_orig, p_manip, pred_label) in zip(meta_list, cond_outputs):
            item = meta["item"]
            sid = item["sample_id"]
            gt = item["ground_truth"]

            if cond_id == "native_baseline":
                baseline_p_manip_map[sid] = p_manip
                delta_m = 0.0
            else:
                delta_m = p_manip - baseline_p_manip_map[sid]

            is_correct = (1 if pred_label == "AI_MANIPULATED" else 0) == gt

            per_image_results.append({
                "sample_id": sid,
                "category": item["category"],
                "ground_truth": gt,
                "ground_truth_label": "AI_MANIPULATED" if gt == 1 else "ORIGINAL_REAL",
                "original_width": meta["ow"],
                "original_height": meta["oh"],
                "original_aspect_ratio": round(meta["o_ratio"], 4),
                "condition_id": cond_id,
                "aspect_ratio_name": r_name,
                "aspect_ratio_val": r_val,
                "jpeg_quality": q if q is not None else "None",
                "transformed_width": meta["tw"],
                "transformed_height": meta["th"],
                "P_original": round(p_orig * 100, 2),
                "P_manipulated": round(p_manip * 100, 2),
                "P_manipulated_raw": p_manip,
                "prediction": pred_label,
                "delta_P_manipulated": round(delta_m * 100, 2),
                "delta_raw": delta_m,
                "correct_prediction": is_correct,
            })

        dt_c = time.time() - t0_c
        print(f"  [{cond_idx+1:2d}/{len(all_conditions):2d}] {cond_id:<32s} completed in {dt_c:.2f}s")

    total_eval_time = time.time() - start_time
    print(f"\nAll evaluations complete in {total_eval_time:.2f}s.")
    if not skip_eval:
        total_eval_time = time.time() - start_time
        print(f"\nAll evaluations complete in {total_eval_time:.2f}s.")

    # Cache raw results immediately
    raw_cache_file = RESULTS_DIR / "phase20_raw_eval_cache.json"
    with open(raw_cache_file, "w", encoding="utf-8") as f:
        json.dump({"per_image_results": per_image_results, "baseline_p_manip_map": baseline_p_manip_map}, f)
    print(f"[OK] Raw evaluations cached to: {raw_cache_file}")
        # Cache raw results immediately
        with open(raw_cache_file, "w", encoding="utf-8") as f:
            json.dump({"per_image_results": per_image_results, "baseline_p_manip_map": baseline_p_manip_map}, f)
        print(f"[OK] Raw evaluations cached to: {raw_cache_file}")

    # 6. COMPUTE PRIMARY METRICS PER CONDITION
    print("\n--- COMPUTING METRICS PER CONDITION ---")
    condition_summaries = {}

    for cinfo in all_conditions:
        cond_id = cinfo["condition_id"]
        c_rows = [r for r in per_image_results if r["condition_id"] == cond_id]

        y_true = [r["ground_truth"] for r in c_rows]
        y_pred = [1 if r["prediction"] == "AI_MANIPULATED" else 0 for r in c_rows]
        y_prob = [r["P_manipulated_raw"] for r in c_rows]

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        # Delta metrics overall
        deltas = [r["delta_raw"] for r in c_rows]
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        std_delta = float(np.std(deltas))

        # Breakdown categories of shift (5 buckets as requested)
        pct_dec_gt10 = sum(1 for d in deltas if d < -0.10) / len(deltas) * 100
        pct_dec_5_10 = sum(1 for d in deltas if -0.10 <= d < -0.05) / len(deltas) * 100
        pct_chg_lt5  = sum(1 for d in deltas if abs(d) <= 0.05) / len(deltas) * 100
        pct_inc_5_10 = sum(1 for d in deltas if 0.05 < d <= 0.10) / len(deltas) * 100
        pct_inc_gt10 = sum(1 for d in deltas if d > 0.10) / len(deltas) * 100

        # Separate for AI_MANIPULATED only
        manip_deltas = [r["delta_raw"] for r in c_rows if r["ground_truth"] == 1]
        manip_mean_d = float(np.mean(manip_deltas))
        manip_median_d = float(np.median(manip_deltas))
        pct_m_dec_gt10 = sum(1 for d in manip_deltas if d < -0.10) / len(manip_deltas) * 100
        pct_m_dec_5_10 = sum(1 for d in manip_deltas if -0.10 <= d < -0.05) / len(manip_deltas) * 100
        pct_m_chg_lt5  = sum(1 for d in manip_deltas if abs(d) <= 0.05) / len(manip_deltas) * 100
        pct_m_inc_5_10 = sum(1 for d in manip_deltas if 0.05 < d <= 0.10) / len(manip_deltas) * 100
        pct_m_inc_gt10 = sum(1 for d in manip_deltas if d > 0.10) / len(manip_deltas) * 100

        # Separate for REAL only
        real_deltas = [r["delta_raw"] for r in c_rows if r["ground_truth"] == 0]
        real_mean_d = float(np.mean(real_deltas))
        real_median_d = float(np.median(real_deltas))

        # Wilcoxon test vs baseline on AI_MANIPULATED
        if cond_id != "native_baseline":
            manip_native_probs = [baseline_p_manip_map[r["sample_id"]] for r in c_rows if r["ground_truth"] == 1]
            manip_variant_probs = [r["P_manipulated_raw"] for r in c_rows if r["ground_truth"] == 1]
            diffs = np.array(manip_variant_probs) - np.array(manip_native_probs)
            if np.all(diffs == 0):
                w_stat, w_p = 0.0, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(manip_variant_probs, manip_native_probs)
                    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = 0.0, 1.0

        condition_summaries[cond_id] = {
            "condition_id": cond_id,
            "aspect_ratio_name": cinfo["aspect_ratio_name"],
            "aspect_ratio_val": cinfo["aspect_ratio_val"],
            "jpeg_quality": cinfo["jpeg_quality"],
            "metrics": metrics,
            "delta_overall": {
                "mean_delta_pct": round(mean_delta * 100, 2),
                "median_delta_pct": round(median_delta * 100, 2),
                "std_delta_pct": round(std_delta * 100, 2),
                "pct_dec_gt10": round(pct_dec_gt10, 2),
                "pct_dec_5_10": round(pct_dec_5_10, 2),
                "pct_chg_lt5": round(pct_chg_lt5, 2),
                "pct_inc_5_10": round(pct_inc_5_10, 2),
                "pct_inc_gt10": round(pct_inc_gt10, 2),
            },
            "delta_manipulated": {
                "mean_delta_pct": round(manip_mean_d * 100, 2),
                "median_delta_pct": round(manip_median_d * 100, 2),
                "pct_dec_gt10": round(pct_m_dec_gt10, 2),
                "pct_dec_5_10": round(pct_m_dec_5_10, 2),
                "pct_chg_lt5": round(pct_m_chg_lt5, 2),
                "pct_inc_5_10": round(pct_m_inc_5_10, 2),
                "pct_inc_gt10": round(pct_m_inc_gt10, 2),
            },
            "delta_real": {
                "mean_delta_pct": round(real_mean_d * 100, 2),
                "median_delta_pct": round(real_median_d * 100, 2),
            },
            "wilcoxon_test": {
                "statistic": w_stat,
                "p_value": w_p,
                "significant_at_bonferroni": bool(w_p < (0.05 / 39)),
            }
        }

    # 7. INTERACTION PAIR TESTING
    print("\n--- STATISTICAL INTERACTION PAIRS ---")
    interaction_pairs = [
        ("pad_9:16_jpeg_95", "pad_9:16_uncompressed", "9:16 + JPEG95 vs 9:16"),
        ("pad_9:16_jpeg_75", "pad_9:16_uncompressed", "9:16 + JPEG75 vs 9:16"),
        ("pad_0.4586_jpeg_95", "pad_0.4586_uncompressed", "0.4586 + JPEG95 vs 0.4586"),
        ("pad_0.4586_jpeg_75", "pad_0.4586_uncompressed", "0.4586 + JPEG75 vs 0.4586"),
    ]
    interaction_test_results = []
    for c_test, c_ref, desc in interaction_pairs:
        p_test = [r["P_manipulated_raw"] for r in per_image_results if r["condition_id"] == c_test and r["ground_truth"] == 1]
        p_ref  = [r["P_manipulated_raw"] for r in per_image_results if r["condition_id"] == c_ref and r["ground_truth"] == 1]
        d = np.array(p_test) - np.array(p_ref)
        w_res = stats.wilcoxon(p_test, p_ref)
        res_row = {
            "pair": desc,
            "condition_test": c_test,
            "condition_reference": c_ref,
            "n": len(p_test),
            "mean_shift_pct": round(float(np.mean(d)) * 100, 2),
            "median_shift_pct": round(float(np.median(d)) * 100, 2),
            "statistic": float(w_res.statistic),
            "p_value": float(w_res.pvalue),
            "significant_at_005": bool(w_res.pvalue < 0.05),
        }
        interaction_test_results.append(res_row)
        print(f"  {desc:<28s}: Mean shift {res_row['mean_shift_pct']:+5.2f}% | p-val: {res_row['p_value']:.2e}")

    # 8. CATEGORY ANALYSIS
    print("\n--- CATEGORY x JPEG QUALITY ANALYSIS ---")
    categories = sorted(list(set(r["category"] for r in per_image_results if r["ground_truth"] == 1)))
    cat_qualities = [None, 95, 90, 75, 50, 30]
    category_summary = {}
    cat_summary = category_summary

    for cat in categories:
        cat_summary[cat] = {}
        for q in cat_qualities:
            cond_id = f"native_jpeg_{q}" if q is not None else "native_baseline"
            rows = [r for r in per_image_results if r["category"] == cat and r["condition_id"] == cond_id]
            n_c = len(rows)
            rec = sum(1 for r in rows if r["prediction"] == "AI_MANIPULATED") / n_c * 100
            mean_p = float(np.mean([r["P_manipulated_raw"] for r in rows])) * 100
            mean_d = float(np.mean([r["delta_raw"] for r in rows])) * 100

            q_label = f"JPEG_{q}" if q is not None else "Native"
            cat_summary[cat][q_label] = {
                "n": n_c,
                "recall": round(rec, 2),
                "mean_p_manip": round(mean_p, 2),
                "mean_delta_prob": round(mean_d, 2),
            }
        print(f"Category: {cat:<24} | Native: {cat_summary[cat]['Native']['recall']:5.2f}% | Q95: {cat_summary[cat]['JPEG_95']['recall']:5.2f}% | Q75: {cat_summary[cat]['JPEG_75']['recall']:5.2f}% | Q30: {cat_summary[cat]['JPEG_30']['recall']:5.2f}%")

    # 9. DIRECT WHATSAPP SUSPECT REPLICATION
    print("\n--- PART 4: DIRECT WHATSAPP SUSPECT REPLICATION ---")
    suspect_results = {}
    if SUSPECT_IMAGE_PATH.exists():
        suspect_sha = compute_sha256(SUSPECT_IMAGE_PATH)
        with Image.open(SUSPECT_IMAGE_PATH) as s_raw:
            s_img = s_raw.convert("RGB")
        sw, sh = s_img.size
        s_ratio = sw / sh

        # Evaluate the 12+ conditions specified in prompt
        # 1. Native
        p_o, p_m, pred = evaluate_batch(model, xform, [s_img], device)[0]
        suspect_results["Native"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}

        # 2-6. JPEG 95, 90, 75, 50, 30
        for q in [95, 90, 75, 50, 30]:
            sq_img = jpeg_roundtrip(s_img, q)
            p_o, p_m, pred = evaluate_batch(model, xform, [sq_img], device)[0]
            suspect_results[f"Native_JPEG_{q}"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}

        # 7-8. Reflect-pad to 9:16 + JPEG 95, 75
        pad_9_16 = pad_to_ratio(s_img, 9.0/16.0, mode="reflect")
        p_o, p_m, pred = evaluate_batch(model, xform, [pad_9_16], device)[0]
        suspect_results["Pad_9:16_Uncompressed"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}
        for q in [95, 75]:
            img_pq = jpeg_roundtrip(pad_9_16, q)
            p_o, p_m, pred = evaluate_batch(model, xform, [img_pq], device)[0]
            suspect_results[f"Pad_9:16_JPEG_{q}"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}

        # 9-10. Reflect-pad to 0.4586 + JPEG 95, 75
        pad_04586 = pad_to_ratio(s_img, 0.4586, mode="reflect")
        p_o, p_m, pred = evaluate_batch(model, xform, [pad_04586], device)[0]
        suspect_results["Pad_0.4586_Uncompressed"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}
        for q in [95, 75]:
            img_pq = jpeg_roundtrip(pad_04586, q)
            p_o, p_m, pred = evaluate_batch(model, xform, [img_pq], device)[0]
            suspect_results[f"Pad_0.4586_JPEG_{q}"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}

        # 11-12. Reflect-pad to 1:1 + JPEG 95, 75
        pad_1_1 = pad_to_ratio(s_img, 1.0, mode="reflect")
        p_o, p_m, pred = evaluate_batch(model, xform, [pad_1_1], device)[0]
        suspect_results["Pad_1:1_Uncompressed"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}
        for q in [95, 75]:
            img_pq = jpeg_roundtrip(pad_1_1, q)
            p_o, p_m, pred = evaluate_batch(model, xform, [img_pq], device)[0]
            suspect_results[f"Pad_1:1_JPEG_{q}"] = {"p_orig": round(p_o*100, 2), "p_manip": round(p_m*100, 2), "pred": pred}

        print(f"Suspect Image: {SUSPECT_IMAGE_PATH.name}")
        print(f"  SHA-256:     {suspect_sha}")
        print(f"  Dimensions:  {sw} x {sh} (ratio: {s_ratio:.4f})")
        for k, v in suspect_results.items():
            print(f"  {k:<26s}: P(Manip) = {v['p_manip']:5.2f}% ({v['pred']})")

    # 10. SAVE CSV
    csv_file = RESULTS_DIR / "phase20_compression_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "category", "ground_truth", "ground_truth_label",
            "original_width", "original_height", "original_aspect_ratio",
            "condition_id", "aspect_ratio_name", "aspect_ratio_val", "jpeg_quality",
            "transformed_width", "transformed_height",
            "P_original", "P_manipulated", "prediction", "delta_P_manipulated",
            "correct_prediction"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_image_results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"\n[OK] Master CSV saved: {csv_file}")

    # 11. SAVE JSON SUMMARY
    summary_file = RESULTS_DIR / "phase20_compression_summary.json"
    full_summary = {
        "metadata": {
            "total_images": total_images,
            "real_images": n_real,
            "manipulated_images": n_manip,
            "total_conditions": len(all_conditions),
            "total_inferences": len(all_conditions) * total_images,
            "execution_time_seconds": round(total_eval_time, 2),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "condition_summaries": condition_summaries,
        "interaction_tests": interaction_test_results,
        "category_summary": category_summary,
        "suspect_results": suspect_results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"[OK] Summary JSON saved: {summary_file}")

    # 12. GENERATE ALL 8+ VISUALIZATIONS
    print("\n--- GENERATING PHASE 20 VISUALIZATIONS ---")

    # 1. jpeg_quality_vs_recall.png
    fig, ax = plt.subplots(figsize=(10, 5.5))
    native_q_keys = [f"native_jpeg_{q}" if q is not None else "native_baseline" for q in NATIVE_JPEG_QUALITIES]
    q_x = [q if q is not None else 105 for q in NATIVE_JPEG_QUALITIES]  # put native as 105
    q_labels = ["Native" if q is None else f"Q{q}" for q in NATIVE_JPEG_QUALITIES]

    rec_manip = [condition_summaries[k]["metrics"]["manipulation_recall"] for k in native_q_keys]
    rec_real  = [condition_summaries[k]["metrics"]["real_recall"] for k in native_q_keys]
    acc_all   = [condition_summaries[k]["metrics"]["accuracy"] for k in native_q_keys]

    ax.plot(range(len(q_labels)), rec_manip, marker="o", linewidth=2.2, color="#dc2626", label="Manipulation Recall")
    ax.plot(range(len(q_labels)), rec_real, marker="s", linewidth=2.0, color="#2563eb", label="Real Recall")
    ax.plot(range(len(q_labels)), acc_all, marker="^", linewidth=1.8, color="#16a34a", linestyle="--", label="Overall Accuracy")

    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_xlabel("JPEG Quality Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 20: Performance vs JPEG Compression Quality", fontsize=12, fontweight="bold")
    ax.set_ylim(60, 102)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    p1 = RESULTS_DIR / "jpeg_quality_vs_recall.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"[OK] Plot 1 saved: {p1}")

    # 2. jpeg_quality_vs_probability.png
    fig, ax = plt.subplots(figsize=(10, 5.5))
    p_mean_manip = [condition_summaries[k]["metrics"]["mean_p_manip"] for k in native_q_keys]
    p_median_manip = [condition_summaries[k]["metrics"]["median_p_manip"] for k in native_q_keys]
    p_mean_real = [condition_summaries[k]["delta_real"]["mean_delta_pct"] for k in native_q_keys]

    ax.plot(range(len(q_labels)), p_mean_manip, marker="o", linewidth=2.2, color="#dc2626", label="Mean P(Manip) - Manipulated")
    ax.plot(range(len(q_labels)), p_median_manip, marker="s", linewidth=2.0, color="#b91c1c", linestyle="--", label="Median P(Manip) - Manipulated")
    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_xlabel("JPEG Quality Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 20: Predicted Manipulation Probability vs JPEG Quality", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    p2 = RESULTS_DIR / "jpeg_quality_vs_probability.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[OK] Plot 2 saved: {p2}")

    # 3. aspect_ratio_jpeg_recall_heatmap.png
    # Matrix: Rows = 6 Aspect Ratios, Columns = 6 JPEG conditions [Uncompressed, Q95, Q90, Q75, Q50, Q30]
    ar_order = ["native", "1:1", "4:3", "16:9", "3:4", "9:16", "0.4586", "2.1805"]
    ar_labels = ["Native", "1:1 Pad", "4:3 Pad", "16:9 Pad", "3:4 Pad", "9:16 Pad", "0.4586 Pad", "2.1805 Pad"]
    # We evaluated: native, 1:1, 16:9, 9:16, 0.4586, 2.1805
    ar_eval_order = ["native", "1:1", "16:9", "9:16", "0.4586", "2.1805"]
    ar_eval_labels = ["Native Ratio", "1:1 Pad", "16:9 Pad", "9:16 Pad", "0.4586 Pad", "2.1805 Pad"]
    q_eval_cols = [None, 95, 90, 75, 50, 30]
    q_col_labels = ["Uncompressed", "Q=95", "Q=90", "Q=75", "Q=50", "Q=30"]

    recall_matrix = np.zeros((len(ar_eval_order), len(q_eval_cols)))
    prob_shift_matrix = np.zeros((len(ar_eval_order), len(q_eval_cols)))

    for i, ar_name in enumerate(ar_eval_order):
        for j, q in enumerate(q_eval_cols):
            if ar_name == "native":
                c_key = f"native_jpeg_{q}" if q is not None else "native_baseline"
            else:
                c_key = f"pad_{ar_name}_jpeg_{q}" if q is not None else f"pad_{ar_name}_uncompressed"
            
            c_info = condition_summaries[c_key]
            recall_matrix[i, j] = c_info["metrics"]["manipulation_recall"]
            prob_shift_matrix[i, j] = c_info["delta_manipulated"]["mean_delta_pct"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(recall_matrix, cmap="YlOrRd_r", vmin=85, vmax=100)
    ax.set_xticks(range(len(q_col_labels)))
    ax.set_xticklabels(q_col_labels, fontsize=10)
    ax.set_yticks(range(len(ar_eval_labels)))
    ax.set_yticklabels(ar_eval_labels, fontsize=10)
    ax.set_title("Aspect Ratio x JPEG Quality: Manipulation Recall (%)", fontsize=12, fontweight="bold")

    for i in range(len(ar_eval_order)):
        for j in range(len(q_col_labels)):
            val = recall_matrix[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if val < 93 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Manipulation Recall (%)")
    plt.tight_layout()
    p3 = RESULTS_DIR / "aspect_ratio_jpeg_recall_heatmap.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"[OK] Plot 3 saved: {p3}")

    # 4. aspect_ratio_jpeg_probability_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = max(abs(prob_shift_matrix.min()), abs(prob_shift_matrix.max()), 10)
    im = ax.imshow(prob_shift_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(q_col_labels)))
    ax.set_xticklabels(q_col_labels, fontsize=10)
    ax.set_yticks(range(len(ar_eval_labels)))
    ax.set_yticklabels(ar_eval_labels, fontsize=10)
    ax.set_title("Aspect Ratio x JPEG Quality: Mean Delta P(Manipulated) (%)", fontsize=12, fontweight="bold")

    for i in range(len(ar_eval_order)):
        for j in range(len(q_col_labels)):
            val = prob_shift_matrix[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if abs(val) > vmax*0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Delta P(Manipulated) % vs Native")
    plt.tight_layout()
    p4 = RESULTS_DIR / "aspect_ratio_jpeg_probability_heatmap.png"
    plt.savefig(p4, dpi=150)
    plt.close()
    print(f"[OK] Plot 4 saved: {p4}")

    # 5. jpeg_probability_shift_distribution.png
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_q_deltas = [95, 90, 75, 50, 30]
    data_to_plot = []
    for q in plot_q_deltas:
        c_key = f"native_jpeg_{q}"
        deltas = [r["delta_raw"] * 100 for r in per_image_results if r["condition_id"] == c_key and r["ground_truth"] == 1]
        data_to_plot.append(deltas)

    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=[f"Q{q}" for q in plot_q_deltas])
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    ax.set_xticks(range(1, len(plot_q_deltas) + 1))
    ax.set_xticklabels([f"Q{q}" for q in plot_q_deltas], fontsize=10)
    for box in bp["boxes"]:
        box.set_facecolor("#fee2e2")
        box.set_edgecolor("#dc2626")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("JPEG Quality Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Delta P(MANIPULATED) (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 20: Distribution of Probability Shifts under JPEG Compression", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p5 = RESULTS_DIR / "jpeg_probability_shift_distribution.png"
    plt.savefig(p5, dpi=150)
    plt.close()
    print(f"[OK] Plot 5 saved: {p5}")

    # 6. category_jpeg_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_matrix = np.zeros((len(categories), len(q_eval_cols)))
    for i, cat in enumerate(categories):
        for j, q in enumerate(q_eval_cols):
            q_lbl = f"JPEG_{q}" if q is not None else "Native"
            cat_matrix[i, j] = category_summary[cat][q_lbl]["recall"]

    im = ax.imshow(cat_matrix, cmap="YlGnBu", vmin=80, vmax=100)
    ax.set_xticks(range(len(q_col_labels)))
    ax.set_xticklabels(q_col_labels, fontsize=10)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in categories], fontsize=10)
    ax.set_title("Category x JPEG Quality: Manipulation Recall (%)", fontsize=12, fontweight="bold")

    for i in range(len(categories)):
        for j in range(len(q_col_labels)):
            val = cat_matrix[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if val < 90 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (%)")
    plt.tight_layout()
    p6 = RESULTS_DIR / "category_jpeg_heatmap.png"
    plt.savefig(p6, dpi=150)
    plt.close()
    print(f"[OK] Plot 6 saved: {p6}")

    # 7. real_false_positive_vs_jpeg.png
    fig, ax = plt.subplots(figsize=(10, 5))
    fpr_vals = [condition_summaries[k]["metrics"]["fpr"] for k in native_q_keys]
    real_rec_vals = [condition_summaries[k]["metrics"]["real_recall"] for k in native_q_keys]

    ax.plot(range(len(q_labels)), fpr_vals, marker="o", linewidth=2.2, color="#e11d48", label="False Positive Rate (FPR)")
    ax.axhline(condition_summaries["native_baseline"]["metrics"]["fpr"], color="#e11d48", linestyle=":", alpha=0.7, label=f"Baseline FPR ({condition_summaries['native_baseline']['metrics']['fpr']}%)")

    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_xlabel("JPEG Quality Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 20: Genuine Image False Positive Rate vs JPEG Quality", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 30)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    p7 = RESULTS_DIR / "real_false_positive_vs_jpeg.png"
    plt.savefig(p7, dpi=150)
    plt.close()
    print(f"[OK] Plot 7 saved: {p7}")

    # 8. suspect_probability_trajectory.png
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Benchmark distribution quantiles for native JPEG conditions
    q_benchmarks = [95, 90, 75, 50, 30]
    q_bench_labels = ["Native", "Q95", "Q90", "Q75", "Q50", "Q30"]

    # Compute benchmark percentiles for AI_MANIPULATED at these qualities
    p_keys = ["native_baseline"] + [f"native_jpeg_{q}" for q in q_benchmarks]
    pct_5 = [np.percentile([r["P_manipulated_raw"]*100 for r in per_image_results if r["condition_id"] == k and r["ground_truth"] == 1], 5) for k in p_keys]
    pct_25 = [np.percentile([r["P_manipulated_raw"]*100 for r in per_image_results if r["condition_id"] == k and r["ground_truth"] == 1], 25) for k in p_keys]
    pct_50 = [np.percentile([r["P_manipulated_raw"]*100 for r in per_image_results if r["condition_id"] == k and r["ground_truth"] == 1], 50) for k in p_keys]
    pct_75 = [np.percentile([r["P_manipulated_raw"]*100 for r in per_image_results if r["condition_id"] == k and r["ground_truth"] == 1], 75) for k in p_keys]
    pct_95 = [np.percentile([r["P_manipulated_raw"]*100 for r in per_image_results if r["condition_id"] == k and r["ground_truth"] == 1], 95) for k in p_keys]

    x_idx = range(len(q_bench_labels))
    ax.fill_between(x_idx, pct_5, pct_95, color="#dbeafe", alpha=0.6, label="Benchmark 5th-95th Percentile")
    ax.fill_between(x_idx, pct_25, pct_75, color="#93c5fd", alpha=0.6, label="Benchmark 25th-75th Percentile")
    ax.plot(x_idx, pct_50, color="#1d4ed8", linestyle="--", linewidth=1.8, label="Benchmark Median")

    # Suspect trajectory
    if suspect_results:
        s_native_vals = [
            suspect_results["Native"]["p_manip"],
            suspect_results["Native_JPEG_95"]["p_manip"],
            suspect_results["Native_JPEG_90"]["p_manip"],
            suspect_results["Native_JPEG_75"]["p_manip"],
            suspect_results["Native_JPEG_50"]["p_manip"],
            suspect_results["Native_JPEG_30"]["p_manip"],
        ]
        ax.plot(x_idx, s_native_vals, marker="o", color="#dc2626", linewidth=2.5, label="Suspect Image (WhatsApp 587x1280)")
        for ii, val in enumerate(s_native_vals):
            ax.text(ii, val + 2.5, f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#dc2626")

    ax.axhline(50.0, color="#ef4444", linestyle=":", linewidth=1.2, label="Decision Boundary (50%)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(q_bench_labels, fontsize=10)
    ax.set_xlabel("JPEG Condition", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(AI_MANIPULATED) (%)", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 20: Suspect Image Trajectory vs In-Distribution Benchmark", fontsize=12, fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    p8 = RESULTS_DIR / "suspect_probability_trajectory.png"
    plt.savefig(p8, dpi=150)
    plt.close()
    print(f"[OK] Plot 8 saved: {p8}")

    print("\n[COMPLETE] Phase 20 execution finished successfully.")


if __name__ == "__main__":
    main()

