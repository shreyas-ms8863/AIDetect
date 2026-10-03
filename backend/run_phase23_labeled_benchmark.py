"""
AIDetect - Phase 23: Independent Labeled Manipulation Benchmark Validation
===========================================================================
Conducts independent labeled manipulation benchmark validation on held-out
independent pairs (MagicBrush Held-Out, 725 images: 266 Real, 459 Manipulated).
Evaluates Native vs 16:9 Reflect Padding vs 1:1, 9:16, 0.4586, and JPEG views.
Evaluates deterministic multi-view fusions, leave-one-out ablations, FN recovery,
FP analysis, category breakdowns, and paired Wilcoxon statistical testing.
Strict research-only script. Absolute freeze on production models and code.
"""

import os
import sys
import io
import time
import json
import csv
import hashlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from backend.manipulation_freq_v1_dataset import AuthoritativeFrequencyTransform

MODELS_DIR = BASE_DIR / "models"
MANIP_CKPT_PATH = MODELS_DIR / "manipulation_frequency_resnet50_v1.pth"
GEN_CKPT_PATH = MODELS_DIR / "frequency_resnet50_v4.pth"
CALIB_PATH = BASE_DIR / "backend" / "phase12_results" / "strategy_e_calibration.json"

RESULTS_DIR = BASE_DIR / "backend" / "phase23_results"
MANIFEST_PATH = RESULTS_DIR / "phase23_manifest.csv"

EXPECTED_MANIP_SHA256 = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"
EXPECTED_GEN_SHA256   = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_CALIB_SHA256 = "30e1da6e2ca709cc6113d0d6726db01aef8a127e627869f195449da52571b1ec"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def pad_to_ratio(image: Image.Image, target_ratio: float, mode: str = "reflect") -> Image.Image:
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


def evaluate_batch(model: torch.nn.Module, xform: AuthoritativeFrequencyTransform, images: List[Image.Image], device: torch.device) -> List[Tuple[float, float, str]]:
    tensors = [xform(img) for img in images]
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    results = []
    for p in probs:
        p_orig, p_manip = float(p[0]), float(p[1])
        pred_label = "AI_MANIPULATED" if p_manip >= 0.50 else "ORIGINAL_REAL"
        results.append((p_orig, p_manip, pred_label))
    return results


def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, Any]:
    n = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    manip_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (manip_recall + real_recall) / 2.0

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
        "balanced_accuracy": round(balanced_accuracy * 100, 2),
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
    print("AIDETECT PHASE 23: INDEPENDENT LABELED MANIPULATION BENCHMARK VALIDATION")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. VERIFY FROZEN CHECKPOINTS
    manip_sha = compute_file_sha256(MANIP_CKPT_PATH)
    gen_sha   = compute_file_sha256(GEN_CKPT_PATH)
    calib_sha = compute_file_sha256(CALIB_PATH)

    print(f"Manipulation Checkpoint SHA: {manip_sha} (Match: {manip_sha == EXPECTED_MANIP_SHA256})")
    print(f"Generation Checkpoint SHA:   {gen_sha} (Match: {gen_sha == EXPECTED_GEN_SHA256})")
    print(f"Calibration JSON SHA:        {calib_sha} (Match: {calib_sha == EXPECTED_CALIB_SHA256})")

    assert manip_sha == EXPECTED_MANIP_SHA256, "Manipulation checkpoint SHA mismatch"
    assert gen_sha == EXPECTED_GEN_SHA256, "Generation checkpoint SHA mismatch"
    assert calib_sha == EXPECTED_CALIB_SHA256, "Calibration JSON SHA mismatch"

    # 2. LOAD MANIFEST
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    manifest_rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            manifest_rows.append(r)

    total_samples = len(manifest_rows)
    orig_count = sum(1 for r in manifest_rows if int(r["ground_truth"]) == 0)
    manip_count = sum(1 for r in manifest_rows if int(r["ground_truth"]) == 1)

    print(f"\nManifest loaded: {total_samples} total samples")
    print(f"  - Original Real: {orig_count} ({orig_count/total_samples*100:.2f}%)")
    print(f"  - AI Manipulated: {manip_count} ({manip_count/total_samples*100:.2f}%)")

    # 3. SETUP MODEL & TRANSFORM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading ResNet-50 Manipulation Model on device: {device}")
    model = ManipulationFrequencyResNet50V1(dropout_p=0.3).to(device)
    ckpt = torch.load(MANIP_CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    xform = AuthoritativeFrequencyTransform()

    # 4. VIEW PREPARATION & EVALUATION
    # Every sample in manifest has sample_id, ground_truth, original_path, manipulated_path
    # For ground_truth == 0, image to load is original_path
    # For ground_truth == 1, image to load is manipulated_path
    sample_records = []
    batch_size = 32

    views_config = [
        ("view_a_native", lambda img: img),
        ("view_b_16x9", lambda img: pad_to_ratio(img, 16.0 / 9.0, mode="reflect")),
        ("view_c_1x1", lambda img: pad_to_ratio(img, 1.0, mode="reflect")),
        ("view_d_9x16", lambda img: pad_to_ratio(img, 9.0 / 16.0, mode="reflect")),
        ("view_e_0.4586", lambda img: pad_to_ratio(img, 0.4586, mode="reflect")),
        ("view_f_jpeg95", lambda img: jpeg_roundtrip(img, 95)),
        ("view_g_jpeg75", lambda img: jpeg_roundtrip(img, 75)),
    ]

    print("\nEvaluating all 725 samples across all views...")
    t0 = time.time()

    # Preload PIL images
    pil_images = []
    for r in manifest_rows:
        img_path = Path(r["original_path"]) if int(r["ground_truth"]) == 0 else Path(r["manipulated_path"])
        pil_images.append(Image.open(img_path).convert("RGB"))

    view_results = {v_name: [] for v_name, _ in views_config}

    for v_name, v_func in views_config:
        print(f"  Running inference for {v_name}...")
        v_images = [v_func(img) for img in pil_images]
        # Evaluate in batches
        for i in range(0, len(v_images), batch_size):
            batch = v_images[i:i + batch_size]
            res = evaluate_batch(model, xform, batch, device)
            view_results[v_name].extend(res)

    t_eval = time.time() - t0
    print(f"Inference completed in {t_eval:.2f}s ({len(views_config)*total_samples} forward passes)")

    # 5. ASSEMBLE DETAILED RESULTS
    y_true = [int(r["ground_truth"]) for r in manifest_rows]

    eval_results_rows = []
    for idx, r in enumerate(manifest_rows):
        gt = int(r["ground_truth"])
        cat = r["category"]
        gen = r["generator_or_editor_if_available"]
        s_id = r["sample_id"]
        p_id = r["pair_id"]

        p_native = view_results["view_a_native"][idx][1]
        p_16x9   = view_results["view_b_16x9"][idx][1]
        p_1x1    = view_results["view_c_1x1"][idx][1]
        p_9x16   = view_results["view_d_9x16"][idx][1]
        p_04586  = view_results["view_e_0.4586"][idx][1]
        p_jpeg95 = view_results["view_f_jpeg95"][idx][1]
        p_jpeg75 = view_results["view_g_jpeg75"][idx][1]

        # Multi-view fusions
        fused_nat_16x9_max = max(p_native, p_16x9)
        fused_all_pad_max  = max(p_native, p_16x9, p_1x1, p_9x16, p_04586)
        fused_nat_16x9_mean = (p_native + p_16x9) / 2.0
        fused_all_pad_mean  = (p_native + p_16x9 + p_1x1 + p_9x16 + p_04586) / 5.0

        # Leave-one-out ablations from all padded views (A, B, C, D, E)
        abl_no_16x9  = max(p_native, p_1x1, p_9x16, p_04586)
        abl_no_1x1   = max(p_native, p_16x9, p_9x16, p_04586)
        abl_no_9x16  = max(p_native, p_16x9, p_1x1, p_04586)
        abl_no_04586 = max(p_native, p_16x9, p_1x1, p_9x16)

        row = {
            "sample_id": s_id,
            "pair_id": p_id,
            "ground_truth": gt,
            "category": cat,
            "generator": gen,
            "p_native": round(p_native, 6),
            "pred_native": 1 if p_native >= 0.50 else 0,
            "p_16x9": round(p_16x9, 6),
            "pred_16x9": 1 if p_16x9 >= 0.50 else 0,
            "p_1x1": round(p_1x1, 6),
            "pred_1x1": 1 if p_1x1 >= 0.50 else 0,
            "p_9x16": round(p_9x16, 6),
            "pred_9x16": 1 if p_9x16 >= 0.50 else 0,
            "p_0.4586": round(p_04586, 6),
            "pred_0.4586": 1 if p_04586 >= 0.50 else 0,
            "p_jpeg95": round(p_jpeg95, 6),
            "pred_jpeg95": 1 if p_jpeg95 >= 0.50 else 0,
            "p_jpeg75": round(p_jpeg75, 6),
            "pred_jpeg75": 1 if p_jpeg75 >= 0.50 else 0,
            "p_fused_nat_16x9_max": round(fused_nat_16x9_max, 6),
            "pred_fused_nat_16x9_max": 1 if fused_nat_16x9_max >= 0.50 else 0,
            "p_fused_all_pad_max": round(fused_all_pad_max, 6),
            "pred_fused_all_pad_max": 1 if fused_all_pad_max >= 0.50 else 0,
            "p_fused_nat_16x9_mean": round(fused_nat_16x9_mean, 6),
            "pred_fused_nat_16x9_mean": 1 if fused_nat_16x9_mean >= 0.50 else 0,
            "p_fused_all_pad_mean": round(fused_all_pad_mean, 6),
            "pred_fused_all_pad_mean": 1 if fused_all_pad_mean >= 0.50 else 0,
            "p_abl_no_16x9": round(abl_no_16x9, 6),
            "pred_abl_no_16x9": 1 if abl_no_16x9 >= 0.50 else 0,
            "p_abl_no_1x1": round(abl_no_1x1, 6),
            "pred_abl_no_1x1": 1 if abl_no_1x1 >= 0.50 else 0,
            "p_abl_no_9x16": round(abl_no_9x16, 6),
            "pred_abl_no_9x16": 1 if abl_no_9x16 >= 0.50 else 0,
            "p_abl_no_04586": round(abl_no_04586, 6),
            "pred_abl_no_04586": 1 if abl_no_04586 >= 0.50 else 0,
        }
        eval_results_rows.append(row)

    # Save CSV
    out_csv_path = RESULTS_DIR / "phase23_external_labeled_results.csv"
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_results_rows)
    print(f"\nSaved evaluation CSV to: {out_csv_path}")

    # 6. CALCULATE COMPREHENSIVE METRICS
    eval_targets = {
        "View A: Native": ("pred_native", "p_native"),
        "View B: 16:9 Reflect Pad": ("pred_16x9", "p_16x9"),
        "View C: 1:1 Reflect Pad": ("pred_1x1", "p_1x1"),
        "View D: 9:16 Reflect Pad": ("pred_9x16", "p_9x16"),
        "View E: 0.4586 Reflect Pad": ("pred_0.4586", "p_0.4586"),
        "View F: Native JPEG Q95": ("pred_jpeg95", "p_jpeg95"),
        "View G: Native JPEG Q75": ("pred_jpeg75", "p_jpeg75"),
        "Fusion: Native + 16:9 Max": ("pred_fused_nat_16x9_max", "p_fused_nat_16x9_max"),
        "Fusion: All Padded Max": ("pred_fused_all_pad_max", "p_fused_all_pad_max"),
        "Fusion: Native + 16:9 Mean": ("pred_fused_nat_16x9_mean", "p_fused_nat_16x9_mean"),
        "Fusion: All Padded Mean": ("pred_fused_all_pad_mean", "p_fused_all_pad_mean"),
        "Ablation: All Pad Max w/o 16:9": ("pred_abl_no_16x9", "p_abl_no_16x9"),
        "Ablation: All Pad Max w/o 1:1": ("pred_abl_no_1x1", "p_abl_no_1x1"),
        "Ablation: All Pad Max w/o 9:16": ("pred_abl_no_9x16", "p_abl_no_9x16"),
        "Ablation: All Pad Max w/o 0.4586": ("pred_abl_no_04586", "p_abl_no_04586"),
    }

    metrics_summary = {}
    for name, (pred_k, prob_k) in eval_targets.items():
        yp = [r[pred_k] for r in eval_results_rows]
        yprob = [r[prob_k] for r in eval_results_rows]
        m = calculate_metrics(y_true, yp, yprob)
        metrics_summary[name] = m

    # 7. FALSE-NEGATIVE RECOVERY ANALYSIS
    # For every manipulated image where Native = FN (gt=1, pred_native=0)
    native_fn_rows = [r for r in eval_results_rows if r["ground_truth"] == 1 and r["pred_native"] == 0]
    total_native_fn = len(native_fn_rows)

    recovered_by_16x9   = [r for r in native_fn_rows if r["pred_16x9"] == 1]
    recovered_by_1x1    = [r for r in native_fn_rows if r["pred_1x1"] == 1]
    recovered_by_9x16   = [r for r in native_fn_rows if r["pred_9x16"] == 1]
    recovered_by_04586  = [r for r in native_fn_rows if r["pred_0.4586"] == 1]
    recovered_by_max_fuse = [r for r in native_fn_rows if r["pred_fused_all_pad_max"] == 1]

    fn_recovery_summary = {
        "native_fn_count": total_native_fn,
        "recovered_16x9_count": len(recovered_by_16x9),
        "recovered_16x9_pct": round(len(recovered_by_16x9) / total_native_fn * 100, 2) if total_native_fn > 0 else 0.0,
        "recovered_1x1_count": len(recovered_by_1x1),
        "recovered_9x16_count": len(recovered_by_9x16),
        "recovered_04586_count": len(recovered_by_04586),
        "recovered_all_pad_max_count": len(recovered_by_max_fuse),
        "recovered_all_pad_max_pct": round(len(recovered_by_max_fuse) / total_native_fn * 100, 2) if total_native_fn > 0 else 0.0,
        "fn_samples_detail": [
            {
                "sample_id": r["sample_id"],
                "category": r["category"],
                "p_native": r["p_native"],
                "p_16x9": r["p_16x9"],
                "p_1x1": r["p_1x1"],
                "p_9x16": r["p_9x16"],
                "p_0.4586": r["p_0.4586"],
                "delta_16x9": round(r["p_16x9"] - r["p_native"], 6)
            }
            for r in native_fn_rows
        ]
    }

    # 8. FALSE-POSITIVE ANALYSIS
    # For every authentic image (ground_truth == 0)
    real_rows = [r for r in eval_results_rows if r["ground_truth"] == 0]
    total_real = len(real_rows)

    native_fps = [r for r in real_rows if r["pred_native"] == 1]
    p16x9_fps  = [r for r in real_rows if r["pred_16x9"] == 1]
    p1x1_fps   = [r for r in real_rows if r["pred_1x1"] == 1]
    p9x16_fps  = [r for r in real_rows if r["pred_9x16"] == 1]
    p04586_fps = [r for r in real_rows if r["pred_0.4586"] == 1]

    new_fps_16x9 = [r for r in real_rows if r["pred_native"] == 0 and r["pred_16x9"] == 1]
    cleared_fps_16x9 = [r for r in real_rows if r["pred_native"] == 1 and r["pred_16x9"] == 0]

    fp_analysis_summary = {
        "total_authentic": total_real,
        "native_fp_count": len(native_fps),
        "native_fpr_pct": round(len(native_fps) / total_real * 100, 2),
        "16x9_fp_count": len(p16x9_fps),
        "16x9_fpr_pct": round(len(p16x9_fps) / total_real * 100, 2),
        "new_fps_16x9": len(new_fps_16x9),
        "cleared_fps_16x9": len(cleared_fps_16x9),
        "delta_fpr_pct": round((len(p16x9_fps) - len(native_fps)) / total_real * 100, 2),
        "1x1_fp_count": len(p1x1_fps),
        "9x16_fp_count": len(p9x16_fps),
        "04586_fp_count": len(p04586_fps)
    }

    # 9. CATEGORY BREAKDOWN
    categories = sorted(list(set(r["category"] for r in eval_results_rows if r["ground_truth"] == 1)))
    category_summary = {}

    for cat in categories:
        cat_rows = [r for r in eval_results_rows if r["ground_truth"] == 1 and r["category"] == cat]
        n_cat = len(cat_rows)
        tp_nat = sum(1 for r in cat_rows if r["pred_native"] == 1)
        tp_16x9 = sum(1 for r in cat_rows if r["pred_16x9"] == 1)
        tp_fuse = sum(1 for r in cat_rows if r["pred_fused_all_pad_max"] == 1)

        nat_rec = round(tp_nat / n_cat * 100, 2) if n_cat > 0 else 0.0
        rec_16x9 = round(tp_16x9 / n_cat * 100, 2) if n_cat > 0 else 0.0
        rec_fuse = round(tp_fuse / n_cat * 100, 2) if n_cat > 0 else 0.0

        fn_nat = sum(1 for r in cat_rows if r["pred_native"] == 0)
        fn_rec_16x9 = sum(1 for r in cat_rows if r["pred_native"] == 0 and r["pred_16x9"] == 1)

        mean_p_nat = round(float(np.mean([r["p_native"] for r in cat_rows])) * 100, 2)
        mean_p_16x9 = round(float(np.mean([r["p_16x9"] for r in cat_rows])) * 100, 2)

        category_summary[cat] = {
            "n_samples": n_cat,
            "native_recall_pct": nat_rec,
            "16x9_recall_pct": rec_16x9,
            "fused_all_pad_recall_pct": rec_fuse,
            "delta_recall_pct": round(rec_16x9 - nat_rec, 2),
            "native_fn_count": fn_nat,
            "recovered_16x9_fn_count": fn_rec_16x9,
            "mean_p_native_pct": mean_p_nat,
            "mean_p_16x9_pct": mean_p_16x9
        }

    # 10. GENERATOR BREAKDOWN
    generators = sorted(list(set(r["generator"] for r in eval_results_rows)))
    generator_summary = {}
    for gen in generators:
        gen_rows = [r for r in eval_results_rows if r["generator"] == gen]
        n_gen = len(gen_rows)
        gt_gen = [r["ground_truth"] for r in gen_rows]
        mean_nat = round(float(np.mean([r["p_native"] for r in gen_rows])) * 100, 2)
        mean_16x9 = round(float(np.mean([r["p_16x9"] for r in gen_rows])) * 100, 2)
        if gt_gen[0] == 1:
            rec_nat = round(sum(1 for r in gen_rows if r["pred_native"] == 1) / n_gen * 100, 2)
            rec_16x9 = round(sum(1 for r in gen_rows if r["pred_16x9"] == 1) / n_gen * 100, 2)
            generator_summary[gen] = {
                "type": "AI_MANIPULATED",
                "n_samples": n_gen,
                "native_recall_pct": rec_nat,
                "16x9_recall_pct": rec_16x9,
                "delta_recall_pct": round(rec_16x9 - rec_nat, 2),
                "mean_p_native_pct": mean_nat,
                "mean_p_16x9_pct": mean_16x9
            }
        else:
            fpr_nat = round(sum(1 for r in gen_rows if r["pred_native"] == 1) / n_gen * 100, 2)
            fpr_16x9 = round(sum(1 for r in gen_rows if r["pred_16x9"] == 1) / n_gen * 100, 2)
            generator_summary[gen] = {
                "type": "ORIGINAL_REAL",
                "n_samples": n_gen,
                "native_fpr_pct": fpr_nat,
                "16x9_fpr_pct": fpr_16x9,
                "delta_fpr_pct": round(fpr_16x9 - fpr_nat, 2),
                "mean_p_native_pct": mean_nat,
                "mean_p_16x9_pct": mean_16x9
            }

    # 11. PAIRED WILCOXON SIGNED-RANK TEST
    p_native_all = np.array([r["p_native"] for r in eval_results_rows])
    p_16x9_all   = np.array([r["p_16x9"] for r in eval_results_rows])
    diffs_all    = p_16x9_all - p_native_all

    # Paired test on manipulated subset
    manip_indices = [idx for idx, r in enumerate(eval_results_rows) if r["ground_truth"] == 1]
    p_nat_manip   = p_native_all[manip_indices]
    p_16x9_manip  = p_16x9_all[manip_indices]
    diffs_manip   = p_16x9_manip - p_nat_manip

    w_stat_manip, p_val_manip = wilcoxon(p_16x9_manip, p_nat_manip, zero_method="wilcox", correction=False)
    # Effect size r = Z / sqrt(N)
    n_manip = len(diffs_manip)
    mean_shift_manip = float(np.mean(diffs_manip)) * 100
    median_shift_manip = float(np.median(diffs_manip)) * 100

    # Categorical McNemar test on classification decisions (Manipulated set)
    # b: native=0, 16x9=1 (recovered)
    # c: native=1, 16x9=0 (lost)
    b_recovered = sum(1 for idx in manip_indices if eval_results_rows[idx]["pred_native"] == 0 and eval_results_rows[idx]["pred_16x9"] == 1)
    c_lost      = sum(1 for idx in manip_indices if eval_results_rows[idx]["pred_native"] == 1 and eval_results_rows[idx]["pred_16x9"] == 0)
    mcnemar_stat = ((abs(b_recovered - c_lost) - 1)**2) / (b_recovered + c_lost) if (b_recovered + c_lost) > 0 else 0.0

    statistical_test_summary = {
        "n_manipulated": n_manip,
        "mean_shift_manip_pct": round(mean_shift_manip, 4),
        "median_shift_manip_pct": round(median_shift_manip, 4),
        "wilcoxon_w_statistic": float(w_stat_manip),
        "wilcoxon_p_value": float(p_val_manip),
        "mcnemar_recovered_b": b_recovered,
        "mcnemar_lost_c": c_lost,
        "mcnemar_statistic": round(float(mcnemar_stat), 4)
    }

    # 12. PHASE 21 REPLICATION TABLE
    # Phase 21 ground truth was 154 manip, 94 real
    p21_nat_rec = 98.70
    p21_16x9_rec = 99.35
    p21_delta_rec = +0.65
    p21_nat_fpr = 13.83
    p21_16x9_fpr = 13.83
    p21_nat_macro_f1 = 92.42
    p21_16x9_macro_f1 = 92.76
    p21_nat_mean_p = 99.63
    p21_16x9_mean_p = 99.64
    p21_nat_fn = 2
    p21_recovered_fn = 1

    p23_nat_m = metrics_summary["View A: Native"]
    p23_16x9_m = metrics_summary["View B: 16:9 Reflect Pad"]

    replication_table = {
        "metric": [
            "Native manipulation recall (%)",
            "16:9 manipulation recall (%)",
            "Delta recall (%)",
            "Native FPR (%)",
            "16:9 FPR (%)",
            "Native macro F1 (%)",
            "16:9 macro F1 (%)",
            "Native mean P(M) (%)",
            "16:9 mean P(M) (%)",
            "Native FN (count)",
            "16:9 recovered FN (count)"
        ],
        "phase21": [
            p21_nat_rec,
            p21_16x9_rec,
            p21_delta_rec,
            p21_nat_fpr,
            p21_16x9_fpr,
            p21_nat_macro_f1,
            p21_16x9_macro_f1,
            p21_nat_mean_p,
            p21_16x9_mean_p,
            p21_nat_fn,
            p21_recovered_fn
        ],
        "phase23": [
            p23_nat_m["manipulation_recall"],
            p23_16x9_m["manipulation_recall"],
            round(p23_16x9_m["manipulation_recall"] - p23_nat_m["manipulation_recall"], 2),
            p23_nat_m["fpr"],
            p23_16x9_m["fpr"],
            p23_nat_m["macro_f1"],
            p23_16x9_m["macro_f1"],
            p23_nat_m["mean_p_manip"],
            p23_16x9_m["mean_p_manip"],
            p23_nat_m["fn"],
            len(recovered_by_16x9)
        ]
    }

    # 13. HYPOTHESES EVALUATION
    # H1: 16:9 reflect padding improves manipulation recall on an independent labeled dataset.
    delta_rec = p23_16x9_m["manipulation_recall"] - p23_nat_m["manipulation_recall"]
    if delta_rec > 0:
        h1_status = "SUPPORTED"
    elif delta_rec == 0 and p23_nat_m["manipulation_recall"] >= 98.0:
        h1_status = "PARTIALLY SUPPORTED (Parity at ceiling recall)"
    else:
        h1_status = "NOT SUPPORTED"

    # H2: 16:9 recovers native false negatives.
    if len(recovered_by_16x9) > 0:
        h2_status = "SUPPORTED"
    elif total_native_fn == 0:
        h2_status = "INCONCLUSIVE (Zero native false negatives occurred)"
    else:
        h2_status = "NOT SUPPORTED"

    # H3: The improvement does not cause a substantial FPR increase.
    delta_fpr = p23_16x9_m["fpr"] - p23_nat_m["fpr"]
    if delta_fpr <= 1.0:
        h3_status = "SUPPORTED"
    elif delta_fpr <= 3.0:
        h3_status = "PARTIALLY SUPPORTED"
    else:
        h3_status = "NOT SUPPORTED"

    # H4: The effect generalizes across manipulation categories.
    cat_gains = [c["delta_recall_pct"] for c in category_summary.values()]
    if all(g >= 0 for g in cat_gains) and any(g > 0 for g in cat_gains):
        h4_status = "SUPPORTED"
    elif all(g >= 0 for g in cat_gains):
        h4_status = "PARTIALLY SUPPORTED (Non-negative across all categories)"
    else:
        h4_status = "PARTIALLY SUPPORTED"

    # H5: The effect is reproducible across editing/generation sources.
    # Note: Only 1 generation source (DALL-E 2 Inpainting) present in dataset
    h5_status = "INCONCLUSIVE — SINGLE GENERATION SOURCE (Requires cross-architecture generator benchmark)"

    hypotheses_summary = {
        "H1": h1_status,
        "H2": h2_status,
        "H3": h3_status,
        "H4": h4_status,
        "H5": h5_status
    }

    # 14. SAVE SUMMARY JSON
    full_summary = {
        "experiment": "AIDetect Phase 23: Independent Labeled Manipulation Benchmark Validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "MagicBrush Held-Out Independent Benchmark",
        "total_images": total_samples,
        "independent_images": total_samples,
        "original_real_count": orig_count,
        "manipulated_count": manip_count,
        "class_balance_real_pct": round(orig_count / total_samples * 100, 2),
        "class_balance_manip_pct": round(manip_count / total_samples * 100, 2),
        "metrics_summary": metrics_summary,
        "fn_recovery_summary": fn_recovery_summary,
        "fp_analysis_summary": fp_analysis_summary,
        "category_summary": category_summary,
        "generator_summary": generator_summary,
        "statistical_test_summary": statistical_test_summary,
        "replication_table": replication_table,
        "hypotheses_summary": hypotheses_summary
    }

    with open(RESULTS_DIR / "phase23_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved summary JSON to: {RESULTS_DIR / 'phase23_summary.json'}")

    # 15. GENERATE ALL 11 PUBLICATION-GRADE VISUALIZATIONS
    print("\nGenerating publication-quality visualization plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Native vs 16:9 Recall
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    views_p1 = ["Native (View A)", "16:9 Reflect (View B)"]
    rec_p1 = [p23_nat_m["manipulation_recall"], p23_16x9_m["manipulation_recall"]]
    bars = ax.bar(views_p1, rec_p1, color=["#1f77b4", "#2ca02c"], width=0.45, edgecolor="black")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Manipulation Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 23: Native vs 16:9 Manipulation Recall\n(Independent Labeled Benchmark, N=459)", fontsize=11, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "native_vs_16x9_recall.png")
    plt.close()

    # Plot 2: All-View Recall Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    views_all = ["Native", "16:9 Reflect", "1:1 Reflect", "9:16 Reflect", "0.4586 Reflect", "JPEG Q95", "JPEG Q75"]
    keys_all  = ["View A: Native", "View B: 16:9 Reflect Pad", "View C: 1:1 Reflect Pad", "View D: 9:16 Reflect Pad", "View E: 0.4586 Reflect Pad", "View F: Native JPEG Q95", "View G: Native JPEG Q75"]
    recs_all  = [metrics_summary[k]["manipulation_recall"] for k in keys_all]
    colors_all = ["#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#e377c2", "#ff7f0e", "#d62728"]
    bars = ax.bar(views_all, recs_all, color=colors_all, width=0.55, edgecolor="black")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Manipulation Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 23: Multi-View Manipulation Recall Comparison", fontsize=12, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_view_recall.png")
    plt.close()

    # Plot 3: Native vs 16:9 FPR
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    fpr_p3 = [p23_nat_m["fpr"], p23_16x9_m["fpr"]]
    bars = ax.bar(views_p1, fpr_p3, color=["#1f77b4", "#2ca02c"], width=0.45, edgecolor="black")
    ax.set_ylim(0, max(20, max(fpr_p3) * 1.4))
    ax.set_ylabel("False Positive Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 23: Native vs 16:9 False Positive Rate (FPR)\n(Authentic Imagery, N=266)", fontsize=11, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.4, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "native_vs_16x9_fpr.png")
    plt.close()

    # Plot 4: Macro F1 Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    views_f1_keys = ["View A: Native", "View B: 16:9 Reflect Pad", "Fusion: Native + 16:9 Max", "Fusion: All Padded Max", "Fusion: All Padded Mean"]
    views_f1_labels = ["Native", "16:9 Reflect", "Nat + 16:9 Max", "All Padded Max", "All Padded Mean"]
    f1_vals = [metrics_summary[k]["macro_f1"] for k in views_f1_keys]
    bars = ax.bar(views_f1_labels, f1_vals, color=["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#17becf"], width=0.5, edgecolor="black")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Macro F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 23: Macro F1 Score across Views and Fusions", fontsize=11, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "macro_f1_comparison.png")
    plt.close()

    # Plot 5: False-Negative Recovery
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    rec_views = ["Total Native FN", "Recovered by 16:9", "Recovered by 1:1", "Recovered by 9:16", "Recovered by 0.4586", "Recovered by All-Pad Max"]
    rec_counts = [
        total_native_fn,
        len(recovered_by_16x9),
        len(recovered_by_1x1),
        len(recovered_by_9x16),
        len(recovered_by_04586),
        len(recovered_by_max_fuse)
    ]
    colors_fn = ["#d62728", "#2ca02c", "#17becf", "#9467bd", "#e377c2", "#33a02c"]
    bars = ax.bar(rec_views, rec_counts, color=colors_fn, width=0.55, edgecolor="black")
    ax.set_ylim(0, max(5, total_native_fn * 1.35))
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_title(f"Phase 23: False-Negative Recovery by View\n(Native FN Base = {total_native_fn})", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{int(yval)}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "false_negative_recovery.png")
    plt.close()

    # Plot 6: Probability Shift Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(diffs_manip * 100, bins=30, color="#1f77b4", edgecolor="black", alpha=0.75, label="Manipulated Samples")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Shift")
    ax.axvline(mean_shift_manip, color="green", linestyle="-", linewidth=2.0, label=f"Mean Shift ({mean_shift_manip:+.2f}%)")
    ax.axvline(median_shift_manip, color="purple", linestyle=":", linewidth=2.0, label=f"Median Shift ({median_shift_manip:+.2f}%)")
    ax.set_xlabel("Probability Shift: P(MANIP | 16:9) - P(MANIP | Native) (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
    ax.set_title(f"Phase 23: Paired Probability Shift Distribution (N={n_manip})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "probability_shift.png")
    plt.close()

    # Plot 7: Category Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cat_names = list(category_summary.keys())
    x = np.arange(len(cat_names))
    width = 0.35
    rec_nat_cat = [category_summary[c]["native_recall_pct"] for c in cat_names]
    rec_16x9_cat = [category_summary[c]["16x9_recall_pct"] for c in cat_names]

    rects1 = ax.bar(x - width/2, rec_nat_cat, width, label="Native Recall", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, rec_16x9_cat, width, label="16:9 Recall", color="#2ca02c", edgecolor="black")
    ax.set_ylabel("Recall (%)", fontsize=10, fontweight="bold")
    ax.set_title("Phase 23: Manipulation Recall by Category", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(N={category_summary[c]['n_samples']})" for c in cat_names], fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "category_comparison.png")
    plt.close()

    # Plot 8: Generator Comparison
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    gen_names = list(generator_summary.keys())
    means_nat = [generator_summary[g]["mean_p_native_pct"] for g in gen_names]
    means_16x9 = [generator_summary[g]["mean_p_16x9_pct"] for g in gen_names]
    x_gen = np.arange(len(gen_names))
    w_gen = 0.35
    b1 = ax.bar(x_gen - w_gen/2, means_nat, w_gen, label="Native Mean P(M)", color="#1f77b4", edgecolor="black")
    b2 = ax.bar(x_gen + w_gen/2, means_16x9, w_gen, label="16:9 Mean P(M)", color="#2ca02c", edgecolor="black")
    ax.set_ylabel("Mean P(MANIPULATED) (%)", fontsize=10, fontweight="bold")
    ax.set_title("Phase 23: Generator-Level Mean Probability Shift", fontsize=11, fontweight="bold")
    ax.set_xticks(x_gen)
    ax.set_xticklabels([f"{g}\n({generator_summary[g]['type']})" for g in gen_names], fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=9)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height()+1.0, f"{bar.get_height():.1f}%", ha="center", fontsize=8.5)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height()+1.0, f"{bar.get_height():.1f}%", ha="center", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "generator_comparison.png")
    plt.close()

    # Plot 9: Phase 21 vs Phase 23 Replication
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    metrics_rep_labels = ["Native Recall", "16:9 Recall", "Native FPR", "16:9 FPR", "Native Macro F1", "16:9 Macro F1"]
    p21_vals = [p21_nat_rec, p21_16x9_rec, p21_nat_fpr, p21_16x9_fpr, p21_nat_macro_f1, p21_16x9_macro_f1]
    p23_vals = [
        p23_nat_m["manipulation_recall"],
        p23_16x9_m["manipulation_recall"],
        p23_nat_m["fpr"],
        p23_16x9_m["fpr"],
        p23_nat_m["macro_f1"],
        p23_16x9_m["macro_f1"]
    ]
    x_rep = np.arange(len(metrics_rep_labels))
    w_rep = 0.35
    b_p21 = ax.bar(x_rep - w_rep/2, p21_vals, w_rep, label="Phase 21 (Internal Test)", color="#7f7f7f", edgecolor="black")
    b_p23 = ax.bar(x_rep + w_rep/2, p23_vals, w_rep, label="Phase 23 (Held-Out Benchmark)", color="#1f77b4", edgecolor="black")
    ax.set_ylabel("Metric Value (%)", fontsize=10, fontweight="bold")
    ax.set_title("Phase 21 vs Phase 23 Side-by-Side Replication", fontsize=11, fontweight="bold")
    ax.set_xticks(x_rep)
    ax.set_xticklabels(metrics_rep_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase21_vs_phase23.png")
    plt.close()

    # Plot 10: Fusion Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    fusion_keys = ["View A: Native", "View B: 16:9 Reflect Pad", "Fusion: Native + 16:9 Max", "Fusion: All Padded Max", "Fusion: Native + 16:9 Mean", "Fusion: All Padded Mean"]
    fusion_labels = ["Native", "16:9 Reflect", "Nat+16:9 Max", "All-Pad Max", "Nat+16:9 Mean", "All-Pad Mean"]
    f_rec = [metrics_summary[k]["manipulation_recall"] for k in fusion_keys]
    f_fpr = [metrics_summary[k]["fpr"] for k in fusion_keys]
    x_f = np.arange(len(fusion_labels))
    w_f = 0.35
    ax.bar(x_f - w_f/2, f_rec, w_f, label="Manip Recall (%)", color="#2ca02c", edgecolor="black")
    ax.bar(x_f + w_f/2, f_fpr, w_f, label="FPR (%)", color="#d62728", edgecolor="black")
    ax.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_title("Phase 23: Multi-View Evidence Fusion Comparison", fontsize=11, fontweight="bold")
    ax.set_xticks(x_f)
    ax.set_xticklabels(fusion_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fusion_comparison.png")
    plt.close()

    # Plot 11: Ablation Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    abl_keys = [
        "Fusion: All Padded Max",
        "Ablation: All Pad Max w/o 16:9",
        "Ablation: All Pad Max w/o 1:1",
        "Ablation: All Pad Max w/o 9:16",
        "Ablation: All Pad Max w/o 0.4586"
    ]
    abl_labels = ["Full Set (All 5)", "w/o 16:9", "w/o 1:1", "w/o 9:16", "w/o 0.4586"]
    abl_recs = [metrics_summary[k]["manipulation_recall"] for k in abl_keys]
    abl_bars = ax.bar(abl_labels, abl_recs, color=["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#e377c2"], width=0.5, edgecolor="black")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Manipulation Recall (%)", fontsize=10, fontweight="bold")
    ax.set_title("Phase 23: Leave-One-View-Out Fusion Ablation", fontsize=11, fontweight="bold")
    for bar in abl_bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_comparison.png")
    plt.close()

    print("All 11 visualization plots successfully generated and saved to backend/phase23_results/")
    print("\nPhase 23 execution complete.")


if __name__ == "__main__":
    main()
