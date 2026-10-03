"""
backend/run_phase13_frozen_confirmation.py
==========================================
AIDetect Phase 13: Frozen Test Confirmation
Validates whether conflict-aware decision strategies discovered during
Phase 12 transfer to the completely locked, untouched Phase 11 test set.

Strict Protocol:
- One-shot confirmation evaluation
- ZERO threshold tuning, ZERO training, ZERO calibration on test data
- 861 authoritative Phase 11 test images (390 REAL, 317 GEN, 154 MANIP)
- Exact frozen models: frequency_resnet50_v4.pth & manipulation_frequency_resnet50_v1.pth
- Identical probabilities passed to Strategy A, Strategy C, and Strategy E
"""

import sys
import os
import time
import json
import csv
import hashlib
import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RESULTS_DIR = BASE_DIR / "phase13_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASET_V4_TEST = PROJECT_DIR / "dataset_v4" / "test"
MANIP_V1_TEST = PROJECT_DIR / "manipulation_v1" / "test"
STRATEGY_E_CALIB_PATH = BASE_DIR / "phase12_results" / "strategy_e_calibration.json"

# Models
GEN_CKPT_PATH = PROJECT_DIR / "models" / "frequency_resnet50_v4.pth"
MANIP_CKPT_PATH = PROJECT_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"

EXPECTED_GEN_HASH = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_MANIP_HASH = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import system components
from backend.forensic_inference import ForensicInferencePipeline
from backend.forensic_decision_engine import ForensicDecisionEngine
from backend.phase12_decision_strategies import (
    strategy_a_current_rule,
    strategy_c_manipulation_priority_in_conflict,
    LogisticMetaClassifierStrategy,
)

def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# DATASET COLLECTION
# ---------------------------------------------------------------------------

def collect_test_samples() -> List[Dict[str, Any]]:
    samples = []
    
    # 1. dataset_v4/test/real/ -> REAL_ORIGINAL
    g1_files = sorted(list((DATASET_V4_TEST / "real").rglob("*.*")))
    for p in g1_files:
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            sub = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "dataset_v4/test/real",
                "source_category": sub,
                "ground_truth": "REAL_ORIGINAL",
                "image_id": p.stem,
            })

    # 2. dataset_v4/test/ai/ -> AI_GENERATED
    g2_files = sorted(list((DATASET_V4_TEST / "ai").rglob("*.*")))
    for p in g2_files:
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            sub = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "dataset_v4/test/ai",
                "source_category": sub,
                "ground_truth": "AI_GENERATED",
                "image_id": p.stem,
            })

    # 3. manipulation_v1/test/original/ -> REAL_ORIGINAL
    g3_files = sorted(list((MANIP_V1_TEST / "original").rglob("*.png")))
    for p in g3_files:
        if p.is_file():
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "manipulation_v1/test/original",
                "source_category": "unaltered_source",
                "ground_truth": "REAL_ORIGINAL",
                "image_id": p.stem.replace("orig_", ""),
            })

    # 4. manipulation_v1/test/manipulated/ -> AI_MANIPULATED
    g4_files = sorted(list((MANIP_V1_TEST / "manipulated").rglob("*.png")))
    for p in g4_files:
        if p.is_file():
            cat = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "manipulation_v1/test/manipulated",
                "source_category": cat,
                "ground_truth": "AI_MANIPULATED",
                "image_id": p.stem.replace("manip_", ""),
            })

    return samples

# ---------------------------------------------------------------------------
# METRICS & CONFUSION MATRIX HELPER
# ---------------------------------------------------------------------------

def compute_comprehensive_metrics(predictions: List[str], ground_truths: List[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    classes = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
    cols = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED", "UNCERTAIN"]
    n_total = len(ground_truths)

    # 3x4 Confusion Matrix
    cm_3x4 = {gt_cls: {pred_col: 0 for pred_col in cols} for gt_cls in classes}
    for p, g in zip(predictions, ground_truths):
        if g in cm_3x4:
            if p in cols:
                cm_3x4[g][p] += 1
            else:
                cm_3x4[g]["UNCERTAIN"] += 1

    correct = sum(cm_3x4[c][c] for c in classes)
    accuracy = correct / n_total if n_total > 0 else 0.0

    uncertain_count = sum(1 for p in predictions if p == "UNCERTAIN")
    uncertain_rate = uncertain_count / n_total if n_total > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = cm_3x4[cls][cls]
        fp = sum(cm_3x4[other_gt][cls] for other_gt in classes if other_gt != cls)
        fn = sum(cm_3x4[cls][col] for col in cols if col != cls)
        unc = cm_3x4[cls]["UNCERTAIN"]
        n_cls = sum(cm_3x4[cls].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            "n_samples": n_cls,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "uncertain_count": unc,
            "uncertain_rate": round(unc / n_cls, 4) if n_cls > 0 else 0.0,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = sum(precisions) / len(classes)
    macro_rec = sum(recalls) / len(classes)
    macro_f1 = sum(f1s) / len(classes)

    metrics_dict = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "uncertain_count": uncertain_count,
        "per_class": per_class,
    }
    return metrics_dict, cm_3x4

def save_confusion_matrix_artifacts(cm_3x4: Dict[str, Dict[str, int]], csv_path: Path, png_path: Path, title: str):
    rows = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
    cols = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED", "UNCERTAIN"]
    
    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth"] + cols + ["total"])
        for r in rows:
            tot = sum(cm_3x4[r][c] for c in cols)
            writer.writerow([r] + [cm_3x4[r][c] for c in cols] + [tot])

    # Plot PNG
    matrix = np.array([[cm_3x4[r][c] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, fontsize=9, rotation=15)
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlabel("Predicted State", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    for i in range(len(rows)):
        for j in range(len(cols)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=11)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------------

def run_phase13_confirmation():
    print("=" * 75)
    print("AIDETECT PHASE 13: FROZEN TEST CONFIRMATION EXPERIMENT")
    print("=" * 75)
    t_start = time.time()
    timestamp = datetime.datetime.now().isoformat()
    print(f"Timestamp   : {timestamp}")
    print(f"Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model   : {torch.cuda.get_device_name(0)}")
    print(f"Results Dir : {RESULTS_DIR}")
    print()

    # 1. Model Checkpoint Verification (Pre)
    print("-" * 65)
    print("1. MODEL CHECKPOINT VERIFICATION (PRE-EVALUATION)")
    print("-" * 65)
    gen_hash_pre = compute_file_sha256(GEN_CKPT_PATH)
    manip_hash_pre = compute_file_sha256(MANIP_CKPT_PATH)

    print(f"Generation Model   : {GEN_CKPT_PATH.name}")
    print(f"  Hash             : {gen_hash_pre}")
    print(f"  Expected         : {EXPECTED_GEN_HASH}")
    gen_match_pre = (gen_hash_pre == EXPECTED_GEN_HASH)
    print(f"  Status           : {'MATCH' if gen_match_pre else 'MISMATCH'}")
    assert gen_match_pre, "STOP: Generation model hash mismatch!"

    print(f"Manipulation Model : {MANIP_CKPT_PATH.name}")
    print(f"  Hash             : {manip_hash_pre}")
    print(f"  Expected         : {EXPECTED_MANIP_HASH}")
    manip_match_pre = (manip_hash_pre == EXPECTED_MANIP_HASH)
    print(f"  Status           : {'MATCH' if manip_match_pre else 'MISMATCH'}")
    assert manip_match_pre, "STOP: Manipulation model hash mismatch!"
    print()

    # 2. Verify Frozen Calibration File for Strategy E
    print("-" * 65)
    print("2. VERIFYING FROZEN STRATEGY E CALIBRATION PARAMETERS")
    print("-" * 65)
    assert STRATEGY_E_CALIB_PATH.exists(), f"STOP: Strategy E calibration file missing: {STRATEGY_E_CALIB_PATH}"
    with open(STRATEGY_E_CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    print(f"Loaded Strategy E calibration:")
    print(f"  Source          : {calib_data.get('source')}")
    print(f"  Created At      : {calib_data.get('created_at')}")
    print(f"  Average Weights : {calib_data.get('average_weights')}")
    print(f"  Average Biases  : {calib_data.get('average_biases')}")
    
    strategy_e_clf = LogisticMetaClassifierStrategy()
    strategy_e_clf.load_calibration_file(str(STRATEGY_E_CALIB_PATH))
    print("  CONFIRMED: Strategy E calibration loaded with zero retraining.")
    print()

    # 3. Collect 861 Locked Test Images
    print("-" * 65)
    print("3. LOCKED TEST SET COLLECTION & AUDIT")
    print("-" * 65)
    test_samples = collect_test_samples()
    print(f"Total Locked Test Samples: {len(test_samples)}")
    class_counts = Counter(s["ground_truth"] for s in test_samples)
    print(f"  - REAL_ORIGINAL : {class_counts['REAL_ORIGINAL']}")
    print(f"  - AI_GENERATED  : {class_counts['AI_GENERATED']}")
    print(f"  - AI_MANIPULATED: {class_counts['AI_MANIPULATED']}")

    assert len(test_samples) == 861, f"Expected 861 test images, got {len(test_samples)}"
    assert class_counts["REAL_ORIGINAL"] == 390
    assert class_counts["AI_GENERATED"] == 317
    assert class_counts["AI_MANIPULATED"] == 154
    print("  CONFIRMED: Exact authoritative Phase 11 test partition.")
    print()

    # 4. Run Model Inference ONCE
    print("-" * 65)
    print("4. RUNNING FROZEN INFERENCE ONCE ON 861 TEST IMAGES")
    print("-" * 65)
    pipeline = ForensicInferencePipeline(device=DEVICE)
    raw_eval_records = []
    
    t_inf_start = time.time()
    for idx, s in enumerate(test_samples):
        res = pipeline.predict(s["filepath"])
        gen = res["generation"]
        manip = res["manipulation"]
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(test_samples):
            elapsed = time.time() - t_inf_start
            print(f"  Processed {idx + 1} / {len(test_samples)} images ({elapsed:.1f}s)...")

        raw_eval_records.append({
            "sample": s,
            "p_gen_real": gen["probability_real"],
            "p_gen_ai": gen["probability_ai_generated"],
            "gen_label": gen["label"],
            "p_manip_orig": manip["probability_original"],
            "p_manip_ai": manip["probability_ai_manipulated"],
            "manip_label": manip["label"],
        })
    print()

    # 5. Evaluate the Three Strategies on IDENTICAL Probabilities
    print("-" * 65)
    print("5. ADJUDICATING PREDICTIONS ACROSS STRATEGIES A, C, AND E")
    print("-" * 65)
    
    pred_csv_rows = []
    strat_a_preds = []
    strat_c_preds = []
    strat_e_preds = []
    ground_truths = [r["sample"]["ground_truth"] for r in raw_eval_records]

    for r in raw_eval_records:
        s = r["sample"]
        p_real = r["p_gen_real"]
        p_gen_ai = r["p_gen_ai"]
        p_orig = r["p_manip_orig"]
        p_manip_ai = r["p_manip_ai"]

        # Strategy A (Current Baseline)
        st_a, conf_a, case_a = strategy_a_current_rule(p_real, p_gen_ai, p_orig, p_manip_ai)
        strat_a_preds.append(st_a)

        # Strategy C (Manip Priority in Conflict)
        st_c, conf_c, case_c = strategy_c_manipulation_priority_in_conflict(p_real, p_gen_ai, p_orig, p_manip_ai, th_gen=0.55, th_manip=0.55)
        strat_c_preds.append(st_c)

        # Strategy E (2D Logistic Regression Fusion)
        st_e, conf_e, case_e = strategy_e_clf.predict_one(p_gen_ai, p_manip_ai)
        strat_e_preds.append(st_e)

        pred_csv_rows.append({
            "image_path": s["rel_path"],
            "source_dataset": s["source_dataset"],
            "source_category": s["source_category"],
            "ground_truth": s["ground_truth"],
            "p_gen_real": p_real,
            "p_gen_ai": p_gen_ai,
            "p_manip_original": p_orig,
            "p_manip_manipulated": p_manip_ai,
            "baseline_state": st_a,
            "strategy_c_state": st_c,
            "strategy_e_state": st_e,
            "baseline_confidence": conf_a,
            "strategy_c_confidence": conf_c,
            "strategy_e_confidence": conf_e,
            "image_id": s["image_id"],
        })

    # Save frozen_test_predictions.csv
    pred_csv_path = RESULTS_DIR / "frozen_test_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_path", "source_dataset", "source_category", "ground_truth",
            "p_gen_real", "p_gen_ai", "p_manip_original", "p_manip_manipulated",
            "baseline_state", "strategy_c_state", "strategy_e_state",
            "baseline_confidence", "strategy_c_confidence", "strategy_e_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pred_csv_rows)
    print(f"Saved: {pred_csv_path.name} ({len(pred_csv_rows)} rows)")
    print()

    # 6. Calculate Metrics & Confusion Matrices
    print("-" * 65)
    print("6. METRICS & CONFUSION MATRICES EVALUATION")
    print("-" * 65)
    metrics_a, cm_a = compute_comprehensive_metrics(strat_a_preds, ground_truths)
    metrics_c, cm_c = compute_comprehensive_metrics(strat_c_preds, ground_truths)
    metrics_e, cm_e = compute_comprehensive_metrics(strat_e_preds, ground_truths)

    # Save artifacts for Strategy A
    save_confusion_matrix_artifacts(cm_a, RESULTS_DIR / "baseline_confusion_matrix.csv", RESULTS_DIR / "baseline_confusion_matrix.png", "Strategy A (Baseline) - Frozen Test Set")
    with open(RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_a, f, indent=2)

    # Save artifacts for Strategy C
    save_confusion_matrix_artifacts(cm_c, RESULTS_DIR / "strategy_c_confusion_matrix.csv", RESULTS_DIR / "strategy_c_confusion_matrix.png", "Strategy C (Manip-Priority) - Frozen Test Set")
    with open(RESULTS_DIR / "strategy_c_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_c, f, indent=2)

    # Save artifacts for Strategy E
    save_confusion_matrix_artifacts(cm_e, RESULTS_DIR / "strategy_e_confusion_matrix.csv", RESULTS_DIR / "strategy_e_confusion_matrix.png", "Strategy E (2D Fusion) - Frozen Test Set")
    with open(RESULTS_DIR / "strategy_e_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_e, f, indent=2)

    print("Strategy Performance Comparison (861 Frozen Test Images):")
    print(f"{'Metric':<25s} | {'Strategy A (Base)':<18s} | {'Strategy C (ManipPri)':<22s} | {'Strategy E (2D Fusion)':<22s}")
    print("-" * 95)
    print(f"{'Overall Accuracy':<25s} | {metrics_a['accuracy']*100:6.2f}%            | {metrics_c['accuracy']*100:6.2f}%                | {metrics_e['accuracy']*100:6.2f}%")
    print(f"{'Macro Precision':<25s} | {metrics_a['macro_precision']:6.4f}             | {metrics_c['macro_precision']:6.4f}              | {metrics_e['macro_precision']:6.4f}")
    print(f"{'Macro Recall':<25s} | {metrics_a['macro_recall']:6.4f}             | {metrics_c['macro_recall']:6.4f}              | {metrics_e['macro_recall']:6.4f}")
    print(f"{'Macro F1 Score':<25s} | {metrics_a['macro_f1']:6.4f}             | {metrics_c['macro_f1']:6.4f}              | {metrics_e['macro_f1']:6.4f}")
    print(f"{'UNCERTAIN Rate':<25s} | {metrics_a['uncertain_rate']*100:6.2f}%            | {metrics_c['uncertain_rate']*100:6.2f}%                | {metrics_e['uncertain_rate']*100:6.2f}%")
    print(f"{'REAL_ORIGINAL Recall':<25s} | {metrics_a['per_class']['REAL_ORIGINAL']['recall']*100:6.2f}%            | {metrics_c['per_class']['REAL_ORIGINAL']['recall']*100:6.2f}%                | {metrics_e['per_class']['REAL_ORIGINAL']['recall']*100:6.2f}%")
    print(f"{'AI_GENERATED Recall':<25s} | {metrics_a['per_class']['AI_GENERATED']['recall']*100:6.2f}%            | {metrics_c['per_class']['AI_GENERATED']['recall']*100:6.2f}%                | {metrics_e['per_class']['AI_GENERATED']['recall']*100:6.2f}%")
    print(f"{'AI_MANIPULATED Recall':<25s} | {metrics_a['per_class']['AI_MANIPULATED']['recall']*100:6.2f}%            | {metrics_c['per_class']['AI_MANIPULATED']['recall']*100:6.2f}%                | {metrics_e['per_class']['AI_MANIPULATED']['recall']*100:6.2f}%")
    print()

    # 7. Manipulation-Specific Analysis (154 Images)
    print("-" * 65)
    print("7. MANIPULATION-SPECIFIC ANALYSIS (154 TEST SAMPLES)")
    print("-" * 65)
    manip_samples = [r for r in pred_csv_rows if r["ground_truth"] == "AI_MANIPULATED"]
    
    for name, st_key in [("Strategy A", "baseline_state"), ("Strategy C", "strategy_c_state"), ("Strategy E", "strategy_e_state")]:
        preds = [r[st_key] for r in manip_samples]
        tp = sum(1 for p in preds if p == "AI_MANIPULATED")
        mis_gen = sum(1 for p in preds if p == "AI_GENERATED")
        fn_real = sum(1 for p in preds if p == "REAL_ORIGINAL")
        unc = sum(1 for p in preds if p == "UNCERTAIN")
        rec = tp / len(manip_samples)
        print(f"{name:12s} -> Correct AI_MANIP: {tp:3d} ({rec*100:5.2f}%) | Misclassified as GEN: {mis_gen:3d} | Called REAL: {fn_real:2d} | UNCERTAIN: {unc:2d}")
    print()

    # 8. Conflict Region Analysis (P_GEN_AI >= 0.55 & P_MANIP >= 0.55)
    print("-" * 65)
    print("8. CONFLICT REGION ANALYSIS ON TEST SET")
    print("-" * 65)
    conflict_samples = [r for r in pred_csv_rows if r["p_gen_ai"] >= 0.55 and r["p_manip_manipulated"] >= 0.55]
    conflict_gt = Counter(r["ground_truth"] for r in conflict_samples)
    print(f"Total Test Samples in Conflict Region: {len(conflict_samples)}")
    print(f"  - REAL_ORIGINAL : {conflict_gt['REAL_ORIGINAL']} ({conflict_gt['REAL_ORIGINAL']/len(conflict_samples)*100:5.2f}%)")
    print(f"  - AI_GENERATED  : {conflict_gt['AI_GENERATED']} ({conflict_gt['AI_GENERATED']/len(conflict_samples)*100:5.2f}%)")
    print(f"  - AI_MANIPULATED: {conflict_gt['AI_MANIPULATED']} ({conflict_gt['AI_MANIPULATED']/len(conflict_samples)*100:5.2f}%)")

    conflict_csv_path = RESULTS_DIR / "conflict_region_analysis.csv"
    with open(conflict_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ground_truth", "p_gen_ai", "p_manip_manipulated", "baseline_state", "strategy_c_state", "strategy_e_state"])
        for r in conflict_samples:
            writer.writerow([r["image_path"], r["ground_truth"], r["p_gen_ai"], r["p_manip_manipulated"], r["baseline_state"], r["strategy_c_state"], r["strategy_e_state"]])
    print(f"Saved: {conflict_csv_path.name}")
    print()

    # 9. Pair-Aware Consistency Analysis (154 Pairs)
    print("-" * 65)
    print("9. PAIR-AWARE CONSISTENCY ANALYSIS (154 PAIRS)")
    print("-" * 65)
    
    # Map original_id to original record
    test_orig_records = {}
    for r in pred_csv_rows:
        if r["source_dataset"] == "manipulation_v1/test/original":
            test_orig_records[r["image_id"]] = r

    test_manip_records = [r for r in pred_csv_rows if r["source_dataset"] == "manipulation_v1/test/manipulated"]
    
    pair_rows = []
    pair_success_counts = {"Strategy A": 0, "Strategy C": 0, "Strategy E": 0}

    for mr in test_manip_records:
        parts = mr["image_id"].split("_")
        orig_id = parts[0]
        orig_r = test_orig_records.get(orig_id)
        if orig_r:
            succ_a = 1 if (orig_r["baseline_state"] == "REAL_ORIGINAL" and mr["baseline_state"] == "AI_MANIPULATED") else 0
            succ_c = 1 if (orig_r["strategy_c_state"] == "REAL_ORIGINAL" and mr["strategy_c_state"] == "AI_MANIPULATED") else 0
            succ_e = 1 if (orig_r["strategy_e_state"] == "REAL_ORIGINAL" and mr["strategy_e_state"] == "AI_MANIPULATED") else 0

            pair_success_counts["Strategy A"] += succ_a
            pair_success_counts["Strategy C"] += succ_c
            pair_success_counts["Strategy E"] += succ_e

            pair_rows.append({
                "original_id": orig_id,
                "manipulated_id": mr["image_id"],
                "category": mr["source_category"],
                "orig_baseline": orig_r["baseline_state"],
                "manip_baseline": mr["baseline_state"],
                "baseline_pair_success": succ_a,
                "orig_strat_c": orig_r["strategy_c_state"],
                "manip_strat_c": mr["strategy_c_state"],
                "strat_c_pair_success": succ_c,
                "orig_strat_e": orig_r["strategy_e_state"],
                "manip_strat_e": mr["strategy_e_state"],
                "strat_e_pair_success": succ_e,
            })

    pair_csv_path = RESULTS_DIR / "pair_comparison.csv"
    with open(pair_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(pair_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    n_pairs = len(pair_rows)
    print(f"Total Test Pairs Evaluated: {n_pairs}")
    for strat, cnt in pair_success_counts.items():
        print(f"  - {strat:<12s} Pair Success: {cnt:3d} / {n_pairs} ({cnt/n_pairs*100:5.2f}%)")
    print(f"Saved: {pair_csv_path.name}")
    print()

    # 10. Independent Branch Diagnostical Metrics
    print("-" * 65)
    print("10. INDEPENDENT BRANCH PERFORMANCE DIAGNOSTICS")
    print("-" * 65)
    # Generation Branch: 707 images (390 REAL, 317 AI)
    gen_eval_samples = [r for r in raw_eval_records if r["sample"]["ground_truth"] in ["REAL_ORIGINAL", "AI_GENERATED"]]
    tp_g = sum(1 for r in gen_eval_samples if r["gen_label"] == "AI_GENERATED" and r["sample"]["ground_truth"] == "AI_GENERATED")
    tn_g = sum(1 for r in gen_eval_samples if r["gen_label"] == "REAL" and r["sample"]["ground_truth"] == "REAL_ORIGINAL")
    fp_g = sum(1 for r in gen_eval_samples if r["gen_label"] == "AI_GENERATED" and r["sample"]["ground_truth"] == "REAL_ORIGINAL")
    fn_g = sum(1 for r in gen_eval_samples if r["gen_label"] == "REAL" and r["sample"]["ground_truth"] == "AI_GENERATED")
    acc_g = (tp_g + tn_g) / len(gen_eval_samples)
    prec_g = tp_g / (tp_g + fp_g)
    rec_g = tp_g / (tp_g + fn_g)
    f1_g = 2 * prec_g * rec_g / (prec_g + rec_g)
    print(f"Generation Branch (n=707): Accuracy={acc_g*100:.2f}%, Precision={prec_g:.4f}, Recall={rec_g:.4f}, F1={f1_g:.4f}")

    # Manipulation Branch: 248 images (94 Orig, 154 Manip)
    manip_eval_samples = [r for r in raw_eval_records if r["sample"]["source_dataset"].startswith("manipulation_v1")]
    tp_m = sum(1 for r in manip_eval_samples if r["manip_label"] == "AI_MANIPULATED" and r["sample"]["ground_truth"] == "AI_MANIPULATED")
    tn_m = sum(1 for r in manip_eval_samples if r["manip_label"] == "ORIGINAL_REAL" and r["sample"]["ground_truth"] == "REAL_ORIGINAL")
    fp_m = sum(1 for r in manip_eval_samples if r["manip_label"] == "AI_MANIPULATED" and r["sample"]["ground_truth"] == "REAL_ORIGINAL")
    fn_m = sum(1 for r in manip_eval_samples if r["manip_label"] == "ORIGINAL_REAL" and r["sample"]["ground_truth"] == "AI_MANIPULATED")
    acc_m = (tp_m + tn_m) / len(manip_eval_samples)
    prec_m = tp_m / (tp_m + fp_m)
    rec_m = tp_m / (tp_m + fn_m)
    f1_m = 2 * prec_m * rec_m / (prec_m + rec_m)
    print(f"Manipulation Branch (n=248): Accuracy={acc_m*100:.2f}%, Precision={prec_m:.4f}, Recall={rec_m:.4f}, F1={f1_m:.4f}")
    print()

    # 11. Strategy Comparison CSV
    strat_comp_rows = [
        {
            "strategy": "Strategy A (Baseline)",
            "accuracy": metrics_a["accuracy"],
            "macro_precision": metrics_a["macro_precision"],
            "macro_recall": metrics_a["macro_recall"],
            "macro_f1": metrics_a["macro_f1"],
            "real_original_recall": metrics_a["per_class"]["REAL_ORIGINAL"]["recall"],
            "ai_generated_recall": metrics_a["per_class"]["AI_GENERATED"]["recall"],
            "ai_manipulated_recall": metrics_a["per_class"]["AI_MANIPULATED"]["recall"],
            "uncertain_rate": metrics_a["uncertain_rate"],
            "pair_success_rate": round(pair_success_counts["Strategy A"] / n_pairs, 4),
        },
        {
            "strategy": "Strategy C (Manip-Priority)",
            "accuracy": metrics_c["accuracy"],
            "macro_precision": metrics_c["macro_precision"],
            "macro_recall": metrics_c["macro_recall"],
            "macro_f1": metrics_c["macro_f1"],
            "real_original_recall": metrics_c["per_class"]["REAL_ORIGINAL"]["recall"],
            "ai_generated_recall": metrics_c["per_class"]["AI_GENERATED"]["recall"],
            "ai_manipulated_recall": metrics_c["per_class"]["AI_MANIPULATED"]["recall"],
            "uncertain_rate": metrics_c["uncertain_rate"],
            "pair_success_rate": round(pair_success_counts["Strategy C"] / n_pairs, 4),
        },
        {
            "strategy": "Strategy E (2D Probability Fusion)",
            "accuracy": metrics_e["accuracy"],
            "macro_precision": metrics_e["macro_precision"],
            "macro_recall": metrics_e["macro_recall"],
            "macro_f1": metrics_e["macro_f1"],
            "real_original_recall": metrics_e["per_class"]["REAL_ORIGINAL"]["recall"],
            "ai_generated_recall": metrics_e["per_class"]["AI_GENERATED"]["recall"],
            "ai_manipulated_recall": metrics_e["per_class"]["AI_MANIPULATED"]["recall"],
            "uncertain_rate": metrics_e["uncertain_rate"],
            "pair_success_rate": round(pair_success_counts["Strategy E"] / n_pairs, 4),
        },
    ]

    strat_comp_csv_path = RESULTS_DIR / "strategy_comparison.csv"
    with open(strat_comp_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(strat_comp_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strat_comp_rows)
    print(f"Saved: {strat_comp_csv_path.name}")
    print()

    # 12. Model Checkpoint Verification (Post)
    print("-" * 65)
    print("12. MODEL CHECKPOINT VERIFICATION (POST-EVALUATION)")
    print("-" * 65)
    gen_hash_post = compute_file_sha256(GEN_CKPT_PATH)
    manip_hash_post = compute_file_sha256(MANIP_CKPT_PATH)

    gen_match_post = (gen_hash_post == EXPECTED_GEN_HASH)
    manip_match_post = (manip_hash_post == EXPECTED_MANIP_HASH)

    print(f"Generation Model   : {'MATCH' if gen_match_post else 'MISMATCH'} ({gen_hash_post})")
    print(f"Manipulation Model : {'MATCH' if manip_match_post else 'MISMATCH'} ({manip_hash_post})")
    assert gen_match_post and manip_match_post, "STOP: Checkpoint hash changed during evaluation!"

    hashes_data = {
        "timestamp": timestamp,
        "models": {
            "generation": {
                "checkpoint": GEN_CKPT_PATH.name,
                "pre_generation_hash": gen_hash_pre,
                "post_generation_hash": gen_hash_post,
                "status": "MATCH" if gen_match_post else "MISMATCH",
            },
            "manipulation": {
                "checkpoint": MANIP_CKPT_PATH.name,
                "pre_manipulation_hash": manip_hash_pre,
                "post_manipulation_hash": manip_hash_post,
                "status": "MATCH" if manip_match_post else "MISMATCH",
            },
        },
    }
    with open(RESULTS_DIR / "phase13_model_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes_data, f, indent=2)
    print(f"Saved: phase13_model_hashes.json")
    print()

    # 13. Summary Report Text File
    summary_txt_path = RESULTS_DIR / "phase13_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("===========================================================================\n")
        f.write("AIDETECT PHASE 13: FROZEN TEST CONFIRMATION SUMMARY REPORT\n")
        f.write("===========================================================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Test Samples: 861 (REAL=390, GEN=317, MANIP=154)\n")
        f.write(f"Generation Model Hash  : {gen_hash_post} (MATCH)\n")
        f.write(f"Manipulation Model Hash: {manip_hash_post} (MATCH)\n\n")
        f.write("1. STRATEGY COMPARISON:\n")
        for sc in strat_comp_rows:
            f.write(f"  {sc['strategy']}:\n")
            f.write(f"    Accuracy       : {sc['accuracy']*100:.2f}%\n")
            f.write(f"    Macro F1       : {sc['macro_f1']:.4f}\n")
            f.write(f"    REAL Recall    : {sc['real_original_recall']*100:.2f}%\n")
            f.write(f"    GEN Recall     : {sc['ai_generated_recall']*100:.2f}%\n")
            f.write(f"    MANIP Recall   : {sc['ai_manipulated_recall']*100:.2f}%\n")
            f.write(f"    UNCERTAIN Rate : {sc['uncertain_rate']*100:.2f}%\n")
            f.write(f"    Pair Success   : {sc['pair_success_rate']*100:.2f}%\n\n")
        f.write("2. CONFLICT REGION (P_GEN_AI >= 0.55 & P_MANIP >= 0.55):\n")
        f.write(f"  Total: {len(conflict_samples)}\n")
        f.write(f"  REAL_ORIGINAL : {conflict_gt['REAL_ORIGINAL']}\n")
        f.write(f"  AI_GENERATED  : {conflict_gt['AI_GENERATED']}\n")
        f.write(f"  AI_MANIPULATED: {conflict_gt['AI_MANIPULATED']}\n\n")
        f.write("3. INDEPENDENT BRANCH PERFORMANCE:\n")
        f.write(f"  Generation Branch   : Acc={acc_g*100:.2f}%, F1={f1_g:.4f}\n")
        f.write(f"  Manipulation Branch : Acc={acc_m*100:.2f}%, F1={f1_m:.4f}\n")
    print(f"Saved: {summary_txt_path.name}")
    print()
    print("=" * 75)
    print("PHASE 13 FROZEN CONFIRMATION COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    run_phase13_confirmation()

