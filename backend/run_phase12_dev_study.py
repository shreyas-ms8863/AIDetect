"""
backend/run_phase12_dev_study.py
================================
AIDetect Phase 12: Decision Engine Calibration / Development Set Study
Analyzes the 2D probability space and evaluates candidate decision strategies
strictly on DEVELOPMENT data (846 samples). Zero test set leakage.
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
from sklearn.model_selection import StratifiedKFold

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
RESULTS_DIR = BASE_DIR / "phase12_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASET_V4 = PROJECT_DIR / "dataset_v4"
MANIP_V1 = PROJECT_DIR / "manipulation_v1"

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
    strategy_b_generation_first,
    strategy_c_manipulation_priority_in_conflict,
    strategy_d_margin_based,
    LogisticMetaClassifierStrategy,
)

def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# DATASET COLLECTION & INTEGRITY AUDIT
# ---------------------------------------------------------------------------

def collect_development_samples() -> List[Dict[str, Any]]:
    samples = []
    
    # 1. dataset_v4/validation/real/ -> REAL_ORIGINAL
    v4_val_real = sorted(list((DATASET_V4 / "validation" / "real").rglob("*.*")))
    for p in v4_val_real:
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            sub = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "dataset_v4/validation/real",
                "source_category": sub,
                "ground_truth": "REAL_ORIGINAL",
                "image_id": p.stem,
            })

    # 2. dataset_v4/validation/ai/ -> AI_GENERATED
    v4_val_ai = sorted(list((DATASET_V4 / "validation" / "ai").rglob("*.*")))
    for p in v4_val_ai:
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            sub = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "dataset_v4/validation/ai",
                "source_category": sub,
                "ground_truth": "AI_GENERATED",
                "image_id": p.stem,
            })

    # 3. manipulation_v1/validation/original/ -> REAL_ORIGINAL
    mv1_val_orig = sorted(list((MANIP_V1 / "validation" / "original").rglob("*.png")))
    for p in mv1_val_orig:
        if p.is_file():
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "manipulation_v1/validation/original",
                "source_category": "unaltered_source",
                "ground_truth": "REAL_ORIGINAL",
                "image_id": p.stem.replace("orig_", ""),
            })

    # 4. manipulation_v1/validation/manipulated/ -> AI_MANIPULATED
    mv1_val_manip = sorted(list((MANIP_V1 / "validation" / "manipulated").rglob("*.png")))
    for p in mv1_val_manip:
        if p.is_file():
            cat = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "manipulation_v1/validation/manipulated",
                "source_category": cat,
                "ground_truth": "AI_MANIPULATED",
                "image_id": p.stem.replace("manip_", ""),
            })

    return samples

# ---------------------------------------------------------------------------
# METRIC COMPUTATION HELPER
# ---------------------------------------------------------------------------

def compute_3class_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    classes = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
    n_total = len(ground_truths)
    
    # Accuracy: exact match among valid ground truth classes
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / n_total if n_total > 0 else 0.0

    uncertain_count = sum(1 for p in predictions if p == "UNCERTAIN")
    uncertain_rate = uncertain_count / n_total if n_total > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != cls and g == cls)
        unc = sum(1 for p, g in zip(predictions, ground_truths) if p == "UNCERTAIN" and g == cls)
        n_cls = sum(1 for g in ground_truths if g == cls)

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

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "uncertain_count": uncertain_count,
        "per_class": per_class,
    }

# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

def run_phase12():
    print("=" * 75)
    print("AIDETECT PHASE 12: DECISION ENGINE CALIBRATION / DEV SET STUDY")
    print("=" * 75)
    t_start = time.time()
    timestamp = datetime.datetime.now().isoformat()
    print(f"Timestamp   : {timestamp}")
    print(f"Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model   : {torch.cuda.get_device_name(0)}")
    print(f"Results Dir : {RESULTS_DIR}")
    print()

    # 1. Model Checkpoint Verification
    print("-" * 65)
    print("1. MODEL CHECKPOINT VERIFICATION")
    print("-" * 65)
    gen_hash = compute_file_sha256(GEN_CKPT_PATH)
    manip_hash = compute_file_sha256(MANIP_CKPT_PATH)

    print(f"Generation Model   : {GEN_CKPT_PATH.name}")
    print(f"  Hash             : {gen_hash}")
    print(f"  Expected         : {EXPECTED_GEN_HASH}")
    gen_match = (gen_hash == EXPECTED_GEN_HASH)
    print(f"  Status           : {'MATCH' if gen_match else 'MISMATCH'}")
    assert gen_match, "Generation checkpoint hash mismatch!"

    print(f"Manipulation Model : {MANIP_CKPT_PATH.name}")
    print(f"  Hash             : {manip_hash}")
    print(f"  Expected         : {EXPECTED_MANIP_HASH}")
    manip_match = (manip_hash == EXPECTED_MANIP_HASH)
    print(f"  Status           : {'MATCH' if manip_match else 'MISMATCH'}")
    assert manip_match, "Manipulation checkpoint hash mismatch!"
    print()

    # 2. Data Split Integrity & Collection
    print("-" * 65)
    print("2. DATASET SPLIT INSPECTION & LEAKAGE AUDIT")
    print("-" * 65)
    dev_samples = collect_development_samples()
    print(f"Total Development Samples: {len(dev_samples)}")
    class_counts = Counter(s["ground_truth"] for s in dev_samples)
    print(f"  - REAL_ORIGINAL : {class_counts['REAL_ORIGINAL']}")
    print(f"  - AI_GENERATED  : {class_counts['AI_GENERATED']}")
    print(f"  - AI_MANIPULATED: {class_counts['AI_MANIPULATED']}")

    assert len(dev_samples) == 846, f"Expected 846 dev samples, got {len(dev_samples)}"
    assert class_counts["REAL_ORIGINAL"] == 385
    assert class_counts["AI_GENERATED"] == 314
    assert class_counts["AI_MANIPULATED"] == 147

    # Audit zero overlap with locked Phase 11 test sets
    test_files = set()
    for p in (DATASET_V4 / "test").rglob("*.*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            test_files.add(p.name)
    for p in (MANIP_V1 / "test").rglob("*.*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            test_files.add(p.name)

    dev_files = {s["filepath"].name for s in dev_samples}
    overlap_with_test = dev_files.intersection(test_files)
    print(f"Overlap with locked Phase 11 test sets: {len(overlap_with_test)}")
    assert len(overlap_with_test) == 0, f"DATA LEAKAGE: {len(overlap_with_test)} files overlap with test set!"
    print("  CONFIRMED: ZERO overlap with Phase 11 test sets.")
    print()

    # 3. Model Inference Execution
    print("-" * 65)
    print(f"3. EXTRACTING FROZEN PROBABILITIES ON {len(dev_samples)} DEV SAMPLES")
    print("-" * 65)
    pipeline = ForensicInferencePipeline(device=DEVICE)
    
    pred_records = []
    t_inf_start = time.time()
    for idx, s in enumerate(dev_samples):
        res = pipeline.predict(s["filepath"])
        gen = res["generation"]
        manip = res["manipulation"]
        fin = res["final"]

        if (idx + 1) % 200 == 0 or (idx + 1) == len(dev_samples):
            elapsed = time.time() - t_inf_start
            print(f"  Processed {idx + 1} / {len(dev_samples)} images ({elapsed:.1f}s)...")

        rec = {
            "image_path": s["rel_path"],
            "source_dataset": s["source_dataset"],
            "source_category": s["source_category"],
            "ground_truth": s["ground_truth"],
            "p_gen_real": gen["probability_real"],
            "p_gen_ai": gen["probability_ai_generated"],
            "gen_prediction": gen["label"],
            "p_manip_original": manip["probability_original"],
            "p_manip_manipulated": manip["probability_ai_manipulated"],
            "manip_prediction": manip["label"],
            "current_final_state": fin["label"],
            "current_final_confidence": fin["confidence"],
            "current_decision_case": fin.get("decision_case", ""),
            "image_id": s["image_id"],
        }
        pred_records.append(rec)

    # Save development_predictions.csv
    pred_csv_path = RESULTS_DIR / "development_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_path", "source_dataset", "source_category", "ground_truth",
            "p_gen_real", "p_gen_ai", "gen_prediction",
            "p_manip_original", "p_manip_manipulated", "manip_prediction",
            "current_final_state", "current_final_confidence", "current_decision_case",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pred_records)
    print(f"Saved: {pred_csv_path.name} ({len(pred_records)} rows)")
    print()

    # 5. Evaluate Current Decision Engine (Baseline)
    print("-" * 65)
    print("5. CURRENT DECISION ENGINE DEVELOPMENT PERFORMANCE")
    print("-" * 65)
    current_preds = [r["current_final_state"] for r in pred_records]
    ground_truths = [r["ground_truth"] for r in pred_records]
    baseline_metrics = compute_3class_metrics(current_preds, ground_truths)

    print(f"Current Accuracy   : {baseline_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision    : {baseline_metrics['macro_precision']:.4f}")
    print(f"Macro Recall       : {baseline_metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score     : {baseline_metrics['macro_f1']:.4f}")
    print(f"UNCERTAIN Rate     : {baseline_metrics['uncertain_rate'] * 100:.2f}% ({baseline_metrics['uncertain_count']} samples)")
    print("Per-Class Breakdown:")
    for cls, m in baseline_metrics["per_class"].items():
        print(f"  - {cls:14s} (n={m['n_samples']:3d}): Recall={m['recall']*100:5.2f}% | Precision={m['precision']*100:5.2f}% | F1={m['f1']:.4f} | UNCERTAIN={m['uncertain_rate']*100:5.2f}%")
    print()

    # 6. Analyze the Four Regions & Conflict Region
    print("-" * 65)
    print("6. FOUR-REGION & CONFLICT REGION STUDY")
    print("-" * 65)
    
    region_counts = defaultdict(lambda: Counter())
    # Region definitions:
    # A: Gen=REAL, Manip=ORIGINAL
    # B: Gen=AI, Manip=ORIGINAL
    # C: Gen=REAL, Manip=MANIPULATED
    # D: Gen=AI, Manip=MANIPULATED
    for r in pred_records:
        g_pred = r["gen_prediction"]
        m_pred = r["manip_prediction"]
        gt = r["ground_truth"]

        if g_pred == "REAL" and m_pred == "ORIGINAL_REAL":
            reg = "Region_A (Gen=REAL, Manip=ORIGINAL)"
        elif g_pred == "AI_GENERATED" and m_pred == "ORIGINAL_REAL":
            reg = "Region_B (Gen=AI, Manip=ORIGINAL)"
        elif g_pred == "REAL" and m_pred == "AI_MANIPULATED":
            reg = "Region_C (Gen=REAL, Manip=MANIPULATED)"
        else:
            reg = "Region_D (Gen=AI, Manip=MANIPULATED) [CONFLICT]"
        region_counts[reg][gt] += 1

    for reg, counts in sorted(region_counts.items()):
        tot = sum(counts.values())
        print(f"{reg:45s} -> Total={tot:3d} | Real={counts['REAL_ORIGINAL']:3d} | Gen={counts['AI_GENERATED']:3d} | Manip={counts['AI_MANIPULATED']:3d}")

    # Conflict Region D specifically (p_gen_ai >= 0.55 & p_manip >= 0.55)
    strict_conflict = [r for r in pred_records if r["p_gen_ai"] >= 0.55 and r["p_manip_manipulated"] >= 0.55]
    strict_conflict_gt = Counter(r["ground_truth"] for r in strict_conflict)
    print()
    print(f"Strict Conflict Region (P_GEN_AI >= 0.55 & P_MANIP >= 0.55):")
    print(f"  Total Samples in Conflict : {len(strict_conflict)}")
    print(f"  Ground-Truth Breakdown    :")
    print(f"    - REAL_ORIGINAL  : {strict_conflict_gt['REAL_ORIGINAL']}")
    print(f"    - AI_GENERATED   : {strict_conflict_gt['AI_GENERATED']}")
    print(f"    - AI_MANIPULATED : {strict_conflict_gt['AI_MANIPULATED']}")

    # Save conflict analysis CSV
    conflict_csv_path = RESULTS_DIR / "development_conflict_analysis.csv"
    with open(conflict_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ground_truth", "p_gen_ai", "p_manip_manipulated", "delta_manip_minus_gen", "source_dataset", "source_category"])
        for r in strict_conflict:
            delta = round(r["p_manip_manipulated"] - r["p_gen_ai"], 4)
            writer.writerow([r["image_path"], r["ground_truth"], r["p_gen_ai"], r["p_manip_manipulated"], delta, r["source_dataset"], r["source_category"]])
    print(f"Saved: {conflict_csv_path.name} ({len(strict_conflict)} rows)")
    print()

    # 7. Candidate Strategy Evaluations
    print("-" * 65)
    print("7. CANDIDATE DECISION STRATEGIES EVALUATION (ON DEV DATA)")
    print("-" * 65)
    
    candidate_results = {}

    # Strategy A: Current Rule
    strat_a_preds = []
    for r in pred_records:
        st, conf, case = strategy_a_current_rule(
            r["p_gen_real"], r["p_gen_ai"], r["p_manip_original"], r["p_manip_manipulated"]
        )
        strat_a_preds.append(st)
    candidate_results["Strategy A: Current Rule (Baseline)"] = compute_3class_metrics(strat_a_preds, ground_truths)

    # Strategy B: Generation-First (th=0.55)
    strat_b_preds = []
    for r in pred_records:
        st, conf, case = strategy_b_generation_first(
            r["p_gen_real"], r["p_gen_ai"], r["p_manip_original"], r["p_manip_manipulated"], th_gen=0.55, th_manip=0.55
        )
        strat_b_preds.append(st)
    candidate_results["Strategy B: Generation-First (th=0.55)"] = compute_3class_metrics(strat_b_preds, ground_truths)

    # Strategy C: Manipulation-Priority in Conflict (th=0.55)
    strat_c_preds = []
    for r in pred_records:
        st, conf, case = strategy_c_manipulation_priority_in_conflict(
            r["p_gen_real"], r["p_gen_ai"], r["p_manip_original"], r["p_manip_manipulated"], th_gen=0.55, th_manip=0.55
        )
        strat_c_preds.append(st)
    candidate_results["Strategy C: Manip-Priority in Conflict (th=0.55)"] = compute_3class_metrics(strat_c_preds, ground_truths)

    # Strategy D: Margin-Based Decision (th=0.55, margin=0.10)
    strat_d_preds = []
    for r in pred_records:
        st, conf, case = strategy_d_margin_based(
            r["p_gen_real"], r["p_gen_ai"], r["p_manip_original"], r["p_manip_manipulated"], th_gen=0.55, th_manip=0.55, margin_threshold=0.10
        )
        strat_d_preds.append(st)
    candidate_results["Strategy D: Margin-Based (margin=0.10)"] = compute_3class_metrics(strat_d_preds, ground_truths)

    # Strategy E: Logistic Regression Meta-Classifier (5-Fold Stratified CV on Dev Set)
    X = np.array([[r["p_gen_ai"], r["p_manip_manipulated"]] for r in pred_records])
    y_map = {"REAL_ORIGINAL": 0, "AI_GENERATED": 1, "AI_MANIPULATED": 2}
    inv_map = {0: "REAL_ORIGINAL", 1: "AI_GENERATED", 2: "AI_MANIPULATED"}
    y = np.array([y_map[r["ground_truth"]] for r in pred_records])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_e_oof_preds = [""] * len(pred_records)

    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticMetaClassifierStrategy(uncertainty_threshold=0.50)
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])
        for idx, p in zip(val_idx, probs):
            max_idx = int(np.argmax(p))
            max_prob = float(p[max_idx])
            if max_prob < 0.50:
                strat_e_oof_preds[idx] = "UNCERTAIN"
            else:
                strat_e_oof_preds[idx] = inv_map[max_idx]

    candidate_results["Strategy E: Logistic Regression (5-Fold CV)"] = compute_3class_metrics(strat_e_oof_preds, ground_truths)

    # Print summary comparison table
    print(f"{'Strategy':45s} | {'Acc':>6s} | {'MacF1':>6s} | {'RealRec':>7s} | {'GenRec':>7s} | {'ManipRec':>8s} | {'Uncert':>6s}")
    print("-" * 100)
    for name, res in candidate_results.items():
        pc = res["per_class"]
        print(f"{name:45s} | {res['accuracy']*100:5.2f}% | {res['macro_f1']:6.4f} | {pc['REAL_ORIGINAL']['recall']*100:6.2f}% | {pc['AI_GENERATED']['recall']*100:6.2f}% | {pc['AI_MANIPULATED']['recall']*100:7.2f}% | {res['uncertain_rate']*100:5.2f}%")
    print()

    # 8. Systematic 2D Threshold Grid Search
    print("-" * 65)
    print("8. 2D THRESHOLD GRID SEARCH ON DEVELOPMENT SET")
    print("-" * 65)
    grid_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    grid_rows = []

    for tg in grid_thresholds:
        for tm in grid_thresholds:
            # Evaluate Strategy C with these thresholds
            preds_c = []
            for r in pred_records:
                st, _, _ = strategy_c_manipulation_priority_in_conflict(
                    r["p_gen_real"], r["p_gen_ai"], r["p_manip_original"], r["p_manip_manipulated"], th_gen=tg, th_manip=tm
                )
                preds_c.append(st)
            m_c = compute_3class_metrics(preds_c, ground_truths)

            # Evaluate Strategy D with margin 0.10
            preds_d = []
            for r in pred_records:
                st, _, _ = strategy_d_margin_based(
                    r["p_gen_real"], r["p_gen_ai"], r["p_manip_original"], r["p_manip_manipulated"], th_gen=tg, th_manip=tm, margin_threshold=0.10
                )
                preds_d.append(st)
            m_d = compute_3class_metrics(preds_d, ground_truths)

            grid_rows.append({
                "th_gen": tg,
                "th_manip": tm,
                "strat_c_acc": m_c["accuracy"],
                "strat_c_macro_f1": m_c["macro_f1"],
                "strat_c_real_rec": m_c["per_class"]["REAL_ORIGINAL"]["recall"],
                "strat_c_gen_rec": m_c["per_class"]["AI_GENERATED"]["recall"],
                "strat_c_manip_rec": m_c["per_class"]["AI_MANIPULATED"]["recall"],
                "strat_c_uncert": m_c["uncertain_rate"],
                "strat_d_acc": m_d["accuracy"],
                "strat_d_macro_f1": m_d["macro_f1"],
                "strat_d_real_rec": m_d["per_class"]["REAL_ORIGINAL"]["recall"],
                "strat_d_gen_rec": m_d["per_class"]["AI_GENERATED"]["recall"],
                "strat_d_manip_rec": m_d["per_class"]["AI_MANIPULATED"]["recall"],
                "strat_d_uncert": m_d["uncertain_rate"],
            })

    threshold_csv_path = RESULTS_DIR / "development_threshold_grid.csv"
    with open(threshold_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(grid_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grid_rows)
    print(f"Saved: {threshold_csv_path.name} ({len(grid_rows)} grid configurations)")
    print()

    # 9. Development Pair Analysis
    print("-" * 65)
    print("9. DEVELOPMENT PAIR ANALYSIS")
    print("-" * 65)
    
    # Map original_id to original record
    val_orig_records = {}
    for r in pred_records:
        if r["source_dataset"] == "manipulation_v1/validation/original":
            val_orig_records[r["image_id"]] = r

    val_manip_records = [r for r in pred_records if r["source_dataset"] == "manipulation_v1/validation/manipulated"]

    pair_records = []
    for mr in val_manip_records:
        # stem format: manip_{orig_id}_turn1 or similar
        parts = mr["image_id"].split("_")
        orig_id = parts[0]
        orig_r = val_orig_records.get(orig_id)
        if orig_r:
            p_gen_ai_shift = round(mr["p_gen_ai"] - orig_r["p_gen_ai"], 4)
            p_manip_shift = round(mr["p_manip_manipulated"] - orig_r["p_manip_manipulated"], 4)
            pair_records.append({
                "original_id": orig_id,
                "manipulated_id": mr["image_id"],
                "category": mr["source_category"],
                "orig_p_gen_ai": orig_r["p_gen_ai"],
                "orig_p_manip": orig_r["p_manip_manipulated"],
                "orig_current_decision": orig_r["current_final_state"],
                "manip_p_gen_ai": mr["p_gen_ai"],
                "manip_p_manip": mr["p_manip_manipulated"],
                "manip_current_decision": mr["current_final_state"],
                "shift_p_gen_ai": p_gen_ai_shift,
                "shift_p_manip": p_manip_shift,
                "both_correct_baseline": 1 if (orig_r["current_final_state"] == "REAL_ORIGINAL" and mr["current_final_state"] == "AI_MANIPULATED") else 0,
            })

    pair_csv_path = RESULTS_DIR / "development_pair_analysis.csv"
    with open(pair_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(pair_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_records)
    
    both_corr = sum(p["both_correct_baseline"] for p in pair_records)
    avg_gen_shift = np.mean([p["shift_p_gen_ai"] for p in pair_records])
    avg_manip_shift = np.mean([p["shift_p_manip"] for p in pair_records])
    print(f"Total Validation Pairs Evaluated: {len(pair_records)}")
    print(f"  - Both Correct under Baseline : {both_corr} ({both_corr/len(pair_records)*100:.2f}%)")
    print(f"  - Average P_GEN_AI Shift     : +{avg_gen_shift:.4f} (synthetic noise injection)")
    print(f"  - Average P_MANIP Shift      : +{avg_manip_shift:.4f} (manipulation detector response)")
    print(f"Saved: {pair_csv_path.name}")
    print()

    # 10. Generate Visualizations
    print("-" * 65)
    print("10. GENERATING VISUALIZATION PLOTS")
    print("-" * 65)

    # Plot 1: Scatter plot P_GEN_AI vs P_MANIPULATED
    plt.figure(figsize=(9, 8))
    colors = {"REAL_ORIGINAL": "#2ca02c", "AI_GENERATED": "#1f77b4", "AI_MANIPULATED": "#d62728"}
    markers = {"REAL_ORIGINAL": "o", "AI_GENERATED": "^", "AI_MANIPULATED": "s"}

    for cls in ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]:
        pts = [r for r in pred_records if r["ground_truth"] == cls]
        x_vals = [r["p_gen_ai"] for r in pts]
        y_vals = [r["p_manip_manipulated"] for r in pts]
        plt.scatter(x_vals, y_vals, c=colors[cls], marker=markers[cls], label=f"{cls} (n={len(pts)})", alpha=0.6, edgecolors="none", s=35)

    plt.axvline(0.55, color="grey", linestyle="--", linewidth=1, label="Threshold Gen (0.55)")
    plt.axhline(0.55, color="grey", linestyle=":", linewidth=1, label="Threshold Manip (0.55)")
    plt.title("Phase 12: Development Probability Space (P_GEN_AI vs P_MANIPULATED)", fontsize=13, fontweight="bold")
    plt.xlabel("Probability AI Generated (Generation Model)", fontsize=11)
    plt.ylabel("Probability AI Manipulated (Manipulation Model)", fontsize=11)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    scatter_path = RESULTS_DIR / "development_probability_scatter.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path.name}")

    # Plot 2: Histogram of P_GEN_AI by ground truth
    plt.figure(figsize=(9, 6))
    bins = np.linspace(0, 1, 25)
    for cls in ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]:
        vals = [r["p_gen_ai"] for r in pred_records if r["ground_truth"] == cls]
        plt.hist(vals, bins=bins, alpha=0.5, label=f"{cls} (n={len(vals)})", color=colors[cls], density=True)
    plt.axvline(0.55, color="black", linestyle="--", label="Threshold = 0.55")
    plt.title("Development Set: P_GEN_AI Distribution by Ground Truth", fontsize=13, fontweight="bold")
    plt.xlabel("Probability AI Generated", fontsize=11)
    plt.ylabel("Normalized Density", fontsize=11)
    plt.legend(framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    hist_gen_path = RESULTS_DIR / "generation_probability_distribution.png"
    plt.savefig(hist_gen_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_gen_path.name}")

    # Plot 3: Histogram of P_MANIPULATED by ground truth
    plt.figure(figsize=(9, 6))
    for cls in ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]:
        vals = [r["p_manip_manipulated"] for r in pred_records if r["ground_truth"] == cls]
        plt.hist(vals, bins=bins, alpha=0.5, label=f"{cls} (n={len(vals)})", color=colors[cls], density=True)
    plt.axvline(0.55, color="black", linestyle="--", label="Threshold = 0.55")
    plt.title("Development Set: P_MANIPULATED Distribution by Ground Truth", fontsize=13, fontweight="bold")
    plt.xlabel("Probability AI Manipulated", fontsize=11)
    plt.ylabel("Normalized Density", fontsize=11)
    plt.legend(framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    hist_manip_path = RESULTS_DIR / "manipulation_probability_distribution.png"
    plt.savefig(hist_manip_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_manip_path.name}")

    # Plot 4: Conflict Region Analysis
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    conf_classes = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
    conf_counts = [strict_conflict_gt[c] for c in conf_classes]
    bars = plt.bar(conf_classes, conf_counts, color=[colors[c] for c in conf_classes])
    plt.title("Ground-Truth in Conflict Region\n(P_GEN >= 0.55 & P_MANIP >= 0.55)", fontsize=11, fontweight="bold")
    plt.ylabel("Sample Count")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{int(h)}", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, max(conf_counts) * 1.25)
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.subplot(1, 2, 2)
    deltas_manip = [r["p_manip_manipulated"] - r["p_gen_ai"] for r in strict_conflict if r["ground_truth"] == "AI_MANIPULATED"]
    deltas_gen = [r["p_manip_manipulated"] - r["p_gen_ai"] for r in strict_conflict if r["ground_truth"] == "AI_GENERATED"]
    plt.hist(deltas_gen, bins=15, alpha=0.6, color=colors["AI_GENERATED"], label=f"AI_GENERATED (n={len(deltas_gen)})", density=True)
    plt.hist(deltas_manip, bins=15, alpha=0.6, color=colors["AI_MANIPULATED"], label=f"AI_MANIPULATED (n={len(deltas_manip)})", density=True)
    plt.axvline(0.0, color="black", linestyle="--", label="Delta = 0")
    plt.title("Margin Distribution in Conflict Region\n(P_MANIP - P_GEN)", fontsize=11, fontweight="bold")
    plt.xlabel("Margin: P_MANIP - P_GEN")
    plt.ylabel("Density")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    conf_plot_path = RESULTS_DIR / "conflict_region_analysis.png"
    plt.savefig(conf_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {conf_plot_path.name}")
    print()

    # 11. Save Metrics JSON & Summary Report
    print("-" * 65)
    print("11. WRITING FINAL METRICS JSON & SUMMARY REPORT")
    print("-" * 65)
    summary_data = {
        "timestamp": timestamp,
        "development_dataset": {
            "total_samples": len(dev_samples),
            "real_original_count": class_counts["REAL_ORIGINAL"],
            "ai_generated_count": class_counts["AI_GENERATED"],
            "ai_manipulated_count": class_counts["AI_MANIPULATED"],
            "locked_phase11_overlap": 0,
        },
        "model_hashes": {
            "generation": {"hash": gen_hash, "status": "MATCH"},
            "manipulation": {"hash": manip_hash, "status": "MATCH"},
        },
        "baseline_current_rule": baseline_metrics,
        "conflict_region_stats": {
            "total_samples": len(strict_conflict),
            "ground_truth_counts": dict(strict_conflict_gt),
        },
        "candidate_strategies": candidate_results,
        "pair_analysis": {
            "total_pairs": len(pair_records),
            "both_correct_baseline": both_corr,
            "avg_shift_gen_ai": round(float(avg_gen_shift), 4),
            "avg_shift_manip": round(float(avg_manip_shift), 4),
        }
    }

    metrics_json_path = RESULTS_DIR / "development_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved: {metrics_json_path.name}")

    summary_txt_path = RESULTS_DIR / "development_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("===========================================================================\n")
        f.write("AIDETECT PHASE 12: DEVELOPMENT SET STUDY SUMMARY\n")
        f.write("===========================================================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Development Samples: {len(dev_samples)} (REAL={class_counts['REAL_ORIGINAL']}, GEN={class_counts['AI_GENERATED']}, MANIP={class_counts['AI_MANIPULATED']})\n")
        f.write(f"Phase 11 Test Set: LOCKED / NOT USED (Overlap = 0)\n\n")
        f.write("1. MODEL INTEGRITY:\n")
        f.write(f"  Generation Model   : {gen_hash} (MATCH)\n")
        f.write(f"  Manipulation Model : {manip_hash} (MATCH)\n\n")
        f.write("2. CURRENT DECISION ENGINE DEVELOPMENT PERFORMANCE:\n")
        f.write(f"  Accuracy       : {baseline_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Macro Precision: {baseline_metrics['macro_precision']:.4f}\n")
        f.write(f"  Macro Recall   : {baseline_metrics['macro_recall']:.4f}\n")
        f.write(f"  Macro F1       : {baseline_metrics['macro_f1']:.4f}\n")
        f.write(f"  UNCERTAIN Rate : {baseline_metrics['uncertain_rate']*100:.2f}%\n\n")
        f.write("3. CONFLICT REGION (P_GEN_AI >= 0.55 & P_MANIP >= 0.55):\n")
        f.write(f"  Total Samples : {len(strict_conflict)}\n")
        f.write(f"  REAL_ORIGINAL : {strict_conflict_gt['REAL_ORIGINAL']}\n")
        f.write(f"  AI_GENERATED  : {strict_conflict_gt['AI_GENERATED']}\n")
        f.write(f"  AI_MANIPULATED: {strict_conflict_gt['AI_MANIPULATED']}\n\n")
        f.write("4. CANDIDATE STRATEGIES ON DEVELOPMENT SET:\n")
        for name, res in candidate_results.items():
            pc = res["per_class"]
            f.write(f"  - {name}:\n")
            f.write(f"      Macro F1       : {res['macro_f1']:.4f}\n")
            f.write(f"      REAL Recall    : {pc['REAL_ORIGINAL']['recall']*100:.2f}%\n")
            f.write(f"      GEN Recall     : {pc['AI_GENERATED']['recall']*100:.2f}%\n")
            f.write(f"      MANIP Recall   : {pc['AI_MANIPULATED']['recall']*100:.2f}%\n")
            f.write(f"      UNCERTAIN Rate : {res['uncertain_rate']*100:.2f}%\n")
    print(f"Saved: {summary_txt_path.name}")
    print()
    print("=" * 75)
    print("PHASE 12 DEVELOPMENT STUDY COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    run_phase12()
