"""
AIDetect — Step 5 & Step 6: Clean-Test Statistical Evaluation & Subgroup Analysis
================================================================================
Performs:
STEP 5: Clean-Test Statistical Evaluation (N=9,000)
  - Evaluates V5-A, V5-B, V5-C, V5-D continuous predictions:
    * AUROC, AUPRC, MCC, EER, TPR @ 1%, 5%, 10% FPR
    * 1000-resample bootstrap 95% CIs for:
      Accuracy, Precision, Recall, F1, Balanced Accuracy, AUROC, AUPRC, MCC
    * Paired bootstrap 95% CIs for model differences:
      ΔF1, ΔAUROC, ΔMCC across all 6 model pairs
  - Updates:
    * evaluation_results/v5/strengthened_evaluation/clean_metrics_extended.json
    * evaluation_results/v5/strengthened_evaluation/clean_bootstrap_ci.csv
    * evaluation_results/v5/strengthened_evaluation/clean_paired_comparisons.csv

STEP 6: Source-Aware Subgroup Analysis
  - Evaluates performance across 3 distinct data sources without retraining:
    1. CIFAKE Test Subset (N=8,356: 4,208 REAL, 4,148 AI)
    2. Non-CIFAKE Test Subset (N=644: 292 REAL, 352 AI)
    3. Defactify External Benchmark (N=800: 400 REAL, 400 AI)
  - Calculates for each source x 4 models:
    * N, N_REAL, N_AI
    * Accuracy, Precision, Recall, F1, Balanced Accuracy, MCC, AUROC
  - Saves to:
    * evaluation_results/v5/strengthened_evaluation/source_subgroup_analysis.csv
    * evaluation_results/v5/strengthened_evaluation/source_subgroup_analysis.json
"""

import os
import sys
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_PRED_CSV = PROJECT_ROOT / "evaluation_results" / "v5" / "v5_clean_test_predictions.csv"
EXTERNAL_PRED_CSV = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_results" / "external_benchmark_v1_per_image.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "v5" / "strengthened_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("V5-A Spatial", "prob_v5_a", "pred_v5_a"),
    ("V5-B Frequency", "prob_v5_b", "pred_v5_b"),
    ("V5-C Hybrid", "prob_v5_c", "pred_v5_c"),
    ("V5-D Gated Residual", "calibrated_prob_v5_d", "pred_v5_d")
]


# -----------------------------------------------------------------------------
# STATISTICAL METRIC UTILITIES (PURE NUMPY)
# -----------------------------------------------------------------------------
def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    desc_idx = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = y_true[desc_idx]
    s_sorted = scores[desc_idx]

    dist_idx = np.where(np.diff(s_sorted))[0]
    t_idx = np.r_[dist_idx, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[t_idx]
    fps = (1 + t_idx) - tps

    n_pos = tps[-1]
    n_neg = fps[-1]

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    thresholds = np.r_[s_sorted[0] + 1e-6, s_sorted[t_idx]]
    return fpr, tpr, thresholds


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = compute_roc_curve(y_true, scores)
    return float(trapezoid_compat(tpr, fpr))


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    desc_idx = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = y_true[desc_idx]
    s_sorted = scores[desc_idx]

    dist_idx = np.where(np.diff(s_sorted))[0]
    t_idx = np.r_[dist_idx, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[t_idx]
    fps = (1 + t_idx) - tps

    n_pos = tps[-1]
    if n_pos == 0:
        return 0.0

    precision = tps / (tps + fps)
    recall = tps / n_pos

    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]

    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    num = float(tp * tn - fp * fn)
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if den == 0 else num / den


def compute_contingency_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = len(y_true)

    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2.0 if (tn + fp) > 0 and (tp + fn) > 0 else acc
    mcc = compute_mcc(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "real_fpr": fp / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ai_fnr": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = int(np.argmin(diff))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# -----------------------------------------------------------------------------
# STEP 5 & 6 EXECUTION
# -----------------------------------------------------------------------------
def run_step5_and_6():
    print("=" * 80)
    print("AIDETECT — STEP 5: CLEAN-TEST STATISTICAL EVALUATION")
    print("=" * 80)

    if not CLEAN_PRED_CSV.is_file():
        raise FileNotFoundError(f"Missing clean test predictions file: {CLEAN_PRED_CSV}")

    df_clean = pd.read_csv(CLEAN_PRED_CSV)
    print(f"[*] Loaded clean predictions: {len(df_clean)} rows.")

    y_true = df_clean["label_int"].values
    N_total = len(y_true)
    assert N_total == 9000

    # 1. Point Estimates for All 4 Models
    clean_point_estimates: Dict[str, Dict[str, Any]] = {}
    model_scores: Dict[str, np.ndarray] = {}
    model_preds: Dict[str, np.ndarray] = {}

    for disp_name, prob_col, pred_col in MODELS:
        scores = df_clean[prob_col].values
        preds = df_clean[pred_col].values
        model_scores[disp_name] = scores
        model_preds[disp_name] = preds

        auc = compute_auroc(y_true, scores)
        ap = compute_auprc(y_true, scores)
        mcc = compute_mcc(y_true, preds)

        fpr, tpr, _ = compute_roc_curve(y_true, scores)
        eer = compute_eer(fpr, tpr)

        tpr_1 = float(np.interp(0.01, fpr, tpr))
        tpr_5 = float(np.interp(0.05, fpr, tpr))
        tpr_10 = float(np.interp(0.10, fpr, tpr))

        base_cm = compute_contingency_metrics(y_true, preds)

        clean_point_estimates[disp_name] = {
            "accuracy": base_cm["accuracy"],
            "precision": base_cm["precision"],
            "recall": base_cm["recall"],
            "f1": base_cm["f1"],
            "balanced_accuracy": base_cm["balanced_accuracy"],
            "mcc": mcc,
            "auroc": auc,
            "auprc": ap,
            "eer": eer,
            "tpr_at_1_pct_fpr": tpr_1,
            "tpr_at_5_pct_fpr": tpr_5,
            "tpr_at_10_pct_fpr": tpr_10,
            "real_fpr": base_cm["real_fpr"],
            "ai_fnr": base_cm["ai_fnr"],
            "tp": base_cm["tp"], "tn": base_cm["tn"], "fp": base_cm["fp"], "fn": base_cm["fn"]
        }

    print("[*] Clean point estimates computed for all 4 models.")

    # 2. 1000-Resample Bootstrap CIs
    print("\n[*] Running 1,000-resample bootstrap on clean test set...")
    B = 1000
    rng = np.random.default_rng(seed=42)
    boot_indices = rng.integers(0, N_total, size=(B, N_total))

    metrics_to_bootstrap = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auroc", "auprc", "mcc"]
    model_names = [m[0] for m in MODELS]

    boot_distributions: Dict[str, Dict[str, np.ndarray]] = {
        m: {metric: np.zeros(B, dtype=float) for metric in metrics_to_bootstrap} for m in model_names
    }

    t0_boot = time.time()
    for b in range(B):
        idx_b = boot_indices[b]
        y_b = y_true[idx_b]

        for disp_name in model_names:
            s_b = model_scores[disp_name][idx_b]
            p_b = model_preds[disp_name][idx_b]

            cm_b = compute_contingency_metrics(y_b, p_b)
            auc_b = compute_auroc(y_b, s_b)
            ap_b = compute_auprc(y_b, s_b)

            boot_distributions[disp_name]["accuracy"][b] = cm_b["accuracy"]
            boot_distributions[disp_name]["precision"][b] = cm_b["precision"]
            boot_distributions[disp_name]["recall"][b] = cm_b["recall"]
            boot_distributions[disp_name]["f1"][b] = cm_b["f1"]
            boot_distributions[disp_name]["balanced_accuracy"][b] = cm_b["balanced_accuracy"]
            boot_distributions[disp_name]["mcc"][b] = cm_b["mcc"]
            boot_distributions[disp_name]["auroc"][b] = auc_b
            boot_distributions[disp_name]["auprc"][b] = ap_b

    print(f"[*] Clean bootstrap finished in {time.time() - t0_boot:.2f} seconds.")

    # Single-model CI rows
    clean_ci_rows = []
    clean_ci_dict: Dict[str, Dict[str, Any]] = {m: {} for m in model_names}

    for m in model_names:
        for metric in metrics_to_bootstrap:
            arr = boot_distributions[m][metric]
            pe = clean_point_estimates[m][metric]
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            se = float(np.std(arr, ddof=1))
            b_mean = float(np.mean(arr))

            clean_ci_dict[m][metric] = {
                "point_estimate": round(pe, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "std_error": round(se, 6),
                "bootstrap_mean": round(b_mean, 6)
            }

            clean_ci_rows.append({
                "model": m,
                "metric": metric,
                "point_estimate": round(pe, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "std_error": round(se, 6),
                "bootstrap_mean": round(b_mean, 6)
            })

    df_clean_ci = pd.DataFrame(clean_ci_rows)
    df_clean_ci.to_csv(OUTPUT_DIR / "clean_bootstrap_ci.csv", index=False)

    # Paired Bootstrap CIs
    paired_comparisons = [
        ("V5-A Spatial", "V5-B Frequency", "A vs B"),
        ("V5-A Spatial", "V5-C Hybrid", "A vs C"),
        ("V5-A Spatial", "V5-D Gated Residual", "A vs D"),
        ("V5-B Frequency", "V5-C Hybrid", "B vs C"),
        ("V5-B Frequency", "V5-D Gated Residual", "B vs D"),
        ("V5-C Hybrid", "V5-D Gated Residual", "C vs D"),
    ]

    diff_metrics = ["f1", "auroc", "mcc", "accuracy", "balanced_accuracy"]
    paired_rows = []
    clean_paired_dict: Dict[str, Any] = {}

    for m1, m2, label in paired_comparisons:
        clean_paired_dict[label] = {}
        for metric in diff_metrics:
            diff_arr = boot_distributions[m1][metric] - boot_distributions[m2][metric]
            pe_diff = clean_point_estimates[m1][metric] - clean_point_estimates[m2][metric]
            ci_low = float(np.percentile(diff_arr, 2.5))
            ci_high = float(np.percentile(diff_arr, 97.5))
            se = float(np.std(diff_arr, ddof=1))
            is_sig = not (ci_low <= 0.0 <= ci_high)
            p_val = 2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0))
            p_val = max(p_val, 1.0 / B)

            clean_paired_dict[label][metric] = {
                "model_1": m1,
                "model_2": m2,
                "point_estimate_diff": round(pe_diff, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "std_error": round(se, 6),
                "statistically_significant_05": is_sig,
                "approx_two_sided_p_value": round(p_val, 4)
            }

            paired_rows.append({
                "comparison": label,
                "model_1": m1,
                "model_2": m2,
                "metric": metric,
                "point_estimate_diff": round(pe_diff, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "std_error": round(se, 6),
                "statistically_significant_05": is_sig,
                "approx_two_sided_p_value": round(p_val, 4)
            })

    df_paired_clean = pd.DataFrame(paired_rows)
    df_paired_clean.to_csv(OUTPUT_DIR / "clean_paired_comparisons.csv", index=False)

    # Update clean_metrics_extended.json with complete point estimates, bootstrap CIs, and paired comparisons
    clean_master = {
        "dataset": "V5 Frozen Test Set (N=9,000)",
        "sample_size": N_total,
        "classes": {"REAL": int(np.sum(y_true == 0)), "AI": int(np.sum(y_true == 1))},
        "threshold": 0.50,
        "point_estimates": clean_point_estimates,
        "bootstrap_confidence_intervals": clean_ci_dict,
        "paired_model_comparisons": clean_paired_dict
    }

    with open(OUTPUT_DIR / "clean_metrics_extended.json", "w", encoding="utf-8") as f:
        json.dump(clean_master, f, indent=2)

    print(f"[*] Step 5 outputs saved successfully to: {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # STEP 6: SOURCE-AWARE SUBGROUP ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AIDETECT — STEP 6: SOURCE-AWARE SUBGROUP ANALYSIS")
    print("=" * 80)

    # 1. CIFAKE subset
    is_cifake = df_clean["source"].str.contains("cifake", case=False).values
    cifake_df = df_clean[is_cifake]

    # 2. Non-CIFAKE subset
    non_cifake_df = df_clean[~is_cifake]

    # 3. Defactify external benchmark
    if not EXTERNAL_PRED_CSV.is_file():
        raise FileNotFoundError(f"Missing external predictions file: {EXTERNAL_PRED_CSV}")
    df_ext = pd.read_csv(EXTERNAL_PRED_CSV)
    piv_ext_prob = df_ext.pivot(index="candidate_id", columns="model", values="calibrated_ai_probability")
    piv_ext_pred = df_ext.pivot(index="candidate_id", columns="model", values="prediction")
    ext_meta = df_ext[["candidate_id", "true_label"]].drop_duplicates().set_index("candidate_id")
    ext_y_true = (ext_meta.loc[piv_ext_prob.index, "true_label"] == "AI").astype(int).values

    subgroups = [
        ("CIFAKE (In-Distribution / Short-cut dominated)", cifake_df, "clean"),
        ("Non-CIFAKE (In-Distribution Natural High-Res)", non_cifake_df, "clean"),
        ("Defactify External Benchmark (Out-of-Distribution)", df_ext, "external")
    ]

    subgroup_rows = []
    subgroup_json_dict = {}

    for sg_name, sg_data, sg_type in subgroups:
        subgroup_json_dict[sg_name] = {}

        if sg_type == "clean":
            y_sg = sg_data["label_int"].values
            n_sg = len(y_sg)
            n_real = int(np.sum(y_sg == 0))
            n_ai = int(np.sum(y_sg == 1))

            for disp_name, prob_col, pred_col in MODELS:
                s_sg = sg_data[prob_col].values
                p_sg = sg_data[pred_col].values

                cm = compute_contingency_metrics(y_sg, p_sg)
                auc = compute_auroc(y_sg, s_sg)
                mcc = compute_mcc(y_sg, p_sg)

                row = {
                    "subgroup": sg_name,
                    "model": disp_name,
                    "N": n_sg,
                    "N_REAL": n_real,
                    "N_AI": n_ai,
                    "accuracy": round(cm["accuracy"], 6),
                    "precision": round(cm["precision"], 6),
                    "recall": round(cm["recall"], 6),
                    "f1": round(cm["f1"], 6),
                    "balanced_accuracy": round(cm["balanced_accuracy"], 6),
                    "mcc": round(mcc, 6),
                    "auroc": round(auc, 6),
                    "real_fpr": round(cm["real_fpr"], 6),
                    "ai_fnr": round(cm["ai_fnr"], 6),
                    "tp": cm["tp"], "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"]
                }
                subgroup_rows.append(row)
                subgroup_json_dict[sg_name][disp_name] = row

        elif sg_type == "external":
            n_sg = len(ext_y_true)
            n_real = int(np.sum(ext_y_true == 0))
            n_ai = int(np.sum(ext_y_true == 1))

            for disp_name, _, _ in MODELS:
                s_sg = piv_ext_prob[disp_name].values
                p_sg = (piv_ext_pred[disp_name].values == "AI").astype(int)

                cm = compute_contingency_metrics(ext_y_true, p_sg)
                auc = compute_auroc(ext_y_true, s_sg)
                mcc = compute_mcc(ext_y_true, p_sg)

                row = {
                    "subgroup": sg_name,
                    "model": disp_name,
                    "N": n_sg,
                    "N_REAL": n_real,
                    "N_AI": n_ai,
                    "accuracy": round(cm["accuracy"], 6),
                    "precision": round(cm["precision"], 6),
                    "recall": round(cm["recall"], 6),
                    "f1": round(cm["f1"], 6),
                    "balanced_accuracy": round(cm["balanced_accuracy"], 6),
                    "mcc": round(mcc, 6),
                    "auroc": round(auc, 6),
                    "real_fpr": round(cm["real_fpr"], 6),
                    "ai_fnr": round(cm["ai_fnr"], 6),
                    "tp": cm["tp"], "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"]
                }
                subgroup_rows.append(row)
                subgroup_json_dict[sg_name][disp_name] = row

    df_subgroups = pd.DataFrame(subgroup_rows)
    df_subgroups.to_csv(OUTPUT_DIR / "source_subgroup_analysis.csv", index=False)

    with open(OUTPUT_DIR / "source_subgroup_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "subgroups_evaluated": [sg[0] for sg in subgroups],
            "results": subgroup_json_dict
        }, f, indent=2)

    print(f"[*] Step 6 outputs saved successfully to: {OUTPUT_DIR}")
    print("=" * 80)
    print("STEP 5 & STEP 6 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
     run_step5_and_6()
