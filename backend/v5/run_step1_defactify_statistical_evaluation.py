"""
AIDetect — Step 1: Defactify Statistical Evaluation
===================================================
Executes zero-inference statistical evaluation of frozen Defactify external benchmark
predictions (N=800 images x 4 models = 3,200 prediction rows).

Calculates for V5-A/B/C/D:
  - AUROC, AUPRC, MCC, EER
  - TPR @ 1%, 5%, 10% FPR
  - 1000-resample bootstrap 95% CIs for:
      Accuracy, Precision, Recall, F1, Balanced Accuracy, AUROC, AUPRC, MCC
  - Paired bootstrap 95% CIs for model differences:
      ΔF1, ΔAUROC, ΔMCC across all 6 model pairs (A-B, A-C, A-D, B-C, B-D, C-D)
  - Generator-wise recall (SD3, MJv6, DALL-E 3, SDXL)
  - Macro generator recall, Worst-generator recall, Generator recall standard deviation

For V5-D additionally calculates external:
  - NLL, Brier score, ECE, MCE, reliability bin data
  - Risk-coverage / selective prediction across confidence thresholds

Outputs into:
  evaluation_results/v5/strengthened_evaluation/
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
INPUT_PER_IMAGE_CSV = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_results" / "external_benchmark_v1_per_image.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "v5" / "strengthened_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "V5-A Spatial",
    "V5-B Frequency",
    "V5-C Hybrid",
    "V5-D Gated Residual"
]

GENERATOR_KEYS = [
    "Stable_Diffusion_3",
    "Midjourney_v6",
    "DALL-E_3",
    "Stable_Diffusion_XL"
]

GENERATOR_NAMES = {
    "Stable_Diffusion_3": "SD3",
    "Midjourney_v6": "Midjourney v6",
    "DALL-E_3": "DALL-E 3",
    "Stable_Diffusion_XL": "SDXL",
    "None_Authentic_Photograph": "REAL (Natural Photos)"
}


# -----------------------------------------------------------------------------
# PURE-NUMPY / STATISTICAL UTILITIES
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

    # Prepend baseline
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]

    # Step-wise integration (Average Precision)
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


def compute_calibration_curve_and_errors(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Calculates ECE, MCE, and 10-bin reliability data."""
    preds = (probs >= 0.50).astype(int)
    corrects = (preds == y_true).astype(float)
    confs = np.maximum(probs, 1.0 - probs)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rel_bins = []

    N = len(y_true)
    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (probs >= b_low) & (probs <= b_high)
        else:
            in_bin = (probs >= b_low) & (probs < b_high)

        cnt = int(np.sum(in_bin))
        if cnt > 0:
            bin_acc = float(np.mean(corrects[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            mean_pred_prob = float(np.mean(probs[in_bin]))
            true_ai_freq = float(np.mean(y_true[in_bin]))
            bin_err = abs(bin_acc - bin_conf)

            ece += (cnt / N) * bin_err
            if bin_err > mce:
                mce = bin_err
        else:
            bin_acc = None
            bin_conf = None
            mean_pred_prob = float((b_low + b_high) / 2.0)
            true_ai_freq = None

        rel_bins.append({
            "bin_idx": i + 1,
            "bin_lower": round(float(b_low), 2),
            "bin_upper": round(float(b_high), 2),
            "sample_count": cnt,
            "mean_predicted_prob": round(mean_pred_prob, 4) if mean_pred_prob is not None else None,
            "true_ai_fraction": round(true_ai_freq, 4) if true_ai_freq is not None else None,
            "mean_confidence": round(bin_conf, 4) if bin_conf is not None else None,
            "accuracy": round(bin_acc, 4) if bin_acc is not None else None
        })

    return float(ece), float(mce), rel_bins


def compute_nll_and_brier(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    nll = -float(np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))
    brier = float(np.mean((probs - y_true) ** 2))
    return nll, brier


def compute_risk_coverage(y_true: np.ndarray, probs: np.ndarray) -> List[Dict[str, Any]]:
    """Calculates coverage, risk, selective accuracy across confidence thresholds."""
    confs = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.50).astype(int)
    corrects = (preds == y_true).astype(int)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    N = len(y_true)
    results = []

    for t in thresholds:
        covered_mask = (confs >= t)
        cov_cnt = int(np.sum(covered_mask))
        cov_pct = (cov_cnt / N) * 100.0
        abst_cnt = N - cov_cnt
        abst_pct = (abst_cnt / N) * 100.0

        if cov_cnt > 0:
            sel_acc = float(np.mean(corrects[covered_mask]))
            sel_risk = 1.0 - sel_acc
            # Selective F1 on covered subset
            y_cov = y_true[covered_mask]
            p_cov = preds[covered_mask]
            sel_cm = compute_contingency_metrics(y_cov, p_cov)
            sel_f1 = sel_cm["f1"]
        else:
            sel_acc = 1.0
            sel_risk = 0.0
            sel_f1 = 0.0

        results.append({
            "confidence_threshold": round(float(t), 2),
            "covered_samples": cov_cnt,
            "coverage_percentage": round(cov_pct, 2),
            "abstained_samples": abst_cnt,
            "abstention_percentage": round(abst_pct, 2),
            "selective_accuracy": round(sel_acc, 6),
            "selective_risk": round(sel_risk, 6),
            "selective_f1": round(sel_f1, 6)
        })
    return results


# -----------------------------------------------------------------------------
# STEP 1 PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def run_step1_evaluation():
    print("=" * 80)
    print("AIDETECT — STEP 1: DEFACTIFY STATISTICAL EVALUATION")
    print("=" * 80)

    if not INPUT_PER_IMAGE_CSV.is_file():
        raise FileNotFoundError(f"Missing input predictions file: {INPUT_PER_IMAGE_CSV}")

    df = pd.read_csv(INPUT_PER_IMAGE_CSV)
    print(f"[*] Loaded external predictions: {len(df)} rows across {df['candidate_id'].nunique()} images.")

    # Pivot continuous scores and binary predictions
    piv_prob = df.pivot(index="candidate_id", columns="model", values="calibrated_ai_probability")
    piv_pred = df.pivot(index="candidate_id", columns="model", values="prediction")
    meta = df[["candidate_id", "original_filename", "true_label", "generator"]].drop_duplicates(subset=["candidate_id"]).set_index("candidate_id")

    # Align every table explicitly by candidate_id.  Do not use positional
    # boolean masks: the input CSV is grouped by source while pandas.pivot()
    # sorts its index lexicographically by candidate_id.
    candidate_ids = piv_prob.index
    meta = meta.reindex(candidate_ids)
    piv_pred = piv_pred.reindex(candidate_ids)
    if meta.isna().any().any() or piv_pred.isna().any().any():
        raise ValueError("Candidate metadata/predictions could not be aligned by candidate_id")

    # Align ground truth array
    y_true = (meta["true_label"] == "AI").astype(int).values
    N_total = len(y_true)

    print(f"[*] Sample alignment confirmed: N={N_total} ({np.sum(y_true==0)} REAL, {np.sum(y_true==1)} AI).")

    # -------------------------------------------------------------------------
    # 1. POINT ESTIMATES FOR ALL 4 MODELS
    # -------------------------------------------------------------------------
    model_point_estimates: Dict[str, Dict[str, Any]] = {}

    for m in MODELS:
        scores = piv_prob[m].values
        preds = (piv_pred[m].values == "AI").astype(int)

        auc = compute_auroc(y_true, scores)
        ap = compute_auprc(y_true, scores)
        mcc = compute_mcc(y_true, preds)

        fpr, tpr, _ = compute_roc_curve(y_true, scores)
        eer = compute_eer(fpr, tpr)

        tpr_1 = float(np.interp(0.01, fpr, tpr))
        tpr_5 = float(np.interp(0.05, fpr, tpr))
        tpr_10 = float(np.interp(0.10, fpr, tpr))

        base_cm = compute_contingency_metrics(y_true, preds)

        model_point_estimates[m] = {
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

    print("\n[*] Point estimates calculated for AUROC, AUPRC, MCC, EER, and TPR @ fixed FPR.")

    # -------------------------------------------------------------------------
    # 2. 1000-RESAMPLE BOOTSTRAP 95% CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    print("\n[*] Running 1,000-resample bootstrap for single-model CIs and paired model differences...")
    B = 1000
    rng = np.random.default_rng(seed=42)
    boot_indices = rng.integers(0, N_total, size=(B, N_total))

    metrics_to_bootstrap = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auroc", "auprc", "mcc"]

    # Store bootstrap distributions: {model: {metric: np.ndarray(B,)}}
    boot_distributions: Dict[str, Dict[str, np.ndarray]] = {
        m: {metric: np.zeros(B, dtype=float) for metric in metrics_to_bootstrap} for m in MODELS
    }

    start_boot = time.time()
    for b in range(B):
        idx_b = boot_indices[b]
        y_b = y_true[idx_b]

        for m in MODELS:
            scores_b = piv_prob[m].values[idx_b]
            preds_b = (piv_pred[m].values[idx_b] == "AI").astype(int)

            cm_b = compute_contingency_metrics(y_b, preds_b)
            auc_b = compute_auroc(y_b, scores_b)
            ap_b = compute_auprc(y_b, scores_b)

            boot_distributions[m]["accuracy"][b] = cm_b["accuracy"]
            boot_distributions[m]["precision"][b] = cm_b["precision"]
            boot_distributions[m]["recall"][b] = cm_b["recall"]
            boot_distributions[m]["f1"][b] = cm_b["f1"]
            boot_distributions[m]["balanced_accuracy"][b] = cm_b["balanced_accuracy"]
            boot_distributions[m]["mcc"][b] = cm_b["mcc"]
            boot_distributions[m]["auroc"][b] = auc_b
            boot_distributions[m]["auprc"][b] = ap_b

    print(f"[*] Bootstrap completed in {time.time() - start_boot:.2f} seconds.")

    # Build single-model CI summary rows
    single_ci_rows = []
    single_ci_dict: Dict[str, Dict[str, Any]] = {m: {} for m in MODELS}

    for m in MODELS:
        for metric in metrics_to_bootstrap:
            arr = boot_distributions[m][metric]
            pe = model_point_estimates[m][metric]
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            se = float(np.std(arr, ddof=1))
            b_mean = float(np.mean(arr))

            single_ci_dict[m][metric] = {
                "point_estimate": round(pe, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "std_error": round(se, 6),
                "bootstrap_mean": round(b_mean, 6)
            }

            single_ci_rows.append({
                "model": m,
                "metric": metric,
                "point_estimate": round(pe, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "std_error": round(se, 6),
                "bootstrap_mean": round(b_mean, 6)
            })

    df_single_ci = pd.DataFrame(single_ci_rows)
    df_single_ci.to_csv(OUTPUT_DIR / "external_bootstrap_ci.csv", index=False)

    # -------------------------------------------------------------------------
    # 3. PAIRED BOOTSTRAP CIs FOR MODEL DIFFERENCES
    # -------------------------------------------------------------------------
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
    paired_dict: Dict[str, Any] = {}

    for m1, m2, label in paired_comparisons:
        paired_dict[label] = {}
        for metric in diff_metrics:
            diff_arr = boot_distributions[m1][metric] - boot_distributions[m2][metric]
            pe_diff = model_point_estimates[m1][metric] - model_point_estimates[m2][metric]
            ci_low = float(np.percentile(diff_arr, 2.5))
            ci_high = float(np.percentile(diff_arr, 97.5))
            se = float(np.std(diff_arr, ddof=1))

            # Statistical significance: does 95% CI cross zero?
            is_sig = not (ci_low <= 0.0 <= ci_high)

            # Two-sided empirical p-value approximation
            p_val = 2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0))
            p_val = max(p_val, 1.0 / B) # bounded by resolution

            paired_dict[label][metric] = {
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

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(OUTPUT_DIR / "external_paired_comparisons.csv", index=False)

    # -------------------------------------------------------------------------
    # 4. GENERATOR-WISE RECALL & MACRO ANALYSIS
    # -------------------------------------------------------------------------
    gen_rows = []
    gen_analysis_dict: Dict[str, Any] = {}

    for m in MODELS:
        gen_recalls = {}
        for g_raw in GENERATOR_KEYS:
            g_disp = GENERATOR_NAMES[g_raw]
            generator_ids = meta.index[
                (meta["generator"] == g_raw) & (meta["true_label"] == "AI")
            ]
            N_g = len(generator_ids)
            if N_g == 0:
                raise ValueError(f"No AI candidates found for generator {g_raw}")

            # AI recall on this generator, with both metadata and predictions
            # selected by the same explicit candidate_id index.
            preds_g = (piv_pred.loc[generator_ids, m].values == "AI").astype(int)
            rec_g = float(np.sum(preds_g == 1)) / N_g
            gen_recalls[g_disp] = rec_g

        rec_values = list(gen_recalls.values())
        macro_rec = float(np.mean(rec_values))
        overall_ai_recall = model_point_estimates[m]["recall"]
        if not np.isclose(macro_rec, overall_ai_recall, atol=1e-12):
            raise AssertionError(
                f"Generator mean {macro_rec:.12f} does not equal overall AI recall "
                f"{overall_ai_recall:.12f} for {m}"
            )
        worst_rec = float(np.min(rec_values))
        std_rec_sample = float(np.std(rec_values, ddof=1))

        gen_analysis_dict[m] = {
            "generator_recalls": {k: round(v, 4) for k, v in gen_recalls.items()},
            "macro_generator_recall": round(macro_rec, 4),
            "worst_generator_recall": round(worst_rec, 4),
            "generator_recall_std_dev": round(std_rec_sample, 4)
        }

        for g_disp, r_val in gen_recalls.items():
            gen_rows.append({
                "model": m,
                "generator": g_disp,
                "N": 100,
                "recall": round(r_val, 4),
                "fnr": round(1.0 - r_val, 4),
                "macro_generator_recall": round(macro_rec, 4),
                "worst_generator_recall": round(worst_rec, 4),
                "generator_recall_std_dev": round(std_rec_sample, 4)
            })

    df_gen_analysis = pd.DataFrame(gen_rows)
    df_gen_analysis.to_csv(OUTPUT_DIR / "generator_analysis.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. V5-D EXTERNAL CALIBRATION & RISK-COVERAGE ANALYSIS
    # -------------------------------------------------------------------------
    v5_d_scores = piv_prob["V5-D Gated Residual"].values
    nll_d, brier_d = compute_nll_and_brier(y_true, v5_d_scores)
    ece_d, mce_d, rel_bins_d = compute_calibration_curve_and_errors(y_true, v5_d_scores, n_bins=10)

    calibration_dict = {
        "model": "V5-D Gated Residual",
        "dataset": "Defactify External Benchmark v1 (N=800)",
        "calibration_configuration": {
            "method": "Temperature Scaling",
            "frozen_temperature_T": 2.1983866642849734,
            "source_config": "backend/models/v5/v5_d_calibration.json"
        },
        "external_calibration_metrics": {
            "nll": round(nll_d, 6),
            "brier_score": round(brier_d, 6),
            "ece": round(ece_d, 6),
            "mce": round(mce_d, 6)
        },
        "reliability_bins": rel_bins_d,
        "methodology_and_limitations": {
            "calibration_fit_source": "Held-out partition of the 9,000-image V5 validation set",
            "selection_limitation": "The reported validation metrics were evaluated on the same half used to select between Platt and Temperature scaling; it was not evaluated on an untouched final validation split.",
            "external_transfer_limitation": "Temperature scaling (T=2.198387) was frozen from in-distribution validation data and evaluated out-of-distribution without refitting. While it drastically reduced extreme overconfidence on REAL false alarms (from 31.75% to 9.00%), out-of-distribution distribution shift causes elevated NLL and ECE compared to in-distribution validation."
        }
    }

    with open(OUTPUT_DIR / "calibration_analysis.json", "w", encoding="utf-8") as f:
        json.dump(calibration_dict, f, indent=2)

    risk_cov_data = compute_risk_coverage(y_true, v5_d_scores)
    df_risk_cov = pd.DataFrame(risk_cov_data)
    df_risk_cov.to_csv(OUTPUT_DIR / "risk_coverage.csv", index=False)

    # -------------------------------------------------------------------------
    # 6. MASTER JSON ARTIFACT (external_metrics_extended.json)
    # -------------------------------------------------------------------------
    master_extended = {
        "dataset": "Defactify External Benchmark v1",
        "sample_size": N_total,
        "classes": {"REAL": int(np.sum(y_true == 0)), "AI": int(np.sum(y_true == 1))},
        "threshold": 0.50,
        "point_estimates": model_point_estimates,
        "bootstrap_confidence_intervals": single_ci_dict,
        "paired_model_comparisons": paired_dict,
        "generator_generalization_analysis": gen_analysis_dict,
        "v5_d_calibration_summary": {
            "nll": round(nll_d, 6),
            "brier": round(brier_d, 6),
            "ece": round(ece_d, 6),
            "mce": round(mce_d, 6)
        }
    }

    with open(OUTPUT_DIR / "external_metrics_extended.json", "w", encoding="utf-8") as f:
        json.dump(master_extended, f, indent=2)

    print(f"\n[*] All Step 1 artifacts successfully saved to: {OUTPUT_DIR}")
    print("=" * 80)
    print("STEP 1 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_step1_evaluation()
