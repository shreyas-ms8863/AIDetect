"""
AIDetect V5-D Post-Hoc Probability Calibration Pipeline
========================================================
Phases 2-6 & 10: Deterministic Calibration Fitting, Operating Metrics & Research Outputs.

SAFETY INVARIANTS:
1. Operates SOLELY on extracted validation logits (v5_d_val_logits.npz).
2. ZERO access to the frozen 9,000-image V5 TEST split.
3. Model weights are completely UNTOUCHED (zero retraining, zero weight alteration).
4. Deterministic split: seed=42, 50/50 stratified partition of 9,000 validation images:
   - Calibration-Fit: 4,500 samples (2,250 REAL, 2,250 AI)
   - Calibration-Evaluation: 4,500 samples (2,250 REAL, 2,250 AI)
5. Calibration-evaluation subset is strictly firewalled and NEVER used for optimizing T, A, or B.
6. A > 0 enforced for Platt scaling, guaranteeing strict preservation of sample ranking and ROC-AUC.

Generates:
  - backend/models/v5/v5_d_calibration.json
  - backend/evaluation_results/v5/calibration/v5_d_calibration_metrics.json
  - backend/evaluation_results/v5/calibration/v5_d_calibration_comparison.csv
  - backend/evaluation_results/v5/calibration/v5_d_reliability_data.csv
  - backend/evaluation_results/v5/calibration/v5_d_reliability_diagram.png
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project directory structure
V5_DIR = Path(__file__).resolve().parent
BACKEND_DIR = V5_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent


# ==============================================================================
# NUMERICALLY STABLE MATH FUNCTIONS
# ==============================================================================

def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def binary_nll_from_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    """
    Compute Binary Cross-Entropy / NLL directly from logit values z = log(p / (1-p)):
      loss = - [y * log(sigmoid(z)) + (1-y) * log(1 - sigmoid(z))]
           = y * softplus(-z) + (1-y) * softplus(z)
    """
    loss = labels * softplus(-logits) + (1.0 - labels) * softplus(logits)
    return float(np.mean(loss))


def binary_nll_from_probs(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    """Compute NLL from probabilities with clipping to ensure finite loss (avoid NaN)."""
    p = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    y = labels.astype(np.float64)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Brier score: mean squared error of predicted probabilities."""
    return float(np.mean((probs - labels) ** 2))


# ==============================================================================
# CALIBRATION & METRICS EVALUATION
# ==============================================================================

def compute_calibration_curve_and_errors(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Compute 10-bin confidence ECE, MCE, and reliability diagram data.
    
    Confidence ECE (Guo et al. 2017):
      confidence c_i = max(p_i, 1 - p_i) in [0.5, 1.0] (or [0, 1] standard partitioning).
      accuracy in bin = fraction of correct predictions.
      ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
      MCE = max_m |acc(B_m) - conf(B_m)|
    
    Reliability diagram data (Class probability calibration across [0, 1]):
      10 equal bins in [0, 1].
      Mean predicted probability vs empirical positive frequency.
    """
    N = len(labels)
    preds = (probs >= 0.5).astype(np.int64)
    confs = np.maximum(probs, 1.0 - probs)
    corrects = (preds == labels).astype(np.float64)

    # 1. Standard Confidence ECE (10 equal bins over [0.0, 1.0])
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        b_lower = bin_boundaries[i]
        b_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confs >= b_lower) & (confs <= b_upper)
        else:
            in_bin = (confs >= b_lower) & (confs < b_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(corrects[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)

            ece += (bin_count / N) * bin_error
            if bin_error > mce:
                mce = bin_error

    # 2. Probability Calibration Curve Data (for plotting & tabular reliability)
    prob_bins = []
    for i in range(n_bins):
        b_lower = bin_boundaries[i]
        b_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (probs >= b_lower) & (probs <= b_upper)
        else:
            in_bin = (probs >= b_lower) & (probs < b_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            mean_prob = float(np.mean(probs[in_bin]))
            true_freq = float(np.mean(labels[in_bin]))
        else:
            mean_prob = float((b_lower + b_upper) / 2.0)
            true_freq = float("nan")

        prob_bins.append({
            "bin_idx": i + 1,
            "bin_lower": round(float(b_lower), 2),
            "bin_upper": round(float(b_upper), 2),
            "count": bin_count,
            "mean_predicted_prob": round(mean_prob, 4) if not np.isnan(mean_prob) else None,
            "empirical_true_freq": round(true_freq, 4) if not np.isnan(true_freq) else None,
        })

    return float(ece), float(mce), {"prob_bins": prob_bins}


def evaluate_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
    scores: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute comprehensive classification and calibration metrics on predictions."""
    preds = (probs >= 0.5).astype(np.int64)

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    
    # Ensure raw and calibrated ROC-AUC use the same underlying continuous score
    auc_score = scores if scores is not None else probs
    auc = float(roc_auc_score(labels, auc_score))
    
    nll = binary_nll_from_probs(labels, probs)
    brier = brier_score(labels, probs)
    ece, mce, rel_data = compute_calibration_curve_and_errors(labels, probs, n_bins=n_bins)

    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(auc, 6),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "reliability_bins": rel_data["prob_bins"]
    }


# ==============================================================================
# PHASE 4: TEMPERATURE SCALING OPTIMIZATION
# ==============================================================================

def fit_temperature_scaling(
    s_fit: np.ndarray,
    y_fit: np.ndarray
) -> float:
    """
    Fit scalar temperature T > 0 on calibration-fit partition:
      P(AI) = sigmoid(s / T), where s = z_ai - z_real.
    Minimizes binary NLL.
    """
    def obj(T_val):
        T = T_val[0]
        scaled_s = s_fit / T
        return binary_nll_from_logits(y_fit, scaled_s)

    # Initial guess T=1.0, bounded [0.01, 10.0]
    res = minimize(
        obj,
        x0=[1.0],
        method="L-BFGS-B",
        bounds=[(0.01, 10.0)]
    )

    if not res.success:
        print(f"[WARNING] Temperature scaling optimization warning: {res.message}")

    fitted_T = float(res.x[0])
    return fitted_T


# ==============================================================================
# PHASE 5: PLATT SCALING OPTIMIZATION
# ==============================================================================

def fit_platt_scaling(
    s_fit: np.ndarray,
    y_fit: np.ndarray,
    l2_reg: float = 1e-5
) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters A and B on calibration-fit partition:
      P(AI) = sigmoid(A * s + B), where s = z_ai - z_real.
    Constraint: A > 0 (preserves monotonic ranking and ROC-AUC).
    Minimizes binary NLL with mild L2 regularization.
    """
    def obj(params):
        A, B = params
        cal_logits = A * s_fit + B
        nll = binary_nll_from_logits(y_fit, cal_logits)
        reg = l2_reg * (A ** 2 + B ** 2)
        return nll + reg

    # Initial guess A=1.0, B=0.0. Bound A > 0 strictly: [1e-4, 20.0], B: [-20.0, 20.0]
    res = minimize(
        obj,
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(1e-4, 20.0), (-20.0, 20.0)]
    )

    if not res.success:
        print(f"[WARNING] Platt scaling optimization warning: {res.message}")

    fitted_A = float(res.x[0])
    fitted_B = float(res.x[1])
    return fitted_A, fitted_B


# ==============================================================================
# RELIABILITY PLOT GENERATION
# ==============================================================================

def generate_reliability_plot(
    output_path: Path,
    raw_eval_metrics: Dict[str, Any],
    temp_eval_metrics: Dict[str, Any],
    platt_eval_metrics: Dict[str, Any],
    raw_eval_probs: np.ndarray,
    temp_eval_probs: np.ndarray,
    platt_eval_probs: np.ndarray,
    selected_method: str
):
    """Generate high-resolution comparative reliability diagram and confidence histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. Reliability Diagram (Left Panel)
    ax_cal = axes[0]
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

    def extract_xy(bins):
        xs = []
        ys = []
        for b in bins:
            if b["empirical_true_freq"] is not None and b["mean_predicted_prob"] is not None:
                xs.append(b["mean_predicted_prob"])
                ys.append(b["empirical_true_freq"])
        return xs, ys

    raw_x, raw_y = extract_xy(raw_eval_metrics["reliability_bins"])
    temp_x, temp_y = extract_xy(temp_eval_metrics["reliability_bins"])
    platt_x, platt_y = extract_xy(platt_eval_metrics["reliability_bins"])

    ax_cal.plot(raw_x, raw_y, "s-", color="#e11d48", label=f"Raw Baseline (ECE={raw_eval_metrics['ece']*100:.2f}%)", linewidth=1.8)
    ax_cal.plot(temp_x, temp_y, "^-", color="#0284c7", label=f"Temperature Scaling (ECE={temp_eval_metrics['ece']*100:.2f}%)", linewidth=1.8)
    ax_cal.plot(platt_x, platt_y, "o-", color="#10b981", label=f"Platt Scaling (ECE={platt_eval_metrics['ece']*100:.2f}%)", linewidth=2.2)

    ax_cal.set_title("Reliability Diagram (Calibration Curve)\nEvaluated on Held-Out Validation Split (N=4,500)", fontsize=11, fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted Probability P(AI)", fontsize=10)
    ax_cal.set_ylabel("Empirical True Fraction P(Y=AI)", fontsize=10)
    ax_cal.set_xlim([-0.02, 1.02])
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    ax_cal.legend(loc="upper left", fontsize=9)

    # 2. Probability Distribution Histogram (Right Panel)
    ax_hist = axes[1]
    bins = np.linspace(0.0, 1.0, 25)
    ax_hist.hist(raw_eval_probs, bins=bins, alpha=0.35, color="#e11d48", label="Raw")
    ax_hist.hist(temp_eval_probs, bins=bins, alpha=0.35, color="#0284c7", label="Temperature")
    ax_hist.hist(platt_eval_probs, bins=bins, alpha=0.35, color="#10b981", label="Platt")

    ax_hist.set_title(f"Predicted Probability Distribution\nSelected Active Method: {selected_method.upper()}", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Predicted Probability P(AI)", fontsize=10)
    ax_hist.set_ylabel("Sample Count", fontsize=10)
    ax_hist.set_yscale("log")
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    ax_hist.legend(loc="upper center", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] Reliability diagram saved to: {output_path}")


# ==============================================================================
# MAIN CALIBRATION PIPELINE
# ==============================================================================

def run_calibration_pipeline(
    input_logits_path: Path,
    output_dir: Path,
    models_dir: Path,
    seed: int = 42
):
    print("=" * 80)
    print("AIDetect V5-D — Post-Hoc Probability Calibration Fitting")
    print("=" * 80)

    # 1. Check input file
    if not input_logits_path.is_file():
        raise FileNotFoundError(
            f"Validation logits file not found at: {input_logits_path}\n"
            f"Please run backend/v5/extract_v5_d_val_logits.py first."
        )

    print(f"[DATA] Loading validation logits from: {input_logits_path}")
    data = np.load(input_logits_path)

    sample_ids = data["sample_id"]
    labels = data["label"]
    z_real = data["z_real"]
    z_ai = data["z_ai"]
    raw_ai_prob = data["raw_ai_probability"]
    raw_real_prob = data["raw_real_probability"]

    N_total = len(labels)
    n_real = int(np.sum(labels == 0))
    n_ai = int(np.sum(labels == 1))

    print(f"[DATA] Extracted samples: {N_total:,} (REAL: {n_real:,}, AI: {n_ai:,})")
    if N_total != 9000 or n_real != 4500 or n_ai != 4500:
        raise ValueError(f"Expected exactly 9,000 samples (4,500 REAL, 4,500 AI), got {N_total}.")

    # Compute logit difference s = z_ai - z_real
    s_all = z_ai - z_real

    # ==========================================================================
    # PHASE 2: DETERMINISTIC CALIBRATION SPLIT
    # ==========================================================================
    print(f"\n[PHASE 2] Deterministic Stratified Split (Seed={seed})...")
    rng = np.random.RandomState(seed)

    real_indices = np.where(labels == 0)[0]
    ai_indices = np.where(labels == 1)[0]

    # Deterministic shuffle within classes
    shuffled_real = rng.permutation(real_indices)
    shuffled_ai = rng.permutation(ai_indices)

    # 50/50 split: exactly 2250 REAL and 2250 AI per partition
    fit_real_idx = shuffled_real[:2250]
    eval_real_idx = shuffled_real[2250:]

    fit_ai_idx = shuffled_ai[:2250]
    eval_ai_idx = shuffled_ai[2250:]

    fit_indices = np.sort(np.concatenate([fit_real_idx, fit_ai_idx]))
    eval_indices = np.sort(np.concatenate([eval_real_idx, eval_ai_idx]))

    # Strict partition audit
    assert len(fit_indices) == 4500, f"Fit partition size: {len(fit_indices)}"
    assert len(eval_indices) == 4500, f"Eval partition size: {len(eval_indices)}"
    assert len(np.intersect1d(fit_indices, eval_indices)) == 0, "Overlap detected between fit and eval partitions!"
    assert np.sum(labels[fit_indices] == 0) == 2250, "Fit REAL count != 2250"
    assert np.sum(labels[fit_indices] == 1) == 2250, "Fit AI count != 2250"
    assert np.sum(labels[eval_indices] == 0) == 2250, "Eval REAL count != 2250"
    assert np.sum(labels[eval_indices] == 1) == 2250, "Eval AI count != 2250"

    print("  Calibration-Fit subset:        4,500 images (2,250 REAL, 2,250 AI) — USED TO FIT T, A, B")
    print("  Calibration-Evaluation subset: 4,500 images (2,250 REAL, 2,250 AI) — HELD OUT FOR EVALUATION")

    # Partition data
    s_fit = s_all[fit_indices]
    y_fit = labels[fit_indices]

    s_eval = s_all[eval_indices]
    y_eval = labels[eval_indices]
    raw_ai_eval = raw_ai_prob[eval_indices]

    # ==========================================================================
    # PHASE 3: RAW BASELINE EVALUATION
    # ==========================================================================
    print("\n[PHASE 3] Evaluating Raw Baseline on Calibration-Evaluation Subset (N=4,500)...")
    raw_eval_scores = s_eval
    raw_eval_metrics = evaluate_predictions(y_eval, raw_ai_eval, scores=raw_eval_scores)
    print(f"  Raw NLL:      {raw_eval_metrics['nll']:.6f}")
    print(f"  Raw Brier:    {raw_eval_metrics['brier']:.6f}")
    print(f"  Raw ECE:      {raw_eval_metrics['ece']*100:.2f}%")
    print(f"  Raw MCE:      {raw_eval_metrics['mce']*100:.2f}%")
    print(f"  Raw Accuracy: {raw_eval_metrics['accuracy']*100:.2f}%")
    print(f"  Raw F1:       {raw_eval_metrics['f1']*100:.2f}%")
    print(f"  Raw ROC-AUC:  {raw_eval_metrics['roc_auc']:.6f}")

    # ==========================================================================
    # PHASE 4: TEMPERATURE SCALING
    # ==========================================================================
    print("\n[PHASE 4] Fitting Temperature Scaling on Calibration-Fit Subset (N=4,500)...")
    fitted_T = fit_temperature_scaling(s_fit, y_fit)
    print(f"  Fitted Temperature T: {fitted_T:.6f}")

    # Evaluate on held-out evaluation subset using logit / T
    temp_eval_scores = s_eval / fitted_T
    temp_eval_probs = expit(temp_eval_scores)
    temp_eval_metrics = evaluate_predictions(y_eval, temp_eval_probs, scores=temp_eval_scores)
    print(f"  Temperature Eval NLL:      {temp_eval_metrics['nll']:.6f}")
    print(f"  Temperature Eval Brier:    {temp_eval_metrics['brier']:.6f}")
    print(f"  Temperature Eval ECE:      {temp_eval_metrics['ece']*100:.2f}%")
    print(f"  Temperature Eval MCE:      {temp_eval_metrics['mce']*100:.2f}%")
    print(f"  Temperature Eval Accuracy: {temp_eval_metrics['accuracy']*100:.2f}%")
    print(f"  Temperature Eval F1:       {temp_eval_metrics['f1']*100:.2f}%")
    print(f"  Temperature Eval ROC-AUC:  {temp_eval_metrics['roc_auc']:.6f}")

    # Invariant check: monotonic scalar preserves ROC-AUC
    auc_diff_temp = abs(temp_eval_metrics['roc_auc'] - raw_eval_metrics['roc_auc'])
    print(f"  [INVARIANT] Temperature ROC-AUC delta: {auc_diff_temp:.2e} (Strictly Preserved)")

    # ==========================================================================
    # PHASE 5: PLATT SCALING
    # ==========================================================================
    print("\n[PHASE 5] Fitting Platt Scaling on Calibration-Fit Subset (N=4,500)...")
    fitted_A, fitted_B = fit_platt_scaling(s_fit, y_fit)
    print(f"  Fitted Platt Parameters: A = {fitted_A:.6f}, B = {fitted_B:.6f}")
    assert fitted_A > 0, f"Platt constraint violation: A must be > 0, got {fitted_A}"

    # Evaluate on held-out evaluation subset using A * logit + B
    platt_eval_scores = fitted_A * s_eval + fitted_B
    platt_eval_probs = expit(platt_eval_scores)
    platt_eval_metrics = evaluate_predictions(y_eval, platt_eval_probs, scores=platt_eval_scores)
    print(f"  Platt Eval NLL:      {platt_eval_metrics['nll']:.6f}")
    print(f"  Platt Eval Brier:    {platt_eval_metrics['brier']:.6f}")
    print(f"  Platt Eval ECE:      {platt_eval_metrics['ece']*100:.2f}%")
    print(f"  Platt Eval MCE:      {platt_eval_metrics['mce']*100:.2f}%")
    print(f"  Platt Eval Accuracy: {platt_eval_metrics['accuracy']*100:.2f}%")
    print(f"  Platt Eval F1:       {platt_eval_metrics['f1']*100:.2f}%")
    print(f"  Platt Eval ROC-AUC:  {platt_eval_metrics['roc_auc']:.6f}")

    # Invariant check: A > 0 preserves ROC-AUC
    auc_diff_platt = abs(platt_eval_metrics['roc_auc'] - raw_eval_metrics['roc_auc'])
    print(f"  [INVARIANT] Platt ROC-AUC delta: {auc_diff_platt:.2e} (Strictly Preserved)")

    # ==========================================================================
    # PHASE 6: SELECT CALIBRATION METHOD
    # ==========================================================================
    print("\n[PHASE 6] Model Selection Decision Engine...")
    print("  Comparison Summary (Evaluation Split N=4,500):")
    print(f"    Raw:         NLL={raw_eval_metrics['nll']:.6f}, Brier={raw_eval_metrics['brier']:.6f}, ECE={raw_eval_metrics['ece']*100:.2f}%")
    print(f"    Temperature: NLL={temp_eval_metrics['nll']:.6f}, Brier={temp_eval_metrics['brier']:.6f}, ECE={temp_eval_metrics['ece']*100:.2f}%")
    print(f"    Platt:       NLL={platt_eval_metrics['nll']:.6f}, Brier={platt_eval_metrics['brier']:.6f}, ECE={platt_eval_metrics['ece']*100:.2f}%")

    # Primary selection rule: NLL / calibration quality
    # If Platt NLL is lower than Temperature by more than 1e-4, select Platt; otherwise prefer simpler Temperature
    nll_diff = temp_eval_metrics['nll'] - platt_eval_metrics['nll']
    if nll_diff > 1e-4:
        selected_method = "platt"
        selection_reason = (
            f"Platt scaling achieved lowest evaluation NLL ({platt_eval_metrics['nll']:.6f} vs "
            f"Temperature {temp_eval_metrics['nll']:.6f}) with superior ECE ({platt_eval_metrics['ece']*100:.2f}%)."
        )
    elif platt_eval_metrics['ece'] < temp_eval_metrics['ece'] - 0.002:
        selected_method = "platt"
        selection_reason = (
            f"Platt scaling achieved superior calibration quality (ECE {platt_eval_metrics['ece']*100:.2f}% vs "
            f"Temperature {temp_eval_metrics['ece']*100:.2f}%)."
        )
    else:
        selected_method = "temperature"
        selection_reason = (
            f"Temperature scaling achieved equivalent calibration performance to Platt scaling with a single parameter."
        )

    print(f"  ==> Selected Calibration Method: {selected_method.upper()}")
    print(f"      Rationale: {selection_reason}")

    # Build calibration config JSON
    calibration_config = {
        "version": "Platt_Scaling_v1" if selected_method == "platt" else "Temperature_Scaling_v1",
        "model": "V5-D Gated Residual",
        "fit_split": "validation_calibration_fit",
        "evaluation_split": "validation_calibration_eval",
        "seed": seed,
        "n_fit": len(fit_indices),
        "n_eval": len(eval_indices),
        "method": selected_method,
        "A": fitted_A,
        "B": fitted_B,
        "temperature_baseline": {
            "T": fitted_T
        },
        "selection_reason": selection_reason,
        "metrics": {
            "raw": {k: v for k, v in raw_eval_metrics.items() if k != "reliability_bins"},
            "temperature": {k: v for k, v in temp_eval_metrics.items() if k != "reliability_bins"},
            "platt": {k: v for k, v in platt_eval_metrics.items() if k != "reliability_bins"}
        }
    }

    # Save to backend/models/v5/v5_d_calibration.json
    models_dir.mkdir(parents=True, exist_ok=True)
    cal_json_path = models_dir / "v5_d_calibration.json"
    with open(cal_json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_config, f, indent=2)
    print(f"\n[OUTPUT] Saved production calibration config to: {cal_json_path}")

    # ==========================================================================
    # PHASE 10: RESEARCH ARTIFACTS & OUTPUTS
    # ==========================================================================
    print("\n[PHASE 10] Generating Research Artifacts...")
    cal_results_dir = output_dir / "calibration"
    cal_results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full metrics JSON
    full_metrics_path = cal_results_dir / "v5_d_calibration_metrics.json"
    full_metrics = {
        "dataset_audit": {
            "total_validation_samples": N_total,
            "real_samples": n_real,
            "ai_samples": n_ai,
            "fit_samples": len(fit_indices),
            "eval_samples": len(eval_indices)
        },
        "fitted_parameters": {
            "temperature_T": fitted_T,
            "platt_A": fitted_A,
            "platt_B": fitted_B
        },
        "selected_method": selected_method,
        "evaluation_metrics": {
            "raw": raw_eval_metrics,
            "temperature": temp_eval_metrics,
            "platt": platt_eval_metrics
        }
    }
    with open(full_metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  [1/4] Saved detailed metrics JSON to: {full_metrics_path}")

    # 2. Comparison CSV
    comparison_csv_path = cal_results_dir / "v5_d_calibration_comparison.csv"
    csv_rows = [
        "Method,NLL,Brier,ECE,MCE,Accuracy,F1,ROC-AUC",
        f"RAW,{raw_eval_metrics['nll']},{raw_eval_metrics['brier']},{raw_eval_metrics['ece']},{raw_eval_metrics['mce']},{raw_eval_metrics['accuracy']},{raw_eval_metrics['f1']},{raw_eval_metrics['roc_auc']}",
        f"TEMPERATURE,{temp_eval_metrics['nll']},{temp_eval_metrics['brier']},{temp_eval_metrics['ece']},{temp_eval_metrics['mce']},{temp_eval_metrics['accuracy']},{temp_eval_metrics['f1']},{temp_eval_metrics['roc_auc']}",
        f"PLATT,{platt_eval_metrics['nll']},{platt_eval_metrics['brier']},{platt_eval_metrics['ece']},{platt_eval_metrics['mce']},{platt_eval_metrics['accuracy']},{platt_eval_metrics['f1']},{platt_eval_metrics['roc_auc']}"
    ]
    with open(comparison_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"  [2/4] Saved comparison CSV to: {comparison_csv_path}")

    # 3. Reliability Data CSV (10 bins)
    rel_csv_path = cal_results_dir / "v5_d_reliability_data.csv"
    rel_headers = [
        "bin_idx", "bin_lower", "bin_upper", "sample_count",
        "raw_mean_prob", "raw_true_freq",
        "temp_mean_prob", "temp_true_freq",
        "platt_mean_prob", "platt_true_freq"
    ]
    rel_rows = [",".join(rel_headers)]
    for i in range(10):
        rb = raw_eval_metrics["reliability_bins"][i]
        tb = temp_eval_metrics["reliability_bins"][i]
        pb = platt_eval_metrics["reliability_bins"][i]
        row_str = (
            f"{rb['bin_idx']},{rb['bin_lower']},{rb['bin_upper']},{rb['count']},"
            f"{rb['mean_predicted_prob']},{rb['empirical_true_freq']},"
            f"{tb['mean_predicted_prob']},{tb['empirical_true_freq']},"
            f"{pb['mean_predicted_prob']},{pb['empirical_true_freq']}"
        )
        rel_rows.append(row_str)
    with open(rel_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rel_rows) + "\n")
    print(f"  [3/4] Saved reliability data CSV to: {rel_csv_path}")

    # 4. Reliability Diagram PNG
    diagram_png_path = cal_results_dir / "v5_d_reliability_diagram.png"
    generate_reliability_plot(
        diagram_png_path,
        raw_eval_metrics,
        temp_eval_metrics,
        platt_eval_metrics,
        raw_ai_eval,
        temp_eval_probs,
        platt_eval_probs,
        selected_method
    )
    print(f"  [4/4] Saved reliability diagram PNG to: {diagram_png_path}")

    print("=" * 80)
    print("Calibration Pipeline Execution Complete.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and Evaluate V5-D Post-Hoc Probability Calibration")
    parser.add_argument(
        "--input-logits",
        type=str,
        default=str(BACKEND_DIR / "evaluation_results" / "v5" / "v5_d_val_logits.npz"),
        help="Path to extracted validation logits .npz"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BACKEND_DIR / "evaluation_results" / "v5"),
        help="Directory to save research outputs"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(BACKEND_DIR / "models" / "v5"),
        help="Directory to save v5_d_calibration.json"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed (default: 42)")

    args = parser.parse_args()

    run_calibration_pipeline(
        input_logits_path=Path(args.input_logits),
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        seed=args.seed
    )

