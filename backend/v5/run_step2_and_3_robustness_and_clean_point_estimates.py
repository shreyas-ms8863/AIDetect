"""
AIDetect — Step 2 & Step 3: Robustness Analysis & Clean-Test Point Estimates
=============================================================================
Computes:
1. Step 2 (Robustness Analysis):
   - Balanced Accuracy
   - MCC
   - F1 degradation from clean (both absolute ΔF1 and % degradation)
   - F1 retention from clean (both ratio and %)
   For all 12 conditions across 4 perturbation families (Resize, Blur, Noise, JPEG) x 4 models (48 rows).
   Saves to:
     - evaluation_results/v5/strengthened_evaluation/robustness_extended.csv
     - evaluation_results/v5/strengthened_evaluation/robustness_extended.json

2. Step 3 (Clean-Test Point Estimates):
   - Derives exact Balanced Accuracy and MCC from v5_final_test_comparison.json (N=9,000)
   - Merges with existing accuracy, precision, recall, f1, TP, TN, FP, FN
   Saves to:
     - evaluation_results/v5/strengthened_evaluation/clean_metrics_extended.json
"""

import json
import math
from pathlib import Path
from typing import Dict, Any

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "backend" / "evaluation_results" / "v5"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "v5" / "strengthened_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEY_MAP = {
    "spatial": "V5-A Spatial",
    "frequency": "V5-B Frequency",
    "hybrid": "V5-C Hybrid",
    "gated_residual": "V5-D Gated Residual"
}

ROBUSTNESS_FILES = [
    ("Resize", INPUT_DIR / "v5_resize_robustness.json", "scale_pct"),
    ("Gaussian Blur", INPUT_DIR / "v5_blur_robustness.json", "radius"),
    ("Gaussian Noise", INPUT_DIR / "v5_noise_robustness.json", "sigma"),
    ("JPEG Re-encoding", INPUT_DIR / "v5_reencoding_robustness.json", "quality")
]


def compute_mcc_from_cm(tp: int, tn: int, fp: int, fn: int) -> float:
    num = float(tp * tn - fp * fn)
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if den == 0 else num / den


def compute_balanced_accuracy_from_cm(tp: int, tn: int, fp: int, fn: int) -> float:
    pos = tp + fn
    neg = tn + fp
    tpr = tp / pos if pos > 0 else 0.0
    tnr = tn / neg if neg > 0 else 0.0
    return (tpr + tnr) / 2.0


def run_step2_and_3():
    print("=" * 80)
    print("AIDETECT — STEP 2 & STEP 3: ROBUSTNESS ANALYSIS & CLEAN POINT ESTIMATES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 3: CLEAN-TEST POINT ESTIMATES
    # -------------------------------------------------------------------------
    clean_json_path = INPUT_DIR / "v5_final_test_comparison.json"
    if not clean_json_path.is_file():
        raise FileNotFoundError(f"Missing clean test file: {clean_json_path}")

    with open(clean_json_path, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    clean_metrics_extended = {
        "dataset": "V5 Frozen Test Set (N=9,000)",
        "sample_size": 9000,
        "classes": {"REAL": 4500, "AI": 4500},
        "threshold": 0.50,
        "models": {}
    }

    clean_f1_lookup: Dict[str, float] = {}

    for model_name, m_info in clean_data["models"].items():
        tm = m_info["test_metrics"]
        tp = tm["true_positive"]
        tn = tm["true_negative"]
        fp = tm["false_positive"]
        fn = tm["false_negative"]
        acc = tm["accuracy"]
        prec = tm["precision"]
        rec = tm["recall"]
        f1 = tm["f1"]

        bal_acc = compute_balanced_accuracy_from_cm(tp, tn, fp, fn)
        mcc = compute_mcc_from_cm(tp, tn, fp, fn)

        clean_metrics_extended["models"][model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "total": tm["total"]
        }
        clean_f1_lookup[model_name] = f1

    # Save Step 3 artifact
    step3_out_path = OUTPUT_DIR / "clean_metrics_extended.json"
    with open(step3_out_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics_extended, f, indent=2)
    print(f"[*] Step 3: Saved clean point estimates to: {step3_out_path}")

    # -------------------------------------------------------------------------
    # STEP 2: ROBUSTNESS ANALYSIS (12 CONDITIONS x 4 MODELS)
    # -------------------------------------------------------------------------
    robustness_rows = []
    robustness_dict_by_perturbation = {}

    for p_name, p_file, p_param in ROBUSTNESS_FILES:
        if not p_file.is_file():
            raise FileNotFoundError(f"Missing robustness file: {p_file}")

        with open(p_file, "r", encoding="utf-8") as f:
            p_data = json.load(f)

        robustness_dict_by_perturbation[p_name] = {}

        for m_key, m_name in MODEL_KEY_MAP.items():
            model_conds = p_data["models"][m_key]
            clean_f1 = clean_f1_lookup[m_name]

            robustness_dict_by_perturbation[p_name][m_name] = {}

            for cond_val, c_metrics in model_conds.items():
                tp = c_metrics["true_positive"]
                tn = c_metrics["true_negative"]
                fp = c_metrics["false_positive"]
                fn = c_metrics["false_negative"]
                acc = c_metrics["accuracy"]
                prec = c_metrics["precision"]
                rec = c_metrics["recall"]
                f1 = c_metrics["f1"]

                bal_acc = compute_balanced_accuracy_from_cm(tp, tn, fp, fn)
                mcc = compute_mcc_from_cm(tp, tn, fp, fn)

                # Degradation & Retention
                f1_deg_abs = clean_f1 - f1
                f1_deg_pct = (f1_deg_abs / clean_f1) * 100.0 if clean_f1 > 0 else 0.0
                f1_ret_ratio = f1 / clean_f1 if clean_f1 > 0 else 0.0
                f1_ret_pct = f1_ret_ratio * 100.0

                row = {
                    "perturbation_family": p_name,
                    "parameter": p_param,
                    "condition_value": cond_val,
                    "model": m_name,
                    "clean_f1": round(clean_f1, 6),
                    "perturbed_f1": round(f1, 6),
                    "f1_degradation_abs": round(f1_deg_abs, 6),
                    "f1_degradation_pct": round(f1_deg_pct, 4),
                    "f1_retention_ratio": round(f1_ret_ratio, 6),
                    "f1_retention_pct": round(f1_ret_pct, 4),
                    "accuracy": round(acc, 6),
                    "balanced_accuracy": round(bal_acc, 6),
                    "precision": round(prec, 6),
                    "recall": round(rec, 6),
                    "mcc": round(mcc, 6),
                    "true_positive": tp,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "total": c_metrics["total"]
                }
                robustness_rows.append(row)

                robustness_dict_by_perturbation[p_name][m_name][cond_val] = row

    # Save CSV
    import pandas as pd
    df_rob = pd.DataFrame(robustness_rows)
    step2_csv_path = OUTPUT_DIR / "robustness_extended.csv"
    df_rob.to_csv(step2_csv_path, index=False)
    print(f"[*] Step 2: Saved extended robustness CSV (48 rows) to: {step2_csv_path}")

    # Save JSON
    step2_json_path = OUTPUT_DIR / "robustness_extended.json"
    with open(step2_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "V5 Robustness Extended Analysis",
            "total_rows": len(robustness_rows),
            "perturbation_families": [p[0] for p in ROBUSTNESS_FILES],
            "models": list(MODEL_KEY_MAP.values()),
            "by_perturbation": robustness_dict_by_perturbation
        }, f, indent=2)
    print(f"[*] Step 2: Saved extended robustness JSON to: {step2_json_path}")

    print("=" * 80)
    print("STEP 2 & STEP 3 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
     run_step2_and_3()
