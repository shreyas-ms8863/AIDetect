"""
AIDetect - Frozen External Benchmark Post-Inference Analysis
============================================================
Performs a comprehensive post-inference error, agreement, confidence,
and cross-dataset generalization analysis of the 800-image Defactify
benchmark evaluation results without modifying any models, thresholds,
or checkpoints.

Reads:
  - diagnostic_outputs/external_benchmark_v1_results/external_benchmark_v1_per_image.csv
  - diagnostic_outputs/external_benchmark_v1_results/external_benchmark_v1_summary.json

Outputs into:
  - diagnostic_outputs/external_benchmark_v1_analysis/
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_results"
ANALYSIS_DIR = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

PER_IMAGE_CSV = RESULTS_DIR / "external_benchmark_v1_per_image.csv"
SUMMARY_JSON = RESULTS_DIR / "external_benchmark_v1_summary.json"

MODELS = [
    "V5-A Spatial",
    "V5-B Frequency",
    "V5-C Hybrid",
    "V5-D Gated Residual"
]

GENERATOR_NAMES = {
    "Stable_Diffusion_3": "SD3",
    "Midjourney_v6": "Midjourney v6",
    "DALL-E_3": "DALL-E 3",
    "Stable_Diffusion_XL": "SDXL",
    "None_Authentic_Photograph": "REAL (Natural Photos)"
}

AI_GENERATOR_KEYS = [
    "Stable_Diffusion_3",
    "Midjourney_v6",
    "DALL-E_3",
    "Stable_Diffusion_XL"
]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not PER_IMAGE_CSV.is_file():
        raise FileNotFoundError(f"Missing per-image results: {PER_IMAGE_CSV}")
    
    df = pd.read_csv(PER_IMAGE_CSV)
    
    # Metadata for unique 800 images
    meta = df[["candidate_id", "original_filename", "true_label", "category", "source", "generator"]].drop_duplicates(subset=["candidate_id"]).set_index("candidate_id")
    
    piv_pred = df.pivot(index="candidate_id", columns="model", values="prediction")
    piv_prob = df.pivot(index="candidate_id", columns="model", values="calibrated_ai_probability")
    piv_corr = df.pivot(index="candidate_id", columns="model", values="correct")
    
    return df, meta, piv_pred, piv_prob, piv_corr


def analyze_model_agreement(meta: pd.DataFrame, piv_pred: pd.DataFrame, piv_corr: pd.DataFrame, piv_prob: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    num_ai = (piv_pred == "AI").sum(axis=1)
    
    agree_4 = int(((num_ai == 4) | (num_ai == 0)).sum())
    agree_3 = int(((num_ai == 3) | (num_ai == 1)).sum())
    agree_2 = int((num_ai == 2).sum())
    
    total = len(piv_pred)
    
    # Pairwise agreement
    pairwise_data = []
    matrix_df = pd.DataFrame(index=MODELS, columns=MODELS)
    for m1 in MODELS:
        for m2 in MODELS:
            eq = int((piv_pred[m1] == piv_pred[m2]).sum())
            pct = round(eq / total * 100.0, 2)
            matrix_df.loc[m1, m2] = f"{eq} ({pct:.1f}%)"
            if m1 < m2:
                pairwise_data.append({
                    "model_1": m1,
                    "model_2": m2,
                    "agreement_count": eq,
                    "agreement_pct": pct
                })
    
    # Agreement breakdown on REAL vs AI
    real_ids = meta[meta["true_label"] == "REAL"].index
    ai_ids = meta[meta["true_label"] == "AI"].index
    
    num_ai_real = num_ai.loc[real_ids]
    num_ai_ai = num_ai.loc[ai_ids]
    
    real_agree_4 = int(((num_ai_real == 0) | (num_ai_real == 4)).sum())
    real_agree_all_real = int((num_ai_real == 0).sum())
    real_agree_all_ai = int((num_ai_real == 4).sum())
    
    ai_agree_4 = int(((num_ai_ai == 0) | (num_ai_ai == 4)).sum())
    ai_agree_all_ai = int((num_ai_ai == 4).sum())
    ai_agree_all_real = int((num_ai_ai == 0).sum())
    
    summary = {
        "total_images": total,
        "agreement_distribution": {
            "all_4_agree_count": agree_4,
            "all_4_agree_pct": round(agree_4 / total * 100.0, 2),
            "three_of_four_agree_count": agree_3,
            "three_of_four_agree_pct": round(agree_3 / total * 100.0, 2),
            "two_vs_two_split_count": agree_2,
            "two_vs_two_split_pct": round(agree_2 / total * 100.0, 2),
        },
        "real_images_agreement": {
            "N": len(real_ids),
            "all_4_agree_count": real_agree_4,
            "all_4_unanimous_REAL_correct": real_agree_all_real,
            "all_4_unanimous_AI_false_alarm": real_agree_all_ai,
        },
        "ai_images_agreement": {
            "N": len(ai_ids),
            "all_4_agree_count": ai_agree_4,
            "all_4_unanimous_AI_correct": ai_agree_all_ai,
            "all_4_unanimous_REAL_missed": ai_agree_all_real,
        },
        "pairwise_matrix": matrix_df.to_dict()
    }
    
    # Flat dataframe for model_agreement.csv
    agreement_rows = [
        {"metric_category": "Overall Agreement", "item": "All 4 Models Agree (Unanimous)", "count": agree_4, "percentage": round(agree_4 / total * 100.0, 2)},
        {"metric_category": "Overall Agreement", "item": "3 Models Agree (1 Dissenter)", "count": agree_3, "percentage": round(agree_3 / total * 100.0, 2)},
        {"metric_category": "Overall Agreement", "item": "2 vs 2 Split Decision", "count": agree_2, "percentage": round(agree_2 / total * 100.0, 2)},
        {"metric_category": "REAL Images (N=400)", "item": "All 4 Models Agree Unanimously", "count": real_agree_4, "percentage": round(real_agree_4 / 400.0 * 100.0, 2)},
        {"metric_category": "REAL Images (N=400)", "item": "All 4 Correctly Say REAL", "count": real_agree_all_real, "percentage": round(real_agree_all_real / 400.0 * 100.0, 2)},
        {"metric_category": "REAL Images (N=400)", "item": "All 4 Wrongly Say AI (Shared False Alarm)", "count": real_agree_all_ai, "percentage": round(real_agree_all_ai / 400.0 * 100.0, 2)},
        {"metric_category": "AI Images (N=400)", "item": "All 4 Models Agree Unanimously", "count": ai_agree_4, "percentage": round(ai_agree_4 / 400.0 * 100.0, 2)},
        {"metric_category": "AI Images (N=400)", "item": "All 4 Correctly Say AI", "count": ai_agree_all_ai, "percentage": round(ai_agree_all_ai / 400.0 * 100.0, 2)},
        {"metric_category": "AI Images (N=400)", "item": "All 4 Wrongly Say REAL (Shared Miss)", "count": ai_agree_all_real, "percentage": round(ai_agree_all_real / 400.0 * 100.0, 2)},
    ]
    for p in pairwise_data:
        agreement_rows.append({
            "metric_category": "Pairwise Agreement",
            "item": f"{p['model_1']} vs {p['model_2']}",
            "count": p["agreement_count"],
            "percentage": p["agreement_pct"]
        })
        
    df_agreement = pd.DataFrame(agreement_rows)
    return summary, df_agreement


def analyze_real_errors(meta: pd.DataFrame, piv_pred: pd.DataFrame, piv_corr: pd.DataFrame, piv_prob: pd.DataFrame, df_raw: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.DataFrame]]:
    real_ids = meta[meta["true_label"] == "REAL"].index
    sub_corr = piv_corr.loc[real_ids]
    sub_prob = piv_prob.loc[real_ids]
    
    N_real = len(real_ids)
    
    # 1. Per-model performance on REAL
    per_model_real = []
    for m in MODELS:
        corr_cnt = int(sub_corr[m].sum())
        fp_cnt = N_real - corr_cnt
        fpr = fp_cnt / N_real
        probs = sub_prob[m].values
        mean_p = float(np.mean(probs))
        med_p = float(np.median(probs))
        ge_50 = float(np.mean(probs >= 0.50) * 100.0)
        ge_70 = float(np.mean(probs >= 0.70) * 100.0)
        ge_90 = float(np.mean(probs >= 0.90) * 100.0)
        
        per_model_real.append({
            "model": m,
            "correct_real": corr_cnt,
            "false_positives": fp_cnt,
            "fpr": round(fpr, 4),
            "mean_ai_prob": round(mean_p, 4),
            "median_ai_prob": round(med_p, 4),
            "pct_ge_50": round(ge_50, 2),
            "pct_ge_70": round(ge_70, 2),
            "pct_ge_90": round(ge_90, 2)
        })
    
    # 2. REAL grouping
    all_4_corr = int((sub_corr.sum(axis=1) == 4).sum())
    a_only_wrong = int(((sub_corr["V5-A Spatial"] == False) & (sub_corr["V5-B Frequency"] == True) & (sub_corr["V5-C Hybrid"] == True) & (sub_corr["V5-D Gated Residual"] == True)).sum())
    b_only_wrong = int(((sub_corr["V5-A Spatial"] == True) & (sub_corr["V5-B Frequency"] == False) & (sub_corr["V5-C Hybrid"] == True) & (sub_corr["V5-D Gated Residual"] == True)).sum())
    c_only_wrong = int(((sub_corr["V5-A Spatial"] == True) & (sub_corr["V5-B Frequency"] == True) & (sub_corr["V5-C Hybrid"] == False) & (sub_corr["V5-D Gated Residual"] == True)).sum())
    d_only_wrong = int(((sub_corr["V5-A Spatial"] == True) & (sub_corr["V5-B Frequency"] == True) & (sub_corr["V5-C Hybrid"] == True) & (sub_corr["V5-D Gated Residual"] == False)).sum())
    
    ad_both_wrong = int(((sub_corr["V5-A Spatial"] == False) & (sub_corr["V5-D Gated Residual"] == False)).sum())
    ad_only_wrong = int(((sub_corr["V5-A Spatial"] == False) & (sub_corr["V5-D Gated Residual"] == False) & (sub_corr["V5-B Frequency"] == True) & (sub_corr["V5-C Hybrid"] == True)).sum())
    
    multi_wrong = int((sub_corr.sum(axis=1) <= 2).sum())
    
    groupings = [
        {"error_group": "All 4 Models Correct (No Error)", "count": all_4_corr, "percentage": round(all_4_corr / N_real * 100.0, 2)},
        {"error_group": "V5-A Spatial Only Wrong", "count": a_only_wrong, "percentage": round(a_only_wrong / N_real * 100.0, 2)},
        {"error_group": "V5-B Frequency Only Wrong", "count": b_only_wrong, "percentage": round(b_only_wrong / N_real * 100.0, 2)},
        {"error_group": "V5-C Hybrid Only Wrong", "count": c_only_wrong, "percentage": round(c_only_wrong / N_real * 100.0, 2)},
        {"error_group": "V5-D Gated Residual Only Wrong", "count": d_only_wrong, "percentage": round(d_only_wrong / N_real * 100.0, 2)},
        {"error_group": "Both V5-A and V5-D Wrong (Total)", "count": ad_both_wrong, "percentage": round(ad_both_wrong / N_real * 100.0, 2)},
        {"error_group": "Exclusively V5-A and V5-D Wrong (B & C Correct)", "count": ad_only_wrong, "percentage": round(ad_only_wrong / N_real * 100.0, 2)},
        {"error_group": "Multiple-Model Errors (>=2 Models Wrong)", "count": multi_wrong, "percentage": round(multi_wrong / N_real * 100.0, 2)},
    ]
    
    # 3. Top 20 highest-confidence REAL false positives per model
    top_20_dict = {}
    for m in MODELS:
        m_df = df_raw[(df_raw["model"] == m) & (df_raw["true_label"] == "REAL") & (df_raw["prediction"] == "AI")].copy()
        m_df = m_df.sort_values(by="calibrated_ai_probability", ascending=False).head(20)
        top_20_dict[m] = m_df[[
            "candidate_id", "original_filename", "category", "source",
            "calibrated_ai_probability", "raw_ai_probability"
        ]].reset_index(drop=True)
    
    summary = {
        "per_model_real": per_model_real,
        "groupings": groupings,
    }
    
    # Combined real_error_analysis.csv
    rows_csv = []
    for pm in per_model_real:
        rows_csv.append({
            "section": "Model REAL Performance",
            "identifier": pm["model"],
            "stat_1_name": "Correct REAL (TN)", "stat_1_val": pm["correct_real"],
            "stat_2_name": "False Positives (FP)", "stat_2_val": pm["false_positives"],
            "stat_3_name": "FPR", "stat_3_val": pm["fpr"],
            "stat_4_name": "Mean AI Prob", "stat_4_val": pm["mean_ai_prob"],
            "stat_5_name": "Median AI Prob", "stat_5_val": pm["median_ai_prob"],
            "stat_6_name": "% AI Prob >= 50%", "stat_6_val": pm["pct_ge_50"],
            "stat_7_name": "% AI Prob >= 90%", "stat_7_val": pm["pct_ge_90"],
        })
    for gr in groupings:
        rows_csv.append({
            "section": "REAL Error Groupings",
            "identifier": gr["error_group"],
            "stat_1_name": "Count", "stat_1_val": gr["count"],
            "stat_2_name": "Percentage", "stat_2_val": gr["percentage"],
            "stat_3_name": "", "stat_3_val": "",
            "stat_4_name": "", "stat_4_val": "",
            "stat_5_name": "", "stat_5_val": "",
            "stat_6_name": "", "stat_6_val": "",
            "stat_7_name": "", "stat_7_val": "",
        })
        
    df_real_error = pd.DataFrame(rows_csv)
    return summary, df_real_error, top_20_dict


def analyze_ai_generators(meta: pd.DataFrame, piv_pred: pd.DataFrame, piv_corr: pd.DataFrame, piv_prob: pd.DataFrame, df_raw: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_rows = []
    recall_dict = {m: {} for m in MODELS}
    fnr_dict = {m: {} for m in MODELS}
    
    hardest_easiest = {}
    
    for m in MODELS:
        sub_m = df_raw[df_raw["model"] == m]
        gen_recalls = {}
        
        for g_raw in AI_GENERATOR_KEYS:
            g_disp = GENERATOR_NAMES[g_raw]
            sub_g = sub_m[sub_m["generator"] == g_raw]
            N_g = len(sub_g)
            
            tp = int((sub_g["prediction"] == "AI").sum())
            fn = int((sub_g["prediction"] == "REAL").sum())
            rec = tp / N_g if N_g > 0 else 0.0
            fnr = fn / N_g if N_g > 0 else 0.0
            mean_p = float(sub_g["calibrated_ai_probability"].mean())
            med_p = float(sub_g["calibrated_ai_probability"].median())
            
            recall_dict[m][g_disp] = round(rec * 100.0, 2)
            fnr_dict[m][g_disp] = round(fnr * 100.0, 2)
            gen_recalls[g_disp] = rec
            
            matrix_rows.append({
                "model": m,
                "generator_raw": g_raw,
                "generator_name": g_disp,
                "N": N_g,
                "TP": tp,
                "FN": fn,
                "recall": round(rec, 4),
                "fnr": round(fnr, 4),
                "mean_ai_probability": round(mean_p, 4),
                "median_ai_probability": round(med_p, 4)
            })
            
        # Hardest / easiest
        sorted_g = sorted(gen_recalls.items(), key=lambda x: x[1])
        hardest_g, hardest_r = sorted_g[0]
        easiest_g, easiest_r = sorted_g[-1]
        
        hardest_easiest[m] = {
            "hardest_generator": hardest_g,
            "hardest_recall": round(hardest_r * 100.0, 2),
            "easiest_generator": easiest_g,
            "easiest_recall": round(easiest_r * 100.0, 2)
        }
        
    df_gen_matrix = pd.DataFrame(matrix_rows)
    df_recall_wide = pd.DataFrame(recall_dict).T
    df_fnr_wide = pd.DataFrame(fnr_dict).T
    
    # Specific trade-offs on AI images
    ai_ids = meta[meta["true_label"] == "AI"].index
    corr_ai = piv_corr.loc[ai_ids].join(meta[["generator"]])
    
    d_fail_a_succ = corr_ai[(corr_ai["V5-D Gated Residual"] == False) & (corr_ai["V5-A Spatial"] == True)]
    a_fail_d_succ = corr_ai[(corr_ai["V5-A Spatial"] == False) & (corr_ai["V5-D Gated Residual"] == True)]
    
    tradeoffs = {
        "D_fail_A_succ_total": len(d_fail_a_succ),
        "D_fail_A_succ_by_generator": d_fail_a_succ["generator"].map(GENERATOR_NAMES).value_counts().to_dict(),
        "A_fail_D_succ_total": len(a_fail_d_succ),
        "A_fail_D_succ_by_generator": a_fail_d_succ["generator"].map(GENERATOR_NAMES).value_counts().to_dict(),
    }
    
    summary = {
        "hardest_easiest_per_model": hardest_easiest,
        "ai_tradeoffs_A_vs_D": tradeoffs,
        "recall_matrix": recall_dict,
        "fnr_matrix": fnr_dict
    }
    
    return summary, df_gen_matrix, df_recall_wide, df_fnr_wide


def analyze_confidence_and_errors(df_raw: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    summary_dict = {}
    
    for m in MODELS:
        sub = df_raw[df_raw["model"] == m].copy()
        # Confidence = max(P(AI), P(REAL))
        sub["conf"] = np.maximum(sub["calibrated_ai_probability"], sub["calibrated_real_probability"])
        
        corr = sub[sub["correct"] == True]
        inc = sub[sub["correct"] == False]
        
        c_mean = float(corr["conf"].mean())
        c_med = float(corr["conf"].median())
        i_mean = float(inc["conf"].mean())
        i_med = float(inc["conf"].median())
        
        hc_err_90 = int((inc["conf"] >= 0.90).sum())
        hc_err_70 = int((inc["conf"] >= 0.70).sum())
        lc_corr_60 = int((corr["conf"] < 0.60).sum())
        
        # REAL FP specifically
        rfp = sub[(sub["true_label"] == "REAL") & (sub["prediction"] == "AI")]
        rfp_ge_90 = int((rfp["calibrated_ai_probability"] >= 0.90).sum())
        rfp_70_89 = int(((rfp["calibrated_ai_probability"] >= 0.70) & (rfp["calibrated_ai_probability"] < 0.90)).sum())
        rfp_50_69 = int(((rfp["calibrated_ai_probability"] >= 0.50) & (rfp["calibrated_ai_probability"] < 0.70)).sum())
        
        # AI FN specifically (P(AI) < 0.50)
        afn = sub[(sub["true_label"] == "AI") & (sub["prediction"] == "REAL")]
        afn_lt_10 = int((afn["calibrated_ai_probability"] < 0.10).sum())
        afn_10_29 = int(((afn["calibrated_ai_probability"] >= 0.10) & (afn["calibrated_ai_probability"] < 0.30)).sum())
        afn_30_49 = int(((afn["calibrated_ai_probability"] >= 0.30) & (afn["calibrated_ai_probability"] < 0.50)).sum())
        
        row_dict = {
            "model": m,
            "correct_mean_conf": round(c_mean, 4),
            "correct_median_conf": round(c_med, 4),
            "incorrect_mean_conf": round(i_mean, 4),
            "incorrect_median_conf": round(i_med, 4),
            "errors_conf_ge_90": hc_err_90,
            "errors_conf_ge_70": hc_err_70,
            "correct_conf_lt_60": lc_corr_60,
            "real_fp_ge_90": rfp_ge_90,
            "real_fp_70_to_89": rfp_70_89,
            "real_fp_50_to_69": rfp_50_69,
            "ai_fn_lt_10": afn_lt_10,
            "ai_fn_10_to_29": afn_10_29,
            "ai_fn_30_to_49": afn_30_49
        }
        rows.append(row_dict)
        summary_dict[m] = row_dict
        
    df_conf = pd.DataFrame(rows)
    return summary_dict, df_conf


def build_disagreement_csvs(meta: pd.DataFrame, piv_pred: pd.DataFrame, piv_corr: pd.DataFrame, piv_prob: pd.DataFrame):
    # Determine correct models string per candidate
    correct_models_series = piv_corr.apply(
        lambda r: ", ".join([col.replace("V5-", "").split()[0] for col in piv_corr.columns if r[col] is True]),
        axis=1
    )
    
    base_df = pd.DataFrame({
        "candidate_id": meta.index,
        "filename": meta["original_filename"],
        "true_label": meta["true_label"],
        "generator": meta["generator"].map(GENERATOR_NAMES),
        "pred_A": piv_pred["V5-A Spatial"],
        "pred_B": piv_pred["V5-B Frequency"],
        "pred_C": piv_pred["V5-C Hybrid"],
        "pred_D": piv_pred["V5-D Gated Residual"],
        "prob_AI_A": piv_prob["V5-A Spatial"].round(6),
        "prob_AI_B": piv_prob["V5-B Frequency"].round(6),
        "prob_AI_C": piv_prob["V5-C Hybrid"].round(6),
        "prob_AI_D": piv_prob["V5-D Gated Residual"].round(6),
        "correct_models": correct_models_series
    })
    
    # 1. A_correct_D_wrong.csv
    mask_a_d = (piv_corr["V5-A Spatial"] == True) & (piv_corr["V5-D Gated Residual"] == False)
    df_a_d = base_df[mask_a_d.values].copy()
    df_a_d.to_csv(ANALYSIS_DIR / "A_correct_D_wrong.csv", index=False)
    
    # 2. D_correct_A_wrong.csv
    mask_d_a = (piv_corr["V5-D Gated Residual"] == True) & (piv_corr["V5-A Spatial"] == False)
    df_d_a = base_df[mask_d_a.values].copy()
    df_d_a.to_csv(ANALYSIS_DIR / "D_correct_A_wrong.csv", index=False)
    
    # 3. C_correct_A_D_wrong.csv
    mask_c_ad = (piv_corr["V5-C Hybrid"] == True) & (piv_corr["V5-A Spatial"] == False) & (piv_corr["V5-D Gated Residual"] == False)
    df_c_ad = base_df[mask_c_ad.values].copy()
    df_c_ad.to_csv(ANALYSIS_DIR / "C_correct_A_D_wrong.csv", index=False)
    
    # 4. B_correct_A_C_D_wrong.csv
    mask_b_acd = (piv_corr["V5-B Frequency"] == True) & (piv_corr["V5-A Spatial"] == False) & (piv_corr["V5-C Hybrid"] == False) & (piv_corr["V5-D Gated Residual"] == False)
    df_b_acd = base_df[mask_b_acd.values].copy()
    df_b_acd.to_csv(ANALYSIS_DIR / "B_correct_A_C_D_wrong.csv", index=False)
    
    print(f"[*] Disagreement CSVs generated:")
    print(f"    - A_correct_D_wrong.csv:       {len(df_a_d)} rows")
    print(f"    - D_correct_A_wrong.csv:       {len(df_d_a)} rows")
    print(f"    - C_correct_A_D_wrong.csv:     {len(df_c_ad)} rows")
    print(f"    - B_correct_A_C_D_wrong.csv:   {len(df_b_acd)} rows")


def analyze_longitudinal_comparison() -> pd.DataFrame:
    # Authoritative historical results specified by user
    data = [
        {
            "model": "V5-A Spatial",
            "clean_v5_test_f1": 97.02,
            "avg_controlled_robustness_f1": 76.49,
            "external_benchmark_f1": 70.84,
            "clean_to_external_delta": round(70.84 - 97.02, 2),
            "robustness_to_external_delta": round(70.84 - 76.49, 2),
            "clean_rank": 2,
            "robustness_rank": 1,
            "external_rank": 1,
            "rank_change_clean_to_ext": "+1"
        },
        {
            "model": "V5-B Frequency",
            "clean_v5_test_f1": 90.65,
            "avg_controlled_robustness_f1": 63.38,
            "external_benchmark_f1": 32.96,
            "clean_to_external_delta": round(32.96 - 90.65, 2),
            "robustness_to_external_delta": round(32.96 - 63.38, 2),
            "clean_rank": 4,
            "robustness_rank": 4,
            "external_rank": 4,
            "rank_change_clean_to_ext": "0"
        },
        {
            "model": "V5-C Hybrid",
            "clean_v5_test_f1": 96.89,
            "avg_controlled_robustness_f1": 71.79,
            "external_benchmark_f1": 62.95,
            "clean_to_external_delta": round(62.95 - 96.89, 2),
            "robustness_to_external_delta": round(62.95 - 71.79, 2),
            "clean_rank": 3,
            "robustness_rank": 3,
            "external_rank": 2,
            "rank_change_clean_to_ext": "+1"
        },
        {
            "model": "V5-D Gated Residual",
            "clean_v5_test_f1": 97.17,
            "avg_controlled_robustness_f1": 73.44,
            "external_benchmark_f1": 58.01,
            "clean_to_external_delta": round(58.01 - 97.17, 2),
            "robustness_to_external_delta": round(58.01 - 73.44, 2),
            "clean_rank": 1,
            "robustness_rank": 2,
            "external_rank": 3,
            "rank_change_clean_to_ext": "-2"
        },
    ]
    df_comp = pd.DataFrame(data)
    df_comp.to_csv(ANALYSIS_DIR / "clean_vs_external_comparison.csv", index=False)
    return df_comp


def generate_plots(df_comp: pd.DataFrame, df_recall_wide: pd.DataFrame, df_conf: pd.DataFrame, real_summary: Dict[str, Any]):
    # 1. Model F1 comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    models = ["V5-A\nSpatial", "V5-B\nFrequency", "V5-C\nHybrid", "V5-D\nGated Residual"]
    f1s = df_comp["external_benchmark_f1"].values
    bars = ax.bar(models, f1s, color=["#2563eb", "#dc2626", "#059669", "#7c3aed"], width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("External Benchmark (Defactify v1) — Model F1 Scores", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "model_f1_comparison.png")
    plt.close(fig)

    # 2. Generator Recall Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    x = np.arange(len(df_recall_wide.columns))
    width = 0.20
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    for i, m in enumerate(MODELS):
        vals = df_recall_wide.loc[m].values
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=colors[i], edgecolor="#1e293b", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_recall_wide.columns, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("AI Detection Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("AI Generator Detection Recall by Model (N=100 per Generator)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "generator_recall_comparison.png")
    plt.close(fig)

    # 3. REAL FPR Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fprs = [pm["fpr"] * 100.0 for pm in real_summary["per_model_real"]]
    bars = ax.bar(models, fprs, color=["#ef4444", "#3b82f6", "#f59e0b", "#10b981"], width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylim(0, 60)
    ax.set_ylabel("False Positive Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("REAL Photograph False Alarm Rate (N=400 MS COCO)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "real_fpr_comparison.png")
    plt.close(fig)

    # 4. Confidence Distributions (Correct vs Incorrect Mean Conf)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(models))
    w = 0.32
    corr_c = [df_conf.loc[df_conf["model"] == m, "correct_mean_conf"].values[0] * 100 for m in MODELS]
    inc_c = [df_conf.loc[df_conf["model"] == m, "incorrect_mean_conf"].values[0] * 100 for m in MODELS]
    ax.bar(x - w/2, corr_c, w, label="Correct Predictions", color="#10b981", edgecolor="#1e293b")
    ax.bar(x + w/2, inc_c, w, label="Incorrect Predictions", color="#f43f5e", edgecolor="#1e293b")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Mean Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_title("Mean Confidence: Correct vs Incorrect Predictions", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="#f8fafc")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "confidence_distributions.png")
    plt.close(fig)

    # 5. Clean vs Robustness vs External F1 Trajectory
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    regimes = ["Clean V5 Test\n(In-Distribution)", "Average Controlled\nRobustness", "External Benchmark\n(Defactify v1)"]
    for i, row in df_comp.iterrows():
        y = [row["clean_v5_test_f1"], row["avg_controlled_robustness_f1"], row["external_benchmark_f1"]]
        ax.plot(regimes, y, "o-", linewidth=2.2, markersize=8, label=row["model"], color=colors[i])
        for xi, yi in enumerate(y):
            ax.annotate(f"{yi:.1f}%", xy=(xi, yi), xytext=(0, 6 if i%2==0 else -14),
                        textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(20, 105)
    ax.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Generalization Degradation Across Evaluation Regimes", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower left")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "clean_vs_robustness_vs_external_f1.png")
    plt.close(fig)

    print("[*] All 5 analysis figures saved to:", ANALYSIS_DIR)


def generate_comprehensive_markdown_report(
    summary_data: Dict[str, Any],
    agreement_summary: Dict[str, Any],
    real_summary: Dict[str, Any],
    gen_summary: Dict[str, Any],
    conf_summary: Dict[str, Any],
    df_comp: pd.DataFrame,
    df_recall_wide: pd.DataFrame,
    df_fnr_wide: pd.DataFrame
):
    md_path = ANALYSIS_DIR / "external_error_analysis.md"
    
    lines = [
        "# AIDetect — Frozen Defactify External Benchmark Error Analysis",
        "",
        "## Executive Summary",
        "",
        "This report provides a strict post-inference factual error and agreement analysis of the **Defactify External Benchmark (v1)**.",
        "The benchmark evaluates four frozen V5 models across **800 out-of-distribution images** (400 natural REAL photographs from MS COCO and 100 images each from SD3, Midjourney v6, DALL-E 3, and SDXL).",
        "",
        "- **Total Images Analyzed:** 800",
        "- **Total Model Predictions Analyzed:** 3,200",
        "- **Model Roster:** V5-A Spatial, V5-B Frequency, V5-C Hybrid, V5-D Gated Residual (Calibrated $T=2.198387$)",
        "- **Model Checkpoint Status:** Frozen and unaltered",
        "- **V5 Test Set Access:** Zero access (frozen 9,000-image test set untouched)",
        "",
        "---",
        "",
        "## 1. Model Agreement & Ensemble Divergence",
        "",
        "Cross-model agreement on the external benchmark demonstrates significant prediction divergence across architectures:",
        "",
        "| Agreement Level | Image Count | Percentage of Benchmark |",
        "| :--- | :---: | :---: |",
        f"| **All 4 Models Agree (Unanimous)** | {agreement_summary['agreement_distribution']['all_4_agree_count']} | {agreement_summary['agreement_distribution']['all_4_agree_pct']}% |",
        f"| **3 of 4 Models Agree (1 Dissenter)** | {agreement_summary['agreement_distribution']['three_of_four_agree_count']} | {agreement_summary['agreement_distribution']['three_of_four_agree_pct']}% |",
        f"| **2 vs 2 Split Decision (Maximal Disagreement)** | {agreement_summary['agreement_distribution']['two_vs_two_split_count']} | {agreement_summary['agreement_distribution']['two_vs_two_split_pct']}% |",
        "",
        "### Pairwise Model Agreement Matrix",
        "",
        "| Model | V5-A Spatial | V5-B Frequency | V5-C Hybrid | V5-D Gated Residual |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    
    mat = agreement_summary["pairwise_matrix"]
    for m1 in MODELS:
        row_str = f"| **{m1}** | " + " | ".join([str(mat[m2][m1]) for m2 in MODELS]) + " |"
        lines.append(row_str)
        
    lines.extend([
        "",
        "### Key Disagreement Counts",
        "",
        f"- **V5-D Correct while V5-A is Wrong:** {len(pd.read_csv(ANALYSIS_DIR / 'D_correct_A_wrong.csv'))} images (98 REAL photos where V5-D suppressed V5-A's false alarm, 10 AI images where V5-D detected what V5-A missed).",
        f"- **V5-A Correct while V5-D is Wrong:** {len(pd.read_csv(ANALYSIS_DIR / 'A_correct_D_wrong.csv'))} images (126 AI images where V5-A detected what V5-D missed, 12 REAL photos where V5-A avoided V5-D's false alarm).",
        f"- **V5-C Correct while both V5-A and V5-D are Wrong:** {len(pd.read_csv(ANALYSIS_DIR / 'C_correct_A_D_wrong.csv'))} images.",
        f"- **V5-B Correct while V5-A, V5-C, and V5-D are all Wrong:** {len(pd.read_csv(ANALYSIS_DIR / 'B_correct_A_C_D_wrong.csv'))} images (56 natural REAL photographs where all spatial-bearing models produced false alarms, but frequency alone stayed correct).",
        "",
        "---",
        "",
        "## 2. REAL Image Error Analysis (The False Alarm Problem)",
        "",
        "Evaluation on the 400 natural MS COCO photographs reveals substantial divergence in false alarm rates across architectures:",
        "",
        "| Model | Correct REAL (TN) | False Positives (FP) | FPR | Mean P(AI) | Median P(AI) | % P(AI) >= 50% | % P(AI) >= 70% | % P(AI) >= 90% |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    for pm in real_summary["per_model_real"]:
        lines.append(
            f"| **{pm['model']}** | {pm['correct_real']}/400 | {pm['false_positives']}/400 | "
            f"{pm['fpr']*100:.2f}% | {pm['mean_ai_prob']:.4f} | {pm['median_ai_prob']:.4f} | "
            f"{pm['pct_ge_50']:.2f}% | {pm['pct_ge_70']:.2f}% | {pm['pct_ge_90']:.2f}% |"
        )
        
    lines.extend([
        "",
        "### REAL Image Error Distribution by Model Agreement",
        "",
        "| Error Group | Count | Percentage of REAL (N=400) | Description |",
        "| :--- | :---: | :---: | :--- |",
    ])
    
    for gr in real_summary["groupings"]:
        lines.append(f"| **{gr['error_group']}** | {gr['count']} | {gr['percentage']:.2f}% | Breakdown across 400 REAL photos |")
        
    lines.extend([
        "",
        "---",
        "",
        "## 3. AI Generator Error Analysis",
        "",
        "### Generator × Model AI Detection Recall Matrix",
        "",
        "| Model | Stable Diffusion 3 | Midjourney v6 | DALL-E 3 | Stable Diffusion XL |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    
    for m in MODELS:
        r = df_recall_wide.loc[m]
        lines.append(f"| **{m}** | {r['SD3']:.2f}% | {r['Midjourney v6']:.2f}% | {r['DALL-E 3']:.2f}% | {r['SDXL']:.2f}% |")
        
    lines.extend([
        "",
        "### Generator × Model False Negative Rate (FNR) Matrix",
        "",
        "| Model | Stable Diffusion 3 | Midjourney v6 | DALL-E 3 | Stable Diffusion XL |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    
    for m in MODELS:
        r = df_fnr_wide.loc[m]
        lines.append(f"| **{m}** | {r['SD3']:.2f}% | {r['Midjourney v6']:.2f}% | {r['DALL-E 3']:.2f}% | {r['SDXL']:.2f}% |")
        
    lines.extend([
        "",
        "### Generator Vulnerability Ranking per Model",
        "",
        "| Model | Easiest Generator (Highest Recall) | Hardest Generator (Lowest Recall) |",
        "| :--- | :--- | :--- |",
    ])
    
    for m, d in gen_summary["hardest_easiest_per_model"].items():
        lines.append(f"| **{m}** | {d['easiest_generator']} ({d['easiest_recall']:.2f}%) | {d['hardest_generator']} ({d['hardest_recall']:.2f}%) |")
        
    lines.extend([
        "",
        "### Direct AI Error Trade-offs Between V5-A and V5-D",
        "",
        f"- **V5-D Fails while V5-A Succeeds:** {gen_summary['ai_tradeoffs_A_vs_D']['D_fail_A_succ_total']} AI images total:",
    ])
    for g, cnt in gen_summary["ai_tradeoffs_A_vs_D"]["D_fail_A_succ_by_generator"].items():
        lines.append(f"  - {g}: {cnt} images")
        
    lines.append(f"- **V5-A Fails while V5-D Succeeds:** {gen_summary['ai_tradeoffs_A_vs_D']['A_fail_D_succ_total']} AI images total:")
    for g, cnt in gen_summary["ai_tradeoffs_A_vs_D"]["A_fail_D_succ_by_generator"].items():
        lines.append(f"  - {g}: {cnt} images")
        
    lines.extend([
        "",
        "---",
        "",
        "## 4. Confidence & Error Analysis",
        "",
        "| Model | Correct Mean Conf | Correct Median Conf | Incorrect Mean Conf | Incorrect Median Conf | High-Conf Errors (>=90%) | High-Conf Errors (>=70%) | Low-Conf Correct (<60%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    for m in MODELS:
        c = conf_summary[m]
        lines.append(
            f"| **{m}** | {c['correct_mean_conf']*100:.2f}% | {c['correct_median_conf']*100:.2f}% | "
            f"{c['incorrect_mean_conf']*100:.2f}% | {c['incorrect_median_conf']*100:.2f}% | "
            f"{c['errors_conf_ge_90']} | {c['errors_conf_ge_70']} | {c['correct_conf_lt_60']} |"
        )
        
    lines.extend([
        "",
        "### Severity of Erroneous Predictions",
        "",
        "| Model | REAL FP >=90% | REAL FP 70–89.99% | REAL FP 50–69.99% | Total REAL FP | AI FN <10% | AI FN 10–29.99% | AI FN 30–49.99% | Total AI FN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    for m in MODELS:
        c = conf_summary[m]
        tot_fp = c['real_fp_ge_90'] + c['real_fp_70_to_89'] + c['real_fp_50_to_69']
        tot_fn = c['ai_fn_lt_10'] + c['ai_fn_10_to_29'] + c['ai_fn_30_to_49']
        lines.append(
            f"| **{m}** | {c['real_fp_ge_90']} | {c['real_fp_70_to_89']} | {c['real_fp_50_to_69']} | {tot_fp} | "
            f"{c['ai_fn_lt_10']} | {c['ai_fn_10_to_29']} | {c['ai_fn_30_to_49']} | {tot_fn} |"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "## 5. Longitudinal Comparison Across Evaluation Regimes",
        "",
        "Comparison of model F1 scores across (1) Clean V5 Test, (2) Average Controlled Robustness, and (3) Defactify External Benchmark:",
        "",
        "| Model | Clean V5 Test F1 | Avg Controlled Robustness F1 | External Benchmark F1 | Clean → External F1 Degradation | Robustness → External F1 Delta | Clean Rank | Robustness Rank | External Rank | Rank Shift |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    for _, r in df_comp.iterrows():
        lines.append(
            f"| **{r['model']}** | {r['clean_v5_test_f1']:.2f}% | {r['avg_controlled_robustness_f1']:.2f}% | "
            f"{r['external_benchmark_f1']:.2f}% | **{r['clean_to_external_delta']:+.2f} pp** | "
            f"{r['robustness_to_external_delta']:+.2f} pp | #{r['clean_rank']} | #{r['robustness_rank']} | #{r['external_rank']} | **{r['rank_change_clean_to_ext']}** |"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "## 6. Research Findings & Answers to Evaluation Questions",
        "",
        "### 1. Does clean-test performance predict external performance?",
        "**Finding: No.** Clean-test F1 across the four models was tightly clustered between 90.65% and 97.17% (a spread of only 6.52 percentage points, with V5-D and V5-A separated by just 0.15 pp). On the external benchmark, performance dispersed widely from 32.96% to 70.84% (a spread of 37.88 pp), and the ranking inverted. Clean-test accuracy was uninformative regarding out-of-distribution generalization.",
        "",
        "### 2. Does V5-D's clean-test advantage transfer externally?",
        "**Finding: No.** V5-D led on clean test (97.17% F1), but dropped to third place on the external benchmark (58.01% F1), falling 12.83 percentage points behind V5-A Spatial (70.84%) and 4.94 percentage points behind V5-C Hybrid (62.95%). Its clean-test lead did not translate into superior external generalization.",
        "",
        "### 3. Does V5-A show more consistent generalization?",
        "**Finding: Yes, in overall F1 and AI recall, but with high false positive rates.** V5-A suffered the smallest degradation from clean test (-26.18 pp vs -39.16 pp for V5-D and -57.69 pp for V5-B) and from controlled robustness (-5.65 pp vs -15.43 pp for V5-D). It attained the highest external F1 (70.84%) and highest AI recall (79.25%). However, V5-A exhibited a critical false alarm problem on natural photographs, misclassifying 178 of 400 REAL images as AI (44.50% FPR).",
        "",
        "### 4. Does frequency-only remain the weakest approach?",
        "**Finding: Yes.** V5-B collapsed under distribution shift, achieving an external F1 of only 32.96% (a degradation of -57.69 pp from clean test). V5-B detected only 22.00% of AI images overall (including only 9.00% of SDXL images and 17.00% of SD3 images). Frequency-only analysis is severely brittle when evaluated on external generator pipelines.",
        "",
        "### 5. Does spatial-frequency fusion provide a consistent advantage?",
        "**Finding: Mixed.** Dual-stream fusion in V5-C Hybrid (62.95% F1) improved upon V5-D Gated Residual (58.01%) and V5-B (32.96%), but trailed spatial-only V5-A (70.84%). In V5-D, learned gating successfully reduced false alarms on REAL photographs (FPR dropped from 44.50% in V5-A to 23.00% in V5-D), but at the substantial cost of failing to detect modern AI generators (FNR increased from 20.75% in V5-A to 49.75% in V5-D).",
        "",
        "### 6. Are errors generator-dependent?",
        "**Finding: Yes.** Generator recall varied substantially across architectures:",
        "- **SDXL** was the most difficult generator across models (31.00% recall for V5-D, 9.00% for V5-B, 50.00% for V5-C).",
        "- **Midjourney v6** was the easiest generator for V5-B (34.00%), V5-C (66.00%), and V5-D (67.00%), but the hardest for V5-A (73.00%).",
        "- **Stable Diffusion 3** exhibited high recall on V5-A (91.00%), but low recall on V5-B (17.00%) and moderate recall on V5-D (56.00%).",
        "",
        "### 7. Is the main external weakness false positives on REAL images, false negatives on AI images, or both?",
        "**Finding: Both, with asymmetric error profiles across models:**",
        "- **V5-A Spatial:** Primary failure mode is **False Positives on REAL photographs** (FPR = 44.50%, 178 false alarms vs 83 missed AI).",
        "- **V5-B Frequency:** Primary failure mode is **False Negatives on AI generators** (FNR = 78.00%, 312 missed AI vs 46 false alarms).",
        "- **V5-D Gated Residual:** Exhibits **balanced but elevated errors in both directions** (FPR = 23.00%, 92 false alarms; FNR = 49.75%, 199 missed AI).",
        "",
        "### 8. What evidence supports domain/generalization limitations?",
        "**Finding: Strong evidence of severe distribution shift:**",
        "1. **High-confidence erroneous predictions:** V5-A produced 184 errors with confidence $\\ge 90\\%$, including assigning $\\ge 90\\%$ AI probability to 127 natural COCO photographs (31.75% of all REAL samples).",
        "2. **Unanimous failure modes:** On 56 natural REAL photographs, all three spatial-bearing models (A, C, D) produced false alarms simultaneously.",
        "3. **Ensemble divergence:** All 4 models agreed on their prediction in only 37.25% of images, showing that out-of-distribution inputs drive models into divergent feature regimes.",
        "",
        "---",
        "",
        "## 7. Artifact Manifest",
        "",
        "The following output artifacts are preserved under `diagnostic_outputs/external_benchmark_v1_analysis/`:",
        "- `model_agreement.csv`: Complete agreement breakdown and pairwise agreement matrix.",
        "- `real_error_analysis.csv`: REAL image error groupings and per-model false positive metrics.",
        "- `generator_model_matrix.csv`: Detailed generator breakdown and recall/FNR matrices.",
        "- `confidence_error_analysis.csv`: Confidence distributions, error severity, and threshold buckets.",
        "- `clean_vs_external_comparison.csv`: Longitudinal comparison table across Clean Test, Robustness, and External Benchmark.",
        "- `A_correct_D_wrong.csv`: 138 disagreement cases where V5-A succeeded and V5-D failed.",
        "- `D_correct_A_wrong.csv`: 108 disagreement cases where V5-D succeeded and V5-A failed.",
        "- `C_correct_A_D_wrong.csv`: 25 disagreement cases where V5-C succeeded while both A and D failed.",
        "- `B_correct_A_C_D_wrong.csv`: 56 disagreement cases where V5-B succeeded while all other models failed.",
        "- `external_analysis_summary.json`: Complete hierarchical data file containing all numerical results.",
        "- Visual figures: `model_f1_comparison.png`, `generator_recall_comparison.png`, `real_fpr_comparison.png`, `confidence_distributions.png`, `clean_vs_robustness_vs_external_f1.png`.",
        ""
    ])
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[*] Markdown report generated at:", md_path)


def main():
    print("=" * 80)
    print("AIDETECT EXTERNAL BENCHMARK POST-INFERENCE ANALYSIS")
    print("=" * 80)
    
    # 1. Load data
    df_raw, meta, piv_pred, piv_prob, piv_corr = load_data()
    print(f"[*] Loaded {len(df_raw)} records ({len(meta)} unique images, {len(MODELS)} models).")
    
    # Check for invalid / missing
    missing_cnt = df_raw.isnull().sum().sum()
    if missing_cnt > 0:
        print(f"[WARNING] Found {missing_cnt} missing values in per-image dataset.")
    else:
        print("[*] Dataset integrity verified: 0 missing/null values.")
        
    # 2. Section 1: Model Agreement
    agreement_summary, df_agreement = analyze_model_agreement(meta, piv_pred, piv_corr, piv_prob)
    df_agreement.to_csv(ANALYSIS_DIR / "model_agreement.csv", index=False)
    
    # 3. Section 2: REAL Image Error Analysis
    real_summary, df_real_error, top_20_dict = analyze_real_errors(meta, piv_pred, piv_corr, piv_prob, df_raw)
    df_real_error.to_csv(ANALYSIS_DIR / "real_error_analysis.csv", index=False)
    
    # 4. Section 3: AI Generator Error Analysis
    gen_summary, df_gen_matrix, df_recall_wide, df_fnr_wide = analyze_ai_generators(meta, piv_pred, piv_corr, piv_prob, df_raw)
    df_gen_matrix.to_csv(ANALYSIS_DIR / "generator_model_matrix.csv", index=False)
    
    # 5. Section 4: Confidence / Error Analysis
    conf_summary, df_conf = analyze_confidence_and_errors(df_raw)
    df_conf.to_csv(ANALYSIS_DIR / "confidence_error_analysis.csv", index=False)
    
    # 6. Section 5: Cross-Model Disagreement CSVs
    build_disagreement_csvs(meta, piv_pred, piv_corr, piv_prob)
    
    # 7. Section 7: Longitudinal Comparison
    df_comp = analyze_longitudinal_comparison()
    
    # 8. Plots
    generate_plots(df_comp, df_recall_wide, df_conf, real_summary)
    
    # 9. Master JSON
    master_summary = {
        "dataset_info": {
            "total_images": len(meta),
            "total_records": len(df_raw),
            "models": MODELS,
            "real_images": 400,
            "ai_images": 400,
            "generator_counts": {GENERATOR_NAMES[k]: 100 if "Stable" in k or "Midjourney" in k or "DALL" in k else 400 for k in GENERATOR_NAMES}
        },
        "model_agreement": agreement_summary,
        "real_error_analysis": real_summary,
        "ai_generator_analysis": gen_summary,
        "confidence_error_analysis": conf_summary,
        "top_20_real_false_positives": {m: top_20_dict[m].to_dict(orient="records") for m in MODELS},
        "longitudinal_comparison": df_comp.to_dict(orient="records")
    }
    
    summary_json_path = ANALYSIS_DIR / "external_analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print("[*] Master analysis JSON saved to:", summary_json_path)
    
    # 10. Master Markdown Report
    generate_comprehensive_markdown_report(
        master_summary,
        agreement_summary,
        real_summary,
        gen_summary,
        conf_summary,
        df_comp,
        df_recall_wide,
        df_fnr_wide
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
