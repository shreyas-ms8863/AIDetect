"""Generate an independent V5 audit log from raw prediction data.

This is an artifact-generation script, not a read-only command: it writes
``statistical_sanity_check.csv``.  Every recomputed value in the log comes
from raw prediction rows or confusion counts and is compared with an
independently stored artifact.  Historical draft discrepancies are recorded
only in the separate historical correction documents and are not presented as
current final results.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRENGTHENED_DIR = PROJECT_ROOT / "evaluation_results" / "v5" / "strengthened_evaluation"
RAW_EXTERNAL_CSV = PROJECT_ROOT / "diagnostic_outputs" / "external_benchmark_v1_results" / "external_benchmark_v1_per_image.csv"
RAW_CLEAN_CSV = STRENGTHENED_DIR / "v5_clean_test_predictions.csv"
CALIBRATION_CONFIG = PROJECT_ROOT / "backend" / "models" / "v5" / "v5_d_calibration.json"

MODELS = [
    ("V5-A Spatial", "prob_v5_a", "pred_v5_a"),
    ("V5-B Frequency", "prob_v5_b", "pred_v5_b"),
    ("V5-C Hybrid", "prob_v5_c", "pred_v5_c"),
    ("V5-D Gated Residual", "calibrated_prob_v5_d", "pred_v5_d"),
]


def metric_values(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    real_fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + tnr) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "real_fpr": real_fpr,
        "mcc": (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def add_check(records, metric, reported, recomputed, notes="", tolerance=1e-6):
    difference = abs(float(reported) - float(recomputed))
    records.append({
        "metric": metric,
        "reported_value": reported,
        "recomputed_value": recomputed,
        "difference": round(difference, 8),
        "status": "PASS" if difference <= tolerance else "FAIL",
        "notes": notes,
    })


def run_audit():
    records = []
    clean_artifact = json.load(open(STRENGTHENED_DIR / "clean_metrics_extended.json", encoding="utf-8"))
    external_artifact = json.load(open(STRENGTHENED_DIR / "external_metrics_extended.json", encoding="utf-8"))
    generator_artifact = pd.read_csv(STRENGTHENED_DIR / "generator_analysis.csv")
    clean = pd.read_csv(RAW_CLEAN_CSV)
    external = pd.read_csv(RAW_EXTERNAL_CSV)

    if len(clean) != 9000:
        raise AssertionError(f"Expected 9,000 clean rows, found {len(clean)}")
    if len(external) != 3200:
        raise AssertionError(f"Expected 3,200 external rows, found {len(external)}")

    # Clean metrics: raw prediction columns -> independent contingency metrics
    clean_labels = clean["label_int"].to_numpy()
    for model, _, pred_col in MODELS:
        calculated = metric_values(clean_labels, clean[pred_col].to_numpy())
        reported = clean_artifact["point_estimates"][model]
        for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"):
            add_check(records, f"clean_{key}_{model}", reported[key], calculated[key], "raw clean prediction CSV vs clean metrics JSON")

    # External metrics: raw per-image rows -> independent contingency metrics
    for model, _, _ in MODELS:
        rows = external[external["model"] == model]
        labels = (rows["true_label"] == "AI").astype(int).to_numpy()
        predictions = (rows["prediction"] == "AI").astype(int).to_numpy()
        calculated = metric_values(labels, predictions)
        reported = external_artifact["point_estimates"][model]
        for key in ("real_fpr", "recall", "f1", "balanced_accuracy", "mcc"):
            add_check(records, f"external_{key}_{model}", reported[key], calculated[key], "raw external prediction CSV vs external metrics JSON")

    # Generator recalls: raw AI rows grouped by model and generator, compared
    # with the stored generator artifact.  The grouping is keyed, never positional.
    ai = external[external["true_label"] == "AI"]
    display_names = {
        "Stable_Diffusion_3": "SD3",
        "Midjourney_v6": "Midjourney v6",
        "DALL-E_3": "DALL-E 3",
        "Stable_Diffusion_XL": "SDXL",
    }
    grouped = ai.groupby(["model", "generator"], sort=False)["prediction"].agg(
        count="size", detected=lambda values: int((values == "AI").sum())
    ).reset_index()
    grouped["recall"] = grouped["detected"] / grouped["count"]
    for _, row in grouped.iterrows():
        generator = display_names[row["generator"]]
        stored = generator_artifact[
            (generator_artifact["model"] == row["model"])
            & (generator_artifact["generator"] == generator)
        ]
        if len(stored) != 1:
            raise AssertionError(f"Missing or duplicate generator artifact row: {row['model']}/{generator}")
        add_check(
            records,
            f"generator_recall_{row['model']}_{generator}",
            float(stored.iloc[0]["recall"]),
            float(row["recall"]),
            "raw external rows grouped by model/generator",
            tolerance=1e-12,
        )

    # Calibration provenance is compared against the calibration configuration,
    # not against a duplicated value in an audit artifact.
    calibration = json.load(open(CALIBRATION_CONFIG, encoding="utf-8"))
    stored_calibration = json.load(open(STRENGTHENED_DIR / "calibration_analysis.json", encoding="utf-8"))
    source_temperature = float(calibration["temperature_baseline"]["T"])
    stored_temperature = float(stored_calibration["calibration_configuration"]["frozen_temperature_T"])
    add_check(records, "calibration_temperature", stored_temperature, source_temperature, str(CALIBRATION_CONFIG), tolerance=1e-12)

    output = STRENGTHENED_DIR / "statistical_sanity_check.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    summary = pd.Series([record["status"] for record in records]).value_counts().to_dict()
    print(f"Wrote historical/current audit log to {output}")
    print(f"Checks: {len(records)}; summary: {summary}")


if __name__ == "__main__":
    run_audit()
