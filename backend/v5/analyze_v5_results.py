"""
V5 MASTER RESULTS ANALYSIS
==========================

Analysis-only script.

Reads:
    ../evaluation_results/v5/v5_final_test_comparison.json
    ../evaluation_results/v5/v5_resize_robustness.json
    ../evaluation_results/v5/v5_blur_robustness.json
    ../evaluation_results/v5/v5_noise_robustness.json
    ../evaluation_results/v5/v5_reencoding_robustness.json

Does NOT:
    - load models
    - load images
    - access datasets
    - use GPU
    - retrain
    - rerun evaluations

Generates:
    ../evaluation_results/v5/analysis/
        v5_master_results.csv
        v5_clean_comparison.csv
        v5_robustness_summary.csv
        v5_model_summary.csv
        v5_model_rankings.csv
        v5_master_analysis.json
        v5_clean_f1.png
        v5_robustness_f1.png
        v5_robustness_accuracy.png
        v5_f1_degradation.png
        v5_balanced_accuracy.png
"""

from pathlib import Path
import json
import csv
import math

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent / "evaluation_results" / "v5"
OUTPUT_DIR = RESULTS_DIR / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "clean": "v5_final_test_comparison.json",
    "resize": "v5_resize_robustness.json",
    "blur": "v5_blur_robustness.json",
    "noise": "v5_noise_robustness.json",
    "reencoding": "v5_reencoding_robustness.json",
}


# ============================================================
# MODEL NAME MAPPING
# ============================================================

MODEL_MAP = {
    "spatial": "V5-A Spatial",
    "frequency": "V5-B Frequency",
    "hybrid": "V5-C Hybrid",
    "gated_residual": "V5-D Gated Residual",
}


MODEL_ORDER = [
    "V5-A Spatial",
    "V5-B Frequency",
    "V5-C Hybrid",
    "V5-D Gated Residual",
]


# ============================================================
# LOAD JSON
# ============================================================

def load_json(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


data = {
    experiment: load_json(filename)
    for experiment, filename in FILES.items()
}


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def calculate_derived(metrics):
    """
    Calculate additional statistics from the recorded
    confusion matrix.
    """

    tp = metrics["true_positive"]
    tn = metrics["true_negative"]
    fp = metrics["false_positive"]
    fn = metrics["false_negative"]

    total = tp + tn + fp + fn

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    balanced_accuracy = (metrics["recall"] + specificity) / 2

    total_errors = fp + fn

    ai_prediction_rate = (tp + fp) / total if total else 0.0

    real_prediction_rate = (tn + fn) / total if total else 0.0

    error_rate = total_errors / total if total else 0.0

    return {
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "balanced_accuracy": balanced_accuracy,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "ai_prediction_rate": ai_prediction_rate,
        "real_prediction_rate": real_prediction_rate,
    }


# ============================================================
# BUILD MASTER TABLE
# ============================================================

rows = []


def add_row(
    experiment,
    condition,
    model,
    metrics,
):
    derived = calculate_derived(metrics)

    row = {
        "experiment": experiment,
        "condition": condition,
        "model": model,

        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],

        "true_positive": metrics["true_positive"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],

        **derived,
    }

    rows.append(row)


# ============================================================
# CLEAN
# ============================================================

clean_models = data["clean"]["models"]

for internal_name, model_data in clean_models.items():

    metrics = model_data["test_metrics"]

    add_row(
        experiment="clean",
        condition="clean",
        model=internal_name,
        metrics=metrics,
    )


# ============================================================
# ROBUSTNESS EXPERIMENTS
# ============================================================

for experiment in ["resize", "blur", "noise", "reencoding"]:

    experiment_data = data[experiment]

    for internal_name, model_name in MODEL_MAP.items():

        if internal_name not in experiment_data["models"]:
            raise KeyError(
                f"{internal_name} missing from {experiment}"
            )

        conditions = experiment_data["models"][internal_name]

        for condition, metrics in conditions.items():

            add_row(
                experiment=experiment,
                condition=str(condition),
                model=model_name,
                metrics=metrics,
            )


# ============================================================
# CLEAN BASELINE LOOKUP
# ============================================================

clean_f1 = {}
clean_accuracy = {}

for row in rows:

    if row["experiment"] == "clean":

        clean_f1[row["model"]] = row["f1"]
        clean_accuracy[row["model"]] = row["accuracy"]


# ============================================================
# ADD DEGRADATION METRICS
# ============================================================

for row in rows:

    model = row["model"]

    row["f1_drop_from_clean"] = (
        clean_f1[model] - row["f1"]
    )

    row["accuracy_drop_from_clean"] = (
        clean_accuracy[model] - row["accuracy"]
    )


# ============================================================
# WRITE CSV
# ============================================================

FIELDNAMES = list(rows[0].keys())

master_csv = OUTPUT_DIR / "v5_master_results.csv"

with open(master_csv, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# CLEAN COMPARISON CSV
# ============================================================

clean_rows = [
    r for r in rows
    if r["experiment"] == "clean"
]

clean_csv = OUTPUT_DIR / "v5_clean_comparison.csv"

with open(clean_csv, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
    )

    writer.writeheader()
    writer.writerows(clean_rows)


# ============================================================
# ROBUSTNESS SUMMARY CSV
# ============================================================

robustness_rows = [
    r for r in rows
    if r["experiment"] != "clean"
]

robustness_csv = OUTPUT_DIR / "v5_robustness_summary.csv"

with open(
    robustness_csv,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
    )

    writer.writeheader()
    writer.writerows(robustness_rows)


# ============================================================
# MODEL SUMMARY
# ============================================================

model_summary = []

for model in MODEL_ORDER:

    model_rows = [
        r for r in rows
        if r["model"] == model
    ]

    robustness_only = [
        r for r in model_rows
        if r["experiment"] != "clean"
    ]

    avg_f1 = (
        sum(r["f1"] for r in robustness_only)
        / len(robustness_only)
    )

    avg_accuracy = (
        sum(r["accuracy"] for r in robustness_only)
        / len(robustness_only)
    )

    avg_balanced_accuracy = (
        sum(
            r["balanced_accuracy"]
            for r in robustness_only
        )
        / len(robustness_only)
    )

    avg_f1_drop = (
        sum(
            r["f1_drop_from_clean"]
            for r in robustness_only
        )
        / len(robustness_only)
    )

    best_robustness_f1 = max(
        robustness_only,
        key=lambda r: r["f1"]
    )

    worst_robustness_f1 = min(
        robustness_only,
        key=lambda r: r["f1"]
    )

    clean = next(
        r for r in model_rows
        if r["experiment"] == "clean"
    )

    model_summary.append({
        "model": model,

        "clean_accuracy": clean["accuracy"],
        "clean_f1": clean["f1"],

        "average_robustness_accuracy": avg_accuracy,
        "average_robustness_f1": avg_f1,
        "average_robustness_balanced_accuracy":
            avg_balanced_accuracy,

        "average_f1_drop":
            avg_f1_drop,

        "best_robustness_experiment":
            best_robustness_f1["experiment"],

        "best_robustness_condition":
            best_robustness_f1["condition"],

        "best_robustness_f1":
            best_robustness_f1["f1"],

        "worst_robustness_experiment":
            worst_robustness_f1["experiment"],

        "worst_robustness_condition":
            worst_robustness_f1["condition"],

        "worst_robustness_f1":
            worst_robustness_f1["f1"],
    })


model_summary_csv = OUTPUT_DIR / "v5_model_summary.csv"

with open(
    model_summary_csv,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=model_summary[0].keys(),
    )

    writer.writeheader()
    writer.writerows(model_summary)


# ============================================================
# RANKINGS
# ============================================================

ranking_rows = []

experiments = [
    ("clean", "clean"),
    ("resize", "75"),
    ("resize", "50"),
    ("resize", "25"),
    ("blur", "1.0"),
    ("blur", "2.0"),
    ("blur", "3.0"),
    ("noise", "0.01"),
    ("noise", "0.05"),
    ("noise", "0.10"),
    ("reencoding", "95"),
    ("reencoding", "75"),
    ("reencoding", "50"),
]


for experiment, condition in experiments:

    matching = [
        r for r in rows
        if r["experiment"] == experiment
        and r["condition"] == condition
    ]

    matching.sort(
        key=lambda r: r["f1"],
        reverse=True,
    )

    for rank, row in enumerate(
        matching,
        start=1,
    ):

        ranking_rows.append({
            "experiment": experiment,
            "condition": condition,
            "rank": rank,
            "model": row["model"],
            "f1": row["f1"],
            "accuracy": row["accuracy"],
            "balanced_accuracy":
                row["balanced_accuracy"],
        })


ranking_csv = OUTPUT_DIR / "v5_model_rankings.csv"

with open(
    ranking_csv,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=ranking_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(ranking_rows)


# ============================================================
# JSON MASTER SUMMARY
# ============================================================

master_json = {
    "description":
        "V5 consolidated statistical analysis",

    "test_samples":
        data["clean"]["test_samples"],

    "class_mapping":
        data["clean"]["methodology"]["class_mapping"],

    "analysis_scope": [
        "clean",
        "resize",
        "blur",
        "noise",
        "reencoding",
    ],

    "models":
        MODEL_ORDER,

    "clean_results":
        clean_rows,

    "robustness_results":
        robustness_rows,

    "model_summary":
        model_summary,

    "ranking":
        ranking_rows,
}


master_json_path = (
    OUTPUT_DIR / "v5_master_analysis.json"
)

with open(
    master_json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        master_json,
        f,
        indent=2,
    )


# ============================================================
# GRAPH HELPERS
# ============================================================

def save_bar_chart(
    values,
    labels,
    ylabel,
    title,
    filename,
    ylim=None,
):

    plt.figure(figsize=(10, 6))

    plt.bar(labels, values)

    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# GRAPH 1 — CLEAN F1
# ============================================================

clean_values = [
    clean_f1[m] * 100
    for m in MODEL_ORDER
]

save_bar_chart(
    values=clean_values,
    labels=MODEL_ORDER,
    ylabel="F1-score (%)",
    title="V5 Clean-Test F1 Comparison",
    filename="v5_clean_f1.png",
    ylim=(0, 100),
)


# ============================================================
# GRAPH 2 — ROBUSTNESS F1
# ============================================================

plt.figure(figsize=(14, 8))

for model in MODEL_ORDER:

    model_rows = [
        r for r in robustness_rows
        if r["model"] == model
    ]

    x = range(len(model_rows))

    plt.plot(
        x,
        [r["f1"] * 100 for r in model_rows],
        marker="o",
        label=model,
    )

plt.xticks(
    range(len(robustness_rows) // 4),
    [],
)

plt.ylabel("F1-score (%)")
plt.title("V5 Robustness F1 Across Transformations")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "v5_robustness_f1.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 3 — ROBUSTNESS ACCURACY
# ============================================================

plt.figure(figsize=(14, 8))

for model in MODEL_ORDER:

    model_rows = [
        r for r in robustness_rows
        if r["model"] == model
    ]

    plt.plot(
        range(len(model_rows)),
        [r["accuracy"] * 100 for r in model_rows],
        marker="o",
        label=model,
    )

plt.ylabel("Accuracy (%)")
plt.title("V5 Robustness Accuracy Across Transformations")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "v5_robustness_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 4 — F1 DEGRADATION
# ============================================================

plt.figure(figsize=(14, 8))

for model in MODEL_ORDER:

    model_rows = [
        r for r in robustness_rows
        if r["model"] == model
    ]

    plt.plot(
        range(len(model_rows)),
        [
            r["f1_drop_from_clean"] * 100
            for r in model_rows
        ],
        marker="o",
        label=model,
    )

plt.ylabel("F1 degradation (percentage points)")
plt.title("V5 F1 Degradation From Clean Test")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "v5_f1_degradation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 5 — BALANCED ACCURACY
# ============================================================

plt.figure(figsize=(14, 8))

for model in MODEL_ORDER:

    model_rows = [
        r for r in robustness_rows
        if r["model"] == model
    ]

    plt.plot(
        range(len(model_rows)),
        [
            r["balanced_accuracy"] * 100
            for r in model_rows
        ],
        marker="o",
        label=model,
    )

plt.ylabel("Balanced accuracy (%)")
plt.title("V5 Balanced Accuracy Across Transformations")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "v5_balanced_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 70)
print("V5 MASTER ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Models:")
for model in MODEL_ORDER:
    print(f"  - {model}")

print()
print("Clean-test F1:")

for row in clean_rows:
    print(
        f"  {row['model']:<24} "
        f"{row['f1'] * 100:.2f}%"
    )

print()
print("Average robustness F1:")

for item in model_summary:
    print(
        f"  {item['model']:<24} "
        f"{item['average_robustness_f1'] * 100:.2f}%"
    )

print()
print("Average F1 degradation:")

for item in model_summary:
    print(
        f"  {item['model']:<24} "
        f"{item['average_f1_drop'] * 100:.2f} pp"
    )

print()
print("Output directory:")
print(f"  {OUTPUT_DIR}")

print()
print("Generated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {path.name}")

print()
print("=" * 70)