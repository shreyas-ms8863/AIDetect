import os
import csv
import json

# ============================================================
# AIDETECT - FINAL AGGREGATE ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# VERIFIED V2 RESULTS
# ============================================================

RESULTS = {

    "Clean": {

        "Spatial V2": {
            "accuracy": 95.93,
            "precision": 94.39,
            "recall": 97.66,
            "f1": 96.00
        },

        "Frequency V2": {
            "accuracy": 88.75,
            "precision": 91.33,
            "recall": 85.63,
            "f1": 88.39
        },

        "Hybrid V2": {
            "accuracy": 95.72,
            "precision": 96.87,
            "recall": 94.49,
            "f1": 95.67
        }
    },


    "JPEG Q50": {

        "Spatial V2": {
            "accuracy": 91.44,
            "precision": 86.70,
            "recall": 97.89,
            "f1": 91.95
        },

        "Frequency V2": {
            "accuracy": 85.64,
            "precision": 83.69,
            "recall": 88.54,
            "f1": 86.04
        },

        "Hybrid V2": {
            "accuracy": 93.46,
            "precision": 93.30,
            "recall": 93.64,
            "f1": 93.47
        }
    },


    "Resize": {

        "Spatial V2": {
            "accuracy": 95.18,
            "precision": 94.93,
            "recall": 95.46,
            "f1": 95.19
        },

        "Frequency V2": {
            "accuracy": 79.96,
            "precision": 72.04,
            "recall": 97.92,
            "f1": 83.01
        },

        "Hybrid V2": {
            "accuracy": 94.63,
            "precision": 97.14,
            "recall": 91.97,
            "f1": 94.48
        }
    },


    "Blur": {

        "Spatial V2": {
            "accuracy": 56.27,
            "precision": 53.42,
            "recall": 98.00,
            "f1": 69.15
        },

        "Frequency V2": {
            "accuracy": 51.15,
            "precision": 50.64,
            "recall": 90.45,
            "f1": 64.93
        },

        "Hybrid V2": {
            "accuracy": 55.61,
            "precision": 53.00,
            "recall": 99.14,
            "f1": 69.07
        }
    },


    "Gaussian Noise": {

        "Spatial V2": {
            "accuracy": 85.91,
            "precision": 79.21,
            "recall": 97.37,
            "f1": 87.35
        },

        "Frequency V2": {
            "accuracy": 51.27,
            "precision": 50.65,
            "recall": 98.91,
            "f1": 66.99
        },

        "Hybrid V2": {
            "accuracy": 90.41,
            "precision": 89.27,
            "recall": 91.86,
            "f1": 90.55
        }
    },


    "JPEG Re-encoding Q90": {

        "Spatial V2": {
            "accuracy": 95.95,
            "precision": 94.62,
            "recall": 97.44,
            "f1": 96.01
        },

        "Frequency V2": {
            "accuracy": 88.08,
            "precision": 91.50,
            "recall": 83.96,
            "f1": 87.57
        },

        "Hybrid V2": {
            "accuracy": 95.25,
            "precision": 96.74,
            "recall": 93.67,
            "f1": 95.18
        }
    }
}


# ============================================================
# CREATE MASTER CSV
# ============================================================

master_rows = []

for test_name, models in RESULTS.items():

    for model_name, metrics in models.items():

        master_rows.append({
            "Test": test_name,
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"]
        })


master_csv = os.path.join(
    OUTPUT_DIR,
    "master_results.csv"
)


with open(
    master_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Test",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    )

    writer.writeheader()
    writer.writerows(master_rows)


# ============================================================
# CALCULATE ACCURACY DROP
# ============================================================

clean_accuracy = {}

for model in [
    "Spatial V2",
    "Frequency V2",
    "Hybrid V2"
]:

    clean_accuracy[model] = RESULTS[
        "Clean"
    ][model]["accuracy"]


drop_rows = []

for test_name, models in RESULTS.items():

    for model_name, metrics in models.items():

        clean = clean_accuracy[model_name]

        drop = clean - metrics["accuracy"]

        drop_rows.append({
            "Test": test_name,
            "Model": model_name,
            "Clean Accuracy": clean,
            "Test Accuracy": metrics["accuracy"],
            "Accuracy Drop": round(drop, 2)
        })


drop_csv = os.path.join(
    OUTPUT_DIR,
    "accuracy_drop.csv"
)


with open(
    drop_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Test",
            "Model",
            "Clean Accuracy",
            "Test Accuracy",
            "Accuracy Drop"
        ]
    )

    writer.writeheader()
    writer.writerows(drop_rows)


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "master_results.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        RESULTS,
        file,
        indent=4
    )


# ============================================================
# PRINT ACCURACY TABLE
# ============================================================

print()
print("=" * 80)
print("AIDETECT FINAL ROBUSTNESS ANALYSIS")
print("=" * 80)

print()

print(
    f"{'Test':<27}"
    f"{'Spatial':>12}"
    f"{'Frequency':>14}"
    f"{'Hybrid':>12}"
)

print("-" * 68)


for test_name, models in RESULTS.items():

    print(
        f"{test_name:<27}"
        f"{models['Spatial V2']['accuracy']:>11.2f}%"
        f"{models['Frequency V2']['accuracy']:>13.2f}%"
        f"{models['Hybrid V2']['accuracy']:>11.2f}%"
    )


# ============================================================
# PRINT ACCURACY DROPS
# ============================================================

print()
print("=" * 80)
print("ACCURACY DROP FROM CLEAN")
print("=" * 80)

print()

for test_name in RESULTS:

    if test_name == "Clean":
        continue

    print(test_name)

    for model in [
        "Spatial V2",
        "Frequency V2",
        "Hybrid V2"
    ]:

        clean = clean_accuracy[model]

        current = RESULTS[
            test_name
        ][model]["accuracy"]

        drop = clean - current

        print(
            f"  {model:<16}: "
            f"{drop:.2f} percentage points"
        )

    print()


# ============================================================
# FILES
# ============================================================

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print()
print("Created files:")
print()
print(master_csv)
print(drop_csv)
print(json_path)
print()