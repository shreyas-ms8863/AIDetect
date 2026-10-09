# analyze_v5_d_vs_a.py

import json
from pathlib import Path

BASE = Path(r"C:\Users\Shreyas\OneDrive\Desktop\AIDetect\backend\evaluation_results\v5")

with open(BASE / "analysis" / "v5_model_rankings.csv", "r", encoding="utf-8") as f:
    import csv
    rows = list(csv.DictReader(f))

conditions = {}

for row in rows:
    condition = f"{row['experiment']}_{row['condition']}"
    model = row["model"]
    f1 = float(row["f1"]) * 100

    conditions.setdefault(condition, {})[model] = f1

print("\nV5-D vs V5-A — F1 Difference")
print("=" * 72)
print(f"{'Condition':<22} {'V5-A':>10} {'V5-D':>10} {'ΔF1 (D-A)':>14}")
print("-" * 72)

deltas = []

for condition, models in conditions.items():
    if "V5-A Spatial" not in models or "V5-D Gated Residual" not in models:
        continue

    a = models["V5-A Spatial"]
    d = models["V5-D Gated Residual"]
    delta = d - a

    deltas.append(delta)

    sign = "+" if delta >= 0 else ""

    print(
        f"{condition:<22} "
        f"{a:>9.2f}% "
        f"{d:>9.2f}% "
        f"{sign}{delta:>12.2f} pp"
    )

print("-" * 72)

robustness_deltas = deltas[1:]  # exclude clean

print(f"\nMean ΔF1 including clean: "
      f"{sum(deltas) / len(deltas):+.2f} pp")

print(f"Mean ΔF1 across robustness tests: "
      f"{sum(robustness_deltas) / len(robustness_deltas):+.2f} pp")

print(
    f"\nV5-D better: "
    f"{sum(x > 0 for x in robustness_deltas)}/"
    f"{len(robustness_deltas)} robustness conditions"
)

print(
    f"V5-A better: "
    f"{sum(x < 0 for x in robustness_deltas)}/"
    f"{len(robustness_deltas)} robustness conditions"
)

print(
    f"Tied: "
    f"{sum(x == 0 for x in robustness_deltas)}/"
    f"{len(robustness_deltas)} robustness conditions"
)