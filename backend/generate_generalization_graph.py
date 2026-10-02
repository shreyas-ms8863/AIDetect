import os
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_analysis",
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VERIFIED RESULTS
# ============================================================

datasets = [
    "CIFAKE\nClean",
    "Tiny-GenImage\n(100 images)",
    "Real-World\n(19 images)"
]

spatial = [
    95.93,
    50.00,
    52.63
]

frequency = [
    88.75,
    58.00,
    15.79
]

hybrid = [
    95.72,
    50.00,
    52.63
]


# ============================================================
# CREATE GRAPH
# ============================================================

x = range(len(datasets))

width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width for i in x],
    spatial,
    width=width,
    label="Spatial V2"
)

plt.bar(
    list(x),
    frequency,
    width=width,
    label="Frequency V2"
)

plt.bar(
    [i + width for i in x],
    hybrid,
    width=width,
    label="Hybrid V2"
)


# ============================================================
# LABELS
# ============================================================

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Evaluation Dataset"
)

plt.title(
    "Model Performance Across Evaluation Datasets"
)

plt.xticks(
    list(x),
    datasets
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "figure5_generalization_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("GENERALIZATION GRAPH GENERATED")
print("=" * 70)

print()
print(output_path)
print()