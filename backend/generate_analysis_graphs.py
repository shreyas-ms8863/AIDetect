import os
import csv

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "final_analysis"
)

OUTPUT_DIR = os.path.join(
    RESULT_DIR,
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DATA
# ============================================================

tests = [
    "Clean",
    "JPEG Q50",
    "Resize",
    "Blur",
    "Gaussian Noise",
    "JPEG Re-encoding Q90"
]

spatial = [
    95.93,
    91.44,
    95.18,
    56.27,
    85.91,
    95.95
]

frequency = [
    88.75,
    85.64,
    79.96,
    51.15,
    51.27,
    88.08
]

hybrid = [
    95.72,
    93.46,
    94.63,
    55.61,
    90.41,
    95.25
]


# ============================================================
# FIGURE 1
# CLEAN MODEL COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

models = [
    "Spatial V2",
    "Frequency V2",
    "Hybrid V2"
]

clean_values = [
    95.93,
    88.75,
    95.72
]

plt.bar(
    models,
    clean_values
)

plt.ylabel("Accuracy (%)")
plt.title(
    "Clean CIFAKE Test Accuracy"
)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure1_clean_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2
# ROBUSTNESS ACCURACY
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    tests,
    spatial,
    marker="o",
    linewidth=2,
    label="Spatial V2"
)

plt.plot(
    tests,
    frequency,
    marker="o",
    linewidth=2,
    label="Frequency V2"
)

plt.plot(
    tests,
    hybrid,
    marker="o",
    linewidth=2,
    label="Hybrid V2"
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Evaluation Condition")

plt.title(
    "Model Accuracy Under Image Transformations"
)

plt.ylim(0, 100)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure2_robustness_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3
# ACCURACY DROP
# ============================================================

transformation_tests = [
    "JPEG Q50",
    "Resize",
    "Blur",
    "Gaussian Noise",
    "JPEG Re-encoding Q90"
]

spatial_drop = [
    4.49,
    0.75,
    39.66,
    10.02,
    -0.02
]

frequency_drop = [
    3.11,
    8.79,
    37.60,
    37.48,
    0.67
]

hybrid_drop = [
    2.26,
    1.09,
    40.11,
    5.31,
    0.47
]

plt.figure(figsize=(11, 6))

x = range(
    len(transformation_tests)
)

width = 0.25

x_spatial = [
    i - width
    for i in x
]

x_frequency = list(x)

x_hybrid = [
    i + width
    for i in x
]

plt.bar(
    x_spatial,
    spatial_drop,
    width=width,
    label="Spatial V2"
)

plt.bar(
    x_frequency,
    frequency_drop,
    width=width,
    label="Frequency V2"
)

plt.bar(
    x_hybrid,
    hybrid_drop,
    width=width,
    label="Hybrid V2"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    list(x),
    transformation_tests,
    rotation=20
)

plt.ylabel(
    "Accuracy Drop (percentage points)"
)

plt.title(
    "Accuracy Degradation Relative to Clean Evaluation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure3_accuracy_drop.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 4
# HEATMAP-STYLE TABLE
# ============================================================

plt.figure(figsize=(11, 5))

data = [
    spatial,
    frequency,
    hybrid
]

plt.imshow(
    data,
    aspect="auto"
)

plt.colorbar(
    label="Accuracy (%)"
)

plt.xticks(
    range(len(tests)),
    tests,
    rotation=20
)

plt.yticks(
    range(3),
    [
        "Spatial V2",
        "Frequency V2",
        "Hybrid V2"
    ]
)

plt.title(
    "Accuracy Heatmap Across Evaluation Conditions"
)

for row in range(3):

    for col in range(len(tests)):

        plt.text(
            col,
            row,
            f"{data[row][col]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure4_accuracy_heatmap.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("RESEARCH GRAPHS GENERATED")
print("=" * 70)

print()

for filename in os.listdir(OUTPUT_DIR):

    if filename.endswith(".png"):

        print(
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )

print()