import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================================
# Output path
# ==========================================================
OUT_DIR = r"C:\Users\<user>\Research center\HeartSense_repro\runs\nhanes\nhanes_figures"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(
    OUT_DIR,
    "nhanes_confusion_matrix_grid_style.png"
)

# ==========================================================
# NHANES pooled confusion matrix values
# ==========================================================
models = {
    "Proposed": {
        "TN": 7626,
        "FP": 679,
        "FN": 374,
        "TP": 501,
    },
    "LightGBM": {
        "TN": 7909,
        "FP": 396,
        "FN": 496,
        "TP": 379,
    },
    "Random Forest": {
        "TN": 8249,
        "FP": 56,
        "FN": 791,
        "TP": 84,
    },
    "CNN": {
        "TN": 5319,
        "FP": 2986,
        "FN": 263,
        "TP": 612,
    },
    "Simple Avg": {
        "TN": 7361,
        "FP": 944,
        "FN": 302,
        "TP": 573,
    },
}

# ==========================================================
# Same BRFSS color style
# ==========================================================
color_matrix = np.array([
    [0, 1],
    [2, 3]
])

custom_cmap = ListedColormap([
    "#6E8B3D",  # TN
    "#6F8FB5",  # FP
    "#B08CC6",  # FN
    "#C07A3D",  # TP
])

# ==========================================================
# Create 2x3 layout
# ==========================================================
fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)

axes = axes.flatten()

# ==========================================================
# Plot each model
# ==========================================================
for idx, (model_name, values) in enumerate(models.items()):

    ax = axes[idx]

    cm_values = np.array([
        [values["TN"], values["FP"]],
        [values["FN"], values["TP"]],
    ])

    # draw color blocks
    ax.imshow(
        color_matrix,
        cmap=custom_cmap,
        aspect="equal"
    )

    # ------------------------------------------------------
    # BIGGER MODEL TITLE
    # ------------------------------------------------------
    ax.set_title(
        model_name,
        fontsize=16,
        fontweight="bold",
        pad=10
    )

    # ------------------------------------------------------
    # Bigger axis text
    # ------------------------------------------------------
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Pred 0", "Pred 1"],
        fontsize=11,
        fontweight="bold"
    )

    ax.set_yticklabels(
        ["True 0", "True 1"],
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel(
        "True Label",
        fontsize=11,
        fontweight="bold"
    )

    # ------------------------------------------------------
    # MUCH BIGGER NUMBERS
    # ------------------------------------------------------
    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm_values[i, j]:,}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="black"
            )

    # ------------------------------------------------------
    # White grid borders
    # ------------------------------------------------------
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=3
    )

    ax.tick_params(which="minor", bottom=False, left=False)

# ==========================================================
# Remove empty subplot
# ==========================================================
fig.delaxes(axes[5])

# ==========================================================
# Main title
# ==========================================================
fig.suptitle(
    "Pooled Confusion Matrix Comparison on NHANES External Validation Cohort at Threshold = 0.5",
    fontsize=16,
    y=0.98
)

# ==========================================================
# Better spacing
# ==========================================================
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ==========================================================
# Save figure
# ==========================================================
plt.savefig(
    OUT_PATH,
    dpi=700,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved figure to:\n{OUT_PATH}")