import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load master CSV
# =========================
path = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_allmodels_allfeatures\master_results_phase2_allfeatures.csv"
df = pd.read_csv(path)

# Basic sanity checks
required_cols = [
    "model",
    "acc_mean", "acc_std",
    "balacc_mean", "balacc_std",
    "precision_mean", "precision_std",
    "recall_mean", "recall_std",
    "f1_mean", "f1_std",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Keep a stable order (CSV order) and ensure strings
df["model"] = df["model"].astype(str)
models = df["model"].tolist()

# Metrics to plot (label, mean_col, std_col)
metrics = [
    ("Accuracy", "acc_mean", "acc_std"),
    ("Balanced Acc", "balacc_mean", "balacc_std"),
    ("Precision", "precision_mean", "precision_std"),
    ("Recall", "recall_mean", "recall_std"),
    ("F1", "f1_mean", "f1_std"),
]

# Convert to numeric to avoid string issues
for _, mcol, scol in metrics:
    df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
    df[scol] = pd.to_numeric(df[scol], errors="coerce")

# =========================
# Plot grouped bars
# =========================
x = np.arange(len(models))
n_metrics = len(metrics)

# Auto width so groups fit nicely within ~80% of spacing
group_width = 0.80
width = group_width / n_metrics

fig, ax = plt.subplots(figsize=(14, 6))

for i, (label, mean_col, std_col) in enumerate(metrics):
    offsets = x - group_width / 2 + (i + 0.5) * width
    ax.bar(
        offsets,
        df[mean_col].values,
        width=width,
        yerr=df[std_col].values,
        capsize=4,
        label=label
    )

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0.0, 1.02)  # change to (0.5, 1.02) if you want zoom-in
ax.legend(title="Metric", loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()

# Save next to the CSV (recommended)
out_dir = os.path.dirname(path)
out_path = os.path.join(out_dir, "figure_grouped_metrics.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"[SAVED] {out_path}")
plt.show()
