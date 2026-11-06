import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# FILE PATHS
# =========================================================
proposed_path = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_fttf_dual_precbiased_gate_selectedfeatures\phase2_fttf_dual_precbiased_gate_selectedfeatures_confmat_050.csv"
cnn_path = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures\CNN1D_confmat_050.csv"
lgbm_path = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures\LightGBM_confmat_050.csv"

# =========================================================
# LOAD FUNCTIONS
# =========================================================
def load_proposed_confmat(path):
    """
    Proposed file format:
      Unnamed: 0, pred_0, pred_1
      true_0, TN, FP
      true_1, FN, TP
    """
    df = pd.read_csv(path)
    print("\nLoaded Proposed:")
    print(df)

    fp = int(df.loc[df["Unnamed: 0"] == "true_0", "pred_1"].values[0])
    fn = int(df.loc[df["Unnamed: 0"] == "true_1", "pred_0"].values[0])

    return {"model": "Proposed", "FP": fp, "FN": fn}


def load_standard_confmat(path, model_name):
    """
    Standard file format:
      model, threshold, tn, fp, fn, tp
    """
    df = pd.read_csv(path)
    print(f"\nLoaded {model_name}:")
    print(df)

    fp = int(df["fp"].values[0])
    fn = int(df["fn"].values[0])

    return {"model": model_name, "FP": fp, "FN": fn}


# =========================================================
# LOAD DATA
# =========================================================
data = []

# From actual files
data.append(load_proposed_confmat(proposed_path))
data.append(load_standard_confmat(lgbm_path, "LightGBM"))
data.append(load_standard_confmat(cnn_path, "CNN1D"))

# Simple Average Fusion added manually from manuscript Table 8
data.append({
    "model": "Simple Avg Fusion",
    "FP": 37835,
    "FN": 20464
})

df_plot = pd.DataFrame(data)

print("\nFinal plot data:")
print(df_plot)

# Optional: save the numbers
df_plot.to_csv("fp_fn_plot_data.csv", index=False)

# =========================================================
# PLOT: side-by-side bars
# =========================================================
x = np.arange(len(df_plot))
width = 0.36

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, df_plot["FP"], width, label="False Positives")
plt.bar(x + width/2, df_plot["FN"], width, label="False Negatives")

plt.xticks(x, df_plot["model"], rotation=10)
plt.ylabel("Count")
plt.title("FP/FN Comparison Across Models at Threshold = 0.50")
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig("fp_fn_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()