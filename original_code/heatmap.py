import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your file
df = pd.read_csv(r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate_selectedfeatures\ablation\ablation_oof_components.csv")

y = df["y_true"].values
p_prec = df["p_prec"].values
p_rec = df["p_rec"].values
p_gate = df["p_gate_only"].values

# Grid
v_values = np.arange(0.10, 0.31, 0.02)
r_values = np.arange(0.90, 1.00, 0.01)

from sklearn.metrics import f1_score

results = []

for v in v_values:
    for r in r_values:
        p = p_gate.copy()

        # rules
        p[p_prec < v] = 0
        p[p_rec > r] = 1

        y_pred = (p >= 0.5).astype(int)
        f1 = f1_score(y, y_pred)

        results.append({"v": v, "r": r, "f1": f1})

res_df = pd.DataFrame(results)

# Pivot for heatmap
pivot = res_df.pivot(index="r", columns="v", values="f1")

plt.figure(figsize=(10, 6))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="F1 Score")

plt.xticks(range(len(pivot.columns)), [f"{v:.2f}" for v in pivot.columns], rotation=45)
plt.yticks(range(len(pivot.index)), [f"{r:.2f}" for r in pivot.index])

plt.xlabel("Veto Threshold (v)")
plt.ylabel("Rescue Threshold (r)")
plt.title("Threshold Sensitivity Heatmap (F1)")

plt.tight_layout()
plt.savefig("threshold_heatmap_f1.png", dpi=300)
plt.show()