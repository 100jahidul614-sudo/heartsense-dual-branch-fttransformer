import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# =========================
# CONFIG
# =========================
OOF_PATH = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_fttf_dual_precbiased_gate_selectedfeatures\phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv"
OUT_DIR = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\threshold_analysis"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(OOF_PATH)
y_true = df["y_true"].values
y_prob = df["y_prob"].values

thresholds = np.linspace(0.01, 0.99, 99)

precision_list = []
recall_list = []
f1_list = []
fp_list = []
fn_list = []

# =========================
# COMPUTE METRICS
# =========================
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    precision_list.append(precision_score(y_true, y_pred, zero_division=0))
    recall_list.append(recall_score(y_true, y_pred, zero_division=0))
    f1_list.append(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fp_list.append(fp)
    fn_list.append(fn)

# =========================
# PLOT 1: Precision / Recall / F1
# =========================
plt.figure(figsize=(8,6))
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.plot(thresholds, f1_list, label="F1-score")
plt.axvline(0.5, linestyle="--", label="Threshold = 0.50")

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_prf_curve.png"), dpi=300)
plt.close()

# =========================
# PLOT 2: FP and FN vs Threshold
# =========================
plt.figure(figsize=(8,6))
plt.plot(thresholds, fp_list, label="False Positives")
plt.plot(thresholds, fn_list, label="False Negatives")
plt.axvline(0.5, linestyle="--", label="Threshold = 0.50")

plt.xlabel("Decision Threshold")
plt.ylabel("Count")
plt.title("False Positives and False Negatives vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_error_curve.png"), dpi=300)
plt.close()

print("Threshold curves saved successfully.")