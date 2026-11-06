import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# =========================
# CONFIG
# =========================
OOF_PATH = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate\phase2_fttf_dual_precbiased_gate_oof.csv"
OUT_DIR  = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\calibration_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

N_BINS = 15
SEED = 42
N_SPLITS = 5

# =========================
# Helpers
# =========================
def clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    p = clip01(p)
    return float(np.mean((p - y_true) ** 2))

def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins.
    ECE = sum_k (n_k/N) * |acc_k - conf_k|
    """
    p = clip01(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(p)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        acc_k = float(np.mean(y_true[mask]))
        conf_k = float(np.mean(p[mask]))
        ece += (n_k / N) * abs(acc_k - conf_k)

    return float(ece)

def crossfit_calibration(y_true: np.ndarray, p_raw: np.ndarray, n_splits: int = 5, seed: int = 42):
    """
    Cross-fitted Platt (logistic) + Isotonic calibration on OOF probabilities.
    Returns calibrated OOF probabilities for both methods.
    """
    y_true = y_true.astype(int)
    p_raw = clip01(p_raw)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    p_platt = np.zeros_like(p_raw, dtype=float)
    p_iso   = np.zeros_like(p_raw, dtype=float)

    for tr_idx, va_idx in skf.split(p_raw.reshape(-1, 1), y_true):
        # Train folds
        p_tr = p_raw[tr_idx]
        y_tr = y_true[tr_idx]

        # Val fold
        p_va = p_raw[va_idx]

        # ---- Platt scaling (logistic regression on 1D probability)
        lr = LogisticRegression(solver="lbfgs", max_iter=500)
        lr.fit(p_tr.reshape(-1, 1), y_tr)
        p_platt[va_idx] = lr.predict_proba(p_va.reshape(-1, 1))[:, 1]

        # ---- Isotonic regression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, y_tr)
        p_iso[va_idx] = iso.predict(p_va)

    return clip01(p_platt), clip01(p_iso)

# =========================
# Load OOF
# =========================
df = pd.read_csv(OOF_PATH)
if not {"y_true", "y_prob"}.issubset(df.columns):
    raise ValueError(f"OOF file must contain y_true and y_prob. Found: {df.columns.tolist()}")

y = df["y_true"].astype(int).to_numpy()
p_raw = df["y_prob"].astype(float).to_numpy()
p_raw = clip01(p_raw)

# =========================
# Cross-fitted calibration
# =========================
p_platt, p_iso = crossfit_calibration(y, p_raw, n_splits=N_SPLITS, seed=SEED)

# =========================
# Metrics
# =========================
rows = []
for name, p in [
    ("Raw", p_raw),
    ("Platt", p_platt),
    ("Isotonic", p_iso),
]:
    rows.append({
        "method": name,
        "brier": brier_score(y, p),
        "ece": ece_score(y, p, n_bins=N_BINS),
    })

met_df = pd.DataFrame(rows).sort_values("brier")
met_path = os.path.join(OUT_DIR, "calibration_metrics_brier_ece.csv")
met_df.to_csv(met_path, index=False)

print("\n[Calibration Metrics]")
print(met_df.to_string(index=False))
print("\n[SAVE]", met_path)

# =========================
# Reliability diagram (calibration plot)
# =========================
plt.figure(figsize=(8, 6))

for name, p, ls in [
    ("Raw", p_raw, "-"),
    ("Platt", p_platt, "--"),
    ("Isotonic", p_iso, "-."),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=N_BINS, strategy="uniform")
    plt.plot(mean_pred, frac_pos, ls, label=name)

# Perfect calibration line
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title(f"Calibration Plot (Reliability Diagram) | bins={N_BINS}")
plt.legend()
plt.grid(True)
plt.tight_layout()

cal_plot_path = os.path.join(OUT_DIR, f"calibration_plot_bins{N_BINS}.png")
plt.savefig(cal_plot_path, dpi=300)
plt.close()
print("[SAVE]", cal_plot_path)

# =========================
# Histogram of predicted probabilities (optional but useful)
# =========================
plt.figure(figsize=(8, 6))
plt.hist(p_raw, bins=50, alpha=0.7, label="Raw")
plt.hist(p_platt, bins=50, alpha=0.5, label="Platt")
plt.hist(p_iso, bins=50, alpha=0.5, label="Isotonic")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities (OOF)")
plt.legend()
plt.grid(True)
plt.tight_layout()

hist_path = os.path.join(OUT_DIR, "probability_histograms.png")
plt.savefig(hist_path, dpi=300)
plt.close()
print("[SAVE]", hist_path)

print("\n✅ Done. Outputs are in:", OUT_DIR)