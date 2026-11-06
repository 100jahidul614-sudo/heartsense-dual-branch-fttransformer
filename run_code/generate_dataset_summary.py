import os
import json
import numpy as np
import pandas as pd

# =========================
# CONFIG (match Phase-2 FINAL)
# =========================
DATA_PATH = r"C:\Users\<user>\Downloads\Dataset\BRFSS2024\archive (1)\main_data_brfss_2024.XPT"
TARGET_COL = "_MICHD"
LEAKAGE_COLS = ["CVDINFR4", "CVDCRHD4"]  # dropped in your Phase-2 pipeline

FEATURES_JSON = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase1_feature_fusion\final_features_allfolds.json"

# BRFSS missing codes (same logic as your pipeline)
MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999, 888, 8888, 88888}

# =========================
# Helpers (aligned with Phase-2)
# =========================
def decode_brfss_target(y_raw: pd.Series) -> pd.Series:
    # BRFSS often uses {1,2} => map 2->0
    return y_raw.replace({2: 0})

def safe_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def brfss_missing_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for code in MISSING_CODES:
        out = out.replace(code, np.nan)
    return out

def drop_all_nan_and_constant(X: pd.DataFrame) -> pd.DataFrame:
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
    return X

def load_xpt(path: str) -> pd.DataFrame:
    import pyreadstat
    df, meta = pyreadstat.read_xport(path, encoding="latin1")
    df.columns = [str(c).strip() for c in df.columns]
    return df

# =========================
# Load data
# =========================
df = load_xpt(DATA_PATH)
print("Raw XPT shape:", df.shape)

# Drop leakage cols if present (as in Phase-2)
drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print("Dropped leakage cols:", drop_cols)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target '{TARGET_COL}' not found. Available CVD-like cols: "
                     f"{[c for c in df.columns if 'CVD' in c.upper() or 'HEART' in c.upper()]}")

y_raw = df[TARGET_COL]
y = decode_brfss_target(y_raw)

mask = y.isin([0, 1])
df = df.loc[mask].copy()
y = y.loc[mask].astype(int)

X = df.drop(columns=[TARGET_COL])
X = safe_numeric_only(X)
X = brfss_missing_to_nan(X)
X = drop_all_nan_and_constant(X)

# Load Phase-1 selected features (final list) and keep only those present
with open(FEATURES_JSON, "r", encoding="utf-8") as f:
    selected_features = json.load(f)

selected_features = [str(c) for c in selected_features if str(c) in X.columns]
X_sel = X[selected_features].copy()

# =========================
# Summary stats
# =========================
N = int(len(y))
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
prev = (n_pos / N) if N > 0 else float("nan")
imb_ratio = (n_neg / max(1, n_pos))

n_features_all = int(X.shape[1])
n_features_sel = int(X_sel.shape[1])

missing_all = float(X.isna().mean().mean())   # average missing ratio over all features
missing_sel = float(X_sel.isna().mean().mean())

# Print results
print("\n===== DATASET SUMMARY (Phase-2 aligned) =====")
print(f"Samples (after target cleaning): {N}")
print(f"Positives (y=1): {n_pos}")
print(f"Negatives (y=0): {n_neg}")
print(f"Prevalence (%): {prev*100:.3f}")
print(f"Imbalance ratio (Neg/Pos): {imb_ratio:.2f}")

print(f"\nNumeric features (after leakage drop + cleaning): {n_features_all}")
print(f"Selected features used by final system (Phase-1 list): {n_features_sel}")

print(f"\nAvg missing ratio (all cleaned numeric): {missing_all*100:.3f}%")
print(f"Avg missing ratio (selected features): {missing_sel*100:.3f}%")

# Save a small CSV for Overleaf table creation
out_dir = os.path.join(os.path.dirname(FEATURES_JSON), "paper_tables")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "dataset_summary_brfss2024_phase2.csv")

summary = pd.DataFrame([{
    "N": N,
    "Positives": n_pos,
    "Negatives": n_neg,
    "Prevalence_%": round(prev*100, 4),
    "Neg_per_Pos": round(imb_ratio, 4),
    "Features_all_numeric": n_features_all,
    "Features_selected": n_features_sel,
    "Missing_avg_all_%": round(missing_all*100, 4),
    "Missing_avg_selected_%": round(missing_sel*100, 4),
}])

summary.to_csv(out_csv, index=False)
print("\n[SAVED]", out_csv)