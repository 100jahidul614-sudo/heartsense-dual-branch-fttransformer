# nhanes_baseline_comparison_models.py
# ------------------------------------------------------------
# NHANES baseline comparison models:
#   1) LightGBM
#   2) Random Forest
#   3) CNN baseline
#
# Uses:
#   - NHANES dataset: nhanes_final_with_target.csv
#   - Target: CVD_TARGET
#   - Phase-1 selected features: final_features_allfolds.json
#   - Leakage columns removed: MCQ160B, MCQ160C, MCQ160D, MCQ160E
#
# Output:
#   - nhanes_baseline_comparison_results.csv
#   - nhanes_baseline_comparison_confmats.csv
#   - OOF probability files for each baseline

import os
import json
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================================
# Paths
# ==========================================================
DATA_PATH = r"C:\Users\<user>\Research center\HeartSense_repro\data\nhanes\nhanes_final_with_target.csv"
TARGET = "CVD_TARGET"

PHASE1_FEATURES_JSON = r"C:\Users\<user>\Research center\HeartSense_repro\runs\nhanes\nhanes_phase1_feature_fusion\final_features_allfolds.json"

OUT_DIR = r"C:\Users\<user>\Research center\HeartSense_repro\runs\nhanes\nhanes_baseline_comparison"

LEAKAGE_COLS = [
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
]

N_SPLITS = 5
SEED = 42

CNN_MAX_EPOCHS = 30
CNN_PATIENCE = 6
CNN_BATCH_SIZE = 512
CNN_LR = 3e-4
CNN_WEIGHT_DECAY = 1e-5
CNN_VAL_FRACTION = 0.10


# ==========================================================
# Utility functions
# ==========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def nhanes_missing_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    missing_codes = {
        7,
        9,
        77,
        99,
        777,
        999,
        7777,
        9999,
        77777,
        99999,
    }

    out = df.copy()
    for code in missing_codes:
        out = out.replace(code, np.nan)

    return out


def drop_all_nan_and_constant(X: pd.DataFrame) -> pd.DataFrame:
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"[INFO] Dropping all-NaN columns: {len(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)

    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[INFO] Dropping constant columns: {len(constant_cols)}")
        X = X.drop(columns=constant_cols)

    return X


def load_phase1_features(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phase-1 feature JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return sorted(list(dict.fromkeys([str(x) for x in obj])))

    if isinstance(obj, dict):
        all_feats = []
        for _, v in obj.items():
            if isinstance(v, list):
                all_feats.extend([str(x) for x in v])
        return sorted(list(dict.fromkeys(all_feats)))

    raise ValueError("Unsupported Phase-1 JSON format.")


def apply_phase1_feature_filter(X: pd.DataFrame, phase1_json: str) -> pd.DataFrame:
    feats = load_phase1_features(phase1_json)
    feats_in_X = [f for f in feats if f in X.columns]

    if len(feats_in_X) == 0:
        raise ValueError("No Phase-1 features found in X. Check paths and columns.")

    missing = sorted(set(feats) - set(feats_in_X))
    if missing:
        print(f"[WARN] Missing Phase-1 features: {len(missing)}. Example: {missing[:10]}")

    print(f"[INFO] Phase-1 selected features used: {len(feats_in_X)} / {len(feats)}")
    return X[feats_in_X].copy()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compute_pos_weight_np(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0:
        return 1.0

    return float(n_neg / max(1, n_pos))


# ==========================================================
# CNN baseline
# ==========================================================
class CNN1DBaseline(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.15):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.head(x).squeeze(-1)
        return x


@torch.no_grad()
def predict_cnn(model, X_np, device, batch_size=512):
    model.eval()

    probs = []

    for i in range(0, len(X_np), batch_size):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)

    return np.concatenate(probs)


def train_cnn_fold(X_train, y_train, X_val, y_val, X_test, seed):
    seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN1DBaseline(n_features=X_train.shape[1]).to(device)

    pos_weight_value = compute_pos_weight_np(y_train)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CNN_LR,
        weight_decay=CNN_WEIGHT_DECAY,
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train.astype(np.float32)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    best_auprc = -1
    best_state = None
    bad = 0

    for epoch in range(1, CNN_MAX_EPOCHS + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_prob = predict_cnn(model, X_val, device, batch_size=CNN_BATCH_SIZE)
        val_auprc = average_precision_score(y_val, val_prob)

        if val_auprc > best_auprc + 1e-6:
            best_auprc = val_auprc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        if bad >= CNN_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_prob = predict_cnn(model, X_test, device, batch_size=CNN_BATCH_SIZE)

    del model

    if device == "cuda":
        torch.cuda.empty_cache()

    return test_prob


# ==========================================================
# Load and prepare dataset
# ==========================================================
def load_dataset():
    print("======================================================")
    print("NHANES Baseline Comparison")
    print("======================================================")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c) for c in df.columns]

    print(f"[INFO] Raw dataset shape: {df.shape}")

    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leakage_present:
        print(f"[INFO] Dropping leakage columns: {leakage_present}")
        df = df.drop(columns=leakage_present)

    df = df[df[TARGET].isin([0, 1])].copy()
    y = df[TARGET].astype(int).to_numpy()

    X = df.drop(columns=[TARGET])
    X = safe_numeric_only(X)

    if "SEQN" in X.columns:
        print("[INFO] Dropping participant ID: SEQN")
        X = X.drop(columns=["SEQN"])

    X = nhanes_missing_to_nan(X)
    X = drop_all_nan_and_constant(X)
    X = apply_phase1_feature_filter(X, PHASE1_FEATURES_JSON)

    print(f"[INFO] Final X shape: {X.shape}")
    print("[INFO] Class distribution:", dict(pd.Series(y).value_counts()))

    return X, y


# ==========================================================
# Main
# ==========================================================
def main():
    warnings.filterwarnings("ignore")
    seed_all(SEED)
    ensure_dir(OUT_DIR)

    X, y = load_dataset()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    oof_lgbm = np.zeros(len(y), dtype=np.float32)
    oof_rf = np.zeros(len(y), dtype=np.float32)
    oof_cnn = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n==================== FOLD {fold}/{N_SPLITS} ====================")

        X_tr_df = X.iloc[tr_idx]
        X_va_df = X.iloc[va_idx]

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Leakage-safe imputation
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_df).astype(np.float32)
        X_va_imp = imputer.transform(X_va_df).astype(np.float32)

        # Scaling for CNN only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp).astype(np.float32)
        X_va_scaled = scaler.transform(X_va_imp).astype(np.float32)

        # Inner split for CNN early stopping
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=CNN_VAL_FRACTION,
            random_state=SEED + fold,
        )

        inner_tr_idx, inner_val_idx = next(sss.split(X_tr_scaled, y_tr))

        X_cnn_train = X_tr_scaled[inner_tr_idx]
        y_cnn_train = y_tr[inner_tr_idx]

        X_cnn_val = X_tr_scaled[inner_val_idx]
        y_cnn_val = y_tr[inner_val_idx]

        # -----------------------------
        # LightGBM
        # -----------------------------
        if HAS_LGBM:
            pos_weight = compute_pos_weight_np(y_tr)

            lgbm = LGBMClassifier(
                objective="binary",
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                class_weight=None,
                scale_pos_weight=pos_weight,
                random_state=SEED + fold,
                n_jobs=-1,
                verbosity=-1,
            )

            lgbm.fit(X_tr_imp, y_tr)
            oof_lgbm[va_idx] = lgbm.predict_proba(X_va_imp)[:, 1].astype(np.float32)

            m_lgbm = compute_metrics(y_va, oof_lgbm[va_idx])
            print(
                "[LightGBM] "
                f"F1={m_lgbm['F1']:.4f} | "
                f"P={m_lgbm['Precision']:.4f} | "
                f"R={m_lgbm['Recall']:.4f} | "
                f"AUROC={m_lgbm['AUROC']:.4f} | "
                f"AUPRC={m_lgbm['AUPRC']:.4f} | "
                f"FP={m_lgbm['FP']} | FN={m_lgbm['FN']}"
            )

        else:
            print("[WARN] LightGBM not installed. Skipping LightGBM.")

        # -----------------------------
        # Random Forest
        # -----------------------------
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED + fold,
            n_jobs=-1,
        )

        rf.fit(X_tr_imp, y_tr)
        oof_rf[va_idx] = rf.predict_proba(X_va_imp)[:, 1].astype(np.float32)

        m_rf = compute_metrics(y_va, oof_rf[va_idx])
        print(
            "[Random Forest] "
            f"F1={m_rf['F1']:.4f} | "
            f"P={m_rf['Precision']:.4f} | "
            f"R={m_rf['Recall']:.4f} | "
            f"AUROC={m_rf['AUROC']:.4f} | "
            f"AUPRC={m_rf['AUPRC']:.4f} | "
            f"FP={m_rf['FP']} | FN={m_rf['FN']}"
        )

        # -----------------------------
        # CNN baseline
        # -----------------------------
        oof_cnn[va_idx] = train_cnn_fold(
            X_train=X_cnn_train,
            y_train=y_cnn_train,
            X_val=X_cnn_val,
            y_val=y_cnn_val,
            X_test=X_va_scaled,
            seed=SEED + fold,
        ).astype(np.float32)

        m_cnn = compute_metrics(y_va, oof_cnn[va_idx])
        print(
            "[CNN] "
            f"F1={m_cnn['F1']:.4f} | "
            f"P={m_cnn['Precision']:.4f} | "
            f"R={m_cnn['Recall']:.4f} | "
            f"AUROC={m_cnn['AUROC']:.4f} | "
            f"AUPRC={m_cnn['AUPRC']:.4f} | "
            f"FP={m_cnn['FP']} | FN={m_cnn['FN']}"
        )

    # ======================================================
    # Final pooled metrics
    # ======================================================
    rows = []

    if HAS_LGBM:
        rows.append({
            "Model": "LightGBM",
            **compute_metrics(y, oof_lgbm),
        })

    rows.append({
        "Model": "Random Forest",
        **compute_metrics(y, oof_rf),
    })

    rows.append({
        "Model": "CNN",
        **compute_metrics(y, oof_cnn),
    })

    results_df = pd.DataFrame(rows)

    results_path = os.path.join(OUT_DIR, "nhanes_baseline_comparison_results.csv")
    conf_path = os.path.join(OUT_DIR, "nhanes_baseline_comparison_confmats.csv")

    results_df[
        [
            "Model",
            "ACC",
            "BalAcc",
            "Precision",
            "Recall",
            "F1",
            "AUROC",
            "AUPRC",
            "FP",
            "FN",
        ]
    ].to_csv(results_path, index=False)

    results_df[
        [
            "Model",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ].to_csv(conf_path, index=False)

    oof_path = os.path.join(OUT_DIR, "nhanes_baseline_oof_probabilities.csv")

    oof_df = pd.DataFrame({
        "y_true": y.astype(int),
        "p_lgbm": oof_lgbm if HAS_LGBM else np.nan,
        "p_rf": oof_rf,
        "p_cnn": oof_cnn,
    })

    oof_df.to_csv(oof_path, index=False)

    print("\n======================================================")
    print("FINAL POOLED BASELINE RESULTS")
    print("======================================================")
    print(
        results_df[
            [
                "Model",
                "ACC",
                "BalAcc",
                "Precision",
                "Recall",
                "F1",
                "AUROC",
                "AUPRC",
                "FP",
                "FN",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVE]", results_path)
    print("[SAVE]", conf_path)
    print("[SAVE]", oof_path)
    print("[DONE] NHANES baseline comparison completed.")


if __name__ == "__main__":
    main()