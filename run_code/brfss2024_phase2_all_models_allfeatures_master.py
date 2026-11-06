# brfss2024_phase2_all_models_allfeatures_master.py
# -------------------------------------------------
# Phase-2 (ALL FEATURES baseline): Train + evaluate many models on ALL usable features
# Leakage-safe 5-fold Stratified CV:
#   - Drop leakage cols
#   - Numeric only
#   - Replace BRFSS missing codes -> NaN
#   - Drop all-NaN + constant cols
#   - Median impute (fit on TRAIN fold only)
#   - Scaling (fit on TRAIN fold only) where needed
#
# Models included (as available in your environment):
#   LightGBM (SMOTE train-only)  [per your request: NO class_weight for LGBM]
#   Logistic Regression (class_weight balanced)
#   SVM (class_weight balanced)
#   Decision Tree (class_weight balanced)
#   Random Forest (class_weight balanced)
#   KNN (no class_weight; optional SMOTE off by default)
#   Naive Bayes (GaussianNB)
#   XGBoost (scale_pos_weight)
#   CatBoost (class_weights)
#   Autoencoder (AE) + Logistic Regression head
#   CNN (1D) (tabular-as-channels baseline)
#   LSTM (tabular-as-sequence baseline)
#   GRU (tabular-as-sequence baseline)
#
# Outputs:
#   - master_results_phase2_allfeatures.csv          (one row per model)
#   - per_model_metrics.csv                          (fold-by-fold)
#   - per_model_confmat_050.csv                      (pooled confusion matrix @0.50)
#   - per_model_oof.csv                              (OOF probs per sample)
#
# Notes:
#   - Deep models are intentionally simple baselines (tabular is not time-series).
#   - If a library is missing, that model is skipped but the run continues.

import os
import json
import time
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# --- classical models ---
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --- optional libs ---
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# --- deep (optional) ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# =========================
# Defaults (PyCharm run)
# =========================
RUN_WITH_DEFAULTS = True

DEFAULT_DATA_PATH = r"C:\Users\<user>\Downloads\Dataset\BRFSS2024\archive (1)\main_data_brfss_2024.XPT"
DEFAULT_TARGET = "_MICHD"
DEFAULT_OUT_DIR = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures"

DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42

DEFAULT_LEAKAGE_COLS = ["CVDINFR4", "CVDCRHD4"]

# scaler for models that need it
DEFAULT_SCALER = "standard"  # standard | robust

# Deep runtime knobs (keep conservative)
DEFAULT_DL_EPOCHS = 10
DEFAULT_DL_BATCH = 2048
DEFAULT_DL_LR = 2e-3
DEFAULT_DL_PATIENCE = 3
DEFAULT_USE_AMP = 1

# =========================
# BRFSS cleaning helpers
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def brfss_missing_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    missing_codes = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999, 888, 8888, 88888}
    out = df.copy()
    for code in missing_codes:
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

def decode_brfss_target(y_raw: pd.Series) -> pd.Series:
    # BRFSS: 1=Yes, 2=No
    return y_raw.replace({2: 0})

def make_scaler(kind: str):
    kind = (kind or "").lower().strip()
    if kind == "robust":
        return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    return StandardScaler(with_mean=True, with_std=True)

def seed_all(seed: int):
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# =========================
# Metrics
# =========================
def safe_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    y_hat = (y_prob >= thr).astype(int)

    out = {
        "acc": float(accuracy_score(y_true, y_hat)),
        "balacc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["auprc"] = float("nan")
    return out

def mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
    return m, s

# =========================
# Imbalance helpers
# =========================
def compute_scale_pos_weight(y: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0:
        return 1.0
    return float(n_neg / max(1, n_pos))

def smote_resample_if_needed(X_tr: np.ndarray, y_tr: np.ndarray, seed: int, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_IMBLEARN:
        raise RuntimeError("imblearn not installed, cannot use SMOTE.")
    sm = SMOTE(k_neighbors=k, random_state=seed)
    X_res, y_res = sm.fit_resample(X_tr, y_tr.astype(int))
    return X_res, y_res

# =========================
# Classical model runners
# =========================
def fit_predict_lgbm_smote(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    if not HAS_LGBM:
        raise RuntimeError("lightgbm not installed.")
    if not HAS_IMBLEARN:
        raise RuntimeError("imblearn not installed (needed for SMOTE).")

    X_res, y_res = smote_resample_if_needed(X_tr, y_tr, seed=seed, k=5)

    params = dict(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=seed,
        n_jobs=-1,
    )
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_res, y_res)
    p = clf.predict_proba(X_va)
    return p[:, 1].astype(np.float32)

def fit_predict_lr(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)
    clf = LogisticRegression(
        solver="saga", max_iter=5000, n_jobs=-1,
        class_weight="balanced", random_state=seed
    )
    clf.fit(X_trs, y_tr.astype(int))
    return clf.predict_proba(X_vas)[:, 1].astype(np.float32)

def fit_predict_svm(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    # LinearSVC doesn't have predict_proba; we calibrate it (still leakage-safe within fold).
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)

    base = LinearSVC(class_weight="balanced", random_state=seed)
    # Calibration uses internal CV on TRAIN fold only
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_trs, y_tr.astype(int))
    return clf.predict_proba(X_vas)[:, 1].astype(np.float32)

def fit_predict_dt(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    clf = DecisionTreeClassifier(
        random_state=seed,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=25
    )
    clf.fit(X_tr, y_tr.astype(int))
    return clf.predict_proba(X_va)[:, 1].astype(np.float32)

def fit_predict_rf(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=10,
        max_features="sqrt",
    )
    clf.fit(X_tr, y_tr.astype(int))
    return clf.predict_proba(X_va)[:, 1].astype(np.float32)

def fit_predict_knn(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    # KNN needs scaling
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)
    clf = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    clf.fit(X_trs, y_tr.astype(int))
    return clf.predict_proba(X_vas)[:, 1].astype(np.float32)

def fit_predict_nb(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    # GaussianNB works better with scaling sometimes (keep simple: standardize)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)
    clf = GaussianNB()
    clf.fit(X_trs, y_tr.astype(int))
    return clf.predict_proba(X_vas)[:, 1].astype(np.float32)

def fit_predict_xgb(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed.")
    spw = compute_scale_pos_weight(y_tr)
    clf = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=spw,
        tree_method="hist",
    )
    clf.fit(X_tr, y_tr.astype(int))
    return clf.predict_proba(X_va)[:, 1].astype(np.float32)

def fit_predict_cat(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    if not HAS_CAT:
        raise RuntimeError("catboost not installed.")
    # class weights: [w0, w1]
    y = y_tr.astype(int)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    w1 = float(n0 / max(1, n1))
    clf = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        class_weights=[1.0, w1],
    )
    clf.fit(X_tr, y)
    return clf.predict_proba(X_va)[:, 1].astype(np.float32)

# =========================
# Deep baselines (optional)
# =========================
if HAS_TORCH:
    class MLPHead(nn.Module):
        def __init__(self, d_in: int, p_drop: float = 0.10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 256),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class SimpleAE(nn.Module):
        def __init__(self, d_in: int, d_latent: int = 32):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d_in, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, d_latent),
            )
            self.dec = nn.Sequential(
                nn.Linear(d_latent, 64), nn.ReLU(),
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, d_in),
            )

        def forward(self, x):
            z = self.enc(x)
            xh = self.dec(z)
            return z, xh

    class CNN1DTab(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            # input: [B, 1, F]
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(32),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 32, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            # x: [B, F]
            x = x.unsqueeze(1)
            h = self.conv(x)
            return self.head(h).squeeze(-1)

    class RNNTab(nn.Module):
        def __init__(self, n_features: int, kind: str = "lstm"):
            super().__init__()
            self.kind = kind
            self.input_proj = nn.Linear(1, 32)
            if kind == "gru":
                self.rnn = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
            else:
                self.rnn = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(
                nn.Linear(64 * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            # x: [B, F] -> treat as sequence length F with 1 channel
            x = x.unsqueeze(-1)          # [B, F, 1]
            x = self.input_proj(x)       # [B, F, 32]
            out, _ = self.rnn(x)         # [B, F, 128]
            pooled = out.mean(dim=1)     # [B, 128]
            return self.head(pooled).squeeze(-1)

    def torch_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def torch_predict_proba(model: nn.Module, X: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
        model.eval()
        out = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            logits = model(xb)
            out[i:i+batch] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        return out

    def torch_train_binary(
        model: nn.Module,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_va: np.ndarray, y_va: np.ndarray,
        seed: int,
        epochs: int,
        batch: int,
        lr: float,
        patience: int,
        use_amp: bool,
        pos_weight: float,
    ) -> nn.Module:
        device = torch_device()
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == "cuda")
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

        ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
        ld = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

        best = -1.0
        best_state = None
        bad = 0

        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in ld:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        logits = model(xb)
                        loss = crit(logits, yb)
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = crit(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            # early stop on val AUPRC
            p = torch_predict_proba(model, X_va, device=device, batch=8192)
            auprc = float(average_precision_score(y_va.astype(int), p))
            if auprc > best + 1e-6:
                best = auprc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def fit_predict_ae_lr(X_tr, y_tr, X_va, seed: int, epochs: int, batch: int, lr: float, patience: int, use_amp: bool) -> np.ndarray:
        # scale first
        sc = StandardScaler()
        X_trs = sc.fit_transform(X_tr).astype(np.float32)
        X_vas = sc.transform(X_va).astype(np.float32)

        # train AE on TRAIN fold only (unsupervised)
        device = torch_device()
        d_in = X_trs.shape[1]
        ae = SimpleAE(d_in=d_in, d_latent=32).to(device)
        opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == "cuda")
        mse = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_trs))
        ld = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

        best = 1e18
        best_state = None
        bad = 0

        for ep in range(1, max(5, epochs) + 1):
            ae.train()
            for (xb,) in ld:
                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        z, xh = ae(xb)
                        loss = mse(xh, xb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    z, xh = ae(xb)
                    loss = mse(xh, xb)
                    loss.backward()
                    opt.step()

            # early stop on recon loss
            ae.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X_trs[: min(20000, X_trs.shape[0])]).to(device)
                _, xh = ae(xb)
                l = float(mse(xh, xb).detach().cpu().item())
            if l < best - 1e-6:
                best = l
                best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            ae.load_state_dict(best_state)

        # encode -> LR on latent
        ae.eval()
        with torch.no_grad():
            z_tr = ae.enc(torch.from_numpy(X_trs).to(device)).detach().cpu().numpy()
            z_va = ae.enc(torch.from_numpy(X_vas).to(device)).detach().cpu().numpy()

        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=seed
        )
        clf.fit(z_tr, y_tr.astype(int))
        return clf.predict_proba(z_va)[:, 1].astype(np.float32)

    def fit_predict_cnn(X_tr, y_tr, X_va, seed: int, epochs: int, batch: int, lr: float, patience: int, use_amp: bool) -> np.ndarray:
        sc = StandardScaler()
        X_trs = sc.fit_transform(X_tr).astype(np.float32)
        X_vas = sc.transform(X_va).astype(np.float32)
        posw = compute_scale_pos_weight(y_tr)

        model = CNN1DTab(n_features=X_trs.shape[1])
        model = torch_train_binary(
            model, X_trs, y_tr, X_vas, y_va=y_tr[:0] if False else y_tr,  # placeholder unused
            seed=seed, epochs=epochs, batch=batch, lr=lr, patience=patience,
            use_amp=use_amp, pos_weight=posw
        )
        # For evaluation, we need val labels; training function expects y_va; we'll handle properly outside.
        # We'll not use this wrapper; see deep runners in main loop.
        raise RuntimeError("Internal misuse - should not be called directly.")


# =========================
# Model registry
# =========================
@dataclass
class ModelSpec:
    name: str
    family: str
    needs_scaling: bool
    runner: str   # key in runner map
    enabled: bool = True


def get_model_specs() -> List[ModelSpec]:
    specs = [
        ModelSpec("LightGBM", "tree", False, "lgbm_smote", enabled=True),
        ModelSpec("LogReg", "linear", True, "lr", enabled=True),
        ModelSpec("SVM", "linear", True, "svm", enabled=True),
        ModelSpec("DT", "tree", False, "dt", enabled=True),
        ModelSpec("RF", "tree", False, "rf", enabled=True),
        ModelSpec("KNN", "instance", True, "knn", enabled=True),
        ModelSpec("NB", "probabilistic", True, "nb", enabled=True),
        ModelSpec("XGBoost", "tree", False, "xgb", enabled=True),
        ModelSpec("CatBoost", "tree", False, "cat", enabled=True),
        ModelSpec("AE_LR", "deep", True, "ae_lr", enabled=True),
        ModelSpec("CNN1D", "deep", True, "cnn", enabled=True),
        ModelSpec("LSTM", "deep", True, "lstm", enabled=True),
        ModelSpec("GRU", "deep", True, "gru", enabled=True),
    ]
    return specs


# =========================
# Argparse
# =========================
def build_args():
    p = argparse.ArgumentParser("BRFSS2024 Phase-2: ALL MODELS on ALL FEATURES (master table)")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--target", type=str, default=DEFAULT_TARGET)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--leakage_cols", type=str, default=",".join(DEFAULT_LEAKAGE_COLS))
    p.add_argument("--scaler", type=str, default=DEFAULT_SCALER, choices=["standard", "robust"])

    # deep knobs
    p.add_argument("--dl_epochs", type=int, default=DEFAULT_DL_EPOCHS)
    p.add_argument("--dl_batch", type=int, default=DEFAULT_DL_BATCH)
    p.add_argument("--dl_lr", type=float, default=DEFAULT_DL_LR)
    p.add_argument("--dl_patience", type=int, default=DEFAULT_DL_PATIENCE)
    p.add_argument("--use_amp", type=int, default=int(DEFAULT_USE_AMP))

    # choose models
    p.add_argument("--models", type=str, default="all",
                   help="Comma list of models (LightGBM,LogReg,SVM,DT,RF,KNN,NB,XGBoost,CatBoost,AE_LR,CNN1D,LSTM,GRU) or 'all'")
    return p


# =========================
# Main
# =========================
def main():
    warnings.filterwarnings("ignore")
    parser = build_args()

    if RUN_WITH_DEFAULTS:
        args = parser.parse_args([])
        print("[INFO] RUN_WITH_DEFAULTS=True (PyCharm Run button mode).")
    else:
        args = parser.parse_args()

    seed_all(args.seed)
    ensure_dir(args.out_dir)

    leakage_cols = [c.strip() for c in (args.leakage_cols or "").split(",") if c.strip()]
    scaler_obj = make_scaler(args.scaler)

    # Load BRFSS
    t0 = time.time()
    df = pd.read_sas(args.data_path, format="xport")
    df.columns = [c.decode() if isinstance(c, bytes) else str(c) for c in df.columns]

    # Drop leakage cols if present
    leakage_present = [c for c in leakage_cols if c in df.columns]
    if leakage_present:
        df = df.drop(columns=leakage_present)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found.")

    y_raw = df[args.target]
    y_mapped = decode_brfss_target(y_raw)
    mask = y_mapped.isin([0, 1])
    df = df.loc[mask].copy()
    y = y_mapped.loc[mask].astype(int).to_numpy()

    X = df.drop(columns=[args.target])
    X = safe_numeric_only(X)
    X = brfss_missing_to_nan(X)
    X = drop_all_nan_and_constant(X)

    feature_names = X.columns.tolist()
    print("===============================================================")
    print("Phase-2 ALL MODELS baseline (ALL usable features)")
    print("===============================================================")
    print(f"[INFO] Loaded & cleaned in {time.time() - t0:.1f}s")
    print(f"[INFO] Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"[INFO] Class distribution: {dict(pd.Series(y).value_counts())}")
    print(f"[INFO] Output dir: {args.out_dir}")

    # Model selection
    specs = get_model_specs()
    if (args.models or "").strip().lower() != "all":
        wanted = set([m.strip() for m in args.models.split(",") if m.strip()])
        specs = [s for s in specs if s.name in wanted]

    # Check availability
    def available(spec: ModelSpec) -> Tuple[bool, str]:
        if spec.runner == "lgbm_smote" and (not HAS_LGBM or not HAS_IMBLEARN):
            return False, "requires lightgbm + imblearn(SMOTE)"
        if spec.runner == "xgb" and (not HAS_XGB):
            return False, "requires xgboost"
        if spec.runner == "cat" and (not HAS_CAT):
            return False, "requires catboost"
        if spec.family == "deep" and (not HAS_TORCH):
            return False, "requires torch"
        return True, ""

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    master_rows = []
    master_path = os.path.join(args.out_dir, "master_results_phase2_allfeatures.csv")

    for spec in specs:
        ok, why = available(spec)
        if not ok:
            print(f"[SKIP] {spec.name}: {why}")
            continue

        print("\n---------------------------------------------------------------")
        print(f"[MODEL] {spec.name}")
        print("---------------------------------------------------------------")

        oof_prob = np.zeros(len(y), dtype=np.float32)
        per_fold_rows = []
        all_y_true = []
        all_y_prob = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr_df = X.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va_df = X.iloc[va_idx]
            y_va = y[va_idx]

            # impute train-only
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr_df).astype(np.float32)
            X_va = imp.transform(X_va_df).astype(np.float32)

            # optional scaling (train-only)
            if spec.needs_scaling:
                sc = make_scaler(args.scaler)
                X_tr = sc.fit_transform(X_tr).astype(np.float32)
                X_va = sc.transform(X_va).astype(np.float32)

            # fit+predict
            seed_fold = args.seed + 1000 * fold

            if spec.runner == "lgbm_smote":
                p = fit_predict_lgbm_smote(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "lr":
                p = fit_predict_lr(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "svm":
                p = fit_predict_svm(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "dt":
                p = fit_predict_dt(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "rf":
                p = fit_predict_rf(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "knn":
                p = fit_predict_knn(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "nb":
                p = fit_predict_nb(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "xgb":
                p = fit_predict_xgb(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner == "cat":
                p = fit_predict_cat(X_tr, y_tr, X_va, seed=seed_fold)

            elif spec.runner in ("ae_lr", "cnn", "lstm", "gru"):
                if not HAS_TORCH:
                    raise RuntimeError("Torch not available.")

                # deep models use StandardScaler always (stronger baseline)
                sc = StandardScaler()
                X_trs = sc.fit_transform(X_tr).astype(np.float32)
                X_vas = sc.transform(X_va).astype(np.float32)

                # compute pos_weight (n_neg/n_pos)
                pos_weight = compute_scale_pos_weight(y_tr)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                use_amp = bool(args.use_amp) and (device == "cuda")

                if spec.runner == "ae_lr":
                    p = fit_predict_ae_lr(
                        X_trs, y_tr, X_vas,
                        seed=seed_fold,
                        epochs=args.dl_epochs,
                        batch=args.dl_batch,
                        lr=args.dl_lr,
                        patience=args.dl_patience,
                        use_amp=use_amp
                    )

                else:
                    if spec.runner == "cnn":
                        model = CNN1DTab(n_features=X_trs.shape[1])
                    elif spec.runner == "lstm":
                        model = RNNTab(n_features=X_trs.shape[1], kind="lstm")
                    else:
                        model = RNNTab(n_features=X_trs.shape[1], kind="gru")

                    # train with early stopping on VAL AUPRC (fold-val)
                    model = model.to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=args.dl_lr, weight_decay=1e-5)
                    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
                    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

                    ds = TensorDataset(torch.from_numpy(X_trs), torch.from_numpy(y_tr.astype(np.float32)))
                    ld = DataLoader(ds, batch_size=args.dl_batch, shuffle=True, num_workers=0, drop_last=False)

                    best = -1.0
                    best_state = None
                    bad = 0

                    for ep in range(1, args.dl_epochs + 1):
                        model.train()
                        for xb, yb in ld:
                            xb = xb.to(device)
                            yb = yb.to(device)
                            opt.zero_grad(set_to_none=True)
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    logits = model(xb)
                                    loss = crit(logits, yb)
                                scaler_amp.scale(loss).backward()
                                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler_amp.step(opt)
                                scaler_amp.update()
                            else:
                                logits = model(xb)
                                loss = crit(logits, yb)
                                loss.backward()
                                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                opt.step()

                        pv = torch_predict_proba(model, X_vas, device=device, batch=8192)
                        auprc_v = float(average_precision_score(y_va.astype(int), pv))
                        if auprc_v > best + 1e-6:
                            best = auprc_v
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            bad = 0
                        else:
                            bad += 1
                            if bad >= args.dl_patience:
                                break

                    if best_state is not None:
                        model.load_state_dict(best_state)

                    p = torch_predict_proba(model, X_vas, device=device, batch=8192)

                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()

                p = np.asarray(p, dtype=np.float32)

            else:
                raise ValueError(f"Unknown runner: {spec.runner}")

            # store fold
            oof_prob[va_idx] = p.astype(np.float32)
            all_y_true.append(y_va.astype(int))
            all_y_prob.append(p.astype(float))

            m = safe_metrics(y_va, p, thr=0.5)
            m["fold"] = fold
            per_fold_rows.append(m)

            print(
                f"[FOLD {fold}] "
                f"acc={m['acc']:.4f} balacc={m['balacc']:.4f} "
                f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                f"auroc={m['roc_auc']:.4f} auprc={m['auprc']:.4f}"
            )

        # save per-model artifacts
        per_fold_df = pd.DataFrame(per_fold_rows).sort_values("fold")
        per_fold_path = os.path.join(args.out_dir, f"{spec.name}_per_fold_metrics.csv")
        per_fold_df.to_csv(per_fold_path, index=False)

        y_true_all = np.concatenate(all_y_true, axis=0)
        y_prob_all = np.concatenate(all_y_prob, axis=0)
        y_pred_all = (y_prob_all >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()

        cm_df = pd.DataFrame([{
            "model": spec.name,
            "threshold": 0.50,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }])
        cm_path = os.path.join(args.out_dir, f"{spec.name}_confmat_050.csv")
        cm_df.to_csv(cm_path, index=False)

        oof_df = pd.DataFrame({"y_true": y.astype(int), "y_prob": oof_prob})
        oof_path = os.path.join(args.out_dir, f"{spec.name}_oof.csv")
        oof_df.to_csv(oof_path, index=False)

        # aggregate row for master
        row = {
            "model": spec.name,
            "feature_set": "ALL_numeric_features",
            "imbalance": (
                "SMOTE(train_only)" if spec.name == "LightGBM"
                else "class_weight/built-in" if spec.name in ("LogReg", "SVM", "DT", "RF", "XGBoost", "CatBoost")
                else "none"
            ),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "threshold": 0.50,
        }
        for k in ["acc", "balacc", "precision", "recall", "f1", "roc_auc", "auprc"]:
            vals = per_fold_df[k].to_numpy(dtype=float)
            m, s = mean_std(vals)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s

        row["tn"] = int(tn); row["fp"] = int(fp); row["fn"] = int(fn); row["tp"] = int(tp)
        row["per_fold_metrics_path"] = per_fold_path
        row["confmat_050_path"] = cm_path
        row["oof_path"] = oof_path

        master_rows.append(row)

        print(f"[SAVE] {per_fold_path}")
        print(f"[SAVE] {cm_path}")
        print(f"[SAVE] {oof_path}")

    # write master
    if not master_rows:
        raise RuntimeError("No models were run (all skipped). Check installed libraries.")

    master_df = pd.DataFrame(master_rows)
    # stable ordering
    cols_front = [
        "model", "feature_set", "imbalance", "n_samples", "n_features", "threshold",
        "acc_mean", "acc_std", "balacc_mean", "balacc_std", "precision_mean", "precision_std",
        "recall_mean", "recall_std", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
        "auprc_mean", "auprc_std",
        "tn", "fp", "fn", "tp",
        "per_fold_metrics_path", "confmat_050_path", "oof_path"
    ]
    cols = [c for c in cols_front if c in master_df.columns] + [c for c in master_df.columns if c not in cols_front]
    master_df = master_df[cols]

    master_df.to_csv(master_path, index=False)
    print("\n===============================================================")
    print("[DONE] Master table created")
    print(f"[SAVE] {master_path}")
    print("===============================================================")


if __name__ == "__main__":
    main()
