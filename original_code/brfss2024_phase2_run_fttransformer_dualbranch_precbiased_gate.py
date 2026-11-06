# brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate_PHASE1SEL_ABLATIONPATCH.py
# ---------------------------------------------------------------------
# Phase 2 (FINAL, PHASE-1 SELECTED FEATURES): Dual-Branch FT-Transformer + Precision-Biased Gate Fusion
#
# PHASE-1 SELECTED FEATURES:
#   - Uses Phase-1 selected feature list from --phase1_features_json
#   - Applies numeric-only + missing handling + drop-all-NaN + drop-constant first,
#     then filters to Phase-1 selected features present in X.
#
# ABLATION PATCH (Q1 mandatory):
#   - Saves pooled OOF component probabilities: p_rec, p_prec, g, p_gate_only, p_final
#   - Generates Ablation Study Table (Table X) as CSV:
#       ablation/ablation_results.csv (F1, Precision, Recall, BalAcc, AUPRC, FP, FN)
#       ablation/ablation_confmats.csv (TN, FP, FN, TP)
#       ablation/ablation_oof_components.csv (y_true, p_rec, p_prec, g, p_gate_only, p_final)
#
# Notes:
#   - Uses the SAME veto/rescue rules as your final model:
#         veto if p_prec < veto_v -> force 0
#         rescue if p_rec > rescue_rhi -> force 1

import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================
# RUN-BUTTON MODE DEFAULTS
# =========================
DEFAULT_DATA_PATH = r"C:\Users\<user>\Downloads\Dataset\BRFSS2024\archive (1)\main_data_brfss_2024.XPT"
DEFAULT_TARGET = "_MICHD"
DEFAULT_LEAKAGE_COLS = ["CVDINFR4", "CVDCRHD4"]

DEFAULT_PHASE1_FEATURES_JSON = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase1_feature_fusion\final_features_allfolds.json"
DEFAULT_PHASE1_STABILITY_CSV = r""  # unused here; kept for compatibility/logging only

DEFAULT_OUT_DIR = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate_selectedfeatures"

# CV
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
DEFAULT_MIN_FOLDS_KEEP = 1

# Training
DEFAULT_MAX_EPOCHS = 30
DEFAULT_PATIENCE = 6
DEFAULT_VAL_FRACTION = 0.10

DEFAULT_BATCH_SIZE = 1024
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-5

# Model
DEFAULT_D_EMBED = 48
DEFAULT_N_HEADS = 4
DEFAULT_N_LAYERS = 2
DEFAULT_FFN_MULT = 4
DEFAULT_DROPOUT = 0.08

# Inference batching
DEFAULT_VAL_BATCH_SIZE = 512
DEFAULT_USE_AMP = True

# Scaling
DEFAULT_SCALER = "standard"  # "standard" or "robust"

# Precision head FP-strict knobs (FINAL)
DEFAULT_PREC_NEG_WEIGHT = 4.0
DEFAULT_ALPHA_POS = 1.30
DEFAULT_USE_FOCAL = 0
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_LABEL_SMOOTH = 0.00

# Optional inference sharpening for precision head
DEFAULT_PREC_TEMPERATURE = 1.0

# GateNet (tiny and fast)
DEFAULT_GATE_EPOCHS = 6
DEFAULT_GATE_LR = 5e-3
DEFAULT_GATE_WEIGHT_DECAY = 0.0
DEFAULT_GATE_BATCH_SIZE = 4096
DEFAULT_GATE_HIDDEN = 16
DEFAULT_GATE_BIAS = 0.9

# FINAL operating point
DEFAULT_VETO_V = 0.20
DEFAULT_RESCUE_RHI = 0.98


# =========================
# Utilities
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
    return y_raw.replace({2: 0})

def make_scaler(kind: str):
    kind = (kind or "").lower().strip()
    if kind == "robust":
        return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    return StandardScaler(with_mean=True, with_std=True)

def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(float(n_neg / max(1, n_pos)), dtype=torch.float32)

def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_phase1_features(features_json_path: str) -> List[str]:
    """
    Supports Phase-1 JSON saved as:
      - list: ["feat1","feat2",...]
      - dict: {"fold_1":[...], "fold_2":[...], ...} or { "1":[...], ... }
    Returns UNION of all features in the JSON.
    """
    if not features_json_path or not os.path.exists(features_json_path):
        raise FileNotFoundError(f"Phase-1 features JSON not found: {features_json_path}")

    with open(features_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        feats = [str(x) for x in obj]
        return sorted(list(dict.fromkeys(feats)))

    if isinstance(obj, dict):
        all_feats = []
        for _, v in obj.items():
            if isinstance(v, list):
                all_feats.extend([str(x) for x in v])
        all_feats = [str(x) for x in all_feats]
        return sorted(list(dict.fromkeys(all_feats)))

    raise ValueError(f"Unsupported Phase-1 JSON format in: {features_json_path}")

def apply_phase1_feature_filter(X: pd.DataFrame, phase1_features_json: str) -> pd.DataFrame:
    feats = load_phase1_features(phase1_features_json)
    feats_in_X = [f for f in feats if f in X.columns]
    if len(feats_in_X) == 0:
        raise ValueError(
            "Phase-1 features loaded but none exist in X columns. "
            "Check leakage-drop / numeric-only conversion."
        )
    missing = sorted(set(feats) - set(feats_in_X))
    if missing:
        print(f"[WARN] {len(missing)} Phase-1 features not in current X (dropped/absent). Example: {missing[:10]}")
    print(f"[INFO] Phase-1 selected features kept: {len(feats_in_X)} / {len(feats)} (present / listed)")
    return X[feats_in_X].copy()


# =========================
# FT-Transformer blocks
# =========================
class TransformerBlock(nn.Module):
    def __init__(self, d_embed: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embed)
        self.attn = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_embed)
        self.ffn = nn.Sequential(
            nn.Linear(d_embed, ffn_mult * d_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_embed, d_embed),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_embed: int, n_heads: int, n_layers: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed

        self.feat_embed = nn.Parameter(torch.randn(n_features, d_embed) * 0.02)
        self.feat_bias = nn.Parameter(torch.zeros(n_features, d_embed))

        self.blocks = nn.ModuleList([TransformerBlock(d_embed, n_heads, ffn_mult, dropout) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def tokenize(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num.unsqueeze(-1) * self.feat_embed.unsqueeze(0) + self.feat_bias.unsqueeze(0)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        return self.ln_final(x)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenize(x_num)
        enc = self.encode(tokens)
        pooled = enc.mean(dim=1)
        pooled = self.dropout(pooled)
        return pooled

class ProbHead(nn.Module):
    def __init__(self, d_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)

class FTTransformerModel(nn.Module):
    def __init__(self, n_features: int, d_embed: int, n_heads: int, n_layers: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.backbone = FTTransformer(n_features, d_embed, n_heads, n_layers, ffn_mult, dropout)
        self.head = ProbHead(d_embed, dropout)

    def forward_logits(self, x_num: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_num)
        return self.head(h)


# =========================
# Losses
# =========================
class FocalBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        w = (1 - pt).pow(self.gamma)
        return (w * bce).mean()

class NegWeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, w_neg: float):
        super().__init__()
        self.w_neg = float(w_neg)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.bce(logits, targets)
        w = torch.ones_like(base)
        w = torch.where(targets < 0.5, w * self.w_neg, w)
        return (w * base).mean()


# =========================
# Metrics helpers
# =========================
@dataclass
class FoldMetrics:
    fold: int
    acc_050: float
    balacc_050: float
    precision_050: float
    recall_050: float
    f1_050: float
    auroc: float
    auprc: float
    best_thr_f1: float
    f1_best: float
    best_val_auprc_rec: float
    best_val_auprc_prec: float
    best_epoch_rec: int
    best_epoch_prec: int
    gate_score_proxy: float
    gate_mean: float
    veto_frac: float
    rescue_frac: float
    veto_v: float
    rescue_rhi: float

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        f1v = f1_score(y_true, y_hat, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_thr = float(t)
    return best_thr, float(best_f1)

def compute_metrics_at_050(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_hat = (y_prob >= 0.50).astype(int)
    acc = float(accuracy_score(y_true, y_hat))
    balacc = float(balanced_accuracy_score(y_true, y_hat))
    prec = float(precision_score(y_true, y_hat, zero_division=0))
    rec = float(recall_score(y_true, y_hat, zero_division=0))
    f1v = float(f1_score(y_true, y_hat, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")
    return {"acc": acc, "balacc": balacc, "precision": prec, "recall": rec, "f1": f1v, "auroc": auroc, "auprc": auprc}

def fmt_metrics(m: dict) -> str:
    return (
        f"ACC={m['acc']:.4f} | BalACC={m['balacc']:.4f} | "
        f"P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f} | "
        f"AUROC={m['auroc']:.4f} | AUPRC={m['auprc']:.4f}"
    )

@torch.no_grad()
def predict_logits_batched(model: FTTransformerModel, X_np: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    n = X_np.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_np[i:i + batch_size]).to(device, non_blocking=True)
        logits = model.forward_logits(xb)
        out[i:i + batch_size] = logits.detach().cpu().numpy().astype(np.float32)
    return out

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))

def logits_to_probs_with_temp(logits: np.ndarray, T: float) -> np.ndarray:
    T = float(T)
    if T <= 0:
        T = 1.0
    return sigmoid_np(logits / T)


# =========================
# GateNet
# =========================
class GateNet(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def build_gate_features(p_prec: np.ndarray, p_rec: np.ndarray) -> np.ndarray:
    d = np.abs(p_prec - p_rec)
    return np.stack([p_prec, p_rec, d], axis=1).astype(np.float32)

def train_gatenet(
    Xg_tr: np.ndarray, yg_tr: np.ndarray,
    Xg_val: np.ndarray, yg_val: np.ndarray,
    gate_hidden: int,
    gate_epochs: int,
    gate_lr: float,
    gate_wd: float,
    gate_batch_size: int,
    device: str,
    use_amp: bool,
) -> Tuple[GateNet, float]:
    gate = GateNet(hidden=gate_hidden).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=gate_lr, weight_decay=gate_wd)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
    bce = nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(torch.from_numpy(Xg_tr), torch.from_numpy(yg_tr.astype(np.float32)))
    va_ds = TensorDataset(torch.from_numpy(Xg_val), torch.from_numpy(yg_val.astype(np.float32)))
    tr_ld = DataLoader(tr_ds, batch_size=gate_batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    va_ld = DataLoader(va_ds, batch_size=gate_batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    best = -1.0
    best_state = None

    for _ in range(1, gate_epochs + 1):
        gate.train()
        for xb, yb in tr_ld:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = gate(xb)
                    loss = bce(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                logits = gate(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()

        gate.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                gl = gate(xb)
                g = torch.sigmoid(gl)
                correct += torch.mean(1.0 - torch.abs(g - yb)).item()
                total += 1.0
        score = correct / max(1.0, total)

        if score > best + 1e-6:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}

    if best_state is not None:
        gate.load_state_dict(best_state)

    return gate, float(best)


# =========================
# Training (branch)
# =========================
def train_one_branch(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_features: int,
    args,
    device: str,
    use_amp: bool,
    pos_weight_override: Optional[float] = None,
    is_precision_head: bool = False,
) -> Tuple[FTTransformerModel, float, int]:

    y_tr_f = y_tr.astype(np.float32)
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_f))
    tr_ld = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=(device == "cuda")
    )

    model = FTTransformerModel(
        n_features=n_features,
        d_embed=args.d_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    ).to(device)

    if pos_weight_override is None:
        pw = compute_pos_weight(y_tr.astype(int)).to(device)
    else:
        pw = torch.tensor(float(pos_weight_override), dtype=torch.float32, device=device)

    if is_precision_head and args.prec_neg_weight > 1.0:
        criterion = NegWeightedBCEWithLogits(pos_weight=pw, w_neg=args.prec_neg_weight)
    else:
        if args.use_focal:
            criterion = FocalBCEWithLogits(pos_weight=pw, gamma=args.focal_gamma)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auprc = -1.0
    best_epoch = 0
    best_state = None
    bad = 0

    for ep in range(1, args.max_epochs + 1):
        model.train()
        for xb, yb in tr_ld:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if args.label_smooth > 0:
                yb = yb * (1.0 - args.label_smooth) + 0.5 * args.label_smooth

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model.forward_logits(xb)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optim)
                scaler_amp.update()
            else:
                logits = model.forward_logits(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

        val_logits = predict_logits_batched(model, X_val, device, batch_size=args.val_batch_size)
        val_prob = logits_to_probs_with_temp(val_logits, T=1.0)
        val_auprc = float(average_precision_score(y_val, val_prob))

        if val_auprc > best_val_auprc + 1e-6:
            best_val_auprc = val_auprc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, float(best_val_auprc), int(best_epoch)


# =========================
# Fusion + Ablation helpers
# =========================
@torch.no_grad()
def gatenet_predict_g_numpy(
    gate: GateNet,
    p_prec: np.ndarray,
    p_rec: np.ndarray,
    device: str,
    gate_bias: float,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, float]:
    gate.eval()
    n = len(p_rec)
    g_out = np.zeros(n, dtype=np.float32)
    Xg = build_gate_features(p_prec, p_rec)

    mean_g = 0.0
    cnt = 0
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        xb = torch.from_numpy(Xg[sl]).to(device, non_blocking=True)
        gl = gate(xb) + float(gate_bias)
        g = torch.sigmoid(gl).detach().cpu().numpy().astype(np.float32)
        g_out[sl] = g
        mean_g += float(g.sum())
        cnt += int(g.shape[0])

    mean_gate = float(mean_g / max(1, cnt))
    return g_out, mean_gate

def apply_veto_rescue_numpy(
    p_in: np.ndarray,
    p_prec: np.ndarray,
    p_rec: np.ndarray,
    veto_v: float,
    rescue_rhi: float,
) -> Tuple[np.ndarray, float, float]:
    p = p_in.copy().astype(np.float32)
    veto = (p_prec < float(veto_v))
    rescue = (p_rec > float(rescue_rhi))
    veto_frac = float(np.mean(veto))
    rescue_frac = float(np.mean(rescue))
    p[veto] = 0.0
    p[rescue] = 1.0
    return p, veto_frac, rescue_frac


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--target", type=str, default=DEFAULT_TARGET)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    # PHASE-1 selected features (USED here)
    p.add_argument("--phase1_features_json", type=str, default=DEFAULT_PHASE1_FEATURES_JSON)
    p.add_argument("--phase1_stability_csv", type=str, default=DEFAULT_PHASE1_STABILITY_CSV)  # unused
    p.add_argument("--min_folds_keep", type=int, default=DEFAULT_MIN_FOLDS_KEEP)              # unused

    p.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)

    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    p.add_argument("--d_embed", type=int, default=DEFAULT_D_EMBED)
    p.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--ffn_mult", type=int, default=DEFAULT_FFN_MULT)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    p.add_argument("--val_batch_size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    p.add_argument("--use_amp", type=int, default=int(DEFAULT_USE_AMP))

    p.add_argument("--scaler", type=str, default=DEFAULT_SCALER, choices=["standard", "robust"])
    p.add_argument("--leakage_cols", type=str, default=",".join(DEFAULT_LEAKAGE_COLS))

    p.add_argument("--alpha_pos", type=float, default=DEFAULT_ALPHA_POS)
    p.add_argument("--prec_neg_weight", type=float, default=DEFAULT_PREC_NEG_WEIGHT)
    p.add_argument("--prec_temperature", type=float, default=DEFAULT_PREC_TEMPERATURE)
    p.add_argument("--use_focal", type=int, default=DEFAULT_USE_FOCAL)
    p.add_argument("--focal_gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    p.add_argument("--label_smooth", type=float, default=DEFAULT_LABEL_SMOOTH)

    p.add_argument("--gate_epochs", type=int, default=DEFAULT_GATE_EPOCHS)
    p.add_argument("--gate_lr", type=float, default=DEFAULT_GATE_LR)
    p.add_argument("--gate_weight_decay", type=float, default=DEFAULT_GATE_WEIGHT_DECAY)
    p.add_argument("--gate_batch_size", type=int, default=DEFAULT_GATE_BATCH_SIZE)
    p.add_argument("--gate_hidden", type=int, default=DEFAULT_GATE_HIDDEN)
    p.add_argument("--gate_bias", type=float, default=DEFAULT_GATE_BIAS)

    p.add_argument("--veto_v", type=float, default=DEFAULT_VETO_V)
    p.add_argument("--rescue_rhi", type=float, default=DEFAULT_RESCUE_RHI)

    return p.parse_args()


# =========================
# Main
# =========================
def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    seed_all(args.seed)

    leakage_cols = [c.strip() for c in args.leakage_cols.split(",") if c.strip()]
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.use_amp) and (device == "cuda")

    print("=======================================================================")
    print("Phase-2 FINAL (PHASE-1 FEATURES): Dual-Branch FT-Transformer + Precision-Biased Gate Fusion")
    print("=======================================================================")
    print(f"[INFO] Device: {device} | AMP: {use_amp} | dropout={args.dropout}")
    print(f"[INFO] CV: n_splits={args.n_splits} | seed={args.seed} | val_fraction={args.val_fraction}")
    print(f"[INFO] Scaler: {args.scaler}")
    print(f"[INFO] Precision strict: prec_neg_weight={args.prec_neg_weight} | alpha_pos={args.alpha_pos} | prec_T={args.prec_temperature}")
    print(f"[INFO] Gate: epochs={args.gate_epochs} | hidden={args.gate_hidden} | gate_bias={args.gate_bias}")
    print(f"[INFO] FINAL veto/rescue: veto_v={args.veto_v} | rescue_rhi={args.rescue_rhi}")
    print(f"[INFO] Phase-1 JSON (USED): {args.phase1_features_json}")
    print(f"[INFO] Output dir: {args.out_dir}")
    print("-------------------------------------------------------")

    # Load BRFSS
    df = pd.read_sas(args.data_path, format="xport")
    df.columns = [c.decode() if isinstance(c, bytes) else str(c) for c in df.columns]

    # Drop leakage columns if present
    leakage_present = [c for c in leakage_cols if c in df.columns]
    if leakage_present:
        df = df.drop(columns=leakage_present)

    # Target
    y_raw = df[args.target]
    y_mapped = decode_brfss_target(y_raw)
    mask = y_mapped.isin([0, 1])
    df = df.loc[mask].copy()
    y = y_mapped.loc[mask].astype(int).to_numpy()

    # Features (numeric-only + missing handling + drop all-NaN/constant)
    X = df.drop(columns=[args.target])
    X = safe_numeric_only(X)
    X = brfss_missing_to_nan(X)
    X = drop_all_nan_and_constant(X)

    # PHASE-1 FEATURE FILTER (KEY CHANGE)
    X = apply_phase1_feature_filter(X, args.phase1_features_json)

    print(f"[INFO] Samples: {X.shape[0]} | Phase-1 selected features used: {X.shape[1]}")
    print("[INFO] Class distribution:", dict(pd.Series(y).value_counts()))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # Final pooled OOF
    oof_prob = np.zeros(len(y), dtype=np.float32)

    # ABLATION pooled OOF components
    oof_p_rec = np.zeros(len(y), dtype=np.float32)
    oof_p_prec = np.zeros(len(y), dtype=np.float32)
    oof_g = np.zeros(len(y), dtype=np.float32)
    oof_gate_only = np.zeros(len(y), dtype=np.float32)

    fold_metrics: List[FoldMetrics] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_df = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va_df = X.iloc[va_idx]
        y_va = y[va_idx]

        # Impute + scale (train fold only)
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr_df).astype(np.float32)
        X_va_imp = imp.transform(X_va_df).astype(np.float32)

        scaler = make_scaler(args.scaler)
        X_tr_scaled = scaler.fit_transform(X_tr_imp).astype(np.float32)
        X_va_scaled = scaler.transform(X_va_imp).astype(np.float32)

        # Inner split (early stopping + gate training)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed + fold)
        inner_tr_idx, inner_val_idx = next(sss.split(X_tr_scaled, y_tr))

        X_inner_tr = X_tr_scaled[inner_tr_idx]
        y_inner_tr = y_tr[inner_tr_idx]
        X_inner_val = X_tr_scaled[inner_val_idx]
        y_inner_val = y_tr[inner_val_idx]

        # Recall head
        rec_model, best_auprc_rec, best_ep_rec = train_one_branch(
            X_tr=X_inner_tr, y_tr=y_inner_tr,
            X_val=X_inner_val, y_val=y_inner_val,
            n_features=X_tr_scaled.shape[1],
            args=args, device=device, use_amp=use_amp,
            pos_weight_override=None,
            is_precision_head=False
        )

        # Precision head
        base_pw = float(compute_pos_weight(y_inner_tr).cpu().item())
        pos_weight_prec = max(1.0, base_pw / max(1e-6, args.alpha_pos))

        prec_model, best_auprc_prec, best_ep_prec = train_one_branch(
            X_tr=X_inner_tr, y_tr=y_inner_tr,
            X_val=X_inner_val, y_val=y_inner_val,
            n_features=X_tr_scaled.shape[1],
            args=args, device=device, use_amp=use_amp,
            pos_weight_override=pos_weight_prec,
            is_precision_head=True
        )

        # INNER-VAL probs for gate training
        rec_logits_iv = predict_logits_batched(rec_model, X_inner_val, device, args.val_batch_size)
        prec_logits_iv = predict_logits_batched(prec_model, X_inner_val, device, args.val_batch_size)

        p_rec_iv = logits_to_probs_with_temp(rec_logits_iv, T=1.0)
        p_prec_iv = logits_to_probs_with_temp(prec_logits_iv, T=float(args.prec_temperature))
        y_iv = y_inner_val.astype(int)

        # Gate supervision: prefer precision head when it is "more correct" than recall head
        yhat_rec = (p_rec_iv >= 0.5).astype(int)
        yhat_prec = (p_prec_iv >= 0.5).astype(int)
        rec_correct = (yhat_rec == y_iv).astype(np.float32)
        prec_correct = (yhat_prec == y_iv).astype(np.float32)
        gate_y = ((prec_correct > rec_correct) + 0.0).astype(np.float32)

        Xg_all = build_gate_features(p_prec_iv, p_rec_iv)
        sss_g = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed + 1000 + fold)
        g_tr_idx, g_va_idx = next(sss_g.split(Xg_all, gate_y.astype(int)))

        gate, gate_score = train_gatenet(
            Xg_tr=Xg_all[g_tr_idx], yg_tr=gate_y[g_tr_idx],
            Xg_val=Xg_all[g_va_idx], yg_val=gate_y[g_va_idx],
            gate_hidden=args.gate_hidden,
            gate_epochs=args.gate_epochs,
            gate_lr=args.gate_lr,
            gate_wd=args.gate_weight_decay,
            gate_batch_size=args.gate_batch_size,
            device=device,
            use_amp=use_amp
        )

        # OUTER fold inference
        rec_logits = predict_logits_batched(rec_model, X_va_scaled, device, args.val_batch_size)
        prec_logits = predict_logits_batched(prec_model, X_va_scaled, device, args.val_batch_size)

        p_rec = logits_to_probs_with_temp(rec_logits, T=1.0)
        p_prec = logits_to_probs_with_temp(prec_logits, T=float(args.prec_temperature))

        # Gate weights for ALL outer-val samples
        g_val, gate_mean = gatenet_predict_g_numpy(
            gate=gate,
            p_prec=p_prec,
            p_rec=p_rec,
            device=device,
            gate_bias=float(args.gate_bias),
            batch_size=8192
        )

        # Gate-only (no veto/rescue)
        p_gate = (g_val * p_prec + (1.0 - g_val) * p_rec).astype(np.float32)

        # Final (with veto/rescue)
        p_fuse, veto_frac, rescue_frac = apply_veto_rescue_numpy(
            p_in=p_gate,
            p_prec=p_prec,
            p_rec=p_rec,
            veto_v=float(args.veto_v),
            rescue_rhi=float(args.rescue_rhi)
        )

        # Save pooled OOF
        oof_prob[va_idx] = p_fuse.astype(np.float32)

        # Save pooled OOF components (ablation)
        oof_p_rec[va_idx] = p_rec.astype(np.float32)
        oof_p_prec[va_idx] = p_prec.astype(np.float32)
        oof_g[va_idx] = g_val.astype(np.float32)
        oof_gate_only[va_idx] = p_gate.astype(np.float32)

        # Metrics on OUTER fold
        m_rec = compute_metrics_at_050(y_va, p_rec)
        m_prec = compute_metrics_at_050(y_va, p_prec)
        m_fuse = compute_metrics_at_050(y_va, p_fuse)

        fp_out = int(np.sum((y_va == 0) & ((p_fuse >= 0.5).astype(int) == 1)))
        fn_out = int(np.sum((y_va == 1) & ((p_fuse >= 0.5).astype(int) == 0)))

        print(f"\n==================== FOLD {fold}/{args.n_splits} ====================")
        print(f"[INFO] Early-stop AUPRC: recall={best_auprc_rec:.4f} (ep={best_ep_rec}), precision={best_auprc_prec:.4f} (ep={best_ep_prec})")
        print(f"[INFO] Gate score (proxy): {gate_score:.4f} | gate_mean(after bias)={gate_mean:.4f}")
        print(f"[INFO] veto_v={args.veto_v:.3f} | rescue_rhi={args.rescue_rhi:.3f} | veto_frac={veto_frac:.4f} | rescue_frac={rescue_frac:.4f}")
        print(f"[OUTER] FP={fp_out} | FN={fn_out}")
        print("---------------------------------------------------------------------")
        print("[RECALL HEAD ] " + fmt_metrics(m_rec))
        print("[PREC  HEAD ] " + fmt_metrics(m_prec))
        print("[GATED  FUSE] " + fmt_metrics(m_fuse))
        print("=====================================================================\n")

        thr, f1b = best_f1_threshold(y_va, p_fuse)

        fold_metrics.append(FoldMetrics(
            fold=fold,
            acc_050=float(m_fuse["acc"]),
            balacc_050=float(m_fuse["balacc"]),
            precision_050=float(m_fuse["precision"]),
            recall_050=float(m_fuse["recall"]),
            f1_050=float(m_fuse["f1"]),
            auroc=float(m_fuse["auroc"]),
            auprc=float(m_fuse["auprc"]),
            best_thr_f1=float(thr),
            f1_best=float(f1b),
            best_val_auprc_rec=float(best_auprc_rec),
            best_val_auprc_prec=float(best_auprc_prec),
            best_epoch_rec=int(best_ep_rec),
            best_epoch_prec=int(best_ep_prec),
            gate_score_proxy=float(gate_score),
            gate_mean=float(gate_mean),
            veto_frac=float(veto_frac),
            rescue_frac=float(rescue_frac),
            veto_v=float(args.veto_v),
            rescue_rhi=float(args.rescue_rhi),
        ))

        del rec_model, prec_model, gate
        if device == "cuda":
            torch.cuda.empty_cache()

    # =========================
    # Save FINAL outputs
    # =========================
    oof_df = pd.DataFrame({
        "y_true": y.astype(int),
        "p_rec": oof_p_rec.astype(np.float32),
        "p_prec": oof_p_prec.astype(np.float32),
        "g": oof_g.astype(np.float32),
        "p_gate_only": oof_gate_only.astype(np.float32),
        "p_final": oof_prob.astype(np.float32),
        "y_prob": oof_prob.astype(np.float32),  # backward compatibility
    })
    oof_path = os.path.join(args.out_dir, "phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv")
    oof_df.to_csv(oof_path, index=False)

    y_hat = (oof_df["p_final"].values >= 0.5).astype(int)
    cm = confusion_matrix(y, y_hat)
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    cm_path = os.path.join(args.out_dir, "phase2_fttf_dual_precbiased_gate_selectedfeatures_confmat_050.csv")
    cm_df.to_csv(cm_path, index=True)

    met_df = pd.DataFrame([m.__dict__ for m in fold_metrics])
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for c in met_df.columns:
        if c == "fold":
            continue
        mean_row[c] = float(pd.to_numeric(met_df[c], errors="coerce").mean())
        std_row[c] = float(pd.to_numeric(met_df[c], errors="coerce").std(ddof=1))
    met_df2 = pd.concat([met_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    met_path = os.path.join(args.out_dir, "phase2_fttf_dual_precbiased_gate_selectedfeatures_metrics.csv")
    met_df2.to_csv(met_path, index=False)

    summary = {
        "device": device,
        "amp": use_amp,
        "scaler": args.scaler,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_mode": "Phase-1 selected features (JSON)",
        "phase1_features_json_used": args.phase1_features_json,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "val_fraction": args.val_fraction,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_embed": args.d_embed,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ffn_mult": args.ffn_mult,
        "dropout": args.dropout,
        "precision_strict": {
            "alpha_pos": args.alpha_pos,
            "prec_neg_weight": args.prec_neg_weight,
            "prec_temperature": args.prec_temperature,
        },
        "gate": {
            "epochs": args.gate_epochs,
            "hidden": args.gate_hidden,
            "lr": args.gate_lr,
            "weight_decay": args.gate_weight_decay,
            "gate_bias": args.gate_bias,
            "inputs": ["p_prec", "p_rec", "|diff|"],
        },
        "final_rules": {
            "veto_v": args.veto_v,
            "rescue_rhi": args.rescue_rhi,
            "threshold_eval": 0.50
        },
        "oof_path": oof_path,
        "oof_columns": ["y_true", "p_rec", "p_prec", "g", "p_gate_only", "p_final", "y_prob"],
        "ablation_components_path": os.path.join(args.out_dir, "ablation", "ablation_oof_components.csv"),
        "metrics_path": met_path,
        "confmat_050_path": cm_path,
        "phase1_stability_csv_ignored": args.phase1_stability_csv,
        "min_folds_keep_ignored": int(args.min_folds_keep),
    }
    summary_path = os.path.join(args.out_dir, "phase2_fttf_dual_precbiased_gate_selectedfeatures_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # =========================
    # ABLATION: build Table X from pooled OOF
    # =========================
    def pooled_conf_counts(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.50):
        y_hat2 = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat2, labels=[0, 1]).ravel()
        return int(tn), int(fp), int(fn), int(tp)

    def eval_row(name: str, y_true: np.ndarray, y_prob: np.ndarray):
        m2 = compute_metrics_at_050(y_true, y_prob)
        tn, fp, fn, tp = pooled_conf_counts(y_true, y_prob, thr=0.50)
        return {
            "Variant": name,
            "F1": float(m2["f1"]),
            "Precision": float(m2["precision"]),
            "Recall": float(m2["recall"]),
            "BalAcc": float(m2["balacc"]),
            "AUPRC": float(m2["auprc"]),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "TP": int(tp),
        }

    p_recall_only = oof_p_rec
    p_precision_only = oof_p_prec
    p_simple_avg = 0.5 * (oof_p_rec + oof_p_prec)
    p_gate_only_all = oof_gate_only
    p_final_all = oof_prob

    ablation_rows = [
        eval_row("Recall head only", y.astype(int), p_recall_only),
        eval_row("Precision head only", y.astype(int), p_precision_only),
        eval_row("Simple average fusion", y.astype(int), p_simple_avg),
        eval_row("GateNet only (no veto/rescue)", y.astype(int), p_gate_only_all),
        eval_row("GateNet + veto/rescue (final)", y.astype(int), p_final_all),
    ]
    ablation_df = pd.DataFrame(ablation_rows)

    abl_dir = os.path.join(args.out_dir, "ablation")
    ensure_dir(abl_dir)

    components_path = os.path.join(abl_dir, "ablation_oof_components.csv")
    pd.DataFrame({
        "y_true": y.astype(int),
        "p_rec": oof_p_rec,
        "p_prec": oof_p_prec,
        "g": oof_g,
        "p_gate_only": oof_gate_only,
        "p_final": oof_prob,
    }).to_csv(components_path, index=False)

    results_path = os.path.join(abl_dir, "ablation_results.csv")
    confmats_path = os.path.join(abl_dir, "ablation_confmats.csv")

    ablation_df[["Variant", "F1", "Precision", "Recall", "BalAcc", "AUPRC", "FP", "FN"]].to_csv(results_path, index=False)
    ablation_df[["Variant", "TN", "FP", "FN", "TP"]].to_csv(confmats_path, index=False)

    print(f"[SAVE] {oof_path}")
    print(f"[SAVE] {met_path}")
    print(f"[SAVE] {cm_path}")
    print(f"[SAVE] {summary_path}")
    print(f"[ABLATION SAVE] {components_path}")
    print(f"[ABLATION SAVE] {results_path}")
    print(f"[ABLATION SAVE] {confmats_path}")
    print("\n[ABLATION TABLE PREVIEW]")
    print(ablation_df[["Variant", "F1", "Precision", "Recall", "BalAcc", "AUPRC", "FP", "FN"]].to_string(index=False))
    print("[DONE] FINAL model run (PHASE-1 FEATURES) + ABLATION completed.")


if __name__ == "__main__":
    main()