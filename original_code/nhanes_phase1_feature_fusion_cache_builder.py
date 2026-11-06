# nhanes_phase1_feature_fusion_cache_builder.py
# ------------------------------------------------
# Phase 1: Leakage-safe multi-selector feature selection + fusion for NHANES
#
# Dataset:
#   - nhanes_final_with_target.csv
#   - Target: CVD_TARGET
#
# Important:
#   - MCQ160B, MCQ160C, MCQ160D, MCQ160E are leakage columns because they were used
#     to construct CVD_TARGET. They MUST be dropped before feature selection.

import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier

# TabNet
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False


# ==========================================================
# RUN-BUTTON MODE
# ==========================================================
RUN_WITH_DEFAULTS = True

DEFAULT_DATA_PATH = r"C:\Users\<user>\Research center\Heartseanse2.0\Dataset2\nhanes_final_with_target.csv"
DEFAULT_TARGET = "CVD_TARGET"
DEFAULT_OUT_DIR = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\nhanes_phase1_feature_fusion"

DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42

# Pearson threshold
DEFAULT_PEARSON_ABS_R_THRESHOLD = 0.08

# NHANES leakage columns used to create the target
DEFAULT_LEAKAGE_COLS = [
    "MCQ160B",  # congestive heart failure
    "MCQ160C",  # coronary heart disease
    "MCQ160D",  # angina
    "MCQ160E",  # heart attack
]

# TabNet training
DEFAULT_TABNET_MAX_EPOCHS = 30
DEFAULT_TABNET_BATCH_SIZE = 2048
DEFAULT_TABNET_VBATCH_SIZE = 512

# Threshold: importance >= alpha * (1/N)
DEFAULT_TABNET_BASELINE_ALPHA = 0.75

# Permutation importance
DEFAULT_PI_REPEATS = 5
DEFAULT_PI_SAMPLE_SIZE = 9000
DEFAULT_PI_N_JOBS = 1

# HGB conservative rule
DEFAULT_HGB_LCB_LAMBDA = 1.7
DEFAULT_HGB_MIN_MARGIN = 2e-4


# ==========================================================
# Utilities
# ==========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def nhanes_missing_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    NHANES commonly uses values such as 7, 9, 77, 99, 777, 999,
    7777, 9999 for refused / don't know / missing depending on variable scale.

    Important:
    This is a broad numeric cleaning rule, similar to the BRFSS cleaning logic.
    It should be applied only to feature columns, not to the final binary target.
    """
    missing_codes = {
        7, 9,
        77, 99,
        777, 999,
        7777, 9999,
        77777, 99999,
    }

    out = df.copy()
    for code in missing_codes:
        out = out.replace(code, np.nan)

    return out


def safe_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def compute_abs_pearson_threshold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    abs_r_threshold: float
) -> List[str]:

    y_series = pd.Series(y_train, index=X_train.index, name="target")
    corrs = X_train.corrwith(y_series)
    corrs = corrs.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    selected = corrs.index[corrs.abs() >= abs_r_threshold].tolist()
    selected_sorted = corrs.loc[selected].abs().sort_values(ascending=False).index.tolist()

    return selected_sorted


# ==========================================================
# TabNet selector
# ==========================================================
def train_tabnet_and_get_thresholded(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    seed: int,
    max_epochs: int,
    batch_size: int,
    virtual_batch_size: int,
    baseline_alpha: float,
) -> List[str]:

    if not HAS_TABNET:
        raise RuntimeError("TabNet is not available. Install pytorch-tabnet and torch.")

    if len(np.unique(y_train)) < 2:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = TabNetClassifier(
        seed=seed,
        device_name=device,
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
    )

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        max_epochs=max_epochs,
        patience=0,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
    )

    importances = getattr(clf, "feature_importances_", None)

    if importances is None:
        raise RuntimeError("TabNet did not expose feature_importances_.")

    imp = pd.Series(importances, index=feature_names).fillna(0.0)

    n = max(1, len(feature_names))
    threshold = float(baseline_alpha) * (1.0 / n)

    selected = imp.index[imp >= threshold].tolist()
    selected_sorted = imp.loc[selected].sort_values(ascending=False).index.tolist()

    return selected_sorted


# ==========================================================
# HGB + permutation importance selector
# ==========================================================
def hgb_permutation_importance_lcb_positive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    seed: int,
    n_repeats: int,
    sample_size: int,
    n_jobs: int,
    lcb_lambda: float,
    min_margin: float,
) -> List[str]:

    if len(np.unique(y_train)) < 2:
        return []

    hgb = HistGradientBoostingClassifier(
        random_state=seed,
        max_depth=None,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=0.0,
        early_stopping=True,
        scoring="roc_auc",
        validation_fraction=0.1,
    )

    hgb.fit(X_train, y_train)

    n = X_train.shape[0]

    if sample_size is not None and sample_size < n:
        rng = np.random.RandomState(seed)

        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]

        n_pos = int(sample_size * (len(idx_pos) / n))
        n_pos = max(1, min(n_pos, len(idx_pos)))

        n_neg = sample_size - n_pos
        n_neg = max(1, min(n_neg, len(idx_neg)))

        samp_pos = rng.choice(idx_pos, size=n_pos, replace=False)
        samp_neg = rng.choice(idx_neg, size=n_neg, replace=False)

        samp_idx = np.concatenate([samp_pos, samp_neg])
        rng.shuffle(samp_idx)

        X_pi = X_train[samp_idx].astype(np.float32, copy=False)
        y_pi = y_train[samp_idx]
    else:
        X_pi = X_train.astype(np.float32, copy=False)
        y_pi = y_train

    pi = permutation_importance(
        hgb,
        X_pi,
        y_pi,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )

    imp_mean = pd.Series(pi.importances_mean, index=feature_names).fillna(0.0)
    imp_std = pd.Series(pi.importances_std, index=feature_names).fillna(0.0)

    score = imp_mean - (float(lcb_lambda) * imp_std)

    selected = score.index[score > float(min_margin)].tolist()
    selected_sorted = score.loc[selected].sort_values(ascending=False).index.tolist()

    return selected_sorted


def fuse_sets_at_least_two(
    a: Set[str],
    b: Set[str],
    c: Set[str]
) -> Tuple[Set[str], Dict[str, Set[str]]]:

    ab = a & b
    ac = a & c
    bc = b & c

    common_ge2 = ab | ac | bc

    unique_a = a - (b | c)
    unique_b = b - (a | c)
    unique_c = c - (a | b)

    final_set = common_ge2 | unique_a | unique_b | unique_c

    details = {
        "common_ge2": common_ge2,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "unique_c": unique_c,
        "ab": ab,
        "ac": ac,
        "bc": bc,
        "abc": a & b & c,
    }

    return final_set, details


# ==========================================================
# Main pipeline
# ==========================================================
def main(
    data_path: str,
    target: str,
    out_dir: str,
    n_splits: int,
    seed: int,
    pearson_abs_r_threshold: float,
    leakage_cols: List[str],
    tabnet_max_epochs: int,
    tabnet_batch_size: int,
    tabnet_vbatch_size: int,
    tabnet_baseline_alpha: float,
    pi_repeats: int,
    hgb_lcb_lambda: float,
    hgb_min_margin: float,
) -> None:

    warnings.filterwarnings("ignore")
    ensure_dir(out_dir)

    print("=======================================================")
    print("NHANES Phase-1: Feature Selection Fusion Cache Builder")
    print("=======================================================")
    print(f"[INFO] Data path: {data_path}")
    print(f"[INFO] Target: {target}")
    print(f"[INFO] Output dir: {out_dir}")
    print("[INFO] Selectors:")
    print(f"  - Pearson: |r| >= {pearson_abs_r_threshold}")
    print(f"  - TabNet : importance >= {tabnet_baseline_alpha}*(1/N_features)")
    print(f"  - HGB+PI : (mu - {hgb_lcb_lambda}*sigma) > {hgb_min_margin}")
    print("[INFO] Per-fold fusion: Common(>=2) + Uniques")
    print("[INFO] Final rule: union across all folds, no stability filtering")
    print("-------------------------------------------------------")

    # =========================
    # Load NHANES CSV
    # =========================
    df = pd.read_csv(data_path)

    df.columns = [str(c) for c in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    print(f"[INFO] Raw dataset shape: {df.shape}")

    # =========================
    # Drop leakage columns
    # =========================
    leakage_present = [c for c in leakage_cols if c in df.columns]

    if leakage_present:
        print(f"[INFO] Dropping leakage columns: {leakage_present}")
        df = df.drop(columns=leakage_present)

    # =========================
    # Validate target
    # =========================
    print("[DEBUG] Raw target value counts:")
    print(df[target].value_counts(dropna=False))

    df = df[df[target].isin([0, 1])].copy()
    y = df[target].astype(int).to_numpy()

    X = df.drop(columns=[target])

    # =========================
    # Numeric features only
    # =========================
    X = safe_numeric_only(X)

    # Do not use participant ID as a feature
    if "SEQN" in X.columns:
        print("[INFO] Dropping participant ID column: SEQN")
        X = X.drop(columns=["SEQN"])

    # =========================
    # NHANES missing-code handling
    # =========================
    X = nhanes_missing_to_nan(X)

    # =========================
    # Drop all-NaN columns
    # =========================
    all_nan_cols = X.columns[X.isna().all()].tolist()

    if all_nan_cols:
        print(f"[INFO] Dropping all-NaN columns: {len(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)

    # =========================
    # Drop constant columns
    # =========================
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()

    if constant_cols:
        print(f"[INFO] Dropping constant columns: {len(constant_cols)}")
        X = X.drop(columns=constant_cols)

    feature_names = X.columns.tolist()

    print(f"[INFO] Clean feature count: {len(feature_names)}")
    print(f"[INFO] Samples: {X.shape[0]}")
    print("[INFO] Class distribution:", dict(pd.Series(y).value_counts()))

    if len(np.unique(y)) < 2:
        raise RuntimeError("Target is single-class after cleaning.")

    # =========================
    # Stratified folds
    # =========================
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    per_fold_selected_final: Dict[str, List[str]] = {}
    fold_overlap_rows: List[Dict[str, int]] = []
    stability_counter: Dict[str, int] = {fn: 0 for fn in feature_names}

    all_folds_union: Set[str] = set()

    for fold, (tr_idx, _va_idx) in enumerate(skf.split(X, y), start=1):

        print("\n-------------------------------------------------------")
        print(f"[FOLD {fold}] Starting")
        print("-------------------------------------------------------")

        X_tr = X.iloc[tr_idx].copy()
        y_tr = y[tr_idx]

        # Leakage-safe imputation: fit only on train fold
        imputer = SimpleImputer(strategy="median")

        X_tr_imp = pd.DataFrame(
            imputer.fit_transform(X_tr),
            columns=feature_names,
            index=X_tr.index,
        )

        # Selector 1: Pearson
        pearson_feats = compute_abs_pearson_threshold(
            X_train=X_tr_imp,
            y_train=y_tr,
            abs_r_threshold=pearson_abs_r_threshold,
        )

        X_tr_np = X_tr_imp.to_numpy(dtype=np.float32)

        # Selector 2: TabNet
        tabnet_feats = train_tabnet_and_get_thresholded(
            X_train=X_tr_np,
            y_train=y_tr,
            feature_names=feature_names,
            seed=seed + fold,
            max_epochs=tabnet_max_epochs,
            batch_size=tabnet_batch_size,
            virtual_batch_size=tabnet_vbatch_size,
            baseline_alpha=tabnet_baseline_alpha,
        )

        # Selector 3: HGB + permutation importance
        hgb_feats = hgb_permutation_importance_lcb_positive(
            X_train=X_tr_np,
            y_train=y_tr,
            feature_names=feature_names,
            seed=seed + 10000 + fold,
            n_repeats=pi_repeats,
            sample_size=DEFAULT_PI_SAMPLE_SIZE,
            n_jobs=DEFAULT_PI_N_JOBS,
            lcb_lambda=hgb_lcb_lambda,
            min_margin=hgb_min_margin,
        )

        A = set(pearson_feats)
        B = set(tabnet_feats)
        C = set(hgb_feats)

        final_set, details = fuse_sets_at_least_two(A, B, C)

        for feature in final_set:
            stability_counter[feature] = stability_counter.get(feature, 0) + 1

        per_fold_selected_final[f"fold_{fold}"] = sorted(list(final_set))
        all_folds_union |= final_set

        fold_overlap_rows.append({
            "fold": fold,
            "N_features_clean": len(feature_names),
            "Pearson_selected": len(A),
            "TabNet_selected": len(B),
            "HGB_selected": len(C),
            "AB_overlap": len(details["ab"]),
            "AC_overlap": len(details["ac"]),
            "BC_overlap": len(details["bc"]),
            "ABC_overlap": len(details["abc"]),
            "Common_ge2": len(details["common_ge2"]),
            "Unique_Pearson": len(details["unique_a"]),
            "Unique_TabNet": len(details["unique_b"]),
            "Unique_HGB": len(details["unique_c"]),
            "FinalUnion_per_fold": len(final_set),
        })

        print(
            f"[FOLD {fold}] "
            f"Pearson={len(A)} | "
            f"TabNet={len(B)} | "
            f"HGB={len(C)} | "
            f"Common(>=2)={len(details['common_ge2'])} | "
            f"Unique(P,T,H)=({len(details['unique_a'])},"
            f"{len(details['unique_b'])},"
            f"{len(details['unique_c'])}) | "
            f"FinalUnion={len(final_set)} | "
            f"AllFoldsUnionSoFar={len(all_folds_union)}"
        )

    # =========================
    # Save outputs
    # =========================
    stability_df = pd.DataFrame(
        [
            {"feature": feature, "folds_selected": count}
            for feature, count in stability_counter.items()
        ]
    ).sort_values(
        ["folds_selected", "feature"],
        ascending=[False, True]
    )

    final_features_allfolds = sorted(list(all_folds_union))

    fold_json_path = os.path.join(out_dir, "fold_selected_features.json")
    with open(fold_json_path, "w", encoding="utf-8") as f:
        json.dump(per_fold_selected_final, f, indent=2)

    print(f"[SAVE] {fold_json_path}")

    stability_csv_path = os.path.join(out_dir, "feature_stability.csv")
    stability_df.to_csv(stability_csv_path, index=False)

    print(f"[SAVE] {stability_csv_path}")

    final_json_path = os.path.join(out_dir, "final_features_allfolds.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_features_allfolds, f, indent=2)

    print(f"[SAVE] {final_json_path}")
    print(f"[INFO] Final all-fold union feature count: {len(final_features_allfolds)}")

    overlap_df = pd.DataFrame(fold_overlap_rows)

    overlap_csv_path = os.path.join(out_dir, "selector_overlap_table.csv")
    overlap_df.to_csv(overlap_csv_path, index=False)

    print(f"[SAVE] {overlap_csv_path}")

    summary = {
        "dataset": "NHANES 2017-March 2020 Pre-pandemic",
        "target": target,
        "target_definition": "CVD_TARGET = 1 if any of MCQ160B/MCQ160C/MCQ160D/MCQ160E indicated cardiovascular disease history; otherwise 0.",
        "leakage_columns_removed": leakage_present,
        "n_splits": n_splits,
        "seed": seed,
        "final_rule": "Union across folds of per-fold final sets; no stability filtering.",
        "pearson_abs_r_threshold": pearson_abs_r_threshold,
        "tabnet_importance_threshold": f"{tabnet_baseline_alpha}*(1/N_features)",
        "tabnet_baseline_alpha": float(tabnet_baseline_alpha),
        "hgb_permutation_threshold": f"(mu - {hgb_lcb_lambda}*sigma) > {hgb_min_margin}",
        "pi_repeats": int(pi_repeats),
        "pi_sample_size": int(DEFAULT_PI_SAMPLE_SIZE),
        "pi_n_jobs": int(DEFAULT_PI_N_JOBS),
        "clean_feature_count_before_selection": int(len(feature_names)),
        "samples": int(X.shape[0]),
        "class_distribution": {
            str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()
        },
        "final_allfolds_feature_count": int(len(final_features_allfolds)),
        "final_allfolds_features_path": final_json_path,
        "overlap_table_path": overlap_csv_path,
        "fold_selected_features_path": fold_json_path,
        "stability_table_path": stability_csv_path,
        "hgb_lcb_lambda": float(hgb_lcb_lambda),
        "hgb_min_margin": float(hgb_min_margin),
    }

    summary_path = os.path.join(out_dir, "phase1_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] {summary_path}")
    print("[DONE] NHANES Phase-1 feature fusion cache built successfully.")


def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(
        description="NHANES Phase-1 feature selection fusion cache builder"
    )

    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--target", type=str, default=DEFAULT_TARGET)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    p.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument(
        "--pearson_abs_r_threshold",
        type=float,
        default=DEFAULT_PEARSON_ABS_R_THRESHOLD,
    )

    p.add_argument(
        "--leakage_cols",
        type=str,
        default=",".join(DEFAULT_LEAKAGE_COLS),
    )

    p.add_argument(
        "--tabnet_max_epochs",
        type=int,
        default=DEFAULT_TABNET_MAX_EPOCHS,
    )

    p.add_argument(
        "--tabnet_batch_size",
        type=int,
        default=DEFAULT_TABNET_BATCH_SIZE,
    )

    p.add_argument(
        "--tabnet_vbatch_size",
        type=int,
        default=DEFAULT_TABNET_VBATCH_SIZE,
    )

    p.add_argument(
        "--tabnet_baseline_alpha",
        type=float,
        default=DEFAULT_TABNET_BASELINE_ALPHA,
    )

    p.add_argument(
        "--pi_repeats",
        type=int,
        default=DEFAULT_PI_REPEATS,
    )

    p.add_argument(
        "--hgb_lcb_lambda",
        type=float,
        default=DEFAULT_HGB_LCB_LAMBDA,
    )

    p.add_argument(
        "--hgb_min_margin",
        type=float,
        default=DEFAULT_HGB_MIN_MARGIN,
    )

    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()

    leakage_cols = [
        c.strip()
        for c in args.leakage_cols.split(",")
        if c.strip()
    ]

    main(
        data_path=args.data_path,
        target=args.target,
        out_dir=args.out_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        pearson_abs_r_threshold=args.pearson_abs_r_threshold,
        leakage_cols=leakage_cols,
        tabnet_max_epochs=args.tabnet_max_epochs,
        tabnet_batch_size=args.tabnet_batch_size,
        tabnet_vbatch_size=args.tabnet_vbatch_size,
        tabnet_baseline_alpha=args.tabnet_baseline_alpha,
        pi_repeats=args.pi_repeats,
        hgb_lcb_lambda=args.hgb_lcb_lambda,
        hgb_min_margin=args.hgb_min_margin,
    )