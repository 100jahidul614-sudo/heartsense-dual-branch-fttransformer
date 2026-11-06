import os
import json
import time
import argparse
import pandas as pd


# =========================
# RUN-BUTTON MODE DEFAULTS
# =========================
RUN_WITH_DEFAULTS = True

DEFAULT_MASTER_CSV = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_allmodels_allfeatures\master_results_phase2_allfeatures.csv"

# Point these to the outputs of the run you want to add
DEFAULT_METRICS_CSV = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate_allfeatures\phase2_fttf_dual_precbiased_gate_allfeatures_metrics.csv"
DEFAULT_CONFMAT_CSV = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate_allfeatures\phase2_fttf_dual_precbiased_gate_allfeatures_confmat_050.csv"
DEFAULT_OOF_CSV = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate_allfeatures\phase2_fttf_dual_precbiased_gate_allfeatures_oof.csv"

# OPTIONAL (recommended): summary JSON from phase2 contains n_samples + n_features
DEFAULT_SUMMARY_JSON = r"C:\Users\<user>\Research center\Heartseanse2.0\artifacts\phase2_fttf_dual_precbiased_gate_allfeatures\phase2_fttf_dual_precbiased_gate_allfeatures_summary.json"

# Meta fields for the master table row
DEFAULT_MODEL_NAME = "FTTF_Dual_PrecBiasedGate_veto0_Allfeatures.20"
DEFAULT_FEATURE_SET = "phase1_final_features_allfolds"
DEFAULT_IMBALANCE = "RecallHead=pos_weight; PrecHead=pos_weight/alpha + neg_weight; Fusion=precision-biased gate + veto/rescue"
DEFAULT_THRESHOLD = 0.50


REQUIRED_MASTER_COLS = [
    "model", "feature_set", "imbalance", "n_samples", "n_features", "threshold",
    "acc_mean", "acc_std", "balacc_mean", "balacc_std", "precision_mean", "precision_std",
    "recall_mean", "recall_std", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
    "auprc_mean", "auprc_std", "tn", "fp", "fn", "tp",
    "per_fold_metrics_path", "confmat_050_path", "oof_path"
]


# =========================
# PATCH: lock-safe CSV writer (Windows/Excel/OneDrive)
# =========================
def safe_to_csv_replace(df: pd.DataFrame, path: str, retries: int = 8, wait_sec: float = 0.75) -> None:
    """
    Windows-safe writer:
      1) write to temp file in same directory
      2) atomic replace into target (os.replace)
    Retries for transient locks (Excel/OneDrive).
    If still locked, writes a fallback file to avoid losing updates.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp = path + ".tmp"
    last_err = None

    for _attempt in range(1, retries + 1):
        try:
            # write temp
            df.to_csv(tmp, index=False)

            # atomic replace
            os.replace(tmp, path)

            # success: ensure temp is gone (usually already replaced)
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass

            return

        except PermissionError as e:
            last_err = e

            # clean temp if possible (so next retry can rewrite)
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass

            time.sleep(wait_sec)

        except OSError as e:
            # Handles other filesystem issues (rare)
            last_err = e
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            time.sleep(wait_sec)

    # Final fallback: do NOT lose updates
    ts = int(time.time())
    fallback = path.replace(".csv", f"_NEW_{ts}.csv")
    df.to_csv(fallback, index=False)

    raise PermissionError(
        f"Permission denied writing master CSV (likely open in Excel or locked by sync):\n"
        f"  {path}\n\n"
        f"I saved your updated master to:\n"
        f"  {fallback}\n\n"
        f"Close Excel/unlock the file, then rerun to overwrite the original."
    ) from last_err


def load_mean_std(metrics_csv: str) -> tuple[dict, dict]:
    df = pd.read_csv(metrics_csv)

    if "fold" not in df.columns:
        raise ValueError("metrics CSV must contain a 'fold' column with rows including 'mean' and 'std'.")

    mean_row = df.loc[df["fold"].astype(str).str.lower() == "mean"]
    std_row = df.loc[df["fold"].astype(str).str.lower() == "std"]

    if len(mean_row) != 1 or len(std_row) != 1:
        raise ValueError("metrics CSV must contain exactly one 'mean' row and one 'std' row in the 'fold' column.")

    mean_row = mean_row.iloc[0].to_dict()
    std_row = std_row.iloc[0].to_dict()
    return mean_row, std_row


def load_confmat(confmat_csv: str) -> tuple[int, int, int, int]:
    cm = pd.read_csv(confmat_csv, index_col=0)

    # expects:
    #          pred_0  pred_1
    # true_0     TN     FP
    # true_1     FN     TP
    for r in ["true_0", "true_1"]:
        if r not in cm.index:
            raise ValueError(f"confmat CSV missing index row '{r}'. Found: {list(cm.index)}")
    for c in ["pred_0", "pred_1"]:
        if c not in cm.columns:
            raise ValueError(f"confmat CSV missing column '{c}'. Found: {list(cm.columns)}")

    tn = int(cm.loc["true_0", "pred_0"])
    fp = int(cm.loc["true_0", "pred_1"])
    fn = int(cm.loc["true_1", "pred_0"])
    tp = int(cm.loc["true_1", "pred_1"])
    return tn, fp, fn, tp


def ensure_master_schema(master_df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_MASTER_COLS:
        if c not in master_df.columns:
            master_df[c] = ""
    return master_df[REQUIRED_MASTER_COLS].copy()


def load_summary(summary_json: str) -> dict:
    if summary_json and os.path.isfile(summary_json):
        with open(summary_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", type=str, default=DEFAULT_MASTER_CSV)
    ap.add_argument("--metrics_csv", type=str, default=DEFAULT_METRICS_CSV)
    ap.add_argument("--confmat_csv", type=str, default=DEFAULT_CONFMAT_CSV)
    ap.add_argument("--oof_csv", type=str, default=DEFAULT_OOF_CSV)
    ap.add_argument("--summary_json", type=str, default=DEFAULT_SUMMARY_JSON)

    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--feature_set", type=str, default=DEFAULT_FEATURE_SET)
    ap.add_argument("--imbalance", type=str, default=DEFAULT_IMBALANCE)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    ap.add_argument("--replace_if_exists", type=int, default=1,
                    help="1=replace existing row for same model name, 0=append anyway")
    args = ap.parse_args()

    if not os.path.isfile(args.master_csv):
        raise FileNotFoundError(f"Missing master_csv: {args.master_csv}")
    if not os.path.isfile(args.metrics_csv):
        raise FileNotFoundError(f"Missing metrics_csv: {args.metrics_csv}")
    if not os.path.isfile(args.confmat_csv):
        raise FileNotFoundError(f"Missing confmat_csv: {args.confmat_csv}")

    master = pd.read_csv(args.master_csv)
    master = ensure_master_schema(master)

    mean_row, std_row = load_mean_std(args.metrics_csv)
    tn, fp, fn, tp = load_confmat(args.confmat_csv)
    summary = load_summary(args.summary_json)

    def get_num(d: dict, k: str, default=float("nan")) -> float:
        try:
            return float(d.get(k, default))
        except Exception:
            return float("nan")

    # Prefer summary.json for n_samples / n_features
    n_samples = int(summary.get("n_samples", 0)) if summary else 0
    n_features = int(summary.get("n_features", 0)) if summary else 0

    new_row = {
        "model": str(args.model),
        "feature_set": str(args.feature_set),
        "imbalance": str(args.imbalance),
        "n_samples": n_samples if n_samples > 0 else "",
        "n_features": n_features if n_features > 0 else "",
        "threshold": float(args.threshold),

        "acc_mean": get_num(mean_row, "acc_050"),
        "acc_std": get_num(std_row, "acc_050"),

        "balacc_mean": get_num(mean_row, "balacc_050"),
        "balacc_std": get_num(std_row, "balacc_050"),

        "precision_mean": get_num(mean_row, "precision_050"),
        "precision_std": get_num(std_row, "precision_050"),

        "recall_mean": get_num(mean_row, "recall_050"),
        "recall_std": get_num(std_row, "recall_050"),

        "f1_mean": get_num(mean_row, "f1_050"),
        "f1_std": get_num(std_row, "f1_050"),

        "roc_auc_mean": get_num(mean_row, "auroc"),
        "roc_auc_std": get_num(std_row, "auroc"),

        "auprc_mean": get_num(mean_row, "auprc"),
        "auprc_std": get_num(std_row, "auprc"),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "per_fold_metrics_path": str(args.metrics_csv),
        "confmat_050_path": str(args.confmat_csv),
        "oof_path": str(args.oof_csv) if args.oof_csv else "",
    }

    # Fallback: infer n_samples from OOF if summary missing
    if (new_row["n_samples"] == "" or new_row["n_samples"] == 0) and args.oof_csv and os.path.isfile(args.oof_csv):
        oof = pd.read_csv(args.oof_csv)
        new_row["n_samples"] = int(len(oof))

    # Replace existing model row if requested
    if int(args.replace_if_exists) == 1:
        mask = master["model"].astype(str) == str(args.model)
        if mask.any():
            master = master.loc[~mask].copy()

    master = pd.concat([master, pd.DataFrame([new_row])], ignore_index=True)

    # PATCHED write (lock-safe)
    safe_to_csv_replace(master, args.master_csv)

    print(f"[OK] Master table updated: {args.master_csv}")
    print(f"[ADDED] model={args.model} | FP={fp} | FN={fn} | TP={tp} | TN={tn}")


if __name__ == "__main__":
    main()
