"""
Builds the three-column results table: PUBLISHED (paper) | VERIFIED 2026 RUN (verified_outputs/) | NEW RUN (runs/).
Package edition of the tool used for the verified runs (evidence/tools_as_run/); that run's published vs
original-saved table is evidence/reports/published_vs_reproduced_20261106_031053.csv.
Read-only. Values are reported as found. A row whose reproduced file does not exist yet shows 'not run'.

Columns:
  published           value printed in the paper (Sensors 2026, 26, 3417)
  verified_2026       the same quantity read from verified_outputs/ (equal to the original saved outputs;
                      NHANES cohort-size rows need data/nhanes/*.csv and are blank until Stage 1 is re-run)
  reproduced          the same quantity read from this reproduction's outputs
  verified_vs_published 'match' if the verified value rounds to the published value at the published precision;
                      'match (truncated, not rounded)' if it only matches when cut off instead of rounded
  reproduced_vs_verified 'identical' / 'within 1e-4' / difference
  note                factual notes on where a published value comes from (no interpretation)

Usage (from the package folder):  python tools/published_vs_reproduced.py
"""
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

REPRO = Path(__file__).resolve().parent.parent
ROOTS = {"saved": {"brfss": "verified_outputs/brfss", "nhanes": "verified_outputs/nhanes", "nh_data": None},
         "repro": {"brfss": "runs/brfss", "nhanes": "runs/nhanes", "nh_data": "data/nhanes"}}
SEL = "phase2_fttf_dual_precbiased_gate_selectedfeatures"
NH_SEL = "nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures"


def p(which, area, rel):
    if ROOTS[which][area] is None:
        raise FileNotFoundError(area)
    return REPRO / ROOTS[which][area] / rel


def csv_row(path, key_col, key, col):
    df = pd.read_csv(path)
    return float(df.loc[df[key_col].astype(str) == key, col].iloc[0])


def getter(which, spec):
    kind = spec[0]
    if kind == "foldmean":          # proposed-model per-fold metrics file, 'mean' row
        _, area, folder, prefix, col = spec
        return csv_row(p(which, area, f"{folder}/{prefix}_metrics.csv"), "fold", "mean", col)
    if kind == "confmat":           # pooled confusion matrix written by the proposed-model script
        _, area, folder, prefix, cell = spec
        df = pd.read_csv(p(which, area, f"{folder}/{prefix}_confmat_050.csv"), index_col=0)
        return float(df.loc[cell[0], cell[1]])
    if kind == "csvrow":
        _, area, rel, key_col, key, col = spec
        return csv_row(p(which, area, rel), key_col, key, col)
    if kind == "counts_prf":        # precision / recall / F1 from Table 9 counts
        _, model, what = spec
        df = pd.read_csv(p(which, "brfss", "confusion_matrix_comparison/confusion_matrix_counts_all_models.csv")).set_index("Model")
        tp, fp, fn = df.loc[model, "TP"], df.loc[model, "FP"], df.loc[model, "FN"]
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        return float({"P": pr, "R": rc, "F1": 2 * pr * rc / (pr + rc)}[what])
    if kind == "shape":
        _, rel, axis = spec
        df = pd.read_csv(p(which, "nh_data", rel), low_memory=False)
        return float(df.shape[axis])
    if kind == "nh_phase1":
        _, key = spec
        s = json.loads(p(which, "nhanes", "nhanes_phase1_feature_fusion/phase1_summary.json").read_text(encoding="utf-8"))
        if key == "pos_pct":
            cd = s["class_distribution"]; return 100 * cd["1"] / (cd["0"] + cd["1"])
        return float(s[key])
    raise ValueError(kind)


M = "master"  # shorthand
MASTER = "phase2_allmodels_allfeatures/master_results_phase2_allfeatures.csv"
ROWS = []


def add(element, metric, published, spec, note=""):
    ROWS.append((element, metric, published, spec, note))


# ---- Abstract
add("Abstract", "F1 @0.50 (fold mean)", "0.43", ("foldmean", "brfss", SEL, SEL, "f1_050"))
add("Abstract", "Recall @0.50 (fold mean)", "0.59", ("foldmean", "brfss", SEL, SEL, "recall_050"))
add("Abstract", "AUPRC (fold mean)", "0.38", ("foldmean", "brfss", SEL, SEL, "auprc"),
    "scikit-learn average precision of the final probabilities (after veto/rescue)")
# ---- Table 3
for col, pub in [("N", "452464"), ("Positives", "42338"), ("Negatives", "410126"), ("Prevalence_%", "9.3572"),
                 ("Neg_per_Pos", "9.6869"), ("Features_all_numeric", "283"), ("Features_selected", "181"),
                 ("Missing_avg_all_%", "51.7566"), ("Missing_avg_selected_%", "41.7640")]:
    add("Table 3", col, pub, ("csvrow", "brfss", "phase1_feature_fusion/paper_tables/dataset_summary_brfss2024_phase2.csv", "N" if col != "N" else "N", None, col))
# ---- Table 6
T6 = {"LightGBM": "0.91 0.56 0.85 0.38 0.56 0.13 0.21", "LogReg": "0.75 0.77 0.85 0.38 0.25 0.79 0.37",
      "SVM": "0.91 0.55 0.85 0.38 0.57 0.11 0.19", "DT": "0.75 0.71 0.76 0.26 0.22 0.66 0.33",
      "RF": "0.85 0.73 0.85 0.37 0.33 0.58 0.42", "KNN": "0.91 0.50 0.78 0.28 0.61 0.01 0.02",
      "NB": "0.66 0.72 0.76 0.19 0.19 0.80 0.30", "AE_LR": "0.74 0.74 0.81 0.29 0.22 0.73 0.34",
      "CNN1D": "0.74 0.77 0.85 0.38 0.24 0.81 0.37", "LSTM": "0.76 0.76 0.84 0.36 0.25 0.76 0.37",
      "GRU": "0.73 0.76 0.84 0.35 0.23 0.80 0.36", "FTTF_Dual_Head": "0.86 0.73 0.79 0.38 0.35 0.59 0.43"}
T6COLS = ["acc_mean", "balacc_mean", "roc_auc_mean", "auprc_mean", "precision_mean", "recall_mean", "f1_mean"]
for model, vals in T6.items():
    for col, pub in zip(T6COLS, vals.split()):
        note = "proposed model row (added to the master table by update_master_table_add_experiment.py)" if model == "FTTF_Dual_Head" else ""
        add("Table 6", f"{model} {col.replace('_mean', '')}", pub, ("csvrow", "brfss", MASTER, "model", model, col), note)
# ---- Tables 8-9
T9 = {"Proposed": ("50487", "16975", "0.334", "0.599", "0.429"), "LightGBM": ("4421", "36777", "0.557", "0.131", "0.212"),
      "Random Forest": ("49186", "17919", "0.332", "0.577", "0.421"), "CNN": ("108803", "8011", "0.240", "0.811", "0.370"),
      "Simple Avg": ("37835", "20464", "0.366", "0.517", "0.429")}
for model, (fp, fn, pr, rc, f1) in T9.items():
    note = "mean of LightGBM, RF and CNN1D probabilities" if model == "Simple Avg" else ""
    add("Tables 8-9", f"{model} FP", fp, ("csvrow", "brfss", "confusion_matrix_comparison/confusion_matrix_counts_all_models.csv", "Model", model, "FP"), note)
    add("Tables 8-9", f"{model} FN", fn, ("csvrow", "brfss", "confusion_matrix_comparison/confusion_matrix_counts_all_models.csv", "Model", model, "FN"), note)
    for what, pub in (("P", pr), ("R", rc), ("F1", f1)):
        add("Table 9", f"{model} {what}", pub, ("counts_prf", model, what), note)
# ---- Table 10
T10 = {"Recall head only": ("0.378", "0.247", "0.804", "0.775", "103572", "8318"),
       "Precision head only": ("0.375", "0.452", "0.320", "0.640", "16422", "28780"),
       "Simple average fusion": ("0.431", "0.335", "0.605", "0.741", "50966", "16705"),
       "GateNet + veto/rescue (final)": ("0.430", "0.349", "0.599", "0.738", "50487", "16975")}
for variant, vals in T10.items():
    note = "mean of the recall- and precision-branch probabilities" if variant == "Simple average fusion" else ""
    for col, pub in zip(["F1", "Precision", "Recall", "BalAcc", "FP", "FN"], vals):
        add("Table 10", f"{variant} {col}", pub, ("csvrow", "brfss", f"{SEL}/ablation/ablation_results.csv", "Variant", variant, col), note)
# ---- Table 12
for method, (b, e) in {"Raw": ("0.0944", "0.1027"), "Platt": ("0.0695", "0.0040"), "Isotonic": ("0.0695", "0.0005")}.items():
    add("Table 12", f"{method} Brier", b, ("csvrow", "brfss", "calibration_analysis/calibration_metrics_brier_ece.csv", "method", method, "brier"))
    add("Table 12", f"{method} ECE", e, ("csvrow", "brfss", "calibration_analysis/calibration_metrics_brier_ece.csv", "method", method, "ece"))
# ---- Table 14
add("Table 14", "Merged participants", "15560", ("shape", "nhanes_2017_2020_merged_raw.csv", 0))
add("Table 14", "Merged raw variables", "282", ("shape", "nhanes_2017_2020_merged_raw.csv", 1))
add("Table 14", "Variables after missingness filter", "173", ("shape", "nhanes_cleaned_step1.csv", 1))
add("Table 14", "Final dataset size", "9180", ("shape", "nhanes_final_with_target.csv", 0))
add("Table 14", "Features before Phase 1", "163", ("nh_phase1", "clean_feature_count_before_selection"))
add("Table 14", "Phase 1 selected features", "141", ("nh_phase1", "final_allfolds_feature_count"))
add("Table 14", "Positive class ratio (%)", "9.53", ("nh_phase1", "pos_pct"))
# ---- Table 15
NHA = f"{NH_SEL}/ablation/nhanes_ablation_results.csv"
for col, pub in [("Precision", "0.4246"), ("Recall", "0.5726"), ("BalAcc", "0.7454"), ("F1", "0.4876"),
                 ("AUROC", "0.8736"), ("AUPRC", "0.4677"), ("FP", "679"), ("FN", "374")]:
    note = ("published value equals the saved 'GateNet only (no veto/rescue)' row" if col in ("AUROC", "AUPRC") else "")
    add("Table 15", f"Proposed {col}", pub, ("csvrow", "nhanes", NHA, "Variant", "GateNet + veto/rescue (final)", col), note)
    if col in ("AUROC", "AUPRC"):
        add("Table 15", f"Proposed {col} (gate-only output)", pub, ("csvrow", "nhanes", NHA, "Variant", "GateNet only (no veto/rescue)", col))
T15 = {"LightGBM": "0.4890 0.4331 0.6927 0.4594 0.8739 0.4522 396 496", "Random Forest": "0.6000 0.0960 0.5446 0.1655 0.8794 0.4285 56 791",
       "CNN": "0.1701 0.6994 0.6699 0.2736 0.7405 0.2271 2986 263"}
for model, vals in T15.items():
    for col, pub in zip(["Precision", "Recall", "BalAcc", "F1", "AUROC", "AUPRC", "FP", "FN"], vals.split()):
        add("Table 15", f"{model} {col}", pub, ("csvrow", "nhanes", "nhanes_baseline_comparison/nhanes_baseline_comparison_results.csv", "Model", model, col))
for col, pub in zip(["Precision", "Recall", "BalAcc", "F1", "AUROC", "AUPRC", "FP", "FN"], "0.3777 0.6549 0.7716 0.4791 0.8696 0.4601 944 302".split()):
    add("Table 15", f"Simple Avg {col}", pub, ("csvrow", "nhanes", NHA, "Variant", "Simple average fusion", col),
        "mean of the recall- and precision-branch probabilities")


def fix_spec(spec):
    # Table 3 rows: single-row CSV, key column/value not needed
    if spec[0] == "csvrow" and spec[4] is None:
        return ("t3", spec[2], spec[5])
    return spec


def value(which, spec):
    try:
        if spec[0] == "t3":
            return float(pd.read_csv(p(which, "brfss", spec[1]))[spec[2]].iloc[0])
        return getter(which, spec)
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa
        return f"error: {e}"


def main():
    out_rows = []
    for element, metric, pub, spec, note in ROWS:
        spec = fix_spec(spec)
        saved, repro = value("saved", spec), value("repro", spec)
        dec = len(pub.split(".")[1]) if "." in pub else 0
        if isinstance(saved, float):
            scale = 10 ** dec
            if round(saved, dec) == float(pub):
                s_vs_p = "match"
            elif math.floor(saved * scale + 1e-9) / scale == float(pub):
                s_vs_p = "match (truncated, not rounded)"
            else:
                s_vs_p = "differs"
        else:
            s_vs_p = "n/a"
        if repro is None:
            r_vs_s = "not run"
        elif isinstance(repro, float) and isinstance(saved, float):
            d = abs(repro - saved)
            r_vs_s = "identical" if d == 0 else ("within 1e-4" if d <= 1e-4 else f"differs by {d:.4g}")
        else:
            r_vs_s = "n/a"
        fmt = lambda v: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else str(v)))
        out_rows.append({"element": element, "metric": metric, "published": pub, "verified_2026": fmt(saved),
                         "reproduced": fmt(repro), "verified_vs_published": s_vs_p, "reproduced_vs_verified": r_vs_s, "note": note})
    df = pd.DataFrame(out_rows)
    out = REPRO / "reports"; out.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out / f"published_vs_reproduced_{ts}.csv", index=False)
    print(df.to_string(index=False))
    print("\nverified_vs_published:", df.verified_vs_published.value_counts().to_dict())
    print("reproduced_vs_verified:", df.reproduced_vs_verified.value_counts().to_dict())
    print(f"report: {out / f'published_vs_reproduced_{ts}.csv'}")


if __name__ == "__main__":
    main()
