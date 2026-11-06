"""
Compares reproduced outputs (runs/, data/nhanes/) with the original saved outputs (reference/), file by file.
Read-only; it never edits either side. Results are reported as they are; nothing is adjusted.

Per file type:
  .csv   values compared after dropping columns whose name contains 'path' (absolute paths differ by design);
         numeric columns -> max |difference|; text columns -> number of differing cells;
         probability columns -> number of decisions that flip at the 0.50 threshold.
  .json  lists compared as sets (and order); dicts compared after removing path-like strings.
  .png/.pdf  byte hash (informational: images can differ in bytes even when the content is the same).
Status: IDENTICAL | MATCH_1e-4 (all numeric |diff| <= 1e-4, no text/decision differences) | DIFFERS |
        MISSING_IN_RUN | REFERENCE_ONLY (file in the original folder that this pipeline does not produce).

Usage (from HeartSense_repro):  .venv\\Scripts\\python.exe tools\\compare_outputs.py --stage 1
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parent.parent
TOL = 1e-4
PROB_COLS = {"y_prob", "p_final", "p_rec", "p_prec", "p_gate_only", "g"}
PAIRS = {  # (reproduced, reference) relative to HeartSense_repro
    "1": [("data/nhanes/nhanes_2017_2020_merged_raw.csv", "reference/nhanes/data/nhanes_2017_2020_merged_raw.csv"),
          ("data/nhanes/nhanes_cleaned_step1.csv", "reference/nhanes/data/nhanes_cleaned_step1.csv"),
          ("data/nhanes/nhanes_final_with_target.csv", "reference/nhanes/data/nhanes_final_with_target.csv"),
          ("runs/nhanes/nhanes_phase1_feature_fusion", "reference/nhanes/nhanes_phase1_feature_fusion"),
          ("runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures", "reference/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures"),
          ("runs/nhanes/nhanes_baseline_comparison", "reference/nhanes/nhanes_baseline_comparison"),
          ("runs/nhanes/nhanes_figures", "reference/nhanes/nhanes_figures")],
    "2": [("runs/brfss/phase1_feature_fusion", "reference/brfss/phase1_feature_fusion"),
          ("runs/brfss/phase2_fttf_dual_precbiased_gate_selectedfeatures", "reference/brfss/phase2_fttf_dual_precbiased_gate_selectedfeatures")],
    "3": [("runs/brfss/phase2_allmodels_allfeatures", "reference/brfss/phase2_allmodels_allfeatures")],
    "4": [("runs/brfss/confusion_matrix_comparison", "reference/brfss/confusion_matrix_comparison"),
          ("runs/brfss/FP", "reference/brfss/FP"),
          ("runs/brfss/threshold_analysis", "reference/brfss/threshold_analysis"),
          ("runs/brfss/threshold_heatmap", "reference/brfss/threshold_heatmap"),
          ("runs/brfss/calibration_analysis", "reference/brfss/calibration_analysis")],
}


# Files in the original output folders that the current original scripts do not write (documented, not regenerated)
REFERENCE_ONLY_NOTES = {
    "nhanes_pooled_confusion_matrix_comparison.png": "earlier 1x5 draft of Fig 15 (13 May 2026); the current script writes "
        "nhanes_confusion_matrix_grid_style.png, the 2x3 layout printed as Fig 15; the code that drew this draft no longer exists",
    "nhanes_pooled_confusion_matrix_comparison_large.png": "larger copy of the same earlier 1x5 draft of Fig 15; not written by the current script",
    "final_features_stable.json": "exploratory copy of the Phase-1 list (same 181 features); not written by the Phase-1 script",
    "figure_radar_metrics_like_paper.png": "radar plot from plot_radar_metrics.py; not in the paper, not in this pipeline",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strip_paths(o):
    if isinstance(o, dict):
        return {k: strip_paths(v) for k, v in o.items() if not (isinstance(v, str) and (":\\" in v or v.startswith("/")))}
    if isinstance(o, list):
        return [strip_paths(v) for v in o]
    return o


def json_diff(a, b, where="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{where}/{k}: only in {'run' if k in a else 'reference'}")
            else:
                json_diff(a[k], b[k], f"{where}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and all(isinstance(x, str) for x in a + b):
        if set(a) != set(b):
            out.append(f"{where}: +{sorted(set(a) - set(b))[:10]} -{sorted(set(b) - set(a))[:10]}")
        elif a != b:
            out.append(f"{where}: same items, different order")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, abs_tol=TOL) or (isinstance(a, float) and math.isnan(a) and math.isnan(b))):
            out.append(f"{where}: {a} vs {b}")
    elif a != b:
        out.append(f"{where}: {str(a)[:60]} vs {str(b)[:60]}")
    return out


def compare_csv(run, ref):
    a, b = pd.read_csv(run), pd.read_csv(ref)
    a = a[[c for c in a.columns if "path" not in c.lower()]]
    b = b[[c for c in b.columns if "path" not in c.lower()]]
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return "DIFFERS", f"shape/columns differ: run {a.shape} vs reference {b.shape}"
    if a.equals(b):
        return "IDENTICAL", f"{a.shape[0]} rows x {a.shape[1]} cols"
    max_d, text_diff, flips, worst = 0.0, 0, 0, ""
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.where(both_nan, 0.0, np.abs(xv - yv))
            d = np.nan_to_num(d, nan=np.inf)
            if d.size and d.max() > max_d:
                max_d, worst = float(d.max()), c
            if c in PROB_COLS:
                flips += int(((xv >= 0.5) != (yv >= 0.5)).sum())
        else:
            text_diff += int((x.astype(str) != y.astype(str)).sum())
    status = "MATCH_1e-4" if max_d <= TOL and text_diff == 0 and flips == 0 else "DIFFERS"
    return status, f"max|diff|={max_d:.3g} (column {worst}); text cells differing={text_diff}; 0.50-decision flips={flips}"


def compare_file(run, ref):
    if not run.exists():
        return "MISSING_IN_RUN", ""
    suf = ref.suffix.lower()
    if suf == ".csv":
        return compare_csv(run, ref)
    if suf == ".json":
        a, b = json.loads(run.read_text(encoding="utf-8")), json.loads(ref.read_text(encoding="utf-8"))
        if a == b:
            return "IDENTICAL", ""
        d = json_diff(strip_paths(a), strip_paths(b))
        return ("MATCH_1e-4", "only absolute paths differ") if not d else ("DIFFERS", "; ".join(d[:8]))
    same = sha(run) == sha(ref)
    if suf in (".png", ".pdf"):
        return ("IDENTICAL", "same bytes") if same else ("IMAGE_DIFFERS", "informational: compare visually")
    return ("IDENTICAL", "") if same else ("DIFFERS", "bytes differ")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(PAIRS))
    args = ap.parse_args()
    rows = []
    for run_rel, ref_rel in PAIRS[args.stage]:
        run_p, ref_p = REPRO / run_rel, REPRO / ref_rel
        if ref_p.is_dir():
            ref_files = sorted(p for p in ref_p.rglob("*") if p.is_file())
            for f in ref_files:
                sub = f.relative_to(ref_p)
                st, det = compare_file(run_p / sub, f)
                if st == "MISSING_IN_RUN" and f.name in REFERENCE_ONLY_NOTES:
                    st, det = "REFERENCE_ONLY", REFERENCE_ONLY_NOTES[f.name]
                elif st == "MISSING_IN_RUN" and f.suffix == ".py":
                    st, det = "REFERENCE_ONLY", "script stored in an output folder; not produced by this pipeline"
                rows.append({"reproduced": f"{run_rel}/{sub.as_posix()}", "status": st, "detail": det})
            if run_p.is_dir():
                for f in sorted(p for p in run_p.rglob("*") if p.is_file()):
                    if not (ref_p / f.relative_to(run_p)).exists():
                        rows.append({"reproduced": f"{run_rel}/{f.relative_to(run_p).as_posix()}", "status": "NEW_IN_RUN", "detail": "no original counterpart"})
        else:
            st, det = compare_file(run_p, ref_p)
            rows.append({"reproduced": run_rel, "status": st, "detail": det})
    df = pd.DataFrame(rows)
    out = REPRO / "reports"; out.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out / f"compare_stage{args.stage}_{ts}.csv", index=False)
    with open(out / f"compare_stage{args.stage}_{ts}.md", "w", encoding="utf-8") as f:
        f.write(f"# Stage {args.stage}: reproduced vs original saved outputs ({ts})\n\n")
        f.write(md_table(df.status.value_counts().rename_axis("status").reset_index(name="files")) + "\n\n")
        f.write(md_table(df) + "\n")
    print(df.status.value_counts().to_string())
    print(df[~df.status.isin(["IDENTICAL", "MATCH_1e-4", "REFERENCE_ONLY"])].to_string(index=False))
    print(f"\nreport: {out / f'compare_stage{args.stage}_{ts}.md'}")


if __name__ == "__main__":
    main()
