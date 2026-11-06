"""
Compares a new run's outputs (runs/, data/nhanes/) with the verified November 2026 outputs shipped in
verified_outputs/, file by file. Read-only. Package edition of the comparison tool used for the verified runs
(that version compared against copies of the original saved outputs; it is in evidence/tools_as_run/).

The verified outputs are themselves bit-identical to the original saved outputs of the paper's project, except
where evidence/reports/ records otherwise (stored absolute paths; 9-decimal rounding in the saved master table;
calibration metrics within 2.4e-12; a PDF creation date).

Per file type:
  .csv   values compared after dropping columns whose name contains 'path' (absolute paths differ by design);
         numeric columns -> max |difference|; text columns -> number of differing cells;
         probability columns -> number of decisions that flip at the 0.50 threshold.
  .json  lists compared as sets (and order); dicts compared after removing path-like strings.
  .png/.pdf  byte hash (informational: images can differ in bytes when library versions or fonts differ).
Files not shipped (row-level BRFSS predictions, derived NHANES CSVs; manifests/excluded_files.sha256) are
compared by SHA-256 with the verified run.
Status: IDENTICAL | MATCH_1e-4 | DIFFERS | IMAGE_DIFFERS | MISSING_IN_RUN | NEW_IN_RUN

Usage (from the package folder):  python tools/compare_outputs.py --stage 1
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VERIFIED = REPO / "verified_outputs"
TOL = 1e-4
PROB_COLS = {"y_prob", "p_final", "p_rec", "p_prec", "p_gate_only", "g"}
PAIRS = {  # run folders/files (relative to the package folder), as in the verified runs
    "1": ["data/nhanes/nhanes_2017_2020_merged_raw.csv", "data/nhanes/nhanes_cleaned_step1.csv",
          "data/nhanes/nhanes_final_with_target.csv", "runs/nhanes/nhanes_phase1_feature_fusion",
          "runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures",
          "runs/nhanes/nhanes_baseline_comparison", "runs/nhanes/nhanes_figures"],
    "2": ["runs/brfss/phase1_feature_fusion", "runs/brfss/phase2_fttf_dual_precbiased_gate_selectedfeatures"],
    "3": ["runs/brfss/phase2_allmodels_allfeatures"],
    "4": ["runs/brfss/confusion_matrix_comparison", "runs/brfss/FP", "runs/brfss/threshold_analysis",
          "runs/brfss/threshold_heatmap", "runs/brfss/calibration_analysis"],
}


def verified_path(run_rel: str) -> Path:
    return VERIFIED / run_rel.split("/", 1)[1] if run_rel.startswith("runs/") else VERIFIED / run_rel


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def excluded() -> dict:
    out = {}
    f = REPO / "manifests" / "excluded_files.sha256"
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            out[rel] = h
    return out


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
                out.append(f"{where}/{k}: only in {'run' if k in a else 'verified'}")
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
        return "DIFFERS", f"shape/columns differ: run {a.shape} vs verified {b.shape}"
    if a.equals(b):
        return "IDENTICAL", f"{a.shape[0]} rows x {a.shape[1]} cols"
    max_d, text_diff, flips, worst = 0.0, 0, 0, ""
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.nan_to_num(np.where(both_nan, 0.0, np.abs(xv - yv)), nan=np.inf)
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
    exc = excluded()
    rows = []
    for run_rel in PAIRS[args.stage]:
        run_p, ver_p = REPO / run_rel, verified_path(run_rel)
        expected = {}  # sub-path -> ("file", Path) | ("hash", sha)
        if ver_p.is_dir():
            for f in sorted(p for p in ver_p.rglob("*") if p.is_file() and p.name != "README.md"):
                expected[f.relative_to(ver_p).as_posix()] = ("file", f)
        for rel, h in exc.items():
            if rel == run_rel:
                expected[""] = ("hash", h)
            elif rel.startswith(run_rel + "/"):
                expected[rel[len(run_rel) + 1:]] = ("hash", h)
        if not expected and ver_p.is_file():
            expected[""] = ("file", ver_p)
        for sub, (kind, ref) in sorted(expected.items()):
            target = run_p / sub if sub else run_p
            if kind == "file":
                st, det = compare_file(target, ref)
            elif not target.exists():
                st, det = "MISSING_IN_RUN", "not shipped; compared by SHA-256"
            else:
                st, det = ("IDENTICAL", "SHA-256 equals the verified run (file not shipped)") if sha(target) == ref \
                    else ("DIFFERS", "SHA-256 differs from the verified run (file not shipped; compare values if needed)")
            rows.append({"reproduced": f"{run_rel}/{sub}" if sub else run_rel, "status": st, "detail": det})
        if run_p.is_dir():
            for f in sorted(p for p in run_p.rglob("*") if p.is_file()):
                if f.relative_to(run_p).as_posix() not in expected:
                    rows.append({"reproduced": f"{run_rel}/{f.relative_to(run_p).as_posix()}", "status": "NEW_IN_RUN",
                                 "detail": "no verified counterpart"})
    df = pd.DataFrame(rows)
    out = REPO / "reports"; out.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out / f"compare_stage{args.stage}_{ts}.csv", index=False)
    with open(out / f"compare_stage{args.stage}_{ts}.md", "w", encoding="utf-8") as f:
        f.write(f"# Stage {args.stage}: new run vs verified outputs ({ts})\n\n")
        f.write(md_table(df.status.value_counts().rename_axis("status").reset_index(name="files")) + "\n\n")
        f.write(md_table(df) + "\n")
    print(df.status.value_counts().to_string())
    print(df[~df.status.isin(["IDENTICAL", "MATCH_1e-4"])].to_string(index=False))
    print(f"\nreport: {out / f'compare_stage{args.stage}_{ts}.md'}")


if __name__ == "__main__":
    main()
