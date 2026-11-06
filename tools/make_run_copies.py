"""
Regenerates run_code/ (path-only execution copies) from original_code/ for the folder this package sits in.
Non-scientific helper. Package edition of the Stage 0 tool used for the verified November 2026 runs
(that version is kept, unchanged apart from path sanitisation, in evidence/tools_as_run/).

Fourteen of the original scripts have their file paths written into the code. Their execution copies differ
from original_code/ ONLY in those path string literals:
  * literals that start with the original project folder  C:\\Users\\<user>\\Research center\\Heartseanse2.0
    are mapped exactly as in the verified runs:
        ...\\Dataset2\\...                      -> <repo>/data/nhanes/...
        ...\\artifacts\\nhanes_...              -> <repo>/runs/nhanes/nhanes_...
        ...\\artifacts\\...                     -> <repo>/runs/brfss/...
        the January copy of the proposed-model OOF file -> the selected-features run's OOF file (same script,
        same 'y_prob' column; the two original runs agree to 1.4e-7 with no decision changes)
  * the hard-coded raw BRFSS file path -> --brfss_xpt (default <repo>/data/brfss/main_data_brfss_2024.XPT)
Every other byte is kept. The tool stops if any other change would occur, and checks that each copy's Python
syntax tree matches its original's after normalisation of the permitted path-string literals.

Usage (from the package folder):
    python tools/make_run_copies.py --overwrite          # write run_code/ for this folder
    python tools/make_run_copies.py --check              # regenerate in memory and compare with run_code/
Verification options (used when the package was built):
    --source_dir DIR  --source_user NAME  --repo_root PATH  --brfss_xpt PATH  --style windows|posix  --out_dir DIR
"""
import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLACEHOLDER_USER = "<user>"
ORIG_T = "C:\\Users\\{u}\\Research center\\Heartseanse2.0"
BRFSS_T = "C:\\Users\\{u}\\Downloads\\Dataset\\BRFSS2024\\archive (1)\\main_data_brfss_2024.XPT"
JAN_OOF = "\\artifacts\\phase2_fttf_dual_precbiased_gate\\phase2_fttf_dual_precbiased_gate_oof.csv"
SEL_OOF = ["runs", "brfss", "phase2_fttf_dual_precbiased_gate_selectedfeatures",
           "phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv"]

# original relative path -> name of the execution copy (same 14 scripts as in the verified runs)
PATH_COPY_SCRIPTS = {
    "nhanes_merge_check.py": "nhanes_merge_check.py",
    "nhanes_cleaning_step1.py": "nhanes_cleaning_step1.py",
    "nhanes_create_target.py": "nhanes_create_target.py",
    "nhanes_baseline_comparison_models.py": "nhanes_baseline_comparison_models.py",
    "nhanes_confusion_matrix_comparison_plot.py": "nhanes_confusion_matrix_comparison_plot.py",
    "generate_dataset_summary.py": "generate_dataset_summary.py",
    "brfss2024_phase2_all_models_allfeatures_master.py": "brfss2024_phase2_all_models_allfeatures_master.py",
    "artifacts/phase2_fttf_allfeatures_normal/plot_grouped_metrics.py": "plot_grouped_metrics.py",
    "simple_Avarage_fusion.py": "simple_Avarage_fusion.py",
    "confusion_metrix_allmodel.py": "confusion_metrix_allmodel.py",
    "FP/FN_bar_chart.py": "FN_bar_chart.py",
    "plot_threshold_curve.py": "plot_threshold_curve.py",
    "heatmap.py": "heatmap.py",
    "plot_calibration_brier_ece.py": "plot_calibration_brier_ece.py",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Mapper:
    def __init__(self, source_user, repo_root, brfss_xpt, style):
        self.orig = ORIG_T.format(u=source_user)
        self.brfss_src = BRFSS_T.format(u=source_user)
        self.root = str(repo_root).rstrip("\\/")
        self.brfss = str(brfss_xpt)
        self.sep = "\\" if style == "windows" else "/"

    def join(self, *parts):
        return self.sep.join([self.root] + [p.replace("\\", self.sep).replace("/", self.sep) for p in parts if p])

    def map_literal(self, lit: str) -> str:
        o = self.orig
        if lit == self.brfss_src:
            return self.brfss
        if lit == o + JAN_OOF:
            return self.join(*SEL_OOF)
        if lit.startswith(o + "\\Dataset2"):
            return self.join("data", "nhanes") + lit[len(o + "\\Dataset2"):].replace("\\", self.sep)
        if lit.startswith(o + "\\artifacts\\nhanes_"):
            return self.join("runs", "nhanes", "nhanes_") + lit[len(o + "\\artifacts\\nhanes_"):].replace("\\", self.sep)
        if lit.startswith(o + "\\artifacts\\"):
            return self.join("runs", "brfss") + self.sep + lit[len(o + "\\artifacts\\"):].replace("\\", self.sep)
        raise SystemExit(f"[STOP] unmapped path literal: {lit}")

    def rewrite_line(self, line: str, mapping: dict) -> str:
        out, i = line, 0
        while True:
            hits = [k for k in (out.find(self.orig, i), out.find(self.brfss_src, i)) if k >= 0]
            if not hits:
                return out
            start = min(hits)
            q = out[start - 1] if start > 0 else ""
            if q not in ("'", '"'):
                raise SystemExit(f"[STOP] path is not at the start of a string literal:\n{line}")
            end = out.find(q, start)
            lit = out[start:end]
            new = self.map_literal(lit)
            mapping[lit] = new
            out = out[:start] + new + out[end:]
            i = start + len(new) + 1


def ast_equal_modulo_paths(src: str, new: str, mapping: dict) -> bool:
    """True if the syntax trees match after normalising the permitted path-string literals (each rewritten path is mapped)."""
    t_src, t_new = ast.parse(src), ast.parse(new)
    for node in ast.walk(t_src):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in mapping:
            node.value = mapping[node.value]
    return ast.dump(t_src) == ast.dump(t_new)


def generate(source_dir: Path, source_user: str, repo_root: str, brfss_xpt: str, style: str):
    m = Mapper(source_user, repo_root, brfss_xpt, style)
    files, changes = {}, []
    for rel, name in PATH_COPY_SCRIPTS.items():
        raw = (source_dir / rel).read_bytes()
        text = raw.decode("utf-8")
        old_lines = text.splitlines(keepends=True)
        mapping = {}
        new_lines = [m.rewrite_line(l, mapping) for l in old_lines]
        new_text = "".join(new_lines)
        for i, (a, b) in enumerate(zip(old_lines, new_lines), 1):
            if a != b:
                changes.append({"script": name, "line": i, "old": a.rstrip("\r\n"), "new": b.rstrip("\r\n")})
        if not ast_equal_modulo_paths(text, new_text, mapping):
            raise SystemExit(f"[STOP] {name}: syntax tree differs beyond path strings")
        if m.orig in new_text or (m.brfss_src != m.brfss and m.brfss_src in new_text):
            raise SystemExit(f"[STOP] {name}: unmapped path left")
        files[name] = new_text.encode("utf-8")
    return files, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default=str(REPO / "original_code"))
    ap.add_argument("--source_user", default=PLACEHOLDER_USER, help="user-name segment in the source paths")
    ap.add_argument("--repo_root", default=str(REPO))
    ap.add_argument("--brfss_xpt", default=None,
                    help="raw BRFSS file (default: $HS_BRFSS_XPT, else <repo_root>/data/brfss/main_data_brfss_2024.XPT)")
    ap.add_argument("--style", choices=["windows", "posix"], default="windows" if os.name == "nt" else "posix")
    ap.add_argument("--out_dir", default=str(REPO / "run_code"))
    ap.add_argument("--overwrite", action="store_true", help="replace existing files in out_dir")
    ap.add_argument("--check", action="store_true", help="only compare with out_dir; write nothing")
    args = ap.parse_args()
    sep = "\\" if args.style == "windows" else "/"
    brfss = args.brfss_xpt or os.environ.get("HS_BRFSS_XPT") or sep.join([args.repo_root.rstrip("\\/"), "data", "brfss", "main_data_brfss_2024.XPT"])
    files, changes = generate(Path(args.source_dir), args.source_user, args.repo_root, brfss, args.style)
    out = Path(args.out_dir)
    record = {"repo_root": args.repo_root, "brfss_xpt": brfss, "style": args.style,
              "path_lines_changed": len(changes), "sha256": {n: sha256_bytes(b) for n, b in files.items()}}
    if args.check:
        bad = [n for n, b in files.items() if not (out / n).is_file() or (out / n).read_bytes() != b]
        print(json.dumps({"compared": len(files), "differing": bad, "result": "PASS" if not bad else "FAIL"}, indent=2))
        sys.exit(0 if not bad else 1)
    existing = [n for n in files if (out / n).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"[STOP] {len(existing)} file(s) already in {out}; add --overwrite to replace them")
    out.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (out / n).write_bytes(b)
    (out / "_generated_for.json").write_text(json.dumps(dict(record, changes=changes), indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "sha256"}, indent=2))
    print(f"[OK] {len(files)} execution copies written to {out}; only path strings differ from original_code/.")


if __name__ == "__main__":
    main()
