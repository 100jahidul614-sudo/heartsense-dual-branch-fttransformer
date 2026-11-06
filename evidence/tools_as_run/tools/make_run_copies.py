"""
Stage 0 helper for the HeartSense reproduction (non-scientific).

What it does (and nothing else):
  1. Copies the 19 original scripts byte-for-byte into  original_code/  (same relative paths).
  2. Copies the 18 original NHANES .xpt modules into   data/nhanes/.
  3. Copies the original saved outputs into            reference/  (read-only comparison targets).
  4. Writes path-only execution copies of the 14 scripts that cannot take paths
     as command-line arguments into                   run_code/.
     Only string literals that start with the original project path are rewritten.
     Every changed line is written to PATH_CHANGES.diff, and the script stops if any
     other line would differ.
  5. Writes SHA-256 manifests for everything it copied or generated.

It never writes inside the original project and refuses to overwrite an existing target.

Usage (Windows, from the HeartSense_repro folder):
    .venv\\Scripts\\python.exe tools\\make_run_copies.py
Usage elsewhere (paths as they appear on that machine):
    python make_run_copies.py --orig_root <Heartseanse2.0> --repro_root <HeartSense_repro>
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

WIN_ORIG = r"C:\Users\<user>\Research center\Heartseanse2.0"
WIN_REPRO = r"C:\Users\<user>\Research center\HeartSense_repro"
WIN_BRFSS_XPT = r"C:\Users\<user>\Downloads\Dataset\BRFSS2024\archive (1)\main_data_brfss_2024.XPT"

# ---------------------------------------------------------------- what is copied
# Scripts that accept every path on the command line: used exactly as written.
CLI_SCRIPTS = [
    "nhanes_phase1_feature_fusion_cache_builder.py",
    "nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py",
    "brfss2024_phase1_feature_fusion_cache_builder.py",
    "brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py",
    "update_master_table_add_experiment.py",
]
# Scripts whose paths are written into the code (or that ignore CLI arguments):
# original relative path -> name of the execution copy in run_code/
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
NHANES_XPT = ["P_DEMO", "P_BMX", "P_BPXO", "P_BPQ", "P_DIQ", "P_MCQ", "P_PAQ", "P_SMQ", "P_ALB_CR",
              "P_BIOPRO", "P_CBC", "P_GHB", "P_GLU", "P_HDL", "P_HSCRP", "P_INS", "P_TCHOL", "P_TRIGLY"]
# Original outputs used as comparison targets: original relative path -> reference relative path
REFERENCE = {
    "Dataset2/nhanes_2017_2020_merged_raw.csv": "nhanes/data/nhanes_2017_2020_merged_raw.csv",
    "Dataset2/nhanes_cleaned_step1.csv": "nhanes/data/nhanes_cleaned_step1.csv",
    "Dataset2/nhanes_final_with_target.csv": "nhanes/data/nhanes_final_with_target.csv",
    "artifacts/nhanes_phase1_feature_fusion": "nhanes/nhanes_phase1_feature_fusion",
    "artifacts/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures": "nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures",
    "artifacts/nhanes_baseline_comparison": "nhanes/nhanes_baseline_comparison",
    "artifacts/nhanes_figures": "nhanes/nhanes_figures",
    "artifacts/phase1_feature_fusion": "brfss/phase1_feature_fusion",
    "artifacts/phase2_fttf_dual_precbiased_gate_selectedfeatures": "brfss/phase2_fttf_dual_precbiased_gate_selectedfeatures",
    "artifacts/phase2_fttf_dual_precbiased_gate": "brfss/phase2_fttf_dual_precbiased_gate_JANUARY_RUN",
    "artifacts/phase2_allmodels_allfeatures": "brfss/phase2_allmodels_allfeatures",
    "artifacts/confusion_matrix_comparison": "brfss/confusion_matrix_comparison",
    "artifacts/threshold_analysis": "brfss/threshold_analysis",
    "artifacts/calibration_analysis": "brfss/calibration_analysis",
    "FP/fp_fn_plot_data.csv": "brfss/FP/fp_fn_plot_data.csv",
    "FP/fp_fn_bar_chart.png": "brfss/FP/fp_fn_bar_chart.png",
    "threshold_heatmap_f1.png": "brfss/threshold_heatmap/threshold_heatmap_f1.png",
}

# ---------------------------------------------------------------- path rules (Windows strings inside the scripts)
# Applied in order; the first matching prefix wins. Anything else under WIN_ORIG is an error.
PATH_RULES = [
    # The analysis scripts read the January copy of the proposed-model OOF file. The April re-run of the
    # same script (artifacts/phase2_fttf_dual_precbiased_gate_selectedfeatures) produced the same
    # predictions (max |diff| 1.4e-7); in the reproduction they read the new run's OOF file, whose
    # 'y_prob' column the final script writes for backward compatibility.
    (WIN_ORIG + "\\artifacts\\phase2_fttf_dual_precbiased_gate\\phase2_fttf_dual_precbiased_gate_oof.csv",
     WIN_REPRO + "\\runs\\brfss\\phase2_fttf_dual_precbiased_gate_selectedfeatures\\phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv"),
    (WIN_ORIG + "\\Dataset2", WIN_REPRO + "\\data\\nhanes"),
    (WIN_ORIG + "\\artifacts\\nhanes_", WIN_REPRO + "\\runs\\nhanes\\nhanes_"),
    (WIN_ORIG + "\\artifacts\\", WIN_REPRO + "\\runs\\brfss\\"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_line(line: str) -> str:
    if WIN_ORIG not in line:
        return line
    out = line
    for old, new in PATH_RULES:
        if old in out:
            out = out.replace(old, new)
            break
    if WIN_ORIG in out:
        raise SystemExit(f"[STOP] unmapped original-project path in line:\n{line}")
    return out


def copy_file(orig: Path, src_rel: str, repro: Path, dst_rel: str, manifest: list, tag: str) -> None:
    src, dst = orig / src_rel, repro / dst_rel
    if dst.exists():
        raise SystemExit(f"[STOP] target already exists, refusing to overwrite: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    hs, hd = sha256(src), sha256(dst)
    if hs != hd:
        raise SystemExit(f"[STOP] hash mismatch after copy: {src}")
    manifest.append({"type": tag, "source_rel": Path(src_rel).as_posix(), "copy_rel": Path(dst_rel).as_posix(),
                     "sha256": hd, "bytes": dst.stat().st_size})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_root", default=WIN_ORIG, help="Heartseanse2.0 folder as seen on this machine")
    ap.add_argument("--repro_root", default=WIN_REPRO, help="HeartSense_repro folder as seen on this machine")
    args = ap.parse_args()
    orig, repro = Path(args.orig_root), Path(args.repro_root)
    if not orig.is_dir():
        raise SystemExit(f"[STOP] original project not found: {orig}")
    if orig.resolve() in repro.resolve().parents or orig.resolve() == repro.resolve():
        raise SystemExit("[STOP] repro_root must not be inside the original project")

    manifest = []
    # 1. original scripts, byte-identical
    for rel in CLI_SCRIPTS + list(PATH_COPY_SCRIPTS):
        copy_file(orig, rel, repro, f"original_code/{rel}", manifest, "original_script")
    # 2. NHANES raw modules
    for m in NHANES_XPT:
        copy_file(orig, f"Dataset2/{m}.xpt", repro, f"data/nhanes/{m}.xpt", manifest, "nhanes_xpt")
    # 3. reference outputs
    for rel, ref in REFERENCE.items():
        src = orig / rel
        if src.is_dir():
            for f in sorted(p for p in src.rglob("*") if p.is_file()):
                sub = f.relative_to(src).as_posix()
                copy_file(orig, f"{rel}/{sub}", repro, f"reference/{ref}/{sub}", manifest, "reference_output")
        else:
            copy_file(orig, rel, repro, f"reference/{ref}", manifest, "reference_output")
    # 4. path-only execution copies + diff
    diff_lines, change_log = [], []
    for rel, name in PATH_COPY_SCRIPTS.items():
        src = orig / rel
        text = src.read_bytes().decode("utf-8")
        old_lines = text.splitlines(keepends=True)
        new_lines = [rewrite_line(l) for l in old_lines]
        for i, (a, b) in enumerate(zip(old_lines, new_lines), 1):
            if a != b:
                change_log.append({"script": name, "line": i, "old": a.rstrip("\r\n"), "new": b.rstrip("\r\n")})
        dst = repro / "run_code" / name
        if dst.exists():
            raise SystemExit(f"[STOP] target already exists, refusing to overwrite: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes("".join(new_lines).encode("utf-8"))
        changed = [(i, a, b) for i, (a, b) in enumerate(zip(old_lines, new_lines), 1) if a != b]
        if changed:
            diff_lines.append(f"=== original_code/{rel}  ->  run_code/{name}  ({len(changed)} line(s) changed)\n")
            for i, a, b in changed:
                diff_lines.append(f"@@ line {i}\n- {a.rstrip(chr(13) + chr(10))}\n+ {b.rstrip(chr(13) + chr(10))}\n")
            diff_lines.append("\n")
        manifest.append({"type": "run_copy", "source_rel": Path(rel).as_posix(), "copy_rel": f"run_code/{name}",
                         "sha256": sha256(dst), "bytes": dst.stat().st_size})
    # safety check: every changed line differs only inside a path literal that started with WIN_ORIG
    for c in change_log:
        old, new = c["old"], c["new"]
        start = old.find(WIN_ORIG)
        prefix_ok = new[:start] == old[:start]
        # the part after the literal's closing quote must be unchanged
        q = old[start - 1] if start > 0 else '"'
        old_tail = old[old.find(q, start):]
        new_tail = new[new.find(q, start):]
        if not (prefix_ok and old_tail == new_tail):
            raise SystemExit(f"[STOP] non-path change detected in {c['script']} line {c['line']}")
    (repro / "PATH_CHANGES.diff").write_text("".join(diff_lines), encoding="utf-8")
    (repro / "manifests").mkdir(exist_ok=True)
    with open(repro / "manifests" / "path_changes.json", "w", encoding="utf-8") as f:
        json.dump(change_log, f, indent=2)
    with open(repro / "manifests" / "stage0_copies.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(repro / "manifests" / "stage0_copies.sha256", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['copy_rel']}\n")
    # 6. baseline hashes of the whole original project (virtual environments excluded), so that every later
    #    stage can confirm the original project is still unchanged
    with open(repro / "manifests" / "original_project_baseline.sha256", "w", encoding="utf-8") as f:
        files = []
        for dirpath, dirnames, filenames in os.walk(orig):
            if Path(dirpath) == orig:
                dirnames[:] = [d for d in dirnames if d not in (".venv", ".venv312")]
            files += [Path(dirpath) / n for n in filenames]
        for p in sorted(files, key=lambda q: q.relative_to(orig).as_posix()):
            f.write(f"{sha256(p)}  {p.relative_to(orig).as_posix()}\n")
    counts = {}
    for m in manifest:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
    print(json.dumps({"copied": counts, "path_lines_changed": len(change_log),
                      "scripts_with_path_changes": sorted({c['script'] for c in change_log})}, indent=2))


if __name__ == "__main__":
    main()
