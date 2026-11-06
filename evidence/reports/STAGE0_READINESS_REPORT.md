# Stage 0 readiness report — 2026-11-05

Stage 0 prepared and verified everything that does not need the GPU. No model has been trained.
The one remaining Stage 0 item is the environment check, which must run on the Windows PC (`00a_gpu_check.ps1`, then `00_setup_env.ps1`).

## Verification results

| Item | Result | Evidence |
| --- | --- | --- |
| `HeartSense_repro` created next to the project, not inside it | PASS | this folder |
| 19 original scripts copied to `original_code\` | PASS, byte-identical (SHA-256) | `manifests\stage0_copies.json` |
| 18 NHANES modules copied to `data\nhanes\` | PASS, byte-identical | same |
| 92 original output files copied to `reference\` | PASS, byte-identical | same |
| 14 execution copies in `run_code\` | PASS: 27 lines differ, all file paths; every other line identical | `PATH_CHANGES.diff`, `manifests\path_changes.json` |
| No execution copy points into `Heartseanse2.0` | PASS | `tools\verify_stage0.py` |
| Original project unchanged | PASS: 277 files match the baseline (virtual environments excluded) | `manifests\original_project_baseline.sha256` |
| Raw BRFSS file | PASS: SHA-256 `7d53cf08...aab36b1c`, 1,093,874,240 bytes | `tools\stages.json` |
| Pre-flight, Stage 1 (NHANES) | PASS | `tools\run_stage.py --stage 1 --check_only` |
| Pre-flight, Stage 2 (BRFSS core) | PASS | `--stage 2 --check_only` |
| Pre-flight, Stages 3-4 | Waiting, as expected: their inputs are produced by Stage 2 | `--stage 3/4 --check_only` |
| Comparison tools | Tested on a sandbox copy: identical outputs report IDENTICAL; a changed feature list and a flipped 0.50 decision report DIFFERS; a 1e-6 change reports MATCH_1e-4 | `tools\compare_outputs.py` |
| GPU, NVIDIA driver, CUDA, Python environment | **Not yet checked** (needs Windows) | run `00a_gpu_check.ps1`, `00_setup_env.ps1` |

## Path changes (summary)

| Original location | Reproduction location |
| --- | --- |
| `Heartseanse2.0\Dataset2\` | `HeartSense_repro\data\nhanes\` |
| `Heartseanse2.0\artifacts\nhanes_*` | `HeartSense_repro\runs\nhanes\nhanes_*` |
| `Heartseanse2.0\artifacts\<other>` | `HeartSense_repro\runs\brfss\<other>` |
| `artifacts\phase2_fttf_dual_precbiased_gate\phase2_fttf_dual_precbiased_gate_oof.csv` (January copy, read by 3 analysis scripts) | `runs\brfss\phase2_fttf_dual_precbiased_gate_selectedfeatures\phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv` (same script, same `y_prob` column; the original January and April runs gave identical predictions, max difference 1.4e-7) |

Five scripts take every path as a command-line argument and run unchanged from `original_code\`: both Phase-1 builders, both proposed-model scripts, and `update_master_table_add_experiment.py`.

## Settings applied by the runner (no effect on any computation)

`MPLBACKEND=Agg` (figures saved to files; `plt.show()` does not block), `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`.
`heatmap.py` and `FN_bar_chart.py` save into their working folder, so the runner starts them in `runs\brfss\threshold_heatmap\` and `runs\brfss\FP\`.

## Published numbers vs original saved outputs (already measurable)

The comparison table was run once with the original outputs in place of a reproduction, to check it.
Of 200 published numbers it checks, 183 equal the original saved outputs after rounding. Another 7 equal them only when truncated instead of rounded, for example recall 0.599 printed as 0.59. The other 10 differ:

| Published element | Published | Original saved output |
| --- | --- | --- |
| Abstract AUPRC | 0.38 | 0.366 (fold mean) |
| Table 6 proposed: ACC / ROC-AUC / PR-AUC / precision | 0.86 / 0.79 / 0.38 / 0.35 | 0.851 / 0.803 / 0.366 / 0.336 |
| Table 10 final: F1 / precision | 0.430 / 0.349 | 0.429 / 0.334 |
| Table 15 proposed: AUROC / AUPRC | 0.8736 / 0.4677 | 0.8062 / 0.4252 (the published values equal the saved gate-only row) |
| Table 15 Simple Avg BalAcc | 0.7716 | 0.7706 |

These differences belong to the original outputs. The reproduction reports them as found and does not try to close them.

## Next steps (on the Windows PC)

1. `powershell -ExecutionPolicy Bypass -File .\00a_gpu_check.ps1`. This installs nothing; it writes `logs\stage0_env\gpu_check_*.txt`.
2. `powershell -ExecutionPolicy Bypass -File .\00_setup_env.ps1`. This creates `.venv` with the `.venv312` package versions (about 3 GB download for PyTorch CUDA 12.8), then runs `tools\check_env.py` and `tools\verify_stage0.py`.
3. Send `logs\stage0_env\` for review. Stage 1 starts only after your approval.
