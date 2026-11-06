# Reproducibility record

The original scripts were re-run on 5–6 November 2026, with seeds, preprocessing, feature selection, architecture and hyperparameters unchanged. The run used a separate folder on the author's computer; the original project was only read, and all 277 of its files were confirmed unchanged after every stage. Only file paths were changed (27 lines in 14 execution copies; `PATH_CHANGES.diff`).

The full report is `evidence/reports/FINAL_REPRODUCIBILITY_REPORT.md`; the stage reports are `STAGE0`–`STAGE4`. Paths in those reports are relative to the reproduction folder. In this package, its `reports/`, `logs/` and `manifests/` folders are under `evidence/`, and `runs/` is `verified_outputs/` (see `evidence/README.md`).

## 1. Result

**Reproduced exactly.** Every trained model's out-of-fold predictions are bit-identical to the original saved outputs: the proposed model on BRFSS and NHANES, 11 BRFSS baselines and 3 NHANES baselines. So are the Phase-1 feature lists (181 BRFSS / 141 NHANES, same order) and all confusion matrices.

| Stage | Content | Run time | Original output files compared |
| --- | --- | --- | --- |
| 1 | NHANES cohort, Phase 1, proposed model, baselines, Fig 15 | 2.9 min | 17 identical; 2 numerically identical (stored paths only); 2 earlier Fig 15 drafts not written by the current script |
| 2 | BRFSS Phase 1, Table 3, proposed model + ablation | 185.0 min | 11 identical; 2 numerically identical (paths only); 3 exploratory files stored in the output folder |
| 3 | 11 BRFSS baselines, master table, Fig 4 | 71.9 min | 34 identical; master table within 5×10⁻¹⁰ (the saved file is rounded to 9 decimals); 1 radar plot from a script outside the pipeline |
| 4 | Error, threshold, sensitivity and calibration analyses | 0.4 min | 9 identical; calibration table within 2.4×10⁻¹² (computed originally from a January copy of the predictions, max difference 1.4×10⁻⁷); 1 PDF differing only in its creation date (0 differing pixels) |

**Overall:** 84 original output files were compared. 71 are identical, 6 numerically equal, 1 differs in metadata only, and 6 are files the current scripts do not write. No genuine differences.

## 2. Environment

The run used Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128 (cuDNN 9.10.02) and an NVIDIA RTX 5070 Ti Laptop GPU with driver 591.91, with the 47 pinned versions captured from the project's `.venv312` interpreter during the reproduction (5 November 2026); the versions installed for the original January 2026 experiments were not recorded (`env/ENVIRONMENT.md`). XGBoost and CatBoost are absent, as in the original environment.

## 3. What was changed to run the code

- Only path string literals, in 14 execution copies. Five scripts take every path as a command-line argument and ran from unchanged copies.
- One mapping is not a pure folder move. Three Stage 4 scripts (`confusion_metrix_allmodel.py`, `plot_threshold_curve.py`, `plot_calibration_brier_ece.py`) read a January 2026 copy of the proposed model's out-of-fold file. They were pointed to the reproduced selected-features run of the same script, whose predictions agree with the January copy to 1.4×10⁻⁷, with no decision change at any threshold used.

## 4. Key verified numbers

| | Value |
| --- | --- |
| BRFSS proposed model, 5-fold mean | F1 0.4294, precision 0.3359, recall 0.5991, balanced accuracy 0.7380, ROC-AUC 0.8030, PR-AUC 0.3659 |
| BRFSS proposed model, pooled at 0.50 | TN 359,639 / FP 50,487 / FN 16,975 / TP 25,363 (F1 0.4292) |
| BRFSS 3-model simple average | FP 37,835 / FN 20,464, F1 0.4287 |
| BRFSS calibration, Brier / ECE | raw 0.0944 / 0.1027; Platt 0.0696 / 0.0041; isotonic 0.0695 / 0.0006 |
| NHANES proposed model, pooled | TN 7,626 / FP 679 / FN 374 / TP 501, F1 0.4876 |
| NHANES AUROC / AUPRC | final output 0.8062 / 0.4252; gate-only output 0.8736 / 0.4677 |

## 5. Published values vs the original saved outputs (recorded as found; the paper is not corrected here)

Of 200 published values checked:
- 183 equal the original saved outputs after rounding;
- 7 equal them only when truncated rather than rounded (recall 0.59 in the abstract and Table 6, Table 6 balanced accuracy 0.73, Table 9 LightGBM F1 0.212, and three Table 12 calibration values);
- 10 differ.

The reproduction equals the saved outputs in all 200 cases.

| Paper element | Published | Original saved output = reproduced | Note |
| --- | --- | --- | --- |
| Abstract PR-AUC | 0.38 | 0.3659 (fold mean) | 0.3632 pooled |
| Table 6, proposed: accuracy | 0.86 | 0.8509 | |
| Table 6, proposed: ROC-AUC | 0.79 | 0.8030 | |
| Table 6, proposed: PR-AUC | 0.38 | 0.3659 | |
| Table 6, proposed: precision | 0.35 | 0.3359 | |
| Table 10, final model: F1 | 0.430 | 0.4292 (pooled) | fold mean 0.4294 |
| Table 10, final model: precision | 0.349 | 0.3344 (pooled) | |
| Table 15, proposed: AUROC | 0.8736 | 0.8062 | the published value equals the saved gate-only row |
| Table 15, proposed: AUPRC | 0.4677 | 0.4252 | the published value equals the saved gate-only row |
| Table 15, Simple Avg: balanced accuracy | 0.7716 | 0.7706 | |

Full table: `evidence/reports/published_vs_reproduced_20261106_031053.csv`.

## 6. Not covered by this reproduction

| Paper element | Reason |
| --- | --- |
| Table 5 (numeric vs one-hot), Table 11 (imputation ablation) | Supplementary ablations; not re-run |
| SHAP (Figs 13–14), Table 13 statistics | Post-hoc analyses with a surrogate model; not re-run |
| Table 7 paired tests | The input file the original script expects does not exist in the project, so it cannot run as set up |

## 7. Evidence

| Evidence | Location |
| --- | --- |
| Per-step command, working folder, start/end time, exit code and console output, plus an environment snapshot | `evidence/logs/stage1_*`–`stage4_*` |
| File-by-file comparisons | `evidence/reports/compare_stage*.md` / `.csv` |
| SHA-256 manifests of every output, of the copies made from the original project, and of the original-project baseline | `evidence/manifests_verified_runs/` |
| Exact tool versions used for the verified runs | `evidence/tools_as_run/` |
| Checks made when this package was built | `evidence/package_build/` |

The package-build checks confirm three things:
- the distribution copies differ from the originals only in the user-folder path, with matching Python syntax trees after normalisation of the permitted path-string literals;
- the execution copies differ only in path strings, with matching Python syntax trees after normalisation of the permitted path-string literals;
- regenerating the execution copies reproduces those used in the runs byte for byte.
