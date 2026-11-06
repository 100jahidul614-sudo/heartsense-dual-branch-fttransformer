# HeartSense: final reproducibility report (Stages 0–4)

**Paper:** *Leakage-Safe Precision-Aware Dual-Branch FT-Transformer for Population-Scale Heart Disease Risk Prediction*, Sensors 2026, 26, 3417, doi:10.3390/s26113417. First author: Jahidul Islam.
**Reproduction:** 5–6 November 2026, in `HeartSense_repro\`, a separate folder next to the original project `Heartseanse2.0\`. The original project was only read, never written.
**Scope:** the 19 original scripts behind the paper's main results:
- NHANES external cohort;
- BRFSS-2024 feature selection;
- the proposed model and its ablation;
- 11 baselines;
- error, threshold, sensitivity and calibration analyses.

## Summary

| | Result |
| --- | --- |
| Reproduction status | **Reproduced exactly.** Every trained model's out-of-fold predictions are bit-identical to the original saved outputs: the proposed model on both datasets, 11 BRFSS baselines and 3 NHANES baselines. So are the feature lists and confusion matrices. |
| Original output files compared | 84: 71 identical, 6 numerically equal (largest difference 5×10⁻¹⁰; stored paths or rounding only), 1 PDF differing only in its creation date, 6 reference-only files that the current pipeline does not write. 0 genuine differences |
| Published numbers checked | 200 printed values. 183 equal the original saved outputs after rounding, 7 only when truncated, 10 differ. All 200 reproduced values equal the saved outputs (112 bit-identical, 88 within 2.4×10⁻¹² or explained by 9-decimal rounding of the saved file) |
| Original project | All 277 files unchanged after every stage (SHA-256 baseline) |
| Code changes | None to the science. 14 execution copies differ from the originals in 27 lines, all file paths. 5 scripts ran unchanged with paths passed as arguments |
| Total run time | 260 min on one laptop GPU: Stage 1 2.9 min, Stage 2 185.0 min, Stage 3 71.9 min, Stage 4 0.4 min |

## 1. Execution environment

| Component | Value |
| --- | --- |
| Computer | The author's Windows 11 laptop: Intel Core Ultra 9 275HX (24 threads), 31.4 GB RAM |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB, compute capability 12.0; driver 591.91 (CUDA 13.1) |
| Python | 3.12.10 (64-bit) |
| PyTorch | 2.9.1+cu128 (CUDA 12.8 build, cuDNN 9.10.02); mixed precision as in the original scripts |
| Key packages | scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, LightGBM 4.6.0, imbalanced-learn 0.14.1, pytorch-tabnet 4.1.0, matplotlib 3.10.8, pyreadstat 1.3.3 |
| Package set | `env\requirements.txt` pins the 47 packages of the original interpreter `.venv312`; the 48th package there is pip itself. A new `.venv` was built to these versions, so `.venv312` was never touched. Versions at the time of the original runs were not recorded |
| Not installed (as in the original) | XGBoost, CatBoost. The baseline script skips them when they are missing, so installing them would change the run |
| Runner settings | `MPLBACKEND=Agg`, `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`; none affects any computation |

Evidence: `logs\stage0_env\` (`env_report.json`, `pip_freeze.txt`, `nvidia_smi.txt`, GPU and CPU reports), plus `environment.json` in each stage's log folder.

## 2. Datasets and provenance

| Dataset | Source | File(s) used | Check |
| --- | --- | --- | --- |
| BRFSS 2024 | CDC Behavioral Risk Factor Surveillance System, 2024 annual data (LLCP2024), obtained through a Kaggle re-upload (`rudritarahman/cdc-brfss-survey-data-2024`) | `main_data_brfss_2024.XPT`, 1,093,874,240 bytes, read in place | SHA-256 `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c`, verified before every BRFSS stage. Whether this file is byte-identical to CDC's own LLCP2024 download was not checked |
| NHANES 2017–March 2020 (pre-pandemic) | CDC National Health and Nutrition Examination Survey, 18 modules: DEMO, BMX, BPXO, BPQ, DIQ, MCQ, PAQ, SMQ, ALB_CR, BIOPRO, CBC, GHB, GLU, HDL, HSCRP, INS, TCHOL, TRIGLY | `P_*.xpt`, copied to `data\nhanes\` | SHA-256 of each copy equals the original |

- **BRFSS cohort:** 457,670 rows; 452,464 with a valid `_MICHD` label (42,338 positive, 9.3572%). `CVDINFR4` and `CVDCRHD4` are removed as label-defining columns. 283 numeric candidate features remain after cleaning.
- **NHANES cohort:** 15,560 participants after merging on SEQN; 173 variables after the >70%-missing filter; 9,180 participants with the CVD target (875 positive, 9.53%); 163 candidate features.

## 3. Original scripts

All 19 are kept byte-identical in `original_code\` (SHA-256 in `manifests\stage0_copies.json`).

| Stage | Script | Lines | Ran as | Paper element |
| --- | --- | --- | --- | --- |
| 1 | `nhanes_merge_check.py`, `nhanes_cleaning_step1.py`, `nhanes_create_target.py` | 50 / 29 / 72 | run copy (paths) | Table 14 cohort |
| 1 | `nhanes_phase1_feature_fusion_cache_builder.py` | 695 | original (arguments) | NHANES Phase 1 (141 features) |
| 1 | `nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | 1,498 | original (arguments) | Tables 14–15 proposed model and ablation |
| 1 | `nhanes_baseline_comparison_models.py` | 603 | run copy | Table 15 baselines |
| 1 | `nhanes_confusion_matrix_comparison_plot.py` | 199 | run copy | Fig 15 |
| 2 | `brfss2024_phase1_feature_fusion_cache_builder.py` | 581 | original (arguments) | BRFSS Phase 1 (181 features) |
| 2 | `generate_dataset_summary.py` | 132 | run copy | Table 3 |
| 2 | `brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | 1,072 | original (arguments) | Proposed model, Tables 8–10, abstract |
| 3 | `brfss2024_phase2_all_models_allfeatures_master.py` | 980 | run copy | 11 baselines, Tables 4 and 6 |
| 3 | `update_master_table_add_experiment.py` | 263 | original (arguments) | Proposed-model row of the master table |
| 3 | `artifacts\phase2_fttf_allfeatures_normal\plot_grouped_metrics.py` | 82 | run copy | Fig 4 |
| 4 | `simple_Avarage_fusion.py`, `confusion_metrix_allmodel.py`, `FP\FN_bar_chart.py` | 31 / 273 / 89 | run copies | Tables 8–9, Figs 5, 7 |
| 4 | `plot_threshold_curve.py`, `heatmap.py`, `plot_calibration_brier_ece.py` | 79 / 51 / 172 | run copies | Figs 8–11, Table 12 |

In total there are 6,951 lines of original code.

- `PATH_CHANGES.diff` and `manifests\path_changes.json` list every changed line.
- Three Stage 4 scripts (`confusion_metrix_allmodel.py`, `plot_threshold_curve.py`, `plot_calibration_brier_ece.py`) originally read a January 2026 copy of the proposed model's out-of-fold file. The run copies read the file reproduced in Stage 2 instead. The two differ by at most 1.4×10⁻⁷, with no decision changes.

## 4. Method as implemented (summary)

1. **Phase 1, feature selection per training fold**, inside each of 5 stratified folds (seed 42), then combined across folds. Each fold uses three selectors:
   - Pearson |r| ≥ 0.08;
   - TabNet importance ≥ 0.75/N;
   - HistGradientBoosting permutation importance, μ − 1.7σ > 2×10⁻⁴ (5 repeats on a 30,000-row subsample).

   The fold's selection is the union of the three. The final list is the union over the five folds: BRFSS 181 features (101/106/105/100/111 per fold); NHANES 141 features (103/108/109/98/107).

   Phase 2 uses this one shared union in every fold. Every respondent is in the training part of four of the five Phase-1 folds, so the complete procedure does not maintain strict outer-fold-independent feature selection.
2. **Phase 2, dual-branch FT-Transformer**, 5-fold CV with fold-wise median imputation and standard scaling.
   - **Architecture:** two separate FT-Transformers, each with 48-dimensional feature tokens, 2 pre-norm transformer blocks with 4 attention heads, mean pooling and an MLP head.
   - **Training:** AdamW, lr 3×10⁻⁴, batch 1,024, ≤30 epochs, early stopping on inner-validation AUPRC with patience 6, mixed precision.
   - **Recall branch:** BCE with pos_weight = negatives/positives.
   - **Precision branch:** pos_weight divided by α_pos = 1.3, and negative samples weighted ×4.
3. **GateNet**: a 3→16→1 MLP on (p_prec, p_rec, |p_prec − p_rec|).
   - It is trained for 6 epochs on the inner validation split. Its target is 1 when the precision branch is right and the recall branch wrong.
   - At inference a bias of 0.9 is added to its logit, and g weights the two branches: p = g·p_prec + (1 − g)·p_rec.
4. **Veto/rescue rule**:
   - Veto: p = 0 when p_prec < 0.20.
   - Rescue: p = 1 when p_rec > 0.98.
   - Decision threshold 0.50.

## 5. Results by stage

### Stage 1: NHANES (2.9 min)

- **Files:** 17 identical, 2 summary JSONs numerically identical (only stored paths differ), 2 earlier Fig 15 drafts reference-only.
- **Proposed model, pooled:** TN 7,626 / FP 679 / FN 374 / TP 501; F1 0.487591.
- **Baselines:** LightGBM, RF and CNN are identical.

### Stage 2: BRFSS core (185.0 min)

- **Files:** 11 identical, 2 summary JSONs numerically identical (paths only), 3 reference-only (exploratory files stored in the output folder).
- **Phase 1:** the 181 features and the five fold lists are identical in content and order.
- **Proposed model:** the out-of-fold file (452,464 × 7) is identical.
  - Pooled: TN 359,639 / FP 50,487 / FN 16,975 / TP 25,363.
  - Fold-mean F1 0.4294, pooled F1 0.4292.
- **Table 10 ablation:** identical.

### Stage 3: BRFSS baselines (71.9 min)

- **Files:** 34 identical, 1 master table within 5×10⁻¹⁰, 1 reference-only (a radar plot from a script outside the pipeline).
- **Per-model files:** for all 11 models, the per-fold metrics, out-of-fold probabilities and confusion matrices are byte-identical, including the four PyTorch models.
- **Master table:** the saved original stores values rounded to 9 decimals.
- **Fig 4:** identical.

### Stage 4: analyses (0.4 min)

- **Files:** 9 identical, 1 calibration CSV within 2.4×10⁻¹² (computed originally from the January out-of-fold copy), 1 PDF that differs only in `/CreationDate` (0 differing pixels when rendered).
- **Hard-coded values:** two figure scripts contain hard-coded counts.
  - `FN_bar_chart.py`: the Simple Avg FP/FN, typed in from Table 8.
  - `nhanes_confusion_matrix_comparison_plot.py`: all five NHANES confusion matrices.
  - Both equal the reproduced computed values.

Stage reports: `reports\STAGE1_REPRODUCTION_REPORT.md` to `STAGE4_REPRODUCTION_REPORT.md`, plus `STAGE0_READINESS_REPORT.md`.

## 6. Key metrics (reproduced = original saved outputs)

| Model (BRFSS, 5-fold mean at 0.50) | Accuracy | BalAcc | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed (FTTF_Dual_Head) | 0.8509 | 0.7380 | 0.3359 | 0.5991 | 0.4294 | 0.8030 | 0.3659 |
| RF | 0.8517 | 0.7284 | 0.3318 | 0.5768 | 0.4212 | 0.8494 | 0.3730 |
| LogReg | 0.7522 | 0.7706 | 0.2453 | 0.7933 | 0.3747 | 0.8491 | 0.3799 |
| CNN1D | 0.7418 | 0.7727 | 0.2400 | 0.8108 | 0.3703 | 0.8505 | 0.3808 |
| LightGBM | 0.9089 | 0.5603 | 0.5572 | 0.1313 | 0.2126 | 0.8526 | 0.3831 |

The full 12-row table is in `runs\brfss\phase2_allmodels_allfeatures\master_results_phase2_allfeatures.csv` and the Stage 3 report.

| Analysis | Reproduced value |
| --- | --- |
| BRFSS 3-model simple average (LightGBM + RF + CNN1D) | FP 37,835, FN 20,464, F1 0.4287 |
| Calibration (Table 12): Brier / ECE | Raw 0.0944 / 0.1027; Platt 0.0696 / 0.0041; Isotonic 0.0695 / 0.0006 |
| NHANES proposed (pooled) | F1 0.4876; final-model AUROC / AUPRC 0.8062 / 0.4252; gate-only 0.8736 / 0.4677 |

## 7. Published values vs original saved outputs (recorded as found, not corrected)

Ten printed values differ from the original saved outputs, which the reproduction matches exactly.

| Paper element | Published | Original saved = reproduced | Note |
| --- | --- | --- | --- |
| Abstract PR-AUC | 0.38 | 0.3659 (fold mean) | 0.3632 pooled |
| Table 6 proposed: accuracy | 0.86 | 0.8509 | |
| Table 6 proposed: ROC-AUC | 0.79 | 0.8030 | |
| Table 6 proposed: PR-AUC | 0.38 | 0.3659 | |
| Table 6 proposed: precision | 0.35 | 0.3359 | |
| Table 10 final model: F1 | 0.430 | 0.4292 (pooled) | the fold mean is 0.4294 |
| Table 10 final model: precision | 0.349 | 0.3344 (pooled) | |
| Table 15 proposed: AUROC | 0.8736 | 0.8062 | the published value equals the saved gate-only row |
| Table 15 proposed: AUPRC | 0.4677 | 0.4252 | the published value equals the saved gate-only row |
| Table 15 Simple Avg: balanced accuracy | 0.7716 | 0.7706 | |

Seven further values match only when truncated rather than rounded:
- recall 0.59 in the abstract and in Table 6;
- Table 6 balanced accuracy 0.73;
- Table 9 LightGBM F1 0.212;
- Table 12 Platt Brier, Platt ECE and Isotonic ECE.

All other 183 checked values match. Source: `reports\published_vs_reproduced_20261106_031053.csv`.

## 8. Paper elements not covered by this reproduction

Each item below was excluded in the approved plan.

| Element | Reason |
| --- | --- |
| Table 5 (numeric vs one-hot) and Table 11 (imputation ablation) | Supplementary ablations; their original scripts exist but were not run |
| SHAP (Figs 13–14) and Table 13 statistics | Post-hoc analyses using a surrogate model; not run. The original SHAP outputs from the January run are kept in `reference\` for completeness |
| Table 7 paired tests | The input the original script expects (`simple_avg_fusion_metrics.csv`) does not exist in the project, so the test cannot run as set up |

## 9. File hashes and evidence

| Manifest | Covers | SHA-256 of the manifest |
| --- | --- | --- |
| `manifests\original_project_baseline.sha256` | 277 original project files (virtual environments excluded) | `6b527361…99f8ec` |
| `manifests\stage0_copies.sha256` / `.json` | 143 copies: 19 original scripts, 14 run copies, 18 NHANES modules, 92 reference outputs | `f1bb2e24…5d3d` / `05baffb9…a4f` |
| `manifests\stage1_outputs.sha256` | 43 Stage 1 files | `0d97c597…f8a6` |
| `manifests\stage2_outputs.sha256` | 32 Stage 2 files | `e3989717…d9ea8` |
| `manifests\stage3_outputs.sha256` | 54 Stage 3 files | `5148adca…9353` |
| `manifests\stage4_outputs.sha256` | 33 Stage 4 files | `58165ec9…9e4f` |
| `PATH_CHANGES.diff` | the 27 path lines | `fc1020c1…4f29` |
| `env\requirements.txt` | the 47 pinned packages | `91b1f629…8747` |

All four stage manifests and the 277-file baseline were re-verified on 6 November 2026 after Stage 4 (`tools\verify_stage0.py`: PASS). Logs for every step are in `logs\stage1_*` to `logs\stage4_*`: command, working folder, start and end times, exit code, full console output and environment. PowerShell transcripts are in `logs\ps_transcripts\`.
