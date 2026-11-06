# Contribution map and attribution

> Items marked **[AUTHOR REVIEW]** need the author's confirmation before this package is shared. This document does
> not infer who wrote a particular section from the style of comments in the code.

## 1. What is established by the evidence in this package

- The 19 scripts in `original_code/` come from the author's research project folder. Their copies are byte-identical to the project files apart from the user-folder path (`evidence/manifests_verified_runs/stage0_copies.json`, `manifests/sanitisation_map.csv`).
- These scripts produced the outputs saved in that project. Re-running them unchanged on the author's computer reproduced those outputs bit for bit (`REPRODUCIBILITY.md`).
- The results correspond to the paper on which Jahidul Islam is the first author (Sensors 2026, 26, 3417). **[AUTHOR REVIEW: confirm name form and author list wording]**

## 2. Contribution map: what each part of the code implements

| Area | Files (`original_code/`) | Lines | What the code implements |
| --- | --- | --- | --- |
| Proposed model | `brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py`, `nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | 1,072 / 1,498 | An FT-Transformer written from PyTorch building blocks (feature tokenizer, pre-norm attention blocks, pooled MLP head). It also covers two asymmetric losses (recall branch; precision branch with α_pos and negative weighting), GateNet with its supervision target and inference bias, the veto/rescue rule, and fold-wise imputation and scaling with an inner validation split. Training uses mixed precision with AUPRC early stopping and deterministic seeding. Every ablation component is logged |
| Feature selection | `brfss2024_phase1_feature_fusion_cache_builder.py`, `nhanes_phase1_feature_fusion_cache_builder.py` | 581 / 695 | Three selectors (Pearson, TabNet mask importance, HistGradientBoosting permutation importance with a lower-confidence-bound rule) run inside each CV training fold, fused per fold and then combined into one union across folds (`CODE_WALKTHROUGH.md` §1.3 notes what this means for fold independence), with overlap and stability reports |
| Baseline benchmark | `brfss2024_phase2_all_models_allfeatures_master.py`, `nhanes_baseline_comparison_models.py`, `update_master_table_add_experiment.py` | 980 / 603 / 263 | 11 BRFSS baselines (classical, ensemble and PyTorch deep models) under identical folds, with SMOTE inside the training folds and a calibrated linear SVM. Metrics are written to one master table |
| Data engineering | `nhanes_merge_check.py`, `nhanes_cleaning_step1.py`, `nhanes_create_target.py`, `generate_dataset_summary.py` | 50 / 29 / 72 / 132 | Merging 18 NHANES modules, missingness filtering, cardiovascular target construction, and BRFSS cohort statistics (Table 3) |
| Analysis and figures | `simple_Avarage_fusion.py`, `confusion_metrix_allmodel.py`, `FP/FN_bar_chart.py` (run as `run_code/FN_bar_chart.py`), `plot_threshold_curve.py`, `heatmap.py`, `plot_calibration_brier_ece.py`, `nhanes_confusion_matrix_comparison_plot.py`, `artifacts/…/plot_grouped_metrics.py` | 31–273 each | Error analysis, threshold curves, veto/rescue sensitivity, cross-fitted calibration (Brier, ECE), and the paper figures |

For a quick inspection of the core implementation, start with the BRFSS Phase-2 script, lines 203–320 (model and losses), 399–480 and 582–630 (GateNet, veto/rescue), and 739–860 (the cross-validation loop). `docs/METHOD.md` maps every step to its lines.

## 3. Authorship and tool assistance

| Part | Attribution |
| --- | --- |
| Research design, experiments and the paper | The author and co-authors, as stated in the paper. **[AUTHOR REVIEW: describe the author's own role, and any co-author contributions to the code]** |
| Original scripts in `original_code/` (development before November 2026) | **[AUTHOR REVIEW REQUIRED]** The extent to which AI coding assistants or other people contributed to writing these scripts is not documented in the project and is not inferred here. Some comments address a reader in the second person (for example `# … [per your request: NO class_weight for LGBM]` in the baseline script, line 13). Comment style alone is not treated as evidence of authorship. Suggested wording, to be confirmed or replaced by the author: *"Designed, implemented, debugged and run by the author; AI coding assistance was used for parts of the code."* |
| Training runs for the verified reproduction | Run by the author on the author's own computer (Windows PowerShell stage scripts; transcripts kept privately), after reviewing each stage's pre-flight check |
| November 2026 reproduction automation, comparison tools, reports, documentation and packaging (`tools/`, `00*.ps1`–`04*.ps1`, `evidence/reports/`, `docs/`, `README.md` and the other package documents) | **Prepared with AI assistance** (Claude, Anthropic) under the author's direction and review. These files are not part of the original research code |

## 4. What this package does not claim

- It does not claim that the published paper is correct or incorrect. Where published numbers differ from the saved outputs, both are listed as found (`REPRODUCIBILITY.md` §5).
- It does not include the scripts for the paper elements that were not re-run here:
  - SHAP (Figs 13–14);
  - Table 13 statistics;
  - Table 7 paired tests (their expected input file does not exist in the project);
  - Table 5 one-hot and Table 11 imputation ablations.
