# Contribution map and attribution

This document distinguishes the original research, the Python implementation, and the November 2026 reproducibility work. The account of individual contributions in Section 3 was confirmed by Jahidul Islam for this repository. The formal contribution declaration in the published paper remains the publication's authoritative record of manuscript authorship and associated roles.

## 1. What is established by the evidence in this package

- The 19 scripts in `original_code/` were collected from the HeartSense research project folder. The distributed copies preserve the research code, with only the personal Windows user-folder paths sanitised (`evidence/manifests_verified_runs/stage0_copies.json`, `manifests/sanitisation_map.csv`).
- The research scripts were rerun in an isolated reproduction workspace. Core saved predictions, feature lists, confusion matrices and other outputs were reproduced, with byte-level matches and the limited rounding, metadata and historical-file differences documented in `REPRODUCIBILITY.md`.
- The project corresponds to the peer-reviewed article by Jahidul Islam and co-authors in *Sensors* (2026), 26(11), 3417. Jahidul Islam is the first author. Publication authorship and the repository's implementation-history statement are related but distinct forms of attribution.

## 2. Contribution map: what each part of the code implements

This table describes the **functionality contained in each script**. It does not imply that any individual manually wrote every line of code.

| Area | Files (`original_code/`) | Lines | What the code implements |
| --- | --- | --- | --- |
| Proposed model | `brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py`, `nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | 1,072 / 1,498 | An FT-Transformer built from PyTorch components: feature tokenizer, pre-norm attention blocks and pooled MLP head. Two separately trained branches use asymmetric losses (recall-biased and precision-biased with `alpha_pos` and negative weighting). The scripts implement GateNet training and inference, its bias, veto/rescue rules, fold-wise imputation and scaling, inner validation, mixed-precision training, AUPRC-based early stopping, deterministic seeding and ablation evaluation. |
| Feature selection | `brfss2024_phase1_feature_fusion_cache_builder.py`, `nhanes_phase1_feature_fusion_cache_builder.py` | 581 / 695 | Three selectors (Pearson correlation, TabNet mask importance, HistGradientBoosting permutation importance with a lower-confidence-bound rule) operate on each Phase-1 CV training fold. Their selections are fused within folds and subsequently combined into a shared union used for Phase 2. As explained in `CODE_WALKTHROUGH.md` §1.3, this does **not** preserve strict outer-fold-independent feature selection. |
| Baseline benchmark | `brfss2024_phase2_all_models_allfeatures_master.py`, `nhanes_baseline_comparison_models.py`, `update_master_table_add_experiment.py` | 980 / 603 / 263 | Eleven executed BRFSS baselines spanning classical, ensemble and PyTorch deep-learning models; fold-wise evaluation, including SMOTE within training folds where used, and calibrated linear SVM probabilities. NHANES baseline comparisons and consolidated results tables are also included. XGBoost and CatBoost were defined but not executed in the original baseline run. |
| Data engineering | `nhanes_merge_check.py`, `nhanes_cleaning_step1.py`, `nhanes_create_target.py`, `generate_dataset_summary.py` | 50 / 29 / 72 / 132 | Merging 18 NHANES modules, missingness filtering, cardiovascular target construction and BRFSS cohort statistics (Table 3). |
| Analysis and figures | `simple_Avarage_fusion.py`, `confusion_metrix_allmodel.py`, `FP/FN_bar_chart.py` (executed as `run_code/FN_bar_chart.py`), `plot_threshold_curve.py`, `heatmap.py`, `plot_calibration_brier_ece.py`, `nhanes_confusion_matrix_comparison_plot.py`, `artifacts/…/plot_grouped_metrics.py` | 31–273 each | Confusion-matrix and error analysis, threshold curves, veto/rescue sensitivity, cross-fitted calibration (Brier score and ECE), and manuscript figures. Some figure scripts contain fixed numerical values; those are identified and checked in `CODE_WALKTHROUGH.md`. |

For an initial technical review, start with the BRFSS Phase-2 script: lines 203–320 (architecture and losses), 399–480 and 582–630 (GateNet and veto/rescue), and 739–860 (cross-validation and training flow). `CODE_WALKTHROUGH.md` and `docs/METHOD.md` give further function- and line-level explanations.

## 3. Research contributions, implementation process and AI assistance

### Jahidul Islam — research initiation and technical development

Jahidul Islam independently conceived the initial HeartSense research idea and proposed it to his supervisors. After discussing the approach and receiving their approval, he led the practical development and experimental implementation of the project.

The Python research implementation was developed **with coding assistance from Claude and ChatGPT**. Jahidul directed the development, personally executed the scripts, examined experimental outputs, identified errors, troubleshot problems and decided how to proceed with subsequent iterations. This describes substantive hands-on implementation and experimental responsibility; it does **not** claim that he manually authored every line or that AI-generated code was absent.

### Academic supervisors — research guidance and manuscript development

Dr Dristi Datta and Dr Fowzia Akhter provided academic supervision, research and methodological guidance, feedback and manuscript review. According to Jahidul's account of the implementation process, neither supervisor directly wrote or modified the project's Python scripts. Their wider scholarly contributions and formal publication roles are acknowledged in the published article.

### November 2026 reproducibility and repository preparation

Jahidul personally ran the reproduction stages on his computer, inspected their results and directed the process. Reproduction automation, environment checks, comparison utilities, technical reports, code walkthroughs and repository packaging were developed **with AI assistance**, including Claude and ChatGPT, under his direction and review. These November 2026 materials are separate from the original research-code implementation and are identified in `tools/`, `evidence/`, `docs/` and the repository documentation.

### Attribution boundaries

- The original scripts are presented as the research project's implementation, **not** as code written entirely by hand by a single person.
- Use of AI coding assistance is disclosed; no claim is made that AI tools independently conceived, approved or conducted the research.
- The code-function map in Section 2 is a guide to implementation, not a claim of line-by-line authorship.
- A Git commit identifies who made that repository commit; it does not establish who originally authored each file.
- This document supplements, and does not replace or amend, the formal author-contribution statement in the published article.

## 4. What this package does not claim

- It does not independently certify every methodological or numerical claim in the published paper. Where published numbers differ from the saved and reproduced outputs, the differences are recorded in `REPRODUCIBILITY.md` §5 without changing the paper.
- It does not claim strict outer-fold-independent feature selection: the Phase-1 selections are combined across folds before Phase-2 evaluation.
- It does not include the underlying raw datasets or the full row-level BRFSS prediction files; data sources, excluded-file hashes and regeneration instructions are documented elsewhere in the repository.
- It does not establish that a complete fresh training run will work on every computer, operating system or CPU-only environment.
- It does not include all scripts for paper elements that were not rerun in this reproduction, including SHAP (Figs 13–14), Table 13 statistics, Table 7 paired tests (the expected input file was unavailable in the project), Table 5 one-hot and Table 11 imputation ablations.
- It does not grant a public software licence or establish permission to redistribute any third-party material. Repository access and wider distribution should be handled separately from this attribution statement.
