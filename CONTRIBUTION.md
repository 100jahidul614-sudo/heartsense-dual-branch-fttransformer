Research Contributions and Implementation Map
This repository presents the research implementation and experimental materials supporting “Leakage-Safe Precision-Aware Dual-Branch FT-Transformer for Population-Scale Heart Disease Risk Prediction”, published in Sensors (2026), 26(11), 3417 (DOI: 10.3390/s26113417). It brings together the original HeartSense Python scripts, available experimental outputs, and technical documentation so that supervisors and research collaborators can inspect the methods and follow the computational workflow.

The underlying model and experiments were developed during the original research project. The subsequent November 2026 work organised, documented, and verified those existing materials for academic review; it was not a newly developed implementation of the published method.
1. Original research and implementation
Jahidul Islam, the paper's first author, initiated the HeartSense research idea, proposed the approach to his academic supervisors, and led its practical development and experimental evaluation. He directed the implementation process, executed the Python scripts, inspected model outputs, troubleshot errors, assessed experimental results, and determined the subsequent development and analysis steps.

The research implementation was developed with coding assistance from Claude and ChatGPT, under Jahidul's direction and technical review. This assistance supported the programming work; the research decisions, hands-on execution, evaluation, and interpretation described above were undertaken by Jahidul.

The repository preserves 19 scripts from the original research project in original_code/. The shared copies retain their research logic, with personal Windows user-folder paths sanitised for distribution; the changes are recorded in manifests/sanitisation_map.csv and evidence/manifests_verified_runs/stage0_copies.json. The run_code/ directory provides 14 execution copies with path adjustments for the verification workspace. These copies are organisational adaptations of the existing scripts, not new model implementations.
2. Code and experiment contribution map
The following map identifies the main scripts and what they implement. It is intended to help readers inspect the actual research code and connect it to the methods and results described in the article.
Research area	Principal files	Implementation and experimental role
Proposed dual-branch model	original_code/brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py, original_code/nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py	PyTorch FT-Transformer components; recall- and precision-biased branches; GateNet training and inference; veto/rescue rules; data preprocessing, training, validation, early stopping, and model evaluation.
Feature selection	original_code/brfss2024_phase1_feature_fusion_cache_builder.py, original_code/nhanes_phase1_feature_fusion_cache_builder.py	Pearson correlation, TabNet mask importance, and HistGradientBoosting permutation-importance selection; within-fold fusion and construction of the shared Phase-2 feature set.
Baseline comparisons	original_code/brfss2024_phase2_all_models_allfeatures_master.py, original_code/nhanes_baseline_comparison_models.py, original_code/update_master_table_add_experiment.py	Classical, ensemble, and deep-learning baseline evaluations; cross-validation and training-fold resampling where applicable; NHANES comparisons; aggregation of model results. The executed BRFSS comparison comprises 11 baseline models.
Dataset preparation	original_code/nhanes_merge_check.py, original_code/nhanes_cleaning_step1.py, original_code/nhanes_create_target.py, original_code/generate_dataset_summary.py	NHANES module merging, cleaning and target construction, plus cohort and dataset summary statistics.
Evaluation and figures	original_code/simple_Avarage_fusion.py, original_code/confusion_metrix_allmodel.py, original_code/FP/FN_bar_chart.py, original_code/plot_threshold_curve.py, original_code/heatmap.py, original_code/plot_calibration_brier_ece.py, original_code/nhanes_confusion_matrix_comparison_plot.py	Confusion-matrix and error analysis, threshold analysis, calibration metrics and visualisations, and evaluation figures. Additional plotting material is documented in CODE_WALKTHROUGH.md.


For a code-first technical review, the BRFSS Phase-2 script is a useful starting point for understanding the architecture, asymmetric losses, GateNet, and training/evaluation flow. CODE_WALKTHROUGH.md and docs/METHOD.md provide detailed implementation guidance.
3. Academic supervision and research contributions
Dr Dristi Datta and Dr Fowzia Akhter provided academic supervision, methodological guidance, feedback, and manuscript review. According to the implementation history supplied for this repository, they did not directly write or modify the Python research scripts. Their broader scholarly roles are recognised in the published article.

This document explains the practical implementation and repository provenance. The article's formal Author Contributions statement remains the authoritative record for publication-level contributor roles.
4. Organisation and verification of the original materials
In November 2026, the existing HeartSense research files were arranged into a structured, reviewable repository. Jahidul ran the verification stages on his computer, examined the outputs, and directed the checks. Claude and ChatGPT assisted with supporting tasks such as execution automation, environment checks, comparison utilities, documentation, and packaging.

The verification work checked the original saved feature selections, predictions, confusion matrices, baseline results, and related analysis outputs. Many results reproduced exactly, including byte-for-byte matches; REPRODUCIBILITY.md records the detailed comparisons, including differences between some saved outputs and numbers printed in the published article. The original research scripts and scientific results have been preserved rather than rewritten to match those printed values.

The repository's documentation is organised to make these materials straightforward to inspect:
- README.md — research overview, repository contents, and starting points.
- CODE_WALKTHROUGH.md and docs/METHOD.md — model implementation, feature-selection procedure, and execution flow.
- REPRODUCIBILITY.md — rerun evidence, output comparisons, environment information, and the scope of completed verification.
- DATA.md — dataset sources, preparation, and access instructions for data not distributed with the repository.
- original_code/, run_code/, artifacts/, and evidence/ — research scripts, execution copies, results, and verification records.
For methodological precision, CODE_WALKTHROUGH.md explains that Phase-1 feature selections were combined across folds into the shared feature set subsequently used in Phase 2. REPRODUCIBILITY.md identifies which analyses were rerun from the available research archive and documents any unavailable historical inputs. These technical notes support an accurate reading of the implementation without changing the published work.

Purpose of this repository: to provide a clear, inspectable account of the original HeartSense implementation, the experiments conducted, the available research evidence, and the later verification of those materials for academic review.
