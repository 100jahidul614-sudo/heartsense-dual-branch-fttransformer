
# Research Contributions and Implementation Map

This repository presents the original research implementation and experimental materials supporting the peer-reviewed paper:

**Leakage-Safe Precision-Aware Dual-Branch FT-Transformer for Population-Scale Heart Disease Risk Prediction**

**Publication:** *Sensors*, 2026, 26(11), 3417  
**DOI:** https://doi.org/10.3390/s26113417

The proposed method is referred to throughout this documentation as the **Dual-Branch FT-Transformer**.

The repository brings together the Python research scripts, model implementations, feature-selection procedures, baseline experiments, evaluation results, and technical documentation developed during the original research.

The research was initially developed under the working name *HeartSense*, which remains in some historical folder names and project artifacts.

The November 2026 repository preparation involved organising and documenting these existing research materials and independently checking their reproducibility. The underlying research implementation was already developed and used during the original study.

## 1. Original Research and Technical Implementation

### Research initiation and development

**Jahidul Islam**, the first author of the published paper, initiated the original research idea and proposed it to his academic supervisors.

Following discussions and methodological guidance from his supervisors, he led the practical development, experimental implementation, and evaluation of the proposed approach.

His responsibilities included:

- Developing and refining the research approach.
- Directing the implementation of the proposed machine-learning architecture.
- Preparing and processing the research datasets.
- Implementing and evaluating feature-selection strategies.
- Executing the model-training and validation experiments.
- Examining model predictions and evaluation metrics.
- Identifying implementation errors, troubleshooting problems, and refining the experimental workflow.
- Conducting baseline comparisons and experimental analyses.
- Interpreting the experimental findings and contributing to manuscript preparation.

### Python implementation and coding assistance

The Python research implementation was developed with coding assistance from **Claude and ChatGPT**, under Jahidul's direction and technical review.

Jahidul personally executed the research scripts, examined the experimental results, investigated errors, evaluated model behaviour, and determined subsequent development and analysis steps.

AI-assisted programming supported the implementation process, while the research direction, experimental decisions, practical execution, validation, and interpretation were managed by Jahidul in collaboration with his academic supervisors.

### Original research scripts

The repository preserves **19 Python scripts from the original research project**, located in:

`original_code/`

These scripts implement the research workflow used to develop and experimentally evaluate the proposed Dual-Branch FT-Transformer.

The archived copies retain the original computational logic. Personal Windows user-directory paths were sanitised where necessary for repository distribution.

The path adjustments are documented in:

- `manifests/sanitisation_map.csv`
- `evidence/manifests_verified_runs/stage0_copies.json`

The repository also provides **14 execution copies** in `run_code/`, with environment-specific path adjustments to support independent execution and verification of the existing implementation.

## 2. Code and Experiment Contribution Map

The following table identifies the principal research scripts, their implementation responsibilities, and their connection to the published methodology.

| Research Area | Principal Python Files | Implementation and Experimental Contribution |
| --- | --- | --- |
| **Proposed Dual-Branch FT-Transformer** | `original_code/brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py`, `original_code/nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | Implementation of the PyTorch FT-Transformer components, recall-biased and precision-biased branches, asymmetric training objectives, GateNet, veto/rescue decision rules, preprocessing, cross-validation, training, inference, and evaluation. |
| **Feature Selection** | `original_code/brfss2024_phase1_feature_fusion_cache_builder.py`, `original_code/nhanes_phase1_feature_fusion_cache_builder.py` | Implementation of Pearson correlation, TabNet mask-importance selection, HistGradientBoosting permutation-importance selection, within-fold feature fusion, and construction of the shared Phase-2 feature set. |
| **Baseline Model Comparisons** | `original_code/brfss2024_phase2_all_models_allfeatures_master.py`, `original_code/nhanes_baseline_comparison_models.py`, `original_code/update_master_table_add_experiment.py` | Evaluation of classical machine-learning, ensemble, and deep-learning baselines; cross-validation, training-fold resampling where applicable, NHANES baseline comparisons, and consolidated performance reporting. |
| **Dataset Preparation** | `original_code/nhanes_merge_check.py`, `original_code/nhanes_cleaning_step1.py`, `original_code/nhanes_create_target.py`, `original_code/generate_dataset_summary.py` | NHANES data integration, cleaning, cardiovascular target construction, and dataset descriptive analysis. |
| **Model Evaluation and Analysis** | `original_code/simple_Avarage_fusion.py`, `original_code/confusion_metrix_allmodel.py`, `original_code/FP/FN_bar_chart.py`, `original_code/plot_threshold_curve.py`, `original_code/heatmap.py`, `original_code/plot_calibration_brier_ece.py`, `original_code/nhanes_confusion_matrix_comparison_plot.py` | Confusion-matrix analysis, false-positive and false-negative analysis, threshold evaluation, calibration assessment, performance visualisation, and supporting manuscript figures. |

### Proposed model implementation

The core model scripts contain the principal components of the proposed architecture:

1. FT-Transformer feature tokenisation and attention-based representation learning.
2. Separately trained recall-oriented and precision-oriented prediction branches.
3. Asymmetric loss functions for controlling the trade-off between sensitivity and precision.
4. GateNet for combining branch predictions.
5. Decision rules involving prediction bias, veto, and rescue mechanisms.
6. Cross-validation, preprocessing, training, validation, and performance evaluation.

The code also implements deterministic seeding, mixed-precision training, inner validation, and early-stopping procedures.

### Feature-selection implementation

The Phase-1 scripts implement three complementary feature-selection approaches:

- Pearson correlation.
- TabNet feature-mask importance.
- HistGradientBoosting permutation importance with a lower-confidence-bound selection rule.

The selectors operate on Phase-1 training folds, and the selected features are fused to produce the shared feature set used in Phase 2.

The exact cross-validation and shared-feature construction procedure is explained in `CODE_WALKTHROUGH.md` and `docs/METHOD.md`.

### Baseline experiments

The original experimental workflow includes **11 executed BRFSS baseline models** and NHANES baseline comparisons.

These experiments provide comparative evaluation of the proposed method against conventional machine-learning, ensemble, and deep-learning approaches.

The original benchmark scripts and saved results are preserved for technical inspection.

## 3. Academic Collaboration and Supervision

The published research was conducted by:

- **Jahidul Islam** — First author.
- **Dr Dristi Datta** — Co-author and academic supervisor.
- **Dr Fowzia Akhter** — Co-author and academic supervisor.

### Jahidul Islam

Jahidul initiated the original research idea and led the practical Python implementation and experimental development.

He directed the coding workflow, executed the experiments, investigated errors, analysed the results, and participated in preparing the research manuscript.

### Dr Dristi Datta and Dr Fowzia Akhter

Dr Dristi Datta and Dr Fowzia Akhter contributed academic supervision, methodological guidance, research feedback, and manuscript review.

According to the project's implementation history, the supervisors did not directly write or modify the Python research scripts.

Their broader scientific contributions are acknowledged in the published article.

The journal's formal **Author Contributions** statement remains the authoritative record of publication-level responsibilities.

## 4. Organisation and Reproducibility Verification

Following publication, the existing research implementation and experimental materials were organised into a structured repository to facilitate technical inspection and independent review.

This work was carried out in November 2026.

### Organisation of existing research materials

The repository preparation included:

- Collecting and organising the original Python research scripts.
- Preserving their computational logic.
- Preparing path-adjusted execution copies where necessary.
- Organising saved experimental outputs and analysis artifacts.
- Documenting the research environment and software dependencies.
- Preparing implementation walkthroughs and execution instructions.
- Recording source-file provenance and integrity checks.

These activities focused on improving the accessibility and traceability of the existing research implementation.

### Reproducibility verification

The original research scripts were subsequently rerun in a controlled environment to verify the available experimental outputs.

The verification process included:

- Dataset preparation and feature-selection outputs.
- BRFSS and NHANES model evaluation.
- Baseline model comparisons.
- Prediction and confusion-matrix calculations.
- Performance metrics and analytical outputs.
- Available evaluation figures and supporting results.

The verification reproduced core saved results, including exact matches for many outputs.

Detailed comparisons, numerical precision differences, historical output variations, and the scope of completed reruns are documented in `REPRODUCIBILITY.md`.

The original scientific implementation and saved experimental results were preserved throughout this process.

### Verification and documentation assistance

Jahidul personally ran the reproduction stages, examined the outputs, reviewed the verification findings, and directed the organisation of the research materials.

Claude and ChatGPT assisted with the supporting automation, environment checks, comparison utilities, technical documentation, and repository preparation.

The verification utilities and repository documentation complement the original research scripts and provide additional transparency for independent technical review.

## 5. Repository Guide for Research Reviewers

The repository is organised to help supervisors and research collaborators navigate the original implementation and supporting evidence.

| Location | Purpose |
| --- | --- |
| `README.md` | Research overview, principal contributions, and repository navigation. |
| `original_code/` | Original Python research implementation. |
| `run_code/` | Path-adjusted execution copies of the research scripts. |
| `CODE_WALKTHROUGH.md` | Detailed explanation of code components and implementation flow. |
| `docs/METHOD.md` | Research methodology and computational procedures. |
| `REPRODUCIBILITY.md` | Re-execution evidence, output verification, environment information, and comparison results. |
| `DATA.md` | Dataset sources, preparation, and data-access instructions. |
| `verified_outputs/` | Verified experimental outputs, evaluation results, and analysis materials. |
| `evidence/` | Supporting verification records and research provenance. |
| `manifests/` | File-integrity and source-mapping records. |

### Suggested technical review sequence

Readers who wish to inspect the core implementation can begin with:

**1. Feature selection**

`original_code/brfss2024_phase1_feature_fusion_cache_builder.py`

**2. Proposed architecture and model training**

`original_code/brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py`

**3. Baseline evaluation**

`original_code/brfss2024_phase2_all_models_allfeatures_master.py`

**4. NHANES evaluation**

`original_code/nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py`

**5. Reproducibility and experimental outputs**

`REPRODUCIBILITY.md`

For detailed code-level explanations, see `CODE_WALKTHROUGH.md`.

## 6. Research Provenance and Documentation

This repository provides an organised record of the Dual-Branch FT-Transformer research implementation, associated experiments, and available verification evidence.

The distinction between the original research materials and the subsequent organisation and reproducibility work is maintained through the repository structure, source manifests, and technical documentation.

The published article remains the authoritative reference for the reported research findings and formal author contributions.

Detailed methodological information, including the Phase-1 shared-feature selection procedure, is provided in `docs/METHOD.md` and `CODE_WALKTHROUGH.md`.

The available experimental evidence, comparisons with reported manuscript values, software environment, and execution requirements are documented in `REPRODUCIBILITY.md`.

---

## Purpose of This Repository

To provide researchers, academic supervisors, and collaborators with a clear and technically inspectable account of the **original Dual-Branch FT-Transformer implementation, experimental workflow, model evaluation, research contributions, and reproducibility evidence** supporting the published study.
