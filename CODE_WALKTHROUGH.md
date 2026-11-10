# Code walkthrough

This walkthrough follows the original implementation in the order the pipeline runs. Every reference points to a function and line in `original_code/`. The distribution copies keep the original line numbering; only the user folder in hard-coded default paths reads `<user>`.

The main reference is the BRFSS script; NHANES equivalents are given where they differ. This document was prepared with AI assistance from the code as it stands. It describes what the code does and makes no claims about who wrote which part (see `CONTRIBUTION.md`).

**Abbreviations used below:**

| Short name | File |
| --- | --- |
| **P1** | `brfss2024_phase1_feature_fusion_cache_builder.py` (581 lines) |
| **P2** | `brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py` (1,072 lines) |
| **NH-P2** | `nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py` (1,498 lines) |
| **BL** | `brfss2024_phase2_all_models_allfeatures_master.py` (980 lines) |

All defaults are module-level constants: P1 lines 55–88, P2 lines 46–105, BL lines 103–122.

## 1. Preprocessing and feature selection

### 1.1 BRFSS-2024: P1 `main()` (line 292) and P2 `main()` (line 687)
| Step | Code | What it does |
| --- | --- | --- |
| Load | P2 l.712; P1 l.325 | `pd.read_sas(..., format="xport")` on the raw LLCP2024 file |
| Leakage control | P2 l.715–718; P1 l.331–334 | Drops `CVDINFR4` and `CVDCRHD4` (`DEFAULT_LEAKAGE_COLS`), the two items that define the target |
| Target | `decode_brfss_target` P2 l.135; used l.721–725 | `_MICHD`: 2 → 0, keep {0, 1} rows. Gives 452,464 respondents, 42,338 positive |
| Numeric features | `safe_numeric_only` P2 l.114; l.729 | Keeps numeric columns only |
| Missing codes | `brfss_missing_to_nan` P2 l.117; P1 l.97 | Replaces the BRFSS codes {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999, 888, 8888, 88888} with NaN in every numeric column |
| Degenerate columns | `drop_all_nan_and_constant` P2 l.124 | Drops all-missing and constant columns, leaving 283 candidate features |

### 1.2 NHANES 2017–March 2020
| Step | Code | What it does |
| --- | --- | --- |
| Merge | `nhanes_merge_check.py` l.35–49 | Reads 18 `P_*.xpt` modules and left-joins them on `SEQN`: 15,560 × 282 |
| Missingness filter | `nhanes_cleaning_step1.py` l.13–21 | Keeps columns with ≤ 70% missing, leaving 173 |
| Target | `nhanes_create_target.py` `create_target` l.30–43 | `CVD_TARGET` = 1 if any of `MCQ160B/C/D/E` (heart failure, CHD, angina, heart attack) is 1. It is 0 if all are 2, otherwise the row is dropped. Gives 9,180 participants, 875 positive |
| In Phase 2 | NH-P2 l.945, 960–961, 978–982 | Reads the CSV, drops the `MCQ160B–E` leakage columns and the ID `SEQN`, and applies `nhanes_missing_to_nan` (l.122) |

### 1.3 Phase 1: multi-selector fusion per training fold, then across folds (P1)
- **Folds:** `StratifiedKFold(5, shuffle=True, random_state=42)` (l.369). The loop at l.378 uses only the training indices; the fold's validation part (`_va_idx`) is never read.
- **Imputation:** median, fitted on the training part (l.382).
- **Three selectors on the training part:**

| Selector | Function | Rule |
| --- | --- | --- |
| Linear | `compute_abs_pearson_threshold` l.114 (called l.389) | abs(Pearson r) ≥ 0.08 |
| Deep | `train_tabnet_and_get_thresholded` l.131 (called l.397) | TabNet mask importance ≥ α/N, α = 0.75 (threshold l.182) |
| Non-linear | `hgb_permutation_importance_lcb_positive` l.193 (called l.408) | HistGradientBoosting + permutation importance on a 30,000-row stratified subsample, 5 repeats. A feature is kept if μ − 1.7σ > 2×10⁻⁴ (score l.256) |

- **Fusion** (`fuse_sets_at_least_two` l.264):
  - The code forms *common* = features chosen by ≥ 2 selectors (l.268), and the features unique to each selector.
  - It returns common ∪ unique (l.274), which is the union of the three selections. Both sets are reported in `selector_overlap_table.csv`.
- **Across folds:** union of the five fold sets (l.434, l.468), written to `final_features_allfolds.json`.
  - BRFSS: 181 features; per fold 101 / 106 / 105 / 100 / 111.
  - NHANES: 141 features; per fold 103 / 108 / 109 / 98 / 107.
- **In Phase 2:** P2 reads this list (`load_phase1_features` l.158, which returns the union of whatever the JSON holds). It applies it to all of its folds (`apply_phase1_feature_filter` l.185, called l.734).
- **Scope of the fold isolation:** Selection happens inside each training fold, but the five fold selections are then combined into one shared union. Every respondent is in the training part of four of the five Phase-1 folds, so the shared list used in each Phase-2 outer fold was chosen partly with that fold's validation data. The complete procedure therefore does not maintain strict outer-fold-independent feature selection.

## 2. Dual-Branch FT-Transformer architecture (P2 l.203–285)

```
x ∈ R^F (standardised) ──► tokenize: t_j = x_j · W_j + b_j      W, b ∈ R^{F×48}    (FTTransformer.tokenize, l.241)
                         ──► 2 × TransformerBlock (pre-LN)                          (l.203–226)
                               h = h + Drop(MHA_4heads(LN(h)))
                               h = h + Drop(FFN(LN(h))),  FFN: 48→192→48, GELU
                         ──► LayerNorm → mean over the F tokens → dropout          (encode/forward, l.244–255)
                         ──► ProbHead: 48→256→64→1 (ReLU, dropout 0.08)            (l.257–271)
                         ──► logit                                                  (FTTransformerModel.forward_logits, l.279)
```

- **Initialisation:** the feature embeddings `feat_embed` start at 0.02·N(0, 1) and `feat_bias` at 0 (l.234–235).
- **Two models:** "dual-branch" means two independent `FTTransformerModel` instances with this same architecture, trained separately per fold with different losses (section 3). They do not share weights.
- **Settings:** `d_embed` 48, `n_heads` 4, `n_layers` 2, `ffn_mult` 4, `dropout` 0.08 (l.72–76). The NHANES script defines the same classes at NH-P2 l.241–360.

## 3. Recall-biased and precision-biased training

### 3.1 Cross-validation structure (P2 l.739–830)
1. **Outer folds:** `StratifiedKFold(5, shuffle=True, random_state=42)` (l.739), with the fold loop at l.752.
2. **Fold preprocessing:** median imputation (l.759) and `StandardScaler` (`make_scaler` l.138, used l.763) are fitted on the outer training part only.
3. **Inner split:** `StratifiedShuffleSplit(test_size=0.10, random_state=42+fold)` (l.768) gives inner-train (90%) and inner-validation (10%) parts of the training fold.
   - The inner-validation part is used for early stopping and as GateNet's training data.
   - The outer validation fold is used only for the out-of-fold predictions and metrics.

### 3.2 `train_one_branch` (l.485)
- **Shared settings:**
  - AdamW, lr 3×10⁻⁴, weight decay 10⁻⁵ (l.525), batch 1,024, at most 30 epochs;
  - mixed precision with gradient clipping at 1.0 (l.548, l.555);
  - after each epoch, AUPRC on the inner-validation part (`average_precision_score`, l.560); the best state is kept, with patience 6 (l.562–570, restored l.573).
- **Seeding:** `seed_all` (l.151) sets the NumPy, PyTorch and CUDA seeds to 42 and makes cuDNN deterministic.

| | Recall branch (l.777) | Precision branch (l.787–797) |
| --- | --- | --- |
| Positive-class weight | `pos_weight = n_neg / n_pos` of the inner-train part (`compute_pos_weight` l.144), about 9.7 on BRFSS | `max(1, (n_neg/n_pos) / α_pos)`, α_pos = 1.3 (l.787–788) |
| Loss | `nn.BCEWithLogitsLoss(pos_weight)` (l.523) | `NegWeightedBCEWithLogits` (l.300): the same BCE, with every negative sample's loss multiplied by `prec_neg_weight` = 4.0 (l.306–310) |
| Intended bias | Missing a positive is expensive, so recall is favoured | Positives are weighted less and false positives cost 4× more, so precision is favoured |

- **Options left at their defaults:** focal loss (`FocalBCEWithLogits` l.287, `use_focal = 0`), label smoothing 0, and the precision-branch temperature 1.0.
- **Early stopping in the verified runs:** best epochs were 27–30 (recall branch) and 15–30 (precision branch) per fold, from `*_metrics.csv`.

## 4. GateNet fusion and veto/rescue

### 4.1 GateNet target and training (P2 l.800–827, l.399–483)
- **Inputs:** both branches predict on the inner-validation part (l.800–801). The gate features are (p_prec, p_rec, |p_prec − p_rec|) (`build_gate_features` l.411).
- **Target** (l.807–812): `gate_y = 1` when the precision branch's 0.50 decision is correct and the recall branch's is wrong, otherwise 0.
- **Model:** `GateNet` (l.399), a 3 → 16 → 1 MLP with ReLU.
- **Training** (`train_gatenet` l.415):
  - BCE loss, AdamW lr 5×10⁻³, 6 epochs, batch 4,096.
  - It trains on 75% of the inner-validation samples and selects the epoch on the other 25% (`StratifiedShuffleSplit` l.815, seed 42+1000+fold).
  - The selection score is mean(1 − |g − gate_y|).

### 4.2 Inference on the outer fold (l.830–860)
1. **Gate weight** (`gatenet_predict_g_numpy` l.582): g = σ(GateNet logit + 0.9). `gate_bias` = 0.9 shifts the weight towards the precision branch.
2. **Gate-only probability** (l.848): p_gate = g·p_prec + (1 − g)·p_rec.
3. **Veto/rescue** (`apply_veto_rescue_numpy` l.609, applied l.851):
   - p = 0 where p_prec < `veto_v` = 0.20;
   - p = 1 where p_rec > `rescue_rhi` = 0.98;
   - the code applies the veto first, then the rescue.
4. **Decision:** the final out-of-fold probability is stored at l.860, and decisions use threshold 0.50.

**What the verified runs show** (from `*_metrics.csv` and `ablation/`):
- The veto applied to 72–74% of BRFSS samples per fold (73–85% on NHANES), and rescue never fired.
- The final and gate-only 0.50 decisions are therefore almost the same. On NHANES the final and gate-only 0.50 decisions are identical (FP 679, FN 374 in both). On BRFSS they differ by one false positive: 50,488 for gate-only versus 50,487 for the final output, with FN 16,975 in both; the veto moved one gate-only false positive below 0.50.
- Ranking metrics differ, because vetoed samples get probability 0. On NHANES the AUROC/AUPRC is 0.8062/0.4252 for the final output and 0.8736/0.4677 for the gate-only output.

### 4.3 Outputs and ablation (l.868–1058)
- **Fold metrics:** computed by `compute_metrics_at_050` (l.350) on the outer fold (l.868–871): accuracy, balanced accuracy, precision, recall and F1 at 0.50, plus ROC-AUC and AUPRC.
- **Written files:** pooled out-of-fold file (l.928), pooled confusion matrix (l.932), per-fold metrics (l.947) and the run summary with all settings (l.950–1000).
- **Ablation** (`eval_row` l.1010, rows l.1032–1037): the same pooled out-of-fold predictions are scored in five ways:
  - recall branch only;
  - precision branch only;
  - simple average of the two branches;
  - GateNet without veto/rescue;
  - the final model.

## 5. Baseline benchmarking and evaluation (BL)

- **Same data handling as P2:** the same leakage removal, missing-code handling and folds (5-fold, seed 42, l.727), but on **all 283 numeric features** rather than the Phase-1 list.
- **Fold preprocessing:** median imputation is fitted per training fold (l.754). Standard scaling is applied to models with `needs_scaling`.

| Model (registry `get_model_specs` l.612) | Implementation |
| --- | --- |
| LightGBM | `fit_predict_lgbm_smote` l.217: SMOTE on the training fold only (l.207), no class weight |
| Logistic regression | `fit_predict_lr` l.243: saga solver, `class_weight="balanced"` |
| Linear SVM | `fit_predict_svm` l.254: `LinearSVC(class_weight="balanced")` + `CalibratedClassifierCV` for probabilities |
| Decision tree / random forest | `fit_predict_dt` l.266, `fit_predict_rf` l.276: class-balanced |
| KNN / Gaussian NB | `fit_predict_knn` l.288, `fit_predict_nb` l.297 |
| AE + LR, 1-D CNN, LSTM, GRU | Model classes l.353–440. Trained by `torch_train_binary` (l.450): early stopping on validation AUPRC, at most 10 epochs, batch 2,048, lr 2×10⁻³, patience 3. Wrappers from l.513 |
| XGBoost, CatBoost | `fit_predict_xgb` l.306, `fit_predict_cat` l.328. Skipped at l.719–735 because the packages were not installed in the original environment, so they are absent from the results |

- **Metrics:** `safe_metrics` (l.169) gives the same measures as P2 at threshold 0.50, after clipping probabilities to [10⁻⁹, 1 − 10⁻⁹].
- **Per-model outputs:** per-fold metrics (l.906), pooled confusion matrix (l.919) and out-of-fold probabilities (l.923), plus one row per model in `master_results_phase2_allfeatures.csv` (l.972).
- **Proposed-model row:** `update_master_table_add_experiment.py` (`main` l.162) appends it from P2's metrics and confusion files.
- **NHANES baselines** (`nhanes_baseline_comparison_models.py`, `main` l.373): LightGBM, RF and a 1-D CNN (`CNN1DBaseline` l.206, `train_cnn_fold` l.252), on the Phase-1 features with the same folds.

**Post-hoc analyses**, all reading the saved out-of-fold files:
- `confusion_metrix_allmodel.py`: pooled confusion matrices, plus a LightGBM/RF/CNN1D probability average.
- `plot_threshold_curve.py`: thresholds 0.01–0.99.
- `heatmap.py`: re-applies veto/rescue to the gate-only output over v ∈ [0.10, 0.30] and r ∈ [0.90, 0.99].
- `plot_calibration_brier_ece.py`: Brier score and 15-bin ECE (`ece_score` l.32). Platt and isotonic calibration are cross-fitted over 5 folds (`crossfit_calibration` l.54).
- **Hard-coded values in two figure scripts:**
  - `run_code/FN_bar_chart.py` (Fig 5; execution copy of `original_code/FP/FN_bar_chart.py`, same line numbers) l.57–62 types in the simple-average FP/FN (37,835 / 20,464) from the manuscript's Table 8.
  - `nhanes_confusion_matrix_comparison_plot.py` types in its five confusion matrices.
  - Both equal the values computed from the out-of-fold files.

## 6. How the implementation was reproduced and verified

| Step | What was done | Evidence |
| --- | --- | --- |
| Isolation | The original project was only read. All work ran in a separate folder, and a SHA-256 baseline of all 277 project files was re-checked after every stage (unchanged) | `evidence/manifests_verified_runs/original_project_baseline.sha256`, `evidence/logs/stage*/summary.json` |
| Exact copies | The 19 scripts were copied byte for byte. Five take all paths as arguments and ran unchanged. The other 14 got execution copies where only path string literals were rewritten (27 lines) | `PATH_CHANGES.diff`, `evidence/manifests_verified_runs/path_changes.json` |
| Same settings | Seeds, preprocessing, feature selection, architecture and hyperparameters as in the scripts; the 47 pinned versions captured from the project's `.venv312` interpreter during the reproduction (5 November 2026); the versions installed for the original January 2026 experiments were not recorded; XGBoost/CatBoost left absent | `env/requirements.txt`, `evidence/logs/stage0_env/` |
| Guarded runs | A pre-flight check per stage confirmed script hashes, inputs present, outputs absent (no overwrite), all paths inside the run folder, and the raw-data SHA-256. Every step was logged with command, timing, exit code, console output and environment | `tools/run_stage.py`, `evidence/logs/` |
| Comparison | Every output was compared file by file with the original saved outputs: CSV values, probability flips at 0.50, JSON contents, image bytes | `evidence/reports/compare_stage*.md` |
| Published values | 200 printed values were checked against saved and reproduced outputs, and differences recorded as found | `evidence/reports/published_vs_reproduced_20261106_031053.csv`, `REPRODUCIBILITY.md` §5 |

**Outcome:**
- Every trained model's out-of-fold predictions are bit-identical to the original saved outputs. That covers both proposed models, 11 BRFSS baselines (including the four PyTorch models) and 3 NHANES baselines, as well as the Phase-1 feature lists (same order) and all confusion matrices.
- Remaining differences are non-numerical or below 10⁻⁹:
  - stored absolute paths;
  - the saved master table's 9-decimal rounding (≤ 5×10⁻¹⁰);
  - calibration values within 2.4×10⁻¹², computed originally from an earlier, 1.4×10⁻⁷-different copy of the predictions;
  - one PDF creation date (0 differing pixels).

**For this package:**
- The scripts were path-sanitised, and the execution copies can be regenerated for any folder.
- Build checks confirm that the 19 scripts differ from the originals only in the user-folder lines, with matching Python syntax trees after normalisation of the permitted path-string literals. Regenerating the execution copies for the original folder reproduces the 14 files used in the verified runs byte for byte (`evidence/package_build/verification.md`).

**To re-run:** see README §5. Each stage ends with `tools/compare_outputs.py`, which compares the new run with `verified_outputs/`.
