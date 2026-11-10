# Dual-Branch FT-Transformer — Implemented Methodology

This document specifies the computational methodology of the Dual-Branch FT-Transformer as implemented in the original research scripts: cohort construction and outcome definitions, Phase-1 feature selection, the two-branch model and its training objectives, GateNet fusion with the veto/rescue rule, the baseline models and the post-hoc analyses. Each step references the corresponding function or line in `original_code/`; the distribution copies keep the original line numbering. The description follows the code as it runs, and the published article remains the reference for the reported findings.

## 1. Data preparation

### BRFSS-2024
The BRFSS file is used in `brfss2024_phase1_feature_fusion_cache_builder.py`, `brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py` and `brfss2024_phase2_all_models_allfeatures_master.py`.

- **Outcome definition:** `_MICHD` is the BRFSS calculated variable for respondents who have ever reported coronary heart disease (CHD) or myocardial infarction (MI). It records self-reported CHD/MI history at the time of the survey, so the task is classification of reported history, not prediction of prospectively observed future events.
- **Target:** `_MICHD` is decoded to 0/1. Rows without a valid label are dropped, leaving 452,464 of 457,670 (42,338 positive, 9.36%).
- **Leakage columns removed:** `CVDINFR4` (heart attack) and `CVDCRHD4` (coronary heart disease), which define `_MICHD`.
- **Cleaning:**
  - Only numeric columns are kept.
  - BRFSS special codes (don't know / refused / not asked) become missing values (`brfss_missing_to_nan`, Phase-2 script line 117).
  - All-missing and constant columns are dropped (`drop_all_nan_and_constant`, line 124).
  - 283 numeric candidate features remain.

### NHANES 2017–March 2020
- `nhanes_merge_check.py` merges 18 modules on `SEQN`: 15,560 participants × 282 variables.
- `nhanes_cleaning_step1.py` drops variables with more than 70% missing, leaving 173.
- `nhanes_create_target.py` builds `CVD_TARGET` from the self-reported MCQ cardiovascular items (`MCQ160B/C/D/E`: heart failure, CHD, angina, heart attack), giving 9,180 participants with 875 positive (9.53%).
- `MCQ160B/C/D/E` are removed as label-defining columns, leaving 163 candidate features.

## 2. Phase 1: multi-selector feature fusion per training fold, then across folds

Script: `brfss2024_phase1_feature_fusion_cache_builder.py` (NHANES: `nhanes_phase1_feature_fusion_cache_builder.py`).

- **Folds:** `StratifiedKFold(5, shuffle=True, random_state=42)` (line 369). For each fold, the selectors see only that fold's training part.

| Selector | Function | Rule (defaults) |
| --- | --- | --- |
| Pearson | `compute_abs_pearson_threshold` (line 114) | abs(r) ≥ 0.08 |
| TabNet | `train_tabnet_and_get_thresholded` (line 131) | mask importance ≥ 0.75 × (1 / N features); 30 epochs, batch 8,192 |
| HistGradientBoosting + permutation importance | `hgb_permutation_importance_lcb_positive` (line 193) | mean − 1.7 × std > 2×10⁻⁴; 5 repeats on a 30,000-row stratified subsample; seed 42 |

- **Fusion** (`fuse_sets_at_least_two`, line 264): the code computes features chosen by at least two selectors (*common*) and by exactly one (*unique*). It keeps common ∪ unique, which is the union of all three selections.
- **Across folds:** the five fold selections are joined by union, without stability filtering, into `final_features_allfolds.json`.
  - BRFSS: 181 features; per fold 101 / 106 / 105 / 100 / 111.
  - NHANES: 141 features; per fold 103 / 108 / 109 / 98 / 107.
- **How Phase 2 uses the list:** Phase 2 receives this union list (`--phase1_features_json`; `apply_phase1_feature_filter`, line 185) and applies it in every one of its folds.
- **Scope of the fold isolation:** Selection happens inside each training fold, but the five fold selections are then combined into one shared union. Every respondent is in the training part of four of the five Phase-1 folds, so the shared list used in each Phase-2 outer fold was chosen partly with that fold's validation data. The complete procedure therefore does not maintain strict outer-fold-independent feature selection.
- **Other outputs:** `fold_selected_features.json`, `selector_overlap_table.csv`, `feature_stability.csv` and `phase1_summary.json`.

NHANES uses the same rules. Its batch sizes are smaller (TabNet 2,048 / 512) and the permutation-importance subsample is 9,000 rows.

## 3. Phase 2: Dual-Branch FT-Transformer

Script: `brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py` (NHANES: `nhanes_phase2_…`). The defaults are at lines 60–105. Seeds are set with `seed_all` (line 151): NumPy, PyTorch and CUDA seeds 42, `cudnn.deterministic = True`, `cudnn.benchmark = False`.

### 3.1 Cross-validation and preprocessing (lines 739–830)
1. **Outer folds:** `StratifiedKFold(5, shuffle=True, random_state=42)` (line 739), with the fold loop at line 752.
2. **Preprocessing per fold:** median imputation (`SimpleImputer`, line 759) and standard scaling are fitted on the training fold and applied to the validation fold.
3. **Inner split:** `StratifiedShuffleSplit(test_size=0.10, random_state=42+fold)` splits the training fold into inner-train and inner-validation. The inner validation set is used for early stopping and for training GateNet.

### 3.2 Backbone (lines 203–285)
- **Tokenizer** (`FTTransformer.tokenize`, line 241): each standardised feature value x_j becomes a token x_j·W_j + b_j, with a learned per-feature W_j, b_j ∈ ℝ⁴⁸.
- **Encoder:** 2 × `TransformerBlock` (line 203).
  - Each block is pre-LayerNorm → 4-head self-attention → residual, then LayerNorm → feed-forward (48 → 192 → 48, GELU) → residual.
  - Dropout is 0.08 throughout.
- **Readout:** final LayerNorm, mean pooling over tokens, then `ProbHead` (line 257): 48 → 256 → 64 → 1 with ReLU and dropout.

### 3.3 Two branches with different error costs (`train_one_branch`, line 485)
| | Recall branch | Precision branch |
| --- | --- | --- |
| Loss | `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` | `NegWeightedBCEWithLogits` (line 300): the same BCE with pos_weight = max(1, (n_neg / n_pos) / α_pos), α_pos = 1.3, and every negative sample's loss × 4.0 |
| Effect | Positives weighted about 9.7×, favouring recall | Positives weighted less and negatives more, so false positives cost more |

- **Shared training settings:** AdamW (lr 3×10⁻⁴, weight decay 10⁻⁵), batch 1,024 (NHANES 512), at most 30 epochs.
- **Early stopping:** on inner-validation AUPRC with patience 6; the best epoch's weights are restored. Mixed precision (`torch.cuda.amp`) is used.
- **Options at their defaults:** focal loss is off (`use_focal = 0`), label smoothing is 0, and the precision-branch temperature is 1.0.

### 3.4 GateNet (lines 399–480, 582–607)
- **Inputs** (`build_gate_features`, line 411): (p_prec, p_rec, |p_prec − p_rec|) on the inner-validation samples.
- **Target:** 1 when the precision branch's 0.50 decision is correct and the recall branch's is not, otherwise 0 (lines ≈ 805–812).
- **Model and training** (`train_gatenet`, line 415): a 3 → 16 → 1 MLP with ReLU, BCE loss, AdamW lr 5×10⁻³, 6 epochs, batch 4,096.
  - It is trained on 75% of the inner-validation samples (`StratifiedShuffleSplit`, seed 42+1000+fold).
  - The epoch with the highest mean (1 − |g − target|) on the other 25% is kept.
  - **NHANES single-class fallback** (`nhanes_phase2_…`, lines 1106–1126): if `gate_y` contains only one class, the NHANES script prints a warning and splits the gate data 75/25 without stratification, from an index shuffled with `np.random.RandomState(42+1000+fold)`; otherwise it uses the stratified split above. The BRFSS script has no such branch. The fallback did not trigger in the verified NHANES run (no single-class warning in its step log).
- **Inference** (`gatenet_predict_g_numpy`, line 582): g = sigmoid(logit + 0.9) and p_gate = g·p_prec + (1 − g)·p_rec. The bias 0.9 shifts the weight towards the precision branch.

### 3.5 Veto/rescue rule (`apply_veto_rescue_numpy`, line 609)
- **Veto:** if p_prec < 0.20, the final probability is set to 0.
- **Rescue:** if p_rec > 0.98, it is set to 1.
- **Decision:** threshold 0.50 on the final probability.
- **In the verified runs:** veto applied to 72–74% of BRFSS samples per fold and 73–85% of NHANES samples. Rescue never fired. On NHANES the final and gate-only 0.50 decisions are identical (FP 679, FN 374 in both). On BRFSS they differ by one false positive: 50,488 for gate-only versus 50,487 for the final output, with FN 16,975 in both. Ranking metrics (AUROC/AUPRC) differ between the gate-only and final outputs, because vetoed samples get probability 0.

### 3.6 NHANES: cohort-specific selection and training
The NHANES analysis applies the same method to the NHANES cohort itself; it does not reuse the BRFSS model. Phase 1 is run on NHANES (163 → 141 features, `nhanes_phase1_feature_fusion_cache_builder.py`). Both branches and GateNet are then trained and evaluated from scratch with 5-fold cross-validation on NHANES (`nhanes_phase2_…`, fold loop line 1009). No BRFSS-trained weights are loaded; the only weight loading in the NHANES script restores each model's own best early-stopping state (lines 654, 790).

### 3.7 Outputs and ablation (lines 862–1070)
- **Per fold:** accuracy, balanced accuracy, precision, recall, F1 at 0.50, ROC-AUC, AUPRC (`average_precision_score`), best epochs, mean gate value, and veto/rescue fractions (`*_metrics.csv`, with a `mean` row).
- **Pooled:** out-of-fold probabilities (`*_oof.csv`: `y_true, p_rec, p_prec, g, p_gate_only, p_final, y_prob`) and the confusion matrix at 0.50 (`*_confmat_050.csv`).
- **Ablation** (`ablation/`): pooled metrics and confusion matrices for five variants: recall branch only, precision branch only, simple average of the branches, GateNet without veto/rescue, and the final model.

## 4. Baselines

Script: `brfss2024_phase2_all_models_allfeatures_master.py`. It runs with `RUN_WITH_DEFAULTS = True`, on all 283 numeric features, with the same folds (5-fold, seed 42) and preprocessing fitted on the training folds.

| Model | Setting in the code |
| --- | --- |
| LightGBM | SMOTE on the training fold only, no class weight (`fit_predict_lgbm_smote`, line 217) |
| Logistic regression | saga solver, `class_weight="balanced"`, standard scaling |
| Linear SVM | `LinearSVC(class_weight="balanced")` with `CalibratedClassifierCV` for probabilities |
| Decision tree, random forest | class-balanced (`balanced`, `balanced_subsample`) |
| KNN, Gaussian NB | standard scaling, no class weights |
| AE + LR, 1-D CNN, LSTM, GRU | PyTorch; at most 10 epochs, batch 2,048, lr 2×10⁻³, patience 3, mixed precision |
| XGBoost, CatBoost | in the code, but skipped because the packages were not installed in the original environment |

- **Per model:** the script writes per-fold metrics, out-of-fold probabilities and a pooled confusion matrix at 0.50, plus one row in `master_results_phase2_allfeatures.csv`.
- **Proposed-model row:** `update_master_table_add_experiment.py` adds the proposed model's row from its metrics and confusion files.
- **NHANES baselines** (`nhanes_baseline_comparison_models.py`): LightGBM, Random Forest and a 1-D CNN on the Phase-1 features, with the same folds.

## 5. Analyses (Stage 4)

| Script | What it computes |
| --- | --- |
| `simple_Avarage_fusion.py` | Mean of LightGBM, RF and CNN1D probabilities at 0.50; prints FP, FN, precision, recall and F1 |
| `confusion_metrix_allmodel.py` | Pooled confusion matrices for the proposed model, LightGBM, RF, CNN and the 3-model average (Table 9, Fig 7) |
| `run_code/FN_bar_chart.py` (execution copy of `original_code/FP/FN_bar_chart.py`) | FP/FN bars (Fig 5). The Simple-Avg FP/FN (37,835 / 20,464) are typed in from the manuscript's Table 8; they equal the computed values |
| `plot_threshold_curve.py` | Precision, recall, F1 and FP/FN of the final probabilities for thresholds 0.01–0.99 (Figs 8–9) |
| `heatmap.py` | Pooled F1 when the veto/rescue rule is re-applied to the gate-only output over v = 0.10–0.30 and r = 0.90–0.99 (Fig 10) |
| `plot_calibration_brier_ece.py` | Brier score and 15-bin ECE for raw, Platt and isotonic calibration, cross-fitted with 5 folds and seed 42 (Table 12, Fig 11) |

`nhanes_confusion_matrix_comparison_plot.py` (Fig 15) also contains typed-in confusion counts; they equal the reproduced NHANES confusion files.
