# Stage 2 reproduction report — BRFSS-2024 core pipeline

**Run:** 2026-11-05, 15:31:25–18:36:26 AEDT (185 min), on the author's PC, with the original scripts, seeds and parameters unchanged.
**Result:** every Stage 2 output written by the original scripts was reproduced **bit for bit**:
- the Table 3 summary;
- the 181-feature Phase-1 selection and its five fold lists;
- the dual-branch FT-Transformer's 452,464 out-of-fold probabilities (recall branch, precision branch, gate weight, gate-only and final);
- fold metrics, pooled confusion matrix and the Table 10 ablation.

## 1. Run record

| Step | Script (unchanged except one path line in 2.2) | Duration | Exit |
| --- | --- | --- | --- |
| 2.1 Phase 1 feature fusion: Pearson, TabNet, HistGradientBoosting permutation importance | `original_code\brfss2024_phase1_feature_fusion_cache_builder.py` | 22.3 min | 0 |
| 2.2 Table 3 dataset summary | `run_code\generate_dataset_summary.py` | 0.5 min | 0 |
| 2.3 Dual-branch FT-Transformer + GateNet + veto/rescue, 5-fold CV + ablation | `original_code\brfss2024_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | 162.2 min | 0 |

- **Inputs:**
  - Raw `main_data_brfss_2024.XPT`, SHA-256 `7d53cf08...aab36b1c`, verified before the run.
  - 452,464 respondents with a valid `_MICHD` label (42,338 positive).
  - 283 numeric features after cleaning.
- **Environment:**
  - Windows 11 with Python 3.12.10.
  - PyTorch 2.9.1+cu128 (CUDA 12.8, cuDNN 9.10), with mixed precision as in the original.
  - NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB), driver 591.91.
  - Package versions identical to the original `.venv312`.
- **Logs:** `logs\stage2_20261105_153124\` (commands, console output, timings, environment) and `logs\ps_transcripts\`.

## 2. Phase-1 feature selection (181 features)

The reproduced `final_features_allfolds.json` holds the same 181 features as the original, in the same order. Each fold-specific selection is identical in content and order, and the five folds' union equals the 181-feature list.

| Fold | Pearson (abs r ≥ 0.08) | TabNet (≥ 0.75/N) | HGB permutation (μ − 1.7σ > 2e-4) | Selected by ≥ 2 | Fold union | Same as original (set and order) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 52 | 55 | 42 | 36 | 101 | yes |
| 2 | 52 | 33 | 71 | 37 | 106 | yes |
| 3 | 53 | 44 | 49 | 30 | 105 | yes |
| 4 | 52 | 42 | 49 | 30 | 100 | yes |
| 5 | 52 | 54 | 58 | 39 | 111 | yes |

`selector_overlap_table.csv` and `feature_stability.csv` (283 rows) are identical.

## 3. Proposed model

The out-of-fold file (452,464 rows × 7 columns: `y_true, p_rec, p_prec, g, p_gate_only, p_final, y_prob`) is **identical** to the original. Compared with the separate January 2026 original run, the largest probability difference is 1.4×10⁻⁷, and all 0.50 decisions are the same.

**Pooled confusion matrix at 0.50:** TN 359,639 · FP 50,487 · FN 16,975 · TP 25,363. That gives precision 0.334384, recall 0.599060, F1 0.429198, balanced accuracy 0.737979 and accuracy 0.850901, all identical to the original.

| Fold | Acc | BalAcc | Precision | Recall | F1 | AUROC | AUPRC | Best epoch (recall / precision branch) | Veto fraction | Rescue fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8579 | 0.7353 | 0.3462 | 0.5845 | 0.4349 | 0.8032 | 0.3717 | 30 / 23 | 0.741 | 0 |
| 2 | 0.8649 | 0.7218 | 0.3556 | 0.5458 | 0.4306 | 0.7972 | 0.3621 | 29 / 24 | 0.744 | 0 |
| 3 | 0.8512 | 0.7414 | 0.3363 | 0.6063 | 0.4326 | 0.8039 | 0.3624 | 29 / 15 | 0.728 | 0 |
| 4 | 0.8471 | 0.7358 | 0.3270 | 0.5987 | 0.4230 | 0.8010 | 0.3571 | 30 / 21 | 0.724 | 0 |
| 5 | 0.8334 | 0.7556 | 0.3142 | 0.6600 | 0.4257 | 0.8098 | 0.3762 | 27 / 30 | 0.716 | 0 |
| Mean ± SD | 0.8509 ± 0.0119 | 0.7380 ± 0.0122 | 0.3359 ± 0.0162 | 0.5991 ± 0.0413 | 0.4294 ± 0.0049 | 0.8030 ± 0.0046 | 0.3659 ± 0.0078 | | 0.731 | 0 |

The per-fold metrics file (all 20 columns, including gate and early-stopping diagnostics) is identical to the original.

## 4. Ablation (Table 10), pooled OOF, identical to the original

| Variant | F1 | Precision | Recall | BalAcc | AUPRC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recall branch only | 0.3781 | 0.2473 | 0.8035 | 0.7755 | 0.3887 | 103,572 | 8,318 |
| Precision branch only | 0.3750 | 0.4522 | 0.3202 | 0.6401 | 0.3888 | 16,422 | 28,780 |
| Simple average of the two branches | 0.4310 | 0.3346 | 0.6054 | 0.7406 | 0.3910 | 50,966 | 16,705 |
| GateNet only (no veto/rescue) | 0.4292 | 0.3344 | 0.5991 | 0.7380 | 0.3890 | 50,488 | 16,975 |
| GateNet + veto/rescue (final model) | 0.4292 | 0.3344 | 0.5991 | 0.7380 | 0.3632 | 50,487 | 16,975 |

## 5. Comparison with the published paper (reported as found)

The reproduced values equal the original saved outputs in every case below. Where the paper differs from those outputs, both are shown.

| Paper element | Published | Original saved = reproduced |
| --- | --- | --- |
| Table 3 (N, positives, negatives, prevalence 9.3572%, ratio 9.6869, 283 → 181 features, missingness 51.7566% / 41.7640%) | as printed | all 9 values match |
| Abstract F1 | 0.43 | 0.4294 (fold mean) |
| Abstract recall | 0.59 | 0.5991 (printed value is truncated, not rounded) |
| Abstract AUPRC | 0.38 | 0.3659 (fold mean), 0.3632 (pooled) |
| Table 10: recall-branch, precision-branch and simple-average rows | as printed | all 18 values match |
| Table 10 final model: recall / BalAcc / FP / FN | 0.599 / 0.738 / 50,487 / 16,975 | match |
| Table 10 final model: F1 / precision | 0.430 / 0.349 | 0.4292 / 0.3344 |

The Table 6 proposed-model row is assembled in Stage 3 (master table) and is compared there.

## 6. File-level comparison

`reports\compare_stage2_20261105_183629.md`: 11 IDENTICAL, 2 MATCH_1e-4, 3 REFERENCE_ONLY, 0 differing.

| File | Status | Explanation |
| --- | --- | --- |
| `phase1_feature_fusion\phase1_summary.json` | MATCH_1e-4 | Maximum numerical difference **0**. Only 4 stored file paths differ (`Heartseanse2.0\artifacts\...` → `HeartSense_repro\runs\brfss\...`) |
| `phase2_..._selectedfeatures\..._summary.json` | MATCH_1e-4 | Maximum numerical difference **0**. Only 5 stored paths differ (data/feature-list/output paths); every hyperparameter entry is identical |
| `phase1_feature_fusion\brfss2024_phase2_run_fttransformer_dualbranch_ugf.py` | REFERENCE_ONLY | A copy of an early exploratory script ("Uncertainty-Gated Fusion, Option A", 18 Jan 2026) saved in the original output folder by hand. It differs from the later root-level script of the same name and is not part of the published pipeline |
| `phase1_feature_fusion\final_features_allfolds.json.py` | REFERENCE_ONLY | A small helper (18 Jan 2026) that rebuilds the union list from the fold lists; it points to an older folder name (`HeartSense2.0`). The Phase-1 script itself already writes the union, and its output was reproduced identically |
| `phase1_feature_fusion\final_features_stable.json` | REFERENCE_ONLY | An earlier saved copy of the Phase-1 list (18 Jan 2026): the same 181 features in a different order. The Phase-1 script does not write this file |

## 7. Integrity

- **Original project unchanged:** the runner re-hashed all 277 files on your PC after the run (`summary.json`), and an independent re-check from the sandbox passed (`tools\verify_stage0.py`: PASS).
- **Preserved files:** every Stage 2 output, log and report is listed with its SHA-256 in `manifests\stage2_outputs.sha256`.
