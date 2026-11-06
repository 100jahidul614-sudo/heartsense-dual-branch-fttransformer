# Stage 1 reproduction report — NHANES 2017–March 2020 cohort

**Run:** 2026-11-05, 15:17:18–15:20:13 AEDT (2.9 min), on the author's PC, with the original scripts and seeds unchanged.
**Result:** every NHANES output written by the original scripts was reproduced **bit for bit**. That covers the cohort files, the Phase-1 feature lists, the proposed model's per-sample predictions, fold metrics, confusion matrix and ablation, and the three baselines.

## 1. Run record

| Step | Script | Duration (s) | Exit |
| --- | --- | --- | --- |
| 1.1 Merge 18 modules on SEQN | `run_code\nhanes_merge_check.py` | 2.4 | 0 |
| 1.2 Drop variables with >70% missing | `run_code\nhanes_cleaning_step1.py` | 1.3 | 0 |
| 1.3 Build CVD_TARGET | `run_code\nhanes_create_target.py` | 3.0 | 0 |
| 1.4 Phase 1 feature fusion (Pearson, TabNet, HGB permutation) | `original_code\nhanes_phase1_feature_fusion_cache_builder.py` | 62.7 | 0 |
| 1.5 Dual-branch FT-Transformer + GateNet + veto/rescue | `original_code\nhanes_phase2_run_fttransformer_dualbranch_precbiased_gate.py` | 81.0 | 0 |
| 1.6 Baselines (LightGBM, Random Forest, CNN) | `run_code\nhanes_baseline_comparison_models.py` | 21.8 | 0 |
| 1.7 Fig 15 | `run_code\nhanes_confusion_matrix_comparison_plot.py` | 2.4 | 0 |

- **Environment:**
  - Windows 11 with Python 3.12.10.
  - PyTorch 2.9.1+cu128 (CUDA 12.8, cuDNN 9.10).
  - NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB), driver 591.91.
  - Intel Core Ultra 9 275HX, 31.4 GB RAM.
  - Package versions identical to the original `.venv312`.
- **Settings added by the runner:** MPLBACKEND=Agg, PYTHONIOENCODING=utf-8, PYTHONUNBUFFERED=1 and PYTHONDONTWRITEBYTECODE=1. None of them affects any computation.
- **Full commands and console output:** `logs\stage1_20261105_151717\` (`steps.csv`, `step_*.log`, `environment.json`, `summary.json`), with the PowerShell transcript in `logs\ps_transcripts\`.

## 2. Cohort (Table 14)

| Quantity | Published | Original output | Reproduced |
| --- | --- | --- | --- |
| Merged participants × variables | 15,560 × 282 | 15,560 × 282 | 15,560 × 282 (file identical) |
| Variables after >70% missingness filter | 173 | 173 | 173 (file identical) |
| Final cohort | 9,180 | 9,180 | 9,180 (file identical) |
| Positive class | 9.53% | 875 / 9,180 = 9.5316% | 875 / 9,180 = 9.5316% |
| Candidate features before Phase 1 | 163 | 163 | 163 |
| Phase-1 selected features | 141 | 141 | 141 |

## 3. Phase-1 feature selection

The reproduced `final_features_allfolds.json` holds the same 141 feature names as the original, in the same order. Each of the five fold selections is also identical in content and order.

| Fold | Original | Reproduced | Same set and order |
| --- | --- | --- | --- |
| 1 | 103 | 103 | yes |
| 2 | 108 | 108 | yes |
| 3 | 109 | 109 | yes |
| 4 | 98 | 98 | yes |
| 5 | 107 | 107 | yes |

The union of the five reproduced fold selections equals the 141-feature list. `selector_overlap_table.csv` (Pearson/TabNet/HGB counts and overlaps per fold) and `feature_stability.csv` are identical.

## 4. Proposed model (dual-branch FT-Transformer + GateNet + veto/rescue)

The out-of-fold prediction file (9,180 rows × 7 columns: `y_true, p_rec, p_prec, g, p_gate_only, p_final, y_prob`) is **identical** to the original, so every probability matches exactly.

| Pooled at threshold 0.50 | Original | Reproduced |
| --- | --- | --- |
| TN / FP / FN / TP | 7,626 / 679 / 374 / 501 | 7,626 / 679 / 374 / 501 |
| F1 | 0.487591 | 0.487591 |
| Precision / Recall / Balanced accuracy | 0.424576 / 0.572571 / 0.745407 | identical |

Per-fold metrics (identical to the original file in every column):

| Fold | Acc | BalAcc | Precision | Recall | F1 | AUROC | AUPRC | Veto fraction | Rescue fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8824 | 0.7126 | 0.4055 | 0.5029 | 0.4490 | 0.7991 | 0.4316 | 0.831 | 0 |
| 2 | 0.8878 | 0.7463 | 0.4329 | 0.5714 | 0.4926 | 0.8056 | 0.4129 | 0.803 | 0 |
| 3 | 0.8943 | 0.7090 | 0.4492 | 0.4800 | 0.4641 | 0.7604 | 0.4043 | 0.847 | 0 |
| 4 | 0.8753 | 0.7956 | 0.4094 | 0.6971 | 0.5159 | 0.8575 | 0.5121 | 0.729 | 0 |
| 5 | 0.8867 | 0.7636 | 0.4332 | 0.6114 | 0.5071 | 0.8147 | 0.4295 | 0.806 | 0 |
| Mean ± SD | 0.8853 ± 0.0071 | 0.7454 ± 0.0362 | 0.4260 ± 0.0183 | 0.5726 ± 0.0873 | 0.4857 ± 0.0284 | 0.8074 ± 0.0348 | 0.4381 ± 0.0429 | 0.803 | 0 |

The published F1 0.4876 is the pooled value (0.487591); the fold mean is 0.4857.

## 5. Final-model vs gate-only AUROC/AUPRC (as found, no correction)

The original script writes both variants into `ablation\nhanes_ablation_results.csv`. They differ only in the veto/rescue rules applied after the gate. The reproduced file is identical to the original:

| Variant (pooled OOF) | F1 | Precision | Recall | FP | FN | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recall head only | 0.4285 | 0.2955 | 0.7794 | 1,626 | 193 | 0.8594 | 0.4141 |
| Precision head only | 0.4720 | 0.4767 | 0.4674 | 449 | 466 | 0.8733 | 0.4593 |
| Simple average of the two branches | 0.4791 | 0.3777 | 0.6549 | 944 | 302 | 0.8696 | 0.4601 |
| GateNet only (no veto/rescue) | 0.4876 | 0.4246 | 0.5726 | 679 | 374 | **0.8736** | **0.4677** |
| GateNet + veto/rescue (final model) | 0.4876 | 0.4246 | 0.5726 | 679 | 374 | **0.8062** | **0.4252** |

- The two variants make the same 0.50 decisions. Veto sets the final probability to 0 for 73–85% of samples per fold without changing a decision, and rescue never fires (rescue fraction 0 in every fold).
- Only the ranking metrics differ: AUROC/AUPRC are 0.8062/0.4252 for the final model and 0.8736/0.4677 for gate only.
- Table 15 of the paper prints 0.8736 / 0.4677 in the "Proposed" row; these are the gate-only values. The final model's values are the ones above.

## 6. Baselines (Table 15)

`nhanes_baseline_comparison_results.csv`, `_confmats.csv` and `nhanes_baseline_oof_probabilities.csv` are identical to the originals:

| Model | TN / FP / FN / TP | F1 | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| LightGBM | 7,909 / 396 / 496 / 379 | 0.4594 | 0.8739 | 0.4522 |
| Random Forest | 8,249 / 56 / 791 / 84 | 0.1655 | 0.8794 | 0.4285 |
| CNN | 5,319 / 2,986 / 263 / 612 | 0.2736 | 0.7405 | 0.2271 |

The results table checks 47 printed numbers from Tables 14–15. Of these, 44 equal the original outputs at the printed precision, and all 47 reproduced values equal the original outputs. The other three are listed below, exactly as found:

- Proposed AUROC and AUPRC: see section 5.
- Simple Avg balanced accuracy: printed as 0.7716; the saved and reproduced value is 0.7706.

## 7. Figure 15

- `nhanes_confusion_matrix_grid_style.png` (2×3 grid, "at Threshold = 0.5") is the layout printed as Fig 15. It was regenerated **byte-identical** to the original.
- The script draws counts typed into its source. They were checked against the reproduced confusion-matrix files and match for all five models.
- The comparison listed two further original images, `nhanes_pooled_confusion_matrix_comparison.png` and `nhanes_pooled_confusion_matrix_comparison_large.png`, as missing. They are an earlier one-row (1×5) draft of the same figure, saved on 13 May 2026, before the final grid version. They show the same counts.
  - The current original script no longer writes them, and the code that drew them does not exist in the project.
  - They are kept in `reference\` and marked REFERENCE_ONLY in the comparison. They were not regenerated, because that would mean writing new plotting code that the original pipeline never contained.

## 8. File-level comparison and integrity

| Status | Files | Detail |
| --- | --- | --- |
| IDENTICAL | 17 | All data, feature-selection, prediction, metric, ablation and baseline files, plus the Fig 15 image |
| MATCH_1e-4 | 2 | `phase1_summary.json` and `nhanes_phase2_..._summary.json`: maximum numerical difference **0**. Only the absolute paths stored in them differ (4 and 5 path entries: `Heartseanse2.0\...` became `HeartSense_repro\...`) |
| REFERENCE_ONLY | 2 | The Fig 15 drafts (section 7) |

- **Original project unchanged:** the runner re-hashed all 277 files after the run on your PC (`summary.json`: `"original_project": "unchanged"`), and an independent re-check from the sandbox at 15:27 also passed.
- **Reports:** the comparison runs that produced these results are `reports\compare_stage1_20261105_152015.md` (original run) and `reports\compare_stage1_20261105_152738_recheck.md` (with the drafts labelled), plus `reports\published_vs_reproduced_20261105_152017.csv`.
- **Preserved files:** every Stage 1 log, output and report is listed with its SHA-256 in `manifests\stage1_outputs.sha256`.
