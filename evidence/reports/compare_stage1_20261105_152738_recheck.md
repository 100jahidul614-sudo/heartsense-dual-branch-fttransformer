# Stage 1: reproduced vs original saved outputs (re-check 2026-11-05 15:27 AEDT, after documenting the two Fig 15 drafts as reference-only; run in the Claude sandbox, read-only)

| status | files |
| --- | --- |
| IDENTICAL | 17 |
| MATCH_1e-4 | 2 |
| REFERENCE_ONLY | 2 |

| reproduced | status | detail |
| --- | --- | --- |
| data/nhanes/nhanes_2017_2020_merged_raw.csv | IDENTICAL | 15560 rows x 282 cols |
| data/nhanes/nhanes_cleaned_step1.csv | IDENTICAL | 15560 rows x 173 cols |
| data/nhanes/nhanes_final_with_target.csv | IDENTICAL | 9180 rows x 174 cols |
| runs/nhanes/nhanes_phase1_feature_fusion/feature_stability.csv | IDENTICAL | 163 rows x 2 cols |
| runs/nhanes/nhanes_phase1_feature_fusion/final_features_allfolds.json | IDENTICAL |  |
| runs/nhanes/nhanes_phase1_feature_fusion/fold_selected_features.json | IDENTICAL |  |
| runs/nhanes/nhanes_phase1_feature_fusion/phase1_summary.json | MATCH_1e-4 | only absolute paths differ |
| runs/nhanes/nhanes_phase1_feature_fusion/selector_overlap_table.csv | IDENTICAL | 5 rows x 14 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/ablation/nhanes_ablation_confmats.csv | IDENTICAL | 5 rows x 5 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/ablation/nhanes_ablation_oof_components.csv | IDENTICAL | 9180 rows x 6 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/ablation/nhanes_ablation_results.csv | IDENTICAL | 5 rows x 9 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures_confmat_050.csv | IDENTICAL | 2 rows x 3 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures_metrics.csv | IDENTICAL | 7 rows x 20 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv | IDENTICAL | 9180 rows x 7 cols |
| runs/nhanes/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures/nhanes_phase2_fttf_dual_precbiased_gate_selectedfeatures_summary.json | MATCH_1e-4 | only absolute paths differ |
| runs/nhanes/nhanes_baseline_comparison/nhanes_baseline_comparison_confmats.csv | IDENTICAL | 3 rows x 5 cols |
| runs/nhanes/nhanes_baseline_comparison/nhanes_baseline_comparison_results.csv | IDENTICAL | 3 rows x 10 cols |
| runs/nhanes/nhanes_baseline_comparison/nhanes_baseline_oof_probabilities.csv | IDENTICAL | 9180 rows x 4 cols |
| runs/nhanes/nhanes_figures/nhanes_confusion_matrix_grid_style.png | IDENTICAL | same bytes |
| runs/nhanes/nhanes_figures/nhanes_pooled_confusion_matrix_comparison.png | REFERENCE_ONLY | earlier 1x5 draft of Fig 15 (13 May 2026); the current script writes nhanes_confusion_matrix_grid_style.png, the 2x3 layout printed as Fig 15; the code that drew this draft no longer exists |
| runs/nhanes/nhanes_figures/nhanes_pooled_confusion_matrix_comparison_large.png | REFERENCE_ONLY | larger copy of the same earlier 1x5 draft of Fig 15; not written by the current script |
