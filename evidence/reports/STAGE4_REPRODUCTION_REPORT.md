# Stage 4 reproduction report — BRFSS-2024 analyses from saved predictions

**Run:** 2026-11-06, 03:10:27–03:10:49 AEDT (0.36 min, no training), on the author's PC. The six original analysis scripts ran unchanged apart from their file paths.
**Result:**
- Every Stage 4 data file and PNG figure was reproduced byte-identical, except one calibration CSV, which agrees to within 2.4×10⁻¹².
- The one PDF differs only in its embedded creation date.

## 1. Run record

| Step | Script (paths only changed) | Paper element | Duration | Exit |
| --- | --- | --- | --- | --- |
| 4.1 | `simple_Avarage_fusion.py` | 3-model simple average (Tables 8–9), printed only | 1.4 s | 0 |
| 4.2 | `confusion_metrix_allmodel.py` | Table 9 / Fig 7 pooled confusion matrices | 2.5 s | 0 |
| 4.3 | `FP\FN_bar_chart.py` (run from `runs\brfss\FP\`) | Fig 5 FP/FN bars | 0.7 s | 0 |
| 4.4 | `plot_threshold_curve.py` | Figs 8–9 threshold curves | 9.7 s | 0 |
| 4.5 | `heatmap.py` (run from `runs\brfss\threshold_heatmap\`) | Fig 10 veto/rescue sensitivity | 4.3 s | 0 |
| 4.6 | `plot_calibration_brier_ece.py` | Table 12 / Fig 11 calibration | 2.2 s | 0 |

- **Inputs:** only outputs reproduced in Stages 2–3: the proposed model's out-of-fold file and ablation components, and the LightGBM, RF and CNN1D out-of-fold and confusion files.
- **Logs:** `logs\stage4_20261106_031026\` and `logs\ps_transcripts\stage4_20261106_031021.txt`. The 03:10:13 transcript is the `-CheckOnly` pre-flight.

## 2. Comparison with the original saved outputs

`reports\compare_stage4_20261106_031051.md`: **9 IDENTICAL, 1 MATCH_1e-4, 1 IMAGE_DIFFERS (metadata only), 0 differing.**

| Output | Status |
| --- | --- |
| `confusion_matrix_counts_all_models.csv` (5 models × TN/FP/FN/TP) | IDENTICAL |
| `confusion_matrix_all_models.png` (Fig 7) | IDENTICAL |
| `confusion_matrix_all_models.pdf` | Creation date only (section 4) |
| `fp_fn_plot_data.csv`, `fp_fn_bar_chart.png` (Fig 5) | IDENTICAL |
| `threshold_prf_curve.png`, `threshold_error_curve.png` (Figs 8–9) | IDENTICAL |
| `threshold_heatmap_f1.png` (Fig 10) | IDENTICAL |
| `calibration_plot_bins15.png`, `probability_histograms.png` (Fig 11) | IDENTICAL |
| `calibration_metrics_brier_ece.csv` (Table 12) | MATCH_1e-4 (section 3) |

**Step 4.1 (printed output):** FP 37,835, FN 20,464, precision 0.366343, recall 0.516652, F1 0.428704. These equal the "Simple Avg" row of the reproduced and original confusion-count files.

## 3. The MATCH_1e-4 file: `calibration_metrics_brier_ece.csv`

| Method | Metric | Original saved | Reproduced | Difference |
| --- | --- | --- | --- | --- |
| Raw | Brier | 0.0944190516394689 | 0.0944190516418446 | +2.376×10⁻¹² |
| Raw | ECE | 0.1027323917460173 | 0.1027323917456637 | −3.536×10⁻¹³ |
| Platt | Brier | 0.0695648647505906 | 0.0695648647527226 | +2.132×10⁻¹² |
| Platt | ECE | 0.0040593175005475 | 0.0040593174996043 | −9.432×10⁻¹³ |
| Isotonic | Brier | 0.0695041962450677 | 0.0695041962450677 | 0 |
| Isotonic | ECE | 0.0005859150953501 | 0.0005859150953501 | 0 |

**Cause:** the original file was computed on 22 Feb 2026 from the January copy of the proposed model's out-of-fold file. The reproduction reads the out-of-fold file rebuilt in Stage 2, which is bit-identical to the April original. The January and April probabilities differ by at most 1.4×10⁻⁷, with no decision changes at any of the 99 thresholds the threshold script uses. That difference propagates to about 10⁻¹² in the Brier score and ECE.

- Isotonic calibration depends only on the order of the probabilities, which is unchanged, so it is identical.
- The calibration figures are byte-identical.
- The threshold curves read the same January file originally, and they are byte-identical too.

## 4. PDF check: `confusion_matrix_all_models.pdf`

The PDF differs only in its creation date, not in anything visible:

- **Bytes:** both files are 28,117 bytes and differ in 9 bytes, all inside `/CreationDate`: `D:20260410185334+11'00'` (original) vs `D:20261106031032+11'00'` (reproduced). Both were written by Matplotlib pdf backend v3.10.8.
- **Rendering:** each single page was rendered at 150 dpi (2177 × 1329 px). Pixel difference: **0 differing pixels**.
- **Text:** the extracted text, including every printed count, is identical.

## 5. Plots with hard-coded values

| Script | Hard-coded content | Check |
| --- | --- | --- |
| `FP\FN_bar_chart.py` (Fig 5) | Simple Avg FP 37,835 and FN 20,464, typed in with the comment "added manually from manuscript Table 8". The other three bars are read from confusion files | Equal to the values computed from the out-of-fold files in step 4.1 and in `confusion_matrix_counts_all_models.csv` |
| `nhanes_confusion_matrix_comparison_plot.py` (Fig 15, Stage 1) | TN/FP/FN/TP for all five models, typed in | Equal to the reproduced NHANES confusion files for all five models (Stage 1 report) |

No other figure script contains result values. Only analysis settings are written in:

- `heatmap.py`: veto grid 0.10–0.30 in steps of 0.02, rescue grid 0.90–0.99 in steps of 0.01.
- `plot_threshold_curve.py`: thresholds 0.01–0.99.
- `plot_calibration_brier_ece.py`: 15 bins, 5-fold cross-fitting, seed 42.

Every other figure script computes its values from the files it reads, including the master table behind Fig 4.

## 6. Comparison with the published paper (reported as found)

| Paper element | Published values checked | Saved = published | Reproduced = saved |
| --- | --- | --- | --- |
| Tables 8–9 (5 models: FP, FN, precision, recall, F1) | 25 | 25 (LightGBM F1 0.212 only when truncated; saved 0.2126) | 25 |
| Table 12 (Brier, ECE × 3 methods) | 6 | 6 (Platt Brier, Platt ECE and Isotonic ECE only when truncated) | 6 (within 2.4×10⁻¹²) |

None of the Stage 4 values differ from the paper beyond truncation. The proposed-model differences recorded in Stages 0–3 (Table 6, Table 10, Table 15 and the abstract AUPRC) come from earlier stages and are unchanged.

## 7. Integrity

- **Original project unchanged:** the runner re-hashed all 277 files after the run (`summary.json`: `"original_project": "unchanged"`). `tools\verify_stage0.py` was then re-run independently: PASS. It checked all 277 files, the 19 original-code copies, the 14 run copies with their 27 documented path lines, the 18 NHANES copies and the raw BRFSS hash.
- **Preserved files:** every Stage 4 output, log and report, plus the tools used, is listed with its SHA-256 in `manifests\stage4_outputs.sha256`.
