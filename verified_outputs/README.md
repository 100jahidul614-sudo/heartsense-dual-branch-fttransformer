# verified_outputs/: outputs of the verified re-run (5–6 November 2026)

This folder is the `runs/` folder of the verified reproduction: `nhanes/` from Stage 1, and `brfss/` from Stages 2–4.

- **Relation to the original saved outputs:** these files are bit-identical to the outputs saved in the original project, except as recorded in `evidence/reports/` (stored absolute paths; 9-decimal rounding of the saved master table; calibration values within 2.4×10⁻¹²; one PDF creation date).
- **Sanitisation:** text files have the user folder replaced by `C:\Users\<user>\` (`PACKAGE_NOTES.md`). This affects only the stored path fields in summary JSONs and the path columns of the master table.
- **Not included:** the 13 row-level BRFSS prediction files (116 MB). Their SHA-256 values are in `manifests/excluded_files.sha256`.

New runs write to `runs/`, and `tools/compare_outputs.py --stage N` compares them with this folder.
