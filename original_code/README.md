# original_code/: original research scripts (distribution copies)

These are the 19 scripts behind the paper's main results, copied from the author's research project. They are **distribution copies**: byte-identical to the originals except that the Windows user folder in hard-coded default paths reads `C:\Users\<user>\` (`PACKAGE_NOTES.md`). Folder structure and file names are as in the original project. The misspellings in `simple_Avarage_fusion.py` and `confusion_metrix_allmodel.py` are the original names.

- **Running them:** five scripts take all their paths as command-line arguments. They are run from here by `tools/run_stage.py`, with the arguments in `tools/stages.json`:
  - both `*_phase1_feature_fusion_cache_builder.py`;
  - both `*_phase2_run_fttransformer_dualbranch_precbiased_gate.py`;
  - `update_master_table_add_experiment.py`.
- **The other 14:** they are run from their path-only execution copies in `run_code/`.
- **Checksums:** `manifests/original_code_distribution.sha256` (package copies) and `manifests/sanitisation_map.csv` (the private originals).

`docs/METHOD.md` describes what each script does, with line references.
