# Package build verification

Checks run by the package builder on the private reproduction folder and on this package. All passed (the build stops on any failure).

| Check | Result |
| --- | --- |
| Files copied from the private folders | 186 (75 with path replacements; 487 replacements in total) |
| Sanitisation is exactly one substitution (user folder → `<user>`) and is reversible for every file | yes |
| 19 original scripts: only lines containing the user-folder path differ | yes |
| 19 original scripts: Python syntax tree matches the private original after normalisation of the permitted path-string literals (user folder) | yes |
| Module-level constants (hyperparameters, seeds, thresholds) of the 6 core scripts identical to the private originals | yes |
| Package tool regenerates the 14 execution copies of the verified runs byte for byte (original folder) | 14/14 |
| Regenerated path-change log equals the verified runs' log (27 lines) | yes |
| Shipped `run_code/` equals the package tool's output for the sanitised root | 14/14 |
| `tools/stages.json`: steps, scripts and arguments identical to the verified runs | yes |
| Portability `windows:D:\review\heartsense`: 14 copies, 29 path lines, syntax trees match after path-literal normalisation, all compile, no placeholder left | yes |
| Portability `posix:/home/reviewer/heartsense`: 14 copies, 29 path lines, syntax trees match after path-literal normalisation, all compile, no placeholder left | yes |

## Core hyperparameters (BRFSS Phase-2 script, unchanged)

| Constant | Value |
| --- | --- |
| `DEFAULT_TARGET` | `_MICHD` |
| `DEFAULT_LEAKAGE_COLS` | `['CVDINFR4', 'CVDCRHD4']` |
| `DEFAULT_PHASE1_STABILITY_CSV` | `` |
| `DEFAULT_N_SPLITS` | `5` |
| `DEFAULT_SEED` | `42` |
| `DEFAULT_MIN_FOLDS_KEEP` | `1` |
| `DEFAULT_MAX_EPOCHS` | `30` |
| `DEFAULT_PATIENCE` | `6` |
| `DEFAULT_VAL_FRACTION` | `0.1` |
| `DEFAULT_BATCH_SIZE` | `1024` |
| `DEFAULT_LR` | `0.0003` |
| `DEFAULT_WEIGHT_DECAY` | `1e-05` |
| `DEFAULT_D_EMBED` | `48` |
| `DEFAULT_N_HEADS` | `4` |
| `DEFAULT_N_LAYERS` | `2` |
| `DEFAULT_FFN_MULT` | `4` |
| `DEFAULT_DROPOUT` | `0.08` |
| `DEFAULT_VAL_BATCH_SIZE` | `512` |
| `DEFAULT_USE_AMP` | `True` |
| `DEFAULT_SCALER` | `standard` |
| `DEFAULT_PREC_NEG_WEIGHT` | `4.0` |
| `DEFAULT_ALPHA_POS` | `1.3` |
| `DEFAULT_USE_FOCAL` | `0` |
| `DEFAULT_FOCAL_GAMMA` | `2.0` |
| `DEFAULT_LABEL_SMOOTH` | `0.0` |
| `DEFAULT_PREC_TEMPERATURE` | `1.0` |
| `DEFAULT_GATE_EPOCHS` | `6` |
| `DEFAULT_GATE_LR` | `0.005` |
| `DEFAULT_GATE_WEIGHT_DECAY` | `0.0` |
| `DEFAULT_GATE_BATCH_SIZE` | `4096` |
| `DEFAULT_GATE_HIDDEN` | `16` |
| `DEFAULT_GATE_BIAS` | `0.9` |
| `DEFAULT_VETO_V` | `0.2` |
| `DEFAULT_RESCUE_RHI` | `0.98` |

Full details per script: `verification.json`.
