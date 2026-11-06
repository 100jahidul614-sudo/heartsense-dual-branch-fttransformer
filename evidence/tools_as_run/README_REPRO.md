# HeartSense_repro — reproduction workspace

Re-runs the original experiments of *Leakage-Safe Precision-Aware Dual-Branch FT-Transformer for
Population-Scale Heart Disease Risk Prediction* (Sensors 2026, 26, 3417) without changing their
scientific implementation. The original project `..\Heartseanse2.0` is only read, never written.

## Layout

| Folder / file | Contents |
| --- | --- |
| `original_code\` | Byte-identical copies of the 19 original scripts (SHA-256 checked) |
| `run_code\` | Execution copies of the 14 scripts whose file paths are written into the code; only those paths differ |
| `PATH_CHANGES.diff` | Every changed line (27 lines, file paths only) |
| `data\nhanes\` | Copies of the 18 NHANES 2017–Mar 2020 modules; the NHANES prep scripts write their CSVs here |
| `reference\` | Copies of the original saved outputs, used only for comparison |
| `runs\` | Every new output (created by the stage scripts) |
| `tools\` | Non-scientific helpers: copying, path checks, stage runner, environment check, comparisons |
| `env\requirements.txt` | Package versions of the original `.venv312` |
| `manifests\` | SHA-256 manifests, the path-change log, and a baseline of the original project |
| `logs\`, `reports\` | Run logs (commands, timings, exit codes, environment) and comparison reports |

The raw BRFSS file is read in place: `C:\Users\<user>\Downloads\Dataset\BRFSS2024\archive (1)\main_data_brfss_2024.XPT`
(SHA-256 `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c`).

## Order of execution (PowerShell, from this folder)

```
powershell -ExecutionPolicy Bypass -File .\00a_gpu_check.ps1          # hardware check, installs nothing
powershell -ExecutionPolicy Bypass -File .\00_setup_env.ps1           # creates .venv with the .venv312 versions
powershell -ExecutionPolicy Bypass -File .\01_stage1_nhanes.ps1       # only after approval
powershell -ExecutionPolicy Bypass -File .\02_stage2_brfss_core.ps1   # only after approval
powershell -ExecutionPolicy Bypass -File .\03_stage3_brfss_baselines.ps1
powershell -ExecutionPolicy Bypass -File .\04_stage4_brfss_analysis.ps1
```

Each stage script first runs a pre-flight check (inputs present, no output overwritten, all paths inside
this folder, raw-data hash). It then asks for `YES`, runs the steps with full logging, and compares the new
outputs with `reference\` (`reports\compare_stageN_*.md`) and with the published numbers
(`reports\published_vs_reproduced_*.csv`). Add `-CheckOnly` to run only the pre-flight.

## Rules

- Seeds, preprocessing, feature selection, architecture and hyperparameters are exactly those of the original
  scripts. Nothing is tuned and runs are not repeated to approach a published number.
- Results are reported as they come out, next to the original saved outputs and the published values.
