# evidence/: records of the verified reproduction and of this package's build

| Folder | Contents |
| --- | --- |
| `reports/` | `FINAL_REPRODUCIBILITY_REPORT.md`; `STAGE0_READINESS_REPORT.md` and `STAGE1`–`STAGE4` reproduction reports; file-by-file comparisons with the original saved outputs (`compare_stage*.md/.csv`); published vs original-saved vs reproduced tables (`published_vs_reproduced_*.csv`; the 20261106_031053 file is the final one) |
| `logs/` | `stage0/` (pre-flight and tool checks before training); `stage0_env/` (environment reports, pip freeze, GPU records); `stage1_*`–`stage4_*` (per-step command, working folder, timings, exit code, full console output, environment snapshot, summary) |
| `manifests_verified_runs/` | SHA-256 manifests written during the reproduction: copies made from the original project (`stage0_copies.*`), the path-change log, every stage's outputs, and the 277-file baseline of the original project |
| `tools_as_run/` | The tools and PowerShell scripts exactly as used for the verified runs (sanitised) |
| `package_build/` | Checks run when this package was built (`verification.json`, `verification.md`, `smoke_test.json`) |

**Path mapping:** the reports and logs use paths of the reproduction folder. In this package:
- `reports\` is `evidence/reports/`;
- `logs\` is `evidence/logs/`;
- `manifests\` is `evidence/manifests_verified_runs/`;
- `runs\` is `verified_outputs/`;
- `reference\` (the copies of the original saved outputs) is not shipped.

The `logs/stage0/` files come from tool tests run before training in an AI assistant's sandbox (paths as mounted there). The PowerShell transcripts of the training runs are not included, because they record the Windows account and computer name.
