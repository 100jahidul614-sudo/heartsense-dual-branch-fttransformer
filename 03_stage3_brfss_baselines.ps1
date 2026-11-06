# Stage 3: BRFSS-2024: 11 baselines, master table, Fig 4 (about 72 min); needs Stage 2 first
# Pre-flight only:  powershell -ExecutionPolicy Bypass -File .\03_stage3_brfss_baselines.ps1 -CheckOnly
# Run:              powershell -ExecutionPolicy Bypass -File .\03_stage3_brfss_baselines.ps1
param([switch]$CheckOnly)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$PY = ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Run 00_setup_env.ps1 first (.venv not found)" }
New-Item -ItemType Directory -Force -Path "logs\ps_transcripts" | Out-Null
Start-Transcript -Path ("logs\ps_transcripts\stage3" + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")

& $PY tools\run_stage.py --stage 3 --check_only
if ($LASTEXITCODE -ne 0) { Stop-Transcript; Write-Host "Pre-flight failed; nothing was run." -ForegroundColor Red; exit 1 }
if ($CheckOnly) { Stop-Transcript; exit 0 }

$ans = Read-Host "Start Stage 3 now? Type YES to continue"
if ($ans -ne "YES") { Stop-Transcript; Write-Host "Cancelled; nothing was run."; exit 0 }

& $PY tools\run_stage.py --stage 3
$rc = $LASTEXITCODE
& $PY tools\compare_outputs.py --stage 3
& $PY tools\published_vs_reproduced.py
Stop-Transcript
exit $rc
