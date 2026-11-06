# Stage 1: NHANES: data build, Phase 1, proposed model, baselines, Fig 15 (about 3 min on the verified GPU)
# Pre-flight only:  powershell -ExecutionPolicy Bypass -File .\01_stage1_nhanes.ps1 -CheckOnly
# Run:              powershell -ExecutionPolicy Bypass -File .\01_stage1_nhanes.ps1
param([switch]$CheckOnly)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$PY = ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Run 00_setup_env.ps1 first (.venv not found)" }
New-Item -ItemType Directory -Force -Path "logs\ps_transcripts" | Out-Null
Start-Transcript -Path ("logs\ps_transcripts\stage1" + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")

& $PY tools\run_stage.py --stage 1 --check_only
if ($LASTEXITCODE -ne 0) { Stop-Transcript; Write-Host "Pre-flight failed; nothing was run." -ForegroundColor Red; exit 1 }
if ($CheckOnly) { Stop-Transcript; exit 0 }

$ans = Read-Host "Start Stage 1 now? Type YES to continue"
if ($ans -ne "YES") { Stop-Transcript; Write-Host "Cancelled; nothing was run."; exit 0 }

& $PY tools\run_stage.py --stage 1
$rc = $LASTEXITCODE
& $PY tools\compare_outputs.py --stage 1
& $PY tools\published_vs_reproduced.py
Stop-Transcript
exit $rc
