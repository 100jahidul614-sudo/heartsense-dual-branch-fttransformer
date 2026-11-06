# Stage 4: BRFSS-2024 analyses from saved predictions, no training (under 1 min); needs Stages 2-3 first
# Pre-flight only:  powershell -ExecutionPolicy Bypass -File .\04_stage4_brfss_analysis.ps1 -CheckOnly
# Run:              powershell -ExecutionPolicy Bypass -File .\04_stage4_brfss_analysis.ps1
param([switch]$CheckOnly)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$PY = ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Run 00_setup_env.ps1 first (.venv not found)" }
New-Item -ItemType Directory -Force -Path "logs\ps_transcripts" | Out-Null
Start-Transcript -Path ("logs\ps_transcripts\stage4" + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")

& $PY tools\run_stage.py --stage 4 --check_only
if ($LASTEXITCODE -ne 0) { Stop-Transcript; Write-Host "Pre-flight failed; nothing was run." -ForegroundColor Red; exit 1 }
if ($CheckOnly) { Stop-Transcript; exit 0 }

$ans = Read-Host "Start Stage 4 now? Type YES to continue"
if ($ans -ne "YES") { Stop-Transcript; Write-Host "Cancelled; nothing was run."; exit 0 }

& $PY tools\run_stage.py --stage 4
$rc = $LASTEXITCODE
& $PY tools\compare_outputs.py --stage 4
& $PY tools\published_vs_reproduced.py
Stop-Transcript
exit $rc
