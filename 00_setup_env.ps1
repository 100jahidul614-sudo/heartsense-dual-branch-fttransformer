# Step 0: create the virtual environment with the package versions used for the verified runs, then record
# the hardware/software environment. Installs into .venv inside this folder only.
# Run from PowerShell in this folder:
#   powershell -ExecutionPolicy Bypass -File .\00_setup_env.ps1                      # uses "py -3.12"
#   powershell -ExecutionPolicy Bypass -File .\00_setup_env.ps1 -Python C:\path\to\python.exe
param([string]$Python = "")
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Log = "logs\env"
New-Item -ItemType Directory -Force -Path $Log | Out-Null
Start-Transcript -Path ("$Log\setup_transcript_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Check($what) { if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: $what (exit $LASTEXITCODE)" } }

if ($Python -ne "") { $Base = $Python; $BaseArgs = @() }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $Base = "py"; $BaseArgs = @("-3.12") }
else { $Base = "python"; $BaseArgs = @() }
& $Base @BaseArgs --version; Check "base python (the verified runs used Python 3.12.10)"

if (Test-Path ".venv") { Stop-Transcript; throw ".venv already exists. Remove it yourself if you want to rebuild it." }
$t0 = Get-Date
& $Base @BaseArgs -m venv .venv; Check "venv creation"
$PY = ".venv\Scripts\python.exe"
& $PY -m pip install --disable-pip-version-check "pip==25.3"; Check "pip pin"
& $PY -m pip install --disable-pip-version-check -r env\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128; Check "pip install"
"Install time (min): " + [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

& $PY -m pip freeze | Out-File -Encoding utf8 "$Log\pip_freeze.txt"
try { nvidia-smi | Out-File -Encoding utf8 "$Log\nvidia_smi.txt" } catch { "nvidia-smi not found on PATH" | Out-File -Encoding utf8 "$Log\nvidia_smi.txt" }
& $PY tools\check_env.py
$envRc = $LASTEXITCODE
& $PY tools\make_run_copies.py --overwrite; Check "run_code generation"
& $PY tools\verify_package.py | Tee-Object -FilePath "$Log\verify_package.json"
$verRc = $LASTEXITCODE
Stop-Transcript
if ($envRc -ne 0 -or $verRc -ne 0) { Write-Host "Environment or package checks reported problems. See $Log" -ForegroundColor Yellow; exit 1 }
Write-Host "Environment ready. run_code\ has been generated for this folder." -ForegroundColor Green
