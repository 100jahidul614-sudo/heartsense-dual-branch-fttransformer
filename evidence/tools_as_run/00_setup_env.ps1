# Stage 0 (environment): create the reproduction virtual environment and record the
# hardware/software environment. Installs into HeartSense_repro\.venv only; the original
# project and its .venv312 are only read.
# Run from PowerShell:  powershell -ExecutionPolicy Bypass -File .\00_setup_env.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Log = "logs\stage0_env"
New-Item -ItemType Directory -Force -Path $Log | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "$Log\setup_transcript_$Stamp.txt"

function Check($what) { if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: $what (exit $LASTEXITCODE)" } }

# Same base interpreter that created the original .venv312 (see Heartseanse2.0\.venv312\pyvenv.cfg)
$Base = "C:\Users\<user>\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $Base)) { Stop-Transcript; throw "Python 3.12 not found at $Base" }
& $Base --version; Check "base python"

if (Test-Path ".venv") { Stop-Transcript; throw ".venv already exists. Remove it yourself if you want to rebuild it." }
$t0 = Get-Date
& $Base -m venv .venv; Check "venv creation"
$PY = ".venv\Scripts\python.exe"
& $PY -m pip install --disable-pip-version-check "pip==25.3"; Check "pip pin"
& $PY -m pip install --disable-pip-version-check -r env\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128; Check "pip install"
"Install time (min): " + [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

& $PY -m pip freeze | Out-File -Encoding utf8 "$Log\pip_freeze.txt"
try { nvidia-smi | Out-File -Encoding utf8 "$Log\nvidia_smi.txt" } catch { "nvidia-smi not found on PATH" | Out-File -Encoding utf8 "$Log\nvidia_smi.txt" }
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | Format-List | Out-File -Encoding utf8 "$Log\windows_video_controllers.txt"
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-File -Encoding utf8 "$Log\windows_cpu.txt"

& $PY tools\check_env.py
$envRc = $LASTEXITCODE
& $PY tools\verify_stage0.py | Tee-Object -FilePath "$Log\verify_stage0.json"
$verRc = $LASTEXITCODE
Stop-Transcript
if ($envRc -ne 0 -or $verRc -ne 0) { Write-Host "Stage 0 checks reported problems. See $Log" -ForegroundColor Yellow; exit 1 }
Write-Host "Stage 0 environment ready. Send logs\stage0_env\env_report.txt for review." -ForegroundColor Green
