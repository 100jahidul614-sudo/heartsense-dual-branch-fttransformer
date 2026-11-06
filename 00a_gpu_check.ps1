# Optional hardware check (installs nothing): records GPU, NVIDIA driver and CUDA versions.
# Run from PowerShell in this folder:  powershell -ExecutionPolicy Bypass -File .\00a_gpu_check.ps1
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$Log = "logs\env"
New-Item -ItemType Directory -Force -Path $Log | Out-Null
$Out = "$Log\gpu_check_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt"
"=== Date: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $Out
"=== Windows video controllers" | Out-File -Append -Encoding utf8 $Out
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate | Format-List | Out-File -Append -Encoding utf8 $Out
"=== CPU / RAM" | Out-File -Append -Encoding utf8 $Out
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-File -Append -Encoding utf8 $Out
"RAM GB: " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1) | Out-File -Append -Encoding utf8 $Out
"=== nvidia-smi" | Out-File -Append -Encoding utf8 $Out
try { nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv | Out-File -Append -Encoding utf8 $Out } catch { "nvidia-smi not found on PATH" | Out-File -Append -Encoding utf8 $Out }
if (Test-Path ".venv\Scripts\python.exe") {
  "=== PyTorch in .venv" | Out-File -Append -Encoding utf8 $Out
  & ".venv\Scripts\python.exe" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| cudnn', torch.backends.cudnn.version()); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" 2>&1 | Out-File -Append -Encoding utf8 $Out
}
Get-Content $Out
Write-Host "`nSaved: $Out" -ForegroundColor Green
