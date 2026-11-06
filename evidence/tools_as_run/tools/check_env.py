"""
Environment check for the HeartSense reproduction (read-only; installs nothing).

Records and checks:
  - Python version and interpreter path;
  - every installed package against env/requirements.txt AND against the original .venv312 (read from
    its site-packages folder, not modified);
  - XGBoost / CatBoost absent (as in the original environment);
  - PyTorch build, CUDA availability, CUDA runtime of the build, cuDNN, GPU name/capability/memory;
  - NVIDIA driver version and the driver's supported CUDA version (nvidia-smi);
  - OS, CPU and RAM.
Writes logs/stage0_env/env_report.json and env_report.txt. Exit code 1 if a version differs or CUDA is unavailable.

Usage (from HeartSense_repro):  .venv\\Scripts\\python.exe tools\\check_env.py
"""
import json
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPRO = TOOLS.parent
CFG = json.loads((TOOLS / "stages.json").read_text(encoding="utf-8"))
ORIG_VENV_SP = Path(CFG["original_project"]) / ".venv312" / "Lib" / "site-packages"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def ram_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                        ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                        ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        s = MS(); s.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return round(s.t / 2**30, 1)
    except Exception:
        return None


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERROR: {e!r}"


def main():
    rep, problems = {}, []
    rep["python"] = {"version": platform.python_version(), "executable": sys.executable}
    if platform.python_version() != "3.12.10":
        problems.append(f"Python {platform.python_version()} (expected 3.12.10)")
    rep["os"] = {"platform": platform.platform(), "processor": platform.processor(),
                 "cpu_count": os.cpu_count(), "ram_gb": ram_gb()}

    installed = {norm(d.metadata["Name"]): d.version for d in distributions()}
    req = {}
    for line in (REPRO / "env" / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            n, v = line.split("=="); req[norm(n)] = v
    orig = {}
    if ORIG_VENV_SP.is_dir():
        for d in ORIG_VENV_SP.glob("*.dist-info"):
            m = re.match(r"(.+?)-([^-]+)\.dist-info$", d.name)
            if m:
                orig[norm(m.group(1))] = m.group(2)
    rep["packages_vs_requirements"] = {n: {"required": v, "installed": installed.get(n)} for n, v in sorted(req.items())}
    for n, v in req.items():
        if installed.get(n) != v:
            problems.append(f"{n}: installed {installed.get(n)} vs required {v}")
    ignore = {"pip"}
    rep["original_venv312_found"] = bool(orig)
    for n, v in orig.items():
        if n not in ignore and installed.get(n) != v:
            problems.append(f"{n}: installed {installed.get(n)} vs original .venv312 {v}")
    for n in ("xgboost", "catboost"):
        if n in installed:
            problems.append(f"{n} is installed (absent in the original environment; the baseline script would run it)")

    try:
        import torch
        t = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
             "cuda_runtime_of_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            t.update(gpu=p.name, capability=f"{p.major}.{p.minor}", memory_gb=round(p.total_memory / 2**30, 1),
                     device_count=torch.cuda.device_count())
        else:
            problems.append("CUDA not available to PyTorch")
        rep["torch"] = t
    except Exception as e:
        problems.append(f"torch import failed: {e!r}")
    rep["nvidia_smi"] = {
        "query": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
        "header": [l.strip() for l in run(["nvidia-smi"]).splitlines() if "Driver Version" in l or "CUDA Version" in l],
    }
    rep["problems"] = problems
    rep["result"] = "PASS" if not problems else "CHECK"
    out = REPRO / "logs" / "stage0_env"
    out.mkdir(parents=True, exist_ok=True)
    (out / "env_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    lines = [f"Python {rep['python']['version']}  ({rep['python']['executable']})",
             f"OS {rep['os']['platform']} | CPU threads {rep['os']['cpu_count']} | RAM {rep['os']['ram_gb']} GB",
             f"PyTorch {rep.get('torch', {}).get('version')} | CUDA available {rep.get('torch', {}).get('cuda_available')} | "
             f"CUDA runtime (build) {rep.get('torch', {}).get('cuda_runtime_of_build')} | cuDNN {rep.get('torch', {}).get('cudnn')}",
             f"GPU {rep.get('torch', {}).get('gpu')} | capability {rep.get('torch', {}).get('capability')} | {rep.get('torch', {}).get('memory_gb')} GB",
             f"nvidia-smi: {rep['nvidia_smi']['query']}"] + rep["nvidia_smi"]["header"] + \
            [f"Packages checked: {len(req)} vs requirements, {len(orig)} vs original .venv312",
             f"RESULT: {rep['result']}"] + [f"  - {p}" for p in problems]
    (out / "env_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
