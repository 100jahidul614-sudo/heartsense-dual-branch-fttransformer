"""
Stage runner for the HeartSense reproduction (non-scientific helper).

For one stage defined in tools/stages.json it:
  1. Pre-flight checks every path (no step runs if any check fails):
       - each script exists; scripts in original_code/ match the original project byte-for-byte;
       - every input exists (or is produced by an earlier step of the same stage);
       - no output exists yet (nothing is ever overwritten);
       - every path is inside HeartSense_repro (the raw BRFSS file is the only allowed exception);
       - nothing points into the original Heartseanse2.0 project;
       - the raw BRFSS file has the expected SHA-256 (BRFSS stages).
  2. Runs the steps in order with the venv's Python, logging command, working folder,
     start/end time, duration, exit code and full console output.
  3. Snapshots the environment (Python, packages, GPU, driver, CUDA) into the log folder.
  4. Confirms afterwards that the original project is still unchanged (baseline hashes).

Usage (from HeartSense_repro):
    .venv\\Scripts\\python.exe tools\\run_stage.py --stage 1 --check_only   # pre-flight only
    .venv\\Scripts\\python.exe tools\\run_stage.py --stage 1                # pre-flight, then run
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPRO = TOOLS.parent
CFG = json.loads((TOOLS / "stages.json").read_text(encoding="utf-8"))
# Environment overrides exist only so the tools can be tested on another machine; on the PC leave them unset.
BRFSS = os.environ.get("HS_BRFSS_XPT", CFG["brfss_xpt"])
ORIG = Path(os.environ.get("HS_ORIG_ROOT", CFG["original_project"]))
REPO_DIRS = ("data", "runs", "reference", "original_code", "run_code")
EXTRA_ENV = {  # recorded in the log; none of them changes any computation
    "MPLBACKEND": "Agg",          # figures are saved to files; plt.show() becomes a no-op instead of blocking
    "PYTHONIOENCODING": "utf-8",  # console output is captured to UTF-8 log files
    "PYTHONUNBUFFERED": "1",      # progress lines reach the log immediately
    "PYTHONDONTWRITEBYTECODE": "1",  # keeps original_code/ free of __pycache__
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(token: str) -> Path:
    if token == "{BRFSS_XPT}":
        return Path(BRFSS)
    return REPRO / Path(token)


def is_path_arg(a: str) -> bool:
    return a == "{BRFSS_XPT}" or a.split("/")[0] in REPO_DIRS


def inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def preflight(stage: dict, verbose=True) -> list:
    problems, produced = [], set()
    if stage.get("uses_brfss_xpt"):
        x = Path(BRFSS)
        if not x.is_file():
            problems.append(f"raw BRFSS file missing: {x}")
        else:
            t = time.time(); h = sha256(x)
            if h != CFG["brfss_xpt_sha256"]:
                problems.append(f"raw BRFSS SHA-256 mismatch: {h}")
            elif verbose:
                print(f"[OK] raw BRFSS file SHA-256 verified ({time.time()-t:.1f}s)")
    copies = {m["copy_rel"]: m for m in json.loads((REPRO / "manifests" / "stage0_copies.json").read_text(encoding="utf-8"))}
    for st in stage["steps"]:
        sid = st["id"]
        script = REPRO / st["script"]
        if not script.is_file():
            problems.append(f"{sid}: script missing {script}")
        elif st["script"].startswith("original_code/"):
            m = copies.get(st["script"])
            if not m or sha256(script) != m["sha256"] or sha256(ORIG / m["source_rel"]) != m["sha256"]:
                problems.append(f"{sid}: {st['script']} is not byte-identical to the original")
        if st["script"].startswith("run_code/"):
            text = script.read_text(encoding="utf-8")
            if "Heartseanse2.0" in text:
                problems.append(f"{sid}: {st['script']} still references the original project")
        paths = [a for a in st["args"] if is_path_arg(a)] + st.get("inputs", []) + st.get("outputs", []) + st.get("updates", [])
        for tok in paths + [st["cwd"]]:
            p = resolve(tok)
            if inside(p, ORIG):
                problems.append(f"{sid}: path points into the original project: {p}")
            if tok != "{BRFSS_XPT}" and not inside(p, REPRO):
                problems.append(f"{sid}: path outside HeartSense_repro: {p}")
        for tok in st.get("inputs", []) + st.get("updates", []):
            p = resolve(tok)
            if not p.exists() and not any(tok == o or tok.startswith(o.rstrip("/") + "/") for o in produced):
                problems.append(f"{sid}: input not found and not produced earlier in this stage: {tok}")
        for tok in st.get("outputs", []):
            if resolve(tok).exists():
                problems.append(f"{sid}: output already exists (refusing to overwrite): {tok}")
            produced.add(tok)
    return problems


def env_snapshot() -> dict:
    snap = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "extra_env": EXTRA_ENV}
    try:
        from importlib.metadata import distributions
        snap["packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
    except Exception as e:
        snap["packages_error"] = repr(e)
    try:
        import torch
        snap["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                         "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        if torch.cuda.is_available():
            snap["torch"]["gpu"] = torch.cuda.get_device_name(0)
            snap["torch"]["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:
        snap["torch_error"] = repr(e)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        snap["nvidia_smi"] = out.stdout.strip()
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout.splitlines()
        snap["nvidia_smi_header"] = [l for l in head if "CUDA Version" in l]
    except Exception as e:
        snap["nvidia_smi_error"] = repr(e)
    return snap


def check_original_unchanged() -> str:
    base = REPRO / "manifests" / "original_project_baseline.sha256"
    if not base.is_file():
        return "baseline missing"
    bad = []
    for line in base.read_text(encoding="utf-8").splitlines():
        h, rel = line.split("  ", 1)
        p = ORIG / rel
        if not p.is_file() or sha256(p) != h:
            bad.append(rel)
    return "unchanged" if not bad else f"CHANGED: {bad[:10]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(CFG["stages"]))
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()
    stage = CFG["stages"][args.stage]
    print(f"=== Stage {args.stage}: {stage['name']}")
    problems = preflight(stage)
    for st in stage["steps"]:
        cmd = [st["script"]] + st["args"]
        print(f"  [{st['id']}] {st['title']}\n        python {' '.join(cmd)}   (cwd: {st['cwd']})")
    if problems:
        print("\n[STOP] pre-flight failed:")
        for p in problems:
            print("   -", p)
        sys.exit(2)
    print("\n[OK] pre-flight passed: all inputs present, no output exists, all paths inside HeartSense_repro.")
    if args.check_only:
        return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = REPRO / "logs" / f"stage{args.stage}_{ts}"
    logdir.mkdir(parents=True)
    (logdir / "environment.json").write_text(json.dumps(env_snapshot(), indent=2), encoding="utf-8")
    env = dict(os.environ, **EXTRA_ENV)
    rows, t_stage = [], time.time()
    for st in stage["steps"]:
        cwd = resolve(st["cwd"]); cwd.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(REPRO / st["script"])] + [str(resolve(a)) if is_path_arg(a) else a for a in st["args"]]
        logf = logdir / f"step_{st['id']}.log"
        start = dt.datetime.now(); t0 = time.time()
        print(f"\n>>> [{st['id']}] {st['title']}  ({start:%H:%M:%S})")
        with open(logf, "w", encoding="utf-8") as lf:
            lf.write(f"# command: {subprocess.list2cmdline(argv)}\n# cwd: {cwd}\n# start: {start.isoformat()}\n\n")
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line); lf.write(line)
            code = proc.wait()
            end = dt.datetime.now(); secs = time.time() - t0
            lf.write(f"\n# end: {end.isoformat()}  duration_s: {secs:.1f}  exit_code: {code}\n")
        rows.append({"step": st["id"], "title": st["title"], "command": subprocess.list2cmdline(argv), "cwd": str(cwd),
                     "start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
                     "duration_s": f"{secs:.1f}", "exit_code": code})
        print(f"<<< [{st['id']}] exit {code} after {secs/60:.1f} min")
        if code != 0:
            break
    with open(logdir / "steps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    status = check_original_unchanged()
    summary = {"stage": args.stage, "total_minutes": round((time.time() - t_stage) / 60, 2),
               "all_steps_ok": all(r["exit_code"] == 0 for r in rows) and len(rows) == len(stage["steps"]),
               "original_project": status}
    (logdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== Stage {args.stage} finished: {summary}\n    logs: {logdir}")
    sys.exit(0 if summary["all_steps_ok"] else 1)


if __name__ == "__main__":
    main()
