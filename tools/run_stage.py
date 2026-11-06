"""
Stage runner (non-scientific helper). Package edition of the runner used for the verified November 2026 runs
(that version is in evidence/tools_as_run/). Differences: the repository root is wherever this package sits;
the raw BRFSS file defaults to data/brfss/main_data_brfss_2024.XPT (or HS_BRFSS_XPT / --brfss_xpt);
original_code/ is checked against manifests/original_code_distribution.sha256 instead of the private original
project; --steps runs a subset of a stage. The steps, scripts and arguments (tools/stages.json) are those of the
verified runs.

For one stage it:
  1. Pre-flight checks (no step runs if any check fails):
       - each script exists; original_code/ scripts match the distribution manifest;
       - run_code/ was generated for this folder by tools/make_run_copies.py and is unchanged since;
       - every input exists (or is produced by an earlier step); no output exists yet (nothing is overwritten);
       - every path is inside the package folder (the raw BRFSS file is the only allowed exception);
       - the raw BRFSS file has the SHA-256 of the file used for the paper and the verified runs (BRFSS stages).
  2. Runs the steps in order with this Python, logging command, working folder, start/end time, duration,
     exit code and full console output to logs/stageN_<timestamp>/.
  3. Snapshots the environment (Python, packages, GPU, driver, CUDA) into the same folder.

Usage (from the package folder):
    python tools/run_stage.py --stage 1 --check_only
    python tools/run_stage.py --stage 1
    python tools/run_stage.py --stage 4 --steps 4.3
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
REPO = TOOLS.parent
CFG = json.loads((TOOLS / "stages.json").read_text(encoding="utf-8"))
REPO_DIRS = ("data", "runs", "original_code", "run_code")
EXTRA_ENV = {  # recorded in the log; none of them changes any computation
    "MPLBACKEND": "Agg",          # figures are saved to files; plt.show() becomes a no-op instead of blocking
    "PYTHONIOENCODING": "utf-8",  # console output is captured to UTF-8 log files
    "PYTHONUNBUFFERED": "1",      # progress lines reach the log immediately
    "PYTHONDONTWRITEBYTECODE": "1",  # keeps original_code/ free of __pycache__
}
BRFSS = None  # set in main()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(token: str) -> Path:
    if token == "{BRFSS_XPT}":
        return Path(BRFSS)
    return REPO / Path(token)


def is_path_arg(a: str) -> bool:
    return a == "{BRFSS_XPT}" or a.split("/")[0] in REPO_DIRS


def inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_sha_list(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            out[rel] = h
    return out


def preflight(steps: list, uses_brfss: bool) -> list:
    problems, produced = [], set()
    if uses_brfss:
        x = Path(BRFSS)
        if not x.is_file():
            problems.append(f"raw BRFSS file missing: {x} (see DATA.md)")
        else:
            t = time.time(); h = sha256(x)
            if h != CFG["brfss_xpt_sha256"]:
                problems.append(f"raw BRFSS SHA-256 differs from the file used for the paper: {h}")
            else:
                print(f"[OK] raw BRFSS file SHA-256 verified ({time.time()-t:.1f}s)")
    dist = load_sha_list(REPO / "manifests" / "original_code_distribution.sha256")
    gen_file = REPO / "run_code" / "_generated_for.json"
    gen = json.loads(gen_file.read_text(encoding="utf-8")) if gen_file.is_file() else None
    for st in steps:
        sid = st["id"]
        script = REPO / st["script"]
        if not script.is_file():
            problems.append(f"{sid}: script missing {script}")
        elif st["script"].startswith("original_code/"):
            if dist.get(st["script"]) != sha256(script):
                problems.append(f"{sid}: {st['script']} differs from the distribution copy (manifest)")
        elif st["script"].startswith("run_code/"):
            name = Path(st["script"]).name
            if gen is None:
                problems.append(f"{sid}: run_code/ not generated for this folder: run tools/make_run_copies.py --overwrite")
            elif Path(gen["repo_root"]).resolve() != REPO.resolve():
                problems.append(f"{sid}: run_code/ was generated for {gen['repo_root']}, not this folder: "
                                "run tools/make_run_copies.py --overwrite")
            elif gen["sha256"].get(name) != sha256(script):
                problems.append(f"{sid}: {st['script']} changed after it was generated")
            elif Path(gen["brfss_xpt"]).resolve() != Path(BRFSS).resolve() and "{BRFSS_XPT}" in st.get("inputs", []):
                problems.append(f"{sid}: run_code/ points to a different BRFSS file: {gen['brfss_xpt']}")
        paths = [a for a in st["args"] if is_path_arg(a)] + st.get("inputs", []) + st.get("outputs", []) + st.get("updates", [])
        for tok in paths + [st["cwd"]]:
            p = resolve(tok)
            if tok != "{BRFSS_XPT}" and not inside(p, REPO):
                problems.append(f"{sid}: path outside the package folder: {p}")
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


def main():
    global BRFSS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(CFG["stages"]))
    ap.add_argument("--steps", default="", help="comma-separated step ids to run (default: all steps of the stage)")
    ap.add_argument("--brfss_xpt", default=os.environ.get("HS_BRFSS_XPT", ""))
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()
    BRFSS = args.brfss_xpt or str(REPO / CFG["brfss_xpt"])
    stage = CFG["stages"][args.stage]
    steps = stage["steps"]
    if args.steps:
        want = [s.strip() for s in args.steps.split(",")]
        steps = [s for s in steps if s["id"] in want]
        if len(steps) != len(want):
            raise SystemExit(f"[STOP] unknown step id in {want}")
    uses_brfss = stage.get("uses_brfss_xpt", False) and any("{BRFSS_XPT}" in s["args"] + s.get("inputs", []) for s in steps)
    print(f"=== Stage {args.stage}: {stage['name']}" + (f"  (steps {args.steps})" if args.steps else ""))
    problems = preflight(steps, uses_brfss)
    for st in steps:
        print(f"  [{st['id']}] {st['title']}\n        python {' '.join([st['script']] + st['args'])}   (cwd: {st['cwd']})")
    if problems:
        print("\n[STOP] pre-flight failed:")
        for p in problems:
            print("   -", p)
        sys.exit(2)
    print("\n[OK] pre-flight passed: all inputs present, no output exists, all paths inside the package folder.")
    if args.check_only:
        return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = REPO / "logs" / f"stage{args.stage}_{ts}"
    logdir.mkdir(parents=True)
    (logdir / "environment.json").write_text(json.dumps(env_snapshot(), indent=2), encoding="utf-8")
    env = dict(os.environ, **EXTRA_ENV)
    rows, t_stage = [], time.time()
    for st in steps:
        cwd = resolve(st["cwd"]); cwd.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(REPO / st["script"])] + [str(resolve(a)) if is_path_arg(a) else a for a in st["args"]]
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
    summary = {"stage": args.stage, "steps": [s["id"] for s in steps],
               "total_minutes": round((time.time() - t_stage) / 60, 2),
               "all_steps_ok": all(r["exit_code"] == 0 for r in rows) and len(rows) == len(steps)}
    (logdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== Stage {args.stage} finished: {summary}\n    logs: {logdir}")
    sys.exit(0 if summary["all_steps_ok"] else 1)


if __name__ == "__main__":
    main()
