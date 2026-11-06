"""
Re-verifies everything Stage 0 prepared (read-only; changes nothing).

Checks:
  1. original_code/, data/nhanes/ and reference/ copies are byte-identical to the original project.
  2. Each run_code/ script differs from its original only on the lines listed in
     manifests/path_changes.json, and no run_code/ script still points into Heartseanse2.0.
  3. The original project still matches manifests/original_project_baseline.sha256.
  4. The raw BRFSS file has the expected SHA-256.

Usage (from HeartSense_repro):  .venv\\Scripts\\python.exe tools\\verify_stage0.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPRO = TOOLS.parent
CFG = json.loads((TOOLS / "stages.json").read_text(encoding="utf-8"))
ORIG = Path(os.environ.get("HS_ORIG_ROOT", CFG["original_project"]))
BRFSS = Path(os.environ.get("HS_BRFSS_XPT", CFG["brfss_xpt"]))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    fails, counts = [], {}
    manifest = json.loads((REPRO / "manifests" / "stage0_copies.json").read_text(encoding="utf-8"))
    changes = json.loads((REPRO / "manifests" / "path_changes.json").read_text(encoding="utf-8"))
    for m in manifest:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
        copy = REPRO / m["copy_rel"]
        if not copy.is_file() or sha256(copy) != m["sha256"]:
            fails.append(f"copy changed or missing: {m['copy_rel']}")
        if m["type"] != "run_copy" and sha256(ORIG / m["source_rel"]) != m["sha256"]:
            fails.append(f"copy differs from original: {m['copy_rel']}")
        if m["type"] == "run_copy":
            orig_lines = (ORIG / m["source_rel"]).read_bytes().decode("utf-8").splitlines()
            run_lines = copy.read_bytes().decode("utf-8").splitlines()
            allowed = {c["line"] for c in changes if c["script"] == Path(m["copy_rel"]).name}
            diff = {i for i, (a, b) in enumerate(zip(orig_lines, run_lines), 1) if a != b}
            if len(orig_lines) != len(run_lines) or diff != allowed:
                fails.append(f"{m['copy_rel']}: differs outside the documented path lines")
            if "Heartseanse2.0" in "\n".join(run_lines):
                fails.append(f"{m['copy_rel']}: still references the original project")
    n_base = 0
    for line in (REPRO / "manifests" / "original_project_baseline.sha256").read_text(encoding="utf-8").splitlines():
        h, rel = line.split("  ", 1); n_base += 1
        p = ORIG / rel
        if not p.is_file() or sha256(p) != h:
            fails.append(f"original project file changed: {rel}")
    xpt_ok = BRFSS.is_file() and sha256(BRFSS) == CFG["brfss_xpt_sha256"]
    if not xpt_ok:
        fails.append("raw BRFSS file missing or SHA-256 mismatch")
    report = {"copies_checked": counts, "path_lines_documented": len(changes),
              "original_project_files_checked": n_base, "brfss_xpt_sha256_ok": xpt_ok,
              "result": "PASS" if not fails else "FAIL", "failures": fails}
    print(json.dumps(report, indent=2))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
