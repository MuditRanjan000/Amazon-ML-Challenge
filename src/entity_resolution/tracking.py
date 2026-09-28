"""Experiment tracking: one JSON line per run, tied to a git commit.

JSONL (not CSV) so different experiments can log different fields without
breaking a header, and every line is self-describing and diff-friendly.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from . import config


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 1e9
    import resource  # POSIX: ru_maxrss is KB on Linux, bytes on macOS
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


def git_commit() -> str:
    if os.environ.get("ER_GIT_COMMIT"):  # code shipped without .git (e.g. a tarball on AWS)
        return os.environ["ER_GIT_COMMIT"]
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=config.REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_run(record: dict, path: Path = None) -> dict:
    """Append `record` (+ timestamp, commit, peak RSS) to the experiment log and return it."""
    path = Path(path or config.RESULTS_DIR / "experiments.jsonl")
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(),
              "peak_rss_gb": round(peak_rss_gb(), 2), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
