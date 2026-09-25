"""Single source of truth for paths and runtime settings.

Everything is overridable through environment variables (or a repo-root `.env`
file with KEY=VALUE lines), so the same code runs on a laptop, an AWS box, or
inside the submission zip without edits. Nothing else in the package may
hardcode a filesystem path.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")

DATA_DIR = Path(os.environ.get(
    "ER_DATA_DIR", REPO_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", REPO_ROOT / "output"))
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", OUTPUT_DIR / "cache"))
SPLIT_DIR = OUTPUT_DIR / "validation_split"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
SPLIT_MANIFEST = RESULTS_DIR / "validation_split_manifest.json"
OFFICIAL_VALIDATOR = Path(os.environ.get(
    "ER_OFFICIAL_VALIDATOR", DATA_DIR.parent / "utils" / "validate_submission.py"))

# Threads for sparse top-k / BLAS. Default leaves one core for the OS.
N_JOBS = int(os.environ.get("ER_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
SEED = int(os.environ.get("ER_SEED", 42))
