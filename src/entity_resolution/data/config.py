"""Local data-location configuration for reproducible execution scripts.

The supplied datasets stay outside the Git checkout. A developer sets
``AMAZON_ML_DATA_DIR`` in an ignored ``.env`` file (or process environment) to
the directory containing the ``train/`` and ``test/`` folders.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV_VAR = "AMAZON_ML_DATA_DIR"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_local_env(env_path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from a local .env without a dependency.

    Existing process environment variables win, so CI and explicit shell values
    remain authoritative. This parser supports the simple format needed for a
    local filesystem path.
    """

    path = env_path or REPOSITORY_ROOT / ".env"
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return a validated dataset root from an explicit value or local config."""

    if data_dir is None:
        load_local_env()
        data_dir = os.environ.get(DATA_DIR_ENV_VAR)
    if not data_dir:
        raise RuntimeError(
            f"Set {DATA_DIR_ENV_VAR} in .env or the process environment. "
            "It must point to the supplied dataset directory containing train/ and test/."
        )

    path = Path(data_dir).expanduser()
    missing = [name for name in ("train", "test") if not (path / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset directory {path} is missing required folder(s): {', '.join(missing)}."
        )
    return path
