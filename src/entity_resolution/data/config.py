"""Compatibility adapter for the shared runtime configuration.

New code uses :mod:`entity_resolution.config` directly.  This adapter keeps
the existing matcher CLI surface while making ``ER_DATA_DIR`` the only data
location setting.
"""

from __future__ import annotations

from pathlib import Path

from entity_resolution.config import DATA_DIR

DATA_DIR_ENV_VAR = "ER_DATA_DIR"


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return a validated dataset root from an explicit value or local config."""

    path = Path(data_dir).expanduser() if data_dir is not None else DATA_DIR
    missing = [name for name in ("train", "test") if not (path / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset directory {path} is missing required folder(s): {', '.join(missing)}; set {DATA_DIR_ENV_VAR}."
        )
    return path
