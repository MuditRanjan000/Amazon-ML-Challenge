"""Thin wrapper around the organisers' validator (student_resource/utils/validate_submission.py).

We deliberately do not re-implement the rules: the official script is the source
of truth, is stdlib-only, and streams the files (low memory).
"""
import importlib.util
from pathlib import Path

from .. import config


def _official():
    path = Path(config.OFFICIAL_VALIDATOR)
    if not path.is_file():
        raise FileNotFoundError(f"Official validator not found at {path} (set ER_OFFICIAL_VALIDATOR)")
    spec = importlib.util.spec_from_file_location("official_validate_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SubmissionValidator:
    def __init__(self, test_dir=None):
        self.test_dir = str(test_dir or Path(config.DATA_DIR) / "test")

    def validate(self, submission_path, candidate_path=None, check_ids: bool = False) -> bool:
        errors, warnings = _official().validate(str(submission_path), candidate_path and str(candidate_path),
                                                self.test_dir, check_ids=check_ids)
        for w in warnings:
            print(f"WARNING: {w}")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        print("PASS" if not errors else f"FAIL - {len(errors)} issue(s)")
        return not errors
