import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.entity_resolution.data.config import DATA_DIR_ENV_VAR, resolve_data_dir


class DataConfigTests(unittest.TestCase):
    def test_explicit_dataset_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train").mkdir()
            (root / "test").mkdir()
            self.assertEqual(resolve_data_dir(root), root)

    def test_missing_environment_value_has_actionable_error(self) -> None:
        with patch.dict(os.environ, {DATA_DIR_ENV_VAR: ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, DATA_DIR_ENV_VAR):
                resolve_data_dir()


if __name__ == "__main__":
    unittest.main()
