import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entity_resolution.data.config import DATA_DIR_ENV_VAR, resolve_data_dir


class DataConfigTests(unittest.TestCase):
    def test_explicit_dataset_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train").mkdir()
            (root / "test").mkdir()
            self.assertEqual(resolve_data_dir(root), root)

    def test_missing_shared_config_value_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            with patch("entity_resolution.data.config.DATA_DIR", missing):
                with self.assertRaisesRegex(FileNotFoundError, DATA_DIR_ENV_VAR):
                    resolve_data_dir()


if __name__ == "__main__":
    unittest.main()
