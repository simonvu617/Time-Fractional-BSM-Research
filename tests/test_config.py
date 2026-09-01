from __future__ import annotations

import unittest
from pathlib import Path

from tfbsm_empirical.data.config import CollectorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CollectorConfigTests(unittest.TestCase):
    def test_repository_config_is_project_local_and_resolves_symbols(self) -> None:
        config = CollectorConfig.from_yaml(PROJECT_ROOT / "configs" / "collector.yaml")

        self.assertTrue(config.output_dir.is_relative_to(PROJECT_ROOT))
        self.assertEqual(config.symbol("spy").symbol, "SPY")
        self.assertEqual(config.option_rights, ("call", "put"))

    def test_unknown_symbol_fails(self) -> None:
        config = CollectorConfig.from_yaml(PROJECT_ROOT / "configs" / "collector.yaml")

        with self.assertRaises(KeyError):
            config.symbol("NOT_CONFIGURED")


if __name__ == "__main__":
    unittest.main()
