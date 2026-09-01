from __future__ import annotations

import unittest

import pandas as pd

from tfbsm_empirical.data.validation import (
    SchemaValidationError,
    validate_frame,
    validate_reconciliation,
)


class ValidationTests(unittest.TestCase):
    def test_missing_required_columns_fail_loudly(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_frame("option_quotes", pd.DataFrame({"timestamp": []}))

    def test_reconciliation_requires_exact_count(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_reconciliation(expected=2, collected=1)


if __name__ == "__main__":
    unittest.main()
