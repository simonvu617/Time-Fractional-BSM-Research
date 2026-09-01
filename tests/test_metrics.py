from __future__ import annotations

import unittest

from tfbsm_empirical.metrics import absolute_repricing_error, repricing_improvement


class RepricingMetricTests(unittest.TestCase):
    def test_absolute_repricing_error(self) -> None:
        self.assertAlmostEqual(absolute_repricing_error(5.25, 5.0), 0.25)

    def test_positive_improvement_favors_tfbsm(self) -> None:
        self.assertAlmostEqual(repricing_improvement(0.30, 0.20), 0.10)

    def test_loss_inputs_must_be_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            repricing_improvement(-0.10, 0.20)


if __name__ == "__main__":
    unittest.main()
