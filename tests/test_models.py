from __future__ import annotations

import unittest

from tfbsm_empirical.models import (
    BSM_SPECIFICATION,
    TFBSM_SPECIFICATION,
    validate_alpha,
)


class ModelSpecificationTests(unittest.TestCase):
    def test_model_families_remain_distinct(self) -> None:
        self.assertEqual(BSM_SPECIFICATION.family, "classical_diffusion")
        self.assertEqual(TFBSM_SPECIFICATION.family, "inverse_stable_market_time")
        self.assertEqual(TFBSM_SPECIFICATION.implementation_status, "not_implemented")

    def test_alpha_one_is_valid_bsm_boundary(self) -> None:
        self.assertEqual(validate_alpha(1.0), 1.0)

    def test_alpha_outside_locked_domain_fails(self) -> None:
        for value in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_alpha(value)


if __name__ == "__main__":
    unittest.main()
