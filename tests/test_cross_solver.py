from __future__ import annotations

import unittest

import numpy as np

from tfbsm_pricing import EuropeanOptionProblem, GridSpec, solve_l2, solve_weighted


class CrossSolverTests(unittest.TestCase):
    def test_fractional_prices_agree_and_have_sane_surfaces(self) -> None:
        grid = GridSpec(-4.0, 3.0, 180, 120)
        for alpha in (0.5, 0.8, 0.95):
            for option_type in ("call", "put"):
                with self.subTest(alpha=alpha, option_type=option_type):
                    problem = EuropeanOptionProblem(
                        1.0, 1.1, 1.0, 0.04, 0.3, alpha, option_type
                    )
                    weighted = solve_weighted(problem, grid)
                    l2 = solve_l2(problem, grid)
                    self.assertLess(abs(weighted.price - l2.price), 8.0e-4)
                    self.assertGreaterEqual(np.min(weighted.grid_price), -1.0e-10)
                    self.assertGreaterEqual(np.min(l2.grid_price), -1.0e-10)
                    if option_type == "call":
                        self.assertGreaterEqual(np.min(np.diff(weighted.grid_price[-1])), -1e-9)
                        self.assertGreaterEqual(np.min(np.diff(l2.grid_price[-1])), -1e-9)


if __name__ == "__main__":
    unittest.main()
