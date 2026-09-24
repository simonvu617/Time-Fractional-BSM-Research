from __future__ import annotations

import math
import unittest

import numpy as np

from tfbsm_pricing.boundaries import boundary_values, payoff
from tfbsm_pricing.grid import GridSpec
from tfbsm_pricing.model import EuropeanOptionProblem
from tfbsm_pricing.weighted_fd import solve_weighted


class ModelTests(unittest.TestCase):
    def test_invalid_model_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EuropeanOptionProblem(0.0, 100.0, 1.0, 0.03, 0.2, 0.8, "call")
        with self.assertRaises(ValueError):
            EuropeanOptionProblem(100.0, 100.0, 1.0, 0.03, 0.2, 1.1, "call")

    def test_payoff_and_boundaries_use_elapsed_time(self) -> None:
        problem = EuropeanOptionProblem(100.0, 100.0, 1.0, 0.05, 0.2, 0.7, "call")
        x = np.log(np.array([80.0, 100.0, 120.0]))
        np.testing.assert_allclose(payoff(problem, x), [0.0, 0.0, 20.0], atol=1e-12)
        left, right = boundary_values(problem, math.log(1.0), math.log(400.0), 1.0)
        self.assertEqual(left, 0.0)
        self.assertAlmostEqual(right, 400.0 - 100.0 * math.exp(-0.05))

    def test_grid_counts_intervals(self) -> None:
        grid = GridSpec(-2.0, 2.0, 8, 5)
        self.assertEqual(grid.x_values().size, 9)
        self.assertEqual(grid.time_values(1.0).size, 6)
        self.assertAlmostEqual(grid.dx, 0.5)

    def test_spot_must_be_inside_pricing_domain(self) -> None:
        problem = EuropeanOptionProblem(100.0, 100.0, 1.0, 0.05, 0.2, 0.8, "call")
        with self.assertRaises(ValueError):
            solve_weighted(problem, GridSpec(-2.0, 2.0, 20, 10))


if __name__ == "__main__":
    unittest.main()
