from __future__ import annotations

import unittest

import numpy as np

from tfbsm_pricing.grid import GridSpec
from tfbsm_pricing.model import EuropeanOptionProblem
from tfbsm_pricing.weighted_fd import (
    optimal_weight,
    solve_weighted,
    weighted_caputo_weights,
)


class WeightedMethodTests(unittest.TestCase):
    def test_l1_weights_and_classical_limit(self) -> None:
        alpha = 0.6
        weights = weighted_caputo_weights(alpha, 4)
        expected = np.array(
            [(j + 1) ** (1 - alpha) - j ** (1 - alpha) for j in range(4)]
        )
        np.testing.assert_allclose(weights, expected)
        np.testing.assert_array_equal(weighted_caputo_weights(1.0, 4), [1, 0, 0, 0])
        self.assertAlmostEqual(optimal_weight(1.0), 0.5)

    def test_krzyzanowski_example_2_is_near_reported_benchmark(self) -> None:
        problem = EuropeanOptionProblem(1.0, 2.0, 4.0, 0.04, 1.0, 0.999, "call")
        grid = GridSpec(-20.0, 10.0, 500, 50)
        implicit = solve_weighted(problem, grid, theta=0.0)
        crank_nicolson = solve_weighted(problem, grid, theta=0.5)
        self.assertLess(abs(crank_nicolson.price - 0.593), abs(implicit.price - 0.593))
        self.assertLess(abs(crank_nicolson.price - 0.593), 0.002)


if __name__ == "__main__":
    unittest.main()
