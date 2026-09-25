from __future__ import annotations

import math
import unittest

import numpy as np

from tfbsm_pricing.model import EuropeanOptionProblem, GridSpec, SolverResult
from tfbsm_pricing.weighted_solver import (
    optimal_weight,
    paper_unconditional_stability_threshold,
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
        problem = EuropeanOptionProblem(
            1.0,
            2.0,
            4.0,
            0.04,
            1.0,
            0.999,
            "call",
            boundary_mode="paper_reproduction",
        )
        grid = GridSpec(-20.0, 10.0, 500, 50)
        implicit = solve_weighted(problem, grid, theta=0.0)
        crank_nicolson = solve_weighted(problem, grid, theta=0.5)
        self.assertLess(abs(crank_nicolson.price - 0.593), abs(implicit.price - 0.593))
        self.assertLess(abs(crank_nicolson.price - 0.593), 0.002)

    def test_krzyzanowski_table_1_temporal_orders(self) -> None:
        cases = (
            (0.99, 0.01, 1.02),
            (0.7, 2.22e-3, 1.32),
            (0.5, 1.67e-3, 1.51),
            (0.3, 1.39e-3, 1.70),
            (0.1, 1.25e-3, 1.85),
        )
        for alpha, dt, paper_order in cases:
            problem = EuropeanOptionProblem(
                1.0,
                2.0,
                1.0,
                0.04,
                1.0,
                alpha,
                "call",
                boundary_mode="paper_reproduction",
            )
            reference = solve_weighted(
                problem, GridSpec(-1.0, 1.0, 10, 2597), theta=0.0
            )
            coarse = solve_weighted(
                problem, GridSpec(-1.0, 1.0, 10, round(1.0 / dt)), theta=0.0
            )
            fine = solve_weighted(
                problem, GridSpec(-1.0, 1.0, 10, round(2.0 / dt)), theta=0.0
            )

            def value(result: SolverResult) -> float:
                return float(np.interp(-0.01, result.x_grid, result.grid_price[-1]))

            order = math.log2(
                abs((value(coarse) - value(reference)) / (value(fine) - value(reference)))
            )
            self.assertAlmostEqual(order, paper_order, delta=0.07)

    def test_paper_unconditional_stability_condition(self) -> None:
        self.assertEqual(paper_unconditional_stability_threshold(0.0), 0.0)
        self.assertAlmostEqual(
            paper_unconditional_stability_threshold(0.25),
            1.0 - math.log2(5.0 / 3.0),
        )
        self.assertEqual(paper_unconditional_stability_threshold(0.5), 1.0)

        alpha = 0.7
        theta_hat = optimal_weight(alpha)
        self.assertAlmostEqual(
            paper_unconditional_stability_threshold(theta_hat), alpha, places=14
        )
        problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.0, 0.3, alpha, "call")
        grid = GridSpec(-4.0, 3.0, 60, 40)
        self.assertFalse(
            solve_weighted(problem, grid, theta=0.5).diagnostics[
                "paper_unconditional_stability_condition"
            ]
        )
        self.assertTrue(
            solve_weighted(problem, grid, theta=theta_hat).diagnostics[
                "paper_unconditional_stability_condition"
            ]
        )


if __name__ == "__main__":
    unittest.main()
