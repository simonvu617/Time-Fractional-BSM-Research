from __future__ import annotations

import math
import unittest

import numpy as np

from tfbsm_pricing import black_scholes_price, solve_l2, solve_weighted
from tfbsm_pricing.model import (
    EuropeanOptionProblem,
    GridSpec,
    boundary_values,
    payoff,
)


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

    def test_both_solvers_converge_to_bsm_for_calls_and_puts(self) -> None:
        grid = GridSpec(math.log(5.0), math.log(500.0), 240, 200)
        for option_type in ("call", "put"):
            problem = EuropeanOptionProblem(
                100.0, 105.0, 0.75, 0.03, 0.25, 1.0, option_type
            )
            exact = black_scholes_price(problem)
            self.assertLess(abs(solve_weighted(problem, grid).price - exact), 0.03)
            self.assertLess(abs(solve_l2(problem, grid).price - exact), 0.03)

    def test_refinement_reduces_bsm_limit_error(self) -> None:
        problem = EuropeanOptionProblem(100.0, 100.0, 1.0, 0.05, 0.2, 1.0, "call")
        exact = black_scholes_price(problem)
        coarse = GridSpec(math.log(5.0), math.log(500.0), 80, 50)
        fine = GridSpec(math.log(5.0), math.log(500.0), 160, 100)
        for solver in (solve_weighted, solve_l2):
            self.assertLess(
                abs(solver(problem, fine).price - exact),
                abs(solver(problem, coarse).price - exact),
            )

    def test_fractional_solvers_agree_and_have_sane_surfaces(self) -> None:
        grid = GridSpec(-4.0, 3.0, 180, 120)
        for alpha in (0.5, 0.8, 0.95):
            for option_type in ("call", "put"):
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

    def test_cross_solver_difference_shrinks_under_refinement(self) -> None:
        for alpha in (0.5, 0.9):
            problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.04, 0.3, alpha, "call")
            differences = []
            for count in (60, 120, 240):
                grid = GridSpec(-5.0, 5.0, count, count)
                differences.append(
                    abs(solve_weighted(problem, grid).price - solve_l2(problem, grid).price)
                )
            self.assertGreater(differences[0], differences[1])
            self.assertGreater(differences[1], differences[2])


if __name__ == "__main__":
    unittest.main()
