from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from tfbsm_pricing import (
    AmericanPutProblem,
    EuropeanOptionProblem,
    StockGridSpec,
    solve_zhou_gao,
)
from validation.subordination import subordinated_bsm_price


def american_binomial_put(problem: AmericanPutProblem, steps: int = 2000) -> float:
    """Independent classical CRR tree, including dividend yield and exercise."""

    dt = problem.T / steps
    up = math.exp(problem.sigma * math.sqrt(dt))
    probability = (math.exp((problem.r - problem.q) * dt) - 1.0 / up) / (up - 1.0 / up)
    discount = math.exp(-problem.r * dt)
    stock = problem.S0 * up ** (2 * np.arange(steps + 1) - steps)
    values = np.maximum(problem.K - stock, 0.0)
    for _ in range(steps):
        stock = stock[:-1] * up
        continuation = discount * (
            (1.0 - probability) * values[:-1] + probability * values[1:]
        )
        values = np.maximum(problem.K - stock, continuation)
    return float(values[0])


class ZhouGaoMethodTests(unittest.TestCase):
    def test_classical_limit_converges_to_independent_american_tree(self) -> None:
        for rate, dividend, sigma in ((0.05, 0.0, 0.2), (0.05, 0.08, 0.2), (0.0, 0.15, 0.1)):
            with self.subTest(r=rate, q=dividend):
                problem = AmericanPutProblem(100, 100, 1, rate, sigma, 1, q=dividend)
                reference = american_binomial_put(problem)
                coarse = solve_zhou_gao(problem, StockGridSpec(400, 200, 100))
                fine = solve_zhou_gao(problem, StockGridSpec(400, 400, 200))
                self.assertLess(abs(fine.price - reference), abs(coarse.price - reference))
                self.assertAlmostEqual(fine.price, reference, delta=0.015)
                if rate == 0:
                    self.assertTrue(fine.diagnostics["one_sided_stock_nodes"])
                    np.testing.assert_array_equal(fine.exercise_boundary[1:], 0.0)

    def test_fractional_surface_satisfies_the_obstacle_pde(self) -> None:
        problems = (
            AmericanPutProblem(100, 100, 1, 0.05, 0.3, 0.8, q=0.02),
            AmericanPutProblem(100, 103.7, 1, 0.03, 0.35, 0.5, q=0.08),
        )
        for problem in problems:
            with self.subTest(alpha=problem.alpha, q=problem.q):
                grid = StockGridSpec(400, 160, 40)
                result = solve_zhou_gao(problem, grid)
                values = result.grid_price
                payoff = np.maximum(problem.K - result.stock_grid, 0.0)
                np.testing.assert_array_equal(values[0], payoff)
                np.testing.assert_array_equal(values[:, 0], problem.K)
                np.testing.assert_array_equal(values[:, -1], 0.0)
                self.assertGreaterEqual(float(np.min(values - payoff)), -1e-10)
                self.assertLessEqual(float(np.max(values)), problem.K + 1e-10)
                self.assertLessEqual(float(np.max(np.diff(values, axis=1))), 1e-10)
                self.assertFalse(result.diagnostics["one_sided_stock_nodes"])

                # Reconstruct D^alpha V and L V directly, without the solver's
                # rearranged matrix/history. This checks -r*V, dividend drift,
                # and memory through previous exercise and continuation levels.
                dt = problem.T / grid.time_steps
                stock = result.stock_grid[1:-1]
                for step in range(1, grid.time_steps + 1):
                    lag = np.arange(step, dtype=float)
                    weights = (lag + 1) ** (1 - problem.alpha) - lag ** (1 - problem.alpha)
                    caputo = weights[::-1] @ np.diff(values[: step + 1, 1:-1], axis=0)
                    caputo /= math.gamma(2 - problem.alpha) * dt**problem.alpha
                    delta = (values[step, 2:] - values[step, :-2]) / (2 * grid.dS)
                    gamma = np.diff(values[step], n=2) / grid.dS**2
                    spatial = (
                        0.5 * problem.sigma**2 * stock**2 * gamma
                        + (problem.r - problem.q) * stock * delta
                        - problem.r * values[step, 1:-1]
                    )
                    residual = caputo - spatial
                    gap = values[step, 1:-1] - payoff[1:-1]
                    self.assertGreaterEqual(float(np.min(residual)), -1e-9)
                    self.assertLess(float(np.max(np.abs(gap * residual))), 1e-8)
                    exercise = result.stock_grid <= result.exercise_boundary[step]
                    np.testing.assert_array_equal(values[step, exercise], payoff[exercise])
                    continuation = stock > result.exercise_boundary[step]
                    np.testing.assert_allclose(residual[continuation], 0.0, atol=1e-9)

    def test_no_early_exercise_case_matches_inverse_stable_benchmark(self) -> None:
        problem = AmericanPutProblem(100, 100, 1, 0, 0.3, 0.6)
        european = EuropeanOptionProblem(100, 100, 1, 0, 0.3, 0.6, "put")
        reference = subordinated_bsm_price(european, order=256)
        result = solve_zhou_gao(problem, StockGridSpec(400, 400, 200))
        self.assertAlmostEqual(result.price, reference, delta=0.02)
        np.testing.assert_array_equal(result.exercise_boundary[1:], 0.0)

    def test_fractional_put_has_early_exercise_premium_and_dividend_response(self) -> None:
        problem = AmericanPutProblem(100, 100, 1, 0.05, 0.3, 0.7)
        european = EuropeanOptionProblem(100, 100, 1, 0.05, 0.3, 0.7, "put")
        grid = StockGridSpec(400, 240, 120)
        result = solve_zhou_gao(problem, grid)
        dividend_result = solve_zhou_gao(replace(problem, q=0.04), grid)
        self.assertGreater(result.price, subordinated_bsm_price(european, order=256) + 0.1)
        self.assertGreater(dividend_result.price, result.price)
        self.assertLess(dividend_result.exercise_boundary[-1], result.exercise_boundary[-1])
        self.assertAlmostEqual(
            solve_zhou_gao(replace(problem, S0=0), grid).price, problem.K
        )

    def test_incompatible_contracts_and_domains_are_rejected(self) -> None:
        problem = AmericanPutProblem(100, 100, 1, 0.05, 0.2, 0.8)
        for change in ({"r": -0.01}, {"q": -0.01}, {"alpha": 0.0}, {"sigma": math.nan}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(problem, **change)
        with self.assertRaisesRegex(ValueError, "exceed K"):
            solve_zhou_gao(problem, StockGridSpec(100, 20, 10))
        with self.assertRaisesRegex(ValueError, "S0"):
            solve_zhou_gao(replace(problem, S0=500), StockGridSpec(400, 20, 10))
        with self.assertRaises(ValueError):
            StockGridSpec(400, 2.5, 10)


if __name__ == "__main__":
    unittest.main()
