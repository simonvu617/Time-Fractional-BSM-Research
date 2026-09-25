from __future__ import annotations

import math
import unittest

import numpy as np

from tfbsm_pricing import (
    black_scholes_price,
    refine_price,
    solve_l2,
    solve_weighted,
)
from tfbsm_pricing.l2_solver import l2_coefficients
from tfbsm_pricing.model import (
    EuropeanOptionProblem,
    GridSpec,
    SolverResult,
    boundary_values,
    mittag_leffler_discount,
    payoff,
)
from validation.subordination import (
    inverse_stable_moment,
    inverse_stable_quadrature,
    subordinated_bsm_price,
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
        self.assertAlmostEqual(
            right, 400.0 - 100.0 * mittag_leffler_discount(0.7, 0.05, 1.0)
        )

        paper_problem = EuropeanOptionProblem(
            100.0,
            100.0,
            1.0,
            0.05,
            0.2,
            0.7,
            "call",
            boundary_mode="paper_reproduction",
        )
        _, paper_right = boundary_values(
            paper_problem, math.log(1.0), math.log(400.0), 1.0
        )
        self.assertAlmostEqual(paper_right, 400.0 - 100.0 * math.exp(-0.05))

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

    def test_mittag_leffler_discount_special_cases(self) -> None:
        self.assertEqual(mittag_leffler_discount(0.7, 0.05, 0.0), 1.0)
        self.assertAlmostEqual(
            mittag_leffler_discount(1.0, 0.05, 2.0), math.exp(-0.1)
        )
        argument = 0.4
        expected = math.exp(argument**2) * math.erfc(argument)
        self.assertAlmostEqual(
            mittag_leffler_discount(0.5, argument, 1.0), expected, places=13
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            mittag_leffler_discount(0.5, 1.0, 1.0)

    def test_inverse_stable_quadrature_moments(self) -> None:
        for alpha in (0.1, 0.5, 0.9, 0.9999):
            expected = 1.0 / math.gamma(1.0 + alpha)
            self.assertAlmostEqual(
                inverse_stable_moment(alpha, 1.0, 1.0, order=256),
                expected,
                delta=1e-5,
            )

    def test_canonical_discount_matches_clock_expectation(self) -> None:
        for alpha in (0.1, 0.5, 0.9, 0.9999):
            clock, weights = inverse_stable_quadrature(alpha, 1.0, order=256)
            expected = float(weights @ np.exp(-0.05 * clock))
            self.assertAlmostEqual(
                mittag_leffler_discount(alpha, 0.05, 1.0), expected, delta=5e-7
            )

    def test_subordination_quadrature_converges_independently(self) -> None:
        cases = (
            EuropeanOptionProblem(1.0, 1.0, 0.25, 0.0, 0.3, 0.1, "call"),
            EuropeanOptionProblem(
                1.0, math.exp(0.5), 2.0, 0.03, 0.4, 0.3, "put"
            ),
            EuropeanOptionProblem(1.0, 1.0, 1.0, 0.03, 0.3, 0.9999, "put"),
        )
        for problem in cases:
            values = [
                subordinated_bsm_price(problem, order)
                for order in (32, 64, 128, 256, 512)
            ]
            changes = np.abs(np.diff(values))
            with self.subTest(alpha=problem.alpha, option=problem.option_type):
                self.assertTrue(np.all(changes[1:] < changes[:-1]))
                self.assertLess(changes[2], 1e-6)
                self.assertLess(changes[3], 1e-7)

    def test_both_solvers_converge_to_subordination_across_alpha_range(self) -> None:
        cases = (
            (0.1, 1.0, 0.25, 0.0, 0.3, "call"),
            (0.3, math.exp(0.5), 2.0, 0.03, 0.4, "put"),
            (0.5, math.exp(0.5), 1.0, 0.0, 0.3, "call"),
            (0.9, math.exp(-0.5), 1.0, 0.03, 0.35, "put"),
            (0.99, math.exp(-0.5), 2.0, 0.04, 0.45, "call"),
            (0.9999, 1.0, 1.0, 0.03, 0.3, "put"),
        )
        for alpha, strike, maturity, rate, sigma, option_type in cases:
            problem = EuropeanOptionProblem(
                1.0, strike, maturity, rate, sigma, alpha, option_type
            )
            benchmark = subordinated_bsm_price(problem, order=512)
            for solver in (solve_weighted, solve_l2):
                errors = [
                    abs(
                        solver(problem, GridSpec(-4.0, 4.0, count, count)).price
                        - benchmark
                    )
                    for count in (80, 160, 320, 640)
                ]
                with self.subTest(alpha=alpha, solver=solver.__name__):
                    self.assertTrue(np.all(np.diff(errors) < 0.0))
                    self.assertLess(errors[-1], errors[0] / 10.0)

    def test_short_maturity_solvers_converge_to_independent_benchmark(self) -> None:
        cases = (
            EuropeanOptionProblem(1.0, 1.0, 7 / 365, 0.03, 0.2, 0.1, "call"),
            EuropeanOptionProblem(
                1.0, math.exp(0.0625), 14 / 365, 0.03, 0.6, 0.5, "put"
            ),
            EuropeanOptionProblem(
                1.0, math.exp(0.0625), 7 / 365, 0.03, 1.0, 0.9, "call"
            ),
            EuropeanOptionProblem(
                1.0,
                math.exp(-0.0625),
                14 / 365,
                0.03,
                0.6,
                0.9999,
                "put",
            ),
        )
        for problem in cases:
            benchmark_values = [
                subordinated_bsm_price(problem, order)
                for order in (64, 128, 256, 512)
            ]
            benchmark_changes = np.abs(np.diff(benchmark_values))
            self.assertTrue(np.all(np.diff(benchmark_changes) < 0.0))
            benchmark = benchmark_values[-1]
            for solver in (solve_weighted, solve_l2):
                errors = [
                    abs(
                        solver(problem, GridSpec(-2.0, 2.0, count, count)).price
                        - benchmark
                    )
                    for count in (64, 128, 256, 512)
                ]
                with self.subTest(
                    days=round(365 * problem.T),
                    alpha=problem.alpha,
                    option=problem.option_type,
                    solver=solver.__name__,
                ):
                    self.assertTrue(np.all(np.diff(errors) < 0.0))
                    self.assertLess(errors[-1], errors[0] / 20.0)
                    self.assertLess(benchmark_changes[-1], errors[-1] / 100.0)

    def test_l2_remains_finite_and_convergent_near_alpha_one(self) -> None:
        bsm_problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.03, 0.3, 1.0, "call")
        bsm_price = black_scholes_price(bsm_problem)
        benchmark_distances = []
        for alpha in (0.99, 0.999, 0.9999, 1.0):
            coefficients = np.concatenate(l2_coefficients(alpha, 800))
            self.assertTrue(np.all(np.isfinite(coefficients[~np.isnan(coefficients)])))
            problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.03, 0.3, alpha, "call")
            benchmark = subordinated_bsm_price(problem, order=512)
            benchmark_distances.append(abs(benchmark - bsm_price))
            result = solve_l2(problem, GridSpec(-4.0, 4.0, 160, 160))
            self.assertTrue(np.isfinite(result.price))
            self.assertTrue(np.all(np.isfinite(result.grid_price)))
        self.assertTrue(np.all(np.diff(benchmark_distances) < 0.0))

        problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.03, 0.3, 0.999, "call")
        benchmark = subordinated_bsm_price(problem, order=512)
        for solver in (solve_weighted, solve_l2):
            errors = [
                abs(solver(problem, GridSpec(-4.0, 4.0, n, n)).price - benchmark)
                for n in (80, 160, 320, 640)
            ]
            self.assertTrue(np.all(np.diff(errors) < 0.0))

    def test_separate_time_space_and_joint_refinement(self) -> None:
        problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.0, 0.3, 0.5, "call")
        benchmark = subordinated_bsm_price(problem, order=512)
        grids = {
            "time": [GridSpec(-4.0, 4.0, 640, n) for n in (80, 160, 320)],
            "space": [GridSpec(-4.0, 4.0, n, 640) for n in (80, 160, 320)],
            "joint": [GridSpec(-4.0, 4.0, n, n) for n in (80, 160, 320)],
        }
        for solver in (solve_weighted, solve_l2):
            for mode, sequence in grids.items():
                errors = [abs(solver(problem, grid).price - benchmark) for grid in sequence]
                with self.subTest(solver=solver.__name__, mode=mode):
                    self.assertTrue(np.all(np.diff(errors) < 0.0))

    def test_refinement_helper_reports_empirical_error(self) -> None:
        problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.0, 0.3, 0.5, "call")
        initial = GridSpec(-4.0, 4.0, 40, 40)
        for solver in (solve_weighted, solve_l2):
            result = refine_price(
                solver, problem, initial, tolerance=5e-4, max_refinements=3
            )
            refinement = result.diagnostics["grid_refinement"]
            self.assertTrue(
                {
                    "grid_levels",
                    "prices",
                    "successive_differences",
                    "observed_convergence_order",
                    "estimated_remaining_discretization_error",
                    "requested_tolerance",
                    "converged",
                    "fixed_domain",
                    "finite_domain_error_included",
                }
                <= refinement.keys()
            )
            self.assertTrue(refinement["converged"])
            self.assertEqual(refinement["refinements"], 3)
            self.assertEqual(len(refinement["grid_levels"]), 4)
            self.assertEqual(result.price, refinement["prices"][-1])
            self.assertTrue(np.all(np.diff(refinement["successive_differences"]) < 0.0))
            self.assertGreater(refinement["observed_convergence_order"], 0.0)
            self.assertLessEqual(
                refinement["estimated_remaining_discretization_error"], 5e-4
            )
            self.assertEqual(refinement["fixed_domain"], (-4.0, 4.0))
            self.assertFalse(refinement["finite_domain_error_included"])

        strict = refine_price(
            solve_l2, problem, initial, tolerance=1e-12, max_refinements=2
        )
        self.assertFalse(strict.diagnostics["grid_refinement"]["converged"])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            refine_price(solve_l2, problem, initial, max_refinements=1)

        prices = iter((0.0, 1.0, 3.0))

        def irregular_solver(
            unused_problem: EuropeanOptionProblem, grid: GridSpec
        ) -> SolverResult:
            price = next(prices)
            return SolverResult(
                price,
                grid.x_values(),
                grid.time_values(unused_problem.T),
                np.zeros((2, 2)),
                {},
            )

        irregular = refine_price(
            irregular_solver, problem, initial, max_refinements=2
        ).diagnostics["grid_refinement"]
        self.assertFalse(irregular["converged"])
        self.assertIsNone(irregular["observed_convergence_order"])
        self.assertIsNone(irregular["estimated_remaining_discretization_error"])

    def test_nonzero_rate_canonical_prices_match_subordination(self) -> None:
        grid = GridSpec(-6.0, 4.0, 400, 400)
        prices = {}
        for option_type in ("call", "put"):
            problem = EuropeanOptionProblem(
                1.0, 1.1, 1.0, 0.05, 0.3, 0.7, option_type
            )
            benchmark = subordinated_bsm_price(problem, order=128)
            weighted = solve_weighted(problem, grid).price
            l2 = solve_l2(problem, grid).price
            self.assertAlmostEqual(weighted, benchmark, delta=2e-4)
            self.assertAlmostEqual(l2, benchmark, delta=2e-4)
            prices[option_type] = weighted
        parity = 1.0 - 1.1 * mittag_leffler_discount(0.7, 0.05, 1.0)
        self.assertAlmostEqual(prices["call"] - prices["put"], parity, delta=1e-4)

    def test_prices_stabilize_as_spatial_domain_expands(self) -> None:
        cases = (
            ("atm_call_short", 1.0, 1.0, 0.1, 0.25, 0.5, "call"),
            ("itm_call_long_high", 1.2, 1.0, 2.0, 0.6, 0.9, "call"),
            ("otm_put_long", 1.2, 1.0, 1.5, 0.3, 0.5, "put"),
            ("itm_put_short_high", 0.8, 1.0, 0.25, 0.6, 0.9, "put"),
        )
        for name, spot, strike, maturity, sigma, alpha, option_type in cases:
            problem = EuropeanOptionProblem(
                spot, strike, maturity, 0.03, sigma, alpha, option_type
            )
            for solver in (solve_weighted, solve_l2):
                prices = []
                for half_width in (2.0, 3.0, 4.0):
                    grid = GridSpec(
                        math.log(spot) - half_width,
                        math.log(spot) + half_width,
                        round(2.0 * half_width / 0.025),
                        240,
                    )
                    prices.append(solver(problem, grid).price)
                with self.subTest(case=name, solver=solver.__name__):
                    self.assertLess(abs(prices[2] - prices[1]), 1e-6)


if __name__ == "__main__":
    unittest.main()
