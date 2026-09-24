from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.typing import NDArray

from tfbsm_pricing.l2_fd import l2_coefficients, solve_l2_pde
from tfbsm_pricing.model import EuropeanOptionProblem, GridSpec


def exact(x: NDArray[np.float64], elapsed: float) -> NDArray[np.float64]:
    return (elapsed + 1.0) ** 2 * (x**3 + x**2 + 1.0)


def boundary(elapsed: float) -> tuple[float, float]:
    values = exact(np.array([0.0, 1.0]), elapsed)
    return float(values[0]), float(values[1])


def source(
    x: NDArray[np.float64], elapsed: float, alpha: float
) -> NDArray[np.float64]:
    polynomial = x**3 + x**2 + 1.0
    caputo_time = (
        2.0 * elapsed ** (2.0 - alpha) / math.gamma(3.0 - alpha)
        + 2.0 * elapsed ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    )
    spatial = (6.0 * x + 2.0) - 0.5 * (3.0 * x**2 + 2.0 * x) - 0.5 * polynomial
    return caputo_time * polynomial - (elapsed + 1.0) ** 2 * spatial


class L2MethodTests(unittest.TestCase):
    def test_l2_weights_reduce_to_bdf2_at_alpha_one(self) -> None:
        a_weight, b_weight, c_weight = l2_coefficients(1.0, 5)
        np.testing.assert_allclose(a_weight[1:], 0.0)
        np.testing.assert_allclose(b_weight[1:], 0.0)
        np.testing.assert_allclose(c_weight[1:], 0.0)

    def test_an_manufactured_solution_reproduces_table_1(self) -> None:
        published = {
            0.1: (
                (10, 29, 2.5543e-4, 2e-8),
                (20, 78, 3.5313e-5, 2e-8),
                (40, 211, 4.8264e-6, 2e-8),
            ),
            0.5: (
                (10, 18, 6.5066e-4, 2e-8),
                (20, 43, 1.1428e-4, 2e-8),
                (40, 101, 2.0731e-5, 2e-8),
            ),
            # The paper prints the first alpha=0.9 error only as 0.0015.
            0.9: (
                (10, 12, 1.5e-3, 5e-5),
                (20, 24, 3.6555e-4, 2e-8),
                (40, 49, 8.7868e-5, 2e-8),
            ),
        }
        for alpha, rows in published.items():
            problem = EuropeanOptionProblem(
                math.exp(0.5), 1.0, 1.0, 0.5, math.sqrt(2.0), alpha, "call"
            )
            for time_steps, space_steps, paper_error, tolerance in rows:
                result = solve_l2_pde(
                    problem,
                    GridSpec(0.0, 1.0, space_steps, time_steps),
                    initial=lambda x: exact(x, 0.0),
                    boundary=boundary,
                    source=lambda x, t, a=alpha: source(x, t, a),
                )
                error = np.max(np.abs(result.grid_price[-1] - exact(result.x_grid, 1.0)))
                self.assertAlmostEqual(float(error), paper_error, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
