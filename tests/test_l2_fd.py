from __future__ import annotations

import unittest

import numpy as np

from tfbsm_pricing.grid import GridSpec
from tfbsm_pricing.l2_fd import l2_coefficients, solve_l2_pde

from tests._support import (
    manufactured_boundary,
    manufactured_exact,
    manufactured_problem,
    manufactured_source,
)


class L2MethodTests(unittest.TestCase):
    def test_l2_weights_reduce_to_bdf2_at_alpha_one(self) -> None:
        a_weight, b_weight, c_weight = l2_coefficients(1.0, 5)
        np.testing.assert_allclose(a_weight[1:], 0.0)
        np.testing.assert_allclose(b_weight[1:], 0.0)
        np.testing.assert_allclose(c_weight[1:], 0.0)

    def test_an_manufactured_solution_reproduces_first_table_entry(self) -> None:
        alpha = 0.1
        grid = GridSpec(0.0, 1.0, 29, 10)
        result = solve_l2_pde(
            manufactured_problem(alpha),
            grid,
            initial=lambda x: manufactured_exact(x, 0.0),
            boundary=manufactured_boundary,
            source=lambda x, t: manufactured_source(x, t, alpha),
        )
        error = np.max(np.abs(result.grid_price[-1] - manufactured_exact(result.x_grid, 1.0)))
        # An et al. Table 1 reports 2.5543e-4 for this grid.
        self.assertAlmostEqual(float(error), 2.5543e-4, delta=2.0e-8)


if __name__ == "__main__":
    unittest.main()
