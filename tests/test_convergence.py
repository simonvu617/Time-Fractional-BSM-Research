from __future__ import annotations

import math
import unittest

from tfbsm_pricing import (
    EuropeanOptionProblem,
    GridSpec,
    black_scholes_price,
    solve_l2,
    solve_weighted,
)


class ConvergenceTests(unittest.TestCase):
    def test_refinement_reduces_classical_limit_error(self) -> None:
        problem = EuropeanOptionProblem(100.0, 100.0, 1.0, 0.05, 0.2, 1.0, "call")
        exact = black_scholes_price(problem)
        coarse = GridSpec(math.log(5.0), math.log(500.0), 80, 50)
        fine = GridSpec(math.log(5.0), math.log(500.0), 160, 100)
        for solver in (solve_weighted, solve_l2):
            with self.subTest(solver=solver.__name__):
                coarse_error = abs(solver(problem, coarse).price - exact)
                fine_error = abs(solver(problem, fine).price - exact)
                self.assertLess(fine_error, coarse_error)


if __name__ == "__main__":
    unittest.main()
