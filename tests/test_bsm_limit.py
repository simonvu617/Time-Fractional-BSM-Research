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


class BlackScholesLimitTests(unittest.TestCase):
    def test_both_solvers_recover_bsm_for_calls_and_puts(self) -> None:
        grid = GridSpec(math.log(5.0), math.log(500.0), 240, 200)
        for option_type in ("call", "put"):
            with self.subTest(option_type=option_type):
                problem = EuropeanOptionProblem(
                    100.0, 105.0, 0.75, 0.03, 0.25, 1.0, option_type
                )
                exact = black_scholes_price(problem)
                self.assertLess(abs(solve_weighted(problem, grid).price - exact), 0.03)
                self.assertLess(abs(solve_l2(problem, grid).price - exact), 0.03)

    def test_classical_put_call_parity(self) -> None:
        call = EuropeanOptionProblem(100.0, 105.0, 0.75, 0.03, 0.25, 1.0, "call")
        put = EuropeanOptionProblem(100.0, 105.0, 0.75, 0.03, 0.25, 1.0, "put")
        parity = 100.0 - 105.0 * math.exp(-0.03 * 0.75)
        self.assertAlmostEqual(
            black_scholes_price(call) - black_scholes_price(put), parity, places=12
        )


if __name__ == "__main__":
    unittest.main()
