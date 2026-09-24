"""Print the reproducible numerical-validation tables used in the documentation."""

from __future__ import annotations

import math
import time

import numpy as np

from tfbsm_pricing import (
    EuropeanOptionProblem,
    GridSpec,
    black_scholes_price,
    solve_l2,
    solve_weighted,
)
from tfbsm_pricing.l2_fd import solve_l2_pde

from tests._support import (
    manufactured_boundary,
    manufactured_exact,
    manufactured_problem,
    manufactured_source,
)


def main() -> None:
    print("## An et al. manufactured-solution reproduction")
    print("| alpha | dt | space steps | paper max error | computed max error |")
    print("|---:|---:|---:|---:|---:|")
    published = {
        0.1: [(10, 29, 2.5543e-4), (20, 78, 3.5313e-5), (40, 211, 4.8264e-6)],
        0.5: [(10, 18, 6.5066e-4), (20, 43, 1.1428e-4), (40, 101, 2.0731e-5)],
        0.9: [(10, 12, 1.5e-3), (20, 24, 3.6555e-4), (40, 49, 8.7868e-5)],
    }
    for alpha, rows in published.items():
        for time_steps, space_steps, paper_error in rows:
            grid = GridSpec(0.0, 1.0, space_steps, time_steps)
            result = solve_l2_pde(
                manufactured_problem(alpha),
                grid,
                initial=lambda x: manufactured_exact(x, 0.0),
                boundary=manufactured_boundary,
                source=lambda x, t, a=alpha: manufactured_source(x, t, a),
            )
            error = float(
                np.max(
                    np.abs(
                        result.grid_price[-1]
                        - manufactured_exact(result.x_grid, 1.0)
                    )
                )
            )
            print(
                f"| {alpha:.1f} | 1/{time_steps} | {space_steps} | "
                f"{paper_error:.7g} | {error:.7g} |"
            )

    print("\n## Krzyzanowski et al. Example 2 representative grid")
    print("| theta | paper percent error | computed price | computed percent error |")
    print("|---:|---:|---:|---:|")
    example = EuropeanOptionProblem(1.0, 2.0, 4.0, 0.04, 1.0, 0.999, "call")
    example_grid = GridSpec(-20.0, 10.0, 500, 50)
    for theta, paper_error in ((0.0, 1.74), (0.25, 1.12), (0.5, 0.61)):
        price = solve_weighted(example, example_grid, theta=theta).price
        error = 100.0 * abs(price - 0.593) / 0.593
        print(f"| {theta:.2f} | {paper_error:.2f}% | {price:.9f} | {error:.3f}% |")

    print("\n## Krzyzanowski et al. Table 1 temporal-order reproduction")
    print("| alpha | paper order | computed order | theoretical 2-alpha |")
    print("|---:|---:|---:|---:|")
    for alpha, dt, paper_order in (
        (0.99, 0.01, 1.02),
        (0.7, 2.22e-3, 1.32),
        (0.5, 1.67e-3, 1.51),
        (0.3, 1.39e-3, 1.70),
        (0.1, 1.25e-3, 1.85),
    ):
        problem = EuropeanOptionProblem(1.0, 2.0, 1.0, 0.04, 1.0, alpha, "call")
        reference = solve_weighted(problem, GridSpec(-1.0, 1.0, 10, 2597), theta=0.0)
        coarse = solve_weighted(
            problem, GridSpec(-1.0, 1.0, 10, round(1.0 / dt)), theta=0.0
        )
        fine = solve_weighted(
            problem, GridSpec(-1.0, 1.0, 10, round(2.0 / dt)), theta=0.0
        )
        evaluate = lambda result: float(
            np.interp(-0.01, result.x_grid, result.grid_price[-1])
        )
        order = math.log2(
            abs(
                (evaluate(coarse) - evaluate(reference))
                / (evaluate(fine) - evaluate(reference))
            )
        )
        print(f"| {alpha:.2f} | {paper_order:.2f} | {order:.3f} | {2-alpha:.2f} |")

    print("\n## alpha=1 analytic BSM convergence")
    print("| intervals (space,time) | weighted error | L2 error |")
    print("|---:|---:|---:|")
    classical = EuropeanOptionProblem(100.0, 100.0, 1.0, 0.05, 0.2, 1.0, "call")
    exact = black_scholes_price(classical)
    for count in (80, 160, 320):
        grid = GridSpec(math.log(5.0), math.log(500.0), count, count)
        weighted_error = abs(solve_weighted(classical, grid).price - exact)
        l2_error = abs(solve_l2(classical, grid).price - exact)
        print(f"| ({count},{count}) | {weighted_error:.7g} | {l2_error:.7g} |")

    print("\n## Cross-solver agreement under refinement")
    print("| alpha | intervals | weighted price | L2 price | absolute difference |")
    print("|---:|---:|---:|---:|---:|")
    for alpha in (0.5, 0.9):
        problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.04, 0.3, alpha, "call")
        for count in (60, 120, 240):
            comparison_grid = GridSpec(-5.0, 5.0, count, count)
            weighted = solve_weighted(problem, comparison_grid).price
            l2 = solve_l2(problem, comparison_grid).price
            print(
                f"| {alpha:.2f} | {count} | {weighted:.9f} | {l2:.9f} | "
                f"{abs(weighted-l2):.3g} |"
            )

    print("\n## Fractional put-call-parity diagnostic")
    print("| alpha | solver | numerical C-P | paper parity | residual |")
    print("|---:|:---|---:|---:|---:|")
    parity_grid = GridSpec(-5.0, 5.0, 240, 180)
    for alpha in (0.5, 0.9, 1.0):
        call = EuropeanOptionProblem(1.0, 1.1, 1.0, 0.04, 0.3, alpha, "call")
        put = EuropeanOptionProblem(1.0, 1.1, 1.0, 0.04, 0.3, alpha, "put")
        claimed = call.S0 - call.K * math.exp(-call.r * call.T)
        for name, solver in (("weighted", solve_weighted), ("L2", solve_l2)):
            difference = solver(call, parity_grid).price - solver(put, parity_grid).price
            print(
                f"| {alpha:.1f} | {name} | {difference:.9f} | "
                f"{claimed:.9f} | {difference-claimed:.3g} |"
            )

    print("\n## Timing smoke test")
    timing_problem = EuropeanOptionProblem(1.0, 1.0, 1.0, 0.04, 0.3, 0.7, "call")
    timing_grid = GridSpec(-5.0, 5.0, 400, 300)
    for name, solver in (("weighted", solve_weighted), ("L2", solve_l2)):
        started = time.perf_counter()
        solver(timing_problem, timing_grid)
        print(f"{name}: {time.perf_counter()-started:.3f} seconds")


if __name__ == "__main__":
    main()
