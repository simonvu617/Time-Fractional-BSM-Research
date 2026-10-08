"""American put solver based on Zhou and Gao (2016), Section 4.

Zhiqiang Zhou and Xuemei Gao, "Numerical Methods for Pricing American Options
with Time-Fractional PDE Models," Mathematical Problems in Engineering,
vol. 2016, Article 5614950, pp. 1-8 (2016).
DOI: https://doi.org/10.1155/2016/5614950
PDF: https://pdfs.semanticscholar.org/294d/bb2b5113e9346fd8e10b7fbeab615ffa75b3.pdf

The implementation uses equations (10)-(14), (34)-(39), and boundary-search
Step 2.1, with beta=0 and delta=sigma. See docs/american_solver.md for the
discount-term correction and the mesh-boundary approximation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .american_model import AmericanPutProblem, AmericanPutResult, StockGridSpec
from .model import TridiagonalFactor


def solve_zhou_gao(
    problem: AmericanPutProblem, grid: StockGridSpec
) -> AmericanPutResult:
    """Solve the Caputo American-put free-boundary PDE with dividend yield q.

    Each time step solves for the continuation region and searches for the
    exercise boundary; it does not merely clip a European solution to payoff.
    The full exercise/continuation history enters every fractional step.
    """

    if grid.S_max <= problem.K:
        raise ValueError("S_max must exceed K for the zero far-field boundary")
    if problem.S0 > grid.S_max:
        raise ValueError("S0 must lie inside [0, S_max]")

    # ZG16 equation (34), PDF page 5: stock space, rather than log stock.
    stock = np.linspace(0.0, grid.S_max, grid.space_steps + 1)
    times = np.linspace(0.0, problem.T, grid.time_steps + 1)
    dt = problem.T / grid.time_steps
    intrinsic = np.maximum(problem.K - stock, 0.0)
    surface = np.empty((times.size, stock.size), dtype=float)
    surface[0] = intrinsic
    boundaries = np.empty(times.size, dtype=float)
    boundaries[0] = problem.K

    # ZG16 (35)-(36), PDF page 5, normalized by b_n^(n). The alpha=1
    # limit is backward Euler; it avoids the undefined 0^0 in the weights.
    scale = math.gamma(2.0 - problem.alpha) * dt**problem.alpha
    weights = np.zeros(grid.time_steps, dtype=float)
    weights[0] = 1.0
    if problem.alpha < 1.0:
        lag = np.arange(1, grid.time_steps, dtype=float)
        # Stable evaluation of (lag+1)^(1-alpha) - lag^(1-alpha).
        weights[1:] = lag ** (1.0 - problem.alpha) * np.expm1(
            (1.0 - problem.alpha) * np.log1p(1.0 / lag)
        )

    interior_stock = stock[1:-1]
    diffusion = 0.5 * problem.sigma**2 * (interior_stock / grid.dS) ** 2
    drift = (problem.r - problem.q) * interior_stock / grid.dS
    lower_l = diffusion - 0.5 * drift
    upper_l = diffusion + 0.5 * drift
    # Central differences from (37). Near S=0, drift can make a coefficient
    # negative regardless of grid refinement. Use a one-sided derivative
    # only on those rows to preserve the discrete comparison principle.
    one_sided = (lower_l < 0.0) | (upper_l < 0.0)
    lower_l[one_sided] = diffusion[one_sided] + np.maximum(-drift[one_sided], 0.0)
    upper_l[one_sided] = diffusion[one_sided] + np.maximum(drift[one_sided], 0.0)

    # ZG16 (38)-(39), PDF page 5, divided by b_n^(n). Equation (39) prints
    # c_j without r*dt; retain it here to discretize the -r*V in (10).
    lower = -scale * lower_l
    diagonal = 1.0 + scale * (lower_l + upper_l + problem.r)
    upper = -scale * upper_l

    # Step 1, (40): first node at or above the strike. Later entries are
    # the last exercise node, i.e. the upper endpoint of Step 2.1's bracket.
    boundary_index = int(np.searchsorted(stock, problem.K))
    roundoff = 64.0 * np.finfo(float).eps * max(1.0, problem.K)
    solve_count = 0
    min_gap = 0.0
    min_residual = 0.0
    max_complementarity = 0.0

    for step in range(1, times.size):
        # Equation (39): history of increments at each fixed stock node,
        # including levels where that stock was in the exercise region.
        history = surface[step - 1, 1:-1].copy()
        if problem.alpha < 1.0 and step > 1:
            increments = (
                surface[step - 1 : 0 : -1, 1:-1]
                - surface[step - 2 :: -1, 1:-1]
            )
            history -= weights[1:step] @ increments

        # ZG16 boundary-search Step 2.1, PDF page 5. At a trial boundary i,
        # (V[i+1]-V[i])/dS < -1 is equivalent to V[i+1] < K-S[i+1].
        # Keep the adjacent feasible solve at i+1, rather than retaining
        # the trial surface that fell below payoff. Step 2.2 is optional in
        # the paper; this implementation reports a mesh-level boundary.
        feasible = None
        for candidate in range(boundary_index - 1, -1, -1):
            trial = _continuation_values(candidate, intrinsic, history, lower, diagonal, upper)
            solve_count += 1
            if trial[candidate + 1] < intrinsic[candidate + 1] - roundoff:
                boundary_index = candidate + 1
                if feasible is None:
                    feasible = _continuation_values(
                        boundary_index, intrinsic, history, lower, diagonal, upper
                    )
                    solve_count += 1
                break
            feasible = trial
        else:
            boundary_index = 0
            if feasible is None:
                feasible = _continuation_values(0, intrinsic, history, lower, diagonal, upper)
                solve_count += 1

        surface[step] = feasible
        boundaries[step] = stock[boundary_index]
        gap = feasible[1:-1] - intrinsic[1:-1]
        residual = (
            lower * feasible[:-2]
            + diagonal * feasible[1:-1]
            + upper * feasible[2:]
            - history
        )
        min_gap = min(min_gap, float(np.min(gap)))
        min_residual = min(min_residual, float(np.min(residual)))
        max_complementarity = max(
            max_complementarity, float(np.max(np.abs(np.minimum(gap, residual))))
        )

    return AmericanPutResult(
        price=float(np.interp(problem.S0, stock, surface[-1])),
        stock_grid=stock,
        time_grid=times,
        grid_price=surface,
        exercise_boundary=boundaries,
        diagnostics={
            "solver": "zhou_gao_l1_american_put",
            "paper": "ZG16",
            "paper_doi": "10.1155/2016/5614950",
            "paper_equations": "(10)-(14), (34)-(40), boundary-search Step 2.1",
            "beta": 0.0,
            "dt": dt,
            "dS": grid.dS,
            "boundary_resolution": grid.dS,
            "boundary_refinement": "mesh only; optional Step 2.2 not used",
            "discount_term_correction": True,
            "one_sided_stock_nodes": tuple(int(i + 1) for i in np.flatnonzero(one_sided)),
            "continuation_solves": solve_count,
            "min_obstacle_gap": min_gap,
            "min_scaled_pde_residual": min_residual,
            "max_scaled_complementarity_residual": max_complementarity,
        },
    )


def _continuation_values(
    boundary: int,
    intrinsic: NDArray[np.float64],
    history: NDArray[np.float64],
    lower: NDArray[np.float64],
    diagonal: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve (38) above a proposed exercise node with value matching."""

    values = intrinsic.copy()
    if boundary < history.size:
        rhs = history[boundary:].copy()
        rhs[0] -= lower[boundary] * intrinsic[boundary]
        factor = TridiagonalFactor.factor(
            lower[boundary + 1 :], diagonal[boundary:], upper[boundary:-1]
        )
        values[boundary + 1 : -1] = factor.solve(rhs)
    return values
