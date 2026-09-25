"""Weighted L1 solver reproduced from KMP20.

KMP20: G. Krzyzanowski, M. Magdziarz, and L. Plociniczak, "A weighted
finite difference method for subdiffusive Black-Scholes model," Computers &
Mathematics with Applications 80(5), 653-670 (2020).
DOI: https://doi.org/10.1016/j.camwa.2020.04.029
Open manuscript: https://arxiv.org/abs/1907.00297v4

Equation and page locators below refer to the publisher PDF.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .model import (
    EuropeanOptionProblem,
    GridSpec,
    SolverResult,
    TridiagonalFactor,
    boundary_values,
    payoff,
)


def weighted_caputo_weights(alpha: float, count: int) -> NDArray[np.float64]:
    """Return ``b_j`` from KMP20 equation (7), PDF page 6."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if count < 1:
        raise ValueError("count must be positive")
    # The analytic alpha -> 1 limit is b_0=1 and b_j=0 for j>0.  Directly
    # evaluating j^(1-alpha) would produce the indeterminate 0^0 at j=0.
    if alpha == 1.0:
        weights = np.zeros(count, dtype=float)
        weights[0] = 1.0
        return weights
    indices = np.arange(count, dtype=float)
    return (indices + 1.0) ** (1.0 - alpha) - indices ** (1.0 - alpha)


def optimal_weight(alpha: float) -> float:
    """Return KMP20's optimal stable weight after Theorem 3.3, PDF page 16."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    power = 2.0 ** (1.0 - alpha)
    return (2.0 - power) / (3.0 - power)


def paper_unconditional_stability_threshold(theta: float) -> float:
    """Return the minimum alpha in KMP20 Theorem 3.2(i), PDF page 10.

    The typeset condition is ``1-log2(2-theta/(1-theta)) <= alpha``.
    For ``theta >= 2/3`` its logarithm is not positive, so the theorem gives no
    unconditional fractional-alpha region.
    """

    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    if theta >= 2.0 / 3.0:
        return math.inf
    return 1.0 - math.log2(2.0 - theta / (1.0 - theta))


def solve_weighted(
    problem: EuropeanOptionProblem,
    grid: GridSpec,
    *,
    theta: float | None = None,
) -> SolverResult:
    """Price an option with KMP20 equations (7)-(11), PDF pages 6-7.

    The paper defines ``theta=0`` as fully implicit and ``theta=1`` as fully
    explicit.  When omitted, ``theta`` is their alpha-dependent optimal stable
    weight.  The complete surface is retained because every fractional time
    step depends on all preceding levels.
    """

    if theta is None:
        theta = optimal_weight(problem.alpha)
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    log_spot = math.log(problem.S0)
    if not grid.x_min <= log_spot <= grid.x_max:
        raise ValueError("log(S0) must lie inside the spatial domain")

    x = grid.x_values()
    times = grid.time_values(problem.T)
    dt = grid.dt(problem.T)
    dx = grid.dx
    interior_size = grid.space_steps - 1

    surface = np.empty((grid.time_steps + 1, grid.space_steps + 1), dtype=float)
    surface[0] = payoff(problem, x)
    for step, elapsed in enumerate(times):
        left, right = boundary_values(problem, grid.x_min, grid.x_max, elapsed)
        surface[step, 0] = left
        surface[step, -1] = right

    # KMP20 equations (9)-(11), PDF page 7: after x=log(S), centered
    # differences of L u = a*u_xx + b*u_x - r*u give these coefficients.
    lower_l = problem.diffusion / dx**2 - problem.drift / (2.0 * dx)
    diagonal_l = -2.0 * problem.diffusion / dx**2 - problem.r
    upper_l = problem.diffusion / dx**2 + problem.drift / (2.0 * dx)

    scale = math.gamma(2.0 - problem.alpha) * dt**problem.alpha
    lower = np.full(
        max(interior_size - 1, 0), -(1.0 - theta) * scale * lower_l
    )
    diagonal = np.full(
        interior_size, 1.0 - (1.0 - theta) * scale * diagonal_l
    )
    upper = np.full(
        max(interior_size - 1, 0), -(1.0 - theta) * scale * upper_l
    )
    factor = TridiagonalFactor.factor(lower, diagonal, upper)
    weights = weighted_caputo_weights(problem.alpha, grid.time_steps)

    for step in range(1, grid.time_steps + 1):
        if step == 1:
            history = surface[0, 1:-1].copy()
        else:
            # KMP20 equation (11), PDF page 7: successive differences of b_j
            # multiply prior levels newest-first; the final b term multiplies
            # the payoff level.
            history_coefficients = weights[: step - 1] - weights[1:step]
            prior_levels = surface[step - 1 : 0 : -1, 1:-1]
            history = history_coefficients @ prior_levels
            history += weights[step - 1] * surface[0, 1:-1]

        previous = surface[step - 1]
        previous_l = (
            lower_l * previous[:-2]
            + diagonal_l * previous[1:-1]
            + upper_l * previous[2:]
        )
        rhs = history + theta * scale * previous_l

        # KMP20 equations (8) and (11), PDF page 7: current boundaries from
        # the implicit operator move to the right-hand side. Previous
        # boundaries are already included in previous_l.
        rhs[0] += (1.0 - theta) * scale * lower_l * surface[step, 0]
        rhs[-1] += (1.0 - theta) * scale * upper_l * surface[step, -1]
        surface[step, 1:-1] = factor.solve(rhs)

    price = float(np.interp(log_spot, x, surface[-1]))
    threshold = paper_unconditional_stability_threshold(theta)
    on_unconditional_side = problem.alpha >= threshold - 16.0 * np.finfo(float).eps
    return SolverResult(
        price=price,
        x_grid=x,
        time_grid=times,
        grid_price=surface,
        diagnostics={
            "solver": "weighted_l1",
            "paper": "KMP20",
            "paper_doi": "10.1016/j.camwa.2020.04.029",
            "paper_equations": "(7)-(11)",
            "theta": theta,
            "dt": dt,
            "dx": dx,
            "paper_unconditional_stability_threshold": threshold,
            "paper_unconditional_stability_condition": on_unconditional_side,
            "time_order_claim": 2.0 - problem.alpha,
            "space_order_claim": 2.0,
        },
    )
