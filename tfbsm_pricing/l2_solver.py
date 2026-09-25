"""L2 solver reproduced from An24.

An24: X. An, Q. Wang, F. Liu, V. V. Anh, and I. W. Turner, "Parameter
estimation for time-fractional Black-Scholes equation with S&P 500 index
option," Numerical Algorithms 95, 1-30 (2024).
DOI and open article: https://doi.org/10.1007/s11075-023-01563-4

Equation and page locators below refer to the publisher PDF.
"""

from __future__ import annotations

import math
from typing import Callable

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


InitialFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
BoundaryFunction = Callable[[float], tuple[float, float]]
SourceFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


def l2_coefficients(
    alpha: float, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the independent ``a_i``, ``b_i``, and ``c_i`` L2 sequences.

    These are An24 equations (8)-(9), PDF page 6. Element zero is unused so
    array indices match the paper's one-based subscripts.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if count < 1:
        raise ValueError("count must be positive")

    index = np.arange(1, count + 1, dtype=float)
    current_1 = index ** (1.0 - alpha)
    next_1 = (index + 1.0) ** (1.0 - alpha)
    current_2 = index ** (2.0 - alpha)
    next_2 = (index + 1.0) ** (2.0 - alpha)
    a_values = (
        (2.0 - alpha) * (0.5 * current_1 - 1.5 * next_1)
        - current_2
        + next_2
    )
    b_values = (
        (4.0 - 2.0 * alpha) * next_1
        + 2.0 * current_2
        - 2.0 * next_2
    )
    c_values = (
        (0.5 * alpha - 1.0) * (current_1 + next_1)
        - current_2
        + next_2
    )
    padding = np.array([np.nan])
    return (
        np.concatenate((padding, a_values)),
        np.concatenate((padding, b_values)),
        np.concatenate((padding, c_values)),
    )


def solve_l2(
    problem: EuropeanOptionProblem,
    grid: GridSpec,
) -> SolverResult:
    """Price an option with An24 equations (7)-(17), PDF pages 6-7."""

    return solve_l2_pde(
        problem,
        grid,
        initial=lambda x: payoff(problem, x),
        boundary=lambda elapsed: boundary_values(
            problem, grid.x_min, grid.x_max, elapsed
        ),
    )


def solve_l2_pde(
    problem: EuropeanOptionProblem,
    grid: GridSpec,
    *,
    initial: InitialFunction,
    boundary: BoundaryFunction,
    source: SourceFunction | None = None,
) -> SolverResult:
    """Solve the paper's transformed PDE, optionally with a validation source.

    The callback form reproduces An24 Section 4.4's manufactured-solution
    experiment; the public option pricer supplies the payoff and boundaries.
    """

    log_spot = math.log(problem.S0)
    if not grid.x_min <= log_spot <= grid.x_max:
        raise ValueError("log(S0) must lie inside the spatial domain")

    x = grid.x_values()
    times = grid.time_values(problem.T)
    dt = grid.dt(problem.T)
    dx = grid.dx
    interior_x = x[1:-1]
    interior_size = grid.space_steps - 1

    surface = np.empty((grid.time_steps + 1, grid.space_steps + 1), dtype=float)
    initial_values = np.asarray(initial(x), dtype=float)
    if initial_values.shape != x.shape:
        raise ValueError("initial callback must return one value per grid point")
    surface[0] = initial_values
    for step, elapsed in enumerate(times):
        left, right = boundary(float(elapsed))
        surface[step, 0] = left
        surface[step, -1] = right

    # An24 equations (12)-(13), PDF page 7, give the centered log-price
    # operator. Its L2 weights and history recurrence remain independent of
    # the KMP20 solver.
    lower_l = problem.diffusion / dx**2 - problem.drift / (2.0 * dx)
    diagonal_l = -2.0 * problem.diffusion / dx**2 - problem.r
    upper_l = problem.diffusion / dx**2 + problem.drift / (2.0 * dx)

    # An24 equations (7) and (14), PDF pages 6-7: L1 starts the L2 method.
    phi_1 = math.gamma(2.0 - problem.alpha) * dt**problem.alpha
    first_factor = _implicit_factor(
        interior_size, phi_1, 1.0, lower_l, diagonal_l, upper_l
    )
    rhs = surface[0, 1:-1].copy()
    if source is not None:
        rhs += phi_1 * _source_values(source, interior_x, times[1])
    rhs[0] += phi_1 * lower_l * surface[1, 0]
    rhs[-1] += phi_1 * upper_l * surface[1, -1]
    surface[1, 1:-1] = first_factor.solve(rhs)

    if grid.time_steps >= 2:
        a_weight, b_weight, c_weight = l2_coefficients(
            problem.alpha, grid.time_steps
        )
        # An24 equations (8)-(9) and (15)-(17), PDF pages 6-7.
        phi_2 = math.gamma(3.0 - problem.alpha) * dt**problem.alpha
        beta = c_weight[1] + (4.0 - problem.alpha) / 2.0
        later_factor = _implicit_factor(
            interior_size, phi_2, beta, lower_l, diagonal_l, upper_l
        )

        for step in range(2, grid.time_steps + 1):
            if step == 2:
                # An24 equation (15), PDF page 7.
                rhs = (2.0 - b_weight[1]) * surface[1, 1:-1]
                rhs -= (0.5 * problem.alpha + a_weight[1]) * surface[0, 1:-1]
            elif step == 3:
                # An24 equation (16), PDF page 7.
                rhs = (2.0 - b_weight[1] - c_weight[2]) * surface[2, 1:-1]
                rhs -= (
                    0.5 * problem.alpha + a_weight[1] + b_weight[2]
                ) * surface[1, 1:-1]
                rhs -= a_weight[2] * surface[0, 1:-1]
            else:
                # An24 equation (17), PDF page 7.
                rhs = (2.0 - b_weight[1] - c_weight[2]) * surface[
                    step - 1, 1:-1
                ]
                rhs -= 0.5 * problem.alpha * surface[step - 2, 1:-1]
                for level in range(2, step - 1):
                    lag = step - level
                    coefficient = (
                        a_weight[lag - 1]
                        + b_weight[lag]
                        + c_weight[lag + 1]
                    )
                    rhs -= coefficient * surface[level, 1:-1]
                rhs -= (
                    a_weight[step - 2] + b_weight[step - 1]
                ) * surface[1, 1:-1]
                rhs -= a_weight[step - 1] * surface[0, 1:-1]

            if source is not None:
                rhs += phi_2 * _source_values(source, interior_x, times[step])
            rhs[0] += phi_2 * lower_l * surface[step, 0]
            rhs[-1] += phi_2 * upper_l * surface[step, -1]
            surface[step, 1:-1] = later_factor.solve(rhs)
    else:
        phi_2 = math.gamma(3.0 - problem.alpha) * dt**problem.alpha

    price = float(np.interp(log_spot, x, surface[-1]))
    return SolverResult(
        price=price,
        x_grid=x,
        time_grid=times,
        grid_price=surface,
        diagnostics={
            "solver": "l2",
            "paper": "An24",
            "paper_doi": "10.1007/s11075-023-01563-4",
            "paper_equations": "(7)-(17)",
            "startup": "published L1 equation (14)",
            "dt": dt,
            "dx": dx,
            "phi_1": phi_1,
            "phi_2": phi_2,
            "time_order_claim": 3.0 - problem.alpha,
            "space_order_claim": 2.0,
            "stability_proof_status": "published proof has an invalid drift-term cancellation",
        },
    )


def _implicit_factor(
    interior_size: int,
    time_scale: float,
    mass: float,
    lower_l: float,
    diagonal_l: float,
    upper_l: float,
) -> TridiagonalFactor:
    # The signs follow An24 component equations (14)-(17), PDF page 7.
    # Equation (20), PDF page 8, prints the later upper entry with the opposite
    # sign; copying that matrix-entry typo would contradict equations (15)-(17).
    lower = np.full(max(interior_size - 1, 0), -time_scale * lower_l)
    diagonal = np.full(interior_size, mass - time_scale * diagonal_l)
    upper = np.full(max(interior_size - 1, 0), -time_scale * upper_l)
    return TridiagonalFactor.factor(lower, diagonal, upper)


def _source_values(
    source: SourceFunction, x: NDArray[np.float64], elapsed: float
) -> NDArray[np.float64]:
    values = np.asarray(source(x, float(elapsed)), dtype=float)
    if values.shape != x.shape:
        raise ValueError("source callback must return one value per interior point")
    return values
