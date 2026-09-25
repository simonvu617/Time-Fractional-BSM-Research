"""Empirical joint grid refinement for either European TFBS solver."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .model import EuropeanOptionProblem, GridSpec, SolverResult


PricingSolver = Callable[[EuropeanOptionProblem, GridSpec], SolverResult]


def refine_price(
    solver: PricingSolver,
    problem: EuropeanOptionProblem,
    initial_grid: GridSpec,
    *,
    tolerance: float = 1e-4,
    max_refinements: int = 3,
) -> SolverResult:
    """Double both grid counts until successive prices meet ``tolerance``.

    The reported difference is an empirical grid-refinement estimate, not a
    rigorous error bound. The returned surface and price are from the finest
    grid evaluated.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_refinements < 1:
        raise ValueError("max_refinements must be at least 1")

    coarse_grid = initial_grid
    coarse = solver(problem, coarse_grid)
    converged = False

    for refinement in range(1, max_refinements + 1):
        fine_grid = GridSpec(
            coarse_grid.x_min,
            coarse_grid.x_max,
            2 * coarse_grid.space_steps,
            2 * coarse_grid.time_steps,
        )
        fine = solver(problem, fine_grid)
        difference = abs(fine.price - coarse.price)
        if difference <= tolerance:
            converged = True
            break
        if refinement < max_refinements:
            coarse_grid, coarse = fine_grid, fine

    diagnostics = dict(fine.diagnostics)
    diagnostics["grid_refinement"] = {
        "coarse_grid": coarse_grid,
        "fine_grid": fine_grid,
        "coarse_price": coarse.price,
        "fine_price": fine.price,
        "absolute_difference": difference,
        "refinements": refinement,
        "requested_tolerance": tolerance,
        "converged": converged,
    }
    return replace(fine, diagnostics=diagnostics)
