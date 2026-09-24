"""Canonical model, grid, boundary data, and solver result types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray


OptionType = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class EuropeanOptionProblem:
    """A zero-dividend European option in the papers' normalized time units.

    Time is measured in years.  The coefficient called ``rho`` by An et al.
    has units year^(alpha-1); that paper sets it to one, and this canonical
    problem makes the same normalization.
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    alpha: float
    option_type: OptionType

    def __post_init__(self) -> None:
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive")
        if self.K <= 0.0:
            raise ValueError("K must be positive")
        if self.T <= 0.0:
            raise ValueError("T must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    @property
    def diffusion(self) -> float:
        """Coefficient of ``u_xx`` after the log-price transformation."""

        return 0.5 * self.sigma * self.sigma

    @property
    def drift(self) -> float:
        """Coefficient of ``u_x`` in the transformed pricing equation."""

        return self.r - self.diffusion


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A uniform log-price and elapsed-time grid, counted in intervals."""

    x_min: float
    x_max: float
    space_steps: int
    time_steps: int

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.space_steps < 2:
            raise ValueError("space_steps must be at least 2")
        if self.time_steps < 1:
            raise ValueError("time_steps must be at least 1")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.space_steps

    def dt(self, maturity: float) -> float:
        return maturity / self.time_steps

    def x_values(self) -> NDArray[np.float64]:
        return np.linspace(self.x_min, self.x_max, self.space_steps + 1)

    def time_values(self, maturity: float) -> NDArray[np.float64]:
        return np.linspace(0.0, maturity, self.time_steps + 1)


@dataclass(frozen=True, slots=True)
class SolverResult:
    """A scalar price and the finite-difference surface used to obtain it."""

    price: float
    x_grid: NDArray[np.float64]
    time_grid: NDArray[np.float64]
    grid_price: NDArray[np.float64]
    diagnostics: Mapping[str, Any]


def payoff(
    problem: EuropeanOptionProblem, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the maturity payoff on a log-price grid."""

    stock = np.exp(x)
    if problem.option_type == "call":
        return np.maximum(stock - problem.K, 0.0)
    return np.maximum(problem.K - stock, 0.0)


def boundary_values(
    problem: EuropeanOptionProblem, x_min: float, x_max: float, elapsed: float
) -> tuple[float, float]:
    """Return the papers' European finite-domain boundary values."""

    discounted_strike = problem.K * math.exp(-problem.r * elapsed)
    if problem.option_type == "call":
        return 0.0, max(math.exp(x_max) - discounted_strike, 0.0)
    return max(discounted_strike - math.exp(x_min), 0.0), 0.0
