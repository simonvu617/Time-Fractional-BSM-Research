"""Canonical model, grid, boundary data, and solver result types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray


OptionType = Literal["call", "put"]
BoundaryMode = Literal["canonical_subdiffusive", "paper_reproduction"]


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
    boundary_mode: BoundaryMode = "canonical_subdiffusive"

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
        if self.boundary_mode not in ("canonical_subdiffusive", "paper_reproduction"):
            raise ValueError("unknown boundary_mode")

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
    """Return finite-domain values for the selected mathematical model.

    The canonical discount is ``E_alpha(-r*t^alpha)``, obtained by applying the
    KMP20 subordination identity (page 3) and clock-density transform (page 4)
    to the operational-time BSM discount. Paper reproduction mode retains the
    ordinary exponential used in the published numerical setup.
    """

    if problem.boundary_mode == "canonical_subdiffusive":
        discount = mittag_leffler_discount(problem.alpha, problem.r, elapsed)
    else:
        discount = math.exp(-problem.r * elapsed)
    discounted_strike = problem.K * discount
    if problem.option_type == "call":
        return 0.0, max(math.exp(x_max) - discounted_strike, 0.0)
    return max(discounted_strike - math.exp(x_min), 0.0), 0.0


def mittag_leffler_discount(alpha: float, rate: float, elapsed: float) -> float:
    """Compute ``E_alpha(-rate * elapsed**alpha)`` by its defining series.

    The direct series is deliberately limited to the range verified here.
    Larger arguments need a separate asymptotic or contour algorithm; refusing
    them avoids silent cancellation error in finite-domain boundary values.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if elapsed < 0.0:
        raise ValueError("elapsed time must be nonnegative")
    if elapsed == 0.0 or rate == 0.0:
        return 1.0
    if alpha == 1.0:
        return math.exp(-rate * elapsed)

    argument = -rate * elapsed**alpha
    if abs(argument) > 0.75:
        raise ValueError(
            "Mittag-Leffler boundary evaluation requires "
            "abs(rate * elapsed**alpha) <= 0.75"
        )
    total = 1.0
    for index in range(1, 512):
        magnitude = math.exp(
            index * math.log(abs(argument))
            - math.lgamma(alpha * index + 1.0)
        )
        term = -magnitude if argument < 0.0 and index % 2 else magnitude
        total += term
        if abs(term) <= 1e-15 * max(1.0, abs(total)):
            return total
    raise ArithmeticError("Mittag-Leffler series did not converge")


def black_scholes_price(problem: EuropeanOptionProblem) -> float:
    """Return the analytic zero-dividend BSM benchmark."""

    root_time = math.sqrt(problem.T)
    d1 = (
        math.log(problem.S0 / problem.K)
        + (problem.r + 0.5 * problem.sigma**2) * problem.T
    ) / (problem.sigma * root_time)
    d2 = d1 - problem.sigma * root_time
    def normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    discounted_strike = problem.K * math.exp(-problem.r * problem.T)
    if problem.option_type == "call":
        return problem.S0 * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    return discounted_strike * normal_cdf(-d2) - problem.S0 * normal_cdf(-d1)
