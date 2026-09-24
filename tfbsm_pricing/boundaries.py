"""European payoff and finite-domain boundary conditions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .model import EuropeanOptionProblem


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
    """Return asymptotic left and right values at elapsed pricing time.

    Elapsed time starts at zero at the payoff and reaches ``T`` at valuation.
    Discounting is therefore by ``exp(-r * elapsed)``.
    """

    discounted_strike = problem.K * math.exp(-problem.r * elapsed)
    if problem.option_type == "call":
        return 0.0, max(math.exp(x_max) - discounted_strike, 0.0)
    return max(discounted_strike - math.exp(x_min), 0.0), 0.0
