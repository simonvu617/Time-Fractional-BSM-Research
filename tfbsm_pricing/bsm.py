"""Analytic zero-dividend Black--Scholes benchmark."""

from __future__ import annotations

import math

from .model import EuropeanOptionProblem


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_price(problem: EuropeanOptionProblem) -> float:
    """Return the classical BSM price for the canonical option inputs.

    ``alpha`` is deliberately ignored: this function is the independent
    analytic benchmark used when the fractional model is evaluated at one.
    """

    root_time = math.sqrt(problem.T)
    d1 = (
        math.log(problem.S0 / problem.K)
        + (problem.r + 0.5 * problem.sigma**2) * problem.T
    ) / (problem.sigma * root_time)
    d2 = d1 - problem.sigma * root_time
    discounted_strike = problem.K * math.exp(-problem.r * problem.T)
    if problem.option_type == "call":
        return problem.S0 * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return discounted_strike * _normal_cdf(-d2) - problem.S0 * _normal_cdf(-d1)
