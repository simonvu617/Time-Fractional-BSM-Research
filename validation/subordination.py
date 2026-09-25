"""Independent inverse-stable subordination benchmark.

KMP20 Section 2.2, PDF pages 3-5, gives
``V_alpha(t)=E[V_BS(S_alpha(t))]`` and the inverse-clock density transform.
DOI: https://doi.org/10.1016/j.camwa.2020.04.029

The positive-stable random-variable representation is from M. Kanter,
"Stable densities under change of scale and total variation inequalities,"
The Annals of Probability 3(4), 697-707 (1975).
DOI: https://doi.org/10.1214/aop/1176996309

This module uses neither finite-difference recurrence.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from tfbsm_pricing.model import EuropeanOptionProblem, black_scholes_price


def inverse_stable_quadrature(
    alpha: float, elapsed: float, order: int = 96
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return inverse-stable clock values and deterministic quadrature weights.

    If ``D_1`` is a positive alpha-stable random variable with Laplace
    transform ``exp(-s**alpha)``, then the inverse clock satisfies
    ``S_alpha(t) = t**alpha * D_1**(-alpha)`` in distribution (KMP20,
    Section 2.2, PDF pages 3-5). Kanter's representation writes ``D_1`` using
    independent uniform and exponential variables. Gauss-Legendre quadrature
    integrates both uniforms after the exponential inverse-CDF transform.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if elapsed <= 0.0:
        raise ValueError("elapsed must be positive")
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    if alpha == 1.0:
        return np.array([elapsed]), np.array([1.0])

    nodes, base_weights = leggauss(order)
    angle = (nodes + 1.0) * (math.pi / 2.0)
    angle_weights = base_weights / 2.0  # includes division by pi
    uniform = (nodes + 1.0) / 2.0
    uniform_weights = base_weights / 2.0

    angle_grid = angle[:, np.newaxis]
    exponential_grid = -np.log1p(-uniform)[np.newaxis, :]
    stable = (
        np.sin(alpha * angle_grid)
        / np.sin(angle_grid) ** (1.0 / alpha)
        * (
            np.sin((1.0 - alpha) * angle_grid) / exponential_grid
        ) ** ((1.0 - alpha) / alpha)
    )
    clock = elapsed**alpha * stable ** (-alpha)
    weights = angle_weights[:, np.newaxis] * uniform_weights[np.newaxis, :]
    return clock.ravel(), weights.ravel()


def inverse_stable_moment(
    alpha: float, elapsed: float, power: float, order: int = 96
) -> float:
    """Evaluate a clock moment for checking the quadrature itself."""

    clock, weights = inverse_stable_quadrature(alpha, elapsed, order)
    return float(weights @ clock**power)


def subordinated_bsm_price(problem: EuropeanOptionProblem, order: int = 96) -> float:
    """Evaluate ``E[V_BS(S_alpha(T))]`` by deterministic quadrature."""

    if problem.alpha == 1.0:
        return black_scholes_price(problem)

    clock, weights = inverse_stable_quadrature(problem.alpha, problem.T, order)
    root_clock = np.sqrt(clock)
    d1 = (
        math.log(problem.S0 / problem.K)
        + (problem.r + 0.5 * problem.sigma**2) * clock
    ) / (problem.sigma * root_clock)
    d2 = d1 - problem.sigma * root_clock

    def normal_cdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.fromiter(
            (0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in values),
            dtype=float,
            count=values.size,
        )
        return result

    discounted_strike = problem.K * np.exp(-problem.r * clock)
    if problem.option_type == "call":
        conditional = problem.S0 * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    else:
        conditional = discounted_strike * normal_cdf(-d2) - problem.S0 * normal_cdf(-d1)
    return float(weights @ conditional)
