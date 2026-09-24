"""Shared manufactured solution from An et al. (2024), pages 13--14."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from tfbsm_pricing.model import EuropeanOptionProblem


def manufactured_problem(alpha: float) -> EuropeanOptionProblem:
    # An et al. specify mu=sigma^2/2=1, r=0.5, and lambda=r-mu=-0.5.
    return EuropeanOptionProblem(
        S0=math.exp(0.5),
        K=1.0,
        T=1.0,
        r=0.5,
        sigma=math.sqrt(2.0),
        alpha=alpha,
        option_type="call",
    )


def manufactured_exact(x: NDArray[np.float64], elapsed: float) -> NDArray[np.float64]:
    return (elapsed + 1.0) ** 2 * (x**3 + x**2 + 1.0)


def manufactured_source(
    x: NDArray[np.float64], elapsed: float, alpha: float
) -> NDArray[np.float64]:
    polynomial = x**3 + x**2 + 1.0
    caputo_time = (
        2.0 * elapsed ** (2.0 - alpha) / math.gamma(3.0 - alpha)
        + 2.0 * elapsed ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    )
    spatial = (6.0 * x + 2.0) - 0.5 * (3.0 * x**2 + 2.0 * x) - 0.5 * polynomial
    return caputo_time * polynomial - (elapsed + 1.0) ** 2 * spatial


def manufactured_boundary(elapsed: float) -> tuple[float, float]:
    values = manufactured_exact(np.array([0.0, 1.0]), elapsed)
    return float(values[0]), float(values[1])
