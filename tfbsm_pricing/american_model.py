"""Inputs and stock-grid output for an American time-fractional put."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AmericanPutProblem:
    """Zhou--Gao's beta=0 model, using sigma and continuous dividend yield q.

    Time is in years with the same normalized fractional-time scale as the
    European solvers. Nonnegative rates give the boundary V(0, tau)=K.
    Scheduled cash dividends and general CEV elasticity are not included.
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    alpha: float
    q: float = 0.0

    def __post_init__(self) -> None:
        for name in ("S0", "K", "T", "r", "sigma", "alpha", "q"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.S0 < 0.0:
            raise ValueError("S0 must be nonnegative")
        for name in ("K", "T", "sigma"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if self.r < 0.0 or self.q < 0.0:
            raise ValueError("r and q must be nonnegative")


@dataclass(frozen=True, slots=True)
class StockGridSpec:
    """Uniform stock grid [0, S_max]; step counts mean intervals."""

    S_max: float
    space_steps: int
    time_steps: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.S_max) or self.S_max <= 0.0:
            raise ValueError("S_max must be finite and positive")
        for name, minimum in (("space_steps", 2), ("time_steps", 1)):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, Integral) or count < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")

    @property
    def dS(self) -> float:
        return self.S_max / self.space_steps


@dataclass(frozen=True, slots=True)
class AmericanPutResult:
    """Price, full surface, and mesh approximation to the exercise boundary.

    ``time_grid`` is time to expiry, increasing from the payoff at zero.
    ``exercise_boundary`` is the last exercise node at each positive time;
    its resolution is dS. Its expiry entry is K by convention.
    """

    price: float
    stock_grid: NDArray[np.float64]
    time_grid: NDArray[np.float64]
    grid_price: NDArray[np.float64]
    exercise_boundary: NDArray[np.float64]
    diagnostics: Mapping[str, Any]
