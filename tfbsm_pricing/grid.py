"""Uniform log-price and elapsed-time grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A uniform finite-difference grid.

    ``space_steps`` and ``time_steps`` are interval counts, so the returned
    grids contain one more point than the corresponding count.
    """

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
