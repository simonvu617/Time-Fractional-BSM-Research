"""Result and diagnostic types shared by the public solver APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SolverResult:
    """A scalar price together with the computed finite-difference surface."""

    price: float
    x_grid: NDArray[np.float64]
    time_grid: NDArray[np.float64]
    grid_price: NDArray[np.float64]
    diagnostics: Mapping[str, Any]
