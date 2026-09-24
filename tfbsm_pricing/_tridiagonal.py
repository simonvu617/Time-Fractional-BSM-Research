"""Small dependency-free factorization for constant tridiagonal systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class TridiagonalFactor:
    """Reusable Thomas factorization of a nonsingular tridiagonal matrix."""

    lower_multipliers: NDArray[np.float64]
    diagonal: NDArray[np.float64]
    upper: NDArray[np.float64]

    @classmethod
    def factor(
        cls,
        lower: NDArray[np.float64],
        diagonal: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> "TridiagonalFactor":
        lower = np.asarray(lower, dtype=float).copy()
        diagonal = np.asarray(diagonal, dtype=float).copy()
        upper = np.asarray(upper, dtype=float).copy()
        n = diagonal.size
        if lower.size != max(n - 1, 0) or upper.size != max(n - 1, 0):
            raise ValueError("invalid tridiagonal dimensions")
        if n == 0:
            raise ValueError("the system must contain at least one row")

        multipliers = np.empty_like(lower)
        tolerance = np.finfo(float).eps
        for row in range(1, n):
            pivot = diagonal[row - 1]
            if abs(pivot) <= tolerance:
                raise np.linalg.LinAlgError("zero pivot in tridiagonal factorization")
            multiplier = lower[row - 1] / pivot
            multipliers[row - 1] = multiplier
            diagonal[row] -= multiplier * upper[row - 1]
        if abs(diagonal[-1]) <= tolerance:
            raise np.linalg.LinAlgError("zero pivot in tridiagonal factorization")
        return cls(multipliers, diagonal, upper)

    def solve(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        rhs = np.asarray(right_hand_side, dtype=float).copy()
        if rhs.ndim != 1 or rhs.size != self.diagonal.size:
            raise ValueError("right-hand side has the wrong shape")

        for row in range(1, rhs.size):
            rhs[row] -= self.lower_multipliers[row - 1] * rhs[row - 1]
        rhs[-1] /= self.diagonal[-1]
        for row in range(rhs.size - 2, -1, -1):
            rhs[row] = (
                rhs[row] - self.upper[row] * rhs[row + 1]
            ) / self.diagonal[row]
        return rhs
