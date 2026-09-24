"""Canonical European time-fractional Black--Scholes model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
