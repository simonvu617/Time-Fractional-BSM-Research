from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ModelSpecification:
    key: str
    label: str
    family: str
    implementation_status: str


BSM_SPECIFICATION = ModelSpecification(
    key="bsm",
    label="Black-Scholes-Merton",
    family="classical_diffusion",
    implementation_status="not_implemented",
)

TFBSM_SPECIFICATION = ModelSpecification(
    key="subdiffusive_tfbsm",
    label="Subdiffusive time-fractional Black-Scholes",
    family="inverse_stable_market_time",
    implementation_status="not_implemented",
)


def validate_alpha(alpha: float) -> float:
    """Validate the locked TFBSM fractional-order domain ``0 < alpha <= 1``."""

    value = float(alpha)
    if not isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("alpha must be finite and satisfy 0 < alpha <= 1")
    return value
