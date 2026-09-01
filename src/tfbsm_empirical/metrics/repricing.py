from __future__ import annotations

from math import isfinite


def absolute_repricing_error(model_value: float, market_value: float) -> float:
    """Return ``abs(model_value - market_value)`` for finite inputs."""

    predicted = float(model_value)
    observed = float(market_value)
    if not isfinite(predicted) or not isfinite(observed):
        raise ValueError("repricing-error inputs must be finite")
    return abs(predicted - observed)


def repricing_improvement(bsm_loss: float, tfbsm_loss: float) -> float:
    """Return BSM loss minus TFBSM loss; positive values favor TFBSM."""

    baseline = float(bsm_loss)
    candidate = float(tfbsm_loss)
    if not isfinite(baseline) or not isfinite(candidate):
        raise ValueError("repricing losses must be finite")
    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("repricing losses must be nonnegative")
    return baseline - candidate
