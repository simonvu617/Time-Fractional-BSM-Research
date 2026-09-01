from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable


def select_expirations(
    trade_date: date,
    expirations: Iterable[date],
    *,
    min_dte: int,
    max_dte: int,
    target_dtes: tuple[int, ...],
    maximum: int,
) -> tuple[date, ...]:
    eligible = sorted(
        {
            expiration
            for expiration in expirations
            if min_dte <= (expiration - trade_date).days <= max_dte
        }
    )
    selected: list[date] = []
    remaining = list(eligible)
    for target in target_dtes:
        if not remaining or len(selected) >= maximum:
            break
        best = min(
            remaining,
            key=lambda expiration: (
                abs((expiration - trade_date).days - target),
                (expiration - trade_date).days,
            ),
        )
        selected.append(best)
        remaining.remove(best)

    if len(selected) < maximum:
        remaining.sort(
            key=lambda expiration: min(
                abs((expiration - trade_date).days - target)
                for target in target_dtes
            )
        )
        selected.extend(remaining[: maximum - len(selected)])
    return tuple(sorted(selected))


def select_strikes(
    strikes: Iterable[float],
    reference_mids: Iterable[float],
    *,
    moneyness_targets: tuple[float, ...],
    per_target: int,
) -> tuple[float, ...]:
    available = sorted({float(value) for value in strikes})
    references = [float(value) for value in reference_mids if float(value) > 0]
    if not available or not references:
        return ()

    selected: list[float] = []
    for reference_mid in references:
        for target_moneyness in moneyness_targets:
            target_strike = reference_mid / target_moneyness
            ranked = sorted(
                available,
                key=lambda strike: (abs(strike - target_strike), strike),
            )
            for strike in ranked[:per_target]:
                if strike not in selected:
                    selected.append(strike)
    return tuple(sorted(selected))


def occ_contract_id(
    symbol: str,
    expiration: date,
    right: str,
    strike: float,
) -> str:
    root_value = symbol.strip().upper()
    if len(root_value) > 6:
        raise ValueError("OCC roots longer than six characters require an explicit vendor mapping")
    normalized_right = right.strip().lower()
    if normalized_right not in {"call", "put"}:
        raise ValueError(f"unsupported option right: {right!r}")
    root = root_value.ljust(6)
    strike_mils = int(
        (Decimal(str(strike)) * Decimal("1000")).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    return (
        f"{root}{expiration.strftime('%y%m%d')}"
        f"{normalized_right[0].upper()}{strike_mils:08d}"
    )


def format_strike(strike: float) -> str:
    value = f"{float(strike):.6f}".rstrip("0").rstrip(".")
    return value or "0"
