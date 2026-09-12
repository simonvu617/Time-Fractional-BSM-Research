# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Select observed option contracts by maturity and stock-price moneyness.

Selection works on copies of dated quote/trade lists and open-interest reports.
It does not filter contracts by option liquidity or rewrite stored market rows.
The union across selection times is a retrospective research sample.
"""

from typing import Iterable

import numpy as np
import pandas as pd

from tfbsm_collector import config, planning, validation

DISCOVERY_COLUMNS = (*planning.CONTRACT_FIELDS, "discovery_sources")

SELECTION_COLUMNS = (
    *planning.CONTRACT_FIELDS,
    "contract_key",
    "dte_days",
    "selection_times",
    "discovery_sources",
)


def normalize_chain(
    frame: pd.DataFrame, symbol: str, cfg: config.CollectorConfig
) -> pd.DataFrame:
    """Normalize and deduplicate a dated contract list for selection.

    Args:
        frame: Vendor table containing the four contract identity fields.
        symbol: Expected underlying ticker.
        cfg: Option rights allowed by the study.

    Returns:
        A sorted table of unique identities with parsed expirations and strikes.
        The input table and stored observations remain unchanged.

    Raises:
        ValueError: A row has a wrong symbol or invalid contract identity.
    """
    if frame.empty:
        return pd.DataFrame(columns=planning.CONTRACT_FIELDS)
    chain = frame.loc[:, planning.CONTRACT_FIELDS].copy()
    chain["expiration"] = pd.to_datetime(
        chain["expiration"], format="mixed", errors="raise"
    ).dt.normalize()
    chain["strike"] = pd.to_numeric(chain["strike"], errors="raise")
    chain["right"] = (
        chain["right"].str.lower().replace({"c": "call", "p": "put"})
    )
    if (
        chain["symbol"].ne(symbol).any()
        or chain[["expiration", "strike"]].isna().any().any()
        or not np.isfinite(chain["strike"]).all()
        or chain["strike"].le(0).any()
        or not chain["right"].isin(["call", "put"]).all()
    ):
        raise ValueError("Invalid identity in dated observed universe")
    for strike in chain["strike"].unique():
        planning.format_strike(strike)
    # Deduplicate contract identities for requesting, never the contract's
    # raw observations. Repeated quotes can matter to later liquidity research.
    return (
        chain.loc[chain["right"].isin(cfg.option_rights)]
        .drop_duplicates()
        .sort_values(["expiration", "strike", "right"])
        .reset_index(drop=True)
    )


def observed_contract_universe(
    frames: dict[str, pd.DataFrame], symbol: str, cfg: config.CollectorConfig
) -> tuple[pd.DataFrame, dict]:
    """Combine dated discovery sources while recording each source's role.

    Args:
        frames: Tables keyed by source, usually quote, trade, and open_interest.
        symbol: Expected underlying ticker.
        cfg: Allowed option rights.

    Returns:
        A pair (universe, counts). The universe has one row per identity and a
        pipe-separated discovery_sources column. Counts describe unique
        identities in each source before combining them. This observed list does
        not establish complete historical listing coverage.
    """
    # OI can reveal a contract with no quote or trade that day. Requiring
    # current-day activity here would remove quiet contracts from the sample.
    tables = {
        name: normalize_chain(frame, symbol, cfg)
        for name, frame in frames.items()
    }
    counts = {name: len(table) for name, table in tables.items()}
    parts = [
        table.assign(discovery_sources=name)
        for name, table in tables.items()
        if not table.empty
    ]
    if not parts:
        return pd.DataFrame(columns=DISCOVERY_COLUMNS), counts
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(
        list(planning.CONTRACT_FIELDS), as_index=False, sort=True
    )["discovery_sources"].agg(lambda sources: "|".join(sorted(set(sources))))
    return combined.loc[:, DISCOVERY_COLUMNS], counts


def stock_selection_references(
    frames: pd.DataFrame | Iterable[pd.DataFrame],
    day: pd.Timestamp,
    cfg: config.CollectorConfig,
) -> list[dict]:
    """Find usable stock midpoints at the configured selection times.

    Args:
        frames: Stock quote table or batches, with timestamp, bid, ask, and both
            condition columns. Ties retain the last row in input order.
        day: Exchange session date, without a time zone.
        cfg: Selection times, accepted sample age, and exchange time zone.

    Returns:
        Dictionaries with selection_time, stock_mid, source timestamps, and
        sample age. Missing or unusable references are omitted. The midpoint
        targets strikes; it is not an assumed executable price.
    """
    opened, closed = planning.session_bounds(day, cfg)
    times = {
        at: pd.Timestamp(f"{day.date()} {at}", tz=cfg.exchange_tz)
        for at in cfg.selection_times
    }
    times = {
        at: clock for at, clock in times.items() if opened <= clock < closed
    }
    latest = {}
    for frame in [frames] if isinstance(frames, pd.DataFrame) else frames:
        if frame.empty:
            continue
        quotes = frame.copy()
        quotes["_clock"] = validation.parse_vendor_clock(
            quotes["timestamp"], cfg.exchange_tz
        )
        quotes = quotes.loc[quotes["_clock"].notna()].sort_values(
            "_clock", kind="stable"
        )
        for at, clock in times.items():
            # Only use samples at or before the reference. Stable ordering gives
            # the last vendor row precedence when timestamps tie across batches.
            prior = quotes.loc[quotes["_clock"].between(opened, clock)]
            if not prior.empty:
                row = prior.iloc[-1]
                if at not in latest or row["_clock"] >= latest[at]["_clock"]:
                    latest[at] = row

    # Assess the actual latest sample before accepting it. Filtering bad quotes
    # first would hide a halt/non-firm quote behind an older usable observation.
    references = []
    for selection_time, at in times.items():
        if selection_time not in latest:
            continue
        row = latest[selection_time]
        conditions = [
            str(row[column]).strip()
            for column in ("bid_condition", "ask_condition")
        ]

        # This study accepts condition codes 0 (regular), 1 (auto-executable),
        # and 50 (national BBO), plus unspecified blanks. This is a selection
        # policy, not Theta's complete list of usable conditions. Raw rows stay
        # saved.
        # https://docs.thetadata.us/Articles/Errors-Exchanges-Conditions/Quote-Conditions.html
        if any(
            text and pd.to_numeric(text, errors="coerce") not in {0, 1, 50}
            for text in conditions
        ):
            continue
        bid, ask = (
            pd.to_numeric(row[side], errors="coerce") for side in ("bid", "ask")
        )
        # Age measures the returned sample's clock, not the original quote
        # event.
        age = (at - row["_clock"]).total_seconds()

        # A missing, nonpositive, or crossed stock quote cannot supply a
        # sensible S/K target. Reject the reference without cleaning the saved
        # market table.
        if not (
            np.isfinite(bid)
            and np.isfinite(ask)
            and 0 < bid <= ask
            and age <= cfg.max_stock_quote_age_seconds
        ):
            continue

        references.append(
            {
                "selection_time": selection_time,
                "stock_mid": float((bid + ask) / 2),
                "observation_timestamp": str(row["timestamp"]),
                "observation_timestamp_utc": row["_clock"].isoformat(),
                "observation_age_seconds": age,
                "timestamp_role": "sample_boundary",
                "quote_event_age_seconds": None,
            }
        )
    return references


def eligible_expirations(
    day: pd.Timestamp, expirations, cfg: config.CollectorConfig
) -> list[pd.Timestamp]:
    """Choose nearby listed expirations within the inclusive DTE bounds.

    Args:
        day: Exchange session date, without a time zone.
        expirations: Unique observed expiration timestamps.
        cfg: Target DTEs, allowed DTE range, and expiration cap.

    Returns:
        Sorted expirations, preferring the shorter maturity on equal target
        distance. Distances are calendar days, not a pricing year fraction.
    """
    remaining = sorted(
        exp
        for exp in expirations
        if cfg.min_dte <= (exp - day).days <= cfg.max_dte
    )
    selected = []
    for target in cfg.target_dtes:
        if not remaining or len(selected) >= cfg.max_expirations_per_day:
            break

        # Targets 29 and 31 days away tie for a 30-day target; prefer 29.
        # Remove the chosen expiry so nearby targets do not select it twice.
        best = min(
            remaining,
            key=lambda exp: (abs((exp - day).days - target), (exp - day).days),
        )
        selected.append(best)
        remaining.remove(best)
    remaining.sort(
        key=lambda exp: (
            min(abs((exp - day).days - target) for target in cfg.target_dtes),
            exp,
        )
    )

    return sorted(
        selected
        + remaining[: max(cfg.max_expirations_per_day - len(selected), 0)]
    )


def select_contracts(
    chain: pd.DataFrame,
    day: pd.Timestamp,
    references: list[dict],
    cfg: config.CollectorConfig,
) -> pd.DataFrame:
    """Select observed contracts across the maturity and S/K target grid.

    Args:
        chain: Normalized dated universe, optionally with discovery_sources.
        day: Exchange session date, without a time zone.
        references: Usable stock midpoint records from
            stock_selection_references.
        cfg: Maturity, moneyness, and strike-count settings.

    Returns:
        A table with SELECTION_COLUMNS. Each selected identity appears once,
        with every selection time that chose its strike. Membership is the
        retrospective union across those times, without an option-volume,
        spread, or open-interest threshold.
    """
    # Do not select on option volume, spread, or OI thresholds: that would
    # remove the quiet or wide-spread contracts needed for the liquidity
    # comparison.
    rows = []
    for expiration in eligible_expirations(
        day, chain["expiration"].unique(), cfg
    ):
        family = chain.loc[chain["expiration"].eq(expiration)]
        strikes = sorted(family["strike"].unique())
        chosen_by_time = {}
        for reference in references:
            chosen = set()
            for target in cfg.moneyness_targets:
                # K = S/(S/K); the target is a ratio, not a percentage strike
                # offset.
                target_strike = reference["stock_mid"] / target

                # Choose actual listed strikes by dollar distance, taking the
                # lower strike on a tie. Achieved moneyness can differ from the
                # target.
                ranked = sorted(
                    strikes,
                    key=lambda strike: (abs(strike - target_strike), strike),
                )
                chosen.update(ranked[: cfg.strikes_per_moneyness_target])
            chosen_by_time[reference["selection_time"]] = chosen

        # Keep the union of strikes chosen as S moves during the day. A contract
        # chosen in the afternoon gets its full requested day, so membership is
        # retrospective and may not have been known at the morning observation.
        selected = (
            set().union(*chosen_by_time.values()) if chosen_by_time else set()
        )

        for contract in family.loc[family["strike"].isin(selected)].itertuples(
            index=False
        ):
            expiry, strike = (
                expiration.strftime("%Y-%m-%d"),
                planning.format_strike(contract.strike),
            )
            rows.append(
                {
                    "symbol": contract.symbol,
                    "expiration": expiry,
                    "strike": strike,
                    "right": contract.right,
                    "contract_key": f"{contract.symbol}|{expiry}|{strike}|{contract.right}",
                    "dte_days": (expiration - day).days,
                    "selection_times": "|".join(
                        at
                        for at, values in chosen_by_time.items()
                        if contract.strike in values
                    ),
                    "discovery_sources": getattr(
                        contract, "discovery_sources", ""
                    ),
                }
            )
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)
