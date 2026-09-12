# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Study settings and the fixed stock/ETF universe.

CollectorConfig separates sampling choices from download and storage controls.
Changing the sampling policy creates a separate collection directory; changing
worker counts or parsing batch sizes does not change the research sample.
"""

import dataclasses
import pathlib
import re

import numpy as np
import pandas as pd

from tfbsm_collector import provenance

# Saved layout/meaning determines resume compatibility. A source-file move
# does not change that contract, so the output schema version stays the same.
OUTPUT_SCHEMA_VERSION = "2026-09-08-selected-quotes-v5"

# Resolve from the repository root so splitting the package does not move
# existing caches into a new data directory.
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "multi_year_bsm_backtest_output"
)

# One minute is the minimum supported stock snapshot interval on Standard;
# hourly is the current study choice shared by all three asset types.
QUOTE_INTERVALS = ("1m", "5m", "10m", "15m", "30m", "1h")

# SOFR is an overnight benchmark. Treasury M/Y suffixes identify months/years.
# Keep the reported curve; maturity matching and discounting happen later.
RATE_SYMBOLS = (
    "SOFR",
    "TREASURY_M1",
    "TREASURY_M3",
    "TREASURY_M6",
    "TREASURY_Y1",
    "TREASURY_Y2",
    "TREASURY_Y3",
    "TREASURY_Y5",
    "TREASURY_Y7",
    "TREASURY_Y10",
    "TREASURY_Y20",
    "TREASURY_Y30",
)


@dataclasses.dataclass(frozen=True)
class SymbolConfig:
    """An underlying and its fixed study labels.

    Attributes:
        symbol: Theta ticker used in requests.
        asset_type: ETF or EQUITY for later cross-asset comparisons.
        universe_bucket: Study group, not historical index membership.
        sector_proxy: Descriptive sector label, not a dated classification.
    """

    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str


# This fixed cross-asset sample is not a reconstruction of all past listings.
# A chosen ticker can legitimately have no history before it was listed.
UNIVERSE = [
    SymbolConfig("SPY", "ETF", "broad_market_etf", "broad_market"),
    SymbolConfig("QQQ", "ETF", "growth_etf", "technology"),
    SymbolConfig("IWM", "ETF", "small_cap_etf", "small_cap"),
    SymbolConfig("DIA", "ETF", "value_etf", "industrial_mix"),
    SymbolConfig("XLF", "ETF", "sector_etf", "financials"),
    SymbolConfig("XLK", "ETF", "sector_etf", "technology"),
    SymbolConfig("XLE", "ETF", "sector_etf", "energy"),
    SymbolConfig("XLV", "ETF", "sector_etf", "health_care"),
    SymbolConfig("XLI", "ETF", "sector_etf", "industrials"),
    SymbolConfig("XLP", "ETF", "sector_etf", "consumer_staples"),
    SymbolConfig("AAPL", "EQUITY", "mega_cap_equity", "technology"),
    SymbolConfig("MSFT", "EQUITY", "mega_cap_equity", "technology"),
    SymbolConfig("NVDA", "EQUITY", "mega_cap_equity", "technology"),
    SymbolConfig("JPM", "EQUITY", "mega_cap_equity", "financials"),
    SymbolConfig("XOM", "EQUITY", "mega_cap_equity", "energy"),
    SymbolConfig("UNH", "EQUITY", "mega_cap_equity", "health_care"),
    SymbolConfig("WMT", "EQUITY", "mega_cap_equity", "consumer_staples"),
    SymbolConfig("CAT", "EQUITY", "large_cap_equity", "industrials"),
    SymbolConfig("AMD", "EQUITY", "large_cap_equity", "technology"),
    SymbolConfig("PLTR", "EQUITY", "mid_cap_equity", "technology"),
    SymbolConfig("RIOT", "EQUITY", "small_cap_equity", "crypto_exposed"),
]


@dataclasses.dataclass(frozen=True)
class CollectorConfig:
    """Validated sampling, subscription, and storage settings for a run.

    Dates and sampling targets are research choices. Constructing this frozen
    configuration checks their consistency without contacting Theta.

    Attributes:
        base_url: URL of the running Theta Terminal v3 service.
        start_date: Earliest study date allowed by the CLI, inclusive.
        end_date: Latest study date allowed by the CLI, inclusive.
        index_history_start: First accessible index date on Standard.
        lookback_sessions: Earlier stock/rate exchange sessions to collect.
        option_rights: Call (right to buy at the strike) and/or put (right to
            sell).
        target_dtes: Preferred calendar-day distances to expiration.
        max_expirations_per_day: Maximum selected expirations per underlying.
        moneyness_targets: Target underlying-price/strike ratios, S/K.
        strikes_per_moneyness_target: Listed strikes nearest each target.
        min_dte: Inclusive minimum calendar days to expiration.
        max_dte: Inclusive maximum DTE, also applied to bulk reports/lists.
        exchange_tz: Time zone for sessions and naive vendor timestamps.
        quote_interval: Sampling interval shared by stocks/options/indices.
        near_close_minutes: Minutes before the actual session close for the
            additional daily comparison snapshot.
        raw_chunk_rows: Maximum CSV records per parsing/storage batch.
        stock_venue: Theta stock feed; utp_cta requests the merged feeds.
        selection_times: Exchange-local HH:MM:SS times for stock references.
        max_stock_quote_age_seconds: Maximum age of a sampled stock record at
            selection time; does not measure the quote event's age.
        max_symbol_day_workers: Concurrent underlying/day tasks.
        max_batch_workers: Concurrent downloads within each batch.
        max_inflight_requests: Shared HTTP cap, at most four for Standard.
        max_requests_per_second: Shared request-start limit; zero disables it.
        store_raw_payloads: Whether to keep full successful CSV responses in
            addition to Parquet. Failed response bytes are always retained.
        refresh_no_data: Whether to retry previously successful empty pulls.
        output_dir: Root for response caches, manifests, and coverage reports.
    """

    base_url: str = "http://127.0.0.1:25503/v3"
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    index_history_start: str = "2022-01-01"

    # The buffer can support a first-day volatility estimate. It does not choose
    # a calibration window or guarantee continuous histories of the same
    # options.
    lookback_sessions: int = 60
    option_rights: tuple[str, ...] = ("call", "put")

    target_dtes: tuple[int, ...] = (7, 14, 30, 60, 120)
    max_expirations_per_day: int = 5

    # Moneyness is S/K: at S=$100, target 0.80 seeks K=$125. Values above one
    # are in the money for calls and out of the money for puts.
    moneyness_targets: tuple[float, ...] = (
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.20,
    )
    strikes_per_moneyness_target: int = 1
    min_dte: int = 7
    max_dte: int = 180
    exchange_tz: str = "America/New_York"
    quote_interval: str = "1h"
    near_close_minutes: int = 5
    raw_chunk_rows: int = 100_000
    stock_venue: str = "utp_cta"

    # These times align with hourly samples starting at 09:30. References after
    # an early close are omitted instead of borrowing a different day's price.
    selection_times: tuple[str, ...] = ("10:30:00", "13:30:00", "15:30:00")

    # A 12:30 sample cannot supply the 13:30 reference under this tolerance.
    # The timestamp does not reveal when the underlying quote last changed.
    max_stock_quote_age_seconds: int = 70
    max_symbol_day_workers: int = 4
    max_batch_workers: int = 4
    # Standard's four-request limit is account-wide. Lower this if another
    # process uses the same account; local worker limits cannot coordinate it.
    max_inflight_requests: int = 4
    max_requests_per_second: float = 0.0
    store_raw_payloads: bool = False
    refresh_no_data: bool = False
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self):
        """Reject inconsistent settings before a run creates files.

        Raises:
            ValueError: A sampling, interval, or resource limit is unsupported.
        """
        if not self.option_rights or set(self.option_rights) - {"call", "put"}:
            raise ValueError("option_rights must contain call and/or put")
        if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
            raise ValueError("start_date must not follow end_date")
        if not 0 <= self.min_dte <= self.max_dte or not self.target_dtes:
            raise ValueError(
                "Provide target_dtes and an ordered, nonnegative DTE range"
            )
        if any(d < 0 for d in self.target_dtes):
            raise ValueError("target_dtes must be nonnegative")
        if not self.moneyness_targets or any(
            not np.isfinite(x) or x <= 0 for x in self.moneyness_targets
        ):
            raise ValueError("moneyness_targets must be finite and positive")
        for name in (
            "max_expirations_per_day",
            "strikes_per_moneyness_target",
            "max_symbol_day_workers",
            "max_batch_workers",
            "max_inflight_requests",
            "max_stock_quote_age_seconds",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not np.isfinite(self.max_requests_per_second)
            or self.max_requests_per_second < 0
        ):
            raise ValueError(
                "max_requests_per_second must be nonnegative; 0 disables pacing"
            )
        if self.max_inflight_requests > 4:
            raise ValueError(
                "Standard allows at most four simultaneous requests across the account"
            )
        if (
            not isinstance(self.near_close_minutes, int)
            or not 1 <= self.near_close_minutes <= 30
        ):
            raise ValueError(
                "near_close_minutes must be an integer from 1 through 30"
            )
        if not isinstance(self.raw_chunk_rows, int) or self.raw_chunk_rows <= 0:
            raise ValueError("raw_chunk_rows must be a positive integer")
        if (
            not isinstance(self.lookback_sessions, int)
            or self.lookback_sessions < 0
        ):
            raise ValueError("lookback_sessions must be a nonnegative integer")
        if (
            not self.selection_times
            or tuple(sorted(set(self.selection_times))) != self.selection_times
            or any(
                not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d", t)
                for t in self.selection_times
            )
        ):
            raise ValueError(
                "selection_times must be unique, ordered HH:MM:SS values"
            )
        if (
            self.quote_interval not in QUOTE_INTERVALS
            or self.stock_venue not in {"utp_cta", "nqb"}
        ):
            raise ValueError("Unsupported quote interval or stock venue")

    def policy(self) -> dict:
        """Return the settings that determine the sample and its interpretation.

        Worker counts, parsing batch size, and requested date scope are excluded:
        changing them does not change an already collected session's sample.
        """
        names = (
            "option_rights",
            "target_dtes",
            "max_expirations_per_day",
            "moneyness_targets",
            "strikes_per_moneyness_target",
            "min_dte",
            "max_dte",
            "exchange_tz",
            "quote_interval",
            "near_close_minutes",
            "stock_venue",
            "selection_times",
            "max_stock_quote_age_seconds",
        )
        return {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            **{name: getattr(self, name) for name in names},
        }

    @property
    def policy_id(self) -> str:
        """The stable directory key for this sampling policy."""
        return provenance.digest_json(self.policy())[:20]
