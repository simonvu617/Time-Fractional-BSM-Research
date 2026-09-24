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

# v9 records each day's selected cross-section separately from the accumulated
# cohort. A weekly-only cache cannot supply missing daily selection evidence.
OUTPUT_SCHEMA_VERSION = "2026-09-18-daily-cross-sections-v9"

# Resolve from the repository root so splitting the package does not move
# existing caches into a new data directory.
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "multi_year_bsm_backtest_output"
)

# Pro permits finer data, but hourly remains the research sampling choice.
# Keep the existing minute-or-coarser options instead of adding tick downloads.
QUOTE_INTERVALS = ("1m", "5m", "10m", "15m", "30m", "1h")

# Pro's concurrency is shared across endpoints, even with mixed asset tiers.
# Historical access still depends on each separate product subscription.
# https://docs.thetadata.us/Articles/Data-And-Requests/Concurrent-Requests.html
PRO_MAX_INFLIGHT_REQUESTS = 8
PRO_HISTORY_START = "2012-06-01"
INDEX_HISTORY_STARTS = {
    "none": None,
    "value": "2023-01-01",
    "standard": "2022-01-01",
    "pro": "2017-01-01",
}
RATE_HISTORY_STARTS = {"free": "2024-01-01", "value": "1970-01-01"}

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
        symbol: Theta option root used in requests.
        asset_type: ETF, EQUITY, or INDEX for later cross-asset comparisons.
        universe_bucket: Study group, not historical index membership.
        sector_proxy: Descriptive sector label, not a dated classification.
        underlying_symbol: Price ticker when it differs from the option root.
    """

    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str
    underlying_symbol: str = ""

    @property
    def underlying(self) -> str:
        """The price ticker, which can differ from the option root."""
        return self.underlying_symbol or self.symbol

    @property
    def price_asset(self) -> str:
        """The Theta price endpoint family for this underlying."""
        return "index" if self.asset_type == "INDEX" else "stock"


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

# An opt-in benchmark: both option roots reference SPX, never SPY. Their
# underlying prices require a separate index entitlement. Historical product
# terms and adjusted deliverables still need verification before pricing.
INDEX_BENCHMARK = (
    SymbolConfig("SPX", "INDEX", "index_benchmark", "broad_market", "SPX"),
    SymbolConfig("SPXW", "INDEX", "index_benchmark", "broad_market", "SPX"),
)


@dataclasses.dataclass(frozen=True)
class CollectorConfig:
    """Validated sampling, subscription, and storage settings for a run.

    Dates and sampling targets are research choices. Constructing this frozen
    configuration checks their consistency without contacting Theta.

    Attributes:
        base_url: URL of the running Theta Terminal v3 service.
        start_date: Inclusive entry-window start; defaults to January 2017.
        end_date: Last date for selecting new contracts. Follow-up extends
            through their expirations, limited to completed historical dates.
        symbols: Underlyings selected from the fixed research universe.
        rate_symbols: Theta rate series shared across the underlyings.
        mode: Collect panels, references only, or date coverage only.
        index_subscription: Separately purchased index tier, or none.
        rate_subscription: Separately purchased rate tier; free starts in 2024.
        lookback_sessions: Earlier stock/rate exchange sessions to collect.
        option_rights: Call (right to buy at the strike) and/or put (right to
            sell).
        target_dtes: Preferred calendar-day distances to expiration.
        max_expirations_per_day: Maximum expirations in one day's new selection;
            previously enrolled expirations remain tracked separately.
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
        enrollment_frequency: Daily refresh by default; weekly is a smaller
            collection. Both follow enrolled contracts at quote_interval.
        max_stock_quote_age_seconds: Maximum age of a sampled stock record at
            selection time; does not measure the quote event's age.
        max_symbol_workers: Concurrent roots, each advancing months in order.
        max_batch_workers: Concurrent downloads within each batch.
        max_inflight_requests: Shared HTTP cap, at most eight for Pro.
        max_requests_per_second: Shared request-start limit; zero disables it.
        store_raw_payloads: Whether to keep full successful CSV responses in
            addition to Parquet. Failed response bytes are always retained.
        refresh_no_data: Whether to retry previously successful empty pulls.
        output_dir: Root for response caches, manifests, and coverage reports.
    """

    base_url: str = "http://127.0.0.1:25503/v3"
    start_date: str = "2017-01-01"
    end_date: str = "2025-12-31"
    symbols: tuple[SymbolConfig, ...] = tuple(UNIVERSE)
    rate_symbols: tuple[str, ...] = RATE_SYMBOLS
    mode: str = "panels"
    index_subscription: str = "none"
    rate_subscription: str = "free"

    # The buffer can support a first-day volatility estimate. It does not choose
    # a calibration window or guarantee continuous histories of the same
    # options.
    lookback_sessions: int = 60
    option_rights: tuple[str, ...] = ("call", "put")

    target_dtes: tuple[int, ...] = (7, 14, 30, 60, 120)
    max_expirations_per_day: int = 5

    # The denser central grid retains nearby strikes around S/K=1. This helps
    # comparisons within the narrow near-money bins used in An et al.; it does
    # not assert that every target has a distinct listed strike.
    # Moneyness is S/K: at S=$100, target 0.90 seeks K=$111.11. The 0.90/1.10
    # wings retain farther-out calls and puts for comparison without filling
    # the outer chain. Values above one are out of the money for puts.
    moneyness_targets: tuple[float, ...] = (
        0.90,
        0.95,
        0.975,
        0.99,
        1.00,
        1.01,
        1.025,
        1.05,
        1.10,
    )
    strikes_per_moneyness_target: int = 1
    min_dte: int = 7
    max_dte: int = 180
    exchange_tz: str = "America/New_York"
    quote_interval: str = "1h"
    near_close_minutes: int = 5
    raw_chunk_rows: int = 100_000
    stock_venue: str = "utp_cta"

    # One clock limits overlapping enrollment as prices move intraday. 10:30
    # aligns with the hourly grid and precedes scheduled early closes.
    selection_times: tuple[str, ...] = ("10:30:00",)
    enrollment_frequency: str = "daily"

    # A 12:30 sample cannot supply the 13:30 reference under this tolerance.
    # The timestamp does not reveal when the underlying quote last changed.
    max_stock_quote_age_seconds: int = 70
    max_symbol_workers: int = 4
    # A single reference batch can now occupy all eight Pro HTTP slots.
    # Four symbol workers provide independent monthly batches.
    max_batch_workers: int = 8
    # This account-wide budget is not eight slots per asset or worker. Lower it
    # when another client uses the same account; local locks cannot track that.
    max_inflight_requests: int = PRO_MAX_INFLIGHT_REQUESTS
    max_requests_per_second: float = 0.0
    store_raw_payloads: bool = False
    refresh_no_data: bool = False
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self):
        """Reject inconsistent settings before a run creates files.

        Raises:
            ValueError: A date, subscription, sampling, or resource setting is
                unsupported.
        """
        if not self.option_rights or set(self.option_rights) - {"call", "put"}:
            raise ValueError("option_rights must contain call and/or put")
        if not all(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            for value in (self.start_date, self.end_date)
        ):
            raise ValueError("Dates must use YYYY-MM-DD")
        start, end = pd.Timestamp(self.start_date), pd.Timestamp(self.end_date)
        if not pd.Timestamp(PRO_HISTORY_START) <= start <= end:
            raise ValueError(
                f"Dates must be ordered and start on or after {PRO_HISTORY_START}"
            )
        # A completed historical date can have an EOD report. Keep this check
        # here so CLI and direct Python runs follow the same rule.
        today = pd.Timestamp.now(self.exchange_tz).normalize().tz_localize(None)
        if end >= today:
            raise ValueError("The end date must be before today in New York")
        if self.mode not in {"panels", "references", "coverage"}:
            raise ValueError("Unsupported collection mode")
        if self.enrollment_frequency not in {"daily", "weekly"}:
            raise ValueError("enrollment_frequency must be daily or weekly")
        if self.index_subscription not in INDEX_HISTORY_STARTS:
            raise ValueError("Unsupported index subscription")
        if self.rate_subscription not in RATE_HISTORY_STARTS:
            raise ValueError("Unsupported interest-rate subscription")
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
            "max_symbol_workers",
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
        if (
            not isinstance(self.max_inflight_requests, int)
            or self.max_inflight_requests > PRO_MAX_INFLIGHT_REQUESTS
        ):
            raise ValueError(
                "Pro requires an integer limit of 1 through 8 shared HTTP requests"
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

        Cohort entry depends on the study's start and end. Different entry
        windows therefore get separate manifests, while identical raw requests
        can still reuse the shared cache.
        """
        names = (
            "start_date",
            "end_date",
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
            "enrollment_frequency",
            "max_stock_quote_age_seconds",
        )
        return {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "entry_schedule": "every_exchange_session_in_entry_window"
            if self.enrollment_frequency == "daily"
            else "first_exchange_session_of_week_in_entry_window",
            "cross_section_membership": "dated_selection_including_existing_contracts",
            "discovery_schedule": "enrollment_days_with_underlying_access",
            "option_oi_retention": "full_discovery_otherwise_tracked_date_windows",
            "contract_followup": "through_expiration",
            "option_eod_retention": "tracked_contract_date_windows",
            **{name: getattr(self, name) for name in names},
        }

    @property
    def index_history_start(self) -> str | None:
        """The index access boundary, or None without a subscription."""
        return INDEX_HISTORY_STARTS[self.index_subscription]

    @property
    def rate_history_start(self) -> str:
        """The configured interest-rate access boundary."""
        return RATE_HISTORY_STARTS[self.rate_subscription]

    @property
    def policy_id(self) -> str:
        """The stable directory key for this sampling policy."""
        return provenance.digest_json(self.policy())[:20]
