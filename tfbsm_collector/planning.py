# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Theta request identities, schemas, and exchange-calendar request planning.

Build requests here before downloading them. Request identities include the
retained option contracts, so a smaller saved sample cannot satisfy a broader
request. Request builders use the actual stock session, including early closes.
"""

import dataclasses
import decimal
import functools

import exchange_calendars as xcals
import pandas as pd

from tfbsm_collector import config, provenance

# A bid/ask is an advertised quote, not a trade. Preserve both sides, sizes,
# exchanges, and conditions so later work can examine spreads and quote quality.
QUOTE_FIELDS = (
    "bid_size",
    "bid_exchange",
    "bid",
    "bid_condition",
    "ask_size",
    "ask_exchange",
    "ask",
    "ask_condition",
)

CONTRACT_FIELDS = ("symbol", "expiration", "strike", "right")

REFERENCE_COLUMNS = {
    # Rates remain in reported percent: 4.25 means 4.25%, not 0.0425.
    "interest_rate_eod": ("created", "rate"),
}

# Rate reports have a date, not a verified intraday publication timestamp.
REPORT_DATE_COLUMNS = {
    "interest_rate_eod": "created",
}


@dataclasses.dataclass(frozen=True)
class Request:
    """A Theta request and the exact scope of its saved response.

    Attributes:
        dataset: Name used to group cached responses on disk.
        endpoint: Theta v3 URL path, beginning with a slash.
        params: Exact query parameters sent to Theta, including CSV format.
        vendor: Fixed vendor label recorded with the request.
        retained_contract_keys: Option identities retained in Parquet, or None
            to keep every row. This is never sent as a query parameter.
        retained_contract_windows: Tuples of contract key, first retained date,
            and last retained date. A monthly response keeps only these dated
            windows; first-day membership remains retrospective.
    """

    dataset: str
    endpoint: str
    params: dict
    vendor: str = dataclasses.field(default="ThetaData", init=False)

    retained_contract_keys: tuple[str, ...] | None = None
    retained_contract_windows: tuple[tuple[str, str, str], ...] | None = None

    def identity(self) -> dict:
        """Return the canonical request and retention policy for cache matching.

        Reordered or repeated retained keys have the same identity. Omitting
        retention preserves the identity of older full-response requests.
        """
        identity = dataclasses.asdict(self)
        if self.retained_contract_windows is None:
            identity.pop("retained_contract_windows")
        else:
            identity["retained_contract_windows"] = [
                list(window)
                for window in sorted(set(self.retained_contract_windows))
            ]
        if self.retained_contract_keys is None:
            # Older full-response caches omit this field. Keep their identities
            # stable.
            identity.pop("retained_contract_keys")
        else:
            # Selection row order cannot change which contracts were actually
            # retained.
            identity["retained_contract_keys"] = sorted(
                set(self.retained_contract_keys)
            )
        return identity

    @property
    def request_id(self) -> str:
        """The cache key for this request and its retained contracts."""
        return provenance.digest_json(self.identity())[:24]

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Required vendor columns; extra fields are retained."""
        kind = self.endpoint.rsplit("/", 1)[-1]
        if self.dataset in {"quoted_contracts", "traded_contracts"}:
            return CONTRACT_FIELDS
        if self.dataset in REFERENCE_COLUMNS:
            return REFERENCE_COLUMNS[self.dataset]
        if "/list/dates" in self.endpoint:
            return ("date",)
        fields = {
            "quote": ("timestamp", *QUOTE_FIELDS),
            "price": ("timestamp", "price"),
            "open_interest": ("timestamp", "open_interest"),
            "ohlc": (
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "count",
                "vwap",
            ),
            "eod": (
                "created",
                "last_trade",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "count",
            ),
        }
        return (
            (*CONTRACT_FIELDS,) if self.endpoint.startswith("/option/") else ()
        ) + fields[kind]

    @property
    def report_date_column(self) -> str | None:
        """The endpoint's date-only field, or None for market timestamps."""
        return (
            "date"
            if "/list/dates" in self.endpoint
            else REPORT_DATE_COLUMNS.get(self.dataset)
        )

    def observation_semantics(self) -> dict:
        """Return clock meanings and interpretation limits saved with the data.

        A sample boundary, quote event, report date, and publication time are
        different clocks. The metadata states which ones the response supports.
        """
        kind = self.endpoint.rsplit("/", 1)[-1]
        if self.report_date_column:
            return {
                "kind": "dated_report",
                "date_column": self.report_date_column,
                "publication_time_verified": False,
            }
        if "/list/contracts/" in self.endpoint:
            return {
                "kind": "date_wide_observed_contracts",
                "intraday_listing_time_verified": False,
            }
        if "/at_time/" in self.endpoint:
            return {
                "kind": "at_time_snapshot",
                "requested_time": self.params["time_of_day"],
                "timestamp_role": "vendor_at_time_timestamp",
                "quote_event_time_available": False,
                "purpose": "near_close_daily_comparison",
            }
        # A 10:30 sample may repeat a quote last updated at 10:12. That boundary
        # alone does not measure a trade gap or establish quote freshness.
        if kind == "quote":
            return {
                "kind": "sampled_quotes",
                "timestamp_role": "sample_boundary",
                "quote_event_time_available": False,
            }
        if kind == "ohlc":
            return {
                "kind": "trade_activity_bars",
                "timestamp_role": "bar_open",
                "trade_window": "bar_open <= trade_time < bar_open + interval",
                "aggregation": "Theta SIP trade-condition rules",
                "missing_bar_means_zero": False,
                "exact_trade_waiting_times_available": False,
            }
        # Open interest counts outstanding contracts at the previous session's
        # close, not today's trading volume. Missing reports do not mean zero.
        # https://docs.thetadata.us/operations/option_history_open_interest.html
        if kind == "open_interest":
            return {
                "kind": "open_interest_report",
                "describes": "previous_trading_session_close",
                "missing_report_means_zero": False,
            }
        if kind == "price":
            return {
                "kind": "sampled_index_prices",
                "unchanged_updates_may_be_omitted": True,
            }

        # Theta's later EOD report is separate from the near-close pricing
        # sample.
        # https://docs.thetadata.us/operations/option_history_eod.html
        return {
            "kind": "end_of_day_report",
            "is_regular_session_close_quote": False,
        }


def interval_seconds(interval: str) -> float:
    """Return seconds for a Theta interval ending in ms, s, m, or h."""
    # Theta's m means minutes; do not delegate this to ambiguous date aliases.
    if interval.endswith("ms"):
        return float(interval[:-2]) / 1000
    return float(interval[:-1]) * {"s": 1, "m": 60, "h": 3600}[interval[-1]]


def format_strike(value) -> str:
    """Return a canonical dollar strike without rounding its value.

    Args:
        value: Numeric or text strike, positive and exact to 0.001 dollars.

    Returns:
        Decimal text with unnecessary trailing zeros removed.

    Raises:
        ValueError: The strike is malformed, nonpositive, nonfinite, or more
            precise than the supported option strike grid.
    """
    try:
        # Decimal avoids binary-float artifacts in contract keys and request
        # URLs.
        strike = decimal.Decimal(str(value))
        if (
            not strike.is_finite()
            or strike <= 0
            or strike != strike.quantize(decimal.Decimal("0.001"))
        ):
            raise ValueError(f"Invalid option strike: {value!r}")
    except decimal.InvalidOperation as exc:
        raise ValueError(f"Invalid option strike: {value!r}") from exc
    return format(strike, ".3f").rstrip("0").rstrip(".")


def option_contract_keys(frame: pd.DataFrame) -> pd.Series:
    """Return symbol|expiration|strike|right identities for matching rows.

    Args:
        frame: Table containing CONTRACT_FIELDS with valid identities.

    Returns:
        A series aligned to the input index. Identity spellings are normalized
        only in these keys; the input vendor fields are unchanged.
    """
    # Hourly rows repeat the same strikes and expirations. Normalize each
    # distinct value once per batch without rounding prices or changing rows.
    strikes = {
        value: format_strike(value) for value in frame["strike"].unique()
    }
    expirations = pd.Series(frame["expiration"].unique())
    dates = dict(
        zip(
            expirations,
            pd.to_datetime(expirations, format="mixed").dt.strftime("%Y-%m-%d"),
        )
    )
    return (
        frame["symbol"]
        + "|"
        + frame["expiration"].map(dates)
        + "|"
        + frame["strike"].map(strikes)
        + "|"
        + frame["right"].str.lower().replace({"c": "call", "p": "put"})
    )


@functools.lru_cache(maxsize=1)
def exchange_calendar():
    """Return the cached XNYS calendar covering the study and its buffers."""
    next_year = pd.Timestamp.now("America/New_York").year + 1
    return xcals.get_calendar(
        "XNYS", start="2012-01-01", end=f"{next_year}-12-31"
    )


def session_bounds(
    day: pd.Timestamp, cfg: config.CollectorConfig
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the actual stock session's open and close in exchange time.

    Args:
        day: Exchange session date, without a time zone.
        cfg: Configuration supplying the exchange time zone.

    Returns:
        A pair of timezone-aware timestamps, including early-close times. Option
        collection uses this same underlying-session window.
    """
    # Use actual session bounds through holidays and daylight saving. Options
    # trading after the underlying stock session are outside this study window.
    calendar = exchange_calendar()
    return (
        calendar.session_open(day).tz_convert(cfg.exchange_tz),
        calendar.session_close(day).tz_convert(cfg.exchange_tz),
    )


def is_enrollment_day(day: pd.Timestamp, cfg: config.CollectorConfig) -> bool:
    """Select once per exchange week, including a partial first study week.

    Calendar closures move Monday's selection to the next exchange session.
    Missing vendor data do not move selection: that would make entry depend on
    later observations. Month boundaries and resume points do not reset a week.
    """
    if not pd.Timestamp(cfg.start_date) <= day <= pd.Timestamp(cfg.end_date):
        return False
    week_start = max(
        pd.Timestamp(cfg.start_date), day - pd.Timedelta(days=day.weekday())
    )
    return day == exchange_calendar().date_to_session(
        week_start, direction="next"
    )


def history_request(
    cfg: config.CollectorConfig,
    asset: str,
    kind: str,
    symbol: str,
    day: pd.Timestamp,
    contract: dict | None = None,
) -> Request:
    """Build one session's sampled prices, open interest, or EOD request.

    Args:
        cfg: Sampling interval, feed, and maturity limits.
        asset: Theta asset path: stock, option, or index.
        kind: Endpoint kind: quote, price, ohlc, open_interest, or eod.
        symbol: Underlying or index ticker.
        day: Exchange session date, without a time zone.
        contract: Option identity with expiration, strike, and right. A bulk
            expiration uses strike='*' and right='both'. None requests all
            contracts within max_dte for option open-interest/EOD reports.

    Returns:
        A request descriptor; no network or filesystem work is performed.

    Raises:
        ValueError: Option quotes have no contract, or a strike is invalid.
    """
    params = {"symbol": symbol, "date": day.strftime("%Y%m%d"), "format": "csv"}
    if asset == "option":
        if contract is None and kind in {"open_interest", "eod"}:
            # OI and EOD are one bulk report per underlying/day, capped at the
            # study's maximum maturity. Quiet contracts stay eligible for
            # discovery.
            params.update(
                expiration="*", strike="*", right="both", max_dte=cfg.max_dte
            )
        elif contract is None:
            raise ValueError("Option history requires an observed contract")
        else:
            strike = contract.get("strike", "*")
            params.update(
                expiration=contract["expiration"],
                strike="*" if strike == "*" else format_strike(strike),
                right=contract["right"],
            )
    if kind in {"quote", "price", "ohlc"}:
        # History snapshots are the latest quotes at boundaries, not hourly
        # averages. Stock and option requests use the same grid for later
        # matching.
        params["interval"] = cfg.quote_interval
        dataset = f"{asset}_{kind}{'' if kind == 'ohlc' else 's'}_{cfg.quote_interval}"
    else:
        dataset = f"{asset}_{kind}"
    if kind == "eod":
        # EOD endpoints need a date range even for one day. Their volume/count
        # measure trading activity, not the number of quote changes.
        params.pop("date")
        params.update(
            start_date=day.strftime("%Y%m%d"), end_date=day.strftime("%Y%m%d")
        )
    elif kind != "open_interest":
        opened, closed = session_bounds(day, cfg)
        params.update(
            start_time=opened.strftime("%H:%M:%S"),
            end_time=closed.strftime("%H:%M:%S"),
        )
        if asset == "stock":
            params["venue"] = cfg.stock_venue
    return Request(dataset, f"/{asset}/history/{kind}", params)


def near_close_request(
    cfg: config.CollectorConfig,
    asset: str,
    symbol: str,
    day: pd.Timestamp,
    expiration: str | None = None,
) -> Request:
    """Build the additional snapshot relative to the actual session close.

    Args:
        cfg: Minutes before close, exchange time zone, and stock feed.
        asset: Theta asset path: stock, option, or index.
        symbol: Underlying or index ticker.
        day: Exchange session date, without a time zone.
        expiration: Selected option expiration; unused for stock/index data.

    Returns:
        An at-time quote request, or price request for an index.

    Raises:
        ValueError: An option request has no selected expiration.
    """
    _, closed = session_bounds(day, cfg)
    # The daily sample is 15:55 normally and 12:55 on a 13:00 close. An hourly
    # grid and the later EOD report cannot substitute for this at-time snapshot.
    at = closed - pd.Timedelta(minutes=cfg.near_close_minutes)
    kind = "price" if asset == "index" else "quote"
    params = {
        "symbol": symbol,
        "start_date": day.strftime("%Y%m%d"),
        "end_date": day.strftime("%Y%m%d"),
        "time_of_day": at.strftime("%H:%M:%S.000"),
        "format": "csv",
    }
    if asset == "option":
        if expiration is None:
            raise ValueError(
                "Option near-close requests require a selected expiration"
            )
        params.update(expiration=expiration, strike="*", right="both")
    elif asset == "stock":
        params["venue"] = cfg.stock_venue
    return Request(
        f"{asset}_{kind}s_near_close", f"/{asset}/at_time/{kind}", params
    )


def date_batches(days, cfg: config.CollectorConfig, *, intraday: bool = True):
    """Yield contiguous session ranges within a month and one session length.

    Split around excluded dates and early closes: a monthly 15:55 request must
    never replace the 12:55 observation on a half day. Calendar months keep all
    sampled requests within Theta's documented one-month limit.
    """
    batch, previous_key, previous_position = [], None, None
    calendar = exchange_calendar().sessions.tz_localize(None)
    for day in sorted(pd.DatetimeIndex(days)):
        position = calendar.get_loc(day)
        key = (
            day.strftime("%Y-%m"),
            session_bounds(day, cfg)[1].strftime("%H:%M") if intraday else "",
        )
        if batch and (key != previous_key or position != previous_position + 1):
            yield pd.DatetimeIndex(batch)
            batch = []
        batch.append(day)
        previous_key, previous_position = key, position
    if batch:
        yield pd.DatetimeIndex(batch)


def ranged(request: Request, days: pd.DatetimeIndex) -> Request:
    """Extend a single-session descriptor to one supported inclusive range."""
    if len(days) == 1:
        return request
    params = {
        key: value for key, value in request.params.items() if key != "date"
    }
    params.update(
        start_date=days[0].strftime("%Y%m%d"),
        end_date=days[-1].strftime("%Y%m%d"),
    )
    return dataclasses.replace(request, params=params)


def request_days(request: Request) -> pd.DatetimeIndex:
    """Return the exchange sessions addressed by a dated request."""
    start = request.params.get("date", request.params.get("start_date"))
    end = request.params.get("date", request.params.get("end_date"))
    return exchange_calendar().sessions_in_range(start, end).tz_localize(None)


def underlying_requests(
    cfg: config.CollectorConfig,
    symbol: config.SymbolConfig,
    days,
    unavailable: dict | None = None,
) -> list[Request]:
    """Batch underlying observations, honoring index access and date catalogues.

    A successful catalogue can exclude quote or trade dates; a failed catalogue
    cannot justify skipping them. Index prices have no trade volume, so no
    synthetic index OHLC activity is manufactured.
    """
    unavailable = unavailable or {}
    asset, ticker = symbol.price_asset, symbol.underlying
    days = pd.DatetimeIndex(days)
    if asset == "index":
        if cfg.index_history_start is None:
            return []
        days = days[days >= pd.Timestamp(cfg.index_history_start)]
    requests_to_make = []
    kinds = (
        ("price", "near_close", "eod")
        if asset == "index"
        else ("quote", "near_close", "ohlc", "eod")
    )
    for kind in kinds:
        catalogue_kind = (
            "price"
            if asset == "index"
            else ("trade" if kind == "ohlc" else "quote")
        )
        excluded = (
            unavailable.get((ticker, catalogue_kind), set())
            if kind != "eod"
            else set()
        )
        wanted = [d for d in days if str(d.date()) not in excluded]
        for batch in date_batches(wanted, cfg, intraday=kind != "eod"):
            request = (
                near_close_request(cfg, asset, ticker, batch[0])
                if kind == "near_close"
                else history_request(cfg, asset, kind, ticker, batch[0])
            )
            requests_to_make.append(ranged(request, batch))
    return requests_to_make


def discovery_requests(
    cfg: config.CollectorConfig, symbol: str, days
) -> list[Request]:
    """Collect full candidate evidence only on weekly enrollment dates.

    Listing requests remain dated even though their small responses are packed
    together on disk. An expiration observed later must not enter an earlier
    day's candidate universe. Open interest also discovers quiet contracts that
    need not appear in that day's quote or trade list.
    """
    requests_to_make = []
    for day in days:
        if not is_enrollment_day(day, cfg):
            continue
        requests_to_make.append(
            history_request(cfg, "option", "open_interest", symbol, day)
        )
        for kind in ("quote", "trade"):
            requests_to_make.append(
                Request(
                    "quoted_contracts"
                    if kind == "quote"
                    else "traded_contracts",
                    f"/option/list/contracts/{kind}",
                    {
                        "symbol": symbol,
                        "date": day.strftime("%Y%m%d"),
                        "max_dte": cfg.max_dte,
                        "format": "csv",
                    },
                )
            )
    return requests_to_make


def _cohort_windows(cohort: pd.DataFrame, days) -> tuple:
    """Clip each contract's retained dates to a request's actual date range."""
    first, last = str(days[0].date()), str(days[-1].date())
    return tuple(
        sorted(
            (
                row.contract_key,
                max(row.first_selected_date, first),
                min(row.expiration, last),
            )
            for row in cohort.itertuples(index=False)
            if row.first_selected_date <= last and row.expiration >= first
        )
    )


def followup_requests(
    cfg: config.CollectorConfig,
    symbol: str,
    days,
    cohort: pd.DataFrame,
    *,
    discovery_days=(),
) -> list[Request]:
    """Request daily cohort observations, reusing enrollment-day OI.

    Entry DTE and moneyness limits apply only when a contract is first chosen.
    Continue requesting it through expiration even when it becomes deep ITM/OTM,
    has no new discovery row, or falls below the entry maturity threshold.

    Args:
        cfg: Sampling and maturity settings.
        symbol: Option root.
        days: Exchange sessions to follow.
        cohort: Enrolled contracts with their first selection dates.
        discovery_days: Dates already requested as full OI discovery snapshots.
            A failed snapshot is retried with that discovery request, not hidden
            by a narrower follow-up response.

    Returns:
        Requests retaining only the cohort's tracked date windows.
    """
    requests_to_make = []
    discovered = set(discovery_days)
    for day in days:
        windows = _cohort_windows(cohort, [day])
        if windows and day not in discovered:
            # Keep one bulk OI download per day; filtering stored rows avoids
            # one HTTP request per contract. A skipped enrollment snapshot must
            # not interrupt OI follow-up for an existing cohort.
            requests_to_make.append(
                dataclasses.replace(
                    history_request(
                        cfg, "option", "open_interest", symbol, day
                    ),
                    retained_contract_windows=windows,
                )
            )
    # Keep the efficient bulk EOD request, but save only enrolled identities on
    # their tracked dates. Weekly discovery documents the broader universe.
    # EOD is a report, so it can span early closes without changing its clock.
    for batch in date_batches(days, cfg, intraday=False):
        windows = _cohort_windows(cohort, batch)
        if windows:
            requests_to_make.append(
                dataclasses.replace(
                    ranged(
                        history_request(cfg, "option", "eod", symbol, batch[0]),
                        batch,
                    ),
                    retained_contract_windows=windows,
                )
            )
    for expiration, family in cohort.groupby("expiration", sort=True):
        active = [
            d
            for d in days
            if family["first_selected_date"].min()
            <= str(d.date())
            <= expiration
        ]
        for batch in date_batches(active, cfg):
            windows = _cohort_windows(family, batch)
            contract = {
                "expiration": expiration,
                "strike": "*",
                "right": "both",
            }
            for kind in ("quote", "ohlc", "near_close"):
                request = (
                    near_close_request(
                        cfg, "option", symbol, batch[0], expiration
                    )
                    if kind == "near_close"
                    else history_request(
                        cfg, "option", kind, symbol, batch[0], contract
                    )
                )
                requests_to_make.append(
                    dataclasses.replace(
                        ranged(request, batch),
                        retained_contract_windows=windows,
                    )
                )
    return requests_to_make


def collection_windows(cfg: config.CollectorConfig) -> dict:
    """Return the study dates and supporting history/event windows.

    Args:
        cfg: Study dates, earlier session count, and maximum option maturity.

    Returns:
        Study and event boundaries, requested_history_start before access
        limits, history_start for accessible stock history, lookback_dates,
        and unavailable_lookback_dates. The later event window covers possible
        option expirations.

    Raises:
        ValueError: The lookback exceeds available calendar history.
    """
    start, end = cfg.start_date, cfg.end_date
    sessions = exchange_calendar().sessions.tz_localize(None)
    before = sessions[sessions < pd.Timestamp(start)]
    if cfg.lookback_sessions > len(before):
        raise ValueError(
            "lookback_sessions exceeds the available exchange-calendar history"
        )
    # Count exchange sessions, not calendar days, for the earlier stock buffer.
    lookback = (
        before[-cfg.lookback_sessions :]
        if cfg.lookback_sessions
        else before[:0]
    )
    requested_history_start = (
        str(lookback[0].date()) if len(lookback) else start
    )
    # A study starting at Pro's first date cannot have a full stock buffer.
    # Record the omitted sessions instead of causing a permission failure or
    # pretending that fewer observations constitute the requested lookback.
    accessible = lookback >= pd.Timestamp(config.PRO_HISTORY_START)
    followup_end = pd.Timestamp(end) + pd.Timedelta(days=cfg.max_dte)
    yesterday = pd.Timestamp.now(cfg.exchange_tz).normalize().tz_localize(
        None
    ) - pd.Timedelta(days=1)
    return {
        "study_start": start,
        "study_end": end,
        "followup_end_bound": str(followup_end.date()),
        "available_followup_end": str(min(followup_end, yesterday).date()),
        "future_followup_possible": followup_end > yesterday,
        "requested_history_start": requested_history_start,
        "history_start": max(requested_history_start, config.PRO_HISTORY_START),
        "corporate_action_end": str(
            (pd.Timestamp(end) + pd.Timedelta(days=cfg.max_dte)).date()
        ),
        "lookback_dates": list(lookback[accessible].strftime("%Y-%m-%d")),
        "unavailable_lookback_dates": list(
            lookback[~accessible].strftime("%Y-%m-%d")
        ),
    }


def reference_access_gaps(
    cfg: config.CollectorConfig,
    windows: dict,
) -> list[dict]:
    """Describe references excluded by access limits or unavailable endpoints.

    Args:
        cfg: Requested rates, separate entitlements, and sampling interval.
        windows: Study and buffer boundaries from collection_windows.

    Returns:
        Known access gaps, including dates never requested. These are distinct
        from missing observations within successful vendor responses.
    """
    dates = list(
        exchange_calendar()
        .sessions_in_range(
            windows["requested_history_start"],
            windows["available_followup_end"]
            if cfg.mode == "panels"
            else windows["study_end"],
        )
        .strftime("%Y-%m-%d")
    )
    gaps = []
    index_excluded = [
        date
        for date in dates
        if cfg.index_history_start is None or date < cfg.index_history_start
    ]
    if index_excluded:
        gaps.append(
            {
                "symbol": "VIX",
                "symbols": sorted(
                    {"VIX"}
                    | {
                        s.underlying
                        for s in cfg.symbols
                        if s.price_asset == "index"
                    }
                ),
                "reason": "index_subscription_unavailable"
                if cfg.index_history_start is None
                else "before_index_subscription_history_start",
                "subscription": cfg.index_subscription,
                "access_start": cfg.index_history_start,
                "unrequested_eod_session_dates": index_excluded,
                "unrequested_intraday_session_dates": [
                    date
                    for date in index_excluded
                    if date >= windows["study_start"]
                ],
            }
        )
    rate_excluded = [date for date in dates if date < cfg.rate_history_start]
    if cfg.rate_symbols and rate_excluded:
        gaps.append(
            {
                "dataset": "interest_rate_eod",
                "symbols": sorted(set(cfg.rate_symbols)),
                "reason": "before_rate_subscription_history_start",
                "subscription": cfg.rate_subscription,
                "access_start": cfg.rate_history_start,
                "unrequested_eod_session_dates": rate_excluded,
            }
        )
    if cfg.mode == "panels" and windows["unavailable_lookback_dates"]:
        gaps.append(
            {
                "dataset": "stock_lookback",
                "reason": "before_stock_pro_history_start",
                "access_start": config.PRO_HISTORY_START,
                "unrequested_session_dates": windows[
                    "unavailable_lookback_dates"
                ],
            }
        )
    if any(s.price_asset == "stock" for s in cfg.symbols):
        # The September 2026 live check returned 404 for both corporate-action
        # routes. Theta's v3 migration guide marks them as coming soon. Keep the
        # missing event window visible; absent dividends must never imply zero.
        # An option can expire after the study end, so the gap includes max_dte.
        for dataset in ("corporate_dividend", "corporate_split"):
            gaps.append(
                {
                    "dataset": dataset,
                    "symbols": [
                        symbol.underlying
                        for symbol in cfg.symbols
                        if symbol.price_asset == "stock"
                    ],
                    "reason": "theta_v3_endpoint_unavailable",
                    "unrequested_start_date": windows["history_start"],
                    "unrequested_end_date": windows["corporate_action_end"],
                }
            )
    if cfg.symbols:
        gaps.append(
            {
                "dataset": "option_contract_terms",
                "symbols": [s.symbol for s in cfg.symbols],
                "reason": "historical_contract_terms_not_supplied_by_current_Theta_API",
                "missing_fields": [
                    "verified_multiplier",
                    "deliverable",
                    "last_trading_timestamp",
                    "verified_exercise_and_settlement_terms",
                ],
            }
        )
    return gaps


def reference_requests(cfg: config.CollectorConfig):
    """Yield the shared Theta reference bundle and optional stock lookback.

    Args:
        cfg: Requested symbols/dates, sampling, and subscription history limits.

    Yields:
        Request descriptors. Rates and VIX are shared across underlyings.
        Inaccessible dates are omitted and described by reference_access_gaps.
    """
    start = cfg.start_date
    windows = collection_windows(cfg)
    end = (
        windows["available_followup_end"]
        if cfg.mode == "panels"
        else cfg.end_date
    )
    window = {
        "start_date": windows["history_start"],
        "end_date": end,
        "format": "csv",
    }
    # Rate access is separate from Stocks/Options Pro. Free rates begin in
    # 2024; a paid rate tier can cover the buffer even before stock history.
    rate_start = max(windows["requested_history_start"], cfg.rate_history_start)
    if rate_start <= end:
        for symbol in sorted(set(cfg.rate_symbols)):
            yield Request(
                "interest_rate_eod",
                "/interest_rate/history/eod",
                {"symbol": symbol, **window, "start_date": rate_start},
            )

    # Stocks/Options Pro does not include indices. Skip inaccessible requests
    # so a missing VIX entitlement cannot stop the stock/option collection.
    if cfg.index_history_start is not None:
        index_start = max(
            windows["requested_history_start"], cfg.index_history_start
        )
        if index_start <= end:
            yield Request(
                "index_eod",
                "/index/history/eod",
                {"symbol": "VIX", **window, "start_date": index_start},
            )
        intraday_start = max(start, cfg.index_history_start)
        if intraday_start <= end:
            # VIX provides S&P 500 volatility context, not each stock's
            # volatility. Intraday values avoid using EOD data that morning.
            days = (
                exchange_calendar()
                .sessions_in_range(intraday_start, end)
                .tz_localize(None)
            )
            for batch in date_batches(days, cfg):
                yield ranged(
                    history_request(cfg, "index", "price", "VIX", batch[0]),
                    batch,
                )
                yield ranged(
                    near_close_request(cfg, "index", "VIX", batch[0]), batch
                )
    if cfg.mode == "panels":
        days = pd.DatetimeIndex(windows["lookback_dates"])
        seen = set()
        for symbol in cfg.symbols:
            if (symbol.price_asset, symbol.underlying) in seen:
                continue
            seen.add((symbol.price_asset, symbol.underlying))
            yield from underlying_requests(cfg, symbol, days)
