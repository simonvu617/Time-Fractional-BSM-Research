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
    # Announcement, ex-dividend, record, and payment dates have different roles.
    # Component/type fields prevent later work from blindly summing an event's
    # breakdown as if every row were a separate cash distribution.
    "corporate_dividend": (
        "announcement_date",
        "ex_dividend_date",
        "record_date",
        "payment_date",
        "amount",
        "event_code",
        "is_component",
        "distribution_type",
    ),
    # Splits change share counts and prices. These terms do not establish an
    # adjusted option's deliverable; the collector does not infer one.
    "corporate_split": (
        "effective_date",
        "before_shares",
        "after_shares",
        "split_ratio",
        "event_code",
    ),
}

# Validate against each endpoint's filtering date. An announcement or payment
# date can legitimately lie outside a request filtered on ex-dividend date.
REPORT_DATE_COLUMNS = {
    "interest_rate_eod": "created",
    "corporate_dividend": "ex_dividend_date",
    "corporate_split": "effective_date",
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
    """

    dataset: str
    endpoint: str
    params: dict
    vendor: str = dataclasses.field(default="ThetaData", init=False)

    retained_contract_keys: tuple[str, ...] | None = None

    def identity(self) -> dict:
        """Return the canonical request and retention policy for cache matching.

        Reordered or repeated retained keys have the same identity. Omitting
        retention preserves the identity of older full-response requests.
        """
        identity = dataclasses.asdict(self)
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
    return (
        frame["symbol"]
        + "|"
        + pd.to_datetime(frame["expiration"], format="mixed").dt.strftime(
            "%Y-%m-%d"
        )
        + "|"
        + frame["strike"].map(format_strike)
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
        kind: Endpoint kind: quote, price, open_interest, or eod.
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
    if kind in {"quote", "price"}:
        # History snapshots are the latest quotes at boundaries, not hourly
        # averages. Stock and option requests use the same grid for later
        # matching.
        params["interval"] = cfg.quote_interval
        dataset = f"{asset}_{kind}s_{cfg.quote_interval}"
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


def shared_day_requests(
    cfg: config.CollectorConfig, symbol: str, day: pd.Timestamp
) -> list[Request]:
    """Return seven stock, discovery, and option-report requests for a day."""
    # Use dated quote/trade contract lists for discovery. A current chain would
    # miss expired contracts; a trade list does not download individual trades.
    return [
        history_request(cfg, "stock", "quote", symbol, day),
        near_close_request(cfg, "stock", symbol, day),
        history_request(cfg, "stock", "eod", symbol, day),
        *[
            Request(
                dataset,
                f"/option/list/contracts/{kind}",
                {
                    "symbol": symbol,
                    "date": day.strftime("%Y%m%d"),
                    "max_dte": cfg.max_dte,
                    "format": "csv",
                },
            )
            for kind, dataset in (
                ("quote", "quoted_contracts"),
                ("trade", "traded_contracts"),
            )
        ],
        history_request(cfg, "option", "open_interest", symbol, day),
        history_request(cfg, "option", "eod", symbol, day),
    ]


def option_quote_requests(
    cfg: config.CollectorConfig,
    symbol: str,
    day: pd.Timestamp,
    selected: pd.DataFrame,
) -> list[Request]:
    """Build hourly and near-close bulk pulls for each selected expiration.

    Args:
        cfg: Collection settings.
        symbol: Underlying ticker.
        day: Exchange session date, without a time zone.
        selected: Selection table with expiration and contract_key columns.

    Returns:
        Two requests per selected expiration, each retaining only that
        expiration's selected contracts. An empty selection returns no pulls.
    """
    requests_to_make = []
    for expiration in sorted(selected["expiration"].unique()):
        family = {"expiration": expiration, "strike": "*", "right": "both"}

        # The HTTP response is bulk, but the stored sample is exact. Recording
        # these keys prevents a $100-only cache from serving a later $105-call
        # request.
        keys = tuple(
            sorted(
                selected.loc[
                    selected["expiration"].eq(expiration), "contract_key"
                ].unique()
            )
        )
        requests_to_make.extend(
            [
                dataclasses.replace(
                    history_request(
                        cfg, "option", "quote", symbol, day, family
                    ),
                    retained_contract_keys=keys,
                ),
                dataclasses.replace(
                    near_close_request(cfg, "option", symbol, day, expiration),
                    retained_contract_keys=keys,
                ),
            ]
        )
    return requests_to_make


def collection_windows(
    cfg: config.CollectorConfig, start: str, end: str
) -> dict:
    """Return the study dates and supporting history/event windows.

    Args:
        cfg: Earlier session count and maximum option maturity.
        start: Inclusive study start in YYYY-MM-DD form.
        end: Inclusive study end in YYYY-MM-DD form.

    Returns:
        Study and event boundaries, requested_history_start before access
        limits, history_start for accessible stock history, lookback_dates,
        and unavailable_lookback_dates. The later event window covers possible
        option expirations.

    Raises:
        ValueError: The lookback exceeds available calendar history.
    """
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
    return {
        "study_start": start,
        "study_end": end,
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
    rate_symbols: list[str],
    *,
    include_stock_lookback: bool = False,
) -> list[dict]:
    """Describe reference sessions excluded by the configured subscriptions.

    Args:
        cfg: Separate index/rate entitlements and sampling interval.
        windows: Study and buffer boundaries from collection_windows.
        rate_symbols: Requested rate identifiers; duplicates are removed.
        include_stock_lookback: Whether inaccessible stock buffer dates matter
            for this run.

    Returns:
        Known access gaps, including dates never requested. These are distinct
        from missing observations within successful vendor responses.
    """
    dates = list(
        exchange_calendar()
        .sessions_in_range(
            windows["requested_history_start"], windows["study_end"]
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
    if rate_symbols and rate_excluded:
        gaps.append(
            {
                "dataset": "interest_rate_eod",
                "symbols": sorted(set(rate_symbols)),
                "reason": "before_rate_subscription_history_start",
                "subscription": cfg.rate_subscription,
                "access_start": cfg.rate_history_start,
                "unrequested_eod_session_dates": rate_excluded,
            }
        )
    if include_stock_lookback and windows["unavailable_lookback_dates"]:
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
    return gaps


def reference_requests(
    cfg: config.CollectorConfig,
    symbols: list[config.SymbolConfig],
    start: str,
    end: str,
    rate_symbols: list[str],
    *,
    include_stock_lookback: bool = False,
):
    """Yield the shared Theta reference bundle and optional stock lookback.

    Args:
        cfg: Collection settings and separate subscription history limits.
        symbols: Underlyings needing corporate actions and stock history.
        start: Inclusive study start in YYYY-MM-DD form.
        end: Inclusive study end in YYYY-MM-DD form.
        rate_symbols: Theta rate series identifiers; duplicates are removed.
        include_stock_lookback: Whether to request earlier stock snapshots and
            EOD reports in addition to rates/actions/VIX.

    Yields:
        Request descriptors. Rates and VIX are shared across underlyings.
        Inaccessible dates are omitted and described by reference_access_gaps.
    """
    windows = collection_windows(cfg, start, end)
    window = {
        "start_date": windows["history_start"],
        "end_date": end,
        "format": "csv",
    }
    for symbol in symbols:
        for kind in ("dividend", "split"):
            # An option selected in December may expire after a January
            # dividend. Keep that event and its announcement date without
            # assuming it was known earlier.
            yield Request(
                f"corporate_{kind}",
                f"/corporate_action/{kind}",
                {
                    "symbol": symbol.symbol,
                    **window,
                    "end_date": windows["corporate_action_end"],
                },
            )
    # Rate access is separate from Stocks/Options Pro. Free rates begin in
    # 2024; a paid rate tier can cover the buffer even before stock history.
    rate_start = max(windows["requested_history_start"], cfg.rate_history_start)
    if rate_start <= end:
        for symbol in sorted(set(rate_symbols)):
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
            for day in (
                exchange_calendar()
                .sessions_in_range(intraday_start, end)
                .tz_localize(None)
            ):
                yield history_request(cfg, "index", "price", "VIX", day)
                yield near_close_request(cfg, "index", "VIX", day)
    if include_stock_lookback:
        for date in windows["lookback_dates"]:
            for symbol in symbols:
                for kind in ("quote", "eod"):
                    yield history_request(
                        cfg, "stock", kind, symbol.symbol, pd.Timestamp(date)
                    )
                yield near_close_request(
                    cfg, "stock", symbol.symbol, pd.Timestamp(date)
                )
