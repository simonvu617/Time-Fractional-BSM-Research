# READING GUIDE
# This file downloads and records inputs for the research. It does not fit a
# model or decide whether TFBSM performs better than classical Black-Scholes.
#
# For the overall sequence, start at main() near the bottom. For the work done
# on one stock/ETF on one trading day, read Collector.collect_day(). The other
# functions supply the requests, contract selection, storage, and bookkeeping.
#
# Normal run:
#   settings -> available-date check -> dividends/splits/rates/VIX
#            -> each trading day and underlying -> coverage/run reports
# One underlying on one day:
#   stock quotes + dated option list + stock trades/EOD
#   -> stock prices at the selection times -> selected option contracts
#   -> option quotes + option trades + open interest -> saved session record
#
# Vocabulary used throughout:
#   underlying: the stock or ETF on which an option is written.
#   strike: the option's agreed exercise price; expiration: its expiry date.
#   symbol-day/session: one underlying on one exchange trading day.
#   quote: displayed buying/selling prices; trade: an actual reported transaction.
#   frame: a pandas table in memory, with named columns and rows.
#   cache: previously saved responses that can be reused instead of downloaded.
#   manifest/ledger: a JSON record of what was requested, saved, or missing.
#   receipt: a file's location, size, and fingerprint, used to check reuse.
# The numbered sections below follow the file's order, not its execution order.

"""Historical stock/option collection, kept in one file for review.

Python 3.11+; dependencies: exchange-calendars, numpy, pandas>=2, pyarrow, requests.
ThetaData is the sole market/reference data vendor; there are no provider fallbacks.

Examples (Theta Terminal v3 must be running for collection):
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --plan
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-02 --quote-interval tick
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --references-only
  python collector.py --symbols SPY AAPL --start 2018-01-01 --end 2025-12-31 --coverage-only

The default universe and DTE/S/K sampling grid are retained. Discovery uses the
contracts quoted on each historical date, not today's chain. The three stock
reference times choose what to download; they are not evaluation samples.
A contract does not need trades, positive OI, or narrow spreads to be collected.

Raw Parquet preserves vendor columns, values, row order, duplicates, conditions,
and exchange/sequence codes. CSV values remain strings to avoid rounding or
silently coercing bad values. Added collector_*_utc columns are parsed clocks;
the vendor clocks remain unchanged. Optional raw CSV saves the response bytes.
Each request has metadata, a checksum, retrieval time, and coverage diagnostics.
No pricing, waiting-time features, regimes, sample filters, or yield proxies run.

Output (under --output-dir):
  raw_cache/<dataset>/.../request=<id>/latest.json (last successful response)
  raw_cache/<dataset>/.../request=<id>/responses/<id>/{data.parquet,meta.json}
  collection/<policy-id>/sessions/<symbol-day>.json
  collection/<policy-id>/contracts/<symbol-day>.parquet
  collection/<policy-id>/availability.csv
  collection/<policy-id>/runs/<run-id>.json
  references/<run-id>.json (dividends, splits, rates, VIX; standard collection)
  coverage/<run-id>.json (vendor date lists and missing sessions; also --coverage-only)
Successful raw requests are reused across policy changes. A day is resumable
only after its selected-contract file and every referenced request are verified.
Refreshes save a new response; earlier run receipts continue to identify the
exact files they used. Failed refreshes never replace the last successful cache.
"complete" means the requests finished, not that the vendor supplied full coverage.
"no_data" is retained separately from request errors. --refresh-no-data retries it.
Permission/configuration failures stop new requests; temporary connection errors
get bounded retries. Ctrl+C cancels queued work while active requests finish or
time out. Handled interruptions still publish coverage and run records.

Official API specification: https://docs.thetadata.us/openapiv3.yaml
Relevant pages under https://docs.thetadata.us/operations/:
  {stock,option}_history_{quote,trade_quote}.html
  option_list_contracts.html; option_history_open_interest.html
  stock_history_eod.html; interest_rate_history_eod.html
Corporate-action schemas: /corporate_action/{dividend,split} in the OpenAPI spec.
Error codes: https://docs.thetadata.us/Articles/Errors-Exchanges-Conditions/Error-Codes.html

Limits: the quote universe is a date-wide observation, not an intraday listing
snapshot. 1s quotes are samples; use --quote-interval tick for quote events when
your subscription supports them. Requests cover the exchange's regular session.
OI is requested on the report date and describes the previous session's close.
Theta EOD is generated around 17:15 ET, not a 16:00 quote. The reference bundle
includes SOFR and every documented Treasury tenor, plus VIX EOD and intraday
prices at the requested quote interval. Rates keep vendor percentage units and
report dates; no discount curve is inferred. Corporate actions retain dates,
missing amounts, distribution components, and split ratios (before / after).
Action range filters use ex-dividend / effective dates, not announcement dates.
Date-only reports never become fabricated publication timestamps or vintages.
Index updates with unchanged prices can be omitted by Theta; absence of an index
update is not a measurement of underlying trade inactivity.

Coverage limits: Theta documents missing pre-2020 underlying history for SPY and
other CTA-only symbols. --coverage-only queries current vendor date lists for
each stock and VIX; it does not prove intraday completeness or plan entitlements.
Adjusted option deliverables, historical symbol mappings, and reference-data
vintages remain unverified. Missing records are never filled from another source.
Exit status: 0 completed requests; 1 request/processing errors; 2 observed
coverage gaps (including unavailable symbol-days or missing rate/index pulls).
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4
from urllib.parse import urlsplit

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

if os.name == "nt":
    import msvcrt
else:
    import fcntl


# 1. Collection settings and the existing research universe.
# Schema versions identify the layout/meaning of our saved files. They let the
# resume checks distinguish compatible data from an older output format.
OUTPUT_SCHEMA_VERSION = "2026-09-07-raw-collection-v1"
RAW_SCHEMA_VERSION = 1
# Keep the original default root so existing raw caches can be reused.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "multi_year_bsm_backtest_output"
QUOTE_INTERVALS = ("tick", "10ms", "100ms", "500ms", "1s", "5s", "10s", "15s",
                   "30s", "1m", "5m", "10m", "15m", "30m", "1h")
# Both statuses mean a request finished successfully. "no_data" means Theta
# returned no rows; it does not mean that an asset had zero prices or activity.
GOOD_REQUEST_STATUSES = {"available", "no_data"}
# These are minimum expected vendor columns, not a list of columns to keep.
# Extra vendor fields also survive in raw storage.
QUOTE_FIELDS = ("bid_size", "bid_exchange", "bid", "bid_condition",
                "ask_size", "ask_exchange", "ask", "ask_condition")
TRADE_FIELDS = ("trade_timestamp", "quote_timestamp", "sequence", "condition", "size",
                "exchange", "price", *QUOTE_FIELDS)
# An option's identity needs all four fields. "right" means call or put.
CONTRACT_FIELDS = ("symbol", "expiration", "strike", "right")
SELECTION_COLUMNS = (*CONTRACT_FIELDS, "contract_key", "dte_days", "selection_times")
# SOFR is an overnight reference rate; Treasury suffixes specify months/years.
# Collecting several maturities keeps the later choice of a pricing rate open.
RATE_SYMBOLS = ("SOFR", "TREASURY_M1", "TREASURY_M3", "TREASURY_M6", "TREASURY_Y1",
                "TREASURY_Y2", "TREASURY_Y3", "TREASURY_Y5", "TREASURY_Y7",
                "TREASURY_Y10", "TREASURY_Y20", "TREASURY_Y30")
REFERENCE_COLUMNS = {
    "interest_rate_eod": ("created", "rate"),
    "corporate_dividend": ("announcement_date", "ex_dividend_date", "record_date", "payment_date",
                           "amount", "event_code", "is_component", "distribution_type"),
    "corporate_split": ("effective_date", "before_shares", "after_shares", "split_ratio", "event_code"),
}
# Each date-range check must use the date on which that endpoint filters.
# A dividend's announcement/payment dates can legitimately fall outside it.
REPORT_DATE_COLUMNS = {"interest_rate_eod": "created", "corporate_dividend": "ex_dividend_date",
                       "corporate_split": "effective_date"}


@dataclass(frozen=True)
class SymbolConfig:
    # Descriptive research labels for one underlying. They are saved with its
    # session record; they do not create vendor requests or compute features.
    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str


# This is the explicit study list, not a historical list of all listed assets.
# --symbols selects a subset. A listed symbol can still lack data on some dates.
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


@dataclass(frozen=True)
class CollectorConfig:
    # Requests go through the user's local Theta Terminal v3 application.
    base_url: str = "http://127.0.0.1:25503/v3"
    # Default study bounds. The CLI can request a smaller slice within them.
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    option_rights: tuple[str, ...] = ("call", "put")
    # DTE = calendar days to expiration. Choose actual listed expirations near
    # these targets, with an overall limit on the number chosen per day.
    target_dtes: tuple[int, ...] = (7, 14, 30, 60, 120)
    max_expirations_per_day: int = 5
    # Moneyness here is S/K: underlying price divided by option strike.
    # At S=$100, target 0.80 implies K=$125; target 1.00 implies K=$100.
    # We choose nearby listed strikes, not invented contracts at those prices.
    moneyness_targets: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
    strikes_per_moneyness_target: int = 1
    # Expirations outside this window cannot enter the download selection.
    min_dte: int = 7
    max_dte: int = 180
    exchange_tz: str = "America/New_York"
    # "1s" requests sampled quotes. "tick" requests quote events. Trades are
    # requested as individual events regardless of this quote setting.
    quote_interval: str = "1s"
    stock_venue: str = "utp_cta"
    # At these New York times, use a recent stock quote to choose option strikes.
    # These times select downloads; all requested session observations are saved.
    selection_times: tuple[str, ...] = ("10:30:00", "13:00:00", "15:00:00")
    # A selection reference cannot use a stock quote older than this many seconds.
    max_stock_quote_age_seconds: int = 70
    # Ask Theta to match a trade to a quote strictly before its timestamp.
    trade_quote_exclusive: bool = True
    # Workers overlap downloads: first across symbol-days, then option requests.
    # The shared client below still caps total requests in flight and start rate.
    max_symbol_day_workers: int = 2
    max_contract_workers: int = 4
    max_requests_per_second: float = 8.0
    max_inflight_requests: int = 8
    # Parquet tables and request metadata are always saved. This flag also keeps
    # the exact successful response bytes, which use additional disk space.
    store_raw_payloads: bool = False
    # Normally an empty response is cached too. Enable this to ask Theta again.
    refresh_no_data: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self):
        # Reject inconsistent settings when the configuration is constructed,
        # before creating files or sending any requests.
        if not self.option_rights or set(self.option_rights) - {"call", "put"}:
            raise ValueError("option_rights must contain call and/or put")
        if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
            raise ValueError("start_date must not follow end_date")
        if not 0 <= self.min_dte <= self.max_dte or not self.target_dtes:
            raise ValueError("Provide target_dtes and an ordered, nonnegative DTE range")
        if any(d < 0 for d in self.target_dtes):
            raise ValueError("target_dtes must be nonnegative")
        if not self.moneyness_targets or any(not np.isfinite(x) or x <= 0 for x in self.moneyness_targets):
            raise ValueError("moneyness_targets must be finite and positive")
        for name in ("max_expirations_per_day", "strikes_per_moneyness_target",
                     "max_symbol_day_workers", "max_contract_workers",
                     "max_requests_per_second", "max_inflight_requests", "max_stock_quote_age_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (not self.selection_times or tuple(sorted(set(self.selection_times))) != self.selection_times
                or any(not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d", t) for t in self.selection_times)):
            raise ValueError("selection_times must be unique, ordered HH:MM:SS values")
        if self.quote_interval not in QUOTE_INTERVALS or self.stock_venue not in {"utp_cta", "nqb"}:
            raise ValueError("Unsupported quote interval or stock venue")

    def policy(self) -> dict:
        # Worker counts, scope, and the separate reference bundle do not change
        # which contracts are selected or invalidate already collected sessions.
        names = ("option_rights", "target_dtes", "max_expirations_per_day", "moneyness_targets",
                 "strikes_per_moneyness_target", "min_dte", "max_dte", "exchange_tz",
                 "quote_interval", "stock_venue", "selection_times", "max_stock_quote_age_seconds",
                 "trade_quote_exclusive")
        return {"output_schema_version": OUTPUT_SCHEMA_VERSION,
                **{name: getattr(self, name) for name in names}}

    @property
    def policy_id(self) -> str:
        # A short fingerprint names the folder for this selection policy.
        # Changing selection settings keeps the old policy's results separate.
        return digest_json(self.policy())[:20]


# 2. Describe a single Theta request before deciding whether to download it.
@dataclass(frozen=True)
class Request:
    # dataset names our output group; endpoint is Theta's URL path; params is
    # the exact set of URL arguments (symbol, date, contract, interval, etc.).
    dataset: str
    endpoint: str
    params: dict
    vendor: str = field(default="ThetaData", init=False)

    def identity(self) -> dict:
        return asdict(self)

    @property
    def request_id(self) -> str:
        # Identical requests get the same cache key even in different runs.
        # Changing a date, strike, interval, or another argument changes this key.
        return digest_json(self.identity())[:24]

    @property
    def required_columns(self) -> tuple[str, ...]:
        # A successful HTTP response must also look like the requested table.
        # This catches error pages or incompatible schemas before cache reuse.
        kind = self.endpoint.rsplit("/", 1)[-1]
        if self.dataset == "quoted_contracts":
            return CONTRACT_FIELDS
        if self.dataset in REFERENCE_COLUMNS:
            return REFERENCE_COLUMNS[self.dataset]
        if "/list/dates" in self.endpoint:
            return ("date",)
        fields = {"quote": ("timestamp", *QUOTE_FIELDS), "trade_quote": TRADE_FIELDS,
                  "price": ("timestamp", "price"),
                  "open_interest": ("timestamp", "open_interest"),
                  "eod": ("created", "last_trade", "open", "high", "low", "close", "volume", "count")}
        return ((*CONTRACT_FIELDS,) if self.endpoint.startswith("/option/") else ()) + fields[kind]


# 3. File writing, receipts, and locking.
# JSON holds readable metadata; Parquet holds compressed column-based tables.
# UTC is the common clock used for collection times and parsed market timestamps.
def utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def digest_json(value) -> str:
    # Sorting dictionary keys makes the fingerprint independent of key order.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    # Fingerprint the actual file bytes, separately from the request's identity.
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def atomic_output(path: Path):
    # Write a complete temporary file beside the destination, then replace it.
    # A failed write therefore cannot leave a half-written final file. The
    # finally block cleans up our temporary file if writing raises an exception.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        yield temp_path
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    with atomic_output(path) as temp:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    with atomic_output(path) as temp:
        frame.to_parquet(temp, index=False, compression="zstd")


def file_receipt(path: Path, root: Path, frame: pd.DataFrame | None = None) -> dict:
    # Relative paths keep receipts usable if the whole output folder is moved.
    # Table receipts additionally remember row counts and column names.
    stat = path.stat()
    receipt = {"path": path.relative_to(root).as_posix(), "size": stat.st_size,
               "mtime_ns": stat.st_mtime_ns, "sha256": file_hash(path)}
    if frame is not None:
        receipt.update(rows=len(frame), columns=list(frame.columns))
    return receipt


def artifact_valid(receipt: dict, root: Path) -> bool:
    """Check every referenced file; rehash changed files without scanning TBs on each resume."""
    try:
        path = root / receipt["path"]
        stat = path.stat()
        # Check cheap file information first. Recompute the byte fingerprint
        # when the modification time changed, rather than rereading every file.
        if stat.st_size != receipt["size"]:
            return False
        if stat.st_mtime_ns != receipt["mtime_ns"] and file_hash(path) != receipt["sha256"]:
            return False
        if "rows" in receipt:
            # The Parquet footer exposes rows/columns without loading all rows.
            with pq.ParquetFile(path) as parquet:
                if (parquet.metadata.num_rows != receipt["rows"]
                        or parquet.schema_arrow.names != receipt["columns"]):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextmanager
def output_lock(output_dir: Path):
    """One writer per output root, released by the OS even after a crash."""
    # This prevents two collector processes from updating the same output tree.
    # Worker threads inside the one process can still download concurrently.
    output_dir.mkdir(parents=True, exist_ok=True)
    # This persistent lock file is intentional. Unlinking it would let another
    # process lock a different inode while the current writer still owns this one.
    with (output_dir / ".collector.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another collector is writing to {output_dir}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# 4. Understand response clocks and identity without rewriting the vendor data.
def parse_vendor_clock(values: pd.Series, exchange_tz: str) -> pd.Series:
    # Naive Theta clocks are exchange local. Aware clocks keep their stated
    # offset. Date-only interest-rate reports deliberately do not use this.
    text = values.astype("string").str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    parsed.loc[aware] = pd.to_datetime(text.loc[aware], format="mixed", errors="coerce", utc=True)
    naive = pd.to_datetime(text.loc[~aware], format="mixed", errors="coerce")
    # Daylight-saving transitions can make a local clock ambiguous or impossible.
    # Keep those parsed values missing (NaT) instead of guessing their UTC time.
    parsed.loc[~aware] = naive.dt.tz_localize(
        exchange_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return parsed


def raw_frame_with_diagnostics(frame: pd.DataFrame, request: Request, cfg: CollectorConfig) -> tuple[pd.DataFrame, dict]:
    # Return the raw table plus separate diagnostics. Counts of duplicates,
    # bad quotes, and clock problems describe the data; they do not delete rows.
    result = frame.copy()
    diagnostics = {"duplicate_rows": int(frame.duplicated().sum()), "clocks": {}}
    clocks = ("timestamp", "trade_timestamp", "quote_timestamp", "last_trade", "created")
    for name in clocks:
        if name not in frame or (name == "created" and request.dataset == "interest_rate_eod"):
            continue
        parsed = parse_vendor_clock(frame[name], cfg.exchange_tz)
        if f"collector_{name}_utc" not in result:
            result[f"collector_{name}_utc"] = parsed
        valid = parsed.dropna()
        diagnostics["clocks"][name] = {
            "unparseable_or_missing": int(parsed.isna().sum()),
            "out_of_order_transitions": int(valid.diff().lt(pd.Timedelta(0)).sum()),
            "first_utc": valid.min().isoformat() if len(valid) else None,
            "last_utc": valid.max().isoformat() if len(valid) else None,
        }
    if {"bid", "ask"}.issubset(frame):
        # Bid = displayed buying price; ask = displayed selling price. A crossed
        # quote has bid > ask. Numeric conversion here is only for counting.
        bid, ask = (pd.to_numeric(frame[side], errors="coerce") for side in ("bid", "ask"))
        diagnostics.update(invalid_bid_ask_rows=int((~np.isfinite(bid) | ~np.isfinite(ask)).sum()),
                           nonpositive_bid_ask_rows=int((bid.le(0) | ask.le(0)).sum()),
                           crossed_quote_rows=int(bid.gt(ask).sum()))
    report_date = REPORT_DATE_COLUMNS.get(request.dataset)
    # A report date tells us which day the record concerns, not the precise
    # time researchers could first have known it. Do not invent that timestamp.
    if report_date and report_date in frame:
        dates = pd.to_datetime(frame[report_date].astype("string").str.strip(), format="mixed", errors="coerce")
        diagnostics["report_dates"] = {
            "column": report_date, "unparseable_or_missing": int(dates.isna().sum()),
            "first": str(dates.min().date()) if dates.notna().any() else None,
            "last": str(dates.max().date()) if dates.notna().any() else None,
            "unique_count": int(dates.nunique()), "publication_time_verified": False,
        }
    if request.dataset == "corporate_dividend" and "amount" in frame:
        # A blank payment amount remains unknown; treating it as zero would
        # silently create a dividend assumption for later pricing work.
        diagnostics["unknown_dividend_amount_rows"] = int(frame["amount"].astype("string").str.strip().eq("").sum())
    if request.endpoint.endswith("/quote") and "collector_timestamp_utc" in result:
        interval = request.params.get("interval", "tick")
        if interval != "tick":
            seconds = interval_seconds(interval)
            unique = result["collector_timestamp_utc"].dropna().drop_duplicates().sort_values()
            gaps = unique.diff().dt.total_seconds().dropna()
            # 09:30:00 -> 09:31:00 at 1s leaves 59 interior observations absent.
            # Leading/trailing gaps are described by first/last clocks, not filled.
            diagnostics["absent_interior_sample_slots"] = int(
                np.maximum(np.ceil(gaps.to_numpy() / seconds - 1e-9) - 1, 0).sum())
    return result, diagnostics


def response_identity_issues(frame: pd.DataFrame, request: Request, cfg: CollectorConfig) -> list[str]:
    """Flag misrouted responses without dropping or correcting vendor records."""
    if frame.empty:
        return []
    issues = []
    if "symbol" in frame and frame["symbol"].ne(request.params["symbol"]).any():
        issues.append("unexpected_symbol")
    if request.endpoint.startswith("/option/") and set(CONTRACT_FIELDS).issubset(frame):
        # First check whether each row describes a valid option at all. When
        # requesting one contract, also require every row to match that contract.
        expiry = pd.to_datetime(frame["expiration"], format="mixed", errors="coerce")
        strike = pd.to_numeric(frame["strike"], errors="coerce")
        right = frame["right"].str.lower().replace({"c": "call", "p": "put"})
        if (expiry.isna().any() or (~np.isfinite(strike) | strike.le(0)).any()
                or not right.isin(["call", "put"]).all()):
            issues.append("invalid_contract_identity")
        if "expiration" in request.params and (
                expiry.ne(pd.Timestamp(request.params["expiration"])).any()
                or strike.ne(float(request.params["strike"])).any()
                or right.ne(request.params["right"]).any()):
            issues.append("unexpected_contract_identity")
    start = request.params.get("date", request.params.get("start_date"))
    # Compare market records on their exchange-local calendar date. A UTC date
    # can differ from the local date, so it is not used directly for this check.
    end = request.params.get("date", request.params.get("end_date"))
    report_date = REPORT_DATE_COLUMNS.get(request.dataset)
    if "/list/dates" in request.endpoint:
        report_date = "date"
    primary = report_date or next((name for name in ("timestamp", "trade_timestamp", "created") if name in frame), None)
    if primary and primary in frame:
        if report_date:
            dates = pd.to_datetime(frame[primary].str.strip(), format="mixed", errors="coerce")
        else:
            dates = parse_vendor_clock(frame[primary], cfg.exchange_tz).dt.tz_convert(
                cfg.exchange_tz).dt.tz_localize(None).dt.normalize()
        if start and end and (dates.notna() & ~dates.between(pd.Timestamp(start), pd.Timestamp(end))).any():
            issues.append("timestamps_outside_requested_dates")
        if dates.isna().all():
            issues.append("no_parseable_report_dates")
        elif report_date and dates.isna().any():
            issues.append("invalid_report_dates")
    return issues


def interval_seconds(interval: str) -> float:
    # API "m" means minutes; pandas also accepts other, ambiguous abbreviations.
    if interval.endswith("ms"):
        return float(interval[:-2]) / 1000
    return float(interval[:-1]) * {"s": 1, "m": 60, "h": 3600}[interval[-1]]


def format_strike(value) -> str:
    # Use decimal arithmetic for the URL and contract key: binary floats can
    # create spurious digits. Reject values beyond the supported 0.001 precision.
    strike = Decimal(str(value))
    if not strike.is_finite() or strike <= 0 or strike != strike.quantize(Decimal("0.001")):
        raise ValueError(f"Invalid option strike: {value!r}")
    return format(strike, ".3f").rstrip("0").rstrip(".")


# 5. Talk to Theta Terminal: connections, shared request limits, and retries.
# This layer returns bytes plus HTTP details. Parsing and saving happen below.
class CollectionStopped(RuntimeError):
    """Stop scheduling work while retaining requests that already finished."""


class ThetaClient:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.local = threading.local()
        self.semaphore = threading.BoundedSemaphore(cfg.max_inflight_requests)
        self.pace_lock = threading.Lock()
        self.next_allowed = 0.0
        self.stop_event = threading.Event()
        self.stop_reason = ""

    def stop(self, reason: str) -> None:
        # The first cause explains the stop; other workers should not overwrite it.
        with self.pace_lock:
            if not self.stop_event.is_set():
                self.stop_reason = reason
                self.stop_event.set()

    def check_running(self) -> None:
        if self.stop_event.is_set():
            raise CollectionStopped(self.stop_reason)

    def ensure_available(self) -> None:
        # Fail once before scheduling years of requests against an offline terminal.
        # A reachable port is only a connection check; it does not verify account
        # permissions, endpoint support, or the requested historical coverage.
        address = urlsplit(self.cfg.base_url)
        try:
            with socket.create_connection((address.hostname, address.port or (443 if address.scheme == "https" else 80)), timeout=2):
                pass
        except OSError as exc:
            raise RuntimeError(f"Theta Terminal is unreachable at {self.cfg.base_url}. Start it and rerun.") from exc

    def session(self) -> requests.Session:
        # Each worker thread reuses its own HTTP connection pool. Avoid sharing
        # mutable session state between threads while still reusing connections.
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            # Retries are explicit so every attempt obeys the same request budget.
            adapter = HTTPAdapter(max_retries=0, pool_connections=2, pool_maxsize=2)
            self.local.session.mount("http://", adapter)
            self.local.session.mount("https://", adapter)
        return self.local.session

    @contextmanager
    def request_slot(self):
        # Two separate limits: the semaphore caps unfinished requests, while
        # next_allowed spaces out starts across all workers, including retries.
        self.check_running()
        with self.semaphore:
            self.check_running()
            with self.pace_lock:
                wait = max(0.0, self.next_allowed - time.monotonic())
                self.next_allowed = max(time.monotonic(), self.next_allowed) + 1 / self.cfg.max_requests_per_second
            # Waiting workers wake promptly on cancellation, instead of starting
            # another HTTP request after the user or another worker stopped the run.
            self.stop_event.wait(wait)
            self.check_running()
            yield

    def download(self, request: Request) -> tuple[bytes | None, dict]:
        # At most six attempts. Retry temporary failures such as rate limits
        # and server errors; return other failures so they can be recorded.
        meta = {}
        for attempt in range(6):
            started = time.perf_counter()
            retry_after = 0.0
            try:
                with self.request_slot():
                    response = self.session().get(
                        self.cfg.base_url.rstrip("/") + request.endpoint,
                        params=request.params, timeout=(10, 120))
                payload = response.content
                # Keep the response's identity and timing even if it failed.
                # These details help explain missing data later.
                meta = {"request_url": response.url, "status_code": response.status_code,
                        "response_headers": dict(response.headers), "attempts": attempt + 1,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_bytes": len(payload)}
                if response.status_code in {200, 472}:
                    # 200 is an ordinary response; Theta uses 472 for no data.
                    # The storage layer records an empty result separately.
                    return payload, meta
                meta["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in {429, 474, 500, 502, 503, 504, 571} or attempt == 5:
                    # Permissions, invalid parameters, and terminal configuration
                    # need action, not thousands more requests. Sustained terminal
                    # disconnections/rate-limit failures also stop new work.
                    if response.status_code in {400, 401, 403, 404, 429, 471, 473, 474, 475, 476, 478, 571}:
                        self.stop(f"Theta HTTP {response.status_code} for {request.endpoint}: "
                                  "check Theta access, terminal state, and request settings, then rerun.")
                    return payload, meta
                try:
                    retry_after = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    pass
            except requests.RequestException as exc:
                payload = None
                meta = {"error": repr(exc), "attempts": attempt + 1, "status_code": None}
                if attempt == 5:
                    self.stop(f"Theta connection failed after six attempts for {request.endpoint}; rerun when it is available.")
                    return payload, meta
            # Increase the delay between attempts, considering Theta's requested
            # Retry-After delay too. Both delays are bounded by the limits below.
            self.stop_event.wait(max(min(30.0, 2 ** attempt), min(max(retry_after, 0), 120.0)))
            self.check_running()
        return None, meta


# 6. Save and reuse individual responses independently of contract selection.
# RequestStore is the common route for stocks, options, and reference data.
class RequestStore:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.root = cfg.output_dir
        self.client = ThetaClient(cfg)
        # Requests with the same key share a lock, so workers cannot write the
        # same cache entry at once. Unrelated requests can usually proceed together.
        self.locks = tuple(threading.Lock() for _ in range(128))

    def directory(self, request: Request) -> Path:
        # Readable dataset/symbol/date folders help browsing. The request ID
        # distinguishes different contracts and parameters within those folders.
        date = request.params.get("date", request.params.get("start_date", "reference"))
        date = str(date).replace("-", "")
        symbol = request.params["symbol"]
        return (self.root / "raw_cache" / request.dataset / f"symbol={symbol}" /
                f"date={date}" / f"request={request.request_id}")

    def cached(self, request: Request) -> dict | None:
        # Return a reusable receipt, or None to trigger collection. Merely finding
        # a file is insufficient: settings, status, and saved artifacts must agree.
        path = self.directory(request) / "meta.json"
        try:
            # latest.json points to an immutable response. Older collector caches
            # used meta.json directly; keep reading those without moving their files.
            index = path.with_name("latest.json")
            if index.exists():
                receipt = read_json(index)["metadata"]
                if not artifact_valid(receipt, self.root):
                    return None
                path = self.root / receipt["path"]
            meta = read_json(path)
            if (meta["request"] != request.identity() or meta["raw_schema_version"] != RAW_SCHEMA_VERSION
                    or meta["status"] not in GOOD_REQUEST_STATUSES
                    or meta.get("timestamp_timezone") != self.cfg.exchange_tz):
                return None
            if self.cfg.refresh_no_data and meta["status"] == "no_data":
                return None
            if not artifact_valid(meta["data"], self.root):
                return None
            if meta.get("payload") and not artifact_valid(meta["payload"], self.root):
                return None
            if self.cfg.store_raw_payloads and not meta.get("payload"):
                return None
            return self.record(request, meta, path)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def record(self, request: Request, meta: dict, meta_path: Path) -> dict:
        # A compact receipt for session/reference manifests. Detailed diagnostics
        # remain in the request's meta.json instead of being copied everywhere.
        return {"request_id": request.request_id, "dataset": request.dataset,
                "status": meta["status"], "row_count": meta.get("row_count", 0),
                "status_code": meta.get("status_code"),
                "error": meta.get("error", ""), "data": meta.get("data"),
                "payload": meta.get("payload"), "metadata": file_receipt(meta_path, self.root)}

    def collect(self, request: Request, *, refresh: bool = False) -> dict:
        # Order: reuse current cache -> import a compatible older cache -> download.
        # Date-catalogue refreshes skip current cache reuse to see newly added dates.
        self.client.check_running()
        with self.locks[hash(request.request_id) % len(self.locks)]:
            self.client.check_running()
            cached = None if refresh else self.cached(request)
            if cached:
                return cached
            legacy = self.legacy_response(request)
            if legacy is not None:
                frame, response_meta, payload = legacy
                return self.save(request, frame, response_meta, payload)
            payload, response_meta = self.client.download(request)
            status = response_meta.get("status_code")
            frame = pd.DataFrame(columns=request.required_columns)
            if status not in {200, 472}:
                return self.save(request, frame, response_meta, payload, "request_error")
            if status == 200:
                try:
                    content_type = response_meta.get("response_headers", {}).get("Content-Type", "").lower()
                    if content_type and not any(t in content_type for t in ("csv", "text/plain", "octet-stream")):
                        raise ValueError(f"Unexpected response content type: {content_type}")
                    # Read every vendor field as text. This preserves long sequence
                    # numbers, decimal spelling, and blanks before any interpretation.
                    frame = pd.read_csv(BytesIO(payload), dtype="string", keep_default_na=False)
                except pd.errors.EmptyDataError:
                    # A successful empty body and 472 are distinct in HTTP metadata.
                    pass
                except Exception as exc:
                    response_meta["error"] = f"Invalid CSV response: {exc}"
                    return self.save(request, frame, response_meta, payload, "invalid_response")
            return self.save(request, frame, response_meta, payload)

    def save(self, request: Request, frame: pd.DataFrame, response_meta: dict,
             payload: bytes | None = None, status: str | None = None) -> dict:
        # Save the table even when it is empty or invalid, with an explicit status.
        # Retained evidence lets us distinguish absent data from a broken request.
        # Every actual attempt gets new files, including failures. Overwriting
        # data.parquet in place would change data referenced by an earlier run.
        cache_directory = self.directory(request)
        directory = cache_directory / "responses" / uuid4().hex
        missing = set(request.required_columns) - set(frame.columns)
        # collector_ is reserved for our added fields, so a vendor field cannot
        # silently masquerade as one of our parsed timestamps.
        if missing or any(str(column).startswith("collector_") for column in frame.columns):
            status = status or "invalid_response"
            response_meta["error"] = f"Missing required columns {sorted(missing)} or reserved collector_ column"
        frame = frame.astype("string").fillna("")
        frame, quality = raw_frame_with_diagnostics(frame, request, self.cfg)
        quality["response_identity_issues"] = response_identity_issues(frame, request, self.cfg)
        if quality["response_identity_issues"]:
            status = status or "invalid_response"
            response_meta["error"] = ", ".join(quality["response_identity_issues"])
        if status is None:
            status = "no_data" if frame.empty else "available"
            # An unparseable primary clock indicates an unusable response, but
            # preserve it for inspection. Bad individual rows are never discarded.
            primary = next((name for name in ("timestamp", "trade_timestamp", "created")
                            if name in quality["clocks"]), None)
            if primary and len(frame) and quality["clocks"][primary]["unparseable_or_missing"] == len(frame):
                status = "invalid_response"
                response_meta["error"] = f"No parseable {primary} values"
        meta = {"request": request.identity(), "request_id": request.request_id,
                "raw_schema_version": RAW_SCHEMA_VERSION, "timestamp_timezone": self.cfg.exchange_tz,
                "fetched_at_utc": response_meta.pop("fetched_at_utc", utc_now()), "saved_at_utc": utc_now(),
                "status": status, "row_count": len(frame), "quality": quality,
                "collector_code_sha256": file_hash(Path(__file__)), **response_meta}
        data_path = directory / "data.parquet"
        write_parquet(data_path, frame)
        meta["data"] = file_receipt(data_path, self.root, frame)
        # Preserve unsuccessful responses even without --store-raw-payloads.
        # They explain schema errors, entitlement failures and vendor messages.
        if payload is not None and (self.cfg.store_raw_payloads or status not in GOOD_REQUEST_STATUSES):
            payload_path = directory / "raw_response.csv"
            with atomic_output(payload_path) as temp:
                temp.write_bytes(payload)
            meta["payload"] = file_receipt(payload_path, self.root)
        # Publish this attempt's receipt after its files exist, then atomically
        # advance the cache pointer only on success. A failed refresh remains
        # visible to its caller without destroying the previous good response.
        write_json(directory / "meta.json", meta)
        record = self.record(request, meta, directory / "meta.json")
        if status in GOOD_REQUEST_STATUSES:
            write_json(cache_directory / "latest.json", {"metadata": record["metadata"]})
        return record

    def legacy_response(self, request: Request):
        """Reuse valid pre-refactor raw quotes/OI/chains without changing old files."""
        # Compatibility with the original collector's folder layout. Import only
        # exact matching requests whose saved fingerprint still checks out.
        if request.dataset not in {"quoted_contracts", "option_open_interest",
                                    "stock_quotes_" + self.cfg.quote_interval, "option_quotes_" + self.cfg.quote_interval}:
            return None
        params = request.params
        day = pd.Timestamp(params["date"]).strftime("%Y-%m-%d")
        directory = self.root / "raw_cache" / request.dataset / f"symbol={params['symbol']}" / f"date={day}"
        if "expiration" in params:
            directory /= f"expiration={params['expiration']}"
            directory /= f"strike={format_strike(params['strike'])}"
            directory /= f"right={params['right']}"
        try:
            old = read_json(directory / "meta.json")
            path = directory / "data.parquet"
            if (old["dataset"] != request.dataset or old["endpoint"] != request.endpoint or old["params"] != params
                    or old["cache_status"] not in {"ok", "no_data"} or file_hash(path) != old["cache_data_sha256"]):
                return None
            if self.cfg.refresh_no_data and old["cache_status"] == "no_data":
                return None
            payload_path = directory / "raw_response.csv"
            payload = payload_path.read_bytes() if payload_path.exists() else None
            if payload is not None and hashlib.sha256(payload).hexdigest() != old.get("payload_sha256"):
                return None
            if self.cfg.store_raw_payloads and payload is None:
                return None
            if payload and old.get("status_code") == 200:
                # Original CSV bytes preserve more detail than an older table
                # whose numbers/timestamps may already have been converted.
                frame = pd.read_csv(BytesIO(payload), dtype="string", keep_default_na=False)
            else:
                frame = pd.read_parquet(path)
                if "timestamp_raw" in frame:
                    frame["timestamp"] = frame["timestamp_raw"]
                frame = frame.loc[:, old["columns"]]
            if set(request.required_columns) - set(frame.columns):
                return None
            meta = {name: old[name] for name in ("request_url", "status_code", "response_headers",
                    "payload_sha256", "payload_bytes", "fetched_at_utc") if name in old}
            meta.update(legacy_source=path.relative_to(self.root).as_posix(),
                        legacy_values_previously_parsed=payload is None)
            return frame, meta, payload
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def read(self, record: dict) -> pd.DataFrame:
        # Selection code only consumes successful responses. Failed responses
        # remain on disk for inspection but are not used to select contracts.
        if record["status"] not in GOOD_REQUEST_STATUSES or not record.get("data"):
            return pd.DataFrame()
        return pd.read_parquet(self.root / record["data"]["path"])


# 7. Build requests using the exchange calendar and each endpoint's arguments.
# Cache the calendar object so every request does not reconstruct it.
@lru_cache(maxsize=1)
def exchange_calendar():
    return xcals.get_calendar("XNYS", start="2017-01-01", end="2026-12-31")


def session_bounds(day: pd.Timestamp, cfg: CollectorConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Read the day's actual open/close, including early closes, instead of
    # assuming every weekday has a full 09:30-16:00 session.
    calendar = exchange_calendar()
    return (calendar.session_open(day).tz_convert(cfg.exchange_tz),
            calendar.session_close(day).tz_convert(cfg.exchange_tz))


def history_request(cfg: CollectorConfig, asset: str, kind: str, symbol: str,
                    day: pd.Timestamp, contract: dict | None = None) -> Request:
    # Construct a request only; this function does not call the network.
    # asset selects stock/option/index; kind selects quote/trade_quote/OI/etc.
    params = {"symbol": symbol, "date": day.strftime("%Y%m%d"), "format": "csv"}
    if asset == "option":
        if contract is None:
            raise ValueError("Option history requires an observed contract")
        params.update(expiration=contract["expiration"], strike=format_strike(contract["strike"]),
                      right=contract["right"])
    if kind in {"quote", "price"}:
        params["interval"] = cfg.quote_interval
        dataset = f"{asset}_{kind}s_{cfg.quote_interval}"
    elif kind == "trade_quote":
        params["exclusive"] = str(cfg.trade_quote_exclusive).lower()
        dataset = f"{asset}_trade_quotes_tick"
    else:
        dataset = f"{asset}_{kind}"
    if kind == "eod":
        # EOD (end of day) uses a date range, even for a single-day pull.
        params.pop("date")
        params.update(start_date=day.strftime("%Y%m%d"), end_date=day.strftime("%Y%m%d"))
    elif kind != "open_interest":
        # OI (open interest) is a report of outstanding contracts, not a stream
        # of trades. Intraday start/end-time arguments apply to the other pulls.
        opened, closed = session_bounds(day, cfg)
        params.update(start_time=opened.strftime("%H:%M:%S"), end_time=closed.strftime("%H:%M:%S"))
        if asset == "stock":
            params["venue"] = cfg.stock_venue
    return Request(dataset, f"/{asset}/history/{kind}", params)


def reference_requests(cfg: CollectorConfig, symbols: list[SymbolConfig], start: str, end: str,
                       rate_symbols: list[str]):
    """One Theta-only bundle. Rates/actions are date reports; VIX prices are intraday."""
    # yield produces requests one at a time. Actions belong to each underlying;
    # interest-rate and VIX requests are shared across the selected underlyings.
    window = {"start_date": start, "end_date": end, "format": "csv"}
    for symbol in symbols:
        for kind in ("dividend", "split"):
            yield Request(f"corporate_{kind}", f"/corporate_action/{kind}", {"symbol": symbol.symbol, **window})
    for symbol in sorted(set(rate_symbols)):
        yield Request("interest_rate_eod", "/interest_rate/history/eod", {"symbol": symbol, **window})
    yield Request("index_eod", "/index/history/eod", {"symbol": "VIX", **window})
    for day in exchange_calendar().sessions_in_range(start, end).tz_localize(None):
        # Sub-minute index history must be requested one day at a time.
        yield history_request(cfg, "index", "price", "VIX", day)


# 8. Choose actual listed contracts to download using the study's sampling grid.
# These functions use working copies; the vendor's raw tables stay unchanged.
def normalize_chain(frame: pd.DataFrame, symbol: str, cfg: CollectorConfig) -> pd.DataFrame:
    # The "chain" is the dated list of quoted option contracts. Convert its
    # identities to comparable types and remove duplicate identities for selection.
    if frame.empty:
        return pd.DataFrame(columns=CONTRACT_FIELDS)
    chain = frame.loc[:, CONTRACT_FIELDS].copy()
    chain["expiration"] = pd.to_datetime(chain["expiration"], format="mixed", errors="raise").dt.normalize()
    chain["strike"] = pd.to_numeric(chain["strike"], errors="raise")
    chain["right"] = chain["right"].str.lower().replace({"c": "call", "p": "put"})
    if (chain["symbol"].ne(symbol).any() or chain[["expiration", "strike"]].isna().any().any()
            or not np.isfinite(chain["strike"]).all() or chain["strike"].le(0).any()
            or not chain["right"].isin(["call", "put"]).all()):
        raise ValueError("Invalid identity in dated quote universe")
    for strike in chain["strike"].unique():
        format_strike(strike)
    return (chain.loc[chain["right"].isin(cfg.option_rights)].drop_duplicates()
            .sort_values(["expiration", "strike", "right"]).reset_index(drop=True))


def stock_selection_references(frame: pd.DataFrame, day: pd.Timestamp, cfg: CollectorConfig) -> list[dict]:
    # Produce up to three stock midpoints, each with its source quote and age.
    # The midpoint is (bid + ask) / 2 and is used only to target option strikes.
    if frame.empty:
        return []
    # Only the stock spot used to choose strikes needs a usable quote. Raw storage
    # never uses this condition policy, and option observations are never filtered.
    quotes = frame.copy()
    quotes["_clock"] = parse_vendor_clock(quotes["timestamp"], cfg.exchange_tz)
    allowed = pd.Series(True, index=quotes.index)
    # Apply the existing allowed-condition policy only to this stock lookup.
    # Blank conditions are accepted; unlisted/non-numeric codes are excluded here.
    for column in ("bid_condition", "ask_condition"):
        text = quotes[column].astype("string").str.strip()
        allowed &= text.eq("") | pd.to_numeric(text, errors="coerce").isin([0, 1, 50])
    quotes = quotes.loc[allowed & quotes["_clock"].notna()].sort_values("_clock", kind="stable")
    opened, closed = session_bounds(day, cfg)
    references = []
    for selection_time in cfg.selection_times:
        at = pd.Timestamp(f"{day.date()} {selection_time}", tz=cfg.exchange_tz)
        if not opened <= at < closed:
            # For example, a 15:00 reference is skipped on a 13:00 early close.
            continue
        prior = quotes.loc[quotes["_clock"].between(opened, at)]
        # Never look forward for the stock price at a selection time. Use the
        # latest allowed quote at or before it, then check its prices and age.
        if prior.empty:
            continue
        row = prior.iloc[-1]
        bid, ask = (pd.to_numeric(row[side], errors="coerce") for side in ("bid", "ask"))
        age = (at - row["_clock"]).total_seconds()
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask
                and age <= cfg.max_stock_quote_age_seconds):
            continue
        references.append({"selection_time": selection_time, "stock_mid": float((bid + ask) / 2),
                           "quote_timestamp": str(row["timestamp"]), "quote_timestamp_utc": row["_clock"].isoformat(),
                           "sample_age_seconds": age})
    return references


def eligible_expirations(day: pd.Timestamp, expirations, cfg: CollectorConfig) -> list[pd.Timestamp]:
    # Restrict to the DTE window, then choose the nearest unused expiration for
    # each target in order. Ties prefer shorter DTE. Removing each choice prevents
    # two targets from selecting the same expiration twice.
    remaining = sorted(exp for exp in expirations if cfg.min_dte <= (exp - day).days <= cfg.max_dte)
    selected = []
    for target in cfg.target_dtes:
        if not remaining or len(selected) >= cfg.max_expirations_per_day:
            break
        best = min(remaining, key=lambda exp: (abs((exp - day).days - target), (exp - day).days))
        selected.append(best)
        remaining.remove(best)
    remaining.sort(key=lambda exp: (min(abs((exp - day).days - target) for target in cfg.target_dtes), exp))
    # If the cap leaves room, add other expirations nearest any target.
    return sorted(selected + remaining[:max(cfg.max_expirations_per_day - len(selected), 0)])


def select_contracts(chain: pd.DataFrame, day: pd.Timestamp, references: list[dict],
                     cfg: CollectorConfig) -> pd.DataFrame:
    # For each chosen expiration, find listed strikes near each S/K target at
    # each stock reference time. The output records both identity and why chosen.
    rows = []
    for expiration in eligible_expirations(day, chain["expiration"].unique(), cfg):
        family = chain.loc[chain["expiration"].eq(expiration)]
        strikes = sorted(family["strike"].unique())
        chosen_by_time = {}
        for reference in references:
            chosen = set()
            for target in cfg.moneyness_targets:
                target_strike = reference["stock_mid"] / target
                # Rearrange target = S/K to K = S/target. Rank listed strikes by
                # dollar distance from K; equal distances prefer the lower strike.
                ranked = sorted(strikes, key=lambda strike: (abs(strike - target_strike), strike))
                chosen.update(ranked[:cfg.strikes_per_moneyness_target])
            chosen_by_time[reference["selection_time"]] = chosen
        selected = set().union(*chosen_by_time.values()) if chosen_by_time else set()
        # Download the union across selection times once. Keep only call/put
        # identities actually present in the dated chain; do not invent pairs.
        for contract in family.loc[family["strike"].isin(selected)].itertuples(index=False):
            expiry, strike = expiration.strftime("%Y-%m-%d"), format_strike(contract.strike)
            rows.append({"symbol": contract.symbol, "expiration": expiry, "strike": strike,
                         "right": contract.right, "contract_key": f"{contract.symbol}|{expiry}|{strike}|{contract.right}",
                         "dte_days": (expiration - day).days,
                         "selection_times": "|".join(at for at, values in chosen_by_time.items() if contract.strike in values)})
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


# 9. Coordinate symbol-days, shared references, and coverage reports.
# A session manifest is the small record connecting selected contracts to their
# request receipts. It can be read without opening the much larger market tables.
class Collector:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.store = RequestStore(cfg)
        self.directory = cfg.output_dir / "collection" / cfg.policy_id

    @contextmanager
    def workers(self, count: int):
        # ThreadPoolExecutor normally finishes its entire queue on interruption.
        # Stop new HTTP work and cancel queued tasks instead; active requests can
        # finish saving their responses before the output lock is released.
        pool = ThreadPoolExecutor(max_workers=count)
        try:
            yield pool
        except BaseException as exc:
            self.store.client.stop(str(exc) or "Collection interrupted by the user.")
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def session_path(self, symbol: str, day: pd.Timestamp) -> Path:
        return self.directory / "sessions" / f"symbol={symbol}__date={day.date()}.json"

    def manifest_valid(self, manifest: dict) -> bool:
        # A saved "complete" label alone is not enough to skip work. Check that
        # the policy matches and every expected request/artifact is accounted for.
        try:
            if (manifest["output_schema_version"] != OUTPUT_SCHEMA_VERSION
                    or manifest["policy_id"] != self.cfg.policy_id
                    or manifest["status"] not in {"complete", "unavailable"}):
                return False
            records = manifest["requests"]
            selected_count = manifest["selected_contract_count"]
            # Four base requests: stock quotes, dated chain, stock trades, stock EOD.
            # Each selected option adds three: quotes, trades, and open interest.
            if (len(records) != 4 + 3 * selected_count
                    or len({r["request_id"] for r in records}) != len(records)
                    or manifest["contracts"]["rows"] != selected_count):
                return False
            if not artifact_valid(manifest["contracts"], self.cfg.output_dir):
                return False
            for record in records:
                if record["status"] not in GOOD_REQUEST_STATUSES:
                    return False
                if self.cfg.refresh_no_data and record["status"] == "no_data":
                    return False
                if self.cfg.store_raw_payloads and not record.get("payload"):
                    return False
                for name in ("data", "metadata", "payload"):
                    if name != "payload" or record.get(name):
                        if not artifact_valid(record[name], self.cfg.output_dir):
                            return False
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def resumable(self, symbol: str, day: pd.Timestamp) -> bool:
        # True means this symbol-day can be skipped. False lets collect_day run
        # again, while RequestStore still reuses its individually valid downloads.
        try:
            manifest = read_json(self.session_path(symbol, day))
            return (manifest.get("symbol") == symbol and manifest.get("trade_day") == str(day.date())
                    and self.manifest_valid(manifest))
        except (OSError, ValueError):
            return False

    def collect_day(self, symbol_cfg: SymbolConfig, day: pd.Timestamp) -> dict:
        # This is the main unit of collection: one underlying on one trading day.
        # Its result is a manifest, while the large raw tables are saved separately.
        symbol = symbol_cfg.symbol
        records, references = [], []
        selected = pd.DataFrame(columns=SELECTION_COLUMNS)
        reason, error, discovered = "", "", 0
        try:
            # Step 1: save the stock quotes needed for selection, plus the dated
            # option universe. This universe describes quotes observed that day;
            # it does not establish when each contract first became available.
            stock = self.store.collect(history_request(self.cfg, "stock", "quote", symbol, day))
            records.append(stock)
            # Discovery is independent of stock availability. Keep the dated
            # universe even when stock quotes cannot support strike selection.
            chain_record = self.store.collect(Request("quoted_contracts", "/option/list/contracts/quote",
                                              {"symbol": symbol, "date": day.strftime("%Y%m%d"), "format": "csv"}))
            records.append(chain_record)
            # Step 2: also save individual stock trades and the day's EOD record.
            # Stock trades are retained independently of the quote sampling rate.
            for kind in ("trade_quote", "eod"):
                records.append(self.store.collect(history_request(self.cfg, "stock", kind, symbol, day)))
            chain = normalize_chain(self.store.read(chain_record), symbol, self.cfg)
            discovered = len(chain)
            # Step 3: obtain the stock references, then choose the option contracts.
            # A chosen contract gets its requested whole-session history, even if
            # a later selection time caused us to choose it.
            references = stock_selection_references(self.store.read(stock), day, self.cfg)
            selected = select_contracts(chain, day, references, self.cfg)
            if chain.empty:
                reason = "no_quoted_contracts"
            elif not references:
                reason = "stock_selection_reference_unavailable"
            elif selected.empty:
                reason = "no_contracts_in_sampling_window"
            # Step 4: collect three data types per selected option concurrently.
            # No trade-count, spread, or OI threshold removes a chosen contract.
            with self.workers(self.cfg.max_contract_workers) as pool:
                futures = []
                for contract in selected.to_dict("records"):
                    for kind in ("quote", "trade_quote", "open_interest"):
                        request = history_request(self.cfg, "option", kind, symbol, day, contract)
                        futures.append(pool.submit(self.store.collect, request))
                for future in as_completed(futures):
                    try:
                        records.append(future.result())
                    except CollectionStopped as exc:
                        # Another worker can notice the stop before the triggering
                        # request finishes saving. Drain results so its receipt and
                        # other completed downloads still enter the session manifest.
                        error = repr(exc)
                    except Exception as exc:
                        error = repr(exc)
                        self.store.client.stop(f"Unable to finish an option request: {exc}")
        except Exception as exc:
            # Completed raw pulls survive; a failed day never satisfies resume.
            error = repr(exc)
        failures = sum(record["status"] not in GOOD_REQUEST_STATUSES for record in records)
        # "unavailable" means selection yielded no contracts without a request
        # error; reason explains why. "complete" means the requests finished,
        # including any valid empty responses, not full observed market coverage.
        status = "request_error" if error or failures else ("unavailable" if selected.empty else "complete")
        contract_path = self.directory / "contracts" / f"symbol={symbol}__date={day.date()}.parquet"
        write_parquet(contract_path, selected)
        # Step 5: record selection coverage and receipts. Times outside this
        # day's session are not counted as missing selection references.
        opened, closed = session_bounds(day, self.cfg)
        scheduled = [at for at in self.cfg.selection_times
                     if opened <= pd.Timestamp(f"{day.date()} {at}", tz=self.cfg.exchange_tz) < closed]
        manifest = {**asdict(symbol_cfg), "trade_day": str(day.date()), "status": status,
                    "reason": reason, "error": error, "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "policy_id": self.cfg.policy_id, "updated_at_utc": utc_now(),
                    "session_open": opened.isoformat(), "session_close": closed.isoformat(),
                    "quoted_contract_count": discovered, "selected_contract_count": len(selected),
                    "stock_selection_references": references,
                    "missing_selection_times": [at for at in scheduled
                                                if at not in {r["selection_time"] for r in references}],
                    "requests": sorted(records, key=lambda r: (r["dataset"], r["request_id"])),
                    "contracts": file_receipt(contract_path, self.cfg.output_dir, selected),
                    "request_error_count": failures, "expected_request_count": 4 + 3 * len(selected)}
        # Publish last. The manifest references exact artifacts, not a filename glob.
        write_json(self.session_path(symbol, day), manifest)
        self.store.client.check_running()
        return manifest

    def collect_references(self, symbols: list[SymbolConfig], start: str, end: str,
                           rate_symbols: list[str], run_id: str) -> list[dict]:
        # Reference data has its own ledger because it serves many symbol-days.
        # Keeping it separate avoids copying rates/VIX into each option table.
        records = []
        # These notes travel with the data so later research can interpret units,
        # missing values, and unverified coverage without relying on this script.
        ledger = {"vendor": "ThetaData", "start": start, "end": end,
                  "requested_rates": sorted(set(rate_symbols)), "rate_units": "percent",
                  "required_datasets": ["corporate_dividend", "corporate_split", "interest_rate_eod",
                                        "index_eod", f"index_prices_{self.cfg.quote_interval}"],
                  "rate_publication_timestamps_verified": False,
                  "corporate_action_range_filters": {"dividend": "ex_dividend_date", "split": "effective_date"},
                  "missing_dividend_amounts": "unknown; not zero", "split_ratio": "before_shares / after_shares",
                  "empty_actions_prove_complete_event_coverage": False,
                  "index_unchanged_updates_may_be_omitted": True,
                  "adjusted_contract_deliverables": "not_documented_by_Theta; not_inferred",
                  "historical_symbol_mappings": "not_verified"}
        path = self.cfg.output_dir / "references" / f"{run_id}.json"
        def publish():
            write_json(path, {**ledger, "updated_at_utc": utc_now(), "requests": records})
        try:
            for request in reference_requests(self.cfg, symbols, start, end, rate_symbols):
                if self.store.client.stop_event.is_set():
                    break
                try:
                    record = self.store.collect(request)
                except CollectionStopped:
                    break
                except Exception as exc:
                    # Record the failed reference and continue the remaining pulls.
                    # main() still reports a failed run; completed data is retained.
                    record = {"request_id": request.request_id, "dataset": request.dataset,
                              "status": "request_error", "error": repr(exc)}
                records.append({**record, "params": request.params})
                # Raw receipts are committed per request. Batch this progress file
                # so a multi-year VIX run does not rewrite it thousands of times.
                if len(records) % 25 == 0 or record["status"] not in GOOD_REQUEST_STATUSES:
                    publish()
                print(f"Reference {request.params['symbol']} {request.dataset}: {record['status']}")
        finally:
            publish()
        return records

    def collect_coverage(self, symbols: list[SymbolConfig], anchors: pd.DatetimeIndex, run_id: str) -> dict:
        # Ask "which dates does Theta list?" before requesting detailed history.
        # anchors are the exchange sessions within the user's requested window.
        requests_to_make = [
            Request(f"stock_{kind}_dates", f"/stock/list/dates/{kind}", {"symbol": symbol.symbol, "format": "csv"})
            for symbol in symbols for kind in ("quote", "trade")
        ]
        requests_to_make.append(Request("index_price_dates", "/index/list/dates", {"symbol": "VIX", "format": "csv"}))
        expected = set(anchors.strftime("%Y-%m-%d"))
        rows, records = [], []
        try:
            for request in requests_to_make:
                if self.store.client.stop_event.is_set():
                    break
                try:
                    # Catalogues may expand as Theta backfills older history. Refresh
                    # just these small lists; history requests remain independently cached.
                    record = self.store.collect(request, refresh=True)
                    frame = self.store.read(record)
                except CollectionStopped:
                    break
                except Exception as exc:
                    frame = pd.DataFrame()
                    record = {"request_id": request.request_id, "dataset": request.dataset,
                              "status": "request_error", "error": repr(exc)}
                records.append({**record, "params": request.params})
                dates = (pd.to_datetime(frame["date"].astype("string").str.strip(), format="mixed", errors="coerce")
                         .dropna() if "date" in frame else pd.Series(dtype="datetime64[ns]"))
                known = record["status"] in GOOD_REQUEST_STATUSES
                available = set(dates.dt.strftime("%Y-%m-%d"))
                # None means the catalogue request failed, so we cannot know the gaps.
                # [] means it succeeded and lists every requested session. A populated
                # list names sessions that the vendor's catalogue does not include.
                missing = sorted(expected - available) if known else None
                row = {"symbol": request.params["symbol"], "dataset": request.dataset,
                       "status": ("listed" if not missing else "coverage_gap") if known else "request_error",
                       "first_available": str(dates.min().date()) if len(dates) else None,
                       "last_available": str(dates.max().date()) if len(dates) else None,
                       "requested_sessions": len(expected),
                       "listed_requested_sessions": len(expected & available) if known else None,
                       "missing_requested_dates": missing, "error": record.get("error", "")}
                rows.append(row)
                print(f"Coverage {row['symbol']} {row['dataset']}: {row['status']}")
        finally:
            # Even an interrupted catalogue check leaves its known results and
            # names the requests for which it did not finish a coverage result.
            report = {"vendor": "ThetaData", "checked_at_utc": utc_now(), "rows": rows, "requests": records,
                      "unfinished_requests": [r.identity() for r in requests_to_make[len(rows):]],
                      "request_errors": sum(row["status"] == "request_error" for row in rows),
                      "series_with_gaps": sum(row["status"] == "coverage_gap" for row in rows),
                      "proves_interval_or_subscription_access": False,
                      "proves_complete_intraday_records": False,
                      "scope": "stock quote/trade and VIX date catalogues; option coverage is recorded during collection",
                      "documented_limits": {
                          "SPY_underlying_history_before_2020": "unavailable per Theta documentation",
                          "other_CTA_only_symbols": "pre-2020 underlying history may be unavailable",
                          "reference_and_adjusted_contract_completeness": "not established by date catalogues"},
                      "documentation": "https://docs.thetadata.us/Articles/Data-And-Requests/Making-Requests.html"}
            write_json(self.cfg.output_dir / "coverage" / f"{run_id}.json", report)
        return report

    def write_availability(self, symbols: list[SymbolConfig], anchors: pd.DatetimeIndex) -> dict:
        # Build a small CSV with one row per requested symbol-day. It summarizes
        # manifests, including days never attempted, without loading raw tables.
        # Counts/bytes describe saved artifacts, not unique economic events.
        path = self.directory / "availability.csv"
        columns = ("symbol", "trade_day", "status", "reason", "quoted_contract_count", "selected_contract_count",
                   "selection_reference_count", "missing_selection_times", "request_count", "request_error_count",
                   "no_data_request_count", "stored_rows", "stored_parquet_bytes", "error")
        counts = dict.fromkeys(("complete", "unavailable", "request_error", "not_attempted"), 0)
        with atomic_output(path) as temp:
            with temp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for day in anchors:
                    for symbol in symbols:
                        row = {"symbol": symbol.symbol, "trade_day": str(day.date()), "status": "not_attempted"}
                        try:
                            manifest = read_json(self.session_path(symbol.symbol, day))
                            if (manifest["symbol"] != symbol.symbol or manifest["trade_day"] != str(day.date())
                                    or manifest["status"] not in {"complete", "unavailable", "request_error"}):
                                raise ValueError("Session identity or status does not match the requested day")
                            row.update({key: manifest.get(key, "") for key in columns if key in manifest})
                            records = manifest["requests"]
                            row.update(selection_reference_count=len(manifest["stock_selection_references"]),
                                       missing_selection_times="|".join(manifest["missing_selection_times"]),
                                       request_count=len(records),
                                       no_data_request_count=sum(r["status"] == "no_data" for r in records),
                                       stored_rows=sum(r.get("row_count", 0) for r in records),
                                       stored_parquet_bytes=sum((r.get("data") or {}).get("size", 0) for r in records))
                        except FileNotFoundError:
                            pass
                        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                            # One damaged manifest must not prevent a report for
                            # all the other days. Its row explicitly records failure.
                            row.update(status="request_error", error=repr(exc))
                        counts[row["status"]] += 1
                        writer.writerow(row)
        return counts


# 10. Command-line options and the overall run sequence.
def package_versions() -> dict:
    # Save library versions with each run so later differences can be traced to
    # the code/environment as well as to the requested data.
    result = {}
    for package in ("pandas", "numpy", "pyarrow", "requests", "exchange-calendars"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not_installed"
    return result


def parse_run_scope(argv: list[str] | None = None):
    # Translate command-line arguments into settings, symbols, and trading days.
    # This performs no downloads or output writes, so --plan can stop here safely.
    defaults = CollectorConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", type=str.upper, choices=[cfg.symbol for cfg in UNIVERSE])
    parser.add_argument("--start", default=defaults.start_date, help="Inclusive first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=defaults.end_date, help="Inclusive last date (YYYY-MM-DD)")
    parser.add_argument("--quote-interval", default=defaults.quote_interval, choices=QUOTE_INTERVALS)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--store-raw-payloads", action="store_true", help="Also preserve exact Theta response bytes")
    parser.add_argument("--refresh-no-data", action="store_true", help="Retry previously empty requests")
    parser.add_argument("--max-inflight-requests", type=int, default=defaults.max_inflight_requests,
                        help="Set within your Theta subscription's concurrency allowance")
    parser.add_argument("--max-requests-per-second", type=float, default=defaults.max_requests_per_second)
    parser.add_argument("--rate-symbols", nargs="+", default=list(RATE_SYMBOLS), type=str.upper, choices=RATE_SYMBOLS,
                        help="Theta rate series to collect; defaults to SOFR and all documented Treasury tenors")
    modes = parser.add_mutually_exclusive_group()
    # With neither special mode, collect date coverage, references, and panels.
    # The special modes limit the work to one of those supporting collections.
    modes.add_argument("--references-only", action="store_true", help="Collect dividends, splits, rates, and VIX without stock/option panels")
    modes.add_argument("--coverage-only", action="store_true", help="Refresh stock/VIX available-date lists and report missing sessions")
    parser.add_argument("--plan", action="store_true", help="Show scope without network requests or output writes")
    args = parser.parse_args(argv)
    try:
        if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (args.start, args.end)):
            raise ValueError("Dates must use YYYY-MM-DD")
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if not pd.Timestamp(defaults.start_date) <= start <= end <= pd.Timestamp(defaults.end_date):
            raise ValueError(f"Dates must be ordered and within {defaults.start_date} through {defaults.end_date}")
        cfg = replace(defaults, quote_interval=args.quote_interval, output_dir=args.output_dir.expanduser().resolve(),
                      store_raw_payloads=args.store_raw_payloads, refresh_no_data=args.refresh_no_data,
                      max_inflight_requests=args.max_inflight_requests,
                      max_requests_per_second=args.max_requests_per_second)
    except ValueError as exc:
        parser.error(str(exc))
    anchors = exchange_calendar().sessions_in_range(start, end).tz_localize(None)
    # These are trading dates used as loop labels, not midnight market events.
    # Weekends/holidays disappear here; actual intraday clocks stay timezone-aware.
    symbols = [cfg for cfg in UNIVERSE if args.symbols is None or cfg.symbol in args.symbols]
    return args, cfg, symbols, anchors


def main(argv: list[str] | None = None) -> int:
    # Start reading here for execution order. "panels" means the stock/option
    # observations organized by underlying and trading day, not fitted models.
    args, cfg, symbols, anchors = parse_run_scope(argv)
    panels = not (args.references_only or args.coverage_only)
    total = len(symbols) * len(anchors) if panels else 0
    print(f"Scope: {', '.join(s.symbol for s in symbols)}; {args.start} to {args.end}; {total} symbol-days")
    print(f"Vendor: ThetaData; quotes: {cfg.quote_interval}; trades: events with matched quotes; stock venue: {cfg.stock_venue}")
    print(f"Output: {cfg.output_dir}")
    if args.coverage_only:
        print("Coverage mode: available dates for stock quotes/trades and VIX; no history downloads")
    else:
        print(f"Required references: dividends/splits, {len(set(args.rate_symbols))} rate series, VIX EOD and {cfg.quote_interval} prices")
    if args.plan:
        # Preview ends before opening a network connection or creating output.
        return 0
    collector = Collector(cfg)
    run_id = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    # A run ID identifies this invocation. A policy ID identifies selection rules;
    # a request ID identifies one data pull. Reruns get new run records while
    # sharing compatible saved requests and session manifests.
    run_path = collector.directory / "runs" / f"{run_id}.json"
    run = {"run_id": run_id, "started_at_utc": utc_now(), "status": "running", "data_vendor": "ThetaData",
           "policy_id": cfg.policy_id, "policy": cfg.policy(), "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
           "scope": {"symbols": [s.symbol for s in symbols], "start": args.start, "end": args.end,
                     "references_only": args.references_only, "coverage_only": args.coverage_only,
                     "rate_symbols": args.rate_symbols},
           "code_sha256": file_hash(Path(__file__)), "python": sys.version, "platform": platform.platform(),
           "packages": package_versions(), "resumed_days": 0, "processed_days": 0}
    failed = reference_failures = reference_gaps = catalogue_errors = catalogue_gaps = 0
    exit_code = 0
    try:
        collector.store.client.ensure_available()
        # Hold the output lock for the run, including shared reference writes.
        with output_lock(cfg.output_dir):
            write_json(run_path, run)
            try:
                if not args.references_only:
                    # First record vendor date availability. Observed gaps are
                    # reported; they do not silently shorten the user's date range.
                    run["date_catalogue"] = f"coverage/{run_id}.json"
                    catalogue = collector.collect_coverage(symbols, anchors, run_id)
                    catalogue_errors, catalogue_gaps = catalogue["request_errors"], catalogue["series_with_gaps"]
                    collector.store.client.check_running()
                if not args.coverage_only:
                    # Fetch the common reference bundle before symbol-day work.
                    # Successful reference pulls also reuse the ordinary raw cache.
                    run["reference_ledger"] = f"references/{run_id}.json"
                    references = collector.collect_references(symbols, args.start, args.end, args.rate_symbols, run_id)
                    reference_failures = sum(r["status"] not in GOOD_REQUEST_STATUSES for r in references)
                    # A dividend/split endpoint can correctly return no events.
                    # Missing price/rate history is a coverage gap, not a zero.
                    reference_gaps = sum(r["status"] == "no_data" and not r["dataset"].startswith("corporate_")
                                         for r in references)
                    collector.store.client.check_running()
                if panels:
                    # Advance one trading day at a time, overlapping its symbols.
                    # Within each symbol, collect_day overlaps the option requests.
                    for day in anchors:
                        pending = []
                        for symbol in symbols:
                            if collector.resumable(symbol.symbol, day):
                                run["resumed_days"] += 1
                            else:
                                pending.append(symbol)
                        with collector.workers(cfg.max_symbol_day_workers) as pool:
                            # Finished sessions were omitted from pending; an
                            # incomplete session still reuses its valid raw downloads.
                            futures = {pool.submit(collector.collect_day, symbol, day): symbol for symbol in pending}
                            for future in as_completed(futures):
                                symbol = futures[future]
                                run["processed_days"] += 1
                                try:
                                    manifest = future.result()
                                    failed += manifest["status"] == "request_error"
                                    print(f"{symbol.symbol} {day.date()}: {manifest['status']}; "
                                          f"{manifest['selected_contract_count']} contracts, "
                                          f"{manifest['request_error_count']} failed requests")
                                except CollectionStopped:
                                    raise
                                except Exception as exc:
                                    failed += 1
                                    print(f"FAILED {symbol.symbol} {day.date()}: {exc!r}")
            except BaseException as exc:
                collector.store.client.stop(str(exc) or "Collection interrupted by the user.")
                run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "partial_failure",
                           error=repr(exc))
                raise
            finally:
                # Publish coverage on success, failure, or Ctrl+C. Keep the original
                # failure if generating this summary itself encounters a disk error.
                try:
                    run["coverage"] = collector.write_availability(symbols, anchors) if panels else {}
                except Exception as exc:
                    run["coverage_error"] = repr(exc)
                    exit_code = 1
                coverage = run.get("coverage", {})
                # Failure takes priority over a known gap. Completed requests do
                # not establish completeness of all vendor history/reference data.
                if run["status"] == "running":
                    if (exit_code or failed or reference_failures or catalogue_errors
                            or coverage.get("request_error", 0) or coverage.get("not_attempted", 0)):
                        exit_code = 1
                    elif reference_gaps or catalogue_gaps or coverage.get("unavailable", 0):
                        exit_code = 2
                    run["status"] = {0: "complete", 1: "partial_failure", 2: "coverage_gaps"}[exit_code]
                run.update(finished_at_utc=utc_now(), failed_days=max(int(failed), coverage.get("request_error", 0)),
                           reference_failures=reference_failures, reference_gaps=reference_gaps,
                           catalogue_errors=catalogue_errors, catalogue_series_with_gaps=catalogue_gaps)
                write_json(run_path, run)
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        print(f"Collector stopped: {exc}", file=sys.stderr)
        if run_path.exists():
            print(f"Run record: {run_path}", file=sys.stderr)
        return 1
    print(f"Finished: {run['processed_days']} processed, {run['resumed_days']} resumed, "
          f"{failed} failed days, {reference_failures} failed reference requests, "
          f"{reference_gaps + catalogue_gaps} reference/catalogue gaps")
    print(f"Run record: {run_path}")
    if run.get("date_catalogue"):
        print(f"Date coverage: {cfg.output_dir / run['date_catalogue']}")
    if run.get("reference_ledger"):
        print(f"Theta references: {cfg.output_dir / run['reference_ledger']}")
    if panels:
        print(f"Panel coverage: {collector.directory / 'availability.csv'}")
    return exit_code


if __name__ == "__main__":
    # Run only when launched as a script. Importing this file exposes its helpers
    # without starting collection; the returned integer becomes the process exit code.
    sys.exit(main())
