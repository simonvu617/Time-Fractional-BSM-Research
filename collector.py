"""Collect ThetaData inputs for the hourly/daily TFBSM pricing study.

Python 3.11+; dependencies: exchange-calendars, numpy, pandas>=2, pyarrow, requests.
Theta Terminal v3 must be running for downloads. No pricing or calibration runs here.

Preview requests without downloading:
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --plan
Collect a small sample without the earlier stock-history buffer:
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --lookback-sessions 0

Start with CollectorConfig for study choices, then Collector.collect_day for one
underlying/day. Finance and collection explanations sit beside the relevant code.

Under --output-dir, collection/<policy-id>/availability.csv summarizes coverage;
contracts/ records the selected sample; universes/ records observed candidates;
raw_cache/ holds retained observations and response metadata. These files describe
what arrived and what was missing, not whether the research models performed well.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO, TextIOWrapper
from itertools import islice
from pathlib import Path
import re
from uuid import uuid4
from urllib.parse import urlsplit
from typing import BinaryIO, Iterable

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
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
OUTPUT_SCHEMA_VERSION = "2026-09-08-selected-quotes-v5"
# Version 2 requires strict CSV record validation. Earlier parsed caches cannot
# prove that no field was silently lost, so their original files stay preserved
# but cannot satisfy new requests without reading verified CSV bytes again.
RAW_SCHEMA_VERSION = 2
# Keep the original default root so existing raw caches can be reused.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "multi_year_bsm_backtest_output"
# Standard stock history supports one-minute and coarser snapshots. The default
# is hourly; the same setting applies to stock, option, and index history.
QUOTE_INTERVALS = ("1m", "5m", "10m", "15m", "30m", "1h")
# Both statuses mean a request finished successfully. "no_data" means Theta
# returned no rows; it does not mean that an asset had zero prices or activity.
GOOD_REQUEST_STATUSES = {"available", "no_data"}
# A quote is an advertised price, not a completed trade. NBBO means the best
# displayed bid/ask across the reporting exchanges. Keeping both sides lets later
# research examine spreads instead of treating a midpoint as an executable price.
# These are required fields; any additional vendor columns are also retained.
QUOTE_FIELDS = (
    "bid_size", "bid_exchange", "bid", "bid_condition",  # Buying side: size, source, price, quote status.
    "ask_size", "ask_exchange", "ask", "ask_condition",  # Selling side; status can mark a halt/non-firm quote.
)
CONTRACT_FIELDS = ("symbol", "expiration", "strike", "right")  # Underlying, expiry date, exercise price, call/put.
DISCOVERY_COLUMNS = (*CONTRACT_FIELDS, "discovery_sources")
SELECTION_COLUMNS = (*CONTRACT_FIELDS, "contract_key", "dte_days", "selection_times", "discovery_sources")
# Rates support later discounting over an option's remaining life. The collector
# saves the reported series; selecting/interpolating a pricing rate happens later.
RATE_SYMBOLS = (
    "SOFR",  # Overnight USD borrowing benchmark; not itself a six-month rate.
    "TREASURY_M1", "TREASURY_M3", "TREASURY_M6", "TREASURY_Y1",  # M = months, Y = years; short end of the curve.
    "TREASURY_Y2", "TREASURY_Y3", "TREASURY_Y5", "TREASURY_Y7",  # Broader curve context retained by this bundle.
    "TREASURY_Y10", "TREASURY_Y20", "TREASURY_Y30",  # Not required just to match our <=180-day option maturities.
)
REFERENCE_COLUMNS = {
    "interest_rate_eod": ("created", "rate"),  # Report date and percent: 4.25 means 4.25%, not 0.0425.
    # Announcement: when the dividend was declared. Ex-date: when shares start
    # trading without that dividend entitlement. Payment: when cash is paid.
    # These differ, so a later pricing study needs more than one date per event.
    # Keep component/type identifiers so a later dividend schedule can distinguish
    # an event's breakdown from separate distributions before adding cash amounts.
    "corporate_dividend": ("announcement_date", "ex_dividend_date", "record_date", "payment_date",
                           "amount", "event_code", "is_component", "distribution_type"),
    # A split changes share count and quoted prices. Retain its terms so later
    # research can identify affected dates; this file does not adjust option deliverables.
    "corporate_split": ("effective_date", "before_shares", "after_shares", "split_ratio", "event_code"),
}
# Each date-range check must use the date on which that endpoint filters.
# A dividend's announcement/payment dates can legitimately fall outside it.
REPORT_DATE_COLUMNS = {"interest_rate_eod": "created", "corporate_dividend": "ex_dividend_date",
                       "corporate_split": "effective_date"}


@dataclass(frozen=True)
class SymbolConfig:
    # Labels travel with each session so results can later be compared by asset
    # group. They are fixed study labels, not historical classifications from Theta.
    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str


# Broad-market ETFs, sector ETFs, and individual companies provide the cross-asset
# comparison in the research question. This chosen list is not a reconstruction of
# all stocks listed in each past year; a name can have no data before its listing.
UNIVERSE = [
    # These group/sector labels are descriptive. They do not change which prices
    # Theta returns and they are not classifications of a day's market regime.
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
    # The date/maturity/strike grid below defines the current study sample.
    # These are adjustable research choices, not requirements imposed by TFBSM.
    base_url: str = "http://127.0.0.1:25503/v3"  # Local Theta Terminal relays the requests to the vendor.
    start_date: str = "2018-01-01"  # Requested study window; available history is checked separately.
    end_date: str = "2025-12-31"
    index_history_start: str = "2022-01-01"  # Standard's index limit; earlier VIX dates become reported gaps.
    # Earlier stock prices can support a volatility estimate at the study's first
    # date. Sixty trading sessions is a buffer, not a chosen calibration window;
    # it adds stock/rate history, not continuous histories of the option contracts.
    lookback_sessions: int = 60  # Exchange sessions, not 60 calendar days; 0 disables the buffer.
    option_rights: tuple[str, ...] = ("call", "put")  # Call: right to buy at the strike; put: right to sell.
    # DTE is calendar days to expiration, including weekends. Sampling several
    # maturities lets later research compare short- and longer-lived options.
    target_dtes: tuple[int, ...] = (7, 14, 30, 60, 120)  # Seek the nearest observed expiry to each target.
    max_expirations_per_day: int = 5  # Bound the daily option panel and number of bulk quote requests.
    # S/K = underlying price / strike. At S=$100: 0.80 targets K=$125; 1.20 targets
    # K=$83.33. S/K>1 is in-the-money for a call, out-of-the-money for a put.
    # Spanning both sides of 1 lets later research compare pricing across moneyness.
    moneyness_targets: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
    strikes_per_moneyness_target: int = 1  # Nearest listed strike; never synthesize a contract at the target.
    min_dte: int = 7  # This sample excludes expiration-day options and all expiries less than a week away.
    max_dte: int = 180  # Also caps broad OI/EOD/list requests, so longer maturities do not use storage.
    exchange_tz: str = "America/New_York"  # Session boundaries follow Eastern time through daylight saving.
    quote_interval: str = "1h"  # Seven normal-session snapshots; supports hourly/daily comparison, not trade gaps.
    near_close_minutes: int = 5  # Adds the daily comparison at 15:55, or 12:55 when the market closes at 13:00.
    raw_chunk_rows: int = 100_000  # Bounds memory per parsing batch; does not change sampling frequency.
    stock_venue: str = "utp_cta"  # Merged stock feeds for NBBO history, rather than the Nasdaq Basic default.
    # Re-select strikes as S moves during the day. These times lie on our hourly
    # grid; a 13:00 reference, for example, has no matching 1h sample from 09:30.
    selection_times: tuple[str, ...] = ("10:30:00", "13:30:00", "15:30:00")
    # Reject an older sampled stock price when choosing strikes: at 13:30, a
    # 12:30 sample fails this limit. This cannot detect how old the underlying
    # quote event was, because Theta's sampled response does not expose that age.
    max_stock_quote_age_seconds: int = 70  # Tolerance around the requested sample time, not an event-age test.
    max_symbol_day_workers: int = 4  # Process several underlying/day panels while requests wait for data.
    max_batch_workers: int = 4  # Overlap the individual downloads within a panel/reference batch.
    max_inflight_requests: int = 4  # One shared cap: Standard allows four across the entire account.
    max_requests_per_second: float = 0.0  # Zero disables artificial pacing; the simultaneous-request cap still applies.
    store_raw_payloads: bool = False  # True keeps full CSVs too, including unselected quotes; this costs extra disk.
    refresh_no_data: bool = False  # True rechecks successful empty replies in case Theta has since backfilled them.
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
                     "max_symbol_day_workers", "max_batch_workers",
                     "max_inflight_requests", "max_stock_quote_age_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.max_requests_per_second) or self.max_requests_per_second < 0:
            raise ValueError("max_requests_per_second must be nonnegative; 0 disables pacing")
        if self.max_inflight_requests > 4:
            raise ValueError("Standard allows at most four simultaneous requests across the account")
        if not isinstance(self.near_close_minutes, int) or not 1 <= self.near_close_minutes <= 30:
            raise ValueError("near_close_minutes must be an integer from 1 through 30")
        if not isinstance(self.raw_chunk_rows, int) or self.raw_chunk_rows <= 0:
            raise ValueError("raw_chunk_rows must be a positive integer")
        if not isinstance(self.lookback_sessions, int) or self.lookback_sessions < 0:
            raise ValueError("lookback_sessions must be a nonnegative integer")
        if (not self.selection_times or tuple(sorted(set(self.selection_times))) != self.selection_times
                or any(not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d", t) for t in self.selection_times)):
            raise ValueError("selection_times must be unique, ordered HH:MM:SS values")
        if self.quote_interval not in QUOTE_INTERVALS or self.stock_venue not in {"utp_cta", "nqb"}:
            raise ValueError("Unsupported quote interval or stock venue")

    def policy(self) -> dict:
        # Worker counts, scope, and the separate reference bundle do not change
        # which contracts are selected or invalidate already collected sessions.
        # For example, changing the S/K grid changes the policy ID; lowering the
        # parsing batch size does not. Saved responses can be reused when their
        # endpoint arguments, retained contract sets, and files match exactly.
        names = ("option_rights", "target_dtes", "max_expirations_per_day", "moneyness_targets",
                 "strikes_per_moneyness_target", "min_dte", "max_dte", "exchange_tz",
                 "quote_interval", "near_close_minutes", "stock_venue", "selection_times", "max_stock_quote_age_seconds")
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
    # None saves the full response. A tuple restricts stored option quotes to
    # these exact contract keys, while the HTTP request can still use strike=*.
    # This is a local storage rule, never an extra parameter sent to Theta.
    retained_contract_keys: tuple[str, ...] | None = None

    def identity(self) -> dict:
        identity = asdict(self)
        if self.retained_contract_keys is None:
            # Full-response requests retain their old cache identity.
            identity.pop("retained_contract_keys")
        else:
            # JSON reads lists, not tuples. Canonical order also means reordering
            # the selection table does not cause another identical download.
            identity["retained_contract_keys"] = sorted(set(self.retained_contract_keys))
        return identity

    @property
    def request_id(self) -> str:
        # Identical requests get the same cache key even in different runs.
        # Changing the retained contracts also changes this key: a narrower saved
        # sample must never satisfy a later request for a broader sample.
        return digest_json(self.identity())[:24]

    @property
    def required_columns(self) -> tuple[str, ...]:
        # A successful HTTP response must also look like the requested table.
        # This catches error pages or incompatible schemas before cache reuse.
        # Columns are checked for presence, not restricted to this list. If Theta
        # supplies an additional field, it remains in the saved raw table.
        kind = self.endpoint.rsplit("/", 1)[-1]
        if self.dataset in {"quoted_contracts", "traded_contracts"}:
            return CONTRACT_FIELDS
        if self.dataset in REFERENCE_COLUMNS:
            return REFERENCE_COLUMNS[self.dataset]
        if "/list/dates" in self.endpoint:
            return ("date",)
        fields = {"quote": ("timestamp", *QUOTE_FIELDS),
                  "price": ("timestamp", "price"),
                  "open_interest": ("timestamp", "open_interest"),
                  # OHLC are trade prices. Volume counts units traded; count is
                  # the number of trades. Neither is the number of quote updates.
                  "eod": ("created", "last_trade", "open", "high", "low", "close", "volume", "count")}
        return ((*CONTRACT_FIELDS,) if self.endpoint.startswith("/option/") else ()) + fields[kind]

    @property
    def report_date_column(self) -> str | None:
        return "date" if "/list/dates" in self.endpoint else REPORT_DATE_COLUMNS.get(self.dataset)

    def observation_semantics(self) -> dict:
        # Keep clocks distinct: a sampled-quote timestamp is a sample boundary,
        # not proof that the underlying quote was updated at that moment.
        # These descriptions are saved alongside each request. Later code can
        # read the meaning of a clock without guessing from the dataset name.
        kind = self.endpoint.rsplit("/", 1)[-1]
        if self.report_date_column:
            return {"kind": "dated_report", "date_column": self.report_date_column,
                    "publication_time_verified": False}
        if "/list/contracts/" in self.endpoint:
            return {"kind": "date_wide_observed_contracts", "intraday_listing_time_verified": False}
        if "/at_time/" in self.endpoint:
            return {"kind": "at_time_snapshot", "requested_time": self.params["time_of_day"],
                    "timestamp_role": "vendor_at_time_timestamp", "quote_event_time_available": False,
                    "purpose": "near_close_daily_comparison"}
        if kind == "quote":
            # A 10:30 sample can repeat a quote last updated at 10:12. The sample
            # timestamp alone cannot establish an 18-minute period without trades.
            return {"kind": "sampled_quotes", "timestamp_role": "sample_boundary",
                    "quote_event_time_available": False}
        if kind == "open_interest":
            # OI counts contracts still outstanding. Today's report describes the
            # previous trading day's close; it is not today's trading volume.
            # https://docs.thetadata.us/operations/option_history_open_interest.html
            return {"kind": "open_interest_report", "describes": "previous_trading_session_close",
                    "missing_report_means_zero": False}
        if kind == "price":
            return {"kind": "sampled_index_prices",
                    "unchanged_updates_may_be_omitted": True}
        # Theta creates its EOD report around 17:15 ET. Its last reported bid/ask
        # is therefore kept distinct from our specifically requested 15:55 quote.
        # https://docs.thetadata.us/operations/option_history_eod.html
        return {"kind": "end_of_day_report", "is_regular_session_close_quote": False}


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
        frame.to_parquet(temp, index=False, compression="zstd")  # Lossless compression; no price rounding.


def file_receipt(path: Path, root: Path, frame: pd.DataFrame | None = None) -> dict:
    # Relative paths keep receipts usable if the whole output folder is moved.
    # Table receipts additionally remember row counts and column names.
    # A SHA-256 fingerprint is a compact identifier of the bytes. It helps detect
    # a changed file; it does not certify the economic accuracy of Theta's data.
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
def csv_frames(payload: BinaryIO, chunk_rows: int) -> Iterable[pd.DataFrame]:
    """Read complete CSV records in bounded batches without inferring an index."""
    # pandas' chunked C parser can silently drop extra fields at a batch boundary.
    # csv.reader recognizes quoted commas/newlines, while our width check rejects
    # both extra and missing fields before a row enters the saved table.
    # TextIOWrapper borrows the response file: detach it on every exit so the
    # caller can still retain the exact bytes after a malformed CSV is rejected.
    text = TextIOWrapper(payload, encoding="utf-8-sig", newline="")
    reader = csv.reader(text, strict=True)
    try:
        header = next((row for row in reader if row), None)
        if header is None:
            return
        if any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV header has empty or duplicate column names")
        rows, emitted = [], False
        for row in reader:
            if not row:  # An empty physical line is not a market observation.
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV record ending on line {reader.line_num}: "
                                 f"expected {len(header)} fields, got {len(row)}")
            rows.append(row)
            if len(rows) == chunk_rows:
                yield pd.DataFrame(rows, columns=header, dtype="string")
                rows, emitted = [], True
        if rows or not emitted:
            # A header-only response must still undergo required-column checks.
            yield pd.DataFrame(rows, columns=header, dtype="string")
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV near line {reader.line_num}: {exc}") from exc
    finally:
        text.detach()


def parse_vendor_clock(values: pd.Series, exchange_tz: str) -> pd.Series:
    # Naive Theta clocks are exchange local. Aware clocks keep their stated
    # offset. Date-only interest-rate reports deliberately do not use this.
    # "Naive" means the text has no timezone offset. "Aware" means it includes
    # one, such as -05:00. Both become separate UTC columns for later alignment;
    # conversion does not alter the original vendor timestamp text.
    text = values.astype("string").str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    parsed.loc[aware] = pd.to_datetime(text.loc[aware], format="mixed", errors="coerce", utc=True)
    naive = pd.to_datetime(text.loc[~aware], format="mixed", errors="coerce")
    # Daylight-saving transitions can make a local clock ambiguous or impossible.
    # Keep those parsed values missing (NaT) instead of guessing their UTC time.
    parsed.loc[~aware] = naive.dt.tz_localize(
        exchange_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    # For example, 09:30 New York is 14:30 UTC in winter and 13:30 UTC in summer.
    # A fixed UTC offset would misalign stock and option observations seasonally.
    return parsed


def raw_frame_with_diagnostics(frame: pd.DataFrame, request: Request, cfg: CollectorConfig) -> tuple[pd.DataFrame, dict]:
    # Return the raw table plus separate diagnostics. Counts of duplicates,
    # bad quotes, and clock problems describe the data; they do not delete rows.
    result = frame.copy()
    diagnostics = {"duplicate_rows": int(frame.duplicated().sum()), "clocks": {}}  # Count repeats; do not drop them.
    # A repeated row is counted, not removed. A sequence code, condition, or
    # repeated price may matter when distinguishing events from sampling artifacts.
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
        # Numeric conversion here is just for these counts; the saved bid/ask
        # columns remain the vendor's text, even when a value cannot be parsed.
        bid, ask = (pd.to_numeric(frame[side], errors="coerce") for side in ("bid", "ask"))
        # A zero bid can occur for an option with no displayed buying interest.
        # A crossed quote (bid > ask) is problematic for midpoint interpretation.
        # Report both cases; removing them here would change the liquidity sample.
        diagnostics.update(invalid_bid_ask_rows=int((~np.isfinite(bid) | ~np.isfinite(ask)).sum()),
                           nonpositive_bid_ask_rows=int((bid.le(0) | ask.le(0)).sum()),
                           crossed_quote_rows=int(bid.gt(ask).sum()))
    report_date = request.report_date_column
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
    if (request.endpoint.endswith("/history/quote") and "collector_timestamp_utc" in result
            and request.params.get("strike") != "*"):
        # Mixing timestamps from different strikes would hide missing samples.
        # Bulk responses are checked per selected contract in session_coverage.
        interval = request.params.get("interval", "tick")
        if interval != "tick":
            seconds = interval_seconds(interval)
            unique = result["collector_timestamp_utc"].dropna().drop_duplicates().sort_values()
            gaps = unique.diff().dt.total_seconds().dropna()
            # At 1h, 10:30 -> 13:30 leaves two missing interior slots (11:30,
            # 12:30): gap / interval - 1. The small tolerance avoids inventing a
            # missing slot from floating-point noise. Coverage checks endpoints too.
            diagnostics["absent_interior_sample_slots"] = int(
                np.maximum(np.ceil(gaps.to_numpy() / seconds - 1e-9) - 1, 0).sum())
    return result, diagnostics


def response_identity_issues(frame: pd.DataFrame, request: Request, cfg: CollectorConfig) -> list[str]:
    """Flag misrouted responses without dropping or correcting vendor records."""
    # A table for the wrong symbol, option, or date can look otherwise plausible.
    # Record such mismatches as a failed response so it cannot silently satisfy
    # a different request. Its original rows remain available for inspection.
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
        # Bulk OI explicitly requests all expirations/strikes/rights. Validate
        # returned identities without treating the wildcard as a literal contract.
        if ((request.params.get("expiration") not in {None, "*"}
                and expiry.ne(pd.Timestamp(request.params["expiration"])).any())
                or (request.params.get("strike") not in {None, "*"}
                    and strike.ne(float(request.params["strike"])).any())
                or (request.params.get("right") in {"call", "put"}
                    and right.ne(request.params["right"]).any())):
            issues.append("unexpected_contract_identity")
        if "max_dte" in request.params:
            requested_day = pd.Timestamp(request.params.get("date", request.params.get("start_date")))
            # The limit is inclusive: an expiry 180 calendar days away is allowed;
            # 181 is outside the requested study scope even if Theta returns it.
            if (expiry - requested_day).dt.days.gt(request.params["max_dte"]).any():
                issues.append("expiration_outside_requested_dte")
    start = request.params.get("date", request.params.get("start_date"))
    # Compare market records on their exchange-local calendar date. A UTC date
    # can differ from the local date, so it is not used directly for this check.
    end = request.params.get("date", request.params.get("end_date"))
    report_date = request.report_date_column
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


class RawDiagnostics:
    """Accumulate small quality summaries while raw rows are written in batches."""
    # Only counts, date summaries, and the last clock of each batch carry forward.
    # The entire bulk response does not need to stay in memory to describe it.

    def __init__(self, request: Request, cfg: CollectorConfig):
        self.request, self.cfg = request, cfg
        self.rows = self.chunks = 0
        self.quality = {"clocks": {}}
        self.issues, self.report_dates = set(), set()
        self.last_clocks = {}
        self.report_missing = 0

    def add(self, raw: pd.DataFrame) -> pd.DataFrame:
        frame, quality = raw_frame_with_diagnostics(raw, self.request, self.cfg)
        self.rows += len(frame)
        self.chunks += 1
        # Global all-missing checks happen in finish(), not separately per chunk.
        self.issues.update(issue for issue in response_identity_issues(raw, self.request, self.cfg)
                           if issue != "no_parseable_report_dates")
        for name, value in quality.items():
            if name not in {"clocks", "report_dates"}:
                self.quality[name] = self.quality.get(name, 0) + value
        for name, info in quality["clocks"].items():
            parsed = frame[f"collector_{name}_utc"]
            if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
                parsed = parse_vendor_clock(raw[name], self.cfg.exchange_tz)
            valid = parsed.dropna()
            target = self.quality["clocks"].setdefault(name, {
                "unparseable_or_missing": 0, "out_of_order_transitions": 0,
                "first_utc": None, "last_utc": None})
            for count in ("unparseable_or_missing", "out_of_order_transitions"):
                target[count] += info[count]
            if valid.empty:
                continue
            previous = self.last_clocks.get(name)
            # A boundary still belongs to the same vendor response. Compare the
            # first clock here with the last one in the previous batch, so an
            # out-of-order row or missing sample at that boundary is not overlooked.
            if previous is not None:
                target["out_of_order_transitions"] += int(valid.iloc[0] < previous)
                if name == "timestamp" and "absent_interior_sample_slots" in quality:
                    gap = (valid.iloc[0] - previous).total_seconds() / interval_seconds(self.request.params["interval"])
                    self.quality["absent_interior_sample_slots"] += max(int(np.ceil(gap - 1e-9)) - 1, 0)
            self.last_clocks[name] = valid.iloc[-1]
            for key, choose in (("first_utc", min), ("last_utc", max)):
                target[key] = choose((v for v in (target[key], info[key]) if v is not None), key=pd.Timestamp)
        column = self.request.report_date_column
        if column and column in raw:
            dates = pd.to_datetime(raw[column].str.strip(), format="mixed", errors="coerce")
            self.report_dates.update(dates.dropna().dt.strftime("%Y-%m-%d"))
            self.report_missing += int(dates.isna().sum())
        return frame

    def finish(self) -> dict:
        # "No usable clock anywhere" can only be decided after reading all
        # batches. A first batch with blank clocks does not establish that alone.
        if self.request.report_date_column:
            self.quality["report_dates"] = {
                "column": self.request.report_date_column, "unparseable_or_missing": self.report_missing,
                "first": min(self.report_dates) if self.report_dates else None,
                "last": max(self.report_dates) if self.report_dates else None,
                "unique_count": len(self.report_dates), "publication_time_verified": False}
            if self.report_missing:
                self.issues.add("invalid_report_dates" if self.report_dates else "no_parseable_report_dates")
        else:
            primary = next((name for name in ("timestamp", "trade_timestamp", "created")
                            if name in self.quality["clocks"]), None)
            if primary and self.rows and self.quality["clocks"][primary]["unparseable_or_missing"] == self.rows:
                self.issues.add("no_parseable_report_dates")
        # Exact global deduplication would need memory proportional to the full
        # response. Count before storage filtering and label the lower bound.
        self.quality.update(read_chunks=self.chunks, duplicate_count_is_lower_bound=self.chunks > 1,
                            duplicate_count_scope="within_read_chunks" if self.chunks > 1 else "whole_response",
                            response_identity_issues=sorted(self.issues))
        clock = self.quality["clocks"].get("timestamp", {})
        if "absent_interior_sample_slots" in self.quality and clock.get("out_of_order_transitions"):
            self.quality["absent_interior_sample_slots"] = None
            self.quality["sample_gap_diagnostic_unavailable_reason"] = "out_of_order_timestamps"
        return self.quality


def interval_seconds(interval: str) -> float:
    # API "m" means minutes; pandas also accepts other, ambiguous abbreviations.
    if interval.endswith("ms"):
        return float(interval[:-2]) / 1000
    return float(interval[:-1]) * {"s": 1, "m": 60, "h": 3600}[interval[-1]]


def format_strike(value) -> str:
    # Decimal handles the contract's dollar strike without float formatting
    # artifacts. This is request-identity formatting, not option-price rounding.
    # Use decimal arithmetic for the URL and contract key: binary floats can
    # create spurious digits. Reject values beyond the supported 0.001 precision.
    try:
        strike = Decimal(str(value))
        if not strike.is_finite() or strike <= 0 or strike != strike.quantize(Decimal("0.001")):
            raise ValueError(f"Invalid option strike: {value!r}")
    except InvalidOperation as exc:
        # Treat malformed vendor strikes like other invalid response values, so
        # the caller can retain the original response for inspection and retry.
        raise ValueError(f"Invalid option strike: {value!r}") from exc
    return format(strike, ".3f").rstrip("0").rstrip(".")


def option_contract_keys(frame: pd.DataFrame) -> pd.Series:
    # Normalize identities only for matching. The vendor columns themselves keep
    # their exact text, including strike spelling, quote precision, and blanks.
    return (frame["symbol"] + "|" + pd.to_datetime(frame["expiration"], format="mixed").dt.strftime("%Y-%m-%d")
            + "|" + frame["strike"].map(format_strike) + "|"
            + frame["right"].str.lower().replace({"c": "call", "p": "put"}))


# 5. Talk to Theta Terminal: connections, shared request limits, and retries.
# This layer returns bytes plus HTTP details. Parsing and saving happen below.
class CollectionStopped(RuntimeError):
    """Stop scheduling work while retaining requests that already finished."""


class ThetaClient:
    # This class handles transport only. RequestStore decides how the returned
    # bytes become tables and whether they are suitable for cache reuse.
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
                self.next_allowed = (max(time.monotonic(), self.next_allowed) + 1 / self.cfg.max_requests_per_second
                                     if self.cfg.max_requests_per_second else time.monotonic())
            # Waiting workers wake promptly on cancellation, instead of starting
            # another HTTP request after the user or another worker stopped the run.
            self.stop_event.wait(wait)
            self.check_running()
            yield

    def download(self, request: Request, payload: BinaryIO) -> dict:
        # At most six attempts. Retry temporary failures such as rate limits
        # and server errors; return other failures so they can be recorded.
        meta = {}
        for attempt in range(6):
            started = time.perf_counter()
            retry_after = 0.0
            payload.seek(0)
            payload.truncate()
            # A retry starts the same response file from scratch. Appending would
            # mix two attempts into one apparent dataset and duplicate observations.
            fingerprint = hashlib.sha256()
            meta = {"attempts": attempt + 1}
            try:
                with self.request_slot():
                    with self.session().get(self.cfg.base_url.rstrip("/") + request.endpoint,
                                            params=request.params, timeout=(10, 120), stream=True) as response:
                        meta.update(request_url=response.url, status_code=response.status_code,
                                    response_headers=dict(response.headers))
                        # Hold the request slot until all bytes arrive. A large
                        # bulk response goes to disk instead of response.content.
                        for piece in response.iter_content(chunk_size=256 * 1024):
                            payload.write(piece)
                            fingerprint.update(piece)
                meta.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                            payload_sha256=fingerprint.hexdigest(), payload_bytes=payload.tell())
                payload.seek(0)
                if response.status_code in {200, 472}:
                    # 200 is an ordinary response; Theta uses 472 for no data.
                    # The storage layer records an empty result separately.
                    # HTTP success alone is not a valid table: schema and identity
                    # checks still happen after the CSV is read.
                    return meta
                preview = payload.read(500).decode("utf-8", errors="replace")
                payload.seek(0)
                meta["error"] = f"HTTP {response.status_code}: {preview}"
                # Retry 429 (OS throttling), 474 (lost vendor connection), 571
                # (vendor restarting), and the listed server errors. None of these
                # establishes that the requested market had no observations.
                if response.status_code not in {429, 474, 500, 502, 503, 504, 571} or attempt == 5:
                    # Permissions, invalid parameters, and terminal configuration
                    # need action, not thousands more requests. Sustained terminal
                    # disconnections/rate-limit failures also stop new work.
                    if response.status_code in {400, 401, 403, 404, 429, 471, 473, 474, 475, 476, 478, 571}:
                        self.stop(f"Theta HTTP {response.status_code} for {request.endpoint}: "
                                  "check Theta access, terminal state, and request settings, then rerun.")
                    return meta
                try:
                    retry_after = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    pass
            except requests.RequestException as exc:
                # Keep a final truncated response as failure evidence, but never
                # parse its valid-looking prefix as a complete market-data pull.
                meta.update(error=repr(exc), status_code=None, response_incomplete=True,
                            payload_sha256=fingerprint.hexdigest(), payload_bytes=payload.tell())
                payload.seek(0)
                if attempt == 5:
                    self.stop(f"Theta connection failed after six attempts for {request.endpoint}; rerun when it is available.")
                    return meta
            # Increase the delay between attempts, considering Theta's requested
            # Retry-After delay too. Both delays are bounded by the limits below.
            self.stop_event.wait(max(min(30.0, 2 ** attempt), min(max(retry_after, 0), 120.0)))  # Wait longer after repeated failures.
            self.check_running()
        return meta


# 6. Save response data with the exact contract scope needed for later reuse.
# RequestStore is the common route for stocks, options, and reference data.
class RequestStore:
    # This library holds request results and their storage scope. Session manifests
    # borrow receipts from it; they do not own separate copies of the data.
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.root = cfg.output_dir
        self.client = ThetaClient(cfg)
        # Bind receipts to the code at startup and avoid rereading this whole
        # source file for every response in a multi-year collection.
        self.code_sha256 = file_hash(Path(__file__))
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
        # A reusable empty response is still useful evidence. It avoids asking
        # the same unavailable question every run unless --refresh-no-data is set.
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
                "params": request.params,
                "status": meta["status"], "row_count": meta.get("row_count", 0),
                "status_code": meta.get("status_code"),
                "observation_semantics": request.observation_semantics(),
                "retention": meta.get("retention", {}),
                "error": meta.get("error", ""), "data": meta.get("data"),
                "payload": meta.get("payload"), "metadata": file_receipt(meta_path, self.root)}

    def collect(self, request: Request, *, refresh: bool = False) -> dict:
        # Order: reuse current cache -> import a compatible older cache -> download.
        # Date-catalogue refreshes skip current cache reuse to see newly added dates.
        # The returned object is a receipt, not the potentially huge market table.
        # Call read() for a small discovery table or iter_frames() for bulk rows.
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
            self.root.mkdir(parents=True, exist_ok=True)
            # The temporary response is closed/deleted on every exit path. Keeping
            # it on disk allows parsing and optional exact-byte retention without
            # loading a whole bulk response into memory or making extra requests.
            with tempfile.TemporaryFile(dir=self.root) as payload:
                # There are two batch sizes: the network moves blocks of bytes;
                # the CSV reader later groups complete records. Batch size is a
                # memory setting, not quote frequency. The temporary file needs space for
                # one response even when exact successful CSV retention is disabled.
                response_meta = self.client.download(request, payload)
                status = response_meta.get("status_code")
                empty = pd.DataFrame(columns=request.required_columns)
                if status not in {200, 472}:
                    return self.save(request, empty, response_meta, payload, "request_error")
                if status == 472:
                    return self.save(request, empty, response_meta, payload)
                try:
                    content_type = response_meta.get("response_headers", {}).get("Content-Type", "").lower()
                    if content_type and not any(t in content_type for t in ("csv", "text/plain", "octet-stream")):
                        raise ValueError(f"Unexpected response content type: {content_type}")
                    # Text fields preserve precision, condition codes, and blanks.
                    frames = csv_frames(payload, self.cfg.raw_chunk_rows)
                except (ValueError, UnicodeError) as exc:
                    response_meta["error"] = f"Invalid CSV response: {exc}"
                    return self.save(request, empty, response_meta, payload, "invalid_response")
                try:
                    return self.save(request, frames, response_meta, payload)
                finally:
                    frames.close()

    def save(self, request: Request, frames: pd.DataFrame | Iterable[pd.DataFrame], response_meta: dict,
             payload: bytes | BinaryIO | None = None, status: str | None = None) -> dict:
        # Save the table even when it is empty or invalid, with an explicit status.
        # Retained evidence lets us distinguish absent data from a broken request.
        # Every collected response gets new files, including failures. Overwriting
        # data.parquet in place would change data referenced by an earlier run.
        cache_directory = self.directory(request)
        directory = cache_directory / "responses" / uuid4().hex
        if isinstance(frames, pd.DataFrame):
            source = frames
            frames = (source.iloc[start:start + self.cfg.raw_chunk_rows]
                      for start in range(0, max(len(source), 1), self.cfg.raw_chunk_rows))
        diagnostics = RawDiagnostics(request, self.cfg)
        retained = None if request.retained_contract_keys is None else set(request.retained_contract_keys)
        excluded_rows = 0
        # One Parquet file can contain many row groups. The writer below appends
        # groups in incoming order; batch boundaries do not create new datasets.
        data_path = directory / "data.parquet"
        writer, columns = None, []
        with atomic_output(data_path) as temp:
            try:
                for raw in frames:
                    missing = set(request.required_columns) - set(raw.columns)
                    # collector_ is reserved for added clocks, never vendor fields.
                    if missing or any(str(column).startswith("collector_") for column in raw.columns):
                        status = status or "invalid_response"
                        response_meta["error"] = f"Missing required columns {sorted(missing)} or reserved collector_ column"
                    frame = diagnostics.add(raw.astype("string").fillna(""))
                    if retained is not None and not missing:
                        # Check every parsed row before filtering: a bad response
                        # must not look valid just because its bad rows were outside
                        # our sample. Retain every observation of each selected
                        # contract, including repeated prices and duplicate rows.
                        keep = option_contract_keys(frame).isin(retained)  # Match symbol + expiry + strike + call/put.
                        excluded_rows += int((~keep).sum())
                        frame = frame.loc[keep]  # Keep all times/fields for selected contracts, even a zero bid or duplicate.
                    # The Arrow table connects pandas to the Parquet writer.
                    # preserve_index=False omits pandas' artificial row labels;
                    # all vendor columns and their order remain in the table.
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        columns = list(frame.columns)
                        writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                    # A batch with only unselected quotes still establishes the
                    # schema, but needs no empty row group or repeated footer data.
                    if len(table):
                        writer.write_table(table)
            except (ValueError, UnicodeError) as exc:
                # A malformed later chunk invalidates the request even if its
                # prefix parsed. Preserve the full response for inspection/retry.
                status = status or "invalid_response"
                response_meta["error"] = f"Invalid CSV/table response: {exc}"
            finally:
                if writer is not None:
                    writer.close()
            if writer is None:
                frame = diagnostics.add(pd.DataFrame(columns=request.required_columns, dtype="string"))
                columns = list(frame.columns)
                frame.to_parquet(temp, index=False, compression="zstd")
        # The receipt describes rows actually written, including a retained prefix
        # of an invalid response, not rows merely seen before a conversion failed.
        with pq.ParquetFile(data_path) as parquet:
            stored_rows, columns = parquet.metadata.num_rows, parquet.schema_arrow.names
        quality = diagnostics.finish()
        if quality["response_identity_issues"]:
            status = status or "invalid_response"
            response_meta["error"] = ", ".join(quality["response_identity_issues"])
        if status is None:
            # A nonempty bulk reply can contain none of our selected contracts.
            # Its receipt then has zero stored rows and explicit exclusions;
            # coverage still reports the absent selected quotes as gaps.
            status = "no_data" if diagnostics.rows == 0 else "available"
        # The request record ties four things together: what was asked, what
        # arrived, how it was interpreted, and exactly which files were saved.
        meta = {"request": request.identity(), "request_id": request.request_id,
                "raw_schema_version": RAW_SCHEMA_VERSION, "timestamp_timezone": self.cfg.exchange_tz,
                "fetched_at_utc": response_meta.pop("fetched_at_utc", utc_now()), "saved_at_utc": utc_now(),
                "status": status, "row_count": stored_rows, "quality": quality,
                "quality_scope": "parsed_rows_before_storage_filter",  # Diagnostics may include excluded strikes.
                "retention": {"mode": "full_response" if retained is None else "selected_contracts",
                              "parsed_rows": diagnostics.rows, "excluded_rows": excluded_rows},
                "observation_semantics": request.observation_semantics(),
                "collector_code_sha256": self.code_sha256, **response_meta}
        meta["data"] = {**file_receipt(data_path, self.root), "rows": stored_rows, "columns": columns}
        # Preserve unsuccessful responses even without --store-raw-payloads.
        # They explain schema errors, entitlement failures and vendor messages.
        # These CSVs contain the full vendor reply. Enabling them also keeps the
        # unselected strikes that the Parquet storage filter deliberately excludes.
        if payload is not None and (self.cfg.store_raw_payloads or status not in GOOD_REQUEST_STATUSES):
            payload_path = directory / "raw_response.csv"
            with atomic_output(payload_path) as temp:
                source = BytesIO(payload) if isinstance(payload, bytes) else payload
                source.seek(0)
                with temp.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=256 * 1024)
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
        # Revalidate the original CSV bytes. An older parsed table alone cannot
        # establish that its parser preserved every input field.
        if ("*" in (request.params.get("expiration"), request.params.get("strike"))
                or request.dataset not in {"quoted_contracts", "option_open_interest",
                                    "stock_quotes_" + self.cfg.quote_interval, "option_quotes_" + self.cfg.quote_interval}):
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
            if payload is None or hashlib.sha256(payload).hexdigest() != old.get("payload_sha256"):
                return None
            if old.get("status_code") == 200:
                frames = list(csv_frames(BytesIO(payload), self.cfg.raw_chunk_rows))
                frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=request.required_columns)
            elif old.get("status_code") == 472:
                frame = pd.DataFrame(columns=request.required_columns)
            else:
                return None
            if set(request.required_columns) - set(frame.columns):
                return None
            meta = {name: old[name] for name in ("request_url", "status_code", "response_headers",
                    "payload_sha256", "payload_bytes", "fetched_at_utc") if name in old}
            meta.update(legacy_source=path.relative_to(self.root).as_posix(),
                        legacy_csv_revalidated=True)
            return frame, meta, payload
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def read(self, record: dict) -> pd.DataFrame:
        # Selection code only consumes successful responses. Failed responses
        # remain on disk for inspection but are not used to select contracts.
        if record["status"] not in GOOD_REQUEST_STATUSES or not record.get("data"):
            return pd.DataFrame()
        return pd.read_parquet(self.root / record["data"]["path"])

    def iter_frames(self, record: dict, columns: list[str] | None = None):
        # In particular, finding three stock references must not reload the entire
        # table that we just carefully wrote in bounded batches.
        if record["status"] in GOOD_REQUEST_STATUSES and record.get("data"):
            with pq.ParquetFile(self.root / record["data"]["path"]) as parquet:
                for batch in parquet.iter_batches(batch_size=self.cfg.raw_chunk_rows, columns=columns):
                    yield batch.to_pandas()


# 7. Build requests using the exchange calendar and each endpoint's arguments.
# Cache the calendar object so every request does not reconstruct it.
@lru_cache(maxsize=1)
def exchange_calendar():
    return xcals.get_calendar("XNYS", start="2012-01-01", end="2026-12-31")


def session_bounds(day: pd.Timestamp, cfg: CollectorConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Use the stock exchange's actual session so an early close does not generate
    # expected 14:30/15:30 samples after trading ended. The window also bounds the
    # option requests; any option trading after the stock session is outside this study.
    # Read the day's actual open/close, including early closes, instead of
    # assuming every weekday has a full 09:30-16:00 session.
    # The calendar gives UTC instants; conversion produces New York local times
    # with the correct daylight-saving offset for that date.
    calendar = exchange_calendar()
    return (calendar.session_open(day).tz_convert(cfg.exchange_tz),
            calendar.session_close(day).tz_convert(cfg.exchange_tz))


def history_request(cfg: CollectorConfig, asset: str, kind: str, symbol: str,
                    day: pd.Timestamp, contract: dict | None = None) -> Request:
    params = {"symbol": symbol, "date": day.strftime("%Y%m%d"), "format": "csv"}
    if asset == "option":
        if contract is None and kind in {"open_interest", "eod"}:
            # One wildcard report supplies OI or daily volume/count within the
            # study's maturity limit. "both" explicitly includes calls and puts.
            params.update(expiration="*", strike="*", right="both", max_dte=cfg.max_dte)
        elif contract is None:
            raise ValueError("Option history requires an observed contract")
        else:
            strike = contract.get("strike", "*")
            params.update(expiration=contract["expiration"], strike="*" if strike == "*" else format_strike(strike),
                          right=contract["right"])
    if kind in {"quote", "price"}:
        # Theta returns the latest quote at each boundary, not an average quote
        # over the hour. Asking stock/options on the same grid supports matching S
        # and option prices later without silently using different observation times.
        params["interval"] = cfg.quote_interval
        dataset = f"{asset}_{kind}s_{cfg.quote_interval}"
    else:
        dataset = f"{asset}_{kind}"
    if kind == "eod":
        # EOD supplies daily trade volume/count and OHLC. It has a different role
        # from the near-close quote used for daily pricing. Theta requires a date
        # range here, even when both endpoints are the same day.
        params.pop("date")
        params.update(start_date=day.strftime("%Y%m%d"), end_date=day.strftime("%Y%m%d"))
    elif kind != "open_interest":
        # OI (open interest) is a report of outstanding contracts, not a stream
        # of trades. Intraday start/end-time arguments apply to the other pulls.
        opened, closed = session_bounds(day, cfg)
        params.update(start_time=opened.strftime("%H:%M:%S"), end_time=closed.strftime("%H:%M:%S"))
        if asset == "stock":
            params["venue"] = cfg.stock_venue  # Explicit merged feed avoids silently using Nasdaq Basic.
    return Request(dataset, f"/{asset}/history/{kind}", params)


def near_close_request(cfg: CollectorConfig, asset: str, symbol: str,
                       day: pd.Timestamp, expiration: str | None = None) -> Request:
    # At-time is a documented snapshot endpoint, so we do not assume an hourly
    # response includes 15:55 or treat a 17:15 EOD quote as the market close.
    # Minute boundaries also avoid Theta's documented slow sub-minute lookup.
    _, closed = session_bounds(day, cfg)
    at = closed - pd.Timedelta(minutes=cfg.near_close_minutes)  # One matching daily clock for stock and options.
    kind = "price" if asset == "index" else "quote"
    params = {"symbol": symbol, "start_date": day.strftime("%Y%m%d"),
              "end_date": day.strftime("%Y%m%d"), "time_of_day": at.strftime("%H:%M:%S.000"), "format": "csv"}
    if asset == "option":
        if expiration is None:
            raise ValueError("Option near-close requests require a selected expiration")
        params.update(expiration=expiration, strike="*", right="both")
    elif asset == "stock":
        params["venue"] = cfg.stock_venue
    return Request(f"{asset}_{kind}s_near_close", f"/{asset}/at_time/{kind}", params)


def shared_day_requests(cfg: CollectorConfig, symbol: str, day: pd.Timestamp) -> list[Request]:
    # Seven shared requests, independent of the number of selected contracts.
    # Keep dated trade LISTS for discovery, without downloading individual trades.
    return [history_request(cfg, "stock", "quote", symbol, day),  # Hourly S observations and strike-selection references.
            near_close_request(cfg, "stock", symbol, day),  # S at the same clock as the daily option comparison.
            history_request(cfg, "stock", "eod", symbol, day),  # Underlying's daily trading activity.
            # Dated lists identify contracts observed that day. Using a present-day
            # chain instead would miss expired contracts and distort historical selection.
            *[Request(dataset, f"/option/list/contracts/{kind}",
                      {"symbol": symbol, "date": day.strftime("%Y%m%d"), "max_dte": cfg.max_dte, "format": "csv"})
              for kind, dataset in (("quote", "quoted_contracts"), ("trade", "traded_contracts"))],
            history_request(cfg, "option", "open_interest", symbol, day),  # Outstanding positions, including quiet contracts.
            history_request(cfg, "option", "eod", symbol, day)]  # Option volume/trade counts without individual trades.


def option_quote_requests(cfg: CollectorConfig, symbol: str, day: pd.Timestamp,
                          selected: pd.DataFrame) -> list[Request]:
    # Theta accepts all strikes/rights for a specific expiration in one request.
    # Download in bulk for speed, but write quotes only for selected identities.
    # The full selection is part of each cache key, so changing the sample cannot
    # silently reuse files that omitted the newly requested contracts.
    requests_to_make = []
    for expiration in sorted(selected["expiration"].unique()):
        family = {"expiration": expiration, "strike": "*", "right": "both"}
        # Same bulk HTTP request, different possible stored sample. Include the
        # exact keys so a file keeping $100 calls cannot satisfy a later $105-call study.
        keys = tuple(sorted(selected.loc[selected["expiration"].eq(expiration), "contract_key"].unique()))
        requests_to_make.extend([
            replace(history_request(cfg, "option", "quote", symbol, day, family), retained_contract_keys=keys),
            replace(near_close_request(cfg, "option", symbol, day, expiration), retained_contract_keys=keys)])
    return requests_to_make


def collection_windows(cfg: CollectorConfig, start: str, end: str) -> dict:
    # The study dates still control option selection. Earlier history supports
    # later calibration choices; later corporate events cover the possible life
    # of every selected option (selection already enforces max_dte).
    sessions = exchange_calendar().sessions.tz_localize(None)
    before = sessions[sessions < pd.Timestamp(start)]
    if cfg.lookback_sessions > len(before):
        raise ValueError("lookback_sessions exceeds the available exchange-calendar history")
    lookback = before[-cfg.lookback_sessions:] if cfg.lookback_sessions else before[:0]  # Skip holidays/weekends.
    return {"study_start": start, "study_end": end,
            "history_start": str(lookback[0].date()) if len(lookback) else start,
            "corporate_action_end": str((pd.Timestamp(end) + pd.Timedelta(days=cfg.max_dte)).date()),  # Events during remaining option life.
            "lookback_dates": list(lookback.strftime("%Y-%m-%d"))}


def reference_requests(cfg: CollectorConfig, symbols: list[SymbolConfig], start: str, end: str,
                       rate_symbols: list[str], *, include_stock_lookback: bool = False):
    """One Theta-only bundle. Rates/actions are date reports; VIX prices are intraday."""
    # Actions are company-specific; the rate curves and market-wide VIX series are
    # downloaded once and shared across underlyings, avoiding 21 duplicate copies.
    windows = collection_windows(cfg, start, end)
    window = {"start_date": windows["history_start"], "end_date": end, "format": "csv"}
    for symbol in symbols:
        for kind in ("dividend", "split"):
            # A December observation can involve an option exposed to a January
            # dividend. Keep the later event and its announcement date; collection
            # does not assert that the event/amount was known on the observation day.
            yield Request(f"corporate_{kind}", f"/corporate_action/{kind}",
                          {"symbol": symbol.symbol, **window, "end_date": windows["corporate_action_end"]})
    for symbol in sorted(set(rate_symbols)):
        # Keep the maturity-specific rates for later discounting. The report date
        # does not prove the value was already published at that day's 09:30 quote.
        yield Request("interest_rate_eod", "/interest_rate/history/eod", {"symbol": symbol, **window})
    # VIX reflects volatility expectations in S&P 500 option prices. It supplies
    # market-condition context, not a volatility estimate for every individual stock.
    # https://www.cboe.com/tradable-products/vix
    index_start = max(windows["history_start"], cfg.index_history_start)
    if index_start <= end:
        yield Request("index_eod", "/index/history/eod", {"symbol": "VIX", **window, "start_date": index_start})
    index_days = (exchange_calendar().sessions_in_range(max(start, cfg.index_history_start), end).tz_localize(None)
                  if max(start, cfg.index_history_start) <= end else [])
    for day in index_days:
        # Intraday VIX can later align with hourly option observations; its daily
        # report alone should not be assumed known earlier that same morning.
        yield history_request(cfg, "index", "price", "VIX", day)
        yield near_close_request(cfg, "index", "VIX", day)
    if include_stock_lookback:
        # Only normal panel runs request earlier stock snapshots/EOD. These
        # raw pulls have ordinary cache receipts and coverage in the reference
        # ledger; no option sample or calibration is performed in the lookback.
        for date in windows["lookback_dates"]:
            for symbol in symbols:
                for kind in ("quote", "eod"):
                    yield history_request(cfg, "stock", kind, symbol.symbol, pd.Timestamp(date))
                yield near_close_request(cfg, "stock", symbol.symbol, pd.Timestamp(date))


# 8. Choose actual listed contracts to download using the study's sampling grid.
# These functions use working copies; the vendor's raw tables stay unchanged.
def normalize_chain(frame: pd.DataFrame, symbol: str, cfg: CollectorConfig) -> pd.DataFrame:
    # The "chain" is a dated list of observed option contracts. Convert its
    # identities to comparable types and remove duplicate identities for selection.
    # Deduplicating this list means requesting each contract once. It does not
    # deduplicate any of that contract's raw quote, trade, or OI observations.
    if frame.empty:
        return pd.DataFrame(columns=CONTRACT_FIELDS)
    chain = frame.loc[:, CONTRACT_FIELDS].copy()
    chain["expiration"] = pd.to_datetime(chain["expiration"], format="mixed", errors="raise").dt.normalize()
    chain["strike"] = pd.to_numeric(chain["strike"], errors="raise")
    chain["right"] = chain["right"].str.lower().replace({"c": "call", "p": "put"})
    if (chain["symbol"].ne(symbol).any() or chain[["expiration", "strike"]].isna().any().any()
            or not np.isfinite(chain["strike"]).all() or chain["strike"].le(0).any()
            or not chain["right"].isin(["call", "put"]).all()):
        raise ValueError("Invalid identity in dated observed universe")
    for strike in chain["strike"].unique():
        format_strike(strike)
    return (chain.loc[chain["right"].isin(cfg.option_rights)].drop_duplicates()
            .sort_values(["expiration", "strike", "right"]).reset_index(drop=True))


def observed_contract_universe(frames: dict[str, pd.DataFrame], symbol: str,
                               cfg: CollectorConfig) -> tuple[pd.DataFrame, dict]:
    # OI can reveal a contract with no quote or trade that day. Combining all
    # three dated sources avoids requiring current-day activity for discovery.
    # This is still an observed universe, not a complete historical listing file.
    # Example: the same call appears in the quote and OI tables. It becomes one
    # candidate with discovery_sources="open_interest|quote". An OI-only put can
    # also remain a candidate even if its later quote requests return no data.
    tables = {name: normalize_chain(frame, symbol, cfg) for name, frame in frames.items()}
    counts = {name: len(table) for name, table in tables.items()}
    parts = [table.assign(discovery_sources=name) for name, table in tables.items() if not table.empty]
    if not parts:
        return pd.DataFrame(columns=DISCOVERY_COLUMNS), counts
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(list(CONTRACT_FIELDS), as_index=False, sort=True)["discovery_sources"].agg(
        lambda sources: "|".join(sorted(set(sources))))
    return combined.loc[:, DISCOVERY_COLUMNS], counts


def stock_selection_references(frames: pd.DataFrame | Iterable[pd.DataFrame], day: pd.Timestamp,
                               cfg: CollectorConfig) -> list[dict]:
    # Produce up to three stock midpoints, each with its source quote and age.
    # The midpoint is (bid + ask) / 2 and is used only to target option strikes.
    # "Reference" here means a stock price used by the selector; it is different
    # from the dividends/rates/VIX reference bundle collected for later research.
    opened, closed = session_bounds(day, cfg)
    times = {at: pd.Timestamp(f"{day.date()} {at}", tz=cfg.exchange_tz) for at in cfg.selection_times}
    times = {at: clock for at, clock in times.items() if opened <= clock < closed}  # No 13:30 selection on a 13:00 close.
    latest = {}
    for frame in [frames] if isinstance(frames, pd.DataFrame) else frames:
        if frame.empty:
            continue
        quotes = frame.copy()
        quotes["_clock"] = parse_vendor_clock(quotes["timestamp"], cfg.exchange_tz)
        quotes = quotes.loc[quotes["_clock"].notna()].sort_values("_clock", kind="stable")
        for at, clock in times.items():
            # <= prevents a later quote from supplying an earlier stock reference.
            # A stable sort keeps original row order when several clocks tie; the
            # last such row wins, including across successive reading batches.
            prior = quotes.loc[quotes["_clock"].between(opened, clock)]
            if not prior.empty:
                row = prior.iloc[-1]  # Latest sample at/before the reference; never borrow a future stock price.
                if at not in latest or row["_clock"] >= latest[at]["_clock"]:
                    latest[at] = row
    # First find the actual last observation; only then assess its usability.
    # Filtering first could hide a halt/non-firm quote behind an older good quote.
    references = []
    for selection_time, at in times.items():
        if selection_time not in latest:
            continue
        row = latest[selection_time]
        conditions = [str(row[column]).strip() for column in ("bid_condition", "ask_condition")]
        # Theta condition 0 = regular, 1 = bid/ask auto-executable, and 50 =
        # national BBO. Blank codes are accepted as unspecified by this selection
        # policy. Halted or non-firm observations cannot supply the reference.
        # Their raw rows are still saved; this check only controls strike selection.
        # The three accepted codes are our policy, not Theta's entire list of
        # firm quote conditions. A blank is allowed as an unspecified status.
        # https://docs.thetadata.us/Articles/Errors-Exchanges-Conditions/Quote-Conditions.html
        if any(text and pd.to_numeric(text, errors="coerce") not in {0, 1, 50} for text in conditions):
            continue
        bid, ask = (pd.to_numeric(row[side], errors="coerce") for side in ("bid", "ask"))
        age = (at - row["_clock"]).total_seconds()  # Time since the returned sample, not since its quote event.
        # A missing/nonpositive/crossed stock quote cannot give a sensible S/K.
        # Reject it for selection rather than calculate target strikes from it.
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask
                and age <= cfg.max_stock_quote_age_seconds):
            continue
        # The stock midpoint supplies S only for choosing strikes. Saving it does
        # not assert a trade was possible at that price or set the option's value.
        references.append({"selection_time": selection_time, "stock_mid": float((bid + ask) / 2),
                           "observation_timestamp": str(row["timestamp"]), "observation_timestamp_utc": row["_clock"].isoformat(),
                           "observation_age_seconds": age,
                           "timestamp_role": "sample_boundary", "quote_event_age_seconds": None})
    return references


def eligible_expirations(day: pd.Timestamp, expirations, cfg: CollectorConfig) -> list[pd.Timestamp]:
    # Restrict to the DTE window, then choose the nearest unused expiration for
    # each target in order. Ties prefer shorter DTE. Removing each choice prevents
    # two targets from selecting the same expiration twice.
    # These are calendar-day distances, not trading-day counts or a maturity
    # year fraction. A future pricing module must choose its own time convention.
    remaining = sorted(exp for exp in expirations if cfg.min_dte <= (exp - day).days <= cfg.max_dte)
    selected = []
    for target in cfg.target_dtes:
        if not remaining or len(selected) >= cfg.max_expirations_per_day:
            break
        # Example: for a 30-day target, 29 and 31 are equally near; choose 29.
        # Shorter-on-ties is a deterministic sampling rule, not a pricing result.
        best = min(remaining, key=lambda exp: (abs((exp - day).days - target), (exp - day).days))
        selected.append(best)
        remaining.remove(best)  # Nearby targets must not repeatedly consume the same listed expiry.
    remaining.sort(key=lambda exp: (min(abs((exp - day).days - target) for target in cfg.target_dtes), exp))
    # If the cap leaves room, add other expirations nearest any target.
    return sorted(selected + remaining[:max(cfg.max_expirations_per_day - len(selected), 0)])


def select_contracts(chain: pd.DataFrame, day: pd.Timestamp, references: list[dict],
                     cfg: CollectorConfig) -> pd.DataFrame:
    # Select on observed stock price, strike and maturity. Requiring high option
    # volume or narrow spreads here would remove the quiet/wide-spread contracts
    # needed to study how the models behave across different liquidity conditions.
    rows = []
    for expiration in eligible_expirations(day, chain["expiration"].unique(), cfg):
        family = chain.loc[chain["expiration"].eq(expiration)]
        strikes = sorted(family["strike"].unique())
        chosen_by_time = {}
        for reference in references:
            chosen = set()
            for target in cfg.moneyness_targets:
                target_strike = reference["stock_mid"] / target  # K = S/(S/K): S=$100 and target=0.80 gives K=$125.
                # Choose a listed strike by dollar distance. If $124/$126 tie,
                # choose $124; actual moneyness will differ slightly from the target.
                ranked = sorted(strikes, key=lambda strike: (abs(strike - target_strike), strike))
                chosen.update(ranked[:cfg.strikes_per_moneyness_target])
            chosen_by_time[reference["selection_time"]] = chosen
        # If S moves enough to select $100 in the morning and $105 later, keep
        # both contracts' full requested days. This is a retrospective sample;
        # its final membership was not necessarily knowable at the morning quote.
        selected = set().union(*chosen_by_time.values()) if chosen_by_time else set()
        # Download the union across selection times once. Keep only call/put
        # identities actually present in the dated chain; do not invent pairs.
        for contract in family.loc[family["strike"].isin(selected)].itertuples(index=False):
            # The key combines the four identity fields into a readable label.
            # selection_times explains why we chose it; discovery_sources explains
            # which vendor tables established it as an observed candidate.
            expiry, strike = expiration.strftime("%Y-%m-%d"), format_strike(contract.strike)
            rows.append({"symbol": contract.symbol, "expiration": expiry, "strike": strike,
                         "right": contract.right, "contract_key": f"{contract.symbol}|{expiry}|{strike}|{contract.right}",
                         "dte_days": (expiration - day).days,  # Calendar-day count; conversion to model time comes later.
                         "selection_times": "|".join(at for at, values in chosen_by_time.items() if contract.strike in values),
                         "discovery_sources": getattr(contract, "discovery_sources", "")})
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


# 9. Coordinate symbol-days, shared references, and coverage reports.
# A session manifest is the small record connecting selected contracts to their
# request receipts. It can be read without opening the much larger market tables.
class Collector:
    # The coordinator connects the pieces above: describe requests, save raw
    # results, choose contracts, and publish a small record of the completed work.
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

    def collect_batch(self, requests_to_make: list[Request]) -> Iterable[tuple[Request, dict]]:
        # Small batches overlap network waits and parsing. The shared client
        # still allows at most four active HTTP requests across every worker.
        # Yield receipts as they finish, so the caller retains earlier results
        # even if a later future raises or the operator interrupts the batch.
        with self.workers(self.cfg.max_batch_workers) as pool:
            futures = {pool.submit(self.store.collect, request): request for request in requests_to_make}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    record = future.result()
                except CollectionStopped:
                    continue
                except Exception as exc:
                    self.store.client.stop(f"Unable to save {request.dataset}: {exc}")
                    record = {"request_id": request.request_id, "dataset": request.dataset,
                              "params": request.params, "status": "request_error", "error": repr(exc)}
                yield request, record

    def manifest_valid(self, manifest: dict) -> bool:
        # A saved "complete" label alone is not enough to skip work. Check that
        # the policy matches and every expected request/artifact is accounted for.
        try:
            if (manifest["output_schema_version"] != OUTPUT_SCHEMA_VERSION
                    or manifest["policy_id"] != self.cfg.policy_id
                    or manifest["status"] not in {"complete", "unavailable"}
                    or manifest["coverage"]["status"] not in {"observations_present", "gaps_observed"}):
                return False
            records = manifest["requests"]
            selected_count = manifest["selected_contract_count"]
            if (manifest["expected_request_count"] != len(records)
                    or manifest["contracts"]["rows"] != selected_count
                    or manifest["universe"]["rows"] != manifest["universe_contract_count"]):
                return False
            if not all(artifact_valid(manifest[name], self.cfg.output_dir) for name in ("contracts", "universe")):
                return False
            selected = pd.read_parquet(self.cfg.output_dir / manifest["contracts"]["path"])
            day = pd.Timestamp(manifest["trade_day"])
            expected = (shared_day_requests(self.cfg, manifest["symbol"], day)
                        + option_quote_requests(self.cfg, manifest["symbol"], day, selected))
            # Check the actual batch identities, not a per-contract request-count
            # formula that stopped being valid when transport became bulk.
            if (len(records) != len(expected)
                    or {r["request_id"] for r in records} != {r.request_id for r in expected}):
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

    def request_coverage(self, request: Request, record: dict | None) -> dict:
        """Describe observed presence separately from successful HTTP/file handling."""
        if record is None or record["status"] not in GOOD_REQUEST_STATUSES:
            return {"status": "unknown", "reason": "request_not_completed_successfully"}
        # OI and corporate actions are event/report driven. A successful
        # empty response does not by itself establish a missing required price.
        # In particular, do not label quiet contracts as failed collections.
        if request.dataset.startswith("corporate_") or request.endpoint.endswith("/open_interest"):
            return {"status": "not_assessed", "reason": "empty_event_reports_can_be_valid"}
        if request.endpoint.endswith("/quote") and request.endpoint.startswith("/stock/"):
            return self.quote_or_contract_report_coverage(request, record)
        if request.endpoint.endswith("/eod"):
            try:
                frame = self.store.read(record)
                column = request.report_date_column or "created"
                dates = (pd.to_datetime(frame[column], format="mixed", errors="coerce")
                         if request.report_date_column else
                         parse_vendor_clock(frame[column], self.cfg.exchange_tz).dt.tz_convert(self.cfg.exchange_tz))
                observed = set(dates.dropna().dt.strftime("%Y-%m-%d"))
                sessions = exchange_calendar().sessions_in_range(request.params["start_date"], request.params["end_date"])
                expected = set(sessions.strftime("%Y-%m-%d"))
                missing = sorted(expected - observed)
                return {"status": "gaps_observed" if missing else "observations_present",
                        "requested_session_count": len(expected), "observed_session_count": len(expected & observed),
                        "missing_requested_session_dates": missing, "date_grid": "XNYS_sessions",
                        # A bank/Fed holiday can lack a report while NYSE is open.
                        # This is a presence comparison, not a publisher-calendar
                        # assertion or permission to fill a missing rate with zero.
                        "publisher_schedule_verified": False,
                        "missing_dates_prove_vendor_error": False}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"status": "unknown", "reason": "coverage_check_failed", "error": repr(exc)}
        return {"status": "gaps_observed" if record["status"] == "no_data" else "observations_present",
                "complete_intraday_history_verified": False}

    def quote_or_contract_report_coverage(self, request: Request, record: dict | None,
                                          selected: pd.DataFrame | None = None) -> dict:
        # A bulk response can contain thousands of rows while omitting one of
        # our selected options. Scan it once, checking each selected identity.
        # For sampled quotes also compare the actual clock grid, including its
        # first and last points. A repeated sampled price still counts as a row.
        if record is None or record["status"] not in GOOD_REQUEST_STATUSES:
            return {"status": "unknown", "reason": "request_not_completed_successfully"}
        option = request.endpoint.startswith("/option/")
        if option:
            chosen = selected
            if request.params.get("expiration") != "*":
                chosen = chosen.loc[chosen["expiration"].eq(request.params["expiration"])]
            wanted = set(chosen["contract_key"])
        else:
            wanted = {request.params["symbol"]}
        if not wanted:
            return {"status": "not_assessed", "reason": "no_selected_contracts"}
        sampled = request.endpoint.endswith("/history/quote")
        expected_times = set()
        if sampled:
            day = pd.Timestamp(request.params["date"])
            opened, closed = session_bounds(day, self.cfg)
            expected_times = set(pd.date_range(opened, closed,
                freq=pd.Timedelta(seconds=interval_seconds(request.params["interval"]))).tz_convert("UTC"))
            # On a normal 1h session this is 09:30, 10:30, ..., 15:30. We compare
            # actual timestamps, because seven rows can still repeat one time and
            # omit another. The separate near-close request is checked on its own.
        clocks = {key: set() for key in wanted}
        column = "created" if request.endpoint.endswith("/eod") else "timestamp"
        columns = [*CONTRACT_FIELDS, column] if option else [column]
        try:
            for frame in self.store.iter_frames(record, columns=columns):
                if option:
                    keys = option_contract_keys(frame)
                else:
                    keys = [request.params["symbol"]] * len(frame)
                for key, clock in zip(keys, parse_vendor_clock(frame[column], self.cfg.exchange_tz)):
                    if key in wanted and pd.notna(clock):
                        clocks[key].add(clock)  # Deduplicate clocks for coverage only; stored quote rows are intact.
            missing = []
            for key in sorted(wanted):
                absent = expected_times - clocks[key]  # Missing samples are reported, never filled with invented quotes.
                if (sampled and absent) or not clocks[key]:
                    missing.append({"contract_key" if option else "symbol": key,
                                    "missing_sample_times": [t.tz_convert(self.cfg.exchange_tz).strftime("%H:%M:%S")
                                                             for t in sorted(absent)] if sampled else [],
                                    "reason": "missing_sample_rows" if sampled else "no_dated_observation"})
            return {"status": "gaps_observed" if missing else "observations_present",
                    "checked_contract_count": len(wanted) if option else 0,
                    "expected_samples_per_series": len(expected_times) if sampled else 1,
                    "missing_observations": missing, "quote_event_age_verified": False,
                    "research_sample_usability_verified": False}  # Presence does not make a crossed/stale quote fit for pricing.
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"status": "unknown", "reason": "coverage_check_failed", "error": repr(exc)}

    def session_coverage(self, symbol: str, day: pd.Timestamp, selected: pd.DataFrame,
                         records: list[dict], missing_times: list[str], failed: bool) -> dict:
        # Required hourly/near-close prices and daily reports are assessed on
        # the selected sample. Extra downloaded strikes do not fill its gaps.
        required = [request for request in shared_day_requests(self.cfg, symbol, day)
                    if request.endpoint.endswith(("/quote", "/eod")) and "/list/" not in request.endpoint]
        required += option_quote_requests(self.cfg, symbol, day, selected)
        by_id = {record["request_id"]: record for record in records}
        checks = []
        for request in required:
            record = by_id.get(request.request_id)
            coverage = (self.quote_or_contract_report_coverage(request, record, selected)
                        if request.endpoint.startswith("/option/") else self.request_coverage(request, record))
            checks.append({"request_id": request.request_id, "dataset": request.dataset, "params": request.params, **coverage})
        missing = [check for check in checks if check["status"] == "gaps_observed"]
        unknown = sum(check["status"] == "unknown" for check in checks)
        status = ("unknown" if failed or unknown else "gaps_observed"
                  if missing or missing_times or selected.empty else "observations_present")
        return {"status": status, "required_price_requests": checks,
                "missing_option_quote_count": len({row["contract_key"] for c in missing
                    if c["dataset"].startswith("option_quotes_") for row in c["missing_observations"]}),
                "missing_option_eod_count": len({row["contract_key"] for c in missing
                    if c["dataset"] == "option_eod" for row in c["missing_observations"]}),
                "missing_stock_dataset_count": sum(c["dataset"].startswith("stock_") for c in missing),
                "unknown_required_request_count": unknown,
                "no_selected_contracts": selected.empty,
                "complete_intraday_history_verified": False,
                "research_sample_usability_verified": False}

    def collect_day(self, symbol_cfg: SymbolConfig, day: pd.Timestamp) -> dict:
        # This is the main unit of collection: one underlying on one trading day.
        # Its result is a manifest, while the large raw tables are saved separately.
        # Read the five steps below as one story. Once a raw request is saved,
        # a later problem in this day does not erase it or require downloading
        # it again on the next attempt.
        symbol = symbol_cfg.symbol
        records, references = [], []
        selected = pd.DataFrame(columns=SELECTION_COLUMNS)
        universe = pd.DataFrame(columns=DISCOVERY_COLUMNS)
        source_counts = dict.fromkeys(("quote", "trade", "open_interest"), 0)
        reason, error = "", ""
        try:
            # Step 1: overlap seven shared pulls: hourly/near-close stock quotes,
            # stock EOD, two dated contract lists, bulk OI, and bulk option EOD.
            # Daily EOD reports supply volume and trade counts without tick trades.
            completed = self.collect_batch(shared_day_requests(self.cfg, symbol, day))
            records.extend(record for _, record in completed)
            by_dataset = {record["dataset"]: record for record in records}
            self.store.client.check_running()
            stock = by_dataset[f"stock_quotes_{self.cfg.quote_interval}"]
            # Step 2: discovery remains independent of stock availability. OI-only
            # contracts remain eligible; their absent quotes are recorded later.
            discovery = {kind: self.store.read(by_dataset[dataset]) for kind, dataset in
                         (("quote", "quoted_contracts"), ("trade", "traded_contracts"),
                          ("open_interest", "option_open_interest"))}
            universe, source_counts = observed_contract_universe(discovery, symbol, self.cfg)
            # Step 3: obtain the stock references, then choose the option contracts.
            # A chosen contract gets its requested regular-session history, even if
            # a later selection time caused us to choose it.
            # That is a retrospective research download design. The resulting set
            # is not a trading universe known at the beginning of that session.
            references = stock_selection_references(self.store.iter_frames(
                stock, columns=["timestamp", "bid", "ask", "bid_condition", "ask_condition"]), day, self.cfg)
            selected = select_contracts(universe, day, references, self.cfg)
            if universe.empty:
                reason = "no_observed_contracts"
            elif not references:
                reason = "stock_selection_reference_unavailable"
            elif selected.empty:
                reason = "no_contracts_in_sampling_window"
            # Step 4: two quote requests per selected expiration, covering all
            # strikes/rights: hourly history and a separate near-close snapshot.
            # No trade-count, spread, or OI threshold removes a chosen contract.
            # The writer keeps only this day's selected contracts. All their
            # observations and vendor fields survive; excluded rows are counted.
            completed = self.collect_batch(option_quote_requests(self.cfg, symbol, day, selected))
            records.extend(record for _, record in completed)
            self.store.client.check_running()
        except Exception as exc:
            # Completed raw pulls survive; a failed day never satisfies resume.
            error = repr(exc)
        failures = sum(record["status"] not in GOOD_REQUEST_STATUSES for record in records)
        # "unavailable" means selection yielded no contracts without a request
        # error; reason explains why. "complete" means the requests finished,
        # including any valid empty responses, not full observed market coverage.
        status = "request_error" if error or failures else ("unavailable" if selected.empty else "complete")
        contract_path = self.directory / "contracts" / f"symbol={symbol}__date={day.date()}.parquet"
        universe_path = self.directory / "universes" / f"symbol={symbol}__date={day.date()}.parquet"
        write_parquet(contract_path, selected)
        write_parquet(universe_path, universe)
        # Keeping both lists lets a reviewer distinguish a contract that was
        # observed but not selected from one absent from all discovery responses.
        # Step 5: record selection coverage and receipts. Times outside this
        # day's session are not counted as missing selection references.
        opened, closed = session_bounds(day, self.cfg)
        scheduled = [at for at in self.cfg.selection_times
                     if opened <= pd.Timestamp(f"{day.date()} {at}", tz=self.cfg.exchange_tz) < closed]
        missing_times = [at for at in scheduled if at not in {r["selection_time"] for r in references}]
        coverage = self.session_coverage(symbol, day, selected, records, missing_times, status == "request_error")
        manifest = {**asdict(symbol_cfg), "trade_day": str(day.date()), "status": status,
                "reason": reason, "error": error, "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "policy_id": self.cfg.policy_id, "updated_at_utc": utc_now(),
                    "session_open": opened.isoformat(), "session_close": closed.isoformat(),
                    "intraday_window": "underlying_regular_trading_session",
                    "quote_interval": self.cfg.quote_interval,
                    "near_close_time": (closed - pd.Timedelta(minutes=self.cfg.near_close_minutes)).strftime("%H:%M:%S"),
                    "option_quote_batching": "all_strikes_and_rights_per_selected_expiration",
                    "option_quote_retention": "selected_contracts",
                    "daily_activity_source": "Theta stock/option EOD volume and count; no individual trades",
                    "discovery_scope": "union of dated quote/trade lists and prior-session OI reports within max_dte; complete listing coverage unverified",
                    "discovery_max_dte": self.cfg.max_dte,
                    "quoted_contract_count": source_counts["quote"], "traded_contract_count": source_counts["trade"],
                    "oi_reported_contract_count": source_counts["open_interest"],
                    "universe_contract_count": len(universe), "selected_contract_count": len(selected),
                    "stock_selection_references": references,
                    "missing_selection_times": missing_times, "coverage": coverage,
                    "requests": sorted(records, key=lambda r: (r["dataset"], r["request_id"])),
                    "contracts": file_receipt(contract_path, self.cfg.output_dir, selected),
                    "universe": file_receipt(universe_path, self.cfg.output_dir, universe),
                    "request_error_count": failures,
                    "expected_request_count": 7 + 2 * selected["expiration"].nunique()}
        # Publish last. The manifest references exact artifacts, not a filename glob.
        write_json(self.session_path(symbol, day), manifest)
        self.store.client.check_running()
        return manifest

    def collect_references(self, symbols: list[SymbolConfig], start: str, end: str,
                           rate_symbols: list[str], run_id: str, *, include_stock_lookback: bool = False) -> dict:
        # Reference data has its own ledger because it serves many symbol-days.
        # Keeping it separate avoids copying rates/VIX into each option table.
        records = []
        # These notes travel with the data so later research can interpret units,
        # missing values, and unverified coverage without relying on this script.
        lookback_datasets = ([f"stock_quotes_{self.cfg.quote_interval}", "stock_quotes_near_close", "stock_eod"]
                             if include_stock_lookback and self.cfg.lookback_sessions else [])
        windows = collection_windows(self.cfg, start, end)
        index_excluded = [date for date in exchange_calendar().sessions_in_range(windows["history_start"], end).strftime("%Y-%m-%d")
                          if date < self.cfg.index_history_start]
        access_gaps = ([{"symbol": "VIX", "reason": "before_standard_index_history_start",
                        "access_start": self.cfg.index_history_start,
                        "unrequested_eod_session_dates": index_excluded,
                        "unrequested_intraday_session_dates": [date for date in index_excluded if date >= start]}]
                       if index_excluded else [])
        ledger = {"vendor": "ThetaData", "start": start, "end": end,
                  "collection_windows": windows, "subscription_coverage_gaps": access_gaps,
                  "stock_lookback_requested": bool(lookback_datasets),
                  "option_contract_continuity_guaranteed": False,
                  "requested_rates": sorted(set(rate_symbols)),
                  "rate_units": "percent",  # 4.25 stays 4.25; converting to a pricing rate/day-count convention comes later.
                  "required_datasets": ["corporate_dividend", "corporate_split", "interest_rate_eod",
                                        "index_eod", f"index_prices_{self.cfg.quote_interval}", "index_prices_near_close", *lookback_datasets],
                  "rate_publication_timestamps_verified": False,
                  "corporate_action_range_filters": {"dividend": "ex_dividend_date", "split": "effective_date"},
                  "missing_dividend_amounts": "unknown; not zero", "split_ratio": "before_shares / after_shares",
                  "empty_actions_prove_complete_event_coverage": False,
                  "later_actions_were_known_at_study_time": "not_assumed; retain announcement dates and unknown values",
                  "index_unchanged_updates_may_be_omitted": True,
                  # Corporate actions can change what an option delivers. A ticker
                  # and strike alone do not prove contracts remain comparable across a split.
                  "adjusted_contract_deliverables": "not_documented_by_Theta; not_inferred",
                  "historical_symbol_mappings": "not_verified"}
        path = self.cfg.output_dir / "references" / f"{run_id}.json"
        def publish():
            ledger.update(updated_at_utc=utc_now(), requests=records,
                          requests_with_observed_gaps=sum(r["coverage"]["status"] == "gaps_observed" for r in records),
                          requests_with_unknown_coverage=sum(r["coverage"]["status"] == "unknown" for r in records))
            write_json(path, ledger)
        try:
            pending = iter(reference_requests(self.cfg, symbols, start, end, rate_symbols,
                                              include_stock_lookback=include_stock_lookback))
            while not self.store.client.stop_event.is_set():
                # A bounded queue keeps eight years of reference work from being
                # scheduled at once. Finished response files are reusable even if
                # a later batch fails or the operator stops the run.
                batch = list(islice(pending, 32))
                if not batch:
                    break
                for request, record in self.collect_batch(batch):
                    records.append({**record, "params": request.params,
                                    "coverage": self.request_coverage(request, record)})
                publish()
                print(f"References: {len(records)} requests recorded")
        finally:
            publish()
        return ledger

    def collect_coverage(self, symbols: list[SymbolConfig], anchors: pd.DatetimeIndex, run_id: str) -> dict:
        # Ask "which dates does Theta list?" before requesting detailed history.
        # anchors are the exchange sessions within the user's requested window.
        # A listed date means the vendor advertises some data for that series.
        # It cannot establish that every event/interval or option is present, or
        # that this account can download the requested historical detail.
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
        # For example, a standalone quote and a quote matched to a trade can
        # describe overlapping market information. Adding their row counts is a
        # storage total, not a count of distinct quote updates or executed trades.
        path = self.directory / "availability.csv"
        columns = ("symbol", "trade_day", "status", "coverage_status", "reason", "quoted_contract_count", "traded_contract_count",
                   "oi_reported_contract_count", "universe_contract_count", "selected_contract_count",
                   "selection_reference_count", "missing_selection_times", "request_count", "request_error_count",
                   "no_data_request_count", "missing_option_quote_count", "missing_option_eod_count", "missing_stock_dataset_count",
                   "unknown_required_request_count", "stored_rows", "excluded_quote_rows", "stored_parquet_bytes", "error")
        counts = dict.fromkeys(("complete", "unavailable", "request_error", "not_attempted"), 0)
        counts.update(days_with_observed_gaps=0, days_with_unknown_coverage=0)
        with atomic_output(path) as temp:
            with temp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for day in anchors:
                    for symbol in symbols:
                        row = {"symbol": symbol.symbol, "trade_day": str(day.date()), "status": "not_attempted",
                               "coverage_status": "not_checked"}
                        try:
                            manifest = read_json(self.session_path(symbol.symbol, day))
                            if (manifest["symbol"] != symbol.symbol or manifest["trade_day"] != str(day.date())
                                    or manifest["status"] not in {"complete", "unavailable", "request_error"}):
                                raise ValueError("Session identity or status does not match the requested day")
                            row.update({key: manifest.get(key, "") for key in columns if key in manifest})
                            records = manifest["requests"]
                            coverage = manifest["coverage"]
                            if coverage["status"] not in {"observations_present", "gaps_observed", "unknown"}:
                                raise ValueError("Unknown session coverage status")
                            row.update(coverage_status=coverage["status"],
                                       **{name: coverage[name] for name in ("missing_option_quote_count", "missing_option_eod_count",
                                          "missing_stock_dataset_count", "unknown_required_request_count")})
                            row.update(selection_reference_count=len(manifest["stock_selection_references"]),
                                       missing_selection_times="|".join(manifest["missing_selection_times"]),
                                       request_count=len(records),
                                       no_data_request_count=sum(r["status"] == "no_data" for r in records),
                                       stored_rows=sum(r.get("row_count", 0) for r in records),
                                       excluded_quote_rows=sum(r.get("retention", {}).get("excluded_rows", 0) for r in records),
                                       stored_parquet_bytes=sum((r.get("data") or {}).get("size", 0) for r in records))
                        except FileNotFoundError:
                            pass
                        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                            # One damaged manifest must not prevent a report for
                            # all the other days. Its row explicitly records failure.
                            row.update(status="request_error", coverage_status="unknown", error=repr(exc))
                        counts[row["status"]] += 1
                        counts["days_with_observed_gaps"] += row["coverage_status"] == "gaps_observed"
                        counts["days_with_unknown_coverage"] += row["coverage_status"] == "unknown"
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
    # "argv" is the list of arguments after the script name. If omitted, argparse
    # reads the actual command line. Supplying a list also permits a small local
    # check to exercise this entry point without launching a separate process.
    defaults = CollectorConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", type=str.upper, choices=[cfg.symbol for cfg in UNIVERSE])
    parser.add_argument("--start", default=defaults.start_date, help="Inclusive first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=defaults.end_date, help="Inclusive last date (YYYY-MM-DD)")
    parser.add_argument("--lookback-sessions", type=int, default=defaults.lookback_sessions,
                        help="Prior stock/rate trading sessions to collect (default: 60; 0 disables the buffer)")
    parser.add_argument("--quote-interval", default=defaults.quote_interval, choices=QUOTE_INTERVALS)
    parser.add_argument("--raw-chunk-rows", type=int, default=defaults.raw_chunk_rows,
                        help="Rows per parsing/storage batch; affects memory use, not which rows are saved")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--store-raw-payloads", action="store_true", help="Also preserve exact Theta response bytes")
    parser.add_argument("--refresh-no-data", action="store_true", help="Retry previously empty requests")
    parser.add_argument("--max-inflight-requests", type=int, default=defaults.max_inflight_requests,
                        help="1 through 4 for Standard; lower this if another client shares the account")
    parser.add_argument("--max-requests-per-second", type=float, default=defaults.max_requests_per_second,
                        help="Optional request-start pacing; default 0 means only the concurrency cap applies")
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
                      raw_chunk_rows=args.raw_chunk_rows, lookback_sessions=args.lookback_sessions,
                      store_raw_payloads=args.store_raw_payloads, refresh_no_data=args.refresh_no_data,
                      max_inflight_requests=args.max_inflight_requests,
                      max_requests_per_second=args.max_requests_per_second)
        collection_windows(cfg, args.start, args.end)
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
    # Execution is deliberately visible here: decide scope -> preview or connect
    # -> check dates -> collect references -> collect symbol-days -> write reports.
    args, cfg, symbols, anchors = parse_run_scope(argv)
    windows = collection_windows(cfg, args.start, args.end)
    panels = not (args.references_only or args.coverage_only)
    total = len(symbols) * len(anchors) if panels else 0
    print(f"Scope: {', '.join(s.symbol for s in symbols)}; {args.start} to {args.end}; {total} symbol-days")
    print(f"Vendor: ThetaData; quotes: {cfg.quote_interval} plus near-close; daily volume/count from EOD; stock venue: {cfg.stock_venue}")
    if panels:
        print(f"Bulk collection: 7 shared requests + 2 per selected expiration/day (at most {7 + 2 * cfg.max_expirations_per_day})")
        print(f"Storage: selected option contracts only; broad option reports/lists capped at {cfg.max_dte} days to expiration")
        print(f"Near-close snapshot: {cfg.near_close_minutes} minutes before the actual close; quote sample age is not event age")
        print(f"Standard: {cfg.max_inflight_requests} simultaneous requests; "
              f"request-start cap: {str(cfg.max_requests_per_second) + '/s' if cfg.max_requests_per_second else 'none'}")
    print(f"Output: {cfg.output_dir}")
    if args.coverage_only:
        print("Coverage mode: available dates for stock quotes/trades and VIX; no history downloads")
    else:
        print(f"Required references: dividends/splits, {len(set(args.rate_symbols))} rate series, VIX EOD and {cfg.quote_interval} prices")
        print(f"Reference history starts {windows['history_start']}; corporate actions through {windows['corporate_action_end']}")
        print(f"Standard VIX history starts {cfg.index_history_start}; earlier requested sessions are reported as access gaps")
        if panels:
            print(f"Stock lookback: {cfg.lookback_sessions} prior sessions; "
                  f"{3 * len(symbols) * cfg.lookback_sessions} additional hourly/near-close/EOD requests")
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
                     "collection_windows": windows,
                     "references_only": args.references_only, "coverage_only": args.coverage_only,
                     "rate_symbols": args.rate_symbols},
           "code_sha256": collector.store.code_sha256, "python": sys.version, "platform": platform.platform(),
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
                    catalogue_dates = exchange_calendar().sessions_in_range(windows["history_start"], args.end).tz_localize(None)
                    catalogue = collector.collect_coverage(symbols, catalogue_dates, run_id)
                    catalogue_errors, catalogue_gaps = catalogue["request_errors"], catalogue["series_with_gaps"]
                    collector.store.client.check_running()
                if not args.coverage_only:
                    # Fetch the common reference bundle before symbol-day work.
                    # Successful reference pulls also reuse the ordinary raw cache.
                    run["reference_ledger"] = f"references/{run_id}.json"
                    references = collector.collect_references(symbols, args.start, args.end, args.rate_symbols, run_id,
                                                              include_stock_lookback=panels)
                    reference_failures = sum(r["status"] not in GOOD_REQUEST_STATUSES
                                             or r["coverage"]["status"] == "unknown" for r in references["requests"])
                    # A dividend/split endpoint can correctly return no events.
                    # Missing price/rate history is a coverage gap, not a zero.
                    reference_gaps = references["requests_with_observed_gaps"] + len(references["subscription_coverage_gaps"])
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
                                          f"coverage: {manifest['coverage']['status']}; "
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
                    # Exit codes summarize collection, not research success:
                    # 1 = something failed; 2 = completed with observed gaps;
                    # 0 = requested work completed without those reported problems.
                    if (exit_code or failed or reference_failures or catalogue_errors
                            or coverage.get("request_error", 0) or coverage.get("not_attempted", 0)
                            or coverage.get("days_with_unknown_coverage", 0)):
                        exit_code = 1
                    elif (reference_gaps or catalogue_gaps or coverage.get("unavailable", 0)
                          or coverage.get("days_with_observed_gaps", 0)):
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
