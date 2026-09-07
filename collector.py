"""Historical stock/option collection, kept in one file for review.

Python 3.11+; dependencies: exchange-calendars, numpy, pandas>=2, pyarrow, requests.
Yahoo is optional: install yfinance only when using --yahoo-actions.

Examples (Theta Terminal v3 must be running for collection):
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --plan
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-02 --quote-interval tick
  python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --references-only --rate-symbols SOFR

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
  raw_cache/<dataset>/.../request=<id>/{data.parquet,meta.json}
  collection/<policy-id>/sessions/<symbol-day>.json
  collection/<policy-id>/contracts/<symbol-day>.parquet
  collection/<policy-id>/availability.csv
  collection/<policy-id>/runs/<run-id>.json
  references/<run-id>.json (only when reference collection is requested)
Successful raw requests are reused across policy changes. A day is resumable
only after its selected-contract file and every referenced request are verified.
"complete" means the requests finished, not that the vendor supplied full coverage.
"no_data" is retained separately from request errors. --refresh-no-data retries it.

Official API specification: https://docs.thetadata.us/openapiv3.yaml
Relevant pages under https://docs.thetadata.us/operations/:
  {stock,option}_history_{quote,trade_quote}.html
  option_list_contracts.html; option_history_open_interest.html
  stock_history_eod.html; interest_rate_history_eod.html

Limits: the quote universe is a date-wide observation, not an intraday listing
snapshot. 1s quotes are samples; use --quote-interval tick for quote events when
your subscription supports them. Requests cover the exchange's regular session.
OI is requested on the report date and describes the previous session's close.
Theta stock EOD is generated around 17:15 ET, not a 16:00 quote. Optional rate
series retain vendor percentage units and report dates; no curve is inferred.
Yahoo actions are an optional, unverified reference snapshot. Authoritative
historical dividends and adjusted-contract deliverables remain unresolved.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4

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


# Configuration and the existing research universe.
OUTPUT_SCHEMA_VERSION = "2026-09-07-raw-collection-v1"
RAW_SCHEMA_VERSION = 1
# Keep the original default root so existing raw caches can be reused.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "multi_year_bsm_backtest_output"
QUOTE_INTERVALS = ("tick", "10ms", "100ms", "500ms", "1s", "5s", "10s", "15s",
                   "30s", "1m", "5m", "10m", "15m", "30m", "1h")
GOOD_REQUEST_STATUSES = {"available", "no_data"}
QUOTE_FIELDS = ("bid_size", "bid_exchange", "bid", "bid_condition",
                "ask_size", "ask_exchange", "ask", "ask_condition")
TRADE_FIELDS = ("trade_timestamp", "quote_timestamp", "sequence", "condition", "size",
                "exchange", "price", *QUOTE_FIELDS)
CONTRACT_FIELDS = ("symbol", "expiration", "strike", "right")
SELECTION_COLUMNS = (*CONTRACT_FIELDS, "contract_key", "dte_days", "selection_times")


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str


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
    base_url: str = "http://127.0.0.1:25503/v3"
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    option_rights: tuple[str, ...] = ("call", "put")
    target_dtes: tuple[int, ...] = (7, 14, 30, 60, 120)
    max_expirations_per_day: int = 5
    moneyness_targets: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
    strikes_per_moneyness_target: int = 1
    min_dte: int = 7
    max_dte: int = 180
    exchange_tz: str = "America/New_York"
    quote_interval: str = "1s"
    stock_venue: str = "utp_cta"
    selection_times: tuple[str, ...] = ("10:30:00", "13:00:00", "15:00:00")
    max_stock_quote_age_seconds: int = 70
    trade_quote_exclusive: bool = True
    max_symbol_day_workers: int = 2
    max_contract_workers: int = 4
    max_requests_per_second: float = 8.0
    max_inflight_requests: int = 8
    store_raw_payloads: bool = False
    refresh_no_data: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self):
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
        # Worker counts, scope, and optional reference providers do not change
        # which contracts are selected or invalidate already collected sessions.
        names = ("option_rights", "target_dtes", "max_expirations_per_day", "moneyness_targets",
                 "strikes_per_moneyness_target", "min_dte", "max_dte", "exchange_tz",
                 "quote_interval", "stock_venue", "selection_times", "max_stock_quote_age_seconds",
                 "trade_quote_exclusive")
        return {"output_schema_version": OUTPUT_SCHEMA_VERSION,
                **{name: getattr(self, name) for name in names}}

    @property
    def policy_id(self) -> str:
        return digest_json(self.policy())[:20]


@dataclass(frozen=True)
class Request:
    dataset: str
    endpoint: str
    params: dict
    vendor: str = "ThetaData"

    def identity(self) -> dict:
        return asdict(self)

    @property
    def request_id(self) -> str:
        return digest_json(self.identity())[:24]

    @property
    def required_columns(self) -> tuple[str, ...]:
        kind = self.endpoint.rsplit("/", 1)[-1]
        if self.dataset == "quoted_contracts":
            return CONTRACT_FIELDS
        if self.vendor == "Yahoo":
            return ("Date", "Dividends", "Stock Splits")
        if self.dataset == "interest_rate_eod":
            return ("created", "rate")
        fields = {"quote": ("timestamp", *QUOTE_FIELDS), "trade_quote": TRADE_FIELDS,
                  "open_interest": ("timestamp", "open_interest"),
                  "eod": ("created", "last_trade", "open", "high", "low", "close", "volume", "count")}
        return ((*CONTRACT_FIELDS,) if self.endpoint.startswith("/option/") else ()) + fields[kind]


def utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def atomic_output(path: Path):
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
        if stat.st_size != receipt["size"]:
            return False
        if stat.st_mtime_ns != receipt["mtime_ns"] and file_hash(path) != receipt["sha256"]:
            return False
        if "rows" in receipt:
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


def parse_vendor_clock(values: pd.Series, exchange_tz: str) -> pd.Series:
    # Naive Theta clocks are exchange local. Aware clocks keep their stated
    # offset. Date-only interest-rate reports deliberately do not use this.
    text = values.astype("string").str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    parsed.loc[aware] = pd.to_datetime(text.loc[aware], format="mixed", errors="coerce", utc=True)
    naive = pd.to_datetime(text.loc[~aware], format="mixed", errors="coerce")
    parsed.loc[~aware] = naive.dt.tz_localize(
        exchange_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return parsed


def raw_frame_with_diagnostics(frame: pd.DataFrame, request: Request, cfg: CollectorConfig) -> tuple[pd.DataFrame, dict]:
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
        bid, ask = (pd.to_numeric(frame[side], errors="coerce") for side in ("bid", "ask"))
        diagnostics.update(invalid_bid_ask_rows=int((~np.isfinite(bid) | ~np.isfinite(ask)).sum()),
                           nonpositive_bid_ask_rows=int((bid.le(0) | ask.le(0)).sum()),
                           crossed_quote_rows=int(bid.gt(ask).sum()))
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
    end = request.params.get("date", request.params.get("end_date"))
    primary = next((name for name in ("timestamp", "trade_timestamp", "created") if name in frame), None)
    if start and end and primary:
        if request.dataset == "interest_rate_eod":
            dates = pd.to_datetime(frame[primary].str.strip(), format="mixed", errors="coerce")
        else:
            dates = parse_vendor_clock(frame[primary], cfg.exchange_tz).dt.tz_convert(
                cfg.exchange_tz).dt.tz_localize(None).dt.normalize()
        if (dates.notna() & ~dates.between(pd.Timestamp(start), pd.Timestamp(end))).any():
            issues.append("timestamps_outside_requested_dates")
        if dates.isna().all():
            issues.append("no_parseable_report_dates")
    return issues


def interval_seconds(interval: str) -> float:
    # API "m" means minutes; pandas also accepts other, ambiguous abbreviations.
    if interval.endswith("ms"):
        return float(interval[:-2]) / 1000
    return float(interval[:-1]) * {"s": 1, "m": 60, "h": 3600}[interval[-1]]


def format_strike(value) -> str:
    strike = Decimal(str(value))
    if not strike.is_finite() or strike <= 0 or strike != strike.quantize(Decimal("0.001")):
        raise ValueError(f"Invalid option strike: {value!r}")
    return format(strike, ".3f").rstrip("0").rstrip(".")


class ThetaClient:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.local = threading.local()
        self.semaphore = threading.BoundedSemaphore(cfg.max_inflight_requests)
        self.pace_lock = threading.Lock()
        self.next_allowed = 0.0

    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            # Retries are explicit so every attempt obeys the same request budget.
            adapter = HTTPAdapter(max_retries=0, pool_connections=2, pool_maxsize=2)
            self.local.session.mount("http://", adapter)
            self.local.session.mount("https://", adapter)
        return self.local.session

    @contextmanager
    def request_slot(self):
        with self.semaphore:
            with self.pace_lock:
                wait = max(0.0, self.next_allowed - time.monotonic())
                self.next_allowed = max(time.monotonic(), self.next_allowed) + 1 / self.cfg.max_requests_per_second
            if wait:
                time.sleep(wait)
            yield

    def download(self, request: Request) -> tuple[bytes | None, dict]:
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
                meta = {"request_url": response.url, "status_code": response.status_code,
                        "response_headers": dict(response.headers), "attempts": attempt + 1,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_bytes": len(payload)}
                if response.status_code in {200, 472}:
                    return payload, meta
                meta["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in {429, 474, 500, 502, 503, 504, 571} or attempt == 5:
                    return payload, meta
                try:
                    retry_after = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    pass
            except requests.RequestException as exc:
                payload = None
                meta = {"error": repr(exc), "attempts": attempt + 1, "status_code": None}
                if attempt == 5:
                    return payload, meta
            time.sleep(max(min(30.0, 2 ** attempt), min(max(retry_after, 0), 120.0)))
        return None, meta


# Raw storage and request provenance are independent of contract selection.
class RequestStore:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.root = cfg.output_dir
        self.client = ThetaClient(cfg)
        self.locks = tuple(threading.Lock() for _ in range(128))

    def directory(self, request: Request) -> Path:
        date = request.params.get("date", request.params.get("start_date", "reference"))
        date = str(date).replace("-", "")
        symbol = request.params["symbol"]
        return (self.root / "raw_cache" / request.dataset / f"symbol={symbol}" /
                f"date={date}" / f"request={request.request_id}")

    def cached(self, request: Request) -> dict | None:
        path = self.directory(request) / "meta.json"
        try:
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
            if self.cfg.store_raw_payloads and request.vendor == "ThetaData" and not meta.get("payload"):
                return None
            return self.record(request, meta, path)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def record(self, request: Request, meta: dict, meta_path: Path) -> dict:
        return {"request_id": request.request_id, "dataset": request.dataset,
                "status": meta["status"], "row_count": meta.get("row_count", 0),
                "error": meta.get("error", ""), "data": meta.get("data"),
                "payload": meta.get("payload"), "metadata": file_receipt(meta_path, self.root)}

    def collect(self, request: Request) -> dict:
        with self.locks[hash(request.request_id) % len(self.locks)]:
            cached = self.cached(request)
            if cached:
                return cached
            if request.vendor == "Yahoo":
                return self.collect_yahoo(request)
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
        directory = self.directory(request)
        missing = set(request.required_columns) - set(frame.columns)
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
        write_json(directory / "meta.json", meta)
        return self.record(request, meta, directory / "meta.json")

    def legacy_response(self, request: Request):
        """Reuse valid pre-refactor raw quotes/OI/chains without changing old files."""
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

    def collect_yahoo(self, request: Request) -> dict:
        # Separate, explicitly requested reference snapshot. Yahoo never supplies
        # a required field to the Theta collection or an inferred zero dividend.
        try:
            import yfinance as yf
            history = yf.Ticker(request.params["symbol"]).history(
                start=request.params["start_date"],
                end=(pd.Timestamp(request.params["end_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False, back_adjust=False, actions=True, repair=False,
                keepna=True, rounding=False, raise_errors=True)
            frame = history.reset_index()
            meta = {"fetched_at_utc": utc_now(), "package_version": version("yfinance"),
                    "reference_only": True, "historical_vintages_verified": False,
                    "empty_does_not_prove_no_actions": True}
            return self.save(request, frame, meta)
        except Exception as exc:
            return self.save(request, pd.DataFrame(columns=request.required_columns),
                             {"error": repr(exc), "reference_only": True}, status="request_error")

    def read(self, record: dict) -> pd.DataFrame:
        if record["status"] not in GOOD_REQUEST_STATUSES or not record.get("data"):
            return pd.DataFrame()
        return pd.read_parquet(self.root / record["data"]["path"])


@lru_cache(maxsize=1)
def exchange_calendar():
    return xcals.get_calendar("XNYS", start="2017-01-01", end="2026-12-31")


def session_bounds(day: pd.Timestamp, cfg: CollectorConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    calendar = exchange_calendar()
    return (calendar.session_open(day).tz_convert(cfg.exchange_tz),
            calendar.session_close(day).tz_convert(cfg.exchange_tz))


def history_request(cfg: CollectorConfig, asset: str, kind: str, symbol: str,
                    day: pd.Timestamp, contract: dict | None = None) -> Request:
    params = {"symbol": symbol, "date": day.strftime("%Y%m%d"), "format": "csv"}
    if asset == "option":
        if contract is None:
            raise ValueError("Option history requires an observed contract")
        params.update(expiration=contract["expiration"], strike=format_strike(contract["strike"]),
                      right=contract["right"])
    if kind == "quote":
        params["interval"] = cfg.quote_interval
        dataset = f"{asset}_quotes_{cfg.quote_interval}"
    elif kind == "trade_quote":
        params["exclusive"] = str(cfg.trade_quote_exclusive).lower()
        dataset = f"{asset}_trade_quotes_tick"
    else:
        dataset = f"{asset}_{kind}"
    if kind == "eod":
        params.pop("date")
        params.update(start_date=day.strftime("%Y%m%d"), end_date=day.strftime("%Y%m%d"))
    elif kind != "open_interest":
        opened, closed = session_bounds(day, cfg)
        params.update(start_time=opened.strftime("%H:%M:%S"), end_time=closed.strftime("%H:%M:%S"))
        if asset == "stock":
            params["venue"] = cfg.stock_venue
    return Request(dataset, f"/{asset}/history/{kind}", params)


def normalize_chain(frame: pd.DataFrame, symbol: str, cfg: CollectorConfig) -> pd.DataFrame:
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
    if frame.empty:
        return []
    # Only the stock spot used to choose strikes needs a usable quote. Raw storage
    # never uses this condition policy, and option observations are never filtered.
    quotes = frame.copy()
    quotes["_clock"] = parse_vendor_clock(quotes["timestamp"], cfg.exchange_tz)
    allowed = pd.Series(True, index=quotes.index)
    for column in ("bid_condition", "ask_condition"):
        text = quotes[column].astype("string").str.strip()
        allowed &= text.eq("") | pd.to_numeric(text, errors="coerce").isin([0, 1, 50])
    quotes = quotes.loc[allowed & quotes["_clock"].notna()].sort_values("_clock", kind="stable")
    opened, closed = session_bounds(day, cfg)
    references = []
    for selection_time in cfg.selection_times:
        at = pd.Timestamp(f"{day.date()} {selection_time}", tz=cfg.exchange_tz)
        if not opened <= at < closed:
            continue
        prior = quotes.loc[quotes["_clock"].between(opened, at)]
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
    remaining = sorted(exp for exp in expirations if cfg.min_dte <= (exp - day).days <= cfg.max_dte)
    selected = []
    for target in cfg.target_dtes:
        if not remaining or len(selected) >= cfg.max_expirations_per_day:
            break
        best = min(remaining, key=lambda exp: (abs((exp - day).days - target), (exp - day).days))
        selected.append(best)
        remaining.remove(best)
    remaining.sort(key=lambda exp: (min(abs((exp - day).days - target) for target in cfg.target_dtes), exp))
    return sorted(selected + remaining[:max(cfg.max_expirations_per_day - len(selected), 0)])


def select_contracts(chain: pd.DataFrame, day: pd.Timestamp, references: list[dict],
                     cfg: CollectorConfig) -> pd.DataFrame:
    rows = []
    for expiration in eligible_expirations(day, chain["expiration"].unique(), cfg):
        family = chain.loc[chain["expiration"].eq(expiration)]
        strikes = sorted(family["strike"].unique())
        chosen_by_time = {}
        for reference in references:
            chosen = set()
            for target in cfg.moneyness_targets:
                target_strike = reference["stock_mid"] / target
                ranked = sorted(strikes, key=lambda strike: (abs(strike - target_strike), strike))
                chosen.update(ranked[:cfg.strikes_per_moneyness_target])
            chosen_by_time[reference["selection_time"]] = chosen
        selected = set().union(*chosen_by_time.values()) if chosen_by_time else set()
        for contract in family.loc[family["strike"].isin(selected)].itertuples(index=False):
            expiry, strike = expiration.strftime("%Y-%m-%d"), format_strike(contract.strike)
            rows.append({"symbol": contract.symbol, "expiration": expiry, "strike": strike,
                         "right": contract.right, "contract_key": f"{contract.symbol}|{expiry}|{strike}|{contract.right}",
                         "dte_days": (expiration - day).days,
                         "selection_times": "|".join(at for at, values in chosen_by_time.items() if contract.strike in values)})
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


# Session manifests describe collection coverage; they contain no research features.
class Collector:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.store = RequestStore(cfg)
        self.directory = cfg.output_dir / "collection" / cfg.policy_id

    def session_path(self, symbol: str, day: pd.Timestamp) -> Path:
        return self.directory / "sessions" / f"symbol={symbol}__date={day.date()}.json"

    def manifest_valid(self, manifest: dict) -> bool:
        try:
            if (manifest["output_schema_version"] != OUTPUT_SCHEMA_VERSION
                    or manifest["policy_id"] != self.cfg.policy_id
                    or manifest["status"] not in {"complete", "unavailable"}):
                return False
            records = manifest["requests"]
            selected_count = manifest["selected_contract_count"]
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
        try:
            manifest = read_json(self.session_path(symbol, day))
            return (manifest.get("symbol") == symbol and manifest.get("trade_day") == str(day.date())
                    and self.manifest_valid(manifest))
        except (OSError, ValueError):
            return False

    def collect_day(self, symbol_cfg: SymbolConfig, day: pd.Timestamp) -> dict:
        symbol = symbol_cfg.symbol
        records, references = [], []
        selected = pd.DataFrame(columns=SELECTION_COLUMNS)
        reason, error, discovered = "", "", 0
        try:
            stock = self.store.collect(history_request(self.cfg, "stock", "quote", symbol, day))
            records.append(stock)
            # Discovery is independent of stock availability. Keep the dated
            # universe even when stock quotes cannot support strike selection.
            chain_record = self.store.collect(Request("quoted_contracts", "/option/list/contracts/quote",
                                              {"symbol": symbol, "date": day.strftime("%Y%m%d"), "format": "csv"}))
            records.append(chain_record)
            for kind in ("trade_quote", "eod"):
                records.append(self.store.collect(history_request(self.cfg, "stock", kind, symbol, day)))
            chain = normalize_chain(self.store.read(chain_record), symbol, self.cfg)
            discovered = len(chain)
            references = stock_selection_references(self.store.read(stock), day, self.cfg)
            selected = select_contracts(chain, day, references, self.cfg)
            if chain.empty:
                reason = "no_quoted_contracts"
            elif not references:
                reason = "stock_selection_reference_unavailable"
            elif selected.empty:
                reason = "no_contracts_in_sampling_window"
            with ThreadPoolExecutor(max_workers=self.cfg.max_contract_workers) as pool:
                futures = []
                for contract in selected.to_dict("records"):
                    for kind in ("quote", "trade_quote", "open_interest"):
                        request = history_request(self.cfg, "option", kind, symbol, day, contract)
                        futures.append(pool.submit(self.store.collect, request))
                for future in as_completed(futures):
                    records.append(future.result())
        except Exception as exc:
            # Completed raw pulls survive; a failed day never satisfies resume.
            error = repr(exc)
        failures = sum(record["status"] not in GOOD_REQUEST_STATUSES for record in records)
        status = "request_error" if error or failures else ("unavailable" if selected.empty else "complete")
        contract_path = self.directory / "contracts" / f"symbol={symbol}__date={day.date()}.parquet"
        write_parquet(contract_path, selected)
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
        return manifest

    def collect_references(self, symbols: list[SymbolConfig], start: str, end: str,
                           rate_symbols: list[str], yahoo_actions: bool, run_id: str) -> list[dict]:
        requests_to_make = [
            Request("interest_rate_eod", "/interest_rate/history/eod",
                    {"symbol": symbol, "start_date": start, "end_date": end, "format": "csv"})
            for symbol in sorted(set(rate_symbols))
        ]
        if yahoo_actions:
            requests_to_make.extend(
                Request("yahoo_actions_reference", "Ticker.history",
                        {"symbol": symbol.symbol, "start_date": start, "end_date": end}, vendor="Yahoo")
                for symbol in symbols)
        records = []
        for request in requests_to_make:
            try:
                record = self.store.collect(request)
            except Exception as exc:
                record = {"request_id": request.request_id, "dataset": request.dataset,
                          "status": "request_error", "error": repr(exc)}
            records.append(record)
            # The ledger is separate: reference failures do not invalidate Theta days.
            write_json(self.cfg.output_dir / "references" / f"{run_id}.json",
                       {"updated_at_utc": utc_now(), "requests": records,
                        "requested_rates": sorted(set(rate_symbols)), "rate_units": "percent",
                        "rate_publication_timestamps_verified": False,
                        "yahoo_actions_requested": yahoo_actions,
                        "authoritative_adjusted_contract_deliverables": "not_collected"})
            print(f"Reference {request.params['symbol']} {request.dataset}: {record['status']}")
        return records

    def write_availability(self, symbols: list[SymbolConfig], anchors: pd.DatetimeIndex) -> dict:
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
                        except FileNotFoundError:
                            pass
                        except (OSError, ValueError) as exc:
                            row.update(status="request_error", error=repr(exc))
                        else:
                            row.update({key: manifest.get(key, "") for key in columns if key in manifest})
                            records = manifest["requests"]
                            row.update(selection_reference_count=len(manifest["stock_selection_references"]),
                                       missing_selection_times="|".join(manifest["missing_selection_times"]),
                                       request_count=len(records),
                                       no_data_request_count=sum(r["status"] == "no_data" for r in records),
                                       stored_rows=sum(r.get("row_count", 0) for r in records),
                                       stored_parquet_bytes=sum((r.get("data") or {}).get("size", 0) for r in records))
                        counts[row["status"]] += 1
                        writer.writerow(row)
        return counts


def package_versions() -> dict:
    result = {}
    for package in ("pandas", "numpy", "pyarrow", "requests", "exchange-calendars", "yfinance"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not_installed"
    return result


def parse_run_scope(argv: list[str] | None = None):
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
    parser.add_argument("--rate-symbols", nargs="+", default=[], type=str.upper,
                        help="Optional Theta rate series, e.g. SOFR; no default rate is assumed")
    parser.add_argument("--yahoo-actions", action="store_true", help="Optional Yahoo daily history/actions reference")
    parser.add_argument("--references-only", action="store_true", help="Fetch only the explicitly requested references")
    parser.add_argument("--plan", action="store_true", help="Show scope without network requests or output writes")
    args = parser.parse_args(argv)
    try:
        if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (args.start, args.end)):
            raise ValueError("Dates must use YYYY-MM-DD")
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if not pd.Timestamp(defaults.start_date) <= start <= end <= pd.Timestamp(defaults.end_date):
            raise ValueError(f"Dates must be ordered and within {defaults.start_date} through {defaults.end_date}")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9_.-]{0,31}", symbol) for symbol in args.rate_symbols):
            raise ValueError("Rate symbols must be explicit series names")
        if args.references_only and not (args.rate_symbols or args.yahoo_actions):
            raise ValueError("--references-only requires --rate-symbols or --yahoo-actions")
        cfg = replace(defaults, quote_interval=args.quote_interval, output_dir=args.output_dir.expanduser().resolve(),
                      store_raw_payloads=args.store_raw_payloads, refresh_no_data=args.refresh_no_data,
                      max_inflight_requests=args.max_inflight_requests,
                      max_requests_per_second=args.max_requests_per_second)
    except ValueError as exc:
        parser.error(str(exc))
    anchors = exchange_calendar().sessions_in_range(start, end).tz_localize(None)
    symbols = [cfg for cfg in UNIVERSE if args.symbols is None or cfg.symbol in args.symbols]
    return args, cfg, symbols, anchors


def main(argv: list[str] | None = None) -> int:
    args, cfg, symbols, anchors = parse_run_scope(argv)
    total = 0 if args.references_only else len(symbols) * len(anchors)
    print(f"Scope: {', '.join(s.symbol for s in symbols)}; {args.start} to {args.end}; {total} symbol-days")
    print(f"Quotes: {cfg.quote_interval}; trades: events with matched quotes; stock venue: {cfg.stock_venue}")
    print(f"Output: {cfg.output_dir}")
    print(f"Optional references: rates={','.join(args.rate_symbols) or 'none'}, Yahoo actions={args.yahoo_actions}")
    if args.plan:
        return 0
    collector = Collector(cfg)
    run_id = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    run_path = collector.directory / "runs" / f"{run_id}.json"
    run = {"run_id": run_id, "started_at_utc": utc_now(), "status": "running",
           "policy_id": cfg.policy_id, "policy": cfg.policy(), "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
           "scope": {"symbols": [s.symbol for s in symbols], "start": args.start, "end": args.end,
                     "references_only": args.references_only, "rate_symbols": args.rate_symbols,
                     "yahoo_actions": args.yahoo_actions},
           "code_sha256": file_hash(Path(__file__)), "python": sys.version, "platform": platform.platform(),
           "packages": package_versions(), "resumed_days": 0, "processed_days": 0}
    failed = reference_failures = 0
    try:
        with output_lock(cfg.output_dir):
            write_json(run_path, run)
            try:
                if not args.references_only:
                    for day in anchors:
                        pending = []
                        for symbol in symbols:
                            if collector.resumable(symbol.symbol, day):
                                run["resumed_days"] += 1
                            else:
                                pending.append(symbol)
                        with ThreadPoolExecutor(max_workers=cfg.max_symbol_day_workers) as pool:
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
                                except Exception as exc:
                                    failed += 1
                                    print(f"FAILED {symbol.symbol} {day.date()}: {exc!r}")
                references = collector.collect_references(
                    symbols, args.start, args.end, args.rate_symbols, args.yahoo_actions, run_id)
                reference_failures = sum(r["status"] not in GOOD_REQUEST_STATUSES for r in references)
                run["status"] = "partial_failure" if failed or reference_failures else "complete"
            except BaseException:
                run["status"] = "interrupted"
                raise
            finally:
                run.update(finished_at_utc=utc_now(), failed_days=int(failed), reference_failures=reference_failures)
                if not args.references_only:
                    run["coverage"] = collector.write_availability(symbols, anchors)
                write_json(run_path, run)
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        print(f"Collector stopped: {exc}", file=sys.stderr)
        return 1
    print(f"Finished: {run['processed_days']} processed, {run['resumed_days']} resumed, "
          f"{failed} failed days, {reference_failures} failed references")
    print(f"Run record: {run_path}")
    if not args.references_only:
        print(f"Coverage: {collector.directory / 'availability.csv'}")
    return 1 if failed or reference_failures else 0


if __name__ == "__main__":
    sys.exit(main())
