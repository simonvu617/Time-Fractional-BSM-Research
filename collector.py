import json
import hashlib
import os
import platform
import re
import sys
import time
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from importlib.metadata import version as pkg_version
from io import BytesIO
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps single-process tests importable.
    fcntl = None


# ============================================================
# CONFIG
# ============================================================
@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str


@dataclass(frozen=True)
class BacktestConfig:
    base_url: str
    start_date: str
    end_date: str
    anchor_freq: str
    option_rights: tuple[str, ...]
    target_dtes: tuple[int, ...]
    max_expirations_per_day: int
    moneyness_targets: tuple[float, ...]
    strikes_per_moneyness_target: int
    min_dte: int
    max_dte: int
    exchange_tz: str
    quote_interval: str
    trade_interval: str
    evaluation_times: tuple[str, ...]
    horizon_minutes: int
    max_stock_quote_age_seconds: int
    max_option_quote_age_seconds: int
    max_rel_spread: float
    min_option_bid_size: int
    min_option_ask_size: int
    min_open_interest: int
    min_recent_option_trades: int
    recent_trade_lookback_minutes: int
    max_symbol_day_workers: int
    max_contract_workers: int
    max_requests_per_second: float
    max_inflight_requests: int
    soft_failure_rate_threshold: float
    stream_flush_row_count: int
    store_raw_payloads: bool
    assemble_csv_outputs: bool
    output_dir: Path
    raw_cache_dir: Path
    market_data_dir: Path

    def __post_init__(self) -> None:
        valid_rights = {"call", "put"}
        invalid_rights = set(self.option_rights) - valid_rights
        if invalid_rights:
            raise ValueError(f"Unsupported option_rights: {sorted(invalid_rights)}")
        if not 0.0 <= self.soft_failure_rate_threshold <= 1.0:
            raise ValueError("soft_failure_rate_threshold must be between 0 and 1")


BASE_URL = "http://127.0.0.1:25503/v3"
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "multi_year_bsm_backtest_output"
RAW_CACHE_DIR = OUTPUT_DIR / "raw_cache"
MARKET_DATA_DIR = OUTPUT_DIR / "market_inputs"
REFERENCE_DATA_DIR = OUTPUT_DIR / "reference_data"
CANONICAL_DATA_DIR = OUTPUT_DIR / "canonical"
DIAGNOSTIC_DATA_DIR = OUTPUT_DIR / "diagnostics"
CHAIN_METADATA_DIR = REFERENCE_DATA_DIR / "chain_metadata"
CANONICAL_PARTS_DIR = CANONICAL_DATA_DIR / "parts"
DIAGNOSTIC_PARTS_DIR = DIAGNOSTIC_DATA_DIR / "parts"
# Bump this whenever canonical/session schemas change in a way that makes old
# parquet chunks unsafe to treat as completed work.
OUTPUT_SCHEMA_VERSION = "2026-05-26-paper_comparison_surface_v6"

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

CFG = BacktestConfig(
    base_url=BASE_URL,
    start_date="2018-01-01",
    end_date="2025-12-31",
    anchor_freq="B",
    option_rights=("call", "put"),
    target_dtes=(7, 14, 30, 60, 120),
    max_expirations_per_day=5,
    moneyness_targets=(0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20),
    strikes_per_moneyness_target=1,
    min_dte=7,
    max_dte=180,
    exchange_tz="America/New_York",
    quote_interval="1s",
    trade_interval="1s",
    evaluation_times=("10:30:00", "13:00:00", "15:00:00"),
    horizon_minutes=30,
    max_stock_quote_age_seconds=70,
    max_option_quote_age_seconds=70,
    max_rel_spread=0.35,
    min_option_bid_size=1,
    min_option_ask_size=1,
    min_open_interest=100,
    min_recent_option_trades=1,
    recent_trade_lookback_minutes=30,
    max_symbol_day_workers=2,
    max_contract_workers=4,
    max_requests_per_second=8.0,
    max_inflight_requests=8,
    soft_failure_rate_threshold=0.05,
    stream_flush_row_count=500,
    store_raw_payloads=False,
    assemble_csv_outputs=False,
    output_dir=OUTPUT_DIR,
    raw_cache_dir=RAW_CACHE_DIR,
    market_data_dir=MARKET_DATA_DIR,
)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def to_float(value: object, default: float = np.nan) -> float:
    # Avoid the repeated one-row Series pattern when coercing scalar fields.
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except Exception:
        return default
    return default if pd.isna(numeric) else float(numeric)


def to_int(value: object, default: int = 0) -> int:
    numeric = to_float(value, np.nan)
    return default if pd.isna(numeric) else int(numeric)


def config_payload() -> dict:
    cfg_payload = asdict(CFG)
    cfg_payload["output_dir"] = str(CFG.output_dir)
    cfg_payload["raw_cache_dir"] = str(CFG.raw_cache_dir)
    cfg_payload["market_data_dir"] = str(CFG.market_data_dir)
    return cfg_payload


_CONFIG_PAYLOAD = config_payload()
_CONFIG_DIGEST = sha256_bytes(json.dumps(_CONFIG_PAYLOAD, sort_keys=True).encode("utf-8"))
_CODE_DIGEST = sha256_bytes(Path(__file__).read_bytes())
_RUN_STARTED_UTC = pd.Timestamp.utcnow().isoformat()
_THREAD_LOCAL = threading.local()
_PATH_LOCK_STRIPES = tuple(threading.Lock() for _ in range(256))
_REQUEST_PACE_LOCK = threading.Lock()
_REQUEST_NEXT_ALLOWED = 0.0
_REQUEST_SEMAPHORE = threading.BoundedSemaphore(max(1, CFG.max_inflight_requests))
_REQUEST_FAILURE_LOCK = threading.Lock()
_REQUEST_FAILURE_COUNTS: dict[str, int] = {}
SESSION_KEY_PATTERN = re.compile(r"^symbol=(?P<symbol>.+)__date=(?P<date>\d{4}-\d{2}-\d{2})$")


def path_lock(path: Path) -> threading.Lock:
    return _PATH_LOCK_STRIPES[hash(str(path.resolve())) % len(_PATH_LOCK_STRIPES)]


@contextmanager
def file_lock(path: Path):
    ensure_dir(path.parent)
    lock_path = path.with_name(f"{path.name}.lock")
    if fcntl is None:
        yield
        return
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def request_slot():
    global _REQUEST_NEXT_ALLOWED
    _REQUEST_SEMAPHORE.acquire()
    try:
        with _REQUEST_PACE_LOCK:
            now = time.monotonic()
            wait_seconds = max(0.0, _REQUEST_NEXT_ALLOWED - now)
            spacing = 1.0 / max(CFG.max_requests_per_second, 0.001)
            _REQUEST_NEXT_ALLOWED = max(now, _REQUEST_NEXT_ALLOWED) + spacing
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        yield
    finally:
        _REQUEST_SEMAPHORE.release()


def adaptive_request_delay(endpoint: str) -> None:
    with _REQUEST_FAILURE_LOCK:
        failures = _REQUEST_FAILURE_COUNTS.get(endpoint, 0)
        delay = min(5.0, 0.25 * (2 ** min(failures - 1, 4))) if failures > 0 else 0.0
    if delay > 0:
        time.sleep(delay)


def record_request_result(endpoint: str, success: bool) -> None:
    with _REQUEST_FAILURE_LOCK:
        if success:
            _REQUEST_FAILURE_COUNTS.pop(endpoint, None)
        else:
            _REQUEST_FAILURE_COUNTS[endpoint] = _REQUEST_FAILURE_COUNTS.get(endpoint, 0) + 1


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def build_request_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    pool_size = max(20, CFG.max_contract_workers * 4)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_request_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "request_session", None)
    if session is None:
        session = build_request_session()
        _THREAD_LOCAL.request_session = session
    return session


XNYS = xcals.get_calendar("XNYS")


def safe_pkg_version(name: str) -> str:
    try:
        return pkg_version(name)
    except Exception:
        return "unknown"


def run_context() -> dict:
    return {
        "collector_name": "multi_year_bsm_backtest",
        "collector_mode": "neutral_surface_scrape",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "config_digest": _CONFIG_DIGEST,
        "code_digest": _CODE_DIGEST,
        "config": dict(_CONFIG_PAYLOAD),
        "timestamp_policy": {
            "assumption_for_naive_vendor_timestamps": f"localize_to_{CFG.exchange_tz}",
            "dst_ambiguous_behavior": "NaT",
            "dst_nonexistent_behavior": "NaT",
            "note": "timestamp semantics are monitored and flagged, not proven by this collector",
        },
        "data_limitations": {
            "is_tick_level": False,
            "uses_requested_intervals": {
                "quotes": CFG.quote_interval,
                "trades": CFG.trade_interval,
            },
            "full_market_archive": False,
            "instrument_identity_is_observed_not_authoritative": True,
            "corporate_actions_source": "yfinance_reference_layer",
            "bsm_implied_volatility_computed_in_collector": False,
            "iv_validity_filter_required_downstream": True,
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "pandas": safe_pkg_version("pandas"),
            "numpy": safe_pkg_version("numpy"),
            "requests": safe_pkg_version("requests"),
            "yfinance": safe_pkg_version("yfinance"),
            "exchange_calendars": safe_pkg_version("exchange-calendars"),
        },
        "run_started_utc": _RUN_STARTED_UTC,
    }


def normalize_vendor_timestamp(series: pd.Series) -> pd.Series:
    ts, _, _ = parse_vendor_timestamp(series)
    return ts


def parse_vendor_timestamp(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ts = pd.to_datetime(series, errors="coerce")
    naive_assumed = pd.Series(False, index=series.index, dtype=bool)
    if getattr(ts.dt, "tz", None) is None:
        naive_assumed = ts.notna()
        ts = ts.dt.tz_localize(CFG.exchange_tz, ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts.dt.tz_convert(CFG.exchange_tz)
    return ts, ts.isna(), naive_assumed


def attach_timestamp_metadata(df: pd.DataFrame, include_raw: bool = False) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    parsed, parse_failed, naive_assumed = parse_vendor_timestamp(df["timestamp"])
    result = df.copy()
    if include_raw:
        result["timestamp_raw"] = result["timestamp"]
    result["timestamp"] = parsed
    result["timestamp_parse_failed"] = parse_failed
    result["timestamp_naive_assumed"] = naive_assumed
    return result


def optimize_frame_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df
    for col in ("bid", "ask", "bid_size", "ask_size", "price", "size", "open_interest"):
        if col in result.columns and result[col].dtype == object:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def session_timestamp(day: pd.Timestamp, time_str: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.strftime('%Y-%m-%d')} {time_str}", tz=CFG.exchange_tz)


def time_label(time_str: str) -> str:
    return time_str.replace(":", "")


def evaluation_schedule(trade_day: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    _, session_close = market_session_bounds(trade_day)
    schedule = []
    for time_str in CFG.evaluation_times:
        entry_ts = session_timestamp(trade_day, time_str)
        exit_ts = min(entry_ts + pd.Timedelta(minutes=CFG.horizon_minutes), session_close)
        schedule.append((time_label(time_str), entry_ts, exit_ts))
    return schedule


def _longest_streak_duration(change_flags: pd.Series, durations: pd.Series) -> float:
    change_arr = pd.Series(change_flags).fillna(True).to_numpy(dtype=bool)
    duration_arr = pd.Series(durations).fillna(0.0).to_numpy(dtype=float)
    if change_arr.size == 0 or duration_arr.size == 0:
        return 0.0
    usable = min(change_arr.size, duration_arr.size)
    change_arr = change_arr[:usable]
    duration_arr = duration_arr[:usable].copy()
    duration_arr[change_arr] = 0.0
    group_ids = np.cumsum(change_arr)
    streak_sums = np.bincount(group_ids, weights=duration_arr)
    return float(streak_sums.max()) if streak_sums.size else 0.0


def window_update_features(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    *,
    window_minutes: int,
    prefix: str,
    value_column: str | None = None,
) -> dict:
    window_end = target_ts
    window_start = target_ts - pd.Timedelta(minutes=window_minutes)
    result = {
        f"{prefix}_window_start": window_start,
        f"{prefix}_window_end": window_end,
        f"{prefix}_window_row_count": 0,
        f"{prefix}_time_since_last_update_seconds": np.nan,
        f"{prefix}_update_count": 0,
        f"{prefix}_median_interarrival_seconds": np.nan,
        f"{prefix}_p95_interarrival_seconds": np.nan,
        f"{prefix}_max_interarrival_seconds": np.nan,
        f"{prefix}_last_update_gap_seconds": np.nan,
        f"{prefix}_longest_no_update_gap_seconds": np.nan,
        f"{prefix}_value_change_count": 0,
        f"{prefix}_zero_return_fraction": np.nan,
        f"{prefix}_longest_no_change_streak_seconds": np.nan,
        f"{prefix}_share_window_unchanged": np.nan,
        f"{prefix}_last_value_change_age_seconds": np.nan,
    }
    if df.empty or "timestamp" not in df.columns:
        return result
    window = df.loc[df["timestamp"].notna() & df["timestamp"].between(window_start, window_end)].sort_values("timestamp").copy()
    if window.empty:
        return result
    result[f"{prefix}_window_row_count"] = int(len(window))
    result[f"{prefix}_update_count"] = int(len(window))
    last_ts = window["timestamp"].iloc[-1]
    result[f"{prefix}_time_since_last_update_seconds"] = float((window_end - last_ts).total_seconds())
    interarrival = window["timestamp"].diff().dt.total_seconds().dropna()
    if not interarrival.empty:
        result[f"{prefix}_median_interarrival_seconds"] = float(interarrival.median())
        result[f"{prefix}_p95_interarrival_seconds"] = float(interarrival.quantile(0.95))
        result[f"{prefix}_max_interarrival_seconds"] = float(interarrival.max())
    boundaries = pd.Series(pd.DatetimeIndex([window_start, *window["timestamp"].tolist(), window_end]))
    boundary_gaps = boundaries.diff().dt.total_seconds().dropna()
    if not boundary_gaps.empty:
        result[f"{prefix}_last_update_gap_seconds"] = float(boundary_gaps.iloc[-1])
        result[f"{prefix}_longest_no_update_gap_seconds"] = float(boundary_gaps.max())
    if value_column is None or value_column not in window.columns:
        return result
    values = pd.to_numeric(window[value_column], errors="coerce")
    valid = window.loc[values.notna(), ["timestamp"]].copy()
    valid[value_column] = values.loc[values.notna()].values
    if valid.empty:
        return result
    valid["next_timestamp"] = valid["timestamp"].shift(-1).fillna(window_end)
    valid["duration_seconds"] = (valid["next_timestamp"] - valid["timestamp"]).dt.total_seconds().clip(lower=0.0)
    valid["value_changed"] = valid[value_column].diff().fillna(0.0).ne(0.0)
    value_changes = int(valid["value_changed"].sum())
    result[f"{prefix}_value_change_count"] = value_changes
    if len(valid) > 1:
        zero_fraction = valid[value_column].diff().fillna(0.0).eq(0.0).iloc[1:].mean()
        result[f"{prefix}_zero_return_fraction"] = float(zero_fraction)
    window_seconds = max(float((window_end - window_start).total_seconds()), 1.0)
    unchanged_duration = float(valid.loc[~valid["value_changed"], "duration_seconds"].sum())
    result[f"{prefix}_share_window_unchanged"] = unchanged_duration / window_seconds
    result[f"{prefix}_longest_no_change_streak_seconds"] = _longest_streak_duration(valid["value_changed"], valid["duration_seconds"])
    changed_rows = valid.loc[valid["value_changed"], "timestamp"]
    if not changed_rows.empty:
        result[f"{prefix}_last_value_change_age_seconds"] = float((window_end - changed_rows.iloc[-1]).total_seconds())
    return result


def build_sample_flags(contract_row: dict) -> dict:
    # The label-specific flags are authoritative. The first-evaluation aliases
    # are explicitly named so downstream code cannot mistake them for all-day
    # inclusion flags.
    lagged_oi = to_float(contract_row.get("lagged_open_interest", np.nan))
    broad_max_quote_age = CFG.max_option_quote_age_seconds * 3
    broad_max_rel_spread = max(0.75, CFG.max_rel_spread * 2.0)
    output: dict[str, object] = {}
    clean_any = False
    broad_any = False
    primary_label = time_label(CFG.evaluation_times[0])

    for time_str in CFG.evaluation_times:
        label = time_label(time_str)
        entry_age = to_float(contract_row.get(f"eval_{label}_entry_quote_age_seconds", np.nan))
        entry_rel_spread = to_float(contract_row.get(f"eval_{label}_entry_quote_rel_spread", np.nan))
        entry_bid = to_float(contract_row.get(f"eval_{label}_entry_quote_bid", np.nan))
        entry_ask = to_float(contract_row.get(f"eval_{label}_entry_quote_ask", np.nan))
        entry_bid_size = to_float(contract_row.get(f"eval_{label}_entry_quote_bid_size", np.nan))
        entry_ask_size = to_float(contract_row.get(f"eval_{label}_entry_quote_ask_size", np.nan))
        entry_mid = to_float(contract_row.get(f"eval_{label}_entry_quote_option_mid", np.nan))
        exit_mid = to_float(contract_row.get(f"eval_{label}_exit_quote_option_mid", np.nan))
        exit_age = to_float(contract_row.get(f"eval_{label}_exit_quote_age_seconds", np.nan))
        recent_trades = to_int(contract_row.get(f"eval_{label}_entry_trade_recent_trade_count", 0))

        def clean_reasons_for(max_quote_age: float, max_rel_spread: float) -> list[str]:
            reasons: list[str] = []
            if pd.isna(entry_age) or float(entry_age) > max_quote_age:
                reasons.append("stale_entry_quote")
            if pd.notna(entry_rel_spread) and float(entry_rel_spread) > max_rel_spread:
                reasons.append("wide_entry_spread")
            if pd.notna(entry_bid) and float(entry_bid) <= 0:
                reasons.append("nonpositive_bid")
            if pd.notna(entry_ask) and pd.notna(entry_bid) and float(entry_ask) <= float(entry_bid):
                reasons.append("crossed_or_locked_entry_market")
            if pd.notna(entry_bid_size) and float(entry_bid_size) < CFG.min_option_bid_size:
                reasons.append("small_bid_size")
            if pd.notna(entry_ask_size) and float(entry_ask_size) < CFG.min_option_ask_size:
                reasons.append("small_ask_size")
            if pd.isna(lagged_oi) or float(lagged_oi) < CFG.min_open_interest:
                reasons.append("low_open_interest")
            if recent_trades < CFG.min_recent_option_trades:
                reasons.append("insufficient_recent_trades")
            return reasons

        clean_reasons = clean_reasons_for(CFG.max_option_quote_age_seconds, CFG.max_rel_spread)
        # Store threshold variants now so robustness checks do not require
        # re-scraping the raw option surface.
        tight_clean_reasons = clean_reasons_for(CFG.max_option_quote_age_seconds * 0.5, CFG.max_rel_spread * 0.5)
        loose_clean_reasons = clean_reasons_for(CFG.max_option_quote_age_seconds * 2.0, CFG.max_rel_spread * 2.0)
        broad_reasons: list[str] = []

        if pd.isna(entry_age) or float(entry_age) > broad_max_quote_age:
            broad_reasons.append("very_stale_entry_quote")
        if pd.notna(entry_rel_spread) and float(entry_rel_spread) > broad_max_rel_spread:
            broad_reasons.append("extreme_entry_spread")
        if pd.notna(entry_bid) and float(entry_bid) < 0:
            broad_reasons.append("negative_bid")
        if pd.notna(entry_ask) and pd.notna(entry_bid) and float(entry_ask) < float(entry_bid):
            broad_reasons.append("crossed_entry_market")
        if pd.isna(lagged_oi) or float(lagged_oi) < 1:
            broad_reasons.append("missing_or_zero_open_interest")

        clean_included = not clean_reasons
        tight_clean_included = not tight_clean_reasons
        loose_clean_included = not loose_clean_reasons
        broad_included = not broad_reasons
        clean_any = clean_any or clean_included
        broad_any = broad_any or broad_included
        output[f"clean_sample_included_{label}"] = clean_included
        output[f"clean_sample_exclusion_reasons_{label}"] = "|".join(clean_reasons)
        output[f"clean_sample_tight_included_{label}"] = tight_clean_included
        output[f"clean_sample_tight_exclusion_reasons_{label}"] = "|".join(tight_clean_reasons)
        output[f"clean_sample_loose_included_{label}"] = loose_clean_included
        output[f"clean_sample_loose_exclusion_reasons_{label}"] = "|".join(loose_clean_reasons)
        output[f"broad_sample_included_{label}"] = broad_included
        output[f"broad_sample_exclusion_reasons_{label}"] = "|".join(broad_reasons)
        output[f"exit_quote_stale_{label}"] = bool(pd.isna(exit_age) or float(exit_age) > CFG.max_option_quote_age_seconds)
        output[f"exit_quote_changed_{label}"] = bool(pd.notna(entry_mid) and pd.notna(exit_mid) and float(entry_mid) != float(exit_mid))
        if label == primary_label:
            output["primary_evaluation_label"] = label
            output["clean_sample_included_first_eval"] = clean_included
            output["clean_sample_exclusion_reasons_first_eval"] = "|".join(clean_reasons)
            output["broad_sample_included_first_eval"] = broad_included
            output["broad_sample_exclusion_reasons_first_eval"] = "|".join(broad_reasons)

    output["clean_sample_included_any_evaluation"] = clean_any
    output["broad_sample_included_any_evaluation"] = broad_any
    return output


def read_csv_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "timestamp" in df.columns and str(df["timestamp"].dtype) != "datetime64[ns, America/New_York]":
        df["timestamp"] = normalize_vendor_timestamp(df["timestamp"])
    for col in ["observation_date", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def boolean_flag_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(bool)
    return pd.Series(False, index=frame.index, dtype=bool)


def normalize_condition_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def filter_condition_rows(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, int]:
    condition_columns = [
        col for col in df.columns
        if any(token in col.lower() for token in ("condition", "qualifier", "indicator"))
    ]
    if not condition_columns:
        return df, 0
    allowed_tokens = {"", "regular", "normal", "auto", "continuous", "nbbo"}
    blocked_pattern = r"(?:^|[^a-z0-9])(?:late|cancel|correct|correction|odd|out_of_seq|deriv|derivative)(?:[^a-z0-9]|$)"
    mask = pd.Series(True, index=df.index, dtype=bool)
    # Blank condition fields usually mean no special sale/quote condition; keep
    # them, but count them so the assumption is visible in diagnostics.
    missing_condition_mask = pd.Series(False, index=df.index, dtype=bool)
    for column in condition_columns:
        normalized = df[column].map(normalize_condition_value)
        missing_condition_mask |= normalized.eq("")
        blocked = normalized.str.contains(blocked_pattern, regex=True, na=False)
        column_ok = normalized.isin(allowed_tokens) & ~blocked
        mask &= column_ok
    filtered = df.loc[mask].copy()
    filtered.attrs["missing_condition_row_count"] = int(missing_condition_mask.sum())
    return filtered, int((~mask).sum())


def market_snapshot(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    *,
    max_age_seconds: int | None = None,
    trade_lookback_minutes: int | None = None,
    prefix: str,
) -> dict:
    snapshot = {
        f"{prefix}_target_timestamp": target_ts,
        f"{prefix}_timestamp": pd.NaT,
        f"{prefix}_age_seconds": np.nan,
        # NaN means freshness was not evaluated for this snapshot type.
        f"{prefix}_is_stale": True if max_age_seconds is not None else np.nan,
    }
    if df.empty or "timestamp" not in df.columns:
        return snapshot
    clean = df.loc[df["timestamp"].notna()]
    if clean.empty:
        return snapshot
    if not clean["timestamp"].is_monotonic_increasing:
        clean = clean.sort_values("timestamp")
    timestamps_ns = pd.DatetimeIndex(clean["timestamp"]).asi8
    target_ns = pd.Timestamp(target_ts).value
    row_pos = int(np.searchsorted(timestamps_ns, target_ns, side="right") - 1)
    if row_pos < 0:
        return snapshot
    clean = clean.iloc[: row_pos + 1]
    row = clean.iloc[-1]
    snapshot[f"{prefix}_timestamp"] = row["timestamp"]
    snapshot[f"{prefix}_age_seconds"] = float((target_ts - row["timestamp"]).total_seconds())
    if max_age_seconds is not None:
        snapshot[f"{prefix}_is_stale"] = bool(snapshot[f"{prefix}_age_seconds"] > max_age_seconds)
    for column in ("bid", "ask", "bid_size", "ask_size", "price", "size", "stock_mid", "option_mid", "option_spread", "rel_spread"):
        if column in clean.columns:
            snapshot[f"{prefix}_{column}"] = row[column]
    if trade_lookback_minutes is not None:
        window_start = target_ts - pd.Timedelta(minutes=trade_lookback_minutes)
        recent = clean.loc[clean["timestamp"].ge(window_start)]
        snapshot[f"{prefix}_recent_trade_count"] = int(len(recent))
        snapshot[f"{prefix}_recent_trade_size"] = float(pd.to_numeric(recent.get("size", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if "price" in recent.columns and "size" in recent.columns and not recent.empty:
            price = pd.to_numeric(recent["price"], errors="coerce")
            size = pd.to_numeric(recent["size"], errors="coerce").fillna(0)
            total_size = float(size.sum())
            snapshot[f"{prefix}_recent_trade_vwap"] = float((price * size).sum() / total_size) if total_size > 0 else np.nan
        else:
            snapshot[f"{prefix}_recent_trade_vwap"] = np.nan
    return snapshot


def nearest_clock_skew_seconds(reference_ts: pd.Series | np.ndarray, other_ts: pd.Series) -> dict:
    output = {
        "clock_alignment_pair_count": 0,
        "clock_alignment_median_skew_seconds": np.nan,
        "clock_alignment_p95_abs_skew_seconds": np.nan,
    }
    def timestamp_ns(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values.astype(np.int64, copy=False)
        clean = values.dropna()
        if clean.empty:
            return np.array([], dtype=np.int64)
        if pd.api.types.is_numeric_dtype(clean):
            raise TypeError(f"{name} must contain datetimes, not numeric timestamp surrogates")
        if not pd.api.types.is_datetime64_any_dtype(clean):
            clean = pd.to_datetime(clean, errors="coerce")
            if pd.Series(clean).isna().any():
                raise TypeError(f"{name} must contain datetime-like values")
        return pd.DatetimeIndex(clean).asi8

    ref = timestamp_ns(reference_ts, "reference_ts")
    oth = timestamp_ns(other_ts, "other_ts")
    if ref.size == 0 or oth.size == 0:
        return output
    ref = np.sort(ref)
    positions = np.searchsorted(ref, oth)
    right_idx = np.clip(positions, 0, ref.size - 1)
    left_idx = np.clip(positions - 1, 0, ref.size - 1)
    right_values = ref[right_idx]
    left_values = ref[left_idx]
    choose_right = positions == 0
    choose_left = positions == ref.size
    # Boundary masks prevent first/last observations from indexing outside the
    # reference timestamp array; middle rows use nearest-neighbor comparison.
    middle = (~choose_right) & (~choose_left)
    choose_right[middle] = np.abs(right_values[middle] - oth[middle]) <= np.abs(left_values[middle] - oth[middle])
    nearest = np.where(choose_right, right_values, left_values)
    skew_seconds = (oth - nearest) / 1_000_000_000
    output["clock_alignment_pair_count"] = int(skew_seconds.size)
    output["clock_alignment_median_skew_seconds"] = float(np.median(skew_seconds))
    output["clock_alignment_p95_abs_skew_seconds"] = float(np.quantile(np.abs(skew_seconds), 0.95))
    return output


def dataset_required_columns(dataset: str) -> tuple[str, ...]:
    if dataset.startswith("stock_quotes_"):
        return ("timestamp", "bid", "ask")
    if dataset.startswith("stock_trades_"):
        return ("timestamp", "price", "size")
    if dataset.startswith("option_quotes_"):
        return ("timestamp", "bid", "ask")
    if dataset.startswith("option_trades_"):
        return ("timestamp", "price", "size")
    if dataset == "option_open_interest":
        return ("timestamp", "open_interest")
    return tuple()


def normalize_strike_for_key(strike: float | None) -> str | None:
    if strike is None:
        return None
    text = f"{strike:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_strike(strike: float) -> str:
    return normalize_strike_for_key(strike) or "0"


def session_key(symbol: str, trade_day: pd.Timestamp) -> str:
    return f"symbol={symbol}__date={pd.Timestamp(trade_day).strftime('%Y-%m-%d')}"


def parquet_chunk_stem(symbol: str, trade_day: pd.Timestamp) -> str:
    return session_key(symbol, trade_day)


def parquet_chunk_path(base_dir: Path, dataset: str, symbol: str, trade_day: pd.Timestamp, suffix: str = "") -> Path:
    stem = parquet_chunk_stem(symbol, trade_day)
    if suffix:
        stem = f"{stem}__{suffix}"
    return base_dir / dataset / f"{stem}.parquet"


def write_parquet_chunk(base_dir: Path, dataset: str, symbol: str, trade_day: pd.Timestamp, rows: list[dict], suffix: str = "") -> Path:
    path = parquet_chunk_path(base_dir, dataset, symbol, trade_day, suffix=suffix)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    try:
        pd.DataFrame(rows).to_parquet(tmp_path, index=False, compression="zstd")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path


class SymbolDayChunkWriter:
    def __init__(self, symbol: str, trade_day: pd.Timestamp):
        self.symbol = symbol
        self.trade_day = trade_day
        self.paths = chunk_paths()
        self.contract_rows: list[dict] = []
        self.quality_rows: list[dict] = []
        self.screening_rows: list[dict] = []
        self.contract_part = 0
        self.quality_part = 0
        self.screening_part = 0

    def _flush_contracts(self) -> None:
        if not self.contract_rows:
            return
        self.contract_part += 1
        write_parquet_chunk(
            self.paths["contracts"],
            "contracts",
            self.symbol,
            self.trade_day,
            self.contract_rows,
            suffix=f"part={self.contract_part:05d}",
        )
        self.contract_rows = []

    def _flush_quality(self) -> None:
        if not self.quality_rows:
            return
        self.quality_part += 1
        write_parquet_chunk(
            self.paths["quality"],
            "quality",
            self.symbol,
            self.trade_day,
            self.quality_rows,
            suffix=f"part={self.quality_part:05d}",
        )
        self.quality_rows = []

    def append_contract_row(self, row: dict) -> None:
        self.contract_rows.append(row)
        if len(self.contract_rows) >= CFG.stream_flush_row_count:
            self._flush_contracts()

    def extend_quality_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.quality_rows.extend(rows)
        if len(self.quality_rows) >= CFG.stream_flush_row_count:
            self._flush_quality()

    def _flush_screening(self) -> None:
        if not self.screening_rows:
            return
        self.screening_part += 1
        write_parquet_chunk(
            self.paths["screening"],
            "screening",
            self.symbol,
            self.trade_day,
            self.screening_rows,
            suffix=f"part={self.screening_part:05d}",
        )
        self.screening_rows = []

    def append_screening_row(self, row: dict) -> None:
        self.screening_rows.append(row)
        if len(self.screening_rows) >= CFG.stream_flush_row_count:
            self._flush_screening()

    def finalize(self) -> None:
        self._flush_contracts()
        self._flush_quality()
        self._flush_screening()


# ============================================================
# DATA ACCESS
# ============================================================
def get_csv(path: str, params: dict) -> tuple[pd.DataFrame, dict, bytes]:
    adaptive_request_delay(path)
    try:
        with request_slot():
            started = time.perf_counter()
            response = get_request_session().get(f"{CFG.base_url}{path}", params=params, timeout=120)
    except Exception:
        record_request_result(path, success=False)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = response.content
    meta = {
        "request_url": response.url,
        "status_code": response.status_code,
        "elapsed_ms": elapsed_ms,
        "response_headers": dict(response.headers),
        "payload_sha256": sha256_bytes(payload),
        "payload_bytes": len(payload),
    }
    if response.status_code != 200:
        record_request_result(path, success=False)
        raise RuntimeError(
            f"Request failed: {response.url}\nstatus={response.status_code}\ntext={response.text[:1000]}"
        )
    record_request_result(path, success=True)
    return pd.read_csv(BytesIO(payload)), meta, payload


def raw_cache_path(dataset: str, symbol: str, day: pd.Timestamp, expiration: str | None = None, strike: float | None = None, right: str | None = None) -> Path:
    base = CFG.raw_cache_dir / dataset / f"symbol={symbol}" / f"date={day.strftime('%Y-%m-%d')}"
    if expiration is not None:
        base = base / f"expiration={expiration}"
    if strike is not None:
        base = base / f"strike={normalize_strike_for_key(strike)}"
    if right is not None:
        base = base / f"right={right.lower()}"
    return base / "data.parquet"


def raw_cache_meta_path(dataset: str, symbol: str, day: pd.Timestamp, expiration: str | None = None, strike: float | None = None, right: str | None = None) -> Path:
    return raw_cache_path(dataset, symbol, day, expiration=expiration, strike=strike, right=right).with_name("meta.json")


def raw_cache_payload_path(dataset: str, symbol: str, day: pd.Timestamp, expiration: str | None = None, strike: float | None = None, right: str | None = None) -> Path:
    return raw_cache_path(dataset, symbol, day, expiration=expiration, strike=strike, right=right).with_name("raw_response.csv")


def raw_cache_meta(dataset: str, symbol: str, day: pd.Timestamp, expiration: str | None = None, strike: float | None = None, right: str | None = None) -> dict:
    meta_path = raw_cache_meta_path(dataset, symbol, day, expiration=expiration, strike=strike, right=right)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def small_cache_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def small_cache_is_usable(path: Path, meta_path: Path, endpoint: str, params: dict, required_columns: tuple[str, ...] = ()) -> bool:
    if not path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {"endpoint", "params", "cache_data_sha256"}
    if not required.issubset(meta):
        return False
    if meta["endpoint"] != endpoint or meta["params"] != params:
        return False
    # Raw vendor cache validity is tied to the request surface and data hash,
    # not to collector comments or non-request configuration fields.
    try:
        stat = path.stat()
        if meta.get("cache_data_size_bytes") == stat.st_size and meta.get("cache_data_mtime_ns") == stat.st_mtime_ns:
            pass
        elif meta["cache_data_sha256"] != sha256_bytes(path.read_bytes()):
            return False
        if required_columns:
            df = pd.read_csv(path)
            return all(col in df.columns for col in required_columns)
        return True
    except Exception:
        return False


def cache_is_usable(data_path: Path, meta_path: Path, dataset: str, path: str, params: dict) -> bool:
    if not data_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {"dataset", "endpoint", "params", "cache_status", "cache_data_sha256"}
    if not required.issubset(meta):
        return False
    if meta["dataset"] != dataset or meta["endpoint"] != path or meta["params"] != params:
        return False
    if meta.get("cache_status") != "ok":
        return False
    # Reuse is safe only when the stored parquet still matches its recorded
    # hash/size metadata.
    try:
        stat = data_path.stat()
        if meta.get("cache_data_size_bytes") == stat.st_size and meta.get("cache_data_mtime_ns") == stat.st_mtime_ns:
            return True
        return meta["cache_data_sha256"] == sha256_bytes(data_path.read_bytes())
    except Exception:
        return False


def finalize_market_frame(
    df: pd.DataFrame,
    *,
    derived_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError(f"Response missing timestamp column: {list(df.columns)}")
    result = optimize_frame_dtypes(attach_timestamp_metadata(df))
    result, filtered_condition_row_count = filter_condition_rows(result, "market")
    missing_condition_row_count = int(result.attrs.get("missing_condition_row_count", 0))
    if {"bid", "ask"}.issubset(result.columns):
        if "stock_mid" in derived_columns:
            result["stock_mid"] = (result["bid"] + result["ask"]) / 2.0
        if "option_mid" in derived_columns:
            result["option_mid"] = (result["bid"] + result["ask"]) / 2.0
        if "stock_spread" in derived_columns:
            result["stock_spread"] = result["ask"] - result["bid"]
        if "option_spread" in derived_columns:
            result["option_spread"] = result["ask"] - result["bid"]
        if "rel_spread" in derived_columns:
            if "option_mid" not in result.columns:
                result["option_mid"] = (result["bid"] + result["ask"]) / 2.0
            if "option_spread" not in result.columns:
                result["option_spread"] = result["ask"] - result["bid"]
            result["rel_spread"] = result["option_spread"] / result["option_mid"].replace(0, np.nan)
    result.attrs["filtered_condition_row_count"] = filtered_condition_row_count
    result.attrs["missing_condition_row_count"] = missing_condition_row_count
    return result.sort_values("timestamp", na_position="last").reset_index(drop=True)


def fetch_market_frame(
    dataset: str,
    symbol: str,
    day: pd.Timestamp,
    path: str,
    params: dict,
    *,
    expiration: str | None = None,
    strike: float | None = None,
    right: str | None = None,
    derived_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    df = cached_request_frame(
        dataset=dataset,
        symbol=symbol,
        day=day,
        path=path,
        params=params,
        expiration=expiration,
        strike=strike,
        right=right,
    )
    return finalize_market_frame(df, derived_columns=derived_columns)


def cached_request_frame(
    dataset: str,
    symbol: str,
    day: pd.Timestamp,
    path: str,
    params: dict,
    expiration: str | None = None,
    strike: float | None = None,
    right: str | None = None,
) -> pd.DataFrame:
    cache_path = raw_cache_path(dataset, symbol, day, expiration=expiration, strike=strike, right=right)
    meta_path = raw_cache_meta_path(dataset, symbol, day, expiration=expiration, strike=strike, right=right)
    payload_path = raw_cache_payload_path(dataset, symbol, day, expiration=expiration, strike=strike, right=right)
    lock = path_lock(cache_path)
    with lock:
        with file_lock(cache_path):
            if cache_is_usable(cache_path, meta_path, dataset, path, params):
                return read_csv_cache(cache_path)

            df, response_meta, payload = get_csv(path, params)
            ensure_dir(cache_path.parent)
            content_type = str(response_meta["response_headers"].get("Content-Type", "")).lower()
            validation_issues: list[str] = []
            if not any(token in content_type for token in ("csv", "text/plain", "application/octet-stream", "")):
                validation_issues.append(f"unexpected_content_type:{content_type}")
            missing_columns = [col for col in dataset_required_columns(dataset) if col not in df.columns]
            if missing_columns:
                validation_issues.append(f"missing_columns:{','.join(missing_columns)}")
            if "timestamp" in df.columns:
                if pd.to_datetime(df["timestamp"], errors="coerce").isna().all() and not df.empty:
                    validation_issues.append("all_timestamps_unparseable")
            if df.empty:
                validation_issues.append("empty_pull")
            cache_status = "ok" if not validation_issues else "suspect"
            cached_df = optimize_frame_dtypes(attach_timestamp_metadata(df, include_raw=True))
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False, suffix=".parquet") as tmp:
                tmp_path = Path(tmp.name)
            try:
                cached_df.to_parquet(tmp_path, index=False, compression="zstd")
                os.replace(tmp_path, cache_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            cache_stat = cache_path.stat()
            if CFG.store_raw_payloads:
                atomic_write_bytes(payload_path, payload)
            meta = {
                "vendor": "ThetaData",
                "dataset": dataset,
                "endpoint": path,
                "params": params,
                "fetched_at_utc": pd.Timestamp.utcnow().isoformat(),
                "columns": list(df.columns),
                "row_count": int(len(df)),
                "column_signature": "|".join(df.columns.astype(str)),
                "empty_pull": bool(df.empty),
                "raw_duplicate_row_count": int(df.duplicated().sum()),
                "raw_null_timestamp_count": int(df["timestamp"].isna().sum()) if "timestamp" in df.columns else 0,
                "cache_status": cache_status,
                "validation_issues": validation_issues,
                "cache_data_sha256": sha256_bytes(cache_path.read_bytes()),
                "cache_data_size_bytes": cache_stat.st_size,
                "cache_data_mtime_ns": cache_stat.st_mtime_ns,
                **response_meta,
                "config_digest": run_context()["config_digest"],
                "code_digest": run_context()["code_digest"],
            }
            write_json(meta_path, meta)
            return read_csv_cache(cache_path)

@lru_cache(maxsize=None)
def get_expirations(symbol: str) -> tuple[pd.Timestamp, ...]:
    ensure_dir(CHAIN_METADATA_DIR)
    cache_path = CHAIN_METADATA_DIR / f"{symbol}_expirations.csv"
    meta_path = small_cache_meta_path(cache_path)
    with path_lock(cache_path):
        with file_lock(cache_path):
            params = {"symbol": symbol}
            if small_cache_is_usable(cache_path, meta_path, "/option/list/expirations", params, ("expiration",)):
                df = pd.read_csv(cache_path)
            else:
                df, _, _ = get_csv("/option/list/expirations", params)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                atomic_write_bytes(cache_path, csv_bytes)
                cache_stat = cache_path.stat()
                write_json(
                    meta_path,
                    {
                        "endpoint": "/option/list/expirations",
                        "params": params,
                        "cache_data_sha256": sha256_bytes(csv_bytes),
                        "cache_data_size_bytes": cache_stat.st_size,
                        "cache_data_mtime_ns": cache_stat.st_mtime_ns,
                        "config_digest": run_context()["config_digest"],
                        "code_digest": run_context()["code_digest"],
                    },
                )
    if "expiration" not in df.columns:
        raise KeyError(f"Expirations response missing expiration column: {list(df.columns)}")
    expirations = pd.to_datetime(df["expiration"], errors="coerce").dropna().dt.normalize().sort_values().drop_duplicates().tolist()
    return tuple(pd.Timestamp(exp) for exp in expirations)


@lru_cache(maxsize=None)
def get_strikes(symbol: str, expiration_str: str) -> tuple[float, ...]:
    ensure_dir(CHAIN_METADATA_DIR / f"symbol={symbol}")
    cache_path = CHAIN_METADATA_DIR / f"symbol={symbol}" / f"expiration={expiration_str}_strikes.csv"
    meta_path = small_cache_meta_path(cache_path)
    with path_lock(cache_path):
        with file_lock(cache_path):
            params = {"symbol": symbol, "expiration": expiration_str}
            if small_cache_is_usable(cache_path, meta_path, "/option/list/strikes", params):
                df = pd.read_csv(cache_path)
            else:
                df, _, _ = get_csv("/option/list/strikes", params)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                atomic_write_bytes(cache_path, csv_bytes)
                cache_stat = cache_path.stat()
                write_json(
                    meta_path,
                    {
                        "endpoint": "/option/list/strikes",
                        "params": params,
                        "cache_data_sha256": sha256_bytes(csv_bytes),
                        "cache_data_size_bytes": cache_stat.st_size,
                        "cache_data_mtime_ns": cache_stat.st_mtime_ns,
                        "config_digest": run_context()["config_digest"],
                        "code_digest": run_context()["code_digest"],
                    },
                )
    strike_col = "strike" if "strike" in df.columns else df.columns[-1]
    strikes = pd.to_numeric(df[strike_col], errors="coerce").dropna().sort_values().drop_duplicates().tolist()
    return tuple(float(strike) for strike in strikes)


def get_stock_quotes(symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    return fetch_market_frame(
        dataset=f"stock_quotes_{CFG.quote_interval}",
        symbol=symbol,
        day=day,
        path="/stock/history/quote",
        params={"symbol": symbol, "date": day.strftime("%Y%m%d"), "interval": CFG.quote_interval},
        derived_columns=("stock_mid", "stock_spread"),
    )


def get_stock_trades(symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    return fetch_market_frame(
        dataset=f"stock_trades_{CFG.trade_interval}",
        symbol=symbol,
        day=day,
        path="/stock/history/trade",
        params={"symbol": symbol, "date": day.strftime("%Y%m%d"), "interval": CFG.trade_interval},
    )


def get_option_quotes(symbol: str, expiration: pd.Timestamp, strike: float, right: str, day: pd.Timestamp) -> pd.DataFrame:
    expiration_str = expiration.strftime("%Y-%m-%d")
    return fetch_market_frame(
        dataset=f"option_quotes_{CFG.quote_interval}",
        symbol=symbol,
        day=day,
        path="/option/history/quote",
        params={
            "symbol": symbol,
            "expiration": expiration_str,
            "strike": format_strike(strike),
            "right": right.lower(),
            "date": day.strftime("%Y%m%d"),
            "interval": CFG.quote_interval,
        },
        expiration=expiration_str,
        strike=strike,
        right=right,
        derived_columns=("option_mid", "option_spread", "rel_spread"),
    )


def get_option_trades(symbol: str, expiration: pd.Timestamp, strike: float, right: str, day: pd.Timestamp) -> pd.DataFrame:
    expiration_str = expiration.strftime("%Y-%m-%d")
    return fetch_market_frame(
        dataset=f"option_trades_{CFG.trade_interval}",
        symbol=symbol,
        day=day,
        path="/option/history/trade",
        params={
            "symbol": symbol,
            "expiration": expiration_str,
            "strike": format_strike(strike),
            "right": right.lower(),
            "date": day.strftime("%Y%m%d"),
            "interval": CFG.trade_interval,
        },
        expiration=expiration_str,
        strike=strike,
        right=right,
    )


def get_option_open_interest(symbol: str, expiration: pd.Timestamp, strike: float, right: str, day: pd.Timestamp) -> pd.DataFrame:
    expiration_str = expiration.strftime("%Y-%m-%d")
    return fetch_market_frame(
        dataset="option_open_interest",
        symbol=symbol,
        day=day,
        path="/option/history/open_interest",
        params={
            "symbol": symbol,
            "expiration": expiration_str,
            "strike": format_strike(strike),
            "right": right.lower(),
            "date": day.strftime("%Y%m%d"),
        },
        expiration=expiration_str,
        strike=strike,
        right=right,
    )


# ============================================================
# NEUTRAL SURFACE COLLECTOR
# ============================================================
def candidate_anchor_dates() -> pd.DatetimeIndex:
    sessions = XNYS.sessions_in_range(CFG.start_date, CFG.end_date).tz_localize(None)
    freq = str(CFG.anchor_freq).strip().upper()
    if freq in {"", "B", "SESSION", "SESSIONS", "XNYS"}:
        return sessions
    requested = pd.date_range(CFG.start_date, CFG.end_date, freq=CFG.anchor_freq).normalize()
    requested_index = pd.Index(pd.Timestamp(ts).normalize() for ts in requested)
    return sessions[sessions.isin(requested_index)]


@lru_cache(maxsize=None)
def market_session_bounds(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    session = pd.Timestamp(day).normalize()
    open_ts = XNYS.session_open(session).tz_convert(CFG.exchange_tz)
    close_ts = XNYS.session_close(session).tz_convert(CFG.exchange_tz)
    return open_ts, close_ts


@lru_cache(maxsize=None)
def session_metadata(day: pd.Timestamp) -> dict:
    session = pd.Timestamp(day).normalize()
    open_ts, close_ts = market_session_bounds(session)
    return {
        "session_open": open_ts,
        "session_close": close_ts,
        "session_length_minutes": (close_ts - open_ts).total_seconds() / 60.0,
        "is_early_close": (close_ts - open_ts).total_seconds() < (6.5 * 60 * 60),
    }


def expected_interval_seconds(interval: str) -> float | None:
    mapping = {"1s": 1.0, "1m": 60.0, "5m": 300.0, "15m": 900.0, "1h": 3600.0}
    return mapping.get(interval)


@lru_cache(maxsize=None)
def previous_trade_day(trade_day: pd.Timestamp) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(XNYS.previous_session(pd.Timestamp(trade_day).normalize())).normalize()
    except Exception:
        return None


def eligible_expirations(trade_day: pd.Timestamp, expirations: tuple[pd.Timestamp, ...]) -> list[pd.Timestamp]:
    eligible = []
    trade_day_norm = pd.Timestamp(trade_day).normalize()
    for expiration in expirations:
        dte = int((pd.Timestamp(expiration).normalize() - trade_day_norm).days)
        if CFG.min_dte <= dte <= CFG.max_dte:
            eligible.append(pd.Timestamp(expiration))
    if not eligible:
        return []
    selected: list[pd.Timestamp] = []
    remaining = list(eligible)
    for target_dte in CFG.target_dtes:
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda exp: (
                abs(int((pd.Timestamp(exp).normalize() - trade_day_norm).days) - int(target_dte)),
                int((pd.Timestamp(exp).normalize() - trade_day_norm).days),
            ),
        )
        if best not in selected:
            selected.append(best)
        remaining = [exp for exp in remaining if exp != best]
        if len(selected) >= CFG.max_expirations_per_day:
            break
    if len(selected) < CFG.max_expirations_per_day:
        # This makes max_expirations_per_day a true ceiling: fill to the cap
        # whenever enough eligible expirations exist, otherwise return fewer.
        leftovers = sorted(
            [exp for exp in eligible if exp not in selected],
            key=lambda exp: min(abs(int((pd.Timestamp(exp).normalize() - trade_day_norm).days) - int(target)) for target in CFG.target_dtes),
        )
        selected.extend(leftovers[: max(CFG.max_expirations_per_day - len(selected), 0)])
    return sorted(selected)


def stock_selection_reference_mids(stock_quotes: pd.DataFrame, trade_day: pd.Timestamp) -> dict[str, float]:
    reference_mids: dict[str, float] = {}
    for label, entry_ts, _ in evaluation_schedule(trade_day):
        snapshot = market_snapshot(
            stock_quotes,
            entry_ts,
            max_age_seconds=CFG.max_stock_quote_age_seconds,
            prefix=f"stock_reference_{label}",
        )
        mid = pd.to_numeric(pd.Series([snapshot.get(f"stock_reference_{label}_stock_mid", np.nan)]), errors="coerce").iloc[0]
        if pd.notna(mid) and float(mid) > 0:
            reference_mids[label] = float(mid)
    return reference_mids


def selected_strikes(strikes: tuple[float, ...], reference_mids: dict[str, float]) -> list[float]:
    if not strikes or not reference_mids:
        return []
    chosen: list[float] = []
    for reference_mid in reference_mids.values():
        for target_moneyness in CFG.moneyness_targets:
            target_strike = float(reference_mid / target_moneyness)
            ranked = sorted(strikes, key=lambda strike: (abs(float(strike) - target_strike), float(strike)))
            for strike in ranked[: CFG.strikes_per_moneyness_target]:
                numeric = float(strike)
                if numeric not in chosen:
                    chosen.append(numeric)
    return sorted(chosen)


def occ_contract_id(symbol: str, expiration: pd.Timestamp, right: str, strike: float) -> str:
    if len(symbol) > 6:
        raise ValueError(f"OCC root requires an explicit vendor root for symbols longer than six characters: {symbol}")
    root = symbol.ljust(6)[:6]
    strike_mils = int((Decimal(str(strike)) * Decimal("1000")).to_integral_value(rounding=ROUND_HALF_UP))
    return f"{root}{expiration.strftime('%y%m%d')}{right[:1].upper()}{strike_mils:08d}"


def count_sync_missing(stock_ts_ns: np.ndarray, option_ts: pd.Series, tolerance_seconds: int = 5) -> float:
    if len(stock_ts_ns) == 0 or option_ts.empty:
        return np.nan
    option_ns = pd.DatetimeIndex(option_ts.dropna()).asi8
    if option_ns.size == 0:
        return np.nan
    positions = np.searchsorted(stock_ts_ns, option_ns)
    tolerance_ns = tolerance_seconds * 1_000_000_000
    nearest = np.full(option_ns.shape, np.iinfo(np.int64).max, dtype=np.int64)
    valid_right = positions < len(stock_ts_ns)
    if valid_right.any():
        nearest[valid_right] = np.minimum(nearest[valid_right], np.abs(stock_ts_ns[positions[valid_right]] - option_ns[valid_right]))
    valid_left = positions > 0
    if valid_left.any():
        left_idx = positions[valid_left] - 1
        nearest[valid_left] = np.minimum(nearest[valid_left], np.abs(stock_ts_ns[left_idx] - option_ns[valid_left]))
    return int((nearest > tolerance_ns).sum())


def summarize_market_frame(
    df: pd.DataFrame,
    dataset: str,
    trade_day: pd.Timestamp,
    interval: str | None = None,
) -> dict:
    df = optimize_frame_dtypes(df)
    dtype_signature = "|".join(f"{col}:{dtype}" for col, dtype in df.dtypes.astype(str).items()) if not df.empty else ""
    summary = {
        "dataset": dataset,
        "row_count": int(len(df)),
        "column_signature": "|".join(df.columns.astype(str)),
        "dtype_signature": dtype_signature,
        "empty_pull": bool(df.empty),
        "filtered_condition_row_count": int(df.attrs.get("filtered_condition_row_count", 0)),
        "missing_condition_row_count": int(df.attrs.get("missing_condition_row_count", 0)),
        "null_timestamp_count": 0,
        "timestamp_parse_failed_count": 0,
        "timestamp_naive_assumed_count": 0,
        "duplicate_timestamp_count": 0,
        "duplicate_timestamp_conflict_count": 0,
        "out_of_order_timestamp_count": 0,
        "missing_interval_count": 0,
        "max_gap_seconds": np.nan,
        "p99_gap_seconds": np.nan,
        "interval_alignment_miss_count": 0,
        "first_timestamp": pd.NaT,
        "last_timestamp": pd.NaT,
        "session_start_gap_seconds": np.nan,
        "session_end_gap_seconds": np.nan,
        "in_session_row_count": 0,
        "outside_session_row_count": 0,
        "session_coverage_ratio": np.nan,
        "partial_bid_ask_rows": 0,
        "missing_bid_only_rows": 0,
        "missing_ask_only_rows": 0,
        "crossed_market_count": 0,
        "locked_market_count": 0,
        "negative_size_count": 0,
        "negative_price_count": 0,
        "zero_size_count": 0,
        "zero_price_count": 0,
        "null_bid_count": 0,
        "null_ask_count": 0,
        "null_price_count": 0,
        "null_size_count": 0,
        "absurd_price_count": 0,
        "absurd_size_count": 0,
        "median_rel_spread": np.nan,
        "p95_rel_spread": np.nan,
        "median_spread": np.nan,
    }
    if df.empty or "timestamp" not in df.columns:
        return summary

    ts = df["timestamp"]
    summary["null_timestamp_count"] = int(ts.isna().sum())
    if "timestamp_parse_failed" in df.columns:
        summary["timestamp_parse_failed_count"] = int(pd.to_numeric(df["timestamp_parse_failed"], errors="coerce").fillna(0).astype(bool).sum())
    if "timestamp_naive_assumed" in df.columns:
        summary["timestamp_naive_assumed_count"] = int(pd.to_numeric(df["timestamp_naive_assumed"], errors="coerce").fillna(0).astype(bool).sum())
    clean = df.loc[ts.notna()]
    if clean.empty:
        return summary

    deltas = clean["timestamp"].diff().dt.total_seconds()
    summary["duplicate_timestamp_count"] = int(clean["timestamp"].duplicated().sum())
    if summary["duplicate_timestamp_count"] > 0:
        non_ts_cols = [col for col in clean.columns if col != "timestamp"]
        if non_ts_cols:
            duplicate_rows = clean[clean["timestamp"].duplicated(keep=False)]
            if not duplicate_rows.empty:
                distinct_per_timestamp = duplicate_rows.groupby("timestamp", dropna=False)[non_ts_cols].apply(
                    lambda frame: len(frame.drop_duplicates())
                )
                summary["duplicate_timestamp_conflict_count"] = int((distinct_per_timestamp > 1).sum())
    summary["out_of_order_timestamp_count"] = int((deltas < 0).fillna(False).sum())
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    deltas = clean["timestamp"].diff().dt.total_seconds()
    summary["first_timestamp"] = clean["timestamp"].iloc[0]
    summary["last_timestamp"] = clean["timestamp"].iloc[-1]

    if dataset != "option_open_interest":
        session_open, session_close = market_session_bounds(trade_day)
        in_session = clean[(clean["timestamp"] >= session_open) & (clean["timestamp"] <= session_close)]
        summary["in_session_row_count"] = int(len(in_session))
        summary["outside_session_row_count"] = int(len(clean) - len(in_session))
        metric_frame = in_session
        if not metric_frame.empty:
            summary["session_start_gap_seconds"] = (metric_frame["timestamp"].iloc[0] - session_open).total_seconds()
            summary["session_end_gap_seconds"] = (session_close - metric_frame["timestamp"].iloc[-1]).total_seconds()
            session_seconds = max((session_close - session_open).total_seconds(), 1.0)
            expected = expected_interval_seconds(interval or "")
            if expected is not None:
                total_expected_intervals = max(int(np.floor(session_seconds / expected)) + 1, 1)
                covered_offsets = (
                    ((metric_frame["timestamp"] - session_open).dt.total_seconds().clip(lower=0.0, upper=session_seconds) // expected)
                    .dropna()
                    .astype(int)
                )
                summary["session_coverage_ratio"] = min(float(covered_offsets.nunique() / total_expected_intervals), 1.0)
            else:
                active_seconds = max((metric_frame["timestamp"].iloc[-1] - metric_frame["timestamp"].iloc[0]).total_seconds(), 0.0)
                summary["session_coverage_ratio"] = active_seconds / session_seconds
        metric_deltas = metric_frame["timestamp"].diff().dt.total_seconds() if len(metric_frame) > 1 else pd.Series(dtype=float)
        if len(metric_frame) > 1:
            summary["max_gap_seconds"] = float(metric_deltas.max())
            summary["p99_gap_seconds"] = float(metric_deltas.quantile(0.99))
            expected = expected_interval_seconds(interval or "")
            if expected is not None:
                summary["missing_interval_count"] = int((metric_deltas > (expected * 1.5)).fillna(False).sum())
                summary["interval_alignment_miss_count"] = int(
                    ((metric_frame["timestamp"] - session_open).dt.total_seconds().round() % expected != 0).sum()
                )

    if {"bid", "ask"}.issubset(clean.columns):
        bid = clean["bid"]
        ask = clean["ask"]
        summary["null_bid_count"] = int(bid.isna().sum())
        summary["null_ask_count"] = int(ask.isna().sum())
        summary["partial_bid_ask_rows"] = int((bid.isna() ^ ask.isna()).sum())
        summary["missing_bid_only_rows"] = int((bid.isna() & ask.notna()).sum())
        summary["missing_ask_only_rows"] = int((ask.isna() & bid.notna()).sum())
        summary["crossed_market_count"] = int(((ask < bid) & bid.notna() & ask.notna()).sum())
        summary["locked_market_count"] = int(((ask == bid) & bid.notna() & ask.notna()).sum())
        spread = ask - bid
        mid = (ask + bid) / 2.0
        rel_spread = spread / mid.replace(0, np.nan)
        summary["median_spread"] = float(spread.dropna().median()) if spread.notna().any() else np.nan
        summary["median_rel_spread"] = float(rel_spread.dropna().median()) if rel_spread.notna().any() else np.nan
        summary["p95_rel_spread"] = float(rel_spread.dropna().quantile(0.95)) if rel_spread.notna().any() else np.nan

    if "price" in clean.columns:
        price = clean["price"]
        summary["null_price_count"] = int(price.isna().sum())
        summary["negative_price_count"] = int((price < 0).fillna(False).sum())
        summary["zero_price_count"] = int((price == 0).fillna(False).sum())
        valid_price = price.dropna()
        if not valid_price.empty:
            relative_ceiling = float(valid_price.quantile(0.999) * 10.0)
            absolute_ceiling = 50_000.0
            ceiling = min(relative_ceiling, absolute_ceiling) if relative_ceiling > 0 else absolute_ceiling
            summary["absurd_price_count"] = int((price > ceiling).fillna(False).sum())

    if "size" in clean.columns:
        size = clean["size"]
        summary["null_size_count"] = int(size.isna().sum())
        summary["negative_size_count"] = int((size < 0).fillna(False).sum())
        summary["zero_size_count"] = int((size == 0).fillna(False).sum())
        summary["absurd_size_count"] = int((size > size.dropna().quantile(0.999) * 10).fillna(False).sum()) if size.notna().any() else 0

    return summary


def describe_stock_day(stock_quotes: pd.DataFrame, trade_day: pd.Timestamp) -> dict:
    summary = summarize_market_frame(stock_quotes, "stock_quotes", trade_day, CFG.quote_interval)
    valid = stock_quotes[
        stock_quotes.get("bid", pd.Series(dtype=float)).gt(0)
        & stock_quotes.get("ask", pd.Series(dtype=float)).gt(0)
        & stock_quotes.get("ask", pd.Series(dtype=float)).ge(stock_quotes.get("bid", pd.Series(dtype=float)))
        & stock_quotes.get("stock_mid", pd.Series(dtype=float)).gt(0)
    ].copy()
    if valid.empty:
        summary["stock_open_mid"] = np.nan
        summary["stock_close_mid"] = np.nan
        summary["stock_median_spread"] = np.nan
        return summary

    mids = pd.to_numeric(valid["stock_mid"], errors="coerce")
    spreads = pd.to_numeric(valid["stock_spread"], errors="coerce")
    rel_spread = spreads / mids.replace(0, np.nan)
    summary["stock_open_mid"] = float(mids.dropna().iloc[0]) if mids.notna().any() else np.nan
    summary["stock_close_mid"] = float(mids.dropna().iloc[-1]) if mids.notna().any() else np.nan
    summary["stock_median_spread"] = float(spreads.dropna().median()) if spreads.notna().any() else np.nan
    summary["stock_median_rel_spread"] = float(rel_spread.dropna().median()) if rel_spread.notna().any() else np.nan
    return summary


def collect_contract_day(
    symbol_cfg: SymbolConfig,
    trade_day: pd.Timestamp,
    expiration: pd.Timestamp,
    strike: float,
    right: str,
    prev_trade_day: pd.Timestamp | None,
    stock_timestamp_ns: np.ndarray,
    stock_day: dict,
    stock_context: dict,
    chain_context: dict,
) -> tuple[dict, list[dict]]:
    contract_key = occ_contract_id(symbol_cfg.symbol, expiration, right, strike)
    oi_error = ""
    if prev_trade_day is not None:
        try:
            option_oi = get_option_open_interest(symbol_cfg.symbol, expiration, strike, right, prev_trade_day)
            oi_diag = summarize_market_frame(option_oi, "option_open_interest", prev_trade_day)
        except Exception as exc:
            option_oi = pd.DataFrame()
            oi_diag = summarize_market_frame(option_oi, "option_open_interest", prev_trade_day)
            oi_error = repr(exc)
    else:
        option_oi = pd.DataFrame()
        oi_diag = summarize_market_frame(option_oi, "option_open_interest", trade_day)
        oi_error = "no_previous_trade_day"
    quality_rows = [
        {
            "symbol": symbol_cfg.symbol,
            "trade_day": trade_day,
            "expiration": expiration,
            "strike": strike,
            "right": right.upper(),
            "contract_id": contract_key,
            "open_interest_trade_day": prev_trade_day,
            **oi_diag,
            "request_error": oi_error,
        }
    ]

    quote_error = ""
    try:
        option_quotes = get_option_quotes(symbol_cfg.symbol, expiration, strike, right, trade_day)
        quote_diag = summarize_market_frame(option_quotes, "option_quotes", trade_day, CFG.quote_interval)
    except Exception as exc:
        option_quotes = pd.DataFrame()
        quote_diag = summarize_market_frame(option_quotes, "option_quotes", trade_day, CFG.quote_interval)
        quote_error = repr(exc)

    trade_error = ""
    try:
        option_trades = get_option_trades(symbol_cfg.symbol, expiration, strike, right, trade_day)
        trade_diag = summarize_market_frame(option_trades, "option_trades", trade_day, CFG.trade_interval)
    except Exception as exc:
        option_trades = pd.DataFrame()
        trade_diag = summarize_market_frame(option_trades, "option_trades", trade_day, CFG.trade_interval)
        trade_error = repr(exc)

    quality_rows.extend(
        [
            {
                "symbol": symbol_cfg.symbol,
                "trade_day": trade_day,
                "expiration": expiration,
                "strike": strike,
                "right": right.upper(),
                "contract_id": contract_key,
                **quote_diag,
                "request_error": quote_error,
            },
            {
                "symbol": symbol_cfg.symbol,
                "trade_day": trade_day,
                "expiration": expiration,
                "strike": strike,
                "right": right.upper(),
                "contract_id": contract_key,
                **trade_diag,
                "request_error": trade_error,
            },
        ]
    )

    open_interest = float(option_oi["open_interest"].dropna().iloc[-1]) if (not option_oi.empty and "open_interest" in option_oi.columns and option_oi["open_interest"].notna().any()) else np.nan
    quote_mid = option_quotes["option_mid"] if "option_mid" in option_quotes.columns else pd.Series(dtype=float)
    quote_spread = option_quotes["option_spread"] if "option_spread" in option_quotes.columns else pd.Series(dtype=float)
    trade_price = option_trades["price"] if "price" in option_trades.columns else pd.Series(dtype=float)
    trade_size = option_trades["size"] if "size" in option_trades.columns else pd.Series(dtype=float)
    total_trade_size = float(trade_size.fillna(0).sum()) if not trade_size.empty else 0.0
    vwap = float((trade_price * trade_size).sum() / total_trade_size) if total_trade_size > 0 else np.nan
    stock_open_mid = stock_day.get("stock_open_mid", np.nan)
    primary_reference_mid = to_float(chain_context.get("primary_reference_mid", np.nan))
    moneyness = primary_reference_mid / strike if strike > 0 and not pd.isna(primary_reference_mid) else np.nan
    option_ts = option_quotes["timestamp"].dropna() if "timestamp" in option_quotes.columns else pd.Series(dtype="datetime64[ns, America/New_York]")
    sync_missing = count_sync_missing(stock_timestamp_ns, option_ts)
    clock_alignment = nearest_clock_skew_seconds(stock_context["stock_quote_timestamps_ns"], option_ts)
    option_quote_trade_alignment = nearest_clock_skew_seconds(
        option_quotes["timestamp"] if "timestamp" in option_quotes.columns else pd.Series(dtype="datetime64[ns, America/New_York]"),
        option_trades["timestamp"] if "timestamp" in option_trades.columns else pd.Series(dtype="datetime64[ns, America/New_York]"),
    )
    observed_quote_quality_pass_count = int(
        sum(
            [
                len(option_quotes) > 10,
                quote_diag["crossed_market_count"] == 0,
                quote_diag["partial_bid_ask_rows"] == 0,
                pd.notna(quote_diag["median_rel_spread"]) and quote_diag["median_rel_spread"] < 1.0,
                quote_diag["outside_session_row_count"] == 0,
                quote_diag["timestamp_naive_assumed_count"] == 0,
            ]
        )
    )
    observed_trade_quality_pass_count = int(
        sum(
            [
                len(option_trades) > 5,
                total_trade_size >= 10,
                trade_diag["outside_session_row_count"] == 0,
                trade_diag["timestamp_naive_assumed_count"] == 0,
                trade_diag["missing_interval_count"] == 0,
            ]
        )
    )
    option_quote_meta = raw_cache_meta(
        f"option_quotes_{CFG.quote_interval}",
        symbol_cfg.symbol,
        trade_day,
        expiration=expiration.strftime("%Y-%m-%d"),
        strike=strike,
        right=right,
    )
    option_trade_meta = raw_cache_meta(
        f"option_trades_{CFG.trade_interval}",
        symbol_cfg.symbol,
        trade_day,
        expiration=expiration.strftime("%Y-%m-%d"),
        strike=strike,
        right=right,
    )
    option_oi_meta = raw_cache_meta(
        "option_open_interest",
        symbol_cfg.symbol,
        prev_trade_day if prev_trade_day is not None else trade_day,
        expiration=expiration.strftime("%Y-%m-%d"),
        strike=strike,
        right=right,
    )

    corporate_actions = stock_context.get("corporate_actions", pd.DataFrame())
    # Corporate-action flags are annotations, not hard exclusions. The modeling
    # layer decides whether to drop dividend/split-sensitive rows.
    if isinstance(corporate_actions, pd.DataFrame) and not corporate_actions.empty and "date" in corporate_actions.columns:
        action_dates = pd.to_datetime(corporate_actions["date"], errors="coerce").dt.normalize()
        dividends = pd.to_numeric(corporate_actions.get("dividends", pd.Series(index=corporate_actions.index, dtype=float)), errors="coerce").fillna(0.0)
        splits = pd.to_numeric(corporate_actions.get("stock_splits", pd.Series(index=corporate_actions.index, dtype=float)), errors="coerce").fillna(0.0)
        dividend_dates = action_dates.loc[dividends > 0].dropna()
        split_dates = action_dates.loc[splits > 0].dropna()
    else:
        dividend_dates = pd.Series(dtype="datetime64[ns]")
        split_dates = pd.Series(dtype="datetime64[ns]")
    trade_date = pd.Timestamp(trade_day).normalize()
    expiration_date = pd.Timestamp(expiration).normalize()
    split_in_contract_horizon = bool(((split_dates >= trade_date) & (split_dates <= expiration_date)).any()) if not split_dates.empty else False
    split_prior_to_trade_day = bool((split_dates < trade_date).any()) if not split_dates.empty else False

    evaluation_fields: dict[str, object] = {}
    dividend_in_any_eval_horizon = False
    for label, eval_context in stock_context["evaluation_contexts"].items():
        entry_ts = eval_context["entry_ts"]
        exit_ts = eval_context["exit_ts"]
        # Per-evaluation moneyness depends on get_stock_quotes deriving
        # stock_mid; if that input is unavailable, the value is intentionally NaN.
        underlying_reference_mid = to_float(eval_context["entry_quote"].get(f"stock_eval_{label}_entry_quote_stock_mid", np.nan))
        entry_date = pd.Timestamp(entry_ts.date())
        exit_date = pd.Timestamp(exit_ts.date())
        dividend_in_eval_horizon = bool(((dividend_dates >= entry_date) & (dividend_dates <= exit_date)).any()) if not dividend_dates.empty else False
        dividend_in_any_eval_horizon = dividend_in_any_eval_horizon or dividend_in_eval_horizon
        evaluation_fields[f"eval_{label}_underlying_reference_mid"] = underlying_reference_mid
        evaluation_fields[f"eval_{label}_moneyness"] = underlying_reference_mid / strike if strike > 0 and pd.notna(underlying_reference_mid) else np.nan
        evaluation_fields[f"eval_{label}_dividend_in_horizon_flag"] = dividend_in_eval_horizon
        evaluation_fields.update(market_snapshot(option_quotes, entry_ts, max_age_seconds=CFG.max_option_quote_age_seconds, prefix=f"eval_{label}_entry_quote"))
        evaluation_fields.update(market_snapshot(option_quotes, exit_ts, max_age_seconds=CFG.max_option_quote_age_seconds, prefix=f"eval_{label}_exit_quote"))
        evaluation_fields.update(market_snapshot(option_trades, entry_ts, trade_lookback_minutes=CFG.recent_trade_lookback_minutes, prefix=f"eval_{label}_entry_trade"))
        evaluation_fields.update(market_snapshot(option_trades, exit_ts, trade_lookback_minutes=CFG.recent_trade_lookback_minutes, prefix=f"eval_{label}_exit_trade"))
        evaluation_fields.update(window_update_features(option_quotes, entry_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"eval_{label}_entry_quote_window", value_column="option_mid"))
        evaluation_fields.update(window_update_features(option_quotes, exit_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"eval_{label}_exit_quote_window", value_column="option_mid"))
        evaluation_fields.update(window_update_features(option_trades, entry_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"eval_{label}_entry_trade_window", value_column="price"))
        evaluation_fields.update(window_update_features(option_trades, exit_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"eval_{label}_exit_trade_window", value_column="price"))
        for key in (
            "metadata",
            "entry_quote",
            "exit_quote",
            "entry_trade",
            "exit_trade",
            "entry_quote_window",
            "exit_quote_window",
            "entry_trade_window",
            "exit_trade_window",
        ):
            evaluation_fields.update(eval_context[key])

    contract_row = {
        "symbol": symbol_cfg.symbol,
        "asset_type": symbol_cfg.asset_type,
        "universe_bucket": symbol_cfg.universe_bucket,
        "sector_proxy": symbol_cfg.sector_proxy,
        "trade_day": trade_day,
        "expiration": expiration,
        "right": right.upper(),
        "strike": strike,
        "contract_id": contract_key,
        "dte_days": int((expiration.normalize() - trade_day.normalize()).days),
        "stock_open_mid": stock_open_mid,
        "primary_reference_mid": primary_reference_mid,
        "moneyness_open": moneyness,
        "quote_row_count": len(option_quotes),
        "trade_row_count": len(option_trades),
        "open_interest_row_count": len(option_oi),
        "quote_first_timestamp": quote_diag["first_timestamp"],
        "quote_last_timestamp": quote_diag["last_timestamp"],
        "trade_first_timestamp": trade_diag["first_timestamp"],
        "trade_last_timestamp": trade_diag["last_timestamp"],
        "median_option_mid": float(quote_mid.dropna().median()) if not quote_mid.dropna().empty else np.nan,
        "median_option_spread": float(quote_spread.dropna().median()) if not quote_spread.dropna().empty else np.nan,
        "trade_vwap": vwap,
        "total_trade_size": total_trade_size,
        "lagged_open_interest": open_interest,
        "option_to_stock_sync_missing_count": sync_missing,
        "quote_missing_interval_count": quote_diag["missing_interval_count"],
        "trade_missing_interval_count": trade_diag["missing_interval_count"],
        "quote_duplicate_timestamp_count": quote_diag["duplicate_timestamp_count"],
        "trade_duplicate_timestamp_count": trade_diag["duplicate_timestamp_count"],
        "quote_out_of_order_timestamp_count": quote_diag["out_of_order_timestamp_count"],
        "trade_out_of_order_timestamp_count": trade_diag["out_of_order_timestamp_count"],
        "crossed_market_count": quote_diag["crossed_market_count"],
        "locked_market_count": quote_diag["locked_market_count"],
        "partial_bid_ask_rows": quote_diag["partial_bid_ask_rows"],
        "quote_column_signature": quote_diag["column_signature"],
        "quote_dtype_signature": quote_diag["dtype_signature"],
        "trade_column_signature": trade_diag["column_signature"],
        "trade_dtype_signature": trade_diag["dtype_signature"],
        "oi_column_signature": oi_diag["column_signature"],
        "oi_dtype_signature": oi_diag["dtype_signature"],
        "selection_status": "observed",
        "screening_reasons": "",
        "observed_quote_quality_pass_count": observed_quote_quality_pass_count,
        "observed_trade_quality_pass_count": observed_trade_quality_pass_count,
        "observed_quote_median_rel_spread": quote_diag["median_rel_spread"],
        "observed_quote_p95_rel_spread": quote_diag["p95_rel_spread"],
        "open_interest_trade_day": prev_trade_day,
        "evaluation_times": "|".join(stock_context["evaluation_contexts"].keys()),
        "stock_quote_vendor_clock_verified": False,
        "option_quote_vendor_clock_verified": False,
        "quote_nbbo_provenance_verified": False,
        "contract_identity_adjustment_verified": not split_in_contract_horizon,
        "contract_identity_split_adjusted_flag": split_in_contract_horizon,
        "stock_split_prior_to_trade_day_flag": split_prior_to_trade_day,
        "stock_split_in_contract_horizon_flag": split_in_contract_horizon,
        "dividend_in_any_eval_horizon_flag": dividend_in_any_eval_horizon,
        **stock_context["carry_context"],
        **clock_alignment,
        "option_quote_trade_clock_alignment_pair_count": option_quote_trade_alignment["clock_alignment_pair_count"],
        "option_quote_trade_clock_alignment_median_skew_seconds": option_quote_trade_alignment["clock_alignment_median_skew_seconds"],
        "option_quote_trade_clock_alignment_p95_abs_skew_seconds": option_quote_trade_alignment["clock_alignment_p95_abs_skew_seconds"],
        "total_strike_count_for_expiration": chain_context["total_strike_count_for_expiration"],
        "total_listed_contract_count_for_expiration": chain_context["total_listed_contract_count_for_expiration"],
        "sampled_strike_count_for_expiration": chain_context["sampled_strike_count_for_expiration"],
        "sampled_contract_count_for_expiration": chain_context["sampled_contract_count_for_expiration"],
        "sampled_contract_fraction_for_expiration": chain_context["sampled_contract_fraction_for_expiration"],
        "strike_rank_in_expiration": chain_context["strike_rank_in_expiration"],
        "strike_percentile_in_expiration": chain_context["strike_percentile_in_expiration"],
        "quote_payload_sha256": option_quote_meta.get("payload_sha256", ""),
        "trade_payload_sha256": option_trade_meta.get("payload_sha256", ""),
        "oi_payload_sha256": option_oi_meta.get("payload_sha256", ""),
        "quote_cache_data_sha256": option_quote_meta.get("cache_data_sha256", ""),
        "trade_cache_data_sha256": option_trade_meta.get("cache_data_sha256", ""),
        "oi_cache_data_sha256": option_oi_meta.get("cache_data_sha256", ""),
    }
    contract_row.update(evaluation_fields)
    contract_row.update(build_sample_flags(contract_row))
    return contract_row, quality_rows


def collect_symbol_day(
    symbol_cfg: SymbolConfig,
    trade_day: pd.Timestamp,
    chunk_writer: SymbolDayChunkWriter | None = None,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    try:
        stock_quotes = get_stock_quotes(symbol_cfg.symbol, trade_day)
        stock_error = ""
    except Exception as exc:
        stock_quotes = pd.DataFrame()
        stock_error = repr(exc)
    try:
        stock_trades = get_stock_trades(symbol_cfg.symbol, trade_day)
        stock_trade_error = ""
    except Exception as exc:
        stock_trades = pd.DataFrame()
        stock_trade_error = repr(exc)

    stock_day = describe_stock_day(stock_quotes, trade_day)
    carry_context = carry_snapshot(symbol_cfg.symbol, trade_day)
    stock_available = not pd.isna(stock_day.get("stock_open_mid", np.nan))
    prev_trade = previous_trade_day(trade_day) if stock_available else None
    stock_clock_alignment = nearest_clock_skew_seconds(
        stock_quotes["timestamp"] if "timestamp" in stock_quotes.columns else pd.Series(dtype="datetime64[ns, America/New_York]"),
        stock_trades["timestamp"] if "timestamp" in stock_trades.columns else pd.Series(dtype="datetime64[ns, America/New_York]"),
    )
    session_row = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "symbol": symbol_cfg.symbol,
        "asset_type": symbol_cfg.asset_type,
        "universe_bucket": symbol_cfg.universe_bucket,
        "sector_proxy": symbol_cfg.sector_proxy,
        "requested_day": trade_day,
        "trade_day": trade_day,
        "stock_request_error": stock_error,
        "stock_trade_request_error": stock_trade_error,
        "stock_data_available": stock_available,
        "stock_trade_data_available": bool(not stock_trades.empty),
        "previous_trade_day_for_oi": prev_trade,
        **session_metadata(trade_day),
        **carry_context,
        "stock_quote_trade_clock_alignment_pair_count": stock_clock_alignment["clock_alignment_pair_count"],
        "stock_quote_trade_clock_alignment_median_skew_seconds": stock_clock_alignment["clock_alignment_median_skew_seconds"],
        "stock_quote_trade_clock_alignment_p95_abs_skew_seconds": stock_clock_alignment["clock_alignment_p95_abs_skew_seconds"],
        **stock_day,
    }
    quality_rows = [
        {
            "symbol": symbol_cfg.symbol,
            "trade_day": trade_day,
            "expiration": pd.NaT,
            "strike": np.nan,
            "right": "",
            "contract_id": "",
            **summarize_market_frame(stock_quotes, "stock_quotes", trade_day, CFG.quote_interval),
            "request_error": stock_error,
        },
        {
            "symbol": symbol_cfg.symbol,
            "trade_day": trade_day,
            "expiration": pd.NaT,
            "strike": np.nan,
            "right": "",
            "contract_id": "",
            **summarize_market_frame(stock_trades, "stock_trades", trade_day, CFG.trade_interval),
            "request_error": stock_trade_error,
        },
    ]
    screening_rows: list[dict] = []
    if not stock_available:
        return session_row, [], [], quality_rows, screening_rows

    all_expirations = get_expirations(symbol_cfg.symbol)
    in_window_expirations = [
        expiration for expiration in all_expirations
        if CFG.min_dte <= int((pd.Timestamp(expiration).normalize() - pd.Timestamp(trade_day).normalize()).days) <= CFG.max_dte
    ]
    expirations = eligible_expirations(trade_day, all_expirations)
    excluded_expiration_count = len(all_expirations) - len(in_window_expirations)
    expiration_rows: list[dict] = []
    contract_rows: list[dict] = []
    collected_contract_count = 0
    contract_task_failure_count = 0
    stock_timestamp_ns = pd.DatetimeIndex(stock_quotes["timestamp"].dropna()).asi8 if "timestamp" in stock_quotes.columns else np.array([], dtype=np.int64)
    expected_contract_count = 0
    total_listed_contract_count_eligible = 0
    total_sampled_contract_targets = 0
    evaluation_contexts: dict[str, dict] = {}
    for label, entry_ts, exit_ts in evaluation_schedule(trade_day):
        configured_exit_ts = entry_ts + pd.Timedelta(minutes=CFG.horizon_minutes)
        evaluation_contexts[label] = {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "metadata": {
                # Late-day evaluations can have a shorter realized horizon when
                # the configured exit would exceed the exchange close.
                f"eval_{label}_entry_timestamp": entry_ts,
                f"eval_{label}_configured_exit_timestamp": configured_exit_ts,
                f"eval_{label}_actual_exit_timestamp": exit_ts,
                f"eval_{label}_configured_horizon_minutes": CFG.horizon_minutes,
                f"eval_{label}_actual_horizon_minutes": float((exit_ts - entry_ts).total_seconds() / 60.0),
                f"eval_{label}_exit_clipped_to_session_close": bool(exit_ts < configured_exit_ts),
            },
            "entry_quote": market_snapshot(stock_quotes, entry_ts, max_age_seconds=CFG.max_stock_quote_age_seconds, prefix=f"stock_eval_{label}_entry_quote"),
            "exit_quote": market_snapshot(stock_quotes, exit_ts, max_age_seconds=CFG.max_stock_quote_age_seconds, prefix=f"stock_eval_{label}_exit_quote"),
            "entry_trade": market_snapshot(stock_trades, entry_ts, trade_lookback_minutes=CFG.recent_trade_lookback_minutes, prefix=f"stock_eval_{label}_entry_trade"),
            "exit_trade": market_snapshot(stock_trades, exit_ts, trade_lookback_minutes=CFG.recent_trade_lookback_minutes, prefix=f"stock_eval_{label}_exit_trade"),
            "entry_quote_window": window_update_features(stock_quotes, entry_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"stock_eval_{label}_entry_quote_window", value_column="stock_mid"),
            "exit_quote_window": window_update_features(stock_quotes, exit_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"stock_eval_{label}_exit_quote_window", value_column="stock_mid"),
            "entry_trade_window": window_update_features(stock_trades, entry_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"stock_eval_{label}_entry_trade_window", value_column="price"),
            "exit_trade_window": window_update_features(stock_trades, exit_ts, window_minutes=CFG.recent_trade_lookback_minutes, prefix=f"stock_eval_{label}_exit_trade_window", value_column="price"),
        }
    for eval_context in evaluation_contexts.values():
        for key in ("metadata", "entry_quote", "exit_quote", "entry_trade", "exit_trade", "entry_quote_window", "exit_quote_window", "entry_trade_window", "exit_trade_window"):
            session_row.update(eval_context[key])
    selection_reference_mids = stock_selection_reference_mids(stock_quotes, trade_day)
    stock_context = {
        # Any field needed by contract rows must be placed here before workers
        # are submitted; contract tasks receive this immutable snapshot.
        "evaluation_contexts": evaluation_contexts,
        "stock_quote_timestamps": stock_quotes["timestamp"] if "timestamp" in stock_quotes.columns else pd.Series(dtype="datetime64[ns, America/New_York]"),
        "stock_quote_timestamps_ns": stock_timestamp_ns,
        "carry_context": carry_context,
        "corporate_actions": get_corporate_actions(symbol_cfg.symbol),
    }
    skipped_expirations = [exp for exp in in_window_expirations if exp not in expirations]
    for skipped_expiration in skipped_expirations:
        skipped_strikes = get_strikes(symbol_cfg.symbol, skipped_expiration.strftime("%Y-%m-%d"))
        screening_rows.append(
            {
                "symbol": symbol_cfg.symbol,
                "trade_day": trade_day,
                "expiration": skipped_expiration,
                "strike": np.nan,
                "right": "",
                "contract_id": "",
                "selection_status": "unsampled_expiration",
                "unsampled_reason": "not_targeted_expiration_bucket",
                "dte_days": int((skipped_expiration - trade_day).days),
                "total_strike_count_for_expiration": len(skipped_strikes),
                "total_listed_contract_count_for_expiration": len(skipped_strikes) * len(CFG.option_rights),
                "sampled_strike_count_for_expiration": 0,
                "sampled_contract_count_for_expiration": 0,
            }
        )
    clean_contract_counts = {label: 0 for label in evaluation_contexts}
    broad_contract_counts = {label: 0 for label in evaluation_contexts}
    with ThreadPoolExecutor(max_workers=CFG.max_contract_workers) as executor:
        contract_futures = {}
        for expiration in expirations:
            strikes = get_strikes(symbol_cfg.symbol, expiration.strftime("%Y-%m-%d"))
            total_listed_contract_count_eligible += len(strikes) * len(CFG.option_rights)
            chosen_strikes = selected_strikes(strikes, selection_reference_mids)
            total_sampled_contract_targets += len(chosen_strikes) * len(CFG.option_rights)
            unsampled_strikes = [float(value) for value in strikes if float(value) not in set(chosen_strikes)]
            expiration_rows.append(
                {
                    "symbol": symbol_cfg.symbol,
                    "trade_day": trade_day,
                    "expiration": expiration,
                    "dte_days": int((expiration - trade_day).days),
                    "strike_count": len(strikes),
                    "selected_strike_count": len(chosen_strikes),
                    "listed_contract_count": len(strikes) * len(CFG.option_rights),
                    "observed_contract_target_count": len(chosen_strikes) * len(CFG.option_rights),
                    "sampled_contract_fraction_of_listed": ((len(chosen_strikes) * len(CFG.option_rights)) / max(len(strikes) * len(CFG.option_rights), 1)),
                    "rights_collected": ",".join(opt.upper() for opt in CFG.option_rights),
                    "excluded_expiration_count_outside_window": excluded_expiration_count,
                }
            )
            strike_rank_map = {float(value): idx + 1 for idx, value in enumerate(sorted(float(v) for v in strikes))}
            strike_count = max(len(strikes), 1)
            for unsampled_strike in unsampled_strikes:
                screening_rows.append(
                    {
                        "symbol": symbol_cfg.symbol,
                        "trade_day": trade_day,
                        "expiration": expiration,
                        "strike": unsampled_strike,
                        "right": "",
                        "contract_id": "",
                        "selection_status": "unsampled_strike",
                        "unsampled_reason": "not_targeted_moneyness_grid",
                        "dte_days": int((expiration - trade_day).days),
                        "total_strike_count_for_expiration": len(strikes),
                        "total_listed_contract_count_for_expiration": len(strikes) * len(CFG.option_rights),
                        "sampled_strike_count_for_expiration": len(chosen_strikes),
                        "sampled_contract_count_for_expiration": len(chosen_strikes) * len(CFG.option_rights),
                        "strike_rank_in_expiration": strike_rank_map.get(float(unsampled_strike), np.nan),
                        "strike_percentile_in_expiration": (strike_rank_map.get(float(unsampled_strike), 0) - 1) / max(strike_count - 1, 1),
                    }
                )
            for strike in chosen_strikes:
                for right in CFG.option_rights:
                    expected_contract_count += 1
                    chain_context = {
                        "primary_reference_mid": next(iter(selection_reference_mids.values()), np.nan),
                        "total_strike_count_for_expiration": len(strikes),
                        "total_listed_contract_count_for_expiration": len(strikes) * len(CFG.option_rights),
                        "sampled_strike_count_for_expiration": len(chosen_strikes),
                        "sampled_contract_count_for_expiration": len(chosen_strikes) * len(CFG.option_rights),
                        "sampled_contract_fraction_for_expiration": (len(chosen_strikes) * len(CFG.option_rights)) / max(len(strikes) * len(CFG.option_rights), 1),
                        "strike_rank_in_expiration": strike_rank_map.get(float(strike), np.nan),
                        "strike_percentile_in_expiration": (strike_rank_map.get(float(strike), 0) - 1) / max(strike_count - 1, 1),
                    }
                    future = executor.submit(
                        collect_contract_day,
                        symbol_cfg,
                        trade_day,
                        expiration,
                        strike,
                        right,
                        prev_trade,
                        stock_timestamp_ns,
                        stock_day,
                        stock_context,
                        chain_context,
                    )
                    # Keep contract identity beside each future so failures are
                    # logged against the correct expiration/strike/right.
                    contract_futures[future] = (expiration, strike, right)
        for future in as_completed(contract_futures):
            expiration, strike, right = contract_futures[future]
            try:
                contract_row, contract_quality_rows = future.result()
            except Exception as exc:
                contract_task_failure_count += 1
                failure_row = {
                    "symbol": symbol_cfg.symbol,
                    "trade_day": trade_day,
                    "expiration": expiration,
                    "strike": strike,
                    "right": str(right).upper(),
                    "contract_id": "",
                    "dataset": "contract_collection",
                    "row_count": 0,
                    "empty_pull": True,
                    "request_error": repr(exc),
                }
                if chunk_writer is not None:
                    chunk_writer.extend_quality_rows([failure_row])
                else:
                    quality_rows.append(failure_row)
                continue
            for label in evaluation_contexts:
                clean_contract_counts[label] += int(bool(contract_row.get(f"clean_sample_included_{label}", False)))
                broad_contract_counts[label] += int(bool(contract_row.get(f"broad_sample_included_{label}", False)))
            if chunk_writer is not None:
                collected_contract_count += 1
                chunk_writer.append_contract_row(contract_row)
                chunk_writer.extend_quality_rows(contract_quality_rows)
            else:
                collected_contract_count += 1
                contract_rows.append(contract_row)
                quality_rows.extend(contract_quality_rows)

    if contract_rows:
        contract_rows.sort(key=lambda row: (row["expiration"], row["strike"], row["right"]))
    if screening_rows:
        screening_rows.sort(key=lambda row: (row["expiration"], row["strike"], row["right"]))

    session_row["eligible_expiration_count"] = len(expirations)
    session_row["in_window_expiration_count"] = len(in_window_expirations)
    session_row["excluded_expiration_count_outside_window"] = excluded_expiration_count
    session_row["skipped_in_window_expiration_count"] = len(skipped_expirations)
    session_row["stock_selection_reference_times"] = "|".join(selection_reference_mids.keys())
    session_row["stock_selection_reference_mid_count"] = len(selection_reference_mids)
    session_row["expected_contract_count"] = expected_contract_count
    session_row["collected_contract_count"] = collected_contract_count
    session_row["screened_out_contract_count"] = 0
    session_row["unsampled_chain_context_count"] = len(screening_rows)
    session_row["total_listed_contract_count_eligible"] = total_listed_contract_count_eligible
    session_row["sampled_contract_target_count"] = total_sampled_contract_targets
    session_row["sampled_contract_fraction_eligible"] = total_sampled_contract_targets / max(total_listed_contract_count_eligible, 1)
    session_row["contract_task_failure_count"] = contract_task_failure_count
    session_row["contract_task_failure_rate"] = contract_task_failure_count / max(expected_contract_count, 1)
    for label in evaluation_contexts:
        session_row[f"clean_contract_count_{label}"] = clean_contract_counts[label]
        session_row[f"broad_contract_count_{label}"] = broad_contract_counts[label]
    return session_row, expiration_rows, contract_rows, quality_rows, screening_rows


def download_market_history(ticker: str, cache_name: str, value_name: str) -> pd.DataFrame:
    ensure_dir(CFG.market_data_dir)
    cache_path = CFG.market_data_dir / cache_name
    with path_lock(cache_path):
        with file_lock(cache_path):
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                if "date" in cached.columns:
                    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                return cached
            history = yf.download(ticker, start=CFG.start_date, end=(pd.Timestamp(CFG.end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
            if history is None or history.empty:
                frame = pd.DataFrame(columns=["date", value_name, "source"])
            else:
                frame = history.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": value_name})
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
                frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
                frame["source"] = f"yfinance_{ticker}_close"
            atomic_write_bytes(cache_path, frame.to_csv(index=False).encode("utf-8"))
        return frame


@lru_cache(maxsize=1)
def get_risk_free_rate_history() -> pd.DataFrame:
    rate_13w = download_market_history("^IRX", "risk_free_13w_history.csv", "risk_free_rate_13w")
    rate_5y = download_market_history("^FVX", "risk_free_5y_history.csv", "risk_free_rate_5y")
    rate_10y = download_market_history("^TNX", "risk_free_10y_history.csv", "risk_free_rate_10y")
    merged = rate_13w[["date", "risk_free_rate_13w"]].copy() if not rate_13w.empty else pd.DataFrame(columns=["date", "risk_free_rate_13w"])
    for frame, column in ((rate_5y, "risk_free_rate_5y"), (rate_10y, "risk_free_rate_10y")):
        part = frame[["date", column]].copy() if not frame.empty else pd.DataFrame(columns=["date", column])
        merged = merged.merge(part, on="date", how="outer") if not merged.empty else part
    if merged.empty:
        return pd.DataFrame(columns=["date", "risk_free_rate_13w", "risk_free_rate_5y", "risk_free_rate_10y", "risk_free_rate_short_term_proxy", "rate_source"])
    for column in ("risk_free_rate_13w", "risk_free_rate_5y", "risk_free_rate_10y"):
        # yfinance Treasury index closes are usually quoted in percent points
        # (for example 5.25, not 0.0525). If they arrive as decimals, keep them.
        raw_rate = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = np.where(raw_rate.abs() > 1.0, raw_rate / 100.0, raw_rate)
    merged["risk_free_rate_short_term_proxy"] = merged["risk_free_rate_13w"]
    merged["rate_source"] = "yfinance_curve_proxies_^IRX_^FVX_^TNX"
    return merged.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def get_vix_history() -> pd.DataFrame:
    frame = download_market_history("^VIX", "vix_history.csv", "vix_close")
    if frame.empty:
        return pd.DataFrame(columns=["date", "vix_close", "vix_regime"])
    frame["vix_regime"] = np.where(
        frame["vix_close"] >= 25.0,
        "high_vol",
        np.where(frame["vix_close"] <= 15.0, "low_vol", "medium_vol"),
    )
    return frame[["date", "vix_close", "vix_regime"]]


@lru_cache(maxsize=None)
def get_underlying_price_history(symbol: str) -> pd.DataFrame:
    ensure_dir(CFG.market_data_dir)
    cache_path = CFG.market_data_dir / f"{symbol}_daily_history.csv"
    with path_lock(cache_path):
        with file_lock(cache_path):
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                if "date" in cached.columns:
                    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                return cached
            history = yf.download(symbol, start=(pd.Timestamp(CFG.start_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d"), end=(pd.Timestamp(CFG.end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
            if history is None or history.empty:
                frame = pd.DataFrame(columns=["date", "close", "symbol"])
            else:
                frame = history.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
                frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
                frame["symbol"] = symbol
            atomic_write_bytes(cache_path, frame.to_csv(index=False).encode("utf-8"))
        return frame


@lru_cache(maxsize=None)
def build_dividend_yield_proxy(symbol: str) -> pd.DataFrame:
    prices = get_underlying_price_history(symbol)
    actions = get_corporate_actions(symbol)
    if prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "dividend_yield_proxy"])
    frame = prices[["date", "symbol", "close"]].copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    dividends = actions[["date", "dividends"]].copy() if not actions.empty and "dividends" in actions.columns else pd.DataFrame(columns=["date", "dividends"])
    dividends["date"] = pd.to_datetime(dividends["date"], errors="coerce")
    dividends["dividends"] = pd.to_numeric(dividends.get("dividends", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    if dividends.empty or dividends["date"].dropna().empty:
        frame["trailing_365d_dividends"] = 0.0
    else:
        daily = (
            dividends.dropna(subset=["date"])
            .groupby("date", as_index=True)["dividends"]
            .sum()
            .sort_index()
        )
        calendar = pd.date_range(min(frame["date"].min(), daily.index.min()), frame["date"].max(), freq="D")
        trailing = daily.reindex(calendar, fill_value=0.0).rolling("365D").sum().rename("trailing_365d_dividends")
        frame = pd.merge_asof(
            frame.sort_values("date"),
            trailing.reset_index().rename(columns={"index": "date"}).sort_values("date"),
            on="date",
            direction="backward",
        )
        frame["trailing_365d_dividends"] = frame["trailing_365d_dividends"].fillna(0.0)
    frame["dividend_yield_proxy"] = frame["trailing_365d_dividends"] / frame["close"].replace(0, np.nan)
    return frame[["date", "symbol", "dividend_yield_proxy"]]


@lru_cache(maxsize=None)
def build_regime_labels(symbol: str) -> pd.DataFrame:
    prices = get_underlying_price_history(symbol)
    vix = get_vix_history()
    if prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "symbol_daily_return_1d", "symbol_realized_vol_20d", "symbol_realized_vol_60d", "vix_close", "vix_regime", "market_regime"])
    frame = prices[["date", "symbol", "close"]].copy().sort_values("date").reset_index(drop=True)
    close_to_close_return = frame["close"].pct_change()
    realized_vol_20d = close_to_close_return.rolling(20).std() * np.sqrt(252.0)
    realized_vol_60d = close_to_close_return.rolling(60).std() * np.sqrt(252.0)
    # Shift market-state inputs so a trade day is labeled only with information
    # that would have been known before that session.
    frame["symbol_daily_return_1d"] = close_to_close_return.shift(1)
    frame["symbol_realized_vol_20d"] = realized_vol_20d.shift(1)
    frame["symbol_realized_vol_60d"] = realized_vol_60d.shift(1)
    frame = frame.merge(vix, on="date", how="left")
    if {"vix_close", "vix_regime"}.issubset(frame.columns):
        frame[["vix_close", "vix_regime"]] = frame[["vix_close", "vix_regime"]].shift(1)
    low_cut = frame["symbol_realized_vol_20d"].expanding(min_periods=20).quantile(0.33).shift(1)
    high_cut = frame["symbol_realized_vol_20d"].expanding(min_periods=20).quantile(0.67).shift(1)
    frame["market_regime"] = "medium_realized_vol"
    frame.loc[frame["symbol_realized_vol_20d"].notna() & high_cut.notna() & frame["symbol_realized_vol_20d"].ge(high_cut), "market_regime"] = "high_realized_vol"
    frame.loc[frame["symbol_realized_vol_20d"].notna() & low_cut.notna() & frame["symbol_realized_vol_20d"].le(low_cut), "market_regime"] = "low_realized_vol"
    frame.loc[pd.to_numeric(frame.get("vix_close", pd.Series(dtype=float)), errors="coerce").fillna(0.0).ge(25.0), "market_regime"] = "stress"
    return frame[["date", "symbol", "symbol_daily_return_1d", "symbol_realized_vol_20d", "symbol_realized_vol_60d", "vix_close", "vix_regime", "market_regime"]]


def merge_no_overwrite(*items: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for item in items:
        duplicates = sorted(set(merged).intersection(item))
        if duplicates:
            raise KeyError(f"Duplicate carry snapshot keys: {duplicates}")
        merged.update(item)
    return merged


def daily_asof_snapshot(
    frame: pd.DataFrame,
    trade_day: pd.Timestamp,
    defaults: dict[str, object],
    *,
    provenance_prefix: str | None = None,
    stale_after_days: int = 7,
) -> dict[str, object]:
    output = defaults.copy()
    trade_date = pd.Timestamp(trade_day)
    if trade_date.tz is not None:
        trade_date = trade_date.tz_convert(CFG.exchange_tz).tz_localize(None)
    trade_date = trade_date.normalize()
    asof_date = pd.NaT
    days_stale = np.nan
    if frame.empty or "date" not in frame.columns:
        if provenance_prefix is not None:
            output[f"{provenance_prefix}_asof_date"] = asof_date
            output[f"{provenance_prefix}_days_stale"] = days_stale
            output[f"{provenance_prefix}_is_stale"] = True
        return output
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    eligible = work.loc[work["date"].le(trade_date)].dropna(subset=["date"])
    if eligible.empty:
        if provenance_prefix is not None:
            output[f"{provenance_prefix}_asof_date"] = asof_date
            output[f"{provenance_prefix}_days_stale"] = days_stale
            output[f"{provenance_prefix}_is_stale"] = True
        return output
    row = eligible.sort_values("date").iloc[-1]
    for key in output:
        if key in row.index:
            output[key] = row[key]
    asof_date = pd.Timestamp(row["date"]).normalize()
    days_stale = int((trade_date - asof_date).days)
    if provenance_prefix is not None:
        output[f"{provenance_prefix}_asof_date"] = asof_date
        output[f"{provenance_prefix}_days_stale"] = days_stale
        output[f"{provenance_prefix}_is_stale"] = bool(days_stale > stale_after_days)
    return output


def regime_snapshot(symbol: str, trade_day: pd.Timestamp) -> dict:
    frame = build_regime_labels(symbol)
    defaults = {
        "symbol_daily_return_1d": np.nan,
        "symbol_realized_vol_20d": np.nan,
        "symbol_realized_vol_60d": np.nan,
        "vix_close": np.nan,
        "vix_regime": "",
        "market_regime": "",
    }
    snapshot = daily_asof_snapshot(frame, trade_day, defaults, provenance_prefix="regime")
    # Regime inputs currently share the same date index. Keep aliases explicit so
    # contract rows can be audited without reopening the daily input files.
    snapshot["vix_asof_date"] = snapshot["regime_asof_date"]
    snapshot["vix_days_stale"] = snapshot["regime_days_stale"]
    snapshot["vix_is_stale"] = snapshot["regime_is_stale"]
    snapshot["symbol_realized_vol_asof_date"] = snapshot["regime_asof_date"]
    snapshot["symbol_realized_vol_days_stale"] = snapshot["regime_days_stale"]
    snapshot["symbol_realized_vol_is_stale"] = snapshot["regime_is_stale"]
    return snapshot


def carry_snapshot(symbol: str, trade_day: pd.Timestamp) -> dict[str, object]:
    risk_free = daily_asof_snapshot(
        get_risk_free_rate_history(),
        trade_day,
        {
            "risk_free_rate_13w": np.nan,
            "risk_free_rate_5y": np.nan,
            "risk_free_rate_10y": np.nan,
            "risk_free_rate_short_term_proxy": np.nan,
            "rate_source": "",
        },
        provenance_prefix="risk_free_rate",
    )
    dividend = daily_asof_snapshot(
        build_dividend_yield_proxy(symbol),
        trade_day,
        {"dividend_yield_proxy": np.nan},
        provenance_prefix="dividend_yield",
    )
    return merge_no_overwrite(risk_free, dividend, regime_snapshot(symbol, trade_day))


def collect_carry_inputs(symbols: list[SymbolConfig]) -> pd.DataFrame:
    risk_free = get_risk_free_rate_history()
    frames = []
    for cfg in symbols:
        dividend_yield = build_dividend_yield_proxy(cfg.symbol)
        regime = build_regime_labels(cfg.symbol)
        merged = dividend_yield.merge(risk_free, on="date", how="left").merge(regime, on=["date", "symbol"], how="left")
        merged["asset_type"] = cfg.asset_type
        merged["universe_bucket"] = cfg.universe_bucket
        merged["sector_proxy"] = cfg.sector_proxy
        merged["borrow_rate_proxy"] = np.nan
        merged["borrow_rate_source"] = "unavailable"
        merged["borrow_constraint_risk_flag"] = bool(cfg.sector_proxy == "crypto_exposed" or "small_cap" in cfg.universe_bucket)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@lru_cache(maxsize=None)
def get_corporate_actions(symbol: str) -> pd.DataFrame:
    ensure_dir(REFERENCE_DATA_DIR)
    cache_path = REFERENCE_DATA_DIR / f"{symbol}_corporate_actions.csv"
    with path_lock(cache_path):
        with file_lock(cache_path):
            if cache_path.exists():
                df = pd.read_csv(cache_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                return df
            ticker = yf.Ticker(symbol)
            actions = ticker.actions
            if actions is None or actions.empty:
                df = pd.DataFrame(columns=["date", "dividends", "stock_splits", "symbol", "source"])
            else:
                df = actions.reset_index().rename(columns={"Date": "date", "Dividends": "dividends", "Stock Splits": "stock_splits"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
                df["symbol"] = symbol
                df["source"] = "yfinance"
            atomic_write_bytes(cache_path, df.to_csv(index=False).encode("utf-8"))
            return df


def collect_reference_actions(symbols: list[SymbolConfig]) -> pd.DataFrame:
    frames = []
    for cfg in symbols:
        df = get_corporate_actions(cfg.symbol).copy()
        if df.empty:
            df = pd.DataFrame([{"date": pd.NaT, "dividends": np.nan, "stock_splits": np.nan, "symbol": cfg.symbol, "source": "yfinance"}])
        df["asset_type"] = cfg.asset_type
        df["universe_bucket"] = cfg.universe_bucket
        df["sector_proxy"] = cfg.sector_proxy
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_validation_targets() -> pd.DataFrame:
    sessions = candidate_anchor_dates()
    if len(sessions) == 0:
        return pd.DataFrame()
    samples = {
        pd.Timestamp(sessions[0]),
        pd.Timestamp(sessions[min(len(sessions) - 1, 1)]),
        pd.Timestamp(sessions[len(sessions) // 2]),
        pd.Timestamp(sessions[-1]),
    }
    early_closes = []
    for sess in sessions:
        meta = session_metadata(pd.Timestamp(sess))
        if meta["is_early_close"]:
            early_closes.append(pd.Timestamp(sess))
        if len(early_closes) >= 3:
            break
    samples.update(early_closes)
    rows = []
    for sess in sorted(samples):
        meta = session_metadata(sess)
        rows.append(
            {
                "session": sess,
                **meta,
                "validation_purpose": "calendar_edge_check" if meta["is_early_close"] else "general_session_check",
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def output_paths() -> dict[str, Path]:
    return {
        "sessions": CANONICAL_DATA_DIR / "requested_sessions.csv",
        "expirations": CANONICAL_DATA_DIR / "chain_expirations.csv",
        "contracts": CANONICAL_DATA_DIR / "contract_universe.csv",
        "carry_inputs": MARKET_DATA_DIR / "carry_inputs.csv",
        "observed_instrument_index": CANONICAL_DATA_DIR / "observed_instrument_index.csv",
        "family_coverage": DIAGNOSTIC_DATA_DIR / "contract_family_coverage.csv",
        "quality": DIAGNOSTIC_DATA_DIR / "raw_pull_quality.csv",
        "screening": DIAGNOSTIC_DATA_DIR / "screening_decisions.csv",
        "summary": DIAGNOSTIC_DATA_DIR / "collection_summary.csv",
        "dataset_quality": DIAGNOSTIC_DATA_DIR / "dataset_quality_summary.csv",
        "schema": DIAGNOSTIC_DATA_DIR / "schema_signatures.csv",
        "symbol_quality": DIAGNOSTIC_DATA_DIR / "symbol_quality_summary.csv",
        "timestamp_validation": DIAGNOSTIC_DATA_DIR / "timestamp_validation_summary.csv",
        "validation_targets": DIAGNOSTIC_DATA_DIR / "validation_targets.csv",
        "reference_actions": REFERENCE_DATA_DIR / "corporate_actions_reference.csv",
        "failures": DIAGNOSTIC_DATA_DIR / "failure_ledger.csv",
        "progress": DIAGNOSTIC_DATA_DIR / "progress.json",
    }


@lru_cache(maxsize=1)
def chunk_paths() -> dict[str, Path]:
    return {
        "sessions": CANONICAL_PARTS_DIR,
        "expirations": CANONICAL_PARTS_DIR,
        "contracts": CANONICAL_PARTS_DIR,
        "quality": DIAGNOSTIC_PARTS_DIR,
        "screening": DIAGNOSTIC_PARTS_DIR,
    }


def completed_session_keys() -> set[tuple[str, str]]:
    dataset_dir = CANONICAL_PARTS_DIR / "sessions"
    if not dataset_dir.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    for path in dataset_dir.glob("*.parquet"):
        try:
            frame = pd.read_parquet(path)
            row = frame.iloc[0].to_dict() if not frame.empty else {}
            # Do not let old-schema chunks satisfy resume checks after a schema
            # bump; stale chunks should be archived or collected into a new run.
            if row.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
                continue
            symbol = str(row.get("symbol", ""))
            trade_day = pd.to_datetime(row.get("trade_day"), errors="coerce")
            if not symbol or pd.isna(trade_day):
                match = SESSION_KEY_PATTERN.match(path.stem)
                if not match:
                    continue
                symbol = match.group("symbol")
                trade_day = pd.Timestamp(match.group("date"))
            expected_contracts = int(row.get("expected_contract_count", 0))
            collected_contracts = int(row.get("collected_contract_count", 0))
            screened_out_contracts = int(row.get("screened_out_contract_count", 0))
            contract_failures = int(row.get("contract_task_failure_count", 0))
            reconciles = collected_contracts + screened_out_contracts + contract_failures == expected_contracts
            failure_rate = contract_failures / max(expected_contracts, 1)
            if reconciles and failure_rate <= CFG.soft_failure_rate_threshold:
                completed.add((symbol, pd.Timestamp(trade_day).strftime("%Y-%m-%d")))
        except Exception:
            continue
    return completed


def write_session_chunks(
    symbol: str,
    trade_day: pd.Timestamp,
    session_rows: list[dict],
    expiration_rows: list[dict],
    contract_rows: list[dict],
    quality_rows: list[dict],
    screening_rows: list[dict] | None = None,
) -> None:
    paths = chunk_paths()
    if session_rows:
        write_parquet_chunk(paths["sessions"], "sessions", symbol, trade_day, session_rows)
    if expiration_rows:
        write_parquet_chunk(paths["expirations"], "expirations", symbol, trade_day, expiration_rows)
    if contract_rows:
        write_parquet_chunk(paths["contracts"], "contracts", symbol, trade_day, contract_rows)
    if quality_rows:
        write_parquet_chunk(paths["quality"], "quality", symbol, trade_day, quality_rows)
    if screening_rows:
        write_parquet_chunk(paths["screening"], "screening", symbol, trade_day, screening_rows)


def iter_parquet_chunk_frames(base_dir: Path, dataset: str):
    dataset_dir = base_dir / dataset
    if not dataset_dir.exists():
        return
    for path in sorted(dataset_dir.glob("*.parquet")):
        yield path, pd.read_parquet(path)


def append_csv_frame(path: Path, frame: pd.DataFrame, state: dict[str, bool], key: str) -> None:
    if frame.empty:
        return
    ensure_dir(path.parent)
    header = not state.get(key, False)
    frame.to_csv(path, mode="w" if header else "a", header=header, index=False)
    state[key] = True


def update_progress_manifest(completed_keys: set[tuple[str, str]]) -> None:
    paths = output_paths()
    write_json(
        paths["progress"],
        {
            "completed_symbol_days": len(completed_keys),
            "last_write_utc": pd.Timestamp.utcnow().isoformat(),
            "config_digest": run_context()["config_digest"],
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
        },
    )


def assemble_outputs() -> dict:
    paths = output_paths()
    ensure_dir(REFERENCE_DATA_DIR)
    ensure_dir(CANONICAL_DATA_DIR)
    ensure_dir(DIAGNOSTIC_DATA_DIR)
    csv_state: dict[str, bool] = {}
    stats = {
        "requested_symbol_days": 0,
        "expiration_rows": 0,
        "contract_rows": 0,
        "quality_rows": 0,
        "screening_rows": 0,
        "stock_available_days": 0,
        "option_quote_rows_with_data": 0,
        "option_trade_rows_with_data": 0,
        "clean_sample_contract_rows": 0,
        "broad_sample_contract_rows": 0,
        "contracts_with_full_quote_observed_checks": 0,
        "contracts_with_full_trade_observed_checks": 0,
    }
    symbols_seen: set[str] = set()
    stock_rel_spreads: list[float] = []
    observed_index: dict[tuple, dict] = {}
    family_coverage: dict[tuple, dict] = {}
    dataset_quality: dict[str, dict] = {}
    schema_counts: dict[tuple[str, str], int] = {}
    symbol_quality: dict[tuple[str, str], dict] = {}
    timestamp_validation: dict[str, dict] = {}

    for _, frame in iter_parquet_chunk_frames(CANONICAL_PARTS_DIR, "sessions"):
        stats["requested_symbol_days"] += len(frame)
        if "symbol" in frame.columns:
            symbols_seen.update(frame["symbol"].dropna().astype(str).tolist())
        if "stock_data_available" in frame.columns:
            stats["stock_available_days"] += int(pd.to_numeric(frame["stock_data_available"], errors="coerce").fillna(0).astype(bool).sum())
        if "stock_median_rel_spread" in frame.columns:
            stock_rel_spreads.extend(pd.to_numeric(frame["stock_median_rel_spread"], errors="coerce").dropna().tolist())
        if CFG.assemble_csv_outputs:
            append_csv_frame(paths["sessions"], frame, csv_state, "sessions")

    for _, frame in iter_parquet_chunk_frames(CANONICAL_PARTS_DIR, "expirations"):
        stats["expiration_rows"] += len(frame)
        if CFG.assemble_csv_outputs:
            append_csv_frame(paths["expirations"], frame, csv_state, "expirations")

    for _, frame in iter_parquet_chunk_frames(CANONICAL_PARTS_DIR, "contracts"):
        stats["contract_rows"] += len(frame)
        quote_rows = pd.to_numeric(frame.get("quote_row_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        trade_rows = pd.to_numeric(frame.get("trade_row_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        full_quote = pd.to_numeric(frame.get("observed_quote_quality_pass_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        full_trade = pd.to_numeric(frame.get("observed_trade_quality_pass_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        clean_sample = boolean_flag_series(
            frame,
            ("clean_sample_included_any_evaluation", "clean_sample_included_first_eval", "clean_sample_included"),
        )
        broad_sample = boolean_flag_series(
            frame,
            ("broad_sample_included_any_evaluation", "broad_sample_included_first_eval", "broad_sample_included"),
        )
        stats["option_quote_rows_with_data"] += int((quote_rows > 0).sum())
        stats["option_trade_rows_with_data"] += int((trade_rows > 0).sum())
        stats["clean_sample_contract_rows"] += int(clean_sample.sum())
        stats["broad_sample_contract_rows"] += int(broad_sample.sum())
        stats["contracts_with_full_quote_observed_checks"] += int((full_quote == 6).sum())
        stats["contracts_with_full_trade_observed_checks"] += int((full_trade == 5).sum())
        if CFG.assemble_csv_outputs:
            append_csv_frame(paths["contracts"], frame, csv_state, "contracts")
        for row in frame.to_dict("records"):
            trade_day = pd.to_datetime(row.get("trade_day"), errors="coerce")
            observed_key = (
                row.get("contract_id"),
                row.get("symbol"),
                row.get("asset_type"),
                row.get("universe_bucket"),
                row.get("sector_proxy"),
                row.get("expiration"),
                row.get("right"),
                row.get("strike"),
            )
            state = observed_index.setdefault(
                observed_key,
                {
                    "contract_id": row.get("contract_id"),
                    "symbol": row.get("symbol"),
                    "asset_type": row.get("asset_type"),
                    "universe_bucket": row.get("universe_bucket"),
                    "sector_proxy": row.get("sector_proxy"),
                    "expiration": row.get("expiration"),
                    "right": row.get("right"),
                    "strike": row.get("strike"),
                    "first_seen_trade_day": trade_day,
                    "last_seen_trade_day": trade_day,
                    "observed_trade_days": set(),
                },
            )
            if pd.notna(trade_day):
                state["first_seen_trade_day"] = min(state["first_seen_trade_day"], trade_day) if pd.notna(state["first_seen_trade_day"]) else trade_day
                state["last_seen_trade_day"] = max(state["last_seen_trade_day"], trade_day) if pd.notna(state["last_seen_trade_day"]) else trade_day
                state["observed_trade_days"].add(trade_day.normalize())

            trade_month = trade_day.to_period("M").strftime("%Y-%m") if pd.notna(trade_day) else ""
            family_key = (row.get("symbol"), row.get("right"), row.get("dte_days"), trade_month)
            family_state = family_coverage.setdefault(
                family_key,
                {
                    "symbol": row.get("symbol"),
                    "right": row.get("right"),
                    "dte_days": row.get("dte_days"),
                    "trade_month": trade_month,
                    "contract_rows": 0,
                    "quote_rows_with_data": 0,
                    "trade_rows_with_data": 0,
                    "median_option_spread_values": [],
                    "median_trade_size_values": [],
                },
            )
            family_state["contract_rows"] += 1
            family_state["quote_rows_with_data"] += int(pd.to_numeric(pd.Series([row.get("quote_row_count")]), errors="coerce").fillna(0).iloc[0] > 0)
            family_state["trade_rows_with_data"] += int(pd.to_numeric(pd.Series([row.get("trade_row_count")]), errors="coerce").fillna(0).iloc[0] > 0)
            spread_value = pd.to_numeric(pd.Series([row.get("median_option_spread")]), errors="coerce").iloc[0]
            trade_size_value = pd.to_numeric(pd.Series([row.get("total_trade_size")]), errors="coerce").iloc[0]
            if pd.notna(spread_value):
                family_state["median_option_spread_values"].append(float(spread_value))
            if pd.notna(trade_size_value):
                family_state["median_trade_size_values"].append(float(trade_size_value))
            symbol = str(row.get("symbol", ""))
            validation_state = timestamp_validation.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "stock_option_skew_values": [],
                    "option_quote_trade_skew_values": [],
                    "stock_option_abs_skew_values": [],
                    "option_quote_trade_abs_skew_values": [],
                },
            )
            stock_option_skew = pd.to_numeric(pd.Series([row.get("clock_alignment_median_skew_seconds")]), errors="coerce").iloc[0]
            option_trade_skew = pd.to_numeric(pd.Series([row.get("option_quote_trade_clock_alignment_median_skew_seconds")]), errors="coerce").iloc[0]
            if pd.notna(stock_option_skew):
                validation_state["stock_option_skew_values"].append(float(stock_option_skew))
                validation_state["stock_option_abs_skew_values"].append(abs(float(stock_option_skew)))
            if pd.notna(option_trade_skew):
                validation_state["option_quote_trade_skew_values"].append(float(option_trade_skew))
                validation_state["option_quote_trade_abs_skew_values"].append(abs(float(option_trade_skew)))

    failures_written = False
    for _, frame in iter_parquet_chunk_frames(DIAGNOSTIC_PARTS_DIR, "quality"):
        stats["quality_rows"] += len(frame)
        if CFG.assemble_csv_outputs:
            append_csv_frame(paths["quality"], frame, csv_state, "quality")
        if "request_error" in frame.columns:
            failures = frame.loc[frame["request_error"].astype(str).ne("")]
            if not failures.empty:
                failures.to_csv(paths["failures"], mode="w" if not failures_written else "a", header=not failures_written, index=False)
                failures_written = True
        for row in frame.to_dict("records"):
            dataset_name = str(row.get("dataset", ""))
            state = dataset_quality.setdefault(
                dataset_name,
                {
                    "dataset": dataset_name,
                    "pulls": 0,
                    "empty_pulls": 0,
                    "filtered_condition_row_count": 0,
                    "missing_condition_row_count": 0,
                    "duplicate_timestamp_count": 0,
                    "duplicate_timestamp_conflict_count": 0,
                    "out_of_order_timestamp_count": 0,
                    "missing_interval_count": 0,
                    "crossed_market_count": 0,
                    "locked_market_count": 0,
                    "partial_bid_ask_rows": 0,
                    "zero_price_count": 0,
                    "zero_size_count": 0,
                    "unique_column_signatures": set(),
                    "unique_dtype_signatures": set(),
                },
            )
            state["pulls"] += 1
            for key in ("empty_pulls", "filtered_condition_row_count", "missing_condition_row_count", "duplicate_timestamp_count", "duplicate_timestamp_conflict_count", "out_of_order_timestamp_count", "missing_interval_count", "crossed_market_count", "locked_market_count", "partial_bid_ask_rows", "zero_price_count", "zero_size_count"):
                value = row.get(key, 0)
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
                state[key] += int(numeric)
            state["unique_column_signatures"].add(str(row.get("column_signature", "")))
            state["unique_dtype_signatures"].add(str(row.get("dtype_signature", "")))
            schema_key = (dataset_name, str(row.get("column_signature", "")))
            schema_counts[schema_key] = schema_counts.get(schema_key, 0) + 1
            symbol_key = (str(row.get("symbol", "")), dataset_name)
            symbol_state = symbol_quality.setdefault(
                symbol_key,
                {
                    "symbol": str(row.get("symbol", "")),
                    "dataset": dataset_name,
                    "pulls": 0,
                    "empty_pulls": 0,
                    "missing_interval_count": 0,
                    "crossed_market_count": 0,
                    "partial_bid_ask_rows": 0,
                },
            )
            symbol_state["pulls"] += 1
            for key in ("empty_pulls", "missing_interval_count", "crossed_market_count", "partial_bid_ask_rows"):
                numeric = pd.to_numeric(pd.Series([row.get(key, 0)]), errors="coerce").fillna(0).iloc[0]
                symbol_state[key] += int(numeric)

    if not failures_written:
        pd.DataFrame().to_csv(paths["failures"], index=False)

    screening_frames = []
    for _, frame in iter_parquet_chunk_frames(DIAGNOSTIC_PARTS_DIR, "screening"):
        stats["screening_rows"] += len(frame)
        screening_frames.append(frame)
    screening_df = pd.concat(screening_frames, ignore_index=True) if screening_frames else pd.DataFrame()

    observed_instrument_index_df = pd.DataFrame(
        [
            {
                **{k: v for k, v in row.items() if k != "observed_trade_days"},
                "observed_trade_days": len(row["observed_trade_days"]),
            }
            for row in observed_index.values()
        ]
    )
    family_coverage_df = pd.DataFrame(
        [
            {
                "symbol": row["symbol"],
                "right": row["right"],
                "dte_days": row["dte_days"],
                "trade_month": row["trade_month"],
                "contract_rows": row["contract_rows"],
                "quote_rows_with_data": row["quote_rows_with_data"],
                "trade_rows_with_data": row["trade_rows_with_data"],
                "median_option_spread": float(np.median(row["median_option_spread_values"])) if row["median_option_spread_values"] else np.nan,
                "median_trade_size": float(np.median(row["median_trade_size_values"])) if row["median_trade_size_values"] else np.nan,
            }
            for row in family_coverage.values()
        ]
    )
    summary_df = pd.DataFrame(
        [
            {
                "symbols": len(symbols_seen),
                "requested_symbol_days": stats["requested_symbol_days"],
                "stock_available_days": stats["stock_available_days"],
                "expiration_rows": stats["expiration_rows"],
                "contract_rows": stats["contract_rows"],
                "quality_rows": stats["quality_rows"],
                "screening_rows": stats["screening_rows"],
                "option_quote_rows_with_data": stats["option_quote_rows_with_data"],
                "option_trade_rows_with_data": stats["option_trade_rows_with_data"],
                "clean_sample_contract_rows": stats["clean_sample_contract_rows"],
                "broad_sample_contract_rows": stats["broad_sample_contract_rows"],
                "contracts_with_full_quote_observed_checks": stats["contracts_with_full_quote_observed_checks"],
                "contracts_with_full_trade_observed_checks": stats["contracts_with_full_trade_observed_checks"],
                "median_stock_rel_spread": float(np.median(stock_rel_spreads)) if stock_rel_spreads else np.nan,
            }
        ]
    )
    dataset_quality_df = pd.DataFrame(
        [
            {
                **{k: v for k, v in row.items() if k not in {"unique_column_signatures", "unique_dtype_signatures"}},
                "unique_column_signatures": len(row["unique_column_signatures"] - {""}),
                "unique_dtype_signatures": len(row["unique_dtype_signatures"] - {""}),
            }
            for row in dataset_quality.values()
        ]
    )
    schema_df = pd.DataFrame(
        [
            {"dataset": dataset, "column_signature": column_signature, "pull_count": count}
            for (dataset, column_signature), count in schema_counts.items()
        ]
    ).sort_values(["dataset", "pull_count"], ascending=[True, False]) if schema_counts else pd.DataFrame()
    symbol_quality_df = pd.DataFrame(symbol_quality.values())
    timestamp_validation_df = pd.DataFrame(
        [
            {
                "symbol": row["symbol"],
                "median_stock_option_skew_seconds": float(np.median(row["stock_option_skew_values"])) if row["stock_option_skew_values"] else np.nan,
                "p95_abs_stock_option_skew_seconds": float(np.quantile(row["stock_option_abs_skew_values"], 0.95)) if row["stock_option_abs_skew_values"] else np.nan,
                "median_option_quote_trade_skew_seconds": float(np.median(row["option_quote_trade_skew_values"])) if row["option_quote_trade_skew_values"] else np.nan,
                "p95_abs_option_quote_trade_skew_seconds": float(np.quantile(row["option_quote_trade_abs_skew_values"], 0.95)) if row["option_quote_trade_abs_skew_values"] else np.nan,
            }
            for row in timestamp_validation.values()
        ]
    )
    validation_targets_df = build_validation_targets()
    reference_actions_df = collect_reference_actions(UNIVERSE)
    carry_inputs_df = collect_carry_inputs(UNIVERSE)

    observed_instrument_index_df.to_csv(paths["observed_instrument_index"], index=False)
    family_coverage_df.to_csv(paths["family_coverage"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    dataset_quality_df.to_csv(paths["dataset_quality"], index=False)
    schema_df.to_csv(paths["schema"], index=False)
    symbol_quality_df.to_csv(paths["symbol_quality"], index=False)
    timestamp_validation_df.to_csv(paths["timestamp_validation"], index=False)
    screening_df.to_csv(paths["screening"], index=False)
    validation_targets_df.to_csv(paths["validation_targets"], index=False)
    reference_actions_df.to_csv(paths["reference_actions"], index=False)
    carry_inputs_df.to_csv(paths["carry_inputs"], index=False)
    update_progress_manifest(completed_session_keys())
    return {
        "summary_df": summary_df,
        "counts": stats,
    }


def load_existing_rows() -> set[tuple[str, str]]:
    paths = output_paths()
    if not paths["progress"].exists():
        return completed_session_keys()
    try:
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
    except Exception:
        return completed_session_keys()
    if progress.get("config_digest") != run_context()["config_digest"]:
        # A mismatched manifest means the already-written parquet set may not
        # correspond to the requested run; fail loudly instead of mixing states.
        raise RuntimeError(
            "Progress manifest config_digest differs from the current run. "
            "Use a new output_dir or explicitly archive/clear the existing canonical parts before reprocessing, "
            "so old-schema parquet chunks cannot mix with new output."
        )
    return completed_session_keys()


def validate_written_symbol_day(symbol: str, trade_day: pd.Timestamp, session_row: dict) -> None:
    session_path = parquet_chunk_path(chunk_paths()["sessions"], "sessions", symbol, trade_day)
    if not session_path.exists():
        raise RuntimeError(f"missing_session_chunk:{session_path}")
    contract_pattern = f"{parquet_chunk_stem(symbol, trade_day)}*.parquet"
    contract_parts = list((chunk_paths()["contracts"] / "contracts").glob(contract_pattern))
    screening_parts = list((chunk_paths()["screening"] / "screening").glob(contract_pattern))
    expected_contracts = int(session_row.get("expected_contract_count", 0))
    collected_contracts = int(session_row.get("collected_contract_count", 0))
    screened_out_contracts = int(session_row.get("screened_out_contract_count", 0))
    contract_failures = int(session_row.get("contract_task_failure_count", 0))
    if collected_contracts > 0 and not contract_parts:
        raise RuntimeError("missing_contract_chunks")
    if screened_out_contracts > 0 and not screening_parts:
        raise RuntimeError("missing_screening_chunks")
    if collected_contracts + screened_out_contracts + contract_failures != expected_contracts:
        raise RuntimeError(
            f"contract_reconciliation_failed expected={expected_contracts} collected={collected_contracts} screened={screened_out_contracts} failed={contract_failures}"
        )
    failure_rate = contract_failures / max(expected_contracts, 1)
    # Tiny transient failure rates are retained in diagnostics but do not force
    # an otherwise reconciled symbol-day to be recollected forever.
    if contract_failures > 0 and failure_rate > CFG.soft_failure_rate_threshold:
        raise RuntimeError(f"contract_task_failures:{contract_failures}; failure_rate={failure_rate:.4f}")


def process_symbol_day(symbol_cfg: SymbolConfig, trade_day: pd.Timestamp) -> tuple[tuple[str, str], dict]:
    chunk_writer = SymbolDayChunkWriter(symbol_cfg.symbol, trade_day)
    session_row, session_expirations, session_contracts, session_quality, session_screening = collect_symbol_day(
        symbol_cfg,
        trade_day,
        chunk_writer=chunk_writer,
    )
    write_session_chunks(
        symbol=symbol_cfg.symbol,
        trade_day=trade_day,
        session_rows=[session_row],
        expiration_rows=session_expirations,
        contract_rows=session_contracts,
        quality_rows=[],
        screening_rows=session_screening,
    )
    if session_quality:
        chunk_writer.extend_quality_rows(session_quality)
    chunk_writer.finalize()
    validate_written_symbol_day(symbol_cfg.symbol, trade_day, session_row)
    day_key = (symbol_cfg.symbol, pd.Timestamp(trade_day).strftime("%Y-%m-%d"))
    return day_key, {"session": session_row, "expirations": len(session_expirations), "contracts": int(session_row.get("collected_contract_count", 0))}


def main() -> None:
    ensure_dir(CFG.output_dir)
    ensure_dir(CFG.raw_cache_dir)
    ensure_dir(CFG.market_data_dir)
    ensure_dir(REFERENCE_DATA_DIR)
    ensure_dir(CANONICAL_DATA_DIR)
    ensure_dir(DIAGNOSTIC_DATA_DIR)
    ensure_dir(CHAIN_METADATA_DIR)
    ensure_dir(CANONICAL_PARTS_DIR)
    ensure_dir(DIAGNOSTIC_PARTS_DIR)
    write_json(CFG.output_dir / "run_context.json", run_context())

    anchors = candidate_anchor_dates()
    completed_symbol_days = load_existing_rows()

    total_configs = len(anchors) * len(UNIVERSE)
    processed = 0

    print("Collecting neutral multi-symbol option surface data")
    print(f"Date range    : {CFG.start_date} to {CFG.end_date}")
    print("Requested freq: exchange sessions")
    print(f"Quote interval: {CFG.quote_interval}")
    print(f"Trade interval: {CFG.trade_interval}")
    print(f"Symbols       : {', '.join(cfg.symbol for cfg in UNIVERSE)}")
    print(f"Resume state  : {len(completed_symbol_days)} symbol-days already saved")

    for trade_day in anchors:
        pending: list[tuple[SymbolConfig, pd.Timestamp]] = []
        for symbol_cfg in UNIVERSE:
            processed += 1
            day_key = (symbol_cfg.symbol, pd.Timestamp(trade_day).strftime("%Y-%m-%d"))
            if day_key in completed_symbol_days:
                print(f"[{processed}/{total_configs}] {symbol_cfg.symbol} {trade_day.date()}... cached summary")
                continue
            print(f"[{processed}/{total_configs}] {symbol_cfg.symbol} {trade_day.date()}...")
            pending.append((symbol_cfg, trade_day))
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=CFG.max_symbol_day_workers) as executor:
            futures = {
                executor.submit(process_symbol_day, symbol_cfg, day): (symbol_cfg, day)
                for symbol_cfg, day in pending
            }
            for future in as_completed(futures):
                symbol_cfg, day = futures[future]
                try:
                    day_key, _ = future.result()
                except Exception as exc:
                    print(f"FAILED {symbol_cfg.symbol} {day.date()}: {exc!r}")
                    continue
                completed_symbol_days.add(day_key)
                if len(completed_symbol_days) % 25 == 0:
                    update_progress_manifest(completed_symbol_days)

    assembled = assemble_outputs()
    paths = output_paths()
    summary_df = assembled["summary_df"]
    counts = assembled["counts"]

    print("\nCollection summary")
    print("-" * 60)
    print(f"requested symbol-days : {counts['requested_symbol_days']}")
    print(f"expiration rows       : {counts['expiration_rows']}")
    print(f"contract rows         : {counts['contract_rows']}")
    print(f"quality rows          : {counts['quality_rows']}")
    print(f"screening rows        : {counts['screening_rows']}")
    if not summary_df.empty:
        row = summary_df.iloc[0]
        print(f"stock-available days  : {int(row['stock_available_days'])}")
        print(f"quote-covered rows    : {int(row['option_quote_rows_with_data'])}")
        print(f"trade-covered rows    : {int(row['option_trade_rows_with_data'])}")
        print(f"clean sample rows     : {int(row['clean_sample_contract_rows'])}")
        print(f"broad sample rows     : {int(row['broad_sample_contract_rows'])}")
        print(f"full quote checks     : {int(row['contracts_with_full_quote_observed_checks'])}")
        print(f"full trade checks     : {int(row['contracts_with_full_trade_observed_checks'])}")
    print("-" * 60)
    if CFG.assemble_csv_outputs:
        print(f"Saved: {paths['sessions']}")
        print(f"Saved: {paths['expirations']}")
        print(f"Saved: {paths['contracts']}")
        print(f"Saved: {paths['quality']}")
    print(f"Saved: {paths['observed_instrument_index']}")
    print(f"Saved: {paths['family_coverage']}")
    print(f"Saved: {paths['summary']}")
    print(f"Saved: {paths['dataset_quality']}")
    print(f"Saved: {paths['schema']}")
    print(f"Saved: {paths['symbol_quality']}")
    print(f"Saved: {paths['timestamp_validation']}")
    print(f"Saved: {paths['screening']}")
    print(f"Saved: {paths['validation_targets']}")
    print(f"Saved: {paths['reference_actions']}")
    print(f"Saved: {paths['carry_inputs']}")
    print(f"Saved: {paths['failures']}")


if __name__ == "__main__":
    main()
