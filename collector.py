#!/usr/bin/env python3
"""TFBSM market-data collector: single-file working review.

This file consolidates the collector from commit 3775af944841462dd3295b2b22bc9298c7e77bda.
Read the sections below in order: configuration, HTTP client, raw cache, timestamps,
contract selection, features, validation, canonical writer, pipeline, and CLI.

Dependencies (Python 3.11+):
    python -m pip install exchange-calendars numpy pandas pyarrow PyYAML requests

Preview one symbol-day without vendor requests or output writes:
    python collector.py --symbol SPY --date 2025-01-03 --dry-run

The original YAML configuration is embedded below. Edit it here or supply
--config PATH to use an external YAML file (paths retain the original convention
of resolving relative to that config file's parent directory's parent).
Default outputs are relative to this script's directory.

Review status: historical contract discovery still uses undated expiration and
strike lists; point-in-time semantics and live ThetaData behavior are unresolved.
This is collection code, with no pricing model, calibration, or backtest.
Original software: MIT License, copyright 2026 Simon Vu; see LICENSE.
"""
from __future__ import annotations

# ==========================================================================================
# EMBEDDED CONFIGURATION
# ==========================================================================================
DEFAULT_CONFIG_YAML = """\
project:
  name: tfbsm_empirical
  output_schema_version: "0.1.0"

collector:
  vendor: ThetaData
  base_url: http://127.0.0.1:25503/v3
  request_timeout_seconds: 120
  output_dir: data/raw/thetadata
  start_date: 2018-01-01
  end_date: 2025-12-31
  calendar: XNYS
  exchange_timezone: America/New_York
  quote_interval: 1s
  trade_interval: 1s
  evaluation_times:
    - "10:30:00"
    - "13:00:00"
    - "15:00:00"
  horizon_minutes: 30
  max_stock_quote_age_seconds: 70
  min_dte: 7
  max_dte: 180
  target_dtes:
    - 7
    - 14
    - 30
    - 60
    - 120
  max_expirations_per_day: 5
  moneyness_targets:
    - 0.80
    - 0.85
    - 0.90
    - 0.95
    - 1.00
    - 1.05
    - 1.10
    - 1.20
  strikes_per_moneyness_target: 1
  option_rights:
    - call
    - put

universe:
  - symbol: SPY
    asset_type: ETF
    universe_bucket: broad_market_etf
    sector_proxy: broad_market
  - symbol: QQQ
    asset_type: ETF
    universe_bucket: growth_etf
    sector_proxy: technology
  - symbol: IWM
    asset_type: ETF
    universe_bucket: small_cap_etf
    sector_proxy: small_cap
  - symbol: DIA
    asset_type: ETF
    universe_bucket: value_etf
    sector_proxy: industrial_mix
  - symbol: XLF
    asset_type: ETF
    universe_bucket: sector_etf
    sector_proxy: financials
  - symbol: XLK
    asset_type: ETF
    universe_bucket: sector_etf
    sector_proxy: technology
  - symbol: XLE
    asset_type: ETF
    universe_bucket: sector_etf
    sector_proxy: energy
  - symbol: XLV
    asset_type: ETF
    universe_bucket: sector_etf
    sector_proxy: health_care
  - symbol: XLI
    asset_type: ETF
    universe_bucket: sector_etf
    sector_proxy: industrials
  - symbol: XLP
    asset_type: ETF
    universe_bucket: sector_etf
    sector_proxy: consumer_staples
  - symbol: AAPL
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: technology
  - symbol: MSFT
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: technology
  - symbol: NVDA
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: technology
  - symbol: JPM
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: financials
  - symbol: XOM
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: energy
  - symbol: UNH
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: health_care
  - symbol: WMT
    asset_type: EQUITY
    universe_bucket: mega_cap_equity
    sector_proxy: consumer_staples
  - symbol: CAT
    asset_type: EQUITY
    universe_bucket: large_cap_equity
    sector_proxy: industrials
  - symbol: AMD
    asset_type: EQUITY
    universe_bucket: large_cap_equity
    sector_proxy: technology
  - symbol: PLTR
    asset_type: EQUITY
    universe_bucket: mid_cap_equity
    sector_proxy: technology
  - symbol: RIOT
    asset_type: EQUITY
    universe_bucket: small_cap_equity
    sector_proxy: crypto_exposed
"""


# ==========================================================================================
# CONFIG (formerly src/tfbsm_empirical/data/config.py)
# ==========================================================================================

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SymbolSpec":
        return cls(
            symbol=str(value["symbol"]).strip().upper(),
            asset_type=str(value["asset_type"]).strip().upper(),
            universe_bucket=str(value["universe_bucket"]).strip(),
            sector_proxy=str(value["sector_proxy"]).strip(),
        )


@dataclass(frozen=True)
class CollectorConfig:
    project_name: str
    output_schema_version: str
    vendor: str
    base_url: str
    request_timeout_seconds: float
    project_root: Path
    output_dir: Path
    start_date: date
    end_date: date
    calendar: str
    exchange_timezone: str
    quote_interval: str
    trade_interval: str
    evaluation_times: tuple[str, ...]
    horizon_minutes: int
    max_stock_quote_age_seconds: int
    min_dte: int
    max_dte: int
    target_dtes: tuple[int, ...]
    max_expirations_per_day: int
    moneyness_targets: tuple[float, ...]
    strikes_per_moneyness_target: int
    option_rights: tuple[str, ...]
    universe: tuple[SymbolSpec, ...]

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP URL")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if not (0 < self.min_dte <= self.max_dte):
            raise ValueError("DTE bounds must satisfy 0 < min_dte <= max_dte")
        if not self.target_dtes or any(value <= 0 for value in self.target_dtes):
            raise ValueError("target_dtes must contain positive values")
        if self.max_expirations_per_day <= 0:
            raise ValueError("max_expirations_per_day must be positive")
        if not self.moneyness_targets or any(value <= 0 for value in self.moneyness_targets):
            raise ValueError("moneyness_targets must contain positive values")
        if self.strikes_per_moneyness_target <= 0:
            raise ValueError("strikes_per_moneyness_target must be positive")
        invalid_rights = set(self.option_rights) - {"call", "put"}
        if invalid_rights:
            raise ValueError(f"unsupported option rights: {sorted(invalid_rights)}")
        if not self.evaluation_times:
            raise ValueError("at least one evaluation time is required")
        for value in self.evaluation_times:
            datetime.strptime(value, "%H:%M:%S")
        symbols = [item.symbol for item in self.universe]
        if not symbols:
            raise ValueError("the collection universe must not be empty")
        duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        if duplicates:
            raise ValueError(f"duplicate universe symbols: {duplicates}")
        if not self.output_dir.is_relative_to(self.project_root):
            raise ValueError("output_dir must remain inside the project root")

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "CollectorConfig":
        config_path = Path(path).expanduser().resolve() if path is not None else None
        payload = yaml.safe_load(
            config_path.read_text(encoding="utf-8") if config_path is not None
            else DEFAULT_CONFIG_YAML
        )
        if not isinstance(payload, Mapping):
            raise ValueError("collector config must contain a mapping")

        project = _mapping(payload, "project")
        collector = _mapping(payload, "collector")
        universe_values = payload.get("universe")
        if not isinstance(universe_values, list):
            raise ValueError("universe must be a list")

        project_root = (
            config_path.parent.parent.resolve() if config_path is not None
            else Path(__file__).resolve().parent
        )
        configured_output = Path(str(collector["output_dir"])).expanduser()
        output_dir = (
            configured_output.resolve()
            if configured_output.is_absolute()
            else (project_root / configured_output).resolve()
        )

        return cls(
            project_name=str(project["name"]),
            output_schema_version=str(project["output_schema_version"]),
            vendor=str(collector["vendor"]),
            base_url=str(collector["base_url"]).rstrip("/"),
            request_timeout_seconds=float(collector["request_timeout_seconds"]),
            project_root=project_root,
            output_dir=output_dir,
            start_date=_as_date(collector["start_date"], "start_date"),
            end_date=_as_date(collector["end_date"], "end_date"),
            calendar=str(collector["calendar"]),
            exchange_timezone=str(collector["exchange_timezone"]),
            quote_interval=str(collector["quote_interval"]),
            trade_interval=str(collector["trade_interval"]),
            evaluation_times=tuple(str(value) for value in collector["evaluation_times"]),
            horizon_minutes=int(collector["horizon_minutes"]),
            max_stock_quote_age_seconds=int(collector["max_stock_quote_age_seconds"]),
            min_dte=int(collector["min_dte"]),
            max_dte=int(collector["max_dte"]),
            target_dtes=tuple(int(value) for value in collector["target_dtes"]),
            max_expirations_per_day=int(collector["max_expirations_per_day"]),
            moneyness_targets=tuple(float(value) for value in collector["moneyness_targets"]),
            strikes_per_moneyness_target=int(collector["strikes_per_moneyness_target"]),
            option_rights=tuple(str(value).lower() for value in collector["option_rights"]),
            universe=tuple(SymbolSpec.from_mapping(value) for value in universe_values),
        )

    def symbol(self, value: str) -> SymbolSpec:
        requested = value.strip().upper()
        for item in self.universe:
            if item.symbol == requested:
                return item
        raise KeyError(f"symbol {requested!r} is not present in the configured universe")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["output_dir"] = str(self.output_dir)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _as_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format") from exc


# ==========================================================================================
# CLIENT (formerly src/tfbsm_empirical/data/client.py)
# ==========================================================================================

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping, Protocol

import pandas as pd
import requests


@dataclass(frozen=True)
class CsvResponse:
    frame: pd.DataFrame
    payload: bytes
    request_url: str
    status_code: int
    elapsed_ms: float
    response_headers: Mapping[str, str]


class MarketDataClient(Protocol):
    def fetch_csv(self, endpoint: str, params: Mapping[str, Any]) -> CsvResponse:
        """Fetch one read-only CSV response from the configured vendor."""


class ThetaDataClient:
    """Minimal read-only adapter for the local ThetaData HTTP terminal."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._owns_session = session is None

    def fetch_csv(self, endpoint: str, params: Mapping[str, Any]) -> CsvResponse:
        if not endpoint.startswith("/"):
            raise ValueError("vendor endpoint must begin with '/'")
        started = time.perf_counter()
        response = self._session.get(
            f"{self.base_url}{endpoint}",
            params=dict(params),
            timeout=self.timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.content
        try:
            frame = pd.read_csv(BytesIO(payload))
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        return CsvResponse(
            frame=frame,
            payload=payload,
            request_url=response.url,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            response_headers=dict(response.headers),
        )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "ThetaDataClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ==========================================================================================
# CACHE (formerly src/tfbsm_empirical/data/cache.py)
# ==========================================================================================

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd



class CacheCorruptionError(RuntimeError):
    pass


class CacheConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    frame: pd.DataFrame
    payload_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any]


class ResponseCache:
    """Immutable, request-addressed storage for exact vendor CSV responses."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(
        self,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> CachedResponse | None:
        payload_path, metadata_path = self._paths(dataset, endpoint, params)
        if not payload_path.exists() and not metadata_path.exists():
            return None
        if not payload_path.exists() or not metadata_path.exists():
            raise CacheCorruptionError(f"incomplete cache entry: {payload_path.parent}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_request = request_digest(endpoint, params)
        if metadata.get("request_digest") != expected_request:
            raise CacheCorruptionError(f"request digest mismatch: {payload_path.parent}")
        payload = payload_path.read_bytes()
        if metadata.get("payload_sha256") != sha256_bytes(payload):
            raise CacheCorruptionError(f"payload hash mismatch: {payload_path}")
        try:
            frame = pd.read_csv(BytesIO(payload))
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        return CachedResponse(
            frame=frame,
            payload_path=payload_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def store(
        self,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
        response: CsvResponse,
    ) -> CachedResponse:
        payload_path, metadata_path = self._paths(dataset, endpoint, params)
        existing = self.load(dataset, endpoint, params)
        if existing is not None:
            existing_digest = str(existing.metadata["payload_sha256"])
            received_digest = sha256_bytes(response.payload)
            if existing_digest != received_digest:
                raise CacheConflictError(
                    "a different payload already exists for this request identity; "
                    "use a new output directory for a refreshed pull"
                )
            return existing

        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_digest = sha256_bytes(response.payload)
        metadata = {
            "dataset": dataset,
            "endpoint": endpoint,
            "params": _jsonable(params),
            "request_digest": request_digest(endpoint, params),
            "payload_sha256": payload_digest,
            "payload_bytes": len(response.payload),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_url": response.request_url,
            "status_code": response.status_code,
            "elapsed_ms": response.elapsed_ms,
            "response_headers": dict(response.response_headers),
        }
        _cache_atomic_write(payload_path, response.payload)
        _cache_atomic_write(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        )
        cached = self.load(dataset, endpoint, params)
        if cached is None:  # pragma: no cover - defensive guard after writes.
            raise CacheCorruptionError(f"cache write did not persist: {payload_path.parent}")
        return cached

    def _paths(
        self,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        safe_dataset = _cache_safe_component(dataset)
        entry = self.root / safe_dataset / f"request={request_digest(endpoint, params)}"
        return entry / "response.csv", entry / "metadata.json"


def request_digest(endpoint: str, params: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"endpoint": endpoint, "params": _jsonable(params)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


def _cache_safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def _cache_atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ==========================================================================================
# TIMESTAMPS (formerly src/tfbsm_empirical/data/timestamps.py)
# ==========================================================================================

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True)
class MarketSession:
    trade_date: date
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp


@dataclass(frozen=True)
class EvaluationPoint:
    label: str
    entry_at: pd.Timestamp
    exit_at: pd.Timestamp


def normalize_vendor_timestamps(
    frame: pd.DataFrame,
    exchange_timezone: str,
) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        return frame.copy()

    result = frame.copy()
    raw = result["timestamp"].copy()
    parsed = pd.to_datetime(raw, errors="coerce")
    try:
        timezone = parsed.dt.tz
    except AttributeError as exc:
        raise ValueError("vendor timestamps contain incompatible timezone formats") from exc

    naive_assumed = pd.Series(False, index=result.index, dtype=bool)
    if timezone is None:
        naive_assumed = parsed.notna()
        parsed = parsed.dt.tz_localize(
            exchange_timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    else:
        parsed = parsed.dt.tz_convert(exchange_timezone)

    result["timestamp_raw"] = raw
    result["timestamp"] = parsed
    result["timestamp_parse_failed"] = parsed.isna()
    result["timestamp_naive_assumed"] = naive_assumed
    return result


def market_session(
    trade_date: date,
    calendar_name: str,
    exchange_timezone: str,
) -> MarketSession:
    calendar = _calendar(calendar_name)
    session = pd.Timestamp(trade_date)
    if not calendar.is_session(session):
        raise ValueError(f"{trade_date.isoformat()} is not a {calendar_name} session")
    opened_at = calendar.session_open(session).tz_convert(exchange_timezone)
    closed_at = calendar.session_close(session).tz_convert(exchange_timezone)
    return MarketSession(trade_date, opened_at, closed_at)


def previous_session_date(trade_date: date, calendar_name: str) -> date:
    calendar = _calendar(calendar_name)
    session = pd.Timestamp(trade_date)
    if not calendar.is_session(session):
        raise ValueError(f"{trade_date.isoformat()} is not a {calendar_name} session")
    return pd.Timestamp(calendar.previous_session(session)).date()


def evaluation_schedule(
    session: MarketSession,
    evaluation_times: tuple[str, ...],
    horizon_minutes: int,
    exchange_timezone: str,
) -> tuple[EvaluationPoint, ...]:
    points: list[EvaluationPoint] = []
    for value in evaluation_times:
        entry_at = pd.Timestamp(
            f"{session.trade_date.isoformat()} {value}",
            tz=exchange_timezone,
        )
        if not session.opened_at <= entry_at <= session.closed_at:
            raise ValueError(f"evaluation time {value} falls outside the market session")
        exit_at = min(
            entry_at + pd.Timedelta(minutes=horizon_minutes),
            session.closed_at,
        )
        points.append(
            EvaluationPoint(
                label=value.replace(":", ""),
                entry_at=entry_at,
                exit_at=exit_at,
            )
        )
    return tuple(points)


@lru_cache(maxsize=None)
def _calendar(name: str):
    return xcals.get_calendar(name)


# ==========================================================================================
# UNIVERSE (formerly src/tfbsm_empirical/data/universe.py)
# ==========================================================================================

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


# ==========================================================================================
# FEATURES (formerly src/tfbsm_empirical/data/features.py)
# ==========================================================================================

from typing import Iterable

import numpy as np
import pandas as pd



def add_quote_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("bid", "ask", "bid_size", "ask_size"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if {"bid", "ask"}.issubset(result.columns):
        result["mid"] = (result["bid"] + result["ask"]) / 2.0
        result["spread"] = result["ask"] - result["bid"]
        result["relative_spread"] = result["spread"] / result["mid"].replace(0, np.nan)
    return result


def snapshot_at(
    frame: pd.DataFrame,
    target: pd.Timestamp,
    *,
    max_age_seconds: int,
) -> dict[str, object]:
    empty = {
        "target_timestamp": target,
        "timestamp": pd.NaT,
        "age_seconds": np.nan,
        "is_stale": True,
    }
    if frame.empty or "timestamp" not in frame.columns:
        return empty
    valid = frame.loc[frame["timestamp"].notna()].sort_values("timestamp")
    eligible = valid.loc[valid["timestamp"].le(target)]
    if eligible.empty:
        return empty
    row = eligible.iloc[-1]
    age_seconds = float((target - row["timestamp"]).total_seconds())
    result: dict[str, object] = {
        "target_timestamp": target,
        "timestamp": row["timestamp"],
        "age_seconds": age_seconds,
        "is_stale": age_seconds > max_age_seconds,
    }
    for column in (
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "mid",
        "spread",
        "relative_spread",
    ):
        if column in row.index:
            result[column] = row[column]
    return result


def reference_mids(
    stock_quotes: pd.DataFrame,
    schedule: Iterable[EvaluationPoint],
    *,
    max_age_seconds: int,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for point in schedule:
        snapshot = snapshot_at(
            stock_quotes,
            point.entry_at,
            max_age_seconds=max_age_seconds,
        )
        mid = pd.to_numeric(pd.Series([snapshot.get("mid")]), errors="coerce").iloc[0]
        if not bool(snapshot["is_stale"]) and pd.notna(mid) and float(mid) > 0:
            values[point.label] = float(mid)
    return values


def waiting_time_features(
    frame: pd.DataFrame,
    *,
    value_column: str = "mid",
) -> dict[str, float | int]:
    output: dict[str, float | int] = {
        "observation_count": 0,
        "update_count": 0,
        "zero_change_fraction": np.nan,
        "median_update_interval_seconds": np.nan,
        "longest_no_update_interval_seconds": np.nan,
    }
    if frame.empty or "timestamp" not in frame.columns or value_column not in frame.columns:
        return output
    valid = frame.loc[
        frame["timestamp"].notna() & frame[value_column].notna(),
        ["timestamp", value_column],
    ].sort_values("timestamp")
    output["observation_count"] = int(len(valid))
    if len(valid) < 2:
        return output

    changes = valid[value_column].ne(valid[value_column].shift())
    changes.iloc[0] = True
    update_times = valid.loc[changes, "timestamp"]
    update_intervals = update_times.diff().dt.total_seconds().dropna()
    tail_interval = float((valid["timestamp"].iloc[-1] - update_times.iloc[-1]).total_seconds())
    observed_no_update_intervals = [float(value) for value in update_intervals]
    observed_no_update_intervals.append(tail_interval)

    output["update_count"] = int(changes.sum())
    output["update_count"] -= 1  # The first observation establishes state; it is not an update.
    output["zero_change_fraction"] = float((~changes.iloc[1:]).mean())
    if not update_intervals.empty:
        output["median_update_interval_seconds"] = float(update_intervals.median())
    output["longest_no_update_interval_seconds"] = max(observed_no_update_intervals)
    return output


# ==========================================================================================
# VALIDATION (formerly src/tfbsm_empirical/data/validation.py)
# ==========================================================================================

from dataclasses import asdict, dataclass

import pandas as pd


class SchemaValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    row_count: int
    columns: tuple[str, ...]
    duplicate_row_count: int
    timestamp_parse_failure_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_frame(dataset: str, frame: pd.DataFrame) -> ValidationReport:
    required = required_columns(dataset)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaValidationError(
            f"{dataset} response is missing required columns: {missing}; "
            f"received={list(frame.columns)}"
        )
    parse_failures = 0
    if "timestamp_parse_failed" in frame.columns:
        parse_failures = int(frame["timestamp_parse_failed"].fillna(False).astype(bool).sum())
    return ValidationReport(
        dataset=dataset,
        row_count=int(len(frame)),
        columns=tuple(str(value) for value in frame.columns),
        duplicate_row_count=int(frame.duplicated().sum()),
        timestamp_parse_failure_count=parse_failures,
    )


def required_columns(dataset: str) -> tuple[str, ...]:
    if dataset == "chain_expirations":
        return ("expiration",)
    if dataset == "chain_strikes":
        return ("strike",)
    if dataset in {"stock_quotes", "option_quotes"}:
        return ("timestamp", "bid", "ask")
    if dataset in {"stock_trades", "option_trades"}:
        return ("timestamp", "price", "size")
    if dataset == "option_open_interest":
        return ("timestamp", "open_interest")
    if dataset == "contract_universe":
        return ("contract_id", "symbol", "expiration", "strike", "right")
    return ()


def validate_reconciliation(*, expected: int, collected: int) -> None:
    if expected != collected:
        raise SchemaValidationError(
            f"contract reconciliation failed: expected={expected}, collected={collected}"
        )


# ==========================================================================================
# WRITER (formerly src/tfbsm_empirical/data/writer.py)
# ==========================================================================================

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class WrittenArtifact:
    dataset: str
    path: Path
    row_count: int
    sha256: str
    partition: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


class CanonicalWriter:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def write_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        partition: Mapping[str, object],
    ) -> WrittenArtifact:
        directory = self.output_root / "canonical" / _writer_safe_component(dataset)
        normalized_partition: dict[str, str] = {}
        for key, value in partition.items():
            safe_key = _writer_safe_component(str(key))
            safe_value = _writer_safe_component(str(value))
            normalized_partition[safe_key] = safe_value
            directory /= f"{safe_key}={safe_value}"
        directory.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            dir=directory,
            delete=False,
            suffix=".parquet",
        ) as handle:
            temp_path = Path(handle.name)
        try:
            frame.to_parquet(temp_path, index=False)
            digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            destination = directory / f"part-{digest}.parquet"
            if destination.exists():
                temp_path.unlink()
            else:
                os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return WrittenArtifact(
            dataset=dataset,
            path=destination,
            row_count=int(len(frame)),
            sha256=digest,
            partition=normalized_partition,
        )

    def write_manifest(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.output_root / "manifests" / f"{_writer_safe_component(name)}.json"
        encoded = json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        _writer_atomic_write(path, encoded)
        return path


def _writer_safe_component(value: str) -> str:
    normalized = value.replace(":", "-").replace("/", "-").replace(" ", "_")
    if not normalized or not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError(f"unsafe path component: {value!r}")
    return normalized


def _writer_atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ==========================================================================================
# PIPELINE (formerly src/tfbsm_empirical/data/pipeline.py)
# ==========================================================================================

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd



class CollectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionResult:
    symbol: str
    trade_date: date
    contract_count: int
    artifact_count: int
    manifest_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date.isoformat(),
            "contract_count": self.contract_count,
            "artifact_count": self.artifact_count,
            "manifest_path": str(self.manifest_path),
        }


class CollectorPipeline:
    """Orchestrate one explicit, auditable symbol-day collection."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        client: MarketDataClient | None = None,
        cache: ResponseCache | None = None,
        writer: CanonicalWriter | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or ThetaDataClient(
            config.base_url,
            config.request_timeout_seconds,
        )
        self.cache = cache or ResponseCache(config.output_dir / "raw")
        self.writer = writer or CanonicalWriter(config.output_dir)

    def plan(self, symbol: str, trade_date: date) -> dict[str, object]:
        symbol_spec = self._validate_request(symbol, trade_date)
        session = market_session(
            trade_date,
            self.config.calendar,
            self.config.exchange_timezone,
        )
        schedule = evaluation_schedule(
            session,
            self.config.evaluation_times,
            self.config.horizon_minutes,
            self.config.exchange_timezone,
        )
        return {
            "symbol": symbol_spec.symbol,
            "asset_type": symbol_spec.asset_type,
            "trade_date": trade_date.isoformat(),
            "session_open": session.opened_at.isoformat(),
            "session_close": session.closed_at.isoformat(),
            "evaluation_points": [
                {
                    "label": point.label,
                    "entry_at": point.entry_at.isoformat(),
                    "exit_at": point.exit_at.isoformat(),
                }
                for point in schedule
            ],
            "target_dtes": list(self.config.target_dtes),
            "moneyness_targets": list(self.config.moneyness_targets),
            "option_rights": list(self.config.option_rights),
            "output_dir": str(self.config.output_dir),
            "network_requests_made": False,
        }

    def collect_symbol_day(self, symbol: str, trade_date: date) -> CollectionResult:
        symbol_spec = self._validate_request(symbol, trade_date)
        session = market_session(
            trade_date,
            self.config.calendar,
            self.config.exchange_timezone,
        )
        schedule = evaluation_schedule(
            session,
            self.config.evaluation_times,
            self.config.horizon_minutes,
            self.config.exchange_timezone,
        )
        previous_date = previous_session_date(trade_date, self.config.calendar)
        artifacts: list[dict[str, object]] = []
        request_date = trade_date.strftime("%Y%m%d")
        base_partition = {"symbol": symbol_spec.symbol, "date": trade_date.isoformat()}

        stock_quotes = self._fetch_and_write(
            dataset="stock_quotes",
            endpoint="/stock/history/quote",
            params={
                "symbol": symbol_spec.symbol,
                "date": request_date,
                "interval": self.config.quote_interval,
            },
            partition=base_partition,
            identity={
                "collector_symbol": symbol_spec.symbol,
                "collector_trade_date": trade_date.isoformat(),
            },
            add_quotes=True,
            artifacts=artifacts,
        )
        self._fetch_and_write(
            dataset="stock_trades",
            endpoint="/stock/history/trade",
            params={
                "symbol": symbol_spec.symbol,
                "date": request_date,
                "interval": self.config.trade_interval,
            },
            partition=base_partition,
            identity={
                "collector_symbol": symbol_spec.symbol,
                "collector_trade_date": trade_date.isoformat(),
            },
            artifacts=artifacts,
        )

        mids = reference_mids(
            stock_quotes,
            schedule,
            max_age_seconds=self.config.max_stock_quote_age_seconds,
        )
        if not mids:
            raise CollectionError(
                "no fresh positive stock reference mid was available at the configured evaluation times"
            )

        expiration_frame = self._fetch_and_write(
            dataset="chain_expirations",
            endpoint="/option/list/expirations",
            params={"symbol": symbol_spec.symbol},
            partition=base_partition,
            identity={"collector_symbol": symbol_spec.symbol},
            artifacts=artifacts,
        )
        listed_expirations = _parse_expirations(expiration_frame["expiration"])
        selected_expirations = select_expirations(
            trade_date,
            listed_expirations,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            target_dtes=self.config.target_dtes,
            maximum=self.config.max_expirations_per_day,
        )
        if not selected_expirations:
            raise CollectionError("no listed expirations satisfy the configured DTE window")

        contracts: list[dict[str, object]] = []
        for expiration in selected_expirations:
            expiration_value = expiration.isoformat()
            strikes_frame = self._fetch_and_write(
                dataset="chain_strikes",
                endpoint="/option/list/strikes",
                params={
                    "symbol": symbol_spec.symbol,
                    "expiration": expiration_value,
                },
                partition={**base_partition, "expiration": expiration_value},
                identity={
                    "collector_symbol": symbol_spec.symbol,
                    "collector_expiration": expiration_value,
                },
                artifacts=artifacts,
            )
            listed_strikes = tuple(
                pd.to_numeric(strikes_frame["strike"], errors="coerce")
                .dropna()
                .astype(float)
                .tolist()
            )
            chosen_strikes = select_strikes(
                listed_strikes,
                mids.values(),
                moneyness_targets=self.config.moneyness_targets,
                per_target=self.config.strikes_per_moneyness_target,
            )
            for strike in chosen_strikes:
                for right in self.config.option_rights:
                    contracts.append(
                        {
                            "contract_id": occ_contract_id(
                                symbol_spec.symbol,
                                expiration,
                                right,
                                strike,
                            ),
                            "symbol": symbol_spec.symbol,
                            "asset_type": symbol_spec.asset_type,
                            "universe_bucket": symbol_spec.universe_bucket,
                            "sector_proxy": symbol_spec.sector_proxy,
                            "trade_date": trade_date.isoformat(),
                            "expiration": expiration_value,
                            "dte": (expiration - trade_date).days,
                            "strike": strike,
                            "right": right,
                        }
                    )

        if not contracts:
            raise CollectionError("the configured surface rules selected no option contracts")

        contract_frame = pd.DataFrame(contracts)
        contract_report = validate_frame("contract_universe", contract_frame)
        contract_artifact = self.writer.write_frame(
            "contract_universe",
            contract_frame,
            partition=base_partition,
        )
        artifacts.append(
            {
                "source": "derived",
                "canonical": contract_artifact.as_dict(),
                "validation": contract_report.as_dict(),
            }
        )

        collected_contracts = 0
        for contract in contracts:
            identity = {
                "collector_symbol": symbol_spec.symbol,
                "collector_trade_date": trade_date.isoformat(),
                "collector_expiration": contract["expiration"],
                "collector_strike": contract["strike"],
                "collector_right": contract["right"],
                "collector_contract_id": contract["contract_id"],
            }
            partition = {
                **base_partition,
                "expiration": contract["expiration"],
                "strike": format_strike(float(contract["strike"])),
                "right": contract["right"],
            }
            common_params = {
                "symbol": symbol_spec.symbol,
                "expiration": contract["expiration"],
                "strike": format_strike(float(contract["strike"])),
                "right": contract["right"],
            }
            self._fetch_and_write(
                dataset="option_quotes",
                endpoint="/option/history/quote",
                params={
                    **common_params,
                    "date": request_date,
                    "interval": self.config.quote_interval,
                },
                partition=partition,
                identity=identity,
                add_quotes=True,
                artifacts=artifacts,
            )
            self._fetch_and_write(
                dataset="option_trades",
                endpoint="/option/history/trade",
                params={
                    **common_params,
                    "date": request_date,
                    "interval": self.config.trade_interval,
                },
                partition=partition,
                identity=identity,
                artifacts=artifacts,
            )
            self._fetch_and_write(
                dataset="option_open_interest",
                endpoint="/option/history/open_interest",
                params={
                    **common_params,
                    "date": previous_date.strftime("%Y%m%d"),
                },
                partition={
                    **partition,
                    "observation_date": previous_date.isoformat(),
                },
                identity={
                    **identity,
                    "collector_observation_date": previous_date.isoformat(),
                },
                artifacts=artifacts,
            )
            collected_contracts += 1

        validate_reconciliation(
            expected=len(contracts),
            collected=collected_contracts,
        )
        manifest_name = f"{symbol_spec.symbol}_{trade_date.isoformat()}"
        manifest_path = self.writer.write_manifest(
            manifest_name,
            {
                "status": "complete",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "project": self.config.project_name,
                "output_schema_version": self.config.output_schema_version,
                "vendor": self.config.vendor,
                "symbol": symbol_spec.symbol,
                "trade_date": trade_date.isoformat(),
                "previous_session_for_open_interest": previous_date.isoformat(),
                "evaluation_schedule": [
                    {
                        "label": point.label,
                        "entry_at": point.entry_at.isoformat(),
                        "exit_at": point.exit_at.isoformat(),
                    }
                    for point in schedule
                ],
                "reference_mids": mids,
                "selected_expirations": [value.isoformat() for value in selected_expirations],
                "expected_contract_count": len(contracts),
                "collected_contract_count": collected_contracts,
                "artifacts": artifacts,
            },
        )
        return CollectionResult(
            symbol=symbol_spec.symbol,
            trade_date=trade_date,
            contract_count=collected_contracts,
            artifact_count=len(artifacts),
            manifest_path=manifest_path,
        )

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "CollectorPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_request(self, symbol: str, trade_date: date) -> SymbolSpec:
        if not self.config.start_date <= trade_date <= self.config.end_date:
            raise ValueError(
                f"trade date {trade_date.isoformat()} is outside the configured study window"
            )
        return self.config.symbol(symbol)

    def _fetch_and_write(
        self,
        *,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
        partition: Mapping[str, object],
        identity: Mapping[str, object],
        artifacts: list[dict[str, object]],
        add_quotes: bool = False,
    ) -> pd.DataFrame:
        cached = self.cache.load(dataset, endpoint, params)
        source = "cache"
        if cached is None:
            response = self.client.fetch_csv(endpoint, params)
            cached = self.cache.store(dataset, endpoint, params, response)
            source = "network"

        frame = cached.frame.copy()
        if "timestamp" in frame.columns:
            frame = normalize_vendor_timestamps(
                frame,
                self.config.exchange_timezone,
            )
        if add_quotes:
            frame = add_quote_features(frame)
        for column, value in identity.items():
            frame[column] = value

        report = validate_frame(dataset, frame)
        canonical = self.writer.write_frame(
            dataset,
            frame,
            partition=partition,
        )
        artifacts.append(
            {
                "source": source,
                "endpoint": endpoint,
                "params": dict(params),
                "raw_payload": str(cached.payload_path),
                "raw_metadata": str(cached.metadata_path),
                "canonical": canonical.as_dict(),
                "validation": report.as_dict(),
            }
        )
        return frame


def _parse_expirations(values: pd.Series) -> tuple[date, ...]:
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    return tuple(sorted({value.date() for value in parsed.dropna()}))


# ==========================================================================================
# CLI (formerly src/tfbsm_empirical/data/cli.py)
# ==========================================================================================

import argparse
import json
from datetime import date



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one auditable TFBSM research symbol-day.",
    )
    parser.add_argument("--config", help="Optional YAML override; defaults to the embedded configuration")
    parser.add_argument("--symbol", required=True, help="Configured option root")
    parser.add_argument("--date", required=True, help="XNYS session in YYYY-MM-DD format")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the collection plan without vendor requests or writes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CollectorConfig.from_yaml(args.config)
    trade_date = date.fromisoformat(args.date)
    with CollectorPipeline(config) as pipeline:
        if args.dry_run:
            print(json.dumps(pipeline.plan(args.symbol, trade_date), indent=2, sort_keys=True))
            return
        result = pipeline.collect_symbol_day(args.symbol, trade_date)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
