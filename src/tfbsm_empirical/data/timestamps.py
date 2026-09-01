from __future__ import annotations

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
