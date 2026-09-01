from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .timestamps import EvaluationPoint


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
