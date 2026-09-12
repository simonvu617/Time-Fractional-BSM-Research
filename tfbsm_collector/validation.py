# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Strict CSV parsing and diagnostics that preserve vendor observations.

Raw fields remain text. Parsed UTC clocks are additional columns, and quote
quality findings remain metadata. Validate the full response before storage
selects contracts, so excluded rows cannot conceal a misrouted response.
"""

import csv
import io
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

from tfbsm_collector import config, planning


def csv_frames(payload: BinaryIO, chunk_rows: int) -> Iterable[pd.DataFrame]:
    """Yield strictly validated CSV records as bounded text-only batches.

    Args:
        payload: Open binary response stream at its current read position. This
            function borrows the stream and leaves it open on exit.
        chunk_rows: Maximum records per batch.

    Yields:
        DataFrames preserving field text, order, blanks, and duplicates. A
        header-only response yields an empty table so its schema is checked.

    Raises:
        ValueError: CSV syntax, header names, or record widths are invalid.
        UnicodeError: The response is not valid UTF-8.
    """
    # A permissive chunked CSV parser can lose extra fields at a batch boundary.
    # Check complete CSV records, including quoted commas/newlines, before
    # storage.
    text = io.TextIOWrapper(payload, encoding="utf-8-sig", newline="")
    reader = csv.reader(text, strict=True)
    try:
        header = next((row for row in reader if row), None)
        if header is None:
            return
        if any(not name.strip() for name in header) or len(set(header)) != len(
            header
        ):
            raise ValueError("CSV header has empty or duplicate column names")
        rows, emitted = [], False
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"CSV record ending on line {reader.line_num}: "
                    f"expected {len(header)} fields, got {len(row)}"
                )
            rows.append(row)
            if len(rows) == chunk_rows:
                yield pd.DataFrame(rows, columns=header, dtype="string")
                rows, emitted = [], True
        if rows or not emitted:
            yield pd.DataFrame(rows, columns=header, dtype="string")
    except csv.Error as exc:
        raise ValueError(
            f"Malformed CSV near line {reader.line_num}: {exc}"
        ) from exc
    finally:
        # The caller may need the exact failure bytes; leave its stream open.
        text.detach()


def parse_vendor_clock(values: pd.Series, exchange_tz: str) -> pd.Series:
    """Parse vendor timestamps into a separate UTC series.

    Args:
        values: Original timestamp text, which is left unchanged.
        exchange_tz: Time zone applied only to timestamps without an offset.

    Returns:
        UTC timestamps aligned to the input index. Unparseable, ambiguous, and
        nonexistent local times become NaT; explicit offsets are honored.
    """
    text = values.astype("string").str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    parsed.loc[aware] = pd.to_datetime(
        text.loc[aware], format="mixed", errors="coerce", utc=True
    )
    naive = pd.to_datetime(text.loc[~aware], format="mixed", errors="coerce")

    # Daylight saving can make a local clock ambiguous or impossible. Mark it
    # missing rather than guessing; a fixed UTC offset would misalign seasons.
    parsed.loc[~aware] = naive.dt.tz_localize(
        exchange_tz, ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")

    return parsed


def raw_frame_with_diagnostics(
    frame: pd.DataFrame, request: planning.Request, cfg: config.CollectorConfig
) -> tuple[pd.DataFrame, dict]:
    """Add parsed clocks and describe quality without filtering raw rows.

    Args:
        frame: One batch of vendor fields stored as strings.
        request: Expected response identity and clock semantics.
        cfg: Exchange time zone and collection settings.

    Returns:
        A pair (frame_copy, diagnostics). Added collector_*_utc columns keep
        parsed clocks separate from vendor text. Counts describe duplicate,
        invalid, crossed, or nonpositive quotes without removing them.
    """
    result = frame.copy()
    # Count repeated rows without deleting them or conflating them with events.
    diagnostics = {
        "duplicate_rows": int(frame.duplicated().sum()),
        "clocks": {},
    }

    clocks = (
        "timestamp",
        "trade_timestamp",
        "quote_timestamp",
        "last_trade",
        "created",
    )
    for name in clocks:
        if name not in frame or (
            name == "created" and request.dataset == "interest_rate_eod"
        ):
            continue
        parsed = parse_vendor_clock(frame[name], cfg.exchange_tz)
        if f"collector_{name}_utc" not in result:
            result[f"collector_{name}_utc"] = parsed
        valid = parsed.dropna()
        diagnostics["clocks"][name] = {
            "unparseable_or_missing": int(parsed.isna().sum()),
            "out_of_order_transitions": int(
                valid.diff().lt(pd.Timedelta(0)).sum()
            ),
            "first_utc": valid.min().isoformat() if len(valid) else None,
            "last_utc": valid.max().isoformat() if len(valid) else None,
        }
    if {"bid", "ask"}.issubset(frame):
        # Parse numbers only for diagnostics. Vendor price text, including
        # blanks and precision, remains intact in the saved table.
        bid, ask = (
            pd.to_numeric(frame[side], errors="coerce")
            for side in ("bid", "ask")
        )

        # A zero bid may indicate no displayed buying interest. A crossed quote
        # (bid > ask) complicates midpoint pricing. Both stay in the liquidity
        # sample.
        diagnostics.update(
            invalid_bid_ask_rows=int(
                (~np.isfinite(bid) | ~np.isfinite(ask)).sum()
            ),
            nonpositive_bid_ask_rows=int((bid.le(0) | ask.le(0)).sum()),
            crossed_quote_rows=int(bid.gt(ask).sum()),
        )
    report_date = request.report_date_column

    # A date-only report does not reveal when its value first became available.
    # Do not invent a publication timestamp for rates or corporate actions.
    if report_date and report_date in frame:
        dates = pd.to_datetime(
            frame[report_date].astype("string").str.strip(),
            format="mixed",
            errors="coerce",
        )
        diagnostics["report_dates"] = {
            "column": report_date,
            "unparseable_or_missing": int(dates.isna().sum()),
            "first": str(dates.min().date()) if dates.notna().any() else None,
            "last": str(dates.max().date()) if dates.notna().any() else None,
            "unique_count": int(dates.nunique()),
            "publication_time_verified": False,
        }
    if request.dataset == "corporate_dividend" and "amount" in frame:
        # A blank cash amount remains unknown; zero would invent a dividend
        # input.
        diagnostics["unknown_dividend_amount_rows"] = int(
            frame["amount"].astype("string").str.strip().eq("").sum()
        )
    if (
        request.endpoint.endswith("/history/quote")
        and "collector_timestamp_utc" in result
        and request.params.get("strike") != "*"
    ):
        interval = request.params.get("interval", "tick")
        if interval != "tick":
            seconds = planning.interval_seconds(interval)
            unique = (
                result["collector_timestamp_utc"]
                .dropna()
                .drop_duplicates()
                .sort_values()
            )
            gaps = unique.diff().dt.total_seconds().dropna()

            # At 1h, 10:30 to 13:30 misses 11:30 and 12:30. The tolerance
            # prevents floating-point noise from adding a slot; session coverage
            # checks edges.
            diagnostics["absent_interior_sample_slots"] = int(
                np.maximum(
                    np.ceil(gaps.to_numpy() / seconds - 1e-9) - 1, 0
                ).sum()
            )
    return result, diagnostics


def response_identity_issues(
    frame: pd.DataFrame, request: planning.Request, cfg: config.CollectorConfig
) -> list[str]:
    """Return response identity/date problems without rewriting records.

    Args:
        frame: Vendor fields stored as strings.
        request: Expected symbol, contract scope, and date range.
        cfg: Exchange time zone used to compare market-record dates.

    Returns:
        Issue identifiers, or an empty list. Wildcard contract requests are
        checked against their actual scope, including the inclusive max_dte.
    """
    if frame.empty:
        return []
    issues = []
    if "symbol" in frame and frame["symbol"].ne(request.params["symbol"]).any():
        issues.append("unexpected_symbol")
    if request.endpoint.startswith("/option/") and set(
        planning.CONTRACT_FIELDS
    ).issubset(frame):
        expiry = pd.to_datetime(
            frame["expiration"], format="mixed", errors="coerce"
        )
        strike = pd.to_numeric(frame["strike"], errors="coerce")
        right = frame["right"].str.lower().replace({"c": "call", "p": "put"})
        if (
            expiry.isna().any()
            or (~np.isfinite(strike) | strike.le(0)).any()
            or not right.isin(["call", "put"]).all()
        ):
            issues.append("invalid_contract_identity")

        if (
            (
                request.params.get("expiration") not in {None, "*"}
                and expiry.ne(pd.Timestamp(request.params["expiration"])).any()
            )
            or (
                request.params.get("strike") not in {None, "*"}
                and strike.ne(float(request.params["strike"])).any()
            )
            or (
                request.params.get("right") in {"call", "put"}
                and right.ne(request.params["right"]).any()
            )
        ):
            issues.append("unexpected_contract_identity")
        # The requested maturity bound is inclusive: 180 days is allowed at
        # max_dte=180, while 181 days makes the response outside its requested
        # scope.
        if "max_dte" in request.params:
            requested_day = pd.Timestamp(
                request.params.get("date", request.params.get("start_date"))
            )

            if (
                (expiry - requested_day)
                .dt.days.gt(request.params["max_dte"])
                .any()
            ):
                issues.append("expiration_outside_requested_dte")
    start = request.params.get("date", request.params.get("start_date"))

    end = request.params.get("date", request.params.get("end_date"))
    report_date = request.report_date_column
    primary = report_date or next(
        (
            name
            for name in ("timestamp", "trade_timestamp", "created")
            if name in frame
        ),
        None,
    )
    if primary and primary in frame:
        if report_date:
            dates = pd.to_datetime(
                frame[primary].str.strip(), format="mixed", errors="coerce"
            )
        else:
            # Market dates must be compared in exchange time, not by their UTC
            # date.
            dates = (
                parse_vendor_clock(frame[primary], cfg.exchange_tz)
                .dt.tz_convert(cfg.exchange_tz)
                .dt.tz_localize(None)
                .dt.normalize()
            )
        if (
            start
            and end
            and (
                dates.notna()
                & ~dates.between(pd.Timestamp(start), pd.Timestamp(end))
            ).any()
        ):
            issues.append("timestamps_outside_requested_dates")
        if dates.isna().all():
            issues.append("no_parseable_report_dates")
        elif report_date and dates.isna().any():
            issues.append("invalid_report_dates")
    return issues


class RawDiagnostics:
    """Bounded quality summaries accumulated across a vendor response.

    Attributes:
        request: Expected response identity and clock meanings.
        cfg: Collection settings.
        rows: Number of parsed rows before any storage filter.
        chunks: Number of processed batches, including empty schema batches.
        quality: Aggregated diagnostic counts and clock summaries.
        issues: Response identity problems observed so far.
        report_dates: Distinct parsed report dates for final coverage metadata.
        last_clocks: Last valid clock by field, for cross-batch comparisons.
        report_missing: Number of missing/unparseable date-only report values.
    """

    def __init__(self, request: planning.Request, cfg: config.CollectorConfig):
        """Initialize empty quality summaries for the request."""
        self.request, self.cfg = request, cfg
        self.rows = self.chunks = 0
        self.quality = {"clocks": {}}
        self.issues, self.report_dates = set(), set()
        self.last_clocks = {}
        self.report_missing = 0

    def add(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Accumulate one raw batch and return its copy with parsed UTC clocks.

        Args:
            raw: Unfiltered vendor fields as strings; left unchanged.

        Returns:
            The batch with additional parsed clock columns. State accumulates
            before any selected-contract storage filter is applied.
        """
        frame, quality = raw_frame_with_diagnostics(raw, self.request, self.cfg)
        self.rows += len(frame)
        self.chunks += 1

        # A batch with blank clocks is not proof the whole response lacks dates.
        # Defer that global check until finish(), after all batches have
        # arrived.
        self.issues.update(
            issue
            for issue in response_identity_issues(raw, self.request, self.cfg)
            if issue != "no_parseable_report_dates"
        )
        for name, value in quality.items():
            if name not in {"clocks", "report_dates"}:
                self.quality[name] = self.quality.get(name, 0) + value
        for name, info in quality["clocks"].items():
            parsed = frame[f"collector_{name}_utc"]
            if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
                parsed = parse_vendor_clock(raw[name], self.cfg.exchange_tz)
            valid = parsed.dropna()
            target = self.quality["clocks"].setdefault(
                name,
                {
                    "unparseable_or_missing": 0,
                    "out_of_order_transitions": 0,
                    "first_utc": None,
                    "last_utc": None,
                },
            )
            for count in ("unparseable_or_missing", "out_of_order_transitions"):
                target[count] += info[count]
            if valid.empty:
                continue
            previous = self.last_clocks.get(name)

            # A batch boundary is still inside one response. Include clock
            # reversals and missing samples that cross it instead of resetting
            # their history.
            if previous is not None:
                target["out_of_order_transitions"] += int(
                    valid.iloc[0] < previous
                )
                if (
                    name == "timestamp"
                    and "absent_interior_sample_slots" in quality
                ):
                    gap = (
                        valid.iloc[0] - previous
                    ).total_seconds() / planning.interval_seconds(
                        self.request.params["interval"]
                    )
                    self.quality["absent_interior_sample_slots"] += max(
                        int(np.ceil(gap - 1e-9)) - 1, 0
                    )
            self.last_clocks[name] = valid.iloc[-1]
            for key, choose in (("first_utc", min), ("last_utc", max)):
                target[key] = choose(
                    (v for v in (target[key], info[key]) if v is not None),
                    key=pd.Timestamp,
                )
        column = self.request.report_date_column
        if column and column in raw:
            dates = pd.to_datetime(
                raw[column].str.strip(), format="mixed", errors="coerce"
            )
            self.report_dates.update(dates.dropna().dt.strftime("%Y-%m-%d"))
            self.report_missing += int(dates.isna().sum())
        return frame

    def finish(self) -> dict:
        """Finalize and return quality metadata after all batches are consumed.

        Adds whole-response missing-clock checks and labels duplicate counts as
        a lower bound when duplicates across batch boundaries were not counted.
        """
        if self.request.report_date_column:
            self.quality["report_dates"] = {
                "column": self.request.report_date_column,
                "unparseable_or_missing": self.report_missing,
                "first": min(self.report_dates) if self.report_dates else None,
                "last": max(self.report_dates) if self.report_dates else None,
                "unique_count": len(self.report_dates),
                "publication_time_verified": False,
            }
            if self.report_missing:
                self.issues.add(
                    "invalid_report_dates"
                    if self.report_dates
                    else "no_parseable_report_dates"
                )
        else:
            primary = next(
                (
                    name
                    for name in ("timestamp", "trade_timestamp", "created")
                    if name in self.quality["clocks"]
                ),
                None,
            )
            if (
                primary
                and self.rows
                and self.quality["clocks"][primary]["unparseable_or_missing"]
                == self.rows
            ):
                self.issues.add("no_parseable_report_dates")

        # Exact cross-batch duplicate counts need response-sized state. Label
        # the bounded-memory count as a lower bound, before any storage
        # exclusions.
        self.quality.update(
            read_chunks=self.chunks,
            duplicate_count_is_lower_bound=self.chunks > 1,
            duplicate_count_scope="within_read_chunks"
            if self.chunks > 1
            else "whole_response",
            response_identity_issues=sorted(self.issues),
        )
        clock = self.quality["clocks"].get("timestamp", {})
        if "absent_interior_sample_slots" in self.quality and clock.get(
            "out_of_order_transitions"
        ):
            self.quality["absent_interior_sample_slots"] = None
            self.quality["sample_gap_diagnostic_unavailable_reason"] = (
                "out_of_order_timestamps"
            )
        return self.quality
