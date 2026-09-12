# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Check observations and publish date/session coverage reports.

A successful bulk request can omit a selected contract or sampled timestamp.
CoverageChecker owns date-catalogue checks, per-response presence checks, and
the final availability CSV. It reports gaps without filling data, dropping
quotes, or deciding whether an observation is suitable for pricing.
"""

import csv

import pandas as pd

from tfbsm_collector import (
    planning,
    provenance,
    storage,
    transport,
    validation,
)


class CoverageChecker:
    """Presence checks over saved responses and the selected research sample.

    Attributes:
        store: Response store supplying bounded reads of saved observations.
        cfg: The store's session, sampling, and time-zone settings.
    """

    def __init__(self, store: storage.RequestStore):
        """Use the same store and settings as the collection workflow."""
        self.store = store
        self.cfg = store.cfg

    def request_coverage(
        self, request: planning.Request, record: dict | None
    ) -> dict:
        """Assess observed presence separately from request success.

        Args:
            request: Descriptor defining the expected series and date window.
            record: Saved request receipt, or None if no result was recorded.

        Returns:
            Coverage metadata with status observations_present, gaps_observed,
            unknown, or not_assessed. Empty OI reports are not
            automatically missing prices, and publisher schedules remain
            unverified.
        """
        if (
            record is None
            or record["status"] not in storage.GOOD_REQUEST_STATUSES
        ):
            return {
                "status": "unknown",
                "reason": "request_not_completed_successfully",
            }

        # An empty OI report does not mean zero open contracts and does not
        # establish a missing price observation.
        if request.endpoint.endswith("/open_interest"):
            return {
                "status": "not_assessed",
                "reason": "empty_event_reports_can_be_valid",
            }
        if request.endpoint.endswith("/quote") and request.endpoint.startswith(
            "/stock/"
        ):
            return self.quote_or_contract_report_coverage(request, record)
        if request.endpoint.endswith("/eod"):
            try:
                frame = self.store.read(record)
                column = request.report_date_column or "created"
                dates = (
                    pd.to_datetime(
                        frame[column], format="mixed", errors="coerce"
                    )
                    if request.report_date_column
                    else validation.parse_vendor_clock(
                        frame[column], self.cfg.exchange_tz
                    ).dt.tz_convert(self.cfg.exchange_tz)
                )
                observed = set(dates.dropna().dt.strftime("%Y-%m-%d"))
                sessions = planning.exchange_calendar().sessions_in_range(
                    request.params["start_date"], request.params["end_date"]
                )
                # This compares presence on exchange sessions, not the
                # publisher's schedule. A Fed/bank holiday can lack a rate while
                # the stock market is open.
                expected = set(sessions.strftime("%Y-%m-%d"))
                missing = sorted(expected - observed)
                return {
                    "status": "gaps_observed"
                    if missing
                    else "observations_present",
                    "requested_session_count": len(expected),
                    "observed_session_count": len(expected & observed),
                    "missing_requested_session_dates": missing,
                    "date_grid": "XNYS_sessions",
                    "publisher_schedule_verified": False,
                    "missing_dates_prove_vendor_error": False,
                }
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {
                    "status": "unknown",
                    "reason": "coverage_check_failed",
                    "error": repr(exc),
                }
        return {
            "status": "gaps_observed"
            if record["status"] == "no_data"
            else "observations_present",
            "complete_intraday_history_verified": False,
        }

    def quote_or_contract_report_coverage(
        self,
        request: planning.Request,
        record: dict | None,
        selected: pd.DataFrame | None = None,
    ) -> dict:
        """Check each wanted series, including missing sampled-quote times.

        Args:
            request: Stock quote or option quote/EOD descriptor.
            record: Saved request receipt, or None if no result was recorded.
            selected: Selection table required for option requests. Bulk data
                outside these identities cannot satisfy the selected sample.

        Returns:
            Coverage status, expected sample counts, and missing observations.
            Presence does not verify quote event age or suitability for pricing.
        """
        if (
            record is None
            or record["status"] not in storage.GOOD_REQUEST_STATUSES
        ):
            return {
                "status": "unknown",
                "reason": "request_not_completed_successfully",
            }
        option = request.endpoint.startswith("/option/")
        if option:
            chosen = selected
            if request.params.get("expiration") != "*":
                chosen = chosen.loc[
                    chosen["expiration"].eq(request.params["expiration"])
                ]
            wanted = set(chosen["contract_key"])
        else:
            wanted = {request.params["symbol"]}
        if not wanted:
            return {"status": "not_assessed", "reason": "no_selected_contracts"}
        sampled = request.endpoint.endswith("/history/quote")
        expected_times = set()
        if sampled:
            day = pd.Timestamp(request.params["date"])
            opened, closed = planning.session_bounds(day, self.cfg)
            # Seven hourly rows can repeat a timestamp and omit another. Compare
            # the actual clock grid, including endpoints, separately from
            # near-close data.
            expected_times = set(
                pd.date_range(
                    opened,
                    closed,
                    freq=pd.Timedelta(
                        seconds=planning.interval_seconds(
                            request.params["interval"]
                        )
                    ),
                ).tz_convert("UTC")
            )

        clocks = {key: set() for key in wanted}
        column = "created" if request.endpoint.endswith("/eod") else "timestamp"
        columns = [*planning.CONTRACT_FIELDS, column] if option else [column]
        try:
            for frame in self.store.iter_frames(record, columns=columns):
                if option:
                    keys = planning.option_contract_keys(frame)
                else:
                    keys = [request.params["symbol"]] * len(frame)
                for key, clock in zip(
                    keys,
                    validation.parse_vendor_clock(
                        frame[column], self.cfg.exchange_tz
                    ),
                ):
                    if key in wanted and pd.notna(clock):
                        # Deduplicate clocks only for presence checks. Stored
                        # quote rows remain intact.
                        clocks[key].add(clock)
            missing = []
            for key in sorted(wanted):
                absent = expected_times - clocks[key]
                if (sampled and absent) or not clocks[key]:
                    missing.append(
                        {
                            "contract_key" if option else "symbol": key,
                            "missing_sample_times": [
                                t.tz_convert(self.cfg.exchange_tz).strftime(
                                    "%H:%M:%S"
                                )
                                for t in sorted(absent)
                            ]
                            if sampled
                            else [],
                            "reason": "missing_sample_rows"
                            if sampled
                            else "no_dated_observation",
                        }
                    )
            return {
                "status": "gaps_observed"
                if missing
                else "observations_present",
                "checked_contract_count": len(wanted) if option else 0,
                "expected_samples_per_series": len(expected_times)
                if sampled
                else 1,
                "missing_observations": missing,
                "quote_event_age_verified": False,
                "research_sample_usability_verified": False,
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {
                "status": "unknown",
                "reason": "coverage_check_failed",
                "error": repr(exc),
            }

    def session_coverage(
        self,
        symbol: str,
        day: pd.Timestamp,
        selected: pd.DataFrame,
        records: list[dict],
        missing_times: list[str],
        failed: bool,
    ) -> dict:
        """Combine price and report coverage for the selected option sample.

        Args:
            symbol: Underlying ticker.
            day: Exchange session date, without a time zone.
            selected: Selected contract table, possibly empty.
            records: Completed request receipts for this session.
            missing_times: Scheduled stock selection times without a reference.
            failed: Whether collection encountered a request or processing
                error.

        Returns:
            Required-request checks and counts of missing quotes, EOD reports,
            stock series, and unknown results. Failures take precedence over
            gaps.
        """
        required = [
            request
            for request in planning.shared_day_requests(self.cfg, symbol, day)
            if request.endpoint.endswith(("/quote", "/eod"))
            and "/list/" not in request.endpoint
        ]
        required += planning.option_quote_requests(
            self.cfg, symbol, day, selected
        )
        by_id = {record["request_id"]: record for record in records}
        checks = []
        for request in required:
            record = by_id.get(request.request_id)
            coverage = (
                self.quote_or_contract_report_coverage(
                    request, record, selected
                )
                if request.endpoint.startswith("/option/")
                else self.request_coverage(request, record)
            )
            checks.append(
                {
                    "request_id": request.request_id,
                    "dataset": request.dataset,
                    "params": request.params,
                    **coverage,
                }
            )
        missing = [
            check for check in checks if check["status"] == "gaps_observed"
        ]
        unknown = sum(check["status"] == "unknown" for check in checks)
        status = (
            "unknown"
            if failed or unknown
            else "gaps_observed"
            if missing or missing_times or selected.empty
            else "observations_present"
        )
        return {
            "status": status,
            "required_price_requests": checks,
            "missing_option_quote_count": len(
                {
                    row["contract_key"]
                    for c in missing
                    if c["dataset"].startswith("option_quotes_")
                    for row in c["missing_observations"]
                }
            ),
            "missing_option_eod_count": len(
                {
                    row["contract_key"]
                    for c in missing
                    if c["dataset"] == "option_eod"
                    for row in c["missing_observations"]
                }
            ),
            "missing_stock_dataset_count": sum(
                c["dataset"].startswith("stock_") for c in missing
            ),
            "unknown_required_request_count": unknown,
            "no_selected_contracts": selected.empty,
            "complete_intraday_history_verified": False,
            "research_sample_usability_verified": False,
        }

    def collect_catalogue(
        self,
        anchors: pd.DatetimeIndex,
        run_id: str,
    ) -> dict:
        """Refresh stock/VIX date catalogues and record advertised date gaps.

        Args:
            anchors: Exchange session dates in the requested coverage window.
            run_id: Identifier naming the coverage report.

        Returns:
            Catalogue outcomes, missing requested dates, and unfinished
            requests. Listed dates do not establish subscription access or
            complete history.
        """
        requests_to_make = [
            planning.Request(
                f"stock_{kind}_dates",
                f"/stock/list/dates/{kind}",
                {"symbol": symbol.symbol, "format": "csv"},
            )
            for symbol in self.cfg.symbols
            for kind in ("quote", "trade")
        ]
        access_gaps = []
        if self.cfg.index_history_start is not None:
            requests_to_make.append(
                planning.Request(
                    "index_price_dates",
                    "/index/list/dates",
                    {"symbol": "VIX", "format": "csv"},
                )
            )
        else:
            # The catalogue itself requires index access too. Record its
            # absence even in --coverage-only mode, before reference collection.
            access_gaps.append(
                {
                    "symbol": "VIX",
                    "dataset": "index_price_dates",
                    "reason": "index_subscription_unavailable",
                }
            )
        expected = set(anchors.strftime("%Y-%m-%d"))
        rows, records = [], []
        try:
            for request in requests_to_make:
                if self.store.client.stop_event.is_set():
                    break
                try:
                    # Small date catalogues can expand after vendor backfills.
                    # Refresh them without invalidating independently cached
                    # historical responses.
                    record = self.store.collect(request, refresh=True)
                    frame = self.store.read(record)
                except transport.CollectionStopped:
                    break
                except Exception as exc:
                    frame = pd.DataFrame()
                    record = {
                        "request_id": request.request_id,
                        "dataset": request.dataset,
                        "params": request.params,
                        "status": "request_error",
                        "error": repr(exc),
                    }
                records.append(record)
                dates = (
                    pd.to_datetime(
                        frame["date"].astype("string").str.strip(),
                        format="mixed",
                        errors="coerce",
                    ).dropna()
                    if "date" in frame
                    else pd.Series(dtype="datetime64[ns]")
                )
                known = record["status"] in storage.GOOD_REQUEST_STATUSES
                available = set(dates.dt.strftime("%Y-%m-%d"))

                # None means the catalogue failed and coverage is unknown. An
                # empty list means the catalogue listed all requested dates, not
                # all observations.
                missing = sorted(expected - available) if known else None
                row = {
                    "symbol": request.params["symbol"],
                    "dataset": request.dataset,
                    "status": ("listed" if not missing else "coverage_gap")
                    if known
                    else "request_error",
                    "first_available": str(dates.min().date())
                    if len(dates)
                    else None,
                    "last_available": str(dates.max().date())
                    if len(dates)
                    else None,
                    "requested_sessions": len(expected),
                    "listed_requested_sessions": len(expected & available)
                    if known
                    else None,
                    "missing_requested_dates": missing,
                    "error": record.get("error", ""),
                }
                rows.append(row)
                print(
                    f"Coverage {row['symbol']} {row['dataset']}: {row['status']}"
                )
        finally:
            report = {
                "vendor": "ThetaData",
                "checked_at_utc": provenance.utc_now(),
                "rows": rows,
                "requests": records,
                "index_subscription": self.cfg.index_subscription,
                "subscription_coverage_gaps": access_gaps,
                "unfinished_requests": [
                    r.identity() for r in requests_to_make[len(rows) :]
                ],
                "request_errors": sum(
                    row["status"] == "request_error" for row in rows
                ),
                "series_with_gaps": sum(
                    row["status"] == "coverage_gap" for row in rows
                )
                + len(access_gaps),
                "proves_interval_or_subscription_access": False,
                "proves_complete_intraday_records": False,
                "scope": "stock quote/trade and VIX date catalogues; option coverage is recorded during collection",
                "documented_limits": {
                    "SPY_underlying_history_before_2020": "unavailable per Theta documentation",
                    "other_CTA_only_symbols": "pre-2020 underlying history may be unavailable",
                    "reference_and_adjusted_contract_completeness": "not established by date catalogues",
                },
                "documentation": "https://docs.thetadata.us/Articles/Data-And-Requests/Making-Requests.html",
            }
            storage.write_json(
                self.cfg.output_dir / "coverage" / f"{run_id}.json", report
            )
        return report

    def write_availability(self, anchors: pd.DatetimeIndex) -> dict:
        """Write one coverage-summary CSV row per requested underlying/session.

        Args:
            anchors: Exchange session dates to include, even if never attempted.

        Returns:
            Counts by collection status and observed/unknown coverage. Damaged
            manifests become explicit failure rows rather than aborting the CSV.
        """
        path = self.store.collection_dir / "availability.csv"
        # Rows and bytes below describe stored records, not distinct market
        # events. Overlapping snapshots and EOD reports must not be interpreted
        # as trade counts.
        columns = (
            "symbol",
            "trade_day",
            "status",
            "coverage_status",
            "reason",
            "quoted_contract_count",
            "traded_contract_count",
            "oi_reported_contract_count",
            "universe_contract_count",
            "selected_contract_count",
            "selection_reference_count",
            "missing_selection_times",
            "request_count",
            "request_error_count",
            "no_data_request_count",
            "missing_option_quote_count",
            "missing_option_eod_count",
            "missing_stock_dataset_count",
            "unknown_required_request_count",
            "stored_rows",
            "excluded_quote_rows",
            "stored_parquet_bytes",
            "error",
        )
        counts = dict.fromkeys(
            ("complete", "unavailable", "request_error", "not_attempted"), 0
        )
        counts.update(days_with_observed_gaps=0, days_with_unknown_coverage=0)
        with storage.atomic_output(path) as temp:
            with temp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for day in anchors:
                    for symbol in self.cfg.symbols:
                        row = {
                            "symbol": symbol.symbol,
                            "trade_day": str(day.date()),
                            "status": "not_attempted",
                            "coverage_status": "not_checked",
                        }
                        try:
                            manifest = storage.read_json(
                                self.store.session_path(symbol.symbol, day)
                            )
                            if (
                                manifest["symbol"] != symbol.symbol
                                or manifest["trade_day"] != str(day.date())
                                or manifest["status"]
                                not in {
                                    "complete",
                                    "unavailable",
                                    "request_error",
                                }
                            ):
                                raise ValueError(
                                    "Session identity or status does not match the requested day"
                                )
                            row.update(
                                {
                                    key: manifest.get(key, "")
                                    for key in columns
                                    if key in manifest
                                }
                            )
                            records = manifest["requests"]
                            coverage = manifest["coverage"]
                            if coverage["status"] not in {
                                "observations_present",
                                "gaps_observed",
                                "unknown",
                            }:
                                raise ValueError(
                                    "Unknown session coverage status"
                                )
                            row.update(
                                coverage_status=coverage["status"],
                                **{
                                    name: coverage[name]
                                    for name in (
                                        "missing_option_quote_count",
                                        "missing_option_eod_count",
                                        "missing_stock_dataset_count",
                                        "unknown_required_request_count",
                                    )
                                },
                            )
                            row.update(
                                selection_reference_count=len(
                                    manifest["stock_selection_references"]
                                ),
                                missing_selection_times="|".join(
                                    manifest["missing_selection_times"]
                                ),
                                request_count=len(records),
                                no_data_request_count=sum(
                                    r["status"] == "no_data" for r in records
                                ),
                                stored_rows=sum(
                                    r.get("row_count", 0) for r in records
                                ),
                                excluded_quote_rows=sum(
                                    r.get("retention", {}).get(
                                        "excluded_rows", 0
                                    )
                                    for r in records
                                ),
                                stored_parquet_bytes=sum(
                                    (r.get("data") or {}).get("size", 0)
                                    for r in records
                                ),
                            )
                        except FileNotFoundError:
                            pass
                        except (
                            OSError,
                            ValueError,
                            KeyError,
                            TypeError,
                            AttributeError,
                        ) as exc:
                            row.update(
                                status="request_error",
                                coverage_status="unknown",
                                error=repr(exc),
                            )
                        counts[row["status"]] += 1
                        counts["days_with_observed_gaps"] += (
                            row["coverage_status"] == "gaps_observed"
                        )
                        counts["days_with_unknown_coverage"] += (
                            row["coverage_status"] == "unknown"
                        )
                        writer.writerow(row)
        return counts
