# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Check selected observations separately from request and storage success.

A successful bulk request can omit a selected contract or sampled timestamp.
CoverageChecker reports those gaps without filling data, dropping quotes, or
deciding whether an observation is suitable for pricing.
"""

import pandas as pd

from tfbsm_collector import planning, storage, validation


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
            unknown, or not_assessed. Empty event/OI reports are not
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

        # Event reports can legitimately be empty. Missing cash events or OI are
        # not automatically failed price collection and are never filled with
        # zero.
        if request.dataset.startswith(
            "corporate_"
        ) or request.endpoint.endswith("/open_interest"):
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
