# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Coordinate symbol-day collection, shared references, and coverage reports.

Collector connects request planning, storage, and contract selection. A session
manifest records the selected sample and request receipts; coverage describes
missing observations separately from download success.
"""

import concurrent.futures
import contextlib
import csv
import dataclasses
import itertools
import pathlib
from typing import Iterable

import pandas as pd

from tfbsm_collector import (
    config,
    coverage,
    planning,
    provenance,
    selection,
    storage,
    transport,
)


class Collector:
    """Coordinator for observed samples, request receipts, and coverage.

    Attributes:
        cfg: Study and resource settings.
        store: Shared transport, response validation, and cache access.
        coverage: Checks required observations in successfully saved responses.
        directory: Collection output directory for this sampling policy.
    """

    def __init__(self, cfg: config.CollectorConfig):
        """Initialize shared storage and policy paths without writing."""
        self.cfg = cfg
        self.store = storage.RequestStore(cfg)
        self.coverage = coverage.CoverageChecker(self.store)
        self.directory = cfg.output_dir / "collection" / cfg.policy_id

    @contextlib.contextmanager
    def workers(self, count: int):
        """Provide a thread pool that stops new collection after interruption.

        Args:
            count: Maximum worker threads in this pool.

        Returns:
            A context manager yielding a ThreadPoolExecutor. On exit, pending
            jobs are cancelled and active jobs finish before releasing
            resources.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=count)
        try:
            yield pool
        # Cancel queued work on interruption while active requests finish saving
        # before the caller releases its output lock.
        except BaseException as exc:
            self.store.client.stop(
                str(exc) or "Collection interrupted by the user."
            )
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def session_path(self, symbol: str, day: pd.Timestamp) -> pathlib.Path:
        """Return the manifest path for one underlying and exchange session."""
        return (
            self.directory
            / "sessions"
            / f"symbol={symbol}__date={day.date()}.json"
        )

    def collect_batch(
        self, requests_to_make: list[planning.Request]
    ) -> Iterable[tuple[planning.Request, dict]]:
        """Yield request receipts as a bounded concurrent batch finishes.

        Args:
            requests_to_make: Small list of descriptors to collect together.

        Yields:
            Pairs (request, receipt) in completion order. Save errors become
            failure receipts and stop new work; completed responses remain
            saved.
        """
        with self.workers(self.cfg.max_batch_workers) as pool:
            futures = {
                pool.submit(self.store.collect, request): request
                for request in requests_to_make
            }
            for future in concurrent.futures.as_completed(futures):
                request = futures[future]
                try:
                    record = future.result()
                except transport.CollectionStopped:
                    continue
                except Exception as exc:
                    self.store.client.stop(
                        f"Unable to save {request.dataset}: {exc}"
                    )
                    record = {
                        "request_id": request.request_id,
                        "dataset": request.dataset,
                        "params": request.params,
                        "status": "request_error",
                        "error": repr(exc),
                    }
                yield request, record

    def manifest_valid(self, manifest: dict) -> bool:
        """Return whether a completed session satisfies the current policy.

        Checks expected request identities, selected/universe tables, coverage
        state, and all referenced artifacts. A complete label alone is not
        enough.
        """
        try:
            if (
                manifest["output_schema_version"]
                != config.OUTPUT_SCHEMA_VERSION
                or manifest["policy_id"] != self.cfg.policy_id
                or manifest["status"] not in {"complete", "unavailable"}
                or manifest["coverage"]["status"]
                not in {"observations_present", "gaps_observed"}
            ):
                return False
            records = manifest["requests"]
            selected_count = manifest["selected_contract_count"]
            if (
                manifest["expected_request_count"] != len(records)
                or manifest["contracts"]["rows"] != selected_count
                or manifest["universe"]["rows"]
                != manifest["universe_contract_count"]
            ):
                return False
            if not all(
                storage.artifact_valid(manifest[name], self.cfg.output_dir)
                for name in ("contracts", "universe")
            ):
                return False
            selected = pd.read_parquet(
                self.cfg.output_dir / manifest["contracts"]["path"]
            )
            day = pd.Timestamp(manifest["trade_day"])
            # Bulk transport changed request counts. Compare the exact expected
            # identities, not a formula assuming a separate download per
            # contract.
            expected = planning.shared_day_requests(
                self.cfg, manifest["symbol"], day
            ) + planning.option_quote_requests(
                self.cfg, manifest["symbol"], day, selected
            )

            if len(records) != len(expected) or {
                r["request_id"] for r in records
            } != {r.request_id for r in expected}:
                return False
            for record in records:
                if record["status"] not in storage.GOOD_REQUEST_STATUSES:
                    return False
                if self.cfg.refresh_no_data and record["status"] == "no_data":
                    return False
                if self.cfg.store_raw_payloads and not record.get("payload"):
                    return False
                for name in ("data", "metadata", "payload"):
                    if name != "payload" or record.get(name):
                        if not storage.artifact_valid(
                            record[name], self.cfg.output_dir
                        ):
                            return False
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def resumable(self, symbol: str, day: pd.Timestamp) -> bool:
        """Return whether this underlying/session can be skipped next run.

        A nonresumable day can still reuse individually valid response caches.
        """
        try:
            manifest = storage.read_json(self.session_path(symbol, day))
            return (
                manifest.get("symbol") == symbol
                and manifest.get("trade_day") == str(day.date())
                and self.manifest_valid(manifest)
            )
        except (OSError, ValueError):
            return False

    def collect_day(
        self, symbol_cfg: config.SymbolConfig, day: pd.Timestamp
    ) -> dict:
        """Collect one underlying/session and publish its selection manifest.

        Args:
            symbol_cfg: Underlying ticker and fixed study labels.
            day: Exchange session date, without a time zone.

        Returns:
            A manifest referencing observed/selected contract tables and
            response receipts. A complete status means requests finished,
            including valid empty replies; the separate coverage field describes
            observed gaps.

        Raises:
            transport.CollectionStopped: The shared run is stopped. Completed
                response files and the attempted session manifest are preserved.
            OSError: A session artifact could not be written.
        """
        symbol = symbol_cfg.symbol
        records, references = [], []
        selected = pd.DataFrame(columns=selection.SELECTION_COLUMNS)
        universe = pd.DataFrame(columns=selection.DISCOVERY_COLUMNS)
        source_counts = dict.fromkeys(("quote", "trade", "open_interest"), 0)
        reason, error = "", ""
        try:
            completed = self.collect_batch(
                planning.shared_day_requests(self.cfg, symbol, day)
            )
            records.extend(record for _, record in completed)
            by_dataset = {record["dataset"]: record for record in records}
            self.store.client.check_running()
            stock = by_dataset[f"stock_quotes_{self.cfg.quote_interval}"]

            # Discovery does not depend on finding a usable stock reference.
            # OI-only contracts remain candidates even when their later quotes
            # are absent.
            discovery = {
                kind: self.store.read(by_dataset[dataset])
                for kind, dataset in (
                    ("quote", "quoted_contracts"),
                    ("trade", "traded_contracts"),
                    ("open_interest", "option_open_interest"),
                )
            }
            universe, source_counts = selection.observed_contract_universe(
                discovery, symbol, self.cfg
            )

            references = selection.stock_selection_references(
                self.store.iter_frames(
                    stock,
                    columns=[
                        "timestamp",
                        "bid",
                        "ask",
                        "bid_condition",
                        "ask_condition",
                    ],
                ),
                day,
                self.cfg,
            )
            # Select the day's retrospective union from stock references. Do not
            # apply option-volume or spread filters before studying liquidity
            # effects.
            selected = selection.select_contracts(
                universe, day, references, self.cfg
            )
            if universe.empty:
                reason = "no_observed_contracts"
            elif not references:
                reason = "stock_selection_reference_unavailable"
            elif selected.empty:
                reason = "no_contracts_in_sampling_window"

            completed = self.collect_batch(
                planning.option_quote_requests(self.cfg, symbol, day, selected)
            )
            records.extend(record for _, record in completed)
            self.store.client.check_running()
        except Exception as exc:
            error = repr(exc)
        failures = sum(
            record["status"] not in storage.GOOD_REQUEST_STATUSES
            for record in records
        )

        # Complete means the requests finished, not complete market coverage.
        # Unavailable means selection found no contracts without a request
        # error.
        status = (
            "request_error"
            if error or failures
            else ("unavailable" if selected.empty else "complete")
        )
        contract_path = (
            self.directory
            / "contracts"
            / f"symbol={symbol}__date={day.date()}.parquet"
        )
        universe_path = (
            self.directory
            / "universes"
            / f"symbol={symbol}__date={day.date()}.parquet"
        )
        storage.write_parquet(contract_path, selected)
        storage.write_parquet(universe_path, universe)

        opened, closed = planning.session_bounds(day, self.cfg)
        scheduled = [
            at
            for at in self.cfg.selection_times
            if opened
            <= pd.Timestamp(f"{day.date()} {at}", tz=self.cfg.exchange_tz)
            < closed
        ]
        missing_times = [
            at
            for at in scheduled
            if at not in {r["selection_time"] for r in references}
        ]
        coverage = self.coverage.session_coverage(
            symbol,
            day,
            selected,
            records,
            missing_times,
            status == "request_error",
        )
        manifest = {
            **dataclasses.asdict(symbol_cfg),
            "trade_day": str(day.date()),
            "status": status,
            "reason": reason,
            "error": error,
            "output_schema_version": config.OUTPUT_SCHEMA_VERSION,
            "policy_id": self.cfg.policy_id,
            "updated_at_utc": provenance.utc_now(),
            "session_open": opened.isoformat(),
            "session_close": closed.isoformat(),
            "intraday_window": "underlying_regular_trading_session",
            "quote_interval": self.cfg.quote_interval,
            "near_close_time": (
                closed - pd.Timedelta(minutes=self.cfg.near_close_minutes)
            ).strftime("%H:%M:%S"),
            "option_quote_batching": "all_strikes_and_rights_per_selected_expiration",
            "option_quote_retention": "selected_contracts",
            "daily_activity_source": "Theta stock/option EOD volume and count; no individual trades",
            "discovery_scope": "union of dated quote/trade lists and prior-session OI reports within max_dte; complete listing coverage unverified",
            "discovery_max_dte": self.cfg.max_dte,
            "quoted_contract_count": source_counts["quote"],
            "traded_contract_count": source_counts["trade"],
            "oi_reported_contract_count": source_counts["open_interest"],
            "universe_contract_count": len(universe),
            "selected_contract_count": len(selected),
            "stock_selection_references": references,
            "missing_selection_times": missing_times,
            "coverage": coverage,
            "requests": sorted(
                records, key=lambda r: (r["dataset"], r["request_id"])
            ),
            "contracts": storage.file_receipt(
                contract_path, self.cfg.output_dir, selected
            ),
            "universe": storage.file_receipt(
                universe_path, self.cfg.output_dir, universe
            ),
            "request_error_count": failures,
            "expected_request_count": 7 + 2 * selected["expiration"].nunique(),
        }

        # Publish the session manifest after every referenced artifact exists.
        storage.write_json(self.session_path(symbol, day), manifest)
        self.store.client.check_running()
        return manifest

    def collect_references(
        self,
        symbols: list[config.SymbolConfig],
        start: str,
        end: str,
        rate_symbols: list[str],
        run_id: str,
        *,
        include_stock_lookback: bool = False,
    ) -> dict:
        """Collect the shared reference bundle, publishing progress by batch.

        Args:
            symbols: Underlyings needing corporate actions and optional
                lookback.
            start: Inclusive study start in YYYY-MM-DD form.
            end: Inclusive study end in YYYY-MM-DD form.
            rate_symbols: Theta rate series identifiers.
            run_id: Identifier naming this run's reference ledger.
            include_stock_lookback: Whether to add earlier stock quotes and EOD.

        Returns:
            A ledger of request receipts, coverage, and interpretation limits.
            Inaccessible Standard VIX dates are recorded without requesting
            them.
        """
        records = []

        lookback_datasets = (
            [
                f"stock_quotes_{self.cfg.quote_interval}",
                "stock_quotes_near_close",
                "stock_eod",
            ]
            if include_stock_lookback and self.cfg.lookback_sessions
            else []
        )
        windows = planning.collection_windows(self.cfg, start, end)
        index_excluded = [
            date
            for date in planning.exchange_calendar()
            .sessions_in_range(windows["history_start"], end)
            .strftime("%Y-%m-%d")
            if date < self.cfg.index_history_start
        ]
        # Earlier VIX sessions are subscription gaps. Recording them keeps the
        # requested study window visible without sending inaccessible history
        # pulls.
        access_gaps = (
            [
                {
                    "symbol": "VIX",
                    "reason": "before_standard_index_history_start",
                    "access_start": self.cfg.index_history_start,
                    "unrequested_eod_session_dates": index_excluded,
                    "unrequested_intraday_session_dates": [
                        date for date in index_excluded if date >= start
                    ],
                }
            ]
            if index_excluded
            else []
        )
        ledger = {
            "vendor": "ThetaData",
            "start": start,
            "end": end,
            "collection_windows": windows,
            "subscription_coverage_gaps": access_gaps,
            "stock_lookback_requested": bool(lookback_datasets),
            "option_contract_continuity_guaranteed": False,
            "requested_rates": sorted(set(rate_symbols)),
            "rate_units": "percent",
            "required_datasets": [
                "corporate_dividend",
                "corporate_split",
                "interest_rate_eod",
                "index_eod",
                f"index_prices_{self.cfg.quote_interval}",
                "index_prices_near_close",
                *lookback_datasets,
            ],
            "rate_publication_timestamps_verified": False,
            "corporate_action_range_filters": {
                "dividend": "ex_dividend_date",
                "split": "effective_date",
            },
            "missing_dividend_amounts": "unknown; not zero",
            "split_ratio": "before_shares / after_shares",
            "empty_actions_prove_complete_event_coverage": False,
            "later_actions_were_known_at_study_time": "not_assumed; retain announcement dates and unknown values",
            "index_unchanged_updates_may_be_omitted": True,
            "adjusted_contract_deliverables": "not_documented_by_Theta; not_inferred",
            "historical_symbol_mappings": "not_verified",
        }
        path = self.cfg.output_dir / "references" / f"{run_id}.json"

        def publish():
            ledger.update(
                updated_at_utc=provenance.utc_now(),
                requests=records,
                requests_with_observed_gaps=sum(
                    r["coverage"]["status"] == "gaps_observed" for r in records
                ),
                requests_with_unknown_coverage=sum(
                    r["coverage"]["status"] == "unknown" for r in records
                ),
            )
            storage.write_json(path, ledger)

        try:
            pending = iter(
                planning.reference_requests(
                    self.cfg,
                    symbols,
                    start,
                    end,
                    rate_symbols,
                    include_stock_lookback=include_stock_lookback,
                )
            )
            while not self.store.client.stop_event.is_set():
                # Bound the scheduled reference queue; do not enqueue eight
                # years of work at once. Completed batches remain reusable after
                # interruption.
                batch = list(itertools.islice(pending, 32))
                if not batch:
                    break
                for request, record in self.collect_batch(batch):
                    records.append(
                        {
                            **record,
                            "params": request.params,
                            "coverage": self.coverage.request_coverage(
                                request, record
                            ),
                        }
                    )
                publish()
                print(f"References: {len(records)} requests recorded")
        finally:
            publish()
        return ledger

    def collect_coverage(
        self,
        symbols: list[config.SymbolConfig],
        anchors: pd.DatetimeIndex,
        run_id: str,
    ) -> dict:
        """Refresh stock/VIX date catalogues and record advertised date gaps.

        Args:
            symbols: Underlyings to check for stock quote/trade availability.
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
            for symbol in symbols
            for kind in ("quote", "trade")
        ]
        requests_to_make.append(
            planning.Request(
                "index_price_dates",
                "/index/list/dates",
                {"symbol": "VIX", "format": "csv"},
            )
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
                        "status": "request_error",
                        "error": repr(exc),
                    }
                records.append({**record, "params": request.params})
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
                "unfinished_requests": [
                    r.identity() for r in requests_to_make[len(rows) :]
                ],
                "request_errors": sum(
                    row["status"] == "request_error" for row in rows
                ),
                "series_with_gaps": sum(
                    row["status"] == "coverage_gap" for row in rows
                ),
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

    def write_availability(
        self, symbols: list[config.SymbolConfig], anchors: pd.DatetimeIndex
    ) -> dict:
        """Write one coverage-summary CSV row per requested underlying/session.

        Args:
            symbols: Underlyings in the requested run scope.
            anchors: Exchange session dates to include, even if never attempted.

        Returns:
            Counts by collection status and observed/unknown coverage. Damaged
            manifests become explicit failure rows rather than aborting the CSV.
        """
        path = self.directory / "availability.csv"
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
                    for symbol in symbols:
                        row = {
                            "symbol": symbol.symbol,
                            "trade_day": str(day.date()),
                            "status": "not_attempted",
                            "coverage_status": "not_checked",
                        }
                        try:
                            manifest = storage.read_json(
                                self.session_path(symbol.symbol, day)
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
