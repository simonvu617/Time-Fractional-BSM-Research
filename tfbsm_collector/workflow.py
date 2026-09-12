# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Own run order, session collection, shared references, and cancellation.

Start at Collector.run: date coverage, reference inputs, then stock/option
sessions. Storage owns files and cache reuse; CoverageChecker owns coverage
reports. Session manifests connect the selected sample to request receipts.
"""

import concurrent.futures
import contextlib
import dataclasses
import itertools
import platform
import sys
import uuid
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
    """

    def __init__(self, cfg: config.CollectorConfig):
        """Initialize shared storage and policy paths without writing."""
        self.cfg = cfg
        self.store = storage.RequestStore(cfg)
        self.coverage = coverage.CoverageChecker(self.store)

    def run(self) -> int:
        """Collect the configured scope and publish a run record, even on failure.

        Returns:
            Zero when no gaps were reported, or two when collection finished
            with coverage gaps. Download success does not validate a model.

        Raises:
            RuntimeError: The terminal is unavailable, access fails, or another
                collector owns the output directory.
            OSError: Run artifacts could not be written.
        """
        cfg = self.cfg
        windows = planning.collection_windows(cfg)
        # Session labels exclude weekends/holidays; requests obtain their
        # timezone-aware market boundaries from the same exchange calendar.
        anchors = (
            planning.exchange_calendar()
            .sessions_in_range(cfg.start_date, cfg.end_date)
            .tz_localize(None)
        )
        run_id = (
            pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        path = self.store.collection_dir / "runs" / f"{run_id}.json"
        run = {
            "run_id": run_id,
            "started_at_utc": provenance.utc_now(),
            "status": "running",
            "data_vendor": "ThetaData",
            "policy_id": cfg.policy_id,
            "policy": cfg.policy(),
            "config": {
                **dataclasses.asdict(cfg),
                "output_dir": str(cfg.output_dir),
            },
            "scope": {
                "symbols": [s.symbol for s in cfg.symbols],
                "start": cfg.start_date,
                "end": cfg.end_date,
                "collection_windows": windows,
                "references_only": cfg.mode == "references",
                "coverage_only": cfg.mode == "coverage",
                "rate_symbols": list(cfg.rate_symbols),
            },
            "code_sha256": self.store.code_sha256,
            "code_files": self.store.code_files,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": provenance.package_versions(),
            **dict.fromkeys(
                (
                    "resumed_days",
                    "processed_days",
                    "failed_days",
                    "reference_failures",
                    "reference_gaps",
                    "catalogue_errors",
                    "catalogue_series_with_gaps",
                ),
                0,
            ),
        }
        self.store.client.ensure_available()
        # Keep one owner for run order, cancellation, and final status. The CLI
        # only parses options; it does not start its own collection workers.
        with storage.output_lock(cfg.output_dir):
            storage.write_json(path, run)
            try:
                if cfg.mode != "references":
                    run["date_catalogue"] = f"coverage/{run_id}.json"
                    catalogue_dates = (
                        planning.exchange_calendar()
                        .sessions_in_range(
                            windows["history_start"], cfg.end_date
                        )
                        .tz_localize(None)
                    )
                    catalogue = self.coverage.collect_catalogue(
                        catalogue_dates, run_id
                    )
                    run["catalogue_errors"] = catalogue["request_errors"]
                    run["catalogue_series_with_gaps"] = catalogue[
                        "series_with_gaps"
                    ]
                    self.store.client.check_running()
                if cfg.mode != "coverage":
                    run["reference_ledger"] = f"references/{run_id}.json"
                    references = self.collect_references(run_id)
                    run["reference_failures"] = sum(
                        r["status"] not in storage.GOOD_REQUEST_STATUSES
                        or r["coverage"]["status"] == "unknown"
                        for r in references["requests"]
                    )
                    run["reference_gaps"] = references[
                        "requests_with_observed_gaps"
                    ] + len(references["access_coverage_gaps"])
                    self.store.client.check_running()
                if cfg.mode == "panels":
                    self._collect_panels(anchors, run)
            except BaseException as exc:
                self.store.client.stop(
                    str(exc) or "Collection interrupted by the user."
                )
                run.update(
                    status="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "partial_failure",
                    error=repr(exc),
                )
                raise
            finally:
                exit_code = self._finish_run(run, anchors)
                storage.write_json(path, run)
                print(f"Run record: {path}")
        print(
            f"Finished: {run['processed_days']} processed, {run['resumed_days']} resumed, "
            f"{run['failed_days']} failed days, {run['reference_failures']} failed reference requests, "
            f"{run['reference_gaps'] + run['catalogue_series_with_gaps']} reference/catalogue gaps"
        )
        for name in ("date_catalogue", "reference_ledger"):
            if name in run:
                print(f"{name}: {cfg.output_dir / run[name]}")
        if cfg.mode == "panels":
            print(
                f"Panel coverage: {self.store.collection_dir / 'availability.csv'}"
            )
        return exit_code

    def _collect_panels(self, anchors: pd.DatetimeIndex, run: dict) -> None:
        """Collect pending underlyings one session at a time and update counts."""
        for day in anchors:
            pending = []
            for symbol in self.cfg.symbols:
                if self.resumable(symbol.symbol, day):
                    run["resumed_days"] += 1
                else:
                    pending.append(symbol)
            with self.workers(self.cfg.max_symbol_day_workers) as pool:
                futures = {
                    pool.submit(self.collect_day, symbol, day): symbol
                    for symbol in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    run["processed_days"] += 1
                    try:
                        manifest = future.result()
                        run["failed_days"] += (
                            manifest["status"] == "request_error"
                        )
                        print(
                            f"{symbol.symbol} {day.date()}: {manifest['status']}; "
                            f"coverage: {manifest['coverage']['status']}; "
                            f"{manifest['selected_contract_count']} contracts, "
                            f"{manifest['request_error_count']} failed requests"
                        )
                    except transport.CollectionStopped:
                        raise
                    except Exception as exc:
                        run["failed_days"] += 1
                        print(f"FAILED {symbol.symbol} {day.date()}: {exc!r}")

    def _finish_run(self, run: dict, anchors: pd.DatetimeIndex) -> int:
        """Summarize saved coverage and choose the run outcome in one place."""
        try:
            run["coverage"] = (
                self.coverage.write_availability(anchors)
                if self.cfg.mode == "panels"
                else {}
            )
        except Exception as exc:
            run["coverage_error"] = repr(exc)
        observed = run.get("coverage", {})
        if run["status"] == "running":
            # Failure takes priority over missing observations. A successful
            # download can still leave a research coverage gap.
            failed = (
                "coverage_error" in run
                or any(
                    run[name]
                    for name in (
                        "failed_days",
                        "reference_failures",
                        "catalogue_errors",
                    )
                )
                or any(
                    observed.get(name, 0)
                    for name in (
                        "request_error",
                        "not_attempted",
                        "days_with_unknown_coverage",
                    )
                )
            )
            gaps = (
                run["reference_gaps"]
                or run["catalogue_series_with_gaps"]
                or any(
                    observed.get(name, 0)
                    for name in ("unavailable", "days_with_observed_gaps")
                )
            )
            run["status"] = (
                "partial_failure"
                if failed
                else "coverage_gaps"
                if gaps
                else "complete"
            )
        run.update(
            finished_at_utc=provenance.utc_now(),
            failed_days=max(
                run["failed_days"], observed.get("request_error", 0)
            ),
        )
        return {"complete": 0, "coverage_gaps": 2}.get(run["status"], 1)

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
            return all(self.store.reusable(record) for record in records)
        except (KeyError, TypeError, ValueError):
            return False

    def resumable(self, symbol: str, day: pd.Timestamp) -> bool:
        """Return whether this underlying/session can be skipped next run.

        A nonresumable day can still reuse individually valid response caches.
        """
        try:
            manifest = storage.read_json(self.store.session_path(symbol, day))
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
            self.store.collection_dir
            / "contracts"
            / f"symbol={symbol}__date={day.date()}.parquet"
        )
        universe_path = (
            self.store.collection_dir
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
        storage.write_json(self.store.session_path(symbol, day), manifest)
        self.store.client.check_running()
        return manifest

    def collect_references(
        self,
        run_id: str,
    ) -> dict:
        """Collect the shared reference bundle, publishing progress by batch.

        Args:
            run_id: Identifier naming this run's reference ledger.

        Returns:
            A ledger of request receipts, coverage, and interpretation limits.
            Inaccessible reference dates are recorded without requesting them.
        """
        records = []
        include_stock_lookback = self.cfg.mode == "panels"
        lookback_datasets = (
            [
                f"stock_quotes_{self.cfg.quote_interval}",
                "stock_quotes_near_close",
                "stock_eod",
            ]
            if include_stock_lookback and self.cfg.lookback_sessions
            else []
        )
        windows = planning.collection_windows(self.cfg)
        access_gaps = planning.reference_access_gaps(self.cfg, windows)
        ledger = {
            "vendor": "ThetaData",
            "start": self.cfg.start_date,
            "end": self.cfg.end_date,
            "collection_windows": windows,
            "index_subscription": self.cfg.index_subscription,
            "rate_subscription": self.cfg.rate_subscription,
            "access_coverage_gaps": access_gaps,
            "stock_lookback_requested": bool(lookback_datasets),
            "option_contract_continuity_guaranteed": False,
            "requested_rates": sorted(set(self.cfg.rate_symbols)),
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
            "missing_dividend_amounts": "unknown; not zero",
            "corporate_actions_available": False,
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
            pending = planning.reference_requests(self.cfg)
            while not self.store.client.stop_event.is_set():
                # Bound the scheduled reference queue instead of enqueuing the
                # full history. Completed batches remain reusable after
                # interruption.
                batch = list(itertools.islice(pending, 32))
                if not batch:
                    break
                for request, record in self.collect_batch(batch):
                    records.append(
                        {
                            **record,
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
