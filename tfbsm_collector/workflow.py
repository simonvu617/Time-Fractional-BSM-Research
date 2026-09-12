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
        self.unavailable_dates = {}
        self.catalogue_path = None

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
                            windows["history_start"],
                            windows["available_followup_end"],
                        )
                        .tz_localize(None)
                    )
                    catalogue = self.coverage.collect_catalogue(
                        catalogue_dates, run_id
                    )
                    self.catalogue_path = run["date_catalogue"]
                    for row in catalogue["rows"]:
                        if row["missing_requested_dates"] is not None:
                            kind = row["dataset"].split("_")[1]
                            self.unavailable_dates[(row["symbol"], kind)] = set(
                                row["missing_requested_dates"]
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
        """Keep independent symbols busy while each cohort advances in order."""
        with self.workers(self.cfg.max_symbol_workers) as pool:
            futures = {
                pool.submit(self.collect_symbol, symbol): symbol
                for symbol in self.cfg.symbols
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    counts = future.result()
                    for key, value in counts.items():
                        run[key] = run.get(key, 0) + value
                except transport.CollectionStopped:
                    raise
                except Exception as exc:
                    run["failed_days"] += 1
                    print(f"FAILED {futures[future].symbol}: {exc!r}")

    def collect_symbol(self, symbol: config.SymbolConfig) -> dict:
        """Advance one option root in monthly checkpoints through expiration.

        Different roots can run concurrently. Months for the same root cannot:
        their incoming cohort is part of the next month's resume identity.
        """
        windows = planning.collection_windows(self.cfg)
        days = (
            planning.exchange_calendar()
            .sessions_in_range(
                self.cfg.start_date, windows["available_followup_end"]
            )
            .tz_localize(None)
        )
        cohort = pd.DataFrame(columns=selection.COHORT_COLUMNS)
        counts = dict(
            resumed_days=0,
            processed_days=0,
            failed_days=0,
            contracts_pending_future_expiration=0,
        )
        # Finish entry before planning the follow-up tail. This also avoids
        # requesting the rest of a month after a short cohort has expired.
        entry = days[days <= pd.Timestamp(self.cfg.end_date)]
        followup = days[days > pd.Timestamp(self.cfg.end_date)]
        batches = itertools.chain(
            planning.date_batches(entry, self.cfg, intraday=False),
            planning.date_batches(followup, self.cfg, intraday=False),
        )
        for month in batches:
            # After the entry window, stop when the last enrolled contract
            # expires. Do not start new cohorts in this follow-up tail.
            if str(month[0].date()) > self.cfg.end_date:
                if cohort.empty:
                    break
                month = month[month <= pd.Timestamp(cohort["expiration"].max())]
                if month.empty:
                    break
            cohort = cohort.loc[cohort["expiration"].ge(str(month[0].date()))]
            incoming = provenance.digest_json(cohort.to_dict("records"))
            path = (
                self.store.collection_dir
                / "months"
                / symbol.symbol
                / f"{month[0]:%Y-%m-%d}_{month[-1]:%Y-%m-%d}.json"
            )
            scope = {
                "policy_id": self.cfg.policy_id,
                "symbol": dataclasses.asdict(symbol),
                "dates": list(month.strftime("%Y-%m-%d")),
                "incoming_cohort": incoming,
                "index_subscription": self.cfg.index_subscription,
                "unavailable_underlying_dates": {
                    kind: sorted(set(month.strftime("%Y-%m-%d")) & excluded)
                    for (
                        ticker,
                        kind,
                    ), excluded in self.unavailable_dates.items()
                    if ticker == symbol.underlying
                },
            }
            try:
                previous = storage.read_json(path) if path.exists() else {}
            except (OSError, ValueError):
                previous = {}
            if self.month_valid(previous, scope):
                cohort = pd.read_parquet(
                    self.cfg.output_dir / previous["cohort"]["path"]
                )
                counts["resumed_days"] += len(month)
                continue
            manifest = self.collect_month(symbol, month, cohort, scope)
            storage.write_json(path, manifest)
            counts["processed_days"] += len(month)
            if manifest["request_error_count"]:
                counts["failed_days"] += len(month)
                # A failed discovery month could change every later cohort.
                # Retry it on resume before advancing this root's history.
                break
            cohort = pd.read_parquet(
                self.cfg.output_dir / manifest["cohort"]["path"]
            )
            print(
                f"{symbol.symbol} {month[0]:%Y-%m}: {len(month)} sessions; "
                f"{manifest['request_count']} requests; {len(cohort)} tracked contracts"
            )
        counts["contracts_pending_future_expiration"] = int(
            cohort["expiration"].gt(windows["available_followup_end"]).sum()
        )
        return counts

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
                run.get("contracts_pending_future_expiration", 0)
                or run["reference_gaps"]
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

    def month_valid(self, manifest: dict, scope: dict) -> bool:
        """Reuse a month only with the same entry history and intact artifacts."""
        try:
            return (
                manifest["scope"] == scope
                and manifest["request_error_count"] == 0
                and all(
                    storage.artifact_valid(manifest[name], self.cfg.output_dir)
                    for name in ("contracts", "universes", "cohort")
                )
                and all(
                    self.store.reusable(record)
                    for record in manifest["requests"]
                )
                and all(
                    self.store.session(
                        scope["symbol"]["symbol"], pd.Timestamp(day)
                    )
                    for day in scope["dates"]
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def collect_month(
        self,
        symbol: config.SymbolConfig,
        days: pd.DatetimeIndex,
        cohort: pd.DataFrame,
        scope: dict,
    ) -> dict:
        """Discover dated entries, extend cohorts, then batch their observations.

        The full first selection day is retained for retrospective research.
        first_selected_date/time records when that membership became observable;
        the data are not a claim that afternoon selections were known at 09:30.
        """
        discovery_days = days
        if symbol.price_asset == "index":
            discovery_days = (
                days[:0]
                if self.cfg.index_history_start is None
                else days[days >= pd.Timestamp(self.cfg.index_history_start)]
            )
        # Before any possible entry, an explicitly unlisted underlying quote
        # date cannot supply the S/K reference. Avoid downloading broad option
        # chains for that prefix. Once a cohort can exist, keep following it
        # even through later underlying-data gaps.
        quote_kind = "price" if symbol.price_asset == "index" else "quote"
        excluded = self.unavailable_dates.get(
            (symbol.underlying, quote_kind), set()
        )
        possible_entries = [
            d
            for d in discovery_days
            if planning.is_enrollment_day(d, self.cfg)
            and str(d.date()) not in excluded
        ]
        if cohort.empty:
            discovery_days = (
                discovery_days[discovery_days >= possible_entries[0]]
                if possible_entries
                else discovery_days[:0]
            )
        requests = planning.underlying_requests(
            self.cfg, symbol, days, self.unavailable_dates
        ) + planning.discovery_requests(self.cfg, symbol.symbol, discovery_days)
        records = dict(
            (request.request_id, record)
            for request, record in self.collect_batch(requests)
        )
        self.store.client.check_running()
        daily = []
        selections, universes = [], []

        def matching(dataset, day):
            return [
                (r, records.get(r.request_id))
                for r in requests
                if r.dataset == dataset and day in planning.request_days(r)
            ]

        def frames(dataset, day):
            for _, record in matching(dataset, day):
                if record:
                    yield from self.store.iter_frames(record, day=day)

        def table(dataset, day):
            parts = list(frames(dataset, day))
            return (
                pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            )

        for day in days:
            entry_day = str(day.date()) <= self.cfg.end_date
            enroll = planning.is_enrollment_day(day, self.cfg)
            discovery = {
                kind: table(dataset, day)
                for kind, dataset in (
                    ("quote", "quoted_contracts"),
                    ("trade", "traded_contracts"),
                    ("open_interest", "option_open_interest"),
                )
            }
            universe, source_counts = selection.observed_contract_universe(
                discovery, symbol.symbol, self.cfg
            )
            price_kind = "prices" if symbol.price_asset == "index" else "quotes"
            price_dataset = (
                f"{symbol.price_asset}_{price_kind}_{self.cfg.quote_interval}"
            )
            reference_builder = (
                selection.index_selection_references
                if symbol.price_asset == "index"
                else selection.stock_selection_references
            )
            references = (
                reference_builder(frames(price_dataset, day), day, self.cfg)
                if enroll
                else []
            )
            chosen = (
                selection.select_contracts(universe, day, references, self.cfg)
                if enroll
                else pd.DataFrame(columns=selection.SELECTION_COLUMNS)
            )
            prior_count = len(cohort)
            cohort = selection.extend_cohort(cohort, chosen, day, symbol)
            active = cohort.loc[cohort["expiration"].ge(str(day.date()))].copy()
            active["dte_days"] = (
                pd.to_datetime(active["expiration"]) - day
            ).dt.days
            selections.append(active.assign(trade_day=str(day.date())))
            universes.append(universe.assign(trade_day=str(day.date())))
            opened, closed = planning.session_bounds(day, self.cfg)
            scheduled = [
                at
                for at in (self.cfg.selection_times if enroll else ())
                if opened
                <= pd.Timestamp(f"{day.date()} {at}", tz=self.cfg.exchange_tz)
                < closed
            ]
            missing = [
                at
                for at in scheduled
                if at
                not in {reference["selection_time"] for reference in references}
            ]
            daily.append(
                {
                    "symbol": symbol.symbol,
                    "underlying_symbol": symbol.underlying,
                    "trade_day": str(day.date()),
                    "entry_window": entry_day,
                    "enrollment_scheduled": enroll,
                    "policy_id": self.cfg.policy_id,
                    "newly_selected_contract_count": len(cohort) - prior_count,
                    "selected_contract_count": len(active),
                    "universe_contract_count": len(universe),
                    "quoted_contract_count": source_counts["quote"],
                    "traded_contract_count": source_counts["trade"],
                    "oi_reported_contract_count": source_counts[
                        "open_interest"
                    ],
                    "stock_selection_references": references,
                    "missing_selection_times": missing,
                    "session_open": opened.isoformat(),
                    "session_close": closed.isoformat(),
                    "selection_timing": "retrospective_first_day_union; entry_clock_saved",
                    "unavailable_underlying_inputs": self.underlying_gaps(
                        symbol, day
                    ),
                }
            )
        quote_requests = planning.followup_requests(
            self.cfg, symbol.symbol, days, cohort
        )
        records.update(
            (request.request_id, record)
            for request, record in self.collect_batch(quote_requests)
        )
        requests += quote_requests
        self.store.client.check_running()
        selected_table = pd.concat(selections, ignore_index=True)
        universe_table = pd.concat(universes, ignore_index=True)
        observations = {
            request.request_id: self.coverage.observation_clocks(
                request, records.get(request.request_id)
            )
            for request in requests
            if "/list/" not in request.endpoint
            and not request.endpoint.endswith("/open_interest")
        }
        for manifest, selected in zip(daily, selections):
            day = pd.Timestamp(manifest["trade_day"])
            required = [r for r in requests if day in planning.request_days(r)]
            receipts = [
                records[r.request_id]
                for r in required
                if r.request_id in records
            ]
            failures = sum(
                r["status"] not in storage.GOOD_REQUEST_STATUSES
                for r in receipts
            )
            failures += len(required) - len(receipts)
            checks = self.coverage.session_coverage(
                symbol.symbol,
                day,
                selected,
                receipts,
                manifest["missing_selection_times"],
                bool(failures),
                requests=required,
                observations=observations,
            )
            if (
                manifest["unavailable_underlying_inputs"]
                and checks["status"] != "unknown"
            ):
                checks["status"] = "gaps_observed"
            manifest.update(
                status="request_error"
                if failures
                else "unavailable"
                if selected.empty
                else "complete",
                reason="no_selected_contracts" if selected.empty else "",
                request_error_count=failures,
                coverage=checks,
                # Shared monthly receipts are referenced, not copied into every
                # day's JSON. The request index resolves their Parquet locations.
                request_ids=[r["request_id"] for r in receipts],
                no_data_request_count=sum(
                    r["status"] == "no_data" for r in receipts
                ),
            )
        all_records = list(records.values())
        self.store.compact(
            all_records, f"symbol={symbol.symbol}/month={days[0]:%Y-%m}"
        )
        # Publish the current shared file locations, not pre-compaction paths.
        by_id = {request.request_id: request for request in requests}
        all_records = [
            self.store.record(by_id[r["request_id"]], self.store.metadata(r))
            if "attempt_id" in r
            else r
            for r in all_records
        ]
        directory = (
            self.store.collection_dir
            / "tables"
            / symbol.symbol
            / f"{days[0]:%Y-%m}"
            / uuid.uuid4().hex
        )
        artifacts = {}
        for name, frame in (
            ("contracts", selected_table),
            ("universes", universe_table),
            (
                "cohort",
                cohort.loc[cohort["expiration"].ge(str(days[-1].date()))],
            ),
        ):
            path = directory / f"{name}.parquet"
            storage.write_parquet(path, frame)
            artifacts[name] = storage.file_receipt(
                path, self.cfg.output_dir, frame
            )
        self.store.save_sessions(daily)
        return {
            "scope": scope,
            **artifacts,
            "updated_at_utc": provenance.utc_now(),
            "request_count": len(requests),
            "requests": all_records,
            "request_error_count": sum(m["request_error_count"] for m in daily),
            "contract_terms": "product labels only; missing exact deliverables and last trading times",
            "intraday_window": "underlying XNYS regular session; not full index options session",
        }

    def underlying_gaps(
        self, symbol: config.SymbolConfig, day: pd.Timestamp
    ) -> list[dict]:
        """Explain intentionally unrequested underlying inputs without zero filling."""
        gaps = []
        if symbol.price_asset == "index" and (
            self.cfg.index_history_start is None
            or str(day.date()) < self.cfg.index_history_start
        ):
            gaps.append(
                {
                    "reason": "index_subscription_history_unavailable",
                    "symbol": symbol.underlying,
                }
            )
        for (ticker, kind), excluded in self.unavailable_dates.items():
            if ticker == symbol.underlying and str(day.date()) in excluded:
                gaps.append(
                    {
                        "reason": "date_not_listed_by_vendor",
                        "kind": kind,
                        "symbol": ticker,
                        "catalogue": self.catalogue_path,
                    }
                )
        return gaps

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
                f"stock_ohlc_{self.cfg.quote_interval}",
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
            "option_contract_followup": "first selection through expiration",
            "observed_contract_continuity_guaranteed": False,
            "requested_rates": sorted(set(self.cfg.rate_symbols)),
            "rate_units": "percent",
            "required_datasets": [
                "corporate_dividend",
                "corporate_split",
                "option_contract_terms",
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
            "contract_terms_gaps": [
                "verified_multiplier",
                "adjusted_deliverable",
                "last_trading_timestamp",
                "historical_exercise_and_settlement_terms",
            ],
            "index_option_roots": {"SPX": "SPX", "SPXW": "SPX"},
            "product_context_sources": [
                "https://docs.thetadata.us/Articles/Data-And-Requests/Symbology.html",
                "https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications",
            ],
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
