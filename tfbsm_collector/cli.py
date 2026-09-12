# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Collect ThetaData inputs for the hourly/daily TFBSM pricing study.

Requires Python 3.11+ and the packages in requirements.txt. Theta Terminal v3
must be running for downloads.

Preview a small run without downloading or creating output:
    python collector.py --symbols SPY --plan

Start reading in config.py for study settings and workflow.py for collection.
Under --output-dir, collection/<policy-id>/availability.csv summarizes coverage;
contracts/ and universes/ record selection, and raw_cache/ holds observations
and response metadata. Pricing and calibration are separate research work.
"""

import argparse
import concurrent.futures
import dataclasses
import pathlib
import platform
import re
import sys
import uuid

import pandas as pd

from tfbsm_collector import (
    config,
    planning,
    provenance,
    storage,
    transport,
    workflow,
)


def parse_run_scope(argv: list[str] | None = None):
    """Parse and validate command-line scope without downloading or writing.

    Args:
        argv: Arguments after the program name, or None to use sys.argv.

    Returns:
        A tuple (args, cfg, symbols, anchors): parsed options, configuration,
        underlying labels, and timezone-naive exchange session dates.

    Raises:
        SystemExit: Help was requested or an argument is invalid.
    """
    defaults = config.CollectorConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        type=str.upper,
        choices=[cfg.symbol for cfg in config.UNIVERSE],
    )
    parser.add_argument(
        "--start",
        default=defaults.start_date,
        help="Inclusive first date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=defaults.end_date,
        help="Inclusive last date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lookback-sessions",
        type=int,
        default=defaults.lookback_sessions,
        help="Prior stock/rate trading sessions to collect (default: 60; 0 disables the buffer)",
    )
    parser.add_argument(
        "--quote-interval",
        default=defaults.quote_interval,
        choices=config.QUOTE_INTERVALS,
    )
    parser.add_argument(
        "--raw-chunk-rows",
        type=int,
        default=defaults.raw_chunk_rows,
        help="Rows per parsing/storage batch; affects memory use, not which rows are saved",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=defaults.output_dir
    )
    parser.add_argument(
        "--store-raw-payloads",
        action="store_true",
        help="Also preserve exact Theta response bytes",
    )
    parser.add_argument(
        "--refresh-no-data",
        action="store_true",
        help="Retry previously empty requests",
    )
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=defaults.max_inflight_requests,
        help="1 through 4 for Standard; lower this if another client shares the account",
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=float,
        default=defaults.max_requests_per_second,
        help="Optional request-start pacing; default 0 means only the concurrency cap applies",
    )
    parser.add_argument(
        "--rate-symbols",
        nargs="+",
        default=list(config.RATE_SYMBOLS),
        type=str.upper,
        choices=config.RATE_SYMBOLS,
        help="Theta rate series to collect; defaults to SOFR and all documented Treasury tenors",
    )
    modes = parser.add_mutually_exclusive_group()

    modes.add_argument(
        "--references-only",
        action="store_true",
        help="Collect dividends, splits, rates, and VIX without stock/option panels",
    )
    modes.add_argument(
        "--coverage-only",
        action="store_true",
        help="Refresh stock/VIX available-date lists and report missing sessions",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show scope without network requests or output writes",
    )
    args = parser.parse_args(argv)
    try:
        if not all(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            for value in (args.start, args.end)
        ):
            raise ValueError("Dates must use YYYY-MM-DD")
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if (
            not pd.Timestamp(defaults.start_date)
            <= start
            <= end
            <= pd.Timestamp(defaults.end_date)
        ):
            raise ValueError(
                f"Dates must be ordered and within {defaults.start_date} through {defaults.end_date}"
            )
        cfg = dataclasses.replace(
            defaults,
            quote_interval=args.quote_interval,
            output_dir=args.output_dir.expanduser().resolve(),
            raw_chunk_rows=args.raw_chunk_rows,
            lookback_sessions=args.lookback_sessions,
            store_raw_payloads=args.store_raw_payloads,
            refresh_no_data=args.refresh_no_data,
            max_inflight_requests=args.max_inflight_requests,
            max_requests_per_second=args.max_requests_per_second,
        )
        planning.collection_windows(cfg, args.start, args.end)
    except ValueError as exc:
        parser.error(str(exc))
    # Anchors are session labels, not midnight market observations. Holidays and
    # weekends are excluded; intraday requests use timezone-aware boundaries.
    anchors = (
        planning.exchange_calendar()
        .sessions_in_range(start, end)
        .tz_localize(None)
    )

    symbols = [
        cfg
        for cfg in config.UNIVERSE
        if args.symbols is None or cfg.symbol in args.symbols
    ]
    return args, cfg, symbols, anchors


def main(argv: list[str] | None = None) -> int:
    """Run the requested preview, coverage check, or collection workflow.

    Args:
        argv: Arguments after the program name, or None to use sys.argv.

    Returns:
        Exit code 0 for a preview or completed work without reported gaps, 1 for
        failure/interruption, or 2 for completed work with observed coverage
        gaps. None of these codes validates a pricing model.
    """
    args, cfg, symbols, anchors = parse_run_scope(argv)
    windows = planning.collection_windows(cfg, args.start, args.end)
    panels = not (args.references_only or args.coverage_only)
    total = len(symbols) * len(anchors) if panels else 0
    print(
        f"Scope: {', '.join(s.symbol for s in symbols)}; {args.start} to {args.end}; {total} symbol-days"
    )
    print(
        f"Vendor: ThetaData; quotes: {cfg.quote_interval} plus near-close; daily volume/count from EOD; stock venue: {cfg.stock_venue}"
    )
    if panels:
        print(
            f"Bulk collection: 7 shared requests + 2 per selected expiration/day (at most {7 + 2 * cfg.max_expirations_per_day})"
        )
        print(
            f"Storage: selected option contracts only; broad option reports/lists capped at {cfg.max_dte} days to expiration"
        )
        print(
            f"Near-close snapshot: {cfg.near_close_minutes} minutes before the actual close; quote sample age is not event age"
        )
        print(
            f"Standard: {cfg.max_inflight_requests} simultaneous requests; "
            f"request-start cap: {str(cfg.max_requests_per_second) + '/s' if cfg.max_requests_per_second else 'none'}"
        )
    print(f"Output: {cfg.output_dir}")
    if args.coverage_only:
        print(
            "Coverage mode: available dates for stock quotes/trades and VIX; no history downloads"
        )
    else:
        print(
            f"Required references: dividends/splits, {len(set(args.rate_symbols))} rate series, VIX EOD and {cfg.quote_interval} prices"
        )
        print(
            f"Reference history starts {windows['history_start']}; corporate actions through {windows['corporate_action_end']}"
        )
        print(
            f"Standard VIX history starts {cfg.index_history_start}; earlier requested sessions are reported as access gaps"
        )
        if panels:
            print(
                f"Stock lookback: {cfg.lookback_sessions} prior sessions; "
                f"{3 * len(symbols) * cfg.lookback_sessions} additional hourly/near-close/EOD requests"
            )
    # Preview stops before constructing storage, contacting Theta, or creating
    # output.
    if args.plan:
        return 0
    collector = workflow.Collector(cfg)
    # A run ID identifies this invocation. Policy and request IDs identify
    # reusable selection rules and data pulls across multiple invocations.
    run_id = (
        pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )

    run_path = collector.directory / "runs" / f"{run_id}.json"
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
            "symbols": [s.symbol for s in symbols],
            "start": args.start,
            "end": args.end,
            "collection_windows": windows,
            "references_only": args.references_only,
            "coverage_only": args.coverage_only,
            "rate_symbols": args.rate_symbols,
        },
        "code_sha256": collector.store.code_sha256,
        "code_files": collector.store.code_files,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": provenance.package_versions(),
        "resumed_days": 0,
        "processed_days": 0,
    }
    failed = reference_failures = reference_gaps = catalogue_errors = (
        catalogue_gaps
    ) = 0
    exit_code = 0
    try:
        collector.store.client.ensure_available()

        # Hold the process lock through references, panels, and final reports.
        with storage.output_lock(cfg.output_dir):
            storage.write_json(run_path, run)
            try:
                if not args.references_only:
                    run["date_catalogue"] = f"coverage/{run_id}.json"
                    # Report catalogue gaps without silently shortening the
                    # requested date range.
                    catalogue_dates = (
                        planning.exchange_calendar()
                        .sessions_in_range(windows["history_start"], args.end)
                        .tz_localize(None)
                    )
                    catalogue = collector.collect_coverage(
                        symbols, catalogue_dates, run_id
                    )
                    catalogue_errors, catalogue_gaps = (
                        catalogue["request_errors"],
                        catalogue["series_with_gaps"],
                    )
                    collector.store.client.check_running()
                if not args.coverage_only:
                    run["reference_ledger"] = f"references/{run_id}.json"
                    references = collector.collect_references(
                        symbols,
                        args.start,
                        args.end,
                        args.rate_symbols,
                        run_id,
                        include_stock_lookback=panels,
                    )
                    reference_failures = sum(
                        r["status"] not in storage.GOOD_REQUEST_STATUSES
                        or r["coverage"]["status"] == "unknown"
                        for r in references["requests"]
                    )

                    reference_gaps = references[
                        "requests_with_observed_gaps"
                    ] + len(references["subscription_coverage_gaps"])
                    collector.store.client.check_running()
                if panels:
                    # Advance by trading day and overlap its underlyings. Each
                    # day can reuse completed request receipts even when its
                    # previous manifest was incomplete.
                    for day in anchors:
                        pending = []
                        for symbol in symbols:
                            if collector.resumable(symbol.symbol, day):
                                run["resumed_days"] += 1
                            else:
                                pending.append(symbol)
                        with collector.workers(
                            cfg.max_symbol_day_workers
                        ) as pool:
                            futures = {
                                pool.submit(
                                    collector.collect_day, symbol, day
                                ): symbol
                                for symbol in pending
                            }
                            for future in concurrent.futures.as_completed(
                                futures
                            ):
                                symbol = futures[future]
                                run["processed_days"] += 1
                                try:
                                    manifest = future.result()
                                    failed += (
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
                                    failed += 1
                                    print(
                                        f"FAILED {symbol.symbol} {day.date()}: {exc!r}"
                                    )
            except BaseException as exc:
                collector.store.client.stop(
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
                try:
                    run["coverage"] = (
                        collector.write_availability(symbols, anchors)
                        if panels
                        else {}
                    )
                except Exception as exc:
                    run["coverage_error"] = repr(exc)
                    exit_code = 1
                coverage = run.get("coverage", {})

                if run["status"] == "running":
                    # A failure takes priority over a known coverage gap.
                    # Success here says nothing about pricing, calibration, or
                    # research model performance.
                    if (
                        exit_code
                        or failed
                        or reference_failures
                        or catalogue_errors
                        or coverage.get("request_error", 0)
                        or coverage.get("not_attempted", 0)
                        or coverage.get("days_with_unknown_coverage", 0)
                    ):
                        exit_code = 1
                    elif (
                        reference_gaps
                        or catalogue_gaps
                        or coverage.get("unavailable", 0)
                        or coverage.get("days_with_observed_gaps", 0)
                    ):
                        exit_code = 2
                    run["status"] = {
                        0: "complete",
                        1: "partial_failure",
                        2: "coverage_gaps",
                    }[exit_code]
                run.update(
                    finished_at_utc=provenance.utc_now(),
                    failed_days=max(
                        int(failed), coverage.get("request_error", 0)
                    ),
                    reference_failures=reference_failures,
                    reference_gaps=reference_gaps,
                    catalogue_errors=catalogue_errors,
                    catalogue_series_with_gaps=catalogue_gaps,
                )
                storage.write_json(run_path, run)
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        print(f"Collector stopped: {exc}", file=sys.stderr)
        if run_path.exists():
            print(f"Run record: {run_path}", file=sys.stderr)
        return 1
    print(
        f"Finished: {run['processed_days']} processed, {run['resumed_days']} resumed, "
        f"{failed} failed days, {reference_failures} failed reference requests, "
        f"{reference_gaps + catalogue_gaps} reference/catalogue gaps"
    )
    print(f"Run record: {run_path}")
    if run.get("date_catalogue"):
        print(f"Date coverage: {cfg.output_dir / run['date_catalogue']}")
    if run.get("reference_ledger"):
        print(f"Theta references: {cfg.output_dir / run['reference_ledger']}")
    if panels:
        print(f"Panel coverage: {collector.directory / 'availability.csv'}")
    return exit_code
