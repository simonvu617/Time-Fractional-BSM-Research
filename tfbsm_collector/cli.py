# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Collect ThetaData inputs for the hourly/daily TFBSM pricing study.

Requires Python 3.11+ and the packages in requirements.txt. Theta Terminal v3
must be running for downloads.

Preview a small run without downloading or creating output:
    python collector.py --symbols SPY --plan

Start reading in config.py for study settings and workflow.py for collection.
Under --output-dir, collection/<policy-id>/availability.csv summarizes coverage;
tables/ records monthly selection and cohorts. parquet/ holds observations;
index.sqlite3 holds response metadata and session coverage. Pricing and calibration are separate research work.
"""

import argparse
import dataclasses
import pathlib
import sys

from tfbsm_collector import config, planning, workflow


def parse_run_scope(
    argv: list[str] | None = None,
) -> tuple[config.CollectorConfig, bool]:
    """Parse and validate command-line scope without downloading or writing.

    Args:
        argv: Arguments after the program name, or None to use sys.argv.

    Returns:
        The validated configuration and whether this is a preview. The same
        configuration owns CLI choices and runtime scope.

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
        choices=[
            cfg.symbol for cfg in (*config.UNIVERSE, *config.INDEX_BENCHMARK)
        ],
    )
    parser.add_argument(
        "--start",
        default=defaults.start_date,
        help="Inclusive first date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=defaults.end_date,
        help="Last date for selecting new contracts; existing cohorts are followed through expiration",
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
        help="1 through 8 for Pro; lower this if another client shares the account",
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=float,
        default=defaults.max_requests_per_second,
        help="Optional request-start pacing; default 0 means only the concurrency cap applies",
    )
    parser.add_argument(
        "--index-subscription",
        choices=config.INDEX_HISTORY_STARTS,
        default=defaults.index_subscription,
        help="Your separately purchased index tier (default: none); Stocks/Options Pro does not include VIX",
    )
    parser.add_argument(
        "--rate-subscription",
        choices=config.RATE_HISTORY_STARTS,
        default=defaults.rate_subscription,
        help="Your separate rate tier (default: free, history from 2024); value permits older dates",
    )
    parser.add_argument(
        "--rate-symbols",
        nargs="+",
        default=list(config.RATE_SYMBOLS),
        type=str.upper,
        choices=config.RATE_SYMBOLS,
        help="Theta rate series to collect; defaults to SOFR and all documented Treasury tenors",
    )
    parser.set_defaults(mode=defaults.mode)
    modes = parser.add_mutually_exclusive_group()

    modes.add_argument(
        "--references-only",
        action="store_const",
        dest="mode",
        const="references",
        help="Collect accessible rates/VIX and report missing references without stock/option panels",
    )
    modes.add_argument(
        "--coverage-only",
        action="store_const",
        dest="mode",
        const="coverage",
        help="Refresh stock/VIX available-date lists and report missing sessions",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show scope without network requests or output writes",
    )
    args = parser.parse_args(argv)
    try:
        cfg = dataclasses.replace(
            defaults,
            mode=args.mode,
            symbols=tuple(
                symbol
                for symbol in (*config.UNIVERSE, *config.INDEX_BENCHMARK)
                if (
                    symbol in config.UNIVERSE
                    if args.symbols is None
                    else symbol.symbol in args.symbols
                )
            ),
            rate_symbols=tuple(sorted(set(args.rate_symbols))),
            start_date=args.start,
            end_date=args.end,
            index_subscription=args.index_subscription,
            rate_subscription=args.rate_subscription,
            quote_interval=args.quote_interval,
            output_dir=args.output_dir.expanduser().resolve(),
            raw_chunk_rows=args.raw_chunk_rows,
            lookback_sessions=args.lookback_sessions,
            store_raw_payloads=args.store_raw_payloads,
            refresh_no_data=args.refresh_no_data,
            max_inflight_requests=args.max_inflight_requests,
            max_requests_per_second=args.max_requests_per_second,
        )
        planning.collection_windows(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    return cfg, args.plan


def _print_scope(cfg: config.CollectorConfig) -> None:
    """Describe requested work and known access gaps before a run or preview."""
    anchors = planning.exchange_calendar().sessions_in_range(
        cfg.start_date, cfg.end_date
    )
    windows = planning.collection_windows(cfg)
    panels = cfg.mode == "panels"
    total = len(cfg.symbols) * len(anchors) if panels else 0
    print(
        f"Scope: {', '.join(s.symbol for s in cfg.symbols)}; {cfg.start_date} to {cfg.end_date}; {total} symbol-days"
    )
    print(
        f"Vendor: ThetaData; quotes and activity bars: {cfg.quote_interval}, plus near-close quotes and daily EOD; stock venue: {cfg.stock_venue}"
    )
    if panels:
        print(
            "Monthly date batches: underlying prices/activity and option quotes/activity per tracked expiration; dated discovery and OI"
        )
        print(
            f"Entry: DTE {cfg.min_dte}-{cfg.max_dte}; track selected contracts through expiration, including below the entry cutoff"
        )
        print(
            f"Follow-up tail: through actual selected expirations, bounded by {windows['followup_end_bound']}; completed dates only"
        )
        print(
            "Storage: shared Parquet files plus index.sqlite3 for request/session metadata"
        )
        print(
            f"Near-close snapshot: {cfg.near_close_minutes} minutes before the actual close; quote sample age is not event age"
        )
        print(
            f"Stocks/Options Pro: {cfg.max_inflight_requests} shared simultaneous requests; "
            f"request-start cap: {str(cfg.max_requests_per_second) + '/s' if cfg.max_requests_per_second else 'none'}"
        )
    print(f"Output: {cfg.output_dir}")
    print(
        f"Separate reference subscriptions: indices={cfg.index_subscription}; "
        f"rates={cfg.rate_subscription} (from {cfg.rate_history_start})"
    )
    if cfg.mode == "coverage":
        print(
            "Coverage mode: available dates for stock quotes/trades and VIX; no history downloads"
        )
        if cfg.index_history_start is None:
            print(
                "VIX catalogue not requested: no index subscription; recorded as an access gap"
            )
    else:
        print(
            f"Required references: dividends/splits, {len(set(cfg.rate_symbols))} rate series, VIX EOD and {cfg.quote_interval} prices"
        )
        print(
            f"Requested reference history starts {windows['requested_history_start']}; corporate actions through {windows['corporate_action_end']}"
        )
        for gap in planning.reference_access_gaps(cfg, windows):
            if "missing_fields" in gap:
                print(
                    f"Known input gap: {gap['dataset']}; "
                    f"{', '.join(gap['missing_fields'])}"
                )
                continue
            if "unrequested_start_date" in gap:
                print(
                    f"Known access gap: {gap['dataset']}; {gap['reason']}; "
                    f"{gap['unrequested_start_date']} to {gap['unrequested_end_date']}"
                )
                continue
            count = len(
                gap.get(
                    "unrequested_eod_session_dates",
                    gap.get("unrequested_session_dates", []),
                )
            )
            print(
                f"Known access gap: {gap['reason']}; {count} exchange sessions"
            )
        if panels:
            print(
                f"Stock lookback: {len(windows['lookback_dates'])} of {cfg.lookback_sessions} prior sessions accessible; "
                "batched underlying prices, activity, and EOD reports"
            )


def main(argv: list[str] | None = None) -> int:
    """Parse options, preview the scope, and hand collection to the workflow.

    Args:
        argv: Arguments after the program name, or None to use sys.argv.

    Returns:
        Exit code 0 for a preview or completed work without reported gaps,
        1 for failure/interruption, or 2 for completed work with coverage gaps.
    """
    cfg, preview = parse_run_scope(argv)
    _print_scope(cfg)
    if preview:
        return 0
    try:
        return workflow.Collector(cfg).run()
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        print(f"Collector stopped: {exc}", file=sys.stderr)
        return 1
