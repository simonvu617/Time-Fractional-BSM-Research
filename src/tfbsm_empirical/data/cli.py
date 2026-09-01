from __future__ import annotations

import argparse
import json
from datetime import date

from .config import CollectorConfig
from .pipeline import CollectorPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one auditable TFBSM research symbol-day.",
    )
    parser.add_argument("--config", required=True, help="Path to collector YAML")
    parser.add_argument("--symbol", required=True, help="Configured option root")
    parser.add_argument("--date", required=True, help="XNYS session in YYYY-MM-DD format")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the collection plan without vendor requests or writes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CollectorConfig.from_yaml(args.config)
    trade_date = date.fromisoformat(args.date)
    with CollectorPipeline(config) as pipeline:
        if args.dry_run:
            print(json.dumps(pipeline.plan(args.symbol, trade_date), indent=2, sort_keys=True))
            return
        result = pipeline.collect_symbol_day(args.symbol, trade_date)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
