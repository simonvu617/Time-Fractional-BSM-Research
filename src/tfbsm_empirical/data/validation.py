from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


class SchemaValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    row_count: int
    columns: tuple[str, ...]
    duplicate_row_count: int
    timestamp_parse_failure_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_frame(dataset: str, frame: pd.DataFrame) -> ValidationReport:
    required = required_columns(dataset)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaValidationError(
            f"{dataset} response is missing required columns: {missing}; "
            f"received={list(frame.columns)}"
        )
    parse_failures = 0
    if "timestamp_parse_failed" in frame.columns:
        parse_failures = int(frame["timestamp_parse_failed"].fillna(False).astype(bool).sum())
    return ValidationReport(
        dataset=dataset,
        row_count=int(len(frame)),
        columns=tuple(str(value) for value in frame.columns),
        duplicate_row_count=int(frame.duplicated().sum()),
        timestamp_parse_failure_count=parse_failures,
    )


def required_columns(dataset: str) -> tuple[str, ...]:
    if dataset == "chain_expirations":
        return ("expiration",)
    if dataset == "chain_strikes":
        return ("strike",)
    if dataset in {"stock_quotes", "option_quotes"}:
        return ("timestamp", "bid", "ask")
    if dataset in {"stock_trades", "option_trades"}:
        return ("timestamp", "price", "size")
    if dataset == "option_open_interest":
        return ("timestamp", "open_interest")
    if dataset == "contract_universe":
        return ("contract_id", "symbol", "expiration", "strike", "right")
    return ()


def validate_reconciliation(*, expected: int, collected: int) -> None:
    if expected != collected:
        raise SchemaValidationError(
            f"contract reconciliation failed: expected={expected}, collected={collected}"
        )
