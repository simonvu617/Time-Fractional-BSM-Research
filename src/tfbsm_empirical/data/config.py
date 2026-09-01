from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    asset_type: str
    universe_bucket: str
    sector_proxy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SymbolSpec":
        return cls(
            symbol=str(value["symbol"]).strip().upper(),
            asset_type=str(value["asset_type"]).strip().upper(),
            universe_bucket=str(value["universe_bucket"]).strip(),
            sector_proxy=str(value["sector_proxy"]).strip(),
        )


@dataclass(frozen=True)
class CollectorConfig:
    project_name: str
    output_schema_version: str
    vendor: str
    base_url: str
    request_timeout_seconds: float
    project_root: Path
    output_dir: Path
    start_date: date
    end_date: date
    calendar: str
    exchange_timezone: str
    quote_interval: str
    trade_interval: str
    evaluation_times: tuple[str, ...]
    horizon_minutes: int
    max_stock_quote_age_seconds: int
    min_dte: int
    max_dte: int
    target_dtes: tuple[int, ...]
    max_expirations_per_day: int
    moneyness_targets: tuple[float, ...]
    strikes_per_moneyness_target: int
    option_rights: tuple[str, ...]
    universe: tuple[SymbolSpec, ...]

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP URL")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if not (0 < self.min_dte <= self.max_dte):
            raise ValueError("DTE bounds must satisfy 0 < min_dte <= max_dte")
        if not self.target_dtes or any(value <= 0 for value in self.target_dtes):
            raise ValueError("target_dtes must contain positive values")
        if self.max_expirations_per_day <= 0:
            raise ValueError("max_expirations_per_day must be positive")
        if not self.moneyness_targets or any(value <= 0 for value in self.moneyness_targets):
            raise ValueError("moneyness_targets must contain positive values")
        if self.strikes_per_moneyness_target <= 0:
            raise ValueError("strikes_per_moneyness_target must be positive")
        invalid_rights = set(self.option_rights) - {"call", "put"}
        if invalid_rights:
            raise ValueError(f"unsupported option rights: {sorted(invalid_rights)}")
        if not self.evaluation_times:
            raise ValueError("at least one evaluation time is required")
        for value in self.evaluation_times:
            datetime.strptime(value, "%H:%M:%S")
        symbols = [item.symbol for item in self.universe]
        if not symbols:
            raise ValueError("the collection universe must not be empty")
        duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        if duplicates:
            raise ValueError(f"duplicate universe symbols: {duplicates}")
        if not self.output_dir.is_relative_to(self.project_root):
            raise ValueError("output_dir must remain inside the project root")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CollectorConfig":
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("collector config must contain a mapping")

        project = _mapping(payload, "project")
        collector = _mapping(payload, "collector")
        universe_values = payload.get("universe")
        if not isinstance(universe_values, list):
            raise ValueError("universe must be a list")

        project_root = config_path.parent.parent.resolve()
        configured_output = Path(str(collector["output_dir"])).expanduser()
        output_dir = (
            configured_output.resolve()
            if configured_output.is_absolute()
            else (project_root / configured_output).resolve()
        )

        return cls(
            project_name=str(project["name"]),
            output_schema_version=str(project["output_schema_version"]),
            vendor=str(collector["vendor"]),
            base_url=str(collector["base_url"]).rstrip("/"),
            request_timeout_seconds=float(collector["request_timeout_seconds"]),
            project_root=project_root,
            output_dir=output_dir,
            start_date=_as_date(collector["start_date"], "start_date"),
            end_date=_as_date(collector["end_date"], "end_date"),
            calendar=str(collector["calendar"]),
            exchange_timezone=str(collector["exchange_timezone"]),
            quote_interval=str(collector["quote_interval"]),
            trade_interval=str(collector["trade_interval"]),
            evaluation_times=tuple(str(value) for value in collector["evaluation_times"]),
            horizon_minutes=int(collector["horizon_minutes"]),
            max_stock_quote_age_seconds=int(collector["max_stock_quote_age_seconds"]),
            min_dte=int(collector["min_dte"]),
            max_dte=int(collector["max_dte"]),
            target_dtes=tuple(int(value) for value in collector["target_dtes"]),
            max_expirations_per_day=int(collector["max_expirations_per_day"]),
            moneyness_targets=tuple(float(value) for value in collector["moneyness_targets"]),
            strikes_per_moneyness_target=int(collector["strikes_per_moneyness_target"]),
            option_rights=tuple(str(value).lower() for value in collector["option_rights"]),
            universe=tuple(SymbolSpec.from_mapping(value) for value in universe_values),
        )

    def symbol(self, value: str) -> SymbolSpec:
        requested = value.strip().upper()
        for item in self.universe:
            if item.symbol == requested:
                return item
        raise KeyError(f"symbol {requested!r} is not present in the configured universe")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["output_dir"] = str(self.output_dir)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _as_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format") from exc
