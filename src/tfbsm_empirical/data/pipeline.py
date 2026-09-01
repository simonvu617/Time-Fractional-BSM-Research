from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .cache import ResponseCache
from .client import MarketDataClient, ThetaDataClient
from .config import CollectorConfig, SymbolSpec
from .features import add_quote_features, reference_mids
from .timestamps import (
    evaluation_schedule,
    market_session,
    normalize_vendor_timestamps,
    previous_session_date,
)
from .universe import (
    format_strike,
    occ_contract_id,
    select_expirations,
    select_strikes,
)
from .validation import validate_frame, validate_reconciliation
from .writer import CanonicalWriter


class CollectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionResult:
    symbol: str
    trade_date: date
    contract_count: int
    artifact_count: int
    manifest_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date.isoformat(),
            "contract_count": self.contract_count,
            "artifact_count": self.artifact_count,
            "manifest_path": str(self.manifest_path),
        }


class CollectorPipeline:
    """Orchestrate one explicit, auditable symbol-day collection."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        client: MarketDataClient | None = None,
        cache: ResponseCache | None = None,
        writer: CanonicalWriter | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or ThetaDataClient(
            config.base_url,
            config.request_timeout_seconds,
        )
        self.cache = cache or ResponseCache(config.output_dir / "raw")
        self.writer = writer or CanonicalWriter(config.output_dir)

    def plan(self, symbol: str, trade_date: date) -> dict[str, object]:
        symbol_spec = self._validate_request(symbol, trade_date)
        session = market_session(
            trade_date,
            self.config.calendar,
            self.config.exchange_timezone,
        )
        schedule = evaluation_schedule(
            session,
            self.config.evaluation_times,
            self.config.horizon_minutes,
            self.config.exchange_timezone,
        )
        return {
            "symbol": symbol_spec.symbol,
            "asset_type": symbol_spec.asset_type,
            "trade_date": trade_date.isoformat(),
            "session_open": session.opened_at.isoformat(),
            "session_close": session.closed_at.isoformat(),
            "evaluation_points": [
                {
                    "label": point.label,
                    "entry_at": point.entry_at.isoformat(),
                    "exit_at": point.exit_at.isoformat(),
                }
                for point in schedule
            ],
            "target_dtes": list(self.config.target_dtes),
            "moneyness_targets": list(self.config.moneyness_targets),
            "option_rights": list(self.config.option_rights),
            "output_dir": str(self.config.output_dir),
            "network_requests_made": False,
        }

    def collect_symbol_day(self, symbol: str, trade_date: date) -> CollectionResult:
        symbol_spec = self._validate_request(symbol, trade_date)
        session = market_session(
            trade_date,
            self.config.calendar,
            self.config.exchange_timezone,
        )
        schedule = evaluation_schedule(
            session,
            self.config.evaluation_times,
            self.config.horizon_minutes,
            self.config.exchange_timezone,
        )
        previous_date = previous_session_date(trade_date, self.config.calendar)
        artifacts: list[dict[str, object]] = []
        request_date = trade_date.strftime("%Y%m%d")
        base_partition = {"symbol": symbol_spec.symbol, "date": trade_date.isoformat()}

        stock_quotes = self._fetch_and_write(
            dataset="stock_quotes",
            endpoint="/stock/history/quote",
            params={
                "symbol": symbol_spec.symbol,
                "date": request_date,
                "interval": self.config.quote_interval,
            },
            partition=base_partition,
            identity={
                "collector_symbol": symbol_spec.symbol,
                "collector_trade_date": trade_date.isoformat(),
            },
            add_quotes=True,
            artifacts=artifacts,
        )
        self._fetch_and_write(
            dataset="stock_trades",
            endpoint="/stock/history/trade",
            params={
                "symbol": symbol_spec.symbol,
                "date": request_date,
                "interval": self.config.trade_interval,
            },
            partition=base_partition,
            identity={
                "collector_symbol": symbol_spec.symbol,
                "collector_trade_date": trade_date.isoformat(),
            },
            artifacts=artifacts,
        )

        mids = reference_mids(
            stock_quotes,
            schedule,
            max_age_seconds=self.config.max_stock_quote_age_seconds,
        )
        if not mids:
            raise CollectionError(
                "no fresh positive stock reference mid was available at the configured evaluation times"
            )

        expiration_frame = self._fetch_and_write(
            dataset="chain_expirations",
            endpoint="/option/list/expirations",
            params={"symbol": symbol_spec.symbol},
            partition=base_partition,
            identity={"collector_symbol": symbol_spec.symbol},
            artifacts=artifacts,
        )
        listed_expirations = _parse_expirations(expiration_frame["expiration"])
        selected_expirations = select_expirations(
            trade_date,
            listed_expirations,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            target_dtes=self.config.target_dtes,
            maximum=self.config.max_expirations_per_day,
        )
        if not selected_expirations:
            raise CollectionError("no listed expirations satisfy the configured DTE window")

        contracts: list[dict[str, object]] = []
        for expiration in selected_expirations:
            expiration_value = expiration.isoformat()
            strikes_frame = self._fetch_and_write(
                dataset="chain_strikes",
                endpoint="/option/list/strikes",
                params={
                    "symbol": symbol_spec.symbol,
                    "expiration": expiration_value,
                },
                partition={**base_partition, "expiration": expiration_value},
                identity={
                    "collector_symbol": symbol_spec.symbol,
                    "collector_expiration": expiration_value,
                },
                artifacts=artifacts,
            )
            listed_strikes = tuple(
                pd.to_numeric(strikes_frame["strike"], errors="coerce")
                .dropna()
                .astype(float)
                .tolist()
            )
            chosen_strikes = select_strikes(
                listed_strikes,
                mids.values(),
                moneyness_targets=self.config.moneyness_targets,
                per_target=self.config.strikes_per_moneyness_target,
            )
            for strike in chosen_strikes:
                for right in self.config.option_rights:
                    contracts.append(
                        {
                            "contract_id": occ_contract_id(
                                symbol_spec.symbol,
                                expiration,
                                right,
                                strike,
                            ),
                            "symbol": symbol_spec.symbol,
                            "asset_type": symbol_spec.asset_type,
                            "universe_bucket": symbol_spec.universe_bucket,
                            "sector_proxy": symbol_spec.sector_proxy,
                            "trade_date": trade_date.isoformat(),
                            "expiration": expiration_value,
                            "dte": (expiration - trade_date).days,
                            "strike": strike,
                            "right": right,
                        }
                    )

        if not contracts:
            raise CollectionError("the configured surface rules selected no option contracts")

        contract_frame = pd.DataFrame(contracts)
        contract_report = validate_frame("contract_universe", contract_frame)
        contract_artifact = self.writer.write_frame(
            "contract_universe",
            contract_frame,
            partition=base_partition,
        )
        artifacts.append(
            {
                "source": "derived",
                "canonical": contract_artifact.as_dict(),
                "validation": contract_report.as_dict(),
            }
        )

        collected_contracts = 0
        for contract in contracts:
            identity = {
                "collector_symbol": symbol_spec.symbol,
                "collector_trade_date": trade_date.isoformat(),
                "collector_expiration": contract["expiration"],
                "collector_strike": contract["strike"],
                "collector_right": contract["right"],
                "collector_contract_id": contract["contract_id"],
            }
            partition = {
                **base_partition,
                "expiration": contract["expiration"],
                "strike": format_strike(float(contract["strike"])),
                "right": contract["right"],
            }
            common_params = {
                "symbol": symbol_spec.symbol,
                "expiration": contract["expiration"],
                "strike": format_strike(float(contract["strike"])),
                "right": contract["right"],
            }
            self._fetch_and_write(
                dataset="option_quotes",
                endpoint="/option/history/quote",
                params={
                    **common_params,
                    "date": request_date,
                    "interval": self.config.quote_interval,
                },
                partition=partition,
                identity=identity,
                add_quotes=True,
                artifacts=artifacts,
            )
            self._fetch_and_write(
                dataset="option_trades",
                endpoint="/option/history/trade",
                params={
                    **common_params,
                    "date": request_date,
                    "interval": self.config.trade_interval,
                },
                partition=partition,
                identity=identity,
                artifacts=artifacts,
            )
            self._fetch_and_write(
                dataset="option_open_interest",
                endpoint="/option/history/open_interest",
                params={
                    **common_params,
                    "date": previous_date.strftime("%Y%m%d"),
                },
                partition={
                    **partition,
                    "observation_date": previous_date.isoformat(),
                },
                identity={
                    **identity,
                    "collector_observation_date": previous_date.isoformat(),
                },
                artifacts=artifacts,
            )
            collected_contracts += 1

        validate_reconciliation(
            expected=len(contracts),
            collected=collected_contracts,
        )
        manifest_name = f"{symbol_spec.symbol}_{trade_date.isoformat()}"
        manifest_path = self.writer.write_manifest(
            manifest_name,
            {
                "status": "complete",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "project": self.config.project_name,
                "output_schema_version": self.config.output_schema_version,
                "vendor": self.config.vendor,
                "symbol": symbol_spec.symbol,
                "trade_date": trade_date.isoformat(),
                "previous_session_for_open_interest": previous_date.isoformat(),
                "evaluation_schedule": [
                    {
                        "label": point.label,
                        "entry_at": point.entry_at.isoformat(),
                        "exit_at": point.exit_at.isoformat(),
                    }
                    for point in schedule
                ],
                "reference_mids": mids,
                "selected_expirations": [value.isoformat() for value in selected_expirations],
                "expected_contract_count": len(contracts),
                "collected_contract_count": collected_contracts,
                "artifacts": artifacts,
            },
        )
        return CollectionResult(
            symbol=symbol_spec.symbol,
            trade_date=trade_date,
            contract_count=collected_contracts,
            artifact_count=len(artifacts),
            manifest_path=manifest_path,
        )

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "CollectorPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_request(self, symbol: str, trade_date: date) -> SymbolSpec:
        if not self.config.start_date <= trade_date <= self.config.end_date:
            raise ValueError(
                f"trade date {trade_date.isoformat()} is outside the configured study window"
            )
        return self.config.symbol(symbol)

    def _fetch_and_write(
        self,
        *,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
        partition: Mapping[str, object],
        identity: Mapping[str, object],
        artifacts: list[dict[str, object]],
        add_quotes: bool = False,
    ) -> pd.DataFrame:
        cached = self.cache.load(dataset, endpoint, params)
        source = "cache"
        if cached is None:
            response = self.client.fetch_csv(endpoint, params)
            cached = self.cache.store(dataset, endpoint, params, response)
            source = "network"

        frame = cached.frame.copy()
        if "timestamp" in frame.columns:
            frame = normalize_vendor_timestamps(
                frame,
                self.config.exchange_timezone,
            )
        if add_quotes:
            frame = add_quote_features(frame)
        for column, value in identity.items():
            frame[column] = value

        report = validate_frame(dataset, frame)
        canonical = self.writer.write_frame(
            dataset,
            frame,
            partition=partition,
        )
        artifacts.append(
            {
                "source": source,
                "endpoint": endpoint,
                "params": dict(params),
                "raw_payload": str(cached.payload_path),
                "raw_metadata": str(cached.metadata_path),
                "canonical": canonical.as_dict(),
                "validation": report.as_dict(),
            }
        )
        return frame


def _parse_expirations(values: pd.Series) -> tuple[date, ...]:
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    return tuple(sorted({value.date() for value in parsed.dropna()}))
