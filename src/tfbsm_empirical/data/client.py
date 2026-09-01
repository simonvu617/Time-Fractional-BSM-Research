from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping, Protocol

import pandas as pd
import requests


@dataclass(frozen=True)
class CsvResponse:
    frame: pd.DataFrame
    payload: bytes
    request_url: str
    status_code: int
    elapsed_ms: float
    response_headers: Mapping[str, str]


class MarketDataClient(Protocol):
    def fetch_csv(self, endpoint: str, params: Mapping[str, Any]) -> CsvResponse:
        """Fetch one read-only CSV response from the configured vendor."""


class ThetaDataClient:
    """Minimal read-only adapter for the local ThetaData HTTP terminal."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._owns_session = session is None

    def fetch_csv(self, endpoint: str, params: Mapping[str, Any]) -> CsvResponse:
        if not endpoint.startswith("/"):
            raise ValueError("vendor endpoint must begin with '/'")
        started = time.perf_counter()
        response = self._session.get(
            f"{self.base_url}{endpoint}",
            params=dict(params),
            timeout=self.timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.content
        try:
            frame = pd.read_csv(BytesIO(payload))
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        return CsvResponse(
            frame=frame,
            payload=payload,
            request_url=response.url,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            response_headers=dict(response.headers),
        )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "ThetaDataClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
