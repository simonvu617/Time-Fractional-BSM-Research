from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .client import CsvResponse


class CacheCorruptionError(RuntimeError):
    pass


class CacheConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    frame: pd.DataFrame
    payload_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any]


class ResponseCache:
    """Immutable, request-addressed storage for exact vendor CSV responses."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(
        self,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> CachedResponse | None:
        payload_path, metadata_path = self._paths(dataset, endpoint, params)
        if not payload_path.exists() and not metadata_path.exists():
            return None
        if not payload_path.exists() or not metadata_path.exists():
            raise CacheCorruptionError(f"incomplete cache entry: {payload_path.parent}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_request = request_digest(endpoint, params)
        if metadata.get("request_digest") != expected_request:
            raise CacheCorruptionError(f"request digest mismatch: {payload_path.parent}")
        payload = payload_path.read_bytes()
        if metadata.get("payload_sha256") != sha256_bytes(payload):
            raise CacheCorruptionError(f"payload hash mismatch: {payload_path}")
        try:
            frame = pd.read_csv(BytesIO(payload))
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        return CachedResponse(
            frame=frame,
            payload_path=payload_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def store(
        self,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
        response: CsvResponse,
    ) -> CachedResponse:
        payload_path, metadata_path = self._paths(dataset, endpoint, params)
        existing = self.load(dataset, endpoint, params)
        if existing is not None:
            existing_digest = str(existing.metadata["payload_sha256"])
            received_digest = sha256_bytes(response.payload)
            if existing_digest != received_digest:
                raise CacheConflictError(
                    "a different payload already exists for this request identity; "
                    "use a new output directory for a refreshed pull"
                )
            return existing

        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_digest = sha256_bytes(response.payload)
        metadata = {
            "dataset": dataset,
            "endpoint": endpoint,
            "params": _jsonable(params),
            "request_digest": request_digest(endpoint, params),
            "payload_sha256": payload_digest,
            "payload_bytes": len(response.payload),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_url": response.request_url,
            "status_code": response.status_code,
            "elapsed_ms": response.elapsed_ms,
            "response_headers": dict(response.response_headers),
        }
        _atomic_write(payload_path, response.payload)
        _atomic_write(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        )
        cached = self.load(dataset, endpoint, params)
        if cached is None:  # pragma: no cover - defensive guard after writes.
            raise CacheCorruptionError(f"cache write did not persist: {payload_path.parent}")
        return cached

    def _paths(
        self,
        dataset: str,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        safe_dataset = _safe_component(dataset)
        entry = self.root / safe_dataset / f"request={request_digest(endpoint, params)}"
        return entry / "response.csv", entry / "metadata.json"


def request_digest(endpoint: str, params: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"endpoint": endpoint, "params": _jsonable(params)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
