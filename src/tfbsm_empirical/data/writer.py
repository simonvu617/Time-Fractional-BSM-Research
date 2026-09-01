from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class WrittenArtifact:
    dataset: str
    path: Path
    row_count: int
    sha256: str
    partition: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


class CanonicalWriter:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def write_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        partition: Mapping[str, object],
    ) -> WrittenArtifact:
        directory = self.output_root / "canonical" / _safe_component(dataset)
        normalized_partition: dict[str, str] = {}
        for key, value in partition.items():
            safe_key = _safe_component(str(key))
            safe_value = _safe_component(str(value))
            normalized_partition[safe_key] = safe_value
            directory /= f"{safe_key}={safe_value}"
        directory.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            dir=directory,
            delete=False,
            suffix=".parquet",
        ) as handle:
            temp_path = Path(handle.name)
        try:
            frame.to_parquet(temp_path, index=False)
            digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            destination = directory / f"part-{digest}.parquet"
            if destination.exists():
                temp_path.unlink()
            else:
                os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return WrittenArtifact(
            dataset=dataset,
            path=destination,
            row_count=int(len(frame)),
            sha256=digest,
            partition=normalized_partition,
        )

    def write_manifest(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.output_root / "manifests" / f"{_safe_component(name)}.json"
        encoded = json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        _atomic_write(path, encoded)
        return path


def _safe_component(value: str) -> str:
    normalized = value.replace(":", "-").replace("/", "-").replace(" ", "_")
    if not normalized or not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError(f"unsafe path component: {value!r}")
    return normalized


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
