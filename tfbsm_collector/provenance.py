# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Stable fingerprints and environment metadata for collection records.

Fingerprints identify requests, source code, and saved bytes. They support
reproducibility; they do not establish the accuracy of the vendor's data.
"""

import hashlib
import importlib.metadata
import json
import pathlib

import pandas as pd


def utc_now() -> str:
    """Return the current UTC instant as ISO 8601 text."""
    return pd.Timestamp.now("UTC").isoformat()


def digest_json(value) -> str:
    """Hash JSON deterministically, ignoring dictionary key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: pathlib.Path) -> str:
    """Return a SHA-256 digest of the file bytes without loading them all."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package_versions() -> dict:
    """Return installed collection dependency versions for the run record."""
    result = {}
    for package in (
        "pandas",
        "numpy",
        "pyarrow",
        "requests",
        "exchange-calendars",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def collector_code_files() -> dict[str, str]:
    """Return repository-relative source paths and their SHA-256 digests.

    Includes the launcher and every Python module in this package. Hashing this
    mapping with digest_json produces the code fingerprint saved with each run
    and response. Paths are relative so moving the checkout preserves identity.
    """
    package = pathlib.Path(__file__).resolve().parent
    root = package.parent
    paths = [root / "collector.py", *sorted(package.rglob("*.py"))]
    return {
        path.relative_to(root).as_posix(): file_hash(path) for path in paths
    }
