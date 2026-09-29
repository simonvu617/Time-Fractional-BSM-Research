# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Atomic Parquet artifacts, compact response receipts, and output locking.

RequestStore converts verified responses to compressed Parquet, publishes
receipts, and reuses compatible caches. Session manifests refer to these files
instead of storing another copy of each response.
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import sqlite3
import tempfile
import threading
import uuid
from typing import BinaryIO, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tfbsm_collector import config, planning, provenance, transport, validation

if os.name == "nt":
    import msvcrt
else:
    import fcntl


# Version 2 requires strict CSV width checks. Older parsed files alone cannot
# prove that their parser preserved every field; verified CSVs can be
# revalidated.
RAW_SCHEMA_VERSION = 2

# Both mean request success. no_data means no rows arrived, never zero price
# or zero activity in the underlying market.
GOOD_REQUEST_STATUSES = {"available", "no_data"}


def read_json(path: pathlib.Path) -> dict:
    """Return a UTF-8 JSON metadata document as a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def atomic_output(path: pathlib.Path):
    """Provide a temporary path that replaces the destination on success.

    Args:
        path: Final output path; missing parent directories are created.

    Returns:
        A context manager yielding a neighboring temporary path. Exceptions
        leave the previous destination intact and remove the temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A neighboring temporary file permits atomic replacement on the same
    # filesystem; a partial write cannot replace an earlier complete artifact.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=path.suffix, delete=False
    ) as handle:
        temp_path = pathlib.Path(handle.name)
    try:
        yield temp_path
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: pathlib.Path, value: dict) -> None:
    """Atomically save readable JSON, rejecting nonfinite numeric values."""
    with atomic_output(path) as temp:
        temp.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )


def write_parquet(path: pathlib.Path, frame: pd.DataFrame) -> None:
    """Atomically save a table with lossless Zstandard compression."""
    with atomic_output(path) as temp:
        frame.to_parquet(temp, index=False, compression="zstd")


def file_receipt(
    path: pathlib.Path, root: pathlib.Path, frame: pd.DataFrame | None = None
) -> dict:
    """Return a portable record of an artifact's identity and table shape.

    Args:
        path: Existing file beneath root.
        root: Output root used to store a relative path.
        frame: Written table, if row/column metadata should be included.

    Returns:
        Relative path, size, modification time, SHA-256, and optional table
        rows/columns. This identifies bytes, not economic correctness.
    """
    stat = path.stat()
    receipt = {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": provenance.file_hash(path),
    }
    if frame is not None:
        receipt.update(rows=len(frame), columns=list(frame.columns))
    return receipt


def artifact_valid(receipt: dict, root: pathlib.Path) -> bool:
    """Check an artifact against a receipt without rereading unchanged data.

    Args:
        receipt: Metadata from file_receipt, possibly with table shape.
        root: Output root containing the referenced relative path.

    Returns:
        Whether the file's size and optional Parquet shape match. A changed
        modification time triggers a hash check; unchanged size/mtime alone do
        not establish that the bytes were freshly rehashed.
    """
    try:
        path = root / receipt["path"]
        stat = path.stat()

        if stat.st_size != receipt["size"]:
            return False
        # Large collections cannot rehash terabytes on every resume. Rehash only
        # when mtime changes; this is a reuse check, not a fresh full integrity
        # audit.
        if (
            stat.st_mtime_ns != receipt["mtime_ns"]
            and provenance.file_hash(path) != receipt["sha256"]
        ):
            return False
        if "rows" in receipt:
            with pq.ParquetFile(path) as parquet:
                if (
                    parquet.metadata.num_rows != receipt["rows"]
                    or parquet.schema_arrow.names != receipt["columns"]
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextlib.contextmanager
def output_lock(output_dir: pathlib.Path):
    """Provide an exclusive process lock for an output root.

    Args:
        output_dir: Output root; created if it does not exist.

    Returns:
        A context manager holding the lock until exit. The lock file remains on
        disk, but the operating system releases ownership after a crash.

    Raises:
        RuntimeError: Another collector holds the output lock.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep this lock file: unlinking it could let another process lock a new
    # file while the existing writer still owns the old one.
    with (output_dir / ".collector.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Another collector is writing to {output_dir}"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RequestStore:
    """Response storage shared by stock, option, and reference collection.

    Attributes:
        cfg: Parsing, retention, and cache settings.
        root: Output root for relative artifact receipts.
        collection_dir: Session manifests and tables for the sampling policy.
        client: Shared Theta transport and cancellation signal.
        code_files: Repository-relative source paths and their SHA-256 hashes.
        code_sha256: Digest of code_files, captured when the store is created.
        locks: Shared locks preventing concurrent writes to one request key.
    """

    def __init__(self, cfg: config.CollectorConfig):
        """Initialize transport and provenance without creating output."""
        self.cfg = cfg
        self.root = cfg.output_dir
        self.collection_dir = self.root / "collection" / cfg.policy_id
        self.client = transport.ThetaClient(cfg)

        # Capture all package sources once. Hashing only this storage module
        # would miss changes to selection, request planning, or the command-line
        # workflow.
        self.code_files = provenance.collector_code_files()
        self.code_sha256 = provenance.digest_json(self.code_files)

        # Requests sharing a key also share a lock. Unrelated requests may use
        # different locks and continue downloading concurrently.
        self.locks = tuple(threading.Lock() for _ in range(128))
        self.index_lock = threading.RLock()

    @contextlib.contextmanager
    def index(self):
        """Open the compact receipt index for one short, serialized transaction.

        Downloads and Parquet conversion happen outside this lock. SQLite
        replaces millions of tiny metadata/pointer files; it is not the market
        data format. Each committed response still identifies its Parquet rows.
        """
        with self.index_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with contextlib.closing(
                sqlite3.connect(self.root / "index.sqlite3", timeout=30)
            ) as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS responses "
                    "(attempt TEXT PRIMARY KEY, request_id TEXT, meta TEXT)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS latest "
                    "(request_id TEXT PRIMARY KEY, attempt TEXT)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS sessions "
                    "(policy TEXT, symbol TEXT, day TEXT, manifest TEXT, "
                    "PRIMARY KEY(policy, symbol, day))"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS code "
                    "(sha256 TEXT PRIMARY KEY, files TEXT)"
                )
                with db:
                    yield db

    def metadata(self, record: dict) -> dict:
        """Read response provenance, resolving any later storage compaction."""
        if "attempt_id" not in record:
            return read_json(self.root / record["metadata"]["path"])
        with self.index() as db:
            row = db.execute(
                "SELECT meta FROM responses WHERE attempt=?",
                (record["attempt_id"],),
            ).fetchone()
        if row is None:
            raise ValueError("Response receipt missing from index")
        return json.loads(row[0])

    def session(self, symbol: str, day: pd.Timestamp) -> dict:
        """Read a session's coverage and selection provenance from the index."""
        with self.index() as db:
            row = db.execute(
                "SELECT manifest FROM sessions WHERE policy=? AND symbol=? AND day=?",
                (self.cfg.policy_id, symbol, str(day.date())),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"No session for {symbol} {day.date()}")
        return json.loads(row[0])

    def save_sessions(self, manifests: list[dict]) -> None:
        """Commit completed session summaries together after their files exist."""
        with self.index() as db:
            db.executemany(
                "INSERT OR REPLACE INTO sessions VALUES(?,?,?,?)",
                [
                    (
                        self.cfg.policy_id,
                        m["symbol"],
                        m["trade_day"],
                        json.dumps(m, separators=(",", ":"), allow_nan=False),
                    )
                    for m in manifests
                ],
            )

    def directory(self, request: planning.Request) -> pathlib.Path:
        """Return the cache directory for the request's dataset and identity."""
        date = request.params.get(
            "date", request.params.get("start_date", "reference")
        )
        date = str(date).replace("-", "")
        symbol = request.params["symbol"]
        return (
            self.root
            / "raw_cache"
            / request.dataset
            / f"symbol={symbol}"
            / f"date={date}"
            / f"request={request.request_id}"
        )

    def reusable(self, record: dict) -> bool:
        """Check a response receipt for both cache and session resume.

        A successful status alone is insufficient: every referenced artifact
        must exist, and the requested raw-payload/empty-refresh policy applies.
        """
        if record["status"] not in GOOD_REQUEST_STATUSES:
            return False
        if self.cfg.refresh_no_data and record["status"] == "no_data":
            return False
        if self.cfg.store_raw_payloads and not record.get("payload"):
            return False
        if "attempt_id" in record:
            try:
                meta = self.metadata(record)
                return all(
                    artifact_valid(meta.get(name), self.root)
                    for name in ("data", "payload")
                    if name == "data" or meta.get(name)
                )
            except (OSError, ValueError, KeyError, TypeError):
                return False
        return all(
            artifact_valid(record.get(name), self.root)
            for name in ("data", "metadata", "payload")
            if name != "payload" or record.get(name)
        )

    def cached(self, request: planning.Request) -> dict | None:
        """Return a reusable request receipt, or None when collection is needed.

        Args:
            request: Exact endpoint, parameters, and retained contract scope.

        Returns:
            A receipt only when request identity, schema, time zone, status, and
            referenced artifacts pass validation. Retention and empty-response
            refresh settings can require a new download even if data exist.
        """
        if (self.root / "index.sqlite3").exists():
            with self.index() as db:
                row = db.execute(
                    "SELECT responses.meta FROM latest JOIN responses "
                    "ON latest.attempt=responses.attempt WHERE latest.request_id=?",
                    (request.request_id,),
                ).fetchone()
            if row:
                meta = json.loads(row[0])
                if (
                    meta["request"] == request.identity()
                    and meta["raw_schema_version"] == RAW_SCHEMA_VERSION
                    and meta["timestamp_timezone"] == self.cfg.exchange_tz
                ):
                    record = self.record(request, meta)
                    if self.reusable(record):
                        return record
        path = self.directory(request) / "meta.json"
        try:
            # Current caches point to immutable attempts. Older caches can still
            # use meta.json directly; do not move or rewrite those existing
            # files.
            index = path.with_name("latest.json")
            if index.exists():
                receipt = read_json(index)["metadata"]
                if not artifact_valid(receipt, self.root):
                    return None
                path = self.root / receipt["path"]
            meta = read_json(path)
            if (
                meta["request"] != request.identity()
                or meta["raw_schema_version"] != RAW_SCHEMA_VERSION
                or meta.get("timestamp_timezone") != self.cfg.exchange_tz
            ):
                return None
            record = self.record(request, meta, path)
            return record if self.reusable(record) else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def record(
        self,
        request: planning.Request,
        meta: dict,
        meta_path: pathlib.Path | None = None,
    ) -> dict:
        """Return a compact manifest receipt referencing full response metadata.

        Args:
            request: Descriptor for the saved response.
            meta: Response metadata, including status and optional artifacts.
            meta_path: Existing metadata file beneath the output root.

        Returns:
            Request identity, outcome, clock/retention summaries, and receipts.
        """
        return {
            "request_id": request.request_id,
            "dataset": request.dataset,
            "params": request.params,
            "status": meta["status"],
            "row_count": meta.get("row_count", 0),
            "status_code": meta.get("status_code"),
            "observation_semantics": request.observation_semantics(),
            "retention": meta.get("retention", {}),
            "error": meta.get("error", ""),
            "data": meta.get("data"),
            "payload": meta.get("payload"),
            **(
                {"metadata": file_receipt(meta_path, self.root)}
                if meta_path
                else {"attempt_id": meta["attempt_id"]}
            ),
        }

    def collect(
        self, request: planning.Request, *, refresh: bool = False
    ) -> dict:
        """Reuse a valid response or download, validate, and save a new attempt.

        Args:
            request: Descriptor of the data and retention scope to collect.
            refresh: Whether to bypass the current cache, as for date
                catalogues.

        Returns:
            A compact receipt. Request or response failures have explicit status
            values and retain their original bytes for inspection.

        Raises:
            transport.CollectionStopped: The shared run has been stopped.
            OSError: Response artifacts could not be read or written.
        """
        self.client.check_running()
        with self.locks[hash(request.request_id) % len(self.locks)]:
            self.client.check_running()
            cached = None if refresh else self.cached(request)
            if cached:
                return cached
            self.root.mkdir(parents=True, exist_ok=True)

            # Bulk responses need only one on-disk staging file, even when exact
            # successful CSV retention is disabled. Closing this scope deletes
            # it.
            with tempfile.TemporaryFile(dir=self.root) as payload:
                response_meta = self.client.download(request, payload)
                status = response_meta.get("status_code")
                empty = pd.DataFrame(columns=request.required_columns)
                if status not in {200, 472}:
                    return self.save(
                        request, empty, response_meta, payload, "request_error"
                    )
                if status == 472:
                    return self.save(request, empty, response_meta, payload)
                try:
                    content_type = (
                        response_meta.get("response_headers", {})
                        .get("Content-Type", "")
                        .lower()
                    )
                    if content_type and not any(
                        t in content_type
                        for t in ("csv", "text/plain", "octet-stream")
                    ):
                        raise ValueError(
                            f"Unexpected response content type: {content_type}"
                        )

                    frames = validation.csv_frames(
                        payload, self.cfg.raw_chunk_rows
                    )
                except (ValueError, UnicodeError) as exc:
                    response_meta["error"] = f"Invalid CSV response: {exc}"
                    return self.save(
                        request,
                        empty,
                        response_meta,
                        payload,
                        "invalid_response",
                    )
                try:
                    return self.save(request, frames, response_meta, payload)
                finally:
                    frames.close()

    def save(
        self,
        request: planning.Request,
        frames: pd.DataFrame | Iterable[pd.DataFrame],
        response_meta: dict,
        payload: bytes | BinaryIO | None = None,
        status: str | None = None,
    ) -> dict:
        """Write an immutable response attempt and publish its artifact receipt.

        Args:
            request: Expected identity and retained option contract scope.
            frames: Raw vendor table or iterable of parsing batches.
            response_meta: HTTP/provenance dictionary. Consumed and updated with
                validation failures; fetched_at_utc is removed when transferred.
            payload: Original bytes or borrowed binary stream. May be rewound
                and read, but is not closed. Failure bytes are always preserved.
            status: Optional preexisting failure status to keep during parsing.

        Returns:
            A receipt for the new attempt, including empty or invalid responses.
            Only successful attempts advance the reusable cache pointer; a
            failed refresh leaves the previous successful response available.
        """
        # An attempt gets new files, including on failure. Earlier run receipts
        # must keep pointing to the exact response they originally described.
        attempt_id = uuid.uuid4().hex
        directory = self.root / "responses"
        if isinstance(frames, pd.DataFrame):
            source = frames
            frames = (
                source.iloc[start : start + self.cfg.raw_chunk_rows]
                for start in range(
                    0, max(len(source), 1), self.cfg.raw_chunk_rows
                )
            )
        diagnostics = validation.RawDiagnostics(request, self.cfg)
        retained = (
            None
            if request.retained_contract_keys is None
            else set(request.retained_contract_keys)
        )
        windows = request.retained_contract_windows
        starts = {key: start for key, start, _ in windows or ()}
        ends = {key: end for key, _, end in windows or ()}
        excluded_rows = 0

        data_path = directory / f"{attempt_id}.parquet"
        writer, columns = None, []
        with atomic_output(data_path) as temp:
            try:
                for raw in frames:
                    missing = set(request.required_columns) - set(raw.columns)

                    if missing or any(
                        str(column).startswith("collector_")
                        for column in raw.columns
                    ):
                        status = status or "invalid_response"
                        response_meta["error"] = (
                            f"Missing required columns {sorted(missing)} or reserved collector_ column"
                        )
                    # Inspect every parsed row before filtering. A misrouted or
                    # malformed unselected row must still invalidate the overall
                    # response.
                    frame = diagnostics.add(raw.astype("string").fillna(""))
                    if (
                        retained is not None or windows is not None
                    ) and not missing:
                        # Filter identities only: keep every timestamp, vendor
                        # field, zero bid, and duplicate belonging to a selected
                        # contract.
                        keys = planning.option_contract_keys(frame)
                        keep = (
                            keys.isin(retained)
                            if retained is not None
                            else pd.Series(True, index=frame.index)
                        )
                        if windows is not None:
                            # Quiet contracts can have an older last_trade.
                            # EOD retention follows the report's creation date.
                            clock = (
                                "created"
                                if request.endpoint.endswith("/eod")
                                else "timestamp"
                            )
                            dates = (
                                frame[f"collector_{clock}_utc"]
                                .dt.tz_convert(self.cfg.exchange_tz)
                                .dt.strftime("%Y-%m-%d")
                            )
                            keep &= (
                                dates.ge(keys.map(starts))
                                & dates.le(keys.map(ends))
                            ).fillna(False)
                        excluded_rows += int((~keep).sum())
                        frame = frame.loc[keep]

                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        columns = list(frame.columns)
                        writer = pq.ParquetWriter(
                            temp, table.schema, compression="zstd"
                        )

                    # An excluded-only batch establishes the schema without
                    # adding an empty Parquet row group. An empty selected
                    # sample still needs coverage checks.
                    if len(table):
                        writer.write_table(table)
            # A malformed later batch invalidates the response even if a prefix
            # was saved successfully. Retain the complete failure payload below.
            except (ValueError, UnicodeError) as exc:
                status = status or "invalid_response"
                response_meta["error"] = f"Invalid CSV/table response: {exc}"
            finally:
                if writer is not None:
                    writer.close()
            if writer is None:
                frame = diagnostics.add(
                    pd.DataFrame(
                        columns=request.required_columns, dtype="string"
                    )
                )
                columns = list(frame.columns)
                frame.to_parquet(temp, index=False, compression="zstd")

        # Use actual written rows in the receipt: a failed conversion can leave
        # fewer stored rows than the diagnostics parsed before the failure.
        with pq.ParquetFile(data_path) as parquet:
            stored_rows, columns = (
                parquet.metadata.num_rows,
                parquet.schema_arrow.names,
            )
        quality = diagnostics.finish()
        if quality["response_identity_issues"]:
            status = status or "invalid_response"
            response_meta["error"] = ", ".join(
                quality["response_identity_issues"]
            )
        if status is None:
            # A nonempty bulk reply with no selected rows is still an available
            # response; selected-contract coverage will report the missing
            # observations.
            status = "no_data" if diagnostics.rows == 0 else "available"

        meta = {
            "attempt_id": attempt_id,
            "request": request.identity(),
            "request_id": request.request_id,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "timestamp_timezone": self.cfg.exchange_tz,
            "fetched_at_utc": response_meta.pop(
                "fetched_at_utc", provenance.utc_now()
            ),
            "saved_at_utc": provenance.utc_now(),
            "status": status,
            "row_count": stored_rows,
            "quality": quality,
            "quality_scope": "parsed_rows_before_storage_filter",
            "retention": {
                "mode": "full_response"
                if retained is None
                and request.retained_contract_windows is None
                else "selected_contract_windows",
                "parsed_rows": diagnostics.rows,
                "excluded_rows": excluded_rows,
            },
            "observation_semantics": request.observation_semantics(),
            "collector_code_sha256": self.code_sha256,
            **response_meta,
        }
        meta["data"] = {
            **file_receipt(data_path, self.root),
            "rows": stored_rows,
            "columns": columns,
        }

        # Failure bytes explain vendor/schema errors. Opting into successful
        # CSVs also retains the unselected strikes excluded from Parquet, at
        # extra disk cost.
        if payload is not None and (
            self.cfg.store_raw_payloads or status not in GOOD_REQUEST_STATUSES
        ):
            payload_path = directory / f"{attempt_id}.csv"
            with atomic_output(payload_path) as temp:
                source = (
                    io.BytesIO(payload)
                    if isinstance(payload, bytes)
                    else payload
                )
                source.seek(0)
                with temp.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=256 * 1024)
            meta["payload"] = file_receipt(payload_path, self.root)

        with self.index() as db:
            db.execute(
                "INSERT OR IGNORE INTO code VALUES(?,?)",
                (self.code_sha256, json.dumps(self.code_files)),
            )
            db.execute(
                "INSERT INTO responses VALUES(?,?,?)",
                (
                    attempt_id,
                    request.request_id,
                    json.dumps(meta, separators=(",", ":"), allow_nan=False),
                ),
            )
            # Advance the reusable pointer only after success. A failed refresh must
            # leave the previous good response available for another run.
            if status in GOOD_REQUEST_STATUSES:
                db.execute(
                    "INSERT OR REPLACE INTO latest VALUES(?,?)",
                    (request.request_id, attempt_id),
                )
        return self.record(request, meta)

    def read(
        self, record: dict, day: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return a successful saved table, or an empty table for a failed pull.

        Use iter_frames for large responses; this method loads the full table.
        """
        if record["status"] not in GOOD_REQUEST_STATUSES or not record.get(
            "data"
        ):
            return pd.DataFrame()
        frames = list(self.iter_frames(record, day=day))
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    def iter_frames(
        self,
        record: dict,
        columns: list[str] | None = None,
        day: pd.Timestamp | None = None,
    ):
        """Yield bounded Parquet batches from a successful saved response.

        Args:
            record: Request receipt containing status and a data artifact.
            columns: Optional vendor/parsed columns to read; None reads them
                all.
            day: Optional exchange-local date slice from a monthly response.

        Yields:
            DataFrames in stored order. Failed or absent data yield no batches.
        """
        if record["status"] in GOOD_REQUEST_STATUSES and record.get("data"):
            data = (
                self.metadata(record)["data"]
                if "attempt_id" in record
                else record["data"]
            )
            with pq.ParquetFile(self.root / data["path"]) as parquet:
                if data.get("row_groups") == []:
                    empty = parquet.schema_arrow.empty_table().to_pandas()
                    yield empty if columns is None else empty.loc[:, columns]
                    return
                for batch in parquet.iter_batches(
                    batch_size=self.cfg.raw_chunk_rows,
                    columns=columns,
                    row_groups=data.get("row_groups"),
                ):
                    frame = batch.to_pandas()
                    if day is not None:
                        column = next(
                            (
                                c
                                for c in ("timestamp", "created", "date")
                                if c in frame
                            ),
                            None,
                        )
                        if column:
                            dates = pd.to_datetime(
                                frame[column], format="mixed", errors="coerce"
                            ).dt.strftime("%Y-%m-%d")
                            frame = frame.loc[dates.eq(str(day.date()))]
                    yield frame

    def compact(self, records: list[dict], label: str) -> None:
        """Pack successful responses into shared Parquet files by schema.

        Row-group locators preserve response order, duplicate rows, and every
        vendor field. Publish new locations in one SQLite transaction before
        deleting the superseded individual files. A failed refresh never
        replaces a successful response. Existing legacy caches are untouched.
        """
        groups = {}
        for record in records:
            if (
                "attempt_id" not in record
                or record["status"] not in GOOD_REQUEST_STATUSES
            ):
                continue
            meta = self.metadata(record)
            if "row_groups" in meta["data"]:
                continue
            path = self.root / meta["data"]["path"]
            schema = pq.read_schema(path)
            key = (record["dataset"], str(schema.remove_metadata()))
            groups.setdefault(key, {})[record["attempt_id"]] = meta
        for (dataset, _), attempts in groups.items():
            path = (
                self.root
                / "parquet"
                / dataset
                / label
                / f"{uuid.uuid4().hex}.parquet"
            )
            old_paths, updated = [], []
            with atomic_output(path) as temp:
                writer, group_index = None, 0
                try:
                    for meta in attempts.values():
                        old = self.root / meta["data"]["path"]
                        with pq.ParquetFile(old) as source:
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    temp,
                                    source.schema_arrow,
                                    compression="zstd",
                                )
                            row_groups = []
                            for batch in source.iter_batches(
                                batch_size=self.cfg.raw_chunk_rows
                            ):
                                writer.write_batch(
                                    batch, row_group_size=len(batch)
                                )
                                row_groups.append(group_index)
                                group_index += 1
                        old_paths.append(old)
                        updated.append((meta, row_groups))
                finally:
                    if writer:
                        writer.close()
            receipt = file_receipt(path, self.root)
            with self.index() as db:
                for meta, row_groups in updated:
                    meta["data"] = {
                        **receipt,
                        "row_groups": row_groups,
                        "response_rows": meta["row_count"],
                    }
                    db.execute(
                        "UPDATE responses SET meta=? WHERE attempt=?",
                        (
                            json.dumps(
                                meta, separators=(",", ":"), allow_nan=False
                            ),
                            meta["attempt_id"],
                        ),
                    )
            for old in old_paths:
                # Only individually saved response files owned by this store
                # are removed; retained failure/raw payloads remain beside them.
                old.unlink(missing_ok=True)
