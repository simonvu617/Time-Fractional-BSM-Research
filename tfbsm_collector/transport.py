# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Theta Terminal HTTP transport with shared limits and bounded retries.

ThetaClient downloads response bytes without interpreting market observations.
All workers share its request budget and stop signal; each thread owns its HTTP
session. Storage and response validation live in separate modules.
"""

import contextlib
import hashlib
import socket
import threading
import time
import urllib.parse
from typing import BinaryIO

import requests
import requests.adapters

from tfbsm_collector import config, planning


class CollectionStopped(RuntimeError):
    """A shared stop signal preventing further collection work."""


class ThetaClient:
    """Shared request limits and per-thread connections to Theta Terminal.

    Attributes:
        cfg: Terminal URL and account request limits.
        local: Thread-local HTTP sessions.
        semaphore: Cap on requests whose response bodies are still arriving.
        pace_lock: Protects start-time scheduling and the first stop reason.
        next_allowed: Next permitted request start on the monotonic clock.
        stop_event: Cancellation signal shared across collection workers.
        stop_reason: First reason recorded when stopping the run.
    """

    def __init__(self, cfg: config.CollectorConfig):
        """Initialize shared limits and state without contacting Theta."""
        self.cfg = cfg
        self.local = threading.local()
        self.semaphore = threading.BoundedSemaphore(cfg.max_inflight_requests)
        self.pace_lock = threading.Lock()
        self.next_allowed = 0.0
        self.stop_event = threading.Event()
        self.stop_reason = ""

    def stop(self, reason: str) -> None:
        """Stop new work, preserving the first recorded reason."""
        # Keep the first failure reason so later cancellations do not overwrite
        # it.
        with self.pace_lock:
            if not self.stop_event.is_set():
                self.stop_reason = reason
                self.stop_event.set()

    def check_running(self) -> None:
        """Check whether new work may proceed.

        Raises:
            CollectionStopped: Another worker or the operator stopped the run.
        """
        if self.stop_event.is_set():
            raise CollectionStopped(self.stop_reason)

    def ensure_available(self) -> None:
        """Check the terminal port before scheduling historical downloads.

        A reachable port does not verify entitlements or historical coverage.

        Raises:
            RuntimeError: The configured terminal address is unreachable.
        """
        address = urllib.parse.urlsplit(self.cfg.base_url)
        try:
            with socket.create_connection(
                (
                    address.hostname,
                    address.port or (443 if address.scheme == "https" else 80),
                ),
                timeout=2,
            ):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"Theta Terminal is unreachable at {self.cfg.base_url}. Start it and rerun."
            ) from exc

    def session(self) -> requests.Session:
        """Return this thread's HTTP session with explicit retry control."""
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()

            # Retry explicitly so every attempt obeys the shared account budget.
            adapter = requests.adapters.HTTPAdapter(
                max_retries=0, pool_connections=2, pool_maxsize=2
            )
            self.local.session.mount("http://", adapter)
            self.local.session.mount("https://", adapter)
        return self.local.session

    @contextlib.contextmanager
    def request_slot(self):
        """Provide a shared HTTP slot, honoring pacing and cancellation.

        Returns:
            A context manager holding one slot. Keep it open until the full
            response body has arrived, including for each retry attempt.

        Raises:
            CollectionStopped: The run stops before the request may begin.
        """
        self.check_running()
        with self.semaphore:
            self.check_running()
            with self.pace_lock:
                wait = max(0.0, self.next_allowed - time.monotonic())
                self.next_allowed = (
                    max(time.monotonic(), self.next_allowed)
                    + 1 / self.cfg.max_requests_per_second
                    if self.cfg.max_requests_per_second
                    else time.monotonic()
                )

            # A stop signal wakes paced workers before they start another
            # request.
            self.stop_event.wait(wait)
            self.check_running()
            yield

    def download(self, request: planning.Request, payload: BinaryIO) -> dict:
        """Stream a response into a borrowed file, retrying transient failures.

        Args:
            request: Theta endpoint and query parameters.
            payload: Writable, seekable binary stream. Truncated on each attempt
                and rewound before return; the caller retains ownership.

        Returns:
            HTTP status, headers, timing, payload hash/size, and optional error
            details after at most six attempts. This does not parse the response
            or establish that an HTTP 200 body contains valid market data.

        Raises:
            CollectionStopped: The shared stop signal interrupts further
                retries.
        """
        meta = {}
        for attempt in range(6):
            started = time.perf_counter()
            retry_after = 0.0
            payload.seek(0)
            # A retry must replace the failed attempt, not append a second
            # response and make duplicated observations appear to be one
            # successful pull.
            payload.truncate()

            fingerprint = hashlib.sha256()
            meta = {"attempts": attempt + 1}
            try:
                with self.request_slot():
                    with self.session().get(
                        self.cfg.base_url.rstrip("/") + request.endpoint,
                        params=request.params,
                        timeout=(10, 120),
                        stream=True,
                    ) as response:
                        meta.update(
                            request_url=response.url,
                            status_code=response.status_code,
                            response_headers=dict(response.headers),
                        )

                        # Hold the shared slot until the response finishes
                        # arriving, not merely until its headers arrive. Bulk
                        # replies stream to disk.
                        for piece in response.iter_content(
                            chunk_size=256 * 1024
                        ):
                            payload.write(piece)
                            fingerprint.update(piece)
                meta.update(
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload_sha256=fingerprint.hexdigest(),
                    payload_bytes=payload.tell(),
                )
                payload.seek(0)
                # Theta uses 472 for no data. Storage still checks a 200
                # response's schema and identity before accepting it as market
                # observations.
                if response.status_code in {200, 472}:
                    return meta
                preview = payload.read(500).decode("utf-8", errors="replace")
                payload.seek(0)
                meta["error"] = f"HTTP {response.status_code}: {preview}"

                # 429 is throttling, 474 a lost vendor connection, and 571 a
                # vendor restart. Retry these and temporary server errors within
                # the attempt cap.
                if (
                    response.status_code
                    not in {429, 474, 500, 502, 503, 504, 571}
                    or attempt == 5
                ):
                    # Access/configuration errors and exhausted terminal retries
                    # need a stop, not thousands of additional historical
                    # requests.
                    if response.status_code in {
                        400,
                        401,
                        403,
                        404,
                        429,
                        471,
                        473,
                        474,
                        475,
                        476,
                        478,
                        571,
                    }:
                        self.stop(
                            f"Theta HTTP {response.status_code} for {request.endpoint}: "
                            "check Theta access, terminal state, and request settings, then rerun."
                        )
                    return meta
                try:
                    retry_after = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    pass
            except requests.RequestException as exc:
                # Keep a truncated final body as failure evidence; never parse
                # its valid-looking prefix as a completed historical download.
                meta.update(
                    error=repr(exc),
                    status_code=None,
                    response_incomplete=True,
                    payload_sha256=fingerprint.hexdigest(),
                    payload_bytes=payload.tell(),
                )
                payload.seek(0)
                if attempt == 5:
                    self.stop(
                        f"Theta connection failed after six attempts for {request.endpoint}; rerun when it is available."
                    )
                    return meta

            self.stop_event.wait(
                max(min(30.0, 2**attempt), min(max(retry_after, 0), 120.0))
            )
            self.check_running()
        return meta
