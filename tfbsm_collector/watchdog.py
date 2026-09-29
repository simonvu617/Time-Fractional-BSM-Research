# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Restart an interrupted collector without creating a second writer.

Run this module instead of the collector for unattended downloads::

    python -m tfbsm_collector.watchdog --state-dir results/theta_run -- \
        --start 2017-01-01 --end 2025-12-31 --output-dir data/output

The collector's output lock remains the final protection for its data. This
supervisor holds a separate lock so two watchdogs cannot repeatedly compete to
launch it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time


COMPLETED_EXIT_CODES = {0, 2}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_state(state_dir: pathlib.Path, state: dict) -> None:
    """Atomically replace the small machine-readable watchdog status."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "watchdog.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(state, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _watchdog_lock(state_dir: pathlib.Path):
    """Hold one nonblocking watchdog lock for this run directory."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / ".watchdog.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Another collector watchdog is already using {state_dir}"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def supervise(
    collector_args: list[str],
    state_dir: pathlib.Path,
    *,
    initial_delay: float = 30.0,
    maximum_delay: float = 900.0,
    stable_seconds: float = 600.0,
) -> int:
    """Run the collector until completion, restarting unexpected exits.

    Exit code 0 means complete without reported gaps and 2 means complete with
    documented coverage gaps. Both are terminal outcomes. Any other exit is
    retried forever; capped backoff prevents a missing Theta Terminal or a
    persistent configuration problem from creating a tight restart loop.
    """
    if not collector_args:
        raise ValueError("collector arguments are required")
    if (
        initial_delay <= 0
        or maximum_delay < initial_delay
        or stable_seconds <= 0
    ):
        raise ValueError("watchdog delays must be positive and ordered")

    command = [
        sys.executable,
        "-u",
        "-B",
        "-m",
        "tfbsm_collector",
        *collector_args,
    ]
    delay = initial_delay
    restarts = 0
    with _watchdog_lock(state_dir):
        while True:
            started = time.monotonic()
            child = subprocess.Popen(command)
            _write_state(
                state_dir,
                {
                    "status": "running",
                    "watchdog_pid": os.getpid(),
                    "collector_pid": child.pid,
                    "restarts": restarts,
                    "started_at_utc": _utc_now(),
                    "command": command,
                },
            )
            try:
                exit_code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                exit_code = child.wait()
                _write_state(
                    state_dir,
                    {
                        "status": "stopped_by_operator",
                        "watchdog_pid": os.getpid(),
                        "collector_pid": child.pid,
                        "collector_exit_code": exit_code,
                        "finished_at_utc": _utc_now(),
                    },
                )
                return 130

            runtime = time.monotonic() - started
            if exit_code in COMPLETED_EXIT_CODES:
                _write_state(
                    state_dir,
                    {
                        "status": (
                            "complete" if exit_code == 0 else "complete_with_gaps"
                        ),
                        "watchdog_pid": os.getpid(),
                        "collector_pid": child.pid,
                        "collector_exit_code": exit_code,
                        "restarts": restarts,
                        "finished_at_utc": _utc_now(),
                    },
                )
                return exit_code

            restarts += 1
            if runtime >= stable_seconds:
                delay = initial_delay
            next_restart = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
                seconds=delay
            )
            _write_state(
                state_dir,
                {
                    "status": "waiting_to_restart",
                    "watchdog_pid": os.getpid(),
                    "collector_pid": child.pid,
                    "collector_exit_code": exit_code,
                    "restarts": restarts,
                    "next_restart_at_utc": next_restart.isoformat(),
                    "delay_seconds": delay,
                },
            )
            print(
                f"Watchdog: collector exited {exit_code}; restarting in {delay:g} seconds.",
                flush=True,
            )
            time.sleep(delay)
            delay = min(maximum_delay, delay * 2)


def main(argv: list[str] | None = None) -> int:
    """Parse watchdog settings and supervise the collector command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=pathlib.Path)
    parser.add_argument("--initial-delay", type=float, default=30.0)
    parser.add_argument("--maximum-delay", type=float, default=900.0)
    parser.add_argument("--stable-seconds", type=float, default=600.0)
    parser.add_argument("collector_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    collector_args = list(args.collector_args)
    if collector_args[:1] == ["--"]:
        collector_args.pop(0)
    try:
        return supervise(
            collector_args,
            args.state_dir.expanduser().resolve(),
            initial_delay=args.initial_delay,
            maximum_delay=args.maximum_delay,
            stable_seconds=args.stable_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
