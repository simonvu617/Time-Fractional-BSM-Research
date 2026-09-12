# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Exercise collection with synthetic Theta responses and temporary outputs.

Run with python -m unittest discover -s tests -p "test_collector.py".
No test needs Theta Terminal, credentials, or market-data downloads.
"""

import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import pathlib
import tempfile
import threading
import unittest
from unittest import mock

import pandas as pd

from tfbsm_collector import (
    cli,
    config,
    planning,
    provenance,
    selection,
    storage,
    transport,
    validation,
    workflow,
)

DAY = pd.Timestamp("2025-01-02")
EXPIRATIONS = (
    "2025-01-10",
    "2025-01-17",
    "2025-01-31",
    "2025-03-07",
    "2025-05-02",
)


def _contracts(expirations=EXPIRATIONS, strikes=("100", "105")):
    return [
        {
            "symbol": "SPY",
            "expiration": expiration,
            "strike": strike,
            "right": right,
        }
        for expiration in expirations
        for strike in strikes
        for right in ("call", "put")
    ]


def _quote(timestamp, midpoint="100", **identity):
    return {
        **identity,
        "timestamp": timestamp,
        "bid_size": "10",
        "bid_exchange": "1",
        "bid": midpoint,
        "bid_condition": "0",
        "ask_size": "11",
        "ask_exchange": "1",
        "ask": midpoint,
        "ask_condition": "0",
    }


def _payload(request):
    """Build one synthetic CSV, including OI-only and unselected contracts."""
    if "/list/dates" in request.endpoint:
        rows = [{"date": "2025-01-02"}]
    elif request.dataset.startswith("corporate_"):
        return (",".join(request.required_columns) + "\n").encode()
    elif request.dataset == "interest_rate_eod":
        rows = [{"created": "2025-01-02", "rate": "4.25"}]
    elif request.dataset in {"quoted_contracts", "traded_contracts"}:
        rows = [
            contract
            for contract in _contracts()
            if not (contract["strike"] == "105" and contract["right"] == "put")
        ]
    elif request.dataset == "option_open_interest":
        rows = [
            {
                **contract,
                "timestamp": "2025-01-02T09:00:00",
                "open_interest": "12",
            }
            for contract in _contracts()
        ]
    elif request.endpoint.endswith("/eod"):
        identities = (
            _contracts() if request.endpoint.startswith("/option/") else [{}]
        )
        rows = [
            {
                **contract,
                "created": "2025-01-02T17:15:00",
                "last_trade": "2025-01-02T15:59:00",
                "open": "1.2",
                "high": "1.4",
                "low": "1.1",
                "close": "1.3",
                "volume": "20",
                "count": "3",
            }
            for contract in identities
        ]
    else:
        sampled = "/history/" in request.endpoint
        clocks = (
            [f"{hour:02}:30:00" for hour in range(9, 16)]
            if sampled
            else ["15:55:00"]
        )
        if request.endpoint.startswith("/option/"):
            identities = _contracts((request.params["expiration"],))
            # The bulk response includes a valid strike outside the selected universe.
            identities.append({**identities[0], "strike": "999"})
            rows = [
                _quote(f"2025-01-02T{clock}", "1.230000", **contract)
                for contract in identities
                for clock in clocks
            ]
            if sampled:
                rows.append(
                    rows[0].copy()
                )  # Raw duplicates must survive storage.
        else:
            rows = [
                _quote(
                    f"2025-01-02T{clock}",
                    "100" if clock < "13:00:00" else "105",
                )
                for clock in clocks
            ]
    return pd.DataFrame(rows).to_csv(index=False).encode()


def _download_fixture(client, request, payload):
    data = _payload(request)
    payload.write(data)
    payload.seek(0)
    return {
        "status_code": 200,
        "response_headers": {"Content-Type": "text/csv"},
        "payload_bytes": len(data),
        "payload_sha256": hashlib.sha256(data).hexdigest(),
    }


class PlanningAndSelectionTest(unittest.TestCase):
    def test_pro_history_clips_stock_buffer_and_reports_reference_access(self):
        cfg = config.CollectorConfig()
        windows = planning.collection_windows(cfg, cfg.start_date, cfg.end_date)
        self.assertEqual(windows["history_start"], "2012-06-01")
        self.assertEqual(windows["lookback_dates"], [])
        self.assertEqual(len(windows["unavailable_lookback_dates"]), 60)
        requests = list(
            planning.reference_requests(
                cfg,
                [config.UNIVERSE[0]],
                cfg.start_date,
                cfg.end_date,
                ["SOFR", "TREASURY_M3"],
                include_stock_lookback=True,
            )
        )
        self.assertEqual(len(requests), 4)
        self.assertFalse(
            any(r.endpoint.startswith("/index/") for r in requests)
        )
        for request in requests:
            if request.dataset == "interest_rate_eod":
                self.assertEqual(request.params["start_date"], "2024-01-01")
            else:
                self.assertEqual(request.params["start_date"], "2012-06-01")
        gaps = planning.reference_access_gaps(
            cfg, windows, ["SOFR", "TREASURY_M3"], include_stock_lookback=True
        )
        self.assertEqual(
            {g["reason"] for g in gaps},
            {
                "index_subscription_unavailable",
                "before_rate_subscription_history_start",
                "before_stock_pro_history_start",
            },
        )
        rate_gap = next(
            g for g in gaps if g.get("dataset") == "interest_rate_eod"
        )
        self.assertEqual(
            rate_gap["unrequested_eod_session_dates"][-1], "2023-12-29"
        )
        later = planning.collection_windows(cfg, "2013-01-02", "2013-01-03")
        self.assertEqual(len(later["lookback_dates"]), 60)
        self.assertEqual(later["unavailable_lookback_dates"], [])

    def test_separate_paid_reference_tiers_use_their_own_history_bounds(self):
        cfg = dataclasses.replace(
            config.CollectorConfig(),
            index_subscription="pro",
            rate_subscription="value",
            lookback_sessions=0,
        )
        start, end = "2016-12-30", "2017-01-03"
        requests = list(
            planning.reference_requests(cfg, [], start, end, ["TREASURY_M3"])
        )
        self.assertEqual(len(requests), 4)
        rates = next(r for r in requests if r.dataset == "interest_rate_eod")
        index = next(r for r in requests if r.dataset == "index_eod")
        self.assertEqual(rates.params["start_date"], start)
        self.assertEqual(index.params["start_date"], "2017-01-01")
        gaps = planning.reference_access_gaps(
            cfg, planning.collection_windows(cfg, start, end), ["TREASURY_M3"]
        )
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["unrequested_eod_session_dates"], [start])

    def test_early_close_and_supporting_windows(self):
        cfg = config.CollectorConfig()
        day = pd.Timestamp("2024-11-29")
        opened, closed = planning.session_bounds(day, cfg)
        self.assertEqual(
            (opened.strftime("%H:%M"), closed.strftime("%H:%M")),
            ("09:30", "13:00"),
        )
        request = planning.near_close_request(cfg, "stock", "SPY", day)
        self.assertEqual(request.params["time_of_day"], "12:55:00.000")
        windows = planning.collection_windows(cfg, "2025-01-02", "2025-01-03")
        self.assertEqual(len(windows["lookback_dates"]), 60)
        self.assertEqual(windows["history_start"], "2024-10-07")
        self.assertEqual(windows["corporate_action_end"], "2025-07-02")

    def test_retained_sample_changes_request_identity_but_order_does_not(self):
        request = planning.history_request(
            config.CollectorConfig(),
            "option",
            "quote",
            "SPY",
            DAY,
            {"expiration": EXPIRATIONS[0], "strike": "*", "right": "both"},
        )
        keys = ("SPY|2025-01-10|100|call", "SPY|2025-01-10|105|put")
        first = dataclasses.replace(request, retained_contract_keys=keys)
        reordered = dataclasses.replace(
            request, retained_contract_keys=keys[::-1] + keys[:1]
        )
        narrower = dataclasses.replace(request, retained_contract_keys=keys[:1])
        self.assertEqual(first.request_id, reordered.request_id)
        self.assertNotEqual(first.request_id, narrower.request_id)
        self.assertNotEqual(first.request_id, request.request_id)
        self.assertNotIn("retained_contract_keys", first.params)
        self.assertNotIn("retained_contract_keys", request.identity())

    def test_bad_latest_sample_cannot_fall_back_or_use_a_future_quote(self):
        cfg = dataclasses.replace(
            config.CollectorConfig(), selection_times=("10:30:00",)
        )
        rows = [
            _quote("2025-01-02T10:29:30"),
            {**_quote("2025-01-02T10:30:00"), "bid_condition": "17"},
            _quote("2025-01-02T10:30:01"),
        ]
        frame = pd.DataFrame(rows, dtype="string")
        original = frame.copy()
        self.assertEqual(
            selection.stock_selection_references(frame, DAY, cfg), []
        )
        pd.testing.assert_frame_equal(frame, original)


class SavedCollectionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tfbsm-collector-test-")
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.cfg = dataclasses.replace(
            config.CollectorConfig(),
            output_dir=self.root,
            raw_chunk_rows=3,
        )
        self.collector = workflow.Collector(self.cfg)
        self.store = self.collector.store

    def option_request(self):
        """Return one bulk request retaining a single contract for storage tests."""
        return dataclasses.replace(
            planning.history_request(
                self.cfg,
                "option",
                "quote",
                "SPY",
                DAY,
                {"expiration": EXPIRATIONS[0], "strike": "*", "right": "both"},
            ),
            retained_contract_keys=("SPY|2025-01-10|100|call",),
        )

    def save_bytes(self, request, data):
        """Save exact CSV bytes through the streaming parser and response writer."""
        with io.BytesIO(data) as payload:
            frames = validation.csv_frames(payload, self.cfg.raw_chunk_rows)
            try:
                return self.store.save(
                    request, frames, {"status_code": 200}, payload
                )
            finally:
                frames.close()

    def test_one_session_preserves_selection_coverage_and_resume(self):
        with mock.patch.object(
            transport.ThetaClient,
            "download",
            autospec=True,
            side_effect=_download_fixture,
        ) as download:
            manifest = self.collector.collect_day(config.UNIVERSE[0], DAY)
        self.assertEqual(download.call_count, 17)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["coverage"]["status"], "observations_present")
        self.assertEqual(manifest["selected_contract_count"], 20)
        selected = pd.read_parquet(self.root / manifest["contracts"]["path"])
        oi_only = selected.loc[
            selected["strike"].eq("105") & selected["right"].eq("put")
        ]
        self.assertEqual(set(oi_only["discovery_sources"]), {"open_interest"})
        for record in manifest["requests"]:
            if record["dataset"] == "option_quotes_1h":
                frame = self.store.read(record)
                self.assertEqual(len(frame), 29)
                self.assertEqual(frame["bid"].unique().tolist(), ["1.230000"])
                self.assertEqual(int(frame.duplicated().sum()), 1)
                self.assertEqual(record["retention"]["excluded_rows"], 7)
                self.assertFalse(frame["strike"].eq("999").any())
                meta = storage.read_json(self.root / record["metadata"]["path"])
                self.assertEqual(
                    meta["collector_code_files"], self.store.code_files
                )
                self.assertEqual(meta["raw_schema_version"], 2)
        self.assertTrue(self.collector.resumable("SPY", DAY))
        restarted = workflow.Collector(
            dataclasses.replace(
                self.cfg,
                max_batch_workers=1,
                max_inflight_requests=4,
                start_date="2018-01-01",
                index_subscription="standard",
            )
        )
        self.assertTrue(restarted.resumable("SPY", DAY))
        counts = self.collector.write_availability(
            [config.UNIVERSE[0]], pd.DatetimeIndex([DAY])
        )
        self.assertEqual(counts["complete"], 1)
        availability = pd.read_csv(
            self.collector.directory / "availability.csv"
        )
        self.assertEqual(availability.loc[0, "excluded_quote_rows"], 40)
        self.assertFalse(list(self.root.rglob("raw_response.csv")))

    def test_excluded_misroute_fails_refresh_and_preserves_previous_cache(self):
        request = self.option_request()
        good = self.save_bytes(request, _payload(request))
        before = self.store.read(good)
        rows = [
            _quote(
                "2025-01-02T09:30:00",
                "1.230000",
                **_contracts((EXPIRATIONS[0],))[0],
            ),
            _quote(
                "2025-01-02T09:30:00",
                "1",
                symbol="AAPL",
                expiration=EXPIRATIONS[0],
                strike="999",
                right="call",
            ),
        ]
        payload = pd.DataFrame(rows).to_csv(index=False).encode()
        failed = self.save_bytes(request, payload)
        self.assertEqual(failed["status"], "invalid_response")
        self.assertIn("unexpected_symbol", failed["error"])
        self.assertEqual(
            (self.root / failed["payload"]["path"]).read_bytes(), payload
        )
        cached = self.store.cached(request)
        self.assertEqual(cached["data"]["path"], good["data"]["path"])
        pd.testing.assert_frame_equal(self.store.read(cached), before)

    def test_malformed_csv_after_batch_boundary_retains_failure_bytes(self):
        request = self.option_request()
        payload = _payload(request)
        payload += payload.splitlines()[1] + b",unexpected-extra-field\n"
        record = self.save_bytes(request, payload)
        self.assertEqual(record["status"], "invalid_response")
        self.assertIn("expected", record["error"])
        self.assertEqual(
            (self.root / record["payload"]["path"]).read_bytes(), payload
        )
        self.assertIsNone(self.store.cached(request))

    def test_bulk_success_cannot_hide_missing_selected_samples(self):
        request = self.option_request()
        selected = pd.DataFrame(
            {
                "expiration": [EXPIRATIONS[0]],
                "contract_key": list(request.retained_contract_keys),
            }
        )
        rows = [
            _quote(
                "2025-01-02T09:30:00",
                "1",
                symbol="SPY",
                expiration=EXPIRATIONS[0],
                strike="999",
                right="call",
            )
        ]
        record = self.save_bytes(
            request, pd.DataFrame(rows).to_csv(index=False).encode()
        )
        self.assertEqual(record["status"], "available")
        self.assertEqual(record["row_count"], 0)
        result = self.collector.coverage.quote_or_contract_report_coverage(
            request, record, selected
        )
        self.assertEqual(result["status"], "gaps_observed")
        self.assertEqual(
            len(result["missing_observations"][0]["missing_sample_times"]), 7
        )


class CliAndProvenanceTest(unittest.TestCase):
    def test_cli_accepts_pro_history_and_later_completed_dates(self):
        _, cfg, _, anchors = cli.parse_run_scope(["--symbols", "SPY"])
        self.assertEqual(cfg.start_date, "2012-06-01")
        self.assertEqual(cfg.end_date, "2025-12-31")
        self.assertEqual(str(anchors[0].date()), cfg.start_date)
        self.assertEqual(str(anchors[-1].date()), cfg.end_date)
        # The fixed default end must not become an artificial access ceiling.
        yesterday = str(
            (pd.Timestamp.now("America/New_York") - pd.Timedelta(days=1)).date()
        )
        cli.parse_run_scope(["--start", "2025-01-02", "--end", yesterday])
        for arguments in (
            ["--start", "2012-05-31"],
            ["--end", "2999-01-01"],
            ["--index-subscription", "invalid"],
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    cli.parse_run_scope(arguments)

    def test_stock_options_only_run_collects_panels_and_records_vix_gap(self):
        with tempfile.TemporaryDirectory(prefix="tfbsm-pro-run-") as directory:
            with (
                mock.patch.object(transport.ThetaClient, "ensure_available"),
                mock.patch.object(
                    transport.ThetaClient,
                    "download",
                    autospec=True,
                    side_effect=_download_fixture,
                ) as download,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = cli.main(
                    [
                        "--symbols",
                        "SPY",
                        "--start",
                        "2025-01-02",
                        "--end",
                        "2025-01-02",
                        "--lookback-sessions",
                        "0",
                        "--output-dir",
                        directory,
                    ]
                )
            self.assertEqual(result, 2)
            self.assertFalse(
                any(
                    call.args[1].endpoint.startswith("/index/")
                    for call in download.call_args_list
                )
            )
            root = pathlib.Path(directory)
            run = storage.read_json(next(root.glob("collection/*/runs/*.json")))
            self.assertEqual(run["status"], "coverage_gaps")
            self.assertEqual(run["processed_days"], 1)
            self.assertEqual(run["coverage"]["complete"], 1)
            self.assertEqual(run["reference_failures"], 0)
            ledger = storage.read_json(root / run["reference_ledger"])
            self.assertEqual(
                ledger["subscription_coverage_gaps"][0]["reason"],
                "index_subscription_unavailable",
            )

    def test_plan_modes_do_not_contact_theta_or_create_output(self):
        with tempfile.TemporaryDirectory(
            prefix="tfbsm-plan-test-"
        ) as directory:
            output = pathlib.Path(directory) / "must-not-exist"
            args = [
                "--symbols",
                "SPY",
                "--start",
                "2025-01-02",
                "--end",
                "2025-01-03",
                "--output-dir",
                str(output),
                "--plan",
            ]
            with mock.patch.object(
                transport.ThetaClient,
                "ensure_available",
                side_effect=AssertionError("Plan contacted Theta"),
            ):
                for mode in ([], ["--coverage-only"], ["--references-only"]):
                    with (
                        self.subTest(mode=mode),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.assertEqual(cli.main(args + mode), 0)
                    self.assertFalse(output.exists())

    def test_provenance_covers_every_module_and_default_root_stays_compatible(
        self,
    ):
        root = pathlib.Path(__file__).resolve().parents[1]
        expected = {"collector.py"} | {
            path.relative_to(root).as_posix()
            for path in (root / "tfbsm_collector").rglob("*.py")
        }
        sources = provenance.collector_code_files()
        self.assertEqual(set(sources), expected)
        self.assertEqual(
            sources["tfbsm_collector/selection.py"],
            provenance.file_hash(root / "tfbsm_collector/selection.py"),
        )
        self.assertEqual(
            config.DEFAULT_OUTPUT_DIR,
            root / "data" / "multi_year_bsm_backtest_output",
        )


class ProRequestLimitTest(unittest.TestCase):
    def test_eight_slots_are_shared_and_ninth_waits(self):
        cfg = config.CollectorConfig()
        self.assertEqual(cfg.max_batch_workers, 8)
        for limit in (0, 9, 4.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                dataclasses.replace(cfg, max_inflight_requests=limit)
        client = transport.ThetaClient(cfg)
        release = threading.Event()
        entered = [threading.Event() for _ in range(9)]

        def occupy(index):
            with client.request_slot():
                entered[index].set()
                if not release.wait(5):
                    raise TimeoutError("Shared request slots did not release")

        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
            futures = [pool.submit(occupy, index) for index in range(8)]
            try:
                for event in entered[:8]:
                    self.assertTrue(
                        event.wait(2), "Pro did not permit eight requests"
                    )
                futures.append(pool.submit(occupy, 8))
                self.assertFalse(entered[8].wait(0.05))
            finally:
                release.set()
            for future in futures:
                future.result(timeout=5)
        self.assertTrue(entered[8].is_set())


if __name__ == "__main__":
    unittest.main()
