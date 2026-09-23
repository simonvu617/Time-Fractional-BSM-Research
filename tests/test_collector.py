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
    """Build date-aware CSV fixtures with quiet contracts and raw duplicates."""
    if "/list/dates" in request.endpoint:
        return (
            pd.DataFrame(
                {
                    "date": planning.exchange_calendar()
                    .sessions_in_range("2025-01-02", "2025-05-02")
                    .strftime("%Y-%m-%d")
                }
            )
            .to_csv(index=False)
            .encode()
        )
    rows = []
    for day in planning.request_days(request):
        date = str(day.date())
        contracts = [
            c
            for c in _contracts()
            if pd.Timestamp(c["expiration"]) >= day
            and (
                "max_dte" not in request.params
                or (pd.Timestamp(c["expiration"]) - day).days
                <= request.params["max_dte"]
            )
        ]
        if request.dataset == "interest_rate_eod":
            rows.append({"created": date, "rate": "4.25"})
        elif request.dataset in {"quoted_contracts", "traded_contracts"}:
            rows.extend(
                c
                for c in contracts
                if not (c["strike"] == "105" and c["right"] == "put")
            )
        elif request.dataset == "option_open_interest":
            rows.extend(
                {**c, "timestamp": f"{date}T09:00:00", "open_interest": "12"}
                for c in contracts
            )
        elif request.endpoint.endswith("/eod"):
            identities = (
                contracts if request.endpoint.startswith("/option/") else [{}]
            )
            rows.extend(
                {
                    **c,
                    "created": f"{date}T17:15:00",
                    "last_trade": f"{date}T15:59:00",
                    "open": "1.2",
                    "high": "1.4",
                    "low": "1.1",
                    "close": "1.3",
                    "volume": "20",
                    "count": "3",
                }
                for c in identities
            )
        else:
            sampled = "/history/" in request.endpoint
            clocks = (
                pd.date_range(
                    f"{date} {request.params['start_time']}",
                    f"{date} {request.params['end_time']}",
                    freq="1h",
                ).strftime("%H:%M:%S")
                if sampled
                else [request.params["time_of_day"]]
            )
            identities = [{}]
            if request.endpoint.startswith("/option/"):
                identities = _contracts((request.params["expiration"],))
                identities.append({**identities[0], "strike": "999"})
            for contract in identities:
                for clock in clocks:
                    if request.endpoint.endswith("/ohlc"):
                        rows.append(
                            {
                                **contract,
                                "timestamp": f"{date}T{clock}",
                                "open": "1.2",
                                "high": "1.4",
                                "low": "1.1",
                                "close": "1.3",
                                "volume": "0",
                                "count": "0",
                                "vwap": "0",
                            }
                        )
                    elif request.endpoint.endswith("/price"):
                        rows.append(
                            {"timestamp": f"{date}T{clock}", "price": "100"}
                        )
                    else:
                        rows.append(
                            _quote(
                                f"{date}T{clock}",
                                "1.230000"
                                if contract
                                else "100"
                                if clock < "13:00:00"
                                else "105",
                                **contract,
                            )
                        )
            if (
                sampled
                and request.endpoint.endswith("/quote")
                and identities != [{}]
            ):
                rows.append(rows[-len(identities) * len(clocks)].copy())
    return (
        pd.DataFrame(rows, columns=None if rows else request.required_columns)
        .to_csv(index=False)
        .encode()
    )


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
    def test_contract_keys_accept_a_header_only_response(self):
        frame = pd.DataFrame(columns=planning.CONTRACT_FIELDS, dtype="string")
        original = frame.copy(deep=True)
        keys = planning.option_contract_keys(frame)
        self.assertTrue(keys.empty)
        pd.testing.assert_index_equal(keys.index, frame.index)
        self.assertTrue(pd.api.types.is_string_dtype(keys.dtype))
        pd.testing.assert_frame_equal(frame, original)

    def test_contract_keys_preserve_precision_rows_and_raw_fields(self):
        frame = pd.DataFrame(
            {
                "symbol": ["SPY"] * 3,
                "expiration": ["20250110", "2025-01-10", "20250110"],
                "strike": ["100.000", "100.125", "100.000"],
                "right": ["C", "p", "C"],
            },
            index=[3, 1, 3],
        )
        original = frame.copy(deep=True)
        keys = planning.option_contract_keys(frame)
        self.assertEqual(
            keys.tolist(),
            [
                "SPY|2025-01-10|100|call",
                "SPY|2025-01-10|100.125|put",
                "SPY|2025-01-10|100|call",
            ],
        )
        self.assertEqual(keys.index.tolist(), [3, 1, 3])
        pd.testing.assert_frame_equal(frame, original)
        for strike in ("100.0001", "NaN", "0"):
            with self.subTest(strike=strike), self.assertRaises(ValueError):
                planning.option_contract_keys(frame.assign(strike=strike))

    def test_weekly_enrollment_uses_exchange_weeks_not_month_or_data_boundaries(
        self,
    ):
        cfg = config.CollectorConfig(
            start_date="2025-01-02",
            end_date="2025-02-04",
            enrollment_frequency="weekly",
        )
        days = planning.exchange_calendar().sessions_in_range(
            cfg.start_date, cfg.end_date
        )
        selected = [
            str(d.date()) for d in days if planning.is_enrollment_day(d, cfg)
        ]
        self.assertEqual(
            selected,
            [
                "2025-01-02",
                "2025-01-06",
                "2025-01-13",
                "2025-01-21",
                "2025-01-27",
                "2025-02-03",
            ],
        )
        # August begins on Friday; a monthly restart must wait until Monday.
        full = dataclasses.replace(cfg, end_date="2025-08-04")
        self.assertFalse(
            planning.is_enrollment_day(pd.Timestamp("2025-08-01"), full)
        )
        self.assertTrue(
            planning.is_enrollment_day(pd.Timestamp("2025-08-04"), full)
        )
        self.assertFalse(
            planning.is_enrollment_day(pd.Timestamp("2025-02-05"), cfg)
        )
        daily = dataclasses.replace(cfg, enrollment_frequency="daily")
        self.assertTrue(all(planning.is_enrollment_day(d, daily) for d in days))
        self.assertEqual(
            [
                str(d.date())
                for d in days
                if planning.is_weekly_enrollment_day(d, daily)
            ],
            selected,
        )
        self.assertFalse(
            planning.is_enrollment_day(pd.Timestamp("2025-02-05"), daily)
        )
        self.assertNotEqual(cfg.policy_id, daily.policy_id)

    def test_followup_keeps_contracts_below_entry_cutoff_and_after_entry_end(
        self,
    ):
        cfg = config.CollectorConfig(
            start_date="2025-01-02", end_date="2025-01-02"
        )
        chain = selection.normalize_chain(
            pd.DataFrame(_contracts()), "SPY", cfg
        )
        chosen = selection.select_contracts(
            chain,
            DAY,
            [{"stock_mid": 100.0, "selection_time": "13:30:00"}],
            cfg,
        )
        cohort = selection.extend_cohort(
            pd.DataFrame(columns=selection.COHORT_COLUMNS),
            chosen,
            DAY,
            config.UNIVERSE[0],
        )
        later = pd.Timestamp("2025-01-08")
        retained = selection.extend_cohort(
            cohort, chosen, later, config.UNIVERSE[0]
        )
        self.assertEqual(len(retained), len(cohort))
        self.assertEqual(set(retained["first_selected_date"]), {"2025-01-02"})
        requests = planning.followup_requests(
            cfg, "SPY", pd.DatetimeIndex([later]), retained
        )
        nearest = [
            r for r in requests if r.params["expiration"] == "2025-01-10"
        ]
        self.assertEqual(len(nearest), 3)
        self.assertTrue(all(r.retained_contract_windows for r in nearest))
        self.assertEqual(set(cohort["contract_multiplier"]), {""})
        self.assertNotEqual(
            cfg.policy_id,
            dataclasses.replace(cfg, start_date="2024-12-30").policy_id,
        )

    def test_month_batches_split_early_closes_and_unlisted_dates(self):
        cfg = config.CollectorConfig()
        days = (
            planning.exchange_calendar()
            .sessions_in_range("2024-11-25", "2024-12-02")
            .tz_localize(None)
        )
        requests = planning.underlying_requests(
            cfg, config.UNIVERSE[0], days, {("SPY", "quote"): {"2024-11-26"}}
        )
        near = [r for r in requests if r.dataset == "stock_quotes_near_close"]
        half_day = next(
            r
            for r in near
            if pd.Timestamp("2024-11-29") in planning.request_days(r)
        )
        self.assertEqual(half_day.params["time_of_day"], "12:55:00.000")
        self.assertTrue(
            all(
                pd.Timestamp("2024-11-26") not in planning.request_days(r)
                for r in near
            )
        )
        self.assertTrue(
            all(
                len(set(planning.request_days(r).strftime("%Y-%m"))) == 1
                for r in requests
            )
        )

    def test_spxw_uses_spx_index_prices_and_keeps_missing_access_explicit(self):
        symbol = config.INDEX_BENCHMARK[1]
        cfg = config.CollectorConfig(symbols=(symbol,), lookback_sessions=0)
        self.assertEqual(planning.underlying_requests(cfg, symbol, [DAY]), [])
        paid = dataclasses.replace(cfg, index_subscription="pro")
        requests = planning.underlying_requests(paid, symbol, [DAY])
        self.assertEqual({r.params["symbol"] for r in requests}, {"SPX"})
        self.assertTrue(all(r.endpoint.startswith("/index/") for r in requests))
        frame = pd.DataFrame(
            {"timestamp": ["2025-01-02T10:30:00"], "price": ["6000"]}
        )
        references = selection.index_selection_references([frame], DAY, paid)
        self.assertEqual(references[0]["stock_mid"], 6000.0)
        self.assertEqual(
            references[0]["underlying_price_source"], "index_price"
        )

    def test_pro_history_clips_stock_buffer_and_reports_reference_access(self):
        cfg = config.CollectorConfig(
            start_date=config.PRO_HISTORY_START,
            symbols=(config.UNIVERSE[0],),
            rate_symbols=("SOFR", "TREASURY_M3"),
        )
        windows = planning.collection_windows(cfg)
        self.assertEqual(windows["history_start"], "2012-06-01")
        self.assertEqual(windows["lookback_dates"], [])
        self.assertEqual(len(windows["unavailable_lookback_dates"]), 60)
        requests = list(planning.reference_requests(cfg))
        self.assertEqual(len(requests), 2)
        self.assertFalse(
            any(r.endpoint.startswith("/index/") for r in requests)
        )
        for request in requests:
            if request.dataset == "interest_rate_eod":
                self.assertEqual(request.params["start_date"], "2024-01-01")
            else:
                self.assertEqual(request.params["start_date"], "2012-06-01")
        gaps = planning.reference_access_gaps(cfg, windows)
        self.assertEqual(
            {g["reason"] for g in gaps},
            {
                "index_subscription_unavailable",
                "before_rate_subscription_history_start",
                "before_stock_pro_history_start",
                "theta_v3_endpoint_unavailable",
                "historical_contract_terms_not_supplied_by_current_Theta_API",
            },
        )
        rate_gap = next(
            g for g in gaps if g.get("dataset") == "interest_rate_eod"
        )
        self.assertEqual(
            rate_gap["unrequested_eod_session_dates"][-1], "2023-12-29"
        )
        later = planning.collection_windows(
            dataclasses.replace(
                cfg, start_date="2013-01-02", end_date="2013-01-03"
            )
        )
        self.assertEqual(len(later["lookback_dates"]), 60)
        self.assertEqual(later["unavailable_lookback_dates"], [])

    def test_separate_paid_reference_tiers_use_their_own_history_bounds(self):
        cfg = dataclasses.replace(
            config.CollectorConfig(),
            index_subscription="pro",
            rate_subscription="value",
            lookback_sessions=0,
            start_date="2016-12-30",
            end_date="2017-01-03",
            symbols=(),
            rate_symbols=("TREASURY_M3",),
            mode="references",
        )
        requests = list(planning.reference_requests(cfg))
        self.assertEqual(len(requests), 4)
        rates = next(r for r in requests if r.dataset == "interest_rate_eod")
        index = next(r for r in requests if r.dataset == "index_eod")
        self.assertEqual(rates.params["start_date"], cfg.start_date)
        self.assertEqual(index.params["start_date"], "2017-01-01")
        gaps = planning.reference_access_gaps(
            cfg, planning.collection_windows(cfg)
        )
        self.assertEqual(len(gaps), 1)
        self.assertEqual(
            gaps[0]["unrequested_eod_session_dates"], [cfg.start_date]
        )

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
        windows = planning.collection_windows(
            dataclasses.replace(
                cfg, start_date="2025-01-02", end_date="2025-01-03"
            )
        )
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
    def test_unlisted_underlying_prefix_skips_new_chains_but_not_existing_cohort(
        self,
    ):
        self.collector.unavailable_dates = {("SPY", "quote"): {str(DAY.date())}}
        scope = {"dates": [str(DAY.date())], "symbol": {"symbol": "SPY"}}
        empty = pd.DataFrame(columns=selection.COHORT_COLUMNS)
        with mock.patch.object(
            transport.ThetaClient,
            "download",
            autospec=True,
            side_effect=_download_fixture,
        ) as download:
            self.collector.collect_month(
                config.UNIVERSE[0], pd.DatetimeIndex([DAY]), empty, scope
            )
            self.assertFalse(
                any(
                    c.args[1].endpoint.startswith("/option/")
                    for c in download.call_args_list
                )
            )
        chosen = selection.select_contracts(
            selection.normalize_chain(
                pd.DataFrame(_contracts()), "SPY", self.cfg
            ),
            DAY,
            [{"stock_mid": 100.0, "selection_time": "10:30:00"}],
            self.cfg,
        )
        cohort = selection.extend_cohort(empty, chosen, DAY, config.UNIVERSE[0])
        with mock.patch.object(
            transport.ThetaClient,
            "download",
            autospec=True,
            side_effect=_download_fixture,
        ) as download:
            self.collector.collect_month(
                config.UNIVERSE[0], pd.DatetimeIndex([DAY]), cohort, scope
            )
            self.assertTrue(
                any(
                    c.args[1].dataset == "option_quotes_1h"
                    for c in download.call_args_list
                )
            )
            oi = [
                c.args[1]
                for c in download.call_args_list
                if c.args[1].dataset == "option_open_interest"
            ]
            self.assertEqual(len(oi), 1)
            self.assertTrue(oi[0].retained_contract_windows)
        self.assertEqual(
            self.store.session("SPY", DAY)["universe_status"], "not_requested"
        )
        self.assertEqual(
            self.store.session("SPY", DAY)["cross_section_status"], "incomplete"
        )
        self.assertIsNone(
            self.store.session("SPY", DAY)["cross_section_contract_count"]
        )
        self.assertGreater(
            self.store.session("SPY", DAY)["selected_contract_count"], 0
        )

    def test_monthly_retention_obeys_each_entry_date_and_survives_compaction(
        self,
    ):
        request = planning.ranged(
            self.option_request(),
            pd.DatetimeIndex([DAY, DAY + pd.Timedelta(days=1)]),
        )
        key = "SPY|2025-01-10|100|call"
        request = dataclasses.replace(
            request,
            retained_contract_keys=None,
            retained_contract_windows=((key, "2025-01-03", "2025-01-10"),),
        )
        record = self.save_bytes(request, _payload(request))
        before = self.store.read(record)
        self.assertEqual(set(before["timestamp"].str[:10]), {"2025-01-03"})
        self.assertEqual(len(before), 8)
        self.store.compact([record], "test")
        pd.testing.assert_frame_equal(self.store.read(record), before)
        self.assertTrue(self.store.reusable(record))
        self.assertTrue(self.store.cached(request))
        # Failed refresh remains separate even after the good receipt was packed.
        bad = self.save_bytes(request, b"wrong,header\n1,2\n")
        self.assertEqual(bad["status"], "invalid_response")
        pd.testing.assert_frame_equal(
            self.store.read(self.store.cached(request)), before
        )

    def test_missing_activity_bar_is_not_zero_and_overnight_is_not_a_quote_gap(
        self,
    ):
        request = planning.history_request(
            self.cfg, "stock", "ohlc", "SPY", DAY
        )
        frame = pd.read_csv(io.BytesIO(_payload(request)), dtype="string")
        record = self.save_bytes(request, frame.to_csv(index=False).encode())
        check = self.collector.coverage.quote_or_contract_report_coverage(
            request, record
        )
        self.assertEqual(check["status"], "observations_present")
        shortened = self.save_bytes(
            request, frame.iloc[1:].to_csv(index=False).encode()
        )
        check = self.collector.coverage.quote_or_contract_report_coverage(
            request, shortened
        )
        self.assertEqual(check["status"], "gaps_observed")
        self.assertFalse(check["missing_bars_mean_zero"])
        quote = planning.ranged(
            planning.history_request(self.cfg, "stock", "quote", "SPY", DAY),
            pd.DatetimeIndex([DAY, DAY + pd.Timedelta(days=1)]),
        )
        record = self.save_bytes(quote, _payload(quote))
        self.assertEqual(
            self.store.metadata(record)["quality"][
                "absent_interior_sample_slots"
            ],
            0,
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tfbsm-collector-test-")
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.cfg = dataclasses.replace(
            config.CollectorConfig(),
            start_date=str(DAY.date()),
            output_dir=self.root,
            raw_chunk_rows=3,
            symbols=(config.UNIVERSE[0],),
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

    def test_empty_selected_option_responses_are_saved_and_reusable(self):
        request = dataclasses.replace(
            planning.near_close_request(
                self.cfg, "option", "SPY", DAY, EXPIRATIONS[0]
            ),
            retained_contract_windows=(
                ("SPY|2025-01-10|100|call", str(DAY.date()), str(DAY.date())),
            ),
        )
        header = (",".join(request.required_columns) + "\n").encode()
        for status_code in (200, 472):
            with self.subTest(status_code=status_code):

                def download(request, payload):
                    if status_code == 200:
                        payload.write(header)
                    return {"status_code": status_code}

                with mock.patch.object(
                    self.store.client, "download", side_effect=download
                ):
                    record = self.store.collect(request, refresh=True)
                self.assertEqual(record["status"], "no_data")
                self.assertEqual(record["row_count"], 0)
                self.assertTrue(self.store.read(record).empty)
                self.assertIsNotNone(self.store.cached(request))
                self.assertFalse(self.store.client.stop_event.is_set())

    def test_month_preserves_selection_coverage_compaction_and_resume(self):
        scope = {"dates": [str(DAY.date())], "symbol": {"symbol": "SPY"}}
        with mock.patch.object(
            transport.ThetaClient,
            "download",
            autospec=True,
            side_effect=_download_fixture,
        ) as download:
            manifest = self.collector.collect_month(
                config.UNIVERSE[0],
                pd.DatetimeIndex([DAY]),
                pd.DataFrame(columns=selection.COHORT_COLUMNS),
                scope,
            )
        self.assertEqual(download.call_count, 23)
        self.assertEqual(manifest["request_error_count"], 0)
        session = self.store.session("SPY", DAY)
        self.assertEqual(session["coverage"]["status"], "observations_present")
        self.assertEqual(session["selected_contract_count"], 20)
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
                meta = self.store.metadata(record)
                self.assertEqual(meta["raw_schema_version"], 2)
                self.assertIn("row_groups", meta["data"])
        self.assertTrue(self.collector.month_valid(manifest, scope))
        restarted = workflow.Collector(
            dataclasses.replace(self.cfg, max_inflight_requests=4)
        )
        self.assertTrue(restarted.month_valid(manifest, scope))
        self.assertFalse(
            restarted.month_valid(
                manifest, {**scope, "incoming_cohort": "changed"}
            )
        )
        counts = self.collector.coverage.write_availability(
            pd.DatetimeIndex([DAY])
        )
        self.assertEqual(counts["complete"], 1)
        self.assertFalse(list(self.root.rglob("raw_response.csv")))
        # Five expiration responses now share one quote file, rather than five
        # separate files plus metadata and pointer files for every request.
        self.assertEqual(
            len(
                list(
                    (self.root / "parquet" / "option_quotes_1h").rglob(
                        "*.parquet"
                    )
                )
            ),
            1,
        )

    def test_daily_cross_sections_refresh_without_reenrolling_contracts(self):
        cfg = dataclasses.replace(
            self.cfg, end_date="2025-01-07", moneyness_targets=(1.0,)
        )
        collector = workflow.Collector(cfg)
        # The last session is follow-up only, after the entry window closes.
        days = planning.exchange_calendar().sessions_in_range(
            cfg.start_date, "2025-01-08"
        )
        scope = {
            "dates": days.strftime("%Y-%m-%d").tolist(),
            "symbol": {"symbol": "SPY"},
        }

        def shifted_prices(client, request, payload):
            data = _payload(request)
            if request.dataset == "stock_quotes_1h":
                frame = pd.read_csv(io.BytesIO(data), dtype="string")
                frame.loc[
                    frame["timestamp"]
                    .str[:10]
                    .isin(["2025-01-03", "2025-01-06"]),
                    ["bid", "ask"],
                ] = "105"
                data = frame.to_csv(index=False).encode()
            payload.write(data)
            payload.seek(0)
            return {"status_code": 200, "payload_bytes": len(data)}

        with mock.patch.object(
            transport.ThetaClient,
            "download",
            autospec=True,
            side_effect=shifted_prices,
        ) as download:
            manifest = collector.collect_month(
                config.UNIVERSE[0],
                days,
                pd.DataFrame(columns=selection.COHORT_COLUMNS),
                scope,
            )
        sessions = [collector.store.session("SPY", d) for d in days]
        self.assertEqual(manifest["request_error_count"], 0)
        self.assertEqual(
            [s["newly_selected_contract_count"] for s in sessions],
            [10, 10, 0, 0, 0],
        )
        self.assertEqual(
            [s["selected_contract_count"] for s in sessions],
            [10, 20, 20, 20, 20],
        )
        self.assertEqual(
            [s["cross_section_contract_count"] for s in sessions],
            [10, 10, 8, 8, None],
        )
        self.assertEqual(sessions[-1]["cross_section_status"], "not_scheduled")
        cross = pd.read_parquet(
            cfg.output_dir / manifest["cross_sections"]["path"]
        )
        tracked = pd.read_parquet(
            cfg.output_dir / manifest["contracts"]["path"]
        )
        self.assertFalse(cross.duplicated(["trade_day", "contract_key"]).any())
        self.assertEqual(set(cross["selection_status"]), {"observed"})
        for date, strike in (
            ("2025-01-02", "100"),
            ("2025-01-03", "105"),
            ("2025-01-06", "105"),
            ("2025-01-07", "100"),
        ):
            self.assertEqual(
                set(cross.loc[cross["trade_day"].eq(date), "strike"]), {strike}
            )
        # A Friday entrant can appear again in Monday's fresh grid. Its daily
        # entry stays Friday, but the weekly comparison starts only on Monday.
        self.assertEqual(
            set(
                tracked.loc[tracked["strike"].eq("105"), "first_selected_date"]
            ),
            {"2025-01-03"},
        )
        weekly = cross.loc[cross["weekly_entry_day"]]
        self.assertEqual(
            set(weekly.loc[weekly["strike"].eq("105"), "trade_day"]),
            {"2025-01-06"},
        )
        self.assertEqual(weekly["contract_key"].nunique(), 18)
        request_ids = [
            call.args[1].request_id for call in download.call_args_list
        ]
        self.assertEqual(len(request_ids), len(set(request_ids)))
        self.assertEqual(
            sum(
                r["dataset"] == "option_quotes_1h" for r in manifest["requests"]
            ),
            5,
        )
        self.assertTrue(collector.month_valid(manifest, scope))
        # Missing cross-section evidence must force recovery, even when the
        # market observations and cohort checkpoint are still intact.
        (cfg.output_dir / manifest["cross_sections"]["path"]).unlink()
        self.assertFalse(collector.month_valid(manifest, scope))

    def test_weekly_discovery_preserves_daily_cohort_reports(self):
        cfg = dataclasses.replace(
            self.cfg,
            end_date="2025-01-06",
            moneyness_targets=(1.0,),
            enrollment_frequency="weekly",
        )
        collector = workflow.Collector(cfg)
        days = planning.exchange_calendar().sessions_in_range(
            cfg.start_date, cfg.end_date
        )

        def shifted_prices(client, request, payload):
            data = _payload(request)
            if (
                request.dataset == "option_open_interest"
                and request.params["date"] == "20250103"
            ):
                frame = pd.read_csv(io.BytesIO(data), dtype="string")
                # A missing tracked OI report remains a gap, not a zero or a
                # reason to drop the option from its remaining observations.
                frame = frame.loc[
                    ~(
                        frame["strike"].eq("100")
                        & frame["right"].eq("put")
                        & frame["expiration"].eq(EXPIRATIONS[0])
                    )
                ]
                data = frame.to_csv(index=False).encode()
            if request.dataset == "option_eod":
                frame = pd.read_csv(io.BytesIO(data), dtype="string")
                # A quiet option's last trade can predate selection. Retention
                # must use the report's created date instead.
                frame["last_trade"] = "2024-12-31T15:59:00"
                data = frame.to_csv(index=False).encode()
            if request.dataset == "stock_quotes_1h":
                frame = pd.read_csv(io.BytesIO(data), dtype="string")
                # Friday's large move must not enroll new contracts. Monday's
                # new strikes enter then, without acquiring earlier EOD rows.
                for date, price in (
                    ("2025-01-03", "200"),
                    ("2025-01-06", "105"),
                ):
                    frame.loc[
                        frame["timestamp"].str.startswith(date), ["bid", "ask"]
                    ] = price
                data = frame.to_csv(index=False).encode()
            payload.write(data)
            payload.seek(0)
            return {"status_code": 200, "payload_bytes": len(data)}

        with mock.patch.object(
            transport.ThetaClient,
            "download",
            autospec=True,
            side_effect=shifted_prices,
        ):
            manifest = collector.collect_month(
                config.UNIVERSE[0],
                days,
                pd.DataFrame(columns=selection.COHORT_COLUMNS),
                {},
            )
        self.assertEqual(manifest["request_error_count"], 0)
        sessions = [collector.store.session("SPY", d) for d in days]
        self.assertEqual(
            [s["enrollment_scheduled"] for s in sessions], [True, False, True]
        )
        # Jan 10 is now below the entry cutoff: its old contracts stay, but
        # new strikes can enter only the four later expirations.
        self.assertEqual(
            [s["newly_selected_contract_count"] for s in sessions], [10, 0, 8]
        )
        self.assertEqual(
            [s["selected_contract_count"] for s in sessions], [10, 10, 18]
        )
        self.assertEqual(sessions[1]["missing_selection_times"], [])
        self.assertEqual(
            [s["universe_status"] for s in sessions],
            ["observed", "not_requested", "observed"],
        )
        self.assertIsNone(sessions[1]["universe_contract_count"])
        self.assertIsNone(sessions[1]["oi_reported_contract_count"])
        self.assertEqual(sessions[1]["coverage"]["missing_option_oi_count"], 1)
        oi_check = next(
            c
            for c in sessions[1]["coverage"]["required_price_requests"]
            if c["dataset"] == "option_open_interest"
        )
        self.assertEqual(
            oi_check["missing_observations"][0]["reason"],
            "missing_oi_report; zero_oi_not_inferred",
        )
        for record in manifest["requests"]:
            if record["dataset"] in {"quoted_contracts", "traded_contracts"}:
                self.assertNotEqual(record["params"]["date"], "20250103")
        oi = [
            r
            for r in manifest["requests"]
            if r["dataset"] == "option_open_interest"
        ]
        self.assertEqual(len(oi), 3)
        friday = next(r for r in oi if r["params"]["date"] == "20250103")
        self.assertEqual(friday["row_count"], 9)
        self.assertEqual(
            self.store.metadata(friday)["retention"]["excluded_rows"], 10
        )
        for record in oi:
            if record is not friday:
                self.assertEqual(
                    self.store.metadata(record)["retention"]["mode"],
                    "full_response",
                )
        eod = next(
            r for r in manifest["requests"] if r["dataset"] == "option_eod"
        )
        saved = collector.store.read(eod)
        self.assertEqual(len(saved), 38)
        self.assertEqual(eod["retention"]["excluded_rows"], 22)
        self.assertEqual(
            set(saved.loc[saved["strike"].eq("105"), "created"].str[:10]),
            {"2025-01-06"},
        )
        self.assertTrue(collector.store.reusable(eod))

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

    def test_report_dates_are_checked_across_batches(self):
        request = planning.Request(
            "interest_rate_eod",
            "/interest_rate/history/eod",
            {
                "symbol": "SOFR",
                "start_date": "2025-01-02",
                "end_date": "2025-01-03",
                "format": "csv",
            },
        )
        # An empty first batch must not conceal a later out-of-range report.
        rows = pd.DataFrame(
            {"created": ["", "", "", "2025-01-10"], "rate": ["4.25"] * 4}
        )
        record = self.save_bytes(request, rows.to_csv(index=False).encode())
        self.assertEqual(record["status"], "invalid_response")
        quality = self.store.metadata(record)["quality"]
        self.assertEqual(quality["report_dates"]["unparseable_or_missing"], 3)
        self.assertEqual(quality["report_dates"]["first"], "2025-01-10")
        self.assertEqual(
            quality["response_identity_issues"],
            ["invalid_report_dates", "timestamps_outside_requested_dates"],
        )


class CliAndProvenanceTest(unittest.TestCase):
    def test_failed_run_records_outcome_and_releases_output_lock(self):
        for error, expected in (
            (transport.CollectionStopped("Test stop"), "partial_failure"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory(
                    prefix="tfbsm-stop-test-"
                ) as directory,
            ):
                with (
                    mock.patch.object(
                        transport.ThetaClient, "ensure_available"
                    ),
                    mock.patch.object(
                        transport.ThetaClient,
                        "download",
                        autospec=True,
                        side_effect=_download_fixture,
                    ),
                    mock.patch.object(
                        workflow.Collector, "_collect_panels", side_effect=error
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
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
                root = pathlib.Path(directory)
                run = storage.read_json(
                    next(root.glob("collection/*/runs/*.json"))
                )
                self.assertEqual(result, 1)
                self.assertEqual(run["status"], expected)
                self.assertIn("finished_at_utc", run)
                self.assertEqual(run["coverage"]["not_attempted"], 1)
                with storage.output_lock(root):
                    pass

    def test_cli_accepts_pro_history_and_later_completed_dates(self):
        cfg, preview = cli.parse_run_scope(["--symbols", "SPY"])
        self.assertEqual(cfg.start_date, "2017-01-01")
        self.assertEqual(cfg.end_date, "2025-12-31")
        self.assertEqual(cfg.enrollment_frequency, "daily")
        weekly, _ = cli.parse_run_scope(["--enrollment-frequency", "weekly"])
        self.assertEqual(weekly.enrollment_frequency, "weekly")
        self.assertEqual(cfg.symbols, (config.UNIVERSE[0],))
        self.assertFalse(preview)
        earlier, _ = cli.parse_run_scope(["--start", config.PRO_HISTORY_START])
        self.assertEqual(earlier.start_date, config.PRO_HISTORY_START)
        # The fixed default end must not become an artificial access ceiling.
        yesterday = str(
            (pd.Timestamp.now("America/New_York") - pd.Timedelta(days=1)).date()
        )
        cli.parse_run_scope(["--start", "2025-01-02", "--end", yesterday])
        for arguments in (
            ["--start", "2012-05-31"],
            ["--end", "2999-01-01"],
            ["--index-subscription", "invalid"],
            ["--enrollment-frequency", "monthly"],
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
                    call.args[1].endpoint.startswith(
                        ("/index/", "/corporate_action/")
                    )
                    for call in download.call_args_list
                )
            )
            root = pathlib.Path(directory)
            run = storage.read_json(next(root.glob("collection/*/runs/*.json")))
            self.assertEqual(run["status"], "coverage_gaps")
            self.assertGreater(run["processed_days"], 1)
            self.assertEqual(run["coverage"]["complete"], run["processed_days"])
            self.assertEqual(run["reference_failures"], 0)
            ledger = storage.read_json(root / run["reference_ledger"])
            self.assertEqual(
                ledger["access_coverage_gaps"][0]["reason"],
                "index_subscription_unavailable",
            )
            action_gaps = {
                gap["dataset"]: gap
                for gap in ledger["access_coverage_gaps"]
                if gap["reason"] == "theta_v3_endpoint_unavailable"
            }
            self.assertEqual(
                set(action_gaps), {"corporate_dividend", "corporate_split"}
            )
            self.assertEqual(
                action_gaps["corporate_dividend"]["symbols"], ["SPY"]
            )
            self.assertEqual(
                action_gaps["corporate_dividend"]["unrequested_end_date"],
                "2025-07-01",
            )
            self.assertEqual(run["reference_gaps"], 4)

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
