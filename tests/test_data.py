from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from tfbsm_empirical.data.client import CsvResponse
from tfbsm_empirical.data.config import CollectorConfig
from tfbsm_empirical.data.pipeline import CollectorPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataPipelineTests(unittest.TestCase):
    def test_dry_plan_is_bounded_and_does_not_claim_network_work(self) -> None:
        config = CollectorConfig.from_yaml(PROJECT_ROOT / "configs" / "collector.yaml")

        with CollectorPipeline(config) as pipeline:
            plan = pipeline.plan("SPY", date(2025, 1, 3))

        self.assertEqual(plan["symbol"], "SPY")
        self.assertFalse(plan["network_requests_made"])
        self.assertEqual(len(plan["evaluation_points"]), 3)

    def test_one_contract_vertical_slice_writes_a_complete_manifest(self) -> None:
        base = CollectorConfig.from_yaml(PROJECT_ROOT / "configs" / "collector.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            config = replace(
                base,
                project_root=output_root,
                output_dir=output_root / "data",
                target_dtes=(7,),
                max_expirations_per_day=1,
                moneyness_targets=(1.0,),
                strikes_per_moneyness_target=1,
                option_rights=("call",),
                universe=(base.symbol("SPY"),),
            )
            pipeline = CollectorPipeline(config, client=FakeClient())

            result = pipeline.collect_symbol_day("SPY", date(2025, 1, 3))

            self.assertEqual(result.contract_count, 1)
            self.assertEqual(result.artifact_count, 8)
            self.assertTrue(result.manifest_path.exists())


class FakeClient:
    def fetch_csv(self, endpoint: str, params: dict[str, object]) -> CsvResponse:
        if endpoint == "/stock/history/quote":
            frame = pd.DataFrame(
                {
                    "timestamp": [
                        "2025-01-03 10:29:30",
                        "2025-01-03 12:59:30",
                        "2025-01-03 14:59:30",
                    ],
                    "bid": [499.0, 500.0, 501.0],
                    "ask": [501.0, 502.0, 503.0],
                }
            )
        elif endpoint == "/stock/history/trade":
            frame = pd.DataFrame(
                {
                    "timestamp": ["2025-01-03 10:29:30"],
                    "price": [500.0],
                    "size": [10],
                }
            )
        elif endpoint == "/option/list/expirations":
            frame = pd.DataFrame({"expiration": ["2025-01-10"]})
        elif endpoint == "/option/list/strikes":
            frame = pd.DataFrame({"strike": [500.0]})
        elif endpoint == "/option/history/quote":
            frame = pd.DataFrame(
                {
                    "timestamp": ["2025-01-03 10:29:30"],
                    "bid": [5.0],
                    "ask": [5.2],
                }
            )
        elif endpoint == "/option/history/trade":
            frame = pd.DataFrame(
                {
                    "timestamp": ["2025-01-03 10:29:30"],
                    "price": [5.1],
                    "size": [1],
                }
            )
        elif endpoint == "/option/history/open_interest":
            frame = pd.DataFrame(
                {
                    "timestamp": ["2025-01-02 16:00:00"],
                    "open_interest": [100],
                }
            )
        else:  # pragma: no cover - makes unexpected orchestration explicit.
            raise AssertionError(f"unexpected endpoint: {endpoint}")

        payload = frame.to_csv(index=False).encode("utf-8")
        return CsvResponse(
            frame=frame,
            payload=payload,
            request_url=f"http://127.0.0.1{endpoint}",
            status_code=200,
            elapsed_ms=1.0,
            response_headers={"Content-Type": "text/csv"},
        )


if __name__ == "__main__":
    unittest.main()
