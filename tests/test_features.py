from __future__ import annotations

import unittest

import pandas as pd

from tfbsm_empirical.data.features import (
    add_quote_features,
    snapshot_at,
    waiting_time_features,
)


class FeatureTests(unittest.TestCase):
    def test_snapshot_never_uses_a_future_quote(self) -> None:
        frame = add_quote_features(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2025-01-03 10:29:00", "2025-01-03 10:31:00"]
                    ).tz_localize("America/New_York"),
                    "bid": [99.0, 101.0],
                    "ask": [101.0, 103.0],
                }
            )
        )

        snapshot = snapshot_at(
            frame,
            pd.Timestamp("2025-01-03 10:30:00", tz="America/New_York"),
            max_age_seconds=70,
        )

        self.assertEqual(snapshot["mid"], 100.0)
        self.assertEqual(snapshot["age_seconds"], 60.0)
        self.assertFalse(snapshot["is_stale"])

    def test_waiting_time_features_are_measured_from_observations(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-03 10:00:00",
                        "2025-01-03 10:00:30",
                        "2025-01-03 10:02:00",
                    ]
                ).tz_localize("America/New_York"),
                "mid": [100.0, 100.0, 101.0],
            }
        )

        measured = waiting_time_features(frame)

        self.assertEqual(measured["observation_count"], 3)
        self.assertEqual(measured["update_count"], 1)
        self.assertEqual(measured["zero_change_fraction"], 0.5)
        self.assertEqual(measured["median_update_interval_seconds"], 120.0)
        self.assertEqual(measured["longest_no_update_interval_seconds"], 120.0)


if __name__ == "__main__":
    unittest.main()
