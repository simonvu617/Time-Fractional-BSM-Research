from __future__ import annotations

import unittest
from datetime import date

from tfbsm_empirical.data.universe import (
    occ_contract_id,
    select_expirations,
    select_strikes,
)


class UniverseTests(unittest.TestCase):
    def test_expiration_selection_tracks_targets_without_duplicates(self) -> None:
        selected = select_expirations(
            date(2025, 1, 2),
            (date(2025, 1, 9), date(2025, 1, 16), date(2025, 1, 31)),
            min_dte=7,
            max_dte=60,
            target_dtes=(7, 14),
            maximum=2,
        )

        self.assertEqual(selected, (date(2025, 1, 9), date(2025, 1, 16)))

    def test_strike_selection_is_nearest_to_declared_moneyness(self) -> None:
        selected = select_strikes(
            (90.0, 100.0, 110.0, 120.0),
            (100.0,),
            moneyness_targets=(1.0, 0.9),
            per_target=1,
        )

        self.assertEqual(selected, (100.0, 110.0))

    def test_occ_identifier_is_stable(self) -> None:
        self.assertEqual(
            occ_contract_id("SPY", date(2025, 1, 17), "call", 500.0),
            "SPY   250117C00500000",
        )


if __name__ == "__main__":
    unittest.main()
