from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tfbsm_empirical.data.cache import CacheCorruptionError, ResponseCache
from tfbsm_empirical.data.client import CsvResponse


class ResponseCacheTests(unittest.TestCase):
    def test_round_trip_and_hash_validation(self) -> None:
        payload = b"timestamp,bid,ask\n2025-01-03 10:30:00,1,2\n"
        response = CsvResponse(
            frame=pd.DataFrame(),
            payload=payload,
            request_url="http://127.0.0.1/example",
            status_code=200,
            elapsed_ms=1.0,
            response_headers={"Content-Type": "text/csv"},
        )
        params = {"symbol": "SPY", "date": "20250103"}

        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(directory)
            stored = cache.store("stock_quotes", "/stock/history/quote", params, response)
            loaded = cache.load("stock_quotes", "/stock/history/quote", params)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded.frame), 1)
            self.assertEqual(stored.payload_path, loaded.payload_path)

            Path(loaded.payload_path).write_bytes(b"tampered")
            with self.assertRaises(CacheCorruptionError):
                cache.load("stock_quotes", "/stock/history/quote", params)


if __name__ == "__main__":
    unittest.main()
