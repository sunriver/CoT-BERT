"""extract_eval_metrics：STS12–STS16 聚合 Spearman。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.normalize import extract_eval_metrics  # noqa: E402


class TestExtractStsYears(unittest.TestCase):
    def test_sts_wmean_spearman(self) -> None:
        results = {
            "STS12": {
                "all": {
                    "spearman": {"wmean": 0.111, "mean": 0.222},
                }
            },
            "STS13": {"all": {"spearman": {"mean": 0.333}}},
        }
        m = extract_eval_metrics(results)
        self.assertAlmostEqual(m["sts12_wmean_spearman"], 0.111)
        self.assertAlmostEqual(m["sts13_wmean_spearman"], 0.333)
        self.assertNotIn("sts14_wmean_spearman", m)


if __name__ == "__main__":
    unittest.main()
