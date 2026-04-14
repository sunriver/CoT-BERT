"""train_config_full 多源解析：eval_test > run_summary > alignment。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.normalize import flatten_train_config, train_config_full_from_sources  # noqa: E402


class TestTrainConfigFullFromSources(unittest.TestCase):
    def test_priority_eval_over_others(self) -> None:
        ev = {"train_config_full": {"model_args": {"temp": 0.1}}}
        rs = {"train_config_full": {"model_args": {"temp": 0.9}}}
        al = {"train_config_full": {"model_args": {"temp": 0.5}}}
        tc = train_config_full_from_sources(ev, rs, al)
        self.assertIsNotNone(tc)
        flat = flatten_train_config(tc)
        self.assertEqual(flat.get("model_temp"), 0.1)

    def test_fallback_run_summary(self) -> None:
        rs = {"train_config_full": {"data_args": {"train_file": "x.json"}}}
        tc = train_config_full_from_sources(None, rs, None)
        self.assertIsNotNone(tc)
        flat = flatten_train_config(tc)
        self.assertEqual(flat.get("data_train_file"), "x.json")

    def test_fallback_alignment(self) -> None:
        al = {"train_config_full": {"model_args": {"dropout": 0.2}}}
        tc = train_config_full_from_sources(None, None, al)
        self.assertIsNotNone(tc)
        flat = flatten_train_config(tc)
        self.assertEqual(flat.get("model_dropout"), 0.2)

    def test_none_when_missing(self) -> None:
        self.assertIsNone(
            train_config_full_from_sources({"foo": 1}, {"bar": 2}, None)
        )

    def test_ignores_non_dict_train_config_full(self) -> None:
        tc = train_config_full_from_sources(
            {"train_config_full": "bad"},
            {"train_config_full": {"model_args": {"a": 1}}},
            None,
        )
        self.assertIsNotNone(tc)
        self.assertEqual(flatten_train_config(tc).get("model_a"), 1)


if __name__ == "__main__":
    unittest.main()
