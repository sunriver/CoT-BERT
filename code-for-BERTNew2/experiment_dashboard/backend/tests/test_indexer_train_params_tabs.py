"""同一 run_id 在 list_runs 不同 tab 下 train_params_flat 一致（与 eval 优先级一致）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.indexer import RawRun, RunIndex  # noqa: E402


class TestIndexerTrainParamsTabParity(unittest.TestCase):
    def test_same_train_params_flat_across_tabs_when_eval_present(self) -> None:
        idx = RunIndex()
        rid = "run_tab_parity_1"
        ev_cfg = {"model_args": {"temp": 0.01}}
        rs_cfg = {"model_args": {"temp": 0.99}}
        ev = {
            "train_config_full": ev_cfg,
            "eval_run_tag": rid,
            "results": {},
            "timestamp": "2020-01-01T00:00:00",
        }
        rs = {
            "train_config_full": rs_cfg,
            "eval_run_tag": rid,
            "datasets": [],
        }
        al = {"eval_run_tag": rid, "results": [], "timestamp": "2020-01-01T00:00:01"}
        idx._runs[rid] = RawRun(
            run_id=rid,
            source_dir="/tmp",
            eval_test_data=ev,
            run_summary_data=rs,
            alignment_data=al,
        )

        def flat_for(tab: str) -> dict:
            items = idx.list_runs(tab=tab)
            match = [x for x in items if x.run_id == rid]
            self.assertEqual(len(match), 1, tab)
            return match[0].train_params_flat or {}

        f_se = flat_for("senteval")
        f_al = flat_for("alignment")
        f_go = flat_for("gold")
        self.assertEqual(f_se, f_al)
        self.assertEqual(f_se, f_go)
        self.assertEqual(f_se.get("model_temp"), 0.01)
        item = next(x for x in idx.list_runs("senteval") if x.run_id == rid)
        self.assertEqual(item.train_params_flat_eval_test.get("model_temp"), 0.01)

    def test_alignment_tab_gets_params_from_run_summary_when_no_eval(self) -> None:
        idx = RunIndex()
        rid = "run_only_rs_al"
        rs = {
            "train_config_full": {"model_args": {"lr": 3e-5}},
            "eval_run_tag": rid,
            "datasets": [],
        }
        al = {"eval_run_tag": rid, "results": []}
        idx._runs[rid] = RawRun(
            run_id=rid,
            source_dir="/tmp",
            run_summary_data=rs,
            alignment_data=al,
        )
        items = idx.list_runs("alignment")
        match = [x for x in items if x.run_id == rid]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0].train_params_flat.get("model_lr"), 3e-5)
        self.assertEqual(match[0].train_params_flat_eval_test, {})

if __name__ == "__main__":
    unittest.main()
