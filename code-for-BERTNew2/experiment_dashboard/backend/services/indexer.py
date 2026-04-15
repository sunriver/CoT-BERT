from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import normalize as norm
from .models import ArtifactRef, PointsFile, PointsScatterResponse, PointsSeries, RunDetail, RunListItem


def _code_for_bert_new2_root() -> Path:
    # experiment_dashboard/backend/services/indexer.py -> parents[2] = experiment_dashboard, [3] = code-for-BERTNew2
    return Path(__file__).resolve().parents[3]


def _discover_sibling_expirments_cot_eval(repo_root: Path, add) -> None:
    """仓库在 …/CoT-BERT/code-for-BERTNew2，数据在同级其它目录下 …/论文项目代码/expirments/… 时，沿 repo 上两级扫描各子目录。"""
    try:
        workspace = repo_root.parent.parent
        if not workspace.is_dir():
            return
        tail = Path("expirments") / "bertNew2" / "cot_bert_eval"
        for child in workspace.iterdir():
            if not child.is_dir():
                continue
            add(child / tail)
    except OSError:
        pass


def default_data_dirs() -> List[Path]:
    root = _code_for_bert_new2_root()
    dirs: List[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        if p.is_dir():
            r = p.resolve()
            k = str(r)
            if k not in seen:
                seen.add(k)
                dirs.append(r)

    add(root / "eval_results")
    # 常见布局：…/论文项目代码/CoT-BERT/code-for-BERTNew2 → expirments 在 repo 上两级
    add(root.parent.parent / "expirments" / "bertNew2" / "cot_bert_eval")
    # 备选：…/某根目录/code-for-BERTNew2 → expirments 与 code-for-BERTNew2 同级
    add(root.parent / "expirments" / "bertNew2" / "cot_bert_eval")
    # 数据与 CoT-BERT 仓库不在同一棵子目录时（例如 …/local/code/论文项目代码/expirments）
    _discover_sibling_expirments_cot_eval(root, add)
    gsub = root / "eval_results" / "gold_cosine_scatter"
    add(gsub)
    return dirs


def configured_data_dirs() -> List[Path]:
    raw = os.environ.get("EXPERIMENT_VIZ_DATA_DIRS", "").strip()
    root = _code_for_bert_new2_root()
    if not raw:
        return default_data_dirs()
    out: List[Path] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.is_dir():
            out.append(path)
    return out if out else default_data_dirs()


@dataclass
class RawRun:
    run_id: str
    eval_test_path: Optional[Path] = None
    eval_test_data: Optional[Dict[str, Any]] = None
    run_summary_path: Optional[Path] = None
    run_summary_data: Optional[Dict[str, Any]] = None
    alignment_path: Optional[Path] = None
    alignment_data: Optional[Dict[str, Any]] = None
    source_dir: Optional[str] = None
    gold_scatter_png_names: List[str] = field(default_factory=list)
    gold_png_latest_mtime: float = 0.0


class RunIndex:
    def __init__(self) -> None:
        self.data_dirs: List[Path] = configured_data_dirs()
        self._runs: Dict[str, RawRun] = {}

    def __len__(self) -> int:
        return len(self._runs)

    def refresh(self) -> None:
        self.data_dirs = configured_data_dirs()
        self._runs = {}
        for d in self.data_dirs:
            self._scan_dir(d)

    def _scan_dir(self, d: Path) -> None:
        if not d.is_dir():
            return
        try:
            for path in sorted(d.glob("eval_test_*.json")):
                self._ingest_eval_test(path, d)
            for path in sorted(d.glob("run_summary*.json")):
                self._ingest_run_summary(path, d)
            for path in sorted(d.glob("alignment_uniformity_benchmark*.json")):
                self._ingest_alignment(path, d)
            for path in sorted(d.glob("gold_cosine_scatter_*.png")):
                self._ingest_gold_scatter_png(path, d)
        except OSError:
            pass

    def _ensure(self, run_id: str, source_dir: Path) -> RawRun:
        if run_id not in self._runs:
            self._runs[run_id] = RawRun(run_id=run_id, source_dir=str(source_dir))
        r = self._runs[run_id]
        if not r.source_dir:
            r.source_dir = str(source_dir)
        return r

    def _ingest_eval_test(self, path: Path, d: Path) -> None:
        data = norm.load_json(path)
        if not data:
            return
        rid = norm.run_id_from_eval_test(data, path)
        r = self._ensure(rid, d)
        r.eval_test_path = path
        r.eval_test_data = data

    def _ingest_run_summary(self, path: Path, d: Path) -> None:
        data = norm.load_json(path)
        if not data:
            return
        rid = norm.run_id_from_run_summary(data, path)
        if not rid:
            return
        r = self._ensure(rid, d)
        r.run_summary_path = path
        r.run_summary_data = data

    def _ingest_alignment(self, path: Path, d: Path) -> None:
        data = norm.load_json(path)
        if not data:
            return
        rid = norm.run_id_from_alignment(data, path)
        if rid in ("alignment_uniformity_benchmark", path.stem) and path.stem.startswith(
            "alignment_uniformity_benchmark_"
        ):
            rid = path.stem[len("alignment_uniformity_benchmark_") :]
        r = self._ensure(rid, d)
        r.alignment_path = path
        r.alignment_data = data

    def _ingest_gold_scatter_png(self, path: Path, d: Path) -> None:
        rid = norm.run_id_from_gold_scatter_filename(path.name)
        if not rid:
            return
        r = self._ensure(rid, d)
        if path.name not in r.gold_scatter_png_names:
            r.gold_scatter_png_names.append(path.name)
            r.gold_scatter_png_names.sort()
        try:
            mt = path.stat().st_mtime
            if mt > r.gold_png_latest_mtime:
                r.gold_png_latest_mtime = mt
        except OSError:
            pass

    def list_runs(self, tab: str = "all") -> List[RunListItem]:
        """tab: all | senteval | alignment | gold"""
        items: List[RunListItem] = []
        for rid, raw in sorted(self._runs.items(), key=lambda x: x[0]):
            ev = raw.eval_test_data
            rs_data = raw.run_summary_data
            al_data = raw.alignment_data

            metrics: Dict[str, Any] = {}
            if ev:
                metrics = norm.extract_eval_metrics(ev.get("results"))

            al_metrics = norm.extract_alignment_cot_metrics(al_data)
            gold_metrics = norm.extract_gold_scatter_metrics(rs_data)

            ts = None
            if ev:
                ts = ev.get("timestamp")
            if not ts and isinstance(al_data, dict):
                ts = al_data.get("timestamp")
            if not ts and isinstance(rs_data, dict):
                ts = rs_data.get("eval_started_at_iso") or rs_data.get("run_timestamp")
            if (
                not ts
                and raw.gold_png_latest_mtime > 0
            ):
                ts = datetime.fromtimestamp(
                    raw.gold_png_latest_mtime, tz=timezone.utc
                ).isoformat()

            tag = rid
            if ev and ev.get("eval_run_tag"):
                tag = ev.get("eval_run_tag")
            elif isinstance(rs_data, dict) and rs_data.get("eval_run_tag"):
                tag = rs_data.get("eval_run_tag")
            elif isinstance(al_data, dict) and al_data.get("eval_run_tag"):
                tag = al_data.get("eval_run_tag")

            git_short = None
            if ev:
                g = ev.get("git")
                if isinstance(g, dict):
                    git_short = g.get("commit_short")

            best_m = None
            if ev:
                tr = ev.get("trainer_state_summary")
                if isinstance(tr, dict):
                    best_m = tr.get("best_metric")
                    if not isinstance(best_m, (int, float)):
                        best_m = None

            has_eval = bool(ev)
            has_al = bool(al_data)
            has_rs = bool(rs_data)

            gold_pngs: List[str] = []
            if rs_data:
                pairs = norm.collect_artifacts_for_run(rid, rs_data, self.data_dirs)
                gold_pngs = [
                    n for n, _ in pairs if n.lower().endswith(".png")
                ]
            gold_pngs = sorted(set(gold_pngs) | set(raw.gold_scatter_png_names))
            gold_by_ds = norm.gold_scatter_pngs_by_dataset(gold_pngs)

            tc_full = norm.train_config_full_from_sources(ev, rs_data, al_data)
            train_flat = norm.flatten_train_config(tc_full)
            train_flat_eval: Dict[str, Any] = {}
            if ev and isinstance(ev.get("train_config_full"), dict):
                train_flat_eval = norm.flatten_train_config(ev["train_config_full"])

            item = RunListItem(
                run_id=rid,
                eval_run_tag=tag if isinstance(tag, str) else str(tag),
                timestamp=ts,
                source_dir=raw.source_dir,
                has_eval_test=has_eval,
                has_alignment=has_al,
                has_run_summary=has_rs,
                stsb_test_spearman=metrics.get("stsb_test_spearman"),
                sick_test_spearman=metrics.get("sick_test_spearman"),
                sts_avg_percent=metrics.get("sts_avg_percent"),
                sts12_wmean_spearman=metrics.get("sts12_wmean_spearman"),
                sts13_wmean_spearman=metrics.get("sts13_wmean_spearman"),
                sts14_wmean_spearman=metrics.get("sts14_wmean_spearman"),
                sts15_wmean_spearman=metrics.get("sts15_wmean_spearman"),
                sts16_wmean_spearman=metrics.get("sts16_wmean_spearman"),
                best_metric=best_m if isinstance(best_m, (int, float)) else None,
                git_commit_short=git_short,
                alignment_cot_align_mean=al_metrics.get("alignment_cot_align_mean"),
                alignment_cot_unif_mean=al_metrics.get("alignment_cot_unif_mean"),
                gold_pearson_mean_cot_mask=gold_metrics.get("gold_pearson_mean_cot_mask"),
                gold_pearson_mean_cot_mask_mlp=gold_metrics.get(
                    "gold_pearson_mean_cot_mask_mlp"
                ),
                gold_pearson_mean_bert_base_cls=gold_metrics.get(
                    "gold_pearson_mean_bert_base_cls"
                ),
                gold_cosine_pngs=gold_pngs,
                gold_cosine_png_by_dataset=gold_by_ds,
                train_params_flat=train_flat,
                train_params_flat_eval_test=train_flat_eval,
            )
            items.append(item)

        t = (tab or "all").strip().lower()
        if t == "senteval":
            items = [x for x in items if x.has_eval_test]
        elif t == "alignment":
            items = [x for x in items if x.has_alignment]
        elif t == "gold":
            items = [
                x
                for x in items
                if x.has_run_summary or (x.gold_cosine_pngs and len(x.gold_cosine_pngs) > 0)
            ]

        items.sort(key=lambda x: (x.timestamp or ""), reverse=True)
        return items

    def get_detail(self, run_id: str) -> Optional[RunDetail]:
        raw = self._runs.get(run_id)
        if not raw:
            return None
        ev = raw.eval_test_data
        results = ev.get("results") if ev else None
        metrics = norm.extract_eval_metrics(results if isinstance(results, dict) else None)
        run_summary = raw.run_summary_data
        alignment = raw.alignment_data

        tc_full = norm.train_config_full_from_sources(ev, run_summary, alignment)
        train_flat = norm.flatten_train_config(tc_full)

        artifacts_pairs = norm.collect_artifacts_for_run(
            run_id, run_summary, self.data_dirs
        )
        by_name: Dict[str, str] = {n: p for n, p in artifacts_pairs}
        for name in raw.gold_scatter_png_names:
            if name in by_name:
                continue
            rp = norm.resolve_basename_in_data_dirs(name, self.data_dirs)
            if rp:
                by_name[name] = str(rp)
        artifacts = [ArtifactRef(name=n, path=p) for n, p in sorted(by_name.items())]

        source_files = {}
        if raw.eval_test_path:
            source_files["eval_test"] = str(raw.eval_test_path.resolve())
        if raw.run_summary_path:
            source_files["run_summary"] = str(raw.run_summary_path.resolve())
        if raw.alignment_path:
            source_files["alignment_uniformity"] = str(raw.alignment_path.resolve())

        tag = None
        tag_src = None
        if ev:
            tag = ev.get("eval_run_tag")
            tag_src = ev.get("eval_run_tag_source")
        if not tag and run_summary:
            tag = run_summary.get("eval_run_tag")
            tag_src = run_summary.get("eval_run_tag_source")
        ts = ev.get("timestamp") if ev else None
        if not ts and isinstance(alignment, dict):
            ts = alignment.get("timestamp")
        if not ts and run_summary:
            ts = run_summary.get("eval_started_at_iso")
        if not ts and raw.gold_png_latest_mtime > 0:
            ts = datetime.fromtimestamp(
                raw.gold_png_latest_mtime, tz=timezone.utc
            ).isoformat()

        return RunDetail(
            run_id=run_id,
            eval_run_tag=str(tag) if tag is not None else run_id,
            eval_run_tag_source=str(tag_src) if tag_src else None,
            timestamp=ts,
            source_files=source_files,
            summary_metrics={k: v for k, v in metrics.items() if k != "summary_table"},
            summary_table=metrics.get("summary_table")
            if isinstance(metrics.get("summary_table"), dict)
            else None,
            train_params_flat=train_flat,
            trainer_state_summary=ev.get("trainer_state_summary") if ev else None,
            git=ev.get("git") if ev else None,
            results_raw=results if isinstance(results, dict) else None,
            run_summary=run_summary,
            alignment_uniformity=alignment,
            artifacts=artifacts,
        )

    def compare(self, run_ids: List[str]) -> Optional[Dict[str, Any]]:
        details: List[RunDetail] = []
        found_ids: List[str] = []
        for r in run_ids:
            d = self.get_detail(r)
            if d:
                details.append(d)
                found_ids.append(d.run_id)
        if not details:
            return None

        metric_keys = [
            "stsb_test_spearman",
            "sick_test_spearman",
            "sts_avg_percent",
            "sts12_wmean_spearman",
            "sts13_wmean_spearman",
            "sts14_wmean_spearman",
            "sts15_wmean_spearman",
            "sts16_wmean_spearman",
        ]
        metric_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for d in details:
            metric_matrix[d.run_id] = {}
            sm = d.summary_metrics
            for mk in metric_keys:
                v = sm.get(mk) if isinstance(sm, dict) else None
                if v is not None and not isinstance(v, (int, float)):
                    v = None
                metric_matrix[d.run_id][mk] = float(v) if isinstance(v, (int, float)) else None
            tr = d.trainer_state_summary or {}
            bm = tr.get("best_metric")
            metric_matrix[d.run_id]["best_metric"] = (
                float(bm) if isinstance(bm, (int, float)) else None
            )
        metric_keys = metric_keys + ["best_metric"]

        param_matrix: Dict[str, Dict[str, Any]] = {
            d.run_id: dict(d.train_params_flat) for d in details
        }
        param_diff_only = norm.param_diff(param_matrix)
        all_param_keys = sorted(
            set().union(*(d.train_params_flat.keys() for d in details))
        )

        return {
            "run_ids": found_ids,
            "metric_keys": metric_keys,
            "metric_matrix": metric_matrix,
            "param_keys": all_param_keys,
            "param_matrix": param_matrix,
            "param_diff_only": param_diff_only,
        }

    def get_points_files(self) -> List[PointsFile]:
        """返回所有可用的 points_*.csv 文件（去重）"""
        files: List[PointsFile] = []
        seen = set()
        for d in self.data_dirs:
            if not d.is_dir():
                continue
            try:
                for p in sorted(d.glob("points_*.csv")):
                    name = p.name
                    if name in seen:
                        continue
                    seen.add(name)
                    mtime = p.stat().st_mtime if p.exists() else None
                    files.append(PointsFile(name=name, mtime=mtime))
            except OSError:
                continue
        return sorted(files, key=lambda f: f.name)

    def get_points_scatter(
        self,
        files: List[str],
        run_id: str = "cot_mask",
        dataset_id: Optional[str] = None,
        limit: int = 2500,
    ) -> PointsScatterResponse:
        """返回指定 CSV 中 run_id 的散点数据（抽样）"""
        import csv
        import random

        series_list: List[PointsSeries] = []
        spearman_dict: Dict[str, float] = {}

        for fname in files:
            for d in self.data_dirs:
                path = d / fname
                if not path.is_file():
                    continue
                data: List[Dict[str, float]] = []
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("run_id") != run_id:
                            continue
                        if dataset_id and row.get("dataset_id") != dataset_id:
                            continue
                        try:
                            g = float(row["gold"])
                            c = float(row["cos_pred"])
                            data.append({"gold": g, "cos": c})
                        except (ValueError, KeyError):
                            continue
                golds = [p["gold"] for p in data]
                coss = [p["cos"] for p in data]
                rho = norm.spearman_correlation(golds, coss)
                if rho is not None:
                    spearman_dict[fname] = rho
                if len(data) > limit:
                    seed = (hash(fname) & 0xFFFFFFFF) ^ 0x9E3779B9
                    rng = random.Random(seed)
                    data = rng.sample(data, limit)
                label = fname.replace("points_", "").replace(".csv", "")
                series_list.append(PointsSeries(file=fname, label=label, data=data))
                break

        return PointsScatterResponse(series=series_list, spearman=spearman_dict or None)


_index: Optional[RunIndex] = None


def get_index() -> RunIndex:
    global _index
    if _index is None:
        _index = RunIndex()
        _index.refresh()
    return _index
