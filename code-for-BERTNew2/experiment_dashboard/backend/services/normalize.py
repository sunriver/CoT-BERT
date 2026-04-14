from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# gold_cosine_scatter_{dataset}_{eval_run_tag}.png
_GOLD_SCATTER_KNOWN_DATASETS = ("sick_test", "sts_test")

_STS_SENTEVAL_YEARS = ("STS12", "STS13", "STS14", "STS15", "STS16")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sts_year_wmean_spearman(results: Dict[str, Any], year_key: str) -> Optional[float]:
    """SentEval STS 各年任务聚合：优先 all.spearman.wmean，否则 mean。"""
    block = results.get(year_key)
    if not isinstance(block, dict):
        return None
    all_b = block.get("all")
    if not isinstance(all_b, dict):
        return None
    sp = all_b.get("spearman")
    if not isinstance(sp, dict):
        return None
    w = _safe_float(sp.get("wmean"))
    if w is not None:
        return w
    return _safe_float(sp.get("mean"))


def extract_eval_metrics(results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(results, dict):
        return out
    st = results.get("STSBenchmark")
    if isinstance(st, dict):
        test = st.get("test")
        if isinstance(test, dict):
            sp = test.get("spearman")
            if isinstance(sp, (list, tuple)) and sp:
                out["stsb_test_spearman"] = _safe_float(sp[0])
            elif isinstance(sp, dict) and "correlation" in sp:
                out["stsb_test_spearman"] = _safe_float(sp.get("correlation"))
    sk = results.get("SICKRelatedness")
    if isinstance(sk, dict):
        test = sk.get("test")
        if isinstance(test, dict):
            sp = test.get("spearman")
            if isinstance(sp, (list, tuple)) and sp:
                out["sick_test_spearman"] = _safe_float(sp[0])
    sm = results.get("summary_table")
    if isinstance(sm, dict):
        scores = sm.get("scores")
        tasks = sm.get("tasks")
        if isinstance(scores, list) and scores:
            out["sts_avg_percent"] = _safe_float(scores[-1])
        out["summary_table"] = sm
    for yk in _STS_SENTEVAL_YEARS:
        v = _sts_year_wmean_spearman(results, yk)
        if v is not None:
            out[f"{yk.lower()}_wmean_spearman"] = v
    return out


def flatten_train_config(train_config_full: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if not isinstance(train_config_full, dict):
        return flat
    ma = train_config_full.get("model_args")
    if isinstance(ma, dict):
        for k, v in ma.items():
            flat[f"model_{k}"] = v
    ta = train_config_full.get("training_args")
    if isinstance(ta, dict):
        for k, v in ta.items():
            if isinstance(v, (dict, list)) and k not in (
                "debug",
                "sharded_ddp",
                "fsdp",
                "fsdp_config",
            ):
                continue
            flat[f"train_{k}"] = v
    da = train_config_full.get("data_args")
    if isinstance(da, dict):
        for k, v in da.items():
            flat[f"data_{k}"] = v
    return flat


def train_config_full_from_sources(
    eval_test: Optional[Dict[str, Any]] = None,
    run_summary: Optional[Dict[str, Any]] = None,
    alignment: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """同一 run 在 SentEval / Gold / Alignment 等多份 JSON 中应共享 train_config_full；按优先级取第一份非空 dict。"""
    for src in (eval_test, run_summary, alignment):
        if not isinstance(src, dict):
            continue
        tc = src.get("train_config_full")
        if isinstance(tc, dict):
            return tc
    return None


def run_id_from_eval_test(data: Dict[str, Any], path: Path) -> str:
    tag = data.get("eval_run_tag")
    if tag is not None and str(tag).strip():
        return str(tag).strip()
    stem = path.stem
    if stem.startswith("eval_test_"):
        return stem[len("eval_test_") :]
    return stem


def run_id_from_run_summary(data: Dict[str, Any], path: Path) -> Optional[str]:
    """无有效 tag 且文件名为裸 ``run_summary.json`` 时返回 None，避免索引里出现 run_id=run_summary 的假行。"""
    for key in ("eval_run_tag", "run_timestamp"):
        tag = data.get(key)
        if tag is not None and str(tag).strip():
            return str(tag).strip()
    stem = path.stem
    if stem.startswith("run_summary_"):
        rest = stem[len("run_summary_") :]
        return rest if rest else None
    if stem == "run_summary":
        return None
    return stem if stem else None


def run_id_from_alignment(data: Dict[str, Any], path: Path) -> str:
    tag = data.get("eval_run_tag")
    if tag is not None and str(tag).strip():
        return str(tag).strip()
    stem = path.stem
    prefix = "alignment_uniformity_benchmark_"
    if stem.startswith(prefix):
        rest = stem[len(prefix) :]
        if rest:
            return rest
    return stem


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_artifact_path(
    stored_path: str, data_dirs: List[Path]
) -> Optional[Path]:
    if not stored_path or not str(stored_path).strip():
        return None
    p = Path(stored_path).expanduser()
    try:
        if p.is_file():
            return p.resolve()
    except OSError:
        pass
    name = p.name
    for d in data_dirs:
        if not d.is_dir():
            continue
        try:
            for candidate in d.rglob(name):
                if candidate.is_file():
                    return candidate.resolve()
        except OSError:
            continue
    return None


def collect_artifacts_for_run(
    run_id: str,
    run_summary: Optional[Dict[str, Any]],
    data_dirs: List[Path],
) -> List[Tuple[str, str]]:
    """Return list of (name, resolved_absolute_path_str)."""
    found: Dict[str, str] = {}

    if isinstance(run_summary, dict):
        csv_path = run_summary.get("csv")
        if isinstance(csv_path, str) and csv_path.strip():
            rp = resolve_artifact_path(csv_path, data_dirs)
            if rp:
                found[rp.name] = str(rp)
        for ds in run_summary.get("datasets") or []:
            if not isinstance(ds, dict):
                continue
            fig = ds.get("figure")
            if isinstance(fig, str) and fig.strip():
                rp = resolve_artifact_path(fig, data_dirs)
                if rp:
                    found[rp.name] = str(rp)

    suffixes = (".png", ".csv", ".jpg", ".jpeg", ".webp")
    for d in data_dirs:
        if not d.is_dir():
            continue
        try:
            for pat in (
                f"*{run_id}*.png",
                f"*{run_id}*.csv",
                f"*{run_id}*.jpg",
            ):
                for f in d.glob(pat):
                    if f.is_file() and f.suffix.lower() in suffixes:
                        found[f.name] = str(f.resolve())
        except OSError:
            continue
    return [(k, v) for k, v in sorted(found.items())]


def param_diff(param_matrix: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """For each param key, include only if values differ across runs."""
    run_ids = list(param_matrix.keys())
    if len(run_ids) < 2:
        return {k: {r: param_matrix[r].get(k) for r in run_ids} for k in _all_keys(param_matrix)}
    keys = _all_keys(param_matrix)
    diff: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        vals = [param_matrix[r].get(k) for r in run_ids]
        if _all_equal(vals):
            continue
        diff[k] = {r: param_matrix[r].get(k) for r in run_ids}
    return diff


def _all_keys(param_matrix: Dict[str, Dict[str, Any]]) -> List[str]:
    s: set[str] = set()
    for d in param_matrix.values():
        s.update(d.keys())
    return sorted(s)


def _all_equal(vals: List[Any]) -> bool:
    if not vals:
        return True
    a = vals[0]
    for b in vals[1:]:
        if a != b:
            return False
    return True


def extract_alignment_cot_metrics(alignment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """对齐 benchmark 中 cot_bert_local（或名称含 cot 的本地模型）的 alignment/uniformity 按数据集取平均。"""
    out: Dict[str, Any] = {}
    if not isinstance(alignment, dict):
        return out
    results = alignment.get("results")
    if not isinstance(results, list):
        return out
    rows = [
        r
        for r in results
        if isinstance(r, dict) and r.get("model_id") == "cot_bert_local"
    ]
    if not rows:
        rows = [
            r
            for r in results
            if isinstance(r, dict)
            and "cot" in str(r.get("model_id", "")).lower()
            and "bert" in str(r.get("model_id", "")).lower()
        ]
    if not rows:
        return out
    als = [r["alignment"] for r in rows if isinstance(r.get("alignment"), (int, float))]
    uns = [r["uniformity"] for r in rows if isinstance(r.get("uniformity"), (int, float))]
    if als:
        out["alignment_cot_align_mean"] = sum(als) / len(als)
    if uns:
        out["alignment_cot_unif_mean"] = sum(uns) / len(uns)
    return out


def _mean_pearson_for_run_id(rs: Optional[Dict[str, Any]], target_run_id: str) -> Optional[float]:
    if not isinstance(rs, dict):
        return None
    vals: List[float] = []
    for ds in rs.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        for run in ds.get("runs") or []:
            if not isinstance(run, dict):
                continue
            if run.get("run_id") != target_run_id:
                continue
            p = run.get("pearson_gold_cos")
            if isinstance(p, (int, float)):
                vals.append(float(p))
    if not vals:
        return None
    return sum(vals) / len(vals)


def dataset_and_run_id_from_gold_scatter_filename(
    filename: str,
) -> Tuple[Optional[str], Optional[str]]:
    """返回 (数据集名如 sick_test, eval_run_tag)。"""
    if not filename or not filename.lower().endswith(".png"):
        return None, None
    stem = filename[:-4]
    prefix = "gold_cosine_scatter_"
    if not stem.startswith(prefix):
        return None, None
    rest = stem[len(prefix) :]
    for ds in _GOLD_SCATTER_KNOWN_DATASETS:
        p = ds + "_"
        if rest.startswith(p):
            tag = rest[len(p) :].strip()
            return ds, (tag or None)
    return None, None


def run_id_from_gold_scatter_filename(filename: str) -> Optional[str]:
    """从 ``gold_cosine_scatter_sick_test_<tag>.png`` 等文件名解析 eval_run_tag。"""
    _, tag = dataset_and_run_id_from_gold_scatter_filename(filename)
    return tag


def gold_scatter_pngs_by_dataset(filenames: List[str]) -> Dict[str, str]:
    """每个数据集保留一个 PNG 文件名（sick_test / sts_test）。"""
    out: Dict[str, str] = {}
    for fn in filenames:
        ds, _ = dataset_and_run_id_from_gold_scatter_filename(fn)
        if ds:
            out[ds] = fn
    return out


def resolve_basename_in_data_dirs(
    basename: str, data_dirs: List[Path]
) -> Optional[Path]:
    for d in data_dirs:
        if not d.is_dir():
            continue
        try:
            direct = d / basename
            if direct.is_file():
                return direct.resolve()
            for candidate in d.rglob(basename):
                if candidate.is_file():
                    return candidate.resolve()
        except OSError:
            continue
    return None


def extract_gold_scatter_metrics(rs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """gold_cosine_scatter 的 run_summary：各 run_id 在多个数据集上 pearson_gold_cos 的平均。"""
    out: Dict[str, Any] = {}
    if not isinstance(rs, dict):
        return out
    for rid in ("cot_mask", "cot_mask_mlp", "bert_base_cls"):
        m = _mean_pearson_for_run_id(rs, rid)
        if m is not None:
            out[f"gold_pearson_mean_{rid}"] = m
    return out
