#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金标（相似度分数）与句向量余弦相似度散点图实验。

- 数据集与多模型 runs 由 YAML 配置（默认 configs/gold_cosine_scatter_default.yaml）。
- 支持 `datasets` 列表（每项含 dataset_id、path、format 等）；仍支持单个 `dataset` 块（兼容旧版）。
- 第一个参数若为存在的 .yaml/.yml，则与默认配置深度合并（与 sick_r_alignment_uniformity 用法一致）。
- 模型为本地 BERT checkpoint 目录；编码复用 sick_r_alignment_uniformity.encode_word_batches 等。

用法（在 code-for-BERTNew2 目录下）:
  python gold_cosine_scatter.py
  python gold_cosine_scatter.py configs/gold_cosine_scatter_linux_cuda.yaml
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "gold_cosine_scatter_default.yaml")

# 配置 Hugging Face 国内镜像，解决连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def merge_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并，d2 优先（与 parse_args_util 行为一致）。"""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            merge_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


def _merge_dataset_lists(
    base_list: List[Any],
    overlay_list: List[Any],
) -> List[Dict[str, Any]]:
    """按 dataset_id 合并：覆盖 yaml 可只写 path，保留 default 中的 format 等。"""
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for item in base_list:
        if not isinstance(item, dict):
            continue
        bid = str(item.get("dataset_id", "")).strip() or f"_b{len(order)}"
        by_id[bid] = dict(item)
        if bid not in order:
            order.append(bid)
    for item in overlay_list:
        if not isinstance(item, dict):
            continue
        oid = str(item.get("dataset_id", "")).strip()
        if not oid:
            continue
        if oid in by_id:
            by_id[oid].update(item)
        else:
            by_id[oid] = dict(item)
            order.append(oid)
    return [by_id[k] for k in order if k in by_id]


def merge_experiment_config(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """合并自定义配置；datasets 按 dataset_id 增量合并，其余与 merge_dict 一致。"""
    for k, v in d2.items():
        if k == "datasets" and isinstance(v, list) and isinstance(
            d1.get("datasets"), list
        ):
            d1["datasets"] = _merge_dataset_lists(d1["datasets"], v)
        elif isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            merge_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


def load_yaml_config(default_file: str, custom_file: str) -> Dict[str, Any]:
    import yaml

    config: Dict[str, Any] = {}
    if os.path.isfile(default_file):
        with open(default_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config = loaded
    if custom_file and os.path.isfile(custom_file):
        with open(custom_file, "r", encoding="utf-8") as f:
            custom = yaml.safe_load(f)
            if isinstance(custom, dict):
                merge_experiment_config(config, custom)
    return config


def _split_config_argv(argv: List[str]) -> Tuple[str, List[str]]:
    if not argv:
        return "", []
    first = argv[0]
    if first.endswith((".yaml", ".yml")):
        if not os.path.isfile(first):
            print(f"找不到配置文件: {first}", file=sys.stderr)
            sys.exit(1)
        return first, argv[1:]
    return "", argv


def iter_dataset_specs(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    返回 [(dataset_id, dataset_cfg), ...]。
    优先使用 `datasets`（列表）；否则回退到单个 `dataset`（兼容旧配置）。
    """
    specs: List[Tuple[str, Dict[str, Any]]] = []
    dss = cfg.get("datasets")
    if isinstance(dss, list) and len(dss) > 0:
        for i, item in enumerate(dss):
            if not isinstance(item, dict):
                continue
            block = dict(item)
            did = block.pop("dataset_id", None)
            if did is None or str(did).strip() == "":
                did = f"dataset_{i}"
            specs.append((str(did), block))
        return specs

    single = cfg.get("dataset")
    if isinstance(single, dict):
        block = dict(single)
        did = block.pop("dataset_id", None)
        if did is None or str(did).strip() == "":
            did = "default"
        specs.append((str(did), block))
    return specs


def load_dataset_pairs(dataset_cfg: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """根据 dataset 配置返回 (s1, s2, gold) 列表。"""
    path = dataset_cfg.get("path")
    if not path:
        raise ValueError("dataset.path 未设置")
    path = os.path.abspath(os.path.join(_SCRIPT_DIR, path)) if not os.path.isabs(
        path
    ) else path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")

    fmt = (dataset_cfg.get("format") or "generic_tsv").strip().lower()
    delim = dataset_cfg.get("delimiter")
    if delim is None:
        delim = "\t"
    elif isinstance(delim, str) and delim.lower() in ("\\t", "tab"):
        delim = "\t"

    columns = dataset_cfg.get("columns") or {}
    gold_scale = dataset_cfg.get("gold_scale") or {}
    multiply = gold_scale.get("multiply") if isinstance(gold_scale, dict) else None

    pairs: List[Tuple[str, str, float]] = []

    if fmt == "sick_annotated":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sa = row["sentence_A"].strip()
                sb = row["sentence_B"].strip()
                gold = float(row["relatedness_score"])
                if multiply is not None:
                    gold = float(gold) * float(multiply)
                pairs.append((sa, sb, float(gold)))

    elif fmt == "sts_benchmark":
        ca = columns.get("sentence_a") or columns.get("sentence1") or "sentence1"
        cb = columns.get("sentence_b") or columns.get("sentence2") or "sentence2"
        cg = columns.get("gold") or columns.get("score") or "score"
        ci = columns.get("column_indices") if isinstance(columns, dict) else None
        if not isinstance(ci, dict):
            ci = {}
        ix_gold = int(ci.get("gold", ci.get("score", 4)))
        ix_a = int(
            ci.get("sentence_a", ci.get("sentence1", ci.get("s1", 5)))
        )
        ix_b = int(
            ci.get("sentence_b", ci.get("sentence2", ci.get("s2", 6)))
        )

        def _delimiter_for_path() -> str:
            with open(path, newline="", encoding="utf-8") as f:
                sample = f.read(4096)
            try:
                return csv.Sniffer().sniff(sample).delimiter
            except csv.Error:
                return delim

        d = _delimiter_for_path()

        def _first_line_tokens() -> List[str]:
            with open(path, newline="", encoding="utf-8") as f:
                line = f.readline()
            if not line:
                return []
            return line.rstrip("\n\r").split(d)

        tokens0 = _first_line_tokens()
        force_positional = dataset_cfg.get("sts_no_header") is True
        force_named = dataset_cfg.get("sts_has_header") is True
        looks_like_named_header = False
        if tokens0 and not force_positional:
            lowered = [t.strip().lower() for t in tokens0]
            looks_like_named_header = (
                "sentence1" in lowered
                or "text1" in lowered
                or (len(tokens0) >= 3 and lowered[0] == "index")
            )

        use_positional = force_positional or (
            not force_named and tokens0 and not looks_like_named_header
        )

        if use_positional:
            # SentEval STS-Benchmark sts-test 常见：无表头，7 列 → score 与两句见 column_indices
            max_ix = max(ix_gold, ix_a, ix_b)
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=d)
                for row in reader:
                    if not row or all(not c.strip() for c in row):
                        continue
                    if len(row) <= max_ix:
                        continue
                    gold = float(row[ix_gold])
                    sa = row[ix_a].strip()
                    sb = row[ix_b].strip()
                    if multiply is not None:
                        gold *= float(multiply)
                    pairs.append((sa, sb, gold))
        else:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=d)
                if reader.fieldnames is None:
                    raise ValueError(f"{path} 无表头且无法按列下标解析")
                for row in reader:
                    if ca in row and cb in row and cg in row:
                        sa = row[ca].strip()
                        sb = row[cb].strip()
                        gold = float(row[cg])
                    elif "main_caption" in row and "vice_caption" in row:
                        sa = row["main_caption"].strip()
                        sb = row["vice_caption"].strip()
                        score_key = cg if cg in row else "score"
                        if score_key not in row:
                            raise KeyError(
                                f"缺少分数列（试过 score / {cg}），表头: {reader.fieldnames}"
                            )
                        gold = float(row[score_key])
                    else:
                        missing = [k for k in (ca, cb, cg) if k not in row]
                        raise KeyError(
                            f"行缺少列 {missing}，当前表头: {reader.fieldnames}"
                        )
                    if multiply is not None:
                        gold *= float(multiply)
                    pairs.append((sa, sb, gold))

    elif fmt == "generic_tsv":
        ca = columns.get("sentence_a") or columns.get("sentence1")
        cb = columns.get("sentence_b") or columns.get("sentence2")
        cg = columns.get("gold") or columns.get("score")
        if not all([ca, cb, cg]):
            raise ValueError("generic_tsv 需在 dataset.columns 中提供 sentence_a/sentence_b/gold（或 sentence1/sentence2/score）")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=str(delim))
            for row in reader:
                gold = float(row[cg])
                if multiply is not None:
                    gold *= float(multiply)
                pairs.append((row[ca].strip(), row[cb].strip(), gold))
    else:
        raise ValueError(f"未知 dataset.format: {fmt}")

    if not pairs:
        raise ValueError(f"未从 {path} 读取到任何句对")
    return pairs


def _resolve_run(encoding_defaults: Optional[Dict], run: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if encoding_defaults:
        out.update(encoding_defaults)
    out.update(run)
    return out


def _cosine_rows(
    emb_map: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str, float]],
    to_unit_interval: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    golds = []
    coss = []
    for sa, sb, g in pairs:
        va = emb_map[sa].astype(np.float64)
        vb = emb_map[sb].astype(np.float64)
        na = np.linalg.norm(va) + 1e-12
        nb = np.linalg.norm(vb) + 1e-12
        c = float(np.dot(va, vb) / (na * nb))
        if to_unit_interval:
            c = (c + 1.0) / 2.0
        golds.append(float(g))
        coss.append(c)
    return np.asarray(golds), np.asarray(coss)


def _figure_path_for_dataset(
    out_dir: str,
    fig_name_template: str,
    dataset_id: str,
    multiple_datasets: bool,
) -> str:
    name = fig_name_template or "gold_cosine_scatter.png"
    if "{dataset_id}" in name:
        name = name.replace("{dataset_id}", dataset_id)
    elif multiple_datasets:
        stem, ext = os.path.splitext(name)
        name = f"{stem}_{dataset_id}{ext}"
    return os.path.join(out_dir, name)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_mask_prediction_mlp_for_gold_scatter(model_name_or_path: str, device: Any):
    """
    将 checkpoint 中 pooler 权重载入 BertForMaskedLM.cls.predictions.transform。
    部分 ckpt 在去掉 pooler. 后仍带 bert. 前缀（与 BertPredictionHeadTransform 的 dense/LayerNorm 不一致），
    在此再剥一层 bert.；若仍缺 LayerNorm 等则 strict=False 与预训练初始化混合加载。
    """
    import torch
    import torch.nn as nn
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig.from_pretrained("bert-base-uncased")
    mlp: nn.Module = BertForMaskedLM.from_pretrained(
        "bert-base-uncased", config=config
    ).cls.predictions.transform

    if "result" not in model_name_or_path:
        mlp.eval()
        return mlp.to(device)

    state_dict = torch.load(
        model_name_or_path + "/pytorch_model.bin", map_location="cpu"
    )
    new_state_dict = {}
    for key, param in state_dict.items():
        if "pooler" not in key:
            continue
        nk = key.replace("pooler.", "")
        if nk.startswith("bert."):
            nk = nk.removeprefix("bert.")
        new_state_dict[nk] = param

    mlp.load_state_dict(new_state_dict, strict=False)
    mlp.eval()
    return mlp.to(device)


def main() -> None:
    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoModel, AutoTokenizer

    from cot_bert_evaluation import denoising
    from sick_r_alignment_uniformity import (
        build_batches,
        encode_word_batches,
        pick_device,
    )

    custom_yaml, _rest = _split_config_argv(sys.argv[1:])
    print(f"默认配置: {DEFAULT_CONFIG_PATH}")
    if custom_yaml:
        print(f"覆盖配置: {custom_yaml}")
    cfg = load_yaml_config(DEFAULT_CONFIG_PATH, custom_yaml)

    dataset_specs = iter_dataset_specs(cfg)
    if not dataset_specs:
        print("配置缺少 datasets 列表或 dataset 块", file=sys.stderr)
        sys.exit(1)

    encoding_defaults = cfg.get("encoding_defaults") or {}
    runs = cfg.get("runs")
    if not runs or not isinstance(runs, list):
        print("配置缺少 runs 列表", file=sys.stderr)
        sys.exit(1)

    batch_size = int(cfg.get("batch_size", 64))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    cosine_to_unit = bool(cfg.get("cosine_to_unit_interval", True))
    out_dir = cfg.get("output_dir") or "./eval_results/gold_cosine_scatter"
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_SCRIPT_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig_ncols = int(cfg.get("fig_ncols", 3))
    fig_size_s = cfg.get("fig_size") or "15,4"
    try:
        ws, hs = [float(x.strip()) for x in str(fig_size_s).split(",")]
    except ValueError:
        ws, hs = 15.0, 4.0

    device = pick_device()
    print(f"Device: {device}")

    multi_ds = len(dataset_specs) > 1
    fig_name_tpl = str(cfg.get("figure_filename") or "gold_cosine_scatter.png")

    all_rows_csv: List[Dict[str, Any]] = []
    dataset_records: List[Dict[str, Any]] = []

    n_runs = len(runs)
    nrows = int(math.ceil(n_runs / fig_ncols))

    for dataset_id, dataset_cfg in dataset_specs:
        pairs = load_dataset_pairs(dataset_cfg)
        print(f"[{dataset_id}] 加载句对数: {len(pairs)}")

        endpoints: List[str] = []
        for sa, sb, _ in pairs:
            endpoints.append(sa)
            endpoints.append(sb)
        unique_order = list(dict.fromkeys(endpoints))

        run_summaries: List[Dict[str, Any]] = []
        fig, axes = plt.subplots(
            nrows,
            fig_ncols,
            figsize=(ws, hs * max(1, nrows)),
            squeeze=False,
        )

        for idx, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            r = _resolve_run(encoding_defaults, run)
            run_id = r.get("run_id") or f"run_{idx}"
            model_path = r.get("model_name_or_path")
            if not model_path:
                print(f"[{dataset_id}] run {run_id} 缺少 model_name_or_path", file=sys.stderr)
                sys.exit(1)
            mp = (
                model_path
                if os.path.isabs(model_path)
                else os.path.join(_SCRIPT_DIR, model_path)
            )
            if not os.path.isdir(mp):
                print(f"找不到 checkpoint 目录: {mp}", file=sys.stderr)
                sys.exit(1)

            mask_emb = bool(r.get("mask_embedding_sentence", False))
            template = r.get("mask_embedding_sentence_template")
            if mask_emb and not template:
                print(
                    f"[{dataset_id}] run {run_id}: mask 模式需要 mask_embedding_sentence_template",
                    file=sys.stderr,
                )
                sys.exit(1)

            mask_num = int(r.get("mask_num", 2))
            pooler = str(r.get("pooler", "cls"))
            org_mlp = bool(r.get("mask_embedding_sentence_org_mlp", False))
            use_org_pooler = bool(
                r.get("mask_embedding_sentence_use_org_pooler", False)
            )
            use_pooler = bool(r.get("mask_embedding_sentence_use_pooler", False))
            use_delta = bool(r.get("mask_embedding_sentence_delta", False))

            need_mlp = org_mlp or use_org_pooler
            mlp_mod = None
            if need_mlp:
                mlp_mod = load_mask_prediction_mlp_for_gold_scatter(mp, device)

            print(f"[{dataset_id}] --- Run: {run_id} @ {mp} ---")
            tokenizer = AutoTokenizer.from_pretrained(mp, use_fast=True)
            model = AutoModel.from_pretrained(mp)
            model.eval()
            model.to(device)

            delta_tensor = None
            template_len = None
            if use_delta:
                if not (mask_emb and template):
                    print(
                        f"[{dataset_id}] run {run_id}: delta 需要 mask + template",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                noise, template_len = denoising(
                    model, template, tokenizer, device, mask_num
                )
                delta_tensor = noise

            batches = build_batches(unique_order, batch_size)
            mat = encode_word_batches(
                model,
                tokenizer,
                device,
                batches,
                mask_embedding_sentence=mask_emb,
                mask_embedding_sentence_template=template,
                mask_num=mask_num,
                pooler=pooler,
                delta_tensor=delta_tensor,
                template_len=template_len,
                mlp=mlp_mod,
                mask_embedding_sentence_org_mlp=org_mlp,
                mask_embedding_sentence_use_org_pooler=use_org_pooler,
                mask_embedding_sentence_use_pooler=use_pooler,
            )
            emb_map: Dict[str, np.ndarray] = {
                s: mat[i] for i, s in enumerate(unique_order)
            }

            gold_arr, cos_arr = _cosine_rows(emb_map, pairs, cosine_to_unit)
            rho = _pearson(gold_arr, cos_arr)
            run_summaries.append(
                {
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "model_name_or_path": mp,
                    "pearson_gold_cos": rho,
                    "n_pairs": len(pairs),
                    "cosine_to_unit_interval": cosine_to_unit,
                }
            )

            for i in range(len(pairs)):
                all_rows_csv.append(
                    {
                        "dataset_id": dataset_id,
                        "run_id": run_id,
                        "gold": gold_arr[i],
                        "cos_pred": cos_arr[i],
                        "sentence_a": pairs[i][0],
                        "sentence_b": pairs[i][1],
                    }
                )

            ax = axes[idx // fig_ncols][idx % fig_ncols]
            ax.scatter(gold_arr, cos_arr, s=8, alpha=0.35)
            ax.set_title(run_id)
            ax.set_xlabel("Gold score")
            ax.set_ylabel(
                "Cosine" + (" (0–1)" if cosine_to_unit else " ([−1,1]→scaled)")
            )
            ax.grid(True, alpha=0.3)

        for j in range(n_runs, nrows * fig_ncols):
            axes[j // fig_ncols][j % fig_ncols].set_visible(False)

        fig.suptitle(f"dataset: {dataset_id}", fontsize=11, y=1.02)
        fig.tight_layout()
        fig_path = _figure_path_for_dataset(
            out_dir, fig_name_tpl, dataset_id, multi_ds
        )
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Wrote figure {fig_path}")

        dataset_records.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_cfg,
                "runs": run_summaries,
                "figure": fig_path,
            }
        )

    csv_name = cfg.get("output_csv") or "points.csv"
    csv_path = os.path.join(out_dir, csv_name)
    if all_rows_csv:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows_csv[0].keys()))
            w.writeheader()
            w.writerows(all_rows_csv)
        print(f"Wrote {csv_path}")
    else:
        csv_path = ""

    json_name = cfg.get("output_json") or "run_summary.json"
    json_path = os.path.join(out_dir, json_name)
    record = {
        "datasets": dataset_records,
        "csv": csv_path or None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
