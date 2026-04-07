#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型 × 多数据集 Alignment / Uniformity 对比（Wang & Isola 风格，与 sick_r_alignment_uniformity 编码一致）。

配置为嵌套 YAML（非 parse_args_util 的扁平键）；默认文件 + 可选第一个参数为覆盖 yaml（深度合并）。

  python run_alignment_uniformity_benchmark.py
  python run_alignment_uniformity_benchmark.py configs/my_benchmark.yaml

说明：
- cls_before_pooler 使用 BERT 特殊 id 101/102；RoBERTa 等请用 cls/avg。
- 各数据集单独 pos_threshold；SICK 与 STS-B 均为「分数 >= 阈值」参与 Alignment。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer

import alignment_uniformity_lib as aul
from cot_bert_evaluation import denoising

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCHMARK_YAML = os.path.join(
    _SCRIPT_DIR, "configs", "alignment_uniformity_benchmark_default.yaml"
)


def _merge_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            _merge_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


def _load_benchmark_config(default_file: str, custom_file: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if os.path.isfile(default_file):
        with open(default_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                cfg = loaded
    if custom_file and os.path.isfile(custom_file):
        with open(custom_file, "r", encoding="utf-8") as f:
            custom = yaml.safe_load(f)
            if isinstance(custom, dict):
                cfg = _merge_dict(dict(cfg), custom)
    return cfg


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


def _flatten_defaults(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """将 defaults 节展平到与模型条目同级的键（供合并）。"""
    return dict(defaults) if defaults else {}


def _effective_model_config(
    defaults_flat: Dict[str, Any], model_entry: Dict[str, Any]
) -> Dict[str, Any]:
    m = dict(defaults_flat)
    for k, v in model_entry.items():
        if v is not None:
            m[k] = v
    return m


def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def run_one(
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    device,
) -> Dict[str, Any]:
    ds_id = dataset_cfg.get("id", dataset_cfg.get("path"))
    ds_type = dataset_cfg["type"]
    ds_path = dataset_cfg["path"]
    pos_threshold = float(dataset_cfg["pos_threshold"])

    model_id = model_cfg.get("id", model_cfg.get("model_name_or_path"))
    model_path = model_cfg["model_name_or_path"]

    pairs, pair_endpoints = aul.load_pair_dataset(ds_path, ds_type)
    unique_order = list(dict.fromkeys(pair_endpoints))

    mask_emb = _bool(model_cfg.get("mask_embedding_sentence"))
    template = model_cfg.get("mask_embedding_sentence_template")
    if template is not None and str(template).strip() == "":
        template = None
    mask_num = int(model_cfg.get("mask_num", 2))
    pooler = str(model_cfg.get("pooler", "cls"))
    delta_flag = _bool(model_cfg.get("mask_embedding_sentence_delta"))
    org_mlp = _bool(model_cfg.get("mask_embedding_sentence_org_mlp"))
    use_org_pooler = _bool(model_cfg.get("mask_embedding_sentence_use_org_pooler"))
    use_pooler = _bool(model_cfg.get("mask_embedding_sentence_use_pooler"))
    batch_size = int(model_cfg.get("batch_size", 64))
    seed = int(model_cfg.get("seed", 42))
    uniformity_pairs = int(model_cfg.get("uniformity_pairs", 100_000))
    l2_norm = _bool(model_cfg.get("l2_normalize"))

    if mask_emb and not template:
        raise ValueError(
            f"模型 {model_id!r} 启用了 mask_embedding_sentence 但未提供 mask_embedding_sentence_template"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    model.to(device)

    mlp_mod = None
    if org_mlp or use_org_pooler:
        mlp_mod = aul.load_mask_prediction_mlp(model_path, device)

    delta_tensor = None
    template_len = None
    if delta_flag:
        if not (mask_emb and template):
            raise ValueError(f"模型 {model_id!r}: mask_embedding_sentence_delta 需要 mask 模板")
        noise, template_len = denoising(
            model, template, tokenizer, device, mask_num
        )
        delta_tensor = noise

    batches = aul.build_batches(unique_order, batch_size)
    mat = aul.encode_word_batches(
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
        tqdm_disable=True,
    )

    emb_map = {s: mat[i] for i, s in enumerate(unique_order)}
    align, n_pos = aul.compute_alignment(
        emb_map, pairs, pos_threshold, l2_norm
    )
    slot_matrix = np.stack([emb_map[s] for s in pair_endpoints], axis=0)
    unif = aul.compute_uniformity(
        slot_matrix, uniformity_pairs, seed, l2_norm
    )

    return {
        "model_id": model_id,
        "dataset_id": ds_id,
        "dataset_type": ds_type,
        "dataset_path": os.path.abspath(ds_path),
        "model_name_or_path": model_path,
        "pos_threshold": pos_threshold,
        "n_positive_pairs": n_pos,
        "alignment": align,
        "uniformity": unif,
        "uniformity_M": uniformity_pairs,
        "n_uniformity_slots": len(pair_endpoints),
        "n_distinct_strings_encoded": len(unique_order),
        "seed": seed,
        "l2_normalize": l2_norm,
        "uniformity_sampling": "pair_endpoints_2P_multiset",
    }


def main() -> None:
    argv = sys.argv[1:]
    dry_parser = argparse.ArgumentParser(
        description="Alignment / Uniformity 多模型多数据集对比",
        add_help=True,
    )
    dry_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将加载的配置，不跑模型",
    )
    args_cli, rest = dry_parser.parse_known_args(argv)
    custom_yaml, _ = _split_config_argv(rest)
    print(f"默认配置: {DEFAULT_BENCHMARK_YAML}")
    if custom_yaml:
        print(f"覆盖配置: {custom_yaml}")

    cfg = _load_benchmark_config(DEFAULT_BENCHMARK_YAML, custom_yaml)
    defaults_section = cfg.get("defaults") or {}
    defaults_flat = _flatten_defaults(defaults_section)
    datasets: List[Dict[str, Any]] = cfg.get("datasets") or []
    models: List[Dict[str, Any]] = cfg.get("models") or []

    if not datasets:
        print("错误: 配置中 datasets 为空", file=sys.stderr)
        sys.exit(1)
    if not models:
        print("错误: 配置中 models 为空", file=sys.stderr)
        sys.exit(1)

    if args_cli.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    device = aul.pick_device()
    print(f"Device: {device}")

    results: List[Dict[str, Any]] = []
    summary_matrix: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ds in datasets:
        ds_key = str(ds.get("id", ds.get("path")))
        summary_matrix[ds_key] = {}
        for m_entry in models:
            eff = _effective_model_config(defaults_flat, m_entry)
            mid = str(m_entry.get("id", m_entry.get("model_name_or_path")))
            print(
                f"\n>>> {mid}  @  {ds_key} ({ds.get('type')}) ...",
                flush=True,
            )
            try:
                row = run_one(eff, ds, device)
                results.append(row)
                summary_matrix[ds_key][mid] = {
                    "alignment": row["alignment"],
                    "uniformity": row["uniformity"],
                }
            except FileNotFoundError as e:
                print(f"跳过: {e}", file=sys.stderr)
                results.append(
                    {
                        "model_id": mid,
                        "dataset_id": ds_key,
                        "error": str(e),
                    }
                )
            except Exception as e:
                print(f"失败 {mid} / {ds_key}: {e}", file=sys.stderr)
                results.append(
                    {
                        "model_id": mid,
                        "dataset_id": ds_key,
                        "error": repr(e),
                    }
                )

    tb = PrettyTable()
    tb.field_names = [
        "model_id",
        "dataset_id",
        "alignment",
        "uniformity",
        "n_pos",
        "n_slots",
    ]
    for r in results:
        if "error" in r:
            tb.add_row(
                [
                    r.get("model_id", ""),
                    r.get("dataset_id", ""),
                    "ERR",
                    "ERR",
                    "-",
                    "-",
                ]
            )
        else:
            tb.add_row(
                [
                    r["model_id"],
                    r["dataset_id"],
                    f"{r['alignment']:.6f}",
                    f"{r['uniformity']:.6f}",
                    r["n_positive_pairs"],
                    r["n_uniformity_slots"],
                ]
            )
    print("\n--- Alignment / Uniformity 汇总（越小越好）---")
    print(tb)

    out_path = defaults_flat.get("output_json")
    if out_path in (None, "", "None"):
        out_path = None
    if out_path:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "defaults": defaults_section,
            "datasets": datasets,
            "model_entries": models,
            "results": results,
            "summary_matrix": summary_matrix,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
