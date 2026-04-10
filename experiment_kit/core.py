#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练元数据、评估 JSON 头部字段等小工具（与 CoT-BERT 布局一致）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterator, Optional


def build_training_run_meta() -> Dict[str, Any]:
    """训练保存时写入 checkpoint 旁 train_config_full.json 的 cot_training_meta 片段基座。"""
    return {"saved_at_iso": datetime.now().isoformat()}


def load_run_meta_from_training_dir(
    model_dir: Optional[str],
    *,
    config_filename: str = "train_config_full.json",
    meta_key: str = "cot_training_meta",
) -> Dict[str, Any]:
    """
    从 ``<model_dir>/<config_filename>`` 读取 ``meta_key`` 指向的字典（缺文件或非 dict 则返回空 dict）。
    """
    if not model_dir or not str(model_dir).strip():
        return {}
    root = os.path.abspath(os.path.expanduser(str(model_dir).strip()))
    if not os.path.isdir(root):
        return {}
    path = os.path.join(root, config_filename)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return {}
        raw = cfg.get(meta_key)
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def eval_payload_with_headers(
    payload: Dict[str, Any],
    train_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将 ``train_meta``（通常为 cot_training_meta）并入评估输出 payload，并设置 ``experiment_id``（若可得）。
    """
    out = dict(payload) if isinstance(payload, dict) else {}
    meta = train_meta if isinstance(train_meta, dict) else {}
    if meta:
        out["cot_training_meta"] = dict(meta)
    eid = None
    if meta:
        for k in ("experiment_id", "eval_output_tag", "run_id"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                eid = v.strip()
                break
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                eid = str(v)
                break
    if eid:
        out["experiment_id"] = eid
    return out


def iter_csv_meta_comment_lines(path: str) -> Iterator[str]:
    """遍历文本文件中以 ``#`` 开头的行（常见于带元数据注释的 CSV）。"""
    if not path or not os.path.isfile(path):
        yield from ()
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                yield line.rstrip("\n\r")
