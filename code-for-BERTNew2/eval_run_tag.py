#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估输出文件名与 JSON 中统一的 eval_run_tag。

读取优先级（resolve_eval_run_tag）：
1. 环境变量 COT_EVAL_RUN_TAG（非空则原样使用）
2. model_dir/train_config_full.json 顶层 cot_training_meta.eval_output_tag（训练保存配置时写入）
3. 同文件 training_args.logging_dir 的 basename（旧 checkpoint）
4. model_dir/trainer_state.json 的 mtime → %Y%m%d-%H%M%S
5. model_dir 目录 mtime
6. 墙钟 %Y%m%d-%H%M%S

训练端在 cot_bert_train.py 写入 train_config_full.json 时会附带 cot_training_meta；
也可用 export COT_EVAL_RUN_TAG=my_tag 覆盖一切。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

_ENV_KEY = "COT_EVAL_RUN_TAG"
_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}

_UNSAFE_TAG_CHARS = re.compile(r'[\\/:*?"<>|\s]+')


def _cache_key(model_dir: Optional[str]) -> tuple[str, str]:
    if isinstance(model_dir, str) and model_dir:
        exp = os.path.expanduser(model_dir)
        if os.path.isdir(exp):
            return (os.path.abspath(exp), os.environ.get(_ENV_KEY, ""))
    return (str(model_dir or ""), os.environ.get(_ENV_KEY, ""))


def _optional_sanitize_tag(tag: str) -> str:
    if not _UNSAFE_TAG_CHARS.search(tag):
        return tag
    return _UNSAFE_TAG_CHARS.sub("_", tag).strip("_") or "run"


def resolve_eval_run_tag(model_dir: Optional[str]) -> Dict[str, Any]:
    """
    返回 dict: tag (str), source (str), train_logging_dir (Optional[str]),
    training_saved_at_iso (Optional[str]).
    """
    ck = _cache_key(model_dir)
    if ck in _CACHE:
        return dict(_CACHE[ck])

    env_tag = os.environ.get(_ENV_KEY, "").strip()
    if env_tag:
        out = {
            "tag": _optional_sanitize_tag(env_tag),
            "source": "env",
            "train_logging_dir": None,
            "training_saved_at_iso": None,
        }
        _CACHE[ck] = out
        return dict(out)

    tag: Optional[str] = None
    source: Optional[str] = None
    train_logging_dir: Optional[str] = None
    training_saved_at_iso: Optional[str] = None

    if isinstance(model_dir, str) and model_dir:
        exp = os.path.expanduser(model_dir)
        if os.path.isdir(exp):
            abspath = os.path.abspath(exp)
            cfg_path = os.path.join(abspath, "train_config_full.json")
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    meta = cfg.get("cot_training_meta") or {}
                    raw_meta_tag = meta.get("eval_output_tag")
                    if isinstance(raw_meta_tag, str) and raw_meta_tag.strip():
                        tag = raw_meta_tag.strip()
                        source = "train_config_meta"
                    iso = meta.get("saved_at_iso")
                    if isinstance(iso, str) and iso.strip():
                        training_saved_at_iso = iso.strip()
                    ta = cfg.get("training_args") or {}
                    raw_ld = ta.get("logging_dir")
                    if raw_ld is not None:
                        train_logging_dir = str(raw_ld)
                    if tag is None and raw_ld is not None:
                        base = os.path.basename(str(raw_ld).rstrip("/\\"))
                        if base:
                            tag = base
                            source = "train_logging_run"
                except (OSError, ValueError, TypeError):
                    pass

            if tag is None:
                tsp = os.path.join(abspath, "trainer_state.json")
                if os.path.isfile(tsp):
                    mt = os.path.getmtime(tsp)
                    tag = datetime.fromtimestamp(mt).strftime("%Y%m%d-%H%M%S")
                    source = "trainer_state_mtime"
                else:
                    mt = os.path.getmtime(abspath)
                    tag = datetime.fromtimestamp(mt).strftime("%Y%m%d-%H%M%S")
                    source = "model_dir_mtime"

    if tag is None:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        source = "wall_clock"

    tag = _optional_sanitize_tag(tag)
    out = {
        "tag": tag,
        "source": source,
        "train_logging_dir": train_logging_dir,
        "training_saved_at_iso": training_saved_at_iso,
    }
    _CACHE[ck] = out
    return dict(out)


def get_eval_run_tag(model_dir: Optional[str]) -> str:
    return str(resolve_eval_run_tag(model_dir)["tag"])


def insert_run_tag_before_ext(path: str, tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return path
    parent, base = os.path.split(path)
    stem, ext = os.path.splitext(base)
    new_base = f"{stem}_{tag}{ext}"
    return os.path.join(parent, new_base) if parent else new_base
