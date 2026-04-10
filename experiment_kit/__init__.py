#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_kit — CoT-BERT 实验元数据与评估产物 Git 同步。

将 CoT-BERT 根目录加入 PYTHONPATH 后（例如 ``code-for-BERTNew2`` 下
``sys.path.insert(0, os.path.join(_SCRIPT_DIR, \"..\"))``），即可 ``import experiment_kit``。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from experiment_kit.core import (
    build_training_run_meta,
    eval_payload_with_headers,
    iter_csv_meta_comment_lines,
    load_run_meta_from_training_dir,
)
from experiment_kit.git_sync import (
    GitPushSettings,
    push_eval_artifacts_to_git,
    resolve_git_push_settings,
    resolve_project_git_push_config_path,
    should_push_eval_results,
)

push_artifacts_to_git = push_eval_artifacts_to_git

COT_DEFAULT_TRAIN_CONFIG_BASENAME = "train_config_full.json"
COT_DEFAULT_TRAIN_META_KEY = "cot_training_meta"


def load_training_meta(model_dir: Optional[str]) -> Dict[str, Any]:
    """读取 checkpoint 目录下 train_config_full.json 中的 cot_training_meta。"""
    return load_run_meta_from_training_dir(
        model_dir,
        config_filename=COT_DEFAULT_TRAIN_CONFIG_BASENAME,
        meta_key=COT_DEFAULT_TRAIN_META_KEY,
    )


def build_cot_training_meta_for_save() -> Dict[str, Any]:
    """与 build_training_run_meta 相同；保留旧名称以兼容训练脚本。"""
    return build_training_run_meta()


__all__ = [
    "build_training_run_meta",
    "build_cot_training_meta_for_save",
    "load_run_meta_from_training_dir",
    "load_training_meta",
    "eval_payload_with_headers",
    "iter_csv_meta_comment_lines",
    "GitPushSettings",
    "resolve_git_push_settings",
    "resolve_project_git_push_config_path",
    "should_push_eval_results",
    "push_eval_artifacts_to_git",
    "push_artifacts_to_git",
    "COT_DEFAULT_TRAIN_CONFIG_BASENAME",
    "COT_DEFAULT_TRAIN_META_KEY",
]

