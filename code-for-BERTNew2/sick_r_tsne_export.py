#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SICK-R 测试集：随机抽样固定数量句对 → 各消融 checkpoint 编码句向量 → 逐 run 拟合 t-SNE → 导出 CSV / JSON。

- 编码与 pooler / mask 模板逻辑与 gold_cosine_scatter.py 一致（复用 load_dataset_pairs、encode_word_batches 等）。
- 每个 run 在**该 run 的句向量**上单独 fit t-SNE（坐标不可跨 run 直接叠在同一张平面比较尺度，适合分面图）。
- `tsne.n_components` 默认 3：坐标 CSV 含 `tsne_x` / `tsne_y` / `tsne_z`；设为 2 时与旧版一致仅两列。
- 用法（在 code-for-BERTNew2 目录下）:
    python sick_r_tsne_export.py
    python sick_r_tsne_export.py configs/sick_r_tsne_ablation_linux_cuda.yaml

依赖: torch, transformers, numpy, scikit-learn（pip install scikit-learn）。
写盘结束后会调用 experiment_kit.git_sync.push_eval_artifacts_to_git（与 gold_cosine_scatter.py 一致）。
"""

from __future__ import annotations

import csv
import gc
import inspect
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cot_bert_evaluation import denoising
from eval_run_tag import resolve_eval_run_tag
from gold_cosine_scatter import (
    _filename_with_timestamp,
    _resolve_run,
    _split_config_argv,
    iter_dataset_specs,
    load_dataset_pairs,
    load_mask_prediction_mlp_for_gold_scatter,
    load_yaml_config,
)
from sick_r_alignment_uniformity import build_batches, encode_word_batches, pick_device

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "sick_r_tsne_ablation_default.yaml")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _pick_dataset_cfg(
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    specs = iter_dataset_specs(cfg)
    if not specs:
        raise ValueError("配置缺少 datasets 或 dataset")
    want = str(cfg.get("dataset_id_for_tsne") or "sick_test").strip()
    for did, block in specs:
        if did == want:
            return did, block
    # 回退：仅一项时用第一项
    if len(specs) == 1:
        return specs[0]
    raise ValueError(
        f"未找到 dataset_id_for_tsne={want!r}，当前有: {[d for d, _ in specs]}"
    )


def _sample_pairs(
    pairs: List[Tuple[str, str, float]],
    n_pairs: int,
    seed: int,
) -> List[Tuple[str, str, float]]:
    n = len(pairs)
    if n_pairs > n:
        raise ValueError(f"n_pairs={n_pairs} 大于全集 {n}")
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n_pairs, replace=False)
    idx_sorted = np.sort(idx)
    return [pairs[i] for i in idx_sorted]


def _unique_sentence_order(
    sampled: List[Tuple[str, str, float]],
) -> Tuple[List[str], Dict[str, int]]:
    endpoints: List[str] = []
    for sa, sb, _ in sampled:
        endpoints.append(sa)
        endpoints.append(sb)
    unique_order = list(dict.fromkeys(endpoints))
    uid_map = {s: i for i, s in enumerate(unique_order)}
    return unique_order, uid_map


def _effective_tsne_perplexity(n_samples: int, requested: float) -> float:
    if n_samples < 2:
        raise ValueError("t-SNE 至少需要 2 个样本点")
    # sklearn: perplexity < n_samples
    cap = max(2.0, float(min(n_samples - 1, requested)))
    # 经验上略小于 n_samples/3 更稳
    third = max(2.0, (n_samples - 1) / 3.0)
    return float(min(cap, third))


def _tsne_fit_transform(
    TSNE: Any,
    X: np.ndarray,
    *,
    random_state: int,
    perplexity: float,
    learning_rate: Any,
    max_iter: int,
    n_components: int,
) -> np.ndarray:
    """
    兼容旧版 scikit-learn：老版本 TSNE 使用 n_iter 而非 max_iter，
    且 learning_rate='auto' 仅在较新版本可用。
    """
    sig = inspect.signature(TSNE.__init__)
    pnames = set(sig.parameters.keys()) - {"self", "kwargs"}

    iter_kw: Dict[str, int] = {}
    if "max_iter" in pnames:
        iter_kw["max_iter"] = max_iter
    elif "n_iter" in pnames:
        iter_kw["n_iter"] = max_iter

    lr_try: List[Any] = (
        [learning_rate, 200.0] if learning_rate == "auto" else [learning_rate]
    )
    init_try: List[Optional[str]] = (
        ["pca", "random"] if "init" in pnames else [None]
    )

    last_err: Optional[Exception] = None
    for init in init_try:
        for lr in lr_try:
            if lr == "auto" and "learning_rate" not in pnames:
                continue
            kw: Dict[str, Any] = {
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                **iter_kw,
            }
            if init is not None:
                kw["init"] = init
            if "learning_rate" in pnames:
                kw["learning_rate"] = lr
            fk = {k: v for k, v in kw.items() if k in pnames}
            try:
                return TSNE(**fk).fit_transform(X)
            except (TypeError, ValueError) as e:
                last_err = e

    if last_err:
        raise last_err
    raise RuntimeError("t-SNE: 当前 scikit-learn 版本下无法构造 TSNE，请升级 sklearn 或检查参数")


def _flush_torch_device_caches() -> None:
    """在已 del 模型等对象后调用，回收 MPS/CUDA 显存，降低随后 CPU 大数组分配触发 SIGBUS 的概率。"""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main() -> None:
    import torch
    from sklearn.manifold import TSNE
    from transformers import AutoModel, AutoTokenizer

    custom_yaml, _ = _split_config_argv(sys.argv[1:])
    print(f"默认配置: {DEFAULT_CONFIG_PATH}")
    if custom_yaml:
        print(f"覆盖配置: {custom_yaml}")
    cfg = load_yaml_config(DEFAULT_CONFIG_PATH, custom_yaml)

    runs = cfg.get("runs")
    if not runs or not isinstance(runs, list):
        print("配置缺少 runs 列表", file=sys.stderr)
        sys.exit(1)

    encoding_defaults = cfg.get("encoding_defaults") or {}
    n_pairs = int(cfg.get("n_pairs", 200))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 64))
    text_max_chars = int(cfg.get("sentence_text_max_chars", 200))

    tsne_cfg = cfg.get("tsne") if isinstance(cfg.get("tsne"), dict) else {}
    perplexity_req = float(tsne_cfg.get("perplexity", 30))
    tsne_learning_rate = tsne_cfg.get("learning_rate", "auto")
    tsne_max_iter = int(tsne_cfg.get("max_iter", 1000))
    tsne_n_components = int(tsne_cfg.get("n_components", 3))
    if tsne_n_components < 2:
        raise ValueError("tsne.n_components 须 >= 2（t-SNE 输出维度）")

    dataset_id, dataset_cfg = _pick_dataset_cfg(cfg)
    all_pairs = load_dataset_pairs(dataset_cfg)
    sampled = _sample_pairs(all_pairs, n_pairs, seed)
    unique_order, uid_map = _unique_sentence_order(sampled)
    n_unique = len(unique_order)

    first_model_dir: Optional[str] = None
    for run in runs:
        if not isinstance(run, dict):
            continue
        r0 = _resolve_run(encoding_defaults, run)
        mp0 = r0.get("model_name_or_path")
        if mp0:
            first_model_dir = (
                mp0 if os.path.isabs(mp0) else os.path.join(_SCRIPT_DIR, mp0)
            )
            break
    run_meta = resolve_eval_run_tag(first_model_dir)
    run_ts = run_meta["tag"]

    out_dir = cfg.get("output_dir") or "./eval_results/sick_r_tsne"
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_SCRIPT_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pairs_csv_name = _filename_with_timestamp(
        str(cfg.get("pairs_csv") or "tsne_sick_r_pairs.csv"), run_ts
    )
    coords_csv_name = _filename_with_timestamp(
        str(cfg.get("coords_csv") or "tsne_sick_r_coords.csv"), run_ts
    )
    meta_json_name = _filename_with_timestamp(
        str(cfg.get("meta_json") or "tsne_sick_r_meta.json"), run_ts
    )
    pairs_path = os.path.join(out_dir, pairs_csv_name)
    coords_path = os.path.join(out_dir, coords_csv_name)
    meta_path = os.path.join(out_dir, meta_json_name)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- 写抽样句对表 ---
    with open(pairs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pair_id", "sentence_a", "sentence_b", "relatedness_score"],
        )
        w.writeheader()
        for i, (sa, sb, g) in enumerate(sampled):
            w.writerow(
                {
                    "pair_id": i,
                    "sentence_a": sa,
                    "sentence_b": sb,
                    "relatedness_score": g,
                }
            )
    print(f"Wrote {pairs_path}")

    device = pick_device()
    print(f"Device: {device}")

    coords_rows: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []

    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        r = _resolve_run(encoding_defaults, run)
        run_id = r.get("run_id") or f"run_{idx}"
        model_path = r.get("model_name_or_path")
        if not model_path:
            print(f"run {run_id} 缺少 model_name_or_path", file=sys.stderr)
            sys.exit(1)
        mp = model_path if os.path.isabs(model_path) else os.path.join(_SCRIPT_DIR, model_path)
        if not os.path.isdir(mp):
            print(f"找不到 checkpoint 目录: {mp}", file=sys.stderr)
            sys.exit(1)

        mask_emb = bool(r.get("mask_embedding_sentence", False))
        template = r.get("mask_embedding_sentence_template")
        if mask_emb and not template:
            print(f"run {run_id}: mask 模式需要 mask_embedding_sentence_template", file=sys.stderr)
            sys.exit(1)

        mask_num = int(r.get("mask_num", 2))
        pooler = str(r.get("pooler", "cls"))
        org_mlp = bool(r.get("mask_embedding_sentence_org_mlp", False))
        use_org_pooler = bool(r.get("mask_embedding_sentence_use_org_pooler", False))
        use_pooler = bool(r.get("mask_embedding_sentence_use_pooler", False))
        use_delta = bool(r.get("mask_embedding_sentence_delta", False))

        need_mlp = org_mlp or use_org_pooler
        mlp_mod = None
        if need_mlp:
            mlp_mod = load_mask_prediction_mlp_for_gold_scatter(mp, device)

        print(f"--- Run: {run_id} @ {mp} ---")
        tokenizer = AutoTokenizer.from_pretrained(mp, use_fast=True)
        model = AutoModel.from_pretrained(mp)
        model.eval()
        model.to(device)

        delta_tensor = None
        template_len = None
        if use_delta:
            if not (mask_emb and template):
                print(f"run {run_id}: delta 需要 mask + template", file=sys.stderr)
                sys.exit(1)
            noise, template_len = denoising(model, template, tokenizer, device, mask_num)
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
        # float32 降低 t-SNE 峰值内存；先释放编码器再跑 CPU t-SNE，减轻 Apple MPS 上 SIGBUS 风险
        X = np.asarray(mat, dtype=np.float32)
        del mat
        del model, tokenizer
        if mlp_mod is not None:
            del mlp_mod
        if delta_tensor is not None:
            del delta_tensor
        _flush_torch_device_caches()

        perp = _effective_tsne_perplexity(n_unique, perplexity_req)
        xy = _tsne_fit_transform(
            TSNE,
            X,
            random_state=seed,
            perplexity=perp,
            learning_rate=tsne_learning_rate,
            max_iter=tsne_max_iter,
            n_components=tsne_n_components,
        )
        run_summaries.append(
            {
                "run_id": run_id,
                "model_name_or_path": mp,
                "n_unique_sentences": n_unique,
                "tsne_perplexity_used": perp,
                "embedding_dim": int(X.shape[1]),
            }
        )

        for pair_id, (sa, sb, g) in enumerate(sampled):
            for side, sent in (("a", sa), ("b", sb)):
                suid = uid_map[sent]
                text = sent if len(sent) <= text_max_chars else sent[: text_max_chars - 1] + "…"
                row_out: Dict[str, Any] = {
                    "run_id": run_id,
                    "sentence_uid": suid,
                    "pair_id": pair_id,
                    "side": side,
                    "relatedness_score": g,
                    "tsne_x": float(xy[suid, 0]),
                    "tsne_y": float(xy[suid, 1]),
                    "sentence_text": text,
                }
                if tsne_n_components >= 3:
                    row_out["tsne_z"] = float(xy[suid, 2])
                coords_rows.append(row_out)

    _coord_fields = [
        "run_id",
        "sentence_uid",
        "pair_id",
        "side",
        "relatedness_score",
        "tsne_x",
        "tsne_y",
    ]
    if tsne_n_components >= 3:
        _coord_fields.append("tsne_z")
    _coord_fields.append("sentence_text")
    with open(coords_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_coord_fields)
        if coords_rows:
            w.writeheader()
            w.writerows(coords_rows)
    print(f"Wrote {coords_path}")

    ds_path = dataset_cfg.get("path")
    if ds_path and not os.path.isabs(str(ds_path)):
        ds_path = os.path.abspath(os.path.join(_SCRIPT_DIR, str(ds_path)))
    else:
        ds_path = str(ds_path) if ds_path else ""

    meta = {
        "dataset_id": dataset_id,
        "dataset_path": ds_path,
        "n_pairs_sampled": n_pairs,
        "n_pairs_available": len(all_pairs),
        "n_unique_sentences": n_unique,
        "seed": seed,
        "eval_run_tag": run_meta["tag"],
        "eval_run_tag_source": run_meta["source"],
        "training_saved_at_iso": run_meta.get("training_saved_at_iso"),
        "train_logging_dir": run_meta.get("train_logging_dir"),
        "export_started_at_iso": datetime.now().isoformat(),
        "pairs_csv": pairs_path,
        "coords_csv": coords_path,
        "tsne_requested_perplexity": perplexity_req,
        "tsne_n_components": tsne_n_components,
        "runs": run_summaries,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote {meta_path}")

    _cot_root = os.path.dirname(_SCRIPT_DIR)
    if _cot_root not in sys.path:
        sys.path.insert(0, _cot_root)
    from experiment_kit.git_sync import push_eval_artifacts_to_git

    push_paths: List[str] = [meta_path, pairs_path, coords_path]
    push_eval_artifacts_to_git(
        push_paths,
        experiment_id=run_ts,
        project_root_for_git_config=_SCRIPT_DIR,
    )


if __name__ == "__main__":
    main()
