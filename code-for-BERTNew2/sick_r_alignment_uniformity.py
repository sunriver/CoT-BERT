#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 SICK-R 测试集上计算 Wang & Isola 风格的 Alignment 与 Uniformity（与 paper.md 第 7.1 节描述一致）。

- Alignment: 相关度 score >= pos_threshold 的句子对上，mean ||f(s1)-f(s2)||^2（越小越好）
- Uniformity: 在**测试集每条句对的两个端点按文件顺序展开**的序列上抽样（长度 2P，同一句多次出现则多次计入抽样空间），
  即下标 i,j 均匀取自 {0..2P-1}，对应句子 s_i,s_j，估计 log(mean(exp(-2*||f(s_i)-f(s_j)||^2)))（越小越好）。
  编码仍对每个**不同字符串**只做一次前向，再按槽位复制向量，与 SentEval 句向量一致且省算力。

denoising 从 cot_bert_evaluation 导入；前向与 pooler 分支与 cot_bert_evaluation.batcher 对齐（mask / cls / avg / avg_first_last / cls_before_pooler / avg_top2）。

编码方式尽量与 cot_bert_evaluation.py 的 batcher 一致：
  - CoT-BERT / MV-RaCL: --mask_embedding_sentence + 模板 + 可选 --mask_embedding_sentence_delta
  - 可选与 eval 相同: --mask_embedding_sentence_org_mlp、--mask_embedding_sentence_use_org_pooler、--mask_embedding_sentence_use_pooler
  - SimCSE 风格: --pooler cls 且不加 mask 模板

配置：默认读取脚本同目录下 configs/sick_r_alignment_uniformity_default.yaml（与 parse_args_util.load_configs 一致）。
  第一个参数若为 .yaml/.yml 且文件存在，则作为覆盖配置与 default 深度合并；其后可再接命令行参数覆盖。

  python sick_r_alignment_uniformity.py
  python sick_r_alignment_uniformity.py configs/sick_r_alignment_uniformity_linux_cuda.yaml
  python sick_r_alignment_uniformity.py my.yaml --seed 123 --uniformity_pairs 50000

纯命令行（不用 yaml）:
  python sick_r_alignment_uniformity.py \\
    --model_name_or_path /path/to/ckpt \\
    --sick_test ../SentEval/data/downstream/SICK/SICK_test_annotated.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from transformers import AutoModel, AutoTokenizer

# 与 cot_bert_evaluation 共用实现，避免两处漂移（导入该模块会执行其全局初始化，如平台信息打印）。
from cot_bert_evaluation import denoising
from parse_args_util import load_configs

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SICK_CONFIG = os.path.join(
    _SCRIPT_DIR, "configs", "sick_r_alignment_uniformity_default.yaml"
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sent_to_word_list(s: str) -> List[str]:
    return s.strip().split()


def apply_mask_template(
    tokenizer, template_raw: str, sentence: str
) -> str:
    """与 cot_bert_evaluation.batcher 中模板逻辑一致（*sent_0* -> *sent 0*）。"""
    s = sentence
    if len(s) > 0 and s[-1] not in ".?\"'":
        s += "."
    template = (
        template_raw.replace("*mask*", tokenizer.mask_token)
        .replace("_", " ")
        .replace("*sep+*", "")
        .replace("*cls*", "")
    )
    return template.replace("*sent 0*", s).strip()


def load_mask_prediction_mlp(model_name_or_path: str, device: torch.device) -> nn.Module:
    """
    与 cot_bert_evaluation 中 mask_embedding_sentence_org_mlp 一致：
    BERT MLM 的 cls.predictions.transform，若 checkpoint 路径含 result 则从 pooler 权重映射加载。
    """
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig.from_pretrained("bert-base-uncased")
    mlp = BertForMaskedLM.from_pretrained(
        "bert-base-uncased", config=config
    ).cls.predictions.transform

    if "result" in model_name_or_path:
        state_dict = torch.load(
            model_name_or_path + "/pytorch_model.bin", map_location="cpu"
        )
        new_state_dict = {}
        for key, param in state_dict.items():
            if "pooler" in key:
                key = key.replace("pooler.", "")
                new_state_dict[key] = param
        mlp.load_state_dict(new_state_dict)

    mlp.eval()
    return mlp.to(device)


def encode_word_batches(
    model,
    tokenizer,
    device: torch.device,
    word_batches: Sequence[Sequence[Sequence[str]]],
    *,
    mask_embedding_sentence: bool,
    mask_embedding_sentence_template: Optional[str],
    mask_num: int,
    pooler: str,
    delta_tensor: Optional[torch.Tensor],
    template_len: Optional[int],
    mlp: Optional[nn.Module],
    mask_embedding_sentence_org_mlp: bool,
    mask_embedding_sentence_use_org_pooler: bool,
    mask_embedding_sentence_use_pooler: bool,
) -> np.ndarray:
    """返回 [total_sentences, hidden] 的 numpy（按 batch 顺序拼接）。"""
    all_rows: List[torch.Tensor] = []

    for batch in tqdm.tqdm(word_batches, desc="Encoding"):
        sentences = [" ".join(s) for s in batch]
        if mask_embedding_sentence and mask_embedding_sentence_template:
            enc_sents = [
                apply_mask_template(tokenizer, mask_embedding_sentence_template, s)
                for s in sentences
            ]
        else:
            enc_sents = sentences

        batch_enc = tokenizer.batch_encode_plus(
            enc_sents, return_tensors="pt", padding=True, truncation=True
        )
        for k in batch_enc:
            if batch_enc[k] is not None:
                batch_enc[k] = batch_enc[k].to(device)

        with torch.no_grad():
            outputs = model(
                **batch_enc, output_hidden_states=True, return_dict=True
            )
            last_hidden = outputs.last_hidden_state

            if mask_embedding_sentence and mask_embedding_sentence_template:
                pooler_out = last_hidden[
                    batch_enc["input_ids"] == tokenizer.mask_token_id
                ]
                pooler_out = pooler_out.view(-1, mask_num, pooler_out.shape[-1])
                pooler_out = pooler_out[:, mask_num - 1, :]
                if mask_embedding_sentence_org_mlp and mlp is not None:
                    pooler_out = mlp(pooler_out)
                if delta_tensor is not None and template_len is not None:
                    blen = batch_enc["attention_mask"].sum(-1) - template_len
                    if mask_embedding_sentence_org_mlp and mlp is not None:
                        pooler_out = pooler_out - mlp(delta_tensor[blen])
                    else:
                        pooler_out = pooler_out - delta_tensor[blen]
                if mask_embedding_sentence_use_org_pooler and mlp is not None:
                    pooler_out = mlp(pooler_out)
                if mask_embedding_sentence_use_pooler:
                    pooler_out = model.pooler.dense(pooler_out)
                    pooler_out = model.pooler.activation(pooler_out)
            else:
                if pooler == "cls":
                    try:
                        pooler_out = outputs.pooler_output
                    except AttributeError:
                        pooler_out = last_hidden[:, 0, :]
                elif pooler == "avg":
                    pm = batch_enc["attention_mask"].unsqueeze(-1).float()
                    pooler_out = (last_hidden * pm).sum(1) / pm.sum(1).clamp(min=1)
                elif pooler == "avg_first_last":
                    first_h = outputs.hidden_states[0]
                    pm = batch_enc["attention_mask"].unsqueeze(-1).float()
                    mix = (first_h + last_hidden) / 2.0
                    pooler_out = (mix * pm).sum(1) / pm.sum(1).clamp(min=1)
                elif pooler == "cls_before_pooler":
                    input_ids = batch_enc["input_ids"].clone()
                    input_ids[
                        (input_ids == 0)
                        | (input_ids == 101)
                        | (input_ids == 102)
                    ] = input_ids.max()
                    index = input_ids.topk(3, dim=-1, largest=False)[1]
                    index2 = torch.arange(input_ids.shape[0], device=input_ids.device)
                    r = last_hidden[index2, index[:, 0], :]
                    for t in range(1, 3):
                        r = r + last_hidden[index2, index[:, t], :]
                    pooler_out = r / 3.0
                elif pooler == "avg_top2":
                    second_last_hidden = outputs.hidden_states[-2]
                    pm = batch_enc["attention_mask"].unsqueeze(-1).float()
                    mix = (last_hidden + second_last_hidden) / 2.0
                    pooler_out = (mix * pm).sum(1) / pm.sum(1).clamp(min=1)
                else:
                    raise NotImplementedError(
                        f"非 mask 模式下暂未实现 pooler={pooler!r}。"
                    )

        all_rows.append(pooler_out.cpu())

    return torch.cat(all_rows, dim=0).numpy()


def load_sick_test_rows(
    path: str,
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """
    读取 SICK_test_annotated.txt（tab 分隔，含表头）。
    返回:
      - pairs: [(sA, sB, score), ...]
      - pair_endpoints: 按文件顺序 [sA, sB, sA, sB, ...]，长度 2P（不在此处对句子去重），
        供 Uniformity 在句对端点 multiset 上均匀抽样。
    """
    pairs: List[Tuple[str, str, float]] = []
    pair_endpoints: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sa = row["sentence_A"].strip()
            sb = row["sentence_B"].strip()
            score = float(row["relatedness_score"])
            pairs.append((sa, sb, score))
            pair_endpoints.append(sa)
            pair_endpoints.append(sb)

    return pairs, pair_endpoints


def compute_alignment(
    emb: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str, float]],
    pos_threshold: float,
    l2_normalize: bool,
) -> Tuple[float, int]:
    sq_dists = []
    for sa, sb, sc in pairs:
        if sc < pos_threshold:
            continue
        va, vb = emb[sa], emb[sb]
        if l2_normalize:
            va = va / (np.linalg.norm(va) + 1e-12)
            vb = vb / (np.linalg.norm(vb) + 1e-12)
        sq_dists.append(np.sum((va - vb) ** 2))
    if not sq_dists:
        raise ValueError(
            f"没有满足 score>={pos_threshold} 的句子对，请检查阈值或数据文件。"
        )
    return float(np.mean(sq_dists)), len(sq_dists)


def compute_uniformity(
    emb_matrix: np.ndarray,
    M: int,
    seed: int,
    l2_normalize: bool,
) -> float:
    """
    emb_matrix: [N, D]，每行对应 Uniformity 的一个抽样槽位（可为句对端点展开，重复句对应重复行向量）。
    从 {0..N-1} 上有放回独立抽两个下标，估计 log E[exp(-2||x-y||^2)]。
    """
    rng = np.random.default_rng(seed)
    n = emb_matrix.shape[0]
    if n < 2:
        raise ValueError("Uniformity 抽样槽位不足 2 个，无法估计。")

    x = emb_matrix.astype(np.float64, copy=True)
    if l2_normalize:
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    i = rng.integers(0, n, size=M)
    j = rng.integers(0, n, size=M)
    diff = x[i] - x[j]
    sq = np.sum(diff * diff, axis=1)
    val = np.log(np.mean(np.exp(-2.0 * sq)))
    return float(val)


def build_batches(
    sentences: List[str], batch_size: int
) -> List[List[List[str]]]:
    batches: List[List[List[str]]] = []
    cur: List[List[str]] = []
    for s in sentences:
        cur.append(sent_to_word_list(s))
        if len(cur) >= batch_size:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    return batches


def _split_config_argv(argv: List[str]) -> Tuple[str, List[str]]:
    """若首参为存在的 yaml 路径则作为 custom_file，其余并入命令行。"""
    if not argv:
        return "", []
    first = argv[0]
    if first.endswith((".yaml", ".yml")):
        if not os.path.isfile(first):
            print(f"找不到配置文件: {first}", file=sys.stderr)
            sys.exit(1)
        return first, argv[1:]
    return "", argv


def main():
    parser = argparse.ArgumentParser(description="SICK-R test: Alignment & Uniformity")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Transformers 模型名或本地 checkpoint 目录（可由默认 yaml 提供）",
    )
    parser.add_argument(
        "--sick_test",
        type=str,
        default="../SentEval/data/downstream/SICK/SICK_test_annotated.txt",
        help="SICK 测试集 annotated 文件路径",
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=4.0,
        help="Alignment 正例对：relatedness_score >= 该阈值（论文默认 4）",
    )
    parser.add_argument(
        "--uniformity_pairs",
        type=int,
        default=100_000,
        help="Uniformity 蒙特卡洛采样对数 M",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--l2_normalize",
        action="store_true",
        help="是否在算距离前对句向量做 L2 归一（超球面设定；默认与 eval 原始向量一致不做）",
    )
    parser.add_argument(
        "--mask_embedding_sentence",
        action="store_true",
        help="与 cot_bert_evaluation 一致：用模板 + 第二处 [MASK] 向量",
    )
    parser.add_argument(
        "--mask_embedding_sentence_template",
        type=str,
        default=None,
        help="评估用 anchor 模板（需与 SentEval 评测一致）",
    )
    parser.add_argument("--mask_num", type=int, default=2)
    parser.add_argument(
        "--mask_embedding_sentence_delta",
        action="store_true",
        help="减去 template denoising（与训练/评估一致时打开）",
    )
    parser.add_argument(
        "--mask_embedding_sentence_org_mlp",
        action="store_true",
        help="对 [MASK] 隐状态先过 BERT MLM transform（与 cot_bert_evaluation 一致）",
    )
    parser.add_argument(
        "--mask_embedding_sentence_use_org_pooler",
        action="store_true",
        help="在 delta 之后再过一层 MLP(transform)（与 cot_bert_evaluation 一致）",
    )
    parser.add_argument(
        "--mask_embedding_sentence_use_pooler",
        action="store_true",
        help="再过 model.pooler.dense + activation（与 cot_bert_evaluation 一致）",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_first_last", "avg_top2"],
        default="cls",
        help="非 mask 模式下的池化方式（与 cot_bert_evaluation 一致）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="可选：将数值与配置写入 JSON",
    )

    custom_file, rest = _split_config_argv(sys.argv[1:])
    print(f"SICK align/uniformity 默认配置: {DEFAULT_SICK_CONFIG}")
    if custom_file:
        print(f"SICK align/uniformity 覆盖配置: {custom_file}")
    args_list = load_configs(
        default_file=DEFAULT_SICK_CONFIG, custom_file=custom_file
    )
    args = parser.parse_args(args_list + rest)

    if not args.model_name_or_path:
        print(
            "错误: 未指定 --model_name_or_path，请在 yaml 或命令行中提供。",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.output_json in ("None", ""):
        args.output_json = None

    if args.mask_embedding_sentence and not args.mask_embedding_sentence_template:
        print(
            "错误: 使用 --mask_embedding_sentence 时必须提供 --mask_embedding_sentence_template",
            file=sys.stderr,
        )
        sys.exit(1)

    sick_path = os.path.abspath(args.sick_test)
    if not os.path.isfile(sick_path):
        print(f"找不到 SICK 文件: {sick_path}", file=sys.stderr)
        sys.exit(1)

    pairs, pair_endpoints = load_sick_test_rows(sick_path)
    unique_encode_order = list(dict.fromkeys(pair_endpoints))

    device = pick_device()
    print(
        f"Device: {device}, pairs: {len(pairs)}, "
        f"uniformity_slots (2P): {len(pair_endpoints)}, "
        f"distinct_strings (encode once): {len(unique_encode_order)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.eval()
    model.to(device)

    need_mlp = args.mask_embedding_sentence_org_mlp or (
        args.mask_embedding_sentence_use_org_pooler
    )
    mlp: Optional[nn.Module] = None
    if need_mlp:
        mlp = load_mask_prediction_mlp(args.model_name_or_path, device)

    delta_tensor = None
    template_len = None
    if args.mask_embedding_sentence_delta:
        if not (args.mask_embedding_sentence and args.mask_embedding_sentence_template):
            print("错误: --mask_embedding_sentence_delta 需要 mask 模板", file=sys.stderr)
            sys.exit(1)
        noise, template_len = denoising(
            model,
            args.mask_embedding_sentence_template,
            tokenizer,
            device,
            args.mask_num,
        )
        delta_tensor = noise

    batches = build_batches(unique_encode_order, args.batch_size)
    mat = encode_word_batches(
        model,
        tokenizer,
        device,
        batches,
        mask_embedding_sentence=args.mask_embedding_sentence,
        mask_embedding_sentence_template=args.mask_embedding_sentence_template,
        mask_num=args.mask_num,
        pooler=args.pooler,
        delta_tensor=delta_tensor,
        template_len=template_len,
        mlp=mlp,
        mask_embedding_sentence_org_mlp=args.mask_embedding_sentence_org_mlp,
        mask_embedding_sentence_use_org_pooler=args.mask_embedding_sentence_use_org_pooler,
        mask_embedding_sentence_use_pooler=args.mask_embedding_sentence_use_pooler,
    )

    emb_map: Dict[str, np.ndarray] = {
        s: mat[i] for i, s in enumerate(unique_encode_order)
    }

    align, n_pos = compute_alignment(
        emb_map, pairs, args.pos_threshold, args.l2_normalize
    )
    slot_matrix = np.stack([emb_map[s] for s in pair_endpoints], axis=0)
    unif = compute_uniformity(
        slot_matrix, args.uniformity_pairs, args.seed, args.l2_normalize
    )

    print("--- SICK-R test (Alignment / Uniformity) ---")
    print(f"pos_threshold (>={args.pos_threshold}): n_pairs = {n_pos}")
    print(f"Alignment  (lower better): {align:.6f}")
    print(f"Uniformity (lower better): {unif:.6f}")
    print(f"M (uniformity MC pairs): {args.uniformity_pairs}, seed: {args.seed}")
    print("Uniformity sampling: uniform over pair-endpoint slots (length 2P, multiset).")

    if args.output_json:
        record = {
            "model_name_or_path": args.model_name_or_path,
            "sick_test": sick_path,
            "pos_threshold": args.pos_threshold,
            "n_positive_pairs": n_pos,
            "alignment": align,
            "uniformity": unif,
            "uniformity_M": args.uniformity_pairs,
            "uniformity_sampling": "pair_endpoints_2P_multiset",
            "n_uniformity_slots": len(pair_endpoints),
            "n_distinct_strings_encoded": len(unique_encode_order),
            "seed": args.seed,
            "l2_normalize": args.l2_normalize,
            "mask_embedding_sentence": args.mask_embedding_sentence,
            "mask_embedding_sentence_template": args.mask_embedding_sentence_template,
            "mask_embedding_sentence_delta": args.mask_embedding_sentence_delta,
            "mask_embedding_sentence_org_mlp": args.mask_embedding_sentence_org_mlp,
            "mask_embedding_sentence_use_org_pooler": (
                args.mask_embedding_sentence_use_org_pooler
            ),
            "mask_embedding_sentence_use_pooler": (
                args.mask_embedding_sentence_use_pooler
            ),
            "mask_num": args.mask_num,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
