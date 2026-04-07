# -*- coding: utf-8 -*-
"""
Wang & Isola 风格的 Alignment / Uniformity 公共实现与数据加载。

- cls_before_pooler 分支使用 BERT 特殊 token id 101/102；RoBERTa 等架构请改用 cls/avg 等通用 pooler。
- 导入方若使用 cot_bert_evaluation.denoising，会触发该模块的全局副作用（如平台信息打印）。
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm


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
    tqdm_disable: bool = False,
) -> np.ndarray:
    """返回 [total_sentences, hidden] 的 numpy（按 batch 顺序拼接）。"""
    all_rows: List[torch.Tensor] = []

    it = word_batches
    if not tqdm_disable:
        it = tqdm.tqdm(word_batches, desc="Encoding")

    for batch in it:
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
    返回 pairs 与 pair_endpoints（2P multiset，按文件顺序）。
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


def load_sts_benchmark_test_rows(
    path: str,
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """
    读取 STS-Benchmark 测试集（SentEval 常见路径：STSBenchmark/sts-test.csv）。

    支持：
    - 官方 7 列 tab：genre, filename, year, old_id, score, sentence1, sentence2
    - 简化 3 列：sentence1, sentence2, score（或 s1, s2, score，无表头也可）
    """
    pairs: List[Tuple[str, str, float]] = []
    pair_endpoints: List[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.reader(f, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"空文件: {path}")

    def _row_to_triple(row: List[str]) -> Optional[Tuple[str, str, float]]:
        row = [c.strip() for c in row]
        if not row or all(not c for c in row):
            return None
        # 表头跳过
        joined_l = " ".join(x.lower() for x in row[:3])
        if "sentence1" in joined_l and "sentence2" in joined_l:
            return None

        if len(row) >= 7:
            try:
                score = float(row[4])
                s1, s2 = row[5].strip(), row[6].strip()
            except (ValueError, IndexError):
                return None
            if s1 and s2:
                return (s1, s2, score)
            return None

        if len(row) >= 3:
            s1, s2, sc = row[0], row[1], row[2]
            try:
                score = float(sc)
            except ValueError:
                return None
            if s1 and s2:
                return (s1, s2, score)
        return None

    for row in rows:
        t = _row_to_triple(row)
        if t is None:
            continue
        sa, sb, score = t
        pairs.append((sa, sb, score))
        pair_endpoints.append(sa)
        pair_endpoints.append(sb)

    if not pairs:
        raise ValueError(
            f"未能从 {path} 解析出任何句对；请检查是否为 STS-Benchmark sts-test 格式（7 列 tab 或 3 列）。"
        )
    return pairs, pair_endpoints


def load_pair_dataset(
    path: str,
    dataset_type: str,
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """dataset_type: sick | sts_benchmark"""
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"数据文件不存在: {ap}")
    if dataset_type == "sick":
        return load_sick_test_rows(ap)
    if dataset_type == "sts_benchmark":
        return load_sts_benchmark_test_rows(ap)
    raise ValueError(f"未知 dataset_type: {dataset_type!r}，应为 sick 或 sts_benchmark")


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
    emb_matrix: [N, D]，每行对应 Uniformity 的一个抽样槽位。
    估计 log E[exp(-2||x-y||^2)]（有放回独立抽下标）。
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
