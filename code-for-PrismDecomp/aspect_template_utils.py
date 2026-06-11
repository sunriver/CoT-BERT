"""
Aspect 模板与伪标签数值化工具（Stage1 / Stage2 共用）。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


def load_aspect_templates(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_aspect_names(config: Dict[str, Any]) -> List[str]:
    return [a["name"] for a in config["aspects"]]


def get_aspect_templates_list(config: Dict[str, Any]) -> List[str]:
    aspects = sorted(config["aspects"], key=lambda x: x["index"])
    return [a["template"] for a in aspects]


def build_fixed_projections(
    hidden_size: int,
    num_aspects: int,
    seed: int = 42,
) -> torch.Tensor:
    """固定随机投影 W_i: (num_aspects, hidden_size)。"""
    gen = torch.Generator()
    gen.manual_seed(seed)
    weights = torch.randn(num_aspects, hidden_size, generator=gen)
    weights = weights / weights.norm(dim=1, keepdim=True)
    return weights


def vector_to_scalar(
    vectors: torch.Tensor,
    projection_weights: torch.Tensor,
) -> torch.Tensor:
    """
    vectors: (batch, hidden) 或 (hidden,)
    projection_weights: (num_aspects, hidden)
    返回: (batch, num_aspects) 或 (num_aspects,)
    """
    if vectors.dim() == 1:
        normed = torch.nn.functional.normalize(vectors.unsqueeze(0), p=2, dim=-1)
        scores = torch.sigmoid(normed @ projection_weights.t())
        return scores.squeeze(0)
    normed = torch.nn.functional.normalize(vectors, p=2, dim=-1)
    return torch.sigmoid(normed @ projection_weights.t())


def encode_template_sentence(
    tokenizer,
    template: str,
    sentence: str,
    max_seq_length: int = 128,
) -> Dict[str, List[int]]:
    """将 [X] 模板与句子组合为 input_ids / attention_mask。"""
    parts = template.split("[X]")
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else " [MASK]."
    bs = tokenizer.encode(prefix, add_special_tokens=False)
    es = tokenizer.encode(suffix, add_special_tokens=False)
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)[:max_seq_length]
    input_ids = [tokenizer.cls_token_id] + bs + sent_ids + es + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def extract_mask_hidden(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """从 last_hidden_state 提取每个样本 [MASK] 位置向量。"""
    reps = []
    for i in range(input_ids.size(0)):
        mask_mask = input_ids[i] == mask_token_id
        if mask_mask.any():
            pos = mask_mask.long().argmax().item()
            reps.append(last_hidden_state[i, pos, :])
        else:
            reps.append(last_hidden_state[i, 0, :])
    return torch.stack(reps, dim=0)


def normalize_scores_corpus(
    scores_list: List[List[float]],
    method: str = "minmax",
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """全库 min-max 归一化，返回归一化分数与统计元数据。"""
    arr = np.array(scores_list, dtype=np.float32)
    stats = {"method": method, "mins": [], "maxs": []}
    if method == "minmax":
        for col in range(arr.shape[1]):
            mi, ma = float(arr[:, col].min()), float(arr[:, col].max())
            stats["mins"].append(mi)
            stats["maxs"].append(ma)
            if ma - mi > 1e-8:
                arr[:, col] = (arr[:, col] - mi) / (ma - mi)
            else:
                arr[:, col] = 0.5
    normalized = arr.tolist()
    return normalized, stats


def load_pseudo_label_stats(stats_path: str) -> Dict[str, Any]:
    if not stats_path or not os.path.exists(stats_path):
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pseudo_labels_jsonl(path: str) -> Dict[str, List[float]]:
    """text -> normalized aspect scores [s0..sK-1]"""
    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lookup[obj["text"]] = obj["s"]
    return lookup
