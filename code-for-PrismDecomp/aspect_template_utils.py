"""
Aspect 模板与主题缓存工具（Stage1 / Stage2 共用）。
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

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


def load_cached_texts(path: str) -> Set[str]:
    texts = set()
    if not path or not os.path.exists(path):
        return texts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.add(json.loads(line)["text"])
    return texts


def load_theme_cache(path: str) -> Dict[str, List[List[float]]]:
    """text -> themes (num_themes, compress_dim)"""
    lookup: Dict[str, List[List[float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lookup[obj["text"]] = obj["themes"]
    return lookup


def load_theme_cache_stats(stats_path: str) -> Dict[str, Any]:
    if not stats_path or not os.path.exists(stats_path):
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_theme_cache(
    theme_lookup: Dict[str, List[List[float]]],
    expected_themes: int,
    expected_dim: int,
) -> None:
    if not theme_lookup:
        raise ValueError("Theme cache is empty.")
    sample = next(iter(theme_lookup.values()))
    if len(sample) != expected_themes:
        raise ValueError(
            f"Expected {expected_themes} themes, got {len(sample)} in cache."
        )
    if len(sample[0]) != expected_dim:
        raise ValueError(
            f"Expected compress_dim={expected_dim}, got {len(sample[0])} in cache."
        )
