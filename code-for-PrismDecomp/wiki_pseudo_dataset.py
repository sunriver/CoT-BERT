"""
Wiki 语料 + Stage1 伪标签 join。
"""

import json
from typing import Dict, List, Optional

from aspect_template_utils import load_pseudo_labels_jsonl


def load_pseudo_labels(path: str) -> Dict[str, List[float]]:
    return load_pseudo_labels_jsonl(path)


def attach_pseudo_labels_to_examples(
    examples: dict,
    pseudo_lookup: Dict[str, List[float]],
    default_scores: Optional[List[float]] = None,
) -> dict:
    """
    为 batched examples 添加 aspect_scores 字段。
    examples['text']: list of str
    """
    num_aspects = len(next(iter(pseudo_lookup.values()))) if pseudo_lookup else 7
    if default_scores is None:
        default_scores = [0.5] * num_aspects

    scores = []
    for text in examples["text"]:
        if text is None:
            text = " "
        s = pseudo_lookup.get(text, default_scores)
        scores.append(s)

    examples["aspect_scores"] = scores
    return examples


def merge_pseudo_into_dataset_map_fn(pseudo_lookup: Dict[str, List[float]]):
    def _fn(examples):
        return attach_pseudo_labels_to_examples(examples, pseudo_lookup)
    return _fn
