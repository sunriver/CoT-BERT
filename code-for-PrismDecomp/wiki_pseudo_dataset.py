"""
Wiki 语料 + Stage1 主题缓存 join（Stage2 只读）。
"""

import logging
from typing import Dict, List, Optional

from aspect_template_utils import load_theme_cache

logger = logging.getLogger(__name__)


def load_theme_targets(path: str) -> Dict[str, List[List[float]]]:
    return load_theme_cache(path)


def attach_theme_targets_to_examples(
    examples: dict,
    theme_lookup: Dict[str, List[List[float]]],
    default_themes: Optional[List[List[float]]] = None,
    num_themes: int = 7,
    compress_dim: int = 8,
) -> dict:
    """
    为 batched examples 添加 theme_targets 字段。
    examples['text']: list of str
    """
    if default_themes is None:
        default_themes = [[0.0] * compress_dim for _ in range(num_themes)]

    themes = []
    missing = 0
    for text in examples["text"]:
        if text is None:
            text = " "
        t = theme_lookup.get(text)
        if t is None:
            missing += 1
            t = default_themes
        themes.append(t)

    if missing > 0:
        logger.warning("Theme cache miss for %d sentences in batch.", missing)

    examples["theme_targets"] = themes
    return examples


def merge_theme_cache_into_dataset_map_fn(
    theme_lookup: Dict[str, List[List[float]]],
    num_themes: int = 7,
    compress_dim: int = 8,
):
    def _fn(examples):
        return attach_theme_targets_to_examples(
            examples,
            theme_lookup,
            num_themes=num_themes,
            compress_dim=compress_dim,
        )

    return _fn
