"""
Aspect 评估：主题向量 cosine 相关 + 可选 SentEval STS。
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

from aspect_template_utils import get_aspect_names, load_aspect_templates
from wiki_pseudo_dataset import load_theme_targets


@torch.no_grad()
def evaluate_theme_cache_correlation(
    model,
    tokenizer,
    theme_cache_path: str,
    template: str,
    aspect_names: List[str],
    device,
    max_samples: int = 2000,
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> Dict[str, float]:
    """
    比较分解器预测的 7×d 主题码与 Stage1 缓存的 cosine 相关（按主题维）。
    """
    lookup = load_theme_targets(theme_cache_path)
    texts = list(lookup.keys())[:max_samples]

    parts = template.split("[X]")
    prefix, suffix = parts[0], parts[1] if len(parts) > 1 else " means [MASK]."
    bs = tokenizer.encode(prefix, add_special_tokens=False)
    es = tokenizer.encode(suffix, add_special_tokens=False)

    all_preds = {i: [] for i in range(len(aspect_names))}
    all_targets = {i: [] for i in range(len(aspect_names))}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = {"input_ids": [], "attention_mask": []}
        for sent in batch_texts:
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            ids = [tokenizer.cls_token_id] + bs + s + es + [tokenizer.sep_token_id]
            batch["input_ids"].append(ids)
            batch["attention_mask"].append([1] * len(ids))
        max_len = max(len(x) for x in batch["input_ids"])
        pad = tokenizer.pad_token_id
        for i in range(len(batch["input_ids"])):
            pl = max_len - len(batch["input_ids"][i])
            batch["input_ids"][i] += [pad] * pl
            batch["attention_mask"][i] += [0] * pl

        inp = {
            "input_ids": torch.tensor(batch["input_ids"], device=device),
            "attention_mask": torch.tensor(batch["attention_mask"], device=device),
        }
        model.eval()
        outputs = model.bert(**inp, return_dict=True)
        mask_id = tokenizer.mask_token_id or 103
        reps = []
        for i in range(inp["input_ids"].size(0)):
            m = inp["input_ids"][i] == mask_id
            pos = m.long().argmax().item() if m.any() else 0
            reps.append(outputs.last_hidden_state[i, pos])
        h = torch.stack(reps, dim=0)
        _, h_i_enh, _ = model.multisemantic_spr(h, compute_orth_loss=False, return_all_semantics=True)
        h_stack = torch.stack(h_i_enh, dim=1)
        pred = model.theme_compressor(h_stack, normalize=True).cpu().numpy()

        for j, text in enumerate(batch_texts):
            target = np.array(lookup[text], dtype=np.float32)
            for i in range(len(aspect_names)):
                cos_pred = pred[j, i]
                cos_tgt = target[i]
                all_preds[i].append(float(np.dot(cos_pred, cos_tgt)))
                all_targets[i].append(float(np.linalg.norm(cos_tgt)))

    metrics = {}
    for i, name in enumerate(aspect_names):
        rho, _ = spearmanr(all_preds[i], all_targets[i])
        metrics[f"theme_align_{name}"] = float(rho) if not np.isnan(rho) else 0.0
    if metrics:
        metrics["theme_align_avg"] = float(np.mean(list(metrics.values())))
    return metrics


def save_eval_report(metrics: Dict[str, float], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
