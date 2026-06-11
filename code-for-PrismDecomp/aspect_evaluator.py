"""
Aspect 评估：伪标签 Spearman 相关 + 可选 SentEval STS。
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

from aspect_template_utils import get_aspect_names, load_aspect_templates
from wiki_pseudo_dataset import load_pseudo_labels


@torch.no_grad()
def evaluate_pseudo_label_correlation(
    model,
    tokenizer,
    pseudo_label_path: str,
    template: str,
    aspect_names: List[str],
    device,
    max_samples: int = 2000,
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> Dict[str, float]:
    """
    比较 scalar_head(h_i) 与 Stage1 缓存 s_i 的 Spearman 相关。
    """
    lookup = load_pseudo_labels(pseudo_label_path)
    texts = list(lookup.keys())[:max_samples]

    from prism_decomp_infer import encode_sentences

    all_preds = {i: [] for i in range(len(aspect_names))}
    all_targets = {i: [] for i in range(len(aspect_names))}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = {
            "input_ids": [],
            "attention_mask": [],
        }
        parts = template.split("[X]")
        prefix, suffix = parts[0], parts[1] if len(parts) > 1 else " means [MASK]."
        bs = tokenizer.encode(prefix, add_special_tokens=False)
        es = tokenizer.encode(suffix, add_special_tokens=False)
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
        h_i_list, h_i_enh, _ = model.multisemantic_spr(h, compute_orth_loss=False, return_all_semantics=True)
        pred = model.aspect_scalar_heads(h_i_enh).cpu().numpy()

        for j, text in enumerate(batch_texts):
            target = lookup[text]
            for i in range(len(aspect_names)):
                all_preds[i].append(pred[j, i])
                all_targets[i].append(target[i])

    metrics = {}
    for i, name in enumerate(aspect_names):
        rho, _ = spearmanr(all_preds[i], all_targets[i])
        metrics[f"pseudo_spearman_{name}"] = float(rho) if not np.isnan(rho) else 0.0
    metrics["pseudo_spearman_avg"] = float(np.mean(list(metrics.values())))
    return metrics


def save_eval_report(metrics: Dict[str, float], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
