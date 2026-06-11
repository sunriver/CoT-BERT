#!/usr/bin/env python3
"""
Stage1: 使用 7 套 aspect 模板 + 冻结 BERT 为 wiki1m 每句生成 7 维数值伪标签。
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from aspect_template_utils import (
    build_fixed_projections,
    encode_template_sentence,
    extract_mask_hidden,
    get_aspect_templates_list,
    load_aspect_templates,
    normalize_scores_corpus,
    vector_to_scalar,
)


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def read_sentences(path: str, max_samples: int = None):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            sentences.append(text)
            if max_samples and len(sentences) >= max_samples:
                break
    return sentences


def collate_template_batch(batch_sentences, tokenizer, template, max_seq_length):
    encoded = [
        encode_template_sentence(tokenizer, template, s, max_seq_length)
        for s in batch_sentences
    ]
    max_len = max(len(e["input_ids"]) for e in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = [], []
    for e in encoded:
        pad_len = max_len - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(e["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


@torch.no_grad()
def compute_pseudo_labels(
    model,
    tokenizer,
    sentences,
    aspect_config,
    device,
    batch_size=32,
    max_seq_length=128,
):
    templates = get_aspect_templates_list(aspect_config)
    num_aspects = len(templates)
    hidden_size = model.config.hidden_size
    seed = aspect_config.get("projection_seed", 42)
    projection = build_fixed_projections(hidden_size, num_aspects, seed=seed).to(device)
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 103

    dataset = SentenceDataset(sentences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    for batch_sents in tqdm(loader, desc="Stage1 pseudo-labels"):
        aspect_vectors = []
        for template in templates:
            batch = collate_template_batch(batch_sents, tokenizer, template, max_seq_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_dict=True)
            hidden = extract_mask_hidden(
                outputs.last_hidden_state, batch["input_ids"], mask_token_id
            )
            aspect_vectors.append(hidden)

        # (batch, num_aspects, hidden) -> scores (batch, num_aspects)
        stacked = torch.stack(aspect_vectors, dim=1)
        batch_scores = []
        for b in range(stacked.size(0)):
            vecs = stacked[b]
            s = vector_to_scalar(vecs, projection)
            batch_scores.append(s.cpu().tolist())
        all_scores.extend(batch_scores)

    norm_method = aspect_config.get("normalization", "minmax")
    normalized, stats = normalize_scores_corpus(all_scores, method=norm_method)
    return normalized, stats


def main():
    parser = argparse.ArgumentParser(description="Stage1: template pseudo-label extraction")
    parser.add_argument("--input", required=True, help="wiki1m txt path")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--stats_output", default=None, help="pseudo_label_stats.json path")
    parser.add_argument("--model_path", required=True, help="frozen BERT path")
    parser.add_argument("--templates", default="configs/aspect_templates.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    aspect_config = load_aspect_templates(args.templates)
    sentences = read_sentences(args.input, args.max_samples)
    print(f"Loaded {len(sentences)} sentences from {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    scores, stats = compute_pseudo_labels(
        model,
        tokenizer,
        sentences,
        aspect_config,
        device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for text, s in zip(sentences, scores):
            f.write(json.dumps({"text": text, "s": s}, ensure_ascii=False) + "\n")

    stats_path = args.stats_output or args.output.replace(".jsonl", "_stats.json")
    stats["num_samples"] = len(sentences)
    stats["templates_config"] = args.templates
    stats["model_path"] = args.model_path
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(sentences)} pseudo-labels to {args.output}")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
