"""
Stage1 主题 hidden 提取（7 prompt + 冻结 BERT）。
"""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from aspect_template_utils import (
    encode_template_sentence,
    extract_mask_hidden,
    get_aspect_templates_list,
    load_aspect_templates,
)


class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def read_sentences(path: str, max_samples: int = None) -> List[str]:
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
def extract_theme_hiddens(
    model,
    tokenizer,
    sentences: List[str],
    aspect_config,
    device,
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> torch.Tensor:
    """
    Returns: (num_sentences, num_themes, hidden_dim)
    """
    templates = get_aspect_templates_list(aspect_config)
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 103
    dataset = SentenceDataset(sentences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_hidden = []
    for batch_sents in loader:
        theme_vectors = []
        for template in templates:
            batch = collate_template_batch(batch_sents, tokenizer, template, max_seq_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_dict=True)
            hidden = extract_mask_hidden(
                outputs.last_hidden_state, batch["input_ids"], mask_token_id
            )
            theme_vectors.append(hidden)
        stacked = torch.stack(theme_vectors, dim=1)  # (B, num_themes, H)
        all_hidden.append(stacked.cpu())

    return torch.cat(all_hidden, dim=0)
