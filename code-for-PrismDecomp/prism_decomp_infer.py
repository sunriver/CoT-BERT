#!/usr/bin/env python3
"""
PrismDecomp 可解释推理：输出 global + 7 维主题码相似度（7×d cosine）。
"""

import argparse
import json

import torch
from transformers import AutoConfig, AutoTokenizer

from aspect_template_utils import get_aspect_names, load_aspect_templates
from cot_prompt_utils import (
    build_single_cot_eval_input,
    parse_all_cot_templates,
    register_cot_templates_on_model,
    uses_cot_multi_view,
)
from prism_decomp_model import BertForPrismDecomp

COT_MAIN_TEMPLATE = (
    '*cls*_The_sentence_of_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*'
)
COT_DIFF_TEMPLATE = (
    '*cls*_The_sentence_:_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*'
)
COT_NEG_TEMPLATE = (
    '*cls*_The_sentence_:_"*sent_0*"_does_not_mean_*mask*_,_so_it_can_not_be_summarized_as_*mask*_._*sep+*'
)


@torch.no_grad()
def encode_sentences(model, tokenizer, sentences, model_args, device, max_length=128):
    if uses_cot_multi_view(model_args):
        all_input_ids, all_masks = [], []
        for sent in sentences:
            ids = build_single_cot_eval_input(sent, model_args, tokenizer, max_length)
            all_input_ids.append(ids)
            all_masks.append([1] * len(ids))
        max_len = max(len(x) for x in all_input_ids)
        pad_id = tokenizer.pad_token_id
        for i in range(len(all_input_ids)):
            pad = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [pad_id] * pad
            all_masks[i] = all_masks[i] + [0] * pad
        batch = {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long, device=device),
        }
    else:
        template = model_args.mask_embedding_sentence_template
        parts = template.split("[X]")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else " means [MASK]."
        bs = tokenizer.encode(prefix, add_special_tokens=False)
        es = tokenizer.encode(suffix, add_special_tokens=False)
        all_input_ids, all_masks = [], []
        for sent in sentences:
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_length]
            ids = [tokenizer.cls_token_id] + bs + s + es + [tokenizer.sep_token_id]
            all_input_ids.append(ids)
            all_masks.append([1] * len(ids))
        max_len = max(len(x) for x in all_input_ids)
        pad_id = tokenizer.pad_token_id
        for i in range(len(all_input_ids)):
            pad = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [pad_id] * pad
            all_masks[i] = all_masks[i] + [0] * pad
        batch = {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long, device=device),
        }

    outputs = model(**batch, sent_emb=True, return_dict=True, return_aspects=True)
    aspect_reprs = outputs.get("aspect_reprs") if hasattr(outputs, "get") else getattr(outputs, "aspect_reprs", None)
    return outputs.pooler_output, aspect_reprs


def cosine_sim(a, b):
    return torch.sum(a * b, dim=-1).item()


def predict_pair(model, tokenizer, sent_a, sent_b, model_args, aspect_names, device):
    pooler, aspects = encode_sentences(
        model, tokenizer, [sent_a, sent_b], model_args, device
    )
    result = {"global": cosine_sim(pooler[0], pooler[1])}
    for i, name in enumerate(aspect_names):
        result[name] = cosine_sim(aspects[0, i], aspects[1, i])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sent_a", default="the man isn't singing")
    parser.add_argument("--sent_b", default="the man is singing")
    parser.add_argument("--templates", default="configs/aspect_templates.yaml")
    parser.add_argument("--compress_dim", type=int, default=8)
    parser.add_argument("--compressor_ckpt", default="preprocessed/theme_compressor.pt")
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

    aspect_cfg = load_aspect_templates(args.templates)
    aspect_names = get_aspect_names(aspect_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    class MinimalArgs:
        num_semantics = len(aspect_names)
        temperature = 0.05
        lambda2 = 0.1
        lambda_sup = 0.0
        compress_dim = args.compress_dim
        compress_mode = aspect_cfg.get("compress_mode", "mlp")
        compressor_hidden = aspect_cfg.get("compressor_hidden", 256)
        compressor_ckpt = args.compressor_ckpt
        sup_loss_type = "cosine"
        mask_embedding_sentence = True
        mask_num = 2
        mask_embedding_sentence_template = COT_MAIN_TEMPLATE
        mask_embedding_sentence_different_template = COT_DIFF_TEMPLATE
        mask_embedding_sentence_negative_template = COT_NEG_TEMPLATE
        mask_embedding_sentence_delta = True
        mask_embedding_sentence_delta_no_delta_eval = True
        scd_temp = 0.05

    model_args = MinimalArgs()
    parse_all_cot_templates(model_args, tokenizer)

    model = BertForPrismDecomp.from_pretrained(
        args.model_path, config=config, model_args=model_args
    )
    register_cot_templates_on_model(model, model_args, tokenizer)
    model.to(device)
    model.eval()

    out = predict_pair(
        model, tokenizer, args.sent_a, args.sent_b,
        model_args, aspect_names, device
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
