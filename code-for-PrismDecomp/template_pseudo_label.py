#!/usr/bin/env python3
"""
Stage1b: 7 套主题 prompt + 冻结 BERT + 共享压缩器 -> wiki_theme_cache.jsonl
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from aspect_template_utils import load_aspect_templates, load_cached_texts
from theme_compressor import load_compressor_checkpoint
from theme_extraction_utils import extract_theme_hiddens, read_sentences


def main():
    parser = argparse.ArgumentParser(description="Stage1b: extract 7×d theme cache")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--compressor_ckpt", required=True)
    parser.add_argument("--templates", default="configs/aspect_templates.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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

    done_texts = set()
    if args.resume and os.path.exists(args.output):
        done_texts = load_cached_texts(args.output)
        sentences = [s for s in sentences if s not in done_texts]
        print(f"Resume: skip {len(done_texts)} cached, remaining {len(sentences)}")

    if not sentences and done_texts:
        print("All sentences already cached.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    compressor, _, meta = load_compressor_checkpoint(args.compressor_ckpt)
    compressor.eval()
    compressor.to(device)
    compress_dim = meta.get("compress_dim", aspect_config.get("compress_dim", 8))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_mode = "a" if args.resume and os.path.exists(args.output) else "w"

    chunk_size = args.batch_size * 4
    total_written = len(done_texts)
    with open(args.output, write_mode, encoding="utf-8") as f:
        for start in tqdm(range(0, len(sentences), chunk_size), desc="Stage1b theme cache"):
            chunk = sentences[start : start + chunk_size]
            hiddens = extract_theme_hiddens(
                model,
                tokenizer,
                chunk,
                aspect_config,
                device,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
            ).to(device)

            with torch.no_grad():
                codes = compressor(hiddens, normalize=True).cpu().tolist()

            for text, theme_vecs in zip(chunk, codes):
                f.write(json.dumps({"text": text, "themes": theme_vecs}, ensure_ascii=False) + "\n")
                total_written += 1

    stats = {
        "num_themes": len(aspect_config.get("aspects", [])),
        "compress_dim": compress_dim,
        "compressor_ckpt": args.compressor_ckpt,
        "model_path": args.model_path,
        "templates_config": args.templates,
        "num_samples": total_written,
    }
    stats_path = args.stats_output or args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote {total_written} theme caches to {args.output}")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
