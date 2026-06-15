#!/usr/bin/env python3
"""
Stage1a: 在 wiki 子集上训练共享主题压缩器（重建 + 软正交）。
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from aspect_template_utils import load_aspect_templates
from theme_compressor import (
    SharedThemeCompressor,
    SharedThemeDecompressor,
    save_compressor_checkpoint,
    theme_code_orth_loss,
)
from theme_extraction_utils import extract_theme_hiddens, read_sentences


def train_compressor(
    theme_hiddens: torch.Tensor,
    compress_dim: int,
    hidden_size: int,
    mode: str,
    projection_seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_orth: float,
    device,
):
    num_samples, num_themes, hidden_dim = theme_hiddens.shape
    compressor = SharedThemeCompressor(
        hidden_dim=hidden_dim,
        compress_dim=compress_dim,
        hidden_size=hidden_size,
        mode=mode,
        projection_seed=projection_seed,
    ).to(device)
    decompressor = None
    if mode == "mlp":
        decompressor = SharedThemeDecompressor(
            hidden_dim=hidden_dim,
            compress_dim=compress_dim,
            hidden_size=hidden_size,
        ).to(device)
        params = list(compressor.parameters()) + list(decompressor.parameters())
    else:
        params = []

    if not params:
        compressor.eval()
        return compressor, decompressor

    optimizer = torch.optim.AdamW(params, lr=lr)
    dataset = TensorDataset(theme_hiddens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for (batch,) in tqdm(loader, desc=f"Stage1a epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)
            b, n, h = batch.shape
            flat = batch.view(b * n, h)

            codes = compressor(flat, normalize=True)
            recon = decompressor(codes)
            recon_loss = F.mse_loss(recon, flat)

            codes_3d = codes.view(b, n, -1)
            orth_loss = theme_code_orth_loss(codes_3d)
            loss = recon_loss + lambda_orth * orth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
        print(f"epoch {epoch + 1}: avg_loss={total_loss / max(steps, 1):.6f}")

    return compressor, decompressor


def main():
    parser = argparse.ArgumentParser(description="Stage1a: train shared theme compressor")
    parser.add_argument("--input", required=True)
    parser.add_argument("--compressor_out", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--templates", default="configs/aspect_templates.yaml")
    parser.add_argument("--compress_dim", type=int, default=None)
    parser.add_argument("--compressor_hidden", type=int, default=None)
    parser.add_argument("--compress_mode", default=None, choices=["mlp", "fixed_linear"])
    parser.add_argument("--projection_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    aspect_config = load_aspect_templates(args.templates)
    compress_dim = args.compress_dim or aspect_config.get("compress_dim", 8)
    hidden_size = args.compressor_hidden or aspect_config.get("compressor_hidden", 256)
    mode = args.compress_mode or aspect_config.get("compress_mode", "mlp")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    sentences = read_sentences(args.input, args.max_samples)
    print(f"Loaded {len(sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    theme_hiddens = extract_theme_hiddens(
        model,
        tokenizer,
        sentences,
        aspect_config,
        device,
        batch_size=args.extract_batch_size,
        max_seq_length=args.max_seq_length,
    )
    print(f"Extracted theme hiddens: {tuple(theme_hiddens.shape)}")

    hidden_dim = theme_hiddens.shape[-1]
    compressor, decompressor = train_compressor(
        theme_hiddens,
        compress_dim=compress_dim,
        hidden_size=hidden_size,
        mode=mode,
        projection_seed=args.projection_seed,
        epochs=args.epochs if mode == "mlp" else 0,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_orth=args.lambda_orth,
        device=device,
    )

    os.makedirs(os.path.dirname(args.compressor_out) or ".", exist_ok=True)
    meta = {
        "compress_dim": compress_dim,
        "hidden_dim": hidden_dim,
        "compressor_hidden": hidden_size,
        "mode": mode,
        "num_themes": len(aspect_config.get("aspects", [])),
        "model_path": args.model_path,
        "templates": args.templates,
        "train_samples": len(sentences),
    }
    save_compressor_checkpoint(
        args.compressor_out,
        compressor.cpu(),
        decompressor.cpu() if decompressor is not None else None,
        meta=meta,
    )
    stats_path = args.compressor_out.replace(".pt", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved compressor to {args.compressor_out}")


if __name__ == "__main__":
    main()
