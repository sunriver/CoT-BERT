"""
共享主题压缩器：7 个主题表示共用同一套压缩/解压权重。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedThemeCompressor(nn.Module):
    """768d theme hidden -> compress_dim，7 主题共享权重。"""

    def __init__(
        self,
        hidden_dim: int = 768,
        compress_dim: int = 8,
        hidden_size: int = 256,
        mode: str = "mlp",
        projection_seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compress_dim = compress_dim
        self.mode = mode

        if mode == "fixed_linear":
            gen = torch.Generator()
            gen.manual_seed(projection_seed)
            weight = torch.randn(compress_dim, hidden_dim, generator=gen)
            weight = weight / weight.norm(dim=1, keepdim=True)
            self.register_buffer("fixed_weight", weight)
            self.encoder = None
        else:
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, compress_dim),
            )

    def forward(self, theme_hidden: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        theme_hidden: (B, 768) or (B, num_themes, 768)
        returns: (B, compress_dim) or (B, num_themes, compress_dim)
        """
        single = theme_hidden.dim() == 2
        if single:
            theme_hidden = theme_hidden.unsqueeze(1)

        b, n, h = theme_hidden.shape
        flat = theme_hidden.reshape(b * n, h)
        if self.mode == "fixed_linear":
            normed = F.normalize(flat, p=2, dim=-1)
            out = normed @ self.fixed_weight.t()
        else:
            out = self.encoder(flat)

        out = out.view(b, n, self.compress_dim)
        if normalize:
            out = F.normalize(out, p=2, dim=-1)
        if single:
            out = out.squeeze(1)
        return out


class SharedThemeDecompressor(nn.Module):
    """compress_dim -> hidden_dim，用于 Stage1a 重建损失。"""

    def __init__(self, hidden_dim: int = 768, compress_dim: int = 8, hidden_size: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(compress_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_dim),
        )

    def forward(self, compressed: torch.Tensor) -> torch.Tensor:
        single = compressed.dim() == 2
        if single:
            compressed = compressed.unsqueeze(1)
        b, n, d = compressed.shape
        flat = compressed.reshape(b * n, d)
        out = self.decoder(flat).view(b, n, -1)
        if single:
            out = out.squeeze(1)
        return out


def theme_code_orth_loss(theme_codes: torch.Tensor) -> torch.Tensor:
    """theme_codes: (B, num_themes, compress_dim) -> ||T^T T - I||_F^2 per sample."""
    normalized = F.normalize(theme_codes, p=2, dim=-1)
    losses = []
    for i in range(normalized.size(0)):
        gram = torch.mm(normalized[i], normalized[i].t())
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        losses.append(torch.norm(gram - eye, p="fro") ** 2)
    return torch.stack(losses).mean()


def save_compressor_checkpoint(
    path: str,
    compressor: SharedThemeCompressor,
    decompressor: Optional[SharedThemeDecompressor] = None,
    meta: Optional[Dict[str, Any]] = None,
):
    payload = {
        "compressor": compressor.state_dict(),
        "meta": meta or {},
    }
    if decompressor is not None:
        payload["decompressor"] = decompressor.state_dict()
    torch.save(payload, path)


def load_compressor_checkpoint(
    path: str,
    hidden_dim: int = 768,
    compress_dim: int = 8,
    hidden_size: int = 256,
    mode: str = "mlp",
    projection_seed: int = 42,
    load_decompressor: bool = False,
):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta", {})
    mode = meta.get("mode", mode)
    compress_dim = meta.get("compress_dim", compress_dim)
    hidden_dim = meta.get("hidden_dim", hidden_dim)
    hidden_size = meta.get("compressor_hidden", hidden_size)

    compressor = SharedThemeCompressor(
        hidden_dim=hidden_dim,
        compress_dim=compress_dim,
        hidden_size=hidden_size,
        mode=mode,
        projection_seed=projection_seed,
    )
    compressor.load_state_dict(payload["compressor"])

    decompressor = None
    if load_decompressor and "decompressor" in payload:
        decompressor = SharedThemeDecompressor(hidden_dim, compress_dim, hidden_size)
        decompressor.load_state_dict(payload["decompressor"])

    return compressor, decompressor, meta
