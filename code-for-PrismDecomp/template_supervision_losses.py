"""
Stage2: 主题向量监督损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThemeVectorSupervisionLoss(nn.Module):
    """L_sup: 对齐 Stage1 缓存的 7×d 主题向量。"""

    def __init__(self, loss_type: str = "cosine"):
        super().__init__()
        self.loss_type = loss_type
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred/target: (batch, num_themes, compress_dim)
        """
        if self.loss_type == "mse":
            return self.mse(pred, target)

        pred_n = F.normalize(pred, p=2, dim=-1)
        target_n = F.normalize(target, p=2, dim=-1)
        cos = (pred_n * target_n).sum(dim=-1)
        return (1.0 - cos).mean()
