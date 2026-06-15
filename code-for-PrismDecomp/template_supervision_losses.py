"""
Stage2: 主题向量监督与对比损失。
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


class ThemeContrastiveLoss(nn.Module):
    """L_theme: 每个主题维度的 batch 内 InfoNCE（pred 对齐 cached teacher）。"""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred/target: (batch, num_themes, compress_dim), target 通常 detach
        """
        batch_size, num_themes, _ = pred.shape
        if batch_size < 2:
            return pred.new_zeros(())

        pred_n = F.normalize(pred, p=2, dim=-1)
        target_n = F.normalize(target.detach(), p=2, dim=-1)

        losses = []
        for i in range(num_themes):
            anchor = pred_n[:, i, :]
            teacher = target_n[:, i, :]
            pos = (anchor * teacher).sum(dim=-1, keepdim=True) / self.temperature
            neg = anchor @ teacher.t() / self.temperature
            eye = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
            neg = neg.masked_fill(eye, float("-inf"))
            logits = torch.cat([pos, neg], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            losses.append(self.ce(logits, labels))

        return torch.stack(losses).mean()
