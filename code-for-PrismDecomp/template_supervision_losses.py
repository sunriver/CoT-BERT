"""
Stage2: 模板伪标签监督损失。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectScalarHeads(nn.Module):
    """每个子语义 h_i 对应一个标量预测头。"""

    def __init__(self, hidden_dim: int, num_aspects: int = 7):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_aspects)
        ])

    def forward(self, h_i_enhanced_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns: (batch, num_aspects) 预测标量
        """
        preds = []
        for i, h_i in enumerate(h_i_enhanced_list):
            preds.append(self.heads[i](h_i).squeeze(-1))
        return torch.stack(preds, dim=1)


class TemplateSupervisionLoss(nn.Module):
    """L_tpl: MSE(pred_i, s_i_cached) 或向量 cosine 对齐。"""

    def __init__(self, scalar_target: bool = True):
        super().__init__()
        self.scalar_target = scalar_target
        self.mse = nn.MSELoss()

    def forward(
        self,
        preds_or_h_list,
        aspect_targets: torch.Tensor,
        scalar_mode: bool = True,
    ) -> torch.Tensor:
        """
        scalar_mode=True: preds_or_h_list 为 (batch, num_aspects) 预测
        scalar_mode=False: preds_or_h_list 为 h_i list, aspect_targets 为 (batch, num_aspects, hidden)
        """
        if scalar_mode:
            return self.mse(preds_or_h_list, aspect_targets)

        losses = []
        for i, h_i in enumerate(preds_or_h_list):
            target_v = aspect_targets[:, i, :]
            target_v = F.normalize(target_v, p=2, dim=-1)
            h_norm = F.normalize(h_i, p=2, dim=-1)
            losses.append(1.0 - (h_norm * target_v).sum(dim=-1).mean())
        return torch.stack(losses).mean()


class PseudoContrastiveLoss(nn.Module):
    """可选: batch 内 |s_i(a)-s_i(b)| 小则 cos(h_i^a,h_i^b) 大。"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        h_i_enhanced_list: List[torch.Tensor],
        aspect_scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = aspect_scores.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=aspect_scores.device, requires_grad=True)

        losses = []
        for i, h_i in enumerate(h_i_enhanced_list):
            h_norm = F.normalize(h_i, p=2, dim=-1)
            sim_matrix = h_norm @ h_norm.t()
            s_i = aspect_scores[:, i]
            label_sim = 1.0 - torch.abs(s_i.unsqueeze(0) - s_i.unsqueeze(1))
            label_sim = label_sim.clamp(0.0, 1.0)
            eye = torch.eye(batch_size, device=sim_matrix.device, dtype=torch.bool)
            pred = (sim_matrix + 1.0) / 2.0
            diff = (pred - label_sim) ** 2
            diff = diff.masked_fill(eye, 0.0)
            losses.append(diff.sum() / (batch_size * (batch_size - 1) + self.eps))
        return torch.stack(losses).mean()
