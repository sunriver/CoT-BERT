"""
TexLeJEPA 模型：BERT backbone + MLP Projector。
双路 Dropout 增广：同一 batch 前向两次得到 (z1, z2) 用于 Invariance + SIGReg。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from core import EppsPulley, SlicingUnivariateTest


class Projector(nn.Module):
    """两层 MLP Projector：增加非线性表达能力与 LayerNorm。"""

    def __init__(self, hidden_size: int, out_size: int = None):
        super().__init__()
        out_size = out_size or hidden_size
        self.linears = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_size)
            # nn.LayerNorm(out_size),
        )

    def forward(self, x):
        return self.linears(x)


class Predictor(nn.Module):
    """预测器 MLP：用于非对称结构中的特征预测。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.linears(x)


class BertForTexLeJEPA(BertPreTrainedModel):
    """
    BERT + Projector，用于 TexLeJEPA（Invariance + SIGReg）。
    - backbone: BertModel
    - projector: MLP (hidden -> hidden, Tanh)
    - 训练时 forward 返回 loss 和 (z1, z2)，由同一 batch 前向两次得到（不同 dropout 产生两视图）
    - 评估时 sent_emb=True 返回单次前向的句子嵌入
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, projector_hidden_size: int = None, lamb: float = 0.5):
        super().__init__(config)
        self.bert = BertModel(config)
        proj_dim = projector_hidden_size or config.hidden_size
        self.projector = Projector(config.hidden_size, proj_dim)
        self.predictor = Predictor(proj_dim)
        # TexLeJEPA 超参数（从 config 中读取，若缺失则使用安全默认值，避免老 checkpoint 报错）
        self.lamb = lamb
        self.num_slices = getattr(config, "texlejepa_num_slices", 256)
        self.epps_t_max = getattr(config, "texlejepa_epps_t_max", 3.0)
        self.epps_n_points = getattr(config, "texlejepa_epps_n_points", 17)
        self.sig_clip_value = getattr(config, "texlejepa_sig_clip_value", 0.01)

        epps = EppsPulley(t_max=self.epps_t_max, n_points=self.epps_n_points)
        self.sig_reg_module = SlicingUnivariateTest(
                univariate_test=epps,
                num_slices=self.num_slices,
                reduction="mean",
                sampler="gaussian",
                clip_value=self.sig_clip_value,
        )

    def _pool_and_project(self, last_hidden_state):
        """取 CLS 并过 projector。last_hidden_state: (batch, seq, hidden)"""
        pooled = last_hidden_state[:, 0, :]  # (batch, hidden)
        return self.projector(pooled)

    def _compute_texlejepa_loss(self, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        计算 TexLeJEPA 损失（非对称 SimSiam 风格）：
        L = (1-λ)*L_inv + λ*L_sig
        其中 L_inv = 0.5 * MSE(p1, z2.detach()) + 0.5 * MSE(p2, z1.detach())
        """
        device = p1.device
        
        # 1. L2 归一化：为了计算 MSE 时的稳定性
        p1_norm = F.normalize(p1, p=2, dim=-1)
        z2_norm = F.normalize(z2.detach(), p=2, dim=-1)
        
        p2_norm = F.normalize(p2, p=2, dim=-1)
        z1_norm = F.normalize(z1.detach(), p=2, dim=-1)
        
        # 2. Invariance loss：非对称 MSE (SimSiam 风格)
        inv_tau = 0.001
        L_inv1 = F.mse_loss(p1_norm, z2_norm) / inv_tau
        L_inv2 = F.mse_loss(p2_norm, z1_norm) / inv_tau
        L_inv = 0.5 * (L_inv1 + L_inv2)

        # 3. SIGReg 正则项：作用在目标特征 z1 上，确保特征分布不退化
        sig = self.sig_reg_module.to(device)
        x = z1.unsqueeze(0)  # (1, batch, hidden)
        L_sig = sig(x)
        if L_sig.dim() > 0:
            L_sig = L_sig.mean()
        
        # 4. 手动加权平衡
        loss = (1.0 - self.lamb) * L_inv + self.lamb * L_sig
        
        if self.training and torch.rand(1).item() < 0.01:
            print(f"[Asym Loss] L_inv: {L_inv.item():.6f}, L_sig: {L_sig.item():.6f}, "
                  f"lamb: {self.lamb:.6f}, Total: {loss.item():.6f}")

        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        def _one_forward():
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            z = self._pool_and_project(out.last_hidden_state)
            p = self.predictor(z)
            return z, p, out

        if sent_emb:
            # 推理模式：直接使用 BERT 的 [CLS] 输出，跳过 Projector (Input -> Encoder -> Features)
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            # 取 [CLS] token 的 embedding，不经过 projector
            cls_embedding = out.last_hidden_state[:, 0, :]
            
            if not return_dict:
                return (cls_embedding,)
            return BaseModelOutputWithPoolingAndCrossAttentions(
                pooler_output=cls_embedding,
                last_hidden_state=out.last_hidden_state,
                hidden_states=out.hidden_states,
            )

        # 训练：双路 Dropout 增广，同一 batch 前向两次
        z1, p1, out1 = _one_forward()
        z2, p2, out2 = _one_forward()
        
        # 计算 TexLeJEPA 损失 (非对称)
        loss = self._compute_texlejepa_loss(p1, z1, p2, z2)
        
        if not return_dict:
            # (loss, z1, z2) 形式，便于调试或自定义使用
            return (loss, z1, z2)
        
        # 训练模式：返回 SequenceClassifierOutput，Trainer 会直接读取其中的 loss
        return SequenceClassifierOutput(
            loss=loss,
            logits=z1,  # 将 z1 作为 logits（句向量），与其它模型风格保持一致
            hidden_states=out2.hidden_states if output_hidden_states else None,
            attentions=out2.attentions if output_attentions else None,
        )
