import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class OrthogonalConstraint(nn.Module):
    """
    软正交约束模块：通过损失函数鼓励语义维度的独立性
    使用软正交损失：L_orth = ||H^T H - I||_F²
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, semantic_reprs):
        """
        计算软正交损失而不是强制正交化
        Args:
            semantic_reprs: (batch_size, num_semantics, hidden_dim)
        Returns:
            semantic_reprs: 原样返回（不修改）
            orth_loss: 软正交损失
        """
        batch_size, num_semantics, hidden_dim = semantic_reprs.shape
        
        # 计算每个batch的软正交损失
        orth_losses = []
        for i in range(batch_size):
            # 取出单个batch的语义表示 (num_semantics, hidden_dim)
            batch_reprs = semantic_reprs[i]
            
            # 归一化
            normalized = F.normalize(batch_reprs, p=2, dim=1)
            
            # 计算Gram矩阵 H^T H
            gram_matrix = torch.mm(normalized, normalized.t())
            
            # 计算与单位矩阵的差异 ||H^T H - I||_F²
            identity = torch.eye(num_semantics, device=semantic_reprs.device)
            orth_loss = torch.norm(gram_matrix - identity, p='fro') ** 2
            
            orth_losses.append(orth_loss)
        
        # 平均软正交损失
        avg_orth_loss = torch.stack(orth_losses).mean()
        
        return semantic_reprs, avg_orth_loss


class SemanticDecomposer(nn.Module):
    """
    语义分解器：将句子表示分解为多个语义维度
    参考光学中白光通过棱镜分解为七色光的原理
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        
        # 可学习的分解矩阵
        self.decomposition_matrix = nn.Parameter(
            torch.randn(hidden_dim, num_semantics * hidden_dim) * 0.1
        )
        
        # 正交约束确保语义独立性
        self.orthogonal_constraint = OrthogonalConstraint()
        
        # 激活函数
        self.activation = nn.GELU()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分解矩阵的权重"""
        with torch.no_grad():
            # 使用Xavier初始化
            nn.init.xavier_uniform_(self.decomposition_matrix)
    
    def forward(self, sentence_repr):
        """
        将句子表示分解为多个语义表示
        Args:
            sentence_repr: (batch_size, hidden_dim) 句子表示
        Returns:
            semantic_reprs: (batch_size, num_semantics, hidden_dim) 分解后的语义表示
            orth_loss: 软正交损失
        """
        # 应用分解矩阵
        decomposed = torch.matmul(sentence_repr, self.decomposition_matrix)
        
        # 重塑为多个语义表示
        semantic_reprs = decomposed.view(-1, self.num_semantics, self.hidden_dim)
        
        # 应用层归一化
        semantic_reprs = self.layer_norm(semantic_reprs)
        
        # 应用激活函数
        semantic_reprs = self.activation(semantic_reprs)
        
        # 应用软正交约束，获取软正交损失
        semantic_reprs, orth_loss = self.orthogonal_constraint(semantic_reprs)
        
        return semantic_reprs, orth_loss


class SPR_Module(nn.Module):
    """
    自正则化（Self-Projection Regularization）模块
    参考CSE-SFP的SPR方法，不使用InfoNCE损失，而是使用自正则化
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # 投影层 f_proj: h -> z
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 预测层 f_pred: z -> p
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dropout层用于数据增强
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, semantic_repr):
        """
        对单个语义表示进行SPR处理
        Args:
            semantic_repr: (batch_size, hidden_dim) 语义表示 h
        Returns:
            processed_repr: 处理后的表示 p
            spr_loss: SPR自正则化损失
        """
        # Step 1: 编码句子 → 得到表示 h (输入)
        h = semantic_repr
        
        # Step 2: 投影 → z = f_proj(h)
        z = self.projection(h)
        
        # Step 3: 预测 → p = f_pred(z)
        p = self.prediction(z)
        
        # Step 4: SPR正则项 L_spr = ||p - h||^2
        # 计算预测结果p与原始语义表示h的L2距离平方
        spr_loss = F.mse_loss(p, h)
        
        # 返回预测结果p作为处理后的表示
        return p, spr_loss


class MultiSemanticSPR(nn.Module):
    """
    多语义SPR模型：结合语义分解和并行SPR处理
    实现软正交 + 自正则化组合损失函数：
    L = L_task + λ₁ Σᵢ L_spr,i + λ₂ ||H^T H - I||_F²
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7, lambda1: float = 1.0, lambda2: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        self.lambda1 = lambda1  # λ₁: 自正则化权重
        self.lambda2 = lambda2  # λ₂: 软正交权重
        
        # 语义分解器
        self.decomposer = SemanticDecomposer(hidden_dim, num_semantics)
        
        # 7个并行的SPR模块
        self.spr_modules = nn.ModuleList([
            SPR_Module(hidden_dim) for _ in range(num_semantics)
        ])
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_semantics * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 语义权重（可学习）
        self.semantic_weights = nn.Parameter(torch.ones(num_semantics) / num_semantics)
    
    def forward(self, sentence_repr):
        """
        多语义SPR前向传播
        实现软正交 + 自正则化组合损失函数
        Args:
            sentence_repr: (batch_size, hidden_dim) 句子表示
        Returns:
            final_repr: 融合后的最终表示
            total_loss: 总损失（包含SPR损失和软正交损失）
            semantic_spr_losses: 各语义维度的SPR损失
        """
        # 分解为多个语义表示，获取软正交损失
        semantic_reprs, orth_loss = self.decomposer(sentence_repr)
        
        # 并行SPR处理
        processed_reprs = []
        semantic_spr_losses = []
        
        for i, spr_module in enumerate(self.spr_modules):
            # 对每个语义表示进行SPR处理
            # h_i -> z_i = f_proj(h_i) -> p_i = f_pred(z_i)
            # L_spr_i = ||p_i - h_i||^2 (子语义自一致性)
            processed_repr, spr_loss = spr_module(semantic_reprs[:, i])
            processed_reprs.append(processed_repr)
            semantic_spr_losses.append(spr_loss)
        
        # 融合处理后的表示
        fused_repr = torch.cat(processed_reprs, dim=-1)
        final_repr = self.fusion_layer(fused_repr)
        
        # 计算加权总SPR损失 Σᵢ L_spr,i
        weighted_losses = [weight * loss for weight, loss in zip(self.semantic_weights, semantic_spr_losses)]
        total_spr_loss = torch.stack(weighted_losses).sum()
        
        # 组合损失函数：L = L_task + λ₁ Σᵢ L_spr,i + λ₂ ||H^T H - I||_F²
        # 这里L_task为0（无监督学习），所以总损失为：
        total_loss = self.lambda1 * total_spr_loss + self.lambda2 * orth_loss
        
        return final_repr, total_loss, semantic_spr_losses


def prism_decomp_init(cls, config):
    """
    棱镜分解模型初始化函数
    """
    # 初始化多语义SPR模块，使用软正交 + 自正则化组合损失函数
    cls.multisemantic_spr = MultiSemanticSPR(
        config.hidden_size, 
        num_semantics=7,
        lambda1=1.0,    # λ₁: 自正则化权重
        lambda2=0.01    # λ₂: 软正交权重（保持轻度正交）
    )
    
    # 设置语义维度数量
    cls.num_semantics = 7
    
    # 定义7个语义维度
    cls.semantic_dimensions = [
        "情感语义", "主题语义", "语法语义", 
        "时序语义", "空间语义", "因果语义", "程度语义"
    ]
    
    cls.init_weights()


def prism_decomp_forward(cls,
                        encoder,
                        input_ids=None,
                        attention_mask=None,
                        token_type_ids=None,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        labels=None,
                        return_dict=None,
):
    """
    棱镜分解前向传播函数
    实现多语义句子表示学习训练逻辑
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    # BERT编码
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    # 提取句子表示
    if hasattr(cls, 'model_args') and cls.model_args is not None and hasattr(cls.model_args, 'mask_embedding_sentence') and cls.model_args.mask_embedding_sentence:
        # 找到[MASK]位置并提取其表示
        mask_token_id = cls.config.mask_token_id if hasattr(cls.config, 'mask_token_id') else 103
        
        # 为每个句子找到mask token位置
        batch_size_flat = input_ids.size(0)
        sentence_repr_list = []
        
        for i in range(batch_size_flat):
            mask_positions = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) > 0:
                # 使用第一个[MASK]的位置
                mask_pos = mask_positions[0].item()
                sentence_repr_list.append(outputs.last_hidden_state[i, mask_pos, :])
            else:
                # 如果没有[MASK]，回退到[CLS]
                sentence_repr_list.append(outputs.last_hidden_state[i, 0, :])
        
        sentence_repr = torch.stack(sentence_repr_list, dim=0)  # (batch_size * num_sent, hidden_dim)
    else:
        # 原始方式：提取CLS token作为句子表示
        sentence_repr = outputs.last_hidden_state[:, 0, :]  # (batch_size * num_sent, hidden_dim)
    
    # 重塑为 (batch_size, num_sent, hidden_dim)
    sentence_repr = sentence_repr.view(batch_size, num_sent, -1)
    
    # 单视图：对单个句子进行多语义SPR处理
    # sentence_repr shape: (batch_size, num_sent, hidden_dim)
    # num_sent 现在为 1（单视图）
    final_repr, total_spr_loss, semantic_spr_losses = cls.multisemantic_spr(sentence_repr[:, 0])
    
    # 直接使用总SPR损失（不再平均多个视图）
    avg_spr_loss = total_spr_loss
    
    # 使用处理后的表示作为输出
    logits = final_repr

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((avg_spr_loss,) + output) if avg_spr_loss is not None else output
    
    return SequenceClassifierOutput(
        loss=avg_spr_loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
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
):
    """
    句子嵌入前向传播（用于评估）
    返回融合后的多语义句子表示
    支持SentEval STS任务：每个句子独立处理
    
    输入:
        input_ids: (batch_size, seq_len) - 一批句子，每个句子已经用模板包裹
        attention_mask: (batch_size, seq_len)
    
    输出:
        pooler_output: (batch_size, hidden_dim) - 每个句子的融合表示
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # BERT编码
    # input_ids shape: (batch_size, seq_len) - 例如 (8, 128)
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    # 提取句子表示 (从每个句子的MASK位置)
    if hasattr(cls, 'model_args') and cls.model_args is not None and hasattr(cls.model_args, 'mask_embedding_sentence') and cls.model_args.mask_embedding_sentence:
        mask_token_id = cls.config.mask_token_id if hasattr(cls.config, 'mask_token_id') else 103
        
        # 为每个句子提取mask位置
        # input_ids.size(0) = batch_size (批次中的句子数)
        sentence_repr_list = []
        for i in range(input_ids.size(0)):
            mask_pos_i = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos_i) > 0:
                sentence_repr_list.append(outputs.last_hidden_state[i, mask_pos_i[0].item(), :])
            else:
                sentence_repr_list.append(outputs.last_hidden_state[i, 0, :])
        sentence_repr = torch.stack(sentence_repr_list, dim=0)  # (batch_size, hidden_dim)
    else:
        sentence_repr = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
    
    # 多语义SPR处理（仅前向传播，不计算损失）
    with torch.no_grad():
        # 分解时忽略软正交损失
        semantic_reprs, _ = cls.multisemantic_spr.decomposer(sentence_repr)
        
        # 处理每个语义维度
        processed_reprs = []
        for i, spr_module in enumerate(cls.multisemantic_spr.spr_modules):
            # 只进行前向传播，不计算损失
            projected = spr_module.projection(semantic_reprs[:, i])
            predicted = spr_module.prediction(projected)
            processed_reprs.append(predicted)
        
        # 融合处理后的表示
        fused_repr = torch.cat(processed_reprs, dim=-1)
        pooler_output = cls.multisemantic_spr.fusion_layer(fused_repr)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForPrismDecomp(BertPreTrainedModel):
    """
    BERT for Prism-like Decomposition (PrismDecomp)
    实现基于棱镜分解的多语义句子表示学习
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [
        r"multisemantic_spr\.",
    ]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)

        prism_decomp_init(self, config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        重写 from_pretrained 方法以抑制预期的警告
        """
        import warnings
        import logging
        
        # 临时抑制相关警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")
            warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used.*")
            
            # 临时降低 transformers 日志级别
            old_level = logging.getLogger("transformers.modeling_utils").level
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            
            try:
                model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            finally:
                # 恢复日志级别
                logging.getLogger("transformers.modeling_utils").setLevel(old_level)
        
        return model

    def forward(self,
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
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return prism_decomp_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
