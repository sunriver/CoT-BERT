import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class OrthogonalConstraint(nn.Module):
    """
    正交约束模块：确保语义维度的独立性
    通过Gram-Schmidt正交化过程实现
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, semantic_reprs):
        """
        对语义表示应用正交约束
        Args:
            semantic_reprs: (batch_size, num_semantics, hidden_dim)
        Returns:
            orthogonal_reprs: 正交化后的语义表示
        """
        batch_size, num_semantics, hidden_dim = semantic_reprs.shape
        
        # 重塑为 (batch_size * num_semantics, hidden_dim) 进行批量处理
        flat_reprs = semantic_reprs.view(-1, hidden_dim)
        
        # 对每个batch分别进行正交化
        orthogonal_reprs = []
        for i in range(batch_size):
            batch_reprs = flat_reprs[i * num_semantics:(i + 1) * num_semantics]
            orthogonal_batch = self._gram_schmidt(batch_reprs)
            orthogonal_reprs.append(orthogonal_batch)
        
        # 重新组合
        orthogonal_reprs = torch.stack(orthogonal_reprs, dim=0)
        return orthogonal_reprs
    
    def _gram_schmidt(self, vectors):
        """
        Gram-Schmidt正交化过程
        Args:
            vectors: (num_semantics, hidden_dim)
        Returns:
            orthogonal_vectors: 正交化后的向量
        """
        num_semantics, hidden_dim = vectors.shape
        orthogonal_vectors = torch.zeros_like(vectors)
        
        for i in range(num_semantics):
            # 取当前向量
            v = vectors[i]
            
            # 减去与前面向量的投影
            for j in range(i):
                if torch.norm(orthogonal_vectors[j]) > self.eps:
                    projection = torch.dot(v, orthogonal_vectors[j]) / torch.dot(orthogonal_vectors[j], orthogonal_vectors[j])
                    v = v - projection * orthogonal_vectors[j]
            
            # 归一化
            norm = torch.norm(v)
            if norm > self.eps:
                orthogonal_vectors[i] = v / norm
            else:
                # 如果向量太小，使用随机向量
                orthogonal_vectors[i] = torch.randn_like(v)
                orthogonal_vectors[i] = orthogonal_vectors[i] / torch.norm(orthogonal_vectors[i])
        
        return orthogonal_vectors


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
        """
        # 应用分解矩阵
        decomposed = torch.matmul(sentence_repr, self.decomposition_matrix)
        
        # 重塑为多个语义表示
        semantic_reprs = decomposed.view(-1, self.num_semantics, self.hidden_dim)
        
        # 应用层归一化
        semantic_reprs = self.layer_norm(semantic_reprs)
        
        # 应用激活函数
        semantic_reprs = self.activation(semantic_reprs)
        
        # 应用正交约束
        semantic_reprs = self.orthogonal_constraint(semantic_reprs)
        
        return semantic_reprs


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
    为每个语义表示分别计算SPR自正则化损失
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        
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
        Args:
            sentence_repr: (batch_size, hidden_dim) 句子表示
        Returns:
            final_repr: 融合后的最终表示
            total_spr_loss: 总SPR损失
            semantic_spr_losses: 各语义维度的SPR损失
        """
        # 分解为多个语义表示
        semantic_reprs = self.decomposer(sentence_repr)
        
        # 并行SPR处理
        processed_reprs = []
        semantic_spr_losses = []
        
        for i, spr_module in enumerate(self.spr_modules):
            # 对每个语义表示进行SPR处理
            # h_i -> z_i = f_proj(h_i) -> p_i = f_pred(z_i)
            # L_spr_i = ||p_i - h_i||^2
            processed_repr, spr_loss = spr_module(semantic_reprs[:, i])
            processed_reprs.append(processed_repr)
            semantic_spr_losses.append(spr_loss)
        
        # 融合处理后的表示
        fused_repr = torch.cat(processed_reprs, dim=-1)
        final_repr = self.fusion_layer(fused_repr)
        
        # 计算加权总SPR损失
        # L_total = Σ(i=1 to 7) λ_i * L_spr_i
        weighted_losses = [weight * loss for weight, loss in zip(self.semantic_weights, semantic_spr_losses)]
        total_spr_loss = torch.stack(weighted_losses).sum()
        
        return final_repr, total_spr_loss, semantic_spr_losses


def prism_decomp_init(cls, config):
    """
    棱镜分解模型初始化函数
    """
    # 初始化多语义SPR模块
    cls.multisemantic_spr = MultiSemanticSPR(config.hidden_size, num_semantics=7)
    
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

    # 提取CLS token作为句子表示
    sentence_repr = outputs.last_hidden_state[:, 0, :]  # (batch_size * num_sent, hidden_dim)
    
    # 重塑为 (batch_size, num_sent, hidden_dim)
    sentence_repr = sentence_repr.view(batch_size, num_sent, -1)
    
    # 对每个句子进行多语义SPR处理
    final_reprs = []
    total_spr_losses = []
    semantic_spr_losses_list = []
    
    for i in range(num_sent):
        final_repr, total_spr_loss, semantic_spr_losses = cls.multisemantic_spr(sentence_repr[:, i])
        final_reprs.append(final_repr)
        total_spr_losses.append(total_spr_loss)
        semantic_spr_losses_list.append(semantic_spr_losses)
    
    # 平均SPR损失
    avg_spr_loss = torch.stack(total_spr_losses).mean()
    
    # 使用第一个句子的表示作为输出（用于无监督学习）
    logits = final_reprs[0]

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
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

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

    # 提取CLS token作为句子表示
    sentence_repr = outputs.last_hidden_state[:, 0, :]
    
    # 多语义SPR处理（仅前向传播，不计算损失）
    with torch.no_grad():
        semantic_reprs = cls.multisemantic_spr.decomposer(sentence_repr)
        
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
