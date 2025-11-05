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
    
    def forward(self, semantic_reprs, compute_loss=True):
        """
        计算软正交损失而不是强制正交化
        Args:
            semantic_reprs: (batch_size, num_semantics, hidden_dim)
            compute_loss: 是否计算软正交损失，默认True（训练时），False（评估时）
        Returns:
            semantic_reprs: 原样返回（不修改）
            orth_loss: 软正交损失（如果compute_loss=False则返回0）
        """
        if not compute_loss:
            # 评估时跳过损失计算，直接返回0
            return semantic_reprs, torch.tensor(0.0, device=semantic_reprs.device)
        
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
    
    def forward(self, sentence_repr, compute_orth_loss=True):
        """
        将句子表示分解为多个语义表示
        Args:
            sentence_repr: (batch_size, hidden_dim) 句子表示
            compute_orth_loss: 是否计算软正交损失，默认True（训练时），False（评估时）
        Returns:
            semantic_reprs: (batch_size, num_semantics, hidden_dim) 分解后的语义表示
            orth_loss: 软正交损失
        """
        # 应用分解矩阵
        decomposed = torch.matmul(sentence_repr, self.decomposition_matrix)
        
        # 重塑为多个语义表示
        semantic_reprs = decomposed.view(-1, self.num_semantics, self.hidden_dim)
        
        # 应用层归一化
        # semantic_reprs = self.layer_norm(semantic_reprs)
        
        # 应用激活函数
        semantic_reprs = self.activation(semantic_reprs)
        
        # 应用软正交约束，获取软正交损失
        semantic_reprs, orth_loss = self.orthogonal_constraint(semantic_reprs, compute_loss=compute_orth_loss)
        
        return semantic_reprs, orth_loss


class SPR_Module(nn.Module):
    """
    投影预测模块：对子语义表示进行投影和预测
    用于生成正样本h+的组件
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
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
    
    def forward(self, semantic_repr):
        """
        对单个语义表示进行投影预测处理
        Args:
            semantic_repr: (batch_size, hidden_dim) 语义表示 h
        Returns:
            processed_repr: 归一化后的预测表示 p_norm
        """
        # Step 1: 投影 → z = f_proj(h)
        z = self.projection(semantic_repr)
        
        # Step 2: 预测 → p = f_pred(z)
        p = self.prediction(z)
        
        # Step 3: 归一化表示
        # p_norm = F.normalize(p, p=2, dim=-1)
        
        # 返回归一化后的预测结果作为处理后的表示
        return p


class Similarity(nn.Module):
    """
    相似度计算模块：用于InfoNCE损失
    计算余弦相似度并应用温度参数
    """
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MultiSemanticSPR(nn.Module):
    """
    多语义SPR模型：结合语义分解和并行投影预测处理
    生成正样本h+用于InfoNCE对比学习
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7, lambda2: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        self.lambda2 = lambda2  # λ₂: 软正交权重
        
        # 语义分解器
        self.decomposer = SemanticDecomposer(hidden_dim, num_semantics)
        
        # 7个并行的投影预测模块
        self.spr_modules = nn.ModuleList([
            SPR_Module(hidden_dim) for _ in range(num_semantics)
        ])
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_semantics * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, sentence_repr, compute_orth_loss=True):
        """
        多语义投影预测前向传播
        生成正样本h+和计算软正交损失
        Args:
            sentence_repr: (batch_size, hidden_dim) 锚点样本h
            compute_orth_loss: 是否计算软正交损失，默认True（训练时），False（评估时）
        Returns:
            h_plus: (batch_size, hidden_dim) 融合后的正样本表示（归一化）
            orth_loss: 软正交损失
        """
        # 分解为多个语义表示，获取软正交损失
        semantic_reprs, orth_loss = self.decomposer(sentence_repr, compute_orth_loss=compute_orth_loss)
        
        # 并行投影预测处理
        processed_reprs = []
        for i, spr_module in enumerate(self.spr_modules):
            # 对每个语义表示进行投影预测处理
            # h_i -> z_i = f_proj(h_i) -> p_i = f_pred(z_i)
            processed_repr = spr_module(semantic_reprs[:, i])
            processed_reprs.append(processed_repr)
        
        # 融合处理后的表示
        fused_repr = torch.cat(processed_reprs, dim=-1)
        h_plus = self.fusion_layer(fused_repr)
        # 归一化最终表示，便于余弦相似度计算
        h_plus = F.normalize(h_plus, p=2, dim=-1)
        
        return h_plus, orth_loss


def prism_decomp_init(cls, config, temperature=0.05, lambda2=0.01):
    """
    棱镜分解模型初始化函数
    Args:
        config: 模型配置
        temperature: InfoNCE损失的温度参数（默认0.05）
        lambda2: 软正交损失权重（默认0.01）
    """
    # 初始化多语义SPR模块，用于生成正样本h+
    cls.multisemantic_spr = MultiSemanticSPR(
        config.hidden_size, 
        num_semantics=7,
        lambda2=lambda2  # λ₂: 软正交权重
    )
    
    # 初始化相似度计算模块（用于InfoNCE损失）
    cls.similarity = Similarity(temp=temperature)
    
    # 设置语义维度数量
    cls.num_semantics = 7
    
    # 定义7个语义维度
    cls.semantic_dimensions = [
        "情感语义", "主题语义", "语法语义", 
        "时序语义", "空间语义", "因果语义", "程度语义"
    ]
    
    # 存储损失函数权重
    cls.lambda2 = lambda2
    cls.temperature = temperature
    
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
    
    # 提取锚点样本h：从MASK位置提取的原始表示
    # sentence_repr shape: (batch_size, num_sent, hidden_dim)
    # num_sent 现在为 1（单视图）
    anchor_h = sentence_repr[:, 0]  # (batch_size, hidden_dim)
    
    # 归一化锚点样本
    anchor_h = F.normalize(anchor_h, p=2, dim=-1)
    
    # 通过多语义SPR生成正样本h+
    # h+ = 融合(投影预测(分解(h)))
    h_plus, orth_loss = cls.multisemantic_spr(anchor_h)
    
    # 计算InfoNCE损失
    # 正样本对：anchor_h[i] 与 h_plus[i]
    # 负样本对：anchor_h[i] 与 anchor_h[j]（i != j，批次内其他样本的锚点）
    
    # 构建相似度矩阵：(batch_size, batch_size + 1)
    # 第一列：正样本对相似度（anchor_h[i] 与 h_plus[i]）
    # 后续列：负样本对相似度（anchor_h[i] 与 anchor_h[j]）
    
    # 由于anchor_h和h_plus都已归一化，直接计算点积并应用温度
    # 计算正样本对相似度（anchor_h[i] 与 h_plus[i]）
    pos_sim = (anchor_h * h_plus).sum(dim=-1, keepdim=True) / cls.temperature  # (batch_size, 1)
    
    # 计算负样本对相似度（anchor_h[i] 与 anchor_h[j], i != j）
    # 使用anchor_h作为负样本（批次内其他样本的锚点）
    # 由于anchor_h已归一化，点积就是余弦相似度
    neg_sim = torch.mm(anchor_h, anchor_h.t()) / cls.temperature  # (batch_size, batch_size)
    
    # 将对角线位置设为负无穷（排除自己作为负样本）
    # 对角线元素是anchor_h[i]与自己的相似度（=1.0），不应该作为负样本
    # 这会导致训练目标错误：模型学习让anchor_h与自己不相似，而不是学习语义表示
    eye_mask = torch.eye(batch_size, device=anchor_h.device, dtype=torch.bool)
    neg_sim = neg_sim.masked_fill(eye_mask, float('-inf'))
    
    # 数值稳定性保护：防止softmax溢出
    neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)
    # 注意：clamp后需要重新设置对角线（因为clamp可能会改变-inf）
    neg_sim = neg_sim.masked_fill(eye_mask, float('-inf'))
    
    # 组合相似度矩阵：[正样本对, 负样本对]
    cos_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, batch_size + 1)
    
    # 标签：第一列（索引0）是正样本对
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_h.device)
    
    # 计算InfoNCE损失
    loss_fct = nn.CrossEntropyLoss()
    infonce_loss = loss_fct(cos_sim, labels)
    
    # 总损失：L = L_infonce + λ₂ * L_orthogonal
    total_loss = infonce_loss + cls.lambda2 * orth_loss
    
    # 使用正样本h+作为输出表示
    logits = h_plus

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output
    
    return SequenceClassifierOutput(
        loss=total_loss,
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
        # 直接调用multisemantic_spr，无需预先归一化（内部会对最终输出归一化）
        # 评估时跳过软正交损失计算以提高效率
        pooler_output, _ = cls.multisemantic_spr(sentence_repr, compute_orth_loss=False)
        # h_plus已经是归一化的

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

        # 从model_args获取温度参数和lambda2参数
        temperature = getattr(self.model_args, 'temperature', 0.05) if self.model_args else 0.05
        lambda2 = getattr(self.model_args, 'lambda2', 0.01) if self.model_args else 0.01
        
        prism_decomp_init(self, config, temperature=temperature, lambda2=lambda2)

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
