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


class SignalEnhancer(nn.Module):
    """
    信号增强模块：对每个子语义h_i进行信号增强
    增强弱信号，防止信号弱的语义被忽略
    实现：h_i_enhanced = h_i * learnable_weight + residual_connection
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 可学习的增强权重
        self.enhance_weight = nn.Parameter(torch.ones(1))
        
        # 残差连接层，用于增强信号
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        with torch.no_grad():
            # 增强权重初始化为1.0（不改变原始信号）
            self.enhance_weight.fill_(1.0)
            # 残差层使用Xavier初始化
            for layer in self.residual:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, h_i):
        """
        对子语义表示进行信号增强
        Args:
            h_i: (batch_size, hidden_dim) 子语义表示
        Returns:
            h_i_enhanced: (batch_size, hidden_dim) 增强后的子语义表示
        """
        # 信号增强：h_i_enhanced = h_i * learnable_weight + residual(h_i)
        enhanced = h_i * self.enhance_weight + self.residual(h_i)
        return enhanced


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


class AttentionFusion(nn.Module):
    """
    注意力融合器：使用多头注意力机制融合子语义表示
    最大化保留各个子语义信息，生成融合后的句子增强语义表示h+
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        self.num_heads = num_heads
        
        # 多头注意力机制
        # 使用query作为原始句子表示h，key和value作为子语义表示h_i+
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False  # PyTorch MultiheadAttention默认使用(batch, seq, features)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        with torch.no_grad():
            # 输出投影层使用Xavier初始化
            for layer in self.output_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, h, h_i_plus_list):
        """
        使用注意力机制融合子语义表示
        Args:
            h: (batch_size, hidden_dim) 原始句子表示，作为query
            h_i_plus_list: list of (batch_size, hidden_dim) 投影预测后的子语义表示列表
        Returns:
            h_plus: (batch_size, hidden_dim) 融合后的句子增强语义表示
        """
        batch_size = h.size(0)
        
        # 将h_i_plus_list转换为tensor: (num_semantics, batch_size, hidden_dim)
        # PyTorch MultiheadAttention需要(seq_len, batch, features)格式
        h_i_plus_tensor = torch.stack(h_i_plus_list, dim=0)  # (num_semantics, batch_size, hidden_dim)
        
        # 将h转换为query格式: (1, batch_size, hidden_dim)
        # 使用h作为query，从所有子语义h_i+中提取信息
        h_query = h.unsqueeze(0)  # (1, batch_size, hidden_dim)
        
        # 多头注意力融合
        # query: h (原始句子表示)
        # key, value: h_i+ (所有子语义表示)
        # 这样可以让原始表示h从各个子语义中提取信息，最大化保留各子语义
        attn_output, attn_weights = self.multihead_attn(
            query=h_query,
            key=h_i_plus_tensor,
            value=h_i_plus_tensor
        )
        # attn_output: (1, batch_size, hidden_dim)
        # attn_weights: (batch_size, 1, num_semantics) 注意力权重，显示各子语义的贡献
        
        # 移除seq维度
        attn_output = attn_output.squeeze(0)  # (batch_size, hidden_dim)
        
        # 残差连接：h + attention_output
        fused = h + attn_output
        
        # 层归一化
        fused = self.layer_norm(fused)
        
        # 输出投影
        h_plus = self.output_proj(fused)
        
        # 归一化最终表示，便于余弦相似度计算
        h_plus = F.normalize(h_plus, p=2, dim=-1)
        
        return h_plus


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
    多语义SPR模型：结合语义分解、信号增强和并行投影预测处理
    生成正样本h+用于InfoNCE对比学习
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7, lambda2: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        self.lambda2 = lambda2  # λ₂: 软正交权重
        
        # 语义分解器
        self.decomposer = SemanticDecomposer(hidden_dim, num_semantics)
        
        # 7个并行的信号增强模块
        self.signal_enhancers = nn.ModuleList([
            SignalEnhancer(hidden_dim) for _ in range(num_semantics)
        ])
        
        # 7个并行的投影预测模块
        self.spr_modules = nn.ModuleList([
            SPR_Module(hidden_dim) for _ in range(num_semantics)
        ])
    
    def forward(self, sentence_repr, compute_orth_loss=True, return_all_semantics=False):
        """
        多语义投影预测前向传播
        生成正样本h+和计算软正交损失
        Args:
            sentence_repr: (batch_size, hidden_dim) 锚点样本h
            compute_orth_loss: 是否计算软正交损失，默认True（训练时），False（评估时）
            return_all_semantics: 是否返回所有子语义h_i和h_i+，默认False
        Returns:
            h_plus: (batch_size, hidden_dim) 融合后的正样本表示（归一化）
            orth_loss: 软正交损失
            如果return_all_semantics=True，还返回：
            h_i_list: list of (batch_size, hidden_dim) 所有子语义h_i
            h_i_plus_list: list of (batch_size, hidden_dim) 所有子语义h_i+
        """
        # 分解为多个语义表示，获取软正交损失
        semantic_reprs, orth_loss = self.decomposer(sentence_repr, compute_orth_loss=compute_orth_loss)
        # semantic_reprs: (batch_size, num_semantics, hidden_dim)
        
        # 存储所有子语义h_i和h_i+
        h_i_list = []
        h_i_plus_list = []
        
        # 对每个子语义进行信号增强和投影预测处理
        for i in range(self.num_semantics):
            # 提取第i个子语义 h_i: (batch_size, hidden_dim)
            h_i = semantic_reprs[:, i]
            
            # 信号增强：h_i -> h_i_enhanced
            h_i_enhanced = self.signal_enhancers[i](h_i)
            
            # 投影预测：h_i_enhanced -> h_i+
            h_i_plus = self.spr_modules[i](h_i_enhanced)
            
            # 归一化h_i和h_i+，便于后续InfoNCE计算
            h_i = F.normalize(h_i, p=2, dim=-1)
            h_i_plus = F.normalize(h_i_plus, p=2, dim=-1)
            
            # 存储
            h_i_list.append(h_i)
            h_i_plus_list.append(h_i_plus)
        
        if return_all_semantics:
            # 返回所有子语义信息（用于子语义InfoNCE损失计算）
            return h_i_list, h_i_plus_list, orth_loss
        else:
            # 为了兼容性，返回None（融合将在AttentionFusion中完成）
            return None, orth_loss


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
    
    # 初始化注意力融合器，用于融合子语义表示
    cls.attention_fusion = AttentionFusion(
        hidden_dim=config.hidden_size,
        num_semantics=7,
        num_heads=8
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
    
    # 通过多语义SPR获取所有子语义h_i和h_i+
    # h_i_list: list of (batch_size, hidden_dim) 所有子语义h_i
    # h_i_plus_list: list of (batch_size, hidden_dim) 所有子语义h_i+
    h_i_list, h_i_plus_list, orth_loss = cls.multisemantic_spr(
        anchor_h, 
        compute_orth_loss=True, 
        return_all_semantics=True
    )
    
    # 使用注意力融合器融合所有子语义h_i+，得到综合语义表示h+
    h_plus = cls.attention_fusion(anchor_h, h_i_plus_list)  # (batch_size, hidden_dim)
    
    # ========== 计算子语义InfoNCE损失 ==========
    # 对每个子语义维度i（0-6）：
    # - 锚点：h_i[batch_idx]
    # - 正样本：h_i+[batch_idx]
    # - 负样本：批次内其他句子的对应子语义h_i[j] (j != batch_idx)
    
    loss_fct = nn.CrossEntropyLoss()
    semantic_infonce_losses = []
    
    for i in range(cls.num_semantics):
        h_i = h_i_list[i]  # (batch_size, hidden_dim)
        h_i_plus = h_i_plus_list[i]  # (batch_size, hidden_dim)
        
        # 计算正样本对相似度（h_i[j] 与 h_i+[j]）
        pos_sim = (h_i * h_i_plus).sum(dim=-1, keepdim=True) / cls.temperature  # (batch_size, 1)
        pos_sim = torch.clamp(pos_sim, min=-50.0, max=50.0)
        
        # 计算负样本对相似度（h_i[j] 与 h_i[k], k != j）
        # 使用批次内其他句子的对应子语义h_i作为负样本
        neg_sim = torch.mm(h_i, h_i.t()) / cls.temperature  # (batch_size, batch_size)
        
        # 将对角线位置设为负无穷（排除自己作为负样本）
        eye_mask = torch.eye(batch_size, device=h_i.device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(eye_mask, float('-inf'))
        
        # 数值稳定性保护
        neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)
        neg_sim = neg_sim.masked_fill(eye_mask, float('-inf'))
        
        # 组合相似度矩阵：[正样本对, 负样本对]
        cos_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, batch_size + 1)
        
        # 标签：第一列（索引0）是正样本对
        labels = torch.zeros(batch_size, dtype=torch.long, device=h_i.device)
        
        # 计算InfoNCE损失
        semantic_infonce_loss = loss_fct(cos_sim, labels)
        semantic_infonce_losses.append(semantic_infonce_loss)
    
    # 平均所有子语义的InfoNCE损失
    avg_semantic_infonce_loss = torch.stack(semantic_infonce_losses).mean()
    
    # ========== 计算综合语义InfoNCE损失 ==========
    # - 锚点：h[batch_idx]
    # - 正样本：h+[batch_idx]（注意力融合后的结果）
    # - 负样本：批次内其他句子的h[j] (j != batch_idx)
    
    # 计算正样本对相似度（anchor_h[i] 与 h_plus[i]）
    pos_sim_global = (anchor_h * h_plus).sum(dim=-1, keepdim=True) / cls.temperature  # (batch_size, 1)
    pos_sim_global = torch.clamp(pos_sim_global, min=-50.0, max=50.0)
    
    # 计算负样本对相似度（anchor_h[i] 与 anchor_h[j], i != j）
    neg_sim_global = torch.mm(anchor_h, anchor_h.t()) / cls.temperature  # (batch_size, batch_size)
    
    # 将对角线位置设为负无穷
    eye_mask_global = torch.eye(batch_size, device=anchor_h.device, dtype=torch.bool)
    neg_sim_global = neg_sim_global.masked_fill(eye_mask_global, float('-inf'))
    
    # 数值稳定性保护
    neg_sim_global = torch.clamp(neg_sim_global, min=-50.0, max=50.0)
    neg_sim_global = neg_sim_global.masked_fill(eye_mask_global, float('-inf'))
    
    # 组合相似度矩阵：[正样本对, 负样本对]
    cos_sim_global = torch.cat([pos_sim_global, neg_sim_global], dim=1)  # (batch_size, batch_size + 1)
    
    # 标签：第一列（索引0）是正样本对
    labels_global = torch.zeros(batch_size, dtype=torch.long, device=anchor_h.device)
    
    # 计算综合语义InfoNCE损失
    global_infonce_loss = loss_fct(cos_sim_global, labels_global)
    
    # ========== 总损失 ==========
    # L = 平均(所有子语义InfoNCE) + 综合语义InfoNCE + λ₂ * L_orthogonal
    total_loss = avg_semantic_infonce_loss + global_infonce_loss + cls.lambda2 * orth_loss
    
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
        # 归一化句子表示
        sentence_repr_norm = F.normalize(sentence_repr, p=2, dim=-1)
        
        # 获取所有子语义h_i和h_i+
        h_i_list, h_i_plus_list, _ = cls.multisemantic_spr(
            sentence_repr_norm,
            compute_orth_loss=False,
            return_all_semantics=True
        )
        
        # 使用注意力融合器融合所有子语义h_i+，得到综合语义表示h+
        pooler_output = cls.attention_fusion(sentence_repr_norm, h_i_plus_list)
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
