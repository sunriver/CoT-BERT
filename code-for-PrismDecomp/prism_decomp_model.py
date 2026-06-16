import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from template_supervision_losses import ThemeVectorSupervisionLoss
from theme_compressor import (
    SharedThemeCompressor,
    load_compressor_checkpoint,
)
from cot_prompt_utils import uses_cot_multi_view


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


class ResidualFusion(nn.Module):
    """
    残差融合器：将所有增强后的子语义与原始h进行残差连接
    直接放大子语义信号，保留原始信息
    
    实现：h+ = h + α * sum(w_i * h_i_enhanced)
    其中w_i是可学习的权重，α是融合权重
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7, fusion_weight: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        
        # 可学习的子语义权重：每个子语义的重要性
        self.semantic_weights = nn.Parameter(torch.ones(num_semantics) / num_semantics)
        
        # 融合权重：控制残差连接的强度
        self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        with torch.no_grad():
            # 子语义权重初始化为均匀分布
            self.semantic_weights.fill_(1.0 / self.num_semantics)
            # 融合权重初始化为给定值
            self.fusion_weight.fill_(0.5)
    
    def forward(self, h, h_i_enhanced_list):
        """
        残差融合：将增强后的子语义与原始h进行残差连接
        Args:
            h: (batch_size, hidden_dim) 原始句子表示
            h_i_enhanced_list: list of (batch_size, hidden_dim) 增强后的子语义表示列表
        Returns:
            h_plus: (batch_size, hidden_dim) 融合后的句子增强语义表示
        """
        # 对子语义权重应用softmax归一化
        weights = F.softmax(self.semantic_weights, dim=0)  # (num_semantics,)
        
        # 加权融合所有增强后的子语义
        fused_semantics = torch.zeros_like(h)  # (batch_size, hidden_dim)
        for i, h_i_enhanced in enumerate(h_i_enhanced_list):
            fused_semantics += weights[i] * h_i_enhanced
        
        # 残差连接：h+ = h + α * fused_semantics
        # 通过残差连接放大子语义信号，同时保留原始h的信息
        h_plus = h + torch.clamp(self.fusion_weight, 0.0, 2.0) * fused_semantics
        
        # 不提前归一化，保持原始信号，只在计算相似度时归一化
        
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


class MLPLayer(nn.Module):
    """CoT summary MASK 表示投影头。"""

    def __init__(self, config, scale=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * scale, config.hidden_size * scale)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        return self.activation(self.dense(features))


def _uses_cot(cls):
    ma = getattr(cls, "model_args", None)
    if ma is None or not getattr(ma, "mask_embedding_sentence", False):
        return False
    return getattr(ma, "mask_num", 1) >= 2 and bool(
        getattr(ma, "mask_embedding_sentence_different_template", "")
    )


def denoising(cls, encoder, template, type="pos-1", device="cuda", evaluation=False):
    """CoT delta 去噪：按可变句子长度估计模板噪声。"""
    freeze = getattr(cls.model_args, "mask_embedding_sentence_delta_freeze", False)
    with torch.set_grad_enabled(not freeze and not evaluation):
        if type == "pos-1":
            bs, es = cls.bs, cls.es
        elif type == "pos-2":
            bs, es = cls.bs2, cls.es2
        elif type == "neg-1":
            bs, es = cls.bs3, cls.es3
        else:
            raise ValueError(f"unknown denoising type {type}")

        input_ids, attention_mask = [], []
        for i in range(cls.total_length - len(template) + 1):
            input_ids.append(
                [template[0]]
                + bs
                + [cls.pad_token_id] * i
                + es
                + [template[-1]]
                + [cls.pad_token_id] * (cls.total_length - len(template) - i)
            )
            attention_mask.append(
                [1] * (len(template) + i)
                + [0] * (cls.total_length - len(template) - i)
            )

        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)

        mask = input_ids == cls.mask_token_id
        ctx = torch.no_grad() if evaluation else torch.enable_grad()
        with ctx:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            noise = outputs.last_hidden_state[mask]

        noise = noise.view(-1, cls.mask_num, noise.shape[-1])
        return noise, len(template)


def _extract_cot_pooler(cls, encoder, input_ids, attention_mask, evaluation=False):
    """
    双 MASK 提取 + delta 去噪 + cot_mlp。
    返回 pooler_output: (batch_size, num_sent, mask_num, hidden_dim)
    """
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
    flat_ids = input_ids.view(-1, input_ids.size(-1))
    flat_mask = attention_mask.view(-1, attention_mask.size(-1))

    ma = cls.model_args
    if getattr(ma, "mask_embedding_sentence_delta", False) and (
        not evaluation or not getattr(ma, "mask_embedding_sentence_delta_no_delta_eval", True)
    ):
        noise1, template_length1 = denoising(
            cls, encoder, cls.mask_embedding_template, type="pos-1",
            device=input_ids.device, evaluation=evaluation,
        )
        noise2, template_length2 = None, None
        noise3, template_length3 = None, None
        if getattr(ma, "mask_embedding_sentence_different_template", ""):
            noise2, template_length2 = denoising(
                cls, encoder, cls.mask_embedding_template2, type="pos-2",
                device=input_ids.device, evaluation=evaluation,
            )
        if getattr(ma, "mask_embedding_sentence_negative_template", ""):
            noise3, template_length3 = denoising(
                cls, encoder, cls.mask_embedding_template3, type="neg-1",
                device=input_ids.device, evaluation=evaluation,
            )

    outputs = encoder(
        input_ids=flat_ids,
        attention_mask=flat_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    last_hidden = outputs.last_hidden_state
    pooler = last_hidden[flat_ids == cls.mask_token_id]
    pooler = pooler.view(-1, cls.mask_num, pooler.shape[-1])
    pooler = pooler.view(batch_size, num_sent, cls.mask_num, -1)

    if getattr(ma, "mask_embedding_sentence_delta", False) and (
        not evaluation or not getattr(ma, "mask_embedding_sentence_delta_no_delta_eval", True)
    ):
        attn = attention_mask.view(batch_size, num_sent, -1)
        entire_length = attn.sum(-1)
        max_idx = noise1.size(0) - 1

        token_length = torch.clamp(entire_length - template_length1, 0, max_idx)
        pooler[:, 0, 0, :] -= noise1[token_length[:, 0], 0, :]
        pooler[:, 0, 1, :] -= noise1[token_length[:, 0], 1, :]

        if noise2 is not None:
            token_length = torch.clamp(entire_length - template_length2, 0, max_idx)
            pooler[:, 1, 0, :] -= noise2[token_length[:, 1], 0, :]
            pooler[:, 1, 1, :] -= noise2[token_length[:, 1], 1, :]

        if noise3 is not None and num_sent >= 3:
            token_length = torch.clamp(entire_length - template_length3, 0, max_idx)
            pooler[:, 2, 0, :] -= noise3[token_length[:, 2], 0, :]
            pooler[:, 2, 1, :] -= noise3[token_length[:, 2], 1, :]

    pooler = pooler.reshape(batch_size * num_sent * cls.mask_num, -1)
    pooler = cls.cot_mlp(pooler)
    pooler = pooler.view(batch_size, num_sent, cls.mask_num, -1)
    return pooler


def compute_cot_loss(cls, pooler_output, num_sent, device):
    """L_cot: summary MASK (m2) 跨列 InfoNCE + hard negative。"""
    z1_m2 = pooler_output[:, 0, 1, :]
    if num_sent < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    z2_m2 = pooler_output[:, 1, 1, :]
    cos_sim_m2 = cls.sim(z1_m2.unsqueeze(1), z2_m2.unsqueeze(0))

    if num_sent == 3:
        z3_m2 = pooler_output[:, 2, 1, :]
        z1_z3 = cls.sim_scd(z1_m2.unsqueeze(1), z3_m2.unsqueeze(0))
        z2_z3 = cls.sim_scd(z2_m2.unsqueeze(1), z3_m2.unsqueeze(0))
        cos_sim_m2 = torch.cat([cos_sim_m2, z1_z3, z2_z3], dim=1)

    labels = torch.arange(cos_sim_m2.size(0), dtype=torch.long, device=device)
    loss = nn.CrossEntropyLoss()(cos_sim_m2, labels)
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    return loss


class MultiSemanticSPR(nn.Module):
    """
    多语义分解增强模型：语义分解 + 局部信号增强
    生成增强后的子语义用于残差融合
    
    简化设计流程：
    1. encoder输出h → 语义分解 → h_i
    2. h_i → 局部信号增强 → h_i_enhanced
    """
    def __init__(self, hidden_dim: int, num_semantics: int = 7, lambda2: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_semantics = num_semantics
        self.lambda2 = lambda2  # λ₂: 软正交权重
        
        # 语义分解器
        self.decomposer = SemanticDecomposer(hidden_dim, num_semantics)
        
        # num_semantics个并行的局部信号增强模块：对分解后的每个子语义进行精细化增强
        self.local_signal_enhancers = nn.ModuleList([
            SignalEnhancer(hidden_dim) for _ in range(num_semantics)
        ])
    
    def forward(self, sentence_repr, compute_orth_loss=True, return_all_semantics=False):
        """
        多语义分解增强前向传播
        生成增强后的子语义用于残差融合
        
        简化设计流程：
        1. 语义分解：h -> h_i（分解为多个子语义）
        2. 局部信号增强：h_i -> h_i_enhanced（对每个子语义进行精细化增强）
        
        Args:
            sentence_repr: (batch_size, hidden_dim) 锚点样本h
            compute_orth_loss: 是否计算软正交损失，默认True（训练时），False（评估时）
            return_all_semantics: 是否返回所有子语义h_i和h_i_enhanced，默认False
        Returns:
            orth_loss: 软正交损失
            如果return_all_semantics=True，还返回：
            h_i_list: list of (batch_size, hidden_dim) 所有子语义h_i（分解后的原始表示）
            h_i_enhanced_list: list of (batch_size, hidden_dim) 所有增强后的子语义h_i_enhanced
        """
        # 步骤1：语义分解 - 将句子表示分解为多个语义表示，获取软正交损失
        semantic_reprs, orth_loss = self.decomposer(sentence_repr, compute_orth_loss=compute_orth_loss)
        # semantic_reprs: (batch_size, num_semantics, hidden_dim)
        
        # 存储所有子语义h_i和h_i_enhanced
        h_i_list = []
        h_i_enhanced_list = []
        
        # 步骤2：对每个子语义进行局部信号增强
        for i in range(self.num_semantics):
            # 提取第i个子语义 h_i: (batch_size, hidden_dim)
            h_i = semantic_reprs[:, i]
            
            # 局部信号增强 - 对每个子语义进行精细化增强
            h_i_enhanced = self.local_signal_enhancers[i](h_i)
            
            # 不提前归一化，保持原始信号，只在计算相似度时归一化
            # 存储原始表示和增强后的表示
            h_i_list.append(h_i)
            h_i_enhanced_list.append(h_i_enhanced)
        
        if return_all_semantics:
            # 返回所有子语义信息（用于残差融合）
            return h_i_list, h_i_enhanced_list, orth_loss
        else:
            # 为了兼容性，返回None（融合将在ResidualFusion中完成）
            return None, orth_loss


def prism_decomp_init(
    cls,
    config,
    temperature=0.05,
    lambda2=0.1,
    num_semantics=7,
    compress_dim=8,
    lambda_sup=0.0,
    sup_loss_type="cosine",
    scd_temp=0.05,
    compress_mode="mlp",
    compressor_hidden=256,
    projection_seed=42,
    compressor_ckpt=None,
):
    """
    棱镜分解模型初始化函数
    Args:
        config: 模型配置
        temperature: InfoNCE损失的温度参数（默认0.05）
        lambda2: 软正交损失权重（默认0.1）
        num_semantics: 语义维度数量（默认7，从配置文件读取）
    """
    # 初始化多语义分解增强模块，用于生成增强后的子语义
    cls.multisemantic_spr = MultiSemanticSPR(
        config.hidden_size, 
        num_semantics=num_semantics,
        lambda2=lambda2  # λ₂: 软正交权重
    )
    
    # 初始化残差融合器，用于将增强后的子语义与原始h进行残差连接
    cls.residual_fusion = ResidualFusion(
        hidden_dim=config.hidden_size,
        num_semantics=num_semantics,
        fusion_weight=0.5  # 融合权重，可学习
    )
    
    # 初始化相似度计算模块（L_cot）
    cls.sim = Similarity(temp=temperature)
    cls.sim_scd = Similarity(temp=scd_temp)
    cls.cot_mlp = MLPLayer(config, scale=1)
    
    # 设置语义维度数量（从配置文件读取）
    cls.num_semantics = num_semantics
    
    # 定义语义维度（根据num_semantics动态生成）
    # 如果num_semantics <= 7，使用预定义的名称；否则使用通用名称
    predefined_dimensions = [
        "情感语义", "主题语义", "语法语义", 
        "时序语义", "空间语义", "因果语义", "程度语义"
    ]
    if num_semantics <= len(predefined_dimensions):
        cls.semantic_dimensions = predefined_dimensions[:num_semantics]
    else:
        cls.semantic_dimensions = predefined_dimensions + [
            f"语义维度{i+1}" for i in range(len(predefined_dimensions), num_semantics)
        ]
    
    # 共享主题压缩器（与 Stage1 同结构）
    cls.theme_compressor = SharedThemeCompressor(
        hidden_dim=config.hidden_size,
        compress_dim=compress_dim,
        hidden_size=compressor_hidden,
        mode=compress_mode,
        projection_seed=projection_seed,
    )
    if compressor_ckpt:
        ckpt_compressor, _, _ = load_compressor_checkpoint(
            compressor_ckpt,
            hidden_dim=config.hidden_size,
            compress_dim=compress_dim,
            hidden_size=compressor_hidden,
            mode=compress_mode,
            projection_seed=projection_seed,
        )
        cls.theme_compressor.load_state_dict(ckpt_compressor.state_dict())

    cls.theme_supervision_loss = ThemeVectorSupervisionLoss(loss_type=sup_loss_type)

    cls.compress_dim = compress_dim
    cls.lambda_sup = lambda_sup
    cls.temperature = temperature
    cls.lambda2 = lambda2
    
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
                        theme_targets=None,
                        return_dict=None,
):
    """
    棱镜分解前向传播：L = L_cot + λ_sup·L_sup（无 L_global）
    theme_targets: (batch, num_semantics, compress_dim) Stage1 缓存，可选
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    cot_mode = _uses_cot(cls)

    if cot_mode:
        pooler_output = _extract_cot_pooler(
            cls, encoder, input_ids, attention_mask, evaluation=False
        )
        anchor_h = pooler_output[:, 0, 1, :]  # z1_m2 summary
        cot_loss = compute_cot_loss(cls, pooler_output, num_sent, anchor_h.device)
    else:
        flat_ids = input_ids.view((-1, input_ids.size(-1)))
        flat_mask = attention_mask.view((-1, attention_mask.size(-1)))
        outputs = encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        mask_token_id = cls.config.mask_token_id if hasattr(cls.config, "mask_token_id") else 103
        sentence_repr_list = []
        for i in range(flat_ids.size(0)):
            mask_mask = flat_ids[i] == mask_token_id
            if mask_mask.any():
                mask_pos = mask_mask.long().argmax().item()
                sentence_repr_list.append(outputs.last_hidden_state[i, mask_pos, :])
            else:
                sentence_repr_list.append(outputs.last_hidden_state[i, 0, :])
        sentence_repr = torch.stack(sentence_repr_list, dim=0).view(batch_size, num_sent, -1)
        anchor_h = sentence_repr[:, 0]
        cot_loss = torch.tensor(0.0, device=anchor_h.device)

    h_i_list, h_i_enhanced_list, _ = cls.multisemantic_spr(
        anchor_h,
        compute_orth_loss=False,
        return_all_semantics=True,
    )

    h_i_enhanced_stacked = torch.stack(h_i_enhanced_list, dim=1)
    T_pred = cls.theme_compressor(h_i_enhanced_stacked, normalize=True)
    h_plus = cls.residual_fusion(anchor_h, h_i_enhanced_list)

    sup_loss = torch.tensor(0.0, device=anchor_h.device)
    if theme_targets is not None:
        theme_targets = theme_targets.to(anchor_h.device, dtype=anchor_h.dtype)
        if getattr(cls, "lambda_sup", 0.0) > 0:
            sup_loss = cls.theme_supervision_loss(T_pred, theme_targets)

    # L = L_cot + λ_sup·L_sup
    total_loss = cot_loss + getattr(cls, "lambda_sup", 0.0) * sup_loss

    logits = h_plus

    if not return_dict:
        output = (logits,)
        return ((total_loss,) + output) if total_loss is not None else output

    return SequenceClassifierOutput(
        loss=total_loss,
        logits=logits,
        hidden_states=None,
        attentions=None,
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
    return_aspects=False,
):
    """
    句子嵌入前向传播（评估/推理）：CoT summary MASK m2 → 分解融合。
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    cot_mode = _uses_cot(cls)

    if cot_mode and input_ids.dim() == 2:
        input_ids = input_ids.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)

    if cot_mode:
        pooler_output = _extract_cot_pooler(
            cls, encoder, input_ids, attention_mask, evaluation=True
        )
        sentence_repr = pooler_output[:, 0, 1, :]
    else:
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
        mask_token_id = cls.config.mask_token_id if hasattr(cls.config, "mask_token_id") else 103
        sentence_repr_list = []
        for i in range(input_ids.size(0)):
            mask_mask = input_ids[i] == mask_token_id
            if mask_mask.any():
                mask_pos = mask_mask.long().argmax().item()
                sentence_repr_list.append(outputs.last_hidden_state[i, mask_pos, :])
            else:
                sentence_repr_list.append(outputs.last_hidden_state[i, 0, :])
        sentence_repr = torch.stack(sentence_repr_list, dim=0)

    with torch.no_grad():
        _, h_i_enhanced_list, _ = cls.multisemantic_spr(
            sentence_repr,
            compute_orth_loss=False,
            return_all_semantics=True,
        )
        pooler_output = cls.residual_fusion(sentence_repr, h_i_enhanced_list)
        pooler_output = F.normalize(pooler_output, p=2, dim=-1)

        aspect_reprs = None
        if return_aspects:
            h_stack = torch.stack(h_i_enhanced_list, dim=1)
            aspect_reprs = cls.theme_compressor(h_stack, normalize=True)

    if not return_dict:
        if return_aspects:
            return (pooler_output, aspect_reprs)
        return (pooler_output,)

    result = BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=None,
        hidden_states=None,
    )
    if return_aspects:
        result["aspect_reprs"] = aspect_reprs
    return result


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

        # 从model_args获取温度参数、lambda2参数和num_semantics参数
        temperature = getattr(self.model_args, 'temperature', 0.05) if self.model_args else 0.05
        lambda2 = getattr(self.model_args, 'lambda2', 0.1) if self.model_args else 0.1
        lambda_sup = getattr(self.model_args, 'lambda_sup', 0.0) if self.model_args else 0.0
        compress_dim = getattr(self.model_args, 'compress_dim', 8) if self.model_args else 8
        sup_loss_type = getattr(self.model_args, 'sup_loss_type', 'cosine') if self.model_args else 'cosine'
        compress_mode = getattr(self.model_args, 'compress_mode', 'mlp') if self.model_args else 'mlp'
        compressor_hidden = getattr(self.model_args, 'compressor_hidden', 256) if self.model_args else 256
        projection_seed = getattr(self.model_args, 'projection_seed', 42) if self.model_args else 42
        compressor_ckpt = getattr(self.model_args, 'compressor_ckpt', None) if self.model_args else None
        scd_temp = getattr(self.model_args, 'scd_temp', 0.05) if self.model_args else 0.05
        num_semantics = getattr(self.model_args, 'num_semantics', 7) if self.model_args else 7
        
        prism_decomp_init(
            self, config,
            temperature=temperature,
            lambda2=lambda2,
            num_semantics=num_semantics,
            compress_dim=compress_dim,
            lambda_sup=lambda_sup,
            sup_loss_type=sup_loss_type,
            scd_temp=scd_temp,
            compress_mode=compress_mode,
            compressor_hidden=compressor_hidden,
            projection_seed=projection_seed,
            compressor_ckpt=compressor_ckpt,
        )
        self.total_length = 80
        if self.model_args:
            self.mask_num = getattr(self.model_args, "mask_num", 2)

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
        return_aspects=False,
        theme_targets=None,
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
                return_aspects=return_aspects,
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
                theme_targets=theme_targets,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
