import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)


class MLPLayer(nn.Module):
    """
    MLP层：用于去噪前的预处理（可选）
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


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


def denoising(cls, encoder, template_ids, prefix_ids, suffix_ids, mask_token_id, pad_token_id, 
              total_length, mask_num, device='cuda', evaluation=False):
    """
    Delta去噪函数：提取位置相关的噪声
    
    Args:
        cls: 模型类实例
        encoder: BERT编码器
        template_ids: 模板的token ID列表（包含[CLS]和[SEP]）
        prefix_ids: 模板前缀的token ID列表（[X]之前的部分）
        suffix_ids: 模板后缀的token ID列表（[X]之后的部分）
        mask_token_id: MASK token的ID
        pad_token_id: PAD token的ID
        total_length: 总长度（用于滑动窗口）
        mask_num: MASK token的数量（本项目中为2）
        device: 设备
        evaluation: 是否为评估模式
    
    Returns:
        noise: 噪声张量，形状为 [max_pad_length, mask_num, hidden_size]
        template_length: 模板长度
    """
    with torch.set_grad_enabled(not cls.model_args.mask_embedding_sentence_delta_freeze and not evaluation):
        # 计算模板长度（不包括句子部分）
        template_length = len(prefix_ids) + len(suffix_ids) + 2  # +2 for [CLS] and [SEP]
        
        # 滑动窗口：创建不同pad长度的输入
        input_ids_list = []
        attention_mask_list = []
        
        max_pad_length = total_length - template_length + 1
        
        for pad_count in range(max_pad_length):
            # 构建输入：[CLS] + prefix + [PAD]... + suffix + [SEP] + [PAD]...
            input_ids = (
                [template_ids[0]] +  # [CLS]
                prefix_ids +
                [pad_token_id] * pad_count +
                suffix_ids +
                [template_ids[-1]] +  # [SEP]
                [pad_token_id] * (total_length - template_length - pad_count)
            )
            attention_mask = (
                [1] * (template_length + pad_count) +
                [0] * (total_length - template_length - pad_count)
            )
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        input_ids = torch.tensor(input_ids_list, device=device, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, device=device, dtype=torch.long)
        
        # 编码获取噪声
        if evaluation:
            with torch.no_grad():
                outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = outputs.last_hidden_state
                mask = input_ids == mask_token_id
                noise = last_hidden[mask]  # [total_masks, hidden_size]
        else:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden = outputs.last_hidden_state
            mask = input_ids == mask_token_id
            noise = last_hidden[mask]  # [total_masks, hidden_size]
        
        # 重新组织噪声：[max_pad_length, mask_num, hidden_size]
        # 每个pad长度对应mask_num个MASK token的噪声
        noise = noise.view(max_pad_length, mask_num, -1)
        
        return noise, template_length


def compute_infonce_loss(anchor, positive, negative, similarity, loss_fct):
    """
    计算 InfoNCE 损失
    
    与 models.py 的实现保持一致：
    - 使用 [batch_size, batch_size] 相似度矩阵，对角线为正样本对
    - 将 negative 作为 hard negative 追加到右侧
    - 使用 torch.arange 作为标签（对角线索引）
    
    Args:
        anchor: [batch_size, hidden_size] 锚句表示
        positive: [batch_size, hidden_size] 正样本表示
        negative: [batch_size, hidden_size] 负样本表示
        similarity: Similarity 实例（共享使用，内部已配置温度参数）
        loss_fct: 交叉熵损失函数
    
    Returns:
        loss: InfoNCE 损失值
    """
    batch_size = anchor.size(0)
    device = anchor.device
    
    # 主相似度矩阵：anchor 与 positive 的相似度
    # 形状: [batch_size, batch_size]
    # 对角线 [i, i] 是正样本对 (anchor[i], positive[i])
    # 非对角线 [i, j] where i != j 是负样本对 (anchor[i], positive[j])
    cos_sim = similarity(anchor.unsqueeze(1), positive.unsqueeze(0))  # [batch_size, batch_size]
    
    # Hard negative 处理：将 negative 追加到右侧（类似 models.py 中的 z3）
    # 计算 anchor 与 negative 的相似度
    anchor_negative_cos = similarity(anchor.unsqueeze(1), negative.unsqueeze(0))  # [batch_size, batch_size]
    # 计算 positive 与 negative 的相似度
    positive_negative_cos = similarity(positive.unsqueeze(1), negative.unsqueeze(0))  # [batch_size, batch_size]
    # 追加到右侧
    cos_sim = torch.cat([cos_sim, anchor_negative_cos, positive_negative_cos], dim=1)  # [batch_size, batch_size * 3]
    
    # 标签：使用对角线索引（与 models.py 一致）
    # labels[i] = i 表示第 i 个样本的正样本在对角线位置 [i, i]
    labels = torch.arange(cos_sim.size(0), dtype=torch.long, device=device)
    
    # 计算 InfoNCE 损失
    loss = loss_fct(cos_sim, labels)
    
    # 检查并处理 NaN 和 Inf
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


def compute_constraint_loss(h1_anchor, h1_positive, h2_anchor, h2_positive, eps=1e-8):
    """
    计算约束项损失 L3
    
    L3 = (||(h2_positive - h1_anchor)|| + ||(h2_anchor - h1_positive)||) 
         / (||h2_anchor|| + ||h2_positive||² + ε)
    
    该损失用于增强第一阶段表示（h1_anchor, h1_positive）来推动
    第二阶段表示（h2_anchor, h2_positive）的优化。
    
    Args:
        h1_anchor: [batch_size, hidden_size] 第一阶段锚句表示
        h1_positive: [batch_size, hidden_size] 第一阶段正样本表示
        h2_anchor: [batch_size, hidden_size] 第二阶段锚句表示
        h2_positive: [batch_size, hidden_size] 第二阶段正样本表示
        eps: 数值稳定性参数
    
    Returns:
        loss: 约束项损失值
    """
    # 计算分子：两个差异向量的 L2 范数之和
    diff_1 = h2_positive - h1_anchor  # [batch_size, hidden_size]
    diff_2 = h2_anchor - h1_positive  # [batch_size, hidden_size]
    
    norm_diff_1 = torch.norm(diff_1, p=2, dim=-1)  # [batch_size]
    norm_diff_2 = torch.norm(diff_2, p=2, dim=-1)  # [batch_size]
    numerator = norm_diff_1 + norm_diff_2
    
    # 计算分母：第二阶段表示的范数
    norm_h2_anchor = torch.norm(h2_anchor, p=2, dim=-1)  # [batch_size]
    norm_h2_positive = torch.norm(h2_positive, p=2, dim=-1)  # [batch_size]
    denominator = norm_h2_anchor + norm_h2_positive ** 2 + eps
    
    # 确保分母不会太小，避免数值不稳定
    denominator = torch.clamp(denominator, min=eps * 10)
    
    # 计算损失
    loss = (numerator / denominator).mean()
    
    # 检查并处理 NaN 和 Inf
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
    
    return loss


def cross_template_cot_init(cls, config, temperature=0.05):
    """
    跨模板CoT模型初始化函数
    """
    # 初始化相似度计算模块（用于InfoNCE损失）
    cls.similarity = Similarity(temp=temperature)
    
    # 可选MLP层（用于去噪前预处理）
    if cls.model_args.mask_embedding_sentence_org_mlp:
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        cls.mlp = None
    
    # 存储温度参数
    cls.temperature = temperature
    
    cls.init_weights()


def cross_template_cot_forward(cls,
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
                                anchor_template=None,
                                positive_template=None,
                                negative_template=None,
                                tokenizer=None,
                                ):
    """
    跨模板CoT前向传播函数
    
    流程：
    1. 使用3个模板（锚句、正样本、负样本）提取6个MASK表示
       - h1_anchor, h2_anchor: 锚句模板的两个MASK
       - h1_positive, h2_positive: 正样本模板的两个MASK
       - h1_negative, h2_negative: 负样本模板的两个MASK
    
    2. 使用delta去噪方法去除位置噪声
    
    3. 计算3个损失：
       - L1: 第一个MASK位置的InfoNCE损失（过程监督）
       - L2: 第二个MASK位置的InfoNCE损失（过程监督）
       - L3: 约束项损失（增强第一阶段表示推动第二阶段表示）
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    if input_ids is None:
        raise ValueError("input_ids 不能为空")
    
    # 输入形状：(batch_size, 3, seq_len) - 3个模板
    if input_ids.dim() == 3:
        batch_size = input_ids.size(0)
        num_templates = input_ids.size(1)  # 应该为3
        assert num_templates == 3, f"期望3个模板，但得到{num_templates}个"
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
    else:
        raise ValueError(f"输入形状不正确，期望(batch_size, 3, seq_len)，但得到{input_ids.shape}")
    
    # 编码所有输入
    encoder_outputs = encoder(
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
    
    last_hidden = encoder_outputs.last_hidden_state  # [batch_size * 3, seq_len, hidden_size]
    
    # 提取MASK token表示
    mask_token_id = cls.config.mask_token_id if hasattr(cls.config, "mask_token_id") else tokenizer.mask_token_id
    mask = input_ids == mask_token_id  # [batch_size * 3, seq_len]
    
    # 提取所有MASK token的表示
    h_all_masks = last_hidden[mask]  # [total_masks, hidden_size]
    
    # 计算每个样本的MASK数量（应该为2）
    mask_counts = mask.sum(dim=-1)  # [batch_size * 3]
    
    # 计算每个样本的MASK位置索引
    mask_cumsum = torch.cumsum(mask_counts, dim=0)  # [batch_size * 3]
    mask_indices = mask_cumsum - 1  # [batch_size * 3] - 最后一个MASK的索引
    
    # 处理没有MASK的情况（向后兼容）
    has_mask = mask_counts > 0
    if not has_mask.all():
        fallback_positions = mask.long().argmax(dim=-1)
        batch_indices_fallback = torch.arange(input_ids.size(0), device=input_ids.device)
        h_fallback = last_hidden[batch_indices_fallback, fallback_positions]
        h_flat = torch.where(has_mask.unsqueeze(-1), h_all_masks[mask_indices], h_fallback)
    else:
        h_flat = h_all_masks[mask_indices]  # [batch_size * 3, hidden_size] - 每个样本的最后一个MASK
    
    # 提取每个样本的两个MASK表示
    # 重新组织：找到每个样本的所有MASK位置
    # 计算每个样本在 h_all_masks 中的索引范围
    mask_start_indices = torch.cat([torch.tensor([0], device=mask_counts.device), mask_cumsum[:-1]])  # [batch_size * 3] - 每个样本的起始索引
    h_masks_list = []
    for i in range(batch_size * 3):
        start_idx = mask_start_indices[i].item()
        end_idx = mask_cumsum[i].item()
        num_masks = mask_counts[i].item()
        
        if num_masks >= 2:
            # 提取该样本的所有MASK表示，取前两个
            sample_masks = h_all_masks[start_idx:end_idx][:2]  # [2, hidden_size]
            h_masks_list.append(sample_masks)
        elif num_masks == 1:
            # 只有一个MASK，重复使用
            sample_mask = h_all_masks[start_idx:end_idx][0]  # [hidden_size]
            h_masks_list.append(torch.stack([sample_mask, sample_mask]))  # [2, hidden_size]
        else:
            # 没有MASK，使用fallback
            h_masks_list.append(torch.stack([h_flat[i], h_flat[i]]))  # [2, hidden_size]
    
    h_masks = torch.stack(h_masks_list)  # [batch_size * 3, 2, hidden_size]
    
    # 分离三个模板的MASK表示
    h_anchor = h_masks[:batch_size]  # [batch_size, 2, hidden_size]
    h_positive = h_masks[batch_size:2*batch_size]  # [batch_size, 2, hidden_size]
    h_negative = h_masks[2*batch_size:]  # [batch_size, 2, hidden_size]
    
    # 提取第一个和第二个MASK表示
    h1_anchor = h_anchor[:, 0, :]  # [batch_size, hidden_size]
    h2_anchor = h_anchor[:, 1, :]  # [batch_size, hidden_size]
    h1_positive = h_positive[:, 0, :]  # [batch_size, hidden_size]
    h2_positive = h_positive[:, 1, :]  # [batch_size, hidden_size]
    h1_negative = h_negative[:, 0, :]  # [batch_size, hidden_size]
    h2_negative = h_negative[:, 1, :]  # [batch_size, hidden_size]
    
    # Delta去噪处理
    if cls.model_args.mask_embedding_sentence_delta:
        # 准备模板信息用于去噪
        device = input_ids.device
        pad_token_id = tokenizer.pad_token_id
        total_length = input_ids.size(-1)
        mask_num = 2
        
        # 解析三个模板
        templates = [anchor_template, positive_template, negative_template]
        template_prefixes = []
        template_suffixes = []
        template_ids_list = []
        
        for template in templates:
            parts = template.split('[X]')
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
            
            # 编码模板部分
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            template_ids = [tokenizer.cls_token_id] + prefix_ids + suffix_ids + [tokenizer.sep_token_id]
            
            template_prefixes.append(prefix_ids)
            template_suffixes.append(suffix_ids)
            template_ids_list.append(template_ids)
        
        # 为每个模板提取噪声
        noise_list = []
        template_lengths = []
        
        for i, (prefix_ids, suffix_ids, template_ids) in enumerate(zip(template_prefixes, template_suffixes, template_ids_list)):
            noise, template_length = denoising(
                cls=cls,
                encoder=encoder,
                template_ids=template_ids,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                total_length=total_length,
                mask_num=mask_num,
                device=device,
                evaluation=False
            )
            noise_list.append(noise)
            template_lengths.append(template_length)
        
        noise_anchor = noise_list[0]  # [max_pad_length, 2, hidden_size]
        noise_positive = noise_list[1]
        noise_negative = noise_list[2]
        
        # 计算每个样本的token长度（用于索引噪声）
        # token_length = entire_length - template_length
        # 其中entire_length是实际输入的总长度（包括句子）
        entire_lengths = attention_mask.sum(dim=-1).view(batch_size, 3)  # [batch_size, 3]
        template_length_anchor = template_lengths[0]
        template_length_positive = template_lengths[1]
        template_length_negative = template_lengths[2]
        
        token_lengths_anchor = entire_lengths[:, 0] - template_length_anchor  # [batch_size]
        token_lengths_positive = entire_lengths[:, 1] - template_length_positive
        token_lengths_negative = entire_lengths[:, 2] - template_length_negative
        
        # 应用去噪：从原始MASK表示中减去对应长度的噪声
        # 第一个MASK使用索引0的噪声，第二个MASK使用索引1的噪声
        h1_anchor_denoised = []
        h2_anchor_denoised = []
        h1_positive_denoised = []
        h2_positive_denoised = []
        h1_negative_denoised = []
        h2_negative_denoised = []
        
        for i in range(batch_size):
            # 锚句模板
            tl_anchor = token_lengths_anchor[i].item()
            tl_anchor = max(0, min(tl_anchor, noise_anchor.size(0) - 1))  # 确保索引有效
            h1_anchor_denoised.append(h1_anchor[i] - noise_anchor[tl_anchor, 0, :])
            h2_anchor_denoised.append(h2_anchor[i] - noise_anchor[tl_anchor, 1, :])
            
            # 正样本模板
            tl_positive = token_lengths_positive[i].item()
            tl_positive = max(0, min(tl_positive, noise_positive.size(0) - 1))
            h1_positive_denoised.append(h1_positive[i] - noise_positive[tl_positive, 0, :])
            h2_positive_denoised.append(h2_positive[i] - noise_positive[tl_positive, 1, :])
            
            # 负样本模板
            tl_negative = token_lengths_negative[i].item()
            tl_negative = max(0, min(tl_negative, noise_negative.size(0) - 1))
            h1_negative_denoised.append(h1_negative[i] - noise_negative[tl_negative, 0, :])
            h2_negative_denoised.append(h2_negative[i] - noise_negative[tl_negative, 1, :])
        
        h1_anchor = torch.stack(h1_anchor_denoised)
        h2_anchor = torch.stack(h2_anchor_denoised)
        h1_positive = torch.stack(h1_positive_denoised)
        h2_positive = torch.stack(h2_positive_denoised)
        h1_negative = torch.stack(h1_negative_denoised)
        h2_negative = torch.stack(h2_negative_denoised)
    
    # 可选MLP处理
    if cls.mlp is not None:
        h1_anchor = cls.mlp(h1_anchor)
        h2_anchor = cls.mlp(h2_anchor)
        h1_positive = cls.mlp(h1_positive)
        h2_positive = cls.mlp(h2_positive)
        h1_negative = cls.mlp(h1_negative)
        h2_negative = cls.mlp(h2_negative)
    
    # 计算三个损失组件
    eps = 1e-8
    loss_fct = nn.CrossEntropyLoss()
    
    # 创建共享的 Similarity 实例（避免重复创建）
    similarity = Similarity(temp=cls.temperature)
    
    # L1: 第一个 MASK 位置的 InfoNCE 损失（过程监督）
    # 正样本对：(h1_anchor, h1_positive)
    # 负样本：h1_negative 以及 batch 内其他样本的 h1_anchor, h1_positive, h1_negative
    L1 = compute_infonce_loss(
        anchor=h1_anchor,
        positive=h1_positive,
        negative=h1_negative,
        similarity=similarity,
        loss_fct=loss_fct
    )
    
    # L2: 第二个 MASK 位置的 InfoNCE 损失（过程监督）
    # 正样本对：(h2_anchor, h2_positive)
    # 负样本：h2_negative 以及 batch 内其他样本的 h2_anchor, h2_positive, h2_negative
    L2 = compute_infonce_loss(
        anchor=h2_anchor,
        positive=h2_positive,
        negative=h2_negative,
        similarity=similarity,
        loss_fct=loss_fct
    )
    
    # L3: 约束项损失
    # 增强第一阶段表示（h1_anchor, h1_positive）来推动第二阶段表示（h2_anchor, h2_positive）的优化
    # L3 = compute_constraint_loss(
    #     h1_anchor=h1_anchor,
    #     h1_positive=h1_positive,
    #     h2_anchor=h2_anchor,
    #     h2_positive=h2_positive,
    #     eps=eps
    # )
    
    # # 总损失：加权求和
    # weight_1 = getattr(cls.model_args, 'process_supervision_weight_1', 1.0)
    # weight_2 = getattr(cls.model_args, 'process_supervision_weight_2', 1.0)
    # weight_3 = getattr(cls.model_args, 'constraint_weight', 1.0)
    
    loss = 0.5 * L1 + 0.5 * L2
    # loss = L2
    
    logits = h2_anchor  # 使用第二个MASK的锚句表示作为logits
    
    if not return_dict:
        output = (logits,)
        return ((loss,) + output) if loss is not None else output
    
    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=None,
        attentions=None,
    )


def cross_template_cot_sentemb_forward(
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
    anchor_template=None,
    tokenizer=None,
):
    """
    句子嵌入前向传播（用于SentEval评估）
    默认使用锚句模板中第二个MASK的位置表示作为句子表示。
    不进行归一化，与 models.py 保持一致（SentEval 内部会自动归一化）。
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if input_ids.dim() == 3:
        batch_size = input_ids.size(0)
        num_templates = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
    else:
        batch_size = input_ids.size(0)
        num_templates = 1

    encoder_outputs = encoder(
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

    last_hidden = encoder_outputs.last_hidden_state  # [batch_size * num_templates, seq_len, hidden_size]

    mask_token_id = cls.config.mask_token_id if hasattr(cls.config, "mask_token_id") else tokenizer.mask_token_id
    mask = input_ids == mask_token_id

    h_all_masks = last_hidden[mask]  # [total_masks, hidden_size]
    mask_counts = mask.sum(dim=-1)  # [batch_size * num_templates]

    mask_cumsum = torch.cumsum(mask_counts, dim=0)
    # 计算每个样本在 h_all_masks 中的索引范围
    mask_start_indices = torch.cat([torch.tensor([0], device=mask_counts.device), mask_cumsum[:-1]])  # [batch_size * num_templates] - 每个样本的起始索引
    # 这里只假设每个样本至少有2个MASK，提取前两个
    h_masks_list = []
    for i in range(batch_size * num_templates):
        start_idx = mask_start_indices[i].item()
        end_idx = mask_cumsum[i].item()
        num_masks = mask_counts[i].item()
        
        if num_masks >= 2:
            # 提取该样本的所有MASK表示，取前两个
            sample_masks = h_all_masks[start_idx:end_idx][:2]  # [2, hidden_size]
            h_masks_list.append(sample_masks)
        elif num_masks == 1:
            # 只有一个MASK，重复使用
            sample_mask = h_all_masks[start_idx:end_idx][0]  # [hidden_size]
            h_masks_list.append(torch.stack([sample_mask, sample_mask]))  # [2, hidden_size]
        else:
            # 如果没有MASK，则取序列平均作为fallback
            h_avg = last_hidden[i].mean(dim=0)
            h_masks_list.append(torch.stack([h_avg, h_avg]))

    h_masks = torch.stack(h_masks_list)  # [batch_size * num_templates, 2, hidden_size]

    # 对于评估，只使用锚句模板（num_templates应为1）
    h_anchor = h_masks[:, 1, :]  # 第二个MASK表示 [batch_size * num_templates, hidden_size]

    # Delta去噪（仅对锚句模板）
    if cls.model_args and getattr(cls.model_args, "mask_embedding_sentence_delta", False):
        device = input_ids.device
        pad_token_id = tokenizer.pad_token_id
        total_length = input_ids.size(-1)
        mask_num = 2

        if anchor_template is None:
            anchor_template = 'The sentence of "[X]" means [MASK], so it can be summarized as [MASK].'

        parts = anchor_template.split("[X]")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        template_ids = [tokenizer.cls_token_id] + prefix_ids + suffix_ids + [tokenizer.sep_token_id]

        noise, template_length = denoising(
            cls=cls,
            encoder=encoder,
            template_ids=template_ids,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            total_length=total_length,
            mask_num=mask_num,
            device=device,
            evaluation=True,
        )

        attention_mask_reshaped = attention_mask.view(batch_size, num_templates, -1)
        entire_lengths = attention_mask_reshaped.sum(dim=-1)  # [batch_size, num_templates]
        token_lengths_anchor = entire_lengths[:, 0] - template_length

        h2_anchor_denoised = []
        for i in range(batch_size):
            tl_anchor = token_lengths_anchor[i].item()
            tl_anchor = max(0, min(tl_anchor, noise.size(0) - 1))
            # 第二个MASK使用索引1
            h2_anchor_denoised.append(h_anchor[i] - noise[tl_anchor, 1, :])

        h_anchor = torch.stack(h2_anchor_denoised)

    # 可选MLP
    if cls.mlp is not None:
        h_anchor = cls.mlp(h_anchor)

    # 直接使用 h_anchor 作为 pooler_output（不归一化，与 models.py 保持一致）
    pooler_output = h_anchor

    if not return_dict:
        return (encoder_outputs[0], pooler_output) + encoder_outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=encoder_outputs.last_hidden_state,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


class BertForCrossTemplateCoT(BertPreTrainedModel):
    """
    BERT for Cross-Template Chain-of-Thought (CrossTemplateCoT)
    实现基于跨模板对比学习的句子表示学习
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [
        r"similarity\.",
    ]
    
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs.get("model_args", None)
        self.bert = BertModel(config)
        
        # 从 model_args 获取温度参数
        temperature = getattr(self.model_args, 'temperature', 0.05) if self.model_args else 0.05
        
        cross_template_cot_init(self, config, temperature=temperature)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        重写 from_pretrained 方法以抑制预期的警告
        """
        import warnings
        import logging
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")
            warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used.*")
            
            old_level = logging.getLogger("transformers.modeling_utils").level
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            
            try:
                model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            finally:
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
                sent_emb: bool = False,
                anchor_template=None,
                positive_template=None,
                negative_template=None,
                tokenizer=None,
                ):
        if sent_emb:
            return cross_template_cot_sentemb_forward(
                self,
                self.bert,
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
                anchor_template=anchor_template,
                tokenizer=tokenizer,
            )
        else:
            return cross_template_cot_forward(
                self,
                self.bert,
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
                anchor_template=anchor_template,
                positive_template=positive_template,
                negative_template=negative_template,
                tokenizer=tokenizer,
            )

