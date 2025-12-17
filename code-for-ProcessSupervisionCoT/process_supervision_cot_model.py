import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


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
    """
    Head for getting sentence representations over BERT's representation.
    """
    def __init__(self, config, scale=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * scale, config.hidden_size * scale)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


def process_supervision_cot_init(cls, config, temperature=0.05):
    """
    过程监督 CoT 模型初始化函数
    Args:
        config: 模型配置
        temperature: InfoNCE损失的温度参数（默认0.05）
    """
    # 初始化相似度计算模块（用于InfoNCE损失）
    cls.similarity = Similarity(temp=temperature)
    
    # 初始化 MLP 层（用于 mask 表示投影）
    cls.mlp = MLPLayer(config, scale=2)  # mask_num=2
    
    # 存储温度参数
    cls.temperature = temperature
    
    cls.init_weights()


def cross_template_icnce_loss(mask_anchor, mask_positive, device, temp=0.05):
    """
    跨模板对比学习损失：约束锚句模板和正样本模板的相同位置 mask 相似
    参考 TNCSE 的 icnce 函数
    Args:
        mask_anchor: [batch_size, hidden_dim] 锚句模板的 mask 表示
        mask_positive: [batch_size, hidden_dim] 正样本模板的 mask 表示
        device: 设备
        temp: 温度参数
    Returns:
        loss: 标量损失值
    """
    y_true = torch.arange(mask_anchor.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1

    sim = F.cosine_similarity(
        mask_anchor.unsqueeze(1),
        mask_positive.unsqueeze(0),
        dim=-1
    )

    sim = sim - torch.eye(mask_anchor.shape[0], device=device) * 1e12
    sim = sim / temp

    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def compute_original_infonce_loss(emb_anchor, emb_positive, emb_negative, 
                                  similarity_module, device, temperature=0.05):
    """
    原有的三模板 InfoNCE 对比学习损失
    使用融合后的表示（mask1 + mask2）进行对比
    Args:
        emb_anchor: [batch_size, hidden_dim] 锚句模板的融合表示
        emb_positive: [batch_size, hidden_dim] 正样本模板的融合表示
        emb_negative: [batch_size, hidden_dim] 负样本模板的融合表示（可选）
        similarity_module: 相似度计算模块
        device: 设备
        temperature: 温度参数
    Returns:
        loss: 标量损失值
    """
    eps = 1e-8
    
    # 归一化
    anchor_norm = F.normalize(emb_anchor, p=2, dim=-1, eps=eps)
    pos_norm = F.normalize(emb_positive, p=2, dim=-1, eps=eps)
    
    # 正样本对相似度
    pos_sim = (anchor_norm * pos_norm).sum(dim=-1, keepdim=True) / temperature
    pos_sim = torch.clamp(pos_sim, min=-50.0, max=50.0)
    
    # 构建负样本候选池
    if emb_negative is not None:
        neg_norm = F.normalize(emb_negative, p=2, dim=-1, eps=eps)
        # 包含负样本和批次内其他样本
        all_emb = torch.cat([emb_anchor, emb_positive, emb_negative], dim=0)
    else:
        all_emb = torch.cat([emb_anchor, emb_positive], dim=0)
    
    all_emb_norm = F.normalize(all_emb, p=2, dim=-1, eps=eps)
    
    # 计算锚句与所有候选的相似度
    neg_sim = torch.mm(anchor_norm, all_emb_norm.t()) / temperature
    neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)
    
    # 排除自身（锚句和正样本）
    batch_size = anchor_norm.size(0)
    batch_range = torch.arange(batch_size, device=device)
    neg_sim[:, batch_range] = float("-inf")  # 排除锚句自身
    neg_sim[:, batch_size + batch_range] = float("-inf")  # 排除正样本
    
    # 组合相似度矩阵
    cos_sim = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)
    
    return loss


def process_supervision_cot_forward(cls,
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
                                    cross_template_weight=0.5,
                                    process_supervision_weight=0.1,
                                    enable_process_supervision=False,
                                    process_supervisor=None,
                                    tokenizer=None,
):
    """
    过程监督 CoT 前向传播函数
    1. 提取三个模板的 mask1 和 mask2 表示
    2. 计算原有 InfoNCE 损失
    3. 计算跨模板对比学习损失
    4. 计算过程监督损失（可选）
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if input_ids is None:
        raise ValueError("input_ids 不能为空")

    if input_ids.dim() == 3:
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
    else:
        batch_size = input_ids.size(0)
        num_sent = 1

    # 通过 BERT 编码器
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

    # 提取 mask token 的表示
    mask_token_id = cls.config.mask_token_id if hasattr(cls.config, "mask_token_id") else 103
    mask = input_ids == mask_token_id
    last_hidden = outputs.last_hidden_state
    
    # 提取所有 MASK token 的表示
    pooler_output = last_hidden[mask]  # [total_masks, hidden_size]
    
    # 重塑为 [batch_size * num_sent, mask_num, hidden_size]
    pooler_output = pooler_output.view(-1, cls.model_args.mask_num, pooler_output.shape[-1])
    
    # 分离 mask1 和 mask2
    mask1_emb = pooler_output[:, 0, :]  # [batch_size * num_sent, hidden_dim]
    mask2_emb = pooler_output[:, 1, :]  # [batch_size * num_sent, hidden_dim]
    
    # 应用 MLP
    mask1_emb = cls.mlp(mask1_emb)
    mask2_emb = cls.mlp(mask2_emb)
    
    # 重塑为 [batch_size, num_sent, hidden_dim]
    mask1_emb = mask1_emb.view(batch_size, num_sent, -1)
    mask2_emb = mask2_emb.view(batch_size, num_sent, -1)
    
    # 分离三个模板
    mask1_anchor = mask1_emb[:, 0, :]  # [batch_size, hidden_dim]
    mask1_positive = mask1_emb[:, 1, :]  # [batch_size, hidden_dim]
    mask2_anchor = mask2_emb[:, 0, :]  # [batch_size, hidden_dim]
    mask2_positive = mask2_emb[:, 1, :]  # [batch_size, hidden_dim]
    
    if num_sent >= 3:
        mask1_negative = mask1_emb[:, 2, :]  # [batch_size, hidden_dim]
        mask2_negative = mask2_emb[:, 2, :]  # [batch_size, hidden_dim]
    else:
        mask1_negative = None
        mask2_negative = None
    
    # 计算损失
    loss = None
    logits = None
    device = mask1_anchor.device
    
    # 1. 原有 InfoNCE 损失（使用融合后的表示）
    emb_anchor = (mask1_anchor + mask2_anchor) / 2.0
    emb_positive = (mask1_positive + mask2_positive) / 2.0
    emb_negative = None
    if mask1_negative is not None:
        emb_negative = (mask1_negative + mask2_negative) / 2.0
    
    loss_original = compute_original_infonce_loss(
        emb_anchor, emb_positive, emb_negative,
        cls.similarity, device, cls.temperature
    )
    
    # 2. 跨模板对比学习损失
    loss_cross_mask1 = cross_template_icnce_loss(
        mask1_anchor, mask1_positive, device, cls.temperature
    )
    loss_cross_mask2 = cross_template_icnce_loss(
        mask2_anchor, mask2_positive, device, cls.temperature
    )
    loss_cross = loss_cross_mask1 + loss_cross_mask2
    
    # 3. 过程监督损失（可选）
    loss_ps = torch.tensor(0.0, device=device)
    if enable_process_supervision and process_supervisor is not None:
        # 这里需要提取推理路径并调用过程监督评估器
        # 暂时返回 0，具体实现见 process_supervision_cot_supervisor.py
        # loss_ps = compute_process_supervision_loss(...)
        pass
    
    # 总损失
    loss = loss_original + cross_template_weight * loss_cross + process_supervision_weight * loss_ps
    
    # 使用锚句模板的融合表示作为 logits
    logits = emb_anchor

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(cls,
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
                   tokenizer=None,
):
    """
    句子嵌入前向传播（用于评估）
    默认使用锚句模板的融合表示（mask1 + mask2）作为句子表示
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if input_ids.dim() == 3:
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
    else:
        batch_size = input_ids.size(0)
        num_sent = 1

    with torch.no_grad():
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

        # 提取 mask token 的表示
        mask_token_id = cls.config.mask_token_id if hasattr(cls.config, "mask_token_id") else 103
        mask = input_ids == mask_token_id
        last_hidden = outputs.last_hidden_state
        
        # 提取所有 MASK token 的表示
        pooler_output = last_hidden[mask]
        pooler_output = pooler_output.view(-1, cls.model_args.mask_num, pooler_output.shape[-1])
        
        # 分离 mask1 和 mask2
        mask1_emb = pooler_output[:, 0, :]
        mask2_emb = pooler_output[:, 1, :]
        
        # 应用 MLP
        mask1_emb = cls.mlp(mask1_emb)
        mask2_emb = cls.mlp(mask2_emb)
        
        # 融合表示
        pooler_output = (mask1_emb + mask2_emb) / 2.0
        
        # 归一化
        eps = 1e-8
        pooler_output = F.normalize(pooler_output, p=2, dim=-1, eps=eps)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class BertForProcessSupervisionCoT(BertPreTrainedModel):
    """
    基于过程监督的 CoT-BERT 模型
    支持三模板 InfoNCE + 跨模板对比学习 + 过程监督
    """
    
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.model_args = model_kwargs.get("model_args", None)
        self.bert = BertModel(config)
        
        # 初始化模型组件
        if self.model_args is not None:
            process_supervision_cot_init(
                self, config, 
                temperature=getattr(self.model_args, 'temp', 0.05)
            )
        
        # 设置 mask_token_id
        if hasattr(config, 'mask_token_id'):
            self.config.mask_token_id = config.mask_token_id
        else:
            self.config.mask_token_id = 103  # BERT 默认的 [MASK] token id
    
    def forward(self, *args, **kwargs):
        return process_supervision_cot_forward(self, self.bert, *args, **kwargs)
    
    def sentemb_forward(self, *args, **kwargs):
        return sentemb_forward(self, self.bert, *args, **kwargs)

