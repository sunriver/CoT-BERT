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


class StageFusion(nn.Module):
    """
    两阶段表示融合层：融合第一阶段的h和第二阶段的h_plus
    使用加权融合方式：h_fused = w1 * h + w2 * h_plus
    初始时w1较大（0.7）以保留第一阶段语义
    """
    def __init__(self, hidden_dim: int, stage1_weight: float = 0.7):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 可学习的权重参数
        self.stage1_weight = nn.Parameter(torch.tensor(stage1_weight))
        self.stage2_weight = nn.Parameter(torch.tensor(1.0 - stage1_weight))
    
    def forward(self, h, h_plus):
        """
        融合两个阶段的表示
        Args:
            h: (batch_size, num_sent, hidden_dim) 第一阶段表示
            h_plus: (batch_size, num_sent, hidden_dim) 第二阶段表示
        Returns:
            h_fused: (batch_size, num_sent, hidden_dim) 融合后的表示
        """
        # 使用softmax确保权重和为1，但保持相对比例
        weights = F.softmax(torch.stack([self.stage1_weight, self.stage2_weight]), dim=0)
        h_fused = weights[0] * h + weights[1] * h_plus
        return h_fused


def two_stage_cot_init(cls, config, temperature=0.05):
    """
    两阶段思维链模型初始化函数
    Args:
        config: 模型配置
        temperature: InfoNCE损失的温度参数（默认0.05）
    """
    # 初始化相似度计算模块（用于InfoNCE损失）
    cls.similarity = Similarity(temp=temperature)
    
    # 初始化两阶段融合层
    cls.stage_fusion = StageFusion(
        hidden_dim=config.hidden_size,
        stage1_weight=0.7  # 初始时第一阶段权重较大，保留第一阶段语义
    )
    
    # 存储温度参数
    cls.temperature = temperature
    
    cls.init_weights()


def two_stage_cot_forward(cls,
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
                          stage1_templates=None,
                          stage2_template=None,
                          tokenizer=None,
):
    """
    两阶段思维链前向传播函数
    1. 第一阶段：分别处理负例/锚句/正例模版，提取各自的 h 表示
    2. 第二阶段：将对应的 h 注入 stage2 模版，得到 h_plus
    3. InfoNCE：以锚句与正例为正样本对，负例及跨样本表示作为负样本
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

    stage1_outputs = encoder(
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
    mask = input_ids == mask_token_id
    # 使用向量化方式提取所有 MASK token 的表示（参考 LMF 实现）
    last_hidden = stage1_outputs.last_hidden_state
    # 直接提取所有 MASK token 的表示
    h_all_masks = last_hidden[mask]  # [total_masks, hidden_size]
    
    # 计算每个样本的 MASK 数量
    mask_counts = mask.sum(dim=-1)  # [batch_size * num_sent]
    
    # 计算每个样本的最后一个 MASK 在 h_all_masks 中的索引
    # 使用累积和来定位每个样本的最后一个 MASK
    mask_cumsum = torch.cumsum(mask_counts, dim=0)  # [batch_size * num_sent]
    # 最后一个 MASK 的索引 = 累积和 - 1（因为索引从0开始）
    mask_indices = mask_cumsum - 1  # [batch_size * num_sent]
    
    # 处理没有 MASK 的情况（向后兼容）
    has_mask = mask_counts > 0
    if not has_mask.all():
        # 对于没有 MASK 的样本，使用 argmax 作为后备
        fallback_positions = mask.long().argmax(dim=-1)  # [batch_size * num_sent]
        batch_indices_fallback = torch.arange(input_ids.size(0), device=input_ids.device)
        h_fallback = last_hidden[batch_indices_fallback, fallback_positions]
        # 只替换没有 MASK 的样本
        h_flat = torch.where(has_mask.unsqueeze(-1), h_all_masks[mask_indices], h_fallback)
    else:
        # 所有样本都有 MASK，直接提取
        h_flat = h_all_masks[mask_indices]  # [batch_size * num_sent, hidden_size]
    
    h = h_flat.view(batch_size, num_sent, -1)

    if stage2_template is None:
        stage2_template = "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."

    stage2_template_ids = tokenizer.encode(stage2_template, add_special_tokens=False)
    it_token_id = tokenizer.convert_tokens_to_ids("[IT_SPECIAL_TOKEN]")
    it_pos_in_template = None
    if it_token_id is not None and it_token_id != tokenizer.unk_token_id:
        for idx, tid in enumerate(stage2_template_ids):
            if tid == it_token_id:
                it_pos_in_template = idx
                break

    flat_batch = h_flat.size(0)
    stage2_input_ids = []
    stage2_attention_masks = []
    for _ in range(flat_batch):
        ids = [tokenizer.cls_token_id] + stage2_template_ids + [tokenizer.sep_token_id]
        stage2_input_ids.append(ids)
        stage2_attention_masks.append([1] * len(ids))

    max_len = max(len(ids) for ids in stage2_input_ids)
    for i in range(flat_batch):
        pad_len = max_len - len(stage2_input_ids[i])
        stage2_input_ids[i].extend([tokenizer.pad_token_id] * pad_len)
        stage2_attention_masks[i].extend([0] * pad_len)

    stage2_input_ids = torch.tensor(stage2_input_ids, device=h_flat.device, dtype=torch.long)
    stage2_attention_mask = torch.tensor(stage2_attention_masks, device=h_flat.device, dtype=torch.long)

    stage2_embeddings = encoder.embeddings(stage2_input_ids)

    if it_pos_in_template is not None:
        it_pos_in_sequence = it_pos_in_template + 1
        position_ids_full = torch.arange(stage2_embeddings.size(1), device=stage2_embeddings.device).unsqueeze(0)
        position_embeddings = encoder.embeddings.position_embeddings(position_ids_full)

        token_type_zero = torch.zeros(
            (1, stage2_embeddings.size(1)),
            dtype=torch.long,
            device=stage2_embeddings.device,
        )
        token_type_embeddings = encoder.embeddings.token_type_embeddings(token_type_zero)

        replacement = (
            h_flat
            + position_embeddings[0, it_pos_in_sequence, :].unsqueeze(0)
            + token_type_embeddings[0, it_pos_in_sequence, :].unsqueeze(0)
        )
        stage2_embeddings[:, it_pos_in_sequence, :] = replacement

    stage2_outputs = encoder(
        input_ids=None,
        attention_mask=stage2_attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=head_mask,
        inputs_embeds=stage2_embeddings,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    mask_stage2 = stage2_input_ids == mask_token_id
    mask_positions_stage2 = mask_stage2.long().argmax(dim=-1)
    batch_indices_stage2 = torch.arange(stage2_input_ids.size(0), device=stage2_input_ids.device)
    h_plus_flat = stage2_outputs.last_hidden_state[batch_indices_stage2, mask_positions_stage2]
    h_plus = h_plus_flat.view(batch_size, num_sent, -1)

    # 融合两个阶段的表示，保留第一阶段语义
    h_fused = cls.stage_fusion(h, h_plus)  # (batch_size, num_sent, hidden_dim)

    eps = 1e-8
    loss = None
    logits = None
    loss_fct = nn.CrossEntropyLoss()

    if num_sent >= 3:
        # 提取三个模版的融合表示
        neg_vec = h_fused[:, 0, :]  # 负例模版的 h_fused
        anchor_vec = h_fused[:, 1, :]  # 锚句模版的 h_fused
        pos_vec = h_fused[:, 2, :]  # 正例模版的 h_fused

        # 归一化
        neg_norm = F.normalize(neg_vec, p=2, dim=-1, eps=eps)
        anchor_norm = F.normalize(anchor_vec, p=2, dim=-1, eps=eps)
        pos_norm = F.normalize(pos_vec, p=2, dim=-1, eps=eps)

        # 正样本对：锚句与正例的融合表示
        pos_sim = (anchor_norm * pos_norm).sum(dim=-1, keepdim=True) / cls.temperature
        pos_sim = torch.clamp(pos_sim, min=-50.0, max=50.0)

        # 构建负样本候选池：
        # 包含所有batch的所有h_fused（neg、anchor、pos），然后排除当前batch的anchor和pos
        h_fused_flat = h_fused.view(-1, h_fused.size(-1))  # (batch_size * 3, hidden_dim)
        h_fused_flat_norm = F.normalize(h_fused_flat, p=2, dim=-1, eps=eps)
        
        # 负样本候选池：所有batch的所有h_fused（包括neg、anchor、pos）
        candidate_bank = h_fused_flat_norm  # (batch_size * 3, hidden_dim)
        
        # 计算锚句与所有候选的相似度
        neg_sim = torch.mm(anchor_norm, candidate_bank.t()) / cls.temperature
        neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)

        # 排除自身：当前batch的anchor和pos不应该作为负样本
        batch_range = torch.arange(batch_size, device=anchor_norm.device)
        # 排除当前batch的anchor (索引: batch_size + batch_range，因为h_fused顺序是[neg, anchor, pos])
        neg_sim[:, batch_size + batch_range] = float("-inf")
        # 排除当前batch的pos (索引: 2 * batch_size + batch_range)
        neg_sim[:, 2 * batch_size + batch_range] = float("-inf")

        # 组合相似度矩阵：[正样本对, 负样本对]
        cos_sim = torch.cat([pos_sim, neg_sim], dim=1)
        labels_infonce = torch.zeros(batch_size, dtype=torch.long, device=anchor_norm.device)
        loss = loss_fct(cos_sim, labels_infonce)
        logits = anchor_vec
    else:
        h_single = h.squeeze(1)
        h_fused_single = h_fused.squeeze(1)

        h_norm = F.normalize(h_single, p=2, dim=-1, eps=eps)
        h_fused_norm = F.normalize(h_fused_single, p=2, dim=-1, eps=eps)

        pos_sim = (h_norm * h_fused_norm).sum(dim=-1, keepdim=True) / cls.temperature
        pos_sim = torch.clamp(pos_sim, min=-50.0, max=50.0)

        neg_sim = torch.mm(h_fused_norm, h_fused_norm.t()) / cls.temperature
        eye_mask = torch.eye(batch_size, device=h_norm.device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(eye_mask, float("-inf"))
        neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)
        neg_sim = neg_sim.masked_fill(eye_mask, float("-inf"))

        cos_sim = torch.cat([pos_sim, neg_sim], dim=1)
        labels_infonce = torch.zeros(batch_size, dtype=torch.long, device=h_norm.device)
        loss = loss_fct(cos_sim, labels_infonce)
        logits = h_fused_single

    if not return_dict:
        output = (logits,) + stage2_outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=stage2_outputs.hidden_states,
        attentions=stage2_outputs.attentions,
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
    stage1_templates=None,
    stage2_template=None,
    tokenizer=None,
):
    """
    句子嵌入前向传播（用于评估）
    默认使用锚句模版对应的 h_plus 作为句子表示
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

    stage1_outputs = encoder(
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
    mask = input_ids == mask_token_id
    # 使用向量化方式提取所有 MASK token 的表示（参考 LMF 实现）
    last_hidden = stage1_outputs.last_hidden_state
    # 直接提取所有 MASK token 的表示
    h_all_masks = last_hidden[mask]  # [total_masks, hidden_size]
    
    # 计算每个样本的 MASK 数量
    mask_counts = mask.sum(dim=-1)  # [batch_size * num_sent]
    
    # 计算每个样本的最后一个 MASK 在 h_all_masks 中的索引
    # 使用累积和来定位每个样本的最后一个 MASK
    mask_cumsum = torch.cumsum(mask_counts, dim=0)  # [batch_size * num_sent]
    # 最后一个 MASK 的索引 = 累积和 - 1（因为索引从0开始）
    mask_indices = mask_cumsum - 1  # [batch_size * num_sent]
    
    # 处理没有 MASK 的情况（向后兼容）
    has_mask = mask_counts > 0
    if not has_mask.all():
        # 对于没有 MASK 的样本，使用 argmax 作为后备
        fallback_positions = mask.long().argmax(dim=-1)  # [batch_size * num_sent]
        batch_indices_fallback = torch.arange(input_ids.size(0), device=input_ids.device)
        h_fallback = last_hidden[batch_indices_fallback, fallback_positions]
        # 只替换没有 MASK 的样本
        h_flat = torch.where(has_mask.unsqueeze(-1), h_all_masks[mask_indices], h_fallback)
    else:
        # 所有样本都有 MASK，直接提取
        h_flat = h_all_masks[mask_indices]  # [batch_size * num_sent, hidden_size]

    if stage2_template is None:
        stage2_template = "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."

    stage2_template_ids = tokenizer.encode(stage2_template, add_special_tokens=False)
    it_token_id = tokenizer.convert_tokens_to_ids("[IT_SPECIAL_TOKEN]")
    it_pos_in_template = None
    if it_token_id is not None and it_token_id != tokenizer.unk_token_id:
        for idx, tid in enumerate(stage2_template_ids):
            if tid == it_token_id:
                it_pos_in_template = idx
                break

    flat_batch = h_flat.size(0)
    stage2_input_ids = []
    stage2_attention_masks = []
    for _ in range(flat_batch):
        ids = [tokenizer.cls_token_id] + stage2_template_ids + [tokenizer.sep_token_id]
        stage2_input_ids.append(ids)
        stage2_attention_masks.append([1] * len(ids))

    max_len = max(len(ids) for ids in stage2_input_ids)
    for i in range(flat_batch):
        pad_len = max_len - len(stage2_input_ids[i])
        stage2_input_ids[i].extend([tokenizer.pad_token_id] * pad_len)
        stage2_attention_masks[i].extend([0] * pad_len)

    stage2_input_ids = torch.tensor(stage2_input_ids, device=h_flat.device, dtype=torch.long)
    stage2_attention_mask = torch.tensor(stage2_attention_masks, device=h_flat.device, dtype=torch.long)

    stage2_embeddings = encoder.embeddings(stage2_input_ids)

    if it_pos_in_template is not None:
        it_pos_in_sequence = it_pos_in_template + 1
        position_ids_full = torch.arange(stage2_embeddings.size(1), device=stage2_embeddings.device).unsqueeze(0)
        position_embeddings = encoder.embeddings.position_embeddings(position_ids_full)
        token_type_zero = torch.zeros(
            (1, stage2_embeddings.size(1)),
            dtype=torch.long,
            device=stage2_embeddings.device,
        )
        token_type_embeddings = encoder.embeddings.token_type_embeddings(token_type_zero)
        replacement = (
            h_flat
            + position_embeddings[0, it_pos_in_sequence, :].unsqueeze(0)
            + token_type_embeddings[0, it_pos_in_sequence, :].unsqueeze(0)
        )
        stage2_embeddings[:, it_pos_in_sequence, :] = replacement

    with torch.no_grad():
        stage2_outputs = encoder(
            input_ids=None,
            attention_mask=stage2_attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=head_mask,
            inputs_embeds=stage2_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

        mask_stage2 = stage2_input_ids == mask_token_id
        mask_positions_stage2 = mask_stage2.long().argmax(dim=-1)
        batch_indices_stage2 = torch.arange(stage2_input_ids.size(0), device=stage2_input_ids.device)
        h_plus_flat = stage2_outputs.last_hidden_state[batch_indices_stage2, mask_positions_stage2]
        h_plus_all = h_plus_flat.view(batch_size, num_sent, -1)

        # 将h reshape为(batch_size, num_sent, hidden_dim)以便融合
        h_all = h_flat.view(batch_size, num_sent, -1)
        
        # 融合两个阶段的表示，保留第一阶段语义
        h_fused_all = cls.stage_fusion(h_all, h_plus_all)  # (batch_size, num_sent, hidden_dim)

        if num_sent >= 3:
            pooler_output = h_fused_all[:, 1, :]
        else:
            pooler_output = h_fused_all.squeeze(1)

        eps = 1e-8
        pooler_output = F.normalize(pooler_output, p=2, dim=-1, eps=eps)

    if not return_dict:
        return (stage2_outputs[0], pooler_output) + stage2_outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=stage2_outputs.last_hidden_state,
        hidden_states=stage2_outputs.hidden_states,
    )


class BertForTwoStageCoT(BertPreTrainedModel):
    """
    BERT for Two-Stage Chain-of-Thought (TwoStageCoT)
    实现基于两阶段思维链的句子表示学习
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
        
        two_stage_cot_init(self, config, temperature=temperature)

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
        sent_emb=False,
        stage1_templates=None,
        stage2_template=None,
        tokenizer=None,
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
                stage1_templates=stage1_templates,
                stage2_template=stage2_template,
                tokenizer=tokenizer,
            )
        else:
            return two_stage_cot_forward(self, self.bert,
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
                stage1_templates=stage1_templates,
                stage2_template=stage2_template,
                tokenizer=tokenizer,
            )

