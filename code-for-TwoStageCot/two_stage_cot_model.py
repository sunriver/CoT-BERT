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


def two_stage_cot_init(cls, config, temperature=0.05):
    """
    两阶段思维链模型初始化函数
    Args:
        config: 模型配置
        temperature: InfoNCE损失的温度参数（默认0.05）
    """
    # 初始化相似度计算模块（用于InfoNCE损失）
    cls.similarity = Similarity(temp=temperature)
    
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
                          stage1_template=None,
                          stage2_template=None,
                          tokenizer=None,
):
    """
    两阶段思维链前向传播函数
    实现两阶段模版处理：
    1. 第一阶段：使用模版1 "The sentence of [X] means [mask]." 获得 h
    2. 第二阶段：使用模版2 "so [IT_SPECIAL_TOKEN] can be summarized as [mask]."
       - 获取模版 embedding 矩阵
       - 将 [IT_SPECIAL_TOKEN] 位置的 token embedding 替换为 h
       - 输入到 BERT 得到 h+
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1) if len(input_ids.shape) == 3 else 1

    # Flatten input for encoding
    if len(input_ids.shape) == 3:
        input_ids = input_ids.view((-1, input_ids.size(-1)))
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    # ========== 第一阶段：使用模版1获得 h ==========
    # 第一阶段已经通过 input_ids 传入，直接编码
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

    # 提取第一阶段句子表示 h：从 [MASK] 位置提取（向量化实现）
    mask_token_id = cls.config.mask_token_id if hasattr(cls.config, 'mask_token_id') else 103
    
    # 创建 MASK 位置的布尔张量
    mask = input_ids == mask_token_id  # (batch_size * num_sent, seq_len)
    
    # 找到每个样本第一个 MASK 的位置（如果没有 MASK，argmax 返回 0，对应 CLS token）
    mask_positions = mask.long().argmax(dim=-1)  # (batch_size * num_sent,)
    
    # 使用高级索引提取对应的 hidden states
    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    h = stage1_outputs.last_hidden_state[batch_indices, mask_positions]  # (batch_size * num_sent, hidden_dim)
    
    # 重塑为 (batch_size, num_sent, hidden_dim)
    if num_sent > 1:
        h = h.view(batch_size, num_sent, -1)
        h = h[:, 0]  # 取第一个视图
    # h shape: (batch_size, hidden_dim)

    # ========== 第二阶段：使用模版2获得 h+ ==========
    # 准备第二阶段模版
    if stage2_template is None:
        stage2_template = "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."
    
    # 获取模版2的 token ids（不包含特殊token）
    stage2_template_ids = tokenizer.encode(stage2_template, add_special_tokens=False)
    
    # 找到 [IT_SPECIAL_TOKEN] 的位置（在模版中）
    # 使用特殊 token [IT_SPECIAL_TOKEN] 而不是普通词 "it"
    it_token_id = tokenizer.convert_tokens_to_ids('[IT_SPECIAL_TOKEN]')
    
    # 找到模版中 [IT_SPECIAL_TOKEN] 的位置
    it_pos_in_template = None
    if it_token_id is not None and it_token_id != tokenizer.unk_token_id:
        for idx, tid in enumerate(stage2_template_ids):
            if tid == it_token_id:
                it_pos_in_template = idx
                break
    
    # 构建完整的第二阶段输入（添加特殊token）
    # [CLS] + stage2_template + [SEP]
    stage2_input_ids = []
    stage2_attention_masks = []
    
    for i in range(batch_size):
        # 构建输入： [CLS] + template_tokens + [SEP]
        stage2_ids = [tokenizer.cls_token_id] + stage2_template_ids + [tokenizer.sep_token_id]
        stage2_input_ids.append(stage2_ids)
        stage2_attention_masks.append([1] * len(stage2_ids))
    
    # Padding 到相同长度
    max_len = max(len(ids) for ids in stage2_input_ids)
    for i in range(batch_size):
        pad_len = max_len - len(stage2_input_ids[i])
        stage2_input_ids[i].extend([tokenizer.pad_token_id] * pad_len)
        stage2_attention_masks[i].extend([0] * pad_len)
    
    stage2_input_ids = torch.tensor(stage2_input_ids, device=input_ids.device, dtype=torch.long)
    stage2_attention_mask = torch.tensor(stage2_attention_masks, device=input_ids.device, dtype=torch.long)
    
    # 获取模版 embedding（包含 token + position + token_type）
    stage2_embeddings = encoder.embeddings(stage2_input_ids)  # (batch_size, seq_len, hidden_dim)
    
    # 找到 [IT_SPECIAL_TOKEN] 在完整序列中的位置（考虑 [CLS]）
    if it_pos_in_template is not None:
        it_pos_in_sequence = it_pos_in_template + 1  # +1 因为前面有 [CLS]
        
        # 只替换 token embedding 部分，保留 position 和 token type embeddings
        # 获取该位置的 position 和 token type embeddings
        position_ids = torch.arange(stage2_embeddings.size(1), device=stage2_embeddings.device).unsqueeze(0)  # (1, seq_len)
        position_embeddings = encoder.embeddings.position_embeddings(position_ids)  # (1, seq_len, hidden_dim)
        
        token_type_ids = torch.zeros(stage2_embeddings.size(1), dtype=torch.long, device=stage2_embeddings.device).unsqueeze(0)  # (1, seq_len)
        token_type_embeddings = encoder.embeddings.token_type_embeddings(token_type_ids)  # (1, seq_len, hidden_dim)
        
        # 替换：h + position_embedding + token_type_embedding
        # h 是第一阶段提取的完整 hidden state，但我们需要用第二阶段的位置和类型信息
        stage2_embeddings[:, it_pos_in_sequence, :] = (
            h + 
            position_embeddings[0, it_pos_in_sequence, :] + 
            token_type_embeddings[0, it_pos_in_sequence, :]
        )
    
    # 使用替换后的 embedding 输入到 BERT
    stage2_outputs = encoder(
        input_ids=None,  # 使用 inputs_embeds 而不是 input_ids
        attention_mask=stage2_attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=head_mask,
        inputs_embeds=stage2_embeddings,  # 使用替换后的 embedding
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )
    
    # 提取第二阶段句子表示 h+：从 [MASK] 位置提取（向量化实现）
    # 创建 MASK 位置的布尔张量
    mask = stage2_input_ids == mask_token_id  # (batch_size, seq_len)
    
    # 找到每个样本第一个 MASK 的位置（如果没有 MASK，argmax 返回 0，对应 CLS token）
    mask_positions = mask.long().argmax(dim=-1)  # (batch_size,)
    
    # 使用高级索引提取对应的 hidden states
    batch_indices = torch.arange(batch_size, device=stage2_input_ids.device)
    h_plus = stage2_outputs.last_hidden_state[batch_indices, mask_positions]  # (batch_size, hidden_dim)

    # ========== 计算 InfoNCE 损失 ==========
    # 正样本对：(h, h+)
    # 负样本对：h+ 和批次中其他句子的 h+
    
    loss_fct = nn.CrossEntropyLoss()
    
    # 归一化用于计算相似度（添加小的epsilon避免零向量导致NaN）
    eps = 1e-8
    h_norm = F.normalize(h, p=2, dim=-1, eps=eps)
    h_plus_norm = F.normalize(h_plus, p=2, dim=-1, eps=eps)
    
    # 计算正样本对相似度（h[i] 与 h_plus[i]）
    pos_sim = (h_norm * h_plus_norm).sum(dim=-1, keepdim=True) / cls.temperature  # (batch_size, 1)
    pos_sim = torch.clamp(pos_sim, min=-50.0, max=50.0)
    
    # 计算负样本对相似度（h_plus[i] 与 h_plus[j], i != j）
    neg_sim = torch.mm(h_plus_norm, h_plus_norm.t()) / cls.temperature  # (batch_size, batch_size)
    
    # 将对角线位置设为负无穷（排除自己）
    eye_mask = torch.eye(batch_size, device=h_norm.device, dtype=torch.bool)
    neg_sim = neg_sim.masked_fill(eye_mask, float('-inf'))
    
    # 数值稳定性保护
    neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)
    neg_sim = neg_sim.masked_fill(eye_mask, float('-inf'))
    
    # 组合相似度矩阵：[正样本对, 负样本对]
    cos_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, batch_size + 1)
    
    # 标签：第一列（索引0）是正样本对
    labels_infonce = torch.zeros(batch_size, dtype=torch.long, device=h_norm.device)
    
    # 计算 InfoNCE 损失
    loss = loss_fct(cos_sim, labels_infonce)
    
    # 使用 h+ 作为输出表示
    logits = h_plus

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
    stage1_template=None,
    stage2_template=None,
    tokenizer=None,
):
    """
    句子嵌入前向传播（用于评估）
    返回两阶段处理后的句子表示 h+
    支持SentEval STS任务：每个句子独立处理
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # 第一阶段：使用模版1编码句子
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

    # 提取第一阶段句子表示 h：从 [MASK] 位置提取（向量化实现）
    mask_token_id = cls.config.mask_token_id if hasattr(cls.config, 'mask_token_id') else 103
    batch_size = input_ids.size(0)
    
    # 创建 MASK 位置的布尔张量
    mask = input_ids == mask_token_id  # (batch_size, seq_len)
    
    # 找到每个样本第一个 MASK 的位置（如果没有 MASK，argmax 返回 0，对应 CLS token）
    mask_positions = mask.long().argmax(dim=-1)  # (batch_size,)
    
    # 使用高级索引提取对应的 hidden states
    batch_indices = torch.arange(batch_size, device=input_ids.device)
    h = stage1_outputs.last_hidden_state[batch_indices, mask_positions]  # (batch_size, hidden_dim)

    # 第二阶段：使用模版2获得 h+
    if stage2_template is None:
        stage2_template = "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."
    
    stage2_template_ids = tokenizer.encode(stage2_template, add_special_tokens=False)
    
    # 找到 [IT_SPECIAL_TOKEN] 的位置（在模版中）
    # 使用特殊 token [IT_SPECIAL_TOKEN] 而不是普通词 "it"
    it_token_id = tokenizer.convert_tokens_to_ids('[IT_SPECIAL_TOKEN]')
    
    # 找到模版中 [IT_SPECIAL_TOKEN] 的位置
    it_pos_in_template = None
    if it_token_id is not None and it_token_id != tokenizer.unk_token_id:
        for idx, tid in enumerate(stage2_template_ids):
            if tid == it_token_id:
                it_pos_in_template = idx
                break
    
    # 构建第二阶段输入
    stage2_input_ids = []
    stage2_attention_masks = []
    
    for i in range(batch_size):
        stage2_ids = [tokenizer.cls_token_id] + stage2_template_ids + [tokenizer.sep_token_id]
        stage2_input_ids.append(stage2_ids)
        stage2_attention_masks.append([1] * len(stage2_ids))
    
    max_len = max(len(ids) for ids in stage2_input_ids)
    for i in range(batch_size):
        pad_len = max_len - len(stage2_input_ids[i])
        stage2_input_ids[i].extend([tokenizer.pad_token_id] * pad_len)
        stage2_attention_masks[i].extend([0] * pad_len)
    
    stage2_input_ids = torch.tensor(stage2_input_ids, device=input_ids.device, dtype=torch.long)
    stage2_attention_mask = torch.tensor(stage2_attention_masks, device=input_ids.device, dtype=torch.long)
    
    # 获取模版 embedding（包含 token + position + token_type）
    stage2_embeddings = encoder.embeddings(stage2_input_ids)
    
    if it_pos_in_template is not None:
        it_pos_in_sequence = it_pos_in_template + 1
        
        # 只替换 token embedding 部分，保留 position 和 token type embeddings
        # 获取该位置的 position 和 token type embeddings
        position_ids = torch.arange(stage2_embeddings.size(1), device=stage2_embeddings.device).unsqueeze(0)  # (1, seq_len)
        position_embeddings = encoder.embeddings.position_embeddings(position_ids)  # (1, seq_len, hidden_dim)
        
        token_type_ids = torch.zeros(stage2_embeddings.size(1), dtype=torch.long, device=stage2_embeddings.device).unsqueeze(0)  # (1, seq_len)
        token_type_embeddings = encoder.embeddings.token_type_embeddings(token_type_ids)  # (1, seq_len, hidden_dim)
        
        # 替换：h + position_embedding + token_type_embedding
        # h 是第一阶段提取的完整 hidden state，但我们需要用第二阶段的位置和类型信息
        stage2_embeddings[:, it_pos_in_sequence, :] = (
            h + 
            position_embeddings[0, it_pos_in_sequence, :] + 
            token_type_embeddings[0, it_pos_in_sequence, :]
        )
    
    # 第二阶段编码
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
        
        # 提取 h+（向量化实现）
        # 创建 MASK 位置的布尔张量
        mask = stage2_input_ids == mask_token_id  # (batch_size, seq_len)
        
        # 找到每个样本第一个 MASK 的位置（如果没有 MASK，argmax 返回 0，对应 CLS token）
        mask_positions = mask.long().argmax(dim=-1)  # (batch_size,)
        
        # 使用高级索引提取对应的 hidden states
        batch_indices = torch.arange(batch_size, device=stage2_input_ids.device)
        pooler_output = stage2_outputs.last_hidden_state[batch_indices, mask_positions]  # (batch_size, hidden_dim)
        
        # 归一化最终输出用于评估（添加小的epsilon避免零向量导致NaN）
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
        stage1_template=None,
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
                stage1_template=stage1_template,
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
                stage1_template=stage1_template,
                stage2_template=stage2_template,
                tokenizer=tokenizer,
            )

