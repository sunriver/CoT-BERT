import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class ProjectionNetwork(nn.Module):
    """
    投影网络：将语义表示投影到新的空间
    参考MLPLayer架构，使用Tanh激活
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class PredictionNetwork(nn.Module):
    """
    预测网络：将两个语义方面拼接后预测
    类似SimCLR的projection head（2层MLP）
    """
    def __init__(self, config):
        super().__init__()
        # 输入是拼接的h1和h2
        self.layer1 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.layer2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, h1, h2):
        x = torch.cat([h1, h2], dim=-1)
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


def denoising(cls, encoder, template, device='cuda', evaluation=False, mask_num=2):
    """
    去噪函数：计算不包含句子的prompt模板的噪声
    参考code-for-LMF的denoising实现
    """
    with torch.set_grad_enabled(not evaluation):
        bs = cls.bs
        es = cls.es
        
        input_ids, attention_mask = [], []
        for i in range(cls.total_length - len(template) + 1):
            input_ids.append([template[0]] + 
                             bs + 
                             [cls.pad_token_id] * i +
                             es +
                             [template[-1]] +
                             [cls.pad_token_id] * (cls.total_length - len(template) - i))
            
            attention_mask.append([1] * (len(template) + i) + [0] * (cls.total_length - len(template) - i))

        input_ids = torch.Tensor(input_ids).to(device).long()
        attention_mask = torch.Tensor(attention_mask).to(device).long()

        if evaluation:
            with torch.no_grad():
                mask = input_ids == cls.mask_token_id    
                outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True, return_dict=True)            
                
                last_hidden = outputs.last_hidden_state
                noise = last_hidden[mask]
        else:
            mask = input_ids == cls.mask_token_id    
            outputs = encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              output_hidden_states=True, return_dict=True)            
            
            last_hidden = outputs.last_hidden_state
            noise = last_hidden[mask]

        noise = noise.view(-1, mask_num, noise.shape[-1])
        # 返回两个mask位置的噪声
        return noise, len(template)


def spr_init(cls, config):
    """
    SPR模型初始化函数
    """
    # 初始化投影网络
    cls.projection1 = ProjectionNetwork(config)
    cls.projection2 = ProjectionNetwork(config)
    
    # 初始化预测网络
    cls.prediction_net = PredictionNetwork(config)
    cls.prediction_net_proj = PredictionNetwork(config)
    
    cls.init_weights()


def spr_forward(cls,
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
    SPR前向传播函数
    实现自投影正则化训练逻辑
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (batch_size * num_sent, len)

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

    # 提取mask位置的hidden states
    last_hidden = outputs.last_hidden_state
    
    # 检查是否有mask_token_id属性
    if hasattr(cls, 'mask_token_id'):
        pooler_output = last_hidden[input_ids == cls.mask_token_id]
        # 重塑为(batch_size * num_sent, mask_num, hidden_size)
        pooler_output = pooler_output.view(-1, cls.mask_num, pooler_output.shape[-1])
        
        # 分离h1*和h2*
        h1_star = pooler_output[:, 0, :]  # 第一个mask位置
        h2_star = pooler_output[:, 1, :]  # 第二个mask位置
    else:
        # 如果没有mask token，使用CLS token作为替代
        h1_star = last_hidden[:, 0, :]  # CLS token
        h2_star = last_hidden[:, 0, :]  # 同样的CLS token（简化处理）

    # 去噪：计算噪声并减去
    if hasattr(cls, 'model_args') and cls.model_args.spr_denoising and hasattr(cls, 'mask_embedding_template'):
        noise, template_length = denoising(cls=cls, encoder=encoder, 
                                          template=cls.mask_embedding_template, 
                                          device=input_ids.device, 
                                          mask_num=cls.mask_num)
        
        # 根据句子长度计算对应的噪声
        token_length = attention_mask.sum(-1) - template_length
        noise_h1 = noise[:, 0, :][token_length]  # 第一个mask的噪声
        noise_h2 = noise[:, 1, :][token_length]  # 第二个mask的噪声
        
        # 去噪
        h1 = h1_star - noise_h1
        h2 = h2_star - noise_h2
    else:
        h1 = h1_star
        h2 = h2_star

    # 投影网络
    h1_proj = cls.projection1(h1)
    h2_proj = cls.projection2(h2)

    # 预测网络
    h_prediction = cls.prediction_net(h1, h2)
    h_prediction_proj = cls.prediction_net_proj(h1_proj, h2_proj)

    # SPR损失：投影前后预测的一致性
    # 使用L2范数正则化
    h_pred_norm = F.normalize(h_prediction, p=2, dim=-1)
    h_pred_proj_norm = F.normalize(h_prediction_proj, p=2, dim=-1)
    
    # 计算L2损失
    loss = F.mse_loss(h_pred_norm, h_pred_proj_norm)

    if not return_dict:
        output = (h_prediction,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    return SequenceClassifierOutput(
        loss=loss,
        logits=h_prediction,
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
    只使用h_prediction作为句子表示
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # 应用prompt模板
    if hasattr(cls, 'bs'):
        new_input_ids = []
        bs = torch.LongTensor(cls.bs).to(input_ids.device)
        es = torch.LongTensor(cls.es).to(input_ids.device)

        for i in input_ids:
            ss = i.shape[0]
            ii = i[i != cls.pad_token_id]

            ni = [ii[:1], bs]
            if ii.shape[0] > 2:
                ni += [ii[1:-1]]
            
            ni += [es, ii[-1:]]
            if ii.shape[0] < i.shape[0]:
                ni += [i[i == cls.pad_token_id]]
            
            ni = torch.cat(ni)
            new_input_ids.append(ni)

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = (input_ids != cls.pad_token_id).long()
        token_type_ids = None

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

    # 提取mask位置
    last_hidden = outputs.last_hidden_state
    pooler_output = last_hidden[input_ids == cls.mask_token_id]

    pooler_output = pooler_output.view(-1, cls.mask_num, pooler_output.shape[-1])
    
    # 分离h1*和h2*
    h1_star = pooler_output[:, 0, :]
    h2_star = pooler_output[:, 1, :]

    # 去噪
    if cls.model_args.spr_denoising:
        noise, template_length = denoising(cls=cls, encoder=encoder, 
                                          template=cls.mask_embedding_template, 
                                          device=input_ids.device, 
                                          evaluation=True, 
                                          mask_num=cls.mask_num)
        
        token_length = attention_mask.sum(-1) - template_length
        noise_h1 = noise[:, 0, :][token_length]
        noise_h2 = noise[:, 1, :][token_length]
        
        h1 = h1_star - noise_h1
        h2 = h2_star - noise_h2
    else:
        h1 = h1_star
        h2 = h2_star

    # 使用预测网络得到句子表示
    pooler_output = cls.prediction_net(h1, h2)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForSPR(BertPreTrainedModel):
    """
    BERT for Self-Projection Regularization (SPR)
    实现自投影正则化的BERT模型
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        self.total_length = 80

        spr_init(self, config)

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
            return spr_forward(self, self.bert,
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
