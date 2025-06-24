import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

import math

import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class UniformCircleLoss0(nn.Module):
    def __init__(self, num_bins=10, temperature=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature

    def stereographic_projection(self, x):
        """修复：返回二维坐标 [batch_size, 2]"""
        scale = 2.0 / (1 + x.pow(2).sum(dim=1, keepdim=True).sqrt())
        proj = x * scale  # [batch_size, 768]
        return proj[:, :2]  # 修复：截取前两维 [batch_size, 2]

    def compute_angles(self, x):
        """修复：确保 angles 形状为 [batch_size]"""
        x_proj = self.stereographic_projection(x)  # [batch_size, 2]
        angles = torch.atan2(x_proj[:, 1], x_proj[:, 0]) % (2 * math.pi)  # [batch_size]
        return angles

    def forward(self, x):
        angles = self.compute_angles(x)  # [batch_size]

        # 生成分箱边界
        bin_edges = torch.linspace(0, 2*math.pi, steps=self.num_bins+1, device=x.device)  # [num_bins+1]
        
        # 修复：确保维度匹配
        angles = angles.unsqueeze(1)  # [batch_size, 1]
        bin_edges = bin_edges.unsqueeze(0)  # [1, num_bins+1]
        
        # 计算环形距离
        dists = torch.fmod(angles - bin_edges + math.pi, 2*math.pi) - math.pi  # [batch_size, num_bins+1]
        
        # 计算概率分布
        logits = -dists
        probs = F.softmax(logits / self.temperature, dim=1)  # [batch_size, num_bins+1]
        
        # 合并首尾 bin（环形特性）
        probs = torch.cat([probs[:, 1:], probs[:, :1]], dim=1)  # 可选
        
        return probs

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniformCircleLoss(nn.Module):
    def __init__(self, num_bins=10, temperature=1.0, loss_type='kl'):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        self.loss_type = loss_type  # 'kl', 'entropy', 'variance'

    def stereographic_projection(self, x):
        scale = 2.0 / (1 + x.pow(2).sum(dim=1, keepdim=True).sqrt())
        proj = x * scale
        return proj[:, :2]

    def compute_angles(self, x):
        x_proj = self.stereographic_projection(x)
        angles = torch.atan2(x_proj[:, 1], x_proj[:, 0]) % (2 * math.pi)
        return angles

    def _compute_soft_bins(self, angles, bin_edges):
        angles = angles.unsqueeze(1)
        bin_edges = bin_edges.unsqueeze(0)
        dists = torch.fmod(angles - bin_edges + math.pi, 2*math.pi) - math.pi
        logits = -dists
        probs = F.softmax(logits / self.temperature, dim=1)
        return probs

    def forward(self, x):
        angles = self.compute_angles(x)
        bin_edges = torch.linspace(0, 2*math.pi, steps=self.num_bins+1, device=x.device)
        bin_probs = self._compute_soft_bins(angles, bin_edges)


        self.loss_type = 'kl'

        if self.loss_type == 'kl':
            uniform_dist = torch.ones_like(bin_probs) / self.num_bins
            loss = F.kl_div(bin_probs.log(), uniform_dist, reduction='batchmean')
        elif self.loss_type == 'entropy':
            entropy = -torch.sum(bin_probs * torch.log(bin_probs + 1e-8), dim=-1)
            loss = entropy.mean()
        elif self.loss_type == 'variance':
            mean_prob = bin_probs.mean(dim=-1)
            loss = torch.var(mean_prob)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config, scale=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*scale, config.hidden_size*scale)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    if cls.model_args.mask_embedding_sentence_org_mlp:
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        cls.mlp = MLPLayer(config, scale=cls.model_args.mask_embedding_sentence_num_masks)

    cls.sim = Similarity(temp = cls.model_args.temp)
       

    cls.init_weights()


from token_util import get_mask_token_id, get_null_token_id
from strategy_manage import get_strategy

def get_noise_inputs(input_ids, attention_mask, sent_positions, pad_token_id = 0):
    """将原始输入中的句子部分替换为PAD"""
    noise_input_ids = input_ids.clone()
    noise_attention_mask = attention_mask.clone()
    # null_token_id = get_null_token_id()
    for i, (start, end) in enumerate(sent_positions):
        noise_input_ids[i, start:end] = pad_token_id
        # noise_input_ids[i, start:end] = null_token_id
        noise_attention_mask[i, start:end] = 0

    device = input_ids.device
    noise_input_ids = torch.Tensor(noise_input_ids).to(device).long()
    noise_attention_mask = torch.Tensor(noise_attention_mask).to(device).long()
    return noise_input_ids, noise_attention_mask

def get_denoised_mask_outputs(encoder, input_ids, attention_mask, sent_positions, pad_token_id, mask_token_id):
    noise_input_ids, noise_attention_mask = get_noise_inputs(input_ids=input_ids, attention_mask=attention_mask, sent_positions=sent_positions, pad_token_id=pad_token_id)
    noise_outputs, noise_mask_outputs = cl_get_mask_outputs(encoder=encoder, input_ids=noise_input_ids, attention_mask=noise_attention_mask, mask_token_id=mask_token_id)
    template0_noise_mask_outputs = noise_mask_outputs[:, 0, :]
    template1_noise_mask_outputs = noise_mask_outputs[:, 1, :]

    template0_noise_mask_outputs_mean = template0_noise_mask_outputs.mean(dim=0)
    template1_noise_mask_outputs_mean = template1_noise_mask_outputs.mean(dim=0)

    noise_mask_outputs_clone = noise_mask_outputs.clone()
    noise_mask_outputs_clone[:, 0, :] = template0_noise_mask_outputs_mean
    noise_mask_outputs_clone[:, 1, :] = template1_noise_mask_outputs_mean
    return noise_outputs, noise_mask_outputs_clone
   


def cl_get_mask_outputs(encoder, input_ids, attention_mask, mask_token_id):
    # batch_size = input_ids.size(0)
    # num_sent = input_ids.size(1)

    # # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    mask = input_ids == mask_token_id
    mask_num = mask[0].sum().item()

    last_hidden = outputs.last_hidden_state
    mask_outputs = last_hidden[mask]
    
    mask_outputs = mask_outputs.view((-1, mask_num, mask_outputs.size(-1)))  # (batch_size * num_sent, mask_num, hidden_size)
    return outputs, mask_outputs



def evaluate(encoder, input_ids, attention_mask, sent_positions, mask_token_id, pad_token_id):
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
       # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)
    sent_positions = sent_positions.view((-1, sent_positions.size(-1)))  # (batch_size * num_sent, len)

    outputs, mask_outputs = cl_get_mask_outputs(encoder=encoder, input_ids=input_ids, attention_mask=attention_mask, mask_token_id=mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)

    # noised_outputs, noise_mask_outputs = get_denoised_mask_outputs(encoder=encoder,input_ids=input_ids, mask_outputs=mask_outputs, sent_positions=sent_positions, mask_token_id=mask_token_id, pad_token_id=pad_token_id)
   
    # noise_input_ids, noise_attention_mask = get_noise_inputs(input_ids, attention_mask, sent_positions, pad_token_id)

    # outputs, noise_mask_outputs = cl_get_mask_outputs(encoder, noise_input_ids, noise_attention_mask, mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)


    # denoised_mask_outputs = mask_outputs - noise_mask_outputs

    denoised_mask_outputs = mask_outputs

    denoised_mask_outputs = denoised_mask_outputs.view((batch_size, num_sent, -1, denoised_mask_outputs.size(-1))) # (batch_size, num_sent, mask_num, hidden_size)
    
    # denoised_mask_outputs = cls.mlp(denoised_mask_outputs)
    # pos_mask_output_pooler = denoised_mask_outputs[:,0,:,:].squeeze(1) 
    # pos_mask_output_pooler, _ = pos_mask_output_pooler.max(dim = 1)
    # pos_mask_output_pooler= pos_mask_output_pooler.sum(dim = 1)
    # pos_mask_output_pooler= pos_mask_output_pooler.mean(dim = 1)
    pos_mask_output_pooler = get_strategy().get_sent_output(denoised_mask_outputs)

    return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pos_mask_output_pooler,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
    )

def cl_forward(cls,
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
               sent_positions=None,
               sent_emb=False
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
    pad_token_id = cls.pad_token_id
    mask_token_id = cls.mask_token_id

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)
    sent_positions = sent_positions.view((-1, sent_positions.size(-1)))  # (batch_size * num_sent, len)

    outputs, mask_outputs = cl_get_mask_outputs(encoder, input_ids, attention_mask, mask_token_id=mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)

    if sent_emb:
        denoised_mask_outputs = mask_outputs
        denoised_mask_outputs = denoised_mask_outputs.view((batch_size, num_sent, -1, denoised_mask_outputs.size(-1))) # (batch_size, num_sent, mask_num, hidden_size)
        pos_mask_output_pooler = get_strategy().get_sent_output(denoised_mask_outputs)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pos_mask_output_pooler,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    if get_strategy().is_denoised_enabled():
        noise_input_ids, noise_attention_mask = get_noise_inputs(input_ids, attention_mask, sent_positions, pad_token_id=pad_token_id)

        outputs, noise_mask_outputs = cl_get_mask_outputs(encoder, noise_input_ids, noise_attention_mask, mask_token_id=mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)

        denoised_mask_outputs = mask_outputs - noise_mask_outputs
    else:
        denoised_mask_outputs = mask_outputs

    denoised_mask_outputs = cls.mlp(denoised_mask_outputs)
    denoised_mask_outputs = denoised_mask_outputs.view((batch_size, num_sent, -1, denoised_mask_outputs.size(-1))) # (batch_size, num_sent, mask_num, hidden_size)

    pos_pairs, neg_pairs = get_strategy().get_pos_neg_pairs(denoised_mask_outputs)
    # 计算正样本对的相似度（余弦相似度）
    pos_similarities = [cls.sim(vec1.unsqueeze(1), vec2.unsqueeze(0)) for vec1, vec2 in pos_pairs]  # 每个元素形状 (batch_size,)
    # pos_similarities = torch.stack(pos_similarities, dim = 0) # (num_pos, batch_size, batch_size)

    # 计算负样本对的相似度
    neg_similarities = [cls.sim(vec1.unsqueeze(1), vec2.unsqueeze(0)) for vec1, vec2 in neg_pairs]
    # neg_similarities = torch.stack(neg_similarities, dim = 0)  # (num_neg, batch_size, batch_size)


    # 合并相似度并计算 logits
    cos_sim = torch.cat([*pos_similarities, *neg_similarities], dim=1).to(input_ids.device)  # (num_pos + num_neg, batch_size)

    loss_fct = nn.CrossEntropyLoss()

    labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

    ce_loss = loss_fct(cos_sim, labels)

    #  负样本对相似度惩罚
    # neg2_similarities = [cls.sim(vec1.unsqueeze(1), vec2.unsqueeze(0)) for vec1, vec2 in neg2_pairs]
    # neg2_sim = torch.stack(neg2_similarities, dim=0) 
    # neg2_target = torch.ones_like(neg2_sim)  # 希望相似度接近1
    # sim_loss = F.mse_loss(neg2_sim, neg2_target)

    # neg2_weight = 1
    # loss = ce_loss + neg2_weight * sim_loss

     # 卡方损失（假设目标为均匀分布）
    sent_embedings = denoised_mask_outputs[:, 0, 0, :].squeeze(1)
    loss2 = cls.uniform_loss(sent_embedings)

     # 总损失
    loss = ce_loss + 0.5 * loss2

    # loss = ce_loss
    # loss = loss2


    # print(f"Current loss: {loss.item()}")  # 监控损失值

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states if not cls.model_args.only_embedding_training else None,
        attentions=outputs.attentions if not cls.model_args.only_embedding_training else None,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

  

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        self.total_length = 80
        # self.chi_square_loss = ChiSquareLoss(bins=10)
        self.uniform_loss = UniformCircleLoss(num_bins=4)
        # self.uniform_loss = DirectionEntropyLoss(alpha=0.1)
        cl_init(self, config)
    


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
        sent_positions=None,
    ):
  
        return cl_forward(self, self.bert,
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
                sent_positions=sent_positions,
                sent_emb=sent_emb
        )
