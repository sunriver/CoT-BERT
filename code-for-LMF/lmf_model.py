import torch
import torch.nn as nn
import torch.distributed as dist

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

import math

class MaskWeightLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(embed_dim = config.hidden_size, num_heads=8)
        self.pool = nn.AdaptiveAvgPool1d(1)  
        # self.fc = nn.Linear(embed_dim, output_dim)  # 调整维度

    def forward1(self, x):
        # input: (batch_size, mask_num, embed_dim)
        hidden_size = self.config.hidden_size
        Q = K = V = x
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_size)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, x):
        # pass
        # x: (batch_size, mask_num, embed_dim)
        attn_output, weights = self.attn(x, x, x)#
        # 沿序列长度维度池化
        pooled_output = self.pool(attn_output.transpose(1, 2)).squeeze(2) 
        return pooled_output
        




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


def get_noise_inputs(orig_input_ids, orig_attention_mask, sent_positions, device='cuda',  pad_token_id = 0):
    """将原始输入中的句子部分替换为PAD"""
    noise_input_ids = orig_input_ids.clone()
    noise_attention_mask = orig_attention_mask.clone()
    for i, (start, end) in enumerate(sent_positions):
        noise_input_ids[i, start:end] = pad_token_id
        # noise_attention_mask[i, start:end] = 0

    noise_input_ids = torch.Tensor(noise_input_ids).to(device).long()
    noise_attention_mask = torch.Tensor(noise_attention_mask).to(device).long()
    return noise_input_ids, noise_attention_mask


from token_util import get_mask_token_id
def cl_get_mask_outputs(encoder, input_ids, attention_mask, mask_token_id):
    # batch_size = input_ids.size(0)
    # num_sent = input_ids.size(1)

    # # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)

    mask_id, mask1_id, mask2_id = get_mask_token_id()
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    mask_num = 2

    last_hidden_state = outputs.last_hidden_state
    
    

    mask1 = (input_ids == mask1_id)  # 假设 mask_token_id 是列表 [mask1_id, mask2_id]
    mask2 = (input_ids == mask2_id)

    mask1_hidden = last_hidden_state[mask1]  # [MASK1] 的隐藏状态
    mask2_hidden = last_hidden_state[mask2]  # [MASK2] 的隐藏状态
    
    mask_outputs = torch.cat([mask1_hidden, mask2_hidden], dim=0)
    
    mask_outputs = mask_outputs.view((-1, mask_num, mask_outputs.size(-1)))  # (batch_size * num_sent, mask_num, hidden_size)
    return outputs, mask_outputs


def cl_get_mask_outputs2(encoder, input_ids, attention_mask, mask_token_id):
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

def get_pos_neg_pairs0(denoised_mask_outputs):
    pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
    pos_mask2_vec = denoised_mask_outputs[:, 0, 1]
    neg_mask1_vec = denoised_mask_outputs[:, 1, 0]
    neg_mask2_vec = denoised_mask_outputs[:, 1, 1]



    pos_pairs = [(pos_mask1_vec, pos_mask2_vec)]
    neg_pairs = [
        (pos_mask1_vec, neg_mask1_vec),
        (pos_mask1_vec, neg_mask2_vec),
        (pos_mask2_vec, neg_mask1_vec),
        (pos_mask2_vec, neg_mask2_vec)
    ]
    
    return pos_pairs, neg_pairs


def get_pos_neg_pairs(denoised_mask_outputs):
    pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
    pos_mask2_vec = denoised_mask_outputs[:, 1, 1]
    neg_mask1_vec = denoised_mask_outputs[:, 0, 1]
    neg_mask2_vec = denoised_mask_outputs[:, 1, 0]



    pos_pairs = [(pos_mask1_vec, pos_mask2_vec)]
    neg_pairs = [
        (pos_mask1_vec, neg_mask1_vec),
        (pos_mask1_vec, neg_mask2_vec),
        (pos_mask2_vec, neg_mask1_vec),
        (pos_mask2_vec, neg_mask2_vec)
    ]
    
    return pos_pairs, neg_pairs


def get_sent_output(denoised_mask_outputs):
    pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
    pos_mask2_vec = denoised_mask_outputs[:, 1, 1]
    sent_mask = (pos_mask1_vec + pos_mask2_vec) / 2
    return sent_mask


def evaluate(encoder, input_ids, attention_mask, sent_positions, mask_token_id, pad_token_id):
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
       # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)
    sent_positions = sent_positions.view((-1, sent_positions.size(-1)))  # (batch_size * num_sent, len)

    outputs, mask_outputs = cl_get_mask_outputs(encoder, input_ids, attention_mask, mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)

    noise_input_ids, noise_attention_mask = get_noise_inputs(input_ids, attention_mask, sent_positions, pad_token_id)

    outputs, noise_mask_outputs = cl_get_mask_outputs(encoder, noise_input_ids, noise_attention_mask, mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)

    denoised_mask_outputs = mask_outputs - noise_mask_outputs

    denoised_mask_outputs = denoised_mask_outputs.view((batch_size, num_sent, -1, denoised_mask_outputs.size(-1))) # (batch_size, num_sent, mask_num, hidden_size)
    
    # pos_mask_output_pooler = denoised_mask_outputs[:,0,:,:].squeeze(1) 
    # pos_mask_output_pooler, _ = pos_mask_output_pooler.max(dim = 1)
    # pos_mask_output_pooler= pos_mask_output_pooler.sum(dim = 1)
    # pos_mask_output_pooler= pos_mask_output_pooler.mean(dim = 1)
    pos_mask_output_pooler = get_sent_output(denoised_mask_outputs)

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

    noise_input_ids, noise_attention_mask = get_noise_inputs(input_ids, attention_mask, sent_positions, pad_token_id=pad_token_id)

    outputs, noise_mask_outputs = cl_get_mask_outputs(encoder, noise_input_ids, noise_attention_mask, mask_token_id=mask_token_id) # (batch_size * num_sent, mask_num, hidden_size)

    denoised_mask_outputs = mask_outputs - noise_mask_outputs

    denoised_mask_outputs = denoised_mask_outputs.view((batch_size, num_sent, -1, denoised_mask_outputs.size(-1))) # (batch_size, num_sent, mask_num, hidden_size)
    if sent_emb:
        # pos_mask_output_pooler = denoised_mask_outputs[:,0,:,:].squeeze(1) 
        # pos_mask_output_pooler, _ = pos_mask_output_pooler.max(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.sum(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.mean(dim = 1)
        pos_mask_output_pooler = get_sent_output(denoised_mask_outputs)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pos_mask_output_pooler,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )
        # return pos_mask_output_pooler

    # outputs = denoised_mask_outputs[:, 0].mean(dim=1)  # (batch_size, hidden_size)  

    # pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
    # pos_mask2_vec = denoised_mask_outputs[:, 0, 1]
    # neg_mask1_vec = denoised_mask_outputs[:, 1, 0]
    # neg_mask2_vec = denoised_mask_outputs[:, 1, 1]

    # pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
    # pos_mask2_vec = denoised_mask_outputs[:, 1, 1]
    # neg_mask1_vec = denoised_mask_outputs[:, 0, 1]
    # neg_mask2_vec = denoised_mask_outputs[:, 1, 0]



    # pos_pairs = [(pos_mask1_vec, pos_mask2_vec)]
    # neg_pairs = [
    #     (pos_mask1_vec, neg_mask1_vec),
    #     (pos_mask1_vec, neg_mask2_vec),
    #     (pos_mask2_vec, neg_mask1_vec),
    #     (pos_mask2_vec, neg_mask2_vec)
    # ]

    pos_pairs, neg_pairs = get_pos_neg_pairs(denoised_mask_outputs)
    # 计算正样本对的相似度（余弦相似度）
    pos_similarities = [cls.sim(vec1.unsqueeze(1), vec2.unsqueeze(0)) for vec1, vec2 in pos_pairs]  # 每个元素形状 (batch_size,)
    # pos_similarities = torch.stack(pos_similarities, dim = 0) # (num_pos, batch_size, batch_size)

    # 计算负样本对的相似度
    neg_similarities = [cls.sim(vec1.unsqueeze(1), vec2.unsqueeze(0)) for vec1, vec2 in neg_pairs]
    # neg_similarities = torch.stack(neg_similarities, dim = 0)  # (num_neg, batch_size, batch_size)


    # 合并相似度并计算 logits
    cos_sim = torch.cat([*pos_similarities, *neg_similarities], dim=1)  # (num_pos + num_neg, batch_size)

    loss_fct = nn.CrossEntropyLoss()

    labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

    loss = loss_fct(cos_sim, labels)

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
