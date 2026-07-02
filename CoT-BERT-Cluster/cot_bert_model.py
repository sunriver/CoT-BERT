import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from lmf_log_util import getMyLogger

logger = getMyLogger(__name__)


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

def compute_constraint_loss(h1_anchor, h1_positive, h2_anchor, h2_positive, eps=1e-8):
    """
    计算约束项损失 L3
    
    L3 = (||(h2_positive - h1_anchor)|| + ||(h2_anchor - h1_positive)||) 
         / (||h2_anchor|| + ||h2_positive|| + ε)
    
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
    denominator = norm_h2_anchor + norm_h2_positive + eps
    
    # 确保分母不会太小，避免数值不稳定
    denominator = torch.clamp(denominator, min=eps * 10)
    
    # 计算损失
    loss = (numerator / denominator).mean()
    
    # 检查并处理 NaN 和 Inf
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
    
    return loss

def denoising(cls, encoder, template, type='pos-1', device='cuda', evaluation=False):
    with torch.set_grad_enabled(not cls.model_args.mask_embedding_sentence_delta_freeze and not evaluation):
        if type == 'pos-1':
            bs = cls.bs
            es = cls.es
        elif type == 'pos-2':
            bs = cls.bs2
            es = cls.es2
        elif type == 'neg-1':
            bs = cls.bs3
            es = cls.es3
        elif type == 'neg-2':
            bs = cls.bs4
            es = cls.es4
        else:
            raise ValueError(f'unknown type {type}')
        
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

        # CoT-BERT Authors: Since we haven't made any modifications related to the auto-prompt, 
        #                   there's a high probability that the following code may not function correctly.        
        if cls.model_args.mask_embedding_sentence_autoprompt:
            inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
            p = torch.arange(input_ids.shape[1]).to(device).view(1, -1)
            b = torch.arange(input_ids.shape[0]).to(device)

            for i, k in enumerate(cls.dict_mbv):
                if cls.fl_mbv[i]:
                    index = ((input_ids == k) * p).max(-1)[1]
                else:
                    index = ((input_ids == k) * - p).min(-1)[1]

                inputs_embeds[b, index] = cls.p_mbv[i]
        else:
            inputs_embeds = None

        if evaluation:
            with torch.no_grad():
                mask = input_ids == cls.mask_token_id    
                outputs = encoder(input_ids=input_ids if inputs_embeds is None else None,
                                  inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True, return_dict=True)            
                
                last_hidden = outputs.last_hidden_state
                noise = last_hidden[mask]
        else:
            mask = input_ids == cls.mask_token_id    
            outputs = encoder(input_ids=input_ids if inputs_embeds is None else None,
                              inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              output_hidden_states=True, return_dict=True)            
            
            last_hidden = outputs.last_hidden_state
            noise = last_hidden[mask]

        noise = noise.view(-1, cls.mask_num, noise.shape[-1])
        # 返回所有MASK位置的噪声，形状为 [max_pad_length, mask_num, hidden_size]
        # 不再切片，以支持双MASK损失计算

        return noise, len(template)


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    if cls.model_args.mask_embedding_sentence_org_mlp:
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        # 设置为scale=1，以支持分别处理每个MASK的表示（与CrossTemplateCoT对齐）
        cls.mlp = MLPLayer(config, scale=1)
    
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def _momentum_bank_is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _momentum_bank_format_stats(stats):
    return " ".join(f"{k}={v}" for k, v in stats.items())


def momentum_bank_init(cls, config, encoder):
    """Initialize momentum key encoder and feature queue buffers."""
    cls.encoder_k = copy.deepcopy(encoder)
    for param in cls.encoder_k.parameters():
        param.requires_grad = False

    hidden_size = config.hidden_size
    queue_size = cls.model_args.queue_size
    num_clusters = cls.model_args.num_clusters

    queue = torch.randn(hidden_size, queue_size)
    queue = F.normalize(queue, dim=0)
    cls.register_buffer("queue", queue)
    cls.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    cls.register_buffer("queue_cluster_ids", torch.full((queue_size,), -1, dtype=torch.long))
    centroids = F.normalize(torch.randn(num_clusters, hidden_size), dim=-1)
    cls.register_buffer("cluster_centroids", centroids)


@torch.no_grad()
def momentum_bank_update_key_encoder(cls):
    """EMA update: theta_k <- m * theta_k + (1 - m) * theta_q."""
    m = cls.model_args.momentum
    encoder_q = cls._momentum_encoder_q
    encoder_k = cls.encoder_k
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)


@torch.no_grad()
def momentum_bank_dequeue_and_enqueue(cls, keys, cluster_ids):
    """FIFO enqueue normalized keys and their cluster ids into the queue."""
    batch_size = keys.shape[0]
    ptr = int(cls.queue_ptr.item())
    queue_size = cls.queue.shape[1]

    end = ptr + batch_size
    if end <= queue_size:
        cls.queue[:, ptr:end] = keys.T
        cls.queue_cluster_ids[ptr:end] = cluster_ids
    else:
        first = queue_size - ptr
        cls.queue[:, ptr:] = keys[:first].T
        cls.queue_cluster_ids[ptr:] = cluster_ids[:first]
        remain = batch_size - first
        cls.queue[:, :remain] = keys[first:].T
        cls.queue_cluster_ids[:remain] = cluster_ids[first:]

    cls.queue_ptr[0] = (ptr + batch_size) % queue_size


def momentum_bank_assign_cluster_ids(cls, embeddings):
    """Assign each embedding to nearest cluster centroid."""
    embeddings = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(embeddings, cls.cluster_centroids.T)
    return sim.argmax(dim=-1)


@torch.no_grad()
def momentum_bank_run_kmeans(cls):
    """Cluster queue features with Faiss K-Means and refresh pseudo-labels."""
    try:
        import faiss
    except ImportError:
        if _momentum_bank_is_main_process():
            logger.warning("[MomentumBank/KMeans] faiss not installed, skipping K-Means update")
        return

    features = cls.queue.T.contiguous().cpu().numpy().astype("float32")
    d = features.shape[1]
    queue_size = features.shape[0]
    k = min(cls.model_args.num_clusters, queue_size)
    if k < 2:
        return

    kmeans = faiss.Kmeans(d, k, niter=20, verbose=False)
    kmeans.train(features)

    centroids = torch.tensor(kmeans.centroids, device=cls.queue.device, dtype=cls.queue.dtype)
    cls.cluster_centroids[:k].copy_(F.normalize(centroids, dim=-1))
    if k < cls.cluster_centroids.shape[0]:
        cls.cluster_centroids[k:].zero_()

    _, cluster_ids = kmeans.index.search(features, 1)
    cluster_ids = torch.tensor(cluster_ids.squeeze(), device=cls.queue.device, dtype=torch.long)
    cls.queue_cluster_ids.copy_(cluster_ids)

    counts = torch.bincount(cluster_ids, minlength=k)
    nonzero_counts = counts[counts > 0].float()
    stats = {
        "k": k,
        "queue_size": queue_size,
        "unique_clusters": int(cluster_ids.unique().numel()),
        "cluster_size_min": int(nonzero_counts.min().item()) if nonzero_counts.numel() > 0 else 0,
        "cluster_size_max": int(nonzero_counts.max().item()) if nonzero_counts.numel() > 0 else 0,
        "cluster_size_mean": round(nonzero_counts.mean().item(), 2) if nonzero_counts.numel() > 0 else 0.0,
    }
    cls._momentum_kmeans_stats = stats
    if _momentum_bank_is_main_process():
        logger.info("[MomentumBank/KMeans] %s", _momentum_bank_format_stats(stats))


def momentum_bank_soft_suppress_queue_logits(cls, q, queue_logits):
    """
    Apply soft suppression on queue logits.
    queue_logits: [B, queue_size] (cosine sim / temperature, pre-softmax)
    """
    beta = cls.model_args.soft_suppression_beta
    threshold = cls.model_args.fn_cluster_sim_threshold

    q_cluster_ids = momentum_bank_assign_cluster_ids(cls, q)
    same_cluster = q_cluster_ids.unsqueeze(1) == cls.queue_cluster_ids.unsqueeze(0)

    raw_sim = queue_logits * cls.model_args.temp
    p_fn = torch.sigmoid(cls.model_args.temp * (raw_sim - threshold))
    p_fn = torch.where(
        same_cluster,
        torch.maximum(p_fn, torch.tensor(0.5, device=p_fn.device)),
        p_fn,
    )

    weights = (1.0 - p_fn).clamp(min=1e-6).pow(beta)

    if cls.training:
        with torch.no_grad():
            cls._momentum_bank_stats = {
                "same_cluster_rate": round(same_cluster.float().mean().item(), 4),
                "p_fn_mean": round(p_fn.mean().item(), 4),
                "p_fn_max": round(p_fn.max().item(), 4),
                "weight_mean": round(weights.mean().item(), 4),
                "weight_min": round(weights.min().item(), 6),
                "strong_suppress_rate": round((weights < 0.1).float().mean().item(), 4),
                "queue_logits_mean": round(queue_logits.mean().item(), 4),
            }

    return queue_logits + torch.log(weights)


def _compute_pooler_output(
    cls,
    encoder,
    input_ids,
    attention_mask,
    token_type_ids,
    position_ids,
    head_mask,
    batch_size,
    num_sent,
    noise1=None,
    noise2=None,
    noise3=None,
    noise4=None,
    template_length1=0,
    template_length2=0,
    template_length3=0,
    template_length4=0,
):
    """Run encoder forward and return [batch_size, num_sent, mask_num, hidden_size] pooler output."""
    inputs_embeds = None
    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
        p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
        b = torch.arange(input_ids.shape[0]).to(input_ids.device)

        for i, k in enumerate(cls.dict_mbv):
            if cls.model_args.mask_embedding_sentence_autoprompt_continue_training_as_positive and i % 2 == 0:
                continue

            if cls.fl_mbv[i]:
                index = ((input_ids == k) * p).max(-1)[1]
            else:
                index = ((input_ids == k) * -p).min(-1)[1]

            inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=None,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.mask_embedding_sentence:
        last_hidden = outputs.last_hidden_state
        pooler_output = last_hidden[input_ids == cls.mask_token_id]
        pooler_output = pooler_output.view(-1, cls.mask_num, pooler_output.shape[-1])

        if cls.model_args.mask_embedding_sentence_delta:
            if cls.model_args.mask_embedding_sentence_org_mlp:
                pooler_output = cls.mlp(pooler_output)

            pooler_output = pooler_output.view(batch_size, num_sent, cls.mask_num, -1)
            attention_mask_view = attention_mask.view(batch_size, num_sent, -1)
            entire_length = attention_mask_view.sum(-1)

            token_length = entire_length - template_length1
            max_idx = noise1.size(0) - 1
            token_length = torch.clamp(token_length, 0, max_idx)
            pooler_output[:, 0, 0, :] -= noise1[token_length[:, 0], 0, :]
            pooler_output[:, 0, 1, :] -= noise1[token_length[:, 0], 1, :]

            if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
                token_length = entire_length - template_length2
                token_length = torch.clamp(token_length, 0, max_idx)
                pooler_output[:, 1, 0, :] -= noise2[token_length[:, 1], 0, :]
                pooler_output[:, 1, 1, :] -= noise2[token_length[:, 1], 1, :]

                if num_sent == 3 and len(cls.model_args.mask_embedding_sentence_negative_template) == 0:
                    pooler_output[:, 2, 0, :] -= noise3[token_length[:, 2], 0, :]
                    pooler_output[:, 2, 1, :] -= noise3[token_length[:, 2], 1, :]

                if len(cls.model_args.mask_embedding_sentence_negative_template) > 0:
                    token_length = entire_length - template_length3
                    token_length = torch.clamp(token_length, 0, max_idx)
                    pooler_output[:, 2, 0, :] -= noise3[token_length[:, 2], 0, :]
                    pooler_output[:, 2, 1, :] -= noise3[token_length[:, 2], 1, :]

                if len(cls.model_args.mask_embedding_sentence_different_negative_template) > 0:
                    token_length = entire_length - template_length4
                    token_length = torch.clamp(token_length, 0, max_idx)
                    pooler_output[:, 3, 0, :] -= noise4[token_length[:, 3], 0, :]
                    pooler_output[:, 3, 1, :] -= noise4[token_length[:, 3], 1, :]
            else:
                token_length = entire_length - template_length1
                token_length = torch.clamp(token_length, 0, max_idx)
                for sent_idx in range(num_sent):
                    pooler_output[:, sent_idx, 0, :] -= noise1[token_length[:, sent_idx], 0, :]
                    pooler_output[:, sent_idx, 1, :] -= noise1[token_length[:, sent_idx], 1, :]

        if (
            not cls.model_args.mask_embedding_sentence_delta
            or not cls.model_args.mask_embedding_sentence_org_mlp
        ):
            pooler_output = pooler_output.view(batch_size * num_sent * cls.mask_num, -1)
            pooler_output = cls.mlp(pooler_output)
            pooler_output = pooler_output.view(batch_size, num_sent, cls.mask_num, -1)
        elif pooler_output.dim() == 3:
            pooler_output = pooler_output.view(batch_size, num_sent, cls.mask_num, -1)

    return pooler_output


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
):
    if cls.model_args.mask_embedding_sentence_delta:
        noise1, template_length1 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template, type='pos-1', device=input_ids.device)

        if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
            noise2, template_length2 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template2, type='pos-2', device=input_ids.device)
        else:
            noise2, template_length2 = None, 0

        if len(cls.model_args.mask_embedding_sentence_negative_template) > 0:
            noise3, template_length3 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template3, type='neg-1', device=input_ids.device)
        else:
            noise3, template_length3 = None, 0

        if len(cls.model_args.mask_embedding_sentence_different_negative_template) > 0:
            noise4, template_length4 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template4, type='neg-2', device=input_ids.device)
        else:
            noise4, template_length4 = None, 0
    else:
        noise1 = noise2 = noise3 = noise4 = None
        template_length1 = template_length2 = template_length3 = template_length4 = 0

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (batch_size * num_sent, len)

    pooler_output = _compute_pooler_output(
        cls=cls,
        encoder=encoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        batch_size=batch_size,
        num_sent=num_sent,
        noise1=noise1,
        noise2=noise2,
        noise3=noise3,
        noise4=noise4,
        template_length1=template_length1,
        template_length2=template_length2,
        template_length3=template_length3,
        template_length4=template_length4,
    )

    # 最终形状: [batch_size, num_sent, mask_num, hidden_size]

    # Separate representation for each MASK position
    # 提取第一个MASK和第二个MASK的表示
    z1_m1 = pooler_output[:, 0, 0, :]  # [batch_size, hidden_size] - 第一个sent（template），第一个MASK（含义）
    z1_m2 = pooler_output[:, 0, 1, :]  # [batch_size, hidden_size] - 第一个sent（template），第二个MASK（总结）
    
    # different_template 是反向因果关系：总结 → 含义
    # 为了与 template 对齐（含义 → 总结），需要交换 z2_m1 和 z2_m2 的提取位置
    z2_m1 = pooler_output[:, 1, 1, :]  # [batch_size, hidden_size] - 第二个sent（different_template），第二个MASK（含义，与z1_m1对齐）
    z2_m2 = pooler_output[:, 1, 0, :]  # [batch_size, hidden_size] - 第二个sent（different_template），第一个MASK（总结，与z1_m2对齐）

    # Hard negative
    if num_sent == 3:
        z3_m1 = pooler_output[:, 2, 0, :]  # [batch_size, hidden_size] - 第三个sent，第一个MASK
        z3_m2 = pooler_output[:, 2, 1, :]  # [batch_size, hidden_size] - 第三个sent，第二个MASK
    elif num_sent == 4:
        z3_m1, z4_m1 = pooler_output[:, 2, 0, :], pooler_output[:, 3, 0, :]
        z3_m2, z4_m2 = pooler_output[:, 2, 1, :], pooler_output[:, 3, 1, :]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative for MASK 1
        if num_sent == 3:
            z3_m1_list = [torch.zeros_like(z3_m1) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_m1_list, tensor=z3_m1.contiguous())
            z3_m1_list[dist.get_rank()] = z3_m1
            z3_m1 = torch.cat(z3_m1_list, 0)
        elif num_sent == 4:
            z3_m1_list = [torch.zeros_like(z3_m1) for _ in range(dist.get_world_size())]
            z4_m1_list = [torch.zeros_like(z4_m1) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_m1_list, tensor=z3_m1.contiguous())
            dist.all_gather(tensor_list=z4_m1_list, tensor=z4_m1.contiguous())
            z3_m1_list[dist.get_rank()] = z3_m1
            z4_m1_list[dist.get_rank()] = z4_m1
            z3_m1 = torch.cat(z3_m1_list, 0)
            z4_m1 = torch.cat(z4_m1_list, 0)

        # Gather hard negative for MASK 2
        if num_sent == 3:
            z3_m2_list = [torch.zeros_like(z3_m2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_m2_list, tensor=z3_m2.contiguous())
            z3_m2_list[dist.get_rank()] = z3_m2
            z3_m2 = torch.cat(z3_m2_list, 0)
        elif num_sent == 4:
            z3_m2_list = [torch.zeros_like(z3_m2) for _ in range(dist.get_world_size())]
            z4_m2_list = [torch.zeros_like(z4_m2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_m2_list, tensor=z3_m2.contiguous())
            dist.all_gather(tensor_list=z4_m2_list, tensor=z4_m2.contiguous())
            z3_m2_list[dist.get_rank()] = z3_m2
            z4_m2_list[dist.get_rank()] = z4_m2
            z3_m2 = torch.cat(z3_m2_list, 0)
            z4_m2 = torch.cat(z4_m2_list, 0)

        # Gather z1 and z2 for MASK 1
        z1_m1_list = [torch.zeros_like(z1_m1) for _ in range(dist.get_world_size())]
        z2_m1_list = [torch.zeros_like(z2_m1) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=z1_m1_list, tensor=z1_m1.contiguous())
        dist.all_gather(tensor_list=z2_m1_list, tensor=z2_m1.contiguous())
        z1_m1_list[dist.get_rank()] = z1_m1
        z2_m1_list[dist.get_rank()] = z2_m1
        z1_m1 = torch.cat(z1_m1_list, 0)
        z2_m1 = torch.cat(z2_m1_list, 0)

        # Gather z1 and z2 for MASK 2
        z1_m2_list = [torch.zeros_like(z1_m2) for _ in range(dist.get_world_size())]
        z2_m2_list = [torch.zeros_like(z2_m2) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=z1_m2_list, tensor=z1_m2.contiguous())
        dist.all_gather(tensor_list=z2_m2_list, tensor=z2_m2.contiguous())
        z1_m2_list[dist.get_rank()] = z1_m2
        z2_m2_list[dist.get_rank()] = z2_m2
        z1_m2 = torch.cat(z1_m2_list, 0)
        z2_m2 = torch.cat(z2_m2_list, 0)

    # 计算第一个MASK位置的InfoNCE损失 (L1)
    if cls.model_args.dot_sim:
        cos_sim_m1 = torch.mm(torch.sigmoid(z1_m1), torch.sigmoid(z2_m1.permute(1, 0)))
    else:
        cos_sim_m1 = cls.sim(z1_m1.unsqueeze(1), z2_m1.unsqueeze(0))

    if cls.model_args.norm_instead_temp:
        cos_sim_m1 *= cls.sim.temp
        cmin, cmax = cos_sim_m1.min(), cos_sim_m1.max()
        # 添加数值稳定性保护：防止除以零
        eps = 1e-8
        denominator = cmax - cmin
        denominator = torch.clamp(denominator, min=eps)
        cos_sim_m1 = (cos_sim_m1 - cmin) / denominator / cls.sim.temp

    if num_sent == 3:
        z1_m1_z3_m1_cos = cls.sim(z1_m1.unsqueeze(1), z3_m1.unsqueeze(0))
        z2_m1_z3_m1_cos = cls.sim(z2_m1.unsqueeze(1), z3_m1.unsqueeze(0))
        cos_sim_m1 = torch.cat([cos_sim_m1, z1_m1_z3_m1_cos, z2_m1_z3_m1_cos], 1)
    elif num_sent == 4:
        z1_m1_z3_m1_cos = cls.sim(z1_m1.unsqueeze(1), z3_m1.unsqueeze(0))
        cos_sim_m1 = torch.cat([cos_sim_m1, z1_m1_z3_m1_cos], 1)

    loss_fct = nn.CrossEntropyLoss()
    labels_m1 = torch.arange(cos_sim_m1.size(0)).long().to(input_ids.device)
    loss1 = loss_fct(cos_sim_m1, labels_m1)
    
    # 检查并处理 NaN 和 Inf
    if torch.isnan(loss1) or torch.isinf(loss1):
        loss1 = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

    # 计算第二个MASK位置的InfoNCE损失 (L2)
    if cls.model_args.dot_sim:
        cos_sim_m2 = torch.mm(torch.sigmoid(z1_m2), torch.sigmoid(z2_m2.permute(1, 0)))
    else:
        cos_sim_m2 = cls.sim(z1_m2.unsqueeze(1), z2_m2.unsqueeze(0))

    if cls.model_args.norm_instead_temp:
        cos_sim_m2 *= cls.sim.temp
        cmin, cmax = cos_sim_m2.min(), cos_sim_m2.max()
        # 添加数值稳定性保护：防止除以零
        eps = 1e-8
        denominator = cmax - cmin
        denominator = torch.clamp(denominator, min=eps)
        cos_sim_m2 = (cos_sim_m2 - cmin) / denominator / cls.sim.temp

    if num_sent == 3:
        z1_m2_z3_m2_cos = cls.sim(z1_m2.unsqueeze(1), z3_m2.unsqueeze(0))
        z2_m2_z3_m2_cos = cls.sim(z2_m2.unsqueeze(1), z3_m2.unsqueeze(0))
        cos_sim_m2 = torch.cat([cos_sim_m2, z1_m2_z3_m2_cos, z2_m2_z3_m2_cos], 1)
    elif num_sent == 4:
        z1_m2_z3_m2_cos = cls.sim(z1_m2.unsqueeze(1), z3_m2.unsqueeze(0))
        cos_sim_m2 = torch.cat([cos_sim_m2, z1_m2_z3_m2_cos], 1)

    # Momentum queue negatives with cluster-based soft suppression
    if getattr(cls.model_args, "use_momentum_bank", False) and cls.training and hasattr(cls, "encoder_k"):
        with torch.no_grad():
            momentum_bank_update_key_encoder(cls)
            pooler_output_k = _compute_pooler_output(
                cls=cls,
                encoder=cls.encoder_k,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                batch_size=batch_size,
                num_sent=num_sent,
                noise1=noise1,
                noise2=noise2,
                noise3=noise3,
                noise4=noise4,
                template_length1=template_length1,
                template_length2=template_length2,
                template_length3=template_length3,
                template_length4=template_length4,
            )
            z2_m2_k = F.normalize(pooler_output_k[:, 1, 1, :], dim=-1)
            k_cluster_ids = momentum_bank_assign_cluster_ids(cls, z2_m2_k)
            momentum_bank_dequeue_and_enqueue(cls, z2_m2_k, k_cluster_ids)

        q_norm = F.normalize(z1_m2, dim=-1)
        queue_logits = torch.matmul(q_norm, cls.queue.clone().detach()) / cls.model_args.temp
        queue_logits = momentum_bank_soft_suppress_queue_logits(cls, z1_m2, queue_logits)
        cos_sim_m2 = torch.cat([cos_sim_m2, queue_logits], dim=1)

    labels_m2 = torch.arange(cos_sim_m2.size(0)).long().to(input_ids.device)
    loss2 = loss_fct(cos_sim_m2, labels_m2)
    
    # 检查并处理 NaN 和 Inf
    if torch.isnan(loss2) or torch.isinf(loss2):
        loss2 = torch.tensor(0.0, device=input_ids.device, requires_grad=True)


    loss = loss1 + loss2
    
    # 最终检查：如果总损失仍然是 NaN，设置为 0
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    
    # 使用第二个MASK的相似度矩阵作为logits（保持与评估一致）
    cos_sim = cos_sim_m2

    # Calculate loss for MLM
    # if not cls.model_args.add_pseudo_instances and mlm_outputs is not None and mlm_labels is not None:
    if not return_dict:
        return ((loss,) + (cos_sim,)) if loss is not None else (cos_sim,)
    
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
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
):

    if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
        noise_all, template_length = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template, type='pos-2', device=input_ids.device, evaluation=True)
        # 评估时只使用第二个MASK的噪声（保持与之前行为一致）
        noise = noise_all[:, cls.mask_num - 1, :]

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
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
            
            try:
                assert ss + bs.shape[0] + es.shape[0] == ni.shape[0]
            except:
                print(ss + bs.shape[0] + es.shape[0])
                print(ni.shape[0])
                print(i.tolist())
                print(ni.tolist())
                assert 0

            new_input_ids.append(ni)

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = (input_ids != cls.pad_token_id).long()
        token_type_ids = None

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)

        with torch.no_grad():
            p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
            b = torch.arange(input_ids.shape[0]).to(input_ids.device)
            for i, k in enumerate(cls.dict_mbv):
                if cls.fl_mbv[i]:
                    index = ((input_ids == k) * p).max(-1)[1]
                else:
                    index = ((input_ids == k) * -p).min(-1)[1]
                inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        last_hidden = outputs.last_hidden_state
        pooler_output = last_hidden[input_ids == cls.mask_token_id]

        pooler_output = pooler_output.view(-1, cls.mask_num, pooler_output.shape[-1])
        pooler_output = pooler_output[:, cls.mask_num - 1, :]

        if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
            token_length = attention_mask.sum(-1) - template_length

            if cls.model_args.mask_embedding_sentence_org_mlp and not cls.model_args.mlp_only_train:
                pooler_output, noise = cls.mlp(pooler_output), cls.mlp(noise)

            pooler_output -= noise[token_length]

        if cls.model_args.mask_embedding_sentence_avg:
            pooler_output = pooler_output.view(input_ids.shape[0], -1)
        else:
            pooler_output = pooler_output.view(input_ids.shape[0], -1, pooler_output.shape[-1]).mean(1)
            
    if not cls.model_args.mlp_only_train and not cls.model_args.mask_embedding_sentence_org_mlp:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        self.total_length = 80

        if self.model_args.mask_embedding_sentence_autoprompt:
            # register p_mbv in init, avoid not saving weight
            self.p_mbv = torch.nn.Parameter(torch.zeros(10))
            for param in self.bert.parameters():
                param.requires_grad = False

        cl_init(self, config)

        if getattr(self.model_args, "use_momentum_bank", False):
            momentum_bank_init(self, config, self.bert)
            self._momentum_encoder_q = self.bert

    def run_kmeans(self):
        momentum_bank_run_kmeans(self)

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
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config)
        self.total_length = 80

        cl_init(self, config)

        if getattr(self.model_args, "use_momentum_bank", False):
            momentum_bank_init(self, config, self.roberta)
            self._momentum_encoder_q = self.roberta

    def run_kmeans(self):
        momentum_bank_run_kmeans(self)

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
            return sentemb_forward(self, self.roberta,
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
            return cl_forward(self, self.roberta,
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

