import os
import sys
import time
import math
import torch
import warnings
import collections
from packaging import version
from typing import Any, Dict, List, Optional, Union

from transformers import Trainer
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer_utils import (
    set_seed,
    TrainOutput,
    speed_metrics,
    HPSearchBackend,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

from transformers.modeling_utils import unwrap_model

# Set path to SentEval
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np

logger = logging.get_logger(__name__)

class ProcessSupervisionCoTTrainer(Trainer):
    """
    过程监督 CoT 训练器：继承Trainer，实现SentEval评估
    支持三模板 InfoNCE + 跨模板对比学习 + 过程监督的训练和评估
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sharded_dpp = False
        self.use_amp = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算过程监督 CoT 损失
        """
        # 获取 tokenizer
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        # 获取模型参数
        cross_template_weight = getattr(self.model_args, 'cross_template_weight', 0.5)
        process_supervision_weight = getattr(self.model_args, 'process_supervision_weight', 0.1)
        enable_process_supervision = getattr(self.model_args, 'enable_process_supervision', False)
        
        # 过程监督评估器（如果需要）
        process_supervisor = None
        if enable_process_supervision:
            # 延迟初始化过程监督评估器
            if not hasattr(self, '_process_supervisor'):
                from process_supervision_cot_supervisor import ProcessSupervisor
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._process_supervisor = ProcessSupervisor(api_key=api_key)
                else:
                    logger.warning("OPENAI_API_KEY not set, process supervision disabled")
                    self._process_supervisor = None
            process_supervisor = self._process_supervisor
        
        # 调用模型前向传播
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids', None),
            cross_template_weight=cross_template_weight,
            process_supervision_weight=process_supervision_weight,
            enable_process_supervision=enable_process_supervision,
            process_supervisor=process_supervisor,
            tokenizer=tokenizer,
        )
        
        loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        """
        评估函数：使用SentEval进行STS任务评估
        支持三模板的性能评估
        """

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            """
            SentEval batcher函数
            batch: 单个任务的一批句子
            对于STS任务，SentEval会分别传两批句子（batch1和batch2）
            每批都单独处理，返回各自的编码
            """
            # batch是一个句子列表，每个句子是token列表
            sentences = [' '.join(s) for s in batch]

            use_template = (
                self.model_args
                and hasattr(self.model_args, 'mask_embedding_sentence')
                and self.model_args.mask_embedding_sentence
            )

            if use_template:
                templates = [
                    getattr(self.model_args, 'mask_embedding_sentence_template', 
                            '*cls*_The_sentence_:_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*'),
                    getattr(self.model_args, 'mask_embedding_sentence_different_template', 
                            '*cls*_The_sentence_:_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*'),
                    getattr(self.model_args, 'mask_embedding_sentence_negative_template', 
                            '*cls*_The_sentence_:_"*sent_0*"_does_not_mean_*mask*_,_so_it_cannot_be_summarized_as_*mask*_._*sep+*'),
                ]

                templated_sentences = []
                for sent in sentences:
                    for template in templates:
                        # 处理模板中的占位符
                        templated = template.replace('*sent_0*', sent)
                        templated = templated.replace('*cls*', self.tokenizer.cls_token)
                        templated = templated.replace('*sep+*', self.tokenizer.sep_token)
                        templated = templated.replace('*mask*', self.tokenizer.mask_token)
                        templated_sentences.append(templated)

                encoded = self.tokenizer.batch_encode_plus(
                    templated_sentences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.model_args, 'max_seq_length', 128),
                )

                batch_size = len(sentences)
                num_views = len(templates)
                seq_len = encoded['input_ids'].size(1)

                batch_input = {}
                for k, v in encoded.items():
                    v = v.to(self.args.device)
                    batch_input[k] = v.view(batch_size, num_views, seq_len)
            else:
                encoded = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.model_args, 'max_seq_length', 128),
                )
                batch_input = {}
                for k, v in encoded.items():
                    v = v.to(self.args.device)
                    batch_input[k] = v.unsqueeze(1)
            
            with torch.no_grad():
                outputs = self.model.sentemb_forward(
                    **batch_input,
                    tokenizer=self.tokenizer,
                )
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']

        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

        self.model.eval()
        results = se.eval(tasks)

        metrics = {}
        if 'STSBenchmark' in results:
            metrics[f'{metric_key_prefix}_stsb_spearman'] = results['STSBenchmark']['dev']['spearman'][0]
            metrics[f'{metric_key_prefix}_stsb_pearson'] = results['STSBenchmark']['dev']['pearson'][0]
        if 'SICKRelatedness' in results:
            metrics[f'{metric_key_prefix}_sickr_spearman'] = results['SICKRelatedness']['dev']['spearman'][0]
            metrics[f'{metric_key_prefix}_sickr_pearson'] = results['SICKRelatedness']['dev']['pearson'][0]
        
        if eval_senteval_transfer or self.args.eval_transfer:
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                if task in results:
                    metrics[f'{metric_key_prefix}_{task.lower()}_acc'] = results[task]['devacc']

        return metrics

