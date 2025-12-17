import sys 
sys.path.append('..') 

import os
import torch
import logging
import transformers
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
from prettytable import PrettyTable

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    MODEL_FOR_MASKED_LM_MAPPING,
)

from process_supervision_cot_model import BertForProcessSupervisionCoT
from transformers.trainer_utils import is_main_process
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.utils import cached_property, is_torch_tpu_available

from lmf_log_util import getMyLogger

# Set path to SentEval
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np

# 跨平台设备配置
from platform_utils import (
    detect_platform, 
    setup_device_config, 
    get_platform_config_file,
    setup_cuda_environment,
    print_platform_info
)

# 打印平台信息
print_platform_info()

# 设置CUDA环境（如果需要）
setup_cuda_environment()

# 获取平台配置
platform_config = setup_device_config()


logger = getMyLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # ProcessSupervisionCoT specific arguments
    temperature: float = field(
        default=0.05,
        metadata={
            "help": "Temperature parameter for InfoNCE loss (default: 0.05)"
        }
    )
    
    # Template arguments (for compatibility with training configs)
    mask_embedding_sentence: bool = field(
        default=True,
        metadata={"help": "Whether to use template with [MASK] token"}
    )
    mask_num: int = field(
        default=2,
        metadata={"help": "Number of mask tokens in the template (default: 2 for mask1 and mask2)"}
    )
    mask_embedding_sentence_template: str = field(
        default='*cls*_The_sentence_:_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Anchor template for sentence representation"}
    )
    mask_embedding_sentence_different_template: str = field(
        default='*cls*_The_sentence_:_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Positive template for sentence representation"}
    )
    mask_embedding_sentence_negative_template: str = field(
        default='*cls*_The_sentence_:_"*sent_0*"_does_not_mean_*mask*_,_so_it_cannot_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Negative template for sentence representation"}
    )
    temp: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter (alias for temperature)"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    
    # Task set selection for SentEval
    task_set: str = field(
        default='sts',
        metadata={
            "help": "What set of tasks to evaluate on. Choices: sts, transfer, full, na"
        }
    )
    
    # Evaluation mode
    mode: str = field(
        default='test',
        metadata={
            "help": "What evaluation mode to use. Choices: dev (fast mode, dev results), test (full mode, test results), fasttest (fast mode, test results)"
        }
    )

    # reset follow flag type Optional[bool] -> bool
    ddp_find_unused_parameters: bool = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
            "`DistributedDataParallel`."
        },
    )
    disable_tqdm: bool = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    remove_unused_columns: bool = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    greater_is_better: bool = field(
        default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                self._n_gpu = 1
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
                self._n_gpu = torch.cuda.device_count()
            else:
                device = torch.device("cpu")
                self._n_gpu = 0
        else:
            if self.deepspeed:
                from transformers.integrations import is_deepspeed_available
                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed
                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


from parse_args_util import load_configs

def print_table(task_names, scores):
    """打印结果表格"""
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    # 评估脚本直接使用传入的配置文件
    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/evaluation_default.yaml'
    if not os.path.exists(config_custom_file):
        raise ValueError(f"评估配置文件不存在: {config_custom_file}")
    
    print(f"使用配置文件: {config_custom_file}")
    args_list = load_configs(default_file='', custom_file=config_custom_file)
    
    # 根据平台自动设置设备相关参数
    platform_type = detect_platform()
    if platform_type == "mac_m4":
        pass
    elif platform_type == "linux_cuda":
        args_list.extend([
            "--no_cuda", "false"
        ])
    else:
        args_list.extend([
            "--no_cuda", "true"
        ])
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)

    logger.warning(
        (
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {training_args.local_rank != -1}, 16-bits training: {training_args.fp16}"
        )
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info("Evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    if model_args.model_name_or_path:
        is_bert_model = (
            'bert' in model_args.model_name_or_path.lower() or
            (hasattr(config, 'architectures') and config.architectures and 
             'Bert' in str(config.architectures))
        )
        
        if is_bert_model:
            model = BertForProcessSupervisionCoT.from_pretrained(
                model_args.model_name_or_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:
            raise NotImplementedError("Only BERT models are supported for ProcessSupervisionCoT")
    else:
        raise NotImplementedError
    
    model.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer
    model.eval()

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]

        use_template = (
            model_args.mask_embedding_sentence
        )

        if use_template:
            template = model_args.mask_embedding_sentence_template
            
            templated_sentences = []
            for sent in sentences:
                templated = template.replace('*sent_0*', sent)
                templated = templated.replace('*cls*', tokenizer.cls_token)
                templated = templated.replace('*sep+*', tokenizer.sep_token)
                templated = templated.replace('*mask*', tokenizer.mask_token)
                templated_sentences.append(templated)

            encoded = tokenizer.batch_encode_plus(
                templated_sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=getattr(model_args, 'max_seq_length', 128),
            )

            batch_input = {}
            for k, v in encoded.items():
                v = v.to(training_args.device)
                batch_input[k] = v.unsqueeze(1)
        else:
            encoded = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=getattr(model_args, 'max_seq_length', 128),
            )
            batch_input = {}
            for k, v in encoded.items():
                v = v.to(training_args.device)
                batch_input[k] = v.unsqueeze(1)
        
        with torch.no_grad():
            outputs = model.sentemb_forward(
                **batch_input,
                tokenizer=tokenizer,
            )
            pooler_output = outputs.pooler_output

        return pooler_output.cpu()

    # Set params for SentEval
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    
    # Select tasks based on task_set
    if training_args.task_set == 'sts':
        tasks = ['STSBenchmark', 'SICKRelatedness']
    elif training_args.task_set == 'transfer':
        tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    elif training_args.task_set == 'full':
        tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    else:
        tasks = ['STSBenchmark', 'SICKRelatedness']

    if training_args.eval_transfer:
        tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

    results = se.eval(tasks)

    # Print results
    task_names = []
    scores = []
    
    if 'STSBenchmark' in results:
        if training_args.mode == 'test':
            stsb_spearman = results['STSBenchmark']['test']['spearman'][0]
            stsb_pearson = results['STSBenchmark']['test']['pearson'][0]
        else:
            stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
            stsb_pearson = results['STSBenchmark']['dev']['pearson'][0]
        task_names.extend(['STS-B Spearman', 'STS-B Pearson'])
        scores.extend([f"{stsb_spearman:.2f}", f"{stsb_pearson:.2f}"])
    
    if 'SICKRelatedness' in results:
        if training_args.mode == 'test':
            sickr_spearman = results['SICKRelatedness']['test']['spearman'][0]
            sickr_pearson = results['SICKRelatedness']['test']['pearson'][0]
        else:
            sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
            sickr_pearson = results['SICKRelatedness']['dev']['pearson'][0]
        task_names.extend(['SICK-R Spearman', 'SICK-R Pearson'])
        scores.extend([f"{sickr_spearman:.2f}", f"{sickr_pearson:.2f}"])
    
    if training_args.eval_transfer or training_args.task_set in ['transfer', 'full']:
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            if task in results:
                if training_args.mode == 'test':
                    acc = results[task]['testacc']
                else:
                    acc = results[task]['devacc']
                task_names.append(task)
                scores.append(f"{acc:.2f}")

    print_table(task_names, scores)


if __name__ == "__main__":
    main()

