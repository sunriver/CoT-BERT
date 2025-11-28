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

from two_stage_cot_model import BertForTwoStageCoT
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

    # TwoStageCoT specific arguments
    temperature: float = field(
        default=0.05,
        metadata={
            "help": "Temperature parameter for InfoNCE loss (default: 0.05)"
        }
    )
    
    # Template arguments (for compatibility with training configs)
    mask_embedding_sentence: bool = field(
        default=False,
        metadata={"help": "Whether to use template with [MASK] token (not used in evaluation)"}
    )
    stage1_anchor_template: str = field(
        default="The sentence of \"[X]\" means [MASK].",
        metadata={"help": "Anchor template for stage 1 sentence representation"}
    )
    stage1_positive_template: str = field(
        default="The sentence : \"[X]\" means [MASK].",
        metadata={"help": "Positive template for stage 1 sentence representation"}
    )
    stage1_negative_template: str = field(
        default="The sentence of \"[X]\" doesn't mean [MASK].",
        metadata={"help": "Negative template for stage 1 sentence representation"}
    )
    stage2_template: str = field(
        default="so the sentence's meaning of \"[IT_SPECIAL_TOKEN]\" can be summarized as [MASK].",
        metadata={"help": "Stage 2 template for sentence representation"}
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
            "If False, will pad the samples dynamically when batching the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    def __post_init__(self):
        # For evaluation, we don't need data files, so skip validation
        pass


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
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
    # to fix typing error for TrainingArguments Optional[bool] in transformers==4.2.1
    # https://github.com/huggingface/transformers/pull/10672
    ddp_find_unused_parameters: bool = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
            "DistributedDataParallel`."
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
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Check for MPS (Metal Performance Shaders) on macOS first, then CUDA, then CPU
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
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

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
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    # 评估脚本直接使用传入的配置文件，不使用训练配置文件作为默认配置
    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/evaluation_default.yaml'
    if not os.path.exists(config_custom_file):
        raise ValueError(f"评估配置文件不存在: {config_custom_file}")
    
    print(f"使用配置文件: {config_custom_file}")
    args_list = load_configs(default_file='', custom_file=config_custom_file)
    
    # 根据平台自动设置设备相关参数
    platform_type = detect_platform()
    if platform_type == "mac_m4":
        # Mac M4: 使用MPS，不设置no_cuda（让MPS可用）
        # 注意：no_cuda只影响CUDA，不影响MPS，所以不设置它
        pass
    elif platform_type == "linux_cuda":
        # Linux CUDA: 使用CUDA，禁用MPS
        args_list.extend([
            "--no_cuda", "false"
        ])
    else:
        # 其他平台: 使用CPU
        args_list.extend([
            "--no_cuda", "true"
        ])
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)

    # Log on each process the small summary:
    logger.warning(
        (
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {training_args.local_rank != -1}, 16-bits training: {training_args.fp16}"
        )
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info("Evaluation parameters %s", training_args)

    # Set seed before initializing model.
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
        # Check if it's a BERT model by checking the config or model path
        is_bert_model = (
            'bert' in model_args.model_name_or_path.lower() or
            (hasattr(config, 'architectures') and config.architectures and 
             'Bert' in str(config.architectures))
        )
        
        if is_bert_model:
            model = BertForTwoStageCoT.from_pretrained(
                model_args.model_name_or_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:
            raise NotImplementedError("Only BERT models are supported for TwoStageCoT")
    else:
        raise NotImplementedError
    
    # 添加 [IT_SPECIAL_TOKEN] 特殊 token
    if '[IT_SPECIAL_TOKEN]' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['[IT_SPECIAL_TOKEN]']})
        model.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    # Setup model for TwoStageCoT
    model.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer  # 保存 tokenizer 供模型使用
    
    # Set model_args for sentemb_forward
    model.model_args = model_args

    # Determine device
    # 优先使用MPS（Mac M4），然后CUDA，最后CPU
    # 注意：no_cuda只影响CUDA，不影响MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"使用MPS设备: {device}")
    elif torch.cuda.is_available() and not training_args.no_cuda:
        device = torch.device("cuda:0")
        logger.info(f"使用CUDA设备: {device}")
    else:
        device = torch.device("cpu")
        logger.info(f"使用CPU设备: {device}")
    
    model = model.to(device)
    model.eval()

    # Evaluation using SentEval
    if training_args.do_eval:
        logger.info("***** Running TwoStageCoT Evaluation with SentEval *****")
        
        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            """
            SentEval batcher函数
            batch: 单个任务的一批句子，每个句子是token列表
            对于STS任务，SentEval会分别传两批句子（batch1和batch2）
            每批都单独处理，返回各自的编码
            """
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            # batch是一个句子列表，每个句子是token列表
            sentences = [' '.join(s) for s in batch]

            use_template = (
                model_args
                and hasattr(model_args, 'mask_embedding_sentence')
                and model_args.mask_embedding_sentence
            )

            if use_template:
                templates = [
                    getattr(model_args, 'stage1_negative_template', "The sentence of \"[X]\" doesn't mean [MASK]."),
                    getattr(model_args, 'stage1_anchor_template', "The sentence of \"[X]\" means [MASK]."),
                    getattr(model_args, 'stage1_positive_template', "The sentence : \"[X]\" means [MASK]."),
                ]

                templated_sentences = []
                for sent in sentences:
                    for template in templates:
                        parts = template.split('[X]')
                        prefix = parts[0]
                        suffix = parts[1] if len(parts) > 1 else ""
                        templated_sentences.append(prefix + sent + suffix)

                encoded = tokenizer.batch_encode_plus(
                    templated_sentences,
                    return_tensors='pt',
                    padding=True,
                )

                batch_size = len(sentences)
                num_views = len(templates)
                seq_len = encoded['input_ids'].size(1)

                batch_input = {}
                for k, tensor in encoded.items():
                    tensor = tensor.to(device)
                    batch_input[k] = tensor.view(batch_size, num_views, seq_len)
            else:
                encoded = tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                )
                batch_input = {}
                for k, tensor in encoded.items():
                    tensor = tensor.to(device)
                    batch_input[k] = tensor.unsqueeze(1)

            # Get sentence embeddings using sentemb_forward
            # sentemb_forward会提取mask位置，进行两阶段处理等流程
            with torch.no_grad():
                stage1_templates = {
                    "negative": getattr(model_args, 'stage1_negative_template', "The sentence of \"[X]\" doesn't mean [MASK]."),
                    "anchor": getattr(model_args, 'stage1_anchor_template', "The sentence of \"[X]\" means [MASK]."),
                    "positive": getattr(model_args, 'stage1_positive_template', "The sentence : \"[X]\" means [MASK]."),
                }
                outputs = model(
                    input_ids=batch_input['input_ids'],
                    attention_mask=batch_input['attention_mask'],
                    token_type_ids=batch_input.get('token_type_ids', None),
                    sent_emb=True,  # 使用sentemb_forward进行mask提取、两阶段处理
                    return_dict=True,
                    stage1_templates=stage1_templates,
                    stage2_template=model_args.stage2_template,
                    tokenizer=tokenizer,
                )
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # Set params for SentEval based on mode
        if training_args.mode == 'dev' or training_args.mode == 'fasttest':
            # Fast mode
            params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
            params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
        elif training_args.mode == 'test':
            # Full mode
            params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
            params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
        else:
            raise ValueError(f"Unknown mode: {training_args.mode}. Choose from: dev, test, fasttest")

        se = senteval.engine.SE(params, batcher, prepare)
        
        # Set tasks based on task_set
        if training_args.task_set == 'sts':
            tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        elif training_args.task_set == 'transfer':
            tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
        elif training_args.task_set == 'full':
            tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
            tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
        else:  # 'na' or default
            # Keep backward compatibility with eval_transfer
            tasks = ['STSBenchmark', 'SICKRelatedness']
            if training_args.eval_transfer:
                tasks += ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

        # Run evaluation
        results = se.eval(tasks)
        
        # Determine result key based on mode
        if training_args.mode == 'dev':
            result_key = 'dev'
            spearman_key = 'spearman'
        elif training_args.mode == 'fasttest' or training_args.mode == 'test':
            result_key = 'test'
            spearman_key = 'spearman'
        else:
            result_key = 'test'
            spearman_key = 'spearman'
        
        # Print results based on task_set
        if training_args.task_set == 'sts' or training_args.task_set == 'full':
            print(f"------ {training_args.mode.capitalize()} Results (STS Tasks) ------")
            scores = []
            task_names = []
            
            # For dev mode, only evaluate STSBenchmark and SICKRelatedness (STS12-16 don't have dev data)
            if result_key == 'dev':
                sts_tasks = ['STSBenchmark', 'SICKRelatedness']
            else:
                sts_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
            
            for task in sts_tasks:
                task_names.append(task)
                if task in results:
                    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                        # STS12-16 only have test data, use 'all' key
                        scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                    else:
                        # STSBenchmark and SICKRelatedness
                        if result_key == 'dev':
                            scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
                        else:
                            scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
                else:
                    scores.append("0.00")
            
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)
        
        # Print transfer task results if evaluated
        if training_args.task_set == 'transfer' or training_args.task_set == 'full':
            print(f"------ {training_args.mode.capitalize()} Results (Transfer Tasks) ------")
            scores = []
            task_names = []
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                task_names.append(task)
                if task in results:
                    if result_key == 'dev':
                        scores.append("%.2f" % (results[task]['devacc']))
                    else:
                        scores.append("%.2f" % (results[task]['acc']))
                else:
                    scores.append("0.00")
            
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)
        
        # Also print if eval_transfer is True (backward compatibility)
        if training_args.eval_transfer and training_args.task_set not in ['transfer', 'full']:
            print(f"------ {training_args.mode.capitalize()} Results (Transfer Tasks) ------")
            scores = []
            task_names = []
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                task_names.append(task)
                if task in results:
                    if result_key == 'dev':
                        scores.append("%.2f" % (results[task]['devacc']))
                    else:
                        scores.append("%.2f" % (results[task]['acc']))
                else:
                    scores.append("0.00")
            
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)
        
        # Log results
        logger.info("***** TwoStageCoT Evaluation Results *****")
        eval_results = {}
        
        # Add STS results if task_set includes STS
        if training_args.task_set in ['sts', 'full']:
            # Extract STSBenchmark and SICKRelatedness results
            stsb_spearman = None
            sickr_spearman = None
            if 'STSBenchmark' in results:
                if result_key == 'dev':
                    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
                else:
                    stsb_spearman = results['STSBenchmark']['test']['spearman'].correlation
            if 'SICKRelatedness' in results:
                if result_key == 'dev':
                    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
                else:
                    sickr_spearman = results['SICKRelatedness']['test']['spearman'].correlation
            
            # Add main STS task results
            if stsb_spearman is not None:
                eval_results["eval_stsb_spearman"] = stsb_spearman
            if sickr_spearman is not None:
                eval_results["eval_sickr_spearman"] = sickr_spearman
            if stsb_spearman is not None and sickr_spearman is not None:
                eval_results["eval_avg_sts"] = (stsb_spearman + sickr_spearman) / 2
            
            # Add all STS task results (STS12-16)
            # Note: STS12-16 only have test data, always use 'all' key
            for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                if task in results:
                    eval_results[f'eval_{task.lower()}_spearman'] = results[task]['all']['spearman']['all']
        
        # Add transfer task results
        if training_args.task_set in ['transfer', 'full'] or training_args.eval_transfer:
            avg_transfer = 0
            transfer_count = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                if task in results:
                    if result_key == 'dev':
                        acc = results[task]['devacc']
                    else:
                        acc = results[task]['acc']
                    avg_transfer += acc
                    transfer_count += 1
                    eval_results[f'eval_{task.lower()}'] = acc
            if transfer_count > 0:
                avg_transfer /= transfer_count
                eval_results['eval_avg_transfer'] = avg_transfer
        
        for key, value in sorted(eval_results.items()):
            logger.info(f"  {key} = {value}")
        
        # Save results
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(output_eval_file, "w") as writer:
            for key, value in sorted(eval_results.items()):
                writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()

