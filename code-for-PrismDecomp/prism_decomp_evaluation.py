import sys 
sys.path.append('..') 

import os
import torch
import logging
import transformers
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    MODEL_FOR_MASKED_LM_MAPPING,
)

from prism_decomp_trainer import PrismDecompTrainer
from prism_decomp_model import BertForPrismDecomp
from transformers.trainer_utils import is_main_process
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.utils import cached_property, is_torch_tpu_available

from lmf_log_util import getMyLogger

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

    # PrismDecomp specific arguments
    num_semantics: int = field(
        default=7,
        metadata={
            "help": "Number of semantic dimensions for decomposition"
        }
    )
    orthogonal_constraint: bool = field(
        default=True,
        metadata={
            "help": "Whether to use orthogonal constraint for semantic independence"
        }
    )
    semantic_weights_learnable: bool = field(
        default=True,
        metadata={
            "help": "Whether semantic weights are learnable parameters"
        }
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
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    # reset follow flag type Optional[bool] -> bool
    # to fix typing error for TrainingArguments Optional[bool] in transformers==4.2.1
    # https://github.com/huggingface/transformers/pull/10672
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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    # 获取平台特定的配置文件
    config_file = get_platform_config_file()
    print(f"使用配置文件: {config_file}")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else ''
    args_list = load_configs(default_file=config_file, custom_file=config_custom_file)
    
    # 根据平台自动设置设备相关参数
    platform_type = detect_platform()
    if platform_type == "mac_m4":
        # Mac M4: 使用MPS，禁用CUDA
        args_list.extend([
            "--no_cuda", "true"
        ])
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
        if 'bert' in model_args.model_name_or_path:
            model = BertForPrismDecomp.from_pretrained(
                model_args.model_name_or_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:
            raise NotImplementedError("Only BERT models are supported for PrismDecomp")
    else:
        raise NotImplementedError
    
    model.resize_token_embeddings(len(tokenizer))

    # Setup model for PrismDecomp
    model.pad_token_id = tokenizer.pad_token_id

    trainer = PrismDecompTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    trainer.model_args = model_args

    # Evaluation
    if training_args.do_eval:
        logger.info("***** Running PrismDecomp Evaluation *****")
        
        # Evaluate on STS tasks
        eval_results = trainer.evaluate(eval_senteval_transfer=True)
        
        # Log results
        logger.info("***** PrismDecomp Evaluation Results *****")
        for key, value in sorted(eval_results.items()):
            logger.info(f"  {key} = {value}")
        
        # Save results
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in sorted(eval_results.items()):
                    writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
