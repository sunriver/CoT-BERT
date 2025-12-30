import sys
sys.path.append("..")

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

from cross_template_cot_trainer import CrossTemplateCoTTrainer
from cross_template_cot_model import BertForCrossTemplateCoT
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
    print_platform_info,
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
    模型及模板相关参数
    """

    # Huggingface 原始参数
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
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
            "help": "Will use the token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)."
        },
    )

    # CrossTemplateCoT 特定参数
    temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for InfoNCE loss (default: 0.05)"},
    )
    temp: float = field(
        default=0.05,
        metadata={"help": "Alias of temperature (for compatibility with CoT-BERT configs)"},
    )

    # 模板与去噪相关参数
    mask_embedding_sentence: bool = field(
        default=True,
        metadata={"help": "Whether to use template with [MASK] token"},
    )
    mask_num: int = field(
        default=2,
        metadata={"help": "Number of [MASK] tokens in templates (default: 2)"},
    )
    mask_embedding_sentence_template: str = field(
        default='*cls*_The_sentence_of_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Anchor template"},
    )
    mask_embedding_sentence_different_template: str = field(
        default='*cls*_The_sentence_:_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Positive template"},
    )
    mask_embedding_sentence_negative_template: str = field(
        default='*cls*_The_sentence_:_"*sent_0*"_does_not_mean_*mask*_,_so_it_cannot_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Negative template"},
    )

    # 内部存储解析后的模板部分（由 main 函数填充）
    mask_embedding_sentence_bs: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_es: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_bs2: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_es2: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_bs3: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_es3: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_different_negative_template: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_bs4: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_es4: str = field(default="", metadata={"help": "Internal"})

    # 去噪和MLP设置
    mask_embedding_sentence_delta: bool = field(
        default=True,
        metadata={"help": "Whether to use delta denoising for [MASK] representations"},
    )
    mask_embedding_sentence_delta_freeze: bool = field(
        default=False,
        metadata={"help": "Whether to freeze delta denoising parameters"},
    )
    mask_embedding_sentence_org_mlp: bool = field(
        default=False,
        metadata={"help": "Whether to use original MLP layer before denoising"},
    )

    # 损失权重
    process_supervision_weight_1: float = field(
        default=1.0,
        metadata={"help": "Weight for process supervision InfoNCE loss 1 (first MASK)"},
    )
    process_supervision_weight_2: float = field(
        default=1.0,
        metadata={"help": "Weight for process supervision InfoNCE loss 2 (second MASK)"},
    )
    constraint_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for constraint loss L3"},
    )


@dataclass
class DataTrainingArguments:
    """
    数据相关参数
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file (.txt or .csv)."},
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated."
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
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # 是否在validation阶段评估transfer任务
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."},
    )

    # 修复 transformers==4.2.1 中 ddp_find_unused_parameters 类型问题
    ddp_find_unused_parameters: bool = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` "
            "passed to `DistributedDataParallel`."
        },
    )
    disable_tqdm: bool = field(
        default=None,
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."},
    )
    greater_is_better: bool = field(
        default=True,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."},
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

    @cached_property
    def _setup_devices(self) -> "torch.device":  # type: ignore[name-defined]
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()  # type: ignore[name-defined]
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


def prepare_features(examples, model_args: ModelArguments, data_args: DataTrainingArguments, tokenizer):
    """
    Cross-Template CoT 模板数据准备函数
    使用3个模板（锚句、正样本、负样本）为每个句子生成输入。
    完全复用 CoT-BERT 的模板处理逻辑。
    """
    total = len(examples["text"])

    # 避免 None
    for idx in range(total):
        if examples["text"][idx] is None:
            examples["text"][idx] = " "

    sentences = examples["text"]

    if model_args.mask_embedding_sentence:
        # 预编码模板部分
        # [:-1] 移除 [SEP], [1:] 移除 [CLS]
        bs1 = tokenizer.encode(model_args.mask_embedding_sentence_bs)[:-1]
        es1 = tokenizer.encode(model_args.mask_embedding_sentence_es)[1:]

        if model_args.mask_embedding_sentence_different_template != "":
            bs2 = tokenizer.encode(model_args.mask_embedding_sentence_bs2)[:-1]
            es2 = tokenizer.encode(model_args.mask_embedding_sentence_es2)[1:]
        else:
            bs2, es2 = bs1, es1

        if model_args.mask_embedding_sentence_negative_template != "":
            bs3 = tokenizer.encode(model_args.mask_embedding_sentence_bs3)[:-1]
            es3 = tokenizer.encode(model_args.mask_embedding_sentence_es3)[1:]
        else:
            bs3, es3 = bs1, es1

        sent_features = {"input_ids": [], "attention_mask": []}

        # 为每个样本生成3个视图（anchor, positive, negative）
        # 注意：这里我们为每个样本生成 3 * total 个输入，然后重新组织
        all_input_ids = []
        
        # 处理所有句子
        for sent in sentences:
            # 基础编码（不加特殊 token）
            s_ids = tokenizer.encode(sent, add_special_tokens=False)[: data_args.max_seq_length]
            
            # 1. Anchor 视图
            all_input_ids.append(bs1 + s_ids + es1)
            # 2. Positive 视图
            all_input_ids.append(bs2 + s_ids + es2)
            # 3. Negative 视图
            all_input_ids.append(bs3 + s_ids + es3)

        # 计算最大长度用于填充
        max_len = max(len(ids) for ids in all_input_ids)

        # 填充并构建 attention mask
        padded_input_ids = []
        attention_masks = []
        for ids in all_input_ids:
            padding_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
            attention_masks.append([1] * len(ids) + [0] * padding_len)

        # 重新组织为 [total, 3, seq_len]
        for i in range(total):
            sent_features["input_ids"].append([
                padded_input_ids[i * 3],
                padded_input_ids[i * 3 + 1],
                padded_input_ids[i * 3 + 2]
            ])
            sent_features["attention_mask"].append([
                attention_masks[i * 3],
                attention_masks[i * 3 + 1],
                attention_masks[i * 3 + 2]
            ])
    else:
        # 无模板模式（SimCSE 风格）
        sent_features = {"input_ids": [], "attention_mask": []}
        for sent in sentences:
            s_ids = tokenizer.encode(sent, add_special_tokens=False)[: data_args.max_seq_length]
            seq = [tokenizer.cls_token_id] + s_ids + [tokenizer.sep_token_id]
            sent_features["input_ids"].append([seq, seq, seq])

        max_len = max(len(seq) for sample in sent_features["input_ids"] for seq in sample)
        for i in range(total):
            sample_ids = []
            sample_mask = []
            for seq in sent_features["input_ids"][i]:
                padding_len = max_len - len(seq)
                sample_ids.append(seq + [tokenizer.pad_token_id] * padding_len)
                sample_mask.append([1] * len(seq) + [0] * padding_len)
            sent_features["input_ids"][i] = sample_ids
            sent_features["attention_mask"].append(sample_mask)

    return sent_features


def main():
    # 获取平台特定配置文件
    config_file = get_platform_config_file()
    print(f"使用配置文件: {config_file}")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else ""
    args_list = load_configs(default_file=config_file, custom_file=config_custom_file)

    # 根据平台自动设置设备相关参数
    platform_type = detect_platform()
    if platform_type == "mac_m4":
        # Mac M4: 使用MPS，禁用CUDA
        args_list.extend(["--no_cuda", "true"])
    elif platform_type == "linux_cuda":
        # Linux CUDA: 使用CUDA
        args_list.extend(["--no_cuda", "false"])
    else:
        # 其他平台: 使用CPU
        args_list.extend(["--no_cuda", "true"])

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # 日志
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

    logger.info("Training/evaluation parameters %s", training_args)

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载数据集
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file

    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir="../data/",
            delimiter="\t" if "tsv" in data_args.train_file else ",",
        )
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="../data/")

    # 加载预训练模型与tokenizer
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
        if "bert" in model_args.model_name_or_path.lower():
            model = BertForCrossTemplateCoT.from_pretrained(
                model_args.model_name_or_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:
            raise NotImplementedError("Only BERT models are supported for CrossTemplateCoT")
    else:
        raise NotImplementedError

    # 设置模型属性
    model.resize_token_embeddings(len(tokenizer))

    if model_args.mask_embedding_sentence:
        model.mask_num = model_args.mask_num
        model.pad_token_id = tokenizer.pad_token_id
        model.mask_token_id = tokenizer.mask_token_id

        # 解析并设置锚句模板属性
        if model_args.mask_embedding_sentence_template != '': 
            template = model_args.mask_embedding_sentence_template
            # 自动转换空格为下划线，以对齐 CoT-BERT 解析逻辑
            if ' ' in template:
                template = template.replace(' ', '_')
            
            assert ' ' not in template, f"Template contains spaces: {template}"
            template = template.replace('*mask*', tokenizer.mask_token)\
                               .replace('*sep+*', '').replace('*cls*', '').replace('*sent_0*', ' ')
            template = template.split(' ')
            model_args.mask_embedding_sentence_bs = template[0].replace('_', ' ')
            model_args.mask_embedding_sentence_es = template[1].replace('_', ' ')
            
            model.bs = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
            model.es = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)
            model.mask_embedding_template = tokenizer.encode(model_args.mask_embedding_sentence_bs + model_args.mask_embedding_sentence_es)

        # 解析并设置正样本模板属性
        if model_args.mask_embedding_sentence_different_template != '':
            template = model_args.mask_embedding_sentence_different_template
            if ' ' in template:
                template = template.replace(' ', '_')
                
            assert ' ' not in template, f"Template contains spaces: {template}"
            template = template.replace('*mask*', tokenizer.mask_token)\
                               .replace('*sep+*', '').replace('*cls*', '').replace('*sent_0*', ' ')
            template = template.split(' ')
            model_args.mask_embedding_sentence_bs2 = template[0].replace('_', ' ')
            model_args.mask_embedding_sentence_es2 = template[1].replace('_', ' ')
            
            model.bs2 = tokenizer.encode(model_args.mask_embedding_sentence_bs2, add_special_tokens=False)
            model.es2 = tokenizer.encode(model_args.mask_embedding_sentence_es2, add_special_tokens=False)
            model.mask_embedding_template2 = tokenizer.encode(model_args.mask_embedding_sentence_bs2 + model_args.mask_embedding_sentence_es2)
        
        # 解析并设置负样本模板属性
        if model_args.mask_embedding_sentence_negative_template != '':
            template = model_args.mask_embedding_sentence_negative_template
            if ' ' in template:
                template = template.replace(' ', '_')
                
            assert ' ' not in template, f"Template contains spaces: {template}"
            template = template.replace('*mask*', tokenizer.mask_token)\
                               .replace('*sep+*', '').replace('*cls*', '').replace('*sent_0*', ' ')
            template = template.split(' ')
            model_args.mask_embedding_sentence_bs3 = template[0].replace('_', ' ')
            model_args.mask_embedding_sentence_es3 = template[1].replace('_', ' ')
            
            model.bs3 = tokenizer.encode(model_args.mask_embedding_sentence_bs3, add_special_tokens=False)
            model.es3 = tokenizer.encode(model_args.mask_embedding_sentence_es3, add_special_tokens=False)
            model.mask_embedding_template3 = tokenizer.encode(model_args.mask_embedding_sentence_bs3 + model_args.mask_embedding_sentence_es3)

    # 准备特征
    column_names = datasets["train"].column_names

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            lambda examples: prepare_features(examples, model_args, data_args, tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    @dataclass
    class OurDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ["input_ids", "attention_mask", "token_type_ids"]
            bs = len(features)

            if bs > 0:
                num_sent = len(features[0]["input_ids"])
            else:
                return {}

            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch = {
                k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
                for k in batch
            }

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    # 设置模型属性
    model.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer

    trainer = CrossTemplateCoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.model_args = model_args

    # 训练
    if training_args.do_train:
        model_path = None
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** CrossTemplateCoT Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == "__main__":
    main()


