import sys
sys.path.append("..")

import os
import torch
import logging
import transformers
from dataclasses import dataclass, field
from typing import Optional
from prettytable import PrettyTable

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    MODEL_FOR_MASKED_LM_MAPPING,
)

from cross_template_cot_model import BertForCrossTemplateCoT
from transformers.trainer_utils import is_main_process
from transformers.utils import cached_property, is_torch_tpu_available

from lmf_log_util import getMyLogger

# SentEval 路径
PATH_TO_SENTEVAL = "../SentEval"
PATH_TO_DATA = "../SentEval/data"

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval  # noqa: E402
import numpy as np  # noqa: E402

# 跨平台设备配置
from platform_utils import (
    detect_platform,
    setup_device_config,
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
    模型与模板相关参数（评估阶段）
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Typically a fine-tuned CrossTemplateCoT checkpoint directory."
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

    # 模板参数（与训练保持一致）
    mask_embedding_sentence: bool = field(
        default=True,
        metadata={"help": "Whether to use template with [MASK] token"},
    )
    mask_embedding_sentence_template: str = field(
        default='*cls*_The_sentence_of_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*',
        metadata={"help": "Anchor template"},
    )

    # 内部存储解析后的模板部分（由 main 函数填充）
    mask_embedding_sentence_bs: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_es: str = field(default="", metadata={"help": "Internal"})
    mask_embedding_sentence_different_template: str = field(
        default='The sentence ："[X]" means [MASK], so it can be summarized as [MASK].',
        metadata={"help": "Positive template"},
    )
    mask_embedding_sentence_negative_template: str = field(
        default='The sentence ："[X]" does not mean [MASK], so it cannot be summarized as [MASK]',
        metadata={"help": "Negative template"},
    )


@dataclass
class DataTrainingArguments:
    """
    评估阶段数据参数（SentEval不直接依赖这些文件路径）
    """

    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=32)
    pad_to_max_length: bool = field(default=False)
    mlm_probability: float = field(default=0.15)

    def __post_init__(self):
        # 评估脚本不强制检查数据文件
        pass


@dataclass
class OurTrainingArguments(TrainingArguments):
    # 评估任务集合
    task_set: str = field(
        default="sts",
        metadata={"help": "What set of tasks to evaluate on. Choices: sts, transfer, full, na"},
    )

    # 评估模式
    mode: str = field(
        default="test",
        metadata={
            "help": "What evaluation mode to use. Choices: dev (fast), test (full), fasttest (fast test results)"
        },
    )

    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev/test sets."},
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


def print_table(task_names, scores):
    """打印结果表格"""
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    # 评估脚本直接使用传入的配置文件
    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation_default.yaml"
    if not os.path.exists(config_custom_file):
        raise ValueError(f"评估配置文件不存在: {config_custom_file}")

    print(f"使用配置文件: {config_custom_file}")
    args_list = load_configs(default_file="", custom_file=config_custom_file)

    # 根据平台自动设置设备相关参数
    platform_type = detect_platform()
    if platform_type == "mac_m4":
        # Mac M4: 使用MPS，不设置no_cuda（让MPS可用）
        pass
    elif platform_type == "linux_cuda":
        args_list.extend(["--no_cuda", "false"])
    else:
        args_list.extend(["--no_cuda", "true"])

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

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载模型与tokenizer
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
            "bert" in model_args.model_name_or_path.lower()
            or (hasattr(config, "architectures") and config.architectures and "Bert" in str(config.architectures))
        )
        if is_bert_model:
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

    # Setup model
    model.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer
    model.model_args = model_args

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

    # 设备选择：优先MPS，其次CUDA，最后CPU
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

    # 使用 SentEval 进行评估
    if training_args.do_eval:
        logger.info("***** Running CrossTemplateCoT Evaluation with SentEval *****")

        def prepare(params, samples):
            return

        def batcher(params, batch):
            """
            SentEval batcher函数
            batch: 单个任务的一批句子，每个句子是token列表
            STS任务会分别传两批句子（batch1和batch2）
            """
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode("utf-8") for word in s] for s in batch]

            sentences = [" ".join(s) for s in batch]

            use_template = (
                model_args
                and hasattr(model_args, "mask_embedding_sentence")
                and model_args.mask_embedding_sentence
            )

            if use_template:
                # 获取解析后的模板部分
                bs1 = tokenizer.encode(model_args.mask_embedding_sentence_bs)[:-1]
                es1 = tokenizer.encode(model_args.mask_embedding_sentence_es)[1:]

                all_input_ids = []
                for sent in sentences:
                    # 基础编码（不加特殊 token）
                    s_ids = tokenizer.encode(sent, add_special_tokens=False)[: training_args.max_seq_length]
                    all_input_ids.append(bs1 + s_ids + es1)

                # 计算最大长度用于填充
                max_len = max(len(ids) for ids in all_input_ids)

                # 填充并构建 attention mask
                padded_input_ids = []
                attention_masks = []
                for ids in all_input_ids:
                    padding_len = max_len - len(ids)
                    padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
                    attention_masks.append([1] * len(ids) + [0] * padding_len)

                batch_input = {
                    "input_ids": torch.tensor(padded_input_ids).to(device).unsqueeze(1),
                    "attention_mask": torch.tensor(attention_masks).to(device).unsqueeze(1),
                }
            else:
                encoded = tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors="pt",
                    padding=True,
                )
                batch_input = {}
                for k, tensor in encoded.items():
                    tensor = tensor.to(device)
                    batch_input[k] = tensor.unsqueeze(1)

            with torch.no_grad():
                anchor_template = getattr(
                    model_args,
                    "mask_embedding_sentence_template",
                    'The sentence of "[X]" means [MASK], so it can be summarized as [MASK].',
                )
                outputs = model(
                    input_ids=batch_input["input_ids"],
                    attention_mask=batch_input["attention_mask"],
                    token_type_ids=batch_input.get("token_type_ids", None),
                    sent_emb=True,
                    return_dict=True,
                    anchor_template=anchor_template,
                    tokenizer=tokenizer,
                )
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # SentEval 参数
        if training_args.mode in ["dev", "fasttest"]:
            params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
            params["classifier"] = {
                "nhid": 0,
                "optim": "rmsprop",
                "batch_size": 128,
                "tenacity": 3,
                "epoch_size": 2,
            }
        elif training_args.mode == "test":
            params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 10}
            params["classifier"] = {
                "nhid": 0,
                "optim": "adam",
                "batch_size": 64,
                "tenacity": 5,
                "epoch_size": 4,
            }
        else:
            raise ValueError(f"Unknown mode: {training_args.mode}. Choose from: dev, test, fasttest")

        se = senteval.engine.SE(params, batcher, prepare)

        # 根据task_set选择任务
        if training_args.task_set == "sts":
            tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
        elif training_args.task_set == "transfer":
            tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
        elif training_args.task_set == "full":
            tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
            tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
        else:  # 'na' or default
            tasks = ["STSBenchmark", "SICKRelatedness"]
            if training_args.eval_transfer:
                tasks += ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]

        results = se.eval(tasks)

        # mode -> result_key
        if training_args.mode == "dev":
            result_key = "dev"
        elif training_args.mode in ["fasttest", "test"]:
            result_key = "test"
        else:
            result_key = "test"

        # 打印STS结果
        if training_args.task_set in ["sts", "full"]:
            print(f"------ {training_args.mode.capitalize()} Results (STS Tasks) ------")
            scores = []
            task_names = []

            if result_key == "dev":
                sts_tasks = ["STSBenchmark", "SICKRelatedness"]
            else:
                sts_tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]

            for task in sts_tasks:
                task_names.append(task)
                if task in results:
                    if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                        scores.append("%.2f" % (results[task]["all"]["spearman"]["all"] * 100))
                    else:
                        if result_key == "dev":
                            scores.append("%.2f" % (results[task]["dev"]["spearman"][0] * 100))
                        else:
                            scores.append("%.2f" % (results[task]["test"]["spearman"].correlation * 100))
                else:
                    scores.append("0.00")

            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

        # 打印transfer任务结果
        if training_args.task_set in ["transfer", "full"]:
            print(f"------ {training_args.mode.Capitalize()} Results (Transfer Tasks) ------")
            scores = []
            task_names = []
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                task_names.append(task)
                if task in results:
                    if result_key == "dev":
                        scores.append("%.2f" % (results[task]["devacc"]))
                    else:
                        scores.append("%.2f" % (results[task]["acc"]))
                else:
                    scores.append("0.00")

            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

        # 记录到日志并保存
        logger.info("***** CrossTemplateCoT Evaluation Results *****")
        eval_results = {}

        if training_args.task_set in ["sts", "full"]:
            stsb_spearman = None
            sickr_spearman = None
            if "STSBenchmark" in results:
                if result_key == "dev":
                    stsb_spearman = results["STSBenchmark"]["dev"]["spearman"][0]
                else:
                    stsb_spearman = results["STSBenchmark"]["test"]["spearman"].correlation
            if "SICKRelatedness" in results:
                if result_key == "dev":
                    sickr_spearman = results["SICKRelatedness"]["dev"]["spearman"][0]
                else:
                    sickr_spearman = results["SICKRelatedness"]["test"]["spearman"].correlation

            if stsb_spearman is not None:
                eval_results["eval_stsb_spearman"] = stsb_spearman
            if sickr_spearman is not None:
                eval_results["eval_sickr_spearman"] = sickr_spearman
            if stsb_spearman is not None and sickr_spearman is not None:
                eval_results["eval_avg_sts"] = (stsb_spearman + sickr_spearman) / 2

            for task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                if task in results:
                    eval_results[f"eval_{task.lower()}_spearman"] = results[task]["all"]["spearman"]["all"]

        if training_args.task_set in ["transfer", "full"] or training_args.eval_transfer:
            avg_transfer = 0.0
            transfer_count = 0
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                if task in results:
                    if result_key == "dev":
                        acc = results[task]["devacc"]
                    else:
                        acc = results[task]["acc"]
                    avg_transfer += acc
                    transfer_count += 1
                    eval_results[f"eval_{task.lower()}"] = acc
            if transfer_count > 0:
                avg_transfer /= transfer_count
                eval_results["eval_avg_transfer"] = avg_transfer

        for key, value in sorted(eval_results.items()):
            logger.info(f"  {key} = {value}")

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(output_eval_file, "w") as writer:
            for key, value in sorted(eval_results.items()):
                writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()


