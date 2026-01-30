"""
TexLeJEPA 训练入口：解析 YAML、加载数据、构建模型与 TexLeJEPATrainer，支持跨平台。
"""
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union

import torch
from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import is_main_process
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from datasets import load_dataset

from utils.platform_utils import (
    detect_platform,
    get_platform_config_file,
    setup_cuda_environment,
    print_platform_info,
)
from utils.parse_args_util import load_configs
from utils.lmf_log_util import getMyLogger

from tex_lejepa_model import BertForTexLeJEPA
from tex_lejepa_trainer import TexLeJEPATrainer

print_platform_info()
setup_cuda_environment()

logger = getMyLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model name or path."})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config path."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache dir."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer."})
    model_revision: str = field(default="main", metadata={"help": "Model revision."})
    use_auth_token: bool = field(default=False, metadata={"help": "Use auth token."})
    # TexLeJEPA 超参数（从 yaml 配置文件中读取）
    texlejepa_num_slices: int = field(default=32, metadata={"help": "Number of slices for SlicingUnivariateTest."})
    texlejepa_epps_t_max: float = field(default=3.0, metadata={"help": "EppsPulley t_max."})
    texlejepa_epps_n_points: int = field(default=17, metadata={"help": "EppsPulley n_points (odd)."})
    texlejepa_sig_clip_value: Optional[float] = field(default=0.01, metadata={"help": "Clip value for SIGReg statistic."})
    texlejepa_lamb: float = field(default=0.5, metadata={"help": "Weight for SIGReg: Loss = (1-lamb)*L_inv + lamb*L_SIGReg."})


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "Training data file (.txt)."})
    max_seq_length: Optional[int] = field(default=32, metadata={"help": "Max sequence length."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Pad to max length."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Preprocessing workers."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cache."})

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError("train_file is required.")
        ext = self.train_file.split(".")[-1]
        assert ext in ["csv", "json", "txt"], "train_file should be csv, json or txt."


def main():
    config_file = get_platform_config_file()
    print(f"使用配置文件: {config_file}")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else ""
    args_list = load_configs(default_file=config_file, custom_file=config_custom_file)

    platform_type = detect_platform()
    if platform_type == "mac_m4":
        args_list.extend(["--no_cuda", "true"])
    elif platform_type == "linux_cuda":
        args_list.extend(["--no_cuda", "false"])
    else:
        args_list.extend(["--no_cuda", "true"])

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists."
            " Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    logger.info("Training arguments: %s", training_args)

    set_seed(training_args.seed)

    data_files = {"train": data_args.train_file}
    ext = data_args.train_file.split(".")[-1]
    if ext == "txt":
        ext = "text"
    if ext == "csv":
        datasets = load_dataset(ext, data_files=data_files, cache_dir="../data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(ext, data_files=data_files, cache_dir="../data/")

    column_names = datasets["train"].column_names
    sent_cname = column_names[0]

    config_kw = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kw)

    # 将 TexLeJEPA 超参数写入 config，确保模型 __init__ 中可以直接访问
    # 注意：这些字段是自定义扩展字段，只能在实例化后手动挂到 config 上
    config.texlejepa_num_slices = getattr(model_args, "texlejepa_num_slices", 32)
    config.texlejepa_epps_t_max = getattr(model_args, "texlejepa_epps_t_max", 3.0)
    config.texlejepa_epps_n_points = getattr(model_args, "texlejepa_epps_n_points", 17)
    config.texlejepa_sig_clip_value = getattr(model_args, "texlejepa_sig_clip_value", 0.01)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )

    def prepare_features(examples):
        total = len(examples[sent_cname])
        for i in range(total):
            if examples[sent_cname][i] is None:
                examples[sent_cname][i] = " "
        sentences = examples[sent_cname]
        return tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

    # 直接通过 from_pretrained 传入 lamb，其余 TexLeJEPA 配置从 config 读取
    model = BertForTexLeJEPA.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in (model_args.model_name_or_path or ""),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        lamb=model_args.texlejepa_lamb,
    )

    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:

        @dataclass
        class OurDataCollatorWithPadding:
            tokenizer: PreTrainedTokenizerBase
            padding: Union[bool, str, PaddingStrategy] = True
            max_length: Optional[int] = None
            pad_to_multiple_of: Optional[int] = None

            def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
                return self.tokenizer.pad(
                    features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )
        data_collator = OurDataCollatorWithPadding(tokenizer, max_length=data_args.max_seq_length)

    trainer = TexLeJEPATrainer(
        model_args=model_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero():
            out_file = os.path.join(training_args.output_dir, "train_results.txt")
            with open(out_file, "w") as f:
                for k, v in sorted(train_result.metrics.items()):
                    logger.info("  %s = %s", k, v)
                    f.write(f"{k} = {v}\n")
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == "__main__":
    main()
