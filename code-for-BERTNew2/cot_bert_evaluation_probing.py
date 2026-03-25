import sys
import os
import json
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_SCRIPT_DIR, ".."))

import torch
import argparse
from argparse import Namespace
from transformers import AutoConfig, AutoTokenizer

from cot_bert_model import BertForCL, RobertaForCL
from parse_args_util import load_configs
from platform_utils import (
    detect_platform,
    setup_device_config,
    setup_cuda_environment,
    print_platform_info,
)


# SentEval paths (relative to this script file)
PATH_TO_SENTEVAL = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "SentEval"))
PATH_TO_DATA = os.path.normpath(os.path.join(PATH_TO_SENTEVAL, "data"))

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


PROBING_TASKS = [
    "Length",
    "WordContent",
    "Depth",
    "TopConstituents",
    "BigramShift",
    "Tense",
    "SubjNumber",
    "ObjNumber",
    "OddManOut",
    "CoordinationInversion",
]


def _inject_template(input_ids, pad_token_id, bs_ids, es_ids):
    """
    复制 cot_bert_model.sentemb_forward 中的 template 注入逻辑：
    [CLS] + bs + sentence_tokens + es + [SEP] (+ padding)
    """
    new_input_ids = []
    bs = torch.LongTensor(bs_ids).to(input_ids.device)
    es = torch.LongTensor(es_ids).to(input_ids.device)

    for row in input_ids:
        ss = row.shape[0]
        ii = row[row != pad_token_id]

        parts = [ii[:1], bs]
        if ii.shape[0] > 2:
            parts += [ii[1:-1]]
        parts += [es, ii[-1:]]

        if ii.shape[0] < row.shape[0]:
            parts += [row[row == pad_token_id]]

        ni = torch.cat(parts)

        # 与原逻辑一致的形状检查
        if ss + bs.shape[0] + es.shape[0] != ni.shape[0]:
            raise RuntimeError("Template injection length mismatch.")

        new_input_ids.append(ni)

    injected = torch.stack(new_input_ids, dim=0)
    attn_mask = (injected != pad_token_id).long()
    return injected, attn_mask


def main():
    print_platform_info()
    setup_cuda_environment()
    setup_device_config()

    # 选择平台默认 probing 配置文件（优先 probing 专用；不存在则回退通用 eval）
    platform_type = detect_platform()
    if platform_type == "mac_m4":
        default_cfg = "configs/evaluation_probing_mac_m4.yaml"
    elif platform_type == "linux_cuda":
        default_cfg = "configs/evaluation_probing_linux.yaml"
    else:
        default_cfg = "configs/evaluation_probing_default.yaml"

    default_cfg_abs = os.path.join(_SCRIPT_DIR, default_cfg)
    if not os.path.isfile(default_cfg_abs):
        # 回退到通用 eval 配置
        if platform_type == "mac_m4":
            default_cfg = "configs/evaluation_mac_m4.yaml"
        elif platform_type == "linux_cuda":
            default_cfg = "configs/evaluation_linux_cuda.yaml"
        else:
            default_cfg = "configs/evaluation_default.yaml"

    # 兼容两种调用：
    # 1) python xxx.py configs/evaluation_probing_linux.yaml --repr_type mask1
    # 2) python xxx.py --repr_type mask1
    custom_cfg = ""
    cli_extra_args = []
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg.endswith(".yaml") or first_arg.endswith(".yml"):
            custom_cfg = first_arg
            cli_extra_args = sys.argv[2:]
        else:
            cli_extra_args = sys.argv[1:]
    args_list = load_configs(default_file=default_cfg, custom_file=custom_cfg)
    args_list.extend(cli_extra_args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--repr_type", type=str, choices=["mask1", "mask2", "cls"], default="mask2")
    parser.add_argument("--mode", type=str, choices=["dev", "test", "fasttest"], default="test")

    # CoT 模板相关（与 sentemb 脚本保持一致）
    parser.add_argument("--mask_num", type=int, default=2)
    parser.add_argument("--mask_embedding_sentence", action="store_true")
    parser.add_argument("--mask_embedding_sentence_template", type=str, default=None)
    parser.add_argument("--mask_embedding_sentence_delta", action="store_true")
    parser.add_argument("--mask_embedding_sentence_org_mlp", action="store_true")
    parser.add_argument("--mask_embedding_sentence_autoprompt", action="store_true")

    # eval 配置里可能包含其它参数（如 task_set），这里忽略未知参数
    args, _unknown = parser.parse_known_args(args_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # 构造 model_args（满足 BertForCL 初始化依赖；probing 不用训练相关字段）
    model_args = Namespace(
        model_name_or_path=args.model_name_or_path,
        mask_embedding_sentence=args.mask_embedding_sentence,
        mask_embedding_sentence_template=args.mask_embedding_sentence_template or "",
        mask_num=args.mask_num,
        mask_embedding_sentence_delta=args.mask_embedding_sentence_delta,
        mask_embedding_sentence_delta_no_delta_eval=True,  # probing 默认不做 delta 去噪（更纯粹比较表示本身）
        mask_embedding_sentence_delta_freeze=False,
        mask_embedding_sentence_org_mlp=args.mask_embedding_sentence_org_mlp,
        mask_embedding_sentence_autoprompt=args.mask_embedding_sentence_autoprompt,
        mask_embedding_sentence_avg=False,
        mlp_only_train=False,
        temp=0.05,
        scd_temp=0.05,
        dot_sim=False,
        norm_instead_temp=False,
        only_embedding_training=False,
        mask_embedding_sentence_different_template="",
        mask_embedding_sentence_negative_template="",
        mask_embedding_sentence_different_negative_template="",
        mask_embedding_sentence_autoprompt_continue_training_as_positive=False,
        cache_dir=None,
        model_revision="main",
        use_auth_token=False,
    )

    # 解析模板得到 bs/es（与 sentemb 脚本一致）
    if model_args.mask_embedding_sentence and model_args.mask_embedding_sentence_template:
        template = model_args.mask_embedding_sentence_template
        template = (
            template.replace("*mask*", tokenizer.mask_token)
            .replace("*sep+*", "")
            .replace("*cls*", "")
            .replace("*sent_0*", " ")
        )
        template = template.split(" ")
        model_args.mask_embedding_sentence_bs = template[0].replace("_", " ")
        model_args.mask_embedding_sentence_es = template[1].replace("_", " ")
        if "roberta" in args.model_name_or_path:
            model_args.mask_embedding_sentence_bs = model_args.mask_embedding_sentence_bs.strip()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_columns = 3

    if "roberta" in args.model_name_or_path:
        model = RobertaForCL.from_pretrained(args.model_name_or_path, config=config, model_args=model_args)
    else:
        model = BertForCL.from_pretrained(args.model_name_or_path, config=config, model_args=model_args)

    # 设备
    if platform_type == "mac_m4" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif platform_type == "linux_cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    # mask 表示需要 bs/es token ids
    bs_ids, es_ids = None, None
    if args.repr_type in ("mask1", "mask2"):
        if not (model_args.mask_embedding_sentence and hasattr(model_args, "mask_embedding_sentence_bs")):
            raise ValueError("repr_type=mask1/mask2 需要 mask_embedding_sentence 与 mask_embedding_sentence_template。")
        bs_ids = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
        es_ids = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)

    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [" ".join(s) for s in batch]
        # probing 数据已是 token list；保持与 sentemb 一致的句号补全会改变 probing 分布，这里不做。

        tok = tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        for k in tok:
            tok[k] = tok[k].to(device)

        with torch.no_grad():
            if args.repr_type == "mask2":
                # probing 中不走 sent_emb 分支（需要 model.bs 等训练期属性）
                injected_ids, injected_attn = _inject_template(
                    tok["input_ids"],
                    pad_token_id=tokenizer.pad_token_id,
                    bs_ids=bs_ids,
                    es_ids=es_ids,
                )

                out = model.bert(
                    input_ids=injected_ids,
                    attention_mask=injected_attn,
                    token_type_ids=None,
                    return_dict=True,
                )

                last_hidden = out.last_hidden_state  # [B, L, H]
                mask_positions = injected_ids == tokenizer.mask_token_id
                reps = last_hidden[mask_positions].view(injected_ids.size(0), args.mask_num, -1)
                return reps[:, args.mask_num - 1, :].detach().cpu()

            if args.repr_type == "cls":
                out = model.bert(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    token_type_ids=tok.get("token_type_ids", None),
                    return_dict=True,
                )
                return out.last_hidden_state[:, 0, :].detach().cpu()

            # mask1：模板注入后取第一个 MASK 的隐藏状态
            injected_ids, injected_attn = _inject_template(
                tok["input_ids"],
                pad_token_id=tokenizer.pad_token_id,
                bs_ids=bs_ids,
                es_ids=es_ids,
            )

            out = model.bert(
                input_ids=injected_ids,
                attention_mask=injected_attn,
                token_type_ids=None,
                return_dict=True,
            )

            last_hidden = out.last_hidden_state  # [B, L, H]
            mask_positions = injected_ids == tokenizer.mask_token_id
            # 每个样本应有 mask_num 个 mask token
            reps = last_hidden[mask_positions].view(injected_ids.size(0), args.mask_num, -1)
            return reps[:, 0, :].detach().cpu()

    # SentEval 参数（探针任务是分类）
    params = {
        "task_path": PATH_TO_DATA,
        "usepytorch": True,
        "kfold": 5,
        "batch_size": 128,
        "classifier": {"nhid": 0, "optim": "adam", "batch_size": 128, "tenacity": 5, "epoch_size": 4},
    }

    results = {}
    for task in PROBING_TASKS:
        se = senteval.engine.SE(params, batcher, prepare)
        results[task] = se.eval(task)

    summary = {task: float(results[task].get("acc", 0.0)) for task in PROBING_TASKS if task in results}

    record = {
        "timestamp": datetime.now().isoformat(),
        "script": os.path.basename(__file__),
        "repr_type": args.repr_type,
        "model_name_or_path": args.model_name_or_path,
        "results": results,
        "summary": summary,
    }

    os.makedirs("eval_results", exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join("eval_results", f"probing_{args.repr_type}_{time_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"[Probing] Saved results to: {out_path}")
    for k, v in summary.items():
        print(f"{k}\t{v:.2f}")


if __name__ == "__main__":
    main()

