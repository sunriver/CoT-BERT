import sys
import os
import json
import inspect
from datetime import datetime

#全局层面把旧 API 重定向到新 API，SentEval 调用时自然就用了兼容版本
inspect.getargspec = inspect.getfullargspec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_SCRIPT_DIR, ".."))

import torch
import argparse
from argparse import Namespace
from transformers import AutoConfig, AutoTokenizer

from cot_bert_model import BertForCL, RobertaForCL
from git_repo_info import get_git_repo_info
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


def _namespace_to_jsonable(ns):
    """将 Namespace 转为可 JSON 序列化的 dict。"""
    if ns is None:
        return {}
    out = {}
    for k, v in vars(ns).items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def _load_checkpoint_eval_metadata(model_dir: str, script_dir: str):
    """
    与 cot_bert_evaluation_sentemb.save_experiment_log 一致：从 checkpoint 目录读取
    train_config_full.json、trainer_state.json 摘要，并附 Git 信息。
    """
    train_config_full_path = None
    train_config_full = None
    trainer_state_summary = None

    if isinstance(model_dir, str) and model_dir:
        train_config_full_path = os.path.join(model_dir, "train_config_full.json")
        if os.path.isfile(train_config_full_path):
            try:
                with open(train_config_full_path, "r", encoding="utf-8") as f:
                    train_config_full = json.load(f)
            except Exception as e:
                print(f"[ProbingLog] Failed to load train_config_full.json from '{train_config_full_path}': {e}")
        else:
            print(f"[ProbingLog] train_config_full.json not found in model dir: {train_config_full_path}")

        trainer_state_path = os.path.join(model_dir, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            try:
                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    trainer_state = json.load(f)

                def _extract_step_from_ckpt(ckpt):
                    if not isinstance(ckpt, str) or "checkpoint-" not in ckpt:
                        return None
                    try:
                        step_str = ckpt.split("checkpoint-")[-1].split("/")[0]
                        return int(step_str)
                    except Exception:
                        return None

                best_ckpt = trainer_state.get("best_model_checkpoint", None)
                best_step = _extract_step_from_ckpt(best_ckpt)
                log_history = trainer_state.get("log_history", [])
                if not isinstance(log_history, list):
                    log_history = []
                last_log_history = log_history[-5:] if len(log_history) >= 5 else log_history

                best_eval_entry = None
                if best_step is not None and log_history:
                    for entry in reversed(log_history):
                        if not isinstance(entry, dict):
                            continue
                        step_val = entry.get("step", entry.get("global_step", None))
                        if step_val == best_step:
                            eval_keys = {
                                k: v
                                for k, v in entry.items()
                                if isinstance(k, str) and k.startswith("eval_")
                            }
                            if eval_keys:
                                best_eval_entry = {"step": step_val, "eval_metrics": eval_keys}
                            else:
                                best_eval_entry = {"step": step_val}
                            break

                trainer_state_summary = {
                    "global_step": trainer_state.get("global_step", None),
                    "epoch": trainer_state.get("epoch", None),
                    "best_model_checkpoint": best_ckpt,
                    "best_model_checkpoint_step": best_step,
                    "best_metric": trainer_state.get("best_metric", None),
                    "last_log_history": last_log_history,
                    "best_eval_entry": best_eval_entry,
                }
            except Exception as e:
                print(f"[ProbingLog] Failed to load trainer_state.json from '{trainer_state_path}': {e}")

    git_repo_info = get_git_repo_info(script_dir)
    return {
        "model_dir": model_dir,
        "train_config_full_path": train_config_full_path,
        "train_config_full": train_config_full,
        "trainer_state_summary": trainer_state_summary,
        "git": git_repo_info,
    }


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
    parser.add_argument(
        "--repr_type",
        type=str,
        choices=["mask1", "mask2", "cls", "all"],
        default="all",
        help="all: 依次探测 mask1、mask2、cls 并输出对比表与合并 JSON",
    )
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

    repr_types = ["mask1", "mask2", "cls"] if args.repr_type == "all" else [args.repr_type]
    needs_mask_template = any(r in ("mask1", "mask2") for r in repr_types)

    bs_ids, es_ids = None, None
    if needs_mask_template:
        if not (model_args.mask_embedding_sentence and hasattr(model_args, "mask_embedding_sentence_bs")):
            raise ValueError(
                "repr_type 含 mask1/mask2 时需要 mask_embedding_sentence 与 mask_embedding_sentence_template。"
            )
        bs_ids = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
        es_ids = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)

    def prepare(params, samples):
        return

    def make_batcher(current_repr: str):
        """按当前表示类型返回 SentEval batcher（闭包）。"""

        def batcher(params, batch):
            sentences = [" ".join(s) for s in batch]
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
                if current_repr == "cls":
                    out = model.bert(
                        input_ids=tok["input_ids"],
                        attention_mask=tok["attention_mask"],
                        token_type_ids=tok.get("token_type_ids", None),
                        return_dict=True,
                    )
                    return out.last_hidden_state[:, 0, :].detach().cpu()

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
                last_hidden = out.last_hidden_state
                mask_positions = injected_ids == tokenizer.mask_token_id
                reps = last_hidden[mask_positions].view(injected_ids.size(0), args.mask_num, -1)
                if current_repr == "mask1":
                    return reps[:, 0, :].detach().cpu()
                # mask2
                return reps[:, args.mask_num - 1, :].detach().cpu()

        return batcher

    # SentEval 参数（探针任务是分类）
    params = {
        "task_path": PATH_TO_DATA,
        "usepytorch": True,
        "kfold": 5,
        "batch_size": 128,
        "classifier": {"nhid": 0, "optim": "adam", "batch_size": 128, "tenacity": 5, "epoch_size": 4},
    }

    results_by_repr = {}
    summary_by_repr = {}
    for rtype in repr_types:
        batcher = make_batcher(rtype)
        results = {}
        for task in PROBING_TASKS:
            se = senteval.engine.SE(params, batcher, prepare)
            results[task] = se.eval(task)
        results_by_repr[rtype] = results
        summary_by_repr[rtype] = {
            task: float(results[task].get("acc", 0.0)) for task in PROBING_TASKS if task in results
        }

    time_str = datetime.now().isoformat()
    comparison = {}
    for task in PROBING_TASKS:
        row = {r: summary_by_repr[r].get(task, 0.0) for r in repr_types}
        best_r = max(repr_types, key=lambda r: row[r])
        row["best"] = best_r
        row["best_acc"] = row[best_r]
        # 相对 mask2 的差值（便于与主评测句向量对齐分析）
        if "mask2" in repr_types:
            row["diff_vs_mask2"] = {r: round(row[r] - row["mask2"], 2) for r in repr_types}
        comparison[task] = row

    checkpoint_meta = _load_checkpoint_eval_metadata(args.model_name_or_path, _SCRIPT_DIR)

    record = {
        "timestamp": time_str,
        "script": os.path.basename(__file__),
        "repr_type": args.repr_type,
        "repr_types_run": repr_types,
        "model_name_or_path": args.model_name_or_path,
        "results_by_repr": results_by_repr,
        "summary_by_repr": summary_by_repr,
        "comparison": comparison,
        # 与 sentemb 评估日志对齐，便于与 STS 等结果对照复现
        **checkpoint_meta,
        "eval_args": _namespace_to_jsonable(args),
        "model_args": _namespace_to_jsonable(model_args),
        "model_config": config.to_dict(),
        "platform_type": platform_type,
        "eval_config_default": default_cfg,
        "eval_config_custom": custom_cfg if custom_cfg else None,
        "device": str(device),
        "path_to_senteval_data": PATH_TO_DATA,
        "probing_senteval_params": params,
        "probing_tasks": list(PROBING_TASKS),
    }

    # 单种表示时保留与旧版兼容的顶层字段
    if len(repr_types) == 1:
        only = repr_types[0]
        record["results"] = results_by_repr[only]
        record["summary"] = summary_by_repr[only]

    os.makedirs("eval_results", exist_ok=True)
    file_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "all" if args.repr_type == "all" else args.repr_type
    out_path = os.path.join("eval_results", f"probing_{suffix}_{file_ts}.json")
    def _json_default(o):
        return str(o)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[Probing] Saved results to: {out_path}")
    print(f"[ProbingLog] Metadata (train_config / trainer_state / git) merged into JSON for comparison with sentemb logs.")

    if len(repr_types) > 1:
        col_w = max(len(t) for t in PROBING_TASKS)
        header = f"{'Task':<{col_w}}  {'MASK1':>8}  {'MASK2':>8}  {'CLS':>8}  {'Best':>8}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for task in PROBING_TASKS:
            m1 = summary_by_repr["mask1"].get(task, 0.0)
            m2 = summary_by_repr["mask2"].get(task, 0.0)
            cl = summary_by_repr["cls"].get(task, 0.0)
            b = comparison[task]["best"]
            print(f"{task:<{col_w}}  {m1:8.2f}  {m2:8.2f}  {cl:8.2f}  {b:>8}")
        print(sep)
        print("diff_vs_mask2 (MASK1, CLS):")
        for task in PROBING_TASKS:
            d = comparison[task].get("diff_vs_mask2", {})
            print(
                f"  {task}: mask1 {d.get('mask1', 0):+.2f}, cls {d.get('cls', 0):+.2f}"
            )
    else:
        only = repr_types[0]
        for k, v in summary_by_repr[only].items():
            print(f"{k}\t{v:.2f}")


if __name__ == "__main__":
    main()

