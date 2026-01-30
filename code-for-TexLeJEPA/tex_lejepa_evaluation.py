"""
TexLeJEPA 评估入口：加载训练好的 checkpoint，在 STS 等任务上跑 SentEval。
"""
import sys
import os
import argparse
import torch
from prettytable import PrettyTable

from utils.platform_utils import detect_platform, get_platform_eval_config_file, setup_cuda_environment, print_platform_info
from utils.parse_args_util import load_configs
from utils.lmf_log_util import getMyLogger
from tex_lejepa_model import BertForTexLeJEPA
from transformers import AutoTokenizer

print_platform_info()
setup_cuda_environment()

logger = getMyLogger(__name__)

# SentEval 路径：相对 code-for-TexLeJEPA 目录，CoT-BERT/SentEval
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_SENTEVAL = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "SentEval"))
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, "data")

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    config_file = get_platform_eval_config_file()
    print(f"使用配置文件: {config_file}")

    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else ""
    args_list = load_configs(default_file=config_file, custom_file=config_custom_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="", help="TexLeJEPA checkpoint 路径")
    parser.add_argument("--mode", type=str, choices=["dev", "test", "fasttest"], default="test")
    parser.add_argument("--task_set", type=str, choices=["sts", "transfer", "full", "na"], default="sts")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args(args_list)
    if not getattr(args, "model_name_or_path", None) or not args.model_name_or_path:
        raise ValueError("请在配置或命令行中指定 --model_name_or_path（TexLeJEPA 训练输出目录）")

    device = torch.device("cpu")
    if detect_platform() == "mac_m4" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif detect_platform() == "linux_cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")

    model = BertForTexLeJEPA.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = model.to(device)
    model.eval()

    if args.task_set == "sts":
        tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
    elif args.task_set == "transfer":
        tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif args.task_set == "full":
        tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"] + ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    else:
        tasks = ["STSBenchmark", "SICKRelatedness"]

    if args.mode in ("dev", "fasttest"):
        params = {"task_path": os.path.abspath(PATH_TO_DATA), "usepytorch": True, "kfold": 5}
        params["classifier"] = {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2}
    else:
        params = {"task_path": os.path.abspath(PATH_TO_DATA), "usepytorch": True, "kfold": 10}
        params["classifier"] = {"nhid": 0, "optim": "adam", "batch_size": 64, "tenacity": 5, "epoch_size": 4}

    def prepare(params, samples):
        pass

    def batcher(params, batch):
        sentences = [" ".join(s) for s in batch]
        enc = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        with torch.no_grad():
            out = model(**enc, sent_emb=True)
            pooler = out.pooler_output
        return pooler.cpu()

    results = {}
    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    if args.mode == "dev":
        print("------ dev ------")
        names, scores = [], []
        for t in ["STSBenchmark", "SICKRelatedness"]:
            names.append(t)
            scores.append("%.2f" % (results.get(t, {}).get("dev", {}).get("spearman", [0])[0] * 100))
        print_table(names, scores)
        names, scores = [], []
        for t in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            names.append(t)
            scores.append("%.2f" % results.get(t, {}).get("devacc", 0))
        names.append("Avg.")
        scores.append("%.2f" % (sum(float(x) for x in scores) / max(1, len(scores))))
        print_table(names, scores)
    else:
        print("------ test / fasttest ------")
        names, scores = [], []
        for t in ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]:
            names.append(t)
            if t in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                scores.append("%.2f" % (results.get(t, {}).get("all", {}).get("spearman", {}).get("all", 0) * 100))
            else:
                scores.append("%.2f" % (getattr(results.get(t, {}).get("test", {}).get("spearman"), "correlation", 0) * 100))
        names.append("Avg.")
        scores.append("%.2f" % (sum(float(x) for x in scores) / max(1, len(scores))))
        print_table(names, scores)
        names, scores = [], []
        for t in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            names.append(t)
            scores.append("%.2f" % results.get(t, {}).get("acc", 0))
        names.append("Avg.")
        scores.append("%.2f" % (sum(float(x) for x in scores) / max(1, len(scores))))
        print_table(names, scores)


if __name__ == "__main__":
    main()
