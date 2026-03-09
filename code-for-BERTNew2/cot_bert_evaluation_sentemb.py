import re
import sys
import os
sys.path.append('..') 

import tqdm
import torch
import logging
import argparse
import numpy as np
from argparse import Namespace
from prettytable import PrettyTable
from transformers import AutoConfig, AutoTokenizer

from cot_bert_model import BertForCL, RobertaForCL
from lmf_log_util import getMyLogger
from parse_args_util import load_configs

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

# Set up logger
logger = getMyLogger(__name__)

# Set PATHs
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def get_platform_eval_config_file():
    """根据平台返回对应的评估配置文件"""
    platform_type = detect_platform()
    
    if platform_type == "mac_m4":
        return "configs/evaluation_mac_m4.yaml"
    elif platform_type == "linux_cuda":
        return "configs/evaluation_linux_cuda.yaml"
    else:
        return "configs/evaluation_default.yaml"

def main():
    # 获取平台特定的配置文件
    config_file = get_platform_eval_config_file()
    print(f"使用配置文件: {config_file}")
    
    # 从配置文件加载参数
    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else ''
    args_list = load_configs(default_file=config_file, custom_file=config_custom_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_num', type=int, default=2)
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument('--mask_embedding_sentence_delta', action='store_true')
    parser.add_argument('--mask_embedding_sentence_org_mlp', action='store_true')
    parser.add_argument('--mask_embedding_sentence_autoprompt', action='store_true')
    parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")
    parser.add_argument("--mode", type=str, 
                        choices=['dev', 'test', 'fasttest'],
                        default='test', 
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts'],
                        default='sts',
                        help="What set of tasks to evaluate on. Currently only supports 'sts'")

    args = parser.parse_args(args_list)

    # 1. 构造 model_args (与 cot_bert_model 所需一致)
    model_args = Namespace(
        model_name_or_path=args.model_name_or_path,
        mask_embedding_sentence=args.mask_embedding_sentence,
        mask_embedding_sentence_template=args.mask_embedding_sentence_template or '',
        mask_num=args.mask_num,
        mask_embedding_sentence_delta=args.mask_embedding_sentence_delta,
        mask_embedding_sentence_delta_no_delta_eval=False,
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
        mask_embedding_sentence_different_template='',
        mask_embedding_sentence_negative_template='',
        mask_embedding_sentence_different_negative_template='',
        mask_embedding_sentence_autoprompt_continue_training_as_positive=False,
        cache_dir=None,
        model_revision='main',
        use_auth_token=False,
    )

    # 2. 加载 tokenizer 并解析 template
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    if model_args.mask_embedding_sentence and model_args.mask_embedding_sentence_template:
        template = model_args.mask_embedding_sentence_template
        template = template.replace('*mask*', tokenizer.mask_token)\
                           .replace('*sep+*', '').replace('*cls*', '').replace('*sent_0*', ' ')
        template = template.split(' ')
        model_args.mask_embedding_sentence_bs = template[0].replace('_', ' ')
        model_args.mask_embedding_sentence_es = template[1].replace('_', ' ')
        if 'roberta' in args.model_name_or_path:
            model_args.mask_embedding_sentence_bs = model_args.mask_embedding_sentence_bs.strip()

    # 3. 加载 BertForCL / RobertaForCL
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # 与训练一致的可选 config 设置
    config.num_columns = 3
    
    if 'roberta' in args.model_name_or_path:
        model = RobertaForCL.from_pretrained(
            args.model_name_or_path,
            config=config,
            model_args=model_args,
        )
    else:
        model = BertForCL.from_pretrained(
            args.model_name_or_path,
            config=config,
            model_args=model_args,
        )

    # 4. 设置模型属性 (与 cot_bert_train.py 938-965 对齐)
    if model_args.mask_embedding_sentence:
        model.pad_token_id = tokenizer.pad_token_id
        model.mask_token_id = tokenizer.mask_token_id
        model.mask_num = model_args.mask_num
        model.bs = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
        model.es = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)
        model.mask_embedding_template = tokenizer.encode(model_args.mask_embedding_sentence_bs + model_args.mask_embedding_sentence_es)

        # 默认设置 bs2/es2 等，防止 denoising 报错 (评估时通常与 bs/es 一致)
        model.bs2 = model.bs
        model.es2 = model.es
        model.mask_embedding_template2 = model.mask_embedding_template
        model.bs3 = model.bs
        model.es3 = model.es
        model.mask_embedding_template3 = model.mask_embedding_template
        model.bs4 = model.bs
        model.es4 = model.es
        model.mask_embedding_template4 = model.mask_embedding_template

        if model_args.mask_embedding_sentence_autoprompt:
            # 从 checkpoint 读 p_mbv
            state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
            if 'p_mbv' in state_dict:
                model.p_mbv.data = state_dict['p_mbv'].data
            
            mask_index = model.mask_embedding_template.index(tokenizer.mask_token_id)
            index_mbv = model.mask_embedding_template[1:mask_index] + model.mask_embedding_template[mask_index+1:-1]
            model.dict_mbv = index_mbv
            model.fl_mbv = [i <= 3 for i, _ in enumerate(index_mbv)]

    # 5. 设置设备
    platform_type = detect_platform()
    if platform_type == "mac_m4" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif platform_type == "linux_cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 6. SentEval batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        
        # NOTE: 当使用 BertForCL 的 sent_emb=True 时，模型内部会自动处理 template 注入。
        # 因此这里不需要手动对句子应用 template，否则会导致重复注入模板（double-template）。
        # 只保留基础的预处理（如补齐句号，可选）。
        for i, s in enumerate(sentences):
            if len(s) > 0 and s[-1] not in '.?"\'':
                sentences[i] = s + '.'

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )

        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            # 核心：直接调用 model 的 sent_emb 分支，内部会走 sentemb_forward
            outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            pooler_output = outputs.pooler_output

        return pooler_output.cpu()

    # 7. SentEval 任务与参数
    args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    
    if args.mode == 'dev' or args.mode == 'fasttest':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
    else:
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

    # 8. 运行评估并收集结果
    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # 9. 打印 STS 结果 (与现有评估格式对齐)
    print("------ %s ------" % (args.mode))
    scores = []
    task_names = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)

if __name__ == "__main__":
    main()
