import re
import sys
import tqdm
import torch
import logging
import argparse
import numpy as np
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer, HfArgumentParser, AutoConfig
from  parse_args_util import load_configs
# from token_util import prepare_eval_features
from strategy_manage import get_strategy
from lmf_model import BertForCL, evaluate

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def cal_avg_cosine(k, n=100000):
    cos = torch.nn.CosineSimilarity(dim=-1)
    s = torch.tensor(k[:100000]).cuda()
    kk = []
    pbar = tqdm.tqdm(total=n)
    with torch.no_grad():
        for i in range(n):
            kk.append(cos(s[i:i+1], s).mean().item())
            pbar.set_postfix({'cosine': sum(kk)/len(kk)})
            pbar.update(1)
    return sum(kk) /len(kk)


def s_eval(args):
    se, task = args[0], args[1]
    return se.eval(task)


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    # args_list = ['--model_name_or_path', '../result/CoT-Bert',
    #             '--mode', 'test',
    #             '--mask_embedding_sentence',
    #             '--mask_num', '2',
    #             '--mask_embedding_sentence_org_mlp', 'false',
    #             '--mask_embedding_sentence_template', '*cls*_The_sentence_of_"*sent_0*"_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_only", action='store_true')
    parser.add_argument('--mlm_head_predict', action='store_true')
    parser.add_argument('--remove_continue_word', action='store_true')
    parser.add_argument('--mask_num', type=int, default=2)
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_org_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument('--mask_embedding_sentence_delta', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_autoprompt', action='store_true')
    parser.add_argument('--mask_embedding_sentence_org_mlp', action='store_true')
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")
    parser.add_argument('--mask_embedding_sentence_num_masks', type=int, default=2)
    
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg',  'avg_first_last'],
                        default='cls', 
                        help="Which pooler to use")
    
    parser.add_argument("--mode", type=str, 
                        choices=['dev', 'test', 'fasttest'],
                        default='test', 
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    
    parser.add_argument('--calc_anisotropy', action='store_true')


    config_custom_file = sys.argv[1] if len(sys.argv) > 1 else ''
    args_list = load_configs(default_file="./configs/evaluation_default.yaml", custom_file=config_custom_file)

    args = parser.parse_args(args_list)

 

    model = AutoModel.from_pretrained(args.model_name_or_path)
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)


    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.remove_continue_word:
        pun_remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', '‚', '¿', '–', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}
        if args.model_name_or_path == 'roberta-base':
            remove_set = {'Ġ.', 'Ġa', 'Ġthe', 'Ġin', 'a', 'Ġ, ', 'Ġis', 'Ġto', 'Ġof', 'Ġand', 'Ġon', 'Ġ\'', 's', '.', 'the', 'Ġman', '-', 'Ġwith', 'Ġfor', 'Ġat', 'Ġwoman', 'Ġare', 'Ġ"', 'Ġthat', 'Ġit', 'Ġdog', 'Ġsaid', 'Ġplaying', 'Ġwas', 'Ġas', 'Ġfrom', 'Ġ:', 'Ġyou', 'Ġan', 'i', 'Ġby'}
        else:
            remove_set = {".", "a", "the", "in", ",", "is", "to", "of", "and", "'", "on", "man", "-", "s", "with", "for", "\"", "at", "##s", "woman", "are", "it", "two", "that", "you", "dog", "said", "playing", "i", "an", "as", "was", "from", ":", "by", "white"}



    def batcher(params, batch, max_length=None):
            sentences = [' '.join(s) for s in batch]

            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'sent_positions']
        
            features = get_strategy().prepare_eval_features(tokenizer, sentences)
            input_ids = features['input_ids']

            bs = len(input_ids)
            num_sent = len(input_ids[0])

            flat_features = []
            max_length = 0
            sent_positions = features['sent_positions']
            for bs_i in range(bs):
                for sent_i in range(num_sent):
                    ids = input_ids[bs_i][sent_i]
                    flat_features.append({'input_ids': ids, 'sent_positions': sent_positions[bs_i][sent_i]})
                    length = len(ids)
                    if length > max_length:
                        max_length = length
                        
            batch = tokenizer.pad(
                flat_features,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            with torch.no_grad():
                outputs = evaluate(model, **batch, mask_token_id=tokenizer.mask_token_id, pad_token_id= tokenizer.pad_token_id)
                # outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)

                pooler_output = outputs.pooler_output

            return pooler_output.cpu()


   
    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        scores = []
        task_names = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        scores = []
        task_names = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
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

        scores = []
        task_names = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")

        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

if __name__ == "__main__":
    main()
