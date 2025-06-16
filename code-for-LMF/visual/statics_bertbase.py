import re
import sys
import tqdm
import torch
import logging
import argparse
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig

current_working_directory = os.getcwd()
print(f"当前工作目录: {current_working_directory}")

# Set PATHs
PATH_TO_SENTEVAL = '/workspace/CoT-BERT/SentEval'
PATH_TO_DATA = '/workspace/CoT-BERT/SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


args = {}
args['tasks'] = ['STSBenchmark']
args['mode'] = 'test'

params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
pretrain_path = "/workspace/pretrain_models/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
model = AutoModel.from_pretrained(pretrain_path).to(device)

# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch, max_length=None):
    sentences = [' '.join(s) for s in batch]
    input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_ids)
        outputs_cls = outputs.last_hidden_state[:, 0, :].cpu().numpy() # [CLS] token

    return outputs_cls


results = {}
for task in args['tasks']:
    se = senteval.engine.SE(params, batcher, prepare)
    result = se.eval(task)
    results[task] = result


from .statics_util import save_results

save_results(results, "statics_bert_base.json")


import json
import matplotlib.pyplot as plt
import pandas as pd
def show_scatter():
    # Load the JSON file
    with open('statics_bert_base.json', 'r') as f:
        data = json.load(f)

    # Extract relevant data
    all_sys_scores = data['STSBenchmark']['all_sys_scores']
    all_gs_scores = data['STSBenchmark']['all_gs_scores']

    # Create a DataFrame for easier plotting (optional, but good practice)
    df = pd.DataFrame({
        'all_sys_scores': all_sys_scores,
        'all_gs_scores': all_gs_scores
    })

    # Create the scatter plot with swapped axes
    plt.figure(figsize=(10, 6))
    plt.scatter(df['all_gs_scores'], df['all_sys_scores'], alpha=0.7) # Swapped x and y
    plt.title('Scatter Plot of all_gs_scores vs. all_sys_scores')
    plt.xlabel('all_gs_scores (Similarity Score Value)') # Swapped labels
    plt.ylabel('all_sys_scores (Similarity Score)')      # Swapped labels
    plt.grid(True)
    plt.show()