#!/bin/bash

# CoT-BERT Mac M4训练脚本
# 基于Chain-of-Thought的句子表示学习

echo "开始CoT-BERT Mac M4训练..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python cot_bert_train.py configs/train_mac_m4.yaml

echo "CoT-BERT Mac M4训练完成！"

