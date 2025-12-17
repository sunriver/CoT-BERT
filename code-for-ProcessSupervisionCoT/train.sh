#!/bin/bash

# ProcessSupervisionCoT训练脚本
# 基于过程监督的 CoT-BERT 句子表示学习

echo "开始ProcessSupervisionCoT训练..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python process_supervision_cot_train.py configs/train_default.yaml

echo "ProcessSupervisionCoT训练完成！"

