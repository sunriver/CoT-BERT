#!/bin/bash

# ProcessSupervisionCoT评估脚本
# 基于过程监督的 CoT-BERT 句子表示学习

echo "开始ProcessSupervisionCoT评估..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python process_supervision_cot_evaluation.py configs/evaluation_default.yaml

echo "ProcessSupervisionCoT评估完成！"

