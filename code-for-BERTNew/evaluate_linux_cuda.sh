#!/bin/bash

# CoT-BERT Linux CUDA评估脚本
# 基于Chain-of-Thought的句子表示学习

echo "开始CoT-BERT Linux CUDA评估..."

# 激活虚拟环境


# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python cot_bert_evaluation.py configs/evaluation_linux_cuda.yaml

echo "CoT-BERT Linux CUDA评估完成！"

