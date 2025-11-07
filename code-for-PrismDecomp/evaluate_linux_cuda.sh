#!/bin/bash

# PrismDecomp Linux CUDA评估脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp Linux CUDA评估..."

# 激活虚拟环境
# source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python prism_decomp_evaluation.py configs/evaluation_linux_cuda.yaml

echo "PrismDecomp Linux CUDA评估完成！"

