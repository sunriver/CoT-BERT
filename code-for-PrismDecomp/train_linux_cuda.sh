#!/bin/bash

# PrismDecomp Linux CUDA训练脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp Linux CUDA训练..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python prism_decomp_train.py configs/train_linux_cuda.yaml

echo "PrismDecomp Linux CUDA训练完成！"
