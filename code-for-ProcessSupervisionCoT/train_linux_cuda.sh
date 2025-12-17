#!/bin/bash

# ProcessSupervisionCoT Linux CUDA 训练脚本

echo "开始 ProcessSupervisionCoT Linux CUDA 训练..."

source ../.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练（使用 Linux CUDA 专用配置）
python process_supervision_cot_train.py configs/train_linux_cuda.yaml

echo "ProcessSupervisionCoT Linux CUDA 训练完成！"


