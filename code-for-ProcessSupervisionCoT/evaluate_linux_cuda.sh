#!/bin/bash

# ProcessSupervisionCoT Linux CUDA 评估脚本

echo "开始 ProcessSupervisionCoT Linux CUDA 评估..."

source ../.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python process_supervision_cot_evaluation.py configs/evaluation_linux_cuda.yaml

echo "ProcessSupervisionCoT Linux CUDA 评估完成！"


