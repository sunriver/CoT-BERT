#!/bin/bash

# CrossTemplateCoT Linux CUDA 评估脚本

echo "开始 CrossTemplateCoT Linux CUDA 评估..."

# 激活虚拟环境（如需）
# source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python cross_template_cot_evaluation.py configs/evaluation_linux_cuda.yaml

echo "CrossTemplateCoT Linux CUDA 评估完成！"


