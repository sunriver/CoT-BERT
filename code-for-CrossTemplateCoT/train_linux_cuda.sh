#!/bin/bash

# CrossTemplateCoT Linux CUDA 训练脚本

echo "开始 CrossTemplateCoT Linux CUDA 训练..."

# 激活虚拟环境（如需）
# source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python cross_template_cot_train.py configs/train_linux_cuda.yaml

echo "CrossTemplateCoT Linux CUDA 训练完成！"


