#!/bin/bash

# ProcessSupervisionCoT Mac M4 训练脚本

echo "开始 ProcessSupervisionCoT Mac M4 训练..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练（使用 Mac M4 专用配置）
python process_supervision_cot_train.py configs/train_mac_m4.yaml

echo "ProcessSupervisionCoT Mac M4 训练完成！"


