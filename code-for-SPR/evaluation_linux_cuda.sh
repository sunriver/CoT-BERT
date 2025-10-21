#!/bin/bash

# Linux NVIDIA CUDA 评估脚本
echo "=========================================="
echo "Linux NVIDIA CUDA 平台 SPR 模型评估"
echo "=========================================="

# 激活虚拟环境
source ../myvenv3.8/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:.."
export CUDA_VISIBLE_DEVICES=0  # 评估时使用单个GPU

# 检查CUDA环境
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 检查平台
python -c "
import platform
print(f'操作系统: {platform.system()}')
print(f'架构: {platform.machine()}')
print(f'Python版本: {platform.python_version()}')
"

# 运行评估
echo "开始评估..."
python spr_evaluation.py configs/evaluation_default.yaml

echo "评估完成！"
