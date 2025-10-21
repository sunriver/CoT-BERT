#!/bin/bash

# Mac M4 调试训练脚本
echo "=========================================="
echo "Mac M4 平台 SPR 模型调试训练"
echo "=========================================="

# 激活虚拟环境
source ../myvenv3.8/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:.."
export MPS_FALLBACK=1  # MPS回退到CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 检查平台
python -c "
import platform
print(f'操作系统: {platform.system()}')
print(f'架构: {platform.machine()}')
print(f'Python版本: {platform.python_version()}')
"

# 运行训练
echo "开始训练..."
python spr_train.py configs/train_mac_m4.yaml

echo "训练完成！"
