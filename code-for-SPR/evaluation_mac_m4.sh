#!/bin/bash

# Mac M4 评估脚本
echo "=========================================="
echo "Mac M4 平台 SPR 模型评估"
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

# 运行评估
echo "开始评估..."
python spr_evaluation.py configs/evaluation_default.yaml

echo "评估完成！"
