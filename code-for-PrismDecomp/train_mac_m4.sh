#!/bin/bash

# PrismDecomp Mac M4训练脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp Mac M4训练..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 设置日志目录（与模型输出目录一致）
LOG_DIR="../result/PrismDecomp-BERT"
# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_mac_m4_${TIMESTAMP}.log"

echo "日志文件保存位置: $LOG_FILE"
echo "=================================================="

# 运行训练，同时输出到终端和日志文件
python prism_decomp_train.py configs/train_mac_m4.yaml 2>&1 | tee "$LOG_FILE"

# 保存退出码
EXIT_CODE=${PIPESTATUS[0]}

echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "PrismDecomp Mac M4训练完成！"
else
    echo "PrismDecomp Mac M4训练失败，退出码: $EXIT_CODE"
fi
echo "日志文件已保存: $LOG_FILE"
exit $EXIT_CODE
