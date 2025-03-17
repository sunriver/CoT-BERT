#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p ./logs

# 获取当前时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 定义日志文件路径
LOG_FILE="./logs/train_log_${TIMESTAMP}.txt"

# 运行程序并输出到控制台和文件
python lmf_train.py 2>&1 | tee "$LOG_FILE"

# 可选：添加日志完成提示
echo "训练完成，日志已保存至：$LOG_FILE"
