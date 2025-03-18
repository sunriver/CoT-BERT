#!/bin/bash
set -eo pipefail  # 关键配置：开启严格错误检测,只有python报错就退出
# 创建日志目录（如果不存在）
mkdir -p ./logs

# 获取当前时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 定义日志文件路径
LOG_FILE="./logs/train_log_${TIMESTAMP}.txt"

# 运行程序并输出到控制台和文件
python lmf_train.py 2>&1 | tee "$LOG_FILE"

# 参数验证与执行（新增功能）
if [ $# -eq 0 ]; then
#    echo "错误：缺少配置文件参数！"
#    echo "用法：./train.sh <config_file>"
#    exit 1
    echo "custom config_file is none"
    python lmf_train.py 2>&1 | tee "$LOG_FILE"
else
    config_file=$1
    echo "custom config_file：$config_file"
    # 传递命令行参数给训练脚本（关键修改点）
    python lmf_train.py "$config_file" 2>&1 | tee "$LOG_FILE"
fi

# 可选：添加日志完成提示
echo "训练完成，日志已保存至：$LOG_FILE"
