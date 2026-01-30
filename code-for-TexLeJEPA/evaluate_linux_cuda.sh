#!/bin/bash
# TexLeJEPA Linux CUDA 专用评估脚本

#set -e
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#source "$SCRIPT_DIR/../.venv/bin/activate"
#cd "$SCRIPT_DIR"
#export PYTHONPATH="${PYTHONPATH}:$SCRIPT_DIR"

echo "TexLeJEPA Linux CUDA 评估..."
python tex_lejepa_evaluation.py configs/evaluation_linux_cuda.yaml

echo "TexLeJEPA Linux CUDA 评估完成。"
