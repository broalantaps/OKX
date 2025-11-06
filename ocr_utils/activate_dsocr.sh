#!/bin/bash
# 激活 dsocr conda 环境的脚本

# 方法1：使用 conda init（推荐）
# 首先初始化 conda（如果还没有初始化）
eval "$(/home/dyh/miniconda3/bin/conda shell.bash hook)"
/home/dyh/miniconda3/envs/dsocr/bin/python word2png_function.py
# 激活环境
conda activate dsocr

# 验证环境是否激活成功
echo "当前 Python 路径: $(which python)"
echo "Python 版本: $(python --version)"
echo "当前 conda 环境: $CONDA_DEFAULT_ENV"

