#!/bin/bash

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 创建输出目录
mkdir -p outputs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled  # 禁用 wandb
export WANDB_PROJECT=medical-llm-distillation

# 登录到 wandb（第一次运行时需要）
wandb login

# 启动训练
deepspeed src/train.py \
    --deepspeed configs/ds_config.json 