#!/bin/bash

# ======================================
# 自动化训练脚本：Open-Sora-Plan + W&B 离线模式
# ======================================

# 1. 环境准备
cd /mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan

echo "🔧 安装依赖包中..."
pip install 'numpy<2.0.0' packaging
pip install --upgrade setuptools
pip install deepspeed==0.12.6 --prefer-binary
pip install colorlog
pip install einops
pip install lpips
pip install scikit-video
pip install -r requirements.txt

# 安装系统依赖
echo "🔧 安装系统包中..."
sudo apt update
sudo apt install -y libgl1-mesa-glx net-tools

# 权限设置
cd /mnt/bn/occupancy3d/workspace
sudo chown -R tiger:tiger mzj
chmod -R u+rwx mzj

# 模型权重文件准备
cd /mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan
mkdir -p ~/.cache/torch/hub/checkpoints/
cp ./alexnet-owt-7be5be79.pth ~/.cache/torch/hub/checkpoints/
cp ./vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/

# ======================================
# 2. 配置 wandb（离线模式）
# ======================================
export WANDB_PROJECT="MZJVAE-TRAIN"
# ✅ 设置为离线模式（避免网络超时）
export WANDB_MODE=offline

echo "🚀 开始训练..."
/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/TORCHRUN opensora/train/train_causalvae.py \
    --exp_name Hunyuan-t528-32GPU \
    --eval_video_path /mnt/bn/occupancy3d/workspace/mzj/data/opensoraplan/video33/1 \
    --model_name hunyuan \
    --model_config scripts/causalvae/wfvae_8dim.json \
    --resolution "(448,448)" \
    --num_frames 9 \
    --batch_size 1 \
    --lr 0.00001 \
    --epochs 10 \
    --disc_start 0 \
    --save_ckpt_step 4000 \
    --eval_steps 2000 \
    --eval_batch_size 1 \
    --eval_num_frames 9 \
    --eval_sample_rate 1 \
    --eval_subset_size 2000 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls opensora.models.causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --eval_num_video_log 4 \
    --pretrained_model_name_or_path /mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/baseline/hunyuan \
    --mix_precision fp16 \

echo "✅ 训练结束"

# ======================================
# 4. 上传离线日志到 W&B 云端（需联网）
# ======================================
# ⚠️ 请在联网状态下执行以下命令：

# 找到所有离线的日志目录
# 默认保存在 wandb/offline-run-* 下
# export WANDB_API_KEY="f4416857501984f14835ded01a1fe0fbb6e7bcb7"
# echo "📤 准备上传离线日志至 W&B 云端..."
# wandb sync wandb/offline-run-*

# echo "🎉 所有日志已同步完成！"