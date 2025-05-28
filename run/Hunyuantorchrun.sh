#!/bin/bash

pip install 'numpy<2.0.0' packaging
pip install --upgrade setuptools
pip install deepspeed==0.12.6 --prefer-binary
pip install colorlog
pip install einops
pip install lpips
pip install scikit-video
cd /mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan
pip install -r requirements.txt
pip install --upgrade diffusers
sudo apt update
sudo apt install libgl1-mesa-glx -y
sudo apt install net-tools
# wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -P ~/.cache/torch/hub/checkpoints/
# wget "https://download.pytorch.org/models/vgg16-397923af.pth" -P ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
cd /mnt/bn/occupancy3d/workspace
sudo chown -R tiger:tiger mzj
chmod -R u+rwx mzj
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
    --exp_name Hunyuan-t528-56GPU \
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