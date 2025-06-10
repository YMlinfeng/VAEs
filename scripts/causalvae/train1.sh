export WANDB_PROJECT=MZJVAE-8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export WANDB_ENTITY="xiao102851"
# export WANDB_API_KEY="f4416857501984f14835ded01a1fe0fbb6e7bcb7"
export WANDB_MODE=offline

torchrun \
    --nnodes=2 --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=10.124.2.139 \
    --master_port=12134 \
    opensora/train/train_causalvae.py \
    --exp_name Mhunyuan \
    --model_name hunyuan \
    --resolution "(384,384)" \
    --num_frames 9 \
    --batch_size 1 \
    --lr 0.00001 \
    --epochs 90 \
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
    --mix_precision fp32 \
    --test \


# --model_config scripts/causalvae/wfvae_8dim.json \