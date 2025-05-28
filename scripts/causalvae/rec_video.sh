# CUDA_VISIBLE_DEVICES=1 python3 examples/rec_video.py \
CUDA_VISIBLE_DEVICES=1 python3 examples/rec_video_withoutcrop.py \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/results/t523-32node" \
    --video_path /mnt/bn/occupancy3d/workspace/mzj/data/opensoraplan/video33/1/1.mp4 \
    --rec_path output/t523-32node_.mp4 \
    --device cuda \
    --sample_rate 1 \
    --num_frames 9 \
    --height 512 \
    --width 512 \
    --fps 1 \
    # --enable_tiling