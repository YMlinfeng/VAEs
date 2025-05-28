# written by mzj

import os
import torch
import torchvision.utils as vutils
import numpy as np
import imageio

def visualize_input_and_recon(
    input_video: torch.Tensor,      # [3, T, H, W]
    recon_video: torch.Tensor,      # [3, T, H, W]
    save_dir: str,
    epoch: int,
    step: int,
    prefix: str = "train",
    wandb_log: bool = True,
):
    """
    可视化输入和重建图像序列（9 帧），拼接成一张图保存并上传 wandb。

    上一行是输入原图，下一行是重建图。
    """
    os.makedirs(save_dir, exist_ok=True)

    # Ensure input shape
    assert input_video.shape == recon_video.shape, "Input and recon must have same shape"
    assert input_video.dim() == 4, "Expected shape [C, T, H, W]"

    # 反归一化 [-1, 1] → [0, 1]
    def denorm(x):
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    input_video = denorm(input_video.detach().cpu())     # [3, T, H, W]
    recon_video = denorm(recon_video.detach().cpu())     # [3, T, H, W]

    # 转为 [T, C, H, W]
    input_frames = input_video.permute(1, 0, 2, 3)
    recon_frames = recon_video.permute(1, 0, 2, 3)

    # 拼接两行图像
    all_frames = torch.cat([input_frames, recon_frames], dim=0)  # [2T, C, H, W]
    grid = vutils.make_grid(all_frames, nrow=input_frames.shape[0], padding=2)

    # 保存路径
    filename = f"{prefix}_epoch{epoch}_step{step}_viz.png"
    save_path = os.path.join(save_dir, filename)
    vutils.save_image(grid, save_path)

    print(f"[Visualized] Saved to {save_path}")

    # 上传到 wandb
    if wandb_log:
        try:
            import wandb
            wandb.log({f"{prefix}/input_vs_recon": wandb.Image(save_path)}, step=step)
        except ImportError:
            print("wandb not installed, skipping wandb log.")


def save_video_comparison(input_video, recon_video, save_path, fps=5):
    """
    Save side-by-side video: 上原图，下重建图
    input_video, recon_video: [C, T, H, W]
    """
    denorm = lambda x: (x + 1) / 2.0
    input_video = denorm(input_video).clamp(0, 1)
    recon_video = denorm(recon_video).clamp(0, 1)

    T = input_video.shape[1]
    frames = []

    for t in range(T):
        in_frame = input_video[:, t]  # [C, H, W]
        re_frame = recon_video[:, t]

        in_img = (in_frame.permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
        re_img = (re_frame.permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)

        concat = np.vstack([in_img, re_img])  # 上下拼接
        frames.append(concat)

    imageio.mimsave(save_path, frames, fps=fps)
    #todo 视频和gif都想保存

import torch
import torchvision.utils as vutils
import numpy as np
import wandb

def denorm(tensor):
    """
    将 tensor 从 [-1, 1] 映射到 [0, 1]
    
    参数:
        tensor (torch.Tensor): 输入张量
    返回:
        torch.Tensor: 反归一化后的张量
    """
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

def log_fixed_val_video(log_dir, input_video: torch.Tensor, recon_video: torch.Tensor, current_step: int, fps: int = 1):
    """
    生成输入与重建视频对比图像及视频，并通过 wandb 上传，不在本地保存。
    
    参数:
        input_video (torch.Tensor): 输入视频张量, shape [3, T, H, W]
        recon_video (torch.Tensor): 重建视频张量, shape [3, T, H, W]
        current_step (int): 当前训练的 step，用于 wandb log
        fps (int): 视频每秒帧数，默认 5
    """
    # 反归一化
    input_video_denorm = denorm(input_video)
    recon_video_denorm = denorm(recon_video)

    # -------------------------
    # (1) 生成拼接图（上行为输入，下行为重建），直接上传 wandb
    # 将张量从 [3, T, H, W] 转换成 [T, 3, H, W]
    input_frames = input_video_denorm.permute(1, 0, 2, 3)
    recon_frames = recon_video_denorm.permute(1, 0, 2, 3)
    # 拼接两行图像: [input_frames; recon_frames] → [2T, 3, H, W]
    all_frames = torch.cat([input_frames, recon_frames], dim=0)
    grid = vutils.make_grid(all_frames, nrow=input_frames.shape[0], padding=2)
    grid_np = grid.detach().cpu().numpy().transpose(1, 2, 0)
    wandb.log({"val_fixed/in_vs_re": wandb.Image(grid_np)}, step=current_step)

    # -------------------------
    # (2) 生成视频: 对于每一帧分别将输入和重建帧上下拼接  
    frames = []
    T = input_video_denorm.shape[1]
    for t in range(T):
        # 提取每一帧, 形状为 [3, H, W]
        in_frame = input_video_denorm[:, t]
        re_frame = recon_video_denorm[:, t]
        # 转换为 HWC 格式的 numpy 数组，并放缩到 [0, 255]
        in_img = (in_frame.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        re_img = (re_frame.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        # 上下拼接 (输入在上，重建在下)
        concat_img = np.vstack([in_img, re_img])
        frames.append(concat_img)

    #? 为什么效果不对，问题还没有解决
    # video_np = np.array(frames)  # shape: [T, H_total, W, C]
    # wandb.log({"val_fixed/input_vs_recon_video": wandb.Video(video_np, fps=fps)}, step=current_step)

    os.makedirs(log_dir, exist_ok=True)
    video_path = os.path.join(log_dir, f"{current_step}.mp4")
    # 保存视频为 MP4 文件，需要 ffmpeg 安装在系统中
    imageio.mimsave(video_path, frames, fps=fps, codec="libx264")
    print(f"Saved fixed validation video to: {video_path}")