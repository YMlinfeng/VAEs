import math
import random
import argparse
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from decord import VideoReader, cpu
from torch.nn import functional as F
from torchvision.transforms import Lambda, Compose
import sys
sys.path.append(".")
from opensora.models.causalvideovae import ae_wrapper
from opensora.dataset.transform import ToTensorVideo  

import imageio
from typing import Union
from numpy.typing import NDArray

def array_to_video(image_array: NDArray[np.uint8], fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
    """
    将一个图像数组保存为 MP4 视频，使用 imageio + libx264 编码器（兼容性最佳）

    参数:
      image_array: ndarray of shape (T, H, W, 3)，RGB 格式，uint8 类型
      fps: 帧率
      output_file: 输出视频文件路径，建议为 .mp4
    """
    assert image_array.ndim == 4 and image_array.shape[-1] == 3, "图像数组必须为 (T, H, W, 3)"
    assert image_array.dtype == np.uint8, "图像数组必须为 uint8 类型"

    writer = imageio.get_writer(output_file, fps=fps, codec='libx264', format='mp4', macro_block_size=None)

    for i, frame in enumerate(image_array):
        writer.append_data(frame)
    writer.close()
    print(f"视频成功保存到: {output_file}")

def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
    x = x.detach().cpu()
    # 将数据 clip 到 [-1,1]，再映射到 [0,1]
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    # 从 (1, T, C, H, W) 改为 (T, H, W, C)
    x = x.permute(0, 2, 3, 4, 1).squeeze(-1) if x.ndim == 5 else x.permute(0, 2, 3, 1)
    x = x.squeeze(0).float().numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return

def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    """
    利用 decord 读取视频帧，返回张量，形状为 (C, T, H, W)
    """
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    # 固定从第 0 帧开始采样
    s = 0
    e = sample_frames_len
    print(f'sample_frames_len {sample_frames_len}, 采样帧数: {num_frames * sample_rate}, 视频总帧数: {total_frames}')
    
    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()  # shape: (T, H, W, C)
    video_data = torch.from_numpy(video_data)
    # 转换至 (C, T, H, W)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

def preprocess(video_data: torch.Tensor, height: int = 512, width: int = 512) -> torch.Tensor:
    """
    直接将 video_data 的每一帧进行缩放（不裁剪），利用 Pillow 的 LANCZOS 重采样法，
    转换为目标尺寸 (height x width)，并归一化到 [-1, 1]。
    
    输入 video_data 的形状为 (C, T, H, W)，像素值范围为 [0,255]。
    输出为形状 (1, C, T, H, W) 的张量。
    """
    # 转换顺序： (C, T, H, W) -> (T, H, W, C)
    video_np = video_data.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    resized_frames = []
    num_frames = video_np.shape[0]
    print(f"开始对 {num_frames} 帧进行缩放，每帧缩放至 {width}x{height}")
    for i, frame in enumerate(video_np):
        # 利用 PIL 进行缩放，指定 LANCZOS 重采样方式以减少模糊损失
        im = Image.fromarray(frame)
        im_resized = im.resize((width, height), resample=Image.Resampling.LANCZOS)
        resized_frames.append(np.array(im_resized))
        if (i + 1) % 50 == 0:
            print(f"已处理 {i+1}/{num_frames} 帧")
    
    # 转换为 numpy 数组，形状为 (T, H, W, C)
    resized_frames = np.stack(resized_frames, axis=0)
    # 归一化，[0,255] -> [0,1]
    resized_frames = resized_frames.astype(np.float32) / 255.0
    # 映射到 [-1, 1]
    resized_frames = 2 * resized_frames - 1.0
    # 转换为 (C, T, H, W)
    resized_frames = np.transpose(resized_frames, (3, 0, 1, 2))
    video_tensor = torch.from_numpy(resized_frames)
    # 增加 batch 维度，变为 (1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor

def main(args: argparse.Namespace):
    device = args.device
    kwarg = {}
    vae = ae_wrapper[args.ae](args.ae_path, **kwarg).eval().to(device)
    
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    
    dtype = torch.bfloat16
    vae.eval()
    vae = vae.to(device, dtype=dtype)
    
    with torch.no_grad():
        # 读取并预处理视频：直接 resize，每帧采用 LANCZOS 重采样
        x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
        print("输入视频 shape:", x_vae.shape)  # (1, C, T, H, W)
        x_vae = x_vae.to(device, dtype=dtype)  # b c t h w
        latents = vae.encode(x_vae)
        latents = latents.to(dtype)
        video_recon = vae.decode(latents)  # b t c h w
        print("重构视频 shape:", video_recon.shape)
    
    # 保存推理后的视频
    custom_to_video(video_recon[0], fps=args.fps, output_file=args.rec_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--rec_path', type=str, default='')
    parser.add_argument('--ae', type=str, default='')
    parser.add_argument('--ae_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='results/pretrained')
    parser.add_argument('--fps', type=int, default=30)
    # 设置目标尺寸为 512x512
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_frames', type=int, default=17)
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--tile_sample_min_size', type=int, default=512)
    parser.add_argument('--tile_sample_min_size_t', type=int, default=33)
    parser.add_argument('--tile_sample_min_size_dec', type=int, default=256)
    parser.add_argument('--tile_sample_min_size_dec_t', type=int, default=33)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--save_memory', action='store_true')

    args = parser.parse_args()
    main(args)