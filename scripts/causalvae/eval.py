#!/usr/bin/env python3
import os
import sys
import numpy as np
import imageio
from PIL import Image

def merge_videos(video1, video2, video3, output_video):
    """
    利用 imageio 逐帧读取三个视频，将每一帧先缩放为 512×512，
    然后左右拼接后写入新视频。帧数以三个视频中最短的为准。
    """
    # 打开视频文件
    reader1 = imageio.get_reader(video1)
    reader2 = imageio.get_reader(video2)
    reader3 = imageio.get_reader(video3)
    
    # 尝试从第一个视频获取 fps 信息
    meta = reader1.get_meta_data()
    fps = meta.get('fps', 1)  # 如未找到 fps 信息则默认 1
    fps = 1
    
    n_frames = min(reader1.count_frames(), reader2.count_frames(), reader3.count_frames())
    print(f"总帧数（取三者最小值）：{n_frames}")

    # 初始化视频写入器
    writer = imageio.get_writer(output_video, fps=fps)

    for i in range(n_frames):
        # 读取每个视频的帧，得到 numpy 数组
        frame1 = reader1.get_data(i)
        frame2 = reader2.get_data(i)
        frame3 = reader3.get_data(i)

        # 利用 Pillow 调整尺寸
        im1 = Image.fromarray(frame1).resize((512, 512))
        im2 = Image.fromarray(frame2).resize((512, 512))
        im3 = Image.fromarray(frame3).resize((512, 512))

        # 新建一张宽 512*3, 高 512 的图像，并依次粘贴三帧
        merged_im = Image.new('RGB', (512 * 3, 512))
        merged_im.paste(im1, (0, 0))
        merged_im.paste(im2, (512, 0))
        merged_im.paste(im3, (1024, 0))

        # 转换为 numpy 数组后写入视频
        writer.append_data(np.array(merged_im))
        if (i + 1) % 50 == 0:
            print(f"处理到帧：{i + 1}/{n_frames}")

    writer.close()
    print(f"视频拼接完成，保存为：{output_video}")

def create_horizontal_concat(video, size=(512, 512)):
    """
    读取视频中所有帧，调整为指定尺寸后，横向拼接生成一张大图（所有帧放一行）。
    返回拼接好的 PIL Image 对象。
    """
    reader = imageio.get_reader(video)
    frames = []
    n_frames = reader.count_frames()
    print(f"视频 {video} 总帧数：{n_frames}")
    for i, frame in enumerate(reader):
        # 通过 Pillow 调整帧尺寸到 512x512
        im = Image.fromarray(frame).resize(size)
        frames.append(im)
        if (i + 1) % 50 == 0:
            print(f"提取帧 {i + 1}/{n_frames}")
    if not frames:
        print(f"视频 {video} 无帧可读取！")
        return None

    # 计算最终宽度
    total_width = size[0] * len(frames)
    concat_im = Image.new('RGB', (total_width, size[1]))
    for idx, im in enumerate(frames):
        concat_im.paste(im, (idx * size[0], 0))
    return concat_im

def merge_horizontal_images(vertical_images, output_image):
    """
    将多张横向拼接好的图像（同宽或不同宽均可）竖向拼接成一张图，
    并保存到 output_image。
    """
    total_height = sum(im.height for im in vertical_images)
    max_width = max(im.width for im in vertical_images)
    merged_im = Image.new('RGB', (max_width, total_height))
    current_y = 0
    for im in vertical_images:
        merged_im.paste(im, (0, current_y))
        current_y += im.height
    merged_im.save(output_image)
    print(f"全帧大图生成完成，已保存到：{output_image}")

def main():
    # 定义三个视频的路径
    video1 = "/mnt/bn/occupancy3d/workspace/mzj/data/opensoraplan/video33/1/1.mp4"
    video2 = "/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/output/t523-32node-ori_.mp4"
    video3 = "/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/output/t523-32node_.mp4"

    # 定义输出路径
    output_dir = "/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_video = os.path.join(output_dir, "t532-32node-all.mp4")
    final_image = os.path.join(output_dir, "t532-32node-all.png")

    print("========== 1. 左右拼接视频 ==========")
    merge_videos(video1, video2, video3, output_video)

    print("\n========== 2. 提取帧并生成全帧大图 ==========")
    # 依次生成每个视频的横向拼接图像
    im_row1 = create_horizontal_concat(video1)
    im_row2 = create_horizontal_concat(video2)
    im_row3 = create_horizontal_concat(video3)
    
    if None in (im_row1, im_row2, im_row3):
        print("部分视频帧提取失败，无法生成大图！")
        sys.exit(1)

    # 将三行图像竖向拼接
    merge_horizontal_images([im_row1, im_row2, im_row3], final_image)

if __name__ == "__main__":
    main()