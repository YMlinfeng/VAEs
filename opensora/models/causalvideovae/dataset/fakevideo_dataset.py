# written by ymlf

import os
import pickle
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
    return img


# ------------------- Dataset 类 -------------------
class MultiViewSequenceDataset(Dataset):
    def __init__(self, pkl_path, resolution=256, sequence_length=3):
        super().__init__()
        self.views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        self.sequence_length = sequence_length
        self.resolution = resolution

        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)['infos']

        # 总帧数（每帧都可以作为起点，除了最后两帧）
        self.num_samples = len(self.data) - (self.sequence_length - 1)

        # 图像处理变换
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, antialias=True),  # Resize 先操作 PIL 图像
            # transforms.Pad(padding, fill=0, padding_mode="constant"),  # 填充到正方形
            # transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),  # PIL -> Tensor
            transforms.Lambda(lambda x: 2.0 * x - 1.0),  # 归一化到 [-1, 1]
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        返回 [C, T=9, H, W] 的视频张量
        """
        frames = []
        for i in range(self.sequence_length):  # 3 帧
            frame_idx = index + i
            frame_info = self.data[frame_idx]

            for view in self.views:  # 每帧的 3 个视角
                cam_data = frame_info['cams'][view]
                img_path = cam_data['data_path']
                if img_path.startswith("data/"):
                    img_path = img_path.replace("data", "/mnt/bn/occupancy3d/real_world_data/preprocess", 1)

                img = load_image(img_path)
                img_tensor = self.transform(img)  # [C, H, W]
                frames.append(img_tensor)

        # 拼成 [T, C, H, W]
        video_tensor = torch.stack(frames, dim=0)  # [9, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T=9, H, W]
        return {"video": video_tensor, "label": ""}
    

'''
在未来的 torchvision v0.17 版本中，所有的图像缩放变换（如 Resize()、RandomResizedCrop() 等）默认会启用抗锯齿（antialias=True），以统一在 PIL 和 Tensor 后端的表现。

当前版本中：

antialias=None 是默认。
None 的实际含义是：
如果输入是 Tensor，则 antialias=False。
如果输入是 PIL Image，则 antialias=True。

'''