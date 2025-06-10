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
import json
import matplotlib.pyplot as plt


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


class MultiViewSequenceDataset(Dataset):
    def __init__(self, pkl_path, resolution=256, sequence_length=3, tem_range=(5, 10)):
        super().__init__()
        self.views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.tem_range = tem_range  # 帧间隔范围（例如 5-10 帧间隔）
        self.pkl_path = pkl_path

        # 如果文件是 jsonl，则用 readlines() 读取，每行解析为 json，同时设置根目录
        if pkl_path.endswith('.pkl'):
            self.data_format = 'pkl'
            with open(pkl_path, 'rb') as f:
                self.data = pickle.load(f)['infos']
        elif pkl_path.endswith('.jsonl'):
            self.data_format = 'jsonl'
            self.data = []
            with open(pkl_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
            # 固定根目录，用于构造 jsonl 中的图片路径
            self.jsonl_root = '/mnt/bn/robotics-training-data-lf/lzy/real_word_data/preprocess/dcr_data'
        else:
            raise ValueError("Unsupported file format. Only .pkl and .jsonl are supported.")

        # 为避免采样时出现越界情况，按照最大间隔计算可采样的样本数
        max_gap = self.tem_range[1]
        self.num_samples = len(self.data) - (self.sequence_length - 1) * max_gap

        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, antialias=True),
            # 如有需要可以启用 Pad 或 CenterCrop
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        返回字典，包含 "video" 键对应的 [C, T, H, W] 视频张量。
        每个样本内 frame 之间间隔 gap 随机采样（tem_range 内），同时确保所有帧属于同一 scene_token。
        """
        gap = random.randint(self.tem_range[0], self.tem_range[1])
        # 检查最后一帧是否越界或不在同一 scene_token 内
        if (index + gap * (self.sequence_length - 1)) >= len(self.data) or \
        (self.data[index]['scene_token'] != self.data[index + gap * (self.sequence_length - 1)]['scene_token']):
            index = index - gap * (self.sequence_length - 1)
            index = max(index, 0)
            # 如果调整后仍不满足同一 scene_token，则退回至连续采样（gap=1）
            if (index + gap * (self.sequence_length - 1)) < len(self.data) and \
            (self.data[index]['scene_token'] != self.data[index + gap * (self.sequence_length - 1)]['scene_token']):
                gap = 1

        frames = []
        paths = []
        # print(f"index: {index}, gap: {gap}")
        for i in range(self.sequence_length):
            frame_idx = index + i * gap
            frame_info = self.data[frame_idx]

            for view in self.views:
                # 根据数据格式选择图片路径拼接方式
                if self.data_format == 'pkl':
                    cam_data = frame_info['cams'][view]
                    img_path = cam_data['data_path']
                    # pkl 格式下，若路径以 "data/" 开头则替换为预处理路径
                    if img_path.startswith("data/"):
                        img_path = img_path.replace("data", "/mnt/bn/occupancy3d/real_world_data/preprocess", 1)
                elif self.data_format == 'jsonl':
                    # jsonl 格式下，图片完整路径为：根目录 / scene_token / 相机对应路径
                    cam_data = frame_info['cams'][view]
                    rel_path = cam_data['data_path']
                    scene_path = frame_info['scene_token']
                    # 构造完整路径（注意分隔符）
                    img_path = self.jsonl_root.rstrip('/') + '/' + scene_path.strip('/') + '/' + rel_path.lstrip('/')
                else:
                    raise ValueError("Unsupported data format")

                paths.append(img_path)
                img = load_image(img_path)
                img_tensor = self.transform(img)  # [C, H, W]
                frames.append(img_tensor)

        video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]，T = sequence_length * num_views
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # 调整为 [C, T, H, W]
        return {"video": video_tensor, "label": "", "paths": paths}
    


def main():
    # 请根据自己的数据文件路径选择 pkl 或 jsonl 文件
    # 如果用固定 gap 的验证集，可以将 tem_range 设置为 (5, 5)
    dataset_path = "/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/dcr_bottom_half.jsonl" # 或 "data.jsonl"
    dataset = MultiViewSequenceDataset(dataset_path, resolution=256, sequence_length=3, tem_range=(5, 10))
    # 测试输出前三个样本
    print(f"dataset len: {len(dataset)}")
    for i in range(3):
        sample = dataset[i]
        video_tensor = sample["video"]  # [C, T, H, W]
        paths = sample["paths"]
        print(f"Case {i}:")
        print("Image paths:")
        for p in paths:
            print("   ", p)
        print("Tensor shape:", video_tensor.shape)
        print("Tensor min:", video_tensor.min().item(), 
            "max:", video_tensor.max().item(), 
            "mean:", video_tensor.mean().item())
        
        # 可视化图片
        num_views = len(dataset.views)
        # 每个样本有 sequence_length * num_views 帧，排列成 sequence_length 行，每行 num_views 列
        fig, axs = plt.subplots(dataset.sequence_length, num_views, figsize=(num_views * 3, dataset.sequence_length * 3))
        # video_tensor 的 shape 为 [C, T, H, W]，T = sequence_length * num_views
        for r in range(dataset.sequence_length):
            for c in range(num_views):
                idx = r * num_views + c
                # 取出第 idx 帧，转成 [H, W, C]
                frame = video_tensor[:, idx, :, :].permute(1, 2, 0).cpu().numpy()
                # 从 [-1,1] 反归一化到 [0,1]
                frame = (frame + 1.0) / 2.0
                axs[r, c].imshow(frame)
                axs[r, c].axis("off")
                axs[r, c].set_title(f"Frame {r} - {dataset.views[c]}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()


'''
在未来的 torchvision v0.17 版本中，所有的图像缩放变换（如 Resize()、RandomResizedCrop() 等）默认会启用抗锯齿（antialias=True），以统一在 PIL 和 Tensor 后端的表现。

当前版本中：

antialias=None 是默认。
None 的实际含义是：
如果输入是 Tensor，则 antialias=False。
如果输入是 PIL Image，则 antialias=True。

'''
