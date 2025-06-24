# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from mmengine.fileio import load


# class NuScenesMultiViewVideoDataset(Dataset):
#     def __init__(
#         self,
#         ann_file,
#         resolution=256,
#         sequence_length=6,
#         views=None,
#         transform=None,
#         load_interval=1,
#         dataset_root="/mnt/bn/occupancy3d/workspace/mzj/"
#     ):
#         super().__init__()
#         self.sequence_length = sequence_length
#         self.views = views or [
#             "CAM_FRONT_LEFT",
#             "CAM_FRONT",
#             "CAM_FRONT_RIGHT",
#             "CAM_BACK_LEFT",
#             "CAM_BACK",
#             "CAM_BACK_RIGHT",
#         ]
#         self.resolution = resolution
#         self.load_interval = load_interval
#         self.dataset_root = dataset_root  
#         if isinstance(resolution, int):
#             resize_size = (resolution, resolution)
#         else:
#             resize_size = resolution

#         self.transform = transform or transforms.Compose([
#             transforms.Resize(resize_size, antialias=True),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: 2.0 * x - 1.0)
#         ])

#         data = load(ann_file)
#         self.data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
#         self.data_infos = self.data_infos[::self.load_interval]
#         self.scene_infos = data["scene_tokens"]

#         self.clips = self._build_clips()

#     def _build_clips(self):
#         clips = []
#         token_to_idx = {info["token"]: idx for idx, info in enumerate(self.data_infos)}

#         for scene in self.scene_infos:
#             for start in range(len(scene) - self.sequence_length + 1):
#                 clip_tokens = scene[start: start + self.sequence_length]
#                 clip_indices = []

#                 valid = True
#                 for token in clip_tokens:
#                     if token not in token_to_idx:
#                         valid = False
#                         break
#                     clip_indices.append(token_to_idx[token])

#                 if valid:
#                     clips.append(clip_indices)

#         return clips

#     def __len__(self):
#         return len(self.clips)

#     def __getitem__(self, index):
#         clip_indices = self.clips[index]

#         frames = []
#         paths = []

#         for frame_idx in clip_indices:
#             frame_info = self.data_infos[frame_idx]
#             cam_infos = frame_info["cams"]

#             for view in self.views:
#                 cam_data = cam_infos[view]
#                 img_path = cam_data["data_path"]

#                 # ✅ 将相对路径转换为绝对路径
#                 img_path = os.path.join(self.dataset_root, img_path.lstrip("../"))

#                 # 加载图像并转换
#                 img = Image.open(img_path).convert("RGB")
#                 img_tensor = self.transform(img)
#                 frames.append(img_tensor)
#                 paths.append(img_path)

#         video_tensor = torch.stack(frames, dim=0)
#         T, V = self.sequence_length, len(self.views)
#         video_tensor = video_tensor.view(T, V, *video_tensor.shape[1:])
#         video_tensor = video_tensor.permute(2, 0, 1, 3, 4)
#         video_tensor = video_tensor.reshape(video_tensor.shape[0], T * V, *video_tensor.shape[3:])

#         return {
#             "video": video_tensor,
#             "paths": paths
#         }


# def main():
#     dataset_path = '/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl'
#     dataset = NuScenesMultiViewVideoDataset(dataset_path, resolution=(224, 400), sequence_length=6)
    
#     print(f"dataset len: {len(dataset)}")
#     os.makedirs("visualizations", exist_ok=True)  # ✅ 保存目录

#     for i in range(6):
#         sample = dataset[i]
#         video_tensor = sample["video"]
#         paths = sample["paths"]

#         print(f"Case {i}:")
#         print("Image paths:")
#         for p in paths:
#             print("   ", p)
#         print("Tensor shape:", video_tensor.shape)
#         print("Tensor min:", video_tensor.min().item(), 
#               "max:", video_tensor.max().item(), 
#               "mean:", video_tensor.mean().item())

#         num_views = len(dataset.views)
#         fig, axs = plt.subplots(dataset.sequence_length, num_views, figsize=(num_views * 3, dataset.sequence_length * 3))

#         for r in range(dataset.sequence_length):
#             for c in range(num_views):
#                 idx = r * num_views + c
#                 frame = video_tensor[:, idx, :, :].permute(1, 2, 0).cpu().numpy()
#                 frame = (frame + 1.0) / 2.0  # 反归一化
#                 axs[r, c].imshow(frame)
#                 axs[r, c].axis("off")
#                 axs[r, c].set_title(f"Frame {r} - {dataset.views[c]}")

#         plt.tight_layout()

#         # ✅ 保存可视化图片
#         save_path = f"visualizations/sample_{i}.png"
#         plt.savefig(save_path)
#         print(f"Saved visualization to {save_path}")

#         plt.close()


# if __name__ == "__main__":
#     main()

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from mmengine.fileio import load
from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mmengine.fileio import load


class NuScenesMultiViewVideoDataset(Dataset):
    def __init__(
        self,
        ann_file,
        resolution=256,
        sequence_length=6,
        views=None,
        transform=None,
        load_interval=1,
        dataset_root="/mnt/bn/occupancy3d/workspace/mzj/",
        log_file="dataset_check_log.txt"
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.views = views or [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            # "CAM_BACK_LEFT",
            # "CAM_BACK",
            # "CAM_BACK_RIGHT",
        ]
        self.resolution = resolution
        self.load_interval = load_interval
        self.dataset_root = dataset_root
        self.log_file = log_file

        if isinstance(resolution, int):
            resize_size = (resolution, resolution)
        else:
            resize_size = resolution

        self.transform = transform or transforms.Compose([
            transforms.Resize(resize_size, antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0)
        ])

        # === Load and preprocess data ===
        data = load(ann_file)
        self.data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        self.data_infos = self.data_infos[::self.load_interval]
        self.scene_infos = data["scene_tokens"]

        # === Build token to scene_token mapping ===
        self.token_to_scene = {}
        for scene in self.scene_infos:
            for token in scene:
                self.token_to_scene[token] = scene  # or self.token_to_scene[token] = scene_id if needed

        # === Add scene_token to each frame info ===
        for info in self.data_infos:
            token = info["token"]
            if token in self.token_to_scene:
                info["scene_token"] = self.token_to_scene[token][0]  # assign scene token (first token of scene)
            else:
                info["scene_token"] = "UNKNOWN"

        self.token_to_index = {info["token"]: idx for idx, info in enumerate(self.data_infos)}
        self.clips = self._build_clips()

    def _build_clips(self):
        clips = []
        for scene in self.scene_infos:
            for start in range(len(scene) - self.sequence_length + 1):
                clip_tokens = scene[start: start + self.sequence_length]
                clip_indices = []
                valid = True
                for token in clip_tokens:
                    if token not in self.token_to_index:
                        valid = False
                        break
                    clip_indices.append(self.token_to_index[token])
                if valid:
                    clips.append(clip_indices)
        return clips

    def __len__(self):
        return len(self.clips)

    def log_error(self, index, error_type, message):
        log_line = f"[{error_type}] Index {index}: {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_line)

    def __getitem__(self, index):
        clip_indices = self.clips[index]

        frames = []
        paths = []

        # ✅ Check: all 6 frames belong to the same scene
        scene_tokens = [self.data_infos[i].get("scene_token", "UNKNOWN") for i in clip_indices]
        if len(set(scene_tokens)) > 1:
            self.log_error(index, "SceneMismatch", f"Scene tokens: {scene_tokens}")

        for frame_idx in clip_indices:
            frame_info = self.data_infos[frame_idx]
            cam_infos = frame_info["cams"]

            # ✅ Check: all 6 views are from the same timestamp
            timestamps = [cam_infos[view]["timestamp"] for view in self.views]
            # if len(set(timestamps)) > 1:
            if max(timestamps) - min(timestamps) > 2e4:  # 超过20ms（单位：微秒）
                self.log_error(index, "TimestampMismatch", f"Timestamps: {timestamps}")

            for view in self.views:
                cam_data = cam_infos[view]
                img_path = cam_data["data_path"]
                img_path = os.path.join(self.dataset_root, img_path.lstrip("../"))

                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.transform(img)
                    frames.append(img_tensor)
                    paths.append(img_path)
                except Exception as e:
                    self.log_error(index, "ImageLoadError", f"{img_path} - {str(e)}")
                    dummy = torch.zeros(3, *self.transform.transforms[0].size)
                    frames.append(dummy)
                    paths.append(img_path)

        video_tensor = torch.stack(frames, dim=0)
        T, V = self.sequence_length, len(self.views)
        video_tensor = video_tensor.view(T, V, *video_tensor.shape[1:])
        video_tensor = video_tensor.permute(2, 0, 1, 3, 4)
        video_tensor = video_tensor.reshape(video_tensor.shape[0], T * V, *video_tensor.shape[3:])

        return {
            "video": video_tensor,
            "paths": paths
        }
    
def main():
    dataset_path = '/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl'
    dataset = NuScenesMultiViewVideoDataset(dataset_path, resolution=(224, 400), sequence_length=7)
    
    print(f"dataset len: {len(dataset)}")
    os.makedirs("visualizations", exist_ok=True)  #

    for i in range(6):
        sample = dataset[i]
        video_tensor = sample["video"]
        paths = sample["paths"]

        print(f"Case {i}:")
        print("Image paths:")
        for p in paths:
            print("   ", p)
        print("Tensor shape:", video_tensor.shape)
        print("Tensor min:", video_tensor.min().item(), 
              "max:", video_tensor.max().item(), 
              "mean:", video_tensor.mean().item())

        num_views = len(dataset.views)
        fig, axs = plt.subplots(dataset.sequence_length, num_views, figsize=(num_views * 3, dataset.sequence_length * 3))

        for r in range(dataset.sequence_length):
            for c in range(num_views):
                idx = r * num_views + c
                frame = video_tensor[:, idx, :, :].permute(1, 2, 0).cpu().numpy()
                frame = (frame + 1.0) / 2.0  # 反归一化
                axs[r, c].imshow(frame)
                axs[r, c].axis("off")
                axs[r, c].set_title(f"Frame {r} - {dataset.views[c]}")

        plt.tight_layout()

        # ✅ 保存可视化图片
        save_path = f"visualizations/sample_{i}.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

        plt.close()


if __name__ == "__main__":
    main()