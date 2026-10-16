
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class SUSTech1KDataset(Dataset):
    def __init__(
        self,
        root,
        split_json,
        split="TRAIN_SET",
        training=True,
        max_ids=None
    ):
        self.root = Path(root)
        self.training = training

        with open(split_json, "r") as f:
            partitions = json.load(f)

        identities = partitions[split]

        if max_ids is not None:
            identities = identities[:max_ids]

        self.samples = []
        self.by_label = defaultdict(list)

        for label, pid in enumerate(identities):
            subject_dir = self.root / pid

            # 按官方 PKL 文件命名规则寻找 Camera
            files = sorted(subject_dir.glob(
                "*/*/06-*-Camera-Sils_aligned.pkl"
            ))

            for cam_path in files:
                lidar_name = (
                    cam_path.name
                    .replace("06-", "01-", 1)
                    .replace(
                        "Camera-Sils_aligned.pkl",
                        "LiDAR-PCDs_depths.pkl"
                    )
                )

                lidar_path = cam_path.with_name(lidar_name)

                if not lidar_path.is_file():
                    continue

                index = len(self.samples)

                self.samples.append({
                    "cam": cam_path,
                    "lidar": lidar_path,
                    "label": label,
                    "pid": pid,
                    "seq": cam_path.parent.parent.name,
                    "view": cam_path.parent.name
                })

                self.by_label[label].append(index)

        if len(self.samples) == 0:
            raise RuntimeError(
                "没有找到成对的 Camera/LiDAR PKL，"
                "请检查数据根目录和文件命名。"
            )

        print(f"{split}: {len(self.samples)} paired sequences")
        print(f"IDs found: {len(self.by_label)}")

    def __len__(self):
        return len(self.samples)

    def load_sequence(self, path, frames, channels):
        # 只加载可信来源的 PKL 文件
        with open(path, "rb") as f:
            data = np.asarray(pickle.load(f))

        if len(data) == 0:
            raise ValueError(f"Empty sequence: {path}")

        # Train：随机抽帧
        # Test：固定均匀抽帧
        if self.training and len(data) >= frames:
            indices = np.sort(
                np.random.choice(
                    len(data), frames, replace=False
                )
            )
        else:
            indices = np.linspace(
                0, len(data) - 1, frames
            ).round().astype(int)

        data = data[indices]

        if channels == 1:
            # Camera: [T,H,W] -> [T,1,H,W]
            if data.ndim == 3:
                data = data[:, None, :, :]
            elif data.ndim == 4 and data.shape[-1] == 1:
                data = data.transpose(0, 3, 1, 2)

        elif channels == 3:
            # LiDAR depth: [T,3,H,W]
            # 兼容 [T,H,W,3]
            if data.ndim == 4 and data.shape[-1] == 3:
                data = data.transpose(0, 3, 1, 2)

        x = torch.from_numpy(
            np.ascontiguousarray(data)
        ).float()

        assert x.ndim == 4
        assert x.shape[1] == channels, (
            f"Unexpected shape {tuple(x.shape)}: {path}"
        )

        if x.shape[-2:] != (64, 64):
            x = F.interpolate(
                x, size=(64, 64),
                mode="bilinear",
                align_corners=False
            )

        # 官方图像型 PKL 通常存储为 0~255
        if x.max() > 1.0:
            x = x / 255.0

        return x

    def get_modality(self, index, modality):
        sample = self.samples[index]

        if modality == "camera":
            return self.load_sequence(
                sample["cam"], 30, 1
            )
        elif modality == "lidar":
            return self.load_sequence(
                sample["lidar"], 10, 3
            )

        raise ValueError(modality)

    def __getitem__(self, index):
        sample = self.samples[index]

        cam = self.get_modality(index, "camera")
        lidar = self.get_modality(index, "lidar")

        return cam, lidar, sample["label"]


class PKBatchSampler(Sampler):
    def __init__(
        self, dataset, p=3, k=2, steps=100
    ):
        self.by_label = {
            label: indices
            for label, indices in dataset.by_label.items()
            if len(indices) >= k
        }

        self.labels = list(self.by_label)
        self.p = p
        self.k = k
        self.steps = steps

        assert len(self.labels) >= p, (
            "Not enough identities with sufficient sequences"
        )

    def __iter__(self):
        for _ in range(self.steps):
            selected_ids = random.sample(
                self.labels, self.p
            )

            indices = []

            for label in selected_ids:
                indices.extend(
                    random.sample(
                        self.by_label[label], self.k
                    )
                )

            yield indices

    def __len__(self):
        return self.steps
