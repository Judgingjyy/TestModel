
import torch
import torch.nn as nn

from crossgait_encoder import (
    ResNet9Encoder,
    temporal_pooling
)

from ministd_CrossGait_stage3 import (
    CrossGaitTrainModel,
    compute_losses
)


class CrossGait(nn.Module):
    def __init__(self, num_classes=2, lidar_channels=1):
        super().__init__()

        # 1. 两个模态各自独立的 Encoder
        self.camera_encoder = ResNet9Encoder(
            in_channels=1
        )
        self.lidar_encoder = ResNet9Encoder(
            in_channels=lidar_channels
        )

        # 2. 接入之前完成的模型
        # PMAM + HP + FCs + CMFA + BNNecks
        self.head = CrossGaitTrainModel(
            num_classes=num_classes
        )

    def encode_sequence(self, x, encoder):
        # [B,T,C,H,W] -> [B,32,T,16,16]
        x = encoder(x)

        # [B,32,T,16,16] -> [B,32,16,16]
        x = temporal_pooling(x)

        return x

    def forward(self, camera_seq, lidar_seq):

        # Camera Encoder + TP
        camera_feat = self.encode_sequence(
            camera_seq, self.camera_encoder
        )

        # LiDAR Encoder + TP
        lidar_feat = self.encode_sequence(
            lidar_seq, self.lidar_encoder
        )

        # PMAM + HP + FCs + CMFA + BNNecks
        output = self.head(camera_feat, lidar_feat)

        return output

    @torch.no_grad()
    def extract_embedding(self, x, modality):
        """
        单模态推理接口。
        调用前请先执行 model.eval()。
        """

        features = self.head.features

        if modality == "camera":
            encoder = self.camera_encoder
            specific_fc = features.cam_fcs

        elif modality == "lidar":
            encoder = self.lidar_encoder
            specific_fc = features.lidar_fcs

        else:
            raise ValueError(
                "modality must be 'camera' or 'lidar'"
            )

        # Encoder + Temporal Pooling
        x = self.encode_sequence(x, encoder)

        # Shared HP + PMAM + Concat
        _, _, fused = features.fusion.encode(x)

        # Modality-specific FC
        specific = specific_fc(fused)

        # Shared CMFA
        shared = features.cmfa.shared_fcs(specific)

        return shared


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 两个身份，每个身份两条序列
    labels = torch.tensor(
        [0, 0, 1, 1],
        device=device
    )

    B = len(labels)

    # 模拟输入：Camera Silhouettes
    camera = torch.rand(
        B, 30, 1, 64, 64,
        device=device
    )

    # 模拟输入：LiDAR Projected Depth
    lidar = torch.rand(
        B, 10, 1, 64, 64,
        device=device
    )

    model = CrossGait(
        num_classes=2,
        lidar_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # ===== 一次完整训练 =====
    model.train()

    output = model(camera, lidar)

    print("Camera Specific:",
          output["cam_specific"].shape)
    print("LiDAR Specific:",
          output["lidar_specific"].shape)

    print("Camera Shared:",
          output["cam_shared"].shape)
    print("LiDAR Shared:",
          output["lidar_shared"].shape)

    print("Camera Logits:",
          output["cam_logits"].shape)
    print("LiDAR Logits:",
          output["lidar_logits"].shape)

    # 计算 CrossGait 四项 Loss
    losses = compute_losses(
        output,
        cam_ids=labels,
        lidar_ids=labels
    )

    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")

    optimizer.zero_grad()

    # 反向传播
    losses["total"].backward()

    # 检查梯度是否传回两个 Encoder
    cam_grad = model.camera_encoder.stem[0].weight.grad
    lidar_grad = model.lidar_encoder.stem[0].weight.grad

    assert cam_grad is not None
    assert lidar_grad is not None
    assert torch.isfinite(cam_grad).all()
    assert torch.isfinite(lidar_grad).all()

    optimizer.step()

    print("Full training step completed!")

    # ===== 单模态推理 =====
    model.eval()

    cam_emb = model.extract_embedding(
        camera, "camera"
    )

    lidar_emb = model.extract_embedding(
        lidar, "lidar"
    )

    print("Camera Retrieval Embedding:", cam_emb.shape)
    print("LiDAR Retrieval Embedding:", lidar_emb.shape)

    assert cam_emb.shape == (B, 16, 6)
    assert lidar_emb.shape == (B, 16, 6)

    print("All tests passed!")
