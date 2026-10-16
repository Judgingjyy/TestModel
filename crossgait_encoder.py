
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 输入和输出尺寸不相同时，投影 Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual Connection
        out = out + identity
        out = self.relu(out)

        return out


class ResNet9Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem: 1 层卷积
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # 四个残差块，每块两层卷积
        self.stage1 = BasicBlock(8, 8, stride=1)
        self.stage2 = BasicBlock(8, 16, stride=2)
        self.stage3 = BasicBlock(16, 24, stride=2)
        self.stage4 = BasicBlock(24, 32, stride=1)

    def forward(self, x):
        # 输入 [B,T,C,H,W]
        B, T, C, H, W = x.shape

        # 将时间维合并进 Batch
        # [B,T,C,H,W] -> [B*T,C,H,W]
        x = x.reshape(B * T, C, H, W)

        x = self.stem(x)     # [B*T,8,64,64]
        x = self.stage1(x)   # [B*T,8,64,64]
        x = self.stage2(x)   # [B*T,16,32,32]
        x = self.stage3(x)   # [B*T,24,16,16]
        x = self.stage4(x)   # [B*T,32,16,16]

        # 恢复时间维
        _, C_new, H_new, W_new = x.shape
        x = x.reshape(B, T, C_new, H_new, W_new)

        # [B,T,C,H,W] -> [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x


def temporal_pooling(x):
    # [B,C,T,H,W] -> [B,C,H,W]
    return x.amax(dim=2)


if __name__ == "__main__":
    torch.manual_seed(42)

    cam_encoder = ResNet9Encoder(in_channels=1)
    lidar_encoder = ResNet9Encoder(in_channels=1)

    # 小 Batch，用来验证 Encoder
    cam = torch.randn(2, 30, 1, 64, 64)
    lidar = torch.randn(2, 10, 1, 64, 64)

    cam_feat = cam_encoder(cam)
    lidar_feat = lidar_encoder(lidar)

    print("Camera Encoder:", cam_feat.shape)
    print("LiDAR Encoder:", lidar_feat.shape)

    cam_pool = temporal_pooling(cam_feat)
    lidar_pool = temporal_pooling(lidar_feat)

    print("Camera TP:", cam_pool.shape)
    print("LiDAR TP:", lidar_pool.shape)

    assert cam_pool.shape == (2, 32, 16, 16)
    assert lidar_pool.shape == (2, 32, 16, 16)

    print("Encoder tests passed!")
