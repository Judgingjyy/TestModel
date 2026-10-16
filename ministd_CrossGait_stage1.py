
import math
import torch
import torch.nn as nn


# ===== 1. Horizontal Pooling =====
class HorizontalPooling(nn.Module): #只分一次了
    def __init__(self, num_bins=4):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        assert H % self.num_bins == 0

        # 按高度划分为 P 个区域
        # [B,C,H,W] -> [B,C,P,H//P,W]
        x = x.reshape(
            B, C, self.num_bins, H // self.num_bins, W
        )

        # [B,C,P,区域像素数]
        x = x.flatten(start_dim=3)

        # 每个区域做 Mean + Max Pooling
        x = x.mean(dim=-1) + x.amax(dim=-1)

        # [B,C,P]
        return x


# ===== 2. PMAM =====
class PMAM(nn.Module):
    def __init__(
        self,
        channels=32,
        height=16,
        width=16,
        num_prototypes=2
    ):
        super().__init__()

        self.channels = channels
        self.hw = height * width

        # 共享可学习原型
        # [K,C]
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels) * 0.02
        )

        # 空间维度线性投影
        self.key_proj = nn.Linear(self.hw, self.hw)
        self.value_proj = nn.Linear(self.hw, self.hw)

    def forward(self, x, return_attention=False):
        # [B,C,H,W]
        B, C, H, W = x.shape

        assert C == self.channels
        assert H * W == self.hw

        # [B,C,H,W] -> [B,C,HW]
        x = x.flatten(start_dim=2)

        # Key: [B,C,HW] -> [B,HW,C]
        key = self.key_proj(x).transpose(1, 2)

        # Value: [B,C,HW] -> [B,HW,C]
        value = self.value_proj(x).transpose(1, 2)

        # Q: [K,C] -> [B,K,C]
        query = self.prototypes.unsqueeze(0)
        query = query.expand(B, -1, -1)

        # Attention Scores
        # [B,K,C] @ [B,C,HW] -> [B,K,HW]
        scores = torch.matmul(
            query, key.transpose(1, 2)
        )

        # 论文公式采用 sqrt(HW)
        scores = scores / math.sqrt(self.hw)

        # 对空间位置做 Softmax
        attention = torch.softmax(scores, dim=-1)

        # [B,K,HW] @ [B,HW,C] -> [B,K,C]
        proto_feat = torch.matmul(attention, value)

        # [B,K,C] -> [B,C,K]
        proto_feat = proto_feat.transpose(1, 2).contiguous()

        if return_attention:
            return proto_feat, attention

        return proto_feat


# ===== 3. HP + PMAM + Concatenation =====
class CrossGaitFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.hp = HorizontalPooling(num_bins=4)
        self.pmam = PMAM(
            channels=32,
            height=16,
            width=16,
            num_prototypes=2
        )

    def encode(self, feature_map):
        # feature_map: [B,32,16,16]

        # 水平分区特征 [B,32,4]
        local_feat = self.hp(feature_map)

        # 原型特征 [B,32,2]
        proto_feat = self.pmam(feature_map)

        # 拼接得到 [B,32,6]，这里我觉得还是很神的
        fused = torch.cat(
            [local_feat, proto_feat], dim=-1
        )

        return local_feat, proto_feat, fused

    def forward(self, cam_feat, lidar_feat):
        # 共享 HP 和 PMAM
        cam_result = self.encode(cam_feat)
        lidar_result = self.encode(lidar_feat)

        return cam_result, lidar_result


# ===== 4. Tensor Shape 检查 =====
if __name__ == "__main__":
    torch.manual_seed(42)

    model = CrossGaitFeatureFusion()

    # 模拟两个 Encoder + TP 的输出
    cam_feat = torch.randn(4, 32, 16, 16)
    lidar_feat = torch.randn(4, 32, 16, 16)

    cam_result, lidar_result = model(
        cam_feat, lidar_feat
    )

    cam_local, cam_proto, cam_fused = cam_result
    lidar_local, lidar_proto, lidar_fused = lidar_result

    print("Camera local:", cam_local.shape)
    print("Camera prototype:", cam_proto.shape)
    print("Camera fused:", cam_fused.shape)

    print("LiDAR local:", lidar_local.shape)
    print("LiDAR prototype:", lidar_proto.shape)
    print("LiDAR fused:", lidar_fused.shape)

    assert cam_local.shape == (4, 32, 4)
    assert cam_proto.shape == (4, 32, 2)
    assert cam_fused.shape == (4, 32, 6)

    assert lidar_local.shape == (4, 32, 4)
    assert lidar_proto.shape == (4, 32, 2)
    assert lidar_fused.shape == (4, 32, 6)

    # 检查 Attention 权重是否归一化
    _, attention = model.pmam(
        cam_feat, return_attention=True
    )

    print("Attention:", attention.shape)

    assert attention.shape == (4, 2, 256)
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones(4, 2),
        atol=1e-5
    )

    # 检查 Prototype 是否能接收梯度
    loss = cam_fused.square().mean()
    loss = loss + lidar_fused.square().mean()

    loss.backward()

    assert model.pmam.prototypes.grad is not None

    print("Prototype Grad:",
          model.pmam.prototypes.grad.abs().mean().item())

    print("All tests passed!")
