
import torch
import torch.nn as nn

from ministd_CrossGait_stage1 import CrossGaitFeatureFusion


# ===== 1. Part-separated FC =====
class PartSeparatedFC(nn.Module):
    def __init__(self, parts, in_dim, out_dim):
        super().__init__()

        # [P,C_in,C_out]
        self.weight = nn.Parameter(
            torch.empty(parts, in_dim, out_dim)
        )

        # 每个 Part 独立初始化
        for p in range(parts):
            nn.init.xavier_uniform_(self.weight[p])

    def forward(self, x):
        # 输入 [B,C_in,P]
        P = self.weight.shape[0]
        assert x.shape[2] == P
        assert x.shape[1] == self.weight.shape[1]

        # [B,C,P] -> [P,B,C]
        x = x.permute(2, 0, 1)

        # Part-wise Matrix Multiplication
        # [P,B,C_in] @ [P,C_in,C_out]
        # -> [P,B,C_out]
        x = torch.matmul(x, self.weight)

        # [P,B,C_out] -> [B,C_out,P]
        x = x.permute(1, 2, 0).contiguous()

        return x


# ===== 2. Cross-modality Feature Adapter =====
class CMFA(nn.Module):
    def __init__(self, parts=6, in_dim=24, out_dim=16):
        super().__init__()

        # 唯一的一组共享参数
        self.shared_fcs = PartSeparatedFC(
            parts=parts,
            in_dim=in_dim,
            out_dim=out_dim
        )

    def forward(self, lidar_specific, cam_specific):

        # 两个模态通过同一个 shared_fcs
        lidar_shared = self.shared_fcs(lidar_specific)
        cam_shared = self.shared_fcs(cam_specific)

        return lidar_shared, cam_shared


# ===== 3. CrossGait 主干 =====
class CrossGaitStage2(nn.Module):
    def __init__(self):
        super().__init__()

        # 共享 PMAM + HP
        self.fusion = CrossGaitFeatureFusion()

        # Modality-specific FCs
        # 两套独立参数
        self.lidar_fcs = PartSeparatedFC(
            parts=6,
            in_dim=32,
            out_dim=24
        )

        self.cam_fcs = PartSeparatedFC(
            parts=6,
            in_dim=32,
            out_dim=24
        )

        # 共享 CMFA
        self.cmfa = CMFA(
            parts=6,
            in_dim=24,
            out_dim=16
        )

    def forward(self, cam_feat, lidar_feat):

        # ===== Step 1: Feature Fusion =====
        cam_result, lidar_result = self.fusion(
            cam_feat, lidar_feat
        )

        # 返回 (local, prototype, fused)
        cam_local, cam_proto, cam_fused = cam_result
        lidar_local, lidar_proto, lidar_fused = lidar_result

        # [B,32,6]

        # ===== Step 2: Modality-specific FCs =====
        cam_specific = self.cam_fcs(cam_fused)
        lidar_specific = self.lidar_fcs(lidar_fused)

        # [B,24,6]

        # ===== Step 3: Shared CMFA =====
        lidar_shared, cam_shared = self.cmfa(
            lidar_specific, cam_specific
        )

        # [B,16,6]

        return {
            "cam_specific": cam_specific,
            "lidar_specific": lidar_specific,
            "cam_shared": cam_shared,
            "lidar_shared": lidar_shared,
            "cam_proto": cam_proto,
            "lidar_proto": lidar_proto
        }


# ===== 4. 验证数据流 =====
if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟两个 Encoder + Temporal Pooling 的输出
    cam_feat = torch.randn(4, 32, 16, 16)
    lidar_feat = torch.randn(4, 32, 16, 16)

    model = CrossGaitStage2()

    output = model(cam_feat, lidar_feat)

    print("Camera Specific:",
          output["cam_specific"].shape)

    print("LiDAR Specific:",
          output["lidar_specific"].shape)

    print("Camera Shared:",
          output["cam_shared"].shape)

    print("LiDAR Shared:",
          output["lidar_shared"].shape)

    assert output["cam_specific"].shape == (4, 24, 6)
    assert output["lidar_specific"].shape == (4, 24, 6)
    assert output["cam_shared"].shape == (4, 16, 6)
    assert output["lidar_shared"].shape == (4, 16, 6)

    # ===== 5. 验证梯度 =====
    # 只用于检查数据流，不是识别训练的 Loss
    loss = (
        output["cam_shared"].square().mean()
        + output["lidar_shared"].square().mean()
    )

    loss.backward()

    # Camera FC 获得梯度
    assert model.cam_fcs.weight.grad is not None

    # LiDAR FC 获得梯度
    assert model.lidar_fcs.weight.grad is not None

    # CMFA 获得来自两个分支的梯度
    assert model.cmfa.shared_fcs.weight.grad is not None

    # PMAM 也能收到梯度
    assert model.fusion.pmam.prototypes.grad is not None

    print("All tests passed!")
