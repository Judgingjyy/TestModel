
import torch
import torch.nn as nn
import torch.nn.functional as F

from ministd_CrossGait_stage2 import CrossGaitStage2


# ==========================================
# 1. BNNeck + Part-wise Classifier
# ==========================================
class PartBNNeck(nn.Module):
    def __init__(self, parts=6, dim=16, num_classes=3):
        super().__init__()

        self.bn = nn.BatchNorm1d(parts * dim)

        # 每个 Part 单独的分类权重
        # [P,D,N]
        self.weight = nn.Parameter(
            torch.empty(parts, dim, num_classes)
        )

        for w in self.weight: #为啥要这样操作那，直接不行嘛
            nn.init.xavier_uniform_(w)

    def forward(self, embedding):
        # [B,D,P]
        B, D, P = embedding.shape

        # Batch Normalization
        x = embedding.flatten(1)  # [B,D*P]
        x = self.bn(x)
        x = x.reshape(B, D, P)

        # L2 Normalization
        x = F.normalize(x, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # [B,D,P] × [P,D,N] -> [B,N,P]
        logits = torch.einsum(
            "bdp,pdn->bnp", x, weight
        )

        return 16.0 * logits


# ==========================================
# 2. Modality-specific Triplet Loss
# ==========================================
def triplet_loss(embedding, labels, margin=0.2):
    # embedding: [B,D,P]
    B, D, P = embedding.shape

    # 分 Part 计算样本两两距离
    # [B,D,P] -> [P,B,D]
    x = embedding.permute(2, 0, 1)

    # [P,B,B]
    dist = torch.cdist(x, x, p=2)

    same = labels[:, None] == labels[None, :]

    eye = torch.eye(
        B, device=labels.device, dtype=torch.bool
    )

    # Positive 不能是自己
    pos_mask = same & ~eye
    neg_mask = ~same

    assert pos_mask.any(dim=1).all()
    assert neg_mask.any(dim=1).all()

    # 枚举 Anchor-Positive-Negative
    # [P,B,B,1] - [P,B,1,B]
    # -> [P,B,B,B]
    loss = F.relu(
        dist.unsqueeze(-1)
        - dist.unsqueeze(-2)
        + margin
    )

    # [B,B,B]：合法三元组
    valid = (
        pos_mask.unsqueeze(-1)
        & neg_mask.unsqueeze(-2)
    )

    valid = valid.unsqueeze(0)

    # 原论文按非零 Loss 的有效三元组归约
    active = valid & (loss > 0)

    loss = loss.masked_fill(~valid, 0)

    return loss.sum() / active.sum().clamp_min(1)


# ==========================================
# 3. Cross-modal Contrastive Loss
# ==========================================
def contrastive_loss(
    cam_emb,
    lidar_emb,
    cam_ids,
    lidar_ids,
    margin=0.2
):
    # [B,D,P] -> [B,D*P]
    cam = cam_emb.flatten(1)
    lidar = lidar_emb.flatten(1)

    # 所有跨模态样本对的平方欧氏距离
    # [B,B]
    dist_sq = torch.cdist(
        cam, lidar, p=2
    ).square()

    # 同身份为 True
    positive = (
        cam_ids[:, None] == lidar_ids[None, :]
    )

    # 同身份：拉近
    pos_loss = dist_sq

    # 不同身份：距离不足 margin 时推远
    neg_loss = F.relu(margin - dist_sq)

    # 根据身份标签选择
    loss = torch.where(
        positive, pos_loss, neg_loss
    )

    return loss.mean()


# ==========================================
# 4. CrossGait + BNNeck
# ==========================================
class CrossGaitTrainModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # 上一课的完整主干
        self.features = CrossGaitStage2()

        # 两个分类分支
        self.cam_bnneck = PartBNNeck(
            num_classes=num_classes
        )

        self.lidar_bnneck = PartBNNeck(
            num_classes=num_classes
        )

    def forward(self, cam_feat, lidar_feat):
        # 已有主干：PMAM、Specific FC、CMFA
        out = self.features(cam_feat, lidar_feat)

        # Shared -> BNNeck -> Logits
        out["cam_logits"] = self.cam_bnneck(
            out["cam_shared"]
        )

        out["lidar_logits"] = self.lidar_bnneck(
            out["lidar_shared"]
        )

        return out


# ==========================================
# 5. 四项 Loss
# ==========================================
def compute_losses(out, cam_ids, lidar_ids):

    # LiDAR 内部 Triplet
    loss_pc = triplet_loss(
        out["lidar_specific"], lidar_ids
    )

    # Camera 内部 Triplet
    loss_sils = triplet_loss(
        out["cam_specific"], cam_ids
    )

    # 两个分类分支 Cross Entropy
    # [B] -> [B,P]
    cam_targets = cam_ids[:, None].expand(
        -1, out["cam_logits"].shape[-1]
    )

    lidar_targets = lidar_ids[:, None].expand(
        -1, out["lidar_logits"].shape[-1]
    )

    ce_cam = F.cross_entropy(
        out["cam_logits"], cam_targets
    )
    ce_lidar = F.cross_entropy(
        out["lidar_logits"], lidar_targets
    )

    loss_ce = (ce_cam + ce_lidar) / 2

    # Shared Embedding 的跨模态对齐
    loss_contrastive = contrastive_loss(
        out["cam_shared"],
        out["lidar_shared"],
        cam_ids,
        lidar_ids
    )

    # CrossGait 总损失
    loss_total = (
        loss_pc
        + loss_sils
        + loss_ce
        + 2.0 * loss_contrastive
    )

    return {
        "pc": loss_pc,
        "sils": loss_sils,
        "ce": loss_ce,
        "contrastive": loss_contrastive,
        "total": loss_total
    }


# ==========================================
# 6. 一次完整训练
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CrossGaitTrainModel(
        num_classes=3
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    # P=3, K=2 -> Batch=6
    labels = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        device=device
    )

    # 模拟 Backbone + TP 后的特征
    # 同身份共享一个基础模板
    base = torch.randn(
        3, 32, 16, 16, device=device
    )
    base = base.repeat_interleave(2, dim=0)

    cam_feat = (
        base + 0.1 * torch.randn_like(base)
    )

    lidar_feat = (
        0.7 * base + 0.2 * torch.randn_like(base)
    )

    model.train()

    # Forward
    out = model(cam_feat, lidar_feat)

    for key, val in out.items():
        print(key, val.shape)

    # 四项 Loss
    losses = compute_losses(
        out, labels, labels
    )

    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")

    # Backward
    optimizer.zero_grad()
    losses["total"].backward()

    # 检查梯度
    grad = model.features.fusion.pmam.prototypes.grad
    assert grad is not None
    assert torch.isfinite(grad).all()

    # 更新参数
    optimizer.step()

    assert torch.isfinite(losses["total"])

    print("One CrossGait training step completed!")
