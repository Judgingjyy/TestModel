
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sustech1k_dataset import (
    SUSTech1KDataset,
    PKBatchSampler
)
from crossgait_full import CrossGait
from ministd_CrossGait_stage3 import compute_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--split", default="SUSTech1K.json"
    )
    parser.add_argument(
        "--steps", type=int, default=20
    )
    parser.add_argument(
        "--resume", default=None
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    with open(args.split, "r") as f:
        partition = json.load(f)

    num_classes = len(partition["TRAIN_SET"])

    dataset = SUSTech1KDataset(
        root=args.root,
        split_json=args.split,
        split="TRAIN_SET",
        training=True
    )

    sampler = PKBatchSampler(
        dataset,
        p=3,
        k=2,
        steps=args.steps
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,  # Windows 先用 0
        pin_memory=device.type == "cuda"
    )

    model = CrossGait(
        num_classes=num_classes,
        lidar_channels=3
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    start_step = 0

    if args.resume:
        ckpt = torch.load(
            args.resume,
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

    model.train()

    for step, batch in enumerate(
        loader, start=start_step + 1
    ):
        cam, lidar, labels = batch

        cam = cam.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        # Forward
        output = model(cam, lidar)

        # 四项 Loss
        losses = compute_losses(
            output,
            cam_ids=labels,
            lidar_ids=labels
        )

        loss = losses["total"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=5.0
        )

        optimizer.step()

        if step == start_step + 1 or step % 5 == 0:
            print(
                f"Step {step} | "
                f"Total {loss.item():.4f} | "
                f"PC {losses['pc'].item():.4f} | "
                f"Sils {losses['sils'].item():.4f} | "
                f"CE {losses['ce'].item():.4f} | "
                f"Contrast {losses['contrastive'].item():.4f}"
            )

    Path("checkpoints").mkdir(exist_ok=True)

    checkpoint_path = "checkpoints/crossgait_sustech1k.pt"

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": start_step + args.steps,
        "num_classes": num_classes
    }, checkpoint_path)

    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()
