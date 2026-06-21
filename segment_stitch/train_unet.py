"""Train a lightweight U-Net on Somites temporal projections and pseudo-masks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

from .io import dataset_tag

LOG = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train U-Net segmentation on extracted tile ground-truth masks.")
    p.add_argument("--train-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-dir", help="Tile-level ground-truth mask directory. Default: output-dir/extracted_masks/<dataset>.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pos-weight", type=float)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def train(args: argparse.Namespace) -> Path:
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader, random_split

    from .datasets import ProjectionMaskDataset
    from .losses import DiceLoss
    from .models import UNet

    set_seed(args.seed)
    tag = dataset_tag(args.train_root)
    out_root = Path(args.output_dir)
    ckpt_dir = out_root / "checkpoints"
    metrics_dir = out_root / "metrics"
    qc_dir = out_root / "qc" / "segmentation"
    for d in (ckpt_dir, metrics_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)
    proj_dir = out_root / "projections" / tag
    mask_dir = Path(args.mask_dir) if args.mask_dir else out_root / "extracted_masks" / tag
    ds = ProjectionMaskDataset(proj_dir, mask_dir)
    if len(ds) < 2:
        raise ValueError(f"Need at least two training tiles, found {len(ds)} in {proj_dir}")
    val_n = max(1, int(round(0.15 * len(ds))))
    train_n = len(ds) - val_n
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=generator)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    LOG.info("Selected device: %s", device)
    model = UNet().to(device)
    pos_weight = torch.tensor([args.pos_weight], device=device) if args.pos_weight else None
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    rows: list[dict[str, float | int]] = []
    best = float("inf")
    best_path = ckpt_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * x.shape[0]
        train_loss /= train_n
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["mask"].to(device)
                logits = model(x)
                val_loss += float((bce(logits, y) + dice(logits, y)).detach().cpu()) * x.shape[0]
        val_loss /= val_n
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        LOG.info("epoch=%03d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)
        state = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(state, ckpt_dir / "last_model.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, best_path)
    with (metrics_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
    with (metrics_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"best_val_loss": best, "best_model": str(best_path)}, fh, indent=2)
    plt.figure(figsize=(6, 4))
    plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label="train")
    plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "loss_curve.png", dpi=150)
    plt.close()
    return best_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
