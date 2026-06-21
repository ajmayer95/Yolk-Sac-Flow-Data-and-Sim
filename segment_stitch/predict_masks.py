"""Run U-Net inference for Somites tile projections."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .io import dataset_tag, save_tiff

LOG = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict vessel masks with a trained U-Net.")
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def predict(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    from .datasets import ProjectionMaskDataset
    from .models import UNet
    from .plotting import save_segmentation_qc

    tag = dataset_tag(args.data_root)
    out_root = Path(args.output_dir)
    pred_dir = out_root / "predictions" / tag
    prob_dir = pred_dir / "prob_masks"
    bin_dir = pred_dir / "binary_masks"
    overlay_dir = pred_dir / "overlays"
    for d in (prob_dir, bin_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)
    ds = ProjectionMaskDataset(out_root / "projections" / tag)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()
    with torch.no_grad():
        for item in ds:
            x = item["image"].unsqueeze(0).to(device)
            tile_id = item["tile_id"]
            prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy().astype("float32")
            binary = (prob >= args.threshold).astype("uint8")
            np.save(prob_dir / f"tile_{tile_id:04d}_prob.npy", prob)
            np.save(bin_dir / f"tile_{tile_id:04d}_mask.npy", binary)
            save_tiff(prob_dir / f"tile_{tile_id:04d}_prob.tif", prob)
            save_tiff(bin_dir / f"tile_{tile_id:04d}_mask.tif", (binary * 255).astype("uint8"))
            save_segmentation_qc(overlay_dir / f"tile_{tile_id:04d}_overlay.png", item["image"].numpy(), prob=prob, binary=binary)
    return pred_dir


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    predict(args)


if __name__ == "__main__":
    main()
