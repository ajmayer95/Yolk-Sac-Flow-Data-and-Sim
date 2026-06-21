"""End-to-end Somites27-to-Somites21 segmentation and stitching workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run supervised projection, mask extraction, U-Net, stitching, and evaluation workflow.")
    p.add_argument("--train-root", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-labels", help="Training-set whole-mosaic label/target TIFF. Default: autodetect, then configured stitched mosaic.")
    p.add_argument("--test-labels", help="Test-set whole-mosaic label/target TIFF. Default: autodetect, then configured stitched mosaic.")
    p.add_argument("--positive-label", type=int, help="Specific integer label to treat as foreground. Default: all >0.")
    p.add_argument("--refine-registration", choices=["none", "phase", "ecc", "orb"], default="none")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")
    return p


def run(args: argparse.Namespace) -> None:
    from .evaluate_segmentation import evaluate as eval_seg
    from .evaluate_stitching import evaluate as eval_stitch
    from .extract_masks import extract_dataset_masks, find_mosaic_labels
    from .predict_masks import predict
    from .projections import generate_dataset_projections
    from .stitch.baseline import stitch_dataset
    from .train_unet import train

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_labels = find_mosaic_labels(args.train_root, args.train_labels)
    test_labels = find_mosaic_labels(args.test_root, args.test_labels)
    LOG.info("Using train mosaic labels: %s", train_labels)
    LOG.info("Using test mosaic labels: %s", test_labels)
    LOG.info("Generating projections")
    generate_dataset_projections(args.train_root, out, overwrite=args.overwrite)
    generate_dataset_projections(args.test_root, out, overwrite=args.overwrite)
    LOG.info("Extracting supervised tile masks from mosaic labels")
    train_mask_dir = extract_dataset_masks(args.train_root, out, train_labels, args.positive_label, args.overwrite)
    test_mask_dir = extract_dataset_masks(args.test_root, out, test_labels, args.positive_label, args.overwrite)
    LOG.info("Training U-Net")
    model_path = train(argparse.Namespace(**{
        "train_root": args.train_root,
        "output_dir": str(out),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "device": args.device,
        "mask_dir": str(train_mask_dir),
        "seed": args.seed,
        "pos_weight": None,
    }))
    LOG.info("Predicting test-set masks")
    predict(argparse.Namespace(model=str(model_path), data_root=args.test_root, output_dir=str(out), device=args.device, threshold=0.5))
    LOG.info("Stitching test mosaic with manual positions")
    stitch_dataset(args.test_root, out)
    LOG.info("Evaluating segmentation and stitching")
    eval_seg(argparse.Namespace(data_root=args.test_root, output_dir=str(out), mask_dir=str(test_mask_dir)))
    eval_stitch(argparse.Namespace(data_root=args.test_root, output_dir=str(out), label_path=str(test_labels), positive_label=args.positive_label))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.refine_registration != "none":
        LOG.warning("Registration refinement is scaffolded but not part of the default baseline yet.")
    run(args)


if __name__ == "__main__":
    main()
