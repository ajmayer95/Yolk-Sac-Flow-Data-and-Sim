"""Evaluate stitched mosaics against the provided stitched_linear.tif reference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .extract_masks import find_mosaic_labels, mosaic_to_binary
from .io import dataset_tag, load_tiff, resolve_dataset_paths
from .metrics import binary_metrics, image_metrics

LOG = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate stitched outputs against image and segmentation mosaic references.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--label-path", help="Whole-mosaic segmentation TIFF for mosaic-level mask evaluation.")
    p.add_argument("--positive-label", type=int, help="Specific integer label to treat as foreground. Default: all >0.")
    return p


def boundary_continuity(mask) -> dict[str, float | None]:
    np = __import__("numpy")
    if mask is None:
        return {"mask_coverage_fraction": None, "boundary_pixel_fraction": None}
    m = np.asarray(mask) > 0
    boundary = np.zeros_like(m, dtype=bool)
    step_y = 320
    step_x = 352
    boundary[::step_y, :] = True
    boundary[:, ::step_x] = True
    return {
        "mask_coverage_fraction": float(m.mean()),
        "boundary_pixel_fraction": float((m & boundary).sum() / max(1, m.sum())),
    }


def evaluate(args: argparse.Namespace) -> dict[str, float | None]:
    np = __import__("numpy")
    tag = dataset_tag(args.data_root)
    paths = resolve_dataset_paths(args.data_root)
    out_root = Path(args.output_dir)
    stitch_dir = out_root / "stitched" / tag
    pred = load_tiff(stitch_dir / "stitched_projection_manual.tif")
    ref = load_tiff(paths.stitched_path)
    summary = image_metrics(pred, ref)
    mask_path = stitch_dir / "stitched_mask_manual.tif"
    mask = load_tiff(mask_path) if mask_path.exists() else None
    summary.update(boundary_continuity(mask))
    if mask is not None:
        labels_path = find_mosaic_labels(args.data_root, getattr(args, "label_path", None))
        label_mask = mosaic_to_binary(load_tiff(labels_path), positive_label=getattr(args, "positive_label", None))
        pred_mask = np.asarray(mask) > 0
        h = min(pred_mask.shape[0], label_mask.shape[0])
        w = min(pred_mask.shape[1], label_mask.shape[1])
        summary.update({f"mosaic_mask_{k}": v for k, v in binary_metrics(pred_mask[:h, :w], label_mask[:h, :w]).items()})
        try:
            from skimage.metrics import structural_similarity
            summary["mosaic_mask_ssim"] = float(
                structural_similarity(pred_mask[:h, :w].astype("float32"), label_mask[:h, :w].astype("float32"), data_range=1.0)
            )
        except Exception:
            summary["mosaic_mask_ssim"] = None
    summary.update({
        "translation_error_mean": 0.0,
        "scale_error_mean": 0.0,
        "transform_reference": "manual_positions",
    })
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "stitching_metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOG.info("Stitching summary: %s", evaluate(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
