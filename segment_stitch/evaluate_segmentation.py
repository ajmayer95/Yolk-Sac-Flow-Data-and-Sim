"""Evaluate predicted masks against graph-derived pseudo-masks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .io import dataset_tag
from .metrics import binary_metrics

LOG = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate predicted masks against extracted tile ground-truth masks.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--mask-dir", help="Tile-level ground-truth mask directory. Default: output-dir/extracted_masks/<dataset>.")
    return p


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    np = __import__("numpy")
    tag = dataset_tag(args.data_root)
    out_root = Path(args.output_dir)
    pred_dir = out_root / "predictions" / tag / "binary_masks"
    mask_dir = Path(args.mask_dir) if args.mask_dir else out_root / "extracted_masks" / tag
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pred_path in sorted(pred_dir.glob("tile_*_mask.npy")):
        tile_id = int(pred_path.stem.split("_")[1])
        mask_path = mask_dir / f"tile_{tile_id:04d}_mask.npy"
        if not mask_path.exists():
            LOG.warning("Missing extracted ground-truth mask for tile %s", tile_id)
            continue
        row = {"tile_id": tile_id}
        row.update(binary_metrics(np.load(pred_path), np.load(mask_path)))
        try:
            from skimage.morphology import skeletonize
            row.update({f"skeleton_{k}": v for k, v in binary_metrics(skeletonize(np.load(pred_path) > 0), skeletonize(np.load(mask_path) > 0)).items()})
        except Exception:
            row["skeleton_f1"] = None
        rows.append(row)
    if not rows:
        raise ValueError(f"No predicted masks found in {pred_dir}")
    keys = sorted({k for r in rows for k in r})
    with (metrics_dir / "segmentation_metrics_per_tile.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for key in keys:
        if key == "tile_id":
            continue
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            summary[key] = float(np.mean(vals))
    with (metrics_dir / "segmentation_metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = evaluate(build_arg_parser().parse_args(argv))
    LOG.info("Segmentation summary: %s", summary)


if __name__ == "__main__":
    main()
