"""Generate metrics summaries and visual QC plots from workflow outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .metrics import binary_metrics, image_metrics

LOG = logging.getLogger(__name__)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _load_image(path: Path):
    if path.suffix == ".npy":
        return __import__("numpy").load(path)
    return __import__("tifffile").imread(str(path))


def _normalize01(arr):
    np = __import__("numpy")
    arr = np.asarray(arr, dtype="float32")
    lo, hi = np.percentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def recompute_segmentation_metrics(output_dir: Path, dataset: str, overwrite: bool = False) -> tuple[Path, Path]:
    np = __import__("numpy")
    per_tile_path = output_dir / "metrics" / "segmentation_metrics_per_tile.csv"
    summary_path = output_dir / "metrics" / "segmentation_metrics_summary.json"
    if per_tile_path.exists() and summary_path.exists() and not overwrite:
        return per_tile_path, summary_path

    pred_dir = output_dir / "predictions" / dataset / "binary_masks"
    gt_dir = output_dir / "extracted_masks" / dataset
    rows = []
    for pred_path in sorted(pred_dir.glob("tile_*_mask.npy")):
        tile_id = int(pred_path.stem.split("_")[1])
        gt_path = gt_dir / f"tile_{tile_id:04d}_mask.npy"
        if not gt_path.exists():
            LOG.warning("Missing extracted mask for tile %s", tile_id)
            continue
        row = {"tile_id": tile_id}
        row.update(binary_metrics(np.load(pred_path), np.load(gt_path)))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No predicted/ground-truth tile mask pairs found under {pred_dir} and {gt_dir}")

    keys = ["tile_id"] + sorted(k for k in rows[0] if k != "tile_id")
    per_tile_path.parent.mkdir(parents=True, exist_ok=True)
    with per_tile_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for key in keys:
        if key == "tile_id":
            continue
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        summary[key] = float(np.mean(vals))
    _write_json(summary_path, summary)
    return per_tile_path, summary_path


def recompute_stitching_metrics(output_dir: Path, dataset: str, reference_path: Path | None, overwrite: bool = False) -> Path:
    np = __import__("numpy")
    summary_path = output_dir / "metrics" / "stitching_metrics_summary.json"
    if summary_path.exists() and not overwrite:
        return summary_path

    stitched_dir = output_dir / "stitched" / dataset
    proj_path = stitched_dir / "stitched_projection_manual.tif"
    mask_path = stitched_dir / "stitched_mask_manual.tif"
    summary = {}
    if reference_path and reference_path.exists() and proj_path.exists():
        reference = _load_image(reference_path)
        summary.update({
            "projection_prediction_path": str(proj_path),
            "projection_ground_truth_path": str(reference_path),
        })
        summary.update({f"projection_{k}": v for k, v in image_metrics(_load_image(proj_path), reference).items()})
    if mask_path.exists():
        mask = np.asarray(_load_image(mask_path)) > 0
        summary["mask_coverage_fraction"] = float(mask.mean())
        if reference_path and reference_path.exists():
            reference_mask = np.asarray(_load_image(reference_path)) > 0
            h = min(mask.shape[0], reference_mask.shape[0])
            w = min(mask.shape[1], reference_mask.shape[1])
            summary.update({
                "mask_prediction_path": str(mask_path),
                "mask_ground_truth_path": str(reference_path),
            })
            summary.update({f"mosaic_mask_{k}": v for k, v in binary_metrics(mask[:h, :w], reference_mask[:h, :w]).items()})
            try:
                from skimage.metrics import structural_similarity

                summary["mosaic_mask_ssim"] = float(
                    structural_similarity(mask[:h, :w].astype("float32"), reference_mask[:h, :w].astype("float32"), data_range=1.0)
                )
            except Exception:
                summary["mosaic_mask_ssim"] = None
    _write_json(summary_path, summary)
    return summary_path


def plot_training_loss(output_dir: Path, qc_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    rows = _read_csv(output_dir / "metrics" / "training_metrics.csv")
    if not rows:
        return None
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    path = qc_dir / "training_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("U-Net Training Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_per_tile_metrics(output_dir: Path, qc_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    rows = _read_csv(output_dir / "metrics" / "segmentation_metrics_per_tile.csv")
    if not rows:
        return None
    tile_ids = [int(r["tile_id"]) for r in rows]
    metrics = [m for m in ("dice", "iou", "precision", "recall") if m in rows[0]]
    path = qc_dir / "segmentation_metrics_per_tile.png"
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 2.6 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        vals = [float(r[metric]) for r in rows]
        ax.bar(range(len(tile_ids)), vals)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(len(tile_ids)))
    axes[-1].set_xticklabels(tile_ids, rotation=90)
    axes[-1].set_xlabel("tile id")
    fig.suptitle("Per-Tile Segmentation Metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_mask_coverage(output_dir: Path, dataset: str, qc_dir: Path) -> Path | None:
    np = __import__("numpy")
    import matplotlib.pyplot as plt

    gt_dir = output_dir / "extracted_masks" / dataset
    pred_dir = output_dir / "predictions" / dataset / "binary_masks"
    tile_ids, gt_cov, pred_cov = [], [], []
    for gt_path in sorted(gt_dir.glob("tile_*_mask.npy")):
        tile_id = int(gt_path.stem.split("_")[1])
        pred_path = pred_dir / f"tile_{tile_id:04d}_mask.npy"
        if not pred_path.exists():
            continue
        tile_ids.append(tile_id)
        gt_cov.append(float((np.load(gt_path) > 0).mean()))
        pred_cov.append(float((np.load(pred_path) > 0).mean()))
    if not tile_ids:
        return None
    path = qc_dir / "mask_coverage_by_tile.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(tile_ids))
    ax.bar(x - 0.2, gt_cov, width=0.4, label="extracted target")
    ax.bar(x + 0.2, pred_cov, width=0.4, label="prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(tile_ids, rotation=90)
    ax.set_ylabel("foreground fraction")
    ax.set_xlabel("tile id")
    ax.set_title("Mask Coverage By Tile")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_overlay_montage(output_dir: Path, dataset: str, qc_dir: Path, max_tiles: int = 12) -> Path | None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    overlay_paths = sorted((output_dir / "predictions" / dataset / "overlays").glob("tile_*_overlay.png"))[:max_tiles]
    if not overlay_paths:
        return None
    cols = min(4, len(overlay_paths))
    rows = (len(overlay_paths) + cols - 1) // cols
    path = qc_dir / "segmentation_overlay_montage.png"
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = __import__("numpy").atleast_1d(axes).ravel()
    for ax, overlay_path in zip(axes, overlay_paths):
        ax.imshow(mpimg.imread(overlay_path))
        ax.set_title(overlay_path.stem.replace("_overlay", ""))
        ax.axis("off")
    for ax in axes[len(overlay_paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_stitched_qc(output_dir: Path, dataset: str, qc_dir: Path, reference_path: Path | None = None) -> Path | None:
    np = __import__("numpy")
    import matplotlib.pyplot as plt

    stitched_dir = output_dir / "stitched" / dataset
    proj_path = stitched_dir / "stitched_projection_manual.tif"
    mask_path = stitched_dir / "stitched_mask_manual.tif"
    if not proj_path.exists() and not mask_path.exists():
        return None
    panels = []
    if proj_path.exists():
        panels.append((
            "Prediction: stitched projection\n(manual tile placement)",
            _normalize01(_load_image(proj_path)),
            "gray",
        ))
    if reference_path and reference_path.exists():
        reference = _load_image(reference_path)
        panels.append((
            "Reference: provided stitched mosaic\n(stitched_linear.tif)",
            _normalize01(reference),
            "gray",
        ))
    if mask_path.exists():
        panels.append((
            "Prediction: stitched mask\n(model tile masks + manual placement)",
            np.asarray(_load_image(mask_path)) > 0,
            "gray",
        ))
    if reference_path and reference_path.exists():
        panels.append((
            "Ground truth/target for mask metrics\n(binarized reference mosaic)",
            np.asarray(_load_image(reference_path)) > 0,
            "gray",
        ))
    path = qc_dir / "stitched_qc_panel.png"
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.4))
    axes = np.atleast_1d(axes)
    for ax, (title, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Stitched Mosaic QC: prediction vs reference/target", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_markdown_report(output_dir: Path, dataset: str, qc_dir: Path, generated: list[Path]) -> Path:
    metrics_dir = output_dir / "metrics"
    report_path = qc_dir / "QC_REPORT.md"
    seg_summary = {}
    stitch_summary = {}
    train_summary = {}
    for path, dest in [
        (metrics_dir / "segmentation_metrics_summary.json", seg_summary),
        (metrics_dir / "stitching_metrics_summary.json", stitch_summary),
        (metrics_dir / "training_summary.json", train_summary),
    ]:
        if path.exists():
            dest.update(json.loads(path.read_text(encoding="utf-8")))
    lines = [
        f"# Segment Stitch QC Report: {dataset}",
        "",
        "## Key Metrics",
        "",
        f"- Best validation loss: `{train_summary.get('best_val_loss', 'NA')}`",
        f"- Tile Dice: `{seg_summary.get('dice', 'NA')}`",
        f"- Tile IoU: `{seg_summary.get('iou', 'NA')}`",
        f"- Stitched mask Dice: `{stitch_summary.get('mosaic_mask_dice', 'NA')}`",
        f"- Stitched mask IoU: `{stitch_summary.get('mosaic_mask_iou', 'NA')}`",
        f"- Stitched projection SSIM: `{stitch_summary.get('projection_ssim', stitch_summary.get('ssim', 'NA'))}`",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- [{p.name}]({p.relative_to(qc_dir)})" for p in generated)
    lines.extend([
        "",
        "## Source Files",
        "",
        f"- Report metrics: `{qc_dir / 'metrics'}`",
        f"- Work metrics: `{metrics_dir}`",
        f"- Predictions: `{output_dir / 'predictions' / dataset}`",
        f"- Stitched outputs: `{output_dir / 'stitched' / dataset}`",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_report(
    output_dir: str | Path = "segment_stitch/work",
    report_dir: str | Path = "segment_stitch/outputs",
    dataset: str = "Somites21",
    reference_mosaic: str | Path | None = None,
    overwrite_metrics: bool = False,
    max_montage_tiles: int = 12,
) -> Path:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir) / dataset
    metrics_out = report_dir / "metrics"
    visuals_out = report_dir / "visuals"
    metrics_out.mkdir(parents=True, exist_ok=True)
    visuals_out.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_mosaic).expanduser().resolve() if reference_mosaic else None

    recompute_segmentation_metrics(output_dir, dataset, overwrite=overwrite_metrics)
    recompute_stitching_metrics(output_dir, dataset, reference_path, overwrite=overwrite_metrics)

    for metrics_file in (output_dir / "metrics").glob("*"):
        if metrics_file.is_file() and metrics_file.suffix.lower() in {".csv", ".json"}:
            (metrics_out / metrics_file.name).write_bytes(metrics_file.read_bytes())

    generated = []
    for maybe_path in [
        plot_training_loss(output_dir, visuals_out),
        plot_per_tile_metrics(output_dir, visuals_out),
        plot_mask_coverage(output_dir, dataset, visuals_out),
        make_overlay_montage(output_dir, dataset, visuals_out, max_tiles=max_montage_tiles),
        make_stitched_qc(output_dir, dataset, visuals_out, reference_path=reference_path),
    ]:
        if maybe_path is not None:
            generated.append(maybe_path)
    report_path = write_markdown_report(output_dir, dataset, report_dir, generated)
    LOG.info("QC report written to %s", report_path)
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate metrics summaries and visual QC plots from segment_stitch outputs.")
    p.add_argument("--output-dir", default="segment_stitch/work", help="Workflow artifact directory to read.")
    p.add_argument("--report-dir", default="segment_stitch/outputs", help="Metrics/visuals report directory to write.")
    p.add_argument("--dataset", default="Somites21")
    p.add_argument("--reference-mosaic", help="Optional stitched/full mosaic TIFF for stitched projection comparison.")
    p.add_argument("--overwrite-metrics", action="store_true")
    p.add_argument("--max-montage-tiles", type=int, default=12)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    generate_report(args.output_dir, args.report_dir, args.dataset, args.reference_mosaic, args.overwrite_metrics, args.max_montage_tiles)


if __name__ == "__main__":
    main()
