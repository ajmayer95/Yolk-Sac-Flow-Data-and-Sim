"""Extract supervised tile masks from whole-mosaic segmentation labels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .io import dataset_tag, find_video_files, load_tiff, load_tile_positions, resolve_dataset_paths, save_tiff, tile_position_entries
from .transforms import parse_transforms, transformed_bounds

LOG = logging.getLogger(__name__)

COMMON_LABEL_NAMES = (
    "mosaic_segmentation.tif",
    "mosaic_segmentation_labels.tif",
    "stitched_segmentation.tif",
    "stitched_labels.tif",
    "segmentation_labels.tif",
    "labels.tif",
    "mask.tif",
    "mosaic_mask.tif",
)


def _is_placeholder_path(path: str | Path) -> bool:
    text = str(path)
    return text.startswith("/path/to/") or "<" in text or ">" in text


def find_mosaic_labels(data_root: str | Path, label_path: str | Path | None = None) -> Path:
    """Resolve a whole-mosaic label image.

    The current demo folders do not advertise a label path in `config.json`, so
    callers can pass one explicitly. If omitted, common filenames in
    `emb1/analyzed/` are tried.
    """
    if label_path and not _is_placeholder_path(label_path):
        path = Path(label_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Mosaic label file does not exist: {path}")
        return path
    if label_path and _is_placeholder_path(label_path):
        LOG.warning("Ignoring placeholder mosaic label path: %s", label_path)
    paths = resolve_dataset_paths(data_root)
    for name in COMMON_LABEL_NAMES:
        candidate = paths.analyzed_dir / name
        if candidate.exists():
            return candidate
    if paths.stitched_path.exists():
        LOG.warning(
            "Using configured stitched mosaic %s as the full-mosaic label source. "
            "This is only appropriate if that TIFF is the intended label/target mosaic.",
            paths.stitched_path,
        )
        return paths.stitched_path
    raise FileNotFoundError(
        "Could not find a whole-mosaic segmentation label TIFF. Pass --label-path "
        "or place one of these files in emb1/analyzed/: "
        + ", ".join(COMMON_LABEL_NAMES)
    )


def mosaic_to_binary(labels, positive_label: int | None = None):
    np = __import__("numpy")
    arr = np.asarray(labels)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mosaic labels, got shape {arr.shape}")
    if positive_label is None:
        return (arr > 0).astype("uint8")
    return (arr == positive_label).astype("uint8")


def sample_label_to_tile(label_mask, transform, tile_shape: tuple[int, int], offset_yx: tuple[float, float]):
    """Sample a mosaic label mask into one raw tile coordinate frame."""
    np = __import__("numpy")
    h, w = tile_shape
    yy, xx = np.indices((h, w), dtype="float32")
    m = np.asarray(transform.matrix_yx, dtype="float32")
    global_y = m[0, 0] * yy + m[0, 1] * xx + m[0, 2] + offset_yx[0]
    global_x = m[1, 0] * yy + m[1, 1] * xx + m[1, 2] + offset_yx[1]
    row = np.rint(global_y).astype("int64")
    col = np.rint(global_x).astype("int64")
    valid = (row >= 0) & (row < label_mask.shape[0]) & (col >= 0) & (col < label_mask.shape[1])
    out = np.zeros((h, w), dtype="uint8")
    out[valid] = label_mask[row[valid], col[valid]].astype("uint8")
    return out


def extract_dataset_masks(
    data_root: str | Path,
    output_dir: str | Path,
    label_path: str | Path | None = None,
    positive_label: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract tile-level supervised masks from the labeled mosaic."""
    np = __import__("numpy")
    paths = resolve_dataset_paths(data_root)
    tag = dataset_tag(data_root)
    out_dir = Path(output_dir) / "extracted_masks" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = find_mosaic_labels(data_root, label_path)
    label_mask = mosaic_to_binary(load_tiff(labels_path), positive_label=positive_label)
    LOG.info("Loaded mosaic labels from %s with shape %s", labels_path, label_mask.shape)

    videos = find_video_files(paths.videos_dir)
    transforms = parse_transforms(tile_position_entries(load_tile_positions(paths.tile_positions_path)))
    tile_ids = sorted(set(videos) & set(transforms))
    if not tile_ids:
        raise ValueError("No tiles have both a raw video and a usable manual transform")
    first_stack = __import__("tifffile").imread(str(videos[tile_ids[0]]))
    tile_shape = tuple(first_stack.shape[-2:])
    min_y, min_x, _, _ = transformed_bounds({tile_id: transforms[tile_id] for tile_id in tile_ids}, tile_shape)
    offset_yx = (-min_y, -min_x)
    LOG.info("Using mosaic label offset y/x = %.2f, %.2f", offset_yx[0], offset_yx[1])

    for tile_id in tile_ids:
        out = out_dir / f"tile_{tile_id:04d}_mask.npy"
        if out.exists() and not overwrite:
            continue
        mask = sample_label_to_tile(label_mask, transforms[tile_id], tile_shape, offset_yx)
        np.save(out, mask)
        save_tiff(out_dir / f"tile_{tile_id:04d}_mask.tif", (mask * 255).astype("uint8"))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract tile masks from a whole-mosaic label/target image.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--label-path", help="Whole-mosaic label/target TIFF. If omitted, common names and the configured stitched mosaic are tried.")
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--positive-label", type=int, help="Specific integer label to treat as foreground. Default: all >0.")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    extract_dataset_masks(args.data_root, args.output_dir, args.label_path, args.positive_label, args.overwrite)


if __name__ == "__main__":
    main()
