"""Manual-position baseline stitching for projections and masks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..io import dataset_tag, load_tile_positions, resolve_dataset_paths, save_tiff, tile_position_entries
from ..transforms import parse_transforms, transformed_bounds

LOG = logging.getLogger(__name__)


def _load_tile_array(path: Path):
    np = __import__("numpy")
    if path.suffix == ".npy":
        return np.load(path)
    return __import__("tifffile").imread(str(path))


def render_mosaic(tile_arrays: dict[int, object], transforms, blend: str = "average"):
    np = __import__("numpy")
    sample = next(iter(tile_arrays.values()))
    if sample.ndim == 3:
        sample = sample[0]
    h, w = sample.shape[-2:]
    min_y, min_x, max_y, max_x = transformed_bounds(transforms, (h, w))
    out_h = int(np.ceil(max_y - min_y))
    out_w = int(np.ceil(max_x - min_x))
    mosaic = np.zeros((out_h, out_w), dtype="float32")
    weight = np.zeros_like(mosaic)
    for tile_id, arr in tile_arrays.items():
        if tile_id not in transforms:
            continue
        img = arr[0] if getattr(arr, "ndim", 0) == 3 else arr
        t = transforms[tile_id]
        y0 = int(round(t.translate_y - min_y))
        x0 = int(round(t.translate_x - min_x))
        y1 = min(out_h, y0 + img.shape[0])
        x1 = min(out_w, x0 + img.shape[1])
        sy0 = max(0, -y0)
        sx0 = max(0, -x0)
        y0 = max(0, y0)
        x0 = max(0, x0)
        crop = img[sy0: sy0 + (y1 - y0), sx0: sx0 + (x1 - x0)].astype("float32")
        if crop.size == 0:
            continue
        if blend == "max":
            mosaic[y0:y1, x0:x1] = np.maximum(mosaic[y0:y1, x0:x1], crop)
            weight[y0:y1, x0:x1] = np.maximum(weight[y0:y1, x0:x1], crop > 0)
        else:
            mosaic[y0:y1, x0:x1] += crop
            weight[y0:y1, x0:x1] += 1
    if blend != "max":
        mosaic = mosaic / np.maximum(weight, 1)
    return mosaic, weight > 0


def stitch_dataset(data_root: str | Path, output_dir: str | Path, blend: str = "average") -> Path:
    np = __import__("numpy")
    from ..plotting import save_overlay

    paths = resolve_dataset_paths(data_root)
    tag = dataset_tag(data_root)
    out_dir = Path(output_dir) / "stitched" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    transforms = parse_transforms(tile_position_entries(load_tile_positions(paths.tile_positions_path)))
    proj_dir = Path(output_dir) / "projections" / tag
    pred_dir = Path(output_dir) / "predictions" / tag / "binary_masks"
    projection_tiles = {int(p.stem.split("_")[1]): _load_tile_array(p) for p in proj_dir.glob("tile_*_mean.npy")}
    if not projection_tiles:
        projection_tiles = {int(p.stem.split("_")[1]): _load_tile_array(p)[0] for p in proj_dir.glob("tile_*_channels.npy")}
    mask_tiles = {int(p.stem.split("_")[1]): _load_tile_array(p) for p in pred_dir.glob("tile_*_mask.npy")}
    proj_mosaic, coverage = render_mosaic(projection_tiles, transforms, blend=blend)
    save_tiff(out_dir / "stitched_projection_manual.tif", proj_mosaic.astype("float32"))
    if mask_tiles:
        mask_mosaic, _ = render_mosaic(mask_tiles, transforms, blend="max")
        mask_mosaic = (mask_mosaic > 0).astype("uint8")
        save_tiff(out_dir / "stitched_mask_manual.tif", (mask_mosaic * 255).astype("uint8"))
        save_overlay(out_dir / "stitched_overlay_manual.png", proj_mosaic, mask_mosaic, "manual stitched mask")
    np.save(out_dir / "coverage_manual.npy", coverage.astype("uint8"))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stitch Somites tiles using manual tile positions.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--blend", choices=["average", "max"], default="average")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stitch_dataset(**vars(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
