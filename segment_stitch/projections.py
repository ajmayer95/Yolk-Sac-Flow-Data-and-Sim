"""Temporal projection generation for tile-wise TIFF stacks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .io import dataset_tag, find_video_files, load_tiff, resolve_dataset_paths, save_tiff

LOG = logging.getLogger(__name__)


def robust_normalize(image, p_low: float = 1, p_high: float = 99, eps: float = 1e-6):
    np = __import__("numpy")
    arr = image.astype("float32", copy=False)
    lo, hi = np.percentile(arr, [p_low, p_high])
    arr = (arr - lo) / (hi - lo + eps)
    return np.clip(arr, 0.0, 1.0).astype("float32")


def projection_channels(projections: dict[str, object]):
    np = __import__("numpy")
    return np.stack(
        [robust_normalize(projections["mean"]), robust_normalize(projections["max"]), robust_normalize(projections["std"])],
        axis=0,
    )


def compute_projections(stack):
    np = __import__("numpy")
    if stack.ndim != 3:
        raise ValueError(f"Expected TIFF stack [T,H,W], got shape {stack.shape}")
    return {
        "mean": stack.mean(axis=0).astype("float32"),
        "max": stack.max(axis=0).astype("float32"),
        "std": stack.std(axis=0).astype("float32"),
        "median": np.median(stack, axis=0).astype("float32"),
    }


def generate_dataset_projections(data_root: str | Path, output_dir: str | Path, save_tif: bool = True, overwrite: bool = False) -> Path:
    np = __import__("numpy")
    paths = resolve_dataset_paths(data_root)
    out_dir = Path(output_dir) / "projections" / dataset_tag(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile_id, video_path in find_video_files(paths.videos_dir).items():
        channels_path = out_dir / f"tile_{tile_id:04d}_channels.npy"
        if channels_path.exists() and not overwrite:
            continue
        LOG.info("Computing projections for tile %s from %s", tile_id, video_path.name)
        projections = compute_projections(load_tiff(video_path))
        for name, arr in projections.items():
            np.save(out_dir / f"tile_{tile_id:04d}_{name}.npy", arr)
            if save_tif:
                save_tiff(out_dir / f"tile_{tile_id:04d}_{name}.tif", arr.astype("float32"))
        np.save(channels_path, projection_channels(projections))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate temporal projections for Somites tile videos.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--no-tif", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    generate_dataset_projections(args.data_root, args.output_dir, save_tif=not args.no_tif, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
