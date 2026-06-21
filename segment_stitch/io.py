"""Input/output helpers for Somites tile video datasets."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)
LOC_RE = re.compile(r"(?:loc|vid|tile)[ _-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    emb_dir: Path
    config_path: Path
    videos_dir: Path
    analyzed_dir: Path
    tile_positions_path: Path
    stitched_path: Path
    graph_canonical_path: Path | None
    graph_analyzed_path: Path | None

    @property
    def name(self) -> str:
        return self.root.name.replace("_demo", "")


def require_dependency(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{name}'. Install the scientific Python "
            "environment for segment_stitch before running this command."
        ) from exc


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_dataset_paths(root: str | Path, embryo: str = "emb1") -> DatasetPaths:
    root = Path(root).expanduser().resolve()
    emb_dir = root / embryo
    config_path = emb_dir / "config.json"
    config = load_json(config_path)
    videos_dir = emb_dir / config.get("video_dir", "videos")
    analyzed_dir = emb_dir / "analyzed"
    tile_positions_path = emb_dir / config.get("tile_positions", "analyzed/tile_positions_manual.json")
    stitched_path = emb_dir / config.get("mosaic_tiff", "analyzed/stitched_linear.tif")
    graph_cfg = config.get("mosaic_graph")
    graph_analyzed = emb_dir / graph_cfg if graph_cfg else analyzed_dir / "mosaic_graph_analyzed.gpickle"
    graph_canonical = analyzed_dir / "mosaic_graph_canonical.gpickle"
    paths = DatasetPaths(
        root=root,
        emb_dir=emb_dir,
        config_path=config_path,
        videos_dir=videos_dir,
        analyzed_dir=analyzed_dir,
        tile_positions_path=tile_positions_path,
        stitched_path=stitched_path,
        graph_canonical_path=graph_canonical if graph_canonical.exists() else None,
        graph_analyzed_path=graph_analyzed if graph_analyzed.exists() else None,
    )
    for label, path in {
        "videos directory": videos_dir,
        "manual tile positions": tile_positions_path,
        "stitched TIFF": stitched_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} at {path}")
    if not paths.graph_analyzed_path and not paths.graph_canonical_path:
        raise FileNotFoundError(f"No mosaic graph found under {analyzed_dir}")
    return paths


def parse_tile_id(path_or_name: str | Path) -> int | None:
    name = Path(path_or_name).stem
    match = LOC_RE.search(name)
    if match:
        return int(match.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def find_video_files(videos_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(videos_dir.glob("*.tif*")):
        tile_id = parse_tile_id(path)
        if tile_id is None:
            LOG.warning("Could not parse tile id from %s; skipping", path.name)
            continue
        files[tile_id] = path
    if not files:
        raise FileNotFoundError(f"No TIFF videos found in {videos_dir}")
    return files


def load_tile_positions(path: Path) -> dict[str, Any]:
    data = load_json(path)
    tiles = data.get("tiles", data)
    if not isinstance(tiles, dict):
        raise ValueError(f"Tile positions JSON has no tile mapping: {path}")
    return data


def tile_position_entries(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tiles = data.get("tiles", data)
    out: dict[int, dict[str, Any]] = {}
    for key, value in tiles.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            LOG.warning("Skipping non-integer tile key %r in manual positions", key)
    return out


def usable_tile_ids(paths: DatasetPaths) -> list[int]:
    videos = find_video_files(paths.videos_dir)
    positions = tile_position_entries(load_tile_positions(paths.tile_positions_path))
    ids = sorted(set(videos) & set(positions))
    if not ids:
        raise ValueError("No overlap between raw video IDs and manual tile-position IDs")
    missing_pos = sorted(set(videos) - set(positions))
    missing_video = sorted(set(positions) - set(videos))
    if missing_pos:
        LOG.warning("Videos missing manual positions: %s", missing_pos)
    if missing_video:
        LOG.info("Manual positions without raw videos: %s", missing_video)
    return ids


def load_tiff(path: Path):
    tifffile = require_dependency("tifffile")
    return tifffile.imread(str(path))


def save_tiff(path: Path, array) -> None:
    tifffile = require_dependency("tifffile")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array)


def load_graph(path: Path | None = None):
    if path is None:
        raise FileNotFoundError("No graph path was provided")
    nx = require_dependency("networkx")
    try:
        return nx.read_gpickle(path)
    except AttributeError:
        import pickle

        with path.open("rb") as fh:
            return pickle.load(fh)


def dataset_tag(root: str | Path) -> str:
    return Path(root).name.replace("_demo", "")
