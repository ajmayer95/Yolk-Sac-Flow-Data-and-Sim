"""Pseudo-mask bootstrapping from mosaic graphs or classical image processing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from .io import dataset_tag, find_video_files, load_graph, load_tile_positions, resolve_dataset_paths, save_tiff, tile_position_entries
from .projections import robust_normalize
from .transforms import parse_transforms, transformed_bounds

LOG = logging.getLogger(__name__)
try:
    from skimage.draw import line as _skimage_line
except ImportError:
    _skimage_line = None


def _first_present(data, keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_yx(point) -> tuple[float, float] | None:
    """Convert graph coordinates to y/x order.

    Graph nodes use explicit x/y attributes, and edge path arrays in these
    datasets follow that same x,y order. Masks and images use row/column
    coordinates, so rasterization needs y,x.
    """
    if len(point) < 2:
        return None
    x, y = float(point[0]), float(point[1])
    return y, x


def _edge_points(graph, u, v, data) -> list[tuple[float, float]]:
    path = _first_present(data, ("path", "paths", "coords", "points"))
    if path is not None:
        pts = []
        for p in path:
            pt = _as_yx(p)
            if pt is not None:
                pts.append(pt)
        if len(pts) >= 2:
            return pts
    nu, nv = graph.nodes[u], graph.nodes[v]
    return [(float(nu.get("y", nu.get("row", 0))), float(nu.get("x", nu.get("col", 0)))),
            (float(nv.get("y", nv.get("row", 0))), float(nv.get("x", nv.get("col", 0))))]


def _points_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    ys = [p[0] for p in points]
    xs = [p[1] for p in points]
    return min(ys), min(xs), max(ys), max(xs)


def _bbox_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 0.0) -> bool:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    return not (ay1 < by0 - pad or ay0 > by1 + pad or ax1 < bx0 - pad or ax0 > bx1 + pad)


def _radius(data, default: int, max_width: int = 4) -> int:
    for key in ("radius", "radius_px", "r", "mean_radius"):
        if key in data:
            try:
                return min(max_width, max(1, int(round(float(data[key])))))
            except Exception:
                pass
    return default


def prepare_graph_edges(graph, default_width: int = 3, max_width: int = 4) -> list[dict[str, object]]:
    """Extract graph edge paths once so per-tile mask generation is faster."""
    edges = []
    for u, v, data in graph.edges(data=True):
        points = _edge_points(graph, u, v, data)
        if len(points) < 2:
            continue
        radius = _radius(data, default_width, max_width=max_width)
        edges.append({"points": points, "bbox": _points_bbox(points), "radius": radius})
    return edges


def rasterize_polyline(mask, points: Iterable[tuple[float, float]], width: int) -> None:
    np = __import__("numpy")
    pts = list(points)
    if len(pts) < 2:
        return
    h, w = mask.shape
    width = max(1, int(width))
    if _skimage_line is not None:
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            rr, cc = _skimage_line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
            for dy in range(-width, width + 1):
                rry = rr + dy
                row_ok = (rry >= 0) & (rry < h)
                if not row_ok.any():
                    continue
                for dx in range(-width, width + 1):
                    ccx = cc + dx
                    ok = row_ok & (ccx >= 0) & (ccx < w)
                    if ok.any():
                        mask[rry[ok], ccx[ok]] = True
    else:
        for y, x in pts:
            yy, xx = int(round(y)), int(round(x))
            y0, y1 = max(0, yy - width), min(h, yy + width + 1)
            x0, x1 = max(0, xx - width), min(w, xx + width + 1)
            mask[y0:y1, x0:x1] = True


def graph_mask_for_tile(
    graph,
    transform,
    tile_shape: tuple[int, int],
    default_width: int = 3,
    graph_offset_yx: tuple[float, float] = (0.0, 0.0),
    graph_edges: list[dict[str, object]] | None = None,
):
    np = __import__("numpy")
    mask = np.zeros(tile_shape, dtype=bool)
    h, w = tile_shape
    inv = np.linalg.inv(np.asarray(transform.matrix_yx, dtype=float))
    offset_y, offset_x = graph_offset_yx
    corners = [transform.local_to_global(y, x) for y, x in ((0, 0), (h, 0), (0, w), (h, w))]
    graph_corners = [(y + offset_y, x + offset_x) for y, x in corners]
    tile_bbox = _points_bbox(graph_corners)
    edges = graph_edges if graph_edges is not None else prepare_graph_edges(graph, default_width=default_width)
    for edge in edges:
        points = edge["points"]
        radius = int(edge["radius"])
        if not _bbox_overlaps(edge["bbox"], tile_bbox, pad=radius + 10):
            continue
        local = []
        for gy, gx in points:
            ly, lx, _ = inv @ np.asarray([gy - offset_y, gx - offset_x, 1.0], dtype=float)
            local.append((float(ly), float(lx)))
        if not any(-20 <= y < h + 20 and -20 <= x < w + 20 for y, x in local):
            continue
        rasterize_polyline(mask, local, radius)
    return mask.astype("uint8")


def classical_mask(channels):
    np = __import__("numpy")
    image = robust_normalize(channels[0])
    try:
        from skimage.filters import frangi, threshold_otsu
        from skimage.morphology import binary_closing, binary_opening, remove_small_objects, disk
        vesselness = frangi(image)
        thresh = threshold_otsu(vesselness)
        mask = vesselness > thresh
        mask = binary_closing(binary_opening(mask, disk(1)), disk(2))
        mask = remove_small_objects(mask, 32)
    except ImportError:
        thresh = float(np.percentile(image, 85))
        mask = image > thresh
    return mask.astype("uint8")


def bootstrap_dataset_masks(
    data_root: str | Path,
    output_dir: str | Path,
    projections_dir: str | Path | None = None,
    method: str = "graph",
    overwrite: bool = False,
) -> Path:
    np = __import__("numpy")
    paths = resolve_dataset_paths(data_root)
    tag = dataset_tag(data_root)
    out_dir = Path(output_dir) / "pseudo_masks" / tag / method
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = find_video_files(paths.videos_dir)
    transforms = parse_transforms(tile_position_entries(load_tile_positions(paths.tile_positions_path)))
    graph = load_graph(paths.graph_analyzed_path or paths.graph_canonical_path) if method == "graph" else None
    proj_dir = Path(projections_dir) if projections_dir else Path(output_dir) / "projections" / tag
    tile_ids = sorted(set(videos) & set(transforms))
    if not tile_ids:
        raise ValueError("No tiles have both a raw video and a usable manual transform")
    first_stack = __import__("tifffile").imread(str(videos[tile_ids[0]]))
    tile_shape = tuple(first_stack.shape[-2:])
    min_y, min_x, _, _ = transformed_bounds({tile_id: transforms[tile_id] for tile_id in tile_ids}, tile_shape)
    graph_offset_yx = (-min_y, -min_x)
    LOG.info("Using graph coordinate offset y/x = %.2f, %.2f", graph_offset_yx[0], graph_offset_yx[1])
    graph_edges = prepare_graph_edges(graph) if method == "graph" else None
    if graph_edges is not None:
        LOG.info("Prepared %d graph edges for pseudo-mask rasterization", len(graph_edges))
    for tile_id in tile_ids:
        out = out_dir / f"tile_{tile_id:04d}_mask.npy"
        if out.exists() and not overwrite:
            continue
        if method == "graph":
            mask = graph_mask_for_tile(
                graph,
                transforms[tile_id],
                tile_shape,
                graph_offset_yx=graph_offset_yx,
                graph_edges=graph_edges,
            )
        elif method == "classical":
            channels = np.load(proj_dir / f"tile_{tile_id:04d}_channels.npy")
            mask = classical_mask(channels)
        else:
            raise ValueError(f"Unknown mask method: {method}")
        np.save(out, mask)
        save_tiff(out_dir / f"tile_{tile_id:04d}_mask.tif", (mask * 255).astype("uint8"))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap graph-derived or classical pseudo-masks.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="segment_stitch/work")
    p.add_argument("--projections-dir")
    p.add_argument("--method", choices=["graph", "classical"], default="graph")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    bootstrap_dataset_masks(args.data_root, args.output_dir, args.projections_dir, args.method, args.overwrite)


if __name__ == "__main__":
    main()
