#!/usr/bin/env python
"""Repair Somite21 cut-graph geometry compatibility aliases in place.

The cut graph stores `radius_m` / `length_m` correctly in meters, but older
geometry readers in this repo prioritize `radius_px_true` / `length_true`.
If those pixel-scale aliases are absent, they may treat meter-valued
`radius` / `length` as pixels and rescale them a second time.

This script backfills the compatibility aliases so the stored meter geometry
is read consistently by both legacy and newer code paths.

Input:
- existing cut-ready Somite21 graph

Output:
- updates the input graph in place by backfilling compatibility aliases
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "somite21_mosaic_cut_pipeline_ready.gpickle"

# Matches the Somite21 preprocessing conversion scale observed in the cut graph.
RESCALE_M_PER_PX = 1.7e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    return parser.parse_args()


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_graph(graph, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def meters_to_px(value_m: float) -> float:
    value_m = float(value_m)
    if not math.isfinite(value_m) or value_m <= 0.0:
        return 1.0e-6 / RESCALE_M_PER_PX
    return value_m / RESCALE_M_PER_PX


def best_incident_real_edge_geometry(graph, target_node, synthetic_node):
    best = None
    for nbr in graph.neighbors(target_node):
        if nbr == synthetic_node:
            continue
        edge_data = graph.edges[target_node, nbr]
        if edge_data.get("synthetic_boundary_edge"):
            continue
        radius_m = edge_data.get("radius_m", edge_data.get("radius"))
        length_m = edge_data.get("length_m", edge_data.get("length"))
        try:
            radius_m = float(radius_m)
            length_m = float(length_m)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(radius_m) and radius_m > 0.0 and math.isfinite(length_m) and length_m > 0.0):
            continue
        score = radius_m**4 / length_m
        if best is None or score > best[0]:
            best = (score, radius_m, length_m, nbr)
    return best


def main() -> None:
    args = parse_args()
    graph_path = args.graph_path.expanduser().resolve()
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    graph = load_graph(graph_path)
    updated = 0
    synthetic_updated = 0
    synthetic_missing = 0
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("synthetic_boundary_edge"):
            synthetic_node = u if str(u).startswith("synthetic_source__") else v if str(v).startswith("synthetic_source__") else None
            target_node = v if synthetic_node == u else u if synthetic_node == v else None
            if synthetic_node is not None and target_node is not None:
                best = best_incident_real_edge_geometry(graph, target_node, synthetic_node)
                if best is not None:
                    _, radius_m, length_m, nbr = best
                    edge_data["radius"] = radius_m
                    edge_data["length"] = length_m
                    edge_data["radius_m"] = radius_m
                    edge_data["length_m"] = length_m
                    edge_data["radius_source"] = "incident_real_edge_max_conductance"
                    edge_data["length_source"] = "incident_real_edge_max_conductance"
                    edge_data["synthetic_geometry_proxy_edge"] = (target_node, nbr)
                    synthetic_updated += 1
                else:
                    synthetic_missing += 1
        radius_m = edge_data.get("radius_m", edge_data.get("radius"))
        length_m = edge_data.get("length_m", edge_data.get("length"))
        try:
            radius_m = float(radius_m)
            length_m = float(length_m)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(radius_m) and radius_m > 0.0 and math.isfinite(length_m) and length_m > 0.0):
            continue
        edge_data["radius_px_true"] = meters_to_px(radius_m)
        edge_data["length_true"] = meters_to_px(length_m)
        updated += 1

    if updated == 0:
        raise RuntimeError(f"No eligible edges were updated in {graph_path}")

    save_graph(graph, graph_path)
    print(f"[ok] updated {updated} edges in {graph_path}")
    print(f"[ok] updated {synthetic_updated} synthetic boundary edges from local real-edge geometry")
    if synthetic_missing:
        print(f"[warn] {synthetic_missing} synthetic boundary edges had no usable incident real-edge geometry")


if __name__ == "__main__":
    main()
