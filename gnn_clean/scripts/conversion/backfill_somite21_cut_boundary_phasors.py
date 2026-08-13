#!/usr/bin/env python
"""Backfill missing AC phasor aliases in-place on a cut-ready Somite21 graph.

Input:
- existing cut-graph dataset, typically `datasets/somite21_mosaic_cut_pipeline_ready.gpickle`

Output:
- updates the input graph in place by writing synthetic-boundary edge phasor fields
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("datasets/somite21_mosaic_cut_pipeline_ready.gpickle"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph_path.expanduser().resolve()
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)

    updated_edges = 0
    skipped_edges = 0
    for u, v, edge_data in graph.edges(data=True):
        if not edge_data.get("synthetic_boundary_edge"):
            continue

        source_node = u if str(u).startswith("synthetic_source__") else v if str(v).startswith("synthetic_source__") else None
        if source_node is None:
            skipped_edges += 1
            continue

        source_data = graph.nodes[source_node]
        bc_harmonics = source_data.get("bc_harmonics")
        if bc_harmonics is None:
            skipped_edges += 1
            continue

        bc_array = np.asarray(bc_harmonics, dtype=np.complex128).reshape(-1)
        if bc_array.size == 0:
            skipped_edges += 1
            continue

        edge_data["bc_harmonics"] = bc_array
        edge_data["Q_DC"] = float(np.real(bc_array[0]))
        edge_data["mean_Q"] = float(np.real(bc_array[0]))
        edge_data["mean_Q_piv"] = float(np.real(bc_array[0]))
        for harmonic_idx in range(1, len(bc_array)):
            phasor = complex(bc_array[harmonic_idx])
            amp = float(abs(phasor))
            phase = float(np.angle(phasor))
            edge_data[f"amp_Q_h{harmonic_idx}_piv"] = amp
            edge_data[f"phase_h{harmonic_idx}_piv"] = phase
            edge_data[f"amp_Q_h{harmonic_idx}"] = amp
            edge_data[f"phase_h{harmonic_idx}"] = phase
            if harmonic_idx == 1:
                edge_data["amp_Q_piv"] = amp
                edge_data["phase_piv"] = phase
        updated_edges += 1

    if updated_edges == 0:
        raise RuntimeError(
            "No synthetic boundary edges were updated. "
            "If this is unexpected, inspect the graph contents before retrying."
        )

    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] updated {updated_edges} synthetic boundary edges in {graph_path}")
    if skipped_edges:
        print(f"[warn] skipped {skipped_edges} synthetic boundary edges without usable node bc_harmonics")


if __name__ == "__main__":
    main()
