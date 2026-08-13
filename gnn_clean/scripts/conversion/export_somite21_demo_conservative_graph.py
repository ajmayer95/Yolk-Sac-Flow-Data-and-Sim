#!/usr/bin/env python
"""Export a new Somite21 conservative-field graph from the demo notebook pipeline.

This reproduces the pipeline from
`Somites21_demo/PerTileFlow/gnn_clean/preprocessing_cut/demo.ipynb`:

1. load `datasets/somite21_mosaic.gpickle`
2. build the cleaned network `g = CleanNetwork(raw)`
3. reconcile per-edge DC with `reconcile('winner', 3)`
4. solve the ML conservative field `Q'` with left/right ring symmetry
5. attach the solved fields and export a plain `networkx.Graph`

The output graph uses the cleaned-network topology and marks every arterial ring
node as `boundary_type="source"` and every vein node as `boundary_type="sink"`.
It creates a new graph pickle, typically `datasets/somite21_mosaic_ml_conservative.gpickle`.
Edges carry:

- `Q_data`: signed ML conservative field in the edge tuple orientation
- `Q_sim`: signed passive-geometry field in the edge tuple orientation
- `Q_meas`: signed reconciled field before the conservation solve
- `Q_DC`: non-negative magnitude of `Q_data`
- `Q_DC_signed_nl_s`: signed `Q_data`
- `flow_from` / `flow_to`: physical direction implied by `Q_data`
- `measured`, `trust`, `sigma_Q`, `r_meas`
"""

from __future__ import annotations

import argparse
import copy
import importlib
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "somite21_mosaic.gpickle"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "somite21_mosaic_ml_conservative.gpickle"
DEFAULT_DEMO_ROOT = (
    PROJECT_ROOT.parent / "Somites21_demo" / "PerTileFlow" / "gnn_clean" / "preprocessing_cut"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-graph", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-graph", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--strategy", type=str, default="winner")
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Impose equal total emergent flux into the left and right aortic rings.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def install_numpy_pickle_compat() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    for new_name, old_name in aliases.items():
        if new_name in sys.modules:
            continue
        try:
            sys.modules[new_name] = importlib.import_module(old_name)
        except Exception:
            pass


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat()
    with path.open("rb") as handle:
        return pickle.load(handle)


def import_demo_pipeline(demo_root: Path):
    if not demo_root.exists():
        raise FileNotFoundError(f"Demo root does not exist: {demo_root}")
    sys.path.insert(0, str(demo_root))
    from scripts.clean_network import CleanNetwork  # type: ignore
    from scripts.mle_pipeline import MLESolve  # type: ignore
    from scripts.reconcile import make_reconcile  # type: ignore

    return CleanNetwork, make_reconcile, MLESolve


def ring_node_to_tip(clean_graph) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for tip, nodes in clean_graph.rings.items():
        for node in nodes:
            mapping[node] = tip
    return mapping


def build_export_graph(
    raw_graph: nx.Graph,
    clean_graph,
    solver,
    *,
    input_graph: Path,
    demo_root: Path,
    strategy: str,
    harmonics: int,
    symmetric: bool,
) -> nx.Graph:
    graph = nx.Graph()
    graph.graph.update(copy.deepcopy(getattr(raw_graph, "graph", {})))
    graph.graph.update(
        {
            "pipeline_conversion_ready": True,
            "pipeline_conversion_kind": "somite21_demo_mle_conservative_graph",
            "pipeline_conversion_note": (
                "Cleaned-network graph exported from the Somites21 demo notebook pipeline: "
                "CleanNetwork(raw) -> reconcile('winner', 3) -> "
                "MLESolve.ml_field(symmetric=True) -> attach()."
            ),
            "pipeline_conversion_source": str(input_graph),
            "pipeline_conversion_demo_root": str(demo_root),
            "pipeline_phase_reference": "dc_only_conservative_field",
            "pipeline_frequency_reference": "winner_tile_harmonic_snr",
            "bc_harmonics_convention": "arterial_ring_and_venous_tip_boundary",
            "mle_strategy": str(strategy),
            "mle_harmonics": int(harmonics),
            "mle_symmetric_aortic_rings": bool(symmetric),
            "mle_corr_abs_qdata_qsim": float(solver.corr),
            "mle_reduced_chi2": float(solver.chi2),
            "mle_measured_edge_count": int(np.sum(solver.meas)),
            "arterial_ring_nodes": {
                str(int(tip)): [str(int(node)) for node in sorted(nodes)]
                for tip, nodes in clean_graph.rings.items()
            },
            "aortae": [int(node) for node in clean_graph.AORTAE],
            "veins": [int(node) for node in clean_graph.VEINS],
        }
    )

    ring_tip = ring_node_to_tip(clean_graph)
    clean_nodes = set(clean_graph.V)

    for node_id, clean_node_data in clean_graph.V.items():
        raw_node_data = copy.deepcopy(raw_graph.nodes[node_id]) if node_id in raw_graph.nodes else {}
        raw_node_data["x_raw"] = raw_node_data.get("x")
        raw_node_data["y_raw"] = raw_node_data.get("y")
        raw_node_data["x_clean_m"] = float(clean_node_data["x"])
        raw_node_data["y_clean_m"] = float(clean_node_data["y"])
        raw_node_data["cut_graph_member"] = True
        if node_id in ring_tip:
            raw_node_data["boundary_type"] = "source"
            raw_node_data["boundary_role"] = "arterial_ring"
            raw_node_data["boundary_origin_tip"] = int(ring_tip[node_id])
        elif node_id in clean_graph.VEINS:
            raw_node_data["boundary_type"] = "sink"
            raw_node_data["boundary_role"] = "venous_tip"
        else:
            raw_node_data.pop("boundary_type", None)
        raw_node_data["f"] = float(clean_graph.V[node_id].get("f", 0.0))
        graph.add_node(node_id, **raw_node_data)

    for edge, clean_edge_data in clean_graph.E.items():
        a, b = edge
        raw_edge_data = copy.deepcopy(raw_graph.edges[edge]) if raw_graph.has_edge(a, b) else {}
        q_data = float(clean_edge_data["Q_data"])
        q_sim = float(clean_edge_data["Q_sim"])
        q_meas = float(clean_edge_data["Q_meas"])
        q_abs = abs(q_data)
        q_sim_abs = abs(q_sim)
        q_meas_abs = abs(q_meas)

        if q_data >= 0.0:
            flow_from, flow_to = a, b
        else:
            flow_from, flow_to = b, a

        raw_edge_data.update(
            {
                "radius": float(clean_edge_data["radius"]),
                "length": float(clean_edge_data["length"]),
                "radius_m": float(clean_edge_data["radius"]),
                "length_m": float(clean_edge_data["length"]),
                "r_meas": float(clean_edge_data["r_meas"]),
                "flow_from": flow_from,
                "flow_to": flow_to,
                "Q_data": q_data,
                "Q_sim": q_sim,
                "Q_meas": q_meas,
                "mean_Q_linear": q_data,
                "Q_DC_signed_nl_s": q_data,
                "Q_DC_complex_original_nl_s": complex(q_data, 0.0),
                "Q_DC": q_abs,
                "mean_Q": q_abs,
                "mean_Q_piv": q_abs,
                "Q_sim_abs": q_sim_abs,
                "Q_meas_abs": q_meas_abs,
                "measured": bool(clean_edge_data["measured"]),
                "trust": bool(clean_edge_data["trust"]),
                "sigma_Q": float(clean_edge_data["sigma_Q"])
                if np.isfinite(clean_edge_data["sigma_Q"])
                else float("nan"),
                "synthetic_boundary_edge": False,
                "cut_graph_member": True,
            }
        )
        graph.add_edge(a, b, **raw_edge_data)

    # Keep only the cleaned topology, even if stale raw metadata refers to removed nodes.
    graph.remove_nodes_from([node for node in graph.nodes if node not in clean_nodes])
    return graph


def main() -> None:
    args = parse_args()
    input_graph = args.input_graph.expanduser().resolve()
    output_graph = args.output_graph.expanduser().resolve()
    demo_root = args.demo_root.expanduser().resolve()

    if output_graph.exists() and not args.overwrite:
        raise FileExistsError(f"{output_graph} already exists. Re-run with --overwrite to replace it.")

    raw_graph = load_pickle(input_graph)
    CleanNetwork, make_reconcile, MLESolve = import_demo_pipeline(demo_root)

    clean_graph = CleanNetwork(raw_graph)
    reconcile = make_reconcile(clean_graph)
    cal = reconcile(str(args.strategy), int(args.harmonics))
    solver = MLESolve(clean_graph, cal)
    solver.ml_field(symmetric=bool(args.symmetric))
    solver.geometry()
    solver.attach()

    export_graph = build_export_graph(
        raw_graph,
        clean_graph,
        solver,
        input_graph=input_graph,
        demo_root=demo_root,
        strategy=str(args.strategy),
        harmonics=int(args.harmonics),
        symmetric=bool(args.symmetric),
    )
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    with output_graph.open("wb") as handle:
        pickle.dump(export_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] wrote {output_graph}")
    print(f"  nodes: {export_graph.number_of_nodes()}")
    print(f"  edges: {export_graph.number_of_edges()}")
    print(f"  sources: {sum(data.get('boundary_type') == 'source' for _, data in export_graph.nodes(data=True))}")
    print(f"  sinks: {sum(data.get('boundary_type') == 'sink' for _, data in export_graph.nodes(data=True))}")
    print(f"  measured edges: {sum(bool(data.get('measured')) for _, _, data in export_graph.edges(data=True))}")
    print(f"  corr(|Q_data|, |Q_sim|): {solver.corr:.6f}")


if __name__ == "__main__":
    main()
