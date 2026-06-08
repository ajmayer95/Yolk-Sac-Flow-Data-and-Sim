"""Run a whole-mosaic network solve and export tile-derived BCs.

This is the first half of the "actual tile graph" experiment:

1. solve the entire analyzed mosaic graph once with the transmission-line
   model;
2. export simulated per-edge flow harmonics across the mosaic;
3. carve each requested tile and export the whole-mosaic pressures at that
   tile's boundary nodes.

The resulting pickle can be reused by
``tile_distensibility_mosaic_comparison.py`` to avoid resolving the global
network repeatedly.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inspect_tile import build_tile_problem
from synthetic_validation_neumann_bc import F0_HZ, MU
from pertile.analysis.transmission_line import solve_transmission_line
from tile_mosaic_simulation import (
    choose_tiles,
    load_graph_from_args,
    write_edge_flow_csv,
    write_tile_boundary_pressure_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Solve the whole mosaic and export tile boundary data.")
    ap.add_argument("--config", default="../emb1/config.json",
                    help="Bundle config JSON with mosaic_graph path.")
    ap.add_argument("--graph", default=None,
                    help="Path to mosaic_graph_analyzed.gpickle.")
    ap.add_argument("--tiles", nargs="*", type=int, default=None,
                    help="Tile IDs. Default: known IDENT tiles if present.")
    ap.add_argument("--all-tiles", action="store_true",
                    help="Run every tile with at least one PIV measurement.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: renders/meeting/mosaic_tile_solve.")
    ap.add_argument("--D", type=float, default=1.3e-3,
                    help="Whole-mosaic distensibility used for the solve.")
    ap.add_argument("--n-harmonics", type=int, default=2,
                    help="Number of AC harmonics to solve/export.")
    ap.add_argument("--f0-hz", type=float, default=F0_HZ)
    ap.add_argument("--mu", type=float, default=MU,
                    help="Viscosity in Pa*s.")
    ap.add_argument("--sink-pressure-bc", type=float, default=None,
                    help="Optional Dirichlet pressure at sink boundary nodes.")
    ap.add_argument("--merged-boundary", action="store_true",
                    help="Use the solver's merged-boundary mode.")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir).resolve() if args.out_dir else
               PROJECT_ROOT / "renders" / "meeting" / "mosaic_tile_solve")
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph_from_args(args)
    tiles = choose_tiles(graph, args.tiles, args.all_tiles)

    print(f"Graph: {graph_path}")
    print(f"Tiles: {tiles}")
    print(f"Whole-mosaic D: {args.D:.3e}  n_harmonics={args.n_harmonics}")
    print(f"Output: {out_dir}")

    t0 = time.time()
    result = solve_transmission_line(
        graph, D=float(args.D), n_harmonics=int(args.n_harmonics),
        f0_hz=float(args.f0_hz), mu=float(args.mu), verbose=True,
        sink_pressure_bc=args.sink_pressure_bc,
        merged_boundary=bool(args.merged_boundary))

    result_path = out_dir / "mosaic_solve_result.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {result_path}")

    edge_csv = out_dir / "mosaic_simulated_edge_flows.csv"
    write_edge_flow_csv(edge_csv, graph, result, int(args.n_harmonics))
    print(f"Wrote {edge_csv}")

    tile_probs = {}
    for i, tid in enumerate(tiles, start=1):
        print(f"[{i}/{len(tiles)}] carving tile {tid}", flush=True)
        tile_probs[int(tid)] = build_tile_problem(graph, int(tid))

    bnd_csv = out_dir / "tile_boundary_pressures_from_mosaic.csv"
    write_tile_boundary_pressure_csv(bnd_csv, tile_probs, result,
                                     int(args.n_harmonics))
    print(f"Wrote {bnd_csv}")
    print(f"Done in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
