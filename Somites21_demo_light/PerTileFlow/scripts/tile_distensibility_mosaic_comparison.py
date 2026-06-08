"""Compare tile distensibility recovery using mosaic-derived data.

For each tile, this script runs profile-likelihood scans under paired
observation/boundary-condition choices:

* measured flows, free tile boundary pressures
* measured flows, boundary pressures fixed from a whole-mosaic solve
* mosaic-simulated flows, free tile boundary pressures
* mosaic-simulated flows, boundary pressures fixed from that solve

The last case is a consistency check: with the correct whole-mosaic D, the
tile subgraph should recover the input D when the inherited boundary
pressures explain the local flows.  Deviations flag carve/topology/orientation
issues or real differences between full-mosaic and per-tile assumptions.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from distensibility_ablation import (
    _metric_row,
    _n_params,
    _observations,
    _profile_free,
    _sigma_vectors,
    _transfer,
    _write_profile_csv,
)
from inspect_tile import build_tile_problem
from synthetic_validation_neumann_bc import F0_HZ, MU, nL_per_m3
from pertile.analysis.transmission_line import solve_transmission_line
from tile_mosaic_simulation import (
    choose_tiles,
    load_graph_from_args,
    observations_from_mosaic_result,
)


def _boundary_pressures_from_result(prob: dict, result,
                                    harmonics: Sequence[int]) -> dict:
    n_b = len(prob["boundary_nodes"])
    P_dc = np.zeros(n_b, dtype=complex)
    P_h = {int(h): np.zeros(n_b, dtype=complex) for h in harmonics}
    for i, node in enumerate(prob["boundary_nodes"]):
        p = np.asarray(result.node_pressures.get(node, []), dtype=complex)
        if len(p):
            P_dc[i] = p[0]
        for h in harmonics:
            h = int(h)
            if h < len(p):
                P_h[h][i] = p[h]

    # DC pressure has an arbitrary gauge; match the tile solver pin.
    if n_b:
        P_dc = P_dc - P_dc[int(prob["pin_idx"])]
    return {"P_dc": P_dc, "P_h": P_h}


def _fixed_pressure_profile(prob: dict, obs: dict, sig_dc: np.ndarray,
                            sig_h: Dict[int, np.ndarray],
                            D_grid: np.ndarray, harmonics: Sequence[int],
                            pressure_ref: dict):
    rows = []
    best = None
    for D in D_grid:
        T = _transfer(prob, float(D), harmonics)
        pred_dc = (T[0] @ pressure_ref["P_dc"]).real
        v_dc = obs["valid"]["dc"]
        r_dc = ((obs["q_dc"][v_dc] - pred_dc[v_dc])
                / sig_dc[v_dc]) if v_dc.any() else np.array([])
        chi = float(np.sum(r_dc * r_dc))
        r_h = {}
        for h in harmonics:
            pred_h = T[h] @ pressure_ref["P_h"][int(h)]
            valid = obs["valid"][int(h)]
            r = ((obs["q_h"][int(h)][valid] - pred_h[valid])
                 / sig_h[int(h)][valid]) if valid.any() else np.array([])
            r_h[int(h)] = r
            chi += float(np.sum(r.real ** 2 + r.imag ** 2))
        item = dict(D=float(D), chi2=float(chi), r_dc=r_dc, r_h=r_h,
                    P_dc=pressure_ref["P_dc"], P_h=pressure_ref["P_h"])
        rows.append(item)
        if best is None or item["chi2"] < best["chi2"]:
            best = item
    return rows, best


def _plot_comparison(path: Path, tile_id: int,
                     profiles: Dict[str, List[dict]], D_true: float) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for name, prof in profiles.items():
        D = np.array([p["D"] for p in prof], dtype=float)
        chi = np.array([p["chi2"] for p in prof], dtype=float)
        finite = np.isfinite(chi)
        if not finite.any():
            continue
        dchi = chi - np.nanmin(chi)
        ax.semilogx(D, dchi, marker="o", ms=3, lw=1.2, label=name)
    ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.7)
    ax.axhline(3.84, color="0.4", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(D_true, color="#C0392B", lw=1.0, alpha=0.8,
               label=f"mosaic D={D_true:.1e}")
    ax.set_xlabel("D (1/Pa)")
    ax.set_ylabel("Delta chi2 from profile minimum")
    ax.set_title(f"Tile {tile_id}: measured vs mosaic-derived recovery")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_tile(graph, result, tile_id: int, D_grid: np.ndarray, args,
              out_dir: Path):
    harmonics = tuple(int(h) for h in args.harmonics)
    prob = build_tile_problem(graph, int(tile_id))
    pressure_ref = _boundary_pressures_from_result(prob, result, harmonics)
    profiles: Dict[str, List[dict]] = {}
    metrics = []

    scenarios = [
        ("measured", _observations(graph, prob, harmonics)),
        ("mosaic_flow", observations_from_mosaic_result(
            prob, result, harmonics, nL_per_m3)),
    ]

    for source_name, obs in scenarios:
        sig_dc, sig_h = _sigma_vectors(obs, args)

        free_name = f"{source_name}_free_boundary"
        prof, best = _profile_free(prob, obs, sig_dc, sig_h, D_grid,
                                   harmonics)
        profiles[free_name] = prof
        row = _metric_row(tile_id, free_name, harmonics, prof, best, prob,
                          obs, _n_params(prob, harmonics, "free"))
        row["observation_source"] = source_name
        row["boundary_mode"] = "free"
        metrics.append(row)

        fixed_name = f"{source_name}_fixed_mosaic_pressure"
        prof_f, best_f = _fixed_pressure_profile(
            prob, obs, sig_dc, sig_h, D_grid, harmonics, pressure_ref)
        profiles[fixed_name] = prof_f
        row = _metric_row(tile_id, fixed_name, harmonics, prof_f, best_f,
                          prob, obs, n_params=1)
        row["observation_source"] = source_name
        row["boundary_mode"] = "fixed_mosaic_pressure"
        metrics.append(row)

    tile_dir = out_dir / f"tile_{int(tile_id):03d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    _write_profile_csv(tile_dir / "profiles.csv", profiles)
    _plot_comparison(tile_dir / "mosaic_comparison_profiles.png",
                     int(tile_id), profiles, float(args.mosaic_D))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare tile D recovery using mosaic-derived flows/BCs.")
    ap.add_argument("--config", default="../emb1/config.json",
                    help="Bundle config JSON with mosaic_graph path.")
    ap.add_argument("--graph", default=None,
                    help="Path to mosaic_graph_analyzed.gpickle.")
    ap.add_argument("--mosaic-result", default=None,
                    help="Existing mosaic_solve_result.pkl from "
                         "mosaic_tile_boundary_solve.py.")
    ap.add_argument("--tiles", nargs="*", type=int, default=None)
    ap.add_argument("--all-tiles", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="Default: renders/meeting/"
                         "tile_distensibility_mosaic_comparison.")
    ap.add_argument("--mosaic-D", type=float, default=1.3e-3,
                    help="D used if this script needs to run the mosaic solve.")
    ap.add_argument("--D-min", type=float, default=1e-6)
    ap.add_argument("--D-max", type=float, default=3e-3)
    ap.add_argument("--D-count", type=int, default=31)
    ap.add_argument("--harmonics", nargs="+", type=int, default=[1, 2],
                    choices=[1, 2],
                    help="AC harmonics included in the tile profiles.")
    ap.add_argument("--f0-hz", type=float, default=F0_HZ)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--sink-pressure-bc", type=float, default=None)
    ap.add_argument("--merged-boundary", action="store_true")
    ap.add_argument("--a-dc", type=float, default=0.061,
                    help="DC additive noise floor in nL/s.")
    ap.add_argument("--a-h1", type=float, default=0.012,
                    help="H1 additive noise floor in nL/s.")
    ap.add_argument("--a-h2", type=float, default=0.030,
                    help="H2 additive noise floor in nL/s.")
    ap.add_argument("--b-dc", type=float, default=0.29)
    ap.add_argument("--b-h1", type=float, default=0.0)
    ap.add_argument("--b-h2", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = (Path(args.out_dir).resolve() if args.out_dir else
               PROJECT_ROOT / "renders" / "meeting"
               / "tile_distensibility_mosaic_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph_from_args(args)
    tiles = choose_tiles(graph, args.tiles, args.all_tiles)
    D_grid = np.logspace(np.log10(args.D_min), np.log10(args.D_max),
                         int(args.D_count))

    if args.mosaic_result:
        with open(args.mosaic_result, "rb") as f:
            result = pickle.load(f)
        args.mosaic_D = float(getattr(result, "D", args.mosaic_D))
        print(f"Loaded mosaic result: {args.mosaic_result}")
    else:
        print("No --mosaic-result supplied; running whole-mosaic solve.")
        result = solve_transmission_line(
            graph, D=float(args.mosaic_D),
            n_harmonics=max(int(h) for h in args.harmonics),
            f0_hz=float(args.f0_hz), mu=float(args.mu), verbose=True,
            sink_pressure_bc=args.sink_pressure_bc,
            merged_boundary=bool(args.merged_boundary))
        result_path = out_dir / "mosaic_solve_result.pkl"
        with open(result_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        args.mosaic_D = float(getattr(result, "D", args.mosaic_D))
        print(f"Wrote {result_path}")

    print(f"Graph: {graph_path}")
    print(f"Tiles: {tiles}")
    print(f"D grid: {args.D_min:.1e} .. {args.D_max:.1e} "
          f"({args.D_count} points)")
    print(f"Output: {out_dir}")

    all_rows = []
    t0 = time.time()
    for i, tid in enumerate(tiles, start=1):
        print(f"\n[{i}/{len(tiles)}] tile {tid}", flush=True)
        try:
            rows = _run_tile(graph, result, int(tid), D_grid, args, out_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_rows.append(dict(tile_id=tid, profile="ERROR",
                                 error=f"{type(e).__name__}: {e}"))
            continue
        all_rows.extend(rows)
        for r in rows:
            print(f"  {r['ablation']:<36} "
                  f"D={r['D_hat']:.2e}  "
                  f"chi2_red={r['chi2_red']:.2f}  "
                  f"width1={r['width_decades_dchi1']:.2g} dec")

    summary_csv = out_dir / "mosaic_comparison_summary.csv"
    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nWrote {summary_csv}")
    print(f"Done in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
