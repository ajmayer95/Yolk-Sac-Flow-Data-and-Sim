"""H2 sensitivity check.

Refits one or a few IDENT tiles under TWO harmonic configurations:
  (i) DC + H1 only   (the joint_lm default)
  (ii) DC + H1 + H2  (adding H2 phasor on each measured edge)

Compares D̂, σ_D, and σ_D / D̂ between the two.  If σ_D tightens by ≲ 15%
across tiles, dropping H2 from the canonical pipeline is defensible
(per desktop-Claude's criterion).  If H2 buys ≥ 30% σ_D reduction on
multiple tiles, H2 should stay.

Usage:
  python scripts/h2_sensitivity_check.py           # all 13 IDENT tiles
  python scripts/h2_sensitivity_check.py 22 26 38  # specific tiles
"""
from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from synthetic_validation_neumann_bc import (
    joint_lm, MU, F0_HZ, PX_SIZE_M, nL_per_m3, GRAPH_PATH,
)
from pertile.analysis import local_pressure_inference as lpi
from pertile.analysis.local_pressure_inference import (
    extract_tile_subgraph_spatial, _edge_geometry,
)
from inspect_tile import build_tile_problem

IDENT_TILES = [4, 8, 10, 12, 15, 22, 23, 26, 32, 37, 38, 39, 48]

OUT_DIR = PROJECT_ROOT / "renders" / "meeting" / "h2_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "h2_sensitivity.csv"

B_KCL = 0.29


def fetch_h2_phasor(graph, edges_in, valid_h1, Q_DC_obs):
    """Build Q_H2_obs (SI m³/s, complex) per edge.  Uses
    Q_H2_amp/Q_H2_phi with legacy fallback on each edge; signs flip to match
    flow_from→flow_to direction (same convention as DC and H1)."""
    n = len(edges_in)
    Q_H2 = np.zeros(n, dtype=complex)
    valid_h2 = np.zeros(n, dtype=bool)
    for i, (u, v) in enumerate(edges_in):
        ed = graph.edges[u, v]
        amp = ed.get("Q_H2_amp") or ed.get("amp_Q_h2_piv")
        phase = ed.get("Q_H2_phi") or ed.get("phase_h2_piv")
        ff = ed.get("flow_from"); ft = ed.get("flow_to")
        if (amp is None or phase is None or not np.isfinite(amp)
                or not np.isfinite(phase)):
            continue
        sign = 1.0 if (ff == u and ft == v) else -1.0
        Q_H2[i] = float(amp) * np.exp(1j * float(phase)) * sign / nL_per_m3
        valid_h2[i] = True
    return Q_H2, valid_h2


def fit_with_harmonics(graph, tile_id, harmonics, b_mult=B_KCL):
    """Fit joint_lm with the specified ac_harmonics tuple."""
    prob = build_tile_problem(graph, tile_id)
    n_edges = len(prob["edges_in"])
    Q_DC_abs = np.where(prob["valid_dc"], np.abs(prob["Q_DC_obs"]), 0)
    Q_H1_abs = np.where(prob["valid_h1"], np.abs(prob["Q_H1_obs"]), 0)
    sigma_dc_e = np.maximum(b_mult * Q_DC_abs, 1e-18)
    sigma_h1_e = np.maximum(b_mult * Q_H1_abs, 1e-18)
    Q_DC_clean = np.where(prob["valid_dc"], prob["Q_DC_obs"], 0)
    Q_H1_clean = np.where(prob["valid_h1"], prob["Q_H1_obs"], 0)

    Q_H_dict = {1: Q_H1_clean}
    sigma_h_dict = {1: sigma_h1_e}

    if 2 in harmonics:
        Q_H2_full, valid_h2 = fetch_h2_phasor(
            graph, prob["edges_in"], prob["valid_h1"], prob["Q_DC_obs"])
        Q_H2_clean = np.where(valid_h2, Q_H2_full, 0)
        Q_H2_abs = np.where(valid_h2, np.abs(Q_H2_full), 0)
        sigma_h2_e = np.maximum(b_mult * Q_H2_abs, 1e-18)
        Q_H_dict[2] = Q_H2_clean
        sigma_h_dict[2] = sigma_h2_e
        n_h2 = int(valid_h2.sum())
    else:
        n_h2 = 0

    lm = joint_lm(
        prob["sub"], prob["edges_in"],
        prob["boundary_nodes"], prob["interior_nodes"],
        Q_DC_clean, Q_H_dict, sigma_dc_e, sigma_h_dict,
        ac_harmonics=tuple(harmonics), pin_dc=True,
        pin_idx=prob["pin_idx"], D_init=1.3e-3, verbose=False)
    return dict(
        D_hat=float(lm["D_hat"]),
        sigma_D=float(lm["sigma_D"]),
        rel_sigma=float(lm["sigma_D"]) / float(lm["D_hat"])
            if lm["D_hat"] > 0 else float("nan"),
        chi2=float(lm["chi2"]), iters=int(lm["iters"]),
        n_h2=n_h2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", type=int, nargs="*", default=IDENT_TILES)
    args = ap.parse_args()

    print(f"Loading graph ...", flush=True)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    print(f"\nFitting {len(args.tiles)} tiles under H1 vs H1+H2 "
          f"(KCL noise b={B_KCL}) ...", flush=True)
    rows = []
    t0 = time.time()
    for tid in args.tiles:
        try:
            r1 = fit_with_harmonics(graph, tid, (1,))
            r12 = fit_with_harmonics(graph, tid, (1, 2))
        except Exception as e:
            print(f"  tile {tid}: ERROR {e}")
            continue
        # Tightening (positive = H2 reduces σ_D, so H2 helps)
        tight = (1.0 - r12["sigma_D"] / r1["sigma_D"]
                  if r1["sigma_D"] > 0 else float("nan"))
        rows.append(dict(
            tile_id=tid, n_h2=r12["n_h2"],
            D_hat_H1=r1["D_hat"], rel_sig_H1=r1["rel_sigma"],
            D_hat_H1H2=r12["D_hat"], rel_sig_H1H2=r12["rel_sigma"],
            D_ratio=r12["D_hat"]/r1["D_hat"] if r1["D_hat"] > 0 else float("nan"),
            sigma_tightening=tight,
        ))
        print(f"  tile {tid:>3}:  H1 D̂={r1['D_hat']:.2e} (σ/D̂={r1['rel_sigma']:.2f})  "
              f"|  H1+H2 D̂={r12['D_hat']:.2e} (σ/D̂={r12['rel_sigma']:.2f})  "
              f"|  σ_D tighten = {100*tight:+.1f}%  (n_h2={r12['n_h2']})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({(time.time()-t0)/60:.1f} min)")

    if len(df) > 0:
        print(f"\n=== Summary ===")
        print(f"  median σ_D tightening from adding H2: "
              f"{100*df.sigma_tightening.median():+.1f}%  "
              f"(IQR [{100*df.sigma_tightening.quantile(0.25):+.1f}%, "
              f"{100*df.sigma_tightening.quantile(0.75):+.1f}%])")
        print(f"  median |D_ratio - 1|: "
              f"{(df.D_ratio - 1).abs().median():.2%}")
        print(f"  tiles where adding H2 changes D̂ by >50%: "
              f"{((df.D_ratio - 1).abs() > 0.5).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
