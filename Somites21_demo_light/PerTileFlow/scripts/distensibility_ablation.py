"""Headless distensibility ablation study.

For each tile, this script scans D and refits only the nuisance pressure
degrees of freedom allowed by each ablation.  The output is a CSV plus
profile-likelihood plots that make practical identifiability visible
without opening the napari viewer.

Examples
--------
  python scripts/distensibility_ablation.py --config ../emb1/config.json
  python scripts/distensibility_ablation.py --graph ../emb1/analyzed/mosaic_graph_analyzed.gpickle --tiles 22 26 38
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from h2_sensitivity_check import fetch_h2_phasor
from inspect_tile import build_tile_problem
from synthetic_validation_neumann_bc import MU, F0_HZ, PX_SIZE_M, nL_per_m3
from pertile.analysis.local_pressure_inference import (
    _build_admittance_system,
    _compute_transfer_matrices,
)


DEFAULT_TILES = [4, 8, 10, 12, 15, 22, 23, 26, 32, 37, 38, 39, 48]


def _resolve_path(cli_val: Optional[str], cfg: dict, cfg_dir: Optional[Path],
                  key: str) -> Optional[Path]:
    val = cli_val if cli_val is not None else cfg.get(key)
    if val is None:
        return None
    p = Path(val)
    if p.is_absolute() or cfg_dir is None:
        return p
    return (cfg_dir / p).resolve()


def _load_graph_path(args) -> Path:
    cfg = {}
    cfg_dir = None
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg_dir = cfg_path.parent
        with open(cfg_path) as f:
            cfg = json.load(f)
    graph = _resolve_path(args.graph, cfg, cfg_dir, "mosaic_graph")
    if graph is None:
        raise SystemExit("Provide --graph or --config with mosaic_graph.")
    return graph


def _available_tiles(graph) -> List[int]:
    seen = set()
    for _, _, d in graph.edges(data=True):
        for m in d.get("measurements_piv") or []:
            tid = m.get("tile_id")
            try:
                seen.add(int(tid))
            except (TypeError, ValueError):
                pass
    return sorted(seen)


def _observations(graph, prob: dict, harmonics: Sequence[int]) -> dict:
    n_edges = len(prob["edges_in"])
    q_dc = np.where(prob["valid_dc"], prob["Q_DC_obs"], 0.0)
    valid = {"dc": prob["valid_dc"].copy()}
    q_h: Dict[int, np.ndarray] = {}

    if 1 in harmonics:
        q_h[1] = np.where(prob["valid_h1"], prob["Q_H1_obs"], 0.0)
        valid[1] = prob["valid_h1"].copy()
    if 2 in harmonics:
        q2, v2 = fetch_h2_phasor(
            graph, prob["edges_in"], prob["valid_h1"], prob["Q_DC_obs"])
        q_h[2] = np.where(v2, q2, 0.0)
        valid[2] = v2.copy()

    for h in harmonics:
        if h not in q_h:
            q_h[h] = np.zeros(n_edges, dtype=complex)
            valid[h] = np.zeros(n_edges, dtype=bool)
    return {"q_dc": q_dc, "q_h": q_h, "valid": valid}


def _sigma_vectors(obs: dict, args) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    q_dc = obs["q_dc"]
    valid_dc = obs["valid"]["dc"]
    a_dc = args.a_dc / nL_per_m3
    sig_dc = np.sqrt(a_dc * a_dc + (args.b_dc * np.abs(q_dc)) ** 2)
    sig_dc = np.where(valid_dc, np.maximum(sig_dc, 1e-30), 1.0)

    sig_h = {}
    for h, a_nl, b in ((1, args.a_h1, args.b_h1), (2, args.a_h2, args.b_h2)):
        if h not in obs["q_h"]:
            continue
        a = a_nl / nL_per_m3
        q = obs["q_h"][h]
        valid = obs["valid"][h]
        sig = np.sqrt(a * a + (b * np.abs(q)) ** 2)
        sig_h[h] = np.where(valid, np.maximum(sig, 1e-30), 1.0)
    return sig_dc, sig_h


def _transfer(prob: dict, D: float, harmonics: Sequence[int]):
    ab = _build_admittance_system(
        prob["sub"], prob["edges_in"], prob["boundary_nodes"],
        prob["interior_nodes"], float(D), MU, F0_HZ, tuple(harmonics),
        PX_SIZE_M)
    return _compute_transfer_matrices(
        ab, prob["edges_in"], prob["boundary_nodes"],
        prob["interior_nodes"], verbose=False)


def _fit_dc(T0, q_dc, valid_dc, sig_dc, pin_idx):
    n_bnd = T0.shape[1]
    keep = np.array([i for i in range(n_bnd) if i != pin_idx])
    P = np.zeros(n_bnd, dtype=complex)
    if valid_dc.any() and keep.size:
        A = T0[valid_dc][:, keep].real / sig_dc[valid_dc, None]
        b = q_dc[valid_dc] / sig_dc[valid_dc]
        sol, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
        P[keep] = sol
    pred = (T0 @ P).real
    r = (q_dc[valid_dc] - pred[valid_dc]) / sig_dc[valid_dc]
    return P, float(np.sum(r * r)), r


def _fit_complex_free(T, q, valid, sig):
    if not valid.any():
        return np.zeros(T.shape[1], dtype=complex), 0.0, np.array([])
    A = T[valid] / sig[valid, None]
    b = q[valid] / sig[valid]
    P, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
    resid = (q[valid] - (T @ P)[valid]) / sig[valid]
    chi2 = float(np.sum(resid.real ** 2 + resid.imag ** 2))
    return P.astype(complex), chi2, resid


def _fit_complex_global(T, q, valid, sig):
    basis = np.ones((T.shape[1], 1), dtype=complex)
    P1, chi2, resid = _fit_complex_free(T @ basis, q, valid, sig)
    return (basis[:, 0] * P1[0]), chi2, resid


def _fit_complex_fixed_phase(T, q, valid, sig, phase_vec):
    if not valid.any():
        return np.zeros(T.shape[1], dtype=complex), 0.0, np.array([])
    try:
        from scipy.optimize import nnls
    except Exception:
        nnls = None
    basis = np.exp(1j * np.asarray(phase_vec, dtype=float))
    A_c = T[valid] * basis[None, :]
    b_c = q[valid]
    A = np.vstack([A_c.real / sig[valid, None],
                   A_c.imag / sig[valid, None]])
    b = np.concatenate([b_c.real / sig[valid], b_c.imag / sig[valid]])
    if nnls is not None:
        amps, _ = nnls(A, b)
    else:
        amps, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
        amps = np.maximum(amps, 0.0)
    P = amps * basis
    resid = (q[valid] - (T @ P)[valid]) / sig[valid]
    chi2 = float(np.sum(resid.real ** 2 + resid.imag ** 2))
    return P.astype(complex), chi2, resid


def _fixed_complex_chi2(T, q, valid, sig, P_ref):
    if not valid.any():
        return 0.0, np.array([])
    resid = (q[valid] - (T @ P_ref)[valid]) / sig[valid]
    return float(np.sum(resid.real ** 2 + resid.imag ** 2)), resid


def _profile_free(prob, obs, sig_dc, sig_h, D_grid, harmonics):
    rows = []
    best = None
    for D in D_grid:
        T = _transfer(prob, D, harmonics)
        P_dc, chi_dc, r_dc = _fit_dc(
            T[0], obs["q_dc"], obs["valid"]["dc"], sig_dc,
            prob["pin_idx"])
        P_h = {}
        r_h = {}
        chi = chi_dc
        for h in harmonics:
            P_h[h], chi_h, r_h[h] = _fit_complex_free(
                T[h], obs["q_h"][h], obs["valid"][h], sig_h[h])
            chi += chi_h
        item = dict(D=float(D), chi2=float(chi), P_dc=P_dc, P_h=P_h,
                    r_dc=r_dc, r_h=r_h)
        rows.append(item)
        if best is None or item["chi2"] < best["chi2"]:
            best = item
    return rows, best


def _profile_ablation(prob, obs, sig_dc, sig_h, D_grid, harmonics,
                      mode: str, ref=None):
    rows = []
    best = None
    for D in D_grid:
        T = _transfer(prob, D, harmonics)
        _P_dc, chi, r_dc = _fit_dc(
            T[0], obs["q_dc"], obs["valid"]["dc"], sig_dc,
            prob["pin_idx"])
        r_h = {}
        P_h = {}
        for h in harmonics:
            if mode == "global_ac":
                P_h[h], chi_h, r_h[h] = _fit_complex_global(
                    T[h], obs["q_h"][h], obs["valid"][h], sig_h[h])
            elif mode == "fixed_phase":
                P_h[h], chi_h, r_h[h] = _fit_complex_fixed_phase(
                    T[h], obs["q_h"][h], obs["valid"][h], sig_h[h],
                    ref["phase"][h])
            elif mode == "fixed_complex":
                P_h[h] = ref["P_h"][h]
                chi_h, r_h[h] = _fixed_complex_chi2(
                    T[h], obs["q_h"][h], obs["valid"][h], sig_h[h],
                    ref["P_h"][h])
            else:
                raise ValueError(mode)
            chi += chi_h
        item = dict(D=float(D), chi2=float(chi), P_h=P_h, r_dc=r_dc,
                    r_h=r_h)
        rows.append(item)
        if best is None or item["chi2"] < best["chi2"]:
            best = item
    return rows, best


def _profile_dc_only(prob, obs, sig_dc, D_grid):
    rows = []
    best = None
    for D in D_grid:
        T = _transfer(prob, D, ())
        _P_dc, chi, r_dc = _fit_dc(
            T[0], obs["q_dc"], obs["valid"]["dc"], sig_dc,
            prob["pin_idx"])
        item = dict(D=float(D), chi2=float(chi), r_dc=r_dc, r_h={})
        rows.append(item)
        if best is None or item["chi2"] < best["chi2"]:
            best = item
    return rows, best


def _metric_row(tile_id: int, name: str, harmonics: Sequence[int],
                profile: List[dict], best: dict, prob: dict, obs: dict,
                n_params: int) -> dict:
    chi = np.array([p["chi2"] for p in profile], dtype=float)
    D = np.array([p["D"] for p in profile], dtype=float)
    finite = np.isfinite(chi)
    chi_min = float(np.nanmin(chi)) if finite.any() else float("nan")
    dchi = chi - chi_min
    ok1 = finite & (dchi <= 1.0)
    ok4 = finite & (dchi <= 3.84)

    def _width(mask):
        if mask.sum() < 1:
            return float("nan"), float("nan"), float("nan")
        lo = float(np.nanmin(D[mask]))
        hi = float(np.nanmax(D[mask]))
        return lo, hi, float(np.log10(hi / lo)) if lo > 0 else float("nan")

    lo1, hi1, w1 = _width(ok1)
    lo4, hi4, w4 = _width(ok4)
    n_dc = int(obs["valid"]["dc"].sum())
    n_h = {h: int(obs["valid"][h].sum()) for h in harmonics}
    n_obs = n_dc + sum(2 * n_h[h] for h in harmonics)
    dof = max(n_obs - int(n_params), 1)
    max_dchi = float(np.nanmax(dchi[finite])) if finite.any() else float("nan")
    return dict(
        tile_id=tile_id,
        ablation=name,
        harmonics="+".join(f"H{h}" for h in harmonics) or "DC",
        D_hat=best["D"],
        chi2_min=chi_min,
        chi2_red=chi_min / dof,
        dof=dof,
        n_obs=n_obs,
        n_params=n_params,
        n_dc=n_dc,
        n_h1=n_h.get(1, 0),
        n_h2=n_h.get(2, 0),
        D_lo_dchi1=lo1,
        D_hi_dchi1=hi1,
        width_decades_dchi1=w1,
        D_lo_dchi384=lo4,
        D_hi_dchi384=hi4,
        width_decades_dchi384=w4,
        max_delta_chi2=max_dchi,
        profile_flat=(max_dchi < 1.0) if np.isfinite(max_dchi) else "",
        dc_resid_std=float(np.std(best.get("r_dc", [])))
            if len(best.get("r_dc", [])) else float("nan"),
        h1_resid_rms=_complex_rms(best.get("r_h", {}).get(1)),
        h2_resid_rms=_complex_rms(best.get("r_h", {}).get(2)),
    )


def _complex_rms(x) -> float:
    if x is None:
        return float("nan")
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x.real ** 2 + x.imag ** 2)))


def _n_params(prob: dict, harmonics: Sequence[int], mode: str) -> int:
    n_b = len(prob["boundary_nodes"])
    # D + DC pressures except pin.
    n = 1 + max(n_b - 1, 0)
    if mode == "dc_only":
        return n
    for _h in harmonics:
        if mode == "free":
            n += 2 * n_b
        elif mode == "global_ac":
            n += 2
        elif mode == "fixed_phase":
            n += n_b
        elif mode == "fixed_complex":
            n += 0
    return n


def _write_profile_csv(path: Path, profiles: Dict[str, List[dict]]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ablation", "D", "chi2",
                                          "delta_chi2"])
        w.writeheader()
        for name, prof in profiles.items():
            chi_min = min(p["chi2"] for p in prof)
            for p in prof:
                w.writerow(dict(ablation=name, D=p["D"], chi2=p["chi2"],
                                delta_chi2=p["chi2"] - chi_min))


def _plot_tile(out_png: Path, tile_id: int, profiles: Dict[str, List[dict]],
               metrics: List[dict]):
    import matplotlib.pyplot as plt

    fig, (ax_prof, ax_bar) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for name, prof in profiles.items():
        D = np.array([p["D"] for p in prof], dtype=float)
        chi = np.array([p["chi2"] for p in prof], dtype=float)
        dchi = chi - np.nanmin(chi)
        ax_prof.semilogx(D, dchi, marker="o", ms=3, lw=1.2, label=name)
    ax_prof.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.7,
                    label="Delta chi2 = 1")
    ax_prof.axhline(3.84, color="0.4", ls="--", lw=0.8, alpha=0.6,
                    label="Delta chi2 = 3.84")
    ax_prof.set_xlabel("D (1/Pa)")
    ax_prof.set_ylabel("Delta chi2 from each ablation minimum")
    ax_prof.set_title(f"Tile {tile_id}: profile likelihood ablations")
    ax_prof.grid(alpha=0.25, which="both")
    ax_prof.legend(fontsize=7)

    names = [m["ablation"] for m in metrics]
    Dhat = np.array([m["D_hat"] for m in metrics], dtype=float)
    widths = np.array([m["width_decades_dchi1"] for m in metrics],
                      dtype=float)
    y = np.arange(len(names))
    ax_bar.barh(y, np.log10(Dhat), color="#4C78A8", alpha=0.85)
    for i, (d, wd) in enumerate(zip(Dhat, widths)):
        label = f"{d:.1e}"
        if np.isfinite(wd):
            label += f"  w={wd:.2g} dec"
        else:
            label += "  w=flat/outside"
        ax_bar.text(np.log10(d) if d > 0 else 0, i, "  " + label,
                    va="center", fontsize=8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=8)
    ax_bar.set_xlabel("log10(D_hat)")
    ax_bar.set_title("Best D and Delta chi2 <= 1 width")
    ax_bar.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _run_tile(graph, tile_id: int, D_grid: np.ndarray, args, out_dir: Path):
    prob = build_tile_problem(graph, tile_id)
    n_b = len(prob["boundary_nodes"])
    profiles: Dict[str, List[dict]] = {}
    metrics: List[dict] = []

    obs_all = _observations(graph, prob, (1, 2))
    sig_dc, sig_h = _sigma_vectors(obs_all, args)

    # Negative control: DC only. D should not be identifiable here.
    prof, best = _profile_dc_only(prob, obs_all, sig_dc, D_grid)
    profiles["dc_only"] = prof
    metrics.append(_metric_row(tile_id, "dc_only", (), prof, best, prob,
                               obs_all, _n_params(prob, (), "dc_only")))

    for harmonics, hlabel in [((1,), "h1"), ((1, 2), "h1h2")]:
        obs = _observations(graph, prob, harmonics)
        sig_dc_h, sig_h_h = _sigma_vectors(obs, args)

        free_name = f"{hlabel}_free_boundary"
        prof_free, best_free = _profile_free(
            prob, obs, sig_dc_h, sig_h_h, D_grid, harmonics)
        profiles[free_name] = prof_free
        metrics.append(_metric_row(
            tile_id, free_name, harmonics, prof_free, best_free, prob, obs,
            _n_params(prob, harmonics, "free")))

        ref = {
            "P_h": best_free["P_h"],
            "phase": {
                h: np.angle(best_free["P_h"][h] + 1e-300)
                for h in harmonics
            },
        }
        for mode, suffix in [
            ("fixed_phase", "fixed_boundary_phase"),
            ("fixed_complex", "fixed_boundary_complex"),
            ("global_ac", "single_global_ac_pressure"),
        ]:
            name = f"{hlabel}_{suffix}"
            prof_m, best_m = _profile_ablation(
                prob, obs, sig_dc_h, sig_h_h, D_grid, harmonics, mode,
                ref=ref)
            profiles[name] = prof_m
            metrics.append(_metric_row(
                tile_id, name, harmonics, prof_m, best_m, prob, obs,
                _n_params(prob, harmonics, mode)))

    tile_dir = out_dir / f"tile_{tile_id:03d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    _write_profile_csv(tile_dir / "profiles.csv", profiles)
    _plot_tile(tile_dir / "ablation_profiles.png", tile_id, profiles, metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Run tile-level distensibility nuisance ablations.")
    ap.add_argument("--config", default=None,
                    help="Config JSON with mosaic_graph path.")
    ap.add_argument("--graph", default=None,
                    help="Path to mosaic_graph_analyzed.gpickle.")
    ap.add_argument("--tiles", nargs="*", type=int, default=None,
                    help="Tile IDs. Default: known IDENT tiles if present.")
    ap.add_argument("--all-tiles", action="store_true",
                    help="Run every tile with at least one PIV measurement.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: renders/meeting/"
                         "distensibility_ablation.")
    ap.add_argument("--D-min", type=float, default=1e-6)
    ap.add_argument("--D-max", type=float, default=1e-3)
    ap.add_argument("--D-count", type=int, default=31)
    ap.add_argument("--a-dc", type=float, default=0.061,
                    help="DC additive noise floor in nL/s.")
    ap.add_argument("--a-h1", type=float, default=0.012,
                    help="H1 additive noise floor in nL/s.")
    ap.add_argument("--a-h2", type=float, default=0.030,
                    help="H2 additive noise floor in nL/s.")
    ap.add_argument("--b-dc", type=float, default=0.29,
                    help="DC multiplicative noise coefficient.")
    ap.add_argument("--b-h1", type=float, default=0.0,
                    help="H1 multiplicative noise coefficient.")
    ap.add_argument("--b-h2", type=float, default=0.0,
                    help="H2 multiplicative noise coefficient.")
    args = ap.parse_args()

    graph_path = _load_graph_path(args)
    out_dir = (Path(args.out_dir).resolve() if args.out_dir else
               PROJECT_ROOT / "renders" / "meeting"
               / "distensibility_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    avail = _available_tiles(graph)
    if args.all_tiles:
        tiles = avail
    elif args.tiles:
        tiles = args.tiles
    else:
        tiles = [t for t in DEFAULT_TILES if t in set(avail)] or avail
    D_grid = np.logspace(np.log10(args.D_min), np.log10(args.D_max),
                         int(args.D_count))

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
            rows = _run_tile(graph, int(tid), D_grid, args, out_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_rows.append(dict(tile_id=tid, ablation="ERROR",
                                 error=f"{type(e).__name__}: {e}"))
            continue
        all_rows.extend(rows)
        for r in rows:
            print(f"  {r['ablation']:<28} "
                  f"D={r['D_hat']:.2e}  "
                  f"chi2_red={r['chi2_red']:.2f}  "
                  f"width1={r['width_decades_dchi1']:.2g} dec  "
                  f"max_dchi={r['max_delta_chi2']:.2g}")

    summary_csv = out_dir / "ablation_summary.csv"
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
