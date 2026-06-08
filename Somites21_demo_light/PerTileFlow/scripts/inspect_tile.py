"""Script A — Tile inspector (production pipeline).

Run the production-like joint inversion on any tile and render the
6-panel diagnostic figure.

Default pipeline (post-2026-05-28 consolidation):
  • DC + H1 + H2 joint fit via joint_lm
  • Per-channel noise floors a_c (initialised from manuscript values,
    refit per tile via one outer FGLS pass)
  • This matches the manuscript Section 7.2 inversion structure.

Legacy mode (--legacy):
  • DC + H1 only, with KCL multiplicative noise σ = b|Q|, b=0.29
  • This is the simplified pipeline used in the meeting scripts
    earlier in the day.  Faster, but doesn't reproduce manuscript
    D̂ values cleanly.

Usage:
  python scripts/inspect_tile.py 22                # production pipeline
  python scripts/inspect_tile.py 22 --legacy       # old DC+H1+KCL pipeline
  python scripts/inspect_tile.py 22 --no-show
"""
from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pertile.analysis import local_pressure_inference as lpi
from pertile.analysis.local_pressure_inference import (
    extract_tile_subgraph_spatial, _edge_geometry,
    _build_admittance_system, _compute_transfer_matrices,
)
from synthetic_validation_neumann_bc import (
    joint_lm, MU, F0_HZ, PX_SIZE_M, HARMONICS, nL_per_m3,
)

GRAPH_PATH = ("~/Library/CloudStorage/"
              "<your-drive>/My Drive/"
              "Somites21/Mosaic/Graphs/mosaic_graph_analyzed.gpickle")
OUT_DIR = PROJECT_ROOT / "renders" / "meeting" / "tile_inspector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

A_DC_FGLS_DEFAULT = 0.06      # FGLS additive floor (nL/s)


def build_tile_problem(graph, tile_id, drop_dangling=True):
    """Extract carve, build subgraph, prepare per-edge Q_obs vectors."""
    spec = lpi.LocalInferenceSpec(
        D_init=1.3e-3, eps_D=1e-3, lambda_reg=0.0, P_scale_Pa=None,
        harmonics=HARMONICS, use_dc=True, use_joint_lm=True,
        include_unmeasured_anatomy=True, mu=MU, px_size_m=PX_SIZE_M,
        f0_hz=F0_HZ, verbose=False, save_to_graph=False,
        prior_mode="magnitude", max_iter=200, tol_rel=1e-7,
        n_outer_iter=1, carve_drop_dangling_boundaries=drop_dangling)
    edges_in, _, boundary_nodes, interior_nodes = \
        extract_tile_subgraph_spatial(
            graph, int(tile_id),
            inset_frac=float(spec.carve_inset_frac),
            restrict_to_tile_piv_nodes=bool(spec.carve_restrict_to_tile_piv),
            drop_dangling_boundaries=drop_dangling)

    interior_set = set(interior_nodes)
    g_attach = {n: 0.0 for n in boundary_nodes}
    for u, v in edges_in:
        Rm, Lm = _edge_geometry(graph.edges[u, v], spec.px_size_m)
        if Rm <= 0 or Lm <= 0:
            continue
        Ge = float(np.pi * Rm ** 4 / (8.0 * spec.mu * Lm))
        if u in g_attach and v in interior_set:
            g_attach[u] += Ge
        if v in g_attach and u in interior_set:
            g_attach[v] += Ge
    pin_node = max(g_attach, key=g_attach.get) if g_attach else boundary_nodes[0]
    pin_idx = boundary_nodes.index(pin_node)

    n_edges = len(edges_in)
    Q_DC_obs = np.full(n_edges, np.nan, dtype=float)
    Q_H1_obs = np.full(n_edges, np.nan, dtype=complex)
    valid_dc = np.zeros(n_edges, dtype=bool)
    valid_h1 = np.zeros(n_edges, dtype=bool)
    for i, (u, v) in enumerate(edges_in):
        ed = graph.edges[u, v]
        mq = ed.get("mean_Q") or ed.get("mean_Q_nL_s")
        ff = ed.get("flow_from"); ft = ed.get("flow_to")
        if mq is None or not np.isfinite(mq) or ff is None or ft is None:
            continue
        # Use flow_from→flow_to as the canonical edge direction
        sign = 1.0 if (ff == u and ft == v) else -1.0
        Q_DC_obs[i] = float(mq) * sign / nL_per_m3      # → SI m³/s
        valid_dc[i] = True
        amp = ed.get("amp_Q"); phase = ed.get("phase")
        if amp is not None and phase is not None and np.isfinite(amp) and np.isfinite(phase):
            Q_H1_obs[i] = (float(amp) * np.exp(1j * float(phase))
                            * sign / nL_per_m3)
            valid_h1[i] = True

    sub = nx.Graph()
    for n in set(interior_nodes) | set(boundary_nodes):
        nd = graph.nodes[n]
        sub.add_node(n, x=float(nd.get("x", 0)), y=float(nd.get("y", 0)),
                      boundary_type=None)
    for u, v in edges_in:
        sub.add_edge(u, v, **dict(graph.edges[u, v]))

    return dict(
        spec=spec, sub=sub, edges_in=edges_in,
        boundary_nodes=boundary_nodes, interior_nodes=interior_nodes,
        pin_idx=pin_idx, pin_node=pin_node,
        Q_DC_obs=Q_DC_obs, Q_H1_obs=Q_H1_obs,
        valid_dc=valid_dc, valid_h1=valid_h1,
    )


def _unpack_noise_model(noise_model):
    """Accept either a legacy `(a, b)` tuple (applied to both DC and
    H1) or a per-channel dict {'dc': (a, b), 'h1': (a, b)}.  Returns
    `((a_dc, b_dc), (a_h1, b_h1))`."""
    if isinstance(noise_model, dict):
        a_dc, b_dc = noise_model.get('dc', (0.0, 0.0))
        a_h1, b_h1 = noise_model.get('h1', noise_model.get('dc', (0.0, 0.0)))
        return (float(a_dc), float(b_dc)), (float(a_h1), float(b_h1))
    # Legacy: single tuple applied to both channels.
    a, b = noise_model
    return (float(a), float(b)), (float(a), float(b))


def fit_P_given_D(prob, D, noise_model):
    """Fit P (DC + H1) by weighted LS at fixed D.  Returns dict with
    P_DC, P_H1, χ²_DC, χ²_H1, residuals.

    `noise_model` is either a legacy `(a, b)` tuple (applied to both
    DC and H1) or a per-channel dict `{'dc': (a, b), 'h1': (a, b)}`.
    """
    ab = _build_admittance_system(
        prob["sub"], prob["edges_in"],
        prob["boundary_nodes"], prob["interior_nodes"],
        float(D), MU, F0_HZ, HARMONICS, PX_SIZE_M)
    T = _compute_transfer_matrices(
        ab, prob["edges_in"], prob["boundary_nodes"],
        prob["interior_nodes"], verbose=False)

    pin_idx = prob["pin_idx"]
    keep = np.array([i for i in range(len(prob["boundary_nodes"]))
                       if i != pin_idx])
    Q_DC = prob["Q_DC_obs"]; v_dc = prob["valid_dc"]
    Q_H1 = prob["Q_H1_obs"]; v_h1 = prob["valid_h1"]

    (a_dc, b_dc), (a_h1, b_h1) = _unpack_noise_model(noise_model)
    sig_dc = np.where(v_dc, np.sqrt(a_dc**2 + b_dc**2 * np.where(v_dc, Q_DC, 0)**2), 1.0)
    sig_h1 = np.where(v_h1, np.sqrt(a_h1**2 + b_h1**2 * np.where(v_h1, np.abs(Q_H1), 0)**2), 1.0)

    # DC: real-valued LS, with pin column dropped
    A_dc = (T[0][v_dc][:, keep].real) / sig_dc[v_dc, None]
    b_dc = Q_DC[v_dc] / sig_dc[v_dc]
    P_DC_solved, *_ = np.linalg.lstsq(A_dc, b_dc, rcond=1e-10)
    P_DC = np.zeros(len(prob["boundary_nodes"]), complex)
    P_DC[keep] = P_DC_solved

    # H1: complex LS (no pin)
    A_h1 = T[1][v_h1] / sig_h1[v_h1, None]
    b_h1 = Q_H1[v_h1] / sig_h1[v_h1]
    P_H1, *_ = np.linalg.lstsq(A_h1, b_h1, rcond=1e-10)
    P_H1 = P_H1.astype(complex)

    # residuals
    r_dc = (Q_DC[v_dc] - (T[0] @ P_DC).real[v_dc]) / sig_dc[v_dc]
    r_h1 = (Q_H1[v_h1] - (T[1] @ P_H1)[v_h1]) / sig_h1[v_h1]
    chi2_dc = float(np.sum(r_dc**2))
    chi2_h1 = float(np.sum(np.abs(r_h1)**2))
    n_dc = int(v_dc.sum())
    n_h1 = int(v_h1.sum())
    n_p_dc = len(keep)
    n_p_h1 = 2 * len(prob["boundary_nodes"])    # real + imag
    dof = max(n_dc + 2 * n_h1 - 1 - n_p_dc - n_p_h1, 1)
    return dict(P_DC=P_DC, P_H1=P_H1,
                 chi2_dc=chi2_dc, chi2_h1=chi2_h1,
                 chi2_total=chi2_dc + chi2_h1, dof=dof,
                 r_dc=r_dc, r_h1=r_h1,
                 sigma_dc=sig_dc, sigma_h1=sig_h1)


def profile_likelihood(prob, noise_model, D_grid=None):
    if D_grid is None:
        D_grid = np.logspace(-6, -1, 41)
    chi2s = np.empty(len(D_grid))
    for i, D in enumerate(D_grid):
        try:
            res = fit_P_given_D(prob, float(D), noise_model)
            chi2s[i] = res["chi2_total"]
        except Exception:
            chi2s[i] = np.nan
    return D_grid, chi2s


def render(tile_id, prob, lm_result, prof_D, prof_chi2, noise_model,
            outpath, show=False, lm_trajectory=None):
    # Accept either legacy (a, b) tuple or per-channel dict.  Display
    # uses the DC channel's (a, b) for the title; per-channel summary
    # can be added by the caller in the result panel.
    (a_DC, b_mult), _ = _unpack_noise_model(noise_model)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)
    ax_carve = fig.add_subplot(gs[0, 0])
    ax_pdc = fig.add_subplot(gs[0, 1])
    ax_ph1 = fig.add_subplot(gs[0, 2])
    ax_prof = fig.add_subplot(gs[1, 0])
    ax_rdc = fig.add_subplot(gs[1, 1])
    ax_rh1 = fig.add_subplot(gs[1, 2])

    # ── (a) carve ──
    sub = prob["sub"]
    boundary_set = set(prob["boundary_nodes"])
    edges_in = prob["edges_in"]
    R_max = 0.0
    for u, v in edges_in:
        R = float(sub.edges[u, v].get("radius", 1.0)) or 1.0
        R_max = max(R_max, R)
    for u, v in edges_in:
        x1, y1 = sub.nodes[u]["x"], sub.nodes[u]["y"]
        x2, y2 = sub.nodes[v]["x"], sub.nodes[v]["y"]
        R = float(sub.edges[u, v].get("radius", 1.0)) or 1.0
        lw = 0.3 + 1.6 * (R / R_max)
        ax_carve.plot([x1, x2], [y1, y2], color="#888888", lw=lw, alpha=0.7,
                       solid_capstyle="round")
    int_x = [sub.nodes[n]["x"] for n in prob["interior_nodes"]]
    int_y = [sub.nodes[n]["y"] for n in prob["interior_nodes"]]
    bnd_x = [sub.nodes[n]["x"] for n in prob["boundary_nodes"]]
    bnd_y = [sub.nodes[n]["y"] for n in prob["boundary_nodes"]]
    ax_carve.scatter(int_x, int_y, s=4, color="black", alpha=0.4,
                      edgecolor="none", zorder=2)
    ax_carve.scatter(bnd_x, bnd_y, s=22, facecolor="none",
                      edgecolor="#C0392B", lw=1.0, zorder=3)
    px = sub.nodes[prob["pin_node"]]
    ax_carve.scatter([px["x"]], [px["y"]], s=70, marker="*",
                      color="#1f77b4", zorder=4, label="DC pin")
    ax_carve.invert_yaxis(); ax_carve.set_aspect("equal", "box")
    ax_carve.set_xticks([]); ax_carve.set_yticks([])
    ax_carve.set_title(f"(a) Tile {tile_id} carve  "
                        f"[{len(prob['interior_nodes'])} int, "
                        f"{len(prob['boundary_nodes'])} bnd, "
                        f"{len(edges_in)} edges]",
                        fontsize=10)
    ax_carve.legend(loc="upper right", fontsize=8)

    # ── (b) P̂_DC bars ──
    P_DC = lm_result["P_DC"].real
    ax_pdc.bar(np.arange(len(P_DC)), P_DC, color="#1f77b4", alpha=0.75)
    ax_pdc.axvline(prob["pin_idx"], color="black", ls=":", lw=0.8,
                    alpha=0.6, label="pin (=0)")
    ax_pdc.set_xlabel("boundary node index")
    ax_pdc.set_ylabel(r"$\hat P_\mathrm{DC}$ [Pa]")
    ax_pdc.set_title(r"(b) $\hat P_\mathrm{DC}$ at boundary "
                      "(DC pin-gauged)", fontsize=10)
    ax_pdc.legend(fontsize=8); ax_pdc.grid(alpha=0.3)

    # ── (c) P̂_H1 complex plane ──
    P_H1 = lm_result["P_H1"]
    for p in P_H1:
        ax_ph1.plot([0, p.real], [0, p.imag], color="#1f77b4",
                     lw=0.6, alpha=0.4)
    ax_ph1.scatter(P_H1.real, P_H1.imag, s=40, color="#1f77b4",
                    alpha=0.85, edgecolor="white", lw=0.4)
    ax_ph1.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax_ph1.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax_ph1.set_xlabel(r"Re $\hat P_\mathrm{H1}$ [Pa]")
    ax_ph1.set_ylabel(r"Im $\hat P_\mathrm{H1}$ [Pa]")
    ax_ph1.set_title(r"(c) $\hat P_\mathrm{H1}$ at boundary "
                      f"(|⟨P⟩|={float(np.mean(np.abs(P_H1))):.3f} Pa)",
                      fontsize=10)
    ax_ph1.set_aspect("equal", adjustable="datalim")
    ax_ph1.grid(alpha=0.3)

    # ── (d) profile likelihood ──
    chi2_min = float(np.nanmin(prof_chi2))
    dchi2 = prof_chi2 - chi2_min
    i_min = int(np.nanargmin(prof_chi2))
    D_best = prof_D[i_min]
    ax_prof.semilogx(prof_D, dchi2, "o-", color="#1f77b4", ms=4, lw=1.4)
    ax_prof.axhline(1.0, color="black", ls=":", lw=0.8,
                     alpha=0.6, label=r"$\Delta\chi^2=1$")
    ax_prof.axvline(D_best, color="#C0392B", lw=1.2,
                     label=fr"$\hat D = {D_best:.2e}$ Pa$^{{-1}}$")
    ax_prof.set_xlabel(r"$D$ [1/Pa]")
    ax_prof.set_ylabel(r"$\Delta\chi^2(D)$")
    ax_prof.set_title("(d) Profile likelihood (D scan)", fontsize=10)
    ax_prof.set_ylim(-0.5, max(8, dchi2[np.isfinite(dchi2)].min()*2 + 5))
    # ── Optional LM trajectory overlay ──
    # Accepts a list of dicts with keys 'D', 'chi2', and (optionally)
    # 'accept' and 'iter'.  Plots accepted steps as a connected orange
    # path on top of the profile-likelihood curve, with iteration
    # numbers at the first and last accepted step.  Rejected steps
    # render as faint grey crosses for completeness.
    if lm_trajectory:
        traj_D = np.array([float(h.get('D', np.nan))
                            for h in lm_trajectory])
        traj_chi2 = np.array([float(h.get('chi2', np.nan))
                                for h in lm_trajectory])
        accepted = np.array([bool(h.get('accept', True))
                              for h in lm_trajectory])
        # Re-anchor: the LM minimises DC+H1+H2 while this profile is
        # DC+H1 only, so the trajectory's absolute chi² sits above the
        # profile by a roughly-constant offset = H2 chi² contribution.
        # Shift the trajectory vertically so its converged step lies
        # on the profile curve at the same D.  Preserves the
        # trajectory's shape, makes both directly comparable.
        offset = 0.0
        if accepted.any():
            i_final = np.where(accepted)[0][-1]
            D_final = traj_D[i_final]
            chi2_final = traj_chi2[i_final]
            # Closest grid point to D_final on the profile.
            if np.isfinite(D_final) and D_final > 0:
                i_closest = int(np.argmin(
                    np.abs(np.log(prof_D / max(D_final, 1e-30)))))
                offset = chi2_final - prof_chi2[i_closest]
        traj_dchi2 = (traj_chi2 - offset) - chi2_min
        # Connected line for accepted steps in iteration order.
        if accepted.any():
            ax_prof.plot(
                traj_D[accepted], traj_dchi2[accepted],
                'o-', color='#E67E22', ms=5, lw=1.2, alpha=0.85,
                label='LM trajectory (accepted)')
        # Faint markers for rejected proposals.
        if (~accepted).any():
            ax_prof.plot(
                traj_D[~accepted], traj_dchi2[~accepted],
                'x', color='gray', ms=6, alpha=0.4,
                label='LM rejected')
        # Iteration-number annotations on first and last accepted step.
        accepted_idx = np.where(accepted)[0]
        if accepted_idx.size:
            for offset, (label_y, j) in enumerate(
                    [(8, accepted_idx[0]),
                     (-12, accepted_idx[-1])]):
                ax_prof.annotate(
                    f"iter {lm_trajectory[j].get('iter', j)}",
                    (traj_D[j], traj_dchi2[j]),
                    xytext=(5, label_y), textcoords='offset points',
                    fontsize=8, color='#E67E22')
        # Stretch the y-axis if the trajectory's first iter is far off
        # the profile minimum.
        finite_traj = traj_dchi2[np.isfinite(traj_dchi2)]
        if finite_traj.size:
            ax_prof.set_ylim(
                -0.5,
                max(ax_prof.get_ylim()[1],
                    float(finite_traj.max()) * 1.05))
    ax_prof.legend(fontsize=8); ax_prof.grid(alpha=0.3, which="both")

    # ── (e) DC residuals ──
    r_dc = lm_result["r_dc"]
    ax_rdc.hist(r_dc, bins=30, color="#7A4F00", alpha=0.75,
                 edgecolor="white", lw=0.4)
    ax_rdc.axvline(0, color="black", ls="--", lw=0.7)
    ax_rdc.set_xlabel("standardised DC residual")
    ax_rdc.set_ylabel("count")
    ax_rdc.set_title(f"(e) DC residuals  n={len(r_dc)}  "
                      f"std={float(np.std(r_dc)):.2f}  "
                      f"kurt={float(_pearson_kurt(r_dc)):.1f}",
                      fontsize=10)
    ax_rdc.grid(alpha=0.3)

    # ── (f) H1 residuals ──
    r_h1 = lm_result["r_h1"]
    r_h1_real = r_h1.real
    r_h1_imag = r_h1.imag
    ax_rh1.hist(r_h1_real, bins=30, color="#2E8B57", alpha=0.55,
                 edgecolor="white", lw=0.4, label="Re")
    ax_rh1.hist(r_h1_imag, bins=30, color="#C0392B", alpha=0.45,
                 edgecolor="white", lw=0.4, label="Im")
    ax_rh1.axvline(0, color="black", ls="--", lw=0.7)
    ax_rh1.set_xlabel("standardised H1 residual")
    ax_rh1.set_ylabel("count")
    pooled = np.concatenate([r_h1_real, r_h1_imag])
    ax_rh1.set_title(f"(f) H1 residuals  n={len(r_h1)}  "
                      f"std={float(np.std(pooled)):.2f}  "
                      f"kurt={float(_pearson_kurt(pooled)):.1f}",
                      fontsize=10)
    ax_rh1.legend(fontsize=8); ax_rh1.grid(alpha=0.3)

    # Suptitle
    suptitle = (f"Tile {tile_id} inspector   |   "
                 f"noise model: a={a_DC*nL_per_m3:.3f} nL/s, b={b_mult:.2f}")
    fig.suptitle(suptitle, fontsize=11, y=0.995)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _pearson_kurt(x):
    x = np.asarray(x); m = np.nanmean(x); s = np.nanstd(x)
    if s == 0 or not np.isfinite(s):
        return np.nan
    return float(np.nanmean(((x - m) / s) ** 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tile_id", type=int)
    ap.add_argument("--legacy", action="store_true",
                     help="use old DC+H1+KCL pipeline (b=0.29) instead of "
                          "production DC+H1+H2+per-channel pipeline")
    ap.add_argument("--a", type=float, default=A_DC_FGLS_DEFAULT,
                     help="(legacy only) additive floor in nL/s")
    ap.add_argument("--b", type=float, default=0.29,
                     help="(legacy only) multiplicative noise fraction")
    ap.add_argument("--n-outer", type=int, default=2,
                     help="(production only) number of outer FGLS passes")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print(f"Loading graph ...", flush=True)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    if args.legacy:
        # Legacy DC+H1+KCL pipeline
        a_SI = args.a / nL_per_m3
        noise_model = (a_SI, args.b)
        print(f"\n[legacy mode] DC+H1, a={args.a:.3f} nL/s, b={args.b:.2f}")
        prob = build_tile_problem(graph, args.tile_id)
        print(f"  {len(prob['edges_in'])} edges,  "
              f"{int(prob['valid_dc'].sum())} DC,  "
              f"{int(prob['valid_h1'].sum())} H1")
        n_edges = len(prob["edges_in"])
        Q_DC_abs = np.where(prob["valid_dc"], np.abs(prob["Q_DC_obs"]), 0)
        Q_H1_abs = np.where(prob["valid_h1"], np.abs(prob["Q_H1_obs"]), 0)
        sigma_dc_e = np.maximum(np.sqrt(a_SI**2 + (args.b*Q_DC_abs)**2), 1e-18)
        sigma_h1_e = np.maximum(np.sqrt(a_SI**2 + (args.b*Q_H1_abs)**2), 1e-18)
        Q_DC_clean = np.where(prob["valid_dc"], prob["Q_DC_obs"], 0)
        Q_H1_clean = np.where(prob["valid_h1"], prob["Q_H1_obs"], 0)
        t0 = time.time()
        lm = joint_lm(prob["sub"], prob["edges_in"],
                       prob["boundary_nodes"], prob["interior_nodes"],
                       Q_DC_clean, {1: Q_H1_clean},
                       sigma_dc_e, {1: sigma_h1_e},
                       ac_harmonics=HARMONICS, pin_dc=True,
                       pin_idx=prob["pin_idx"], D_init=1.3e-3, verbose=False)
        print(f"  D̂ = {lm['D_hat']:.3e}  σ_D/D̂ = "
              f"{lm['sigma_D']/lm['D_hat']:.2%}  "
              f"χ² = {lm['chi2']:.2f}  ({time.time()-t0:.1f}s)")
        full = fit_P_given_D(prob, lm["D_hat"], noise_model)
        full["P_DC"] = lm["P_DC"]; full["P_H1"] = lm["P_H"][1]
        print(f"  χ²_red = {full['chi2_total']/full['dof']:.2f}")
        prof_D, prof_chi2 = profile_likelihood(prob, noise_model)
        out = (Path(args.out) if args.out is not None
                else OUT_DIR / f"tile{args.tile_id:02d}_LEGACY"
                                f"_a{args.a:.3f}_b{args.b:.2f}.png")
        render(args.tile_id, prob, full, prof_D, prof_chi2,
                (a_SI, args.b), out, show=(not args.no_show))
        print(f"\nWrote {out}")
        return

    # === Production pipeline ===
    from production_fit import production_fit
    print(f"\n[production pipeline] DC+H1+H2, per-channel a_c, "
          f"{args.n_outer} outer FGLS pass(es)")
    t0 = time.time()
    res = production_fit(graph, args.tile_id, n_outer=args.n_outer)
    if res.get("error"):
        print(f"  FIT FAILED: {res['error']}")
        return
    prob = res["prob"]
    print(f"  {len(prob['edges_in'])} edges,  "
          f"{int(prob['valid_dc'].sum())} DC,  "
          f"{int(prob['valid_h1'].sum())} H1")
    print(f"  D̂ = {res['D_hat']:.3e} Pa^-1   σ_D = {res['sigma_D']:.3e}   "
          f"σ/D̂ = {res['rel_sigma_D']:.2%}")
    print(f"  iters = {res['iters']}   converged = {res['converged']}   "
          f"χ² = {res['chi2']:.2f}  ({time.time()-t0:.1f}s)")
    print(f"  Fitted a_DC = {res['a_DC_fit']:.3f}  "
          f"a_H1 = {res['a_H1_fit']:.3f}  "
          f"a_H2 = {res['a_H2_fit']:.3f} nL/s")

    # Build a `full` dict that matches what render() expects.  Use the
    # production fit's residuals and P̂.
    n_rows = (int(prob['valid_dc'].sum())
                + 2*int(prob['valid_h1'].sum()))
    if 2 in res['harmonics']:
        # add H2 row count (n_h2 = len(r_h[2]))
        n_h2 = len(res['r_h'][2])
        n_rows += 2 * n_h2
    n_params = 1 + (len(prob['boundary_nodes']) - 1) \
                + 2 * len(prob['boundary_nodes']) * len(res['harmonics'])
    dof = max(n_rows - n_params, 1)
    # Weighted residuals (using final σ_c floors)
    sig_dc_final = res['sigma_dc_final']
    sig_h_final = res['sigma_h_final']
    r_dc_w = res['r_dc'] / sig_dc_final[prob['valid_dc']]
    r_h1_w = res['r_h'][1] / sig_h_final[1][prob['valid_h1']]
    chi2_total = float(np.sum(r_dc_w**2) + np.sum(np.abs(r_h1_w)**2))
    if 2 in res['harmonics']:
        valid_h2_mask = np.array([not np.isnan(x) for x in res['r_h'][2].real])
        # We don't actually need to filter — r_h[2] already only contains valid edges
        r_h2_w = res['r_h'][2] / sig_h_final[2][:len(res['r_h'][2])]
        chi2_total += float(np.sum(np.abs(r_h2_w)**2))

    full = dict(
        P_DC=res['P_DC'], P_H1=res['P_H'][1],
        chi2_total=chi2_total, dof=dof,
        r_dc=r_dc_w, r_h1=r_h1_w,
        sigma_dc=sig_dc_final, sigma_h1=sig_h_final[1],
    )
    print(f"  χ²_total = {chi2_total:.2f}  dof = {dof}  "
          f"χ²_red = {chi2_total/dof:.2f}")

    # Profile likelihood — use the final fitted noise floors
    print("\nProfile-likelihood scan ...", flush=True)
    t0 = time.time()
    noise_for_profile = (res['a_DC_fit'] / nL_per_m3, 0.0)
    prof_D, prof_chi2 = profile_likelihood(prob, noise_for_profile)
    print(f"  scan done ({time.time()-t0:.1f}s)")

    out = (Path(args.out) if args.out is not None
            else OUT_DIR / f"tile{args.tile_id:02d}_PROD"
                            f"_aDC{res['a_DC_fit']:.3f}.png")
    render(args.tile_id, prob, full, prof_D, prof_chi2,
            (res['a_DC_fit']/nL_per_m3, 0.0), out, show=(not args.no_show))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
