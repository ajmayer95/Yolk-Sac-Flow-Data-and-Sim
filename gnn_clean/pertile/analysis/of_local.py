"""
Local-tangent optical-flow analysis for vessel flow measurement.

Uses cKDTree-based geometry to project Farneback OF vectors onto the
local tangent at each vessel pixel, then computes plug-flow Q(t).
Optionally refines the centerline position using a symmetry scan.

This is the same approach validated in the synthetic vessel PIV viewer.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


def _centerline_geometry_xy(pts: np.ndarray):
    """Compute arc lengths, unit tangents, and normals for a centerline.

    Parameters
    ----------
    pts : (N, 2) float — [row, col] or [y, x] ordered points

    Returns
    -------
    s        : (N,) cumulative arc length
    tangents : (N, 2) unit tangent vectors
    normals  : (N, 2) CW-rotated unit normals
    """
    pts = np.asarray(pts, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])

    raw = np.empty_like(pts)
    raw[1:-1] = pts[2:] - pts[:-2]
    raw[0]    = pts[1]  - pts[0]
    raw[-1]   = pts[-1] - pts[-2]
    nrm = np.sqrt((raw ** 2).sum(axis=1, keepdims=True))
    tangents = raw / np.maximum(nrm, 1e-8)

    # CW normal: (t_col, -t_row)
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    return s, tangents, normals


def build_vessel_geometry(
    centerline: np.ndarray,
    R: float,
    H: int,
    W: int,
) -> Dict:
    """Build per-pixel vessel geometry from a centerline.

    Parameters
    ----------
    centerline : (N, 2) float [row, col] in tile pixel space
    R : float — vessel radius in pixels
    H, W : int — frame dimensions

    Returns
    -------
    dict with keys:
        rr_in, cc_in  : (n_vessel,) int — vessel pixel coordinates
        rho_in        : (n_vessel,) float — signed perpendicular distance
        t_row_in, t_col_in : (n_vessel,) float — geometric tangent components
        n_row_in, n_col_in : (n_vessel,) float — normal components
        nearest       : (n_vessel,) int — index of nearest CL point
        cl_tree       : cKDTree of centerline
    """
    from scipy.spatial import cKDTree

    s_cl, tangents_cl, normals_cl = _centerline_geometry_xy(centerline)
    cl_tree = cKDTree(centerline)

    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                              cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree.query(pts_g)

    # Vessel mask: within R, excluding endpoints (no caps)
    mask = (
        (dist_g.reshape(H, W) <= R)
        & (nearest_g.reshape(H, W) > 0)
        & (nearest_g.reshape(H, W) < len(centerline) - 1)
    )

    rr_in, cc_in = np.where(mask)
    pts_in = np.column_stack([rr_in.astype(float), cc_in.astype(float)])
    _, nearest = cl_tree.query(pts_in)

    t_row_in = tangents_cl[nearest, 0]
    t_col_in = tangents_cl[nearest, 1]
    n_row_in = normals_cl[nearest, 0]
    n_col_in = normals_cl[nearest, 1]

    rho_in = ((rr_in - centerline[nearest, 0]) * n_row_in
              + (cc_in - centerline[nearest, 1]) * n_col_in)

    return dict(
        rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
        t_row_in=t_row_in, t_col_in=t_col_in,
        n_row_in=n_row_in, n_col_in=n_col_in,
        nearest=nearest, cl_tree=cl_tree,
        s_cl=s_cl, tangents_cl=tangents_cl, normals_cl=normals_cl,
        mask=mask,
    )


def refine_centerline_shift(
    flow_avg: np.ndarray,
    centerline: np.ndarray,
    R: float,
    H: int,
    W: int,
    max_shift: float = 10.0,
    n_steps: int = 21,
    dt: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Refine centerline position by scanning perpendicular shifts.

    Picks the shift that maximises the symmetry of the radial velocity
    profile (v(+ρ) ≈ v(-ρ)).

    Parameters
    ----------
    flow_avg : (H, W, 2) float — time-averaged Farneback flow field
    centerline : (N, 2) float [row, col]
    R : float — vessel radius in pixels
    H, W : int — frame dimensions
    max_shift : float — maximum shift to scan
    n_steps : int — number of shifts to test
    dt : float — frame gap used for flow

    Returns
    -------
    cl_refined : (N, 2) float — shifted centerline
    best_shift : float — applied perpendicular shift
    """
    _, _, normals_cl = _centerline_geometry_xy(centerline)
    shifts = np.linspace(-max_shift, max_shift, n_steps)
    best_shift = 0.0
    best_score = -np.inf

    for shift in shifts:
        cl_shifted = centerline + shift * normals_cl
        # Bounds check
        if (cl_shifted[:, 0].min() < 0 or cl_shifted[:, 0].max() >= H
                or cl_shifted[:, 1].min() < 0 or cl_shifted[:, 1].max() >= W):
            continue

        geom = build_vessel_geometry(cl_shifted, R, H, W)
        if len(geom['rr_in']) < 10:
            continue

        rho = geom['rho_in']
        rr_in = geom['rr_in']
        cc_in = geom['cc_in']
        t_r = geom['t_row_in']
        t_c = geom['t_col_in']

        # Project flow onto tangent
        v_ax = (flow_avg[rr_in, cc_in, 1] * t_r
                + flow_avg[rr_in, cc_in, 0] * t_c) / dt

        # Symmetry score: v(+ρ) vs v(-ρ) in radial bins
        n_half = 10
        edges = np.linspace(0, R, n_half + 1)
        v_pos = np.zeros(n_half)
        v_neg = np.zeros(n_half)
        for ib in range(n_half):
            lo, hi = edges[ib], edges[ib + 1]
            mp = (rho >= lo) & (rho < hi)
            mn = (rho >= -hi) & (rho < -lo)
            if mp.sum() >= 2:
                v_pos[ib] = float(np.mean(np.abs(v_ax[mp])))
            if mn.sum() >= 2:
                v_neg[ib] = float(np.mean(np.abs(v_ax[mn])))
        score = -float(np.mean((v_pos - v_neg) ** 2))

        if score > best_score:
            best_score = score
            best_shift = float(shift)

    cl_refined = centerline + best_shift * normals_cl
    return cl_refined, best_shift


def compute_of_qt_local(
    frames_u8: np.ndarray,
    centerline: np.ndarray,
    R_px: float,
    fb_winsize: int = 7,
    fb_levels: int = 3,
    fb_poly_n: int = 5,
    fb_poly_sigma: float = 1.1,
    fb_pyr_scale: float = 0.5,
    fb_iterations: int = 3,
    dt: int = 1,
    refine_cl: bool = True,
    n_warmup: int = 80,
    use_flow_tangent: bool = True,
) -> Dict:
    """Run Farneback OF with local-tangent projection and return Q(t).

    Parameters
    ----------
    frames_u8 : (T, H, W) uint8 — pre-processed video frames
    centerline : (N, 2) float [row, col] in tile pixel space
    R_px : float — vessel radius in pixels
    fb_* : Farneback parameters
    dt : int — frame gap
    refine_cl : bool — whether to refine centerline with symmetry scan
    n_warmup : int — frames for warmup (mean flow for refinement)
    use_flow_tangent : bool — use flow-derived tangent instead of geometric

    Returns
    -------
    dict with keys:
        Q_t       : (n_pairs,) float — per-frame plug-flow Q [px²/frame]
        Q_mean    : float — time-averaged Q
        v_mean    : float — time-averaged mean axial velocity
        cl_used   : (N, 2) — centerline actually used (possibly refined)
        cl_shift  : float — applied perpendicular shift (0 if not refined)
        profile_r : (n_bins,) — radial bin centers
        profile_v : (n_bins,) — time-averaged v(ρ)
    """
    import cv2

    T, H, W = frames_u8.shape
    fb_kwargs = dict(
        pyr_scale=fb_pyr_scale, levels=fb_levels,
        winsize=fb_winsize | 1, iterations=fb_iterations,
        poly_n=fb_poly_n, poly_sigma=fb_poly_sigma, flags=0,
    )

    cl_shift = 0.0
    cl_used = centerline.copy()

    # Warmup: compute mean flow for refinement + flow-derived tangent
    n_warmup = min(n_warmup, T - dt)
    if n_warmup > 0 and (refine_cl or use_flow_tangent):
        indices = np.linspace(0, T - dt - 1, min(n_warmup, T - dt), dtype=int)
        flow_sum = np.zeros((H, W, 2), dtype=np.float64)
        for idx in indices:
            flow = cv2.calcOpticalFlowFarneback(
                frames_u8[idx], frames_u8[idx + dt], None, **fb_kwargs)
            flow_sum += flow
        flow_avg = flow_sum / len(indices)

        # Refine centerline
        if refine_cl:
            cl_used, cl_shift = refine_centerline_shift(
                flow_avg, centerline, R_px, H, W, dt=float(dt))

    # Build geometry on (possibly refined) centerline
    geom = build_vessel_geometry(cl_used, R_px, H, W)
    rr_in = geom['rr_in']
    cc_in = geom['cc_in']
    rho_in = geom['rho_in']
    t_row_in = geom['t_row_in']
    t_col_in = geom['t_col_in']

    # Optionally replace geometric tangent with flow-derived tangent
    if use_flow_tangent and n_warmup > 0:
        flow_vr = flow_avg[rr_in, cc_in, 1]
        flow_vc = flow_avg[rr_in, cc_in, 0]
        flow_tang = np.column_stack([flow_vr, flow_vc])
        flow_mag = np.sqrt(flow_tang[:, 0] ** 2 + flow_tang[:, 1] ** 2)
        flow_mag_safe = np.maximum(flow_mag, 1e-8)
        flow_tang /= flow_mag_safe[:, None]

        # Consistent orientation with geometric tangent
        dot = flow_tang[:, 0] * t_row_in + flow_tang[:, 1] * t_col_in
        flow_tang[dot < 0] *= -1

        # Fall back to geometric where flow is too weak
        weak = flow_mag < 0.05
        flow_tang[weak, 0] = t_row_in[weak]
        flow_tang[weak, 1] = t_col_in[weak]

        t_row_in = flow_tang[:, 0]
        t_col_in = flow_tang[:, 1]

    # Per-frame Q(t) via plug-flow
    n_pairs = T - dt
    A_px2 = np.pi * R_px ** 2
    Q_t = np.full(n_pairs, np.nan)

    # Radial profile accumulators
    n_bins = 30
    bin_edges = np.linspace(-R_px, R_px, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile_sum = np.zeros(n_bins)
    profile_cnt = np.zeros(n_bins)
    bin_members = [
        (rho_in >= lo) & (rho_in < hi)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
    ]

    for t in range(n_pairs):
        flow = cv2.calcOpticalFlowFarneback(
            frames_u8[t], frames_u8[t + dt], None, **fb_kwargs)
        of_row = flow[rr_in, cc_in, 1] / dt
        of_col = flow[rr_in, cc_in, 0] / dt
        v_axial = of_row * t_row_in + of_col * t_col_in

        Q_t[t] = float(np.nanmean(v_axial)) * A_px2

        for i_bin, in_bin in enumerate(bin_members):
            vals = v_axial[in_bin]
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                profile_sum[i_bin] += float(np.mean(finite)) * int(in_bin.sum())
                profile_cnt[i_bin] += int(in_bin.sum())

    with np.errstate(invalid='ignore'):
        profile_v = np.where(profile_cnt > 0,
                             profile_sum / profile_cnt, np.nan)

    Q_mean = float(np.nanmean(Q_t))
    v_mean = Q_mean / A_px2 if A_px2 > 0 else 0.0

    return dict(
        Q_t=Q_t,
        Q_mean=Q_mean,
        v_mean=v_mean,
        cl_used=cl_used,
        cl_shift=cl_shift,
        profile_r=bin_centers,
        profile_v=profile_v,
        R_px=R_px,
        n_vessel_px=len(rr_in),
    )
