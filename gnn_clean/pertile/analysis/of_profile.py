"""Unified optical-flow velocity profile and Q(t) computation.

This module provides ``compute_of_profile_and_Q``, a standalone function that
runs the full Farneback-based flow-analysis pipeline on a vessel segment:

    frames  ->  background subtraction  ->  centerline geometry
            ->  (optional) centerline refinement from flow symmetry
            ->  (optional) flow-derived tangent field
            ->  per-frame OF + tangent projection
            ->  radial velocity profile  +  per-frame Q(t)

The function is designed to be called from:
  1. ``synthetic_vessel_piv_app.py`` (synthetic viewer)
  2. ``mosaic/_kirchhoff.py  _on_analyze_click_of`` (mosaic analyze-click)
  3. ``mosaic/_kirchhoff.py  _kirchhoff_analyze_edge`` (Kirchhoff test)

No viewer / Qt dependency is required; the function gracefully skips
``QApplication.processEvents()`` when Qt is unavailable.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from .synthetic_video import _centerline_geometry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_events():
    """Call QApplication.processEvents() if Qt is available (non-blocking)."""
    try:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


def _normalise_for_of(frames, bg_percentile=10):
    """Background-subtract float frames and convert to uint8 for Farneback.

    If *frames* is already uint8 the data is returned as-is.  For float input
    a low percentile (default 10th) is used instead of the temporal mean so
    that steady-flow signal is preserved.
    """
    if frames.dtype == np.uint8:
        return frames
    if frames.size == 0 or frames.shape[0] == 0:
        return np.zeros_like(frames, dtype=np.uint8)
    bg = np.percentile(frames, bg_percentile, axis=0)  # (H, W)
    frames_bt = frames - bg
    vmin = float(np.percentile(frames_bt, 1))
    vmax = float(np.percentile(frames_bt, 99))
    frames_norm = np.clip((frames_bt - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    return (frames_norm * 255).astype(np.uint8)


def _build_vessel_mask(centerline, vessel_radius_px, H, W):
    """Build a vessel mask (distance <= R, excluding endpoint caps).

    Returns
    -------
    rr_in, cc_in : int arrays — row/col indices of vessel pixels
    rho_in       : float array — signed perpendicular distance from centreline
    t_row_in, t_col_in : float arrays — local tangent components
    n_row_in, n_col_in : float arrays — local normal components
    nearest      : int array — index of nearest centreline point per pixel
    """
    s_cl, tangents_cl, normals_cl = _centerline_geometry(centerline)
    cl_tree = cKDTree(centerline)

    # All image pixels
    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                             cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree.query(pts_g)

    # Mask: within R and not at endpoint caps (nearest != first or last CL pt)
    mask = ((dist_g.reshape(H, W) <= vessel_radius_px)
            & (nearest_g.reshape(H, W) > 0)
            & (nearest_g.reshape(H, W) < len(centerline) - 1))

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
        nearest=nearest,
    )


def _refine_centerline(
    frames_u8, cl_prior, R_prior, fb_kwargs, dt=1,
    n_warmup=80, max_shift=10.0, n_shift_steps=21,
    f0_hz=None, frame_dt_s=None,
    tangent_smooth_sigma=5.0,
    use_flow_tangent=True,
):
    """Refine an approximate centerline using OF-derived symmetry.

    Steps:
      1. Compute mean flow field (systolic frames preferred when f0 known).
      2. Scan perpendicular shifts — velocity-weighted symmetry score +
         sub-pixel parabolic interpolation.
      3. Optionally derive per-pixel tangent from flow direction, smoothed
         along the arc with Gaussian sigma.

    Returns dict with:
        cl_refined, shift_applied,
        rr_in, cc_in, rho_in, t_row_in, t_col_in,
        n_vessel_px, vfield_mean (H, W, 2)
    """
    import cv2

    T, H, W = frames_u8.shape
    n_warmup = min(n_warmup, T - dt)

    # -- 1. Mean flow field -- prefer systolic frames for best SNR -----------
    if f0_hz is not None and frame_dt_s is not None and f0_hz > 0:
        phi = (np.arange(T - dt) * frame_dt_s * f0_hz) % 1.0
        systolic = phi < 0.3  # first 30% of cardiac cycle
        systolic_idx = np.where(systolic)[0]
        if len(systolic_idx) >= n_warmup // 2:
            indices = systolic_idx[
                np.linspace(0, len(systolic_idx) - 1, n_warmup, dtype=int)]
        else:
            indices = np.linspace(0, T - dt - 1, n_warmup, dtype=int)
    else:
        indices = np.linspace(0, T - dt - 1, n_warmup, dtype=int)

    flow_sum = np.zeros((H, W, 2), dtype=np.float64)
    for idx in indices:
        flow = cv2.calcOpticalFlowFarneback(
            frames_u8[idx], frames_u8[idx + dt], None, **fb_kwargs)
        flow_sum += flow
    flow_avg = flow_sum / len(indices)

    # Geometry of prior centreline
    s_cl, tangents_cl, normals_cl = _centerline_geometry(cl_prior)

    # Image grid (shared across shifts)
    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                             cc_g.ravel().astype(float)])

    # -- 2. Infer centerline from velocity ridge -----------------------------
    cl_refined, offsets_raw, offsets_smooth = _infer_centerline_from_velocity_ridge(
        flow_avg, cl_prior, R_prior, H, W, dt=dt, smooth_sigma=8.0,
    )
    best_shift = float(np.mean(offsets_smooth))  # mean offset for reporting
    s_cl_r, tang_r, norm_r = _centerline_geometry(cl_refined)
    cl_tree_r = cKDTree(cl_refined)

    dist_g, nearest_g = cl_tree_r.query(pts_g)
    mask_r = ((dist_g.reshape(H, W) <= R_prior * 1.2)
              & (nearest_g.reshape(H, W) > 0)
              & (nearest_g.reshape(H, W) < len(cl_refined) - 1))

    rr_in, cc_in = np.where(mask_r)
    pts_in = np.column_stack([rr_in.astype(float), cc_in.astype(float)])
    _, nearest = cl_tree_r.query(pts_in)

    # -- Flow-derived tangent, smoothed along arc ----------------------------
    if use_flow_tangent:
        flow_vr = flow_avg[rr_in, cc_in, 1]
        flow_vc = flow_avg[rr_in, cc_in, 0]
        flow_tang = np.column_stack([flow_vr, flow_vc])
        flow_mag = np.sqrt(flow_tang[:, 0] ** 2 + flow_tang[:, 1] ** 2)
        flow_mag_safe = np.maximum(flow_mag, 1e-8)
        flow_tang /= flow_mag_safe[:, None]

        # Ensure consistent orientation with geometric tangent
        cl_tang = tang_r[nearest]
        dot = (flow_tang * cl_tang).sum(axis=1)
        flow_tang[dot < 0] *= -1

        # Fall back to geometric tangent where flow is too weak
        weak = flow_mag < 0.03
        flow_tang[weak] = cl_tang[weak]

        # Smooth tangent field along arc: aggregate per CL point, Gaussian
        # smooth along arc, map back to vessel pixels.
        n_cl = len(cl_refined)
        tang_cl_sum = np.zeros((n_cl, 2))
        tang_cl_w = np.zeros(n_cl)
        for k in range(len(nearest)):
            j = nearest[k]
            w = float(flow_mag[k])
            tang_cl_sum[j] += w * flow_tang[k]
            tang_cl_w[j] += w
        has_data = tang_cl_w > 1e-8
        tang_cl_avg = np.copy(tang_r)  # start with geometric
        tang_cl_avg[has_data] = (tang_cl_sum[has_data]
                                 / tang_cl_w[has_data, None])
        if tangent_smooth_sigma > 0 and n_cl > 3:
            tang_cl_avg[:, 0] = gaussian_filter1d(
                tang_cl_avg[:, 0], tangent_smooth_sigma)
            tang_cl_avg[:, 1] = gaussian_filter1d(
                tang_cl_avg[:, 1], tangent_smooth_sigma)
            nrm = np.sqrt((tang_cl_avg ** 2).sum(axis=1, keepdims=True))
            tang_cl_avg /= np.maximum(nrm, 1e-8)
        flow_tang = tang_cl_avg[nearest]

        t_row_in = flow_tang[:, 0]
        t_col_in = flow_tang[:, 1]
    else:
        t_row_in = tang_r[nearest, 0]
        t_col_in = tang_r[nearest, 1]

    n_row_in = t_col_in.copy()   # normal = (t_col, -t_row)
    n_col_in = -t_row_in.copy()

    rho_in = ((rr_in - cl_refined[nearest, 0]) * n_row_in
              + (cc_in - cl_refined[nearest, 1]) * n_col_in)

    # Mean velocity field for diagnostics
    vfield_mean = np.zeros((H, W, 2), dtype=np.float64)
    vfield_mean[:, :, 0] = flow_avg[:, :, 0] / max(dt, 1)
    vfield_mean[:, :, 1] = flow_avg[:, :, 1] / max(dt, 1)

    return dict(
        cl_refined=cl_refined,
        shift_applied=best_shift,
        rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
        t_row_in=t_row_in, t_col_in=t_col_in,
        n_vessel_px=int(len(rr_in)),
        vfield_mean=vfield_mean,
    )


def _infer_centerline_from_velocity_ridge(
    flow_avg, cl_prior, R_prior, H, W, dt=1, smooth_sigma=8.0,
):
    """Infer the vessel centerline from the velocity field ridge.

    At each arc position along the prior centerline, take a perpendicular
    slice of the OF velocity field and find the ρ where |v_axial| is maximum.
    Smooth the resulting offsets and build a refined centerline.

    Parameters
    ----------
    flow_avg : (H, W, 2) mean flow field
    cl_prior : (N, 2) approximate centerline [row, col]
    R_prior : float vessel radius
    H, W : image dimensions
    dt : frame step used for flow_avg
    smooth_sigma : float, Gaussian σ for smoothing the ridge along arc (px)

    Returns
    -------
    cl_ridge : (N, 2) refined centerline
    offsets_raw : (N,) raw perpendicular offset at each CL point (before smooth)
    offsets_smooth : (N,) smoothed offsets
    """
    s_cl, tangents_cl, normals_cl = _centerline_geometry(cl_prior)
    N = len(cl_prior)

    # Sample perpendicular slices at each centerline point
    n_samples = int(2 * R_prior + 6)  # sample ρ from -R-3 to R+3
    rho_range = np.linspace(-R_prior - 3, R_prior + 3, n_samples)

    offsets_raw = np.zeros(N)

    for i in range(N):
        row_c, col_c = cl_prior[i]
        n_r, n_c = normals_cl[i]
        t_r, t_c = tangents_cl[i]

        # Sample positions along perpendicular
        rows_s = row_c + rho_range * n_r
        cols_s = col_c + rho_range * n_c

        # Check bounds
        in_bounds = ((rows_s >= 0) & (rows_s < H - 1) &
                     (cols_s >= 0) & (cols_s < W - 1))
        if in_bounds.sum() < 5:
            continue

        # Bilinear interpolation of flow magnitude at sample points
        v_mag = np.zeros(n_samples)
        for j in range(n_samples):
            if not in_bounds[j]:
                continue
            r_f, c_f = float(rows_s[j]), float(cols_s[j])
            r0, c0 = int(r_f), int(c_f)
            r1, c1 = min(r0 + 1, H - 1), min(c0 + 1, W - 1)
            fr, fc = r_f - r0, c_f - c0

            vx = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 0] +
                  fr * (1 - fc) * flow_avg[r1, c0, 0] +
                  (1 - fr) * fc * flow_avg[r0, c1, 0] +
                  fr * fc * flow_avg[r1, c1, 0])
            vy = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 1] +
                  fr * (1 - fc) * flow_avg[r1, c0, 1] +
                  (1 - fr) * fc * flow_avg[r0, c1, 1] +
                  fr * fc * flow_avg[r1, c1, 1])
            v_mag[j] = np.sqrt(vx ** 2 + vy ** 2) / max(dt, 1)

        # Find ρ with maximum |v| (within |ρ| < R)
        inside = in_bounds & (np.abs(rho_range) < R_prior)
        if inside.sum() < 3:
            continue
        v_inside = v_mag.copy()
        v_inside[~inside] = -1
        best_j = int(np.argmax(v_inside))

        # Sub-pixel parabolic refinement
        if 0 < best_j < n_samples - 1 and v_inside[best_j] > 0:
            vm, v0, vp = v_inside[best_j - 1], v_inside[best_j], v_inside[best_j + 1]
            denom = vm - 2 * v0 + vp
            if abs(denom) > 1e-12 and vm < v0 and vp < v0:
                delta = 0.5 * (vm - vp) / denom
                delta = np.clip(delta, -0.5, 0.5)
                step = rho_range[1] - rho_range[0]
                offsets_raw[i] = rho_range[best_j] + delta * step
            else:
                offsets_raw[i] = rho_range[best_j]
        else:
            offsets_raw[i] = rho_range[best_j]

    # Cap raw offsets to ±R/3 — the center shouldn't shift by more than
    # a third of the radius.  Large offsets indicate ridge-finder failure.
    max_offset = R_prior / 3.0
    offsets_raw = np.clip(offsets_raw, -max_offset, max_offset)

    # Smooth offsets along arc
    if smooth_sigma > 0 and N > 3:
        offsets_smooth = gaussian_filter1d(offsets_raw, smooth_sigma)
    else:
        offsets_smooth = offsets_raw.copy()

    # Clamp smoothed offsets too (smoothing can push past cap)
    offsets_smooth = np.clip(offsets_smooth, -max_offset, max_offset)

    # Safety: if mean offset is large (> R/4), the ridge is unreliable.
    # Fall back to zero offset (keep prior centerline).
    mean_abs_offset = float(np.mean(np.abs(offsets_smooth)))
    if mean_abs_offset > R_prior / 4.0:
        offsets_smooth[:] = 0.0

    # Build refined centerline
    cl_ridge = cl_prior + offsets_smooth[:, None] * normals_cl

    return cl_ridge, offsets_raw, offsets_smooth


def _compute_flow_tangent(
    flow_avg, rr_in, cc_in, nearest, tang_geom, centerline,
    tangent_smooth_sigma=5.0, dt=1,
):
    """Derive per-pixel tangent from mean flow field, smoothed along arc.

    Parameters
    ----------
    flow_avg : (H, W, 2) mean flow field (OpenCV convention: [dx, dy])
    rr_in, cc_in : vessel pixel coordinates
    nearest : index of nearest centreline point per vessel pixel
    tang_geom : (N_cl, 2) geometric tangents from _centerline_geometry
    centerline : (N_cl, 2) centreline points
    tangent_smooth_sigma : Gaussian smoothing sigma along arc (CL points)
    dt : frame step (for normalising flow)

    Returns
    -------
    t_row, t_col : (n_vessel,) tangent components
    """
    flow_vr = flow_avg[rr_in, cc_in, 1]  # row component
    flow_vc = flow_avg[rr_in, cc_in, 0]  # col component
    flow_tang = np.column_stack([flow_vr, flow_vc])
    flow_mag = np.sqrt(flow_tang[:, 0] ** 2 + flow_tang[:, 1] ** 2)
    flow_mag_safe = np.maximum(flow_mag, 1e-8)
    flow_tang /= flow_mag_safe[:, None]

    # Ensure consistent orientation with geometric tangent
    cl_tang = tang_geom[nearest]
    dot = (flow_tang * cl_tang).sum(axis=1)
    flow_tang[dot < 0] *= -1

    # Fall back to geometric tangent where flow is too weak
    weak = flow_mag < 0.03
    flow_tang[weak] = cl_tang[weak]

    # Smooth tangent field along arc
    n_cl = len(centerline)
    tang_cl_sum = np.zeros((n_cl, 2))
    tang_cl_w = np.zeros(n_cl)
    for k in range(len(nearest)):
        j = nearest[k]
        w = float(flow_mag[k])
        tang_cl_sum[j] += w * flow_tang[k]
        tang_cl_w[j] += w
    has_data = tang_cl_w > 1e-8
    tang_cl_avg = np.copy(tang_geom)
    tang_cl_avg[has_data] = tang_cl_sum[has_data] / tang_cl_w[has_data, None]
    if tangent_smooth_sigma > 0 and n_cl > 3:
        tang_cl_avg[:, 0] = gaussian_filter1d(
            tang_cl_avg[:, 0], tangent_smooth_sigma)
        tang_cl_avg[:, 1] = gaussian_filter1d(
            tang_cl_avg[:, 1], tangent_smooth_sigma)
        nrm = np.sqrt((tang_cl_avg ** 2).sum(axis=1, keepdims=True))
        tang_cl_avg /= np.maximum(nrm, 1e-8)

    flow_tang_smooth = tang_cl_avg[nearest]
    return flow_tang_smooth[:, 0], flow_tang_smooth[:, 1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_of_profile_and_Q(
    frames,                    # (T, H, W) uint8 or float — raw video frames
    centerline,                # (N, 2) float [row, col] — vessel centreline
    vessel_radius_px,          # float — vessel radius in pixels
    *,
    # Background subtraction
    bg_percentile=10,          # percentile for background estimation (0=skip)
    # Farneback params
    fb_winsize=5,
    fb_levels=1,
    fb_iterations=3,
    fb_poly_n=5,
    fb_poly_sigma=1.1,
    fb_pyr_scale=0.5,
    dt=1,                      # frame step
    # Profile
    n_bins=30,
    fit_fraction=0.75,         # fit Poiseuille to central fraction of R
    v_min_threshold=0.03,      # exclude pixels with |v_axial| below this (px/fr)
    # Centreline refinement
    refine_centerline=False,   # symmetry-based shift scan
    max_shift=10.0,
    n_shift_steps=21,
    # Tangent
    use_flow_tangent=True,     # use flow-derived tangent (vs geometric)
    tangent_smooth_sigma=5.0,  # smooth tangent along arc
    # Cardiac phase (for systolic frame selection in refinement)
    f0_hz=None,
    frame_dt_s=None,
):
    """Run the full Farneback optical-flow pipeline on a vessel segment.

    Parameters
    ----------
    frames : (T, H, W) array
        Video frames.  If float, background subtraction + uint8 conversion is
        applied.  If already uint8, used as-is.
    centerline : (N, 2) float array  [row, col]
        Vessel centreline in frame coordinates.
    vessel_radius_px : float
        Vessel radius in pixels.
    bg_percentile : int
        Percentile for background estimation (0 to skip).
    fb_winsize, fb_levels, fb_iterations, fb_poly_n, fb_poly_sigma, fb_pyr_scale
        Farneback optical flow parameters.
    dt : int
        Frame step for consecutive OF pairs.
    n_bins : int
        Number of radial bins for the velocity profile.
    fit_fraction : float
        Fraction of R to use for the central Poiseuille fit (0 < f <= 1).
    v_min_threshold : float
        Minimum |v_axial| (px/frame) to include a pixel in profile bins.
        Pixels below this are likely Farneback returning ~zero due to no
        local gradient (no particle). Set to 0 to disable.
    refine_centerline : bool
        If True, run symmetry-based perpendicular shift scan to refine the
        centreline position.
    max_shift, n_shift_steps : float, int
        Shift scan range and resolution (only used when refine_centerline=True).
    use_flow_tangent : bool
        If True, derive per-pixel tangent from the mean flow field and smooth
        along the arc.  Otherwise use the geometric tangent from the centreline.
    tangent_smooth_sigma : float
        Gaussian sigma (in centreline points) for smoothing the flow-derived
        tangent along the arc.
    f0_hz : float or None
        Heart rate.  Used for systolic-frame selection during refinement.
    frame_dt_s : float or None
        Inter-frame interval in seconds.

    Returns
    -------
    dict with keys:
        Q_plug_t        (n_pairs,)  per-frame plug-flow Q = mean(v_axial) * pi*R^2
        Q_pois_t        (n_pairs,)  per-frame Poiseuille Q = v0(t) * pi*R^2 / 2
        Q_plug_mean     float       time-averaged plug-flow Q
        Q_pois_mean     float       time-averaged Poiseuille Q
        v0_fixed        float       centreline velocity from R-fixed fit
        v0_free         float       centreline velocity from R-free fit
        R_fit_free      float       flow-derived radius from R-free fit
        bin_centers     (n_bins,)   radial bin centres
        profile_v       (n_bins,)   time-averaged radial velocity profile
        cl_used         (N, 2)      centreline actually used (may be refined)
        cl_shift        float       shift applied (0 if no refinement)
        rr_in, cc_in    int arrays  vessel pixel coordinates
        rho_in          float array signed perpendicular distances
        t_row_in, t_col_in  float   tangent components at vessel pixels
        vfield_mean     (H, W, 2)   time-averaged velocity field
        n_vessel_px     int
        fb_winsize      int         (for downstream smoothing correction)
    """
    import cv2

    frames = np.asarray(frames)
    centerline = np.asarray(centerline, dtype=float)
    R = float(vessel_radius_px)
    T, H_full, W_full = frames.shape

    # Force odd winsize
    fb_winsize = int(fb_winsize) | 1

    # ── 0. Crop to ROI around centreline (2×R margin) ────────────────────
    margin = int(np.ceil(R * 2)) + 4
    r0 = max(0, int(np.floor(centerline[:, 0].min())) - margin)
    r1 = min(H_full, int(np.ceil(centerline[:, 0].max())) + margin + 1)
    c0 = max(0, int(np.floor(centerline[:, 1].min())) - margin)
    c1 = min(W_full, int(np.ceil(centerline[:, 1].max())) + margin + 1)
    frames = frames[:, r0:r1, c0:c1]
    centerline = centerline - np.array([[r0, c0]], dtype=float)
    T, H, W = frames.shape

    fb_kwargs = dict(
        pyr_scale=fb_pyr_scale, levels=fb_levels,
        winsize=fb_winsize, iterations=fb_iterations,
        poly_n=fb_poly_n, poly_sigma=fb_poly_sigma, flags=0,
    )

    # ── 1. Background subtraction ──────────────────────────────────────────
    if frames.dtype != np.uint8 and bg_percentile > 0:
        frames_u8 = _normalise_for_of(frames, bg_percentile)
    elif frames.dtype == np.uint8:
        frames_u8 = frames
    else:
        # Float frames but bg_percentile==0 — just rescale to uint8
        vmin = float(np.percentile(frames, 1))
        vmax = float(np.percentile(frames, 99))
        frames_u8 = np.clip(
            (frames - vmin) / max(vmax - vmin, 1e-8) * 255, 0, 255
        ).astype(np.uint8)

    # ── 2. Centreline geometry / optional refinement ───────────────────────
    cl_shift = 0.0
    cl_used = centerline.copy()

    if refine_centerline:
        geom = _refine_centerline(
            frames_u8, centerline, R, fb_kwargs, dt=dt,
            max_shift=max_shift, n_shift_steps=n_shift_steps,
            f0_hz=f0_hz, frame_dt_s=frame_dt_s,
            tangent_smooth_sigma=tangent_smooth_sigma,
            use_flow_tangent=use_flow_tangent,
        )
        cl_used = geom['cl_refined']
        cl_shift = geom['shift_applied']
        rr_in = geom['rr_in']
        cc_in = geom['cc_in']
        rho_in = geom['rho_in']
        t_row_in = geom['t_row_in']
        t_col_in = geom['t_col_in']
        n_vessel_px = geom['n_vessel_px']
    else:
        # Build vessel mask from centreline + R
        geom = _build_vessel_mask(cl_used, R, H, W)
        rr_in = geom['rr_in']
        cc_in = geom['cc_in']
        rho_in = geom['rho_in']
        t_row_in = geom['t_row_in']
        t_col_in = geom['t_col_in']
        nearest = geom['nearest']
        n_vessel_px = int(len(rr_in))

        # Optionally replace geometric tangent with flow-derived tangent
        if use_flow_tangent:
            # Compute mean flow from ~80 frames for tangent estimation
            n_warmup = min(80, T - dt)
            if f0_hz is not None and frame_dt_s is not None and f0_hz > 0:
                phi = (np.arange(T - dt) * frame_dt_s * f0_hz) % 1.0
                systolic = phi < 0.3
                systolic_idx = np.where(systolic)[0]
                if len(systolic_idx) >= n_warmup // 2:
                    indices = systolic_idx[
                        np.linspace(0, len(systolic_idx) - 1,
                                    n_warmup, dtype=int)]
                else:
                    indices = np.linspace(0, T - dt - 1, n_warmup, dtype=int)
            else:
                indices = np.linspace(0, T - dt - 1, n_warmup, dtype=int)

            flow_sum = np.zeros((H, W, 2), dtype=np.float64)
            for idx in indices:
                flow = cv2.calcOpticalFlowFarneback(
                    frames_u8[idx], frames_u8[idx + dt], None, **fb_kwargs)
                flow_sum += flow
            flow_avg = flow_sum / len(indices)

            _, tang_geom, _ = _centerline_geometry(cl_used)
            t_row_in, t_col_in = _compute_flow_tangent(
                flow_avg, rr_in, cc_in, nearest, tang_geom, cl_used,
                tangent_smooth_sigma=tangent_smooth_sigma, dt=dt,
            )

    if n_vessel_px < 10:
        # Not enough vessel pixels — return NaN result
        nan = float('nan')
        return dict(
            Q_plug_t=np.array([]), Q_pois_t=np.array([]),
            Q_plug_mean=nan, Q_pois_mean=nan,
            v0_fixed=nan, v0_free=nan, R_fit_free=R,
            bin_centers=np.array([]), profile_v=np.array([]),
            cl_used=cl_used, cl_shift=cl_shift,
            rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
            t_row_in=t_row_in, t_col_in=t_col_in,
            vfield_mean=np.zeros((H, W, 2)), n_vessel_px=n_vessel_px,
            fb_winsize=fb_winsize,
        )

    # ── 3. Per-frame OF ────────────────────────────────────────────────────
    A_px2 = np.pi * R ** 2
    n_pairs = T - dt

    # Radial bin edges for profile (signed rho from centreline)
    bin_edges = np.linspace(-R, R, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pre-build bin membership for per-frame R-fixed fit
    bin_members = [
        (rho_in >= lo) & (rho_in < hi)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
    ]

    # Accumulators
    profile_sum = np.zeros(n_bins)
    profile_cnt = np.zeros(n_bins)
    Q_plug_t = np.full(n_pairs, np.nan)
    per_frame_v0_fixed = np.full(n_pairs, np.nan)

    # Time-averaged velocity field
    vfield_sum = np.zeros((H, W, 2), dtype=np.float64)
    vfield_cnt = np.zeros((H, W), dtype=np.float64)

    r_cut = R * fit_fraction  # radial cutoff for central Poiseuille fit

    for t in range(n_pairs):
        if t % 50 == 0:
            _process_events()

        flow = cv2.calcOpticalFlowFarneback(
            frames_u8[t], frames_u8[t + dt], None, **fb_kwargs)

        # Project OF vectors onto local tangent -> axial velocity
        # flow[:,:,0] = dx (col), flow[:,:,1] = dy (row)
        of_row = flow[rr_in, cc_in, 1] / dt   # (n_vessel,)
        of_col = flow[rr_in, cc_in, 0] / dt   # (n_vessel,)
        v_axial = of_row * t_row_in + of_col * t_col_in  # (n_vessel,)

        # Accumulate raw flow vectors at vessel pixels
        vfield_sum[rr_in, cc_in, 0] += flow[rr_in, cc_in, 0] / dt
        vfield_sum[rr_in, cc_in, 1] += flow[rr_in, cc_in, 1] / dt
        vfield_cnt[rr_in, cc_in] += 1.0

        # Mask: only include pixels where OF detected real motion
        of_mag = np.sqrt(of_row ** 2 + of_col ** 2)
        active = of_mag >= v_min_threshold   # (n_vessel,) bool

        # Plug-flow Q: mean tangential velocity x area (active pixels only)
        if active.any():
            Q_plug_t[t] = float(np.nanmean(v_axial[active])) * A_px2

        # Accumulate into radial rho bins (active pixels only)
        for i_bin, in_bin in enumerate(bin_members):
            mask = in_bin & active
            vals = v_axial[mask]
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                profile_sum[i_bin] += float(np.mean(finite)) * int(mask.sum())
                profile_cnt[i_bin] += int(mask.sum())

        # Per-frame R-fixed Poiseuille fit: v = v0 * (1 - r^2/R^2)
        # Fit only to central fraction (|r| < R * fit_fraction), active only
        r_valid_t, v_valid_t = [], []
        for i_bin, in_bin in enumerate(bin_members):
            if abs(bin_centers[i_bin]) > r_cut:
                continue
            mask = in_bin & active
            vals = v_axial[mask]
            finite = vals[np.isfinite(vals)]
            if len(finite) >= 2:
                r_valid_t.append(bin_centers[i_bin])
                v_valid_t.append(float(np.mean(finite)))
        if len(r_valid_t) >= 3:
            rv = np.array(r_valid_t)
            vv = np.array(v_valid_t)
            A_mat = (1.0 - (rv / R) ** 2)[:, None]
            try:
                v0, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
                per_frame_v0_fixed[t] = float(v0[0])
            except Exception:
                pass

    # ── 4. Time-averaged radial profile ────────────────────────────────────
    with np.errstate(invalid='ignore'):
        profile_v = np.where(profile_cnt > 0,
                             profile_sum / profile_cnt, np.nan)

    # ── 5. Time-averaged velocity field ────────────────────────────────────
    with np.errstate(invalid='ignore', divide='ignore'):
        vfield_mean = np.where(
            vfield_cnt[..., None] > 0,
            vfield_sum / vfield_cnt[..., None], 0.0)

    # ── 6. Time-averaged Poiseuille fits ───────────────────────────────────
    valid = np.isfinite(profile_v)
    v0_fixed = np.nan
    v0_free = np.nan
    R_fit_free = R

    if valid.sum() >= 3:
        rv = bin_centers[valid]
        vv = profile_v[valid]

        # R-fixed linear fit: v = v0 * (1 - r^2/R^2)
        A_mat = (1.0 - (rv / R) ** 2)[:, None]
        try:
            coeffs, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
            v0_fixed = float(coeffs[0])
        except Exception:
            pass

        # R-free nonlinear fit
        def pois_model(r, v0, Rf):
            return v0 * np.maximum(0.0, 1.0 - (r / Rf) ** 2)

        try:
            p0 = [max(vv.max(), 1e-3), R]
            popt, _ = curve_fit(
                pois_model, rv, vv, p0=p0,
                bounds=([0, R * 0.3], [np.inf, R * 3.0]),
                maxfev=2000,
            )
            v0_free, R_fit_free = float(popt[0]), float(popt[1])
        except Exception:
            pass

        # Smoothed-Poiseuille 3-parameter fit: v(r) = v₀ × max(0, 1-(r²+σ²)/R²)
        # With reduced smoothing (winsize=3, poly_n=3), σ is small enough
        # (~1-2px) that the R-σ degeneracy is manageable.
        # σ bounded to [0.5, 3.0] — physical range for Farneback with small windows.
        # R bounded to [0.85R, 1.15R] — segmentation is close but not exact.
        # Fit to central 75% only (avoids wall artifacts).
        v0_deconv = np.nan
        R_deconv = R
        sigma_of_fit = np.nan
        fit_mask_dec = np.abs(rv) <= R * fit_fraction
        if fit_mask_dec.sum() >= 5:
            rv_dec = rv[fit_mask_dec]
            vv_dec = vv[fit_mask_dec]
            try:
                def pois_smooth(r, v0, Rf, sigma):
                    return v0 * np.maximum(0.0, 1.0 - (r ** 2 + sigma ** 2) / Rf ** 2)

                sigma_guess = min(fb_winsize * 0.7, 2.0)
                v_peak = float(vv_dec.max())
                v0_guess = v_peak / max(1.0 - sigma_guess ** 2 / R ** 2, 0.3)
                p0s = [v0_guess, R, sigma_guess]
                popt_s, _ = curve_fit(
                    pois_smooth, rv_dec, vv_dec, p0=p0s,
                    bounds=([v_peak * 0.5, R * 0.85, 0.5],
                            [v_peak * 5.0, R * 1.15, 3.0]),
                    maxfev=3000,
                )
                v0_deconv = float(popt_s[0])
                R_deconv = float(popt_s[1])
                sigma_of_fit = float(popt_s[2])
            except Exception:
                pass

    # ── 7. Per-frame Q from Poiseuille fit (R fixed) ───────────────────────
    Q_pois_t = per_frame_v0_fixed * A_px2 / 2.0

    Q_plug_mean = float(np.nanmean(Q_plug_t))
    Q_pois_mean = float(np.nanmean(Q_pois_t))

    # Q from deconvolved fit: uses recovered true v₀ and true R
    Q_deconv_mean = np.nan
    if np.isfinite(v0_deconv) and np.isfinite(R_deconv):
        Q_deconv_mean = float(v0_deconv * np.pi * R_deconv ** 2 / 2.0)

    # ── 7. Poiseuille fit with ORIGINAL centerline (before ridge refinement) ─
    # This lets us compare: how much does the CL shift help?
    Q_pois_orig_mean = np.nan
    v0_orig = np.nan
    profile_v_orig = np.full(n_bins, np.nan)
    if refine_centerline and abs(cl_shift) > 0.1:
        # Re-bin the OF velocity data using original (un-shifted) CL geometry
        geom_orig = _build_vessel_mask(centerline, R, H, W)
        if geom_orig is not None and len(geom_orig['rr_in']) >= 10:
            rho_orig = geom_orig['rho_in']
            rr_orig = geom_orig['rr_in']
            cc_orig = geom_orig['cc_in']
            t_row_orig = geom_orig['t_row_in']
            t_col_orig = geom_orig['t_col_in']

            # Use the same mean velocity field to compute profile with orig CL
            v_ax_orig = (vfield_mean[rr_orig, cc_orig, 1] * t_row_orig
                         + vfield_mean[rr_orig, cc_orig, 0] * t_col_orig)

            # Bin by rho_orig
            for i_bin in range(n_bins):
                lo, hi = bin_edges[i_bin], bin_edges[i_bin + 1]
                in_bin = (rho_orig >= lo) & (rho_orig < hi)
                vals = v_ax_orig[in_bin]
                finite = vals[np.isfinite(vals)]
                if len(finite) >= 2:
                    profile_v_orig[i_bin] = float(np.mean(finite))

            # Poiseuille fit to central fraction
            valid_orig = np.isfinite(profile_v_orig)
            fit_orig = valid_orig & (np.abs(bin_centers) <= r_cut)
            if fit_orig.sum() >= 3:
                rv_o = bin_centers[fit_orig]
                vv_o = profile_v_orig[fit_orig]
                A_mat_o = (1.0 - (rv_o / R) ** 2)[:, None]
                try:
                    c_o, *_ = np.linalg.lstsq(A_mat_o, vv_o, rcond=None)
                    v0_orig = float(c_o[0])
                    Q_pois_orig_mean = float(v0_orig * A_px2 / 2.0)
                except Exception:
                    pass
    else:
        # No refinement applied — original = refined
        Q_pois_orig_mean = Q_pois_mean
        v0_orig = v0_fixed
        profile_v_orig = profile_v.copy()

    return dict(
        Q_plug_t=Q_plug_t,
        Q_pois_t=Q_pois_t,
        Q_plug_mean=Q_plug_mean,
        Q_pois_mean=Q_pois_mean,
        Q_deconv_mean=Q_deconv_mean,
        v0_fixed=v0_fixed,
        v0_free=v0_free,
        v0_deconv=v0_deconv,
        R_fit_free=R_fit_free,
        R_deconv=R_deconv,
        sigma_of_fit=sigma_of_fit,
        bin_centers=bin_centers,
        profile_v=profile_v,
        cl_used=cl_used,
        cl_shift=cl_shift,
        rr_in=rr_in,
        cc_in=cc_in,
        rho_in=rho_in,
        t_row_in=t_row_in,
        t_col_in=t_col_in,
        vfield_mean=vfield_mean,
        n_vessel_px=n_vessel_px,
        fb_winsize=fb_winsize,
        roi_offset=(r0, c0),  # ROI origin in original frame coords
        # Original-CL Poiseuille (before ridge refinement)
        Q_pois_orig_mean=Q_pois_orig_mean,
        v0_orig=v0_orig,
        profile_v_orig=profile_v_orig,
    )


# ===========================================================================
# Re-bin an existing OF result with a different centerline
# ===========================================================================

def _rebin_profile(unified_result, original_centerline, R, *,
                   n_bins=30, fit_fraction=0.75):
    """Re-bin the time-averaged velocity field from an existing unified result
    using a different centerline (e.g., the original segmentation CL).

    Uses vfield_mean from the unified result — no extra Farneback needed.
    Returns profile, Q_plug_mean, Q_pois_mean, v0_fixed, but NOT per-frame Q_t
    (would require re-running the full OF loop).

    Parameters
    ----------
    unified_result : dict from compute_of_profile_and_Q
    original_centerline : (N, 2) [row, col] — the centerline to re-bin with
    R : float — vessel radius in pixels
    """
    vfield = unified_result.get('vfield_mean')
    roi_offset = unified_result.get('roi_offset', (0, 0))
    if vfield is None:
        return dict(bin_centers=np.zeros(n_bins), profile_v=np.full(n_bins, np.nan),
                    v0_fixed=np.nan, Q_plug_mean=np.nan, Q_pois_mean=np.nan,
                    Q_plug_t=None, Q_pois_t=None)

    r0_off, c0_off = roi_offset
    H_roi, W_roi = vfield.shape[:2]

    # Build vessel geometry in ROI coords using the original centerline
    cl_roi = original_centerline.copy()
    cl_roi[:, 0] -= r0_off
    cl_roi[:, 1] -= c0_off

    geom = _build_vessel_mask(cl_roi, R, H_roi, W_roi)
    if geom is None:
        return dict(bin_centers=np.zeros(n_bins), profile_v=np.full(n_bins, np.nan),
                    v0_fixed=np.nan, Q_plug_mean=np.nan, Q_pois_mean=np.nan,
                    Q_plug_t=None, Q_pois_t=None)

    rr_in = geom['rr_in']
    cc_in = geom['cc_in']
    rho_in = geom['rho_in']
    t_row_in = geom['t_row_in']
    t_col_in = geom['t_col_in']

    # Project mean velocity field onto tangent
    of_row = vfield[rr_in, cc_in, 1]  # dy component
    of_col = vfield[rr_in, cc_in, 0]  # dx component
    v_axial = of_row * t_row_in + of_col * t_col_in

    # Bin by rho
    bin_edges = np.linspace(-R, R, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile_v = np.full(n_bins, np.nan)
    for i_bin, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (rho_in >= lo) & (rho_in < hi)
        vals = v_axial[in_bin]
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            profile_v[i_bin] = float(np.mean(finite))

    # Plug-flow Q (mean v × πR²)
    A_px2 = np.pi * R ** 2
    v_mean = float(np.nanmean(v_axial[np.isfinite(v_axial)])) if np.any(np.isfinite(v_axial)) else 0.0
    Q_plug_mean = v_mean * A_px2

    # Poiseuille fit (R-fixed, central fraction)
    fit_limit = R * fit_fraction
    valid = np.isfinite(profile_v) & (np.abs(bin_centers) <= fit_limit)
    v0_fixed = np.nan
    if valid.sum() >= 3:
        rv = bin_centers[valid]
        vv = profile_v[valid]
        A_mat = (1.0 - (rv / R) ** 2)[:, None]
        try:
            coeffs, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
            v0_fixed = float(coeffs[0])
        except Exception:
            pass

    Q_pois_mean = v0_fixed * A_px2 / 2.0 if np.isfinite(v0_fixed) else np.nan

    return dict(
        bin_centers=bin_centers,
        profile_v=profile_v,
        v0_fixed=v0_fixed,
        Q_plug_mean=Q_plug_mean,
        Q_pois_mean=Q_pois_mean,
        Q_plug_t=None,   # not available without re-running OF
        Q_pois_t=None,   # not available without re-running OF
    )


# ===========================================================================
# 1D cross-correlation PIV per radial strip
# ===========================================================================

def _xcorr_symmetry_score(
    cl_candidate, R, bin_edges, bin_centers, n_bins,
    frames_u8, warmup_idx, dt, max_disp, H, W, subpixel=True,
):
    """Compute velocity-weighted symmetry score for a candidate centerline.

    Returns negative MSE (higher = more symmetric). Returns -inf on failure.
    """
    from .synthetic_video import _centerline_geometry
    from scipy.spatial import cKDTree

    n_d = 2 * max_disp + 1

    # Check bounds
    if (cl_candidate[:, 0].min() < 1 or cl_candidate[:, 0].max() >= H - 1
            or cl_candidate[:, 1].min() < 1 or cl_candidate[:, 1].max() >= W - 1):
        return -np.inf

    s_sh, _, n_sh = _centerline_geometry(cl_candidate)
    cl_tree = cKDTree(cl_candidate)

    # Build vessel mask + strips
    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                              cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree.query(pts_g)
    inside = ((dist_g <= R * 1.05) & (nearest_g > 0)
              & (nearest_g < len(cl_candidate) - 1)).reshape(H, W)
    rr_in, cc_in = np.where(inside)
    if len(rr_in) < 10:
        return -np.inf
    pts_in = np.column_stack([rr_in.astype(float), cc_in.astype(float)])
    _, nearest = cl_tree.query(pts_in)
    rho_in = ((rr_in - cl_candidate[nearest, 0]) * n_sh[nearest, 0]
              + (cc_in - cl_candidate[nearest, 1]) * n_sh[nearest, 1])
    s_in = s_sh[nearest]

    strip_data = []
    for ib in range(n_bins):
        mask = (rho_in >= bin_edges[ib]) & (rho_in < bin_edges[ib + 1])
        if mask.sum() < 3:
            strip_data.append(None)
            continue
        idx = np.where(mask)[0]
        order = np.argsort(s_in[idx])
        strip_data.append((rr_in[idx[order]], cc_in[idx[order]],
                           s_in[idx[order]]))

    # Quick xcorr on warmup frames
    corr_acc = [np.zeros(n_d) for _ in range(n_bins)]
    corr_cnt = [0] * n_bins
    for t in warmup_idx:
        f0 = frames_u8[t].astype(np.float32)
        f1 = frames_u8[t + dt].astype(np.float32)
        for ib in range(n_bins):
            sd = strip_data[ib]
            if sd is None:
                continue
            rows, cols, _ = sd
            I0 = f0[rows, cols]
            I1 = f1[rows, cols]
            n_px = len(I0)
            if n_px < 2 * max_disp + 1:
                continue
            I0z = I0 - I0.mean()
            I1z = I1 - I1.mean()
            if np.sum(I0z ** 2) < 1e-6 or np.sum(I1z ** 2) < 1e-6:
                continue
            for di, d in enumerate(range(-max_disp, max_disp + 1)):
                if d >= 0:
                    a = I0z[:n_px - d] if d > 0 else I0z
                    b = I1z[d:] if d > 0 else I1z
                else:
                    a = I0z[-d:]
                    b = I1z[:n_px + d]
                if len(a) >= max_disp:
                    corr_acc[ib][di] += float(np.sum(a * b))
            corr_cnt[ib] += 1

    # Extract velocities and compute symmetry
    disp_arr = np.arange(-max_disp, max_disp + 1, dtype=float)
    v_profile = np.full(n_bins, np.nan)
    for ib in range(n_bins):
        if corr_cnt[ib] < 3 or strip_data[ib] is None:
            continue
        cc = corr_acc[ib]
        i_pk = int(np.argmax(cc))
        disp = float(disp_arr[i_pk])
        if subpixel and 1 <= i_pk <= n_d - 2:
            cm, c0, cp = cc[i_pk - 1], cc[i_pk], cc[i_pk + 1]
            if cm < c0 and cp < c0:
                den = 2.0 * (cm - 2.0 * c0 + cp)
                if abs(den) > 1e-12:
                    disp += float(np.clip(-(cp - cm) / den, -0.5, 0.5))
        _, _, s_pos = strip_data[ib]
        ds = float(np.mean(np.diff(s_pos))) if len(s_pos) > 1 else 1.0
        v_profile[ib] = disp * ds / dt

    # Velocity-weighted symmetry: compare +ρ vs -ρ bins
    n_half = n_bins // 2
    v_pos = np.zeros(n_half)
    v_neg = np.zeros(n_half)
    for ib in range(n_bins):
        bc = bin_centers[ib]
        if not np.isfinite(v_profile[ib]):
            continue
        hi = min(int(abs(bc) / (R / n_half)), n_half - 1)
        if bc >= 0:
            v_pos[hi] = v_profile[ib]
        else:
            v_neg[hi] = v_profile[ib]

    w_bin = 0.5 * (np.abs(v_pos) + np.abs(v_neg))
    w_sum = w_bin.sum()
    if w_sum < 1e-8:
        return -np.inf
    return -float(np.sum(w_bin * (v_pos - v_neg) ** 2) / w_sum)


def _xcorr_refine_centerline(
    frames_u8, centerline, R, bin_edges, bin_centers,
    *, max_disp, dt, max_shift, n_warmup, subpixel,
):
    """3-parameter centerline refinement: offset + tilt + curvature.

    Parameterises the correction as a quadratic perpendicular offset:
        ρ(s) = a₀ + a₁·(s/S - 0.5) + a₂·(s/S - 0.5)²
    where a₀=shift, a₁=tilt, a₂=curvature correction.

    Uses scipy.optimize.minimize (Nelder-Mead) on the xcorr symmetry score.
    Returns (a0, a1, a2) coefficients.
    """
    from .synthetic_video import _centerline_geometry
    from scipy.optimize import minimize

    s_cl, _, normals_cl = _centerline_geometry(centerline)
    S = float(s_cl[-1]) if s_cl[-1] > 0 else 1.0
    T, H, W = frames_u8.shape
    n_bins = len(bin_centers)

    warmup_idx = np.linspace(0, T - dt - 1, min(n_warmup, T - dt), dtype=int)

    # Normalised arc coordinate: u ∈ [-0.5, 0.5]
    u = s_cl / S - 0.5  # (N_cl,)

    _call_count = [0]

    def objective(params):
        a0, a1, a2 = params
        rho_s = a0 + a1 * u + a2 * u ** 2  # (N_cl,)
        cl_cand = centerline + rho_s[:, None] * normals_cl
        score = _xcorr_symmetry_score(
            cl_cand, R, bin_edges, bin_centers, n_bins,
            frames_u8, warmup_idx, dt, max_disp, H, W, subpixel,
        )
        _call_count[0] += 1
        return -score  # minimise negative symmetry

    # Initial guess: no correction
    x0 = np.array([0.0, 0.0, 0.0])

    # Bounds: offset ±max_shift, tilt ±2*max_shift, curvature ±4*max_shift
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options=dict(
            xatol=0.2, fatol=1e-6,
            maxiter=80, maxfev=80,
            initial_simplex=np.array([
                [0.0, 0.0, 0.0],
                [max_shift * 0.5, 0.0, 0.0],
                [0.0, max_shift, 0.0],
                [0.0, 0.0, max_shift * 2],
            ]),
        ),
    )

    a0, a1, a2 = result.x
    # Clip to reasonable range
    a0 = float(np.clip(a0, -max_shift, max_shift))
    a1 = float(np.clip(a1, -2 * max_shift, 2 * max_shift))
    a2 = float(np.clip(a2, -4 * max_shift, 4 * max_shift))

    print(f'  CL refine: {_call_count[0]} evals, '
          f'a₀={a0:+.1f}px (shift), a₁={a1:+.1f} (tilt), '
          f'a₂={a2:+.1f} (curvature)')

    return a0, a1, a2


def compute_xcorr_profile_and_Q(
    frames,
    centerline,
    vessel_radius_px: float,
    *,
    bg_percentile: int = 10,
    dt: int = 1,
    n_bins: int = 30,
    max_disp: int = 8,
    fit_fraction: float = 0.75,
    subpixel: bool = True,
    # Phase-sorted correlation averaging (Kloosterman-style)
    f0_hz: Optional[float] = None,
    frame_dt_s: Optional[float] = None,
    n_phase_bins: int = 10,
    # Centerline refinement
    refine_centerline: bool = False,
    max_shift: float = 10.0,
    n_shift_steps: int = 21,
    n_warmup: int = 80,
):
    """1D cross-correlation with phase-sorted correlation averaging.

    Combines two ideas:
      1. Radial strips at constant ρ — zero cross-radial averaging
      2. Correlation averaging within cardiac phase bins — √N SNR boost

    For each radial bin and phase bin, the 1D correlation functions from
    all frame pairs at that phase are accumulated before peak-finding.
    This dramatically improves SNR for sparse particles (Kloosterman 2014).

    Without f0_hz, falls back to averaging all correlations (no phase sorting).

    Parameters
    ----------
    frames : (T, H, W) array
        Raw video frames.
    centerline : (N, 2) float [row, col]
    vessel_radius_px : float
    bg_percentile : int
    dt : int
        Frame step for correlation pairs.
    n_bins : int
        Number of radial bins spanning [-R, R].
    max_disp : int
        Maximum displacement to search (pixels).
    fit_fraction : float
        Fraction of R for central Poiseuille fit.
    subpixel : bool
        Parabolic sub-pixel peak interpolation.
    f0_hz : float or None
        Cardiac frequency. If provided, enables phase-sorted averaging.
    frame_dt_s : float or None
        Frame interval in seconds. Required if f0_hz is set.
    n_phase_bins : int
        Number of cardiac phase bins (default 10 → good for 3 harmonics).

    Returns
    -------
    dict with keys:
        Q_pois_phi    : (n_phase_bins,) Q per phase bin [px²/frame]
        Q_pois_mean   : float, time-averaged Q
        v0_fixed      : float, Poiseuille v₀ from time-averaged profile
        bin_centers   : (n_bins,) radial bin centers
        profile_v     : (n_bins,) time-averaged velocity profile
        profile_v_phi : (n_phase_bins, n_bins) phase-resolved profiles
        phase_centers : (n_phase_bins,) phase bin centers [0, 1)
    """
    from .synthetic_video import _centerline_geometry
    from scipy.spatial import cKDTree

    centerline = np.asarray(centerline, dtype=float)
    R = float(vessel_radius_px)

    # ── 1. Normalise ──────────────────────────────────────────────────
    frames_u8 = _normalise_for_of(frames, bg_percentile=bg_percentile)
    T, H, W = frames_u8.shape

    bin_edges = np.linspace(-R, R, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ── 2. Centerline refinement (perpendicular shift only) ─────────
    cl_shift = 0.0
    cl_used = centerline.copy()
    if refine_centerline:
        _, _, normals_raw = _centerline_geometry(centerline)
        # Scan shifts, pick most symmetric profile
        shifts = np.linspace(-max_shift, max_shift, n_shift_steps)
        scores = np.full(len(shifts), -np.inf)
        warmup_idx = np.linspace(0, T - dt - 1,
                                  min(n_warmup, T - dt), dtype=int)
        for si, sh in enumerate(shifts):
            cl_cand = centerline + sh * normals_raw
            scores[si] = _xcorr_symmetry_score(
                cl_cand, R, bin_edges, bin_centers, n_bins,
                frames_u8, warmup_idx, dt, max_disp, H, W, subpixel,
            )
        i_best = int(np.argmax(scores))
        cl_shift = float(shifts[i_best])
        # Sub-pixel parabolic interpolation
        if 0 < i_best < len(scores) - 1:
            sm, s0, sp = scores[i_best - 1], scores[i_best], scores[i_best + 1]
            if np.isfinite(sm) and np.isfinite(sp) and sm < s0 and sp < s0:
                den = 2.0 * (sm - 2.0 * s0 + sp)
                if abs(den) > 1e-12:
                    delta = -(sp - sm) / den
                    step = shifts[1] - shifts[0]
                    cl_shift += float(np.clip(delta, -0.5, 0.5)) * step
        cl_used = centerline + cl_shift * normals_raw
        print(f'  CL refined: shift={cl_shift:+.1f}px '
              f'({len(shifts)} steps, {len(warmup_idx)} warmup frames)')

    # ── 3. Vessel geometry ────────────────────────────────────────────
    s_cl, tangents_cl, normals_cl = _centerline_geometry(cl_used)
    cl_tree = cKDTree(cl_used)

    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                              cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree.query(pts_g)
    inside = (
        (dist_g <= R * 1.05)
        & (nearest_g > 0)
        & (nearest_g < len(cl_used) - 1)
    )
    inside = inside.reshape(H, W)
    rr_in, cc_in = np.where(inside)
    if len(rr_in) == 0:
        return _xcorr_nan(T, dt, n_bins, bin_centers, n_phase_bins)

    pts_in = np.column_stack([rr_in.astype(float), cc_in.astype(float)])
    _, nearest = cl_tree.query(pts_in)

    n_row_in = normals_cl[nearest, 0]
    n_col_in = normals_cl[nearest, 1]
    rho_in = ((rr_in - centerline[nearest, 0]) * n_row_in
              + (cc_in - centerline[nearest, 1]) * n_col_in)
    s_in = s_cl[nearest]

    # ── 3. Build radial strips (sorted by arc position) ───────────────
    strip_data = []
    for i_bin in range(n_bins):
        lo, hi = bin_edges[i_bin], bin_edges[i_bin + 1]
        mask = (rho_in >= lo) & (rho_in < hi)
        if mask.sum() < 5:
            strip_data.append(None)
            continue
        idx = np.where(mask)[0]
        order = np.argsort(s_in[idx])
        idx_sorted = idx[order]
        strip_data.append((
            rr_in[idx_sorted],
            cc_in[idx_sorted],
            s_in[idx_sorted],
        ))

    # ── 4. Phase assignment ───────────────────────────────────────────
    n_pairs = T - dt
    n_d = 2 * max_disp + 1  # correlation array length
    A_px2 = np.pi * R ** 2

    use_phase = (f0_hz is not None and frame_dt_s is not None
                 and f0_hz > 0 and n_phase_bins > 1)
    if use_phase:
        phi = (np.arange(n_pairs) * frame_dt_s * f0_hz) % 1.0
        phase_edges = np.linspace(0, 1, n_phase_bins + 1)
        phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])
        phase_idx = np.clip(
            np.digitize(phi, phase_edges) - 1, 0, n_phase_bins - 1)
    else:
        n_phase_bins = 1
        phase_centers = np.array([0.5])
        phase_idx = np.zeros(n_pairs, dtype=int)

    # ── 5. Accumulate correlation maps per (radial bin, phase bin) ─────
    # corr_acc[phase][bin] = accumulated correlation array (n_d,)
    # corr_cnt[phase][bin] = number of pairs contributing
    corr_acc = [[np.zeros(n_d) for _ in range(n_bins)]
                for _ in range(n_phase_bins)]
    corr_cnt = [[0] * n_bins for _ in range(n_phase_bins)]

    _process_events = _get_process_events()

    for t in range(n_pairs):
        if t % 100 == 0 and _process_events is not None:
            _process_events()

        ph = phase_idx[t]
        f0_img = frames_u8[t].astype(np.float32)
        f1_img = frames_u8[t + dt].astype(np.float32)

        for i_bin in range(n_bins):
            sd = strip_data[i_bin]
            if sd is None:
                continue
            rows, cols, s_pos = sd

            I0 = f0_img[rows, cols]
            I1 = f1_img[rows, cols]
            n_px = len(I0)
            if n_px < 2 * max_disp + 1:
                continue

            I0_z = I0 - I0.mean()
            I1_z = I1 - I1.mean()
            norm0 = float(np.sqrt(np.sum(I0_z ** 2)))
            norm1 = float(np.sqrt(np.sum(I1_z ** 2)))
            if norm0 < 1e-6 or norm1 < 1e-6:
                continue

            # Compute unnormalised correlation at each shift
            # (accumulate raw dot products — normalise after averaging)
            corr_frame = np.zeros(n_d)
            for d_idx, d in enumerate(range(-max_disp, max_disp + 1)):
                if d >= 0:
                    a = I0_z[:n_px - d] if d > 0 else I0_z
                    b = I1_z[d:] if d > 0 else I1_z
                else:
                    a = I0_z[-d:]
                    b = I1_z[:n_px + d]
                if len(a) < max_disp:
                    continue
                corr_frame[d_idx] = float(np.sum(a * b))

            corr_acc[ph][i_bin] += corr_frame
            corr_cnt[ph][i_bin] += 1

    # ── 6. Peak-find on averaged correlations ─────────────────────────
    disp_arr = np.arange(-max_disp, max_disp + 1, dtype=float)
    profile_v_phi = np.full((n_phase_bins, n_bins), np.nan)

    for ph in range(n_phase_bins):
        for i_bin in range(n_bins):
            if corr_cnt[ph][i_bin] < 3:
                continue
            sd = strip_data[i_bin]
            if sd is None:
                continue
            _, _, s_pos = sd

            cc = corr_acc[ph][i_bin]
            # Find peak
            i_peak = int(np.argmax(cc))
            disp = float(disp_arr[i_peak])

            # Sub-pixel parabolic interpolation
            if subpixel and 1 <= i_peak <= n_d - 2:
                cm, c0, cp = cc[i_peak - 1], cc[i_peak], cc[i_peak + 1]
                if cm < c0 and cp < c0:
                    denom = 2.0 * (cm - 2.0 * c0 + cp)
                    if abs(denom) > 1e-12:
                        delta = -(cp - cm) / denom
                        disp += float(np.clip(delta, -0.5, 0.5))

            ds_mean = float(np.mean(np.diff(s_pos))) if len(s_pos) > 1 else 1.0
            profile_v_phi[ph, i_bin] = disp * ds_mean / dt

    # ── 7. Time-averaged profile (mean over phase bins) ───────────────
    with np.errstate(invalid='ignore'):
        profile_v = np.nanmean(profile_v_phi, axis=0)

    # ── 8. Q per phase bin (Poiseuille fit, R-fixed, central fraction) ─
    r_cut = R * fit_fraction
    fit_mask = np.abs(bin_centers) <= r_cut
    Q_pois_phi = np.full(n_phase_bins, np.nan)
    v0_phi = np.full(n_phase_bins, np.nan)

    for ph in range(n_phase_bins):
        pv = profile_v_phi[ph]
        valid = np.isfinite(pv) & fit_mask
        if valid.sum() >= 3:
            rv = bin_centers[valid]
            vv = pv[valid]
            A_mat = (1.0 - (rv / R) ** 2)[:, None]
            try:
                v0, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
                v0_phi[ph] = float(v0[0])
                Q_pois_phi[ph] = float(v0[0]) * A_px2 / 2.0
            except Exception:
                pass

    # Time-averaged fit
    valid_avg = np.isfinite(profile_v) & fit_mask
    v0_fixed = np.nan
    if valid_avg.sum() >= 3:
        rv = bin_centers[valid_avg]
        vv = profile_v[valid_avg]
        A_mat = (1.0 - (rv / R) ** 2)[:, None]
        try:
            coeffs, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
            v0_fixed = float(coeffs[0])
        except Exception:
            pass

    # Reconstruct Q(t) from harmonic fit to Q(φ) if phase-sorted
    Q_pois_t = np.full(n_pairs, np.nan)
    Q_plug_t = np.full(n_pairs, np.nan)
    if use_phase and np.any(np.isfinite(Q_pois_phi)):
        # Interpolate Q(φ) → Q(t) using the phase of each frame
        from scipy.interpolate import interp1d
        valid_ph = np.isfinite(Q_pois_phi)
        if valid_ph.sum() >= 3:
            # Wrap-around interpolation: extend by one period
            pc_ext = np.concatenate([phase_centers - 1, phase_centers,
                                      phase_centers + 1])
            qp_ext = np.concatenate([Q_pois_phi, Q_pois_phi, Q_pois_phi])
            valid_ext = np.concatenate([valid_ph, valid_ph, valid_ph])
            interp_fn = interp1d(pc_ext[valid_ext], qp_ext[valid_ext],
                                  kind='linear', fill_value='extrapolate')
            Q_pois_t = interp_fn(phi)

        # Plug-flow from phase-averaged profiles
        Q_plug_phi = np.nanmean(profile_v_phi, axis=1) * A_px2
        if np.any(np.isfinite(Q_plug_phi)):
            valid_pl = np.isfinite(Q_plug_phi)
            pc_ext = np.concatenate([phase_centers - 1, phase_centers,
                                      phase_centers + 1])
            qpl_ext = np.concatenate([Q_plug_phi, Q_plug_phi, Q_plug_phi])
            valid_ext = np.concatenate([valid_pl, valid_pl, valid_pl])
            interp_fn = interp1d(pc_ext[valid_ext], qpl_ext[valid_ext],
                                  kind='linear', fill_value='extrapolate')
            Q_plug_t = interp_fn(phi)
    else:
        # No phase sorting — single-bin results
        if np.isfinite(v0_fixed):
            Q_pois_t[:] = v0_fixed * A_px2 / 2.0
        Q_plug_mean_val = float(np.nanmean(profile_v)) * A_px2
        Q_plug_t[:] = Q_plug_mean_val

    Q_pois_mean = float(np.nanmean(Q_pois_phi)) if np.any(
        np.isfinite(Q_pois_phi)) else np.nan
    Q_plug_mean = float(np.nanmean(profile_v)) * A_px2

    n_frames_per_bin = [sum(1 for p in phase_idx if p == ph)
                        for ph in range(n_phase_bins)]
    print(f'  Phase bins: {n_phase_bins}, '
          f'frames/bin: {min(n_frames_per_bin)}–{max(n_frames_per_bin)}')

    return dict(
        Q_plug_t=Q_plug_t,
        Q_pois_t=Q_pois_t,
        Q_plug_mean=Q_plug_mean,
        Q_pois_mean=Q_pois_mean,
        v0_fixed=v0_fixed,
        bin_centers=bin_centers,
        profile_v=profile_v,
        profile_v_phi=profile_v_phi,
        Q_pois_phi=Q_pois_phi,
        phase_centers=phase_centers,
        n_phase_bins=n_phase_bins,
        cl_used=cl_used,
        cl_shift=cl_shift,
    )


def _xcorr_nan(T, dt, n_bins, bin_centers, n_phase_bins=10):
    n_pairs = T - dt
    return dict(
        Q_plug_t=np.full(n_pairs, np.nan),
        Q_pois_t=np.full(n_pairs, np.nan),
        Q_plug_mean=np.nan,
        Q_pois_mean=np.nan,
        v0_fixed=np.nan,
        bin_centers=bin_centers,
        profile_v=np.full(n_bins, np.nan),
        profile_v_phi=np.full((n_phase_bins, n_bins), np.nan),
        Q_pois_phi=np.full(n_phase_bins, np.nan),
        phase_centers=np.linspace(0.05, 0.95, n_phase_bins),
        n_phase_bins=n_phase_bins,
    )


def _get_process_events():
    """Return QApplication.processEvents or None if Qt unavailable."""
    try:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            return app.processEvents
    except Exception:
        pass
    return None


# =========================================================================
# Correlation-averaged 2D PIV
# =========================================================================

def compute_piv_corr_averaged(
    frames,
    centerline,
    vessel_radius_px: float,
    *,
    bg_percentile: int = 10,
    dt: int = 1,
    win: int = 4,
    max_disp: int = 4,
    n_bins: int = 30,
    fit_fraction: float = 0.75,
    f0_hz: float = 2.5,
    frame_dt_s: float = 0.004,
    n_phase_bins: int = 20,
    max_frames_per_bin: int = 40,
    pixel_stride: int = 2,
    refine_centerline: bool = False,
    max_shift: float = 10.0,
    n_shift_steps: int = 21,
):
    """Correlation-averaged 2D PIV with phase sorting.

    For each vessel pixel, accumulates 2D cross-correlation maps across all
    frame pairs within each cardiac phase bin, then peak-finds on the averaged
    map.  This gives per-pixel, per-phase displacement vectors with √N SNR
    boost from the correlation averaging.

    Parameters
    ----------
    frames : (T, H, W) array
    centerline : (N, 2) [row, col]
    vessel_radius_px : float
    win : int
        Half-width of the interrogation window (full size = 2*win).
    max_disp : int
        Max displacement in pixels (search area = win + max_disp on each side).
    n_phase_bins : int
        Number of cardiac phase bins.
    max_frames_per_bin : int
        Cap on frame pairs per bin (for speed).
    pixel_stride : int
        Process every Nth vessel pixel (for speed).

    Returns
    -------
    dict with per-pixel velocities, phase-resolved profiles, Q(t), etc.
    """
    from scipy.signal import fftconvolve

    centerline = np.asarray(centerline, dtype=float)
    R = float(vessel_radius_px)

    # ── 1. Normalise ──────────────────────────────────────────────────
    frames_u8 = _normalise_for_of(frames, bg_percentile=bg_percentile)
    T, H, W = frames_u8.shape

    # ── 2. Optional centerline refinement ─────────────────────────────
    cl_shift = 0.0
    cl_used = centerline.copy()
    if refine_centerline:
        cl_shift = _xcorr_refine_centerline(
            frames_u8, centerline, R,
            dt=dt, max_disp=max_disp, n_warmup=80,
            max_shift=max_shift, n_shift_steps=n_shift_steps,
        )
        _, _, normals_cl = _centerline_geometry(centerline)
        cl_used = centerline + cl_shift * normals_cl

    # ── 3. Vessel geometry ────────────────────────────────────────────
    geom = _build_vessel_mask(cl_used, R, H, W)
    if geom is None:
        return _piv_nan(T, dt, n_bins, n_phase_bins, cl_used)

    rr_all = geom['rr_in']
    cc_all = geom['cc_in']
    rho_all = geom['rho_in']
    t_row_all = geom['t_row_in']
    t_col_all = geom['t_col_in']

    # Subsample pixels for speed
    idx = np.arange(0, len(rr_all), pixel_stride)
    rr_px = rr_all[idx]
    cc_px = cc_all[idx]
    rho_px = rho_all[idx]
    t_row_px = t_row_all[idx]
    t_col_px = t_col_all[idx]
    n_pix = len(rr_px)

    # ── 4. Phase-sort frames ──────────────────────────────────────────
    frame_phase = (np.arange(T - dt) * frame_dt_s * f0_hz) % 1.0
    bin_edges = np.linspace(0, 1, n_phase_bins + 1)
    phase_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(
        np.digitize(frame_phase, bin_edges) - 1, 0, n_phase_bins - 1)

    # Frame indices per bin (capped)
    bin_frames = []
    for b in range(n_phase_bins):
        frames_in_bin = np.where(bin_idx == b)[0]
        if len(frames_in_bin) > max_frames_per_bin:
            frames_in_bin = frames_in_bin[
                np.linspace(0, len(frames_in_bin) - 1,
                            max_frames_per_bin, dtype=int)]
        bin_frames.append(frames_in_bin)

    print(f'  Phase bins: {n_phase_bins}, '
          f'frames/bin: {min(len(b) for b in bin_frames)}'
          f'–{max(len(b) for b in bin_frames)}, '
          f'{n_pix} pixels (stride={pixel_stride})')

    # ── 5. Bounds check: which pixels have valid windows ──────────────
    half_w = win
    half_s = win + max_disp
    valid_px = (
        (rr_px - half_s >= 0) & (rr_px + half_s < H) &
        (cc_px - half_s >= 0) & (cc_px + half_s < W)
    )
    valid_idx = np.where(valid_px)[0]
    n_valid = len(valid_idx)

    if n_valid == 0:
        return _piv_nan(T, dt, n_bins, n_phase_bins, cl_used)

    corr_size = 2 * max_disp + 1

    # ── 6. Accumulate correlation maps ────────────────────────────────
    # corr_sum[pixel, phase_bin, dr, dc]
    corr_sum = np.zeros((n_valid, n_phase_bins, corr_size, corr_size),
                         dtype=np.float64)
    corr_count = np.zeros((n_valid, n_phase_bins), dtype=np.int32)

    _pe = _get_process_events()
    win_full = 2 * win  # full interrogation window size

    for b in range(n_phase_bins):
        for t in bin_frames[b]:
            f1 = frames_u8[t].astype(np.float32)
            f2 = frames_u8[t + dt].astype(np.float32)

            for vi, ip in enumerate(valid_idx):
                r, c = int(rr_px[ip]), int(cc_px[ip])

                # Interrogation window from frame 1
                w1 = f1[r - half_w:r + half_w,
                         c - half_w:c + half_w]
                # Search window from frame 2
                w2 = f2[r - half_s:r + half_s,
                         c - half_s:c + half_s]

                # Zero-mean normalise
                w1m = w1 - w1.mean()
                w2m = w2 - w2.mean()

                w1_std = w1m.std()
                if w1_std < 1e-6:
                    continue

                # 2D cross-correlation (valid mode → corr_size × corr_size)
                cc = fftconvolve(w2m, w1m[::-1, ::-1], mode='valid')

                if cc.shape == (corr_size, corr_size):
                    corr_sum[vi, b] += cc
                    corr_count[vi, b] += 1

        if _pe is not None:
            _pe()

    # ── 7. Peak-find on averaged correlation maps ─────────────────────
    vr_pix = np.full((n_valid, n_phase_bins), np.nan)
    vc_pix = np.full((n_valid, n_phase_bins), np.nan)

    for vi in range(n_valid):
        for b in range(n_phase_bins):
            if corr_count[vi, b] < 3:
                continue
            cc = corr_sum[vi, b] / corr_count[vi, b]

            # Integer peak
            peak_flat = np.argmax(cc)
            pr, pc = divmod(peak_flat, corr_size)

            # Sub-pixel parabolic refinement
            dr_sub, dc_sub = 0.0, 0.0
            if 0 < pr < corr_size - 1:
                denom = cc[pr - 1, pc] - 2 * cc[pr, pc] + cc[pr + 1, pc]
                if abs(denom) > 1e-12:
                    dr_sub = (cc[pr - 1, pc] - cc[pr + 1, pc]) / (2 * denom)
                    dr_sub = np.clip(dr_sub, -0.5, 0.5)
            if 0 < pc < corr_size - 1:
                denom = cc[pr, pc - 1] - 2 * cc[pr, pc] + cc[pr, pc + 1]
                if abs(denom) > 1e-12:
                    dc_sub = (cc[pr, pc - 1] - cc[pr, pc + 1]) / (2 * denom)
                    dc_sub = np.clip(dc_sub, -0.5, 0.5)

            vr_pix[vi, b] = (pr - max_disp + dr_sub) / dt
            vc_pix[vi, b] = (pc - max_disp + dc_sub) / dt

    # ── 8. Project onto tangent → v_axial per pixel per phase ─────────
    t_row_valid = t_row_px[valid_idx]
    t_col_valid = t_col_px[valid_idx]
    rho_valid = rho_px[valid_idx]

    # vr = row displacement, vc = col displacement
    # tangent is (t_row, t_col), flow[...,1]=dy(row), flow[...,0]=dx(col)
    v_axial = vr_pix * t_row_valid[:, None] + vc_pix * t_col_valid[:, None]

    # ── 9. Radial profile per phase bin ───────────────────────────────
    bin_edges_r = np.linspace(-R, R, n_bins + 1)
    bin_centers_r = 0.5 * (bin_edges_r[:-1] + bin_edges_r[1:])

    profile_v_phi = np.full((n_phase_bins, n_bins), np.nan)
    for b in range(n_phase_bins):
        va = v_axial[:, b]
        for ib in range(n_bins):
            in_bin = ((rho_valid >= bin_edges_r[ib]) &
                      (rho_valid < bin_edges_r[ib + 1]))
            vals = va[in_bin]
            finite = vals[np.isfinite(vals)]
            if len(finite) >= 2:
                profile_v_phi[b, ib] = float(np.mean(finite))

    # Time-averaged profile
    with np.errstate(invalid='ignore'):
        profile_v = np.nanmean(profile_v_phi, axis=0)

    # ── 10. Q per phase bin ───────────────────────────────────────────
    A_px2 = np.pi * R ** 2
    fit_limit = R * fit_fraction

    Q_plug_phi = np.full(n_phase_bins, np.nan)
    Q_pois_phi = np.full(n_phase_bins, np.nan)

    for b in range(n_phase_bins):
        va = v_axial[:, b]
        finite = va[np.isfinite(va)]
        if len(finite) >= 5:
            Q_plug_phi[b] = float(np.mean(finite)) * A_px2

        # Poiseuille fit to central fraction
        pv = profile_v_phi[b]
        fit_mask = np.isfinite(pv) & (np.abs(bin_centers_r) <= fit_limit)
        if fit_mask.sum() >= 3:
            rv = bin_centers_r[fit_mask]
            vv = pv[fit_mask]
            A_mat = (1.0 - (rv / R) ** 2)[:, None]
            try:
                v0, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
                Q_pois_phi[b] = float(v0[0]) * A_px2 / 2.0
            except Exception:
                pass

    # Time-averaged Poiseuille fit
    valid = np.isfinite(profile_v)
    fit_mask_avg = valid & (np.abs(bin_centers_r) <= fit_limit)
    v0_fixed = np.nan
    if fit_mask_avg.sum() >= 3:
        rv = bin_centers_r[fit_mask_avg]
        vv = profile_v[fit_mask_avg]
        A_mat = (1.0 - (rv / R) ** 2)[:, None]
        try:
            coeffs, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
            v0_fixed = float(coeffs[0])
        except Exception:
            pass

    # Build Q(t) from phase bins → interpolate to frame times
    Q_pois_t = np.interp(frame_phase, phase_centers, Q_pois_phi,
                          period=1.0)
    Q_plug_t = np.interp(frame_phase, phase_centers, Q_plug_phi,
                          period=1.0)

    # Mean velocity field for arrows (time-averaged across all phase bins)
    v_axial_mean = np.nanmean(v_axial, axis=1)  # (n_valid,)

    # Map back to full pixel arrays
    vr_mean = np.full(n_pix, np.nan)
    vc_mean = np.full(n_pix, np.nan)
    v_ax_mean_full = np.full(n_pix, np.nan)
    vr_mean[valid_idx] = np.nanmean(vr_pix, axis=1)
    vc_mean[valid_idx] = np.nanmean(vc_pix, axis=1)
    v_ax_mean_full[valid_idx] = v_axial_mean

    # Store example correlation maps for diagnostics (center pixel)
    center_vi = None
    if n_valid > 0:
        center_dist = np.abs(rho_valid)
        center_vi = int(np.argmin(center_dist))
    corr_examples = {}
    if center_vi is not None:
        corr_examples['center'] = (
            corr_sum[center_vi] /
            np.maximum(corr_count[center_vi][:, None, None], 1)
        )
        # Wall pixel (~0.8R)
        wall_dist = np.abs(np.abs(rho_valid) - 0.8 * R)
        wall_vi = int(np.argmin(wall_dist))
        corr_examples['wall'] = (
            corr_sum[wall_vi] /
            np.maximum(corr_count[wall_vi][:, None, None], 1)
        )

    Q_plug_mean = float(np.nanmean(Q_plug_phi))
    Q_pois_mean = float(np.nanmean(Q_pois_phi))

    return dict(
        # Per-pixel arrays (subsampled)
        rr_px=rr_px, cc_px=cc_px, rho_px=rho_px,
        t_row_px=t_row_px, t_col_px=t_col_px,
        vr_mean=vr_mean, vc_mean=vc_mean,
        v_axial_mean=v_ax_mean_full,
        valid_idx=valid_idx,
        # Profile
        bin_centers=bin_centers_r,
        profile_v=profile_v,
        profile_v_phi=profile_v_phi,
        # Q
        Q_plug_t=Q_plug_t, Q_pois_t=Q_pois_t,
        Q_plug_phi=Q_plug_phi, Q_pois_phi=Q_pois_phi,
        Q_plug_mean=Q_plug_mean, Q_pois_mean=Q_pois_mean,
        v0_fixed=v0_fixed,
        # Phase
        phase_centers=phase_centers,
        n_phase_bins=n_phase_bins,
        # Geometry
        cl_used=cl_used, cl_shift=cl_shift,
        n_vessel_px=n_pix, n_valid_px=n_valid,
        # Diagnostics
        corr_examples=corr_examples,
        corr_count=corr_count,
    )


def _piv_nan(T, dt, n_bins, n_phase_bins, cl_used):
    """NaN result for empty vessels."""
    bc = np.linspace(-1, 1, n_bins)
    return dict(
        rr_px=np.array([]), cc_px=np.array([]),
        rho_px=np.array([]), t_row_px=np.array([]),
        t_col_px=np.array([]),
        vr_mean=np.array([]), vc_mean=np.array([]),
        v_axial_mean=np.array([]), valid_idx=np.array([]),
        bin_centers=bc, profile_v=np.full(n_bins, np.nan),
        profile_v_phi=np.full((n_phase_bins, n_bins), np.nan),
        Q_plug_t=np.full(T - dt, np.nan),
        Q_pois_t=np.full(T - dt, np.nan),
        Q_plug_phi=np.full(n_phase_bins, np.nan),
        Q_pois_phi=np.full(n_phase_bins, np.nan),
        Q_plug_mean=np.nan, Q_pois_mean=np.nan,
        v0_fixed=np.nan,
        phase_centers=np.linspace(0.05, 0.95, n_phase_bins),
        n_phase_bins=n_phase_bins,
        cl_used=cl_used, cl_shift=0.0,
        n_vessel_px=0, n_valid_px=0,
        corr_examples={}, corr_count=np.array([]),
    )
