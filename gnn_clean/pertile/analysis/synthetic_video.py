"""
Synthetic 2D vessel video generator for PIV / optical-flow validation.

Generates (T, H, W) video stacks of discrete fluorescent beads flowing
through a vessel with Poiseuille flow along an arbitrary centerline.
DOC weighting, defocus broadening, motion blur, and photon noise are
all modelled.

This module is independent of the kymograph pipeline; it is the backend
for ``pertile.viewer.synthetic_vessel_piv_app``.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from .synthetic_kymo import compute_doc_z_c, compute_psf_sigma_px


# =============================================================================
# Centerline geometry helpers
# =============================================================================

def make_centerline(
    shape: str = 'straight',
    length: float = 80.0,
    vessel_radius_px: float = 12.0,
    curvature: float = 0.0,    # signed 1/radius (positive curves downward)
    amplitude: float = 10.0,   # lateral amplitude for sinusoid / spline (px)
    n_periods: float = 1.0,
    margin: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a vessel centerline at ~1px spacing.

    Returns (N, 2) array of [row, col] image coordinates.
    The centerline starts at approximately (margin + R, margin).

    Parameters
    ----------
    shape : 'straight' | 'arc' | 'sinusoid' | 'spline'
    length : float
        Arc length of the centerline in pixels.
    vessel_radius_px : float
        Used only for vertical positioning (adds enough top margin).
    curvature : float
        For 'arc': signed curvature κ = 1/radius_of_curvature.  Positive
        curves the vessel downward (larger row); negative curves upward.
    amplitude : float
        For 'sinusoid': peak lateral displacement from centre line.
        For 'spline': RMS lateral deviation of random control points.
    n_periods : float
        For 'sinusoid': number of full sinusoidal periods over *length*.
    margin : float
        Pixel margin around the vessel on all sides.
    seed : int or None
        Random seed for 'spline' shape.
    """
    R = vessel_radius_px
    n = max(int(length) + 1, 3)
    t = np.linspace(0.0, 1.0, n)

    if shape == 'straight':
        rows = np.full(n, margin + R)
        cols = margin + t * length

    elif shape == 'arc':
        if abs(curvature) < 1e-5:
            rows = np.full(n, margin + R)
            cols = margin + t * length
        else:
            theta = curvature * length * t          # angle swept
            x = np.sin(theta) / curvature           # col offset
            y = (np.cos(theta) - 1.0) / curvature  # row offset
            cols = margin + x - x.min()
            rows = margin + R + y - y.min()

    elif shape == 'sinusoid':
        cols = margin + t * length
        rows = margin + R + amplitude + amplitude * np.sin(
            2.0 * np.pi * n_periods * t)

    elif shape == 'spline':
        from scipy.interpolate import CubicSpline
        rng = np.random.default_rng(seed)
        # 1–2 interior control points → at most 2 curvature changes.
        # Endpoints pinned at 0 with clamped (zero-slope) BCs.
        n_interior = rng.integers(1, 3)  # 1 or 2
        t_interior = np.sort(rng.uniform(0.15, 0.85, size=n_interior))
        t_ctrl = np.concatenate([[0.0], t_interior, [1.0]])
        y_ctrl = np.zeros(len(t_ctrl))
        y_ctrl[1:-1] = rng.normal(0.0, amplitude, size=n_interior)
        cs = CubicSpline(t_ctrl, y_ctrl, bc_type='clamped')
        lateral = cs(t)
        cols = margin + t * length
        rows = margin + R + abs(lateral.min()) + lateral

    else:
        raise ValueError(
            f'shape must be straight/arc/sinusoid/spline, got {shape!r}')

    return np.column_stack([rows, cols])


def _centerline_geometry(
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute arc lengths, unit tangents, and CW normals for a centerline.

    Parameters
    ----------
    pts : (N, 2) float [row, col]

    Returns
    -------
    s        : (N,) cumulative arc length
    tangents : (N, 2) unit tangent vectors [row_component, col_component]
    normals  : (N, 2) CW-rotated unit normals (tangent_col, -tangent_row);
               positive normal ρ points in the direction of larger row index
               when the vessel runs leftward→rightward.
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

    # CW normal: rotate tangent 90° clockwise in image coords
    # (row, col) → (col, -row) i.e. normal = (t_col, -t_row)
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    return s, tangents, normals


# =============================================================================
# Direct 2D vessel video generator (no kymograph intermediary)
# =============================================================================

def generate_vessel_video(
    T: int = 800,
    W: int = 80,
    vessel_radius_px: float = 12.0,
    radius_variation: float = 0.0,
    centerline: Optional[np.ndarray] = None,
    # ---- Pulsatile velocity waveform (centerline v_max) ----
    v_mean: float = 1.7,
    v_amplitude: float = 2.0,
    v_harmonic: float = 0.7,
    phi_1: float = 0.0,
    phi_2: float = 0.375 * np.pi,
    f0_hz: float = 2.5,
    frame_dt_s: float = 0.004,
    # ---- Optics ----
    NA: float = 0.3,
    wavelength_um: float = 0.605,
    bead_diameter_um: float = 0.5,
    n_medium: float = 1.0,
    magnification: float = 10.0,
    px_size_um: float = 1.7,
    # ---- Particles ----
    particle_density: float = 15.0,
    streak_width: Optional[float] = None,
    intensity_mean: float = 1.0,
    intensity_spread: float = 0.3,
    aggregate_prob: float = 0.10,   # probability a particle is part of an aggregate
    # ---- Depth-of-field ----
    dof_z_c_px: Optional[float] = None,
    z_0_um: float = 0.0,
    doc_model: str = 'squared_lorentzian',
    doc_p: float = 2.0,
    single_depth_z: Optional[float] = None,
    oof_width_factor: float = 4.0,
    oof_intensity_factor: float = 0.5,
    # ---- Artifacts ----
    n_bright_spots: int = 0,
    bright_spot_intensity: float = 2.0,
    bright_spot_width: float = 1.0,
    # ---- Noise ----
    background: float = 0.4,
    vessel_autofluorescence: float = 0.0,
    noise_sigma: float = 0.24,
    poisson_noise: bool = True,
    gain: float = 21.0,
    flicker_sigma: float = 0.08,
    # ---- Other ----
    zero_outside: bool = True,
    seed: Optional[int] = 42,
    Q_harmonics: Optional[np.ndarray] = None,
) -> Dict:
    """Generate a synthetic 2D vessel video with discrete flowing particles.

    Produces a (T, H_img, W_img) video stack directly — no kymograph intermediary.
    Particles are 3D fluorescent beads following Poiseuille flow along an
    arbitrary centerline; DOC weights their in-focus intensity and PSF broadens
    them with depth.

    Parameters
    ----------
    T : int
        Number of frames.
    W : int
        Desired arc length of the vessel (pixels) when ``centerline`` is None;
        ignored when a centerline is supplied (image size comes from the
        centerline's bounding box).
    vessel_radius_px : float
        Vessel radius R in pixels.
    centerline : (N, 2) array_like [row, col] or None
        Vessel centreline at ~1 px spacing.  When None a straight horizontal
        vessel of length *W* is constructed automatically.
    v_mean, v_amplitude, v_harmonic, phi_1, phi_2, f0_hz, frame_dt_s :
        Pulsatile centerline velocity waveform.
    particle_density : float
        Expected number of new particles entering the visible region per frame.
    streak_width : float or None
        In-focus PSF sigma in pixels.  If None, computed from optics.
    dof_z_c_px : float or None
        DOC half-depth z_c in pixels.  If None, computed from optics.

    Returns
    -------
    dict with keys:
        'frames'      : (T, H_img, W_img) float64 — raw (pre-normalisation) video
        'v_max_t'     : (T,) float64 — centerline velocity waveform
        'Q_t'         : (T,) float64 — true Q = πR²v_max/2  [px²/frame]
        'centerline'  : (N, 2) float64 — vessel centreline used
        'inside_mask' : (H_img, W_img) bool — pixels within R of centreline
        'params'      : dict — all generation parameters
    """
    from scipy.spatial import cKDTree

    R = vessel_radius_px
    rng = np.random.default_rng(seed if seed is not None else 42)

    # ── Optics defaults ───────────────────────────────────────────────────
    if streak_width is None:
        streak_width = compute_psf_sigma_px(NA, wavelength_um, px_size_um)
    if dof_z_c_px is None:
        dof_z_c_px = compute_doc_z_c(NA, wavelength_um, bead_diameter_um,
                                      n_medium, magnification) / px_size_um
    z_c = max(dof_z_c_px, 0.1)
    z_0_px = z_0_um / max(px_size_um, 1e-6)

    # ── Velocity waveform ─────────────────────────────────────────────────
    t_arr = np.arange(T, dtype=np.float64)
    phase_t = 2.0 * np.pi * f0_hz * t_arr * frame_dt_s

    # If Q_harmonics provided, use arbitrary Fourier waveform instead of simple 2-harmonic
    if Q_harmonics is not None and len(Q_harmonics) > 0:
        # Reconstruct Q(t) from harmonics, then derive v_max(t)
        omega0 = 2.0 * np.pi * f0_hz
        t_sec = t_arr * frame_dt_s
        Q_t = np.full(T, float(Q_harmonics[0].real))
        for n in range(1, len(Q_harmonics)):
            Q_t += (Q_harmonics[n].real * np.cos(n * omega0 * t_sec)
                    - Q_harmonics[n].imag * np.sin(n * omega0 * t_sec))
        # v_max = 2Q / (pi R^2)  (Poiseuille: v_max = 2 * v_mean)
        v_max_t = 2.0 * Q_t / (np.pi * R**2)
    else:
        v_max_t = (v_mean
                   + v_amplitude * np.sin(phase_t + phi_1)
                   + v_harmonic  * np.sin(2.0 * phase_t + phi_2))
        Q_t = np.pi * R ** 2 * v_max_t / 2.0

    # ── Centerline and image geometry ─────────────────────────────────────
    if centerline is None:
        # Straight vessel: centerline runs horizontally at row R+2, col 0..W-1
        n_cl = max(W + 1, 3)
        centerline = np.column_stack([
            np.full(n_cl, float(R) + 2.0),
            np.linspace(0.0, float(W - 1), n_cl),
        ])
        H_img = int(2 * np.ceil(R)) + 6
        W_img = W
    else:
        centerline = np.asarray(centerline, dtype=float)
        # Shift so vessel fits with R+2 margin on all sides
        margin_px = float(R) + 2.0
        shift_r = max(0.0, margin_px - float(centerline[:, 0].min()))
        shift_c = max(0.0, margin_px - float(centerline[:, 1].min()))
        if shift_r > 0.0 or shift_c > 0.0:
            centerline = centerline + np.array([[shift_r, shift_c]])
        H_img = int(np.ceil(float(centerline[:, 0].max()) + margin_px)) + 1
        W_img = int(np.ceil(float(centerline[:, 1].max()) + margin_px)) + 1

    s_cl, tangents_cl, normals_cl = _centerline_geometry(centerline)
    S = float(s_cl[-1])  # total arc length

    # ── Radius profile R(s) ───────────────────────────────────────────────
    # Length-scale of variation ~ L_CORR_PX (in pixels). Control points placed
    # so they span at least this interval — avoids high-frequency wiggle on
    # short vessels. Each interior control point sampled from
    # Normal(0, σ=radius_variation·R), clipped to ±1.5σ so the spline cannot
    # overshoot into visible step-like artifacts.
    if radius_variation > 0:
        from scipy.interpolate import CubicSpline
        L_CORR_PX = 60.0
        n_ctrl_r = max(3, int(np.ceil(S / L_CORR_PX)) + 1)
        t_ctrl_r = np.linspace(0, S, n_ctrl_r)
        sigma = float(radius_variation) * R
        y_ctrl_r = np.clip(rng.normal(0, sigma, size=n_ctrl_r), -1.5 * sigma, 1.5 * sigma)
        y_ctrl_r[0] = y_ctrl_r[-1] = 0.0   # pin endpoints
        cs_r = CubicSpline(t_ctrl_r, y_ctrl_r, bc_type='clamped')
        R_of_s = R + cs_r(s_cl)
        R_of_s = np.clip(R_of_s, R * 0.75, R * 1.25)
    else:
        R_of_s = np.full(len(s_cl), float(R))

    # Vessel mask: per-pixel distance ≤ R(s) at nearest CL point,
    # excluding semi-circular end-caps.
    cl_tree = cKDTree(centerline)
    rr_g, cc_g = np.mgrid[0:H_img, 0:W_img]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                              cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree.query(pts_g)
    R_at_nearest = R_of_s[nearest_g]
    inside_mask = (
        (dist_g <= R_at_nearest)
        & (nearest_g > 0)
        & (nearest_g < len(centerline) - 1)
    ).reshape(H_img, W_img)

    frames = np.zeros((T, H_img, W_img), dtype=np.float64)

    # ── Particle pool ─────────────────────────────────────────────────────
    # Entry times sampled from an inhomogeneous Poisson process with rate
    # proportional to |v_max(t)| (flux through the boundary).  This prevents
    # parabolic density waves: during fast phases more particles enter AND
    # cross faster, keeping the in-vessel density approximately constant.
    v_abs = max(abs(v_mean), 0.5)
    crossing_frames = int(np.ceil(S / v_abs)) + 10

    # Build velocity array over the extended window [-crossing_frames, T)
    period_frames = max(1, round(1.0 / (f0_hz * frame_dt_s)))
    pre_indices = np.arange(-crossing_frames, 0)
    v_pre = v_max_t[pre_indices % period_frames % T]  # periodic extension
    v_ext = np.concatenate([v_pre, v_max_t])          # length = crossing_frames + T

    # Flux-proportional rate; normalised so mean rate == particle_density
    flux = np.abs(v_ext)
    mean_flux = max(flux.mean(), 1e-6)
    rate = particle_density * flux / mean_flux          # (crossing_frames + T,)

    n_particles = rng.poisson(rate.sum())
    p_entry = rate / rate.sum()
    t_indices = rng.choice(len(rate), size=n_particles, replace=True, p=p_entry)
    # t_indices in [0, crossing_frames+T); shift so 0 → -crossing_frames
    t_starts = (t_indices - crossing_frames).astype(float)
    t_starts += rng.uniform(-0.5, 0.5, size=n_particles)  # sub-frame jitter

    # Flux-weighted cross-section sampling.
    # For uniform volume concentration in steady state the entry flux through
    # each annulus must be proportional to v(r,z).  We accept candidates from
    # the uniform disk with probability pscale = 1 - (r²+z²)/R² so that
    # P(r,z) ∝ v(r,z).  Mean acceptance ≈ 0.5 → oversample by ~4×.
    if single_depth_z is not None:
        r_max = np.sqrt(max(R ** 2 - single_depth_z ** 2, 0.0))
        ps_max = max(1.0 - single_depth_z ** 2 / R ** 2, 1e-6)
        r_p = np.empty(n_particles)
        z_p = np.full(n_particles, single_depth_z, dtype=float)
        filled = 0
        while filled < n_particles:
            batch = max((n_particles - filled) * 3, 200)
            rc = rng.uniform(-r_max, r_max, size=batch)
            ps_cand = np.maximum(0.0,
                                 1.0 - (rc ** 2 + single_depth_z ** 2) / R ** 2)
            accept = rng.uniform(size=batch) < (ps_cand / ps_max)
            n_ok = accept.sum()
            n_take = min(n_ok, n_particles - filled)
            r_p[filled:filled + n_take] = rc[accept][:n_take]
            filled += n_take
    else:
        r_p = np.empty(n_particles)
        z_p = np.empty(n_particles)
        filled = 0
        while filled < n_particles:
            batch = max((n_particles - filled) * 4, 200)
            rc = rng.uniform(-R, R, size=batch)
            zc = rng.uniform(-R, R, size=batch)
            d2 = rc ** 2 + zc ** 2
            in_circle = d2 < R ** 2
            ps_cand = np.maximum(0.0, 1.0 - d2 / R ** 2)
            accept = in_circle & (rng.uniform(size=batch) < ps_cand)
            n_ok = accept.sum()
            n_take = min(n_ok, n_particles - filled)
            r_p[filled:filled + n_take] = rc[accept][:n_take]
            z_p[filled:filled + n_take] = zc[accept][:n_take]
            filled += n_take

    # Fractional radial positions (for variable-R rendering)
    r_frac = r_p / R   # in-plane fractional ∈ (-1, 1)
    z_frac = z_p / R   # depth fractional ∈ (-1, 1)

    # Poiseuille velocity scale at mean radius: 1 - (r²+z²)/R²
    pscale = np.maximum(0.0, 1.0 - (r_frac ** 2 + z_frac ** 2))

    # Defocus: width grows, intensity falls with depth
    u2 = ((z_p - z_0_px) / z_c) ** 2
    widths = streak_width * np.clip(np.sqrt(1.0 + u2), 1.0, oof_width_factor)
    int_scale = np.clip(1.0 / (1.0 + u2), oof_intensity_factor, 1.0)
    if intensity_spread > 0:
        base_int = rng.lognormal(
            np.log(intensity_mean) - 0.5 * intensity_spread ** 2,
            intensity_spread, size=n_particles)
    else:
        base_int = np.full(n_particles, intensity_mean)

    # Bead aggregation: geometric number of beads stuck together.
    # Each extra bead adds intensity; apparent width grows as sqrt(n_beads)
    # (convolution of PSFs for a compact clump).
    if aggregate_prob > 0.0:
        p_geom = 1.0 - aggregate_prob
        n_beads = np.clip(
            rng.geometric(p=max(p_geom, 0.01), size=n_particles), 1, 10
        ).astype(float)
        base_int = base_int * n_beads
        widths = widths * np.sqrt(n_beads)

    intensities = base_int * int_scale

    # Arc entry positions (slightly before s=0)
    s_entry = rng.uniform(-2.0 * streak_width, 0.0, size=n_particles)

    # ── Render each particle ──────────────────────────────────────────────
    # Particles are rendered as elliptical Gaussian blobs: the PSF is
    # convolved with the intra-frame motion blur (a box of length v*ps
    # along the local tangent).  Approximating the box by a Gaussian:
    #   σ_tangent  = sqrt(sw² + (v·ps)²/12)
    #   σ_normal   = sw
    # The blob is evaluated in the (tangent, normal) basis at each pixel.
    for i in range(n_particles):
        t0 = t_starts[i]
        t0_floor = int(np.floor(t0))
        t_lo = max(0, t0_floor)
        if t_lo >= T:
            continue

        sw = float(widths[i])
        I_i = float(intensities[i])
        ps = float(pscale[i])
        r_frac_i = float(r_frac[i])

        # Arc position at t_lo: advance with v_mean for any negative pre-frames
        s_initial = float(s_entry[i])
        if t0_floor < 0:
            s_initial += ps * v_mean * float(-t0_floor)

        # Vectorised arc positions for all frames [t_lo, T)
        v_frames = v_max_t[t_lo:T]
        s_p_arr = s_initial + ps * np.cumsum(v_frames)  # (T - t_lo,)

        # Kill particle permanently once it leaves [0, S].
        # A particle that exits to the left (s<0 during backflow) or right
        # (s>S) has left the field of view.  It must not re-enter later when
        # the flow reverses — fresh particles from the flux-proportional
        # entry process fill that role instead.
        inside = (s_p_arr >= 0.0) & (s_p_arr <= S)
        if not inside.any():
            continue
        # Find the contiguous run starting from first-visible frame.
        # Once the particle exits, all later frames are discarded.
        first_in = int(np.argmax(inside))
        after_first = inside[first_in:]
        if not after_first.all():
            first_out = int(np.argmin(after_first))  # first False after entry
            inside[first_in + first_out:] = False

        vis_idx = np.where(inside)[0]
        s_vis = s_p_arr[vis_idx]

        # Batch-interpolate centreline position, tangent, CW normal, and local R
        row_c = np.interp(s_vis, s_cl, centerline[:, 0])
        col_c = np.interp(s_vis, s_cl, centerline[:, 1])
        t_r   = np.interp(s_vis, s_cl, tangents_cl[:, 0])
        t_c   = np.interp(s_vis, s_cl, tangents_cl[:, 1])
        n_r   = np.interp(s_vis, s_cl, normals_cl[:, 0])
        n_c   = np.interp(s_vis, s_cl, normals_cl[:, 1])
        R_loc = np.interp(s_vis, s_cl, R_of_s)  # local radius

        # 2-D particle centres: fractional radius × local R(s)
        row_centers = row_c + r_frac_i * R_loc * n_r   # (n_vis,)
        col_centers = col_c + r_frac_i * R_loc * n_c   # (n_vis,)

        # Per-frame motion-blur σ along tangent:
        #   Continuity: v_local = v_max × (R_mean/R_local)²
        #   motion_length = |v_local * ps|,  σ_box ≈ L/√12
        #   σ_t = sqrt(sw² + L²/12)
        v_vis = v_frames[vis_idx]   # centreline velocity at each visible frame
        v_correction = (R / R_loc) ** 2  # continuity: faster in narrow sections
        motion_len = np.abs(v_vis * ps * v_correction)
        sigma_t = np.sqrt(sw ** 2 + motion_len ** 2 / 12.0)  # (n_vis,)
        sigma_n = sw   # perpendicular: PSF only

        # Render elliptical blobs in (tangent, normal) basis
        for k in range(len(vis_idx)):
            t = t_lo + int(vis_idx[k])
            rp = float(row_centers[k])
            cp = float(col_centers[k])
            st = float(sigma_t[k])
            sn = sigma_n

            # Bounding box must cover the larger σ_t direction in image coords
            # Max extent in each image axis:
            tr_k = float(t_r[k])
            tc_k = float(t_c[k])
            nr_k = float(n_r[k])
            nc_k = float(n_c[k])
            ext_row = 3.0 * (abs(tr_k) * st + abs(nr_k) * sn) + 1.0
            ext_col = 3.0 * (abs(tc_k) * st + abs(nc_k) * sn) + 1.0

            r0 = max(0, int(rp - ext_row))
            r1 = min(H_img, int(rp + ext_row) + 2)
            c0 = max(0, int(cp - ext_col))
            c1 = min(W_img, int(cp + ext_col) + 2)
            if r0 >= r1 or c0 >= c1:
                continue

            # Pixel offsets from particle centre
            dr = np.arange(r0, r1, dtype=float) - rp   # (nr,)
            dc = np.arange(c0, c1, dtype=float) - cp   # (nc,)
            DR, DC = np.meshgrid(dr, dc, indexing='ij')  # (nr, nc)

            # Project onto tangent / normal
            d_t = DR * tr_k + DC * tc_k   # component along tangent
            d_n = DR * nr_k + DC * nc_k   # component along normal

            blob = I_i * np.exp(
                -0.5 * (d_t / st) ** 2 - 0.5 * (d_n / sn) ** 2
            )
            frames[t, r0:r1, c0:c1] += blob

    # ── Zero outside vessel ───────────────────────────────────────────────
    if zero_outside:
        frames[:, ~inside_mask] = 0.0

    # ── Static bright spots ───────────────────────────────────────────────
    if n_bright_spots > 0:
        spot_s   = rng.uniform(0.05 * S, 0.95 * S, size=n_bright_spots)
        spot_rho = rng.uniform(-R * 0.8, R * 0.8, size=n_bright_spots)
        spot_I   = rng.lognormal(
            np.log(bright_spot_intensity) - 0.5 * 0.3 ** 2,
            0.3, size=n_bright_spots)
        bsw = bright_spot_width
        for ss, sr, si in zip(spot_s, spot_rho, spot_I):
            row_c = float(np.interp(ss, s_cl, centerline[:, 0]))
            col_c = float(np.interp(ss, s_cl, centerline[:, 1]))
            n_r   = float(np.interp(ss, s_cl, normals_cl[:, 0]))
            n_c   = float(np.interp(ss, s_cl, normals_cl[:, 1]))
            rp = row_c + sr * n_r
            cp = col_c + sr * n_c
            r0 = max(0, int(rp - 3 * bsw))
            r1 = min(H_img, int(rp + 3 * bsw) + 2)
            c0 = max(0, int(cp - 3 * bsw))
            c1 = min(W_img, int(cp + 3 * bsw) + 2)
            if r0 < r1 and c0 < c1:
                rr_b = np.arange(r0, r1, dtype=float)
                cc_b = np.arange(c0, c1, dtype=float)
                blob = (si
                        * np.exp(-0.5 * ((rr_b - rp) / bsw) ** 2)[:, None]
                        * np.exp(-0.5 * ((cc_b - cp) / bsw) ** 2)[None, :])
                frames[:, r0:r1, c0:c1] += blob[None, ...]

    # ── Temporal flicker ──────────────────────────────────────────────────
    if flicker_sigma > 0:
        flicker = 1.0 + rng.normal(0.0, flicker_sigma, size=T)
        frames *= flicker[:, None, None]

    # ── Background ────────────────────────────────────────────────────────
    if vessel_autofluorescence > 0:
        frames[:, inside_mask] += vessel_autofluorescence
    frames += background
    np.clip(frames, 0.0, None, out=frames)

    # ── Photon / readout noise ────────────────────────────────────────────
    if poisson_noise:
        frames = rng.poisson(
            np.maximum(frames * gain, 0.0)).astype(np.float64) / gain
    if noise_sigma > 0:
        frames += rng.normal(0.0, noise_sigma, size=frames.shape)

    # ── Final vessel mask (zero outside after all noise) ──────────────────
    if zero_outside:
        frames[:, ~inside_mask] = 0.0

    return {
        'frames':       frames,
        'v_max_t':      v_max_t,
        'Q_t':          Q_t,
        'vessel_radius_px': R,
        'centerline':   centerline,
        'R_of_s':       R_of_s,
        'inside_mask':  inside_mask,
        # Per-particle data for visualisation
        'r_p':          r_p,            # (n_particles,) in-plane transverse offset (px)
        'z_p':          z_p,            # (n_particles,) depth offset (px)
        't_starts':     t_starts,       # (n_particles,) entry time (frame units, can be <0)
        's_entry':      s_entry,        # (n_particles,) initial arc position (px)
        'pscale':       pscale,         # (n_particles,) Poiseuille velocity scale
        'widths':       widths,         # (n_particles,) apparent PSF width (px)
        'intensities':  intensities,    # (n_particles,) apparent intensity
        'S':            S,              # total arc length (px)
        'v_max_t_ext':  v_max_t,        # alias — same as v_max_t (for particle tracking)
        'params': dict(
            T=T, W=W, vessel_radius_px=R, H=H_img, W_img=W_img,
            v_mean=v_mean, v_amplitude=v_amplitude, v_harmonic=v_harmonic,
            phi_1=phi_1, phi_2=phi_2, f0_hz=f0_hz, frame_dt_s=frame_dt_s,
            NA=NA, wavelength_um=wavelength_um, px_size_um=px_size_um,
            streak_width=streak_width, dof_z_c_px=dof_z_c_px,
            z_0_um=z_0_um, doc_model=doc_model, doc_p=doc_p,
            single_depth_z=single_depth_z,
            oof_width_factor=oof_width_factor,
            oof_intensity_factor=oof_intensity_factor,
            aggregate_prob=aggregate_prob,
            radius_variation=radius_variation,
            particle_density=particle_density,
            background=background, vessel_autofluorescence=vessel_autofluorescence,
            noise_sigma=noise_sigma,
            poisson_noise=poisson_noise, gain=gain,
            flicker_sigma=flicker_sigma, seed=seed,
        ),
    }
