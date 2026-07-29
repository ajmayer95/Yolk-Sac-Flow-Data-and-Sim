"""
Kymograph sampling functions for vessel blood flow analysis.

Provides functions to sample kymographs from video stacks along vessel
centerlines, including radial offset sampling for profile analysis.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Literal, Dict


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation sampling from 2D image."""
    h, w = img.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    xf = x - x0
    yf = y - y0
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    return Ia * (1 - xf) * (1 - yf) + Ib * xf * (1 - yf) + Ic * (1 - xf) * yf + Id * xf * yf


def sample_kymo(
    stack: np.ndarray,
    coords_xy: np.ndarray,
    mode: Literal["band", "line"] = "band",
    band_radius: int = 5,
    band_step: int = 1,
    sampling: Literal["bilinear", "nearest"] = "bilinear",
    reduce: Literal["median", "mean", "max"] = "median",
) -> np.ndarray:
    """
    Sample kymograph from image stack along vessel centerline.

    Args:
        stack: (T, H, W) video frames
        coords_xy: (M, 2) array of [x, y] centerline coordinates
        mode: "band" for perpendicular band sampling, "line" for single line
        band_radius: Half-width of perpendicular band in pixels
        band_step: Step size within band (for subsampling)
        sampling: Interpolation method ("bilinear" or "nearest")
        reduce: Reduction method across band ("median", "mean", or "max")

    Returns:
        kymo: (T, M) kymograph array
    """
    T = stack.shape[0]
    M = coords_xy.shape[0]
    kymo = np.zeros((T, M), np.float32)

    # Compute perpendicular normals
    if mode == "band" and band_radius > 0:
        d = np.gradient(coords_xy, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
        offs = np.arange(-band_radius, band_radius + 1, max(1, int(band_step)), dtype=float)
    else:
        n = None
        offs = np.array([0.0], float)

    Xc, Yc = coords_xy[:, 0], coords_xy[:, 1]

    for t in range(T):
        frm = stack[t].astype(np.float32)
        vals = []
        for off in offs:
            if n is None:
                x, y = Xc, Yc
            else:
                x, y = Xc + off * n[:, 0], Yc + off * n[:, 1]

            if sampling == "bilinear":
                v = _bilinear_sample(frm, x, y)
            else:
                xi = np.clip(np.rint(x).astype(int), 0, frm.shape[1] - 1)
                yi = np.clip(np.rint(y).astype(int), 0, frm.shape[0] - 1)
                v = frm[yi, xi].astype(np.float32)
            vals.append(v)

        vals = np.stack(vals, axis=0)  # (N_band, M)

        if reduce == "median":
            kymo[t, :] = np.median(vals, axis=0)
        elif reduce == "mean":
            kymo[t, :] = np.mean(vals, axis=0)
        else:
            kymo[t, :] = np.max(vals, axis=0)

    return kymo


def sample_kymo_at_offset(
    coords: np.ndarray,
    stack: np.ndarray,
    center_offset: int = 0,
    band_radius: int = 2,
    band_reduce: Literal["median", "mean", "max"] = "median",
    smooth_sigma: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Sample RAW kymograph at a specific radial offset from centerline.

    This is used for radial velocity profile analysis where we sample
    at multiple offsets perpendicular to the vessel centerline.

    Args:
        coords: (N, 2) array of [x, y] centerline coordinates
        stack: (T, H, W) video frames
        center_offset: Offset in pixels (positive = one side, negative = other)
        band_radius: Half-width of band around that offset
        band_reduce: Reduction method ("median", "mean", "max")
        smooth_sigma: Gaussian smoothing sigma for offset curves (removes cusps at kinks).
            If None, uses adaptive smoothing based on offset magnitude.

    Returns:
        kymo: (T, M) kymograph array, or None on failure
    """
    try:
        # Compute normal vectors to centerline using gradient
        d = np.gradient(coords, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)

        # Shift centerline to desired offset
        coords_offset = coords + center_offset * n

        # Smooth offset curves to remove cusps at kinks/bends
        # This prevents sampling artifacts at sharp vessel corners
        if center_offset != 0:
            try:
                from scipy.ndimage import gaussian_filter1d

                # Adaptive sigma based on offset magnitude (larger offset = more smoothing needed)
                if smooth_sigma is None:
                    smooth_sigma = max(2.0, abs(center_offset) * 0.3)

                coords_offset = np.column_stack([
                    gaussian_filter1d(coords_offset[:, 0], sigma=smooth_sigma, mode='nearest'),
                    gaussian_filter1d(coords_offset[:, 1], sigma=smooth_sigma, mode='nearest')
                ])
            except ImportError:
                pass  # scipy not available, use unsmoothed coords

        # Sample kymograph at this offset with narrow band
        kymo = sample_kymo(
            stack,
            coords_offset,
            mode="band",
            band_radius=band_radius,
            band_step=1,
            sampling="bilinear",
            reduce=band_reduce,
        )

        return kymo
    except Exception as e:
        print(f"Warning: Failed to sample at offset {center_offset}: {e}")
        return None


def apply_temporal_detrend(
    kymo: np.ndarray,
    window_size: int = 25,
    frame_rate: float = 250.0,
) -> np.ndarray:
    """
    Apply temporal detrending to kymograph using Butterworth high-pass filter.

    This removes stationary features (vessel walls, background) while
    preserving moving features (blood cells). The cutoff frequency is
    derived from the window_size to match the original median-filter
    behaviour: f_cutoff = 1 / (window_size * dt).

    Args:
        kymo: (T, M) kymograph array
        window_size: Temporal window (frames) — converted to cutoff frequency.
            0 = no detrending.
        frame_rate: Sampling rate in Hz (default 250).

    Returns:
        kymo_detrend: (T, M) detrended kymograph
    """
    if window_size == 0:
        return kymo

    try:
        from scipy.signal import butter, sosfiltfilt

        T = kymo.shape[0]
        if T < 3:
            return kymo

        # Convert window_size to cutoff: the median filter removed
        # frequencies below ~1/window_period. Match that as a high-pass.
        nyquist = frame_rate / 2.0
        f_cutoff = frame_rate / max(window_size, 3)
        # Clamp to valid Butterworth range (0 < Wn < 1 in normalized freq)
        Wn = f_cutoff / nyquist
        if Wn >= 1.0:
            Wn = 0.95
        if Wn <= 0.0:
            return kymo

        sos = butter(2, Wn, btype='high', output='sos')
        kymo_detrend = sosfiltfilt(sos, kymo, axis=0).astype(kymo.dtype)

        return kymo_detrend
    except Exception as e:
        print(f"Warning: Temporal detrending failed: {e}")
        return kymo


def _detrend_median(kymo: np.ndarray, window_size: int) -> np.ndarray:
    """Median-filter baseline subtraction (original method, slow)."""
    from scipy.ndimage import median_filter
    T = kymo.shape[0]
    if T < 3 or window_size == 0:
        return kymo
    effective_window = min(window_size, T)
    if effective_window % 2 == 0:
        effective_window -= 1
    effective_window = max(1, effective_window)
    baseline = median_filter(kymo, size=(effective_window, 1), mode='reflect')
    return kymo - baseline


def _detrend_butter(kymo: np.ndarray, window_size: int,
                    frame_rate: float = 250.0) -> np.ndarray:
    """Butterworth high-pass detrend (fast)."""
    from scipy.signal import butter, sosfiltfilt
    T = kymo.shape[0]
    if T < 3 or window_size == 0:
        return kymo
    nyquist = frame_rate / 2.0
    f_cutoff = frame_rate / max(window_size, 3)
    Wn = f_cutoff / nyquist
    if Wn >= 1.0:
        Wn = 0.95
    if Wn <= 0.0:
        return kymo
    sos = butter(2, Wn, btype='high', output='sos')
    return sosfiltfilt(sos, kymo, axis=0).astype(kymo.dtype)


def compare_detrending(
    kymo: np.ndarray,
    frame_rate: float = 250.0,
    f0_hz: Optional[float] = None,
    gst_windows: Optional[list] = None,
    vessel_id: str = "",
    out_dir: Optional[str] = None,
):
    """Show side-by-side detrending comparison: median vs Butterworth at multiple windows.

    Generates a figure with rows = detrending variants, cols = [kymograph, v(t)].
    """
    import matplotlib.pyplot as plt
    import time as _time
    from .gst import compute_gst_velocity, compute_weighted_median_velocity

    if gst_windows is None:
        from .config import GST_WINDOWS
        gst_windows = GST_WINDOWS

    T, M = kymo.shape
    dt = 1.0 / frame_rate

    # Detrend window sizes to test (in frames)
    if f0_hz is not None and f0_hz > 0:
        base_window = int(4.0 / f0_hz / dt)
    else:
        base_window = int(3.0 * frame_rate)
    base_window = max(50, min(base_window, 2000))

    # Test a few window sizes around the base
    windows = sorted(set([
        max(25, base_window // 2),
        base_window,
        min(2000, base_window * 2),
    ]))

    methods = []
    for w in windows:
        methods.append(('median', w))
        methods.append(('butter', w))

    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(10, 2.0 * n_methods),
                             dpi=120, squeeze=False)
    fig.suptitle(f"Detrending comparison — vessel {vessel_id}", fontsize=10, y=0.995)

    t_s = np.arange(T) * dt
    vmin_k, vmax_k = np.percentile(kymo, [2, 98])

    for row, (method, w) in enumerate(methods):
        t0 = _time.time()
        if method == 'median':
            kymo_d = _detrend_median(kymo, w)
        else:
            kymo_d = _detrend_butter(kymo, w, frame_rate)
        elapsed = _time.time() - t0

        # GST → v(t)
        vel_map, conf_px = compute_gst_velocity(kymo_d, windows=gst_windows)
        v_hat = compute_weighted_median_velocity(vel_map, conf_px=conf_px)

        # Cutoff frequency for label
        f_cut = frame_rate / max(w, 1)

        # Kymograph panel
        ax = axes[row, 0]
        ax.imshow(kymo_d.T, aspect='auto', cmap='gray',
                  extent=[0, T * dt, M, 0],
                  vmin=np.percentile(kymo_d, 2), vmax=np.percentile(kymo_d, 98))
        ax.set_ylabel(f"{method}\nw={w} f_c={f_cut:.2f}Hz\n({elapsed*1000:.0f}ms)",
                      fontsize=7)
        if row == 0:
            ax.set_title("Detrended kymograph", fontsize=8)
        if row < n_methods - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=7)
        ax.tick_params(labelsize=6)

        # v(t) panel
        ax = axes[row, 1]
        v_finite = v_hat[np.isfinite(v_hat)]
        ax.plot(t_s[:len(v_hat)], v_hat, linewidth=0.5, color='C0')
        ax.axhline(0, color='gray', linewidth=0.3)
        if len(v_finite) > 0:
            mean_v = np.nanmean(v_hat)
            std_v = np.nanstd(v_hat)
            ax.axhline(mean_v, color='red', linewidth=0.5, linestyle='--')
            ax.set_ylabel(f"μ={mean_v:.2f} σ={std_v:.2f}", fontsize=7)
        if row == 0:
            ax.set_title("v(t) [px/frame]", fontsize=8)
        if row < n_methods - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.tight_layout()

    if out_dir is not None:
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(out_dir) / f"detrend_compare_{vessel_id}.png"
        fig.savefig(str(fname), bbox_inches='tight')
        print(f"  Saved: {fname}")

    plt.show()
    return fig


def get_vessel_coords_from_graph(
    G,
    edge_key: Tuple[int, int],
    px_spacing: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Extract vessel centerline coordinates from graph edge.

    Args:
        G: NetworkX graph with edge 'pts' attribute
        edge_key: (u, v) edge tuple
        px_spacing: Desired spacing between points (for resampling)

    Returns:
        coords: (M, 2) array of [x, y] coordinates
        radius: Vessel radius in pixels
    """
    u, v = edge_key
    edge_data = G.edges[u, v]

    # Get centerline points - try 'pts' first (viewer), then 'path' (collapse_to_vessel_graph)
    pts = edge_data.get('pts', None)
    if pts is None:
        pts = edge_data.get('path', None)
    if pts is None:
        # Fall back to node positions
        pts = np.array([G.nodes[u]['pos'], G.nodes[v]['pos']])

    coords = np.array(pts)

    # Get radius - use MINIMUM along the entire chain (conservative for Poiseuille fitting)
    # This ensures we don't overestimate the vessel radius, which would affect
    # velocity profile fitting and flow rate calculations

    # First check for per-point radii array (from collapse_to_vessel_graph)
    path_radii = edge_data.get('radii', None)
    if path_radii is not None and len(path_radii) > 0:
        # Use minimum radius along the entire centerline path
        radius = float(np.min(path_radii))
    else:
        # Fall back to node radii at endpoints
        r_u = G.nodes[u].get('radius', 5.0)
        r_v = G.nodes[v].get('radius', 5.0)

        # Also check for single edge radius attribute
        edge_radius = edge_data.get('radius', None)
        if edge_radius is not None:
            radius = min(r_u, r_v, edge_radius)
        else:
            radius = min(r_u, r_v)

    return coords, float(radius)


def compute_streak_quality(
    kymo: np.ndarray,
    smoothing_sigma: float = 3.0,
) -> np.ndarray:
    """
    Compute streak quality score per column (arc position) of kymograph.

    Combines multiple metrics:
    1. Temporal variance - high variance indicates motion (blood cells)
    2. Directional structure - diagonal streaks have oriented gradients

    Args:
        kymo: (T, M) kymograph array
        smoothing_sigma: Gaussian smoothing for gradient computation

    Returns:
        quality: (M,) array of quality scores in [0, 1]
    """
    try:
        from scipy.ndimage import gaussian_filter, sobel
    except ImportError:
        # Fallback: just use temporal variance
        temporal_var = np.var(kymo, axis=0)
        return temporal_var / (temporal_var.max() + 1e-6)

    T, M = kymo.shape

    # 1. Temporal variance (normalized)
    # High temporal variance = motion at this arc position
    temporal_var = np.var(kymo, axis=0)
    var_score = temporal_var / (temporal_var.max() + 1e-6)

    # 2. Directional structure using gradients
    # Diagonal streaks have both dt and dx components
    # Horizontal bands have only dx, vertical only dt
    kymo_smooth = gaussian_filter(kymo.astype(np.float64), sigma=smoothing_sigma)

    # Compute gradients
    grad_t = sobel(kymo_smooth, axis=0)  # Temporal gradient
    grad_x = sobel(kymo_smooth, axis=1)  # Spatial gradient

    # Gradient magnitude per column
    grad_mag = np.sqrt(grad_t**2 + grad_x**2)
    grad_strength = np.mean(grad_mag, axis=0)
    grad_score = grad_strength / (grad_strength.max() + 1e-6)

    # 3. Anisotropy: ratio of directional components
    # True streaks have balanced t and x gradients (diagonal)
    # Horizontal bands have strong x, weak t
    abs_grad_t = np.mean(np.abs(grad_t), axis=0)
    abs_grad_x = np.mean(np.abs(grad_x), axis=0)

    # Anisotropy score: 1 when balanced, 0 when one dominates
    min_grad = np.minimum(abs_grad_t, abs_grad_x)
    max_grad = np.maximum(abs_grad_t, abs_grad_x)
    aniso_score = min_grad / (max_grad + 1e-6)

    # Combine scores (geometric mean to require all to be good)
    quality = (var_score * grad_score * aniso_score) ** (1/3)

    return quality


def find_good_arc_region(
    kymo: np.ndarray,
    min_quality: float = 0.3,
    min_length_frac: float = 0.2,
    min_arc_length_px: int = 25,
    edge_margin_frac: float = 0.1,
) -> Dict[str, any]:
    """
    Find the best contiguous arc region with high streak quality.

    Uses a sliding-window approach: for each candidate window length
    (from min_length to full available), finds the window with highest
    mean quality. Then selects the window that maximizes the objective
    mean_quality × sqrt(length / available_length), balancing quality
    and coverage.

    Args:
        kymo: (T, M) kymograph array
        min_quality: Minimum quality threshold for "good" columns
        min_length_frac: Minimum length as fraction of total arc
        min_arc_length_px: Minimum absolute arc length in pixels (default 25)
        edge_margin_frac: Fraction to exclude at each end (junction regions)

    Returns:
        Dict with:
            arc_start: Start index of good region
            arc_end: End index of good region (exclusive)
            arc_length: Length of good region
            quality: Per-column quality scores
            mean_quality: Mean quality in selected region
            good_mask: Boolean mask of good columns
    """
    T, M = kymo.shape

    # Compute quality per column
    quality = compute_streak_quality(kymo)

    # Exclude edges (junction regions often have artifacts)
    edge_margin = int(M * edge_margin_frac)
    edge_margin = max(1, edge_margin)

    # Valid region (excluding edges)
    valid_start = edge_margin
    valid_end = M - edge_margin
    available_length = valid_end - valid_start

    # Good columns mask (for output)
    valid_mask = np.zeros(M, dtype=bool)
    valid_mask[valid_start:valid_end] = True
    good_mask = (quality >= min_quality) & valid_mask

    # Minimum window length
    min_length = max(min_arc_length_px, int(M * min_length_frac))

    if available_length < min_length:
        # Vessel too short — use whatever is available
        return {
            'arc_start': valid_start,
            'arc_end': valid_end,
            'arc_length': available_length,
            'quality': quality,
            'mean_quality': float(np.mean(quality[valid_start:valid_end])),
            'good_mask': good_mask,
        }

    # Sliding window: for each window length, find best position using
    # cumulative sums for O(1) per-window mean computation.
    q_valid = quality[valid_start:valid_end]
    n = len(q_valid)
    cumsum = np.concatenate([[0.0], np.cumsum(q_valid)])

    best_score = -1.0
    best_start = 0
    best_end = n

    # Try window lengths from min_length to full available
    for wlen in range(min_length, n + 1):
        # Mean quality for each window position: O(n) per wlen
        window_sums = cumsum[wlen:] - cumsum[:n - wlen + 1]
        mean_q = window_sums / wlen

        # Best window at this length
        best_idx = np.argmax(mean_q)
        best_mean = mean_q[best_idx]

        # Objective: balance quality and coverage
        # coverage^0.25 weakly rewards length — quality dominates,
        # so a shorter high-quality region beats a longer mediocre one
        coverage = wlen / n
        score = best_mean * coverage ** 0.25

        if score > best_score:
            best_score = score
            best_start = best_idx
            best_end = best_idx + wlen

    # Convert back to global indices
    arc_start = valid_start + best_start
    arc_end = valid_start + best_end

    return {
        'arc_start': arc_start,
        'arc_end': arc_end,
        'arc_length': arc_end - arc_start,
        'quality': quality,
        'mean_quality': float(np.mean(quality[arc_start:arc_end])),
        'good_mask': good_mask,
    }


def refine_arc_by_velocity_continuity(
    vel_map: np.ndarray,
    conf_px: np.ndarray,
    min_length_px: int = 10,
    smooth_kernel: int = 5,
    magnitude_tolerance: float = 0.5,
) -> Dict[str, any]:
    """
    Find the largest contiguous segment with spatially continuous velocity.

    Computes per-column coherence-weighted median velocity, then finds the
    longest run of columns where the velocity is consistent (same sign and
    magnitude within tolerance of the segment median).

    Args:
        vel_map: (T, M) velocity map from GST (px/frame)
        conf_px: (T, M) coherence map
        min_length_px: Minimum segment length in pixels
        smooth_kernel: Kernel size for median-filtering per-column velocities
        magnitude_tolerance: Fractional tolerance for velocity consistency.
            Columns must be within [1/(1+tol), 1+tol] × segment median |v|.

    Returns:
        Dict with:
            start, end: Column indices (0-based within input)
            v_col: Per-column coherence-weighted median velocity (raw)
            v_smooth: Smoothed per-column velocity
            refined: Whether refinement actually narrowed the arc
    """
    from scipy.ndimage import median_filter

    T, M = vel_map.shape
    no_change = {'start': 0, 'end': M, 'v_col': np.full(M, np.nan),
                 'v_smooth': np.full(M, np.nan), 'refined': False}

    if M < min_length_px:
        return no_change

    # Per-column coherence-weighted median velocity
    v_col = np.full(M, np.nan)
    for j in range(M):
        col_v = vel_map[:, j]
        col_c = conf_px[:, j]
        valid = np.isfinite(col_v) & np.isfinite(col_c) & (col_c > 0.1)
        if valid.sum() >= T * 0.3:
            weights = col_c[valid]
            vals = col_v[valid]
            order = np.argsort(vals)
            sorted_v = vals[order]
            sorted_w = weights[order]
            cum_w = np.cumsum(sorted_w)
            idx = np.searchsorted(cum_w, cum_w[-1] / 2.0)
            v_col[j] = sorted_v[min(idx, len(sorted_v) - 1)]

    valid_cols = np.isfinite(v_col)
    if valid_cols.sum() < min_length_px:
        no_change['v_col'] = v_col
        return no_change

    # Smooth per-column velocity
    v_smooth = v_col.copy()
    if not np.all(valid_cols):
        indices = np.arange(M)
        valid_idx = indices[valid_cols]
        for j in indices[~valid_cols]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - j))]
            v_smooth[j] = v_col[nearest]
    v_smooth = median_filter(v_smooth, size=smooth_kernel)

    # Find longest contiguous run with consistent velocity
    sign = np.sign(v_smooth)
    abs_v = np.abs(v_smooth)

    lo_frac = 1.0 / (1.0 + magnitude_tolerance)
    hi_frac = 1.0 + magnitude_tolerance

    best_start, best_end, best_length = 0, M, 0

    # Iterate over same-sign runs
    run_s = 0
    while run_s < M:
        if not valid_cols[run_s] or sign[run_s] == 0:
            run_s += 1
            continue
        s = sign[run_s]
        run_e = run_s + 1
        while run_e < M and valid_cols[run_e] and sign[run_e] == s:
            run_e += 1

        if run_e - run_s >= min_length_px:
            seg_v = abs_v[run_s:run_e]
            seg_med = np.median(seg_v)
            if seg_med > 1e-10:
                consistent = (seg_v >= lo_frac * seg_med) & (seg_v <= hi_frac * seg_med)
                # Find longest contiguous True run
                k = 0
                while k < len(consistent):
                    if not consistent[k]:
                        k += 1
                        continue
                    k_end = k + 1
                    while k_end < len(consistent) and consistent[k_end]:
                        k_end += 1
                    if k_end - k > best_length:
                        best_length = k_end - k
                        best_start = run_s + k
                        best_end = run_s + k_end
                    k = k_end

        run_s = run_e

    refined = best_length >= min_length_px and best_length < M
    if best_length < min_length_px:
        best_start, best_end = 0, M
        refined = False

    return {
        'start': best_start,
        'end': best_end,
        'v_col': v_col,
        'v_smooth': v_smooth,
        'refined': refined,
    }


def restrict_kymo_to_arc(
    kymo: np.ndarray,
    arc_start: int,
    arc_end: int,
) -> np.ndarray:
    """
    Restrict kymograph to a specific arc region.

    Args:
        kymo: (T, M) kymograph array
        arc_start: Start index
        arc_end: End index (exclusive)

    Returns:
        kymo_restricted: (T, arc_end - arc_start) cropped kymograph
    """
    return kymo[:, arc_start:arc_end]


def apply_column_normalization(
    kymo: np.ndarray,
    method: Literal["subtract_mean", "divide_mean", "zscore"] = "subtract_mean",
) -> np.ndarray:
    """
    Apply column-wise normalization to remove horizontal bands (stationary features).

    Horizontal bands in kymographs are caused by stationary vessel walls, background
    structures, or illumination gradients. Column normalization removes these by
    normalizing each column (arc position) independently.

    Args:
        kymo: (T, M) kymograph array
        method: Normalization method:
            - "subtract_mean": Subtract column mean (preserves scale, removes DC)
            - "divide_mean": Divide by column mean (normalizes brightness)
            - "zscore": Z-score normalization (mean=0, std=1 per column)

    Returns:
        kymo_norm: (T, M) normalized kymograph
    """
    kymo = kymo.astype(np.float32)

    if method == "subtract_mean":
        # Subtract mean of each column (arc position)
        col_mean = np.mean(kymo, axis=0, keepdims=True)
        return kymo - col_mean

    elif method == "divide_mean":
        # Divide by mean of each column (normalize brightness variations)
        col_mean = np.mean(kymo, axis=0, keepdims=True)
        # Avoid division by zero
        col_mean = np.maximum(col_mean, 1e-6)
        return kymo / col_mean

    elif method == "zscore":
        # Z-score: (x - mean) / std per column
        col_mean = np.mean(kymo, axis=0, keepdims=True)
        col_std = np.std(kymo, axis=0, keepdims=True)
        # Avoid division by zero
        col_std = np.maximum(col_std, 1e-6)
        return (kymo - col_mean) / col_std

    else:
        raise ValueError(f"Unknown normalization method: {method}")
