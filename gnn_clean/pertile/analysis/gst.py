"""
Gradient Structure Tensor (GST) velocity computation for kymograph analysis.

The GST method computes local orientation and coherence from image gradients,
which can be used to extract velocity information from kymographs where
motion appears as oriented streaks.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from typing import Tuple, List, Optional, Union


def calcGST(
    img: np.ndarray,
    w: Union[int, Tuple[int, int]] = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientation and coherence via structure tensor on image.

    Args:
        img: (T, M) kymograph or (H, W) image array
        w: Window size for box filter smoothing. Either a single int for a
            square kernel, or a (w_time, w_space) tuple for a rectangular
            kernel. For kymographs, axis 0 = time, axis 1 = space.

    Returns:
        coh: Coherence map (0-1), higher values indicate more consistent orientation
        ori: Orientation map in degrees (0-180)
    """
    I = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Parse window size
    if isinstance(w, (tuple, list)):
        ksize = (int(w[0]), int(w[1]))
    else:
        ksize = (int(w), int(w))

    # Compute gradients using Sobel filter
    dx = cv2.Sobel(I, cv2.CV_32F, 1, 0, -1)
    dy = cv2.Sobel(I, cv2.CV_32F, 0, 1, -1)

    # Structure tensor components with box filter smoothing
    J11 = cv2.boxFilter(dx * dx, cv2.CV_32F, ksize)
    J22 = cv2.boxFilter(dy * dy, cv2.CV_32F, ksize)
    J12 = cv2.boxFilter(dx * dy, cv2.CV_32F, ksize)

    # Eigenvalue computation
    tmp = J11 + J22
    diff = J11 - J22
    lam2 = np.sqrt(diff * diff + 4 * J12 * J12)
    lam1 = 0.5 * (tmp + lam2)
    lam2 = 0.5 * (tmp - lam2)

    # Coherence: (lam1 - lam2) / (lam1 + lam2)
    denom = lam1 + lam2
    denom[denom == 0] = 1e-12
    coh = cv2.divide(lam1 - lam2, denom)
    coh[~np.isfinite(coh)] = 0.0
    coh[I == 0] = 0.0

    # Orientation in degrees
    ori = cv2.phase(J22 - J11, 2 * J12, angleInDegrees=True) * 0.5
    ori[~np.isfinite(ori)] = 0.0

    return coh, ori


def gst_stacks(
    img: np.ndarray,
    windows: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run GST over multiple window sizes and return stacks.

    Args:
        img: (T, M) kymograph array
        windows: List of window sizes to try

    Returns:
        coh_stack: (W, T, M) coherence values for each window size
        ori_stack: (W, T, M) orientation values for each window size
    """
    coh_s, ori_s = [], []
    for w in windows:
        c, o = calcGST(img, w=w)
        c = np.where(np.isfinite(c), c, np.nan)
        o = np.where(np.isfinite(o), o, np.nan)
        coh_s.append(c)
        ori_s.append(o)
    return np.stack(coh_s, axis=0), np.stack(ori_s, axis=0)


def compute_gst_velocity(
    kymo: np.ndarray,
    windows: Optional[List[int]] = None,
    coh_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GST velocity map from kymograph.

    Uses multi-scale GST and selects the best window size per pixel
    based on maximum coherence.

    Args:
        kymo: (T, M) kymograph array
        windows: List of window sizes (default: [5, 7, 9, 11, 13, 15, 17])
        coh_thr: Minimum coherence threshold (default: 0.0, no threshold)

    Returns:
        vel_map: (T, M) velocity map in px/frame
        conf_px: (T, M) coherence/confidence map

    Note:
        Velocity is computed as tan(orientation - 90), representing
        the slope of motion streaks in the kymograph. Positive values
        indicate motion in one direction, negative in the other.
    """
    if windows is None:
        windows = [5, 7, 9, 11, 13, 15, 17]

    # Compute GST at multiple scales
    coh_stack, ori_stack = gst_stacks(kymo, windows)

    # Select best window (highest coherence) per pixel
    idx_best = np.nanargmax(coh_stack, axis=0)
    ori_best = np.take_along_axis(ori_stack, idx_best[None, ...], axis=0)[0]
    conf_px = np.take_along_axis(coh_stack, idx_best[None, ...], axis=0)[0]

    # Convert orientation to velocity (px/frame)
    # Velocity = tan(angle - 90 degrees)
    vel_map = np.tan(np.deg2rad(ori_best - 90.0))

    # Optionally mask by coherence threshold
    if coh_thr > 0:
        vel_map = np.where(conf_px >= coh_thr, vel_map, np.nan)

    return vel_map, conf_px


def compute_weighted_median_velocity(
    vel_map: np.ndarray,
    cov_thr: float = 0.25,
    conf_px: Optional[np.ndarray] = None,
    coherence_threshold: float = 0.0,
    column_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute velocity time series v_hat(t) from velocity map.

    When conf_px is provided and coherence_threshold > 0, uses per-pixel
    coherence gating to reject noisy pixels. When column_mask is provided,
    only uses selected columns (allows non-contiguous selection).

    Args:
        vel_map: (T, M) velocity map
        cov_thr: Minimum fraction of valid pixels required (0-1)
        conf_px: (T, M) coherence map from GST (optional)
        coherence_threshold: Min coherence for a pixel to count (0 = no gating)
        column_mask: (M,) boolean mask of columns to include (optional)

    Returns:
        v_hat: (T,) velocity time series (px/frame)
    """
    T, M = vel_map.shape
    v_hat = np.full(T, np.nan, dtype=np.float32)

    for t in range(T):
        vt = vel_map[t, :]
        valid = np.isfinite(vt)

        # Apply column mask (non-contiguous spatial selection)
        if column_mask is not None:
            valid = valid & column_mask

        # Apply per-pixel coherence gating
        if conf_px is not None and coherence_threshold > 0:
            valid = valid & (conf_px[t, :] >= coherence_threshold)

        n_valid = np.sum(valid)
        n_total = M if column_mask is None else np.sum(column_mask)
        n_total = max(n_total, 1)

        if n_valid / n_total >= cov_thr:
            # Always use median - robust to outliers even after gating
            v_hat[t] = np.median(vt[valid])

    return v_hat


def shear_kymograph(
    kymo: np.ndarray,
    v_t: Union[float, np.ndarray],
) -> np.ndarray:
    """Shear a kymograph to remove bulk motion.

    At frame t, shifts row t by the accumulated displacement sum(v_t[0:t])
    using sub-pixel cubic interpolation, so that particles moving at v_t
    appear stationary (near-vertical streaks).

    Spatial wrapping is used (mode='wrap') so that large displacements don't
    push content out of bounds — the GST only measures local streak orientation,
    not absolute streak position, so wrapping is harmless for velocity estimation.

    Parameters
    ----------
    kymo : (T, M) float array
    v_t : float or (T,) array
        Velocity to shear out.  Scalar → constant shear; array → time-varying.

    Returns
    -------
    kymo_sheared : (T, M) float64 array
    """
    T, M = kymo.shape
    t_idx = np.arange(T, dtype=np.float64)
    x_idx = np.arange(M, dtype=np.float64)

    # Accumulated displacement at each frame
    if np.isscalar(v_t):
        x_shift = float(v_t) * t_idx                              # (T,)
    else:
        v_arr = np.asarray(v_t, dtype=np.float64)
        x_shift = np.zeros(T, dtype=np.float64)
        x_shift[1:] = np.cumsum(v_arr[:-1])                       # (T,)

    # Sample: kymo_sheared[t, x] = kymo[t, x + x_shift[t]]
    t_grid = np.broadcast_to(t_idx[:, None], (T, M)).ravel()
    x_grid = np.broadcast_to(x_idx[None, :], (T, M)).ravel()
    x_source = x_grid + np.repeat(x_shift, M)

    return map_coordinates(
        kymo.astype(np.float64),
        [t_grid, x_source],
        order=3, mode='wrap',
    ).reshape(T, M)


def compute_gst_velocity_shear(
    kymo: np.ndarray,
    windows: Optional[List[int]] = None,
    coh_thr: float = 0.0,
    coherence_threshold: float = 0.3,
    n_iter: int = 4,
    damping: float = 0.5,
    v_shear_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GST velocity estimation with iterative shear correction.

    Reduces the tan() nonlinearity bias at high velocities by shearing the
    kymograph so streaks become near-vertical before each GST pass, where the
    angular sensitivity dv/dθ = sec²(θ) is smallest.

    When the initial GST estimate is biased (it typically overestimates at high
    velocities), naive iteration oscillates: the correction over-shoots in one
    direction, then the next step over-shoots back.  Damping (step size < 1)
    converts this oscillation into geometric convergence.

    Algorithm (per iteration):
        1. Shear the original kymograph by the current accumulated estimate
        2. Run GST on the sheared kymograph → residual vel_map
        3. Accumulate: v_total += damping × v_step
        4. Repeat until convergence (or n_iter exhausted)

    At low velocities the tan() bias is negligible (dv/dθ = 1 + v² ≈ 1 for
    small v), but each shear pass applies cubic interpolation which accumulates
    as noise.  The `v_shear_min` threshold enables a hybrid mode: frames where
    the final shear estimate is below the threshold fall back to the standard
    GST result from the first (unsheared) pass, avoiding unnecessary
    interpolation degradation.

    Parameters
    ----------
    kymo : (T, M) float32 array
    windows : list of int, GST window sizes
    coh_thr : float, per-pixel coherence mask threshold for vel_map
    coherence_threshold : float, coherence threshold for weighted median
    n_iter : int
        Maximum number of shear-GST passes (default 4).
        Convergence is checked automatically; stops early when the step
        is smaller than 0.01 px/frame RMS.
    damping : float
        Step size in (0, 1].  damping=1.0 is undamped (fast but may
        oscillate if initial estimate is far off).  damping=0.5 is safe
        for all practical velocity ranges.
    v_shear_min : float
        Hybrid threshold (px/frame).  Frames where |v_shear| < v_shear_min
        use the standard (unsheared) GST result instead.  0.0 disables the
        hybrid (pure shear correction, default behaviour).  A value of ~2.0
        is a good starting point: standard GST is already near-unbiased
        below that speed, while shear correction helps above it.

    Returns
    -------
    v_hat : (T,) corrected velocity time series (px/frame)
    conf_px : (T, M) coherence map from final GST pass
    vel_map : (T, M) residual velocity map from final GST pass
    """
    T = kymo.shape[0]
    v_total = np.zeros(T, dtype=np.float64)
    vel_map = conf_px = None
    v_step = np.zeros(T, dtype=np.float64)
    v_std: Optional[np.ndarray] = None  # standard GST result (first pass)

    for it in range(n_iter):
        kymo_work = shear_kymograph(kymo, v_total).astype(np.float32)
        vel_map, conf_px = compute_gst_velocity(
            kymo_work, windows=windows, coh_thr=coh_thr
        )
        v_step = compute_weighted_median_velocity(
            vel_map, conf_px=conf_px, coherence_threshold=coherence_threshold
        )
        v_step_safe = np.where(np.isfinite(v_step), v_step, 0.0)

        # Save unsheared GST result from the first pass for hybrid fallback
        if it == 0 and v_shear_min > 0:
            v_std = v_step.copy()

        v_total += damping * v_step_safe

        # Early stopping: converged when step is tiny
        if np.sqrt(np.mean(v_step_safe ** 2)) < 0.01:
            break

    # Build output, marking frames with no valid estimate as NaN
    v_hat = np.where(np.isfinite(v_step), v_total, np.nan)

    # Hybrid: substitute standard GST where shear estimate is below threshold
    if v_shear_min > 0 and v_std is not None:
        use_std = np.abs(v_hat) < v_shear_min
        v_hat = np.where(use_std, v_std, v_hat)

    return v_hat.astype(np.float32), conf_px, vel_map


def compute_column_quality_mask(
    conf_px: np.ndarray,
    min_coherence: float = 0.3,
    min_fraction: float = 0.3,
) -> np.ndarray:
    """
    Compute a per-column quality mask from the GST coherence map.

    A column is "good" if a sufficient fraction of its pixels exceed
    the coherence threshold. This allows non-contiguous spatial selection.

    Args:
        conf_px: (T, M) coherence map from GST
        min_coherence: Minimum coherence for a pixel to count as "good"
        min_fraction: Minimum fraction of frames that must be good

    Returns:
        mask: (M,) boolean array, True for good columns
    """
    # Fraction of frames with coherence above threshold per column
    good_frac = np.mean(conf_px >= min_coherence, axis=0)
    return good_frac >= min_fraction
