"""
SVD + Radon transform velocity estimation for kymograph analysis.

Alternative to the Gradient Structure Tensor (GST) method.  Estimates the
dominant streak velocity by:

1. SVD denoising — keep top-k singular-value modes to suppress noise while
   preserving streak structure.
2. Shear-and-sum Radon — for each candidate velocity, shear the patch so
   that streaks at that velocity become vertical, then sum along time.
   The candidate with maximum projection variance is the best velocity.

Unlike GST (per-pixel), Radon estimates **one velocity per temporal window**
(spatially averaged), producing a (T,) time series directly.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# SVD denoising
# ---------------------------------------------------------------------------

def svd_denoise(patch: np.ndarray, k: int = 5) -> np.ndarray:
    """Keep the top-k SVD modes of a 2-D patch.

    Parameters
    ----------
    patch : (W, M) array
    k : number of singular values to retain

    Returns
    -------
    denoised : (W, M) array
    """
    k = min(k, *patch.shape)
    U, s, Vh = np.linalg.svd(patch, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vh[:k, :]


# ---------------------------------------------------------------------------
# Shear-and-sum Radon projection
# ---------------------------------------------------------------------------

def _shear_and_project(
    patch: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Compute projection variance for each candidate velocity.

    For velocity *v*, every row *t* is shifted by ``-v * (t - t_center)``
    pixels (sub-pixel via cubic interpolation).  The column-sum of the
    sheared patch is the Radon projection; its variance measures how well
    *v* matches the dominant streak slope.

    Parameters
    ----------
    patch : (W, M) float64 array  (denoised kymograph window)
    velocities : (N_v,) candidate velocities in px/frame

    Returns
    -------
    radon_profile : (N_v,) variance of projection for each velocity
    """
    W, M = patch.shape
    t_center = W // 2
    t_offsets = np.arange(W, dtype=np.float64) - t_center  # (W,)

    # Base coordinate grids (row, col) for map_coordinates
    t_grid = np.arange(W, dtype=np.float64)[:, None].repeat(M, axis=1)  # (W, M)
    x_base = np.arange(M, dtype=np.float64)[None, :].repeat(W, axis=0)  # (W, M)

    radon_profile = np.empty(len(velocities), dtype=np.float64)

    for iv, v in enumerate(velocities):
        # Shifted x-coordinates
        x_shifted = x_base + (v * t_offsets)[:, None]  # shift right by v*dt

        # Interpolate
        sheared = map_coordinates(
            patch,
            [t_grid.ravel(), x_shifted.ravel()],
            order=3, mode='constant', cval=0.0,
        ).reshape(W, M)

        # Project (sum along time) and compute variance
        projection = sheared.sum(axis=0)
        radon_profile[iv] = np.var(projection)

    return radon_profile


# ---------------------------------------------------------------------------
# Confidence metric
# ---------------------------------------------------------------------------

def _radon_confidence(radon_profile: np.ndarray) -> float:
    """Confidence from Radon variance profile.

    Combines peak prominence (SNR above median baseline) with peak
    narrowness (FWHM fraction).  Returns a value in [0, 1].
    """
    peak_idx = np.argmax(radon_profile)
    peak_val = radon_profile[peak_idx]
    baseline = np.median(radon_profile)

    if baseline <= 0:
        return 0.0

    snr = (peak_val - baseline) / baseline

    # FWHM: fraction of velocity bins above half-max
    half_max = 0.5 * (peak_val + baseline)
    fwhm_frac = np.sum(radon_profile >= half_max) / len(radon_profile)
    narrowness = max(1.0 - fwhm_frac, 0.01)

    conf = (1.0 - np.exp(-snr / 2.0)) * narrowness
    return float(np.clip(conf, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Sub-pixel refinement
# ---------------------------------------------------------------------------

def _subpixel_peak(
    profile: np.ndarray,
    velocities: np.ndarray,
    peak_idx: int,
) -> float:
    """Parabolic interpolation around the discrete peak."""
    if peak_idx == 0 or peak_idx >= len(profile) - 1:
        return velocities[peak_idx]

    y0, y1, y2 = profile[peak_idx - 1], profile[peak_idx], profile[peak_idx + 1]
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return velocities[peak_idx]

    offset = (y0 - y2) / denom
    dv = velocities[1] - velocities[0]
    return float(velocities[peak_idx] + offset * dv)


# ---------------------------------------------------------------------------
# Single-patch estimation
# ---------------------------------------------------------------------------

def radon_velocity_single(
    patch: np.ndarray,
    v_min: float = -6.0,
    v_max: float = 6.0,
    dv: float = 0.1,
    svd_rank: int = 5,
    denoise: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate velocity from a single (W, M) kymograph patch.

    Parameters
    ----------
    patch : (W, M) array
    v_min, v_max : velocity search range (px/frame)
    dv : velocity step
    svd_rank : SVD modes to keep (ignored if denoise=False)
    denoise : whether to apply SVD denoising

    Returns
    -------
    velocity : best velocity estimate (px/frame)
    confidence : [0, 1]
    velocities : (N_v,) candidate array
    radon_profile : (N_v,) variance profile
    """
    patch = np.nan_to_num(patch, nan=0.0).astype(np.float64)

    if denoise and svd_rank > 0:
        patch = svd_denoise(patch, k=svd_rank)

    velocities = np.arange(v_min, v_max + dv * 0.5, dv)
    radon_profile = _shear_and_project(patch, velocities)

    peak_idx = int(np.argmax(radon_profile))
    velocity = _subpixel_peak(radon_profile, velocities, peak_idx)
    confidence = _radon_confidence(radon_profile)

    return velocity, confidence, velocities, radon_profile


# ---------------------------------------------------------------------------
# Full kymograph → (T,) time series
# ---------------------------------------------------------------------------

def compute_radon_velocity(
    kymo: np.ndarray,
    window: int = 32,
    stride: int = 1,
    v_min: float = -6.0,
    v_max: float = 6.0,
    dv: float = 0.1,
    svd_rank: int = 5,
    return_profiles: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute Radon velocity time series from a kymograph.

    Slides a temporal window across the kymograph and estimates one
    velocity per window position via SVD + shear-and-sum Radon.

    Parameters
    ----------
    kymo : (T, M) kymograph (axis 0 = time, axis 1 = space)
    window : temporal window size in frames
    stride : step between consecutive windows (1 = every frame)
    v_min, v_max : velocity search range (px/frame)
    dv : velocity resolution (px/frame)
    svd_rank : SVD modes to keep (0 = no denoising)
    return_profiles : if True, also return the full (N_t, N_v) profiles

    Returns
    -------
    vel_t : (T,) velocity time series (px/frame), NaN at boundaries
    conf_t : (T,) confidence time series [0, 1], NaN at boundaries
    profiles : (N_t, N_v) Radon variance profiles (only if return_profiles)
               or None
    """
    T, M = kymo.shape
    half_w = window // 2

    velocities = np.arange(v_min, v_max + dv * 0.5, dv)
    n_v = len(velocities)

    # Compute at strided positions
    centers = np.arange(half_w, T - half_w, stride)
    n_centers = len(centers)

    vel_at_centers = np.empty(n_centers, dtype=np.float64)
    conf_at_centers = np.empty(n_centers, dtype=np.float64)
    profiles_arr = np.empty((n_centers, n_v), dtype=np.float64) if return_profiles else None

    for i, tc in enumerate(centers):
        patch = kymo[tc - half_w: tc + half_w, :].astype(np.float64)
        patch = np.nan_to_num(patch, nan=0.0)

        if svd_rank > 0:
            patch = svd_denoise(patch, k=svd_rank)

        profile = _shear_and_project(patch, velocities)
        peak_idx = int(np.argmax(profile))
        vel_at_centers[i] = _subpixel_peak(profile, velocities, peak_idx)
        conf_at_centers[i] = _radon_confidence(profile)

        if profiles_arr is not None:
            profiles_arr[i] = profile

    # Interpolate to full (T,) arrays
    vel_t = np.full(T, np.nan, dtype=np.float64)
    conf_t = np.full(T, np.nan, dtype=np.float64)

    if stride == 1:
        vel_t[centers] = vel_at_centers
        conf_t[centers] = conf_at_centers
    else:
        # Linear interpolation for strided computation
        vel_t[centers[0]:centers[-1] + 1] = np.interp(
            np.arange(centers[0], centers[-1] + 1), centers, vel_at_centers
        )
        conf_t[centers[0]:centers[-1] + 1] = np.interp(
            np.arange(centers[0], centers[-1] + 1), centers, conf_at_centers
        )

    return vel_t, conf_t, profiles_arr
