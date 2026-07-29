"""
Radial velocity profile analysis for blood flow measurement.

This module provides the main entry point for analyzing blood flow
in vessels using radial velocity profiles sampled at multiple offsets
from the vessel centerline.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple, Dict, Any

from .gst import compute_gst_velocity, compute_weighted_median_velocity, compute_column_quality_mask
from .kymo import (
    sample_kymo_at_offset, apply_temporal_detrend, apply_column_normalization,
    compute_streak_quality, find_good_arc_region, restrict_kymo_to_arc
)
from .harmonic import estimate_f0_in_band, fit_harmonics
from .config import (
    GST_WINDOWS, COV_THRESHOLD,
    DETREND_WINDOWS, DEFAULT_DETREND_WINDOW, N_HARMONICS,
    FMIN_HZ, FMAX_HZ, BAND_RADIUS_PX, OFFSET_STEP_PX, MARGIN_PX, PX_SIZE_UM, FRAME_DT_S,
    COHERENCE_GATE_THRESHOLD, COLUMN_COHERENCE_MIN, BAND_COHERENCE_MIN,
    ENVELOPE_INNER_FRACTION, ENVELOPE_SIGMA_TOLERANCE,
    DOC_Z_C_UM, DOC_MODEL,
)


def compute_radial_velocity_profiles(
    coords: np.ndarray,
    stack: np.ndarray,
    vessel_radius_px: float,
    frame_dt: float = FRAME_DT_S,
    *,
    offsets: Optional[np.ndarray] = None,
    detrend_windows: Optional[List[int]] = None,
    consensus_f0: Optional[float] = None,
    fmin_hz: float = FMIN_HZ,
    fmax_hz: float = FMAX_HZ,
    profile_band_radius: int = BAND_RADIUS_PX,
    gst_windows: Optional[List[int]] = None,
    centerline_only: bool = False,
    extend_radius_px: float = 0.0,
    use_arc_restriction: bool = True,
    arc_quality_threshold: float = 0.25,
    use_column_norm: bool = True,
    column_norm_method: str = "subtract_mean",
    min_arc_length_px: int = 25,
    coherence_gate: float = COHERENCE_GATE_THRESHOLD,
    column_coherence_min: float = COLUMN_COHERENCE_MIN,
    cached_centerline_kymo: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Compute velocity profiles at multiple radial offsets from vessel centerline.

    This is the core radial profile analysis for blood flow measurement.

    Args:
        coords: (N, 2) array of [x, y] centerline coordinates in pixels
        stack: (T, H, W) video frames (grayscale, float32, [0-1] normalized)
        vessel_radius_px: Vessel radius in pixels
        frame_dt: Time between frames in seconds
        offsets: Radial offsets to sample (default: auto-generate from radius)
        detrend_windows: Temporal detrending window sizes to try
        consensus_f0: If provided, use this f0 instead of estimating per-offset
        fmin_hz: Lower frequency bound for f0 estimation
        fmax_hz: Upper frequency bound for f0 estimation
        profile_band_radius: Width of sampling band perpendicular to centerline
        gst_windows: Window sizes for GST computation
        centerline_only: If True and consensus_f0 is None, only compute centerline
            (offset=0) for fast f0 estimation. Used in PASS 1 of consensus workflow.
        extend_radius_px: Pixels to extend sampling beyond segmented radius (default 0).
            Use >0 (e.g., 5) to sample beyond the original radius and auto-detect
            the true vessel boundary from noise characteristics.
        use_arc_restriction: If True, restrict velocity analysis to high-quality
            arc regions where streaks are clear (default True).
        arc_quality_threshold: Minimum streak quality score (0-1) for arc restriction.
        use_column_norm: If True, apply column normalization to remove horizontal
            bands (stationary features like vessel walls). Default True.
        column_norm_method: Column normalization method ("subtract_mean", "divide_mean",
            or "zscore"). Default "subtract_mean".
        min_arc_length_px: Minimum arc length in pixels for arc restriction (default 25).
        cached_centerline_kymo: If provided, reuse this raw kymograph for offset=0
            instead of re-sampling from the video stack. Used to avoid redundant
            computation when PASS 1 already sampled the centerline.
        verbose: Print progress messages

    Returns:
        profile_data: List of dicts with analysis results per offset
        f0_consensus: Consensus fundamental frequency (Hz)

    Each profile dict contains:
        - offset: radial offset (px)
        - kymo: (T, M) raw kymograph array (full arc)
        - kymo_restricted: (T, M') restricted kymograph (good arc only)
        - arc_start, arc_end: Arc restriction bounds (indices into kymo)
        - arc_quality: Per-column quality scores
        - v_hat: (T,) velocity time series (px/frame)
        - v_mean, v_std: velocity statistics
        - best_detrend_window: optimal detrending window
        - best_hr_snr: harmonic regression SNR (dB)
        - consensus_f0_hz: frequency used
        - mean_coherence: average GST coherence
    """
    if gst_windows is None:
        gst_windows = GST_WINDOWS

    if detrend_windows is None:
        detrend_windows = DETREND_WINDOWS

    # CRITICAL: Convert coordinates from Cartesian y (y=0 at bottom) to image y (y=0 at top)
    # Graph stores nodes with y_cart where 0 is bottom, but image stack uses y_image where 0 is top
    image_height = stack.shape[1]
    coords_image = coords.copy()
    coords_image[:, 1] = image_height - coords[:, 1]


    # Use image coordinates for all sampling operations
    coords = coords_image

    # Generate radial offsets if not provided
    # Sample from -0.85*R to +0.85*R, plus extend_radius_px beyond
    # Use adaptive step size based on vessel radius to balance accuracy vs speed
    if offsets is None:
        inner_max = int(np.round(0.85 * vessel_radius_px))
        outer_max = inner_max + int(np.round(extend_radius_px))

        # Adaptive step size: target ~5-7 bands across the diameter
        # For small vessels (R<8): use step=2 (minimum, ensures overlap)
        # For medium vessels (8<=R<15): use step=3
        # For large vessels (R>=15): use step=4 (maximum)
        MIN_BANDS_TARGET = 5  # Minimum bands needed for reliable profile fit
        diameter = 2 * outer_max

        # Compute step to achieve target bands, with bounds
        if diameter > 0:
            adaptive_step = max(2, min(4, diameter // MIN_BANDS_TARGET))
        else:
            adaptive_step = OFFSET_STEP_PX

        # Generate offsets symmetrically around 0 with adaptive step spacing
        # This ensures: ..., -2*step, -step, 0, step, 2*step, ...
        max_steps = outer_max // adaptive_step
        offsets = np.arange(-max_steps, max_steps + 1) * adaptive_step
        offsets = offsets.astype(int)

    # Override to centerline-only if requested for fast f0 estimation
    if centerline_only and consensus_f0 is None:
        offsets = np.array([0], dtype=int)
        if verbose:
            print("Using centerline-only mode for fast f0 estimation (PASS 1)")

    # PASS 1 uses conservative 3-second window (preserves ≥0.3 Hz, covers full HR range)
    default_detrend = int(3.0 / frame_dt)

    if verbose:
        step_used = offsets[1] - offsets[0] if len(offsets) > 1 else 0
        print(f"Analyzing {len(offsets)} radial offsets: {offsets[0]} to {offsets[-1]} px (step={step_used})")

    # PASS 1: Estimate f0 at each radial offset (if not provided)
    raw_kymographs = []

    if consensus_f0 is None:
        if verbose:
            print("Pass 1: Estimating consensus f0...")

        f0_estimates = []
        _centerline_coherence = None  # Track for early return if f0 fails

        for offset in offsets:
            kymo_raw = sample_kymo_at_offset(
                coords, stack, offset, band_radius=profile_band_radius
            )
            if kymo_raw is None:
                raw_kymographs.append((offset, None))
                continue

            raw_kymographs.append((offset, kymo_raw))

            # Apply column normalization (removes horizontal bands)
            kymo_proc = kymo_raw
            if use_column_norm:
                kymo_proc = apply_column_normalization(kymo_proc, method=column_norm_method)

            # Apply default detrending
            kymo_detrend = apply_temporal_detrend(kymo_proc, window_size=default_detrend,
                                                  frame_rate=1.0/frame_dt)

            # Compute GST velocity
            vel_map, conf_px = compute_gst_velocity(kymo_detrend, windows=gst_windows)

            # Save centerline coherence for early return if f0 estimation fails
            if offset == 0 and conf_px is not None:
                _centerline_coherence = float(np.nanmean(conf_px))

            # Compute v_hat(t) with coherence gating
            col_mask = None
            if column_coherence_min > 0:
                col_mask = compute_column_quality_mask(
                    conf_px, min_coherence=coherence_gate, min_fraction=column_coherence_min)
            v_hat = compute_weighted_median_velocity(
                vel_map, cov_thr=COV_THRESHOLD,
                conf_px=conf_px, coherence_threshold=coherence_gate,
                column_mask=col_mask)

            n_valid = np.isfinite(v_hat).sum()
            if n_valid < 32:
                continue

            # Estimate f0
            try:
                f0_hz = estimate_f0_in_band(v_hat, frame_dt, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
                if np.isfinite(f0_hz):
                    f0_estimates.append(f0_hz)
                    if verbose:
                        print(f"  Offset {offset:+3d} px: f0={f0_hz:.3f} Hz")
            except Exception as e:
                if verbose:
                    print(f"  Offset {offset:+3d} px: f0 estimation failed - {e}")

        if len(f0_estimates) == 0:
            # f0 estimation failed, but if centerline_only we can still return
            # coherence data so the batch can decide whether to skip this vessel
            if centerline_only and _centerline_coherence is not None:
                return {
                    'f0_hz': np.nan,
                    'success': True,
                    'profile_data': [{'mean_coherence': _centerline_coherence, 'v_hat': None, 'offset': 0}],
                }
            raise ValueError("Could not estimate f0 at any radial offset!")

        consensus_f0 = float(np.median(f0_estimates))
        if verbose:
            print(f"Consensus f0: {consensus_f0:.3f} Hz (n={len(f0_estimates)})")
    else:
        if verbose:
            print(f"Using provided consensus f0: {consensus_f0:.3f} Hz")
        # Sample raw kymographs (reuse cached centerline if available)
        for offset in offsets:
            if offset == 0 and cached_centerline_kymo is not None:
                kymo_raw = cached_centerline_kymo
                if verbose:
                    print("  Using cached centerline kymograph from PASS 1")
            else:
                kymo_raw = sample_kymo_at_offset(
                    coords, stack, offset, band_radius=profile_band_radius
                )
            raw_kymographs.append((offset, kymo_raw))

    # Determine arc restriction from centerline kymograph (offset=0)
    arc_restrictions = {}  # offset -> {arc_start, arc_end, quality}

    if use_arc_restriction:
        # Find centerline kymograph
        centerline_kymo = None
        for offset, kymo in raw_kymographs:
            if offset == 0 and kymo is not None:
                centerline_kymo = kymo
                break

        if centerline_kymo is not None:
            # Compute arc restriction from centerline
            centerline_arc = find_good_arc_region(
                centerline_kymo,
                min_quality=arc_quality_threshold,
                min_length_frac=0.15,
                min_arc_length_px=min_arc_length_px,
                edge_margin_frac=0.08,
            )

            if verbose:
                M = centerline_kymo.shape[1]
                print(f"Arc restriction (from centerline): [{centerline_arc['arc_start']}, {centerline_arc['arc_end']}) "
                      f"of {M} ({centerline_arc['arc_length']} px, quality={centerline_arc['mean_quality']:.2f})")

            # Hard minimum: GST needs enough pixels for reliable velocity estimation
            # (window sizes range 5-17, need space for at least small windows to average)
            HARD_MIN_ARC_LENGTH = 15
            if centerline_arc['arc_length'] < HARD_MIN_ARC_LENGTH:
                if verbose:
                    print(f"  WARNING: Arc too short ({centerline_arc['arc_length']} px < {HARD_MIN_ARC_LENGTH} px minimum)")
                    print(f"  Skipping velocity analysis - insufficient arc length for reliable GST")
                # Return empty results - vessel too short to analyze
                return [], np.nan

            # Apply same arc restriction to all offsets (use centerline as reference)
            # But also compute per-offset quality for potential per-offset adjustment
            for offset, kymo in raw_kymographs:
                if kymo is None:
                    continue

                # Compute this offset's quality
                quality = compute_streak_quality(kymo)

                # Use centerline bounds as default
                arc_start = centerline_arc['arc_start']
                arc_end = centerline_arc['arc_end']

                # Optionally: Per-offset refinement (find good region within centerline bounds)
                # For now, use the same bounds for all offsets for consistency
                arc_restrictions[offset] = {
                    'arc_start': arc_start,
                    'arc_end': arc_end,
                    'quality': quality,
                    'mean_quality': float(np.mean(quality[arc_start:arc_end])),
                }
        else:
            if verbose:
                print("Warning: No centerline kymograph found, arc restriction disabled")
    else:
        # No arc restriction - use full kymograph
        for offset, kymo in raw_kymographs:
            if kymo is not None:
                M = kymo.shape[1]
                arc_restrictions[offset] = {
                    'arc_start': 0,
                    'arc_end': M,
                    'quality': np.ones(M),
                    'mean_quality': 1.0,
                }

    # PASS 2: Extract velocity at each offset with single detrending window
    # Detrending window = 4/f0 (preserves cardiac signal, removes slow drift).
    # If f0 unknown (PASS 1), use 3 seconds (conservative, preserves ≥0.3 Hz).
    if consensus_f0 is not None and np.isfinite(consensus_f0) and consensus_f0 > 0:
        detrend_frames = int(4.0 / consensus_f0 / frame_dt)
    else:
        detrend_frames = int(3.0 / frame_dt)
    # Clamp to reasonable range
    detrend_frames = max(50, min(detrend_frames, 2000))

    if verbose:
        print(f"Pass 2: Extracting velocity (detrend_window={detrend_frames} frames)...")

    profile_data = []
    n_coh_rejected = 0

    for offset, kymo_raw in raw_kymographs:
        if kymo_raw is None:
            continue

        # Get arc restriction for this offset
        arc_info = arc_restrictions.get(offset, None)
        if arc_info is None:
            M = kymo_raw.shape[1]
            arc_start, arc_end = 0, M
            arc_quality = np.ones(M)
        else:
            arc_start = arc_info['arc_start']
            arc_end = arc_info['arc_end']
            arc_quality = arc_info['quality']

        # Restrict kymograph to good arc region
        kymo_restricted = restrict_kymo_to_arc(kymo_raw, arc_start, arc_end)

        # Column normalization
        if use_column_norm:
            kymo_restricted = apply_column_normalization(kymo_restricted, method=column_norm_method)

        # Detrend with single window
        kymo_detrend = apply_temporal_detrend(kymo_restricted, window_size=detrend_frames,
                                              frame_rate=1.0/frame_dt)

        # Compute GST velocity + coherence
        vel_map, conf_px = compute_gst_velocity(kymo_detrend, windows=gst_windows)

        # --- Early coherence rejection (before v(t) extraction) ---
        # Compute mean coherence from the coherence map
        if conf_px is not None:
            mean_coherence = float(np.nanmean(conf_px))
        else:
            mean_coherence = 0.0

        # Reject low-coherence bands early (always keep centerline)
        if mean_coherence < BAND_COHERENCE_MIN and offset != 0:
            n_coh_rejected += 1
            if verbose:
                print(f"  Offset {offset:+3d} px: rejected early (mean_coherence={mean_coherence:.2f} < {BAND_COHERENCE_MIN:.2f})")
            continue

        # --- Only extract v(t) for bands that pass coherence gate ---
        col_mask = None
        if column_coherence_min > 0:
            col_mask = compute_column_quality_mask(
                conf_px, min_coherence=coherence_gate, min_fraction=column_coherence_min)

        v_hat_offset = compute_weighted_median_velocity(
            vel_map, cov_thr=COV_THRESHOLD,
            conf_px=conf_px, coherence_threshold=coherence_gate,
            column_mask=col_mask,
        )

        n_valid = np.isfinite(v_hat_offset).sum()
        if n_valid < 32:
            continue

        with np.errstate(invalid='ignore'):
            v_mean = np.nanmean(v_hat_offset)
            v_std = np.nanstd(v_hat_offset)

        # Recompute mean coherence over included columns only
        if conf_px is not None:
            if col_mask is not None and np.any(col_mask):
                mean_coherence = float(np.nanmean(conf_px[:, col_mask]))
                n_good_cols = int(np.sum(col_mask))
                n_total_cols = len(col_mask)
            else:
                mean_coherence = float(np.nanmean(conf_px))
                n_good_cols = conf_px.shape[1]
                n_total_cols = n_good_cols
        else:
            n_good_cols = kymo_restricted.shape[1]
            n_total_cols = n_good_cols

        # Harmonic SNR for diagnostics (not for window selection anymore)
        best_hr_snr = -np.inf
        if consensus_f0 is not None:
            try:
                hr_result = fit_harmonics(
                    v_hat_offset, frame_dt, consensus_f0,
                    K=N_HARMONICS, loss="huber", include_dc=True,
                )
                best_hr_snr = hr_result.get('hr_snr_db', -np.inf)
            except Exception:
                pass

        # Full kymograph for visualization (same processing)
        kymo_raw_proc = kymo_raw
        if use_column_norm:
            kymo_raw_proc = apply_column_normalization(kymo_raw_proc, method=column_norm_method)
        kymo_raw_detrend = apply_temporal_detrend(kymo_raw_proc, window_size=detrend_frames,
                                                    frame_rate=1.0/frame_dt)

        profile_data.append({
            'offset': offset,
            'kymo': kymo_raw,
            'kymo_detrend': kymo_raw_detrend,
            'kymo_restricted': kymo_restricted,
            'vel_map': vel_map,
            'conf_px': conf_px,
            'column_mask': col_mask,
            'arc_start': arc_start,
            'arc_end': arc_end,
            'arc_quality': arc_quality,
            'v_hat': v_hat_offset,
            'v_mean': v_mean,
            'v_std': v_std,
            'best_detrend_window': detrend_frames,
            'best_hr_snr': best_hr_snr,
            'consensus_f0_hz': consensus_f0,
            'mean_coherence': mean_coherence,
            'n_good_cols': n_good_cols,
            'n_total_cols': n_total_cols,
        })

    if len(profile_data) == 0:
        raise ValueError("Failed to compute velocity profiles at any offset")

    if verbose:
        n_total = sum(1 for _, k in raw_kymographs if k is not None)
        print(f"Successfully computed {len(profile_data)}/{n_total} profiles"
              f" ({n_coh_rejected} rejected by coherence < {BAND_COHERENCE_MIN:.1f})")
        centerline = [p for p in profile_data if p['offset'] == 0]
        if centerline:
            p = centerline[0]
            print(f"  Centerline: {p['n_good_cols']}/{p['n_total_cols']} columns pass "
                  f"(mean coh={p['mean_coherence']:.2f})")

    return profile_data, consensus_f0


def _estimate_effective_radius(
    profile_data: List[Dict[str, Any]],
    original_radius_px: float,
    frame_dt: float = FRAME_DT_S,
    noise_ratio_max: float = 3.0,      # Upper bound: sigma > 3x inner = too noisy
    noise_ratio_min: float = 0.15,     # Lower bound: sigma < 0.15x inner = static tissue
    velocity_threshold: float = 0.0,   # Disabled - rely on noise bounds instead
) -> Dict[str, float]:
    """
    Estimate effective vessel radius from noise bounds.

    Key insight: blood flow has characteristic noise level from blood cell passage.
    - Inside vessel: sigma is similar to inner bands (within bounds)
    - Static tissue outside: sigma is MUCH lower (just camera noise)
    - Noisy region outside: sigma is MUCH higher

    A band is considered "inside vessel" if:
      noise_ratio_min * inner_sigma <= sigma <= noise_ratio_max * inner_sigma

    This two-sided criterion catches both:
    - Static tissue (too quiet = below minimum)
    - Noisy regions (too loud = above maximum)

    Args:
        profile_data: List of profile dicts from compute_radial_velocity_profiles
        original_radius_px: Original segmented radius in pixels
        frame_dt: Time between frames in seconds
        noise_ratio_max: Upper bound on sigma/inner_sigma (too noisy = outside)
        noise_ratio_min: Lower bound on sigma/inner_sigma (too quiet = static tissue)
        velocity_threshold: Fraction of max velocity below which band is "outside"

    Returns:
        Dict with:
            effective_radius_px: Estimated true radius based on signal transition
            r_positive: Max good offset on positive side (from segmented centerline)
            r_negative: Max good offset on negative side (positive value)
            r_offset_inferred: Inferred centerline offset = (r_positive - r_negative) / 2
            R_inferred: Inferred radius = (r_positive + r_negative) / 2
    """
    default_result = {
        'effective_radius_px': original_radius_px,
        'r_positive': original_radius_px,
        'r_negative': original_radius_px,
        'r_offset_inferred': 0.0,
        'R_inferred': original_radius_px,
    }
    # Get consensus f0 from profile data
    f0_hz = None
    for p in profile_data:
        f0_hz = p.get('consensus_f0_hz')
        if f0_hz is not None and np.isfinite(f0_hz):
            break

    if f0_hz is None or not np.isfinite(f0_hz):
        return default_result

    # Compute sigma_noise and mean velocity for each band
    # band_data: List of (abs_offset, sigma, v_mean, signed_offset)
    band_data = []

    for p in profile_data:
        v_hat = p.get('v_hat')
        if v_hat is None:
            continue

        offset = p['offset']
        v_mean = float(np.nanmean(v_hat))

        # Compute sigma_noise from harmonic residuals
        hr_result = fit_harmonics(
            v_hat, frame_dt, f0_hz, K=N_HARMONICS,
            loss="huber", include_dc=True
        )
        resid = hr_result.get('resid', None)
        if resid is not None:
            sigma = float(np.nanstd(resid))
        else:
            sigma = float(np.nanstd(v_hat))

        if np.isfinite(sigma) and sigma > 0 and np.isfinite(v_mean):
            band_data.append((abs(offset), sigma, v_mean, offset))

    if len(band_data) < 5:
        return default_result

    # Sort by absolute offset
    band_data.sort(key=lambda x: x[0])

    # Compute reference values from INNER bands (within 50% of original radius)
    inner_limit = max(original_radius_px * 0.5, 1.0)
    inner_sigmas = [sig for (abs_off, sig, _, _) in band_data if abs_off <= inner_limit]
    inner_velocities = [abs(v) for (abs_off, _, v, _) in band_data if abs_off <= inner_limit]

    if len(inner_sigmas) < 2:
        # Fallback: use the 3 innermost bands
        inner_sigmas = [sig for (_, sig, _, _) in band_data[:min(3, len(band_data))]]
        inner_velocities = [abs(v) for (_, _, v, _) in band_data[:min(3, len(band_data))]]

    if not inner_sigmas or not inner_velocities:
        return default_result

    # Reference sigma is median of inner bands
    sigma_inner = float(np.median(inner_sigmas))
    if sigma_inner <= 0:
        return default_result

    # Reference velocity is max of inner bands (centerline velocity)
    v_max_inner = float(np.max(inner_velocities))
    if v_max_inner <= 0:
        return default_result

    # Thresholds - two-sided noise bounds
    sigma_max = noise_ratio_max * sigma_inner   # Too noisy = outside vessel
    sigma_min = noise_ratio_min * sigma_inner   # Too quiet = static tissue
    v_threshold = velocity_threshold * v_max_inner

    # Group data by SIGNED offset for per-side analysis
    # We need to find contiguous good regions from center on each side
    positive_data = []  # [(offset, sigma, v_mean), ...] for offset > 0
    negative_data = []  # [(offset, sigma, v_mean), ...] for offset < 0
    center_data = []    # offset == 0

    for _, sig, v_mean, signed_off in band_data:
        if signed_off > 0:
            positive_data.append((signed_off, sig, v_mean))
        elif signed_off < 0:
            negative_data.append((abs(signed_off), sig, v_mean))  # Store as positive for easier comparison
        else:
            center_data.append((0, sig, v_mean))

    def find_contiguous_radius(side_data):
        """Find max contiguous radius from center where criteria pass."""
        if not side_data:
            return 0.0

        # Sort by offset from center
        side_data.sort(key=lambda x: x[0])

        # Find contiguous good region starting from smallest offset
        max_good_offset = 0.0
        for offset, sigma, v_mean in side_data:
            # Two-sided noise criterion: not too quiet (static) or too noisy
            noise_ok = sigma_min <= sigma <= sigma_max
            velocity_ok = abs(v_mean) >= v_threshold or velocity_threshold == 0.0

            if noise_ok and velocity_ok:
                max_good_offset = offset
            else:
                # First bad band - stop here (don't include anything beyond)
                break

        return max_good_offset

    # Find effective radius on each side
    r_positive = find_contiguous_radius(positive_data)
    r_negative = find_contiguous_radius(negative_data)

    # Check center band(s)
    center_ok = True
    if center_data:
        for _, sigma, v_mean in center_data:
            noise_ok = sigma_min <= sigma <= sigma_max
            vel_ok = abs(v_mean) >= v_threshold or velocity_threshold == 0.0
            if not (noise_ok and vel_ok):
                center_ok = False
                break

    if not center_ok:
        # Center is bad - something is very wrong, use original radius
        return default_result

    # Effective radius is the MINIMUM of the two sides
    # This ensures we don't include artifacts on either side
    if r_positive == 0 and r_negative == 0:
        # Only center is good - use minimum band spacing as radius
        # But don't go more than 1px below segmented radius
        effective_radius = max(1.0, original_radius_px - 1.0)
        return {
            'effective_radius_px': effective_radius,
            'r_positive': 0.0,
            'r_negative': 0.0,
            'r_offset_inferred': 0.0,
            'R_inferred': effective_radius,
        }

    # Use MAXIMUM of the two sides to include all potentially good data
    # The NLLS fit will determine the true radius and handle asymmetry via r_offset
    # This allows thick vessels to use data from the side with better signal
    if r_positive == 0:
        max_good_offset = r_negative
    elif r_negative == 0:
        max_good_offset = r_positive
    else:
        max_good_offset = max(r_positive, r_negative)

    # Add 1 px margin to include the last good band
    effective_radius = max_good_offset + 1.0

    # Enforce minimum: don't shrink more than 1px below segmented radius
    # (segmentation provides a reasonable lower bound on vessel size)
    min_radius = original_radius_px - 1.0
    effective_radius = max(effective_radius, min_radius)

    # Compute inferred centerline and radius from asymmetric bounds
    # r_positive is distance to positive boundary, r_negative is distance to negative boundary
    # Inferred centerline offset: midpoint of bounds = (r_positive - r_negative) / 2
    # Inferred radius: half the span = (r_positive + r_negative) / 2
    r_offset_inferred = (r_positive - r_negative) / 2.0
    R_inferred = (r_positive + r_negative) / 2.0

    print(f"    -> r_positive={r_positive:.1f}, r_negative={r_negative:.1f}")
    print(f"    -> r_offset_inferred={r_offset_inferred:.1f} px, R_inferred={R_inferred:.1f} px")
    print(f"    -> effective_radius: {effective_radius:.1f} px (for NLLS fit)")

    return {
        'effective_radius_px': effective_radius,
        'r_positive': r_positive,
        'r_negative': r_negative,
        'r_offset_inferred': r_offset_inferred,
        'R_inferred': R_inferred,
    }


# =============================================================================
# Cycle-based uncertainty quantification
# =============================================================================

def _segment_into_cycles(
    v_hat: np.ndarray,
    frame_dt: float,
    f0_hz: float,
    min_cycle_fraction: float = 0.7,
) -> List[Tuple[int, int]]:
    """
    Segment velocity time series into heartbeat cycles using phase crossings.

    Uses the fitted harmonic fundamental to define phase, then finds zero-crossings
    of the phase (or equivalently, positive-going crossings of the fundamental).
    This handles heart rate variability better than fixed frame counts.

    Args:
        v_hat: (T,) velocity time series
        frame_dt: Time between frames in seconds
        f0_hz: Fundamental frequency (heart rate) in Hz
        min_cycle_fraction: Minimum fraction of expected cycle length to accept

    Returns:
        List of (start_idx, end_idx) tuples defining each complete cycle
    """
    T = len(v_hat)
    t = np.arange(T) * frame_dt

    # Expected samples per cycle
    expected_samples_per_cycle = 1.0 / (f0_hz * frame_dt)
    min_samples = int(min_cycle_fraction * expected_samples_per_cycle)
    max_samples = int(1.5 * expected_samples_per_cycle)  # Allow some variability

    # Fit harmonic to get phase
    hr_result = fit_harmonics(v_hat, frame_dt, f0_hz, K=1, loss="huber", include_dc=True)

    if not hr_result['harmonics']:
        # Fallback: use fixed frame counts
        samples_per_cycle = int(round(expected_samples_per_cycle))
        cycles = []
        for start in range(0, T - min_samples, samples_per_cycle):
            end = min(start + samples_per_cycle, T)
            if end - start >= min_samples:
                cycles.append((start, end))
        return cycles

    # Extract fundamental phase: v(t) = A*cos(wt) + B*sin(wt) = C*cos(wt - phi)
    # Phase advances as: theta(t) = 2*pi*f0*t - phi
    # Find where theta crosses 0 (mod 2*pi), i.e., positive-going zero crossings
    A1 = hr_result['harmonics'][0]['A']
    B1 = hr_result['harmonics'][0]['B']
    phi = np.arctan2(-B1, A1)  # Phase offset

    # Compute instantaneous phase
    theta = 2 * np.pi * f0_hz * t - phi
    # Wrap to [0, 2*pi)
    theta_wrapped = np.mod(theta, 2 * np.pi)

    # Find cycle boundaries: where phase wraps from ~2*pi to ~0
    # This is where theta_wrapped[i] > theta_wrapped[i+1] (by a lot)
    phase_diff = np.diff(theta_wrapped)
    wrap_indices = np.where(phase_diff < -np.pi)[0] + 1  # +1 because diff shifts by 1

    # Build cycles from wrap points
    cycles = []
    boundaries = [0] + list(wrap_indices) + [T]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        length = end - start

        # Accept if length is reasonable
        if min_samples <= length <= max_samples:
            cycles.append((start, end))

    return cycles


def _compute_cycle_means(
    v_hat: np.ndarray,
    cycles: List[Tuple[int, int]],
    frame_dt: float,
    f0_hz: float,
    use_harmonic_dc: bool = True,
) -> np.ndarray:
    """
    Compute mean velocity for each heartbeat cycle.

    Args:
        v_hat: (T,) velocity time series
        cycles: List of (start, end) tuples from _segment_into_cycles
        frame_dt: Time between frames
        f0_hz: Heart rate in Hz
        use_harmonic_dc: If True, use DC component from per-cycle harmonic fit
                        (more robust to within-cycle spikes). If False, use raw mean.

    Returns:
        cycle_means: (N_cycles,) array of per-cycle mean velocities
    """
    cycle_means = []

    for start, end in cycles:
        v_cycle = v_hat[start:end]
        valid = np.isfinite(v_cycle)

        if valid.sum() < 4:
            continue

        if use_harmonic_dc:
            # Fit harmonic to this cycle and extract DC component
            # This suppresses occasional GST spikes within a cycle
            hr = fit_harmonics(v_cycle, frame_dt, f0_hz, K=1, loss="huber", include_dc=True)
            if np.isfinite(hr['a0']):
                cycle_means.append(hr['a0'])
            else:
                cycle_means.append(np.nanmean(v_cycle))
        else:
            cycle_means.append(np.nanmean(v_cycle))

    return np.array(cycle_means, dtype=float)


def _compute_cycle_based_uncertainty(
    v_hat: np.ndarray,
    frame_dt: float,
    f0_hz: float,
    use_harmonic_dc: bool = True,
) -> Tuple[float, float, int]:
    """
    Compute uncertainty on time-averaged velocity using cycle-to-cycle variability.

    Uses sample standard deviation (not SEM) to give conservative error bars
    that represent actual cycle-to-cycle variability rather than uncertainty
    in the mean. This is more appropriate for small n (~6-9 cycles) where
    SEM can underestimate true uncertainty.

    Args:
        v_hat: (T,) velocity time series (px/frame or um/s)
        frame_dt: Time between frames in seconds
        f0_hz: Heart rate in Hz
        use_harmonic_dc: Use harmonic DC component per cycle (more robust)

    Returns:
        v_mean: Mean velocity (average of cycle means)
        sigma_v_mean: Cycle-to-cycle variability (sample std, not SEM)
        n_cycles: Number of complete cycles used
    """
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        # Fallback: treat entire series as one "cycle"
        v_mean = float(np.nanmean(v_hat))
        sigma_v_mean = float(np.nanstd(v_hat))  # Very conservative
        return v_mean, sigma_v_mean, 1

    # Segment into cycles
    cycles = _segment_into_cycles(v_hat, frame_dt, f0_hz)

    if len(cycles) < 2:
        # Not enough cycles for uncertainty estimate
        v_mean = float(np.nanmean(v_hat))
        sigma_v_mean = float(np.nanstd(v_hat))
        return v_mean, sigma_v_mean, len(cycles)

    # Compute per-cycle means
    cycle_means = _compute_cycle_means(v_hat, cycles, frame_dt, f0_hz, use_harmonic_dc)

    if len(cycle_means) < 2:
        v_mean = float(np.nanmean(v_hat))
        sigma_v_mean = float(np.nanstd(v_hat))
        return v_mean, sigma_v_mean, len(cycle_means)

    # Mean and cycle-to-cycle variability (use std, not SEM)
    # Using std instead of SEM gives more conservative error bars and
    # represents actual cycle-to-cycle variability rather than uncertainty in mean
    v_mean = float(np.mean(cycle_means))
    sigma_cycles = float(np.std(cycle_means, ddof=1))  # Sample std
    n_cycles = len(cycle_means)
    sigma_v_mean = sigma_cycles  # Use std directly (not SEM = std/sqrt(n))

    return v_mean, sigma_v_mean, n_cycles


def _compute_harmonic_residual_uncertainty(
    v_hat: np.ndarray,
    frame_dt: float,
    f0_hz: float,
    n_harmonics: int = N_HARMONICS,
) -> Tuple[float, float, int, float]:
    """
    Compute velocity mean and uncertainty from harmonic fit residuals.

    This approach explicitly separates:
    - Signal: periodic cardiac component (captured by harmonic fit)
    - Noise: everything else (GST error, RBC sparsity, sensor noise)

    Returns σ_v̄ = σ_rms / √n_cycles (standard error of the mean velocity)
    for use as weight in parabolic profile fit.  Also returns the raw
    σ_rms for use in band-quality filtering (Phase 4).

    Args:
        v_hat: (T,) velocity time series (px/frame or um/s)
        frame_dt: Time between frames in seconds
        f0_hz: Heart rate in Hz
        n_harmonics: Number of harmonics to fit (default 3)

    Returns:
        v_mean: Mean velocity (DC component from harmonic fit)
        sigma_v: σ_v̄ = σ_rms / √n_cycles (standard error of mean)
        n_cycles: Number of complete cycles in the time series
        sigma_v_rms: Raw RMS of harmonic fit residuals (per-frame noise)
    """
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        # Fallback: simple mean and std
        v_mean = float(np.nanmean(v_hat))
        sigma_v_rms = float(np.nanstd(v_hat))
        return v_mean, sigma_v_rms, 1, sigma_v_rms

    # Fit harmonic model
    hr_result = fit_harmonics(
        v_hat, frame_dt, f0_hz, K=n_harmonics,
        loss="huber", include_dc=True
    )

    # DC component is the mean velocity
    v_mean = float(hr_result.get('a0', np.nanmean(v_hat)))

    # Get residuals
    resid = hr_result.get('resid', np.array([]))
    if len(resid) == 0 or not np.any(np.isfinite(resid)):
        sigma_v_rms = float(np.nanstd(v_hat))
        return v_mean, sigma_v_rms, 1, sigma_v_rms

    # σ_rms = RMS of residuals (per-frame noise level)
    sigma_v_rms = float(np.sqrt(np.nanmean(resid**2)))

    # Count cycles — each cardiac cycle contributes one quasi-independent
    # measurement of the mean velocity (frames within a cycle are correlated)
    T_total = len(v_hat) * frame_dt
    n_cycles = max(1, int(T_total * f0_hz))

    # σ_v̄ = standard error of the mean velocity
    sigma_v = sigma_v_rms / np.sqrt(n_cycles)

    return v_mean, sigma_v, n_cycles, sigma_v_rms


# =============================================================================
# Spectral purity weighting
# =============================================================================

def compute_spectral_purity_weights(
    v_hat_array: np.ndarray,
    f0_hz: float,
    frame_dt: float,
    bandwidth_hz: float = 0.5,
) -> np.ndarray:
    """
    Compute spectral purity weights for each radial band based on how clean
    the peak is at the heart rate frequency f0.

    Spectral purity = (power in f0 ± bandwidth) / (total power)

    Args:
        v_hat_array: (n_bands, T) velocity time series for each radial band
        f0_hz: Heart rate frequency (Hz)
        frame_dt: Time between frames (seconds)
        bandwidth_hz: Frequency bandwidth around f0 to integrate (Hz)

    Returns:
        weights: (n_bands,) normalized spectral purity weights [0, 1]
    """
    n_bands, T = v_hat_array.shape

    # Compute PSD for each band
    fs = 1.0 / frame_dt  # Sampling frequency
    freqs = np.fft.rfftfreq(T, d=frame_dt)

    purity_scores = np.zeros(n_bands)

    for i in range(n_bands):
        v_signal = v_hat_array[i]

        # Skip bands with too many NaNs
        if np.isnan(v_signal).sum() > 0.5 * T:
            purity_scores[i] = 0.0
            continue

        # Replace NaNs with mean (simple interpolation)
        v_clean = v_signal.copy()
        if np.any(np.isnan(v_clean)):
            v_clean[np.isnan(v_clean)] = np.nanmean(v_clean)

        # Compute PSD
        fft = np.fft.rfft(v_clean - np.mean(v_clean))
        psd = np.abs(fft) ** 2

        # Find frequency bins within f0 ± bandwidth
        f0_mask = (freqs >= f0_hz - bandwidth_hz) & (freqs <= f0_hz + bandwidth_hz)

        # Spectral purity = power near f0 / total power
        power_f0 = np.sum(psd[f0_mask])
        power_total = np.sum(psd)

        if power_total > 0:
            purity_scores[i] = power_f0 / power_total
        else:
            purity_scores[i] = 0.0

    # Normalize weights to sum to 1
    weights = purity_scores / (np.sum(purity_scores) + 1e-10)

    return weights


# =============================================================================
# Depth-of-correlation weighted Poiseuille profile
# =============================================================================

def doc_weight(
    z: np.ndarray,
    z_c: float,
    z_0: float = 0.0,
    model: str = 'squared_lorentzian',
    p: float = 2.0,
) -> np.ndarray:
    """Compute depth-of-correlation weighting W(z).

    Parameters
    ----------
    z : array
        Depth coordinates (same units as z_c).
    z_c : float
        DOC half-depth (characteristic scale).
    z_0 : float
        Focal plane offset from vessel midplane.
    model : str
        'squared_lorentzian': W = 1/(1 + u²)²  — Olsen & Adrian (PIV)
        'lorentzian':         W = 1/(1 + u²)
        'power_lorentzian':   W = 1/(1 + u²)^p  — adjustable exponent
        'gaussian':           W = exp(-u²/2)
        where u = (z − z₀)/z_c.
    p : float
        Exponent for 'power_lorentzian' model (ignored by other models).
    """
    if model == 'none':
        return np.ones_like(np.asarray(z, dtype=float))
    u2 = ((z - z_0) / z_c) ** 2
    if model == 'squared_lorentzian':
        return 1.0 / (1.0 + u2) ** 2
    elif model == 'lorentzian':
        return 1.0 / (1.0 + u2)
    elif model == 'power_lorentzian':
        return 1.0 / (1.0 + u2) ** p
    elif model == 'gaussian':
        return np.exp(-u2 / 2.0)
    else:
        raise ValueError(
            f"Unknown DOC model '{model}'. "
            f"Choose from: none, squared_lorentzian, lorentzian, gaussian, "
            f"power_lorentzian")


def fit_effective_doc_weight(
    offsets_px: np.ndarray,
    v_gst_mean: np.ndarray,
    v_max_mean: float,
    R_px: float,
    z_c_init_px: float = 8.0,
    z_0_px: float = 0.0,
    n_quad: int = 64,
) -> dict:
    """Estimate effective DOC weighting W_eff(z) from GST measurements.

    Fits a power-Lorentzian W(z) = 1/(1 + ((z-z₀)/z_eff)²)^p to explain
    the GST-measured radial velocity profile.  The two free parameters
    (z_eff, p) are found by least-squares against the known 3-D Poiseuille
    velocity field.

    Parameters
    ----------
    offsets_px : (n_bands,) array
        Radial band offsets in pixels.
    v_gst_mean : (n_bands,) array
        Mean GST velocity per band (px/frame).
    v_max_mean : float
        Mean centerline velocity v_max (px/frame).
    R_px : float
        Vessel radius in pixels.
    z_c_init_px : float
        Initial guess for z_eff (use optical z_c).
    z_0_px : float
        Focal plane offset (fixed, not fitted).

    Returns
    -------
    dict with keys:
        z_eff : fitted effective DOC half-depth (px)
        p : fitted power exponent
        v_fitted : (n_bands,) predicted velocity profile at input offsets
        v_fitted_fine : (200,) predicted profile on fine r grid
        r_fine : (200,) fine r grid
        residuals : (n_bands,) fit residuals
        success : bool
    """
    from scipy.optimize import least_squares

    offsets_px = np.asarray(offsets_px, dtype=np.float64)
    v_gst_mean = np.asarray(v_gst_mean, dtype=np.float64)

    # Filter NaN/invalid bands
    valid = np.isfinite(v_gst_mean) & (np.abs(offsets_px) < R_px)
    if valid.sum() < 2:
        return {'z_eff': np.nan, 'p': np.nan, 'v_fitted': v_gst_mean * np.nan,
                'v_fitted_fine': np.zeros(200), 'r_fine': np.linspace(-R_px, R_px, 200),
                'residuals': np.full_like(v_gst_mean, np.nan), 'success': False}

    r_fit = offsets_px[valid]
    v_fit = v_gst_mean[valid]

    # Quadrature nodes
    z_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)

    def _forward(params):
        """Predict v(r_i) for given (z_eff, p)."""
        z_eff, p = params
        v_pred = np.zeros(len(r_fit))
        for i, r_i in enumerate(r_fit):
            h = np.sqrt(max(R_px**2 - r_i**2, 0.0))
            if h < 1e-6:
                continue
            z = h * z_nodes
            w = h * w_nodes
            u2 = ((z - z_0_px) / z_eff) ** 2
            W = 1.0 / (1.0 + u2) ** p
            rho2 = r_i**2 + z**2
            v_factor = np.maximum(0.0, 1.0 - rho2 / R_px**2)
            v_pred[i] = v_max_mean * np.sum(W * v_factor * w) / np.sum(W * w)
        return v_pred

    def _residuals(params):
        return _forward(params) - v_fit

    result = least_squares(
        _residuals, x0=[z_c_init_px, 2.0],
        bounds=([0.5, 0.1], [50.0, 10.0]),
        method='trf',
    )

    z_eff, p = result.x

    # Compute fitted profile on fine grid for plotting
    r_fine = np.linspace(-R_px, R_px, 200)
    v_fine = np.zeros(200)
    for i, r_i in enumerate(r_fine):
        h2 = R_px**2 - r_i**2
        if h2 <= 0:
            continue
        h = np.sqrt(h2)
        z = h * z_nodes
        w = h * w_nodes
        u2 = ((z - z_0_px) / z_eff) ** 2
        W = 1.0 / (1.0 + u2) ** p
        rho2 = r_i**2 + z**2
        v_factor = np.maximum(0.0, 1.0 - rho2 / R_px**2)
        v_fine[i] = v_max_mean * np.sum(W * v_factor * w) / np.sum(W * w)

    # Also compute at original offsets
    v_at_offsets = np.full_like(v_gst_mean, np.nan)
    v_at_offsets[valid] = _forward(result.x)

    return {
        'z_eff': z_eff,
        'p': p,
        'v_fitted': v_at_offsets,
        'v_fitted_fine': v_fine,
        'r_fine': r_fine,
        'residuals': result.fun,
        'success': result.success,
    }


def doc_weighted_poiseuille_profile(
    r_px: np.ndarray,
    v_max: float,
    R_px: float,
    z_c_px: float,
    r_offset_px: float = 0.0,
    z_0_px: float = 0.0,
    n_quad: int = 64,
    model: str = 'squared_lorentzian',
) -> np.ndarray:
    """Depth-averaged Poiseuille velocity with DOC weighting.

    Computes v_meas(r) = ∫ W(z−z₀) v(r,z) dz / ∫ W(z−z₀) dz
    where v(r,z) = v_max * max(0, 1 - (r² + z²)/R²)
    and W(z) = doc_weight(z, z_c, z₀, model).

    Integration is over the vessel chord z ∈ [-h, h] where h = sqrt(R² - r²).
    """
    r_px = np.atleast_1d(np.asarray(r_px, dtype=np.float64))
    r_eff = np.abs(r_px - r_offset_px)
    v_out = np.zeros_like(r_eff)

    # Precompute quadrature nodes on [-1, 1]
    z_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)

    for i, r in enumerate(r_eff):
        if r >= R_px:
            continue
        h = np.sqrt(R_px**2 - r**2)  # chord half-length
        # Map [-1,1] → [-h, h]
        z = h * z_nodes
        w = h * w_nodes

        W = doc_weight(z, z_c_px, z_0_px, model=model)
        # 3D Poiseuille velocity
        rho2 = r**2 + z**2
        v_z = v_max * np.maximum(0.0, 1.0 - rho2 / R_px**2)

        v_out[i] = np.sum(W * v_z * w) / np.sum(W * w)

    return v_out


def _poiseuille_shape(
    r_data_um: np.ndarray,
    R_um: float,
    r_offset_um: float,
    z_c_um: float,
    px_size_um: float,
    z_0_um: float = 0.0,
    doc_model: str = 'squared_lorentzian',
) -> np.ndarray:
    """Compute normalized Poiseuille shape factor, with or without DOC.

    Returns shape s(r) such that v(r) = v_max * s(r).
    When z_c_um <= 0: standard 2D Poiseuille s(r) = 1 - (r/R)².
    When z_c_um > 0: DOC-weighted depth average with focal plane offset z₀.
    """
    if z_c_um > 0:
        r_px = r_data_um / px_size_um
        R_px = R_um / px_size_um
        z_c_px = z_c_um / px_size_um
        r_off_px = r_offset_um / px_size_um
        z_0_px = z_0_um / px_size_um
        return doc_weighted_poiseuille_profile(
            r_px, 1.0, R_px, z_c_px, r_off_px, z_0_px, model=doc_model)
    else:
        r_norm = (r_data_um - r_offset_um) / R_um
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        return 1.0 - r_norm_clipped ** 2


# =============================================================================
# Poiseuille profile fitting with covariance estimation
# =============================================================================

def _fit_poiseuille_profile_nlls(
    offsets_px: np.ndarray,
    v_means: np.ndarray,
    sigma_v_mean: np.ndarray,
    vessel_radius_px: float,
    px_size_um: float = 1.0,
    spectral_purity_weights: Optional[np.ndarray] = None,
    max_radius_px: Optional[float] = None,
    use_grid_search_only: bool = False,
    fit_r_offset: bool = True,
    z_c_um: float = 0.0,
    z_0_um: float = 0.0,
    doc_model: str = 'squared_lorentzian',
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit Poiseuille velocity profile using nonlinear least squares with covariance.

    When z_c_um > 0, fits a DOC-weighted depth-averaged Poiseuille profile
    instead of the focal-plane profile. This accounts for widefield microscopy
    integrating over all depths in the vessel.  z_0_um is the focal plane
    offset from vessel midplane (source of systematic uncertainty).

    Profile: v(r) = v_max * (1 - |r_norm|^2)  (parabolic/Poiseuille)
    where r_norm = (r - r_offset) / R

    Fits 3 parameters: (v_max, R, r_offset) with n fixed at 2.

    Strategy: Grid search over r_offset to find good initial guess,
    then continuous optimization to get proper covariance.

    Args:
        offsets_px: Radial offsets in pixels
        v_means: Time-averaged velocities at each offset (um/s)
        sigma_v_mean: Per-band uncertainty (standard error of mean, um/s)
        vessel_radius_px: Initial vessel radius estimate in pixels
        px_size_um: Pixel size in micrometers
        spectral_purity_weights: Optional (n_bands,) weights from spectral purity.
            If provided, these are combined with inverse variance weighting.
        max_radius_px: Maximum allowed fitted radius in pixels. If None, uses
            1.5 * vessel_radius_px (legacy behavior).
        use_grid_search_only: If True (default), skip NLLS optimization and use
            grid search result directly. This uses R fixed at R_seg which avoids
            degenerate solutions where NLLS drifts R to bounds and v_max → 0.

    Returns:
        Dict with:
            v_max: Peak velocity (um/s)
            R_px: Fitted radius (pixels)
            R_um: Fitted radius (um)
            r_offset_px: Centerline offset (pixels)
            chi2_reduced: Reduced chi-squared
            cov_theta: 3x3 covariance matrix for (v_max, R_um, r_offset)
            param_std: Standard deviations of (v_max, R_um, r_offset)
            success: Whether fit converged
    """
    from scipy.optimize import least_squares

    # Filter valid data
    valid = (np.isfinite(v_means) & np.isfinite(sigma_v_mean) &
             (sigma_v_mean > 0) & (v_means != 0))

    n_valid = valid.sum()
    # Minimum 3 bands needed for 1-param fit (v_max only with R and r_offset fixed)
    MIN_BANDS_ABSOLUTE = 3
    if n_valid < MIN_BANDS_ABSOLUTE:
        return _make_failed_fit_result(v_means, vessel_radius_px, px_size_um)

    r_data = offsets_px[valid]
    v_data = v_means[valid]
    sigma_data = sigma_v_mean[valid]

    # Convert radius to um for fitting (better numerical conditioning)
    R_init_um = vessel_radius_px * px_size_um
    r_data_um = r_data * px_size_um

    # NOTE: No radius truncation here — the upstream caller (analyze_vessel)
    # already applies a shifted 90% R_seg filter centered on the estimated
    # vessel center.  A second truncation here with a different center
    # estimate would inconsistently re-remove bands that were deliberately kept.

    # Prepare spectral purity weights for valid data
    if spectral_purity_weights is not None:
        spec_weights_valid = spectral_purity_weights[valid]
    else:
        spec_weights_valid = np.ones(len(r_data))

    n_bands = len(v_data)

    # Debug: show what we're working with
    if verbose:
        print(f"\n  [NLLS INPUT] {n_bands} bands after filtering:")
        print(f"    offsets (px): {r_data.round(1)}")
        print(f"    v_means (um/s): {v_data.round(2)}")
        print(f"    sigma (um/s): {sigma_data.round(2)}")

    # -------------------------------------------------------------------------
    # Decide between fit modes based on data and fit_r_offset setting
    # -------------------------------------------------------------------------
    # When fit_r_offset=False (default):
    #   2-parameter fit (v_max, R) with r_offset=0 for >=4 bands
    #   1-parameter fit (v_max only) for 3 bands
    # When fit_r_offset=True:
    #   3-parameter fit (v_max, R, r_offset) for >=6 bands
    #   2-parameter fit (v_max, r_offset with R fixed) for 5 bands
    #   1-parameter fit (v_max only) for 3-4 bands
    MIN_BANDS_1PARAM = 3

    if n_bands < MIN_BANDS_1PARAM:
        if verbose:
            print(f"  [NLLS] FAILED: Only {n_bands} bands, need at least {MIN_BANDS_1PARAM}")
        return _make_failed_fit_result(v_means, vessel_radius_px, px_size_um)

    if fit_r_offset:
        # Legacy mode: allow r_offset as a free parameter
        MIN_BANDS_3PARAM = 6
        MIN_BANDS_2PARAM = 5
        use_1param_fit = (n_bands < MIN_BANDS_2PARAM)
        use_2param_fit = (n_bands >= MIN_BANDS_2PARAM) and (n_bands < MIN_BANDS_3PARAM)
    else:
        # Default: fix r_offset=0, fit (v_max, R) or (v_max)
        MIN_BANDS_2PARAM_VR = 4  # (v_max, R) needs >=4 bands for dof>=2
        use_1param_fit = (n_bands < MIN_BANDS_2PARAM_VR)
        use_2param_fit = False  # Never use (v_max, r_offset) mode when r_offset is fixed

    if use_1param_fit:
        # 1-parameter fit: fix R and r_offset, fit only v_max
        n_params_actual = 1
        dof = n_bands - n_params_actual
        R_fixed_um = R_init_um
        r_offset_fixed_um = 0.0
        if verbose:
            print(f"  [FIT MODE] 1-param fit (v_max only, R={R_fixed_um/px_size_um:.1f} px, r0=0 fixed) "
                  f"({n_bands} bands, dof={dof})")
    elif use_2param_fit:
        # 2-parameter fit: fix R, fit v_max and r_offset (only when fit_r_offset=True)
        n_params_actual = 2
        dof = n_bands - n_params_actual
        R_fixed_um = R_init_um
        if verbose:
            print(f"  [FIT MODE] 2-param fit (v_max, r0; R={R_fixed_um/px_size_um:.1f} px fixed) "
                  f"({n_bands} bands, dof={dof})")
    elif not fit_r_offset:
        # 2-parameter fit: fix r_offset=0, fit v_max and R
        n_params_actual = 2
        dof = n_bands - n_params_actual
        r_offset_fixed_um = 0.0
        if verbose:
            print(f"  [FIT MODE] 2-param fit (v_max, R; r0=0 fixed) "
                  f"({n_bands} bands, dof={dof})")
    else:
        # 3-parameter fit: fit v_max, R, and r_offset
        n_params_actual = 3
        dof = n_bands - n_params_actual
        if verbose:
            print(f"  [FIT MODE] 3-param fit (v_max, R, r0) ({n_bands} bands, dof={dof})")

    if dof < 1:
        if verbose:
            print(f"  [NLLS] FAILED: dof={dof} < 1")
        return _make_failed_fit_result(v_means, vessel_radius_px, px_size_um)

    # -------------------------------------------------------------------------
    # Phase 1: Grid search for good initial guess
    # -------------------------------------------------------------------------
    # Search over r_offset only; n is fixed at 2.0 for Poiseuille flow
    # When fit_r_offset=False, only evaluate at r_offset=0
    if fit_r_offset:
        max_offset_px = max(vessel_radius_px, 8)  # At least ±8 px, or ±radius
        offset_values_px = np.arange(-max_offset_px, max_offset_px + 0.5, 1.0)
    else:
        offset_values_px = np.array([0.0])
    offset_values_um = offset_values_px * px_size_um

    best_chi2 = np.inf
    best_init = None

    for offset_test_um in offset_values_um:
        shape = _poiseuille_shape(r_data_um, R_init_um, offset_test_um, z_c_um, px_size_um, z_0_um, doc_model=doc_model)

        # WLS for v_max: combine inverse variance with spectral purity
        inv_var_weights = 1.0 / sigma_data**2
        weights = inv_var_weights * spec_weights_valid
        weights = weights / (np.sum(weights) + 1e-10)  # Normalize
        w_shape_sq = np.sum(weights * shape**2)
        if w_shape_sq < 1e-10:
            continue

        v_max_test = np.sum(weights * v_data * shape) / w_shape_sq

        # Chi-squared
        v_fit = v_max_test * shape
        chi2 = np.sum(((v_data - v_fit) / sigma_data)**2)

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_init = (v_max_test, R_init_um, offset_test_um)

    if best_init is None:
        if verbose:
            print(f"  [NLLS] FAILED: Grid search found no valid initial guess")
        return _make_failed_fit_result(v_means, vessel_radius_px, px_size_um)

    # -------------------------------------------------------------------------
    # Grid-search-only mode: skip NLLS, use grid search result directly
    # -------------------------------------------------------------------------
    # This avoids degenerate NLLS solutions where R drifts to bounds and v_max → 0
    # Grid search uses fixed R = R_seg which is more constrained and stable
    if use_grid_search_only:
        v_max_fit, R_um_fit, r_off_um_fit = best_init
        R_px_fit = R_um_fit / px_size_um
        r_off_px_fit = r_off_um_fit / px_size_um
        chi2_reduced = best_chi2 / dof if dof > 0 else np.inf

        if verbose:
            print(f"  [GRID SEARCH] Using grid search result (R fixed at {R_px_fit:.1f} px):")
            print(f"    v_max = {v_max_fit:.2f} um/s, r_offset = {r_off_px_fit:.1f} px, χ²_red = {chi2_reduced:.3f}")

        # Estimate covariance from grid search (approximate)
        # Since v_max is analytically optimal for each r_offset, we can estimate
        # its variance from the weighted least squares formula
        shape = _poiseuille_shape(r_data_um, R_um_fit, r_off_um_fit, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
        inv_var_weights = 1.0 / sigma_data**2
        weights = inv_var_weights * spec_weights_valid
        w_shape_sq = np.sum(weights * shape**2)
        # Variance of v_max from WLS: Var(v_max) ≈ 1 / sum(w * shape^2)
        var_v_max = 1.0 / (w_shape_sq + 1e-10)
        std_v_max = np.sqrt(var_v_max)

        # For R and r_offset, we don't have good variance estimates from grid search
        # Use placeholder values (these are fixed anyway in grid search mode)
        cov_theta = np.diag([var_v_max, 0.0, 0.0])

        return {
            'v_max': v_max_fit,
            'R_px': R_px_fit,
            'R_um': R_um_fit,
            'r_offset_px': r_off_px_fit,
            'r_offset_um': r_off_um_fit,
            'chi2_reduced': chi2_reduced,
            'cov_theta': cov_theta,
            'param_std': np.array([std_v_max, 0.0, 0.0]),
            'success': True,
            'n_bands': n_bands,
            'dof': dof,
            'fit_mode': 'grid_search',
        }

    # -------------------------------------------------------------------------
    # Phase 2: Continuous optimization (NLLS)
    # -------------------------------------------------------------------------
    # Compute effective sigma for combined weighting
    if spectral_purity_weights is not None:
        sigma_eff = sigma_data / np.sqrt(spec_weights_valid + 1e-10)
    else:
        sigma_eff = sigma_data

    if use_1param_fit:
        # =====================================================================
        # 1-PARAMETER FIT: R and r_offset fixed, fit only v_max
        # =====================================================================
        # With R and r_offset fixed, v_max has closed-form WLS solution:
        # v_max = sum(w * v * f) / sum(w * f^2) where f = shape factor

        shape = _poiseuille_shape(r_data_um, R_fixed_um, r_offset_fixed_um, z_c_um, px_size_um, z_0_um, doc_model=doc_model)

        # Weights: 1/sigma^2
        weights = 1.0 / sigma_eff**2

        # WLS solution for v_max
        v_max_fit = np.sum(weights * v_data * shape) / np.sum(weights * shape**2)
        R_um_fit = R_fixed_um
        r_off_um_fit = r_offset_fixed_um
        success = True

        # Variance of v_max from WLS: Var(v_max) = 1 / sum(w * f^2)
        var_v_max = 1.0 / np.sum(weights * shape**2)

        # Build covariance matrix (only v_max has variance from fit)
        # R and r_offset use segmentation uncertainty (~1 px each)
        sigma_R_seg_um = 1.0 * px_size_um
        sigma_r0_seg_um = 1.0 * px_size_um  # uncertainty in centerline position

        cov_theta = np.diag([var_v_max, sigma_R_seg_um**2, sigma_r0_seg_um**2])
        param_std = np.array([np.sqrt(var_v_max), sigma_R_seg_um, sigma_r0_seg_um])

    elif use_2param_fit:
        # =====================================================================
        # 2-PARAMETER FIT: R fixed at R_seg, fit only (v_max, r_offset)
        # =====================================================================
        # r_offset bound scales with vessel size: max(2px, 0.3R)
        # Larger vessels have more centerline uncertainty
        max_r_offset_um_2p = max(2.0, 0.3 * vessel_radius_px) * px_size_um

        def residual_func_2p(theta):
            """Weighted residuals for 2-param fit (v_max, r_offset), R fixed."""
            v_max, r_off_um = theta

            # Bounds enforcement for r_offset
            if abs(r_off_um) > max_r_offset_um_2p:
                return np.full(n_bands, 1e6)

            shape = _poiseuille_shape(r_data_um, R_fixed_um, r_off_um, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
            v_fit = v_max * shape

            return (v_data - v_fit) / sigma_eff

        # Initial guess: (v_max, r_offset) from grid search
        x0_2p = np.array([best_init[0], best_init[2]])  # v_max, r_offset

        # Bounds for 2-parameter fit
        # r_offset bound scales with vessel size: max(2px, 0.3R)
        max_r_offset_um = max(2.0, 0.3 * vessel_radius_px) * px_size_um
        bounds_lower_2p = [-np.inf, -max_r_offset_um]
        bounds_upper_2p = [np.inf, max_r_offset_um]

        try:
            result = least_squares(
                residual_func_2p, x0_2p,
                bounds=(bounds_lower_2p, bounds_upper_2p),
                method='trf',
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
                max_nfev=1000,
            )

            if not result.success:
                theta_opt_2p = x0_2p
                success = False
            else:
                theta_opt_2p = result.x
                success = True

        except Exception:
            theta_opt_2p = x0_2p
            success = False

        # Extract fitted parameters (R is fixed, not fitted)
        v_max_fit = theta_opt_2p[0]
        R_um_fit = R_fixed_um  # Fixed, not fitted
        r_off_um_fit = theta_opt_2p[1]

    elif not fit_r_offset and not use_1param_fit:
        # =====================================================================
        # 2-PARAMETER FIT: fit (v_max, R) with r_offset fixed at 0
        # =====================================================================
        r_off_um_fit = 0.0

        if max_radius_px is not None:
            _min_R_um = max(1.0, vessel_radius_px - 1.0) * px_size_um
            _max_R_um = max_radius_px * px_size_um
        else:
            _min_R_um = 0.5 * R_init_um
            _max_R_um = 1.5 * R_init_um

        _sigma_R_prior_um = 1.0 * px_size_um

        def residual_func_vr(theta):
            """Weighted residuals for 2-param fit (v_max, R), r_offset=0."""
            v_max_t, R_um_t = theta
            if R_um_t < _min_R_um or R_um_t > _max_R_um:
                return np.full(n_bands + 1, 1e6)
            shape = _poiseuille_shape(r_data_um, R_um_t, 0.0, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
            v_fit = v_max_t * shape
            data_resid = (v_data - v_fit) / sigma_eff
            R_prior_resid = (R_um_t - R_init_um) / _sigma_R_prior_um
            return np.append(data_resid, R_prior_resid)

        x0_vr = np.array([best_init[0], best_init[1]])  # v_max, R
        bounds_lower_vr = [-np.inf, _min_R_um]
        bounds_upper_vr = [np.inf, _max_R_um]

        try:
            result = least_squares(
                residual_func_vr, x0_vr,
                bounds=(bounds_lower_vr, bounds_upper_vr),
                method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=1000,
            )
            if not result.success:
                theta_opt_vr = x0_vr
                success = False
            else:
                theta_opt_vr = result.x
                success = True
        except Exception:
            theta_opt_vr = x0_vr
            success = False

        v_max_fit, R_um_fit = theta_opt_vr
        R_drift_px = (R_um_fit - R_init_um) / px_size_um
        if verbose and abs(R_drift_px) > 0.1:
            print(f"  [R PRIOR] R_seg={vessel_radius_px:.1f} px → R_fit={R_um_fit/px_size_um:.1f} px "
                  f"(drift={R_drift_px:+.1f} px, prior σ=1.0 px)")

    else:
        # =====================================================================
        # 3-PARAMETER FIT: fit (v_max, R, r_offset)
        # =====================================================================
        # Compute bounds once for use in residual_func
        if max_radius_px is not None:
            # Use explicit bounds: [R_seg - 1, max_radius_px]
            _min_R_um = max(1.0, vessel_radius_px - 1.0) * px_size_um
            _max_R_um = max_radius_px * px_size_um
        else:
            # Legacy: ±50% of initial estimate
            _min_R_um = 0.5 * R_init_um
            _max_R_um = 1.5 * R_init_um

        # r_offset should only shift by ~2 pixels from skeleton centerline
        _max_r_offset_um = 2.0 * px_size_um

        # Soft prior on R toward R_seg: penalizes drift with ~1 px uncertainty.
        # This prevents R from drifting to bounds and inflating Q (which scales as R²)
        # while still allowing small corrections when data strongly supports them.
        _sigma_R_prior_um = 1.0 * px_size_um  # ±1 px prior uncertainty

        def residual_func_3p(theta):
            """Weighted residuals for 3-param fit (v_max, R, r_offset) with R prior."""
            v_max, R_um, r_off_um = theta

            # Bounds enforcement (soft, via large residuals)
            if R_um < _min_R_um or R_um > _max_R_um:
                return np.full(n_bands + 1, 1e6)
            if abs(r_off_um) > _max_r_offset_um:
                return np.full(n_bands + 1, 1e6)

            shape = _poiseuille_shape(r_data_um, R_um, r_off_um, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
            v_fit = v_max * shape

            data_resid = (v_data - v_fit) / sigma_eff

            # Prior residual: pulls R toward R_seg
            R_prior_resid = (R_um - R_init_um) / _sigma_R_prior_um

            return np.append(data_resid, R_prior_resid)

        # Initial guess from grid search
        x0 = np.array(best_init)  # v_max, R, r_offset

        # Bounds for 3-parameter fit
        # r_offset should only shift by ~2 pixels from skeleton centerline
        max_r_offset_um = 2.0 * px_size_um
        bounds_lower = [-np.inf, _min_R_um, -max_r_offset_um]
        bounds_upper = [np.inf, _max_R_um, max_r_offset_um]

        try:
            result = least_squares(
                residual_func_3p, x0,
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
                max_nfev=1000,
            )

            if not result.success:
                theta_opt = x0
                success = False
            else:
                theta_opt = result.x
                success = True

        except Exception:
            theta_opt = x0
            success = False

        # Extract fitted parameters
        v_max_fit, R_um_fit, r_off_um_fit = theta_opt
        R_drift_px = (R_um_fit - R_init_um) / px_size_um
        if verbose and abs(R_drift_px) > 0.1:
            print(f"  [R PRIOR] R_seg={vessel_radius_px:.1f} px → R_fit={R_um_fit/px_size_um:.1f} px "
                  f"(drift={R_drift_px:+.1f} px, prior σ=1.0 px)")
    R_px_fit = R_um_fit / px_size_um
    r_off_px_fit = r_off_um_fit / px_size_um

    # -------------------------------------------------------------------------
    # Phase 3: Compute covariance matrix
    # -------------------------------------------------------------------------
    # Covariance from Jacobian: Cov(theta) ≈ (J^T W J)^{-1}
    # where W = diag(1/sigma^2) and J is the Jacobian of the model
    # Note: For 1-param fit, covariance is already computed above (WLS formula)

    if use_1param_fit:
        # Covariance already computed in 1-param fit section
        pass

    elif use_2param_fit:
        # 2-parameter Jacobian (v_max, r_offset) with R fixed
        def compute_jacobian_2p(theta_2p):
            """Numerical Jacobian for 2-parameter Poiseuille model (R fixed)."""
            v_max, r_off_um = theta_2p
            eps = 1e-6
            J = np.zeros((n_bands, 2))

            def model_2p(th):
                vm, ro = th
                shape = _poiseuille_shape(r_data_um, R_fixed_um, ro, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
                return vm * shape

            v0 = model_2p(theta_2p)
            for i in range(2):
                theta_plus = theta_2p.copy()
                theta_plus[i] += eps
                J[:, i] = (model_2p(theta_plus) - v0) / eps

            return J

        try:
            J_2p = compute_jacobian_2p(theta_opt_2p)
            W = np.diag(1.0 / sigma_data**2)
            JtWJ = J_2p.T @ W @ J_2p

            # Add small regularization for numerical stability
            reg = 1e-10 * np.eye(2)
            cov_theta_2p = np.linalg.inv(JtWJ + reg)

            # Expand 2x2 covariance to 3x3 for compatibility
            # Order: (v_max, R, r_offset)
            # For R: use segmentation uncertainty (~1 px) since R was fixed, not fitted
            # This allows Q uncertainty propagation to work
            sigma_R_seg_um = 1.0 * px_size_um  # ~1 px segmentation uncertainty

            cov_theta = np.zeros((3, 3))
            cov_theta[0, 0] = cov_theta_2p[0, 0]  # var(v_max)
            cov_theta[1, 1] = sigma_R_seg_um ** 2  # var(R) from segmentation
            cov_theta[2, 2] = cov_theta_2p[1, 1]  # var(r_offset)
            cov_theta[0, 2] = cov_theta_2p[0, 1]  # cov(v_max, r_offset)
            cov_theta[2, 0] = cov_theta_2p[1, 0]  # cov(r_offset, v_max)
            # Note: cov(v_max, R) and cov(r_offset, R) are 0 since R was fixed

            # param_std: (σ_vmax, σ_R_seg, σ_r0)
            param_std = np.array([
                np.sqrt(max(0.0, cov_theta_2p[0, 0])),
                sigma_R_seg_um,  # R fixed at R_seg with ~1px segmentation uncertainty
                np.sqrt(max(0.0, cov_theta_2p[1, 1]))
            ])

        except Exception:
            cov_theta = np.full((3, 3), np.nan)
            param_std = np.full(3, np.nan)

    else:
        # 3-parameter Jacobian (v_max, R, r_offset) including R prior
        def compute_jacobian_3p(theta):
            """Numerical Jacobian for 3-parameter Poiseuille model (data rows only)."""
            v_max, R_um, r_off_um = theta
            eps = 1e-6
            J = np.zeros((n_bands, 3))

            def model_3p(th):
                vm, R, ro = th
                shape = _poiseuille_shape(r_data_um, R, ro, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
                return vm * shape

            v0 = model_3p(theta)
            for i in range(3):
                theta_plus = theta.copy()
                theta_plus[i] += eps
                J[:, i] = (model_3p(theta_plus) - v0) / eps

            return J

        try:
            J = compute_jacobian_3p(theta_opt)
            W = np.diag(1.0 / sigma_data**2)
            JtWJ = J.T @ W @ J

            # Add R prior contribution to Fisher information: 1/σ_R_prior² on the R diagonal
            # This is equivalent to the extra residual row in residual_func_3p
            _sigma_R_prior_um = 1.0 * px_size_um
            R_prior_info = np.zeros((3, 3))
            R_prior_info[1, 1] = 1.0 / _sigma_R_prior_um**2
            JtWJ = JtWJ + R_prior_info

            # Add small regularization for numerical stability
            reg = 1e-10 * np.eye(3)
            cov_theta = np.linalg.inv(JtWJ + reg)

            # param_std: (σ_vmax, σ_R, σ_r0)
            param_var = np.diag(cov_theta)
            param_var = np.maximum(param_var, 0.0)
            param_std = np.sqrt(param_var)

        except Exception:
            cov_theta = np.full((3, 3), np.nan)
            param_std = np.full(3, np.nan)

    # -------------------------------------------------------------------------
    # Phase 4: Compute chi-squared
    # -------------------------------------------------------------------------
    shape_fit = _poiseuille_shape(r_data_um, R_um_fit, r_off_um_fit, z_c_um, px_size_um, z_0_um, doc_model=doc_model)
    v_fit = v_max_fit * shape_fit

    chi2 = np.sum(((v_data - v_fit) / sigma_data)**2)
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    # Override scipy's success flag if chi2_reduced indicates a good fit
    # A fit with chi2_red < 3 is acceptable regardless of convergence details
    if not success and np.isfinite(chi2_reduced) and chi2_reduced < 3.0:
        success = True

    # Determine fit mode string for diagnostics
    if use_1param_fit:
        fit_mode = '1param'
    elif use_2param_fit:
        fit_mode = '2param'
    else:
        fit_mode = '3param'

    return {
        'v_max': v_max_fit,
        'R_px': R_px_fit,
        'R_um': R_um_fit,
        'r_offset_px': r_off_px_fit,
        'r_offset_um': r_off_um_fit,
        'chi2_reduced': chi2_reduced,
        'chi2': chi2,
        'dof': dof,
        'cov_theta': cov_theta,  # 3x3 covariance for (v_max, R_um, r_offset_um)
        'param_std': param_std,  # Std devs: (σ_vmax, σ_R, σ_r0)
        'success': success,
        'n_bands': n_bands,
        'n_params': n_params_actual,  # 1, 2, or 3
        'R_fixed': use_1param_fit or use_2param_fit,  # True if R was fixed at R_seg
        'r_offset_fixed': use_1param_fit,  # True if r_offset was also fixed
        'fit_mode': fit_mode,
    }


def _make_failed_fit_result(v_means, vessel_radius_px, px_size_um):
    """Return a failed fit result with NaN values."""
    v_max = float(np.nanmax(np.abs(v_means))) if len(v_means) > 0 else np.nan
    return {
        'v_max': v_max,
        'R_px': vessel_radius_px,
        'R_um': vessel_radius_px * px_size_um,
        'r_offset_px': 0.0,
        'r_offset_um': 0.0,
        'chi2_reduced': np.nan,
        'chi2': np.nan,
        'dof': 0,
        'cov_theta': np.full((3, 3), np.nan),
        'param_std': np.full(3, np.nan),
        'success': False,
        'n_bands': 0,
        'n_params': 0,
        'R_fixed': False,
    }


def _propagate_uncertainty_to_Q(
    v_max: float,
    R_um: float,
    cov_theta: np.ndarray,
) -> Tuple[float, float]:
    """
    Propagate parameter uncertainty to flow rate Q using Jacobian.

    For Poiseuille flow (n=2): Q = π * v_max * R² * (1/2)

    Args:
        v_max: Peak velocity (um/s)
        R_um: Radius (um)
        cov_theta: 3x3 covariance matrix for (v_max, R_um, r_offset)

    Returns:
        Q_nL_s: Flow rate in nL/s
        sigma_Q: Uncertainty in Q (nL/s)
    """
    # Q = π * v_max * R² * n/(n+2) with n=2 → Q = π * v_max * R² / 2
    # Convert to nL/s: um/s * um² = um³/s, divide by 1e6 to get nL/s

    profile_factor = 0.5  # n/(n+2) = 2/4 = 0.5 for Poiseuille
    Q_um3_per_s = np.pi * v_max * R_um**2 * profile_factor
    Q_nL_s = Q_um3_per_s / 1e6

    # Jacobian: J = [∂Q/∂v_max, ∂Q/∂R, ∂Q/∂r0]
    # Note: cov_theta is for (v_max, R_um, r_offset_um)

    # ∂Q/∂v_max = π * R² / 2
    dQ_dv = np.pi * R_um**2 * profile_factor / 1e6

    # ∂Q/∂R = π * v_max * R (derivative of R²)
    dQ_dR = np.pi * v_max * R_um / 1e6  # 2 * 0.5 = 1

    # ∂Q/∂r0 = 0 (Q doesn't depend on centerline offset)
    dQ_dr0 = 0.0

    J = np.array([dQ_dv, dQ_dR, dQ_dr0])

    # σ_Q² = J @ Cov @ J^T
    if np.any(np.isnan(cov_theta)):
        sigma_Q = np.nan
    else:
        try:
            var_Q = J @ cov_theta @ J.T
            sigma_Q = np.sqrt(max(var_Q, 0.0))
        except Exception:
            sigma_Q = np.nan

    return Q_nL_s, sigma_Q


def _fit_poiseuille_profile(
    offsets_px: np.ndarray,
    v_means: np.ndarray,
    vessel_radius_px: float,
    r_offset_px: float = 0.0,
) -> Tuple[float, float]:
    """
    Fit Poiseuille velocity profile: v(r) = v_max * (1 - ((r-r0)/R)²).

    Args:
        offsets_px: Radial offsets in pixels
        v_means: Mean velocities at each offset (um/s)
        vessel_radius_px: Vessel radius in pixels
        r_offset_px: Centerline offset in pixels

    Returns:
        v_max: Fitted maximum velocity (um/s)
        r2: Fit quality (R²)
    """
    from scipy.optimize import curve_fit

    valid = np.isfinite(v_means)
    if valid.sum() < 2:
        return float(np.nanmax(np.abs(v_means))) if len(v_means) > 0 else 0.0, 0.0

    r_data = offsets_px[valid]
    v_data = v_means[valid]

    def poiseuille(r_px, v_max):
        r_norm = (r_px - r_offset_px) / vessel_radius_px
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        return v_max * (1 - r_norm_clipped**2)

    try:
        # Initial guess from centerline (near r_offset)
        center_idx = np.argmin(np.abs(r_data - r_offset_px))
        v0 = v_data[center_idx]
        popt, _ = curve_fit(poiseuille, r_data, v_data, p0=[v0])
        v_max_fit = popt[0]

        # Compute R²
        v_fit = poiseuille(r_data, v_max_fit)
        ss_res = np.sum((v_data - v_fit)**2)
        ss_tot = np.sum((v_data - np.nanmean(v_data))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return float(v_max_fit), float(r2)
    except Exception:
        return float(np.nanmax(np.abs(v_means))), 0.0


def _compute_inner_bands_estimate(
    offsets_px: np.ndarray,
    v_hat_um_s: np.ndarray,
    sigma_v_mean_um_s: np.ndarray,
    R_inferred_px: float,
    r_offset_inferred_px: float,
    px_size_um: float,
    frame_dt: float,
    f0_hz: Optional[float],
    inner_fraction: float = 0.5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute Q estimate using inner bands (within inner_fraction of inferred radius).

    This method is more robust to junction effects that corrupt outer bands.
    We use WLS (same as full fit) but only on bands within the inner region:
    1. Select bands within inner_fraction × R_inferred of the inferred centerline
    2. Use WLS with Poiseuille shape factors to compute v_max(t)
    3. Compute Q(t) = v_max(t) × πR² × 0.5 using R_inferred

    The inferred centerline and radius come from signal-based boundary detection:
    - r_offset_inferred = (r_positive - r_negative) / 2 (midpoint of good signal region)
    - R_inferred = (r_positive + r_negative) / 2 (half-span of good signal region)

    This is independent of the NLLS fit and uses only the data-driven bounds.

    Args:
        offsets_px: Radial offsets in pixels (n_bands,)
        v_hat_um_s: Velocity time series in um/s (n_bands, T)
        sigma_v_mean_um_s: Per-band uncertainty in um/s (n_bands,)
        R_inferred_px: Inferred radius from signal bounds (pixels)
        r_offset_inferred_px: Inferred centerline offset from signal bounds (pixels)
        px_size_um: Pixel size in micrometers
        frame_dt: Time between frames in seconds
        f0_hz: Heart rate in Hz (for harmonic fitting)
        inner_fraction: Fraction of R_inferred to use for inner bands (default 0.5)

    Returns:
        Dict with:
            Q_t_inner: Q(t) time series from inner bands (nL/s)
            Q_mean_inner: Mean Q from inner bands (nL/s)
            v_max_inner: v_max estimate from inner bands (um/s)
            chi2_inner: Reduced chi² of inner bands vs implied parabola
            n_inner_bands: Number of bands used
            inner_offsets_px: Offsets of bands used
    """
    empty_result = {
        'Q_t_inner': np.array([]),
        'Q_mean_inner': np.nan,
        'v_max_inner': np.nan,
        'chi2_inner': np.nan,
        'n_inner_bands': 0,
        'inner_offsets_px': np.array([]),
    }

    if len(offsets_px) < 1 or not np.isfinite(R_inferred_px) or R_inferred_px <= 0:
        return empty_result

    T = v_hat_um_s.shape[1] if v_hat_um_s.ndim == 2 else 0
    if T == 0:
        return empty_result

    R_inferred_um = R_inferred_px * px_size_um

    # Select inner bands: all bands within inner_fraction × R_inferred of the inferred centerline
    # For inner_fraction=0.5, this gives bands within 50% of R from the true center
    inner_radius_px = inner_fraction * R_inferred_px
    dist_from_center = np.abs(offsets_px - r_offset_inferred_px)
    inner_mask = dist_from_center <= inner_radius_px

    if np.sum(inner_mask) < 1:
        # Fallback: use the 3 closest bands if no bands within inner region
        sorted_idx = np.argsort(dist_from_center)
        n_use = min(3, len(offsets_px))
        inner_mask = np.zeros(len(offsets_px), dtype=bool)
        inner_mask[sorted_idx[:n_use]] = True

    inner_offsets = offsets_px[inner_mask]
    inner_v_hat = v_hat_um_s[inner_mask]  # (n_inner, T)
    inner_sigma = sigma_v_mean_um_s[inner_mask]

    n_inner = len(inner_offsets)

    # Simple approach: use the band with maximum velocity as v_max
    # This is more robust than relying on the inferred centerline from sigma bounds
    inner_v_means = np.array([np.nanmean(v) for v in inner_v_hat])

    # Find the band with maximum velocity magnitude (this is the true center)
    center_idx = np.argmax(np.abs(inner_v_means))

    # The band with max velocity is at the flow center - use it directly as v_max
    # (no shape factor correction needed since we're defining this as the center)
    v_max_inner = inner_v_means[center_idx]

    # For Q(t), use the center band's time series directly
    center_v_hat = inner_v_hat[center_idx]  # (T,)
    v_max_t = center_v_hat

    # Q(t) = v_max(t) × πR² × 0.5 / 1e6  [nL/s]
    Q_factor = np.pi * R_inferred_um ** 2 * 0.5 / 1e6
    Q_t_inner = v_max_t * Q_factor
    Q_mean_inner = float(np.nanmean(Q_t_inner))

    # Recompute shape factors relative to the actual velocity center (max velocity band)
    # This is more meaningful for chi² since we're using max velocity as v_max
    velocity_center_px = inner_offsets[center_idx]
    f_r_from_vmax = 1.0 - np.clip(np.abs((inner_offsets - velocity_center_px) / R_inferred_px), 0.0, 1.0) ** 2

    # Compute chi² for inner bands: how well do they match the implied parabola?
    # Expected velocity at each inner band: v_expected = v_max_inner × f(r)
    v_expected = v_max_inner * f_r_from_vmax
    residuals = inner_v_means - v_expected
    dof_inner = max(n_inner - 1, 1)
    chi2_inner = np.sum((residuals / inner_sigma) ** 2) / dof_inner

    if verbose:
        print(f"\n  [INNER BANDS] Alternative estimate using {n_inner} inner bands (r < {inner_fraction:.0%} R):")
        print(f"    R_inferred={R_inferred_px:.1f} px, r_offset_inferred={r_offset_inferred_px:.1f} px")
        print(f"    Inner offsets (px): {inner_offsets}")
        print(f"    Inner v_means (um/s): {np.array2string(inner_v_means, precision=1)}")
        print(f"    Max velocity band: offset={inner_offsets[center_idx]} px, v={inner_v_means[center_idx]:.1f} um/s")
        print(f"    Shape factors f(r) from vmax: {np.array2string(f_r_from_vmax, precision=3)}")
        print(f"    v_max_inner: {v_max_inner:.2f} um/s")
        print(f"    Q_mean_inner: {Q_mean_inner:.3f} nL/s")
        print(f"    chi2_inner: {chi2_inner:.2f}")

    return {
        'Q_t_inner': Q_t_inner,
        'Q_mean_inner': Q_mean_inner,
        'v_max_inner': v_max_inner,
        'chi2_inner': chi2_inner,
        'n_inner_bands': n_inner,
        'inner_offsets_px': inner_offsets,
        'velocity_center_px': velocity_center_px,  # Centerline from max velocity band
    }


def _compute_Q_from_profiles(
    profile_data: List[Dict[str, Any]],
    vessel_radius_px: float,
    px_size_um: float,
    frame_dt: float,
    sigma_R_px: float = 1.0,
    R_inferred_px: Optional[float] = None,
    r_offset_inferred_px: Optional[float] = None,
    max_radius_px: Optional[float] = None,
    uncertainty_method: str = "harmonic_residual",
    z_c_um: float = 0.0,
    z_0_um: float = 0.0,
    doc_model: str = 'squared_lorentzian',
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute Q(t) using Poiseuille profile with proper uncertainty estimation.

    Pipeline:
    1. For each radial band, compute velocity uncertainty using specified method
    2. Fit Poiseuille profile (n=2) using NLLS with σ_v̄ weights
    3. Propagate parameter covariance to σ_Q using Jacobian

    Args:
        profile_data: List of profile dicts with 'offset', 'v_hat', 'consensus_f0_hz'
        vessel_radius_px: Vessel radius in pixels (initial estimate)
        px_size_um: Pixel size in micrometers
        frame_dt: Time between frames in seconds
        sigma_R_px: Prior uncertainty in vessel radius in pixels (default: 1.0)
        R_inferred_px: Inferred radius from signal bounds (for inner bands estimate)
        r_offset_inferred_px: Inferred centerline offset from signal bounds
        max_radius_px: Maximum allowed fitted radius in pixels. If provided,
            bounds are [R_seg-1, max_radius_px] instead of legacy ±50%.

    Returns:
        Dict with:
            Q_t: (T,) volumetric flow time series in nL/s
            Q_mean: Mean volumetric flow (nL/s)
            sigma_Q: Uncertainty in Q from profile fit covariance (nL/s)
            v_max: Fitted peak velocity (um/s)
            R_fit_px: Fitted vessel radius (pixels) - may differ from input
            R_fit_um: Fitted vessel radius (um)
            r_offset_px: Centerline offset (pixels)
            chi2_reduced: Reduced chi-squared (should be ~1.0 if calibrated)
            cov_theta: 3x3 parameter covariance matrix
            param_std: (3,) parameter standard deviations
            n_cycles: Number of heartbeat cycles used for uncertainty
            sigma_v_mean: (n_bands,) per-band standard error of mean velocity
            fit_success: Whether NLLS optimization converged
    """
    # -------------------------------------------------------------------------
    # Extract data dimensions and heart rate
    # -------------------------------------------------------------------------
    T = None
    f0_hz = None
    for p in profile_data:
        if p.get('v_hat') is not None:
            T = len(p['v_hat'])
            f0_hz = p.get('consensus_f0_hz', None)
            break

    empty_result = {
        'Q_t': np.array([]), 'Q_mean': np.nan, 'sigma_Q': np.nan,
        'v_max': np.nan, 'R_fit_px': vessel_radius_px,
        'R_fit_um': vessel_radius_px * px_size_um,
        'r_offset_px': np.nan,
        'chi2_reduced': np.nan, 'cov_theta': np.full((3, 3), np.nan),
        'param_std': np.full(3, np.nan), 'n_cycles': 0,
        'sigma_v_mean': np.array([]), 'fit_success': False,
        'envelope_info': None,
        # Legacy fields for compatibility
        'v_max_mean': np.nan, 'r_offset': np.nan,
        'sigma_noise': np.array([]), 'sigma_Q_random': np.nan,
        'sigma_Q_systematic': np.nan, 'sigma_Q_total': np.nan, 'N_eff': np.nan,
    }

    if T is None:
        return empty_result

    scale = px_size_um / frame_dt  # Convert px/frame to um/s

    # -------------------------------------------------------------------------
    # Collect per-band data with cycle-based uncertainty
    # -------------------------------------------------------------------------
    offsets_px = []
    v_means_um_s = []
    sigma_v_mean_um_s = []  # Standard error of mean (σ_rms / √n_cycles)
    sigma_v_rms_um_s = []   # Raw per-frame RMS noise (for band quality filter)
    coherences = []          # Per-band mean GST coherence (for band quality filter)
    v_hat_um_s = []
    n_cycles_list = []

    for p in profile_data:
        v_hat = p.get('v_hat')
        if v_hat is None or len(v_hat) != T:
            continue

        offset = p['offset']

        # Compute uncertainty for this band using specified method
        if f0_hz is not None and np.isfinite(f0_hz) and f0_hz > 0:
            if uncertainty_method == "harmonic_residual":
                # Use harmonic fit residuals — cleaner separation of signal vs noise
                v_mean_band, sigma_v_band, n_cyc, sigma_rms_band = _compute_harmonic_residual_uncertainty(
                    v_hat, frame_dt, f0_hz
                )
            else:
                # Use cycle-to-cycle variability (legacy method)
                v_mean_band, sigma_v_band, n_cyc = _compute_cycle_based_uncertainty(
                    v_hat, frame_dt, f0_hz, use_harmonic_dc=True
                )
                sigma_rms_band = sigma_v_band  # No separate RMS for legacy method
            # Convert to um/s
            v_mean_band *= scale
            sigma_v_band *= scale
            sigma_rms_band *= scale
        else:
            # Fallback: use simple mean and std
            v_mean_band = float(np.nanmean(v_hat)) * scale
            sigma_v_band = float(np.nanstd(v_hat)) * scale
            sigma_rms_band = sigma_v_band
            n_cyc = 1

        # Ensure minimum uncertainty to avoid numerical issues
        sigma_v_band = max(sigma_v_band, 0.01)  # µm/s minimum (SEM is ~3-5× smaller than RMS)

        offsets_px.append(offset)
        v_means_um_s.append(v_mean_band)
        sigma_v_mean_um_s.append(sigma_v_band)
        sigma_v_rms_um_s.append(sigma_rms_band)
        coherences.append(p.get('mean_coherence', 0.0))
        v_hat_um_s.append(v_hat * scale)
        n_cycles_list.append(n_cyc)

    if len(offsets_px) < 1:
        return empty_result

    offsets_px = np.array(offsets_px)
    v_means_um_s = np.array(v_means_um_s)
    sigma_v_mean_um_s = np.array(sigma_v_mean_um_s)
    sigma_v_rms_um_s = np.array(sigma_v_rms_um_s)
    coherences = np.array(coherences)
    v_hat_um_s = np.array(v_hat_um_s)  # Shape: (n_bands, T)
    n_cycles = int(np.median(n_cycles_list)) if n_cycles_list else 1

    # σ_v̄ = σ_rms / √n_cycles (standard error of mean velocity per band)
    # Used for weighting the parabolic profile fit — bands with noisy velocity
    # time series get higher σ_v̄ and thus less weight.
    if verbose:
        print(f"  [UNCERTAINTY] Method: {uncertainty_method}, n_cycles={n_cycles}")
        print(f"    σ_v̄ (um/s): {np.array2string(sigma_v_mean_um_s, precision=2, suppress_small=True)}")
        print(f"    σ_rms (um/s): {np.array2string(sigma_v_rms_um_s, precision=1, suppress_small=True)}")

    # Filter to bands inside 90% of the segmented radius, centered on the
    # velocity-weighted peak rather than the skeleton center (offset=0).
    # The skeleton often doesn't coincide with the true vessel center, so a
    # symmetric |offset| cutoff rejects valid bands on the shifted side.
    #
    # Preliminary r₀: σ_v-weighted mean of offsets near the velocity peak.
    # Only bands with finite, reasonable σ_v contribute (excludes far-outside
    # bands whose σ_v is orders of magnitude larger).
    truncation_factor = 0.9
    truncation_radius_px = truncation_factor * vessel_radius_px

    # Estimate preliminary vessel center from velocity profile
    _finite = np.isfinite(sigma_v_mean_um_s) & (sigma_v_mean_um_s > 0)
    if np.sum(_finite) >= 3:
        _sig = sigma_v_mean_um_s[_finite]
        _sig_med = np.median(_sig)
        # Use only bands with reasonable σ_v (within 5× median) for center estimate
        _reasonable = _finite & (sigma_v_mean_um_s < 5.0 * _sig_med)
        if np.sum(_reasonable) >= 3:
            _w = np.abs(v_means_um_s[_reasonable]) / sigma_v_mean_um_s[_reasonable] ** 2
            _w_sum = np.sum(_w)
            r0_prelim = np.sum(_w * offsets_px[_reasonable]) / _w_sum if _w_sum > 0 else 0.0
        else:
            r0_prelim = 0.0
        # Clamp to ±0.4R so the shift doesn't get unreasonable
        r0_prelim = np.clip(r0_prelim, -0.4 * vessel_radius_px, 0.4 * vessel_radius_px)
    else:
        r0_prelim = 0.0

    within_radius = np.abs(offsets_px - r0_prelim) <= truncation_radius_px
    n_excluded = np.sum(~within_radius)
    r0_label = f", centered at r₀={r0_prelim:+.1f}px" if abs(r0_prelim) > 0.1 else ""
    if verbose and n_excluded > 0:
        print(f"  [FILTER] Excluding {n_excluded} bands outside {truncation_factor:.0%} R_seg ({truncation_radius_px:.1f} px{r0_label})")
    offsets_px = offsets_px[within_radius]
    v_means_um_s = v_means_um_s[within_radius]
    sigma_v_mean_um_s = sigma_v_mean_um_s[within_radius]
    sigma_v_rms_um_s = sigma_v_rms_um_s[within_radius]
    coherences = coherences[within_radius]
    v_hat_um_s = v_hat_um_s[within_radius]

    # Two-criterion band quality filter (coherence + RMS residual)
    # A band is KEPT if it has good streaks (coherence ≥ 0.3) OR reasonable
    # RMS noise (< 10× median).  A band FAILS only if BOTH coherence is low
    # AND residual RMS is extreme — this avoids penalizing flat-velocity bands
    # that have good spatial signal but no pulsatility.
    BAND_COHERENCE_MIN = 0.3
    RMS_OUTLIER_FACTOR = 10.0
    if len(sigma_v_rms_um_s) > 3:
        rms_median = np.median(sigma_v_rms_um_s)
        good_coherence = coherences >= BAND_COHERENCE_MIN
        good_rms = sigma_v_rms_um_s < RMS_OUTLIER_FACTOR * rms_median
        good_band = good_coherence | good_rms
        n_filtered = np.sum(~good_band)
        if n_filtered > 0:
            if verbose:
                for idx in np.where(~good_band)[0]:
                    print(f"  [FILTER] Band at offset {offsets_px[idx]:+.0f}px: "
                          f"coh={coherences[idx]:.2f}, σ_rms={sigma_v_rms_um_s[idx]:.1f} "
                          f"(>{RMS_OUTLIER_FACTOR:.0f}×med={rms_median:.1f})")
            offsets_px = offsets_px[good_band]
            v_means_um_s = v_means_um_s[good_band]
            sigma_v_mean_um_s = sigma_v_mean_um_s[good_band]
            sigma_v_rms_um_s = sigma_v_rms_um_s[good_band]
            coherences = coherences[good_band]
            v_hat_um_s = v_hat_um_s[good_band]

    if len(offsets_px) < 3:
        return empty_result

    # -------------------------------------------------------------------------
    # Outlier detection: remove bands with extreme velocities (artifacts)
    # Using MAD (Median Absolute Deviation) which is robust to outliers
    # -------------------------------------------------------------------------
    if len(v_means_um_s) >= 5:
        v_median = np.median(v_means_um_s)
        v_mad = np.median(np.abs(v_means_um_s - v_median))
        # Scale MAD to approximate std dev for normal distribution
        mad_scale = 1.4826  # For normal distribution: σ ≈ 1.4826 × MAD
        v_mad_scaled = mad_scale * v_mad

        # Remove points > 4 MAD from median (roughly 4σ outliers)
        MAD_THRESHOLD = 4.0
        if v_mad_scaled > 0:
            z_scores = np.abs(v_means_um_s - v_median) / v_mad_scaled
            good_velocity = z_scores < MAD_THRESHOLD
            n_outliers = np.sum(~good_velocity)

            if n_outliers > 0:
                outlier_indices = np.where(~good_velocity)[0]
                outlier_velocities = v_means_um_s[~good_velocity]
                if verbose:
                    print(f"  [OUTLIER] Removing {n_outliers} velocity outliers (>{MAD_THRESHOLD} MAD from median):")
                    print(f"    Median velocity: {v_median:.2f} um/s, MAD: {v_mad_scaled:.2f} um/s")
                    for idx, v_out in zip(outlier_indices, outlier_velocities):
                        print(f"    Band at offset {offsets_px[idx]:.1f}px: v={v_out:.2f} um/s (z={z_scores[idx]:.1f})")

                offsets_px = offsets_px[good_velocity]
                v_means_um_s = v_means_um_s[good_velocity]
                sigma_v_mean_um_s = sigma_v_mean_um_s[good_velocity]
                sigma_v_rms_um_s = sigma_v_rms_um_s[good_velocity]
                coherences = coherences[good_velocity]
                v_hat_um_s = v_hat_um_s[good_velocity]

    if len(offsets_px) < 3:
        return empty_result

    # -------------------------------------------------------------------------
    # Poiseuille envelope filter: reject bands where |v̄| significantly
    # exceeds what the inner bands predict via a Poiseuille extrapolation.
    #
    # Physics: v(r) = v_max × (1 - (r/R)²) can only decrease from center.
    # We fit a quick 1-parameter Poiseuille (v_max only) to the inner bands
    # and use it as an upper envelope. Bands exceeding it by >k×σ are rejected.
    # This is intentionally soft — noise-driven non-monotonicity is tolerated
    # (especially in flat profiles), but clear violations are caught.
    # -------------------------------------------------------------------------
    inner_radius_px = ENVELOPE_INNER_FRACTION * vessel_radius_px
    inner_mask = np.abs(offsets_px) <= inner_radius_px
    _envelope_info = None  # Store for plotting

    if np.sum(inner_mask) >= 3 and ENVELOPE_SIGMA_TOLERANCE > 0:
        r_inner = offsets_px[inner_mask]
        v_inner = v_means_um_s[inner_mask]
        sigma_inner = sigma_v_mean_um_s[inner_mask]
        w_inner = 1.0 / sigma_inner ** 2

        # 2-parameter WLS: v_max + r₀, R = R_seg
        # Grid search over r₀ to accommodate vessel centers offset from skeleton.
        # For each candidate r₀, v_max is solved analytically via WLS.
        r0_max = min(inner_radius_px, vessel_radius_px * 0.4)
        r0_step = max(0.5, r0_max / 10)
        r0_candidates = np.arange(-r0_max, r0_max + r0_step * 0.5, r0_step)
        if 0.0 not in r0_candidates:
            r0_candidates = np.sort(np.append(r0_candidates, 0.0))

        best_r0 = 0.0
        best_vmax = 0.0
        best_wss = np.inf

        for r0_try in r0_candidates:
            shape_try = np.maximum(
                0, 1 - ((r_inner - r0_try) / vessel_radius_px) ** 2
            )
            denom_try = np.sum(w_inner * shape_try ** 2)
            if denom_try > 0:
                vmax_try = np.sum(w_inner * v_inner * shape_try) / denom_try
                wss = np.sum(w_inner * (v_inner - vmax_try * shape_try) ** 2)
                if wss < best_wss:
                    best_wss = wss
                    best_r0 = r0_try
                    best_vmax = vmax_try

        v_max_envelope = best_vmax
        r0_envelope = best_r0

        if v_max_envelope != 0:
            # Envelope at all band positions using (possibly shifted) center
            shape_all = np.maximum(
                0, 1 - ((offsets_px - r0_envelope) / vessel_radius_px) ** 2
            )
            v_envelope = v_max_envelope * shape_all

            # Store for plotting (before any bands are removed)
            _envelope_info = {
                'v_max': v_max_envelope,
                'R_px': vessel_radius_px,
                'r_offset_px': r0_envelope,
            }

            # Excess: how much |v̄| exceeds |envelope| (negative = within envelope)
            excess = np.abs(v_means_um_s) - np.abs(v_envelope)

            # Tolerance = k × min(σ_inner_med, |v_max_envelope|)
            sigma_ref = np.median(sigma_inner)
            tolerance = ENVELOPE_SIGMA_TOLERANCE * min(sigma_ref, abs(v_max_envelope))
            _envelope_info['tolerance_um_s'] = tolerance

            within_envelope = excess <= tolerance
            n_envelope_rejected = np.sum(~within_envelope)

            scale_label = "σ_med" if sigma_ref <= abs(v_max_envelope) else "|v_max|"
            r0_label = f", r₀={r0_envelope:+.1f}px" if abs(r0_envelope) > 0.1 else ""
            if verbose:
                print(f"  [ENVELOPE] Inner-band Poiseuille envelope "
                      f"(v_max={v_max_envelope:.1f} um/s{r0_label} from {np.sum(inner_mask)} inner bands, "
                      f"σ_med={sigma_ref:.0f}, tol={ENVELOPE_SIGMA_TOLERANCE}×{scale_label}={tolerance:.0f} um/s)")

            if n_envelope_rejected > 0:
                if verbose:
                    for idx in np.where(~within_envelope)[0]:
                        print(f"    REJECT offset {offsets_px[idx]:+.0f}px: "
                              f"|v̄|={np.abs(v_means_um_s[idx]):.1f}, "
                              f"envelope={np.abs(v_envelope[idx]):.1f}, "
                              f"excess={excess[idx]:.0f} > {tolerance:.0f}")

                offsets_px = offsets_px[within_envelope]
                v_means_um_s = v_means_um_s[within_envelope]
                sigma_v_mean_um_s = sigma_v_mean_um_s[within_envelope]
                sigma_v_rms_um_s = sigma_v_rms_um_s[within_envelope]
                coherences = coherences[within_envelope]
                v_hat_um_s = v_hat_um_s[within_envelope]
            elif verbose:
                print(f"    All {len(offsets_px)} bands within envelope")

    if len(offsets_px) < 3:
        return empty_result

    # -------------------------------------------------------------------------
    # Compute spectral purity weights
    # -------------------------------------------------------------------------
    if f0_hz is not None and np.isfinite(f0_hz) and f0_hz > 0:
        spectral_purity_weights = compute_spectral_purity_weights(
            v_hat_um_s, f0_hz, frame_dt, bandwidth_hz=0.5
        )
    else:
        spectral_purity_weights = None

    # -------------------------------------------------------------------------
    # Fit Poiseuille profile with NLLS and covariance estimation
    # -------------------------------------------------------------------------
    fit_result = _fit_poiseuille_profile_nlls(
        offsets_px, v_means_um_s, sigma_v_mean_um_s,
        vessel_radius_px, px_size_um,
        spectral_purity_weights=spectral_purity_weights,
        max_radius_px=max_radius_px,
        z_c_um=z_c_um, z_0_um=z_0_um,
        doc_model=doc_model,
        verbose=verbose,
    )

    v_max = fit_result['v_max']
    R_fit_um = fit_result['R_um']
    R_fit_px = fit_result['R_px']
    r_offset_px = fit_result['r_offset_px']
    chi2_reduced = fit_result['chi2_reduced']
    cov_theta = fit_result['cov_theta']
    param_std = fit_result['param_std']
    fit_success = fit_result['success']

    # -------------------------------------------------------------------------
    # Iterative filter: remove points outside vessel based on fitted r_offset
    # -------------------------------------------------------------------------
    # After initial fit, we know r_offset. Points with |r - r_offset| > R_fit
    # are outside the inferred vessel and should be excluded, then re-fit.
    #
    # IMPORTANT: Only do this if the initial fit is good (χ² < 2).
    # If the initial fit is bad (corrupted by outliers), iterative filtering
    # would remove good data based on a bad estimate, making things worse.
    CHI2_THRESHOLD_FOR_REFILTER = 2.0
    do_iterative_refilter = (
        fit_success and
        np.isfinite(r_offset_px) and
        np.isfinite(R_fit_px) and
        np.isfinite(chi2_reduced) and
        chi2_reduced < CHI2_THRESHOLD_FOR_REFILTER
    )

    if do_iterative_refilter:
        dist_from_center = np.abs(offsets_px - r_offset_px)
        inside_vessel = dist_from_center < R_fit_px
        n_outside = np.sum(~inside_vessel)

        if n_outside > 0 and np.sum(inside_vessel) >= 3:
            if verbose:
                print(f"  [FILTER] Removing {n_outside} bands outside inferred vessel "
                      f"(|r - {r_offset_px:.1f}| > {R_fit_px:.1f} px)")
            offsets_px = offsets_px[inside_vessel]
            v_means_um_s = v_means_um_s[inside_vessel]
            sigma_v_mean_um_s = sigma_v_mean_um_s[inside_vessel]
            sigma_v_rms_um_s = sigma_v_rms_um_s[inside_vessel]
            coherences = coherences[inside_vessel]
            v_hat_um_s = v_hat_um_s[inside_vessel]

            # Also filter spectral purity weights
            if spectral_purity_weights is not None:
                spectral_purity_weights = spectral_purity_weights[inside_vessel]

            # Re-fit with filtered data
            fit_result = _fit_poiseuille_profile_nlls(
                offsets_px, v_means_um_s, sigma_v_mean_um_s,
                vessel_radius_px, px_size_um,
                spectral_purity_weights=spectral_purity_weights,
                max_radius_px=max_radius_px,
                z_c_um=z_c_um, z_0_um=z_0_um,
                doc_model=doc_model,
                verbose=verbose,
            )
            v_max = fit_result['v_max']
            R_fit_um = fit_result['R_um']
            R_fit_px = fit_result['R_px']
            r_offset_px = fit_result['r_offset_px']
            chi2_reduced = fit_result['chi2_reduced']
            cov_theta = fit_result['cov_theta']
            param_std = fit_result['param_std']
            fit_success = fit_result['success']

            if verbose:
                print(f"  [RE-FIT] After filtering: n_bands={len(offsets_px)}, "
                      f"R={R_fit_px:.1f} px, r_offset={r_offset_px:.1f} px")
    elif not do_iterative_refilter and fit_success and chi2_reduced >= CHI2_THRESHOLD_FOR_REFILTER:
        if verbose:
            print(f"  [SKIP REFILTER] χ²_r={chi2_reduced:.2f} >= {CHI2_THRESHOLD_FOR_REFILTER}, "
                  f"not re-filtering (initial fit may be bad)")

    # -------------------------------------------------------------------------
    # Inflate covariance: χ² adjustment + radius floor
    # -------------------------------------------------------------------------
    # Uncertainty adjustment:
    # Floor: Even with perfect data, there's ~1 px irreducible uncertainty
    # from segmentation, optical resolution, etc.
    # Note: χ² inflation removed - let the Bayesian inference handle model misfit

    sigma_R_floor_um = 1.0 * px_size_um  # 1 pixel baseline uncertainty
    sigma_R_fit_um = param_std[1] if len(param_std) > 1 else 0.0

    # Step 1: Apply floor
    sigma_R_with_floor = max(sigma_R_fit_um, sigma_R_floor_um)
    floor_factor = sigma_R_with_floor / sigma_R_fit_um if sigma_R_fit_um > 0 else 1.0

    # Inflate R variance and covariances in the covariance matrix
    # cov_theta indices: [0]=v_max, [1]=R, [2]=r_offset
    cov_theta_inflated = cov_theta.copy()
    if np.isfinite(cov_theta_inflated).all():
        # Inflate variance: σ_R² → σ_R² × factor²
        cov_theta_inflated[1, 1] *= floor_factor ** 2
        # Inflate covariances: Cov(R, X) → Cov(R, X) × factor
        for i in [0, 2]:
            cov_theta_inflated[1, i] *= floor_factor
            cov_theta_inflated[i, 1] *= floor_factor

    # Update param_std with inflated R uncertainty
    param_std_inflated = param_std.copy()
    param_std_inflated[1] *= floor_factor

    # -------------------------------------------------------------------------
    # Propagate uncertainty to Q (using inflated covariance)
    # -------------------------------------------------------------------------
    Q_mean, sigma_Q = _propagate_uncertainty_to_Q(v_max, R_fit_um, cov_theta_inflated)

    # -------------------------------------------------------------------------
    # Apply spatial coverage penalty for short vessels
    # -------------------------------------------------------------------------
    # Short vessels have limited spatial averaging along the vessel length.
    # Even with good temporal measurements, they're less representative of
    # the true vessel flow profile. Apply a penalty based on vessel length.
    n_spatial_points = len(profile_data[0]['kymo'][0]) if len(profile_data) > 0 else 20
    MIN_REFERENCE_LENGTH = 20  # pixels - typical "good" vessel length

    if n_spatial_points < MIN_REFERENCE_LENGTH:
        spatial_penalty = np.sqrt(MIN_REFERENCE_LENGTH / n_spatial_points)
        sigma_Q *= spatial_penalty
    else:
        spatial_penalty = 1.0

    # -------------------------------------------------------------------------
    # Compute Q(t) time series using Poiseuille profile shape
    # -------------------------------------------------------------------------
    f_r = _poiseuille_shape(
        offsets_px * px_size_um, R_fit_um, r_offset_px * px_size_um,
        z_c_um, px_size_um, z_0_um,
    )

    # Skip bands too close to wall
    valid_bands = f_r >= 0.1
    if not np.any(valid_bands):
        valid_bands = np.ones(len(f_r), dtype=bool)

    # WLS weights from cycle-based uncertainty
    weights = 1.0 / (sigma_v_mean_um_s**2)
    weights[~np.isfinite(weights)] = 0.0
    weights[~valid_bands] = 0.0
    if np.all(weights == 0):
        weights = np.ones(len(weights))
        weights[~valid_bands] = 0.0

    # v_max(t) = Σ(w × v(t) × f(r)) / Σ(w × f(r)²)
    wf = weights * f_r
    denom = np.sum(weights * f_r**2)

    if denom < 1e-10:
        Q_nL_s = np.full(T, np.nan)
        v_max_t = np.full(T, np.nan)
    else:
        numerator = np.sum(wf[:, np.newaxis] * v_hat_um_s, axis=0)
        v_max_t = numerator / denom

        # Q(t) = v_max(t) × πR² × n/(n+2) / 1e6  [nL/s], with n=2 → factor = 0.5
        Q_factor = np.pi * R_fit_um**2 * 0.5 / 1e6
        Q_nL_s = v_max_t * Q_factor

    # -------------------------------------------------------------------------
    # Compute legacy uncertainty estimates for compatibility
    # -------------------------------------------------------------------------
    N_eff = float(n_cycles)
    Q_mean_from_ts = float(np.nanmean(Q_nL_s))

    # Random uncertainty from Q(t) residuals (legacy method)
    if f0_hz is not None and np.isfinite(f0_hz):
        Q_hr = fit_harmonics(Q_nL_s, frame_dt, f0_hz, K=N_HARMONICS,
                            loss="huber", include_dc=True)
        Q_resid = Q_hr.get('resid', Q_nL_s - Q_mean_from_ts)
        sigma_Q_per_frame = float(np.nanstd(Q_resid))
    else:
        sigma_Q_per_frame = float(np.nanstd(Q_nL_s))

    sigma_Q_random = sigma_Q_per_frame / np.sqrt(max(N_eff, 1.0))
    sigma_Q_systematic = abs(Q_mean_from_ts) * (2.0 * sigma_R_px * px_size_um / R_fit_um) if R_fit_um > 0 else np.nan
    sigma_Q_total_legacy = np.sqrt(sigma_Q_random**2 + sigma_Q_systematic**2)

    # -------------------------------------------------------------------------
    # Alternative estimate: Inner-bands method
    # Use only centerline and nearby bands to estimate v_max directly,
    # then compute Q using inferred R from signal bounds.
    # This is more robust to junction effects that corrupt outer bands.
    # Falls back to NLLS fit values if inferred values not provided.
    # -------------------------------------------------------------------------
    R_for_inner = R_inferred_px if (R_inferred_px is not None and np.isfinite(R_inferred_px)) else R_fit_px
    r_offset_for_inner = r_offset_inferred_px if (r_offset_inferred_px is not None and np.isfinite(r_offset_inferred_px)) else r_offset_px

    inner_result = _compute_inner_bands_estimate(
        offsets_px, v_hat_um_s, sigma_v_mean_um_s,
        R_for_inner, r_offset_for_inner, px_size_um, frame_dt, f0_hz,
        verbose=verbose,
    )

    return {
        'Q_t': Q_nL_s,
        'Q_mean': Q_mean,
        'sigma_Q': sigma_Q,
        'v_max': v_max,
        'R_fit_px': R_fit_px,
        'R_fit_um': R_fit_um,
        'r_offset_px': r_offset_px,
        'chi2_reduced': chi2_reduced,
        'cov_theta': cov_theta_inflated,
        'param_std': param_std_inflated,
        'sigma_R_inflation': floor_factor,
        'sigma_R_floor_um': sigma_R_floor_um,
        'n_cycles': n_cycles,
        'sigma_v_mean': sigma_v_mean_um_s,
        'offsets_fit_px': offsets_px,
        'v_means_fit': v_means_um_s,
        'fit_success': fit_success,
        'envelope_info': _envelope_info,  # For plotting envelope on profile
        # Inner-bands alternative estimate
        'Q_t_inner': inner_result['Q_t_inner'],
        'Q_mean_inner': inner_result['Q_mean_inner'],
        'v_max_inner': inner_result['v_max_inner'],
        'chi2_inner': inner_result['chi2_inner'],
        'n_inner_bands': inner_result['n_inner_bands'],
        'inner_offsets_px': inner_result['inner_offsets_px'],
        'velocity_center_px': inner_result.get('velocity_center_px', np.nan),
        # Legacy fields for compatibility
        'v_max_mean': float(np.nanmean(v_max_t)) if len(v_max_t) > 0 else np.nan,
        'r_offset': r_offset_px,
        'sigma_noise': sigma_v_mean_um_s,
        'sigma_Q_random': sigma_Q_random,
        'sigma_Q_systematic': sigma_Q_systematic,
        'sigma_Q_total': sigma_Q_total_legacy,
        'N_eff': N_eff,
    }


def analyze_vessel(
    coords: np.ndarray,
    stack: np.ndarray,
    vessel_radius_px: float,
    frame_dt: float = FRAME_DT_S,
    px_size_um: float = PX_SIZE_UM,
    *,
    consensus_f0: Optional[float] = None,
    fmin_hz: float = FMIN_HZ,
    fmax_hz: float = FMAX_HZ,
    centerline_only: bool = False,
    extend_radius_px: float = 0.0,
    gst_windows: Optional[List[int]] = None,
    uncertainty_method: str = "harmonic_residual",
    use_coherence_gating: bool = True,
    z_c_um: float = DOC_Z_C_UM,
    z_0_um: float = 0.0,
    doc_model: str = DOC_MODEL,
    cached_centerline_kymo: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze blood flow in a vessel and return summary metrics.

    Computes Q(t) by integrating radial velocity profiles, then performs
    harmonic regression to extract mean_Q, amp_Q, PI, and phase.

    Args:
        coords: (N, 2) array of [x, y] centerline coordinates
        stack: (T, H, W) video frames
        vessel_radius_px: Vessel radius in pixels (from segmentation)
        frame_dt: Time between frames in seconds
        px_size_um: Pixel size in micrometers (for unit conversion)
        consensus_f0: Pre-computed heart rate (Hz), or None to estimate
        fmin_hz: Lower frequency bound for f0 estimation
        fmax_hz: Upper frequency bound for f0 estimation
        centerline_only: If True and consensus_f0 is None, only compute centerline
            for fast f0 estimation (used in PASS 1 of consensus workflow)
        extend_radius_px: Pixels to extend sampling beyond segmented radius.
            Use >0 (e.g., 5) to sample beyond the original radius and
            auto-detect the true vessel boundary from noise characteristics.
        gst_windows: Window sizes for GST computation. If None, uses config default.
            Use fewer windows (e.g., [11]) in PASS 2 for speed since f0 is known.
        uncertainty_method: Method for computing velocity uncertainty per band.
            - "harmonic_residual": Use RMS of harmonic fit residuals / sqrt(N_cycles).
              Cleaner separation of signal (cardiac) vs noise (measurement).
              Typically gives χ²_red closer to 1.0.
            - "cycle_based": Use cycle-to-cycle variability of mean velocity.
              Conflates physiological variation with measurement noise.
        use_coherence_gating: Enable per-pixel coherence gating and non-contiguous
            column masking. Helps with short/noisy vessels. Set False to revert to
            original unweighted median behavior.
        verbose: Print progress messages

    Returns:
        Dict with analysis results:
            - mean_Q: Mean volumetric flow (nL/s)
            - amp_Q: Pulsatile amplitude (nL/s)
            - PI: Pulsatility index (2 * amp / |mean|)
            - phase: Phase of fundamental harmonic (radians)
            - f0_hz: Heart rate in Hz
            - snr_db: Signal quality (SNR in dB)
            - Q_t: Full Q(t) time series
            - profile_data: Full radial profile data
            - radius_px: Effective radius used for fitting (may differ from input)
            - original_radius_px: Original segmented radius (input value)
            - success: True if analysis succeeded
    """
    try:
        # Store original radius from segmentation
        original_radius_px = vessel_radius_px

        # Run radial profile analysis (with extended sampling if requested)
        coh_gate = COHERENCE_GATE_THRESHOLD if use_coherence_gating else 0.0
        col_coh_min = COLUMN_COHERENCE_MIN if use_coherence_gating else 0.0

        profile_data, f0 = compute_radial_velocity_profiles(
            coords,
            stack,
            vessel_radius_px,
            frame_dt,
            consensus_f0=consensus_f0,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            centerline_only=centerline_only,
            extend_radius_px=extend_radius_px,
            gst_windows=gst_windows,
            coherence_gate=coh_gate,
            column_coherence_min=col_coh_min,
            cached_centerline_kymo=cached_centerline_kymo,
            verbose=verbose,
        )

        # Check if profile analysis succeeded (may fail for too-short vessels)
        if len(profile_data) == 0:
            raise ValueError("Arc too short for velocity analysis")

        # PASS 1 early return: if centerline_only, just return f0 estimate
        # No need to compute Q(t) or do harmonic regression
        if centerline_only and consensus_f0 is None:
            return {
                'mean_Q': np.nan,
                'sigma_mean_Q': np.nan,
                'amp_Q': np.nan,
                'sigma_amp_Q': np.nan,
                'PI': np.nan,
                'sigma_PI': np.nan,
                'phase': np.nan,
                'sigma_phase': np.nan,
                'f0_hz': f0,  # The estimated f0 from centerline
                'snr_db': np.nan,
                'Q_t': np.array([]),
                'profile_data': profile_data,
                'v_max': np.nan,
                'r_offset': np.nan,
                'chi2_reduced': np.nan,
                'radius_px': original_radius_px,
                'original_radius_px': original_radius_px,
                'R_fit_px': np.nan,
                'R_fit_um': np.nan,
                'sigma_Q': np.nan,
                'cov_theta': np.full((4, 4), np.nan),
                'param_std': np.full(4, np.nan),
                'n_cycles': 0,
                'fit_success': False,
                'Q_t_inner': np.array([]),
                'Q_mean_inner': np.nan,
                'v_max_inner': np.nan,
                'chi2_inner': np.nan,
                'n_inner_bands': 0,
                'inner_offsets_px': np.array([]),
                'R_inferred_px': np.nan,
                'r_offset_inferred_px': np.nan,
                'velocity_center_px': np.nan,
                'sigma_noise': np.array([]),
                'sigma_Q_random': np.nan,
                'sigma_Q_systematic': np.nan,
                'sigma_Q_total': np.nan,
                'N_eff': np.nan,
                'success': True,  # f0 estimation succeeded
            }

        # Simplified radius handling: skip effective radius estimation, constrain NLLS bounds
        # Instead of estimating effective radius from noise transition, we:
        # 1. Use original_radius_px as the vessel radius for fitting
        # 2. Constrain NLLS bounds to [R_seg - 1, R_seg + extend_radius_px]
        # This prevents runaway radius inflation while still allowing some flexibility
        if extend_radius_px > 0:
            max_radius_px = original_radius_px + extend_radius_px
            if verbose:
                print(f"  Radius: {original_radius_px:.1f} px (segmented), "
                      f"NLLS bounds: [{original_radius_px - 1:.1f}, {max_radius_px:.1f}] px", flush=True)
        else:
            max_radius_px = None  # Use legacy ±50% bounds

        if verbose:
            print(f"  Radial samples: {len(profile_data)}", flush=True)

        # Compute Q(t) using Poiseuille profile weighted least squares
        # Use original radius for fitting, let NLLS adjust within bounds
        Q_result = _compute_Q_from_profiles(
            profile_data, original_radius_px, px_size_um, frame_dt,
            R_inferred_px=original_radius_px, r_offset_inferred_px=0.0,
            max_radius_px=max_radius_px,
            uncertainty_method=uncertainty_method,
            z_c_um=z_c_um, z_0_um=z_0_um,
            doc_model=doc_model,
            verbose=verbose,
        )

        Q_t = Q_result['Q_t']
        v_max = Q_result['v_max']  # From profile fit (grid search), not time-averaged WLS
        r_offset = Q_result['r_offset']
        chi2_reduced = Q_result['chi2_reduced']
        sigma_noise = Q_result['sigma_noise']
        sigma_Q_random = Q_result['sigma_Q_random']
        sigma_Q_systematic = Q_result['sigma_Q_systematic']
        sigma_Q_total = Q_result['sigma_Q_total']
        N_eff = Q_result['N_eff']
        # Fields from cycle-based uncertainty
        sigma_Q = Q_result.get('sigma_Q', np.nan)  # Primary uncertainty from covariance
        R_fit_px = Q_result.get('R_fit_px', original_radius_px)
        R_fit_um = Q_result.get('R_fit_um', original_radius_px * px_size_um)
        cov_theta = Q_result.get('cov_theta', np.full((3, 3), np.nan))
        param_std = Q_result.get('param_std', np.full(3, np.nan))
        n_cycles = Q_result.get('n_cycles', 0)
        fit_success = Q_result.get('fit_success', False)
        # Per-band cycle-based uncertainty for plotting
        sigma_v_mean = Q_result.get('sigma_v_mean', np.array([]))
        offsets_fit_px = Q_result.get('offsets_fit_px', np.array([]))
        v_means_fit = Q_result.get('v_means_fit', np.array([]))
        # Inner bands alternative estimate
        Q_t_inner = Q_result.get('Q_t_inner', np.array([]))
        Q_mean_inner = Q_result.get('Q_mean_inner', np.nan)
        v_max_inner = Q_result.get('v_max_inner', np.nan)
        chi2_inner = Q_result.get('chi2_inner', np.nan)
        n_inner_bands = Q_result.get('n_inner_bands', 0)
        inner_offsets_px = Q_result.get('inner_offsets_px', np.array([]))
        velocity_center_px = Q_result.get('velocity_center_px', np.nan)

        if len(Q_t) == 0 or not np.isfinite(Q_t).any():
            raise ValueError("Failed to compute Q(t) from Poiseuille profiles")

        # Harmonic regression on Q(t) to extract mean, amplitude, phase
        hr_result = fit_harmonics(
            Q_t,
            frame_dt,
            f0,
            K=N_HARMONICS,
            loss="huber",
            include_dc=True,
        )

        # DC component: use simple nanmean for transparency and consistency.
        # The robust harmonic a0 can diverge from nanmean for low-SNR vessels
        # (Huber fit downweights many frames), even when Q_t has no NaN values.
        # AC components (amplitude, phase) still come from the harmonic fit.
        mean_Q = float(np.nanmean(Q_t))

        # Extract fundamental harmonic amplitude, phase, and their uncertainties
        harmonics = hr_result.get('harmonics', [])
        amp_Q = np.nan
        phase = np.nan
        sigma_amp_Q_harmonic = np.nan  # From harmonic fit only
        sigma_phase = np.nan
        for h in harmonics:
            if h.get('k') == 1:
                A = h.get('A', 0)
                B = h.get('B', 0)
                amp_Q = np.sqrt(A**2 + B**2)
                phase = np.arctan2(-B, A)  # Phase convention
                sigma_amp_Q_harmonic = h.get('sigma_amp', np.nan)
                sigma_phase = h.get('sigma_phi', np.nan)
                break

        # =====================================================================
        # σ_Q_mean: from NLLS profile fit covariance (propagated σ_v_max)
        # =====================================================================
        # Q = π R² v_max / 2  ⟹  σ_Q_mean = (π R² / 2) × σ_v_max
        # This does NOT diverge for steady-flow vessels (Q_amp → 0).
        sigma_v_max = param_std[0] if len(param_std) > 0 else np.nan  # µm/s
        R_um = R_fit_um

        if np.isfinite(sigma_v_max) and sigma_v_max > 0 and np.isfinite(R_um) and R_um > 0:
            sigma_Q_mean = (np.pi * R_um**2 / 2.0) * sigma_v_max / 1e6  # nL/s
        else:
            # Fallback: 10% of |Q̄|
            sigma_Q_mean = 0.10 * abs(mean_Q) if abs(mean_Q) > 1e-12 else np.nan

        # σ_Q_amp: from harmonic fit covariance of Q(t)
        sigma_Q_amp = sigma_amp_Q_harmonic  # Already computed from k=1 harmonic

        # For compatibility: primary uncertainty = σ_Q_mean
        sigma_Q = sigma_Q_mean
        sigma_mean_Q = sigma_Q_mean

        rel_unc = sigma_Q_mean / abs(mean_Q) if abs(mean_Q) > 1e-12 else np.nan
        if verbose:
            print(f"  [σ_Q from NLLS] σ_v_max={sigma_v_max:.2f} µm/s, R={R_um:.1f} µm")
            print(f"    σ_Q_mean = {sigma_Q_mean:.3f} nL/s ({100*rel_unc:.1f}% of |Q̄|)")
            if np.isfinite(sigma_Q_amp):
                print(f"    σ_Q_amp  = {sigma_Q_amp:.3f} nL/s")

        # Amplitude uncertainty: from harmonic fit
        sigma_amp_Q = sigma_Q_amp if np.isfinite(sigma_Q_amp) else sigma_Q_mean

        # Compute pulsatility index: PI = 2 * amplitude / |mean|
        # σ_PI = 2 * sqrt((σ_amp/|mean|)² + (amp*σ_mean/mean²)²)
        if np.isfinite(amp_Q) and np.isfinite(mean_Q) and abs(mean_Q) > 1e-12:
            PI = 2 * amp_Q / abs(mean_Q)
            # Propagate uncertainty to PI using total uncertainties
            if np.isfinite(sigma_amp_Q) and np.isfinite(sigma_mean_Q):
                sigma_PI = 2 * np.sqrt((sigma_amp_Q / abs(mean_Q))**2 +
                                        (amp_Q * sigma_mean_Q / mean_Q**2)**2)
            else:
                sigma_PI = np.nan
        else:
            PI = np.nan
            sigma_PI = np.nan

        snr_db = hr_result.get('hr_snr_db', np.nan)

        # -----------------------------------------------------------------
        # Two-tier vessel quality classification
        # -----------------------------------------------------------------
        # snr_pulse: pulsatile SNR = amplitude / uncertainty in amplitude
        snr_pulse = amp_Q / sigma_amp_Q if (np.isfinite(amp_Q) and np.isfinite(sigma_amp_Q) and sigma_amp_Q > 0) else 0.0

        # Mean coherence across all bands used in the fit
        _coh_vals = [p.get('mean_coherence', 0.0) for p in profile_data if p.get('v_hat') is not None]
        mean_coherence_vessel = float(np.mean(_coh_vals)) if _coh_vals else 0.0

        # Tier A: pulsatile — full metrics (amplitude, phase, PI) are reliable
        #   snr_pulse >= 3.0 AND fit converged
        # Tier B: good mean flow — pulsatility unresolved but mean_Q is reliable
        #   mean_coherence >= 0.5 AND mean_Q is finite AND fit converged
        # Excluded: bad data — neither condition met
        if snr_pulse >= 3.0 and fit_success:
            quality_tier = 'A'
        elif mean_coherence_vessel >= 0.5 and np.isfinite(mean_Q) and fit_success:
            quality_tier = 'B'
        else:
            quality_tier = 'X'  # Excluded

        return {
            'mean_Q': float(mean_Q),
            'sigma_mean_Q': float(sigma_mean_Q),  # Total uncertainty in mean (random + systematic)
            'amp_Q': float(amp_Q),
            'sigma_amp_Q': float(sigma_amp_Q),  # Total uncertainty in amplitude (random + systematic)
            'PI': float(PI),
            'sigma_PI': float(sigma_PI),
            'phase': float(phase),
            'sigma_phase': float(sigma_phase),
            'f0_hz': f0,
            'snr_db': snr_db,
            # Quality classification
            'quality_tier': quality_tier,  # 'A' (pulsatile), 'B' (mean flow only), 'X' (excluded)
            'snr_pulse': float(snr_pulse),  # amp_Q / sigma_amp_Q
            'mean_coherence_vessel': float(mean_coherence_vessel),
            'Q_t': Q_t,
            'harmonics': harmonics,  # All harmonic coefficients (k, A, B, amp, phi)
            'profile_data': profile_data,
            'v_max': v_max,  # Mean centerline velocity from WLS (um/s)
            'r_offset': r_offset,  # Centerline offset from fit (pixels)
            'chi2_reduced': chi2_reduced,  # Reduced chi-squared (~1.0 = good fit)
            # Radius estimates
            'radius_px': original_radius_px,  # Segmented radius used for fitting (pixels)
            'original_radius_px': original_radius_px,  # Original segmented radius (pixels)
            'R_fit_px': R_fit_px,  # Fitted radius from profile (pixels)
            'R_fit_um': R_fit_um,  # Fitted radius from profile (um)
            # Uncertainty estimates
            'sigma_Q': sigma_Q,  # Primary uncertainty = σ_Q_mean (nL/s)
            'sigma_Q_mean': float(sigma_Q_mean),  # From NLLS covariance: (πR²/2)×σ_v_max
            'sigma_Q_amp': float(sigma_Q_amp) if np.isfinite(sigma_Q_amp) else np.nan,  # From Q(t) harmonic fit
            'cov_theta': cov_theta,  # 3x3 parameter covariance matrix
            'param_std': param_std,  # Parameter std devs: (σ_vmax, σ_R, σ_r0)
            'n_cycles': n_cycles,  # Number of heartbeat cycles used
            'fit_success': fit_success,  # Whether NLLS optimization converged
            # Per-band data for plotting with cycle-based uncertainty
            'sigma_v_mean': sigma_v_mean,  # Per-band cycle-based uncertainty (um/s)
            'offsets_fit_px': offsets_fit_px,  # Radial offsets used in fit (px)
            'v_means_fit': v_means_fit,  # Time-averaged velocities used in fit (um/s)
            'envelope_info': Q_result.get('envelope_info', None),  # Envelope filter params for plotting
            # Inner bands alternative estimate
            'Q_t_inner': Q_t_inner,  # Q(t) from inner bands only (nL/s)
            'Q_mean_inner': Q_mean_inner,  # Mean Q from inner bands (nL/s)
            'v_max_inner': v_max_inner,  # v_max from inner bands (um/s)
            'chi2_inner': chi2_inner,  # Reduced chi² for inner bands
            'n_inner_bands': n_inner_bands,  # Number of inner bands used
            'inner_offsets_px': inner_offsets_px,  # Offsets of inner bands (px)
            'R_inferred_px': original_radius_px,  # Legacy: was from signal bounds, now same as original
            'r_offset_inferred_px': 0.0,  # Legacy: was from signal bounds, now unused
            'velocity_center_px': velocity_center_px,  # Centerline from max velocity band (px)
            # Legacy uncertainty estimates (for compatibility)
            'sigma_noise': sigma_noise,  # Per-band measurement noise (um/s)
            'sigma_Q_random': sigma_Q_random,  # Random uncertainty in mean Q (nL/s)
            'sigma_Q_systematic': sigma_Q_systematic,  # Systematic uncertainty from radius (nL/s)
            'sigma_Q_total': sigma_Q_total,  # Total uncertainty (nL/s)
            'N_eff': N_eff,  # Effective number of independent measurements (heartbeats)
            'success': True,
        }

    except Exception as e:
        if verbose:
            print(f"Analysis failed: {e}")
        return {
            'mean_Q': np.nan,
            'sigma_mean_Q': np.nan,
            'amp_Q': np.nan,
            'sigma_amp_Q': np.nan,
            'PI': np.nan,
            'sigma_PI': np.nan,
            'phase': np.nan,
            'sigma_phase': np.nan,
            'f0_hz': np.nan,
            'snr_db': np.nan,
            'quality_tier': 'X',
            'snr_pulse': 0.0,
            'mean_coherence_vessel': 0.0,
            'Q_t': np.array([]),
            'profile_data': [],
            'v_max': np.nan,
            'r_offset': np.nan,
            'chi2_reduced': np.nan,
            'radius_px': np.nan,
            'original_radius_px': vessel_radius_px,  # Keep original even on failure
            'R_fit_px': np.nan,
            'R_fit_um': np.nan,
            # New uncertainty fields
            'sigma_Q': np.nan,
            'sigma_Q_mean': np.nan,
            'sigma_Q_amp': np.nan,
            'cov_theta': np.full((4, 4), np.nan),
            'param_std': np.full(4, np.nan),
            'n_cycles': 0,
            'fit_success': False,
            # Inner bands fields
            'Q_t_inner': np.array([]),
            'Q_mean_inner': np.nan,
            'v_max_inner': np.nan,
            'chi2_inner': np.nan,
            'n_inner_bands': 0,
            'inner_offsets_px': np.array([]),
            'R_inferred_px': np.nan,
            'r_offset_inferred_px': np.nan,
            'velocity_center_px': np.nan,
            # Legacy fields
            'sigma_noise': np.array([]),
            'sigma_Q_random': np.nan,
            'sigma_Q_systematic': np.nan,
            'sigma_Q_total': np.nan,
            'N_eff': np.nan,
            'success': False,
            'error': str(e),
        }


def trace_vessel_chain(G: nx.Graph, u: int, v: int) -> List[int]:
    """
    Trace a vessel chain from edge (u, v) through degree-2 nodes.

    Walks in both directions from the starting edge until hitting:
    - Junction nodes (degree != 2)
    - Terminal nodes (degree == 1)
    - Boundary nodes (source/sink markers)

    Args:
        G: NetworkX graph
        u: First node of starting edge
        v: Second node of starting edge

    Returns:
        List of node IDs forming the vessel chain
    """
    def is_boundary(n: int) -> bool:
        """Check if node is marked as a boundary (source/sink)."""
        return G.nodes[n].get('boundary_type') is not None

    def walk_one(start: int, other: int) -> List[int]:
        """Walk from start away from other until hitting a junction or boundary."""
        path = [start]

        # If start is a boundary node, it's an endpoint - don't walk through it
        if is_boundary(start):
            return path

        curr, last = start, other
        seen = {start}
        while True:
            deg = G.degree[curr]
            nbrs = [w for w in G.neighbors(curr) if w != last]
            # Stop at junctions (degree != 2) or if no single neighbor
            if deg != 2 or len(nbrs) != 1:
                break
            nxt = nbrs[0]
            if nxt in seen:
                break
            # Stop if next node is a boundary (source/sink)
            if is_boundary(nxt):
                path.append(nxt)
                break
            seen.add(nxt)
            last, curr = curr, nxt
            path.append(curr)
        return path

    left = walk_one(u, v)
    right = walk_one(v, u)
    chain = list(reversed(left)) + right

    # Remove duplicates while preserving order
    cleaned = []
    for n in chain:
        if not cleaned or cleaned[-1] != n:
            cleaned.append(n)
    return cleaned


def get_chain_coords(
    G,
    chain: List[Tuple[int, int]],
    px_spacing: float = 1.0,
    margin_px: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Get resampled coordinates for a vessel chain with uniform spacing.

    Args:
        G: NetworkX graph with edge 'pts' attribute
        chain: List of (u, v) edges
        px_spacing: Desired spacing between points in pixels (default: 1.0)
        margin_px: Margin to trim at vessel ends (pixels). If None, uses MARGIN_PX
            from config. Use 0 for no trimming, or small values like 3 for
            maximum vessel length with minimal junction exclusion.

    Returns:
        coords: (N, 2) resampled centerline coordinates with uniform spacing
        mean_radius: Average vessel radius
    """
    all_pts = []
    radii = []

    # Collect coordinates and radii, excluding junction edges (first and last)
    for i, (u, v) in enumerate(chain):
        edge_data = G.edges[u, v]
        # Try 'pts' first (viewer format), then 'path' (collapse_to_vessel_graph format)
        pts = edge_data.get('pts', None)
        if pts is None:
            pts = edge_data.get('path', None)
        if pts is not None:
            all_pts.append(np.array(pts))

        # Exclude first and last edge radii (junction edges)
        if len(chain) > 2 and (i == 0 or i == len(chain) - 1):
            continue  # Skip junction edges for radius computation

        # Try 'radius' first, then 'conductivity' as fallback
        radius = edge_data.get('radius', None)
        if radius is None:
            radius = edge_data.get('conductivity', None)
        if radius is not None:
            radii.append(radius)

    if not all_pts:
        # Fall back to node positions for coordinates
        nodes = []
        for u, v in chain:
            if u not in nodes:
                nodes.append(u)
            if v not in nodes:
                nodes.append(v)
        XY_nodes = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nodes])

        # But still check for radii from edge attributes
        if radii:
            mean_radius = float(np.median(radii))
            print(f"  WARNING: No 'pts'/'path' attribute on edges, using node positions for coordinates")
        else:
            mean_radius = 5.0
            # Show what attributes are available on first edge
            if chain:
                sample_edge = G.edges[chain[0]]
                edge_attrs = list(sample_edge.keys())
                print(f"  WARNING: No 'pts'/'path' or 'radius'/'conductivity' attributes found on edges")
                print(f"  Available edge attributes: {edge_attrs}")
                print(f"  Defaulting radius to {mean_radius:.1f} px")
    else:
        # Concatenate points, avoiding duplicates at junctions
        coords_list = []
        for i, pts in enumerate(all_pts):
            if i == 0:
                coords_list.append(pts)
            else:
                # Skip first point if it's close to last point of previous segment
                if len(coords_list) > 0:
                    last_pt = coords_list[-1][-1]
                    if np.linalg.norm(pts[0] - last_pt) < 1.0:
                        coords_list.append(pts[1:])
                    else:
                        coords_list.append(pts)
                else:
                    coords_list.append(pts)
        XY_nodes = np.vstack(coords_list)

        # Use median instead of mean for robustness to outliers
        if radii:
            mean_radius = float(np.median(radii))
        else:
            mean_radius = 5.0
            # Show what attributes are available on first edge
            if chain:
                sample_edge = G.edges[chain[0]]
                edge_attrs = list(sample_edge.keys())
                print(f"  WARNING: No 'radius' or 'conductivity' attribute found on edges")
                print(f"  Available edge attributes: {edge_attrs}")
                print(f"  Defaulting radius to {mean_radius:.1f} px")

    # Resample curve with uniform spacing (like graph_kymo_editor.py)
    # Compute arc length at each node
    seg_len = np.hypot(np.diff(XY_nodes[:, 0]), np.diff(XY_nodes[:, 1]))
    s_nodes = np.concatenate(([0.0], np.cumsum(seg_len)))
    s_total = float(s_nodes[-1])

    # Check if chain is long enough for margins
    # For short vessels, use proportional margin (max 15% of length per side)
    # to leave enough usable arc after trimming
    default_margin = margin_px if margin_px is not None else MARGIN_PX
    max_margin_frac = 0.15  # Max 15% of vessel length per side
    proportional_margin = int(s_total * max_margin_frac)
    effective_margin = min(default_margin, proportional_margin)

    if s_total < 2 * effective_margin + 10:  # Need at least 10 px after trimming
        effective_margin = max(0, int((s_total - 10) / 2))

    # Create uniform sample points along arc length
    s_samp = np.arange(0.0, s_total + 1e-9, px_spacing)

    # Find which segment each sample falls in
    j = np.searchsorted(s_nodes, s_samp, side='right') - 1
    j = np.clip(j, 0, len(s_nodes) - 2)

    # Linear interpolation within each segment
    span = s_nodes[j + 1] - s_nodes[j]
    alpha = (s_samp - s_nodes[j]) / np.maximum(span, 1e-9)
    pts_full = (1.0 - alpha)[:, None] * XY_nodes[j] + alpha[:, None] * XY_nodes[j + 1]

    # Trim margins to exclude junction nodes
    if effective_margin > 0:
        keep = (s_samp >= effective_margin) & (s_samp <= s_total - effective_margin)
        pts_trimmed = pts_full[keep]
    else:
        pts_trimmed = pts_full

    # Smooth coordinates to reduce artifacts from irregular node spacing
    # Apply Gaussian filter with sigma=1.0 to reduce high-frequency noise
    try:
        from scipy.ndimage import gaussian_filter1d
        sigma = 1.0
        pts_smooth = np.column_stack([
            gaussian_filter1d(pts_trimmed[:, 0], sigma=sigma, mode='nearest'),
            gaussian_filter1d(pts_trimmed[:, 1], sigma=sigma, mode='nearest')
        ])
    except ImportError:
        pts_smooth = pts_trimmed

    # Canonicalize coordinate order so results don't depend on edge direction.
    # For undirected graphs, G.edges[u,v] and G.edges[v,u] return the same path
    # data, so we can't rely on chain node ordering. Instead, check which end of
    # the path is geometrically closer to the lower-numbered node.
    start_node = chain[0][0]
    end_node = chain[-1][1]
    lower_node = min(start_node, end_node)
    lower_pos = np.array([G.nodes[lower_node]['x'], G.nodes[lower_node]['y']])
    d_first = np.linalg.norm(pts_smooth[0] - lower_pos)
    d_last = np.linalg.norm(pts_smooth[-1] - lower_pos)
    was_reversed = False
    if d_last < d_first:
        pts_smooth = pts_smooth[::-1]
        was_reversed = True

    return pts_smooth, mean_radius, was_reversed
