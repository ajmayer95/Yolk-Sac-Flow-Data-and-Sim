"""
Flow analysis visualization plots.

Provides popup figures for visualizing:
1. Representative time series and kymographs at different radial positions
2. Profile fit, Q(t), and frequency spectrum
3. WLS regression snapshot at a particular time point
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from typing import List, Dict, Any, Optional, Tuple

from .config import PX_SIZE_UM, FRAME_DT_S
from .harmonic import fit_harmonics


def _ensure_interactive():
    """Ensure matplotlib is in interactive mode for popup figures."""
    # Try to use TkAgg backend if available (works well with napari)
    try:
        current_backend = matplotlib.get_backend()
        if 'inline' in current_backend.lower() or current_backend == 'agg':
            matplotlib.use('TkAgg')
    except Exception:
        pass
    plt.ion()  # Interactive mode


def _show_figure(fig, block=False):
    """Show a figure ensuring it actually displays."""
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)  # Give time for figure to render


def plot_radial_kymographs(
    profile_data: List[Dict[str, Any]],
    vessel_radius_px: float,
    offsets_display: Optional[np.ndarray] = None,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    block: bool = False,
) -> plt.Figure:
    """
    Figure 1: Kymographs and time series at representative radial positions.

    Row 1: Filtered kymographs showing spatial-temporal patterns
    Row 2: Velocity distribution heatmap with weighted median overlay

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    vessel_radius_px : float
        Vessel radius in pixels
    offsets_display : array, optional
        Specific offsets to display (default: -Rmax, -Rmax/2, 0, Rmax/2, Rmax)
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig : matplotlib.Figure
    """
    offsets_arr = np.array([p['offset'] for p in profile_data])

    # Default display offsets: -Rmax, -Rmax/2, 0, Rmax/2, Rmax (in order)
    if offsets_display is None:
        r50 = int(np.round(0.5 * vessel_radius_px))
        rmax = int(np.round(vessel_radius_px))
        offsets_display = np.array([-rmax, -r50, 0, r50, rmax])

    # Find indices matching offsets_display
    show_indices = []
    show_offsets = []
    for target in offsets_display:
        idx = np.argmin(np.abs(offsets_arr - target))
        if idx not in show_indices:
            show_indices.append(idx)
            show_offsets.append(offsets_arr[idx])

    n_show = len(show_indices)
    if n_show == 0:
        print("No valid offsets to display")
        return None

    # Time array
    T = len(profile_data[0]['v_hat'])
    t_arr = np.arange(T) * frame_dt_s

    # Convert velocities to mm/s
    def to_mm_s(v_px_frame):
        return v_px_frame * px_size_um / frame_dt_s / 1000.0

    # Compute global y-limits from v_hat time series
    all_v = []
    for idx in show_indices:
        v = to_mm_s(profile_data[idx]['v_hat'])
        all_v.extend(v[np.isfinite(v)])

    if all_v:
        v_min, v_max = np.min(all_v), np.max(all_v)
        pad = (v_max - v_min) * 0.15
        v_ylim = (v_min - pad, v_max + pad)
    else:
        v_ylim = (-0.1, 0.1)

    # Create figure with explicit number to prevent overwriting
    plt.close(1)  # Close if exists
    fig = plt.figure(num=1, figsize=(3.5 * n_show, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, n_show, height_ratios=[1, 1, 1.2], hspace=0.25, wspace=0.15)

    for i, idx in enumerate(show_indices):
        p = profile_data[idx]
        offset = p['offset']
        # Use detrended kymograph for display
        kymo = p.get('kymo_detrend', p.get('kymo'))
        v_hat = p['v_hat']
        vel_map = p.get('vel_map')
        hr_snr = p.get('best_hr_snr', np.nan)
        detrend_win = p.get('best_detrend_window', 'N/A')

        # Row 1: Kymograph
        ax_kymo = fig.add_subplot(gs[0, i])
        if kymo is not None:
            ax_kymo.imshow(kymo, cmap='gray', aspect='auto', interpolation='nearest')

        detrend_label = "raw" if detrend_win == 0 else f"filtered (win={detrend_win})"
        snr_label = f"SNR={hr_snr:.1f}dB" if np.isfinite(hr_snr) else "SNR=N/A"
        ax_kymo.set_title(f'Offset: {offset} px ({offset * px_size_um:.1f} um)\n{detrend_label}, {snr_label}', fontsize=9)
        ax_kymo.set_xlabel('Arc [px]', fontsize=9)
        if i == 0:
            ax_kymo.set_ylabel('Time [frames]', fontsize=9)

        # Row 2: Spatial velocity heatmap (vel_map as image)
        ax_vmap = fig.add_subplot(gs[1, i])
        v_hat_mm = to_mm_s(v_hat)

        if vel_map is not None and vel_map.ndim == 2 and vel_map.size > 0:
            vel_map_mm = to_mm_s(vel_map)
            # Color scale: [0, 2*median(v)] or [2*median(v), 0]
            finite_vals = vel_map_mm[np.isfinite(vel_map_mm)]
            if len(finite_vals) > 0:
                med_v = np.median(finite_vals)
                if med_v >= 0:
                    vlim0, vlim1 = 0.0, max(2.0 * med_v, 0.01)
                else:
                    vlim0, vlim1 = min(2.0 * med_v, -0.01), 0.0
            else:
                vlim0, vlim1 = -1.0, 1.0
            # Build RGBA with alpha = coherence
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vlim0, vmax=vlim1)
            cmap = plt.cm.RdBu_r
            rgba = cmap(norm(np.nan_to_num(vel_map_mm, nan=0.0)))
            conf_px = p.get('conf_px')
            if conf_px is not None and conf_px.shape == vel_map_mm.shape:
                rgba[..., 3] = np.clip(conf_px, 0, 1)
            rgba[~np.isfinite(vel_map_mm), 3] = 0.0
            im_vmap = ax_vmap.imshow(rgba, aspect='auto', interpolation='nearest')
            # Invisible scalar image for colorbar
            im_cbar = ax_vmap.imshow(np.full_like(vel_map_mm, np.nan), cmap='RdBu_r',
                                     aspect='auto', vmin=vlim0, vmax=vlim1)
            ax_vmap.set_xlabel('Arc [px]', fontsize=9)
            if i == 0:
                ax_vmap.set_ylabel('Time [frames]', fontsize=9)
            plt.colorbar(im_cbar, ax=ax_vmap, fraction=0.046, pad=0.04, label='v [mm/s]')
        else:
            ax_vmap.text(0.5, 0.5, 'No vel_map', ha='center', va='center', transform=ax_vmap.transAxes)
            ax_vmap.set_xlabel('Arc [px]', fontsize=9)
            if i == 0:
                ax_vmap.set_ylabel('Time [frames]', fontsize=9)

        # Row 3: Velocity distribution with median overlay
        ax_vel = fig.add_subplot(gs[2, i])

        if vel_map is not None and vel_map.size > 0:
            vel_map_mm = to_mm_s(vel_map)
            # Mask low-coherence pixels for histogram too
            conf_px_hist = p.get('conf_px')
            if conf_px_hist is not None and conf_px_hist.shape == vel_map_mm.shape:
                vel_map_mm = vel_map_mm.copy()
                vel_map_mm[conf_px_hist < 0.3] = np.nan

            # Build histogram for each frame
            n_bins = 64
            bins = np.linspace(v_ylim[0], v_ylim[1], n_bins + 1)
            H = np.zeros((n_bins, T), float)

            for tt in range(T):
                vt = vel_map_mm[tt] if vel_map_mm.ndim > 1 else vel_map_mm
                if isinstance(vt, np.ndarray):
                    vt = vt[np.isfinite(vt)]
                    if vt.size > 0:
                        H[:, tt] = np.histogram(vt, bins=bins)[0]

            # Show heatmap
            extent = [0.0, t_arr[-1], v_ylim[0], v_ylim[1]]
            im = ax_vel.imshow(H, origin='lower', aspect='auto', extent=extent,
                              cmap='viridis', alpha=0.8, vmin=0)

            # Overlay weighted median v_hat(t)
            ax_vel.plot(t_arr, v_hat_mm, lw=2.0, alpha=1.0, color='cyan',
                       label=r'$\hat{v}(t)$ (median)')
            plt.colorbar(im, ax=ax_vel, fraction=0.046, pad=0.04, label='Counts')
            ax_vel.set_ylim(v_ylim)
        else:
            # Fallback: simple line plot
            ax_vel.plot(t_arr, v_hat_mm, '-', color='steelblue', linewidth=1.5, alpha=0.8)
            ax_vel.axhline(np.nanmean(v_hat_mm), color='k', linestyle='--', linewidth=1.5)
            ax_vel.set_ylim(v_ylim)
            ax_vel.grid(True, alpha=0.3)

        ax_vel.set_xlabel('Time [s]', fontsize=9)
        if i == 0:
            ax_vel.set_ylabel('Velocity [mm/s]', fontsize=9)

    fig.suptitle(
        f'Kymographs & Velocity at Radial Offsets\n'
        f'Vessel R={vessel_radius_px:.1f} px ({vessel_radius_px * px_size_um:.1f} um)',
        fontsize=12, fontweight='bold'
    )

    _ensure_interactive()
    _show_figure(fig, block)
    return fig


def plot_flow_analysis(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    block: bool = False,
    show_inner_bands: bool = False,
    show_direct_integration: bool = False,
) -> plt.Figure:
    """
    Figure 2: Flow analysis with profile fit, Q(t), and spectrum.

    Panels:
    - Top left/center: Q(t) time series with systolic/diastolic markers
    - Top right: Frequency spectrum
    - Bottom left: Velocity profile with fits
    - Bottom center: Method comparison bar chart (if multiple methods enabled)
    - Bottom right: Summary metrics

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel
    vessel_radius_px : float
        Vessel radius in pixels
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    block : bool
        Whether to block on plt.show()
    show_inner_bands : bool
        Whether to show inner bands alternative estimate (default False)
    show_direct_integration : bool
        Whether to show direct integration estimate (default False)

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Extract flow results
    Q_t = flow_results.get('Q_t', np.array([]))
    mean_Q = flow_results.get('mean_Q', np.nan)
    amp_Q = flow_results.get('amp_Q', np.nan)
    PI = flow_results.get('PI', np.nan)
    f0_hz = flow_results.get('f0_hz', np.nan)
    v_max = flow_results.get('v_max', np.nan)
    r_offset = flow_results.get('r_offset', 0.0)
    chi2_reduced = flow_results.get('chi2_reduced', np.nan)
    # Radius info (effective vs original from segmentation)
    # R_fit_px is the NLLS fitted radius (true inferred radius from profile fit)
    # radius_px is from signal quality estimation (intermediate value)
    # original_radius_px is from segmentation
    R_fit_px = flow_results.get('R_fit_px', np.nan)
    effective_radius_px = R_fit_px if np.isfinite(R_fit_px) else flow_results.get('radius_px', vessel_radius_px)
    original_radius_px = flow_results.get('original_radius_px', vessel_radius_px)

    # Inner-bands alternative estimate
    Q_t_inner = flow_results.get('Q_t_inner', np.array([]))
    Q_mean_inner = flow_results.get('Q_mean_inner', np.nan)
    v_max_inner = flow_results.get('v_max_inner', np.nan)
    chi2_inner = flow_results.get('chi2_inner', np.nan)
    n_inner_bands = flow_results.get('n_inner_bands', 0)
    inner_offsets_px = flow_results.get('inner_offsets_px', np.array([]))
    # Inferred values used for inner bands (from signal-based boundary detection)
    R_inferred_px = flow_results.get('R_inferred_px', np.nan)
    r_offset_inferred_px = flow_results.get('r_offset_inferred_px', np.nan)
    velocity_center_px = flow_results.get('velocity_center_px', np.nan)  # Center from max velocity

    # Profile data - prefer cycle-based uncertainty from NLLS fit if available
    sigma_v_mean = flow_results.get('sigma_v_mean', np.array([]))
    offsets_fit_px = flow_results.get('offsets_fit_px', np.array([]))
    v_means_fit = flow_results.get('v_means_fit', np.array([]))

    # Use NLLS fit data if available (has cycle-based uncertainty)
    if len(sigma_v_mean) > 0 and len(offsets_fit_px) > 0:
        offsets_arr = offsets_fit_px
        v_means = v_means_fit  # Already in um/s
        v_errs = sigma_v_mean  # Cycle-based uncertainty (um/s)
        # Convert: v_means_fit is in um/s, v_errs is in um/s
        v_means_mm = v_means / 1000.0  # um/s -> mm/s
        v_errs_mm = v_errs / 1000.0  # um/s -> mm/s
    else:
        # Fall back to profile_data with temporal std
        offsets_arr = np.array([p['offset'] for p in profile_data])
        v_means = np.array([p['v_mean'] for p in profile_data])
        v_stds = np.array([p['v_std'] for p in profile_data])
        # Convert from px/frame to mm/s
        v_means_mm = v_means * px_size_um / frame_dt_s / 1000.0
        v_errs_mm = v_stds * px_size_um / frame_dt_s / 1000.0
    # Use effective radius for fitting (vessel_radius_px is what was passed in)
    vessel_radius_um = effective_radius_px * px_size_um
    vessel_area_um2 = np.pi * vessel_radius_um**2

    T = len(Q_t) if len(Q_t) > 0 else len(profile_data[0]['v_hat'])
    t_arr = np.arange(T) * frame_dt_s

    # Create figure with explicit number to prevent overwriting
    plt.close(2)  # Close if exists
    fig = plt.figure(num=2, figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Q(t) time series (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])

    # Get phase for harmonic fit
    phase = flow_results.get('phase', np.nan)

    if len(Q_t) > 0 and np.any(np.isfinite(Q_t)):
        # Canonicalize sign: dominant direction should be positive
        # If mean is negative, flip sign so positive = dominant direction
        Q_plot = Q_t.copy()
        mean_Q_plot = mean_Q
        amp_Q_plot = amp_Q
        phase_plot = phase
        if mean_Q < 0:
            Q_plot = -Q_plot
            mean_Q_plot = -mean_Q
            # Flip phase by pi when flipping sign
            phase_plot = phase + np.pi if np.isfinite(phase) else phase

        ax1.plot(t_arr[:len(Q_plot)], Q_plot, 'b-', linewidth=1.5, alpha=0.6, label='Q(t) full fit')

        # Overlay harmonic fit using all harmonics: Q_fit = mean + sum_k[A_k*cos + B_k*sin]
        harmonics = flow_results.get('harmonics', [])

        # If harmonics not stored, compute them from Q_t
        if len(harmonics) == 0 and np.isfinite(f0_hz) and len(Q_t) > 32:
            hr_result = fit_harmonics(Q_t, frame_dt_s, f0_hz, K=3)
            harmonics = hr_result.get('harmonics', [])

        if np.isfinite(f0_hz) and len(harmonics) > 0:
            t_fit = t_arr[:len(Q_plot)]
            Q_fit = np.full_like(t_fit, mean_Q_plot)
            for h in harmonics:
                k = h.get('k', 1)
                A = h.get('A', 0)
                B = h.get('B', 0)
                # Flip sign of coefficients if we flipped Q
                if mean_Q < 0:
                    A, B = -A, -B
                omega_k = 2 * np.pi * k * f0_hz
                Q_fit = Q_fit + A * np.cos(omega_k * t_fit) + B * np.sin(omega_k * t_fit)
            ax1.plot(t_fit, Q_fit, 'r-', linewidth=2.5, alpha=0.9,
                    label=f'Harmonic fit (f0={f0_hz:.2f} Hz)')

        ax1.axhline(mean_Q_plot, color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_Q_plot:.1f} nL/s')

        # Overlay inner-bands Q(t) if enabled and available
        if show_inner_bands and len(Q_t_inner) > 0 and np.any(np.isfinite(Q_t_inner)):
            Q_inner_plot = Q_t_inner.copy()
            Q_mean_inner_plot = Q_mean_inner
            if mean_Q < 0:  # Match sign convention
                Q_inner_plot = -Q_inner_plot
                Q_mean_inner_plot = -Q_mean_inner

            ax1.plot(t_arr[:len(Q_inner_plot)], Q_inner_plot, 'orange', linewidth=1.5, alpha=0.6,
                    label='Q(t) inner bands')

            # Compute harmonic fit for inner-bands Q(t)
            if np.isfinite(f0_hz) and len(Q_t_inner) > 32:
                hr_inner = fit_harmonics(Q_t_inner, frame_dt_s, f0_hz, K=3)
                harmonics_inner = hr_inner.get('harmonics', [])
                if len(harmonics_inner) > 0:
                    t_fit = t_arr[:len(Q_inner_plot)]
                    Q_fit_inner = np.full_like(t_fit, Q_mean_inner_plot)
                    for h in harmonics_inner:
                        k = h.get('k', 1)
                        A = h.get('A', 0)
                        B = h.get('B', 0)
                        if mean_Q < 0:
                            A, B = -A, -B
                        omega_k = 2 * np.pi * k * f0_hz
                        Q_fit_inner = Q_fit_inner + A * np.cos(omega_k * t_fit) + B * np.sin(omega_k * t_fit)
                    ax1.plot(t_fit, Q_fit_inner, 'darkorange', linewidth=2.5, alpha=0.9,
                            label=f'Harmonic (inner)')

            ax1.axhline(Q_mean_inner_plot, color='darkorange', linestyle='--', linewidth=2,
                       label=f'Mean (inner): {Q_mean_inner_plot:.1f} nL/s')

        Q_sys = np.nanmax(Q_plot)
        Q_dia = np.nanmin(Q_plot)
        ax1.axhline(Q_sys, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Systolic: {Q_sys:.1f} nL/s')
        ax1.axhline(Q_dia, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Diastolic: {Q_dia:.1f} nL/s')
        ax1.fill_between(t_arr[:len(Q_plot)], Q_dia, Q_sys, alpha=0.1, color='blue')

    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Flow Rate Q [nL/s]', fontsize=11)
    ax1.set_title(f'Time-Resolved Flow Rate | PI={PI:.2f}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Frequency spectrum
    ax2 = fig.add_subplot(gs[0, 2])

    valid_Q = Q_t[np.isfinite(Q_t)] if len(Q_t) > 0 else []
    if len(valid_Q) > 32:
        freqs, psd = periodogram(valid_Q, fs=1.0/frame_dt_s, scaling='spectrum')
        max_psd = np.max(psd[psd > 0]) if np.any(psd > 0) else 1.0
        psd_norm = psd / max_psd
        ax2.semilogy(freqs, psd_norm, 'b-', linewidth=1.5)

        if np.isfinite(f0_hz):
            for k in range(1, 4):
                fk = k * f0_hz
                if fk <= freqs[-1]:
                    color = 'red' if k == 1 else 'orange'
                    style = '-' if k == 1 else '--'
                    label = f'f0={f0_hz:.2f} Hz' if k == 1 else f'{k}f0'
                    ax2.axvline(fk, color=color, linestyle=style, linewidth=1.5, alpha=0.8,
                               label=label)

        ax2.set_ylim(1e-4, 3.0)
        ax2.set_xlim(0, 10)
        ax2.legend(loc='upper right', fontsize=8)

    ax2.set_xlabel('Frequency [Hz]', fontsize=11)
    ax2.set_ylabel('PSD', fontsize=11)
    ax2.set_title(f'Q(t) Spectrum', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    # Panel 3: Velocity profile
    ax3 = fig.add_subplot(gs[1, 0])

    # Plot data points with cycle-based uncertainty (or temporal std fallback)
    # Label indicates source: σ_v̄ = std(cycle means)/√n for cycle-based
    n_cycles = flow_results.get('n_cycles', 0)
    err_label = f'Data (σ_v̄, n={n_cycles} cycles)' if n_cycles > 0 else 'Data (temporal std)'

    # Identify points inside vs outside R_fit (accounting for centerline offset)
    # Points are "inside" if |r - r_offset| < R_fit
    r_offset_px = r_offset if np.isfinite(r_offset) else 0.0
    dist_from_center = np.abs(offsets_arr - r_offset_px)
    inside_mask = dist_from_center <= effective_radius_px
    outside_mask = ~inside_mask

    # Plot inside points (steelblue, filled)
    if np.any(inside_mask):
        ax3.errorbar(offsets_arr[inside_mask] * px_size_um, np.abs(v_means_mm[inside_mask]),
                    yerr=v_errs_mm[inside_mask],
                    fmt='o', markersize=8, color='steelblue', capsize=3, label=err_label)

    # Plot outside points (gray, hollow) - these are excluded from fit
    if np.any(outside_mask):
        ax3.errorbar(offsets_arr[outside_mask] * px_size_um, np.abs(v_means_mm[outside_mask]),
                    yerr=v_errs_mm[outside_mask],
                    fmt='o', markersize=8, markerfacecolor='none', markeredgecolor='gray',
                    ecolor='gray', capsize=3, alpha=0.6,
                    label=f'Outside R_fit (n={np.sum(outside_mask)})')

    # Plot Poiseuille fit (use effective radius)
    r_fit = np.linspace(-effective_radius_px, effective_radius_px, 100)
    # v_max is already in um/s from WLS fit - convert to mm/s
    v_max_mm = v_max / 1000.0 if np.isfinite(v_max) else np.nan

    if np.isfinite(v_max):
        # Include centerline offset from fit
        r_norm = (r_fit - r_offset) / effective_radius_px
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        v_fit = np.abs(v_max_mm) * (1 - r_norm_clipped**2)
        ax3.plot(r_fit * px_size_um, v_fit, 'g-', linewidth=2.5,
                label=f'Full fit (χ²ᵣ={chi2_reduced:.2f})')

    # Plot inner-bands parabola if enabled and available
    if show_inner_bands:
        # Use velocity_center_px (from max velocity band) as the center, R_inferred for radius
        v_max_inner_mm = v_max_inner / 1000.0 if np.isfinite(v_max_inner) else np.nan
        R_inner = R_inferred_px if np.isfinite(R_inferred_px) else effective_radius_px
        # Use velocity center (where max velocity is) instead of inferred centerline from sigma
        r_offset_inner = velocity_center_px if np.isfinite(velocity_center_px) else (
            r_offset_inferred_px if np.isfinite(r_offset_inferred_px) else r_offset)
        if np.isfinite(v_max_inner) and R_inner > 0:
            # Generate parabola centered on the max velocity band
            r_inner_fit = np.linspace(r_offset_inner - R_inner, r_offset_inner + R_inner, 100)
            r_norm_inner = (r_inner_fit - r_offset_inner) / R_inner
            r_norm_inner_clipped = np.clip(np.abs(r_norm_inner), 0.0, 1.0)
            v_fit_inner = np.abs(v_max_inner_mm) * (1 - r_norm_inner_clipped**2)
            ax3.plot(r_inner_fit * px_size_um, v_fit_inner, 'orange', linewidth=2.5, linestyle='--',
                    label=f'Inner bands (χ²ᵣ={chi2_inner:.2f})')

            # Highlight the inner bands used with markers
            if len(inner_offsets_px) > 0:
                inner_mask = np.isin(offsets_arr, inner_offsets_px)
                if np.any(inner_mask):
                    ax3.scatter(offsets_arr[inner_mask] * px_size_um, np.abs(v_means_mm[inner_mask]),
                               s=150, facecolors='none', edgecolors='orange', linewidths=2,
                               zorder=5, label=f'Inner bands (n={n_inner_bands})')

    # Convert r_offset to um for plotting
    r_offset_um = r_offset * px_size_um if np.isfinite(r_offset) else 0.0

    # Nominal centerline (r=0, dashed gray)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Inferred centerline (solid blue) - only show if offset is significant
    if np.isfinite(r_offset) and abs(r_offset) > 0.3:
        ax3.axvline(r_offset_um, color='blue', linestyle='-', linewidth=1.5, alpha=0.7,
                    label=f'Inferred CL={r_offset_um:.0f}μm')

    # Show both original (segmented) and effective (inferred) radius
    original_radius_um = original_radius_px * px_size_um
    effective_radius_um = effective_radius_px * px_size_um

    # Original radius from segmentation (dashed gray, centered on r=0)
    ax3.axvline(-original_radius_um, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(original_radius_um, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Segmented R={original_radius_um:.0f}μm')

    # Effective/inferred radius (solid darker line, centered on inferred centerline)
    if abs(effective_radius_px - original_radius_px) > 0.5:
        # Center bounds on inferred centerline (r_offset)
        ax3.axvline(r_offset_um - effective_radius_um, color='darkgreen', linestyle='-', linewidth=1.5, alpha=0.7)
        ax3.axvline(r_offset_um + effective_radius_um, color='darkgreen', linestyle='-', linewidth=1.5, alpha=0.7,
                    label=f'Inferred R={effective_radius_um:.0f}μm')

    ax3.set_xlabel('Radial Position [um]', fontsize=11)
    ax3.set_ylabel('|Velocity| [mm/s]', fontsize=11)
    ax3.set_title('Velocity Profile Fit', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Method comparison bar chart
    ax4 = fig.add_subplot(gs[1, 1])

    # Compute Q by direct integration of measured velocity profile (if enabled)
    # For circular cross-section, integrating along a diameter:
    # Q = ∫_{-R}^{R} v(x) · 2√(R² - x²) dx
    # where 2√(R² - x²) is the chord length at position x from center
    Q_integrated = np.nan
    if show_direct_integration and len(offsets_arr) >= 3:
        # offsets_arr is in pixels, v_means is in um/s
        # Compute position relative to inferred centerline
        x_px = offsets_arr - r_offset
        sort_idx = np.argsort(x_px)
        x_sorted_px = x_px[sort_idx]
        v_sorted = v_means[sort_idx]  # um/s

        # Convert to um
        x_sorted_um = x_sorted_px * px_size_um
        R_um = effective_radius_um

        # Chord length at each position: 2√(R² - x²), clipped to vessel bounds
        x_clipped = np.clip(np.abs(x_sorted_um), 0, R_um - 1e-6)
        chord_lengths = 2.0 * np.sqrt(np.maximum(R_um**2 - x_clipped**2, 0))

        # Integrand: v(x) × chord_length(x)
        integrand = np.abs(v_sorted) * chord_lengths  # um/s × um = um²/s

        # Trapezoidal integration
        Q_integrated_um3_s = np.trapz(integrand, x_sorted_um)  # um³/s
        Q_integrated = np.abs(Q_integrated_um3_s) / 1e6  # Convert to nL/s

    # Build bar chart based on which methods are enabled
    methods = ['Full\nFit']
    values = [np.abs(mean_Q)]
    colors_bar = ['darkblue']

    if show_inner_bands and np.isfinite(Q_mean_inner):
        methods.append('Inner\nBands')
        values.append(np.abs(Q_mean_inner))
        colors_bar.append('darkorange')

    if show_direct_integration and np.isfinite(Q_integrated):
        methods.append('Direct\nIntegration')
        values.append(Q_integrated)
        colors_bar.append('coral')

    bars = ax4.bar(methods, values, color=colors_bar, edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_ylabel('|Q| [nL/s]', fontsize=11)
    ax4.set_title('Flow Rate Estimates', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Summary metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Show radius info (original vs inferred if different)
    if abs(effective_radius_px - original_radius_px) > 0.5:
        radius_str = f"Radius: {effective_radius_um:.1f} um (was {original_radius_um:.1f})"
    else:
        radius_str = f"Radius: {effective_radius_um:.1f} um"
    # Show centerline offset if significant
    r_offset_str = ""
    if np.isfinite(r_offset) and abs(r_offset) > 0.3:
        r_offset_str = f"CL offset: {r_offset_um:.1f} um ({r_offset:.1f} px)\n"
    # Format inner-bands Q if enabled and available
    Q_inner_str = ""
    if show_inner_bands and np.isfinite(Q_mean_inner):
        Q_inner_str = f"Q (inner, n={n_inner_bands}): {Q_mean_inner:.1f} nL/s\n"
    # Format integrated Q if enabled and available
    Q_int_str = ""
    if show_direct_integration and np.isfinite(Q_integrated):
        Q_int_str = f"Q (integrated): {Q_integrated:.1f} nL/s\n"
    # Format chi² values
    chi2_str = f"χ²ᵣ: {chi2_reduced:.2f}\n"
    if show_inner_bands and np.isfinite(chi2_inner):
        chi2_str += f"Inner χ²ᵣ: {chi2_inner:.2f}\n"
    summary_text = (
        f"FLOW RATE SUMMARY\n"
        f"{'='*30}\n\n"
        f"{radius_str}\n"
        f"{r_offset_str}"
        f"Cross-section: {vessel_area_um2:.0f} um²\n\n"
        f"Q: {mean_Q:.1f} nL/s\n"
        f"{Q_inner_str}"
        f"{Q_int_str}"
        f"Amplitude: {amp_Q:.1f} nL/s\n"
        f"Pulsatility: {PI:.2f}\n"
        f"Heart rate: {f0_hz:.2f} Hz\n\n"
        f"{chi2_str}"
    )

    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(
        f'Flow Rate Analysis | R={vessel_radius_um:.0f} um | A={vessel_area_um2:.0f} um2',
        fontsize=14, fontweight='bold'
    )

    _ensure_interactive()
    _show_figure(fig, block)
    return fig


def plot_wls_snapshot(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    block: bool = False,
) -> plt.Figure:
    """
    Figure 3: WLS regression snapshot showing how velocities are averaged.

    Panels:
    A) Snapshot at systolic frame with scatter of v_i and fitted profile
    B) Profile shape f(r) and spread-based weights
    C) v_max(t) time series with individual bands
    D) Resulting Q(t)

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel
    vessel_radius_px : float
        Vessel radius in pixels
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Extract parameters
    r_offset = flow_results.get('r_offset', 0.0)
    Q_t = flow_results.get('Q_t', np.array([]))

    offsets_arr = np.array([p['offset'] for p in profile_data])
    n_bands = len(offsets_arr)
    T = len(profile_data[0]['v_hat'])
    t_arr = np.arange(T) * frame_dt_s

    vessel_radius_um = vessel_radius_px * px_size_um

    # Conversion helper
    def to_um_s(v):
        return v * px_size_um / frame_dt_s

    # Get spread for weights (use v_std as proxy if spread not available)
    spread_arr = np.array([p.get('mean_vel_spread', p['v_std']) for p in profile_data])
    spread_eps = 1.0
    weights = np.array([1.0 / (s + spread_eps) if np.isfinite(s) and s > 0 else 0.0
                       for s in spread_arr])

    # Poiseuille profile shape function (includes centerline offset from fit)
    def f_r(r_px):
        r_norm = (r_px - r_offset) / vessel_radius_px
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        return 1.0 - r_norm_clipped**2

    f_vals = np.array([f_r(off) for off in offsets_arr])

    # Build v_hat matrix (bands x time) in um/s
    v_hat_matrix = np.array([to_um_s(p['v_hat']) for p in profile_data])

    # Compute v_max(t) using WLS
    mask = (f_vals > 0.1) & (weights > 0)
    if np.sum(mask) < 1:
        print("Insufficient bands for WLS visualization")
        return None

    w_masked = weights[mask]
    f_masked = f_vals[mask]
    v_masked = v_hat_matrix[mask, :]

    denom = np.sum(w_masked * f_masked**2)
    if denom < 1e-10:
        print("WLS denominator too small")
        return None

    wf = w_masked * f_masked
    v_max_t = np.sum(wf[:, np.newaxis] * v_masked, axis=0) / denom

    # Find representative frames
    valid_vmax = v_max_t[np.isfinite(v_max_t)]
    if len(valid_vmax) < 10:
        print("Insufficient valid v_max values")
        return None

    t_systole = np.nanargmax(v_max_t)
    t_diastole = np.nanargmin(v_max_t)

    # Create figure with explicit number to prevent overwriting
    plt.close(3)  # Close if exists
    fig = plt.figure(num=3, figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Velocity profile at systole
    ax_a = fig.add_subplot(gs[0, 0])

    v_snap_sys = v_hat_matrix[:, t_systole] / 1000.0  # Convert to mm/s

    # Plot data points with error bars
    v_std_arr = np.array([p['v_std'] for p in profile_data])
    v_std_sys = v_std_arr * px_size_um / frame_dt_s / 1000.0  # Convert to mm/s
    ax_a.errorbar(offsets_arr * px_size_um, v_snap_sys, yerr=v_std_sys,
                  fmt='o', color='steelblue', markersize=8, capsize=4, alpha=0.8,
                  label='Data')

    # Plot fitted profile
    r_fit = np.linspace(-vessel_radius_px, vessel_radius_px, 100)
    v_fit_sys = v_max_t[t_systole] / 1000.0 * np.array([f_r(r) for r in r_fit])
    ax_a.plot(r_fit * px_size_um, v_fit_sys, 'r-', linewidth=2.5,
             label=f'Fit: v_max={v_max_t[t_systole]/1000.0:.2f} mm/s')

    ax_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_a.axvline(-vessel_radius_um, color='gray', linestyle=':', alpha=0.5)
    ax_a.axvline(vessel_radius_um, color='gray', linestyle=':', alpha=0.5)
    ax_a.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax_a.set_xlabel('Radial Position [um]', fontsize=11)
    ax_a.set_ylabel('Velocity [mm/s]', fontsize=11)
    ax_a.set_title(f'A) Velocity Profile at Systole (t={t_systole*frame_dt_s:.2f}s)',
                  fontsize=12, fontweight='bold')
    ax_a.legend(loc='best', fontsize=10)
    ax_a.grid(True, alpha=0.3)

    # Panel B: Velocity profile at diastole
    ax_b = fig.add_subplot(gs[0, 1])

    v_snap_dia = v_hat_matrix[:, t_diastole] / 1000.0  # Convert to mm/s

    # Plot data points with error bars
    ax_b.errorbar(offsets_arr * px_size_um, v_snap_dia, yerr=v_std_sys,
                  fmt='o', color='steelblue', markersize=8, capsize=4, alpha=0.8,
                  label='Data')

    # Plot fitted profile
    v_fit_dia = v_max_t[t_diastole] / 1000.0 * np.array([f_r(r) for r in r_fit])
    ax_b.plot(r_fit * px_size_um, v_fit_dia, 'g-', linewidth=2.5,
             label=f'Fit: v_max={v_max_t[t_diastole]/1000.0:.2f} mm/s')

    ax_b.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_b.axvline(-vessel_radius_um, color='gray', linestyle=':', alpha=0.5)
    ax_b.axvline(vessel_radius_um, color='gray', linestyle=':', alpha=0.5)
    ax_b.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax_b.set_xlabel('Radial Position [um]', fontsize=11)
    ax_b.set_ylabel('Velocity [mm/s]', fontsize=11)
    ax_b.set_title(f'B) Velocity Profile at Diastole (t={t_diastole*frame_dt_s:.2f}s)',
                  fontsize=12, fontweight='bold')
    ax_b.legend(loc='best', fontsize=10)
    ax_b.grid(True, alpha=0.3)

    # Panel C: v_max(t) with individual bands
    ax_c = fig.add_subplot(gs[1, 0])

    # Plot a few individual bands (faded, scaled by 1/f(r))
    n_show_bands = min(5, np.sum(mask))
    shown = 0
    for i, (off, is_used) in enumerate(zip(offsets_arr, mask)):
        if is_used and shown < n_show_bands:
            if f_vals[i] > 0.1:
                v_band = v_hat_matrix[i, :] / f_vals[i] / 1000.0  # Convert to mm/s
                ax_c.plot(t_arr, v_band, '-', alpha=0.3, linewidth=1,
                         label=f'r={off}px (scaled)' if shown == 0 else None)
                shown += 1

    # Plot v_max(t)
    v_max_t_mm = v_max_t / 1000.0  # Convert to mm/s
    ax_c.plot(t_arr, v_max_t_mm, 'b-', linewidth=2.5, alpha=0.9, label='v_max(t) [WLS]')
    ax_c.axhline(np.nanmean(v_max_t_mm), color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {np.nanmean(v_max_t_mm):.2f} mm/s')

    # Mark systole/diastole
    ax_c.axvline(t_systole * frame_dt_s, color='red', linestyle=':', alpha=0.7, label='Systole')
    ax_c.axvline(t_diastole * frame_dt_s, color='green', linestyle=':', alpha=0.7, label='Diastole')

    ax_c.set_xlabel('Time [s]', fontsize=11)
    ax_c.set_ylabel('v_max [mm/s]', fontsize=11)
    ax_c.set_title('C) Centerline Velocity v_max(t) from WLS\n'
                  'Faded: individual bands scaled by 1/f(r)', fontsize=12, fontweight='bold')
    ax_c.legend(loc='best', fontsize=9)
    ax_c.grid(True, alpha=0.3)

    # Panel D: Resulting Q(t)
    ax_d = fig.add_subplot(gs[1, 1])

    # Q = v_max * pi*R^2 * 0.5 [convert to nL/s] (Poiseuille: n/(n+2) = 2/4 = 0.5)
    vessel_area_um2 = np.pi * vessel_radius_um**2
    Q_factor = vessel_area_um2 * 0.5 / 1e6  # nL/s per um/s
    Q_t_computed = v_max_t * Q_factor

    # Canonicalize sign: dominant direction should be positive
    Q_mean_raw = np.nanmean(Q_t_computed)
    if Q_mean_raw < 0:
        Q_t_computed = -Q_t_computed

    ax_d.plot(t_arr, Q_t_computed, 'b-', linewidth=2, alpha=0.9,
             label='Q(t) = v_max * πR² / 2')

    Q_mean = np.nanmean(Q_t_computed)
    Q_sys = np.nanmax(Q_t_computed)
    Q_dia = np.nanmin(Q_t_computed)
    ax_d.axhline(Q_mean, color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {Q_mean:.1f} nL/s')
    ax_d.fill_between(t_arr, Q_dia, Q_sys, alpha=0.1, color='blue')

    PI_val = (Q_sys - Q_dia) / Q_mean if Q_mean != 0 else np.nan

    ax_d.set_xlabel('Time [s]', fontsize=11)
    ax_d.set_ylabel('Flow Rate Q [nL/s]', fontsize=11)
    ax_d.set_title(f'D) Flow Rate Q(t)\n'
                  f'Poiseuille, R={vessel_radius_um:.0f}um, PI={PI_val:.2f}',
                  fontsize=12, fontweight='bold')
    ax_d.legend(loc='best', fontsize=10)
    ax_d.grid(True, alpha=0.3)

    fig.suptitle(
        'WLS Regression: v_max(t) = Sum(w_i * v_i(t) * f(r_i)) / Sum(w_i * f(r_i)^2)',
        fontsize=14, fontweight='bold'
    )

    _ensure_interactive()
    _show_figure(fig, block)
    return fig


def plot_vessel_geometry(
    coords: np.ndarray,
    stack: np.ndarray,
    vessel_radius_px: float,
    offsets_display: Optional[np.ndarray] = None,
    profile_data: Optional[List[Dict[str, Any]]] = None,
    frame_idx: int = 0,
    block: bool = False,
) -> plt.Figure:
    """
    Figure 0: Vessel geometry showing centerline and radial offset bands.

    Parameters
    ----------
    coords : ndarray
        (N, 2) centerline coordinates [x, y]
    stack : ndarray
        (T, H, W) video stack
    vessel_radius_px : float
        Vessel radius in pixels
    offsets_display : array, optional
        Offsets to display (default: -Rmax, -Rmax/2, 0, Rmax/2, Rmax)
    profile_data : list of dict, optional
        Output from compute_radial_velocity_profiles. If provided, will show
        arc restriction with greyed-out unused portions.
    frame_idx : int
        Frame to display
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig : matplotlib.Figure
    """
    if offsets_display is None:
        r50 = int(np.round(0.5 * vessel_radius_px))
        rmax = int(np.round(vessel_radius_px))
        offsets_display = np.array([-rmax, -r50, 0, r50, rmax])

    # Get frame
    frame = stack[frame_idx] if frame_idx < len(stack) else stack[0]

    # Compute normal vectors
    d = np.gradient(coords, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)

    # Build arc restriction lookup from profile_data
    arc_bounds = {}  # offset -> (arc_start, arc_end)
    if profile_data is not None:
        for p in profile_data:
            offset = p.get('offset', 0)
            arc_start = p.get('arc_start', 0)
            arc_end = p.get('arc_end', len(coords))
            arc_bounds[offset] = (arc_start, arc_end)

    # Create figure with explicit number to prevent overwriting
    plt.close(0)  # Close if exists
    fig, ax = plt.subplots(num=0, figsize=(12, 10))

    # Show frame
    ax.imshow(frame, cmap='gray', aspect='equal')

    # Plot centerline with arc restriction
    centerline_arc = arc_bounds.get(0, (0, len(coords)))
    arc_start, arc_end = centerline_arc

    # Plot greyed-out portions first (if arc restriction exists)
    if arc_start > 0 or arc_end < len(coords):
        # Grey out start portion
        if arc_start > 0:
            ax.plot(coords[:arc_start+1, 0], coords[:arc_start+1, 1],
                   '-', color='gray', linewidth=2, alpha=0.4)
        # Grey out end portion
        if arc_end < len(coords):
            ax.plot(coords[arc_end-1:, 0], coords[arc_end-1:, 1],
                   '-', color='gray', linewidth=2, alpha=0.4)
        # Plot used portion in cyan
        ax.plot(coords[arc_start:arc_end, 0], coords[arc_start:arc_end, 1],
               'c-', linewidth=2, label=f'Centerline (arc {arc_start}-{arc_end})', alpha=0.9)
    else:
        ax.plot(coords[:, 0], coords[:, 1], 'c-', linewidth=2, label='Centerline', alpha=0.9)

    # Plot offset bands with arc restriction
    colors = ['lime', 'yellow', 'orange', 'red', 'magenta']
    for i, offset in enumerate(offsets_display):
        if offset == 0:
            continue
        color = colors[i % len(colors)]
        coords_offset = coords + offset * n

        # Get arc bounds for this offset (use centerline bounds if not available)
        offset_arc = arc_bounds.get(offset, centerline_arc)
        arc_start, arc_end = offset_arc

        # Plot greyed-out portions first
        if arc_start > 0 or arc_end < len(coords):
            # Grey out start portion
            if arc_start > 0:
                ax.plot(coords_offset[:arc_start+1, 0], coords_offset[:arc_start+1, 1],
                       '-', color='gray', linewidth=1.5, alpha=0.3)
            # Grey out end portion
            if arc_end < len(coords):
                ax.plot(coords_offset[arc_end-1:, 0], coords_offset[arc_end-1:, 1],
                       '-', color='gray', linewidth=1.5, alpha=0.3)
            # Plot used portion in color
            ax.plot(coords_offset[arc_start:arc_end, 0], coords_offset[arc_start:arc_end, 1],
                   '-', color=color, linewidth=1.5, alpha=0.7,
                   label=f'Offset: {offset:+d} px')
        else:
            ax.plot(coords_offset[:, 0], coords_offset[:, 1], '-',
                   color=color, linewidth=1.5, alpha=0.7,
                   label=f'Offset: {offset:+d} px')

    # Zoom to vessel region
    margin = vessel_radius_px * 1.5
    x_min = coords[:, 0].min() - margin
    x_max = coords[:, 0].max() + margin
    y_min = coords[:, 1].min() - margin
    y_max = coords[:, 1].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Flip y for image coordinates

    ax.set_xlabel('X [px]', fontsize=11)
    ax.set_ylabel('Y [px]', fontsize=11)

    # Update title to show arc restriction info
    if profile_data is not None and len(arc_bounds) > 0:
        c_arc = arc_bounds.get(0, (0, len(coords)))
        arc_len = c_arc[1] - c_arc[0]
        ax.set_title(f'Vessel Centerline & Radial Sampling Bands\n'
                    f'Frame {frame_idx} | Radius: {vessel_radius_px:.1f} px | '
                    f'Arc: {c_arc[0]}-{c_arc[1]} ({arc_len} px)',
                    fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Vessel Centerline & Radial Sampling Bands\n'
                    f'Frame {frame_idx} | Radius: {vessel_radius_px:.1f} px',
                    fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)

    _ensure_interactive()
    _show_figure(fig, block)
    return fig


def show_all_flow_figures(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    coords: Optional[np.ndarray] = None,
    stack: Optional[np.ndarray] = None,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
) -> None:
    """
    Show all flow analysis figures.

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel
    vessel_radius_px : float
        Vessel radius in pixels
    coords : ndarray, optional
        (N, 2) centerline coordinates for geometry plot
    stack : ndarray, optional
        (T, H, W) video stack for geometry plot
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    out_dir : str, optional
        Directory to save figures. If None, figures are only displayed.
    vessel_id : str, optional
        Vessel identifier for filename (e.g., "506_512")
    """
    from pathlib import Path

    DPI = 400  # High DPI for publication-quality figures

    # Create panels subfolder if saving
    panels_dir = None
    if out_dir and vessel_id:
        panels_dir = Path(out_dir) / f"flow_{vessel_id}_panels"
        panels_dir.mkdir(exist_ok=True)

    # --- TEMPORARILY DISABLED: Non-slide figures ---
    # # Figure 0: Vessel geometry (if coords and stack provided)
    # if coords is not None and stack is not None:
    #     print("Figure 0: Vessel centerline and radial sampling bands...")
    #     fig0 = plot_vessel_geometry(coords, stack, vessel_radius_px,
    #                                 profile_data=profile_data, block=False)
    #     if out_dir and vessel_id:
    #         fig0.savefig(Path(out_dir) / f"flow_{vessel_id}_fig0_geometry.png", dpi=DPI, bbox_inches='tight')

    # # Figure 1: Kymographs and time series
    # print("Figure 1: Kymographs and velocity time series at radial positions...")
    # fig1 = plot_radial_kymographs(profile_data, vessel_radius_px,
    #                       px_size_um=px_size_um, frame_dt_s=frame_dt_s, block=False)
    # if out_dir and vessel_id:
    #     fig1.savefig(Path(out_dir) / f"flow_{vessel_id}_fig1_kymographs.png", dpi=DPI, bbox_inches='tight')

    # # Figure 2: Flow analysis (profile, Q(t), spectrum)
    # print("Figure 2: Profile fit, Q(t), and frequency spectrum...")
    # fig2 = plot_flow_analysis(profile_data, flow_results, vessel_radius_px,
    #                   px_size_um=px_size_um, frame_dt_s=frame_dt_s, block=False)
    # if out_dir and vessel_id:
    #     fig2.savefig(Path(out_dir) / f"flow_{vessel_id}_fig2_flow.png", dpi=DPI, bbox_inches='tight')

    # # Figure 3: WLS snapshot
    # print("Figure 3: WLS regression demonstration...")
    # fig3 = plot_wls_snapshot(profile_data, flow_results, vessel_radius_px,
    #                  px_size_um=px_size_um, frame_dt_s=frame_dt_s, block=False)
    # if out_dir and vessel_id:
    #     fig3.savefig(Path(out_dir) / f"flow_{vessel_id}_fig3_wls.png", dpi=DPI, bbox_inches='tight')
    #     print(f"  Saved combined figures to {out_dir}")

    # # Presentation figures (cleaner, for slides)
    # print("Presentation figures: Profile fit and Q(t) with uncertainties...")
    # fig_pres1, fig_pres2 = plot_flow_presentation(
    #     profile_data, flow_results, vessel_radius_px,
    #     px_size_um=px_size_um, frame_dt_s=frame_dt_s,
    #     out_dir=out_dir, vessel_id=vessel_id, block=False
    # )

    # --- END TEMPORARILY DISABLED ---

    # Presentation kymographs (3x3 at center and ±R/2)
    print("Presentation kymographs: 3x3 at center and ±R/2...")
    fig_kymo_pres = plot_kymographs_presentation(
        profile_data, flow_results, vessel_radius_px,
        px_size_um=px_size_um, frame_dt_s=frame_dt_s,
        out_dir=out_dir, vessel_id=vessel_id, block=False
    )

    # Slide figures: velocity profile fit and cycle uncertainty
    print("Slide figures: velocity profile fit and cycle-based uncertainty...")
    _, _ = plot_slide_velocity_profile_fit(
        profile_data, flow_results, vessel_radius_px,
        px_size_um=px_size_um, frame_dt_s=frame_dt_s,
        out_dir=out_dir, vessel_id=vessel_id, block=False
    )

    # Simple glanceable profile plot
    print("Simple profile plot...")
    _ = plot_simple_velocity_profile(
        profile_data, flow_results, vessel_radius_px,
        px_size_um=px_size_um, frame_dt_s=frame_dt_s,
        out_dir=out_dir, vessel_id=vessel_id, block=False
    )

    # Slide figures: flow rate (Q(t) with band contributions, spectrum)
    print("Slide figures: flow rate Q(t) and spectrum...")
    _, _ = plot_slide_flow_rate(
        profile_data, flow_results, vessel_radius_px,
        px_size_um=px_size_um, frame_dt_s=frame_dt_s,
        out_dir=out_dir, vessel_id=vessel_id, block=False
    )

    # # Save individual panels (temporarily disabled)
    # if panels_dir:
    #     _save_flow_panels(profile_data, flow_results, vessel_radius_px,
    #                       px_size_um, frame_dt_s, panels_dir, DPI)

    print("\nSlide figures generated. Close windows or press Enter to continue...")


def _save_flow_panels(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float,
    frame_dt_s: float,
    panels_dir,
    dpi: int = 200,
) -> None:
    """Save individual panels as separate PNG files."""
    import matplotlib.pyplot as plt

    # Extract data
    Q_t = flow_results.get('Q_t', np.array([]))
    f0_hz = flow_results.get('f0_hz', np.nan)
    mean_Q = flow_results.get('mean_Q', np.nan)
    amp_Q = flow_results.get('amp_Q', np.nan)
    PI = flow_results.get('PI', np.nan)
    r_offset = flow_results.get('r_offset', 0.0)
    v_max_fit = flow_results.get('v_max_fit', np.nan)
    chi2_reduced = flow_results.get('chi2_reduced', np.nan)

    T = len(profile_data[0]['v_hat']) if profile_data else 0
    t_arr = np.arange(T) * frame_dt_s
    vessel_radius_um = vessel_radius_px * px_size_um

    # Conversion
    def to_um_s(v):
        return v * px_size_um / frame_dt_s

    # Panel: Q(t) time series
    if len(Q_t) > 0:
        fig_qt, ax_qt = plt.subplots(figsize=(6, 4))
        ax_qt.plot(t_arr[:len(Q_t)], Q_t, 'b-', linewidth=1.5)
        ax_qt.axhline(mean_Q, color='r', linestyle='--', linewidth=1.5, label=f'Mean={mean_Q:.1f} nL/s')
        ax_qt.set_xlabel('Time [s]', fontsize=11)
        ax_qt.set_ylabel('Q(t) [nL/s]', fontsize=11)
        ax_qt.set_title(f'Flow Rate Time Series (f0={f0_hz:.2f} Hz)', fontsize=12)
        ax_qt.legend(loc='best')
        ax_qt.grid(True, alpha=0.3)
        fig_qt.tight_layout()
        fig_qt.savefig(panels_dir / "Q_timeseries.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig_qt)
        print(f"    Saved panel: Q_timeseries.png")

    # Panel: Frequency spectrum
    if len(Q_t) > 0 and np.isfinite(f0_hz):
        from scipy.signal import welch
        fs = 1.0 / frame_dt_s
        _nperseg = min(256, len(Q_t)//2)
        freqs, psd = welch(Q_t - np.nanmean(Q_t), fs=fs, nperseg=_nperseg, nfft=max(2048, _nperseg * 4))

        max_psd = np.max(psd[psd > 0]) if np.any(psd > 0) else 1.0
        psd_norm = psd / max_psd

        fig_psd, ax_psd = plt.subplots(figsize=(5, 4))
        ax_psd.semilogy(freqs, psd_norm, 'b-', linewidth=1.5)
        for h in range(1, 4):
            ax_psd.axvline(h * f0_hz, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax_psd.set_xlabel('Frequency [Hz]', fontsize=11)
        ax_psd.set_ylabel('Normalized PSD', fontsize=11)
        ax_psd.set_title(f'Power Spectral Density (f0={f0_hz:.2f} Hz)', fontsize=12)
        ax_psd.set_xlim(0, min(10, freqs[-1]))
        ax_psd.set_ylim(1e-4, 3.0)
        ax_psd.grid(True, alpha=0.3)
        fig_psd.tight_layout()
        fig_psd.savefig(panels_dir / "spectrum.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig_psd)
        print(f"    Saved panel: spectrum.png")

    # Panel: Velocity profile
    offsets_arr = np.array([p['offset'] for p in profile_data])
    v_hat_mean = np.array([np.nanmean(p['v_hat']) for p in profile_data])
    v_hat_mean_um = to_um_s(v_hat_mean) / 1000.0  # mm/s

    fig_prof, ax_prof = plt.subplots(figsize=(5, 4))
    ax_prof.scatter(offsets_arr * px_size_um, np.abs(v_hat_mean_um), s=60, c='steelblue', alpha=0.8)

    # Fit curve (Poiseuille, n=2)
    if np.isfinite(v_max_fit):
        r_fit = np.linspace(-vessel_radius_px, vessel_radius_px, 100)
        r_norm = (r_fit - r_offset) / vessel_radius_px
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        v_fit = np.abs(to_um_s(v_max_fit) / 1000.0) * (1 - r_norm_clipped**2)
        ax_prof.plot(r_fit * px_size_um, v_fit, 'r-', linewidth=2,
                    label=f'Poiseuille, χ²ᵣ={chi2_reduced:.2f}')
        ax_prof.legend(loc='best')

    ax_prof.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_prof.axvline(-vessel_radius_um, color='gray', linestyle=':', alpha=0.5)
    ax_prof.axvline(vessel_radius_um, color='gray', linestyle=':', alpha=0.5)
    ax_prof.set_xlabel('Radial Position [um]', fontsize=11)
    ax_prof.set_ylabel('|Velocity| [mm/s]', fontsize=11)
    ax_prof.set_title('Velocity Profile Fit', fontsize=12)
    ax_prof.grid(True, alpha=0.3)
    fig_prof.tight_layout()
    fig_prof.savefig(panels_dir / "velocity_profile.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig_prof)
    print(f"    Saved panel: velocity_profile.png")

    # Panel: Summary metrics
    fig_sum, ax_sum = plt.subplots(figsize=(4, 4))
    ax_sum.axis('off')

    vessel_area_um2 = np.pi * vessel_radius_um**2

    summary_text = (
        f"FLOW SUMMARY\n"
        f"{'='*25}\n\n"
        f"Vessel radius: {vessel_radius_um:.1f} um\n"
        f"Cross-section: {vessel_area_um2:.0f} um²\n\n"
        f"Mean Q: {mean_Q:.1f} nL/s\n"
        f"Amplitude: {amp_Q:.1f} nL/s\n"
        f"Pulsatility: {PI:.2f}\n"
        f"Heart rate: {f0_hz:.2f} Hz\n\n"
        f"Profile: Poiseuille (n=2)\n"
        f"Fit χ²ᵣ: {chi2_reduced:.2f}\n"
    )

    ax_sum.text(0.1, 0.9, summary_text, transform=ax_sum.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig_sum.tight_layout()
    fig_sum.savefig(panels_dir / "summary.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig_sum)
    print(f"    Saved panel: summary.png")


def plot_flow_presentation(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
    block: bool = False,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create two presentation-quality figures for flow analysis.

    Figure 1: Velocity Profile Fit
        - Left: Schematic of radial sampling bands on vessel cross-section
        - Right: v̄(r) with σ_v̄ error bars, Poiseuille fit, χ²_red displayed

    Figure 2: Flow Rate Estimation
        - Left: Template projection showing how v_max(t) is computed from bands
        - Right: Q(t) time series with σ_Q uncertainty prominently displayed

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel (must include sigma_Q, chi2_reduced)
    vessel_radius_px : float
        Vessel radius in pixels
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    out_dir : str, optional
        Directory to save figures
    vessel_id : str, optional
        Vessel identifier for filename
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig1, fig2 : matplotlib.Figure
        The two presentation figures
    """
    from pathlib import Path

    # Extract key results
    Q_t = flow_results.get('Q_t', np.array([]))
    mean_Q = flow_results.get('mean_Q', np.nan)
    amp_Q = flow_results.get('amp_Q', np.nan)
    f0_hz = flow_results.get('f0_hz', np.nan)
    v_max = flow_results.get('v_max', np.nan)
    r_offset = flow_results.get('r_offset', 0.0)
    chi2_reduced = flow_results.get('chi2_reduced', np.nan)
    sigma_Q = flow_results.get('sigma_Q', np.nan)
    n_cycles = flow_results.get('n_cycles', 0)

    # Radius info
    R_fit_px = flow_results.get('R_fit_px', np.nan)
    effective_radius_px = R_fit_px if np.isfinite(R_fit_px) else vessel_radius_px
    effective_radius_um = effective_radius_px * px_size_um

    # Profile data with cycle-based uncertainty
    sigma_v_mean = flow_results.get('sigma_v_mean', np.array([]))
    offsets_fit_px = flow_results.get('offsets_fit_px', np.array([]))
    v_means_fit = flow_results.get('v_means_fit', np.array([]))

    # Use NLLS fit data if available
    if len(sigma_v_mean) > 0 and len(offsets_fit_px) > 0:
        offsets_arr = offsets_fit_px
        v_means = v_means_fit  # um/s
        v_errs = sigma_v_mean  # um/s
        v_means_mm = v_means / 1000.0
        v_errs_mm = v_errs / 1000.0
    else:
        # Fallback to profile_data
        offsets_arr = np.array([p['offset'] for p in profile_data])
        v_means = np.array([p['v_mean'] for p in profile_data])
        v_stds = np.array([p['v_std'] for p in profile_data])
        v_means_mm = v_means * px_size_um / frame_dt_s / 1000.0
        v_errs_mm = v_stds * px_size_um / frame_dt_s / 1000.0

    T = len(Q_t) if len(Q_t) > 0 else len(profile_data[0]['v_hat'])
    t_arr = np.arange(T) * frame_dt_s

    DPI = 200

    # =====================================================================
    # FIGURE 1: Velocity Profile Analysis
    # =====================================================================
    plt.close('pres_fig1')
    fig1 = plt.figure(num='pres_fig1', figsize=(14, 6))
    gs1 = fig1.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # Left panel: Vessel cross-section schematic with sampling bands
    ax1a = fig1.add_subplot(gs1[0, 0])

    # Draw vessel circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_circ = effective_radius_um * np.cos(theta)
    y_circ = effective_radius_um * np.sin(theta)
    ax1a.plot(x_circ, y_circ, 'k-', linewidth=2, label='Vessel wall')
    ax1a.fill(x_circ, y_circ, color='lightcoral', alpha=0.2)

    # Draw sampling bands as horizontal lines
    n_bands_show = min(7, len(offsets_arr))
    band_indices = np.linspace(0, len(offsets_arr)-1, n_bands_show, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bands_show))

    for i, idx in enumerate(band_indices):
        r = offsets_arr[idx] * px_size_um
        # Chord half-length at radius r
        if abs(r) < effective_radius_um:
            half_chord = np.sqrt(effective_radius_um**2 - r**2)
            ax1a.plot([-half_chord, half_chord], [r, r], '-', color=colors[i],
                     linewidth=3, alpha=0.8)
            # Add small arrow indicating velocity measurement
            ax1a.annotate('', xy=(half_chord*0.3, r), xytext=(-half_chord*0.3, r),
                         arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))

    # Mark centerline
    r_offset_um = r_offset * px_size_um
    ax1a.axhline(r_offset_um, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1a.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax1a.set_xlim(-effective_radius_um*1.3, effective_radius_um*1.3)
    ax1a.set_ylim(-effective_radius_um*1.3, effective_radius_um*1.3)
    ax1a.set_aspect('equal')
    ax1a.set_xlabel('Position [μm]', fontsize=12)
    ax1a.set_ylabel('Radial offset r [μm]', fontsize=12)
    ax1a.set_title('Radial Sampling Bands', fontsize=14, fontweight='bold')

    # Add annotation
    ax1a.text(0.05, 0.95, f'R = {effective_radius_um:.0f} μm\n{len(offsets_arr)} bands',
             transform=ax1a.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Velocity profile with fit
    ax1b = fig1.add_subplot(gs1[0, 1])

    # Identify inside vs outside points
    r_offset_px = r_offset if np.isfinite(r_offset) else 0.0
    dist_from_center = np.abs(offsets_arr - r_offset_px)
    inside_mask = dist_from_center <= effective_radius_px

    # Plot data with error bars
    if np.any(inside_mask):
        ax1b.errorbar(offsets_arr[inside_mask] * px_size_um, np.abs(v_means_mm[inside_mask]),
                     yerr=v_errs_mm[inside_mask],
                     fmt='o', markersize=10, color='steelblue', capsize=4, capthick=2,
                     elinewidth=2, label=f'Data ± σ_v̄ (n={n_cycles} cycles)')

    # Plot outside points (gray)
    outside_mask = ~inside_mask
    if np.any(outside_mask):
        ax1b.errorbar(offsets_arr[outside_mask] * px_size_um, np.abs(v_means_mm[outside_mask]),
                     yerr=v_errs_mm[outside_mask],
                     fmt='o', markersize=8, markerfacecolor='none', markeredgecolor='gray',
                     ecolor='gray', capsize=3, alpha=0.5, label='Outside R (excluded)')

    # Plot Poiseuille fit
    if np.isfinite(v_max):
        r_fit = np.linspace(-effective_radius_px, effective_radius_px, 100)
        v_max_mm = v_max / 1000.0
        r_norm = (r_fit - r_offset) / effective_radius_px
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        v_fit = np.abs(v_max_mm) * (1 - r_norm_clipped**2)
        ax1b.plot(r_fit * px_size_um, v_fit, 'r-', linewidth=3,
                 label='Poiseuille fit')

    # Vessel boundary lines
    ax1b.axvline(-effective_radius_um, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax1b.axvline(effective_radius_um, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax1b.axvline(r_offset_um, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

    ax1b.set_xlabel('Radial Position r [μm]', fontsize=12)
    ax1b.set_ylabel('|v̄| [mm/s]', fontsize=12)
    ax1b.set_title('Velocity Profile Fit', fontsize=14, fontweight='bold')
    ax1b.legend(loc='upper right', fontsize=10)
    ax1b.grid(True, alpha=0.3)

    # Prominent chi-squared and sigma annotation box
    chi2_color = 'green' if chi2_reduced < 2 else ('orange' if chi2_reduced < 5 else 'red')
    stats_text = (
        f"χ²_red = {chi2_reduced:.2f}\n"
        f"v_max = {np.abs(v_max)/1000:.2f} mm/s\n"
        f"R_fit = {effective_radius_um:.0f} μm"
    )
    ax1b.text(0.02, 0.98, stats_text, transform=ax1b.transAxes, fontsize=12,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=chi2_color,
                      linewidth=2, alpha=0.9))

    fig1.suptitle('Figure 1: Poiseuille Profile Fit to Radial Velocity Data',
                  fontsize=14, fontweight='bold', y=0.98)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # =====================================================================
    # FIGURE 2: Flow Rate Estimation with Uncertainty
    # =====================================================================
    plt.close('pres_fig2')
    fig2 = plt.figure(num='pres_fig2', figsize=(14, 6))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # Left panel: Template projection visualization
    ax2a = fig2.add_subplot(gs2[0, 0])

    # Show profile shape function f(r) and weights
    if np.isfinite(v_max) and effective_radius_px > 0:
        r_plot = np.linspace(-effective_radius_px, effective_radius_px, 100)
        r_norm = (r_plot - r_offset) / effective_radius_px
        f_shape = np.maximum(0, 1 - r_norm**2)  # Poiseuille shape

        ax2a.fill_between(r_plot * px_size_um, 0, f_shape, alpha=0.3, color='steelblue',
                         label=r'$f(r) = 1 - (r/R)^2$')
        ax2a.plot(r_plot * px_size_um, f_shape, 'b-', linewidth=2)

        # Mark the band positions with dots sized by their contribution
        if len(offsets_arr) > 0:
            r_band_norm = (offsets_arr - r_offset) / effective_radius_px
            f_bands = np.maximum(0, 1 - r_band_norm**2)
            # Size proportional to f² (contribution to weighted sum)
            sizes = 50 + 200 * f_bands**2
            ax2a.scatter(offsets_arr * px_size_um, f_bands, s=sizes, c='red',
                        alpha=0.7, edgecolors='darkred', linewidths=1.5,
                        label='Band contributions')

    ax2a.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax2a.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2a.axvline(r_offset_um, color='blue', linestyle='--', alpha=0.5)

    ax2a.set_xlabel('Radial Position r [μm]', fontsize=12)
    ax2a.set_ylabel('Shape factor f(r)', fontsize=12)
    ax2a.set_title('Template Projection', fontsize=14, fontweight='bold')
    ax2a.legend(loc='upper right', fontsize=10)
    ax2a.set_ylim(-0.1, 1.2)
    ax2a.grid(True, alpha=0.3)

    # Add equation
    eq_text = (
        r"$v_{\max}(t) = \frac{\sum_i w_i f_i v_i(t)}{\sum_i w_i f_i^2}$"
        "\n\n"
        r"$Q(t) = \frac{\pi R^2}{2} v_{\max}(t)$"
    )
    ax2a.text(0.5, 0.15, eq_text, transform=ax2a.transAxes, fontsize=13,
             ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Right panel: Q(t) time series with uncertainty
    ax2b = fig2.add_subplot(gs2[0, 1])

    if len(Q_t) > 0 and np.any(np.isfinite(Q_t)):
        # Canonicalize sign
        Q_plot = np.abs(Q_t) if mean_Q < 0 else Q_t
        mean_Q_plot = np.abs(mean_Q)

        ax2b.plot(t_arr[:len(Q_plot)], Q_plot, 'b-', linewidth=1.5, alpha=0.7, label='Q(t)')

        # Harmonic fit overlay
        harmonics = flow_results.get('harmonics', [])
        if len(harmonics) == 0 and np.isfinite(f0_hz) and len(Q_t) > 32:
            hr_result = fit_harmonics(Q_t, frame_dt_s, f0_hz, K=3)
            harmonics = hr_result.get('harmonics', [])

        if np.isfinite(f0_hz) and len(harmonics) > 0:
            t_fit = t_arr[:len(Q_plot)]
            Q_fit = np.full_like(t_fit, mean_Q_plot)
            for h in harmonics:
                k = h.get('k', 1)
                A = h.get('A', 0)
                B = h.get('B', 0)
                if mean_Q < 0:
                    A, B = -A, -B
                omega_k = 2 * np.pi * k * f0_hz
                Q_fit = Q_fit + A * np.cos(omega_k * t_fit) + B * np.sin(omega_k * t_fit)
            ax2b.plot(t_fit, np.abs(Q_fit), 'r-', linewidth=2.5, alpha=0.9,
                     label=f'Harmonic fit (f₀={f0_hz:.2f} Hz)')

        # Mean line
        ax2b.axhline(mean_Q_plot, color='darkblue', linestyle='--', linewidth=2)

        # Uncertainty band (if sigma_Q available)
        if np.isfinite(sigma_Q):
            ax2b.axhspan(mean_Q_plot - sigma_Q, mean_Q_plot + sigma_Q,
                        alpha=0.2, color='blue', label=f'Mean ± σ_Q')
            ax2b.axhline(mean_Q_plot + sigma_Q, color='blue', linestyle=':', alpha=0.5)
            ax2b.axhline(mean_Q_plot - sigma_Q, color='blue', linestyle=':', alpha=0.5)

    ax2b.set_xlabel('Time [s]', fontsize=12)
    ax2b.set_ylabel('Flow Rate Q [nL/s]', fontsize=12)
    ax2b.set_title('Time-Resolved Flow Rate', fontsize=14, fontweight='bold')
    ax2b.legend(loc='upper right', fontsize=10)
    ax2b.grid(True, alpha=0.3)

    # Prominent stats box with sigma_Q
    rel_unc = 100 * sigma_Q / np.abs(mean_Q) if np.abs(mean_Q) > 0 and np.isfinite(sigma_Q) else np.nan
    stats_text2 = (
        f"Q̄ = {mean_Q:.1f} nL/s\n"
        f"σ_Q = {sigma_Q:.1f} nL/s\n"
        f"Rel. unc. = {rel_unc:.1f}%\n"
        f"χ²_red = {chi2_reduced:.2f}"
    )
    ax2b.text(0.02, 0.98, stats_text2, transform=ax2b.transAxes, fontsize=12,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='steelblue',
                      linewidth=2, alpha=0.9))

    fig2.suptitle('Figure 2: Flow Rate from Template Projection',
                  fontsize=14, fontweight='bold', y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figures
    if out_dir and vessel_id:
        out_path = Path(out_dir)
        fig1.savefig(out_path / f"flow_{vessel_id}_pres1_profile.png", dpi=DPI, bbox_inches='tight')
        fig2.savefig(out_path / f"flow_{vessel_id}_pres2_flowrate.png", dpi=DPI, bbox_inches='tight')
        print(f"  Saved presentation figures to {out_dir}")

    _ensure_interactive()
    if block:
        plt.show(block=True)
    else:
        plt.show(block=False)

    return fig1, fig2


def plot_kymographs_presentation(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
    block: bool = False,
) -> plt.Figure:
    """
    Create presentation-quality 3x3 kymograph figure.

    Shows center and ±R/2 offsets with:
    - Row 1: Kymographs (grayscale)
    - Row 2: Velocity maps (RdBu colormap)
    - Row 3: Velocity distribution histogram with median v̂(t) overlay

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel (for R_fit, r_offset)
    vessel_radius_px : float
        Vessel radius in pixels
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    out_dir : str, optional
        Directory to save figures
    vessel_id : str, optional
        Vessel identifier for filename
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig : matplotlib.Figure
        The 3x3 combined figure
    """
    from pathlib import Path

    DPI = 400

    # Get fitted radius and offset
    R_fit_px = flow_results.get('R_fit_px', vessel_radius_px)
    r_offset = flow_results.get('r_offset', 0.0)
    if not np.isfinite(R_fit_px):
        R_fit_px = vessel_radius_px
    if not np.isfinite(r_offset):
        r_offset = 0.0

    R_fit_um = R_fit_px * px_size_um

    # Find offsets at center and ±R/2
    offsets_arr = np.array([p['offset'] for p in profile_data])
    r_half = R_fit_px / 2.0

    # Target offsets: -R/2, 0, +R/2 (relative to fitted center)
    target_offsets = [r_offset - r_half, r_offset, r_offset + r_half]

    show_indices = []
    show_offsets = []
    for target in target_offsets:
        idx = np.argmin(np.abs(offsets_arr - target))
        show_indices.append(idx)
        show_offsets.append(offsets_arr[idx])

    # Time array
    T = len(profile_data[0]['v_hat'])
    t_arr = np.arange(T) * frame_dt_s

    # Convert velocities to mm/s
    v_scale = px_size_um / frame_dt_s / 1000.0  # px/frame → mm/s

    def to_mm_s(v_px_frame):
        return v_px_frame * v_scale

    # Compute y-limits based on median v_hat values only (not full distribution)
    all_vhat = []
    for idx in show_indices:
        v = to_mm_s(profile_data[idx]['v_hat'])
        finite_v = v[np.isfinite(v)]
        if len(finite_v) > 0:
            all_vhat.extend(finite_v)

    if all_vhat:
        all_vhat = np.array(all_vhat)
        v_p2, v_p98 = np.percentile(all_vhat, [2, 98])
        v_range = v_p98 - v_p2
        pad = v_range * 0.15
        v_ylim = (v_p2 - pad, v_p98 + pad)
    else:
        v_ylim = (-1, 1)

    # =====================================================================
    # COMBINED 3x3 FIGURE (no titles)
    # =====================================================================
    plt.close('pres_kymo')
    fig = plt.figure(num='pres_kymo', figsize=(12, 24))
    # Add extra column for colorbars (to the right of each row)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.5, 1], width_ratios=[1, 1, 1, 0.06],
                          hspace=0.25, wspace=0.3)

    # Font sizes for high-resolution figure - VERY LARGE for legibility
    label_fs = 28
    tick_fs = 22
    cbar_fs = 22

    # Pre-compute shared scales for velocity maps and histograms
    all_vel_maps_mm = []
    all_conf_px = []
    all_histograms = []
    n_bins = 64
    bins = np.linspace(v_ylim[0], v_ylim[1], n_bins + 1)

    COHERENCE_MASK_THRESHOLD = 0.3  # For histogram: exclude low-coherence pixels

    for idx in show_indices:
        p = profile_data[idx]
        vel_map = p.get('vel_map')
        if vel_map is not None:
            vel_map_mm = to_mm_s(vel_map)
            conf_px = p.get('conf_px')
            all_vel_maps_mm.append(vel_map_mm)
            all_conf_px.append(conf_px)

            # Build histogram (mask low coherence for histogram only)
            vel_for_hist = vel_map_mm.copy()
            if conf_px is not None and conf_px.shape == vel_map_mm.shape:
                vel_for_hist[conf_px < COHERENCE_MASK_THRESHOLD] = np.nan
            H = np.zeros((n_bins, T), float)
            for tt in range(T):
                vt = vel_for_hist[tt] if vel_for_hist.ndim > 1 else vel_for_hist
                if isinstance(vt, np.ndarray):
                    vt = vt[np.isfinite(vt)]
                    if vt.size > 0:
                        H[:, tt] = np.histogram(vt, bins=bins)[0]
            all_histograms.append(H)
        else:
            all_vel_maps_mm.append(None)
            all_conf_px.append(None)
            all_histograms.append(None)

    # Compute shared color scales: [0, 2*median(v)] or [2*median(v), 0]
    vlim_shared = (-1.0, 1.0)
    if any(v is not None for v in all_vel_maps_mm):
        all_vals = np.concatenate([v.ravel() for v in all_vel_maps_mm if v is not None])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) > 0:
            med_v = np.median(all_vals)
            if med_v >= 0:
                vlim_shared = (0.0, max(2.0 * med_v, 0.01))
            else:
                vlim_shared = (min(2.0 * med_v, -0.01), 0.0)

    hist_max = 1.0
    if any(h is not None for h in all_histograms):
        hist_max = max(np.nanmax(h) for h in all_histograms if h is not None)

    # Store image handles for colorbars
    im_vmap_last = None
    im_hist_last = None

    for col, idx in enumerate(show_indices):
        p = profile_data[idx]
        v_hat = p['v_hat']
        v_hat_mm = to_mm_s(v_hat)
        kymo = p.get('kymo_detrend', p.get('kymo'))
        vel_map_mm = all_vel_maps_mm[col]
        H = all_histograms[col]

        # Row 1: Kymograph (no title)
        ax_kymo = fig.add_subplot(gs[0, col])
        if kymo is not None:
            ax_kymo.imshow(kymo, cmap='gray', aspect='auto', interpolation='nearest')

            # Add arc restriction markers (vertical lines showing used region)
            arc_start = p.get('arc_start', 0)
            arc_end = p.get('arc_end', kymo.shape[1])
            M = kymo.shape[1]
            if arc_start > 0 or arc_end < M:
                # Draw vertical lines at arc boundaries
                ax_kymo.axvline(arc_start, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax_kymo.axvline(arc_end - 1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                # Shade unused regions
                T_kymo = kymo.shape[0]
                if arc_start > 0:
                    ax_kymo.fill_betweenx([0, T_kymo], 0, arc_start, color='gray', alpha=0.3)
                if arc_end < M:
                    ax_kymo.fill_betweenx([0, T_kymo], arc_end, M, color='gray', alpha=0.3)

        ax_kymo.set_xlabel('Arc [px]', fontsize=label_fs)
        ax_kymo.tick_params(labelsize=tick_fs)
        if col == 0:
            ax_kymo.set_ylabel('Time [frames]', fontsize=label_fs)

        # Row 2: Streaks + low-opacity heatmap + velocity-colored arrows
        ax_vmap = fig.add_subplot(gs[1, col])

        # Get restricted kymograph for underlay (matches velocity map dimensions)
        kymo_restricted = p.get('kymo_restricted')
        vel_map_raw = p.get('vel_map')  # px/frame
        conf_px_col = all_conf_px[col]

        if vel_map_mm is not None:
            # Grayscale streaks as primary background
            if kymo_restricted is not None:
                ax_vmap.imshow(kymo_restricted, cmap='gray', aspect='auto',
                               interpolation='nearest', alpha=1.0)

            # Low-opacity velocity heatmap overlay (cap alpha so streaks show through)
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vlim_shared[0], vmax=vlim_shared[1])
            cmap_rdb = plt.cm.RdBu_r
            rgba = cmap_rdb(norm(np.nan_to_num(vel_map_mm, nan=0.0)))
            if conf_px_col is not None and conf_px_col.shape == vel_map_mm.shape:
                rgba[..., 3] = np.clip(conf_px_col * 0.4, 0, 0.4)
            else:
                rgba[..., 3] = 0.3
            rgba[~np.isfinite(vel_map_mm), 3] = 0.0
            ax_vmap.imshow(rgba, aspect='auto', interpolation='nearest')
            # Invisible scalar image for colorbar
            im_vmap_last = ax_vmap.imshow(np.full_like(vel_map_mm, np.nan),
                                          cmap='RdBu_r', aspect='auto',
                                          vmin=vlim_shared[0], vmax=vlim_shared[1])

            # Velocity-colored arrows (denser since this is the restricted arc)
            if vel_map_raw is not None:
                _overlay_gst_arrows(ax_vmap, vel_map_raw, conf_px_col,
                                    color='velocity', coh_min=0.1,
                                    vlim=vlim_shared, v_scale=v_scale)

        ax_vmap.set_xlabel('Arc [px]', fontsize=label_fs)
        ax_vmap.tick_params(labelsize=tick_fs)
        if col == 0:
            ax_vmap.set_ylabel('Time [frames]', fontsize=label_fs)

        # Row 3: Velocity distribution histogram with shared scale
        ax_vel = fig.add_subplot(gs[2, col])

        if H is not None:
            extent = [0.0, t_arr[-1], v_ylim[0], v_ylim[1]]
            im_hist_last = ax_vel.imshow(H, origin='lower', aspect='auto', extent=extent,
                                         cmap='viridis', alpha=0.8, vmin=0, vmax=hist_max)
            ax_vel.plot(t_arr, v_hat_mm, lw=2.0, alpha=1.0, color='cyan')
            ax_vel.set_ylim(v_ylim)
        else:
            ax_vel.plot(t_arr, v_hat_mm, '-', color='cyan', linewidth=1.5, alpha=0.9)
            ax_vel.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax_vel.set_ylim(v_ylim)
            ax_vel.grid(True, alpha=0.3)

        ax_vel.set_xlim(0, t_arr[-1])
        ax_vel.set_xlabel('Time [s]', fontsize=label_fs)
        ax_vel.tick_params(labelsize=tick_fs)
        if col == 0:
            ax_vel.set_ylabel('Velocity [mm/s]', fontsize=label_fs)

    # Add vertical colorbars in the 4th column, aligned with their rows
    # Row 2 colorbar: velocity map (RdBu)
    if im_vmap_last is not None:
        ax_cbar_vmap = fig.add_subplot(gs[1, 3])
        cbar_vmap = fig.colorbar(im_vmap_last, cax=ax_cbar_vmap, orientation='vertical')
        cbar_vmap.set_label('v [mm/s]', fontsize=cbar_fs)
        cbar_vmap.ax.tick_params(labelsize=tick_fs)

    # Row 3 colorbar: histogram counts (viridis)
    if im_hist_last is not None:
        ax_cbar_hist = fig.add_subplot(gs[2, 3])
        cbar_hist = fig.colorbar(im_hist_last, cax=ax_cbar_hist, orientation='vertical')
        cbar_hist.set_label('Counts', fontsize=cbar_fs)
        cbar_hist.ax.tick_params(labelsize=tick_fs)

    fig.tight_layout()

    # Save figure
    if out_dir and vessel_id:
        out_path = Path(out_dir)
        fig.savefig(out_path / f"flow_{vessel_id}_slide_kymographs_radial_offsets.png",
                   dpi=DPI, bbox_inches='tight')
        print(f"  Saved slide_kymographs_radial_offsets")

    # =====================================================================
    # SEPARATE CENTERLINE KYMOGRAPH + ARROWS FIGURE
    # =====================================================================
    center_idx = show_indices[1]  # centerline band
    p_ctr = profile_data[center_idx]
    kymo_ctr = p_ctr.get('kymo_detrend', p_ctr.get('kymo'))
    vel_map_ctr = p_ctr.get('vel_map')  # px/frame, restricted to arc
    conf_px_ctr = all_conf_px[1]
    kymo_restricted_ctr = p_ctr.get('kymo_restricted')

    if kymo_ctr is not None and vel_map_ctr is not None:
        plt.close('centerline_arrows')
        fig_ctr = plt.figure(num='centerline_arrows', figsize=(10, 14))
        ax_ctr = fig_ctr.add_subplot(111)

        # Grayscale restricted kymograph
        if kymo_restricted_ctr is not None:
            ax_ctr.imshow(kymo_restricted_ctr, cmap='gray', aspect='auto',
                          interpolation='nearest')

        # Low-opacity velocity heatmap
        from matplotlib.colors import Normalize as _NormCtr
        vel_map_ctr_mm = to_mm_s(vel_map_ctr)
        norm_ctr = _NormCtr(vmin=vlim_shared[0], vmax=vlim_shared[1])
        rgba_ctr = plt.cm.RdBu_r(norm_ctr(np.nan_to_num(vel_map_ctr_mm, nan=0.0)))
        if conf_px_ctr is not None and conf_px_ctr.shape == vel_map_ctr_mm.shape:
            rgba_ctr[..., 3] = np.clip(conf_px_ctr * 0.4, 0, 0.4)
        else:
            rgba_ctr[..., 3] = 0.3
        rgba_ctr[~np.isfinite(vel_map_ctr_mm), 3] = 0.0
        ax_ctr.imshow(rgba_ctr, aspect='auto', interpolation='nearest')

        # Invisible scalar image for colorbar
        im_ctr = ax_ctr.imshow(np.full_like(vel_map_ctr_mm, np.nan),
                               cmap='RdBu_r', aspect='auto',
                               vmin=vlim_shared[0], vmax=vlim_shared[1])
        plt.colorbar(im_ctr, ax=ax_ctr, label='v [mm/s]', shrink=0.6)

        # Higher density arrows for this standalone view
        _overlay_gst_arrows(ax_ctr, vel_map_ctr, conf_px_ctr,
                            color='velocity', coh_min=0.1,
                            n_arrows_x=15, n_arrows_t=25,
                            vlim=vlim_shared, v_scale=v_scale)

        ax_ctr.set_xlabel('Arc [px]', fontsize=14)
        ax_ctr.set_ylabel('Time [frames]', fontsize=14)
        ax_ctr.set_title('Centerline kymograph + GST arrows', fontsize=16)
        fig_ctr.tight_layout()

        if out_dir and vessel_id:
            fig_ctr.savefig(out_path / f"flow_{vessel_id}_centerline_arrows.png",
                            dpi=DPI, bbox_inches='tight')
            print(f"  Saved centerline_arrows")

    _ensure_interactive()
    if block:
        plt.show(block=True)
    else:
        plt.show(block=False)

    return fig


def plot_slide_velocity_profile_fit(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
    block: bool = False,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create two slide figures for velocity profile fitting.

    Figure 1 (slide_velocity_profile_fit):
        v̄(r) profile with σ_v̄ error bars, Poiseuille fit, χ²_profile

    Figure 2 (slide_cycle_uncertainty):
        v(t) for center band showing cycle segmentation and per-cycle means
        to illustrate how σ_v̄ is computed

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel
    vessel_radius_px : float
        Vessel radius in pixels
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    out_dir : str, optional
        Directory to save figures
    vessel_id : str, optional
        Vessel identifier for filename
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig1, fig2 : matplotlib.Figure
    """
    from pathlib import Path

    DPI = 600

    # Font sizes — compact for small-figure readability
    label_fs = 12
    tick_fs = 10
    legend_fs = 8
    annot_fs = 9

    # Extract key results
    v_max = flow_results.get('v_max', np.nan)
    r_offset = flow_results.get('r_offset', 0.0)
    chi2_reduced = flow_results.get('chi2_reduced', np.nan)
    f0_hz = flow_results.get('f0_hz', np.nan)

    # Radius info
    R_fit_px = flow_results.get('R_fit_px', np.nan)
    effective_radius_px = R_fit_px if np.isfinite(R_fit_px) else vessel_radius_px
    effective_radius_um = effective_radius_px * px_size_um

    # Profile data with cycle-based uncertainty
    sigma_v_mean = flow_results.get('sigma_v_mean', np.array([]))
    offsets_fit_px = flow_results.get('offsets_fit_px', np.array([]))
    v_means_fit = flow_results.get('v_means_fit', np.array([]))

    # Use NLLS fit data if available
    if len(sigma_v_mean) > 0 and len(offsets_fit_px) > 0:
        offsets_arr = offsets_fit_px
        v_means = v_means_fit  # um/s
        v_errs = sigma_v_mean  # um/s
        v_means_mm = v_means / 1000.0
        v_errs_mm = v_errs / 1000.0
    else:
        # Fallback to profile_data
        offsets_arr = np.array([p['offset'] for p in profile_data])
        v_means = np.array([p['v_mean'] for p in profile_data])
        v_stds = np.array([p['v_std'] for p in profile_data])
        v_means_mm = v_means * px_size_um / frame_dt_s / 1000.0
        v_errs_mm = v_stds * px_size_um / frame_dt_s / 1000.0

    if not np.isfinite(r_offset):
        r_offset = 0.0

    # =====================================================================
    # FIGURE 1: Velocity Profile Fit
    # =====================================================================
    plt.close('slide_profile')
    fig1 = plt.figure(num='slide_profile', figsize=(2.5, 2))
    ax1 = fig1.add_subplot(111)

    # --- Pre-compute envelope band membership ---
    envelope_info = flow_results.get('envelope_info', None)
    in_band = np.ones(len(offsets_arr), dtype=bool)  # default: all pass
    env_R_px_plot = effective_radius_px
    env_vmax_plot = v_max
    env_tol_plot = 0.0

    if envelope_info is not None:
        env_vmax_plot = envelope_info['v_max']
        env_R_px_plot = envelope_info['R_px']
        env_r0_px_plot = envelope_info.get('r_offset_px', 0.0)
        env_tol_plot = envelope_info.get('tolerance_um_s', 0.0)

        shape_at_data = np.maximum(0, 1 - ((offsets_arr - env_r0_px_plot) / env_R_px_plot) ** 2)
        v_env_at_data = np.abs(env_vmax_plot) * shape_at_data  # um/s
        v_abs = np.abs(v_means)  # um/s

        in_band = ((v_abs <= v_env_at_data + env_tol_plot) &
                   (v_abs >= np.maximum(0, v_env_at_data - env_tol_plot)))

    # Identify inside vs outside vessel
    r_offset_px = r_offset
    dist_from_center = np.abs(offsets_arr - r_offset_px)
    inside_mask = dist_from_center <= effective_radius_px

    # Combined masks
    in_band_inside = inside_mask & in_band
    in_R_not_band = inside_mask & ~in_band
    outside_mask = ~inside_mask

    # --- Show excluded bands as gray x markers ---
    scale_pf = px_size_um / frame_dt_s  # px/frame → um/s
    filtered_offsets_set = set(np.round(offsets_arr, 1).tolist())
    excl_r, excl_v = [], []
    for p in profile_data:
        offset = p['offset']
        if round(float(offset), 1) not in filtered_offsets_set:
            v_hat_p = p.get('v_hat')
            if v_hat_p is not None and len(v_hat_p) > 0:
                excl_r.append(offset * px_size_um)
                excl_v.append(np.abs(np.nanmean(v_hat_p) * scale_pf / 1000.0))
    if excl_r:
        ax1.plot(excl_r, excl_v, 'x', markersize=5, color='silver', alpha=0.4, zorder=1)

    # --- Plot data points ---
    if np.any(in_band_inside):
        ax1.errorbar(offsets_arr[in_band_inside] * px_size_um,
                     np.abs(v_means_mm[in_band_inside]),
                     yerr=v_errs_mm[in_band_inside],
                     fmt='o', markersize=5, color='forestgreen', capsize=3, capthick=1,
                     elinewidth=1, zorder=3)

    if np.any(in_R_not_band):
        ax1.errorbar(offsets_arr[in_R_not_band] * px_size_um,
                     np.abs(v_means_mm[in_R_not_band]),
                     yerr=v_errs_mm[in_R_not_band],
                     fmt='s', markersize=4, color='steelblue', capsize=3, capthick=1,
                     elinewidth=1, zorder=3)

    if np.any(outside_mask):
        ax1.errorbar(offsets_arr[outside_mask] * px_size_um,
                     np.abs(v_means_mm[outside_mask]),
                     yerr=v_errs_mm[outside_mask],
                     fmt='o', markersize=4, markerfacecolor='none', markeredgecolor='gray',
                     ecolor='gray', capsize=2, alpha=0.4, zorder=2)

    # --- Poiseuille fit ---
    if np.isfinite(v_max):
        r_fit = np.linspace(-effective_radius_px, effective_radius_px, 100)
        v_max_mm = v_max / 1000.0
        r_norm = (r_fit - r_offset) / effective_radius_px
        r_norm_clipped = np.clip(np.abs(r_norm), 0.0, 1.0)
        v_fit = np.abs(v_max_mm) * (1 - r_norm_clipped**2)
        ax1.plot(r_fit * px_size_um, v_fit, 'r-', linewidth=1.5)

    # --- Vessel boundary lines (no labels) ---
    r_offset_um = r_offset * px_size_um
    ax1.axvline(-effective_radius_um + r_offset_um, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(effective_radius_um + r_offset_um, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    segmented_radius_um = vessel_radius_px * px_size_um
    ax1.axvline(-segmented_radius_um, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axvline(segmented_radius_um, color='orange', linestyle=':', linewidth=1, alpha=0.5)

    # --- Poiseuille envelope with both upper and lower bands ---
    if envelope_info is not None:
        r_env = np.linspace(env_r0_px_plot - env_R_px_plot,
                            env_r0_px_plot + env_R_px_plot, 200)
        shape_env = np.maximum(0, 1 - ((r_env - env_r0_px_plot) / env_R_px_plot) ** 2)
        v_env_mm = np.abs(env_vmax_plot) / 1000.0 * shape_env
        v_env_upper_mm = v_env_mm + env_tol_plot / 1000.0
        v_env_lower_mm = np.maximum(0, v_env_mm - env_tol_plot / 1000.0)

        ax1.plot(r_env * px_size_um, v_env_mm, color='forestgreen', linestyle='--',
                 linewidth=1, alpha=0.5)
        ax1.fill_between(r_env * px_size_um, v_env_lower_mm, v_env_upper_mm,
                         color='forestgreen', alpha=0.08)

    # --- Labels (minimal) ---
    ax1.set_xlabel(r'r ($\mu$m)', fontsize=label_fs)
    ax1.set_ylabel(r'$|\bar{v}|$ (mm/s)', fontsize=label_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.grid(True, alpha=0.2)

    fig1.tight_layout()

    # =====================================================================
    # FIGURE 2: Harmonic Fit Uncertainty
    # =====================================================================
    plt.close('slide_cycles')
    fig2 = plt.figure(num='slide_cycles', figsize=(3, 2))

    # Two subplots: top = v(t) with fit, bottom = residuals
    ax2a = fig2.add_subplot(211)
    ax2b = fig2.add_subplot(212, sharex=ax2a)

    # Find center band (closest to r_offset)
    center_idx = np.argmin(np.abs(offsets_arr - r_offset))
    center_p = profile_data[center_idx]
    v_hat = center_p['v_hat']

    # Convert to mm/s
    scale = px_size_um / frame_dt_s / 1000.0
    v_hat_mm = v_hat * scale

    T = len(v_hat)
    t_arr = np.arange(T) * frame_dt_s

    # Fit harmonics to get signal and residuals
    if np.isfinite(f0_hz) and f0_hz > 0:
        from ..analysis.harmonic import fit_harmonics
        hr_result = fit_harmonics(v_hat, frame_dt_s, f0_hz, K=3, loss="huber", include_dc=True)

        v_fit = hr_result.get('signal', np.full_like(v_hat, np.nan))
        v_resid = hr_result.get('resid', v_hat - np.nanmean(v_hat))
        v_mean = hr_result.get('a0', np.nanmean(v_hat))

        # Get amplitude from first harmonic
        v_amp = 0.0
        for h in hr_result.get('harmonics', []):
            if h.get('k') == 1:
                v_amp = h.get('amp', 0.0)
                break

        v_fit_mm = v_fit * scale
        v_resid_mm = v_resid * scale
        v_mean_mm = v_mean * scale
        v_amp_mm = v_amp * scale

        # RMS of residuals = σ_v
        sigma_residual = np.sqrt(np.nanmean(v_resid**2))
        sigma_residual_mm = sigma_residual * scale

        # TOP PLOT: v(t) with harmonic fit
        ax2a.plot(t_arr, v_hat_mm, 'b-', linewidth=0.8, alpha=0.4)
        ax2a.plot(t_arr, v_fit_mm, 'r-', linewidth=1.2)
        ax2a.axhline(v_mean_mm, color='darkred', linestyle='--', linewidth=0.8, alpha=0.5)

        ax2a.set_ylabel('v (mm/s)', fontsize=label_fs)
        ax2a.tick_params(labelsize=tick_fs)
        ax2a.grid(True, alpha=0.2)

        # BOTTOM PLOT: Residuals
        ax2b.plot(t_arr, v_resid_mm, 'gray', linewidth=0.8, alpha=0.6)
        ax2b.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2b.axhspan(-sigma_residual_mm, sigma_residual_mm, alpha=0.15, color='red')

        ax2b.set_xlabel('Time (s)', fontsize=label_fs)
        ax2b.set_ylabel('Residual (mm/s)', fontsize=label_fs)
        ax2b.tick_params(labelsize=tick_fs)
        ax2b.grid(True, alpha=0.2)

    else:
        # Fallback: just plot raw data
        ax2a.plot(t_arr, v_hat_mm, 'b-', linewidth=1, alpha=0.6)
        ax2a.set_ylabel('v (mm/s)', fontsize=label_fs)

    ax2a.set_xlim(0, t_arr[-1])
    fig2.tight_layout()

    # Save figures
    if out_dir and vessel_id:
        out_path = Path(out_dir)
        for ext in ('png', 'pdf'):
            kw = dict(bbox_inches='tight')
            if ext == 'png':
                kw['dpi'] = DPI
            fig1.savefig(out_path / f"flow_{vessel_id}_slide_velocity_profile_fit.{ext}", **kw)
            fig2.savefig(out_path / f"flow_{vessel_id}_slide_cycle_uncertainty.{ext}", **kw)
        print(f"  Saved slide_velocity_profile_fit and slide_cycle_uncertainty (PNG+PDF)")

    _ensure_interactive()
    if block:
        plt.show(block=True)
    else:
        plt.show(block=False)

    return fig1, fig2


def plot_simple_velocity_profile(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
    block: bool = False,
) -> plt.Figure:
    """
    Clean, glanceable velocity profile: parabolic fit, data + error bars,
    vessel walls, massive labels. No grid, no envelope, no categories.
    """
    from pathlib import Path

    # --- Data ---
    sigma_v_mean = flow_results.get('sigma_v_mean', np.array([]))
    offsets_fit_px = flow_results.get('offsets_fit_px', np.array([]))
    v_means_fit = flow_results.get('v_means_fit', np.array([]))
    v_max = flow_results.get('v_max', np.nan)
    r_offset = flow_results.get('r_offset', 0.0)
    R_fit_px = flow_results.get('R_fit_px', np.nan)
    effective_radius_px = R_fit_px if np.isfinite(R_fit_px) else vessel_radius_px

    if len(sigma_v_mean) > 0 and len(offsets_fit_px) > 0:
        r_px = offsets_fit_px
        v_mm = np.abs(v_means_fit) / 1000.0
        v_err_mm = sigma_v_mean / 1000.0
    else:
        r_px = np.array([p['offset'] for p in profile_data])
        v_raw = np.array([p['v_mean'] for p in profile_data])
        v_std = np.array([p['v_std'] for p in profile_data])
        scale = px_size_um / frame_dt_s
        v_mm = np.abs(v_raw) * scale / 1000.0
        v_err_mm = v_std * scale / 1000.0

    if not np.isfinite(r_offset):
        r_offset = 0.0

    r_um = r_px * px_size_um
    R_um = effective_radius_px * px_size_um
    r_off_um = r_offset * px_size_um

    # --- Figure ---
    plt.close('simple_profile')
    fig, ax = plt.subplots(num='simple_profile', figsize=(4, 3))

    # Data with error bars
    ax.errorbar(r_um, v_mm, yerr=v_err_mm,
                fmt='o', markersize=6, color='#2563EB', capsize=4, capthick=1.5,
                elinewidth=1.5, markeredgecolor='#2563EB', markerfacecolor='#60A5FA',
                zorder=3, label='Data')

    # Poiseuille fit — 1-param least-squares v_max for vessel_radius_px
    # (The NLLS v_max is fit with R_fit which may differ from R_seg)
    R_plot_px = vessel_radius_px  # use segmented radius for simple view
    R_plot_um = R_plot_px * px_size_um
    inside = np.abs(r_px - r_offset) <= R_plot_px
    if np.sum(inside) >= 2:
        r_in = r_px[inside]
        v_in = np.abs(v_means_fit[inside]) if len(v_means_fit) > 0 else np.abs(
            np.array([p['v_mean'] for p in profile_data]))[inside] * px_size_um / frame_dt_s
        basis = np.maximum(0, 1 - ((r_in - r_offset) / R_plot_px)**2)
        v_max_simple = float(np.dot(v_in, basis) / np.dot(basis, basis))  # um/s
    elif np.isfinite(v_max):
        v_max_simple = abs(v_max)
    else:
        v_max_simple = np.nan

    if np.isfinite(v_max_simple):
        r_fit = np.linspace(r_offset - R_plot_px * 1.05,
                            r_offset + R_plot_px * 1.05, 300)
        r_norm = (r_fit - r_offset) / R_plot_px
        v_fit = v_max_simple / 1000.0 * np.maximum(0, 1 - r_norm**2)
        ax.plot(r_fit * px_size_um, v_fit, '-', color='#DC2626', linewidth=2.5,
                zorder=2)

    # Vessel walls — bold, prominent (use segmented radius)
    wall_kw = dict(color='#1E293B', linewidth=2.5, linestyle='-', alpha=0.8, zorder=4)
    ax.axvline(-R_plot_um + r_off_um, **wall_kw)
    ax.axvline(R_plot_um + r_off_um, **wall_kw)

    # Light shading outside vessel walls
    xlim_lo = min(r_um.min(), -R_plot_um + r_off_um) - 10
    xlim_hi = max(r_um.max(), R_plot_um + r_off_um) + 10
    ax.axvspan(xlim_lo, -R_plot_um + r_off_um, color='#E2E8F0', alpha=0.5, zorder=0)
    ax.axvspan(R_plot_um + r_off_um, xlim_hi, color='#E2E8F0', alpha=0.5, zorder=0)

    # Massive labels
    ax.set_xlabel(r'$r$ ($\mu$m)', fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$|\bar{v}|$ (mm/s)', fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=14, width=1.5, length=6)

    # Clean styling
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    fig.tight_layout()

    # Save
    if out_dir and vessel_id:
        out_path = Path(out_dir)
        for ext in ('png', 'pdf'):
            kw = dict(bbox_inches='tight')
            if ext == 'png':
                kw['dpi'] = 600
            fig.savefig(out_path / f"flow_{vessel_id}_simple_profile.{ext}", **kw)
        print(f"  Saved simple_profile (PNG+PDF)")

    _ensure_interactive()
    if block:
        plt.show(block=True)
    else:
        plt.show(block=False)

    return fig


def plot_slide_flow_rate(
    profile_data: List[Dict[str, Any]],
    flow_results: Dict[str, Any],
    vessel_radius_px: float,
    px_size_um: float = PX_SIZE_UM,
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
    block: bool = False,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create two slide figures for flow rate estimation.

    Figure 1 (slide_flowrate_Qt): Q(t) with overlaid band contributions showing weighted sum
    Figure 2 (slide_flowrate_spectrum): Power spectrum showing heart rate harmonics

    Parameters
    ----------
    profile_data : list of dict
        Output from compute_radial_velocity_profiles
    flow_results : dict
        Output from analyze_vessel
    vessel_radius_px : float
        Vessel radius in pixels
    px_size_um : float
        Pixel size in microns
    frame_dt_s : float
        Frame interval in seconds
    out_dir : str, optional
        Directory to save figures
    vessel_id : str, optional
        Vessel identifier for filename
    block : bool
        Whether to block on plt.show()

    Returns
    -------
    fig1, fig2 : matplotlib.Figure
    """
    from pathlib import Path
    from scipy.signal import welch

    DPI = 600

    # Font sizes — compact for small-figure readability
    label_fs = 12
    tick_fs = 10
    legend_fs = 9

    # Extract key results (already in nL/s)
    Q_t = flow_results.get('Q_t', np.array([]))
    mean_Q = flow_results.get('mean_Q', np.nan)
    sigma_Q = flow_results.get('sigma_Q', np.nan)
    f0_hz = flow_results.get('f0_hz', np.nan)

    T = len(Q_t) if len(Q_t) > 0 else len(profile_data[0]['v_hat'])
    t_arr = np.arange(T) * frame_dt_s

    # =====================================================================
    # FIGURE 1: Q(t) time series with harmonic fit
    # =====================================================================
    plt.close('slide_Qt')
    fig1 = plt.figure(num='slide_Qt', figsize=(3, 1.5))
    ax1 = fig1.add_subplot(111)

    # Get centerline v(t) - find band closest to center
    offsets = np.array([p['offset'] for p in profile_data])
    center_idx = np.argmin(np.abs(offsets))
    v_center = profile_data[center_idx].get('v_hat', np.array([]))

    R_fit_px = flow_results.get('R_fit_px', vessel_radius_px)
    if not np.isfinite(R_fit_px):
        R_fit_px = vessel_radius_px

    if len(v_center) > 0 and len(Q_t) > 0:
        v_center_um_s = v_center * px_size_um / frame_dt_s
        R_um = R_fit_px * px_size_um
        Q_center = (np.pi * R_um**2 / 2) * v_center_um_s / 1e6
        ax1.plot(t_arr[:len(Q_center)], Q_center,
                color='gray', linewidth=0.8, alpha=0.4)

    if len(Q_t) > 0 and np.any(np.isfinite(Q_t)):
        ax1.plot(t_arr[:len(Q_t)], Q_t, 'b-', linewidth=1, alpha=0.7)

        harmonics = flow_results.get('harmonics', [])
        if len(harmonics) == 0 and np.isfinite(f0_hz) and len(Q_t) > 32:
            from .harmonic import fit_harmonics
            hr_result = fit_harmonics(Q_t, frame_dt_s, f0_hz, K=3)
            harmonics = hr_result.get('harmonics', [])

        if np.isfinite(f0_hz) and len(harmonics) > 0:
            t_fit = t_arr[:len(Q_t)]
            Q_fit = np.full_like(t_fit, mean_Q)
            for h in harmonics:
                k = h.get('k', 1)
                A = h.get('A', 0)
                B = h.get('B', 0)
                omega_k = 2 * np.pi * k * f0_hz
                Q_fit = Q_fit + A * np.cos(omega_k * t_fit) + B * np.sin(omega_k * t_fit)
            ax1.plot(t_fit, Q_fit, 'r-', linewidth=1.5, alpha=0.9)

        # Show both DC estimates: nanmean (used) and harmonic a0 (for comparison)
        ax1.axhline(mean_Q, color='darkblue', linestyle='--', linewidth=1, alpha=0.6,
                     label=f'nanmean = {mean_Q:.3f}')
        # Compute harmonic a0 for comparison
        if np.isfinite(f0_hz) and len(Q_t) > 32:
            from .harmonic import fit_harmonics as _fit_h
            try:
                _hr = _fit_h(Q_t, frame_dt_s, f0_hz, K=3, loss="huber", include_dc=True)
                _a0 = _hr.get('a0', np.nan)
                if np.isfinite(_a0) and abs(_a0 - mean_Q) > 1e-6:
                    ax1.axhline(_a0, color='red', linestyle=':', linewidth=1, alpha=0.6,
                                label=f'Huber a₀ = {_a0:.3f}')
            except Exception:
                pass
        ax1.legend(fontsize=7, loc='upper right')
        if np.isfinite(sigma_Q):
            ax1.axhspan(mean_Q - sigma_Q, mean_Q + sigma_Q,
                       alpha=0.1, color='blue')

    ax1.set_xlabel('Time (s)', fontsize=label_fs)
    ax1.set_ylabel('Q (nL/s)', fontsize=label_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_xlim(0, t_arr[-1])
    ax1.grid(True, alpha=0.2)

    fig1.tight_layout()

    # =====================================================================
    # FIGURE 2: Power Spectrum
    # =====================================================================
    plt.close('slide_spectrum')
    fig2, ax2 = plt.subplots(num='slide_spectrum', figsize=(2, 1.5))

    if len(Q_t) > 0 and np.isfinite(f0_hz):
        fs = 1.0 / frame_dt_s
        Q_centered = Q_t - np.nanmean(Q_t)
        nperseg = min(512, len(Q_t)//2)
        nfft = max(2048, nperseg * 4)
        freqs, psd = welch(Q_centered, fs=fs, nperseg=nperseg, nfft=nfft)

        max_psd = np.max(psd[psd > 0]) if np.any(psd > 0) else 1.0
        psd_norm = psd / max_psd

        ax2.semilogy(freqs, psd_norm, 'b-', linewidth=1.2)
        ax2.set_ylim(1e-4, 3.0)
        ax2.set_xlim(0, min(12, freqs[-1]))

        # Mark harmonics
        for h in range(1, 4):
            freq_h = h * f0_hz
            ax2.axvline(freq_h, color='red', linestyle='--', alpha=0.5, linewidth=1)

        ax2.set_xlabel('Frequency (Hz)', fontsize=label_fs)
        ax2.set_ylabel('Normalized PSD', fontsize=label_fs)
        ax2.tick_params(labelsize=tick_fs)
        ax2.grid(True, alpha=0.2)

    fig2.tight_layout()

    # Save figures
    if out_dir and vessel_id:
        out_path = Path(out_dir)
        for ext in ('png', 'pdf'):
            kw = dict(bbox_inches='tight')
            if ext == 'png':
                kw['dpi'] = DPI
            fig1.savefig(out_path / f"flow_{vessel_id}_slide_flowrate_Qt.{ext}", **kw)
            fig2.savefig(out_path / f"flow_{vessel_id}_slide_flowrate_spectrum.{ext}", **kw)
        print(f"  Saved slide_flowrate_Qt, slide_flowrate_spectrum (PNG+PDF)")

    _ensure_interactive()
    if block:
        plt.show(block=True)
    else:
        plt.show(block=False)

    return fig1, fig2


def plot_Qt_clean(
    flow_results: Dict[str, Any],
    frame_dt_s: float = FRAME_DT_S,
    out_dir: Optional[str] = None,
    vessel_id: Optional[str] = None,
    block: bool = False,
) -> plt.Figure:
    """Clean Q(t) time series with 3-harmonic fit overlay.

    Minimal, publication-quality figure: raw Q(t) in light blue,
    harmonic fit in red, proper axis labels with units, no legend clutter.
    """
    from pathlib import Path

    Q_t = np.asarray(flow_results.get('Q_t', []), dtype=float)
    mean_Q = flow_results.get('mean_Q', np.nan)
    f0_hz = flow_results.get('f0_hz', np.nan)

    if len(Q_t) == 0 or not np.any(np.isfinite(Q_t)):
        return None

    # Sign-flip so Q(t) is positive (canonical convention)
    flipped = np.nanmean(Q_t) < 0
    if flipped:
        Q_t = -Q_t
        mean_Q = -mean_Q

    t_arr = np.arange(len(Q_t)) * frame_dt_s

    plt.close('Qt_clean')
    fig, ax = plt.subplots(num='Qt_clean', figsize=(3, 1.25))

    # Raw Q(t)
    ax.plot(t_arr, Q_t, color='#6baed6', linewidth=0.4, alpha=0.8, label='Q(t)')

    # 3-harmonic fit — always refit from (sign-corrected) Q_t
    # because stored harmonics may have been computed before sign convention
    if np.isfinite(f0_hz) and len(Q_t) > 32:
        from .harmonic import fit_harmonics
        hr_result = fit_harmonics(Q_t, frame_dt_s, f0_hz, K=3)
        harmonics = hr_result.get('harmonics', [])
        a0 = hr_result.get('a0', mean_Q)

        if len(harmonics) > 0:
            Q_fit = np.full_like(t_arr, a0)
            for h in harmonics:
                k = h.get('k', 1)
                A = h.get('A', 0)
                B = h.get('B', 0)
                omega_k = 2 * np.pi * k * f0_hz
                Q_fit += A * np.cos(omega_k * t_arr) + B * np.sin(omega_k * t_arr)
            ax.plot(t_arr, Q_fit, color='#e31a1c', linewidth=1.0, alpha=0.9,
                    label=f'Harmonic fit ({len(harmonics)}H)')

    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Q (nL/s)', fontsize=8)
    ax.set_xlim(0, t_arr[-1])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='best', framealpha=0.7)

    fig.tight_layout()

    if out_dir and vessel_id:
        out_path = Path(out_dir)
        for ext in ('png', 'pdf'):
            kw = dict(bbox_inches='tight')
            if ext == 'png':
                kw['dpi'] = 600
            fig.savefig(out_path / f"flow_{vessel_id}_Qt_clean.{ext}", **kw)
        print(f"  Saved Qt_clean (PNG+PDF)")

    _ensure_interactive()
    plt.show(block=block)
    return fig


# =========================================================================
# Kymograph Diagnostics
# =========================================================================


def _overlay_gst_arrows(
    ax, vel_map, conf_px, x_offset: int = 0,
    coh_min: float = 0.2, color: str = 'yellow', alpha: float = 0.5,
    n_arrows_x: int = 12, n_arrows_t: int = 18,
    vlim=None, v_scale: float = 1.0,
):
    """Overlay quiver arrows showing GST streak direction on a kymograph axis.

    Args:
        ax: Matplotlib axis (y = time increasing downward).
        vel_map: (T, M) velocity in px/frame from GST.
        conf_px: (T, M) coherence from GST.
        x_offset: Added to x indices to map into full-kymograph coordinates.
        coh_min: Minimum coherence for drawing an arrow.
        color: Arrow color string, or 'velocity' to color by velocity using RdBu_r.
        alpha: Arrow transparency (used when color != 'velocity'; when 'velocity',
            alpha is proportional to coherence).
        n_arrows_x: Target number of arrows across the spatial dimension.
        n_arrows_t: Target number of arrows across the temporal dimension.
        vlim: (vmin, vmax) for velocity colormap when color='velocity' (in mm/s).
        v_scale: Conversion factor px/frame → mm/s (only used when color='velocity').
    """
    from matplotlib.colors import Normalize as _Norm

    T_vm, M_vm = vel_map.shape
    step_x = max(1, int(round(M_vm / n_arrows_x)))
    step_t = max(1, int(round(T_vm / n_arrows_t)))
    gx_arr = np.arange(step_x // 2, M_vm, step_x)
    gt_arr = np.arange(step_t // 2, T_vm, step_t)
    gx, gt = np.meshgrid(gx_arr, gt_arr)
    vel_sub = vel_map[gt, gx]
    coh_sub = conf_px[gt, gx] if conf_px is not None else np.ones_like(vel_sub)
    # Normalize to unit-length arrows: (dx, dt) = (v, 1) / ||(v, 1)||
    arrow_len = np.sqrt(vel_sub**2 + 1.0)
    dx = vel_sub / arrow_len
    dt = 1.0 / arrow_len   # always positive → always points down
    x_plot = gx + x_offset
    t_plot = gt.astype(float)
    mask = (coh_sub > coh_min) & np.isfinite(vel_sub)
    if not mask.any():
        return

    if color == 'velocity':
        # Color each arrow by its velocity using RdBu_r
        vel_mm = vel_sub * v_scale
        if vlim is None:
            vlim = (-1.0, 1.0)
        norm = _Norm(vmin=vlim[0], vmax=vlim[1])
        cmap = plt.cm.RdBu_r
        arrow_colors = cmap(norm(np.nan_to_num(vel_mm, nan=0.0)))
        # Alpha proportional to coherence
        arrow_colors[..., 3] = np.clip(coh_sub, 0.3, 1.0)
        # Black outline for contrast, then colored arrows on top
        ax.quiver(
            x_plot[mask], t_plot[mask], dx[mask], dt[mask],
            angles='xy', scale=25, width=0.004,
            headwidth=3.5, headlength=3, pivot='mid',
            color='black', alpha=0.8,
        )
        ax.quiver(
            x_plot[mask], t_plot[mask], dx[mask], dt[mask],
            angles='xy', scale=25, width=0.003,
            headwidth=3.5, headlength=3, pivot='mid',
            color=arrow_colors[mask],
        )
    else:
        ax.quiver(
            x_plot[mask], t_plot[mask], dx[mask], dt[mask],
            angles='xy', scale=45, width=0.003,
            headwidth=3, headlength=2.5, pivot='mid',
            color=color, alpha=alpha,
        )


def plot_kymo_diagnostics(
    kymo: np.ndarray,
    vel_map: np.ndarray,
    conf_px: np.ndarray,
    arc_start: int,
    arc_end: int,
    quality: np.ndarray,
    column_mask: Optional[np.ndarray] = None,
    v_hat: Optional[np.ndarray] = None,
    f0_hz: Optional[float] = None,
    frame_dt_s: float = FRAME_DT_S,
    px_size_um: float = PX_SIZE_UM,
    vessel_id: str = "",
    out_dir: Optional[str] = None,
    block: bool = False,
    per_window_gst: Optional[Dict[int, Dict]] = None,
    per_window_vhat: Optional[Dict[int, np.ndarray]] = None,
    vel_refinement: Optional[Dict] = None,
    radon_results: Optional[Dict] = None,
) -> plt.Figure:
    """
    Comprehensive kymograph diagnostic figure for debugging arc selection.

    Layout:
        Row 1: Raw kymograph with arc region boundaries
        Rows 2..N+1: Per-window GST velocity (alpha=coherence) if per_window_gst provided
        Row N+2: Combined best-of velocity map (alpha=coherence)
        Row N+3: SVD+Radon velocity strip (if radon_results provided)
        Row N+4: Radon variance profiles at sample times (if radon_results provided)
        Row N+5: Per-column metrics + per-column velocity profile
        Row N+6: v(t) per-window comparison + Radon overlay (if per_window_vhat provided)
        Row N+7: v(t) from selected region (if provided)
    """
    from pathlib import Path
    from matplotlib.colors import Normalize
    import matplotlib.cm as mpl_cm

    T, M_kymo = kymo.shape
    DPI = 200

    has_vt = v_hat is not None and len(v_hat) > 0
    has_vt_compare = per_window_vhat is not None and len(per_window_vhat) > 0
    has_radon = radon_results is not None and len(radon_results) > 0
    window_keys = sorted(per_window_gst.keys()) if per_window_gst else []
    n_window_rows = len(window_keys)
    n_radon_rows = 2 if has_radon else 0  # velocity strip + variance profiles
    # Row layout: kymo + per-window rows + combined vel + [radon rows] + quality + [vt compare] + [v(t)]
    n_rows = 1 + n_window_rows + 1 + n_radon_rows + 1 + (1 if has_vt_compare else 0) + (1 if has_vt else 0)
    height_ratios = ([1] + [1]*n_window_rows + [1]
                     + ([1, 0.6] if has_radon else [])
                     + [0.8]
                     + ([1.0] if has_vt_compare else [])
                     + ([0.8] if has_vt else []))
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.5 * n_rows),
                             gridspec_kw={'height_ratios': height_ratios})
    fig.canvas.manager.set_window_title(f'Kymograph Diagnostics: {vessel_id}')

    arc_color = 'red'
    arc_lw = 1.5
    v_scale = px_size_um / frame_dt_s / 1000.0  # px/frame → mm/s

    # Determine if GST maps are full-length or arc-restricted
    M_gst = vel_map.shape[1]
    gst_is_restricted = (M_gst != M_kymo)

    if gst_is_restricted:
        gst_extent = [arc_start, arc_end, T, 0]
    else:
        gst_extent = [0, M_kymo, T, 0]

    # Per-window GST maps may span the original (pre-refinement) arc
    if vel_refinement is not None and vel_refinement.get('refined', False):
        _pw_s = vel_refinement['orig_arc_start']
        _pw_e = vel_refinement['orig_arc_end']
        pw_gst_extent = [_pw_s, _pw_e, T, 0]
    else:
        pw_gst_extent = gst_extent

    # Shared velocity color scale from combined map
    vel_combined_mm = vel_map * v_scale
    finite_vc = vel_combined_mm[np.isfinite(vel_combined_mm)]
    if len(finite_vc) > 0:
        med_vc = np.median(finite_vc)
        if med_vc >= 0:
            vlim_shared = (0.0, max(2.0 * med_vc, 0.01))
        else:
            vlim_shared = (min(2.0 * med_vc, -0.01), 0.0)
    else:
        vlim_shared = (-1.0, 1.0)
    norm_shared = Normalize(vmin=vlim_shared[0], vmax=vlim_shared[1])

    row = 0

    # --- Row 1: Raw kymograph ---
    ax = axes[row]
    _is_refined = vel_refinement is not None and vel_refinement.get('refined', False)
    ax.imshow(kymo, aspect='auto', cmap='gray', origin='upper',
              extent=[0, M_kymo, T, 0])
    if _is_refined:
        # Show original arc as red dashed, refined (= current arc_start/end) as green solid
        _orig_s = vel_refinement['orig_arc_start']
        _orig_e = vel_refinement['orig_arc_end']
        ax.axvline(_orig_s, color=arc_color, linestyle='--', linewidth=arc_lw, label='Original arc')
        ax.axvline(_orig_e, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.axvline(arc_start, color='limegreen', linestyle='-', linewidth=1.5, label='Refined arc')
        ax.axvline(arc_end, color='limegreen', linestyle='-', linewidth=1.5)
    else:
        ax.axvline(arc_start, color=arc_color, linestyle='--', linewidth=arc_lw, label='Arc region')
        ax.axvline(arc_end, color=arc_color, linestyle='--', linewidth=arc_lw)
    _overlay_gst_arrows(ax, vel_map, conf_px,
                        x_offset=arc_start if gst_is_restricted else 0)
    ax.set_ylabel('Time [frames]')
    ax.set_title(f'Raw kymograph ({T} frames x {M_kymo} columns) — {vessel_id}', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    row += 1

    # --- Per-window GST rows ---
    for w in window_keys:
        ax = axes[row]
        wdata = per_window_gst[w]
        vel_w_mm = wdata['vel_map'] * v_scale
        coh_w = wdata['conf_px']
        rgba_w = plt.cm.RdBu_r(norm_shared(np.nan_to_num(vel_w_mm, nan=0.0)))
        rgba_w[..., 3] = np.clip(coh_w, 0, 1)
        rgba_w[~np.isfinite(vel_w_mm), 3] = 0.0
        ax.imshow(rgba_w, aspect='auto', origin='upper', extent=pw_gst_extent)
        im = ax.imshow(np.full_like(vel_w_mm, np.nan), aspect='auto', cmap='RdBu_r',
                       origin='upper', extent=pw_gst_extent,
                       vmin=vlim_shared[0], vmax=vlim_shared[1])
        if _is_refined:
            ax.axvline(vel_refinement['orig_arc_start'], color=arc_color,
                       linestyle='--', linewidth=arc_lw)
            ax.axvline(vel_refinement['orig_arc_end'], color=arc_color,
                       linestyle='--', linewidth=arc_lw)
            ax.axvline(arc_start, color='limegreen', linestyle='-', linewidth=1.5)
            ax.axvline(arc_end, color='limegreen', linestyle='-', linewidth=1.5)
        else:
            ax.axvline(arc_start, color=arc_color, linestyle='--', linewidth=arc_lw)
            ax.axvline(arc_end, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.set_xlim(0, M_kymo)
        ax.set_ylabel('Time [frames]')
        mean_coh_w = np.mean(coh_w[np.isfinite(coh_w)])
        ax.set_title(f'GST w={w} (mean coh={mean_coh_w:.3f})', fontsize=14)
        plt.colorbar(im, ax=ax, label='v [mm/s]', shrink=0.8)
        row += 1

    # --- Combined best-of: streaks + low-opacity heatmap + velocity arrows ---
    ax = axes[row]
    # Grayscale kymograph underlay (restricted to arc)
    kymo_arc = kymo[:, arc_start:arc_end]
    ax.imshow(kymo_arc, aspect='auto', cmap='gray', origin='upper', extent=gst_extent)
    # Low-opacity velocity heatmap
    rgba_vd = plt.cm.RdBu_r(norm_shared(np.nan_to_num(vel_combined_mm, nan=0.0)))
    rgba_vd[..., 3] = np.clip(conf_px * 0.4, 0, 0.4)
    rgba_vd[~np.isfinite(vel_combined_mm), 3] = 0.0
    ax.imshow(rgba_vd, aspect='auto', origin='upper', extent=gst_extent)
    im = ax.imshow(np.full_like(vel_combined_mm, np.nan), aspect='auto', cmap='RdBu_r',
                   origin='upper', extent=gst_extent,
                   vmin=vlim_shared[0], vmax=vlim_shared[1])
    if _is_refined:
        _orig_s = vel_refinement['orig_arc_start']
        _orig_e = vel_refinement['orig_arc_end']
        ax.axvline(_orig_s, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.axvline(_orig_e, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.axvline(arc_start, color='limegreen', linestyle='-', linewidth=1.5)
        ax.axvline(arc_end, color='limegreen', linestyle='-', linewidth=1.5)
    else:
        ax.axvline(arc_start, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.axvline(arc_end, color=arc_color, linestyle='--', linewidth=arc_lw)
    _overlay_gst_arrows(ax, vel_map, conf_px,
                        x_offset=arc_start if gst_is_restricted else 0,
                        color='velocity', coh_min=0.1,
                        vlim=vlim_shared, v_scale=v_scale)
    ax.set_xlim(0, M_kymo)
    ax.set_ylabel('Time [frames]')
    ax.set_title('Combined best-of + GST arrows', fontsize=14)
    plt.colorbar(im, ax=ax, label='v [mm/s]', shrink=0.8)
    row += 1

    # --- SVD+Radon rows ---
    if has_radon:
        # Pick the middle window (W=32 if present, else first)
        radon_keys = sorted(radon_results.keys())
        radon_primary = 32 if 32 in radon_results else radon_keys[0]
        rd = radon_results[radon_primary]
        rv_hat = rd['v_hat']
        rc_conf = rd['conf']

        # Row A: Radon velocity strip — v(t) as color vs time
        ax = axes[row]
        # Grayscale kymograph underlay (restricted to arc)
        kymo_arc_radon = kymo[:, arc_start:arc_end]
        ax.imshow(kymo_arc_radon, aspect='auto', cmap='gray', origin='upper',
                  extent=gst_extent)
        # Velocity color strip: each row gets uniform color from Radon v(t)
        rv_mm = rv_hat * v_scale
        M_arc = arc_end - arc_start
        # Build (T, M_arc) RGBA from the (T,) velocity
        rv_2d = np.tile(rv_mm[:, None], (1, M_arc))
        rgba_r = plt.cm.RdBu_r(norm_shared(np.nan_to_num(rv_2d, nan=0.0)))
        # Alpha from confidence (broadcast)
        rc_2d = np.tile(rc_conf[:, None], (1, M_arc))
        rgba_r[..., 3] = np.clip(rc_2d * 0.6, 0, 0.6)
        rgba_r[~np.isfinite(rv_2d), 3] = 0.0
        ax.imshow(rgba_r, aspect='auto', origin='upper', extent=gst_extent)
        im_r = ax.imshow(np.full_like(rv_2d, np.nan), aspect='auto', cmap='RdBu_r',
                         origin='upper', extent=gst_extent,
                         vmin=vlim_shared[0], vmax=vlim_shared[1])
        ax.set_xlim(0, M_kymo)
        ax.set_ylabel('Time [frames]')
        med_conf = float(np.nanmedian(rc_conf))
        ax.set_title(f'SVD+Radon W={radon_primary} (rank=5, median conf={med_conf:.3f})',
                     fontsize=14)
        plt.colorbar(im_r, ax=ax, label='v [mm/s]', shrink=0.8)
        row += 1

        # Row B: Radon variance profiles at sample time points
        ax = axes[row]
        r_profiles = rd.get('profiles')
        if r_profiles is not None:
            n_profiles = min(5, r_profiles.shape[0])
            half_w = radon_primary // 2
            # Indices into the profiles array (evenly spaced)
            prof_indices = np.linspace(0, r_profiles.shape[0] - 1, n_profiles, dtype=int)
            # Reconstruct velocity axis
            dv = 0.1
            velocities_ax = np.arange(-6.0, 6.0 + dv * 0.5, dv)
            if len(velocities_ax) != r_profiles.shape[1]:
                velocities_ax = np.linspace(-6.0, 6.0, r_profiles.shape[1])
            vel_ax_mm = velocities_ax * v_scale

            colors_prof = plt.cm.cool(np.linspace(0.1, 0.9, n_profiles))
            for i, pi in enumerate(prof_indices):
                prof = r_profiles[pi]
                # Normalize to [0, 1]
                pmin, pmax = prof.min(), prof.max()
                prof_norm = (prof - pmin) / (pmax - pmin + 1e-12)
                t_sec = (pi + half_w) * frame_dt_s
                peak_v = velocities_ax[np.argmax(prof)] * v_scale
                ax.plot(vel_ax_mm, prof_norm, color=colors_prof[i], linewidth=1.2,
                        alpha=0.8, label=f't={t_sec:.2f}s (v={peak_v:.2f}mm/s)')
                ax.axvline(peak_v, color=colors_prof[i], linestyle=':', alpha=0.3)

            ax.set_xlabel('Velocity [mm/s]')
            ax.set_ylabel('Radon variance (norm.)')
            ax.set_title('SVD+Radon variance profiles at sample times', fontsize=14)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)
        row += 1

    # --- Per-column quality metrics ---
    ax = axes[row]
    col_x = np.arange(M_kymo)

    # Streak quality (combined) — always full kymograph length
    ax.plot(col_x, quality, 'k-', linewidth=2, label='Streak quality')

    # Column-mean coherence and high-coherence fraction
    if gst_is_restricted:
        mean_coh = np.full(M_kymo, np.nan)
        mean_coh[arc_start:arc_end] = np.mean(conf_px, axis=0)
        high_coh_frac = np.full(M_kymo, np.nan)
        high_coh_frac[arc_start:arc_end] = np.mean(conf_px >= 0.3, axis=0)
    else:
        mean_coh = np.mean(conf_px, axis=0)
        high_coh_frac = np.mean(conf_px >= 0.3, axis=0)

    ax.plot(col_x, mean_coh, 'b-', linewidth=1.5, alpha=0.7, label='Mean coherence')
    ax.plot(col_x, high_coh_frac, 'g-', linewidth=1.5, alpha=0.7,
            label='Frac(coh > 0.3)')

    # Column mask
    if column_mask is not None:
        if len(column_mask) == M_kymo:
            mask_disp = column_mask.astype(float)
        else:
            mask_disp = np.zeros(M_kymo)
            mask_disp[arc_start:arc_end] = column_mask.astype(float)
        ax.fill_between(col_x, 0, mask_disp * 0.15, alpha=0.3, color='orange',
                        label='Column mask')

    # Arc region shading
    if _is_refined:
        _orig_s = vel_refinement['orig_arc_start']
        _orig_e = vel_refinement['orig_arc_end']
        ax.axvspan(_orig_s, _orig_e, alpha=0.08, color='red')
        ax.axvline(_orig_s, color=arc_color, linestyle='--', linewidth=arc_lw, label='Original arc')
        ax.axvline(_orig_e, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.axvspan(arc_start, arc_end, alpha=0.1, color='limegreen')
        ax.axvline(arc_start, color='limegreen', linestyle='-', linewidth=2, label='Refined arc')
        ax.axvline(arc_end, color='limegreen', linestyle='-', linewidth=2)
    else:
        ax.axvspan(arc_start, arc_end, alpha=0.08, color='red')
        ax.axvline(arc_start, color=arc_color, linestyle='--', linewidth=arc_lw)
        ax.axvline(arc_end, color=arc_color, linestyle='--', linewidth=arc_lw)

    # Overlay per-column velocity profile from vel_refinement
    if vel_refinement is not None and vel_refinement.get('v_col') is not None:
        v_col = vel_refinement['v_col']
        v_smooth = vel_refinement['v_smooth']
        # orig_arc_start is the arc start before refinement narrowed it
        orig_arc_start = vel_refinement.get('orig_arc_start', arc_start)

        # Per-column velocity on twin axis (within original arc region)
        ax2 = ax.twinx()
        arc_x = np.arange(orig_arc_start, orig_arc_start + len(v_col))
        v_col_mm = v_col * v_scale
        v_smooth_mm = v_smooth * v_scale
        ax2.plot(arc_x, v_col_mm, 'm.', markersize=2, alpha=0.5, label='v_col')
        ax2.plot(arc_x, v_smooth_mm, 'm-', linewidth=1.5, alpha=0.8, label='v_smooth')
        ax2.set_ylabel('Col velocity [mm/s]', color='m')
        ax2.tick_params(axis='y', labelcolor='m')

    ax.set_xlabel('Arc position [px]')
    ax.set_ylabel('Score')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, M_kymo)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.set_title('Per-column quality metrics + velocity profile', fontsize=14)
    ax.grid(True, alpha=0.3)

    # --- Per-window v(t) comparison ---
    if has_vt_compare:
        row += 1
        ax = axes[row]
        t_arr_cmp = np.arange(T) * frame_dt_s
        sorted_windows = sorted(per_window_vhat.keys())
        cmap_windows = mpl_cm.viridis(np.linspace(0.1, 0.9, len(sorted_windows)))
        for i, w in enumerate(sorted_windows):
            vw = per_window_vhat[w]
            if vw is not None and len(vw) == T:
                v_mm_w = vw * v_scale
                ax.plot(t_arr_cmp, v_mm_w, linewidth=0.7, alpha=0.7,
                        color=cmap_windows[i], label=f'w={w}')
        # Overlay Radon v(t) curves
        if has_radon:
            radon_keys_sorted = sorted(radon_results.keys())
            radon_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(radon_keys_sorted)))
            for i, rw in enumerate(radon_keys_sorted):
                rv = radon_results[rw]['v_hat']
                if rv is not None and len(rv) == T:
                    rv_mm = rv * v_scale
                    ax.plot(t_arr_cmp, rv_mm, linewidth=0.9, alpha=0.8,
                            color=radon_colors[i], linestyle='--',
                            label=f'Radon W={rw}')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title('v(t): GST windows + SVD+Radon comparison', fontsize=14)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [mm/s]')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    # --- v(t) combined row ---
    row += 1
    if has_vt:
        ax = axes[row]
        t_arr = np.arange(len(v_hat)) * frame_dt_s
        v_mm = v_hat * v_scale

        # Overlay pre-refinement v(t) if available
        if _is_refined and vel_refinement.get('v_hat_pre') is not None:
            v_pre = vel_refinement['v_hat_pre']
            t_pre = np.arange(len(v_pre)) * frame_dt_s
            v_pre_mm = v_pre * v_scale
            ax.plot(t_pre, v_pre_mm, color='salmon', linewidth=0.8, alpha=0.7,
                    label='Pre-refinement')

        ax.plot(t_arr, v_mm, 'c-', linewidth=0.8, alpha=0.8,
                label='Refined' if _is_refined else 'Combined')
        ax.axhline(0, color='gray', linewidth=0.5)
        title = 'v(t) — refined vs original arc' if _is_refined else 'v(t) combined (multi-scale best-of)'
        if f0_hz is not None and np.isfinite(f0_hz):
            title += f' — f0 = {f0_hz:.2f} Hz'
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [mm/s]')
        if _is_refined:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"kymo_debug_{vessel_id}.png" if vessel_id else "kymo_debug.png"
        fig.savefig(out_path / fname, dpi=DPI, bbox_inches='tight')
        print(f"  Saved {fname}")

    _ensure_interactive()
    plt.show(block=block)

    return fig


def plot_gst_window_comparison(
    kymo: np.ndarray,
    per_window_gst: Dict,
    per_window_vhat: Dict,
    frame_dt_s: float = FRAME_DT_S,
    px_size_um: float = PX_SIZE_UM,
    vessel_id: str = "",
    out_dir: Optional[str] = None,
):
    """
    Two-figure comparison of GST window configurations.

    Figure 1: Velocity map grid — rows are spatial configs, columns are temporal sizes.
              Shows velocity with alpha=coherence.
    Figure 2: v(t) comparison — square windows vs asymmetric windows.
    """
    from pathlib import Path
    from matplotlib.colors import Normalize
    import matplotlib.cm as mpl_cm

    T, M = kymo.shape
    v_scale = px_size_um / frame_dt_s / 1000.0  # px/frame → mm/s
    t_arr = np.arange(T) * frame_dt_s
    DPI = 150

    # Separate square and asymmetric configs
    square_keys = sorted([k for k in per_window_gst if isinstance(k, int)])
    asym_keys = sorted([k for k in per_window_gst if isinstance(k, tuple)])

    # Shared velocity color scale from all configs
    all_vels = []
    for k, d in per_window_gst.items():
        vm = d['vel_map'] * v_scale
        finite = vm[np.isfinite(vm)]
        if len(finite) > 0:
            all_vels.append(finite)
    if all_vels:
        all_v = np.concatenate(all_vels)
        med_v = np.median(all_v)
        if med_v >= 0:
            vlim = (0.0, max(2.0 * med_v, 0.01))
        else:
            vlim = (min(2.0 * med_v, -0.01), 0.0)
    else:
        vlim = (-1.0, 1.0)
    norm = Normalize(vmin=vlim[0], vmax=vlim[1])

    # =====================================================================
    # FIGURE 1: Velocity map grid
    # =====================================================================
    # Organize asymmetric by unique w_space values
    asym_by_ws = {}
    for wt, ws in asym_keys:
        asym_by_ws.setdefault(ws, []).append((wt, ws))

    # Rows: "square" + one row per w_space value
    row_labels = ['square (w×w)']
    row_configs = [[(w, per_window_gst[w]) for w in square_keys]]
    for ws in sorted(asym_by_ws.keys()):
        row_labels.append(f'w_space={ws}')
        row_configs.append([(k, per_window_gst[k]) for k in asym_by_ws[ws]])

    max_cols = max(len(r) for r in row_configs)
    n_rows = len(row_configs)

    fig1, axes1 = plt.subplots(n_rows, max_cols, figsize=(3.5 * max_cols, 3 * n_rows),
                                squeeze=False)
    fig1.canvas.manager.set_window_title(f'GST Window Comparison: {vessel_id}')

    for ri, (label, configs) in enumerate(zip(row_labels, row_configs)):
        for ci, (key, data) in enumerate(configs):
            ax = axes1[ri, ci]
            vel_mm = data['vel_map'] * v_scale
            coh = data['conf_px']

            # RGBA with alpha = coherence
            rgba = plt.cm.RdBu_r(norm(np.nan_to_num(vel_mm, nan=0.0)))
            rgba[..., 3] = np.clip(coh, 0, 1)
            rgba[~np.isfinite(vel_mm), 3] = 0.0
            ax.imshow(rgba, aspect='auto', origin='upper')

            mean_coh = np.nanmean(coh)
            if isinstance(key, tuple):
                title = f'({key[0]},{key[1]}) coh={mean_coh:.3f}'
            else:
                title = f'w={key} coh={mean_coh:.3f}'
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(label, fontsize=10)

        # Hide unused axes
        for ci in range(len(configs), max_cols):
            axes1[ri, ci].axis('off')

    fig1.tight_layout()

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"gst_velmap_grid_{vessel_id}.png"
        fig1.savefig(out_path / fname, dpi=DPI, bbox_inches='tight')
        print(f"  Saved {fname}")

    # =====================================================================
    # FIGURE 2: v(t) comparison
    # =====================================================================
    fig2, (ax_sq, ax_asym) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig2.canvas.manager.set_window_title(f'GST v(t) Comparison: {vessel_id}')

    # Top: square windows
    sq_vhat_keys = sorted([k for k in per_window_vhat if isinstance(k, int)])
    colors_sq = mpl_cm.viridis(np.linspace(0.1, 0.9, len(sq_vhat_keys)))
    for i, w in enumerate(sq_vhat_keys):
        vw = per_window_vhat[w]
        if vw is not None and len(vw) == T:
            ax_sq.plot(t_arr, vw * v_scale, linewidth=0.7, alpha=0.7,
                       color=colors_sq[i], label=f'w={w}')
    ax_sq.axhline(0, color='gray', linewidth=0.5)
    ax_sq.set_title('v(t) — square windows (w×w)', fontsize=14)
    ax_sq.set_ylabel('Velocity [mm/s]')
    ax_sq.legend(loc='upper right', fontsize=8, ncol=5)
    ax_sq.grid(True, alpha=0.3)

    # Bottom: asymmetric windows
    asym_vhat_keys = sorted([k for k in per_window_vhat if isinstance(k, tuple)])
    colors_asym = mpl_cm.plasma(np.linspace(0.1, 0.9, len(asym_vhat_keys)))
    for i, k in enumerate(asym_vhat_keys):
        vw = per_window_vhat[k]
        if vw is not None and len(vw) == T:
            ax_asym.plot(t_arr, vw * v_scale, linewidth=0.7, alpha=0.7,
                         color=colors_asym[i], label=f'({k[0]},{k[1]})')
    ax_asym.axhline(0, color='gray', linewidth=0.5)
    ax_asym.set_title('v(t) — asymmetric windows (w_time, w_space)', fontsize=14)
    ax_asym.set_xlabel('Time [s]')
    ax_asym.set_ylabel('Velocity [mm/s]')
    ax_asym.legend(loc='upper right', fontsize=8, ncol=4)
    ax_asym.grid(True, alpha=0.3)

    fig2.tight_layout()

    if out_dir is not None:
        fname = f"gst_vt_comparison_{vessel_id}.png"
        fig2.savefig(out_path / fname, dpi=DPI, bbox_inches='tight')
        print(f"  Saved {fname}")

    _ensure_interactive()
    plt.show(block=False)


def plot_quality_diagnostic(
    kymo: np.ndarray,
    vel_map: np.ndarray,
    conf_px: np.ndarray,
    v_hat: np.ndarray,
    arc_start: int,
    arc_end: int,
    quality_score: float,
    mean_coherence: float,
    frame_dt_s: float = FRAME_DT_S,
    px_size_um: float = PX_SIZE_UM,
    f0_hz: Optional[float] = None,
    vessel_id: str = "",
    out_dir: Optional[str] = None,
) -> plt.Figure:
    """Simple 4-panel quality diagnostic: kymograph, velocity, coherence, v(t)."""
    from pathlib import Path

    T_kymo, M_kymo = kymo.shape
    T_arc, M_arc = vel_map.shape

    t_s = np.arange(T_kymo) * frame_dt_s
    t_v = np.arange(len(v_hat)) * frame_dt_s
    vel_ums = vel_map * px_size_um / frame_dt_s
    v_hat_ums = v_hat * px_size_um / frame_dt_s

    fig, axes = plt.subplots(4, 1, figsize=(6, 7), dpi=150,
                             gridspec_kw={'height_ratios': [1, 1, 1, 0.8]})

    # Panel 1: Kymograph with arc boundaries
    ax = axes[0]
    ax.imshow(kymo, aspect='auto', cmap='gray',
              extent=[0, M_kymo, t_s[-1], 0])
    ax.axvline(arc_start, color='lime', lw=1, ls='--')
    ax.axvline(arc_end, color='lime', lw=1, ls='--')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Kymograph  [{vessel_id}]', fontsize=10)

    # Panel 2: Velocity map (arc region only)
    ax = axes[1]
    vlim = np.nanpercentile(np.abs(vel_ums), 95) if np.any(np.isfinite(vel_ums)) else 1.0
    ax.imshow(vel_ums, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim,
              extent=[0, M_arc, t_s[-1] if T_arc == T_kymo else T_arc * frame_dt_s, 0])
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Velocity (um/s)', fontsize=10)

    # Panel 3: Coherence map (arc region only)
    ax = axes[2]
    ax.imshow(conf_px, aspect='auto', cmap='inferno', vmin=0, vmax=1,
              extent=[0, M_arc, t_s[-1] if T_arc == T_kymo else T_arc * frame_dt_s, 0])
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Coherence  (mean={mean_coherence:.3f})', fontsize=10)

    # Panel 4: v(t)
    ax = axes[3]
    ax.plot(t_v, v_hat_ums, 'k-', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('v (um/s)')
    score_str = f'Q={quality_score:.3f}'
    if f0_hz is not None:
        score_str += f', f0={f0_hz:.2f} Hz'
    ax.set_title(f'v(t)  ({score_str})', fontsize=10)
    ax.axhline(0, color='gray', lw=0.5, ls=':')

    fig.tight_layout()

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"quality_{vessel_id}.png"
        fig.savefig(out_path / fname, dpi=150, bbox_inches='tight')
        print(f"  Saved {fname}")

    _ensure_interactive()
    plt.show(block=False)
    return fig
