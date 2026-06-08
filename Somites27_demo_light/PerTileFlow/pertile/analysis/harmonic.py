"""
Harmonic regression for blood flow velocity signal analysis.

Provides robust harmonic regression with IRLS (Iteratively Reweighted
Least Squares) for fitting periodic blood flow signals and computing
signal-to-noise ratios.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Literal


def _mad_sigma(x: np.ndarray) -> float:
    """Compute robust standard deviation estimate using MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * (mad + 1e-12)


def _irls_weights(
    resid: np.ndarray,
    loss: str,
    sigma: float,
    huber_delta_sigma: float = 1.5,
    tukey_c_sigma: float = 4.685,
) -> np.ndarray:
    """Compute IRLS weights for robust regression."""
    r = resid / max(sigma, 1e-12)

    if loss == "huber":
        k = float(huber_delta_sigma)
        w = np.ones_like(r)
        m = np.abs(r) > k
        w[m] = k / (np.abs(r[m]) + 1e-12)
        return w
    elif loss == "tukey":
        c = float(tukey_c_sigma)
        u = r / c
        w = np.zeros_like(u)
        m = np.abs(u) < 1
        u2 = u[m] * u[m]
        w[m] = (1 - u2) ** 2
        return w
    else:
        return np.ones_like(r)


def _design_matrix(
    t: np.ndarray,
    f0: float,
    K: int,
    include_dc: bool = True,
) -> np.ndarray:
    """Build design matrix for harmonic regression."""
    cols = []
    if include_dc:
        cols.append(np.ones_like(t))
    for k in range(1, K + 1):
        w = 2 * np.pi * k * f0
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    return np.column_stack(cols)


def estimate_f0_in_band(
    v_hat: np.ndarray,
    frame_dt: float,
    fmin_hz: float = 1.0,
    fmax_hz: float = 3.5,
    method: Literal["periodogram"] = "periodogram",
    df: Optional[float] = None,
    refine: bool = True,
    return_periodogram: bool = False,
):
    """
    Estimate fundamental frequency (heart rate) from velocity time series.

    Args:
        v_hat: (T,) velocity time series
        frame_dt: Time between frames in seconds
        fmin_hz: Lower frequency bound for search
        fmax_hz: Upper frequency bound for search
        method: Estimation method (currently only "periodogram")
        df: Frequency grid spacing (default: 1/T)
        refine: Apply parabolic refinement around peak
        return_periodogram: If True, return (f0, freqs, power) tuple

    Returns:
        f0: Estimated fundamental frequency in Hz
        OR (f0, freqs, power) if return_periodogram=True
    """
    _nan = (np.nan, np.array([]), np.array([])) if return_periodogram else np.nan

    y = np.asarray(v_hat, float)
    t = np.arange(y.size, dtype=float) * float(frame_dt)
    m = np.isfinite(y)

    if m.sum() < 8:
        return _nan

    y = y[m]
    t = t[m]
    y = y - np.nanmedian(y)

    if y.size < 8 or np.allclose(y, 0):
        return _nan

    # Frequency grid
    T = t[-1] - t[0]
    if T <= 0:
        return _nan

    if df is None or df <= 0:
        df = 1.0 / max(4.0 * T, 1e-12)  # 4x oversampling for precise peak location

    f = np.arange(max(fmin_hz, df), fmax_hz + 1e-12, df, dtype=float)
    if f.size < 3:
        return _nan

    # Windowed periodogram
    w = np.hanning(y.size)
    yw = y * w
    Y = np.array([np.abs(np.sum(yw * np.exp(-2j * np.pi * ff * t))) ** 2
                   for ff in f], float)

    k = int(np.argmax(Y))
    f0 = float(f[k])

    # Parabolic refinement in log-power
    if refine and 1 <= k <= (len(f) - 2):
        xs = f[k - 1 : k + 2]
        ps = np.log(Y[k - 1 : k + 2] + 1e-18)
        try:
            a, b, c = np.polyfit(xs, ps, 2)
            if a < 0:
                f_ref = -b / (2 * a)
                if fmin_hz <= f_ref <= fmax_hz:
                    f0 = float(f_ref)
        except Exception:
            pass

    if return_periodogram:
        return f0, f, Y
    return f0


def periodogram_snr_at_f0(freqs: np.ndarray, power: np.ndarray,
                           f0: float, half_width: float = 0.15) -> float:
    """Compute SNR of a periodogram at a target frequency.

    Power near f0 (±half_width Hz) is compared to the median power across
    the full spectrum.  Returns the ratio (>1 means signal present).
    """
    median_power = np.median(power)
    if median_power <= 0:
        return 0.0
    mask = np.abs(freqs - f0) <= half_width
    if not np.any(mask):
        k = np.argmin(np.abs(freqs - f0))
        return float(power[k] / median_power)
    return float(np.mean(power[mask]) / median_power)


def consensus_f0_stacked(
    periodograms: list,
    fmin_hz: float = 1.0,
    fmax_hz: float = 3.5,
    refine: bool = True,
    debug_plot: bool = False,
) -> float:
    """Compute consensus f0 by stacking peak-normalized periodograms.

    Each periodogram is peak-normalized, then stacked.  The peak is found
    as the tallest *local maximum* (not the global max), which naturally
    excludes the 1/f red-noise boundary spike at fmin.

    Args:
        periodograms: list of (freqs, power) tuples from estimate_f0_in_band
        fmin_hz: Lower search bound (for safety clamp)
        fmax_hz: Upper search bound
        refine: Apply parabolic refinement on stacked peak
        debug_plot: If True, show matplotlib plot of stacked spectrum

    Returns:
        f0: Consensus frequency in Hz (np.nan if no valid periodograms)
    """
    from scipy.signal import find_peaks

    # Filter out empty periodograms
    valid = [(fr, pw) for fr, pw in periodograms
             if len(fr) > 0 and len(pw) > 0 and np.any(pw > 0)]
    if not valid:
        return np.nan

    # Use the frequency grid from the first valid periodogram as reference.
    f_ref = valid[0][0]
    stacked = np.zeros_like(f_ref, dtype=float)
    n_added = 0

    for fr, pw in valid:
        if len(fr) == len(f_ref) and np.allclose(fr, f_ref):
            pw_on_grid = pw
        else:
            pw_on_grid = np.interp(f_ref, fr, pw, left=0.0, right=0.0)

        peak = pw_on_grid.max()
        if peak > 0:
            stacked += pw_on_grid / peak
            n_added += 1

    if n_added == 0:
        return np.nan

    stacked /= n_added

    # Find local maxima — this excludes boundary 1/f spikes which are
    # monotonically decreasing from fmin, not true peaks
    peak_indices, properties = find_peaks(stacked, prominence=0)
    if len(peak_indices) > 0:
        # Pick the local maximum with highest prominence
        prominences = properties['prominences']
        best = peak_indices[np.argmax(prominences)]
        k = int(best)
    else:
        # Fallback: global max (shouldn't normally happen)
        k = int(np.argmax(stacked))

    f0 = float(f_ref[k])

    # Parabolic refinement
    if refine and 1 <= k <= (len(f_ref) - 2):
        xs = f_ref[k - 1 : k + 2]
        ps = np.log(stacked[k - 1 : k + 2] + 1e-18)
        try:
            a, b, _c = np.polyfit(xs, ps, 2)
            if a < 0:
                f_ref_val = -b / (2 * a)
                if fmin_hz <= f_ref_val <= fmax_hz:
                    f0 = float(f_ref_val)
        except Exception:
            pass

    if debug_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(f_ref, stacked)
            ax.axvline(f0, color='r', ls='--', label=f'f0={f0:.3f} Hz')
            if len(peak_indices) > 0:
                ax.plot(f_ref[peak_indices], stacked[peak_indices],
                        'v', color='orange', ms=8, label='local maxima')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Stacked power (peak-normalized)')
            ax.set_title(f'Stacked periodograms (N={n_added})')
            ax.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  Debug plot failed: {e}")

    return f0


def fit_harmonics(
    v_hat: np.ndarray,
    frame_dt: float,
    f0: float,
    K: int = 3,
    loss: Literal["huber", "tukey", "none"] = "huber",
    loss_params: Optional[Dict[str, float]] = None,
    include_dc: bool = True,
) -> Dict[str, Any]:
    """
    Fit harmonic model to velocity time series using robust IRLS.

    Fits the model: v(t) = a0 + sum_k [A_k*cos(2*pi*k*f0*t) + B_k*sin(2*pi*k*f0*t)]

    Args:
        v_hat: (T,) velocity time series (px/frame or physical units)
        frame_dt: Time between frames in seconds
        f0: Fundamental frequency in Hz
        K: Number of harmonics to fit
        loss: Robust loss function ("huber", "tukey", or "none")
        loss_params: Parameters for loss function
        include_dc: Include DC (mean) component in fit

    Returns:
        Dict with:
            - a0: DC component (mean velocity)
            - harmonics: List of dicts with k, A, B, amp, phi for each harmonic
            - signal: (T,) fitted signal
            - resid: (T,) residuals
            - r2: R-squared fit quality
            - hr_snr_db: Harmonic regression SNR in dB
            - f0: Input frequency
            - amp1: First harmonic amplitude
            - amp_rms_ac: RMS of AC component
    """
    y0 = np.asarray(v_hat, float)
    T = int(y0.size)
    t0 = np.arange(T, dtype=float) * float(frame_dt)

    m_valid = np.isfinite(y0)
    n_valid = int(m_valid.sum())
    min_samples = max(8, 2 * int(K) + (1 if include_dc else 0) + 1)

    # Return NaN result if insufficient data
    if n_valid < min_samples or not np.isfinite(f0):
        return dict(
            a0=np.nan,
            harmonics=[],
            signal=np.full(T, np.nan, float),
            resid=np.full(T, np.nan, float),
            r2=np.nan,
            hr_snr_db=np.nan,
            f0=float(f0),
            amp_rms_ac=np.nan,
            amp1=np.nan,
        )

    y = y0[m_valid]
    t = t0[m_valid]
    X = _design_matrix(t, f0, int(K), include_dc=bool(include_dc))

    # Initial OLS fit
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sigma = _mad_sigma(resid)

    # Loss parameters
    loss_params = loss_params or {}
    huber_delta_sigma = float(loss_params.get("huber_delta_sigma", 1.5))
    tukey_c_sigma = float(loss_params.get("tukey_c_sigma", 4.685))

    # IRLS iterations
    if loss != "none":
        for _ in range(8):
            w = _irls_weights(
                resid,
                loss,
                sigma,
                huber_delta_sigma=huber_delta_sigma,
                tukey_c_sigma=tukey_c_sigma,
            )
            W = np.diag(w)
            try:
                beta_new, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
            except Exception:
                break
            yhat_new = X @ beta_new
            resid_new = y - yhat_new
            sigma_new = _mad_sigma(resid_new)

            if np.allclose(beta_new, beta, rtol=1e-4, atol=1e-6):
                beta, yhat, resid, sigma = beta_new, yhat_new, resid_new, sigma_new
                break
            beta, yhat, resid, sigma = beta_new, yhat_new, resid_new, sigma_new

    # Compute R² and SNR
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-18
    ss_res = float(np.sum(resid**2))
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))

    # SNR from R²: SNR = r2 / (1 - r2)
    eps = 1e-9
    num = max(r2, eps)
    den = max(1.0 - r2, eps)
    hr_snr_db = 10.0 * np.log10(num / den)

    # Compute coefficient covariances for uncertainty propagation
    # Cov(beta) = σ² * (X.T @ X)^(-1), where σ² = ss_res / (n - p)
    n_params = X.shape[1]
    dof = max(n_valid - n_params, 1)
    sigma_resid_sq = ss_res / dof
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        cov_beta = sigma_resid_sq * XtX_inv
    except np.linalg.LinAlgError:
        cov_beta = np.full((n_params, n_params), np.nan)

    # Unpack coefficients
    idx0 = 0
    if include_dc:
        a0 = float(beta[0])
        sigma_a0 = float(np.sqrt(cov_beta[0, 0])) if np.isfinite(cov_beta[0, 0]) else np.nan
        idx0 = 1
    else:
        a0 = 0.0
        sigma_a0 = 0.0

    AB = beta[idx0:]
    harmonics = []
    P_sig = 0.0
    amp1 = np.nan
    sigma_amp1 = np.nan
    sigma_phase1 = np.nan

    for k in range(1, int(K) + 1):
        A = float(AB[2 * (k - 1)])
        B = float(AB[2 * (k - 1) + 1])
        amp = float(np.hypot(A, B))
        phi = float(np.arctan2(-B, A))

        # Propagate uncertainties to amplitude and phase
        # σ_A, σ_B from covariance diagonal
        idx_A = idx0 + 2 * (k - 1)
        idx_B = idx0 + 2 * (k - 1) + 1
        sigma_A = float(np.sqrt(cov_beta[idx_A, idx_A])) if np.isfinite(cov_beta[idx_A, idx_A]) else np.nan
        sigma_B = float(np.sqrt(cov_beta[idx_B, idx_B])) if np.isfinite(cov_beta[idx_B, idx_B]) else np.nan

        # Error propagation: amp = sqrt(A² + B²)
        # σ_amp = sqrt((A*σ_A)² + (B*σ_B)²) / amp
        if amp > 1e-12 and np.isfinite(sigma_A) and np.isfinite(sigma_B):
            sigma_amp = np.sqrt((A * sigma_A)**2 + (B * sigma_B)**2) / amp
            # Error propagation: phase = arctan2(-B, A)
            # σ_phase = sqrt((B*σ_A)² + (A*σ_B)²) / amp²
            sigma_phi = np.sqrt((B * sigma_A)**2 + (A * sigma_B)**2) / (amp**2)
        else:
            sigma_amp = np.nan
            sigma_phi = np.nan

        harmonics.append(dict(k=k, A=A, B=B, amp=amp, phi=phi,
                              sigma_A=sigma_A, sigma_B=sigma_B,
                              sigma_amp=sigma_amp, sigma_phi=sigma_phi))
        P_sig += 0.5 * (A * A + B * B)
        if k == 1:
            amp1 = amp
            sigma_amp1 = sigma_amp
            sigma_phase1 = sigma_phi

    amp_rms_ac = float(np.sqrt(max(P_sig, 0.0)))

    # Expand to full length
    signal_full = np.full(T, np.nan, float)
    resid_full = np.full(T, np.nan, float)
    signal_full[m_valid] = yhat
    resid_full[m_valid] = resid

    return dict(
        a0=a0,
        sigma_a0=sigma_a0,
        harmonics=harmonics,
        signal=signal_full,
        resid=resid_full,
        r2=r2,
        hr_snr_db=hr_snr_db,
        f0=float(f0),
        amp_rms_ac=amp_rms_ac,
        amp1=amp1,
        sigma_amp1=sigma_amp1,
        sigma_phase1=sigma_phase1,
    )
