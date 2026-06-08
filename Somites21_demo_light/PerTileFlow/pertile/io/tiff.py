"""
TIFF stack loading utilities.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import tifffile
from scipy.ndimage import uniform_filter1d


def load_tiff_stack(
    path: Union[str, Path],
    max_frames: Optional[int] = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Load a TIFF stack as a 3D array (T, H, W).

    Parameters
    ----------
    path : str or Path
        Path to TIFF file or directory of TIFFs
    max_frames : int, optional
        Maximum number of frames to load (None = all)
    dtype : np.dtype
        Output dtype (default float32)

    Returns
    -------
    np.ndarray
        Video stack with shape (T, H, W)
    """
    path = Path(path)

    if path.is_dir():
        # Directory of individual TIFFs
        tiff_files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {path}")

        frames = []
        for i, f in enumerate(tiff_files):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(tifffile.imread(f))
        stack = np.stack(frames, axis=0)
    else:
        # Single multi-page TIFF
        stack = tifffile.imread(path)
        if max_frames is not None:
            stack = stack[:max_frames]

    return stack.astype(dtype)


def cut_before_fade(
    stack: np.ndarray,
    *,
    p_drop: float = 0.75,
    p_back: float = 0.90,
    smooth: int = 15,
    backoff: int = 5,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Auto-truncate video stack before illumination fade-out.

    Detects when mean intensity drops below threshold and cuts before that.

    Parameters
    ----------
    stack : np.ndarray
        Video stack (T, H, W)
    p_drop : float
        Fraction of peak intensity that triggers detection (default 0.75)
    p_back : float
        Fraction of peak to backtrack to (default 0.90)
    smooth : int
        Smoothing window for intensity trace (frames)
    backoff : int
        Extra frames to cut before detected fade start
    verbose : bool
        Print info about truncation

    Returns
    -------
    stack : np.ndarray
        Truncated stack (or original if no fade detected)
    cut_frame : int or None
        Frame where cut was made (None if no cut)
    """
    T = stack.shape[0]

    # Compute mean intensity per frame
    m = stack.reshape(T, -1).mean(axis=1).astype(float)
    ms = uniform_filter1d(m, size=max(1, smooth), mode='nearest') if smooth > 1 else m

    vmax = float(ms.max()) + 1e-12
    thr_drop = vmax * p_drop
    thr_back = vmax * p_back

    # Find first frame below drop threshold
    idx_drop = np.where(ms <= thr_drop)[0]

    if idx_drop.size == 0:
        # No fade detected
        if verbose:
            print(f"  Fade detection: none (keeping all {T} frames)")
        return stack, None

    # Walk back from drop point to where intensity was still high
    t_drop = int(idx_drop[0])
    i = t_drop
    while i > 0 and ms[i] < thr_back:
        i -= 1
    cut = max(0, i - backoff)

    # Sanity check - don't cut too aggressively
    if cut < 16:
        if verbose:
            print(f"  Fade detection: cut too early ({cut}), keeping all {T} frames")
        return stack, None

    if verbose:
        print(f"  Fade detection: {T} → {cut} frames (cut at {cut}, drop detected at {t_drop})")

    return stack[:cut], cut
