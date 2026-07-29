"""Spatial smoothing and propagation of per-tile heart rate (f0) estimates.

When pulsatility is low, per-tile f0 estimation from the periodogram can lock
onto noise or subharmonics.  This module provides two strategies:

1. ``smooth_tile_f0`` — post-hoc correction using SNR-weighted neighbor averaging.
2. ``propagate_tile_f0`` — BFS flood-fill from the highest-pulsatility tile,
   used proactively during batch mosaic analysis so that pass 2 always has a
   reliable f0 before it runs.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


def smooth_tile_f0(
    tile_f0: Dict[int, float],
    tile_snr: Dict[int, float],
    tile_centers: Dict[int, Tuple[float, float]],
    *,
    snr_threshold: float = 0.0,
    delta_threshold: float = 0.3,
    neighbor_radius: float = 960.0,
    n_iterations: int = 3,
    verbose: bool = True,
) -> Tuple[Dict[int, float], List[int]]:
    """Correct per-tile f0 using SNR-weighted spatial smoothing.

    Parameters
    ----------
    tile_f0 : dict
        Mapping vid -> consensus f0 (Hz) from batch analysis pass 1.
    tile_snr : dict
        Mapping vid -> median SNR (dB) across measured vessels.
    tile_centers : dict
        Mapping vid -> (cx, cy) tile center in mosaic coordinates.
    snr_threshold : float
        Tiles with median SNR below this are considered low-confidence
        and eligible for correction.  Default 0 dB.
    delta_threshold : float
        A tile is flagged anomalous if its f0 deviates from the
        neighbor-weighted f0 by more than this (Hz).  Default 0.3 Hz.
    neighbor_radius : float
        Maximum center-to-center distance (px) for two tiles to be
        neighbors.  Default 960 (~1.5 tile widths at 640 px).
    n_iterations : int
        Smoothing iterations.  Multiple rounds let corrections propagate
        through clusters of bad tiles.  Default 3.
    verbose : bool
        Print a summary table.

    Returns
    -------
    corrected_f0 : dict
        Mapping vid -> corrected f0 for all tiles (unchanged tiles keep
        their original value).
    anomalous_tiles : list
        Tile vids that were corrected.
    """
    vids = sorted(set(tile_f0) & set(tile_centers))
    if not vids:
        return dict(tile_f0), []

    # Build neighbor lists (precompute once)
    neighbors: Dict[int, List[Tuple[int, float]]] = {v: [] for v in vids}
    for i, va in enumerate(vids):
        ax, ay = tile_centers[va]
        for vb in vids[i + 1:]:
            bx, by = tile_centers[vb]
            dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            if dist < neighbor_radius:
                neighbors[va].append((vb, dist))
                neighbors[vb].append((va, dist))

    corrected = dict(tile_f0)
    all_anomalous: set = set()

    for iteration in range(n_iterations):
        newly_fixed = []

        for vid in vids:
            if not neighbors[vid]:
                continue

            # Compute SNR-weighted neighbor f0
            w_sum = 0.0
            wf_sum = 0.0
            for nb_vid, dist in neighbors[vid]:
                snr_nb = tile_snr.get(nb_vid, -90.0)
                # Weight = clipped SNR / distance; bad tiles contribute ~0
                w = max(snr_nb - (snr_threshold - 5.0), 0.0) / dist
                w_sum += w
                wf_sum += w * corrected[nb_vid]

            if w_sum < 1e-12:
                continue

            neighbor_f0 = wf_sum / w_sum
            local_f0 = corrected[vid]
            local_snr = tile_snr.get(vid, -90.0)

            if (abs(local_f0 - neighbor_f0) > delta_threshold
                    and local_snr < snr_threshold):
                corrected[vid] = neighbor_f0
                all_anomalous.add(vid)
                newly_fixed.append(vid)

        if not newly_fixed:
            break

    anomalous = sorted(all_anomalous)

    if verbose and anomalous:
        print(f"\n{'=' * 60}")
        print("SPATIAL f0 CORRECTION")
        print(f"{'=' * 60}")
        print(f"  SNR threshold: {snr_threshold:.1f} dB")
        print(f"  Delta threshold: {delta_threshold:.2f} Hz")
        print(f"  Iterations used: {min(iteration + 1, n_iterations)}")
        print(f"  Tiles corrected: {len(anomalous)}/{len(vids)}")
        print(f"\n  {'Tile':>5} {'Old f0':>8} {'New f0':>8} {'medSNR':>8}")
        print(f"  {'-' * 31}")
        for vid in anomalous:
            old = tile_f0[vid]
            new = corrected[vid]
            snr = tile_snr.get(vid, float('nan'))
            print(f"  {vid:>5} {old:>8.3f} {new:>8.3f} {snr:>8.1f}")
        print(f"{'=' * 60}")
    elif verbose:
        print("Spatial f0 check: no anomalous tiles found.")

    return corrected, anomalous


# ---------------------------------------------------------------------------
# BFS flood-fill propagation (proactive, used during batch analysis)
# ---------------------------------------------------------------------------

def _build_neighbor_graph(
    tile_centers: Dict[int, Tuple[float, float]],
    vids: List[int],
    neighbor_radius: float,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build symmetric neighbor adjacency from tile center distances."""
    neighbors: Dict[int, List[Tuple[int, float]]] = {v: [] for v in vids}
    for i, va in enumerate(vids):
        ax, ay = tile_centers[va]
        for vb in vids[i + 1:]:
            bx, by = tile_centers[vb]
            dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            if dist < neighbor_radius:
                neighbors[va].append((vb, dist))
                neighbors[vb].append((va, dist))
    return neighbors


def propagate_tile_f0(
    tile_f0_pass1: Dict[int, float],
    tile_pulsatility: Dict[int, float],
    tile_centers: Dict[int, Tuple[float, float]],
    *,
    neighbor_radius: float = 960.0,
    search_window_frac: float = 0.05,
    verbose: bool = True,
) -> Tuple[List[int], Dict[int, float]]:
    """Propagate f0 from highest-pulsatility tile via BFS flood fill.

    Starting from the tile with the strongest pulsatile signal, traverses
    the tile adjacency graph in breadth-first order.  Each tile either keeps
    its own pass-1 f0 (if it agrees with its parent within ±search_window_frac)
    or inherits the parent's f0.

    This ensures that low-pulsatility tiles receive a reliable f0 before
    pass 2 runs, eliminating the need for post-hoc correction.

    Parameters
    ----------
    tile_f0_pass1 : dict
        vid -> median f0 (Hz) from pass 1 per-vessel periodograms.
        Tiles with no pass-1 result may be absent.
    tile_pulsatility : dict
        vid -> best periodogram/harmonic SNR across vessels in that tile.
        Higher = more confident f0 estimate.
    tile_centers : dict
        vid -> (cx, cy) in mosaic pixel coordinates.
    neighbor_radius : float
        Maximum center-to-center distance for adjacency (default 960 px).
    search_window_frac : float
        A tile keeps its own f0 if within ±this fraction of the parent's f0.
        Default 0.05 (±5%).
    verbose : bool
        Print propagation summary.

    Returns
    -------
    bfs_order : list[int]
        Tile vids in the order they should be analyzed in pass 2.
    final_f0 : dict
        vid -> assigned f0 for every tile in tile_centers.
    """
    all_vids = sorted(tile_centers.keys())
    if not all_vids:
        return [], {}

    neighbors = _build_neighbor_graph(tile_centers, all_vids, neighbor_radius)

    # Assign f0 to every tile via BFS from highest-pulsatility seed
    final_f0: Dict[int, float] = {}
    bfs_order: List[int] = []
    visited: set = set()
    overridden: List[Tuple[int, float, float]] = []  # (vid, own_f0, assigned_f0)

    # Process potentially disconnected components
    remaining = set(all_vids)
    while remaining:
        # Pick seed: unvisited tile with highest pulsatility
        seed = max(
            remaining,
            key=lambda v: tile_pulsatility.get(v, -np.inf),
        )
        seed_f0 = tile_f0_pass1.get(seed, np.nan)
        if not np.isfinite(seed_f0):
            # No f0 estimate at all — skip this component
            remaining.discard(seed)
            visited.add(seed)
            continue

        # BFS from seed
        queue: deque = deque()
        queue.append(seed)
        visited.add(seed)
        remaining.discard(seed)
        final_f0[seed] = seed_f0
        bfs_order.append(seed)

        while queue:
            current = queue.popleft()
            parent_f0 = final_f0[current]

            # Sort neighbors by distance (closest first) for deterministic order
            for nb_vid, _dist in sorted(neighbors[current], key=lambda x: x[1]):
                if nb_vid in visited:
                    continue
                visited.add(nb_vid)
                remaining.discard(nb_vid)

                nb_own_f0 = tile_f0_pass1.get(nb_vid, np.nan)

                # Accept own f0 if it agrees with parent within ±window
                if (np.isfinite(nb_own_f0)
                        and abs(nb_own_f0 - parent_f0) / parent_f0 <= search_window_frac):
                    final_f0[nb_vid] = nb_own_f0
                else:
                    # Inherit parent's f0
                    final_f0[nb_vid] = parent_f0
                    if np.isfinite(nb_own_f0):
                        overridden.append((nb_vid, nb_own_f0, parent_f0))

                bfs_order.append(nb_vid)
                queue.append(nb_vid)

    if verbose:
        print(f"\n{'=' * 60}")
        print("f0 PROPAGATION (BFS flood fill)")
        print(f"{'=' * 60}")
        seed_vid = bfs_order[0] if bfs_order else None
        seed_pulse = tile_pulsatility.get(seed_vid, np.nan) if seed_vid else np.nan
        print(f"  Seed tile: {seed_vid} (pulsatility SNR={seed_pulse:.1f} dB)")
        print(f"  Seed f0: {final_f0.get(seed_vid, np.nan):.3f} Hz")
        print(f"  Tiles processed: {len(bfs_order)}")
        print(f"  Tiles overridden: {len(overridden)}")
        if overridden:
            print(f"\n  {'Tile':>5} {'Own f0':>8} {'Assigned':>8} {'Delta':>8}")
            print(f"  {'-' * 35}")
            for vid, own, assigned in overridden:
                print(f"  {vid:>5} {own:>8.3f} {assigned:>8.3f} {own - assigned:>+8.3f}")
        print(f"{'=' * 60}\n")

    return bfs_order, final_f0
