"""Graph + geometry helpers used by the read-only viewer.

Slim build — only the 2 viewer-used functions from the full pertile
flow module (get_chain_coords, trace_vessel_chain) are kept; the
kymograph-sampling / GST / harmonic-fit / Poiseuille pipeline that
makes up the rest of the upstream analysis stack is excluded.
"""
from typing import List, Optional, Tuple
import numpy as np
import networkx as nx

from .config import MARGIN_PX


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
