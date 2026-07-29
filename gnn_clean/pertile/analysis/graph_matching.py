"""
Graph matching module for comparing flow measurements between different videos/timepoints.

Matches nodes and edges between two graphs based on spatial position, topology,
and geometry. Designed for comparing overlapping regions of vascular networks.

Usage:
    from pertile.analysis.graph_matching import match_graphs, analyze_reproducibility

    # Load graphs
    G1 = nx.read_gpickle("vid26_graph.gpickle")
    G2 = nx.read_gpickle("vid27_graph.gpickle")

    # Match and compare
    result = match_graphs(G1, G2, config)
    repro = analyze_reproducibility(result.comparison_df)
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from pathlib import Path
from scipy.spatial import KDTree
import networkx as nx


# Default configuration
DEFAULT_CONFIG = {
    # Offset estimation
    "offset_bin_size_um": 5.0,  # Bin size for 2D histogram voting
    "offset_refinement_radius_um": 10.0,  # Radius for mean refinement around peak

    # Node matching
    "node_match_tolerance_um": 10.0,  # Max distance for node match
    "require_mutual_nearest": True,  # Require mutual nearest neighbor
    "use_junction_nodes_only": True,  # Only match degree != 2 nodes for offset

    # Edge matching
    "edge_radius_tolerance_frac": 0.5,  # Max relative difference in radius
    "edge_length_tolerance_frac": 0.5,  # Max relative difference in length

    # Attribute names (handles different conventions)
    "pos_attr": "pos",  # Node position attribute
    "radius_attr": "mean_radius_um",
    "length_attr": "length_um",
    "flow_attr": "mean_Q_nL_s",
    "sigma_flow_attr": "sigma_Q_nL_s",

    # Output
    "verbose": True,
}


@dataclass
class MatchResult:
    """Results from graph matching."""

    # Estimated offset (add to graph1 coords to align with graph2)
    offset: np.ndarray
    offset_confidence: float  # Peak height / background

    # Node matches: node_id_graph1 -> node_id_graph2
    node_matches: Dict[int, int]
    node_match_distances: Dict[int, float]  # Distance after alignment

    # Edge matches: (u1, v1) -> (u2, v2)
    edge_matches: Dict[Tuple[int, int], Tuple[int, int]]

    # Comparison DataFrame with matched edge attributes
    comparison_df: pd.DataFrame

    # Metadata
    n_nodes_g1: int
    n_nodes_g2: int
    n_edges_g1: int
    n_edges_g2: int

    # Config used
    config: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file or use defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)

    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_node_positions(G: nx.Graph, pos_attr: str = "pos") -> Dict[int, np.ndarray]:
    """
    Extract node positions from graph.

    Handles multiple conventions:
    - pos_attr as tuple/array: node[pos_attr] = (x, y, ...)
    - Separate x, y, z attributes: node['x'], node['y'], node['z']

    Returns dict mapping node_id -> position array (2D or 3D).
    """
    positions = {}
    for node in G.nodes():
        node_data = G.nodes[node]

        # Try the specified pos_attr first
        pos = node_data.get(pos_attr)
        if pos is not None:
            positions[node] = np.array(pos)
            continue

        # Try separate x, y, z attributes
        x = node_data.get('x')
        y = node_data.get('y')
        z = node_data.get('z')

        if x is not None and y is not None:
            if z is not None:
                positions[node] = np.array([x, y, z])
            else:
                positions[node] = np.array([x, y])

    return positions


def get_junction_nodes(G: nx.Graph) -> Set[int]:
    """
    Get nodes that are junctions (degree != 2).

    These are more reliable for matching since intermediate nodes
    along edges may be placed arbitrarily.
    """
    return {node for node in G.nodes() if G.degree(node) != 2}


def compute_overlap_region(
    pos1: Dict[int, np.ndarray],
    pos2: Dict[int, np.ndarray],
    margin: float = 0.0,
) -> Tuple[Optional[Tuple[float, float, float, float]], Dict[str, Any]]:
    """
    Compute the overlapping bounding box of two sets of node positions.

    Args:
        pos1: Node positions from graph 1
        pos2: Node positions from graph 2
        margin: Extra margin to add around overlap region

    Returns:
        overlap_bbox: (x_min, x_max, y_min, y_max) or None if no overlap
        stats: Dictionary with overlap statistics
    """
    coords1 = np.array(list(pos1.values()))
    coords2 = np.array(list(pos2.values()))

    # Bounding boxes
    bbox1 = (coords1[:, 0].min(), coords1[:, 0].max(),
             coords1[:, 1].min(), coords1[:, 1].max())
    bbox2 = (coords2[:, 0].min(), coords2[:, 0].max(),
             coords2[:, 1].min(), coords2[:, 1].max())

    # Compute overlap
    x_min = max(bbox1[0], bbox2[0]) - margin
    x_max = min(bbox1[1], bbox2[1]) + margin
    y_min = max(bbox1[2], bbox2[2]) - margin
    y_max = min(bbox1[3], bbox2[3]) + margin

    # Check if there's actual overlap
    if x_min >= x_max or y_min >= y_max:
        return None, {'overlap_exists': False}

    overlap_bbox = (x_min, x_max, y_min, y_max)

    # Compute overlap statistics
    overlap_area = (x_max - x_min) * (y_max - y_min)
    area1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2])
    area2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2])

    # Count nodes in overlap region for each graph
    in_overlap1 = np.sum(
        (coords1[:, 0] >= x_min) & (coords1[:, 0] <= x_max) &
        (coords1[:, 1] >= y_min) & (coords1[:, 1] <= y_max)
    )
    in_overlap2 = np.sum(
        (coords2[:, 0] >= x_min) & (coords2[:, 0] <= x_max) &
        (coords2[:, 1] >= y_min) & (coords2[:, 1] <= y_max)
    )

    stats = {
        'overlap_exists': True,
        'overlap_bbox': overlap_bbox,
        'bbox_g1': bbox1,
        'bbox_g2': bbox2,
        'overlap_area': overlap_area,
        'overlap_frac_g1': overlap_area / area1,
        'overlap_frac_g2': overlap_area / area2,
        'n_nodes_g1_in_overlap': int(in_overlap1),
        'n_nodes_g2_in_overlap': int(in_overlap2),
        'frac_nodes_g1_in_overlap': in_overlap1 / len(pos1),
        'frac_nodes_g2_in_overlap': in_overlap2 / len(pos2),
    }

    return overlap_bbox, stats


def filter_to_overlap_region(
    G: nx.Graph,
    pos: Dict[int, np.ndarray],
    overlap_bbox: Tuple[float, float, float, float],
) -> Tuple[nx.Graph, Dict[int, np.ndarray]]:
    """
    Create a subgraph containing only nodes within the overlap region.

    Args:
        G: Input graph
        pos: Node positions
        overlap_bbox: (x_min, x_max, y_min, y_max)

    Returns:
        G_sub: Subgraph with nodes in overlap region
        pos_sub: Positions of nodes in subgraph
    """
    x_min, x_max, y_min, y_max = overlap_bbox

    # Find nodes in overlap region
    nodes_in_overlap = [
        node for node, p in pos.items()
        if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max
    ]

    # Create subgraph
    G_sub = G.subgraph(nodes_in_overlap).copy()

    # Filter positions
    pos_sub = {node: pos[node] for node in nodes_in_overlap}

    return G_sub, pos_sub


def estimate_offset_from_images(
    img1_path: str,
    img2_path: str,
    downsample: float = 0.25,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Estimate translation offset between two images using cross-correlation.

    Args:
        img1_path: Path to first image (reference)
        img2_path: Path to second image (to be aligned)
        downsample: Downsample factor for faster correlation
        verbose: Print progress

    Returns:
        offset: (dx, dy) offset to apply to image2/graph2 coordinates
        peak_value: Cross-correlation peak value (confidence)
    """
    from PIL import Image
    from scipy.ndimage import zoom
    from scipy.signal import correlate

    # Load images
    img1 = np.array(Image.open(img1_path)).astype(float)
    img2 = np.array(Image.open(img2_path)).astype(float)

    if verbose:
        print(f"Image 1: {img1.shape}")
        print(f"Image 2: {img2.shape}")

    # Normalize
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-10)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-10)

    # Downsample for speed
    img1_small = zoom(img1_norm, downsample)
    img2_small = zoom(img2_norm, downsample)

    if verbose:
        print(f"Downsampled to {img1_small.shape}")

    # Cross-correlation
    corr = correlate(img1_small, img2_small, mode='full')

    # Find peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_value = corr[peak_idx]

    h_small, w_small = img1_small.shape

    # Convert to offset in original pixel coordinates
    # Cross-correlation gives the shift of img2 relative to img1
    # We want the offset to ADD to img2 to align with img1, so negate
    dy = -((peak_idx[0] - (h_small - 1)) / downsample)
    dx = -((peak_idx[1] - (w_small - 1)) / downsample)

    offset = np.array([dx, dy])

    if verbose:
        print(f"Cross-correlation peak: {peak_idx}")
        print(f"Offset (dx, dy): ({dx:.1f}, {dy:.1f})")

    return offset, peak_value


def find_vessel_paths(G: nx.Graph, junctions: set) -> List[Dict]:
    """
    Find all vessel paths connecting junction nodes.

    A vessel path is a sequence of edges connecting two junctions
    through intermediate degree-2 nodes.

    Args:
        G: Graph
        junctions: Set of junction node IDs (degree != 2)

    Returns:
        List of path dicts with 'start', 'end', 'nodes', 'edges'
    """
    paths = []
    seen_paths = set()

    for start_node in junctions:
        for neighbor in G.neighbors(start_node):
            # Trace path until we hit another junction
            path_nodes = [start_node, neighbor]
            path_edges = [(start_node, neighbor)]

            current = neighbor
            prev = start_node

            while current not in junctions and G.degree(current) == 2:
                neighbors = list(G.neighbors(current))
                next_node = neighbors[0] if neighbors[1] == prev else neighbors[1]
                path_nodes.append(next_node)
                path_edges.append((current, next_node))
                prev = current
                current = next_node

            if current in junctions:
                # Create canonical key to avoid duplicates
                key = tuple(sorted([start_node, current]))
                if key not in seen_paths:
                    seen_paths.add(key)
                    paths.append({
                        'start': start_node,
                        'end': current,
                        'nodes': path_nodes,
                        'edges': path_edges,
                        'length': len(path_edges),
                    })

    return paths


def match_vessel_paths(
    paths1: List[Dict],
    paths2: List[Dict],
    junc_matches: Dict[int, int],
) -> List[Dict]:
    """
    Match vessel paths between two graphs based on matched junction endpoints.

    Args:
        paths1: Vessel paths from graph 1
        paths2: Vessel paths from graph 2
        junc_matches: Junction matching {g1_node: g2_node}

    Returns:
        List of matched path pairs
    """
    matched = []

    for p1 in paths1:
        start1, end1 = p1['start'], p1['end']

        if start1 in junc_matches and end1 in junc_matches:
            start2_exp = junc_matches[start1]
            end2_exp = junc_matches[end1]

            for p2 in paths2:
                start2, end2 = p2['start'], p2['end']
                if (start2 == start2_exp and end2 == end2_exp) or \
                   (start2 == end2_exp and end2 == start2_exp):
                    matched.append({
                        'path_g1': p1,
                        'path_g2': p2,
                    })
                    break

    return matched


@dataclass
class VesselMatchResult:
    """Results from vessel-level graph matching."""

    # Offset applied to G2 to align with G1
    offset: np.ndarray

    # Overlap region in G1 coordinates
    overlap_bbox: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

    # Junction matching
    junction_matches: Dict[int, int]  # G1 junction -> G2 junction
    junction_match_distances: Dict[int, float]

    # Vessel path matching
    matched_paths: List[Dict]  # List of {'path_g1': {...}, 'path_g2': {...}}

    # Statistics
    n_junctions_g1_overlap: int
    n_junctions_g2_overlap: int
    n_paths_g1: int
    n_paths_g2: int


def match_vessels_with_offset(
    G1: nx.Graph,
    G2: nx.Graph,
    offset: Union[np.ndarray, Tuple[float, float]],
    junction_tolerance: float = 20.0,
    verbose: bool = True,
) -> VesselMatchResult:
    """
    Match vessels between two graphs using a known spatial offset.

    This function:
    1. Applies the offset to G2 coordinates
    2. Finds the overlap region
    3. Matches junction nodes (degree != 2) using mutual nearest neighbor
    4. Finds vessel paths between matched junctions

    Args:
        G1: First graph (reference)
        G2: Second graph (to be aligned)
        offset: (dx, dy) offset to add to G2 coordinates to align with G1
        junction_tolerance: Max distance for junction matching (pixels)
        verbose: Print progress

    Returns:
        VesselMatchResult with all matching information
    """
    offset = np.array(offset)

    # Get node positions
    pos1 = get_node_positions(G1)
    pos2 = get_node_positions(G2)

    # Apply offset to G2
    pos2_shifted = {n: p + offset for n, p in pos2.items()}

    # Compute overlap region
    coords1 = np.array(list(pos1.values()))
    coords2_shifted = np.array(list(pos2_shifted.values()))

    x_overlap = (max(coords1[:, 0].min(), coords2_shifted[:, 0].min()),
                 min(coords1[:, 0].max(), coords2_shifted[:, 0].max()))
    y_overlap = (max(coords1[:, 1].min(), coords2_shifted[:, 1].min()),
                 min(coords1[:, 1].max(), coords2_shifted[:, 1].max()))

    overlap_bbox = (x_overlap[0], x_overlap[1], y_overlap[0], y_overlap[1])

    if verbose:
        print(f"Offset applied: ({offset[0]:.1f}, {offset[1]:.1f})")
        print(f"Overlap region: x=[{x_overlap[0]:.0f}, {x_overlap[1]:.0f}], "
              f"y=[{y_overlap[0]:.0f}, {y_overlap[1]:.0f}]")

    # Check for valid overlap
    if x_overlap[1] <= x_overlap[0] or y_overlap[1] <= y_overlap[0]:
        if verbose:
            print("WARNING: No overlap region found!")
        return VesselMatchResult(
            offset=offset,
            overlap_bbox=overlap_bbox,
            junction_matches={},
            junction_match_distances={},
            matched_paths=[],
            n_junctions_g1_overlap=0,
            n_junctions_g2_overlap=0,
            n_paths_g1=0,
            n_paths_g2=0,
        )

    def in_overlap(p):
        return (x_overlap[0] <= p[0] <= x_overlap[1] and
                y_overlap[0] <= p[1] <= y_overlap[1])

    # Get junction nodes
    junc1_all = get_junction_nodes(G1)
    junc2_all = get_junction_nodes(G2)

    # Filter to overlap region
    junc1_overlap = {n for n in junc1_all if n in pos1 and in_overlap(pos1[n])}
    junc2_overlap = {n for n in junc2_all if n in pos2_shifted and in_overlap(pos2_shifted[n])}

    if verbose:
        print(f"Junctions in overlap: G1={len(junc1_overlap)}, G2={len(junc2_overlap)}")

    # Match junctions using mutual nearest neighbor
    junc1_list = list(junc1_overlap)
    junc2_list = list(junc2_overlap)

    if len(junc1_list) == 0 or len(junc2_list) == 0:
        if verbose:
            print("WARNING: No junctions in overlap region!")
        return VesselMatchResult(
            offset=offset,
            overlap_bbox=overlap_bbox,
            junction_matches={},
            junction_match_distances={},
            matched_paths=[],
            n_junctions_g1_overlap=len(junc1_overlap),
            n_junctions_g2_overlap=len(junc2_overlap),
            n_paths_g1=0,
            n_paths_g2=0,
        )

    junc1_coords = np.array([pos1[n] for n in junc1_list])
    junc2_coords = np.array([pos2_shifted[n] for n in junc2_list])

    tree1 = KDTree(junc1_coords)
    tree2 = KDTree(junc2_coords)

    junction_matches = {}
    junction_match_distances = {}

    for i, n1 in enumerate(junc1_list):
        dist, idx = tree2.query(junc1_coords[i])
        if dist < junction_tolerance:
            n2 = junc2_list[idx]
            # Check mutual nearest neighbor
            _, idx_back = tree1.query(junc2_coords[idx])
            if junc1_list[idx_back] == n1:
                junction_matches[n1] = n2
                junction_match_distances[n1] = dist

    if verbose:
        print(f"Matched junctions: {len(junction_matches)} / {len(junc1_overlap)} "
              f"({100*len(junction_matches)/len(junc1_overlap):.1f}%)")
        if junction_match_distances:
            dists = list(junction_match_distances.values())
            print(f"  Distances: median={np.median(dists):.1f}, max={max(dists):.1f}")

    # Find vessel paths in overlap region
    paths1 = find_vessel_paths(G1, junc1_all)
    paths2 = find_vessel_paths(G2, junc2_all)

    # Filter to paths with both endpoints in overlap
    paths1_overlap = [p for p in paths1
                      if p['start'] in junc1_overlap and p['end'] in junc1_overlap]
    paths2_overlap = [p for p in paths2
                      if p['start'] in junc2_overlap and p['end'] in junc2_overlap]

    if verbose:
        print(f"Vessel paths in overlap: G1={len(paths1_overlap)}, G2={len(paths2_overlap)}")

    # Match vessel paths
    matched_paths = match_vessel_paths(paths1_overlap, paths2_overlap, junction_matches)

    if verbose:
        print(f"Matched vessel paths: {len(matched_paths)}")

    return VesselMatchResult(
        offset=offset,
        overlap_bbox=overlap_bbox,
        junction_matches=junction_matches,
        junction_match_distances=junction_match_distances,
        matched_paths=matched_paths,
        n_junctions_g1_overlap=len(junc1_overlap),
        n_junctions_g2_overlap=len(junc2_overlap),
        n_paths_g1=len(paths1_overlap),
        n_paths_g2=len(paths2_overlap),
    )


def plot_matched_vessels(
    G1: nx.Graph,
    G2: nx.Graph,
    result: VesselMatchResult,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize matched vessel paths between two graphs.

    Args:
        G1, G2: Input graphs
        result: VesselMatchResult from match_vessels_with_offset()
        output_path: Path to save figure
        title: Plot title (auto-generated if None)
        show: Whether to display
    """
    import matplotlib.pyplot as plt

    pos1 = get_node_positions(G1)
    pos2 = get_node_positions(G2)

    # Apply offset to G2
    pos2_shifted = {n: p + result.offset for n, p in pos2.items()}

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot all edges very faintly
    for u, v in G1.edges():
        if u in pos1 and v in pos1:
            ax.plot([pos1[u][0], pos1[v][0]], [pos1[u][1], pos1[v][1]],
                    'b-', linewidth=0.2, alpha=0.2)
    for u, v in G2.edges():
        if u in pos2_shifted and v in pos2_shifted:
            ax.plot([pos2_shifted[u][0], pos2_shifted[v][0]],
                    [pos2_shifted[u][1], pos2_shifted[v][1]],
                    'r-', linewidth=0.2, alpha=0.2)

    # Plot matched vessel paths with distinct colors
    n_paths = len(result.matched_paths)
    if n_paths > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_paths, 20)))

        for i, mp in enumerate(result.matched_paths):
            p1 = mp['path_g1']
            p2 = mp['path_g2']
            color = colors[i % 20]

            # Draw G1 path (solid)
            for u, v in p1['edges']:
                if u in pos1 and v in pos1:
                    ax.plot([pos1[u][0], pos1[v][0]], [pos1[u][1], pos1[v][1]],
                            color=color, linewidth=2.5, alpha=0.9)

            # Draw G2 path (dashed)
            for u, v in p2['edges']:
                if u in pos2_shifted and v in pos2_shifted:
                    ax.plot([pos2_shifted[u][0], pos2_shifted[v][0]],
                            [pos2_shifted[u][1], pos2_shifted[v][1]],
                            color=color, linewidth=2.5, alpha=0.9, linestyle='--')

    # Mark matched junctions
    for n1, n2 in result.junction_matches.items():
        if n1 in pos1:
            ax.scatter([pos1[n1][0]], [pos1[n1][1]],
                       c='blue', s=50, zorder=5, edgecolors='white', linewidths=1)
        if n2 in pos2_shifted:
            ax.scatter([pos2_shifted[n2][0]], [pos2_shifted[n2][1]],
                       c='red', s=50, zorder=5, edgecolors='white', linewidths=1)

    # Focus on overlap region
    x_min, x_max, y_min, y_max = result.overlap_bbox
    margin = 20
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_aspect('equal')
    if title is None:
        title = f'Matched Vessel Paths: {n_paths} vessels\n(solid=G1, dashed=G2)'
    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    return fig


def plot_skeleton_overlay(
    G1: nx.Graph,
    G2: nx.Graph,
    pos1: Dict[int, np.ndarray],
    pos2: Dict[int, np.ndarray],
    overlap_bbox: Optional[Tuple[float, float, float, float]] = None,
    output_path: Optional[str] = None,
    title: str = "Skeleton Overlay",
    show: bool = True,
):
    """
    Plot both graph skeletons overlaid to visualize spatial relationship.

    Args:
        G1, G2: Input graphs
        pos1, pos2: Node positions
        overlap_bbox: Optional bounding box to highlight
        output_path: Path to save figure
        title: Plot title
        show: Whether to display
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 14))

    # Get junction nodes for each graph
    junctions1 = get_junction_nodes(G1)
    junctions2 = get_junction_nodes(G2)

    # Plot edges for G1 (blue)
    for u, v in G1.edges():
        if u in pos1 and v in pos1:
            x = [pos1[u][0], pos1[v][0]]
            y = [pos1[u][1], pos1[v][1]]
            ax.plot(x, y, 'b-', linewidth=0.8, alpha=0.6)

    # Plot edges for G2 (red)
    for u, v in G2.edges():
        if u in pos2 and v in pos2:
            x = [pos2[u][0], pos2[v][0]]
            y = [pos2[u][1], pos2[v][1]]
            ax.plot(x, y, 'r-', linewidth=0.8, alpha=0.6)

    # Plot junction nodes for G1 (blue dots)
    junc_coords1 = np.array([pos1[n] for n in junctions1 if n in pos1])
    if len(junc_coords1) > 0:
        ax.scatter(junc_coords1[:, 0], junc_coords1[:, 1],
                   c='blue', s=15, zorder=5, alpha=0.7, label='G1 junctions')

    # Plot junction nodes for G2 (red dots)
    junc_coords2 = np.array([pos2[n] for n in junctions2 if n in pos2])
    if len(junc_coords2) > 0:
        ax.scatter(junc_coords2[:, 0], junc_coords2[:, 1],
                   c='red', s=15, zorder=5, alpha=0.7, label='G2 junctions')

    # Highlight overlap region if provided
    if overlap_bbox is not None:
        x_min, x_max, y_min, y_max = overlap_bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor='green', linewidth=2,
                              linestyle='--', label='Overlap region')
        ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlay plot to: {output_path}")

    if show:
        plt.show()

    return fig


def estimate_offset(
    pos1: Dict[int, np.ndarray],
    pos2: Dict[int, np.ndarray],
    bin_size: float = 5.0,
    refinement_radius: float = 10.0,
    check_zero_offset: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Estimate translation offset using 2D histogram voting.

    For each pair of nodes (one from each graph), compute the offset
    that would align them. The true offset will have many pairs agreeing,
    creating a peak in the offset histogram.

    Args:
        pos1: Node positions from graph 1 {node_id: (x, y, ...)}
        pos2: Node positions from graph 2 {node_id: (x, y, ...)}
        bin_size: Bin size for histogram in same units as positions
        refinement_radius: Radius around peak for mean refinement
        check_zero_offset: If True, first check if zero offset gives good matches
        verbose: Print progress

    Returns:
        offset: Array to add to pos1 coordinates to align with pos2
        confidence: Peak height relative to median (higher = more confident)
    """
    coords1 = np.array(list(pos1.values()))
    coords2 = np.array(list(pos2.values()))

    n1, ndim = coords1.shape
    n2 = coords2.shape[0]

    # First, check if graphs are already aligned (zero offset)
    if check_zero_offset:
        tree2 = KDTree(coords2)
        zero_matches = 0
        for c1 in coords1:
            dist, _ = tree2.query(c1)
            if dist < bin_size * 2:
                zero_matches += 1

        zero_match_frac = zero_matches / n1
        if verbose:
            print(f"  Zero-offset check: {zero_matches}/{n1} ({100*zero_match_frac:.1f}%) nodes match within {bin_size*2:.1f} units")

        # If >20% match with zero offset, graphs are likely already aligned
        if zero_match_frac > 0.2:
            if verbose:
                print(f"  -> Graphs appear to be already aligned, using zero offset")
            return np.zeros(ndim), zero_match_frac * n1  # confidence = number of matches

    if verbose:
        print(f"  Estimating offset from {n1} × {n2} = {n1*n2} candidate pairs")

    # Compute all pairwise offset candidates
    # offset = pos2 - pos1 (what to add to pos1 to get pos2)
    offset_candidates = []
    for c1 in coords1:
        for c2 in coords2:
            offset_candidates.append(c2 - c1)

    offset_candidates = np.array(offset_candidates)

    # Determine histogram range
    ranges = []
    for d in range(ndim):
        margin = bin_size * 2
        ranges.append((offset_candidates[:, d].min() - margin,
                      offset_candidates[:, d].max() + margin))

    # Compute number of bins
    n_bins = [int((r[1] - r[0]) / bin_size) + 1 for r in ranges]

    # Build histogram
    histogram, edges = np.histogramdd(offset_candidates, bins=n_bins, range=ranges)

    # Find peak
    peak_idx = np.unravel_index(np.argmax(histogram), histogram.shape)
    peak_value = histogram[peak_idx]

    # Compute offset at peak (bin center)
    offset_coarse = np.array([
        edges[d][peak_idx[d]] + bin_size / 2
        for d in range(ndim)
    ])

    # Confidence: peak height relative to median non-zero bin
    nonzero_bins = histogram[histogram > 0]
    if len(nonzero_bins) > 1:
        confidence = peak_value / np.median(nonzero_bins)
    else:
        confidence = 1.0

    # Refine with mean of candidates near peak
    distances_to_peak = np.linalg.norm(offset_candidates - offset_coarse, axis=1)
    near_peak_mask = distances_to_peak < refinement_radius

    if near_peak_mask.sum() > 0:
        offset_refined = offset_candidates[near_peak_mask].mean(axis=0)
    else:
        offset_refined = offset_coarse

    if verbose:
        print(f"  Coarse offset: {offset_coarse}")
        print(f"  Refined offset: {offset_refined}")
        print(f"  Peak votes: {int(peak_value)}, confidence: {confidence:.1f}×")

    return offset_refined, confidence


def match_nodes(
    G1: nx.Graph,
    G2: nx.Graph,
    pos1: Dict[int, np.ndarray],
    pos2: Dict[int, np.ndarray],
    offset: np.ndarray,
    tolerance: float = 10.0,
    require_mutual: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Match nodes between graphs using spatial proximity after alignment.

    Args:
        G1, G2: Input graphs
        pos1, pos2: Node positions
        offset: Translation offset to add to pos1
        tolerance: Maximum distance for a match (in position units)
        require_mutual: If True, require mutual nearest neighbor
        verbose: Print progress

    Returns:
        node_matches: Dict mapping node1_id -> node2_id
        match_distances: Dict mapping node1_id -> distance to matched node
    """
    # Apply offset to align pos1 with pos2
    pos1_aligned = {node: pos + offset for node, pos in pos1.items()}

    # Build arrays and indices
    nodes1 = list(pos1_aligned.keys())
    nodes2 = list(pos2.keys())

    coords1_aligned = np.array([pos1_aligned[n] for n in nodes1])
    coords2 = np.array([pos2[n] for n in nodes2])

    # Build KD-trees
    tree1 = KDTree(coords1_aligned)
    tree2 = KDTree(coords2)

    # Find matches
    node_matches = {}
    match_distances = {}

    for i, node1 in enumerate(nodes1):
        # Find nearest in graph2
        dist_to_2, idx_2 = tree2.query(coords1_aligned[i])

        if dist_to_2 > tolerance:
            continue

        node2 = nodes2[idx_2]

        if require_mutual:
            # Check if node2's nearest in graph1 is node1
            dist_back, idx_back = tree1.query(coords2[idx_2])
            if nodes1[idx_back] != node1:
                continue  # Not mutual nearest neighbor

        node_matches[node1] = node2
        match_distances[node1] = dist_to_2

    if verbose:
        print(f"  Matched {len(node_matches)} / {len(nodes1)} nodes "
              f"({100*len(node_matches)/len(nodes1):.1f}%)")
        if match_distances:
            dists = list(match_distances.values())
            print(f"  Match distances: median={np.median(dists):.2f}, "
                  f"max={np.max(dists):.2f}")

    return node_matches, match_distances


def match_edges(
    G1: nx.Graph,
    G2: nx.Graph,
    node_matches: Dict[int, int],
    radius_attr: str = "mean_radius_um",
    length_attr: str = "length_um",
    radius_tol: float = 0.5,
    length_tol: float = 0.5,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Match edges between graphs based on matched endpoint nodes.

    Args:
        G1, G2: Input graphs
        node_matches: Node matching from match_nodes()
        radius_attr, length_attr: Edge attribute names
        radius_tol, length_tol: Maximum relative difference (fraction)
        verbose: Print progress

    Returns:
        edge_matches: Dict mapping (u1, v1) -> (u2, v2)
    """
    edge_matches = {}

    # Track reasons for non-matches
    n_no_node_match = 0
    n_no_edge = 0
    n_geom_mismatch = 0

    for u1, v1 in G1.edges():
        # Check if both endpoints have matches
        if u1 not in node_matches or v1 not in node_matches:
            n_no_node_match += 1
            continue

        u2 = node_matches[u1]
        v2 = node_matches[v1]

        # Check if corresponding edge exists in G2
        if G2.has_edge(u2, v2):
            edge2 = (u2, v2)
        elif G2.has_edge(v2, u2):
            edge2 = (v2, u2)
        else:
            n_no_edge += 1
            continue

        # Optional: Check geometry consistency
        attr1 = G1.edges[u1, v1]
        attr2 = G2.edges[edge2]

        # Check radius similarity
        r1 = attr1.get(radius_attr)
        r2 = attr2.get(radius_attr)
        if r1 is not None and r2 is not None and r1 > 0 and r2 > 0:
            rel_diff_r = abs(r1 - r2) / max(r1, r2)
            if rel_diff_r > radius_tol:
                n_geom_mismatch += 1
                continue

        # Check length similarity
        L1 = attr1.get(length_attr)
        L2 = attr2.get(length_attr)
        if L1 is not None and L2 is not None and L1 > 0 and L2 > 0:
            rel_diff_L = abs(L1 - L2) / max(L1, L2)
            if rel_diff_L > length_tol:
                n_geom_mismatch += 1
                continue

        edge_matches[(u1, v1)] = edge2

    if verbose:
        print(f"  Matched {len(edge_matches)} / {G1.number_of_edges()} edges "
              f"({100*len(edge_matches)/G1.number_of_edges():.1f}%)")
        print(f"    No node match: {n_no_node_match}")
        print(f"    No corresponding edge: {n_no_edge}")
        print(f"    Geometry mismatch: {n_geom_mismatch}")

    return edge_matches


def build_comparison_df(
    G1: nx.Graph,
    G2: nx.Graph,
    edge_matches: Dict[Tuple[int, int], Tuple[int, int]],
    node_matches: Dict[int, int],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build DataFrame comparing attributes of matched edges.

    Args:
        G1, G2: Input graphs
        edge_matches: Edge matching from match_edges()
        node_matches: Node matching for position info
        config: Configuration dict with attribute names

    Returns:
        DataFrame with columns for both graphs' edge attributes
    """
    radius_attr = config.get("radius_attr", "mean_radius_um")
    length_attr = config.get("length_attr", "length_um")
    flow_attr = config.get("flow_attr", "mean_Q_nL_s")
    sigma_attr = config.get("sigma_flow_attr", "sigma_Q_nL_s")

    rows = []
    for edge1, edge2 in edge_matches.items():
        u1, v1 = edge1
        u2, v2 = edge2
        attr1 = G1.edges[edge1]
        attr2 = G2.edges[edge2]

        # Handle alternate attribute names
        Q1 = attr1.get(flow_attr) or attr1.get('mean_Q')
        Q2 = attr2.get(flow_attr) or attr2.get('mean_Q')
        sigma1 = attr1.get(sigma_attr) or attr1.get('sigma_Q')
        sigma2 = attr2.get(sigma_attr) or attr2.get('sigma_Q')

        # Get node positions for skeleton comparison
        x1_u = G1.nodes[u1].get('x')
        y1_u = G1.nodes[u1].get('y')
        x1_v = G1.nodes[v1].get('x')
        y1_v = G1.nodes[v1].get('y')
        x2_u = G2.nodes[u2].get('x')
        y2_u = G2.nodes[u2].get('y')
        x2_v = G2.nodes[v2].get('x')
        y2_v = G2.nodes[v2].get('y')

        rows.append({
            'edge_g1': edge1,
            'edge_g2': edge2,

            # Node positions (skeleton)
            'x1_u': x1_u, 'y1_u': y1_u,
            'x1_v': x1_v, 'y1_v': y1_v,
            'x2_u': x2_u, 'y2_u': y2_u,
            'x2_v': x2_v, 'y2_v': y2_v,

            # Node degrees (topology)
            'deg1_u': G1.degree(u1), 'deg1_v': G1.degree(v1),
            'deg2_u': G2.degree(u2), 'deg2_v': G2.degree(v2),

            # Geometry
            'radius_g1': attr1.get(radius_attr),
            'radius_g2': attr2.get(radius_attr),
            'length_g1': attr1.get(length_attr),
            'length_g2': attr2.get(length_attr),

            # Flow measurements
            'Q_g1': Q1,
            'Q_g2': Q2,
            'sigma_Q_g1': sigma1,
            'sigma_Q_g2': sigma2,

            # MCMC predictions (if available)
            'Q_pred_g1': attr1.get('Q_predicted'),
            'Q_pred_g2': attr2.get('Q_predicted'),

            # Quality metrics
            'chi2_g1': attr1.get('chi2_reduced'),
            'chi2_g2': attr2.get('chi2_reduced'),
            'snr_g1': attr1.get('snr'),
            'snr_g2': attr2.get('snr'),
        })

    return pd.DataFrame(rows)


def analyze_skeleton_match(
    result: 'MatchResult',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze the quality of skeleton (topology + geometry) matching.

    This focuses purely on whether the graph structures match,
    independent of flow measurements.

    Args:
        result: MatchResult from match_graphs()
        verbose: Print summary

    Returns:
        Dictionary with skeleton match statistics
    """
    df = result.comparison_df

    if len(df) == 0:
        if verbose:
            print("No matched edges to analyze")
        return {}

    stats = {}

    # Position matching quality
    if 'x1_u' in df.columns and 'x2_u' in df.columns:
        # Distance between matched node positions
        dist_u = np.sqrt((df['x1_u'] - df['x2_u'])**2 + (df['y1_u'] - df['y2_u'])**2)
        dist_v = np.sqrt((df['x1_v'] - df['x2_v'])**2 + (df['y1_v'] - df['y2_v'])**2)
        all_dists = np.concatenate([dist_u.values, dist_v.values])

        stats['position_dist_median'] = np.nanmedian(all_dists)
        stats['position_dist_max'] = np.nanmax(all_dists)
        stats['position_dist_mean'] = np.nanmean(all_dists)

    # Topology matching quality (degree agreement)
    if 'deg1_u' in df.columns:
        deg_match_u = (df['deg1_u'] == df['deg2_u']).mean()
        deg_match_v = (df['deg1_v'] == df['deg2_v']).mean()
        stats['degree_agreement'] = (deg_match_u + deg_match_v) / 2

    # Geometry matching quality
    r1 = pd.to_numeric(df['radius_g1'], errors='coerce')
    r2 = pd.to_numeric(df['radius_g2'], errors='coerce')
    valid_r = (~np.isnan(r1)) & (~np.isnan(r2)) & (r1 > 0) & (r2 > 0)
    if valid_r.sum() > 5:
        stats['radius_correlation'] = np.corrcoef(r1[valid_r], r2[valid_r])[0, 1]
        rel_diff_r = np.abs(r1[valid_r] - r2[valid_r]) / np.maximum(r1[valid_r], r2[valid_r])
        stats['radius_rel_diff_median'] = np.median(rel_diff_r)

    L1 = pd.to_numeric(df['length_g1'], errors='coerce')
    L2 = pd.to_numeric(df['length_g2'], errors='coerce')
    valid_L = (~np.isnan(L1)) & (~np.isnan(L2)) & (L1 > 0) & (L2 > 0)
    if valid_L.sum() > 5:
        stats['length_correlation'] = np.corrcoef(L1[valid_L], L2[valid_L])[0, 1]
        rel_diff_L = np.abs(L1[valid_L] - L2[valid_L]) / np.maximum(L1[valid_L], L2[valid_L])
        stats['length_rel_diff_median'] = np.median(rel_diff_L)

    # Overall match counts
    stats['n_node_matches'] = len(result.node_matches)
    stats['n_edge_matches'] = len(result.edge_matches)
    stats['node_match_rate'] = len(result.node_matches) / result.n_nodes_g1
    stats['edge_match_rate'] = len(result.edge_matches) / result.n_edges_g1

    if verbose:
        print("\n" + "=" * 60)
        print("SKELETON MATCH ANALYSIS")
        print("=" * 60)

        print(f"\nMATCH COUNTS:")
        print(f"  Nodes matched: {stats['n_node_matches']} / {result.n_nodes_g1} "
              f"({100*stats['node_match_rate']:.1f}%)")
        print(f"  Edges matched: {stats['n_edge_matches']} / {result.n_edges_g1} "
              f"({100*stats['edge_match_rate']:.1f}%)")

        if 'position_dist_median' in stats:
            print(f"\nPOSITION AGREEMENT:")
            print(f"  Median distance: {stats['position_dist_median']:.2f}")
            print(f"  Max distance: {stats['position_dist_max']:.2f}")

        if 'degree_agreement' in stats:
            print(f"\nTOPOLOGY AGREEMENT:")
            print(f"  Degree match rate: {100*stats['degree_agreement']:.1f}%")

        if 'radius_correlation' in stats:
            print(f"\nGEOMETRY AGREEMENT:")
            print(f"  Radius correlation: {stats['radius_correlation']:.3f}")
            print(f"  Radius rel diff median: {100*stats['radius_rel_diff_median']:.1f}%")

        if 'length_correlation' in stats:
            print(f"  Length correlation: {stats['length_correlation']:.3f}")
            print(f"  Length rel diff median: {100*stats['length_rel_diff_median']:.1f}%")

    return stats


def match_graphs(
    G1: nx.Graph,
    G2: nx.Graph,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> MatchResult:
    """
    Main function: Match nodes and edges between two graphs.

    Args:
        G1, G2: NetworkX graphs to match
        config: Configuration dict (overrides config_path)
        config_path: Path to JSON config file

    Returns:
        MatchResult with matches and comparison DataFrame
    """
    # Load config
    if config is None:
        config = load_config(config_path)

    verbose = config.get("verbose", True)

    if verbose:
        print("=" * 60)
        print("GRAPH MATCHING")
        print("=" * 60)
        print(f"Graph 1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
        print(f"Graph 2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

    # Extract positions
    pos_attr = config.get("pos_attr", "pos")
    pos1_all = get_node_positions(G1, pos_attr)
    pos2_all = get_node_positions(G2, pos_attr)

    if verbose:
        print(f"\nNodes with positions: G1={len(pos1_all)}, G2={len(pos2_all)}")

    # Detect overlap region
    overlap_bbox, overlap_stats = compute_overlap_region(pos1_all, pos2_all)

    if verbose:
        if overlap_stats['overlap_exists']:
            print(f"\nOVERLAP REGION DETECTED:")
            print(f"  G1 bbox: x=[{overlap_stats['bbox_g1'][0]:.1f}, {overlap_stats['bbox_g1'][1]:.1f}], "
                  f"y=[{overlap_stats['bbox_g1'][2]:.1f}, {overlap_stats['bbox_g1'][3]:.1f}]")
            print(f"  G2 bbox: x=[{overlap_stats['bbox_g2'][0]:.1f}, {overlap_stats['bbox_g2'][1]:.1f}], "
                  f"y=[{overlap_stats['bbox_g2'][2]:.1f}, {overlap_stats['bbox_g2'][3]:.1f}]")
            print(f"  Overlap: x=[{overlap_bbox[0]:.1f}, {overlap_bbox[1]:.1f}], "
                  f"y=[{overlap_bbox[2]:.1f}, {overlap_bbox[3]:.1f}]")
            print(f"  Overlap covers {100*overlap_stats['overlap_frac_g1']:.1f}% of G1 area, "
                  f"{100*overlap_stats['overlap_frac_g2']:.1f}% of G2 area")
            print(f"  G1 nodes in overlap: {overlap_stats['n_nodes_g1_in_overlap']} "
                  f"({100*overlap_stats['frac_nodes_g1_in_overlap']:.1f}%)")
            print(f"  G2 nodes in overlap: {overlap_stats['n_nodes_g2_in_overlap']} "
                  f"({100*overlap_stats['frac_nodes_g2_in_overlap']:.1f}%)")
        else:
            print("\n  WARNING: No spatial overlap between graphs!")

    # For offset estimation, optionally use only junction nodes
    if config.get("use_junction_nodes_only", True):
        junctions1 = get_junction_nodes(G1)
        junctions2 = get_junction_nodes(G2)
        pos1_junc = {n: p for n, p in pos1_all.items() if n in junctions1}
        pos2_junc = {n: p for n, p in pos2_all.items() if n in junctions2}
        if verbose:
            print(f"\nJunction nodes: G1={len(pos1_junc)}, G2={len(pos2_junc)}")
    else:
        pos1_junc = pos1_all
        pos2_junc = pos2_all

    # Step 1: Estimate offset
    if verbose:
        print("\nStep 1: Estimating translation offset...")

    offset, confidence = estimate_offset(
        pos1_junc, pos2_junc,
        bin_size=config.get("offset_bin_size_um", 5.0),
        refinement_radius=config.get("offset_refinement_radius_um", 10.0),
        verbose=verbose,
    )

    # Step 2: Match nodes (using all nodes, not just junctions)
    if verbose:
        print("\nStep 2: Matching nodes...")

    node_matches, match_distances = match_nodes(
        G1, G2, pos1_all, pos2_all, offset,
        tolerance=config.get("node_match_tolerance_um", 10.0),
        require_mutual=config.get("require_mutual_nearest", True),
        verbose=verbose,
    )

    # Step 3: Match edges
    if verbose:
        print("\nStep 3: Matching edges...")

    edge_matches = match_edges(
        G1, G2, node_matches,
        radius_attr=config.get("radius_attr", "mean_radius_um"),
        length_attr=config.get("length_attr", "length_um"),
        radius_tol=config.get("edge_radius_tolerance_frac", 0.5),
        length_tol=config.get("edge_length_tolerance_frac", 0.5),
        verbose=verbose,
    )

    # Step 4: Build comparison DataFrame
    if verbose:
        print("\nStep 4: Building comparison table...")

    comparison_df = build_comparison_df(G1, G2, edge_matches, node_matches, config)

    if verbose:
        n_both_Q = comparison_df.dropna(subset=['Q_g1', 'Q_g2']).shape[0]
        print(f"  Edges with Q_obs in both graphs: {n_both_Q}")

    return MatchResult(
        offset=offset,
        offset_confidence=confidence,
        node_matches=node_matches,
        node_match_distances=match_distances,
        edge_matches=edge_matches,
        comparison_df=comparison_df,
        n_nodes_g1=G1.number_of_nodes(),
        n_nodes_g2=G2.number_of_nodes(),
        n_edges_g1=G1.number_of_edges(),
        n_edges_g2=G2.number_of_edges(),
        config=config,
    )


@dataclass
class ReproducibilityResult:
    """Results from reproducibility analysis."""

    # Sample size
    n_matched_edges: int
    n_both_measured: int

    # Flow reproducibility
    sigma_reproducibility: float  # Std of differences / sqrt(2)
    correlation: float  # Pearson correlation
    sign_agreement: float  # Fraction with same sign
    median_relative_error: float  # Median |ΔQ| / |Q_mean|

    # Uncertainty calibration
    z_score_mean: float
    z_score_std: float  # Should be ~1 if uncertainties calibrated

    # Geometry reproducibility
    radius_correlation: Optional[float]
    length_correlation: Optional[float]

    # Detailed arrays
    Q_g1: np.ndarray
    Q_g2: np.ndarray
    delta_Q: np.ndarray
    z_scores: np.ndarray


def analyze_reproducibility(
    comparison_df: pd.DataFrame,
    verbose: bool = True,
) -> ReproducibilityResult:
    """
    Analyze measurement reproducibility from matched edges.

    Args:
        comparison_df: DataFrame from match_graphs()
        verbose: Print summary statistics

    Returns:
        ReproducibilityResult with statistics
    """
    n_matched = len(comparison_df)

    # Filter to edges with Q measurements in both
    both_measured = comparison_df.dropna(subset=['Q_g1', 'Q_g2'])
    n_both = len(both_measured)

    if verbose:
        print("\n" + "=" * 60)
        print("REPRODUCIBILITY ANALYSIS")
        print("=" * 60)
        print(f"Matched edges: {n_matched}")
        print(f"Both have Q_obs: {n_both}")

    if n_both < 5:
        print("WARNING: Too few matched edges with measurements for analysis")
        return ReproducibilityResult(
            n_matched_edges=n_matched,
            n_both_measured=n_both,
            sigma_reproducibility=np.nan,
            correlation=np.nan,
            sign_agreement=np.nan,
            median_relative_error=np.nan,
            z_score_mean=np.nan,
            z_score_std=np.nan,
            radius_correlation=None,
            length_correlation=None,
            Q_g1=np.array([]),
            Q_g2=np.array([]),
            delta_Q=np.array([]),
            z_scores=np.array([]),
        )

    Q1 = both_measured['Q_g1'].values
    Q2 = both_measured['Q_g2'].values

    # Flow difference
    delta_Q = Q2 - Q1

    # Reproducibility std (assuming equal variance in both measurements)
    sigma_repro = np.std(delta_Q) / np.sqrt(2)

    # Correlation
    correlation = np.corrcoef(Q1, Q2)[0, 1]

    # Sign agreement
    sign_agree = np.mean(np.sign(Q1) == np.sign(Q2))

    # Relative error (using mean of absolute values)
    Q_mean_abs = (np.abs(Q1) + np.abs(Q2)) / 2
    # Avoid division by very small flows
    valid_for_rel = Q_mean_abs > 0.1
    if valid_for_rel.sum() > 0:
        rel_error = np.abs(delta_Q[valid_for_rel]) / Q_mean_abs[valid_for_rel]
        median_rel_error = np.median(rel_error)
    else:
        median_rel_error = np.nan

    # Z-scores using propagated uncertainties
    sigma1 = both_measured['sigma_Q_g1'].values
    sigma2 = both_measured['sigma_Q_g2'].values

    # Handle missing uncertainties
    valid_sigma = (~np.isnan(sigma1)) & (~np.isnan(sigma2)) & (sigma1 > 0) & (sigma2 > 0)

    if valid_sigma.sum() > 5:
        sigma_expected = np.sqrt(sigma1[valid_sigma]**2 + sigma2[valid_sigma]**2)
        z_scores = delta_Q[valid_sigma] / sigma_expected
        z_mean = np.mean(z_scores)
        z_std = np.std(z_scores)
    else:
        z_scores = np.array([])
        z_mean = np.nan
        z_std = np.nan

    # Geometry reproducibility
    r1 = pd.to_numeric(comparison_df['radius_g1'], errors='coerce').values
    r2 = pd.to_numeric(comparison_df['radius_g2'], errors='coerce').values
    valid_r = (~np.isnan(r1)) & (~np.isnan(r2))
    if valid_r.sum() > 5:
        r_corr = np.corrcoef(r1[valid_r], r2[valid_r])[0, 1]
    else:
        r_corr = None

    L1 = pd.to_numeric(comparison_df['length_g1'], errors='coerce').values
    L2 = pd.to_numeric(comparison_df['length_g2'], errors='coerce').values
    valid_L = (~np.isnan(L1)) & (~np.isnan(L2))
    if valid_L.sum() > 5:
        L_corr = np.corrcoef(L1[valid_L], L2[valid_L])[0, 1]
    else:
        L_corr = None

    if verbose:
        print(f"\nFLOW REPRODUCIBILITY:")
        print(f"  σ_reproducibility: {sigma_repro:.2f} nL/s")
        print(f"  Correlation: {correlation:.3f}")
        print(f"  Sign agreement: {100*sign_agree:.1f}%")
        print(f"  Median |ΔQ|/|Q_mean|: {100*median_rel_error:.1f}%")

        print(f"\nUNCERTAINTY CALIBRATION:")
        print(f"  z-score mean: {z_mean:.2f} (should be ~0)")
        print(f"  z-score std: {z_std:.2f} (should be ~1)")
        if z_std > 1.5:
            print(f"  -> Uncertainties underestimated by ~{z_std:.1f}×")

        print(f"\nGEOMETRY REPRODUCIBILITY:")
        if r_corr is not None:
            print(f"  Radius correlation: {r_corr:.3f}")
        if L_corr is not None:
            print(f"  Length correlation: {L_corr:.3f}")

    return ReproducibilityResult(
        n_matched_edges=n_matched,
        n_both_measured=n_both,
        sigma_reproducibility=sigma_repro,
        correlation=correlation,
        sign_agreement=sign_agree,
        median_relative_error=median_rel_error,
        z_score_mean=z_mean,
        z_score_std=z_std,
        radius_correlation=r_corr,
        length_correlation=L_corr,
        Q_g1=Q1,
        Q_g2=Q2,
        delta_Q=delta_Q,
        z_scores=z_scores,
    )


def plot_reproducibility(
    result: MatchResult,
    repro: ReproducibilityResult,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Create diagnostic plots for graph matching and reproducibility.

    Args:
        result: MatchResult from match_graphs()
        repro: ReproducibilityResult from analyze_reproducibility()
        output_path: Path to save figure (optional)
        show: Whether to display figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    df = result.comparison_df

    # Plot 1: Q_g1 vs Q_g2 scatter
    ax = axes[0, 0]
    valid = df.dropna(subset=['Q_g1', 'Q_g2'])
    if len(valid) > 0:
        ax.scatter(valid['Q_g1'], valid['Q_g2'], alpha=0.5, s=20)

        # Add 1:1 line
        lims = [min(valid['Q_g1'].min(), valid['Q_g2'].min()),
                max(valid['Q_g1'].max(), valid['Q_g2'].max())]
        ax.plot(lims, lims, 'k--', lw=1, label='1:1')

        ax.set_xlabel('Q (nL/s) - Graph 1')
        ax.set_ylabel('Q (nL/s) - Graph 2')
        ax.set_title(f'Flow Comparison\nr={repro.correlation:.3f}')
        ax.legend()

    # Plot 2: ΔQ histogram
    ax = axes[0, 1]
    if len(repro.delta_Q) > 0:
        ax.hist(repro.delta_Q, bins=30, density=True, alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', lw=1)
        ax.axvline(repro.delta_Q.mean(), color='r', linestyle='-', lw=2,
                   label=f'mean={repro.delta_Q.mean():.2f}')
        ax.set_xlabel('ΔQ = Q_g2 - Q_g1 (nL/s)')
        ax.set_ylabel('Density')
        ax.set_title(f'Flow Differences\nσ_repro={repro.sigma_reproducibility:.2f} nL/s')
        ax.legend()

    # Plot 3: Z-score histogram
    ax = axes[0, 2]
    if len(repro.z_scores) > 5:
        ax.hist(repro.z_scores, bins=20, density=True, alpha=0.7, label='z-scores')

        # Add N(0,1) reference
        from scipy import stats
        x = np.linspace(-4, 4, 100)
        ax.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1)')

        ax.set_xlabel('z-score')
        ax.set_ylabel('Density')
        ax.set_title(f'Z-score Distribution\nmean={repro.z_score_mean:.2f}, std={repro.z_score_std:.2f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)

    # Plot 4: Radius comparison
    ax = axes[1, 0]
    valid_r = df.dropna(subset=['radius_g1', 'radius_g2'])
    if len(valid_r) > 0:
        ax.scatter(valid_r['radius_g1'], valid_r['radius_g2'], alpha=0.5, s=20)
        lims = [min(valid_r['radius_g1'].min(), valid_r['radius_g2'].min()),
                max(valid_r['radius_g1'].max(), valid_r['radius_g2'].max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel('Radius (μm) - Graph 1')
        ax.set_ylabel('Radius (μm) - Graph 2')
        r_corr = repro.radius_correlation or 0
        ax.set_title(f'Radius Comparison\nr={r_corr:.3f}')

    # Plot 5: Length comparison
    ax = axes[1, 1]
    valid_L = df.dropna(subset=['length_g1', 'length_g2'])
    if len(valid_L) > 0:
        ax.scatter(valid_L['length_g1'], valid_L['length_g2'], alpha=0.5, s=20)
        lims = [min(valid_L['length_g1'].min(), valid_L['length_g2'].min()),
                max(valid_L['length_g1'].max(), valid_L['length_g2'].max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel('Length (μm) - Graph 1')
        ax.set_ylabel('Length (μm) - Graph 2')
        L_corr = repro.length_correlation or 0
        ax.set_title(f'Length Comparison\nr={L_corr:.3f}')

    # Plot 6: |ΔQ| vs |Q_mean|
    ax = axes[1, 2]
    if len(repro.Q_g1) > 0:
        Q_mean = (np.abs(repro.Q_g1) + np.abs(repro.Q_g2)) / 2
        abs_delta = np.abs(repro.delta_Q)
        ax.scatter(Q_mean, abs_delta, alpha=0.5, s=20)
        ax.set_xlabel('|Q_mean| (nL/s)')
        ax.set_ylabel('|ΔQ| (nL/s)')
        ax.set_title(f'Error vs Flow Magnitude\nMedian rel error: {100*repro.median_relative_error:.1f}%')

        # Add trend line
        if len(Q_mean) > 10:
            z = np.polyfit(Q_mean, abs_delta, 1)
            x_fit = np.linspace(Q_mean.min(), Q_mean.max(), 100)
            ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=2,
                    label=f'slope={z[0]:.3f}')
            ax.legend()

    plt.suptitle(f'Graph Matching: {result.n_nodes_g1} ↔ {result.n_nodes_g2} nodes, '
                 f'{len(result.edge_matches)} matched edges',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    return fig


def save_match_result(result: MatchResult, output_dir: str, prefix: str = "match"):
    """
    Save match results to files.

    Saves:
        - {prefix}_comparison.csv: Edge comparison table
        - {prefix}_node_matches.json: Node mapping
        - {prefix}_summary.json: Summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison DataFrame
    csv_path = output_dir / f"{prefix}_comparison.csv"
    result.comparison_df.to_csv(csv_path, index=False)

    # Save node matches
    node_path = output_dir / f"{prefix}_node_matches.json"
    with open(node_path, 'w') as f:
        json.dump({str(k): v for k, v in result.node_matches.items()}, f, indent=2)

    # Save summary
    summary = {
        'offset': result.offset.tolist(),
        'offset_confidence': result.offset_confidence,
        'n_node_matches': len(result.node_matches),
        'n_edge_matches': len(result.edge_matches),
        'n_nodes_g1': result.n_nodes_g1,
        'n_nodes_g2': result.n_nodes_g2,
        'n_edges_g1': result.n_edges_g1,
        'n_edges_g2': result.n_edges_g2,
        'config': result.config,
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {output_dir}/")
    print(f"  - {prefix}_comparison.csv")
    print(f"  - {prefix}_node_matches.json")
    print(f"  - {prefix}_summary.json")


# CLI entry point
if __name__ == "__main__":
    import sys
    import pickle

    if len(sys.argv) < 3:
        print("Usage: python graph_matching.py graph1.gpickle graph2.gpickle [config.json]")
        print("")
        print("Matches nodes and edges between two graphs and analyzes reproducibility.")
        print("")
        print("To generate default config:")
        print("  python graph_matching.py --generate-config output_config.json")
        sys.exit(1)

    if sys.argv[1] == "--generate-config":
        config_path = sys.argv[2]
        save_config(DEFAULT_CONFIG, config_path)
        print(f"Saved default config to: {config_path}")
        sys.exit(0)

    # Load graphs
    graph1_path = sys.argv[1]
    graph2_path = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Loading graph 1: {graph1_path}")
    with open(graph1_path, 'rb') as f:
        G1 = pickle.load(f)

    print(f"Loading graph 2: {graph2_path}")
    with open(graph2_path, 'rb') as f:
        G2 = pickle.load(f)

    # Match graphs
    result = match_graphs(G1, G2, config_path=config_path)

    # Analyze reproducibility
    repro = analyze_reproducibility(result.comparison_df)

    # Plot results
    output_path = graph1_path.replace('.gpickle', '_vs_') + Path(graph2_path).stem + '_match.png'
    plot_reproducibility(result, repro, output_path=output_path, show=True)

    # Save results
    output_dir = Path(graph1_path).parent / "matching"
    save_match_result(result, output_dir)
