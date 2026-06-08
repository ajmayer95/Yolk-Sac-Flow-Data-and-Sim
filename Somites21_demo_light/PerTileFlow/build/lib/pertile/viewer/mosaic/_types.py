"""Data types and tile-position parsing for the Mosaic Viewer."""

import re
import json
import pickle
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field


# Result fields to store on mosaic graph edges from analyze_vessel()
_RESULT_FIELDS = [
    'mean_Q', 'amp_Q', 'PI', 'PI_filt', 'PI_f0', 'PI_smooth', 'phase', 'phase_global', 'f0_hz', 'snr_db',
    'sigma_Q', 'sigma_mean_Q', 'sigma_Q_mean', 'sigma_Q_amp',
    'rel_uncertainty', 'chi2_reduced',
    'v_max', 'r_offset', 'radius_px', 'original_radius_px',
    'R_fit_px', 'path_length_px', 'fit_success',
    'mean_Q_nL_s', 'sigma_Q_nL_s',
    'quality_tier', 'snr_pulse', 'mean_coherence_vessel', 'psd_snr_f0',
]


@dataclass
class TileInfo:
    """Information about a single tile in the mosaic."""
    vid: int
    tile_width: float
    tile_height: float
    # Corner coordinates in mosaic space (x1,y1 is top-right, x2,y2 is top-left, etc.)
    # These can be None if using JSON format with translate_x/translate_y
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    x3: Optional[float] = None
    y3: Optional[float] = None
    x4: Optional[float] = None
    y4: Optional[float] = None
    image_path: str = ""
    # Optional: override graph dimensions (detected from graph if not set)
    _graph_width: Optional[float] = field(default=None, repr=False)
    _graph_height: Optional[float] = field(default=None, repr=False)
    # JSON format: direct translate values (alternative to corner coords)
    _translate_x: Optional[float] = field(default=None, repr=False)
    _translate_y: Optional[float] = field(default=None, repr=False)

    @property
    def top_left_x(self) -> float:
        """X coordinate of top-left corner (minimum x)."""
        # JSON format: use translate_x directly
        if self._translate_x is not None:
            return self._translate_x
        # Legacy format: compute from corner coords
        return min(self.x1, self.x2, self.x3, self.x4)

    @property
    def top_left_y(self) -> float:
        """Y coordinate of top-left corner (minimum y)."""
        # JSON format: use translate_y directly
        if self._translate_y is not None:
            return self._translate_y
        # Legacy format: compute from corner coords
        return min(self.y1, self.y2, self.y3, self.y4)

    @property
    def graph_width(self) -> float:
        """Original graph coordinate width."""
        if self._graph_width is not None:
            return self._graph_width
        # Default: assume graph coords match tile coords (scale=1)
        return self.tile_width

    @property
    def graph_height(self) -> float:
        """Original graph coordinate height."""
        if self._graph_height is not None:
            return self._graph_height
        # Default: assume graph coords match tile coords (scale=1)
        return self.tile_height

    def set_graph_dimensions(self, width: float, height: float):
        """Set the graph coordinate dimensions for proper scaling."""
        self._graph_width = width
        self._graph_height = height

    @property
    def scale_x(self) -> float:
        """Scale factor for x coordinates (tile_width / graph_width)."""
        return self.tile_width / self.graph_width

    @property
    def scale_y(self) -> float:
        """Scale factor for y coordinates (tile_height / graph_height)."""
        return self.tile_height / self.graph_height


@dataclass
class TileStats:
    """Aggregated statistics for a tile."""
    vid: int
    median_Q: Optional[float] = None
    max_Q: Optional[float] = None
    median_PI: Optional[float] = None
    f0: Optional[float] = None
    n_vessels: int = 0


def parse_tile_positions(tile_file: str, tile_width: float = 640.0, tile_height: float = 704.0) -> Dict[int, TileInfo]:
    """
    Parse tile position file (JSON or legacy tab-separated format).

    Supports two formats:
    1. JSON format (tile_positions_manual.json):
       {"tiles": {"1": {"translate_x": ..., "translate_y": ..., ...}, ...}}

    2. Legacy tab-separated format:
       tile_width, 0, 0, 0, 0, tile_height, x1, y1, x2, y2, x3, y3, x4, y4, -1, image_path

    Args:
        tile_file: Path to tile positions file
        tile_width: Default tile width for JSON format (default: 640)
        tile_height: Default tile height for JSON format (default: 704)

    Returns:
        Dict mapping vid -> TileInfo
    """
    tile_path = Path(tile_file)

    # Detect JSON format by extension or content
    if tile_path.suffix.lower() == '.json':
        return _parse_tile_positions_json(tile_file, tile_width, tile_height)

    # Try to detect JSON by content
    with open(tile_file, 'r') as f:
        first_char = f.read(1).strip()
        if first_char == '{':
            return _parse_tile_positions_json(tile_file, tile_width, tile_height)

    # Legacy tab-separated format
    return _parse_tile_positions_legacy(tile_file)


def _parse_tile_positions_json(tile_file: str, tile_width: float, tile_height: float) -> Dict[int, TileInfo]:
    """Parse JSON format tile positions (tile_positions_manual.json)."""
    with open(tile_file, 'r') as f:
        data = json.load(f)

    tiles = {}
    tiles_data = data.get('tiles', data)  # Support both {"tiles": {...}} and flat format

    for vid_str, tile_data in tiles_data.items():
        try:
            vid = int(vid_str)
        except ValueError:
            continue  # Skip non-numeric keys

        # Get translate values (position in mosaic space)
        translate_x = tile_data.get('translate_x', 0.0)
        translate_y = tile_data.get('translate_y', 0.0)

        # Get scale factors (default to 1.0)
        scale_x = tile_data.get('scale_x', 1.0)
        scale_y = tile_data.get('scale_y', 1.0)

        # Create TileInfo with translate values
        tiles[vid] = TileInfo(
            vid=vid,
            tile_width=tile_width * scale_x,
            tile_height=tile_height * scale_y,
            _translate_x=translate_x,
            _translate_y=translate_y,
        )

    return tiles


def _parse_tile_positions_legacy(tile_file: str) -> Dict[int, TileInfo]:
    """Parse legacy tab-separated tile positions format."""
    tiles = {}

    with open(tile_file, 'r') as f:
        lines = f.readlines()

    # Auto-detect header: check if first field of first line is numeric
    start_idx = 0
    if lines:
        first_line = lines[0].strip()
        if first_line:
            first_field = first_line.split('\t')[0]
            try:
                float(first_field)
                # First field is numeric - no header
                start_idx = 0
            except ValueError:
                # First field is not numeric - skip header
                start_idx = 1

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 16:
            continue

        tile_width = float(parts[0])
        tile_height = float(parts[5])
        x1, y1 = float(parts[6]), float(parts[7])
        x2, y2 = float(parts[8]), float(parts[9])
        x3, y3 = float(parts[10]), float(parts[11])
        x4, y4 = float(parts[12]), float(parts[13])
        image_path = parts[15]

        # Extract vid from image path (processed{N}.png)
        match = re.search(r'processed(\d+)\.png', image_path)
        if match:
            vid = int(match.group(1))
            tiles[vid] = TileInfo(
                vid=vid,
                tile_width=tile_width,
                tile_height=tile_height,
                x1=x1, y1=y1,
                x2=x2, y2=y2,
                x3=x3, y3=y3,
                x4=x4, y4=y4,
                image_path=image_path,
            )

    return tiles


def load_graph(graph_path: str):
    """Load a graph from pickle file."""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def transform_graph_to_mosaic(G, tile_info: TileInfo):
    """
    Transform graph node coordinates from tile space to mosaic space.

    Args:
        G: NetworkX graph with node attributes 'x' and 'y'
        tile_info: TileInfo with position and scaling info

    Returns:
        New graph with transformed coordinates (in image space, y=0 at top)
    """
    import networkx as nx

    # Create a copy of the graph
    G_mosaic = G.copy()

    # Transform each node's coordinates
    for node in G_mosaic.nodes():
        # Original coordinates in tile/graph space
        # Graph uses cartesian coords: y=0 at bottom
        x_graph = G_mosaic.nodes[node]['x']
        y_graph = G_mosaic.nodes[node]['y']

        # Flip y within the tile (graph coords have y=0 at bottom, image has y=0 at top)
        y_graph_flipped = tile_info.graph_height - y_graph

        # Scale to tile image space
        x_tile = x_graph * tile_info.scale_x
        y_tile = y_graph_flipped * tile_info.scale_y

        # Translate to mosaic space (tile positions are already in image coords)
        x_mosaic = tile_info.top_left_x + x_tile
        y_mosaic = tile_info.top_left_y + y_tile

        # Store transformed coordinates (now in image space, ready for napari)
        G_mosaic.nodes[node]['x'] = x_mosaic
        G_mosaic.nodes[node]['y'] = y_mosaic
        G_mosaic.nodes[node]['x_original'] = x_graph
        G_mosaic.nodes[node]['y_original'] = y_graph
        G_mosaic.nodes[node]['vid'] = tile_info.vid

    # Also tag edges with vid
    for u, v in G_mosaic.edges():
        G_mosaic[u][v]['vid'] = tile_info.vid

    return G_mosaic
