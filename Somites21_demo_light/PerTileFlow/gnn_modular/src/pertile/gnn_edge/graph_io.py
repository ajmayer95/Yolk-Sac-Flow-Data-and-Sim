"""Graph loading helpers for the modular GNN edge-flow workflow."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

from .utils import install_numpy_pickle_compat


def _load_config(config_path: Optional[str]) -> Tuple[dict, Optional[Path]]:
    if not config_path:
        return {}, None
    cfg_path = Path(config_path).expanduser().resolve()
    with open(cfg_path) as f:
        return json.load(f), cfg_path.parent


def _resolve_path(
    cli_value: Optional[str],
    cfg: dict,
    cfg_dir: Optional[Path],
    key: str,
) -> Optional[Path]:
    if cli_value is not None:
        return Path(cli_value).expanduser().resolve()

    value = cfg.get(key)
    if value is None:
        return None

    path = Path(value).expanduser()
    if path.is_absolute() or cfg_dir is None:
        return path
    return (cfg_dir / path).resolve()


def load_graph_from_args(args) -> Tuple[object, Path]:
    """Load the mosaic graph from ``--graph`` or ``--config``."""
    cfg, cfg_dir = _load_config(getattr(args, "config", None))
    graph_path = _resolve_path(
        getattr(args, "graph", None),
        cfg,
        cfg_dir,
        "mosaic_graph",
    )
    if graph_path is None:
        raise SystemExit("Provide --graph or --config with mosaic_graph.")

    install_numpy_pickle_compat()
    with open(graph_path, "rb") as f:
        return pickle.load(f), graph_path
