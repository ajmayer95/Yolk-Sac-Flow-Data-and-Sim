"""
Configuration loading from JSON files.
"""

import json
from pathlib import Path
from typing import Optional, Union, Any


# Default config path (relative to project root)
DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.json"


class Config:
    """Configuration container with dot-access to nested values."""

    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from JSON file.

    Parameters
    ----------
    path : str or Path, optional
        Path to config JSON. If None, uses default.json

    Returns
    -------
    Config
        Configuration object with dot-access
    """
    if path is None:
        path = DEFAULT_CONFIG
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    return Config(data)


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge two config dicts (override takes precedence).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# Convenience: load default config for quick access
_default_cfg = None


def get_default_config() -> Config:
    """Get the default configuration (cached)."""
    global _default_cfg
    if _default_cfg is None:
        _default_cfg = load_config()
    return _default_cfg
