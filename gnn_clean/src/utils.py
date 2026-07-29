"""Shared utilities for the real-data GNN pressure-field pipeline."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def install_numpy_pickle_compat() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    for new_name, old_name in aliases.items():
        if new_name not in sys.modules:
            try:
                sys.modules[new_name] = importlib.import_module(old_name)
            except Exception:
                pass


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(int(seed))
        except Exception:
            pass


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def safe_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _parse_scalar(text: str):
    value = text.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result = {}
        for part in inner.split(","):
            key, item = part.split(":", 1)
            result[key.strip()] = _parse_scalar(item)
        return result
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_yaml(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lines = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        content = raw.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        lines.append((line_number, indent, content.strip()))

    def parse_block(index: int, indent: int):
        if index >= len(lines):
            return {}, index
        _, current_indent, stripped = lines[index]
        if stripped.startswith("- "):
            result = []
            while index < len(lines):
                line_number, item_indent, item_text = lines[index]
                if item_indent < indent:
                    break
                if item_indent != indent or not item_text.startswith("- "):
                    raise ValueError(f"Unsupported YAML at {path}:{line_number}")
                payload = item_text[2:].strip()
                if not payload:
                    value, index = parse_block(index + 1, indent + 2)
                    result.append(value)
                    continue
                if payload.startswith("{") and payload.endswith("}"):
                    result.append(_parse_scalar(payload))
                    index += 1
                    continue
                if ":" in payload:
                    key, raw_value = payload.split(":", 1)
                    key = key.strip()
                    raw_value = raw_value.strip()
                    item = {}
                    if raw_value:
                        item[key] = _parse_scalar(raw_value)
                        index += 1
                    else:
                        nested, index = parse_block(index + 1, indent + 2)
                        item[key] = nested
                    while index < len(lines):
                        next_line_number, next_indent, next_text = lines[index]
                        if next_indent <= indent:
                            break
                        if next_text.startswith("- "):
                            break
                        if ":" not in next_text:
                            raise ValueError(
                                f"Unsupported YAML at {path}:{next_line_number}"
                            )
                        sub_key, sub_value = next_text.split(":", 1)
                        sub_key = sub_key.strip()
                        sub_value = sub_value.strip()
                        if sub_value:
                            item[sub_key] = _parse_scalar(sub_value)
                            index += 1
                        else:
                            nested, index = parse_block(index + 1, next_indent + 2)
                            item[sub_key] = nested
                    result.append(item)
                    continue
                result.append(_parse_scalar(payload))
                index += 1
            return result, index

        result = {}
        while index < len(lines):
            line_number, current_indent, stripped = lines[index]
            if current_indent < indent:
                break
            if current_indent != indent or stripped.startswith("- "):
                raise ValueError(f"Unsupported YAML at {path}:{line_number}")
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML at {path}:{line_number}")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if raw_value:
                result[key] = _parse_scalar(raw_value)
                index += 1
            else:
                value, index = parse_block(index + 1, indent + 2)
                result[key] = value
        return result, index

    parsed, _ = parse_block(0, lines[0][1] if lines else 0)
    return parsed


def dump_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_for_yaml(payload), indent=2, sort_keys=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clean_for_yaml(value):
    if isinstance(value, dict):
        return {str(key): clean_for_yaml(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_yaml(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_for_yaml(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.complexfloating, complex)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_yaml(path: Path, payload: object) -> None:
    dump_yaml(path, clean_for_yaml(payload))


def nested_value(payload: dict, dotted: str) -> float:
    value = payload
    for part in dotted.split("."):
        value = value[part]
    return float(value)
