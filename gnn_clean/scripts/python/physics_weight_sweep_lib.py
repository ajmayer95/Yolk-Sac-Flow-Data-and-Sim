"""Shared helpers for the Step 2 physics-weight sweep."""

from __future__ import annotations

import math
from pathlib import Path


LAMBDA_VALUES = (0.1, 1.0, 10.0, 100.0)
LAMBDA_B_FIXED = 100.0
WEIGHTING_REGIME_ORDER = (
    "flow_prioritized",
    "balanced",
    "conservation_prioritized",
    "correction_regularized",
)


def format_lambda_token(value: float) -> str:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Lambda value must be positive and finite, got {value!r}")
    text = f"{value:g}"
    return text.replace(".", "p")


def build_gnn_run_name(lambda_q: float, lambda_k: float, lambda_delta: float) -> str:
    return (
        f"q_{format_lambda_token(lambda_q)}"
        f"__k_{format_lambda_token(lambda_k)}"
        f"__delta_{format_lambda_token(lambda_delta)}"
    )


def build_poiseuille_run_name(lambda_q: float, lambda_k: float) -> str:
    return f"poiseuille__q_{format_lambda_token(lambda_q)}__k_{format_lambda_token(lambda_k)}"


def generate_gnn_run_configs() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for lambda_q in LAMBDA_VALUES:
        for lambda_k in LAMBDA_VALUES:
            for lambda_delta in LAMBDA_VALUES:
                rows.append(
                    {
                        "run_name": build_gnn_run_name(lambda_q, lambda_k, lambda_delta),
                        "model_family": "gnn",
                        "lambda_q": float(lambda_q),
                        "lambda_k": float(lambda_k),
                        "lambda_b": float(LAMBDA_B_FIXED),
                        "lambda_delta": float(lambda_delta),
                        "message_passing_layers": 2,
                    }
                )
    return rows


def generate_poiseuille_run_configs() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for lambda_q in LAMBDA_VALUES:
        for lambda_k in LAMBDA_VALUES:
            rows.append(
                {
                    "run_name": build_poiseuille_run_name(lambda_q, lambda_k),
                    "model_family": "poiseuille_baseline",
                    "lambda_q": float(lambda_q),
                    "lambda_k": float(lambda_k),
                    "lambda_b": float(LAMBDA_B_FIXED),
                    "lambda_delta": float("nan"),
                    "message_passing_layers": 0,
                }
            )
    return rows


def classify_weighting_regime(lambda_q: float, lambda_k: float, lambda_delta: float) -> str:
    if lambda_q >= 10.0 * lambda_k and lambda_q >= 10.0 * lambda_delta:
        return "flow_prioritized"
    if lambda_k >= 10.0 * lambda_q and lambda_k >= 10.0 * lambda_delta:
        return "conservation_prioritized"
    if lambda_delta >= 10.0 * lambda_q and lambda_delta >= 10.0 * lambda_k:
        return "correction_regularized"
    return "balanced"


def percentile_scale(value: float, p05: float, p95: float) -> float:
    if not (math.isfinite(value) and math.isfinite(p05) and math.isfinite(p95)):
        return float("nan")
    denom = p95 - p05
    if abs(denom) <= 1.0e-30:
        return 0.0
    scaled = (value - p05) / denom
    return min(max(scaled, 0.0), 1.0)


def dominates(a: dict[str, float], b: dict[str, float], keys: tuple[str, str]) -> bool:
    a_values = [float(a[key]) for key in keys]
    b_values = [float(b[key]) for key in keys]
    if any(not math.isfinite(value) for value in a_values + b_values):
        return False
    return all(x <= y for x, y in zip(a_values, b_values)) and any(
        x < y for x, y in zip(a_values, b_values)
    )


def nondominated_sort(rows: list[dict[str, object]], keys: tuple[str, str]) -> list[int | None]:
    pending = set(range(len(rows)))
    ranks: list[int | None] = [None] * len(rows)
    rank = 1
    while pending:
        current_front: list[int] = []
        for idx in sorted(pending):
            row = rows[idx]
            values = [float(row[key]) for key in keys]
            if any(not math.isfinite(value) for value in values):
                continue
            dominated_flag = False
            for other_idx in pending:
                if other_idx == idx:
                    continue
                if dominates(rows[other_idx], row, keys):
                    dominated_flag = True
                    break
            if not dominated_flag:
                current_front.append(idx)
        if not current_front:
            for idx in sorted(pending):
                ranks[idx] = None
            break
        for idx in current_front:
            ranks[idx] = rank
            pending.remove(idx)
        rank += 1
    return ranks


def category_scores(flow_scaled: float, kirchhoff_scaled: float) -> dict[str, float]:
    if not (math.isfinite(flow_scaled) and math.isfinite(kirchhoff_scaled)):
        return {
            "score_flow_prioritized": float("nan"),
            "score_conservation_prioritized": float("nan"),
            "score_balanced": float("nan"),
            "score_correction_regularized": float("nan"),
        }
    return {
        "score_flow_prioritized": 0.75 * flow_scaled + 0.25 * kirchhoff_scaled,
        "score_conservation_prioritized": 0.25 * flow_scaled + 0.75 * kirchhoff_scaled,
        "score_balanced": 0.5 * flow_scaled + 0.5 * kirchhoff_scaled,
        "score_correction_regularized": 0.5 * flow_scaled + 0.5 * kirchhoff_scaled,
    }


def score_for_regime(regime: str, scores: dict[str, float]) -> float:
    mapping = {
        "flow_prioritized": "score_flow_prioritized",
        "conservation_prioritized": "score_conservation_prioritized",
        "balanced": "score_balanced",
        "correction_regularized": "score_correction_regularized",
    }
    key = mapping.get(regime)
    if key is None:
        return float("nan")
    return float(scores.get(key, float("nan")))


def representative_sort_key(row: dict[str, object]) -> tuple[float, float, float, float, str]:
    def number(key: str) -> float:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("inf")
        return value if math.isfinite(value) else float("inf")

    return (
        number("pareto_rank"),
        number("selection_score"),
        number("delta_rms"),
        number("delta_saturation_fraction"),
        str(row.get("run_name", "")),
    )


def parse_scalar(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and lowered not in {"nan", "inf", "-inf"}:
        return int(number)
    return number


def ensure_unique_run_names(rows: list[dict[str, object]]) -> None:
    names = [str(row["run_name"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate run names detected.")


def expected_run_files(model_family: str) -> tuple[str, ...]:
    if model_family == "gnn":
        return ("summary.csv", "summary.yaml", "edge_predictions.csv", "node_predictions.csv")
    if model_family == "poiseuille_baseline":
        return ("summary.csv", "summary.yaml", "edge_predictions.csv", "node_predictions.csv")
    raise ValueError(f"Unsupported model_family: {model_family}")


def launcher_metadata_path(run_dir: Path) -> Path:
    return run_dir / "launcher_run_config.yaml"
