"""GNN experiment configuration, model construction, and artifact helpers."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from distensibility.io import write_json
from models.baselines import EdgeLocalMLP, VanillaGCN
from models.gnn import PhysicsInformedGNN


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def run_name(config: dict) -> str:
    name = (
        f"{config['model_name']}__K{int(config['K'])}"
        f"__{config['harmonic_mode']}__seed{int(config['training']['seed'])}"
    )
    if (
        config["model_name"] == "vanilla_gcn"
        and config.get("model", {}).get("vanilla_gcn_pressure_decoder", False)
    ):
        name += "__pressure_decoder"
    return name


def create_run_directory(
    project_root: Path, dataset_path: Path, config: dict
) -> Path:
    return (
        project_root
        / config["output"]["root"]
        / dataset_path.stem
        / run_name(config)
    )


def build_model(data, config):
    common = {
        "node_dim": int(data.node_features.shape[1]),
        "edge_dim": int(data.edge_features.shape[1]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "activation_name": config["model"]["activation"],
        "dropout": float(config["model"]["dropout"]),
        "correction_bound": float(config["model"]["correction_bound"]),
        "harmonic_correction_bound": float(
            config["model"]["harmonic_correction_bound"]
        ),
    }
    name = config["model_name"]
    if name == "physics_informed_gnn":
        return PhysicsInformedGNN(
            **common,
            K=int(config["K"]),
            n_harmonics=int(data.n_harmonics),
        )
    if name == "vanilla_gcn":
        return VanillaGCN(
            **common,
            K=int(config["K"]),
            n_channels=int(data.n_channels),
        )
    if name == "edge_local_mlp":
        return EdgeLocalMLP(
            **common,
            n_channels=int(data.n_channels),
        )
    raise ValueError(f"Unknown model: {name}")


def expand_experiment_grid(config: dict) -> list[dict]:
    """Expand valid model/K/harmonic combinations."""
    resolved = []
    for seed in config["grid"]["seeds"]:
        for requested_name in config["grid"]["model_names"]:
            for K in config["grid"]["K_values"]:
                if int(K) == 0:
                    if requested_name != "edge_local_mlp":
                        continue
                    model_name = "edge_local_mlp"
                else:
                    if requested_name == "edge_local_mlp":
                        continue
                    model_name = requested_name
                for harmonic_mode in config["grid"]["harmonic_modes"]:
                    item = {
                        "model_name": model_name,
                        "K": int(K),
                        "harmonic_mode": harmonic_mode,
                        "model": dict(config["model"]),
                        "training": dict(config["training"]),
                        "loss": dict(config["loss"]),
                        "data": dict(config["data"]),
                        "output": dict(config["output"]),
                        "loss_toggles": dict(
                            config["model_defaults"][model_name]
                        ),
                    }
                    item["training"]["seed"] = int(seed)
                    if harmonic_mode == "dc_only":
                        item["loss_toggles"]["use_harmonic_loss"] = False
                    resolved.append(item)
    return resolved


def load_manifest_datasets(project_root: Path, manifest_path: str):
    manifest = project_root / manifest_path
    with manifest.open(newline="") as handle:
        return [
            project_root / "data" / "synthetic" / row["file"]
            for row in csv.DictReader(handle)
        ]


def save_resolved_config(path: Path, config: dict) -> None:
    write_json(path, config)


def save_outputs(run_dir: Path, data, outputs, metrics, history, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    velocity = outputs["velocity_m_s"].detach().cpu().numpy()
    velocity_complex = velocity[..., 0] + 1j * velocity[..., 1]
    pressure = outputs.get("pressure_pa")
    if pressure is None:
        pressure_np = np.full(len(data.node_id), np.nan, dtype=np.float32)
    else:
        pressure_np = pressure.detach().cpu().numpy()
    delta = outputs["delta_dc"].detach().cpu().numpy()
    harmonic = outputs["harmonic_corrections"].detach().cpu().numpy()
    np.savez_compressed(
        run_dir / "predicted_velocities.npz",
        predicted_velocity_m_s=velocity_complex,
        harmonic_index=np.arange(data.n_channels, dtype=np.int8),
        edge_source_index=data.edge_index[0].cpu().numpy(),
        edge_target_index=data.edge_index[1].cpu().numpy(),
    )
    # Keep predicted_pressure_pa consistent with classical predictions.npz.
    np.savez_compressed(
        run_dir / "pressure_field.npz",
        predicted_pressure_pa=pressure_np,
        pressure_field_pa=pressure_np,
        node_id=data.node_id,
        harmonic_index=np.asarray([0], dtype=np.int8),
    )
    np.savez_compressed(
        run_dir / "corrections.npz",
        delta_dc=delta,
        conductance_multiplier=np.exp(delta),
        harmonic_corrections_normalized=harmonic,
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "training_history.json", {"history": history})
    save_resolved_config(run_dir / "config.yaml", config)


def read_resolved_config(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from distensibility.simulation import load_yaml

        return load_yaml(path)
