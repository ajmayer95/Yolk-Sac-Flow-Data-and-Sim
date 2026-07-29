"""Configuration, model construction, and artifact helpers for real-data GNNs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.baselines import DirectEdgeDelta, EdgeLocalMLP, VanillaGCN
from models.gnn import PhysicsInformedGNN
from utils import load_yaml, write_yaml


MODEL_LABELS = {
    "physics_informed_gnn": "Physics-informed conductance GNN",
    "vanilla_gcn": "Node-message conductance GNN",
    "edge_local_k0": "K=0 edge-local conductance MLP",
    "direct_delta": "Direct per-edge delta optimization",
}


def correction_limits(config: dict) -> tuple[float, float]:
    model_cfg = config["model"]
    bound = float(model_cfg.get("correction_bound", 0.25))
    correction_min = float(model_cfg.get("correction_min", model_cfg.get("delta_min", -bound)))
    correction_max = float(model_cfg.get("correction_max", model_cfg.get("delta_max", bound)))
    return correction_min, correction_max


def run_name(config: dict) -> str:
    hidden = int(config["model"]["hidden_dim"])
    lr = f"{float(config['training']['learning_rate']):.0e}".replace("+", "")
    wd = f"{float(config['training']['weight_decay']):.0e}".replace("+", "")
    lambda_delta = f"{float(config['loss']['lambda_delta']):.0e}".replace("+", "")
    lambda_alpha = f"{float(config['loss'].get('lambda_alpha', 0.0)):.0e}".replace("+", "")
    correction_min, correction_max = correction_limits(config)
    bound = max(abs(float(correction_min)), abs(float(correction_max)))
    bound_label = f"{bound:.0e}" if bound < 1.0 else str(int(round(bound)))
    name = (
        f"{config['model_name']}__K{int(config['K'])}"
        f"__h{hidden}"
        f"__lr{lr}"
        f"__wd{wd}"
        f"__ld{lambda_delta}"
        f"__la{lambda_alpha}"
        f"__cb{bound_label}"
        f"__dc_only__seed{int(config['training']['seed'])}"
    )
    if config.get("data", {}).get("use_tilewise_flow_normalization", False):
        name += "__tile_flow_norm"
    return name


def create_run_directory(project_root: Path, graph_path: Path, config: dict) -> Path:
    return project_root / config["output"]["root"] / graph_path.stem / run_name(config)


def build_model(data, config):
    correction_min, correction_max = correction_limits(config)
    common = {
        "node_dim": int(data.node_features.shape[1]),
        "edge_dim": int(data.edge_features.shape[1]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "activation_name": config["model"]["activation"],
        "dropout": float(config["model"]["dropout"]),
        "correction_bound": float(config["model"].get("correction_bound", max(abs(correction_min), abs(correction_max)))),
        "correction_min": correction_min,
        "correction_max": correction_max,
        "correction_parameterization": str(
            config["model"].get("correction_parameterization", "tanh")
        ),
        "predict_gamma": bool(config["model"].get("predict_gamma", False)),
        "gamma_min": float(config["model"].get("gamma_min", -0.5)),
        "gamma_max": float(config["model"].get("gamma_max", 0.5)),
        "gamma_parameterization": str(config["model"].get("gamma_parameterization", "tanh")),
    }
    name = config["model_name"]
    if name == "physics_informed_gnn":
        return PhysicsInformedGNN(**common, K=int(config["K"]))
    if name == "vanilla_gcn":
        return VanillaGCN(**common, K=int(config["K"]))
    if name == "edge_local_k0":
        return EdgeLocalMLP(**common)
    if name == "direct_delta":
        return DirectEdgeDelta(n_edges=int(data.n_edges), **common)
    raise ValueError(f"Unknown model: {name}")


def save_resolved_config(path: Path, config: dict) -> None:
    write_yaml(path, config)


def read_resolved_config(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return load_yaml(path)


def save_outputs(run_dir: Path, data, outputs, metrics, diagnostics, history, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "metrics.yaml", metrics)
    write_yaml(run_dir / "diagnostics.yaml", diagnostics)
    save_resolved_config(run_dir / "config.yaml", config)
    if config["output"].get("save_training_history", False):
        write_yaml(run_dir / "training_history.yaml", {"history": history})
    if config["output"].get("save_predicted_velocities", False):
        velocity = outputs["velocity_m_s"].detach().cpu().numpy()
        velocity_complex = velocity[..., 0] + 1j * velocity[..., 1]
        np.savez_compressed(
            run_dir / "predicted_velocities.npz",
            predicted_velocity_m_s=velocity_complex,
            harmonic_index=np.arange(data.n_channels, dtype=np.int8),
            edge_source_index=data.edge_index[0].cpu().numpy(),
            edge_target_index=data.edge_index[1].cpu().numpy(),
            edge_ids=np.asarray([(str(u), str(v)) for u, v in data.edge_ids], dtype=object),
        )
    if "flow_m3_s" in outputs:
        flow = outputs["flow_m3_s"].detach().cpu().numpy()
        observed_flow = (data.velocity_observed_m_s[:, 0, 0] * data.area_m2).detach().cpu().numpy()
        np.savez_compressed(
            run_dir / "predicted_flows.npz",
            predicted_flow_m3_s=flow,
            observed_flow_m3_s=observed_flow,
            edge_source_index=data.edge_index[0].cpu().numpy(),
            edge_target_index=data.edge_index[1].cpu().numpy(),
            edge_ids=np.asarray([(str(u), str(v)) for u, v in data.edge_ids], dtype=object),
        )
    if config["output"].get("save_pressure_artifact", False):
        pressure = outputs.get("pressure_pa")
        if pressure is None:
            pressure_np = np.full(len(data.node_id), np.nan, dtype=np.float32)
        else:
            pressure_np = pressure.detach().cpu().numpy()
        np.savez_compressed(
            run_dir / "pressure_field.npz",
            predicted_pressure_pa=pressure_np,
            pressure_field_pa=pressure_np,
            node_id=np.asarray([str(node) for node in data.node_id], dtype=object),
            harmonic_index=np.asarray([0], dtype=np.int8),
        )
    if config["output"].get("save_corrections", False):
        delta = outputs["delta_e"].detach().cpu().numpy()
        np.savez_compressed(
            run_dir / "corrections.npz",
            delta_e=delta,
            delta_dc=delta,
            conductance_ratio=outputs["conductance_ratio"].detach().cpu().numpy(),
            conductance_multiplier=outputs["conductance_ratio"].detach().cpu().numpy(),
        )
    if getattr(data, "reference_pressure_pa", None) is not None:
        reference_pressure = data.reference_pressure_pa.detach().cpu().numpy()
        reference_velocity = data.velocity_reference_m_s.detach().cpu().numpy()[:, 0, 0]
        reference_boundary_injection = (
            data.reference_boundary_injection_m3_s.detach().cpu().numpy()
        )
        delta_zero_pressure = data.delta_zero_reference_pressure_pa.detach().cpu().numpy()
        delta_zero_velocity = data.delta_zero_reference_velocity_m_s.detach().cpu().numpy()[
            :, 0, 0
        ]
        np.savez_compressed(
            run_dir / "poiseuille_reference.npz",
            reference_pressure_pa=reference_pressure,
            poiseuille_reference_pressure_pa=reference_pressure,
            reference_velocity_m_s=reference_velocity,
            poiseuille_reference_velocity_m_s=reference_velocity,
            reference_boundary_injection_m3_s=reference_boundary_injection,
            poiseuille_reference_boundary_injection_m3_s=reference_boundary_injection,
            delta_zero_reference_pressure_pa=delta_zero_pressure,
            delta_zero_reference_velocity_m_s=delta_zero_velocity,
            node_id=np.asarray([str(node) for node in data.node_id], dtype=object),
            edge_ids=np.asarray([(str(u), str(v)) for u, v in data.edge_ids], dtype=object),
        )
    if getattr(data, "flow_normalization", None):
        flow_norm = data.flow_normalization
        np.savez_compressed(
            run_dir / "tilewise_flow_normalization.npz",
            tile_ids=np.asarray(flow_norm["tile_ids"]),
            tile_flux_scale=np.asarray(flow_norm["tile_flux_scale"]),
            tile_flux_scale_raw=np.asarray(flow_norm["tile_flux_scale_raw"]),
            tile_valid_edge_count=np.asarray(flow_norm["tile_valid_edge_count"]),
            membership_edge_index=np.asarray(flow_norm["membership_edge_index"]),
            membership_tile_id=np.asarray(flow_norm["membership_tile_id"]),
            membership_weight=np.asarray(flow_norm["membership_weight"]),
            membership_observed_velocity_dc_m_s=np.asarray(
                flow_norm["membership_observed_velocity_dc_m_s"]
            ),
            membership_normalized_velocity_dc_m_s=np.asarray(
                flow_norm["membership_normalized_velocity_dc_m_s"]
            ),
            normalized_velocity_dc_m_s=np.asarray(flow_norm["normalized_velocity_dc_m_s"]),
            observed_velocity_dc_m_s=np.asarray(flow_norm["observed_velocity_dc_m_s"]),
            reference_velocity_dc_m_s=np.asarray(flow_norm["reference_velocity_dc_m_s"]),
        )
        write_yaml(
            run_dir / "tilewise_flow_normalization_diagnostics.yaml",
            flow_norm["diagnostics"],
        )
