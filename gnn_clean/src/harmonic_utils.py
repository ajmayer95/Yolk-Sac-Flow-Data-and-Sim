"""Reusable harmonic utilities for fixed-baseline experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import torch

from pertile.analysis.config import FRAME_DT_S
from pertile.analysis.harmonic import fit_harmonics
from real_data import PX_SIZE_M
from utils import safe_float


NL_PER_M3 = 1.0e12
DEG_PER_RAD = 180.0 / math.pi


def wrap_phase_rad(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return math.atan2(math.sin(value), math.cos(value))


def float_rmse(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(values[finite] ** 2)))


def complex_rmse(values: np.ndarray) -> float:
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.abs(values[finite]) ** 2)))


def principal_phase_residual_deg(pred_phase: np.ndarray, obs_phase: np.ndarray) -> np.ndarray:
    residual = pred_phase - obs_phase
    return np.asarray([wrap_phase_rad(float(value)) * DEG_PER_RAD for value in residual], dtype=np.float64)


def phase_range_deg(phases_rad: np.ndarray) -> float:
    finite = phases_rad[np.isfinite(phases_rad)]
    if finite.size == 0:
        return float("nan")
    anchor = np.angle(np.mean(np.exp(1j * finite)))
    wrapped = np.asarray([wrap_phase_rad(float(value - anchor)) for value in finite], dtype=np.float64)
    return float((wrapped.max() - wrapped.min()) * DEG_PER_RAD)


def phase_target_residual_deg(
    pressures: np.ndarray,
    targets: np.ndarray,
    arterial_idx: np.ndarray,
) -> np.ndarray:
    if arterial_idx.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return principal_phase_residual_deg(np.angle(pressures[arterial_idx]), np.angle(targets))


def best_f0_hz(graph, data) -> float:
    tile_f0s = graph.graph.get("tile_f0s", {})
    if isinstance(tile_f0s, dict):
        values = [safe_float(value) for value in tile_f0s.values()]
        values = [value for value in values if math.isfinite(value) and value > 0.0]
        if values:
            return float(np.median(np.asarray(values, dtype=np.float64)))
    edge_f0s: list[float] = []
    for u, v in data.edge_ids:
        edge_data = graph.edges[u, v]
        for key in ("f0_hz_piv", "f0_hz"):
            value = safe_float(edge_data.get(key))
            if math.isfinite(value) and value > 0.0:
                edge_f0s.append(value)
                break
    if edge_f0s:
        return float(np.median(np.asarray(edge_f0s, dtype=np.float64)))
    raise ValueError("Could not infer a harmonic frequency from the graph.")


def edge_geometry_m(edge_data: dict) -> tuple[float, float]:
    radius_px = safe_float(edge_data.get("radius_px_true"))
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        radius_px = safe_float(edge_data.get("R_fit_px"))
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        radius_px = safe_float(edge_data.get("radius_px"))
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        radius_px = safe_float(edge_data.get("radius"))
    length_px = safe_float(edge_data.get("length_true"))
    if not math.isfinite(length_px) or length_px <= 0.0:
        length_px = safe_float(edge_data.get("length"))
    if not math.isfinite(length_px) or length_px <= 0.0:
        length_px = safe_float(edge_data.get("path_length_px"))
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        radius_px = 1.0e-6 / PX_SIZE_M
    if not math.isfinite(length_px) or length_px <= 0.0:
        length_px = 1.0e-6 / PX_SIZE_M
    radius_m = max(float(radius_px) * PX_SIZE_M, 1.0e-12)
    length_m = max(float(length_px) * PX_SIZE_M, 1.0e-12)
    return radius_m, length_m


def edge_distensibility_values(
    radii_m: np.ndarray,
    d0: float,
    alpha: float,
    reference_radius_m: float | None = None,
) -> tuple[np.ndarray, float]:
    finite = radii_m[np.isfinite(radii_m) & (radii_m > 0.0)]
    reference = float(reference_radius_m) if reference_radius_m is not None else (
        float(np.mean(finite)) if finite.size else 1.0
    )
    reference = max(reference, 1.0e-30)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        ratio = np.clip(radii_m / reference, 1.0e-30, None)
        d_edge = float(d0) * np.power(ratio, float(alpha))
    d_edge = np.asarray(d_edge, dtype=np.float64)
    d_edge[~np.isfinite(d_edge)] = float(d0)
    d_edge = np.clip(d_edge, 0.0, None)
    return d_edge, reference


def signed_measurement_phasor_nl_s(edge_data: dict, u, v, harmonic_number: int, global_f0_hz: float) -> tuple[complex, bool, float]:
    bc_harmonics = edge_data.get("bc_harmonics")
    if bc_harmonics is not None:
        try:
            bc_array = np.asarray(bc_harmonics, dtype=np.complex128).reshape(-1)
        except Exception:
            bc_array = np.asarray([], dtype=np.complex128)
        if harmonic_number >= 1 and bc_array.size > harmonic_number:
            phasor = complex(bc_array[harmonic_number])
            if np.isfinite(phasor.real) and np.isfinite(phasor.imag):
                flow_from = edge_data.get("flow_from")
                flow_to = edge_data.get("flow_to")
                sign = 1.0
                if flow_from == v and flow_to == u:
                    sign = -1.0
                elif flow_from == u and flow_to == v:
                    sign = 1.0
                return complex(sign * phasor), True, float(global_f0_hz)

    amp = safe_float(edge_data.get(f"amp_Q_h{harmonic_number}_piv"))
    if not math.isfinite(amp):
        amp = safe_float(edge_data.get("amp_Q_piv" if harmonic_number == 1 else f"amp_Q_h{harmonic_number}"))
    phase = safe_float(edge_data.get(f"phase_h{harmonic_number}_piv"))
    if not math.isfinite(phase):
        phase = safe_float(edge_data.get("phase_piv" if harmonic_number == 1 else f"phase_h{harmonic_number}"))
    if math.isfinite(amp) and math.isfinite(phase):
        phasor = amp * np.exp(1j * phase)
        flow_from = edge_data.get("flow_from")
        flow_to = edge_data.get("flow_to")
        sign = 1.0
        if flow_from == v and flow_to == u:
            sign = -1.0
        elif flow_from == u and flow_to == v:
            sign = 1.0
        return complex(sign * phasor), True, float(global_f0_hz)

    q_t = edge_data.get("Q_t_piv")
    if q_t is None or len(q_t) < 8:
        q_t = edge_data.get("Q_t")
    if q_t is not None and len(q_t) >= 8:
        qt_arr = np.asarray(q_t, dtype=np.float64)
        if np.nanmean(qt_arr) < 0.0:
            qt_arr = -qt_arr
        fit = fit_harmonics(
            qt_arr,
            FRAME_DT_S,
            float(global_f0_hz),
            K=harmonic_number,
            loss="huber",
            include_dc=True,
        )
        phasor = 0.0j
        for entry in fit.get("harmonics", []):
            if int(entry.get("k", -1)) == harmonic_number:
                phasor = complex(float(entry["A"]), -float(entry["B"]))
                break
        flow_from = edge_data.get("flow_from")
        flow_to = edge_data.get("flow_to")
        sign = 1.0
        if flow_from == v and flow_to == u:
            sign = -1.0
        elif flow_from == u and flow_to == v:
            sign = 1.0
        return complex(sign * phasor), True, float(global_f0_hz)

    return 0.0j, False, float(global_f0_hz)


def build_harmonic_measurements(graph, data, harmonic_number: int, global_f0_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.zeros(len(data.edge_ids), dtype=np.complex128)
    valid = np.zeros(len(data.edge_ids), dtype=bool)
    edge_f0 = np.full(len(data.edge_ids), float(global_f0_hz), dtype=np.float64)
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        phasor, ok, f0_hz = signed_measurement_phasor_nl_s(
            graph.edges[u, v],
            u,
            v,
            harmonic_number=harmonic_number,
            global_f0_hz=global_f0_hz,
        )
        observed[edge_idx] = phasor
        valid[edge_idx] = ok and np.isfinite(phasor.real) and np.isfinite(phasor.imag)
        edge_f0[edge_idx] = f0_hz
    return observed, valid, edge_f0


def validate_edge_frequencies(edge_f0_hz: np.ndarray, valid_mask: np.ndarray, *, rtol: float = 1.0e-6, atol: float = 1.0e-9) -> float:
    finite = np.isfinite(edge_f0_hz) & valid_mask
    if not np.any(finite):
        raise ValueError("No valid harmonic edge frequencies were found.")
    values = np.asarray(edge_f0_hz[finite], dtype=np.float64)
    reference = float(values[0])
    if not np.allclose(values, reference, rtol=rtol, atol=atol):
        raise ValueError(
            "Valid harmonic edges do not share a consistent fundamental frequency: "
            f"min={values.min():.12g} Hz, max={values.max():.12g} Hz"
        )
    return reference


def full_complex_matrix_diagnostics(
    matrix: np.ndarray,
    *,
    rcond: float = 1.0e-12,
    precomputed_rank: int | None = None,
) -> dict[str, float | bool]:
    if matrix.size == 0:
        return {
            "matrix_rank": 0,
            "matrix_rows": 0,
            "matrix_cols": 0,
            "min_singular_value": float("nan"),
            "max_singular_value": float("nan"),
            "condition_number": float("nan"),
            "is_full_column_rank": False,
        }
    matrix = np.asarray(matrix, dtype=np.complex128)
    if not (
        np.all(np.isfinite(matrix.real))
        and np.all(np.isfinite(matrix.imag))
    ):
        rows, cols = matrix.shape
        return {
            "matrix_rank": 0,
            "matrix_rows": int(rows),
            "matrix_cols": int(cols),
            "min_singular_value": float("nan"),
            "max_singular_value": float("nan"),
            "condition_number": float("nan"),
            "is_full_column_rank": False,
        }
    rows, cols = matrix.shape
    sigma_max = float("nan")
    sigma_min = float("nan")
    condition = float("nan")
    rank = int(precomputed_rank) if precomputed_rank is not None else -1
    try:
        from scipy import sparse
        from scipy.sparse.csgraph import structural_rank
        from scipy.sparse.linalg import svds

        sparse_matrix = sparse.csr_matrix(matrix)
        if rank < 0:
            rank = int(structural_rank(sparse_matrix))
        sigma_max = float(svds(sparse_matrix, k=1, which="LM", return_singular_vectors=False)[0])
        if max(rows, cols) <= 2000 and matrix.size <= 2_000_000:
            sigma_min = float(svds(sparse_matrix, k=1, which="SM", return_singular_vectors=False)[0])
    except Exception:
        if max(rows, cols) <= 2000 and matrix.size <= 2_000_000:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if singular_values.size:
                sigma_max = float(np.max(singular_values))
                sigma_min = float(np.min(singular_values))
                if rank < 0:
                    threshold = max(float(rcond) * sigma_max, 1.0e-30)
                    rank = int(np.count_nonzero(singular_values > threshold))
            else:
                rank = 0 if rank < 0 else rank
        elif rank < 0:
            rank = 0
    if rank < 0:
        if math.isfinite(sigma_min) and math.isfinite(sigma_max):
            threshold = max(float(rcond) * sigma_max, 1.0e-30)
            rank = int(cols if sigma_min > threshold else max(cols - 1, 0))
        else:
            rank = 0
    if math.isfinite(sigma_max) and math.isfinite(sigma_min):
        condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    return {
        "matrix_rank": rank,
        "matrix_rows": int(rows),
        "matrix_cols": int(cols),
        "min_singular_value": sigma_min,
        "max_singular_value": sigma_max,
        "condition_number": condition,
        "is_full_column_rank": bool(rank >= min(rows, cols)),
    }


def complex_to_real_system_torch(matrix: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real_matrix = torch.cat([matrix.real, -matrix.imag], dim=1)
    imag_matrix = torch.cat([matrix.imag, matrix.real], dim=1)
    stacked = torch.cat([real_matrix, imag_matrix], dim=0)
    target = torch.cat([rhs.real, rhs.imag], dim=0)
    return stacked, target


def solve_real_stacked_lsqr(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    atol: float = 1.0e-8,
    btol: float = 1.0e-8,
    iter_lim: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    real_matrix_t, real_rhs_t = complex_to_real_system_torch(matrix, rhs)
    real_matrix = real_matrix_t.detach().cpu().numpy().astype(np.float64, copy=False)
    real_rhs = real_rhs_t.detach().cpu().numpy().astype(np.float64, copy=False)
    if not (np.all(np.isfinite(real_matrix)) and np.all(np.isfinite(real_rhs))):
        raise ValueError("non-finite entries detected in real-stacked least-squares system")
    sparse_matrix = sparse.csr_matrix(real_matrix)
    result = lsqr(
        sparse_matrix,
        real_rhs,
        atol=float(atol),
        btol=float(btol),
        iter_lim=iter_lim,
    )
    solution_real = np.asarray(result[0], dtype=np.float64)
    half = solution_real.size // 2
    solution_complex = solution_real[:half] + 1j * solution_real[half:]
    info: dict[str, float | int | bool | str] = {
        "backend": "scipy_sparse_lsqr_real_stacked",
        "istop": int(result[1]),
        "iterations": int(result[2]),
        "r1norm": float(result[3]),
        "r2norm": float(result[4]),
        "anorm": float(result[5]),
        "acond": float(result[6]),
        "arnorm": float(result[7]),
        "xnorm": float(result[8]),
        "success": int(result[1]) in {1, 2},
    }
    return solution_complex.astype(np.complex128, copy=False), info


def solve_real_stacked_dense_lstsq(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    backend = str(backend).strip().lower()
    real_matrix_t, real_rhs_t = complex_to_real_system_torch(matrix, rhs)
    if backend == "torch":
        if not (bool(torch.all(torch.isfinite(real_matrix_t))) and bool(torch.all(torch.isfinite(real_rhs_t)))):
            raise ValueError("non-finite entries detected in real-stacked least-squares system")
        rhs_norm_reference = float(torch.linalg.vector_norm(real_rhs_t).detach().cpu())
        if real_matrix_t.device.type == "cuda":
            real_matrix_gpu = real_matrix_t.to(dtype=torch.float64)
            real_rhs_gpu = real_rhs_t.to(dtype=torch.float64)
            try:
                lstsq_result_gpu = torch.linalg.lstsq(real_matrix_gpu, real_rhs_gpu)
                solution_real_gpu = lstsq_result_gpu.solution
                residual_gpu = real_matrix_gpu @ solution_real_gpu - real_rhs_gpu
                residual_norm_gpu = float(torch.linalg.vector_norm(residual_gpu).detach().cpu())
                relative_residual_gpu = residual_norm_gpu / max(rhs_norm_reference, 1.0e-30)
                if bool(torch.all(torch.isfinite(solution_real_gpu))) and math.isfinite(relative_residual_gpu):
                    solution_real = solution_real_gpu.detach().cpu().numpy().astype(np.float64, copy=False)
                    half = solution_real.size // 2
                    solution_complex = solution_real[:half] + 1j * solution_real[half:]
                    singular_values_t = lstsq_result_gpu.singular_values
                    singular_values = (
                        singular_values_t.detach().cpu().numpy().astype(np.float64, copy=False)
                        if singular_values_t is not None
                        else np.asarray([], dtype=np.float64)
                    )
                    rank = _extract_lstsq_rank(lstsq_result_gpu.rank)
                    sigma_max = float(np.max(singular_values)) if np.size(singular_values) else float("nan")
                    sigma_min = float(np.min(singular_values)) if np.size(singular_values) else float("nan")
                    info: dict[str, float | int | bool | str] = {
                        "backend": "torch_linalg_lstsq_real_stacked_cuda",
                        "istop": float("nan"),
                        "iterations": float("nan"),
                        "r1norm": residual_norm_gpu,
                        "r2norm": residual_norm_gpu,
                        "anorm": sigma_max,
                        "acond": float(sigma_max / sigma_min) if sigma_min > 0.0 and math.isfinite(sigma_max) and math.isfinite(sigma_min) else float("nan"),
                        "arnorm": float("nan"),
                        "xnorm": float(torch.linalg.vector_norm(solution_real_gpu).detach().cpu()),
                        "rank": rank,
                        "success": True,
                        "relative_residual": relative_residual_gpu,
                    }
                    return solution_complex.astype(np.complex128, copy=False), info
            except RuntimeError:
                pass
        # Fallback path: use CPU Torch lstsq with a rank-revealing driver.
        real_matrix_solve = real_matrix_t.to(device="cpu", dtype=torch.float64)
        real_rhs_solve = real_rhs_t.to(device="cpu", dtype=torch.float64)
        lstsq_result = torch.linalg.lstsq(real_matrix_solve, real_rhs_solve, driver="gelsd")
        solution_real_t = lstsq_result.solution
        residual_t = real_matrix_solve @ solution_real_t - real_rhs_solve
        solution_real = solution_real_t.detach().cpu().numpy().astype(np.float64, copy=False)
        half = solution_real.size // 2
        solution_complex = solution_real[:half] + 1j * solution_real[half:]
        residual_norm = float(torch.linalg.vector_norm(residual_t).detach().cpu())
        singular_values_t = lstsq_result.singular_values
        singular_values = (
            singular_values_t.detach().cpu().numpy().astype(np.float64, copy=False)
            if singular_values_t is not None
            else np.asarray([], dtype=np.float64)
        )
        rank = _extract_lstsq_rank(lstsq_result.rank)
        sigma_max = float(np.max(singular_values)) if np.size(singular_values) else float("nan")
        sigma_min = float(np.min(singular_values)) if np.size(singular_values) else float("nan")
        info = {
            "backend": "torch_linalg_lstsq_real_stacked_cpu_gelsd",
            "istop": float("nan"),
            "iterations": float("nan"),
            "r1norm": residual_norm,
            "r2norm": residual_norm,
            "anorm": sigma_max,
            "acond": float(sigma_max / sigma_min) if sigma_min > 0.0 and math.isfinite(sigma_max) and math.isfinite(sigma_min) else float("nan"),
            "arnorm": float("nan"),
            "xnorm": float(torch.linalg.vector_norm(solution_real_t).detach().cpu()),
            "rank": rank,
            "success": True,
            "relative_residual": residual_norm / max(rhs_norm_reference, 1.0e-30),
        }
        return solution_complex.astype(np.complex128, copy=False), info
    if backend != "numpy":
        raise ValueError(f"Unsupported dense least-squares backend: {backend}")
    real_matrix = real_matrix_t.detach().cpu().numpy().astype(np.float64, copy=False)
    real_rhs = real_rhs_t.detach().cpu().numpy().astype(np.float64, copy=False)
    if not (np.all(np.isfinite(real_matrix)) and np.all(np.isfinite(real_rhs))):
        raise ValueError("non-finite entries detected in real-stacked least-squares system")
    solution_real, residuals, rank, singular_values = np.linalg.lstsq(real_matrix, real_rhs, rcond=None)
    half = solution_real.size // 2
    solution_complex = solution_real[:half] + 1j * solution_real[half:]
    residual_norm = float(np.sqrt(float(residuals[0]))) if np.size(residuals) else float(np.linalg.norm(real_matrix @ solution_real - real_rhs))
    sigma_max = float(np.max(singular_values)) if np.size(singular_values) else float("nan")
    sigma_min = float(np.min(singular_values)) if np.size(singular_values) else float("nan")
    info = {
        "backend": "numpy_linalg_lstsq_real_stacked",
        "istop": float("nan"),
        "iterations": float("nan"),
        "r1norm": residual_norm,
        "r2norm": residual_norm,
        "anorm": sigma_max,
        "acond": float(sigma_max / sigma_min) if sigma_min > 0.0 and math.isfinite(sigma_max) and math.isfinite(sigma_min) else float("nan"),
        "arnorm": float("nan"),
        "xnorm": float(np.linalg.norm(solution_real)),
        "rank": int(rank),
        "success": True,
    }
    return solution_complex.astype(np.complex128, copy=False), info


def _extract_lstsq_rank(rank_tensor: torch.Tensor | None) -> int:
    if rank_tensor is None:
        return -1
    rank_flat = rank_tensor.detach().reshape(-1)
    if rank_flat.numel() == 0:
        return -1
    return int(rank_flat[0].item())


def solve_real_dense_lstsq_system(
    real_matrix: torch.Tensor,
    real_rhs: torch.Tensor,
    *,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve a real least-squares system whose unknown is [Re(P), Im(P)].

    This is used when complex residual blocks are combined with genuinely real
    phase-only constraint rows.
    """
    backend = str(backend).strip().lower()
    if not (
        bool(torch.all(torch.isfinite(real_matrix)))
        and bool(torch.all(torch.isfinite(real_rhs)))
    ):
        raise ValueError("non-finite entries detected in real least-squares system")

    matrix_cpu = real_matrix.to(device="cpu", dtype=torch.float64)
    rhs_cpu = real_rhs.to(device="cpu", dtype=torch.float64)

    if backend == "torch":
        rhs_norm_reference = float(torch.linalg.vector_norm(real_rhs).detach().cpu())
        if real_matrix.device.type == "cuda":
            matrix_gpu = real_matrix.to(dtype=torch.float64)
            rhs_gpu = real_rhs.to(dtype=torch.float64)
            try:
                result_gpu = torch.linalg.lstsq(matrix_gpu, rhs_gpu)
                solution_real_t = result_gpu.solution
                residual_t = matrix_gpu @ solution_real_t - rhs_gpu
                residual_norm = float(torch.linalg.vector_norm(residual_t).detach().cpu())
                relative_residual = residual_norm / max(rhs_norm_reference, 1.0e-30)
                if bool(torch.all(torch.isfinite(solution_real_t))) and math.isfinite(relative_residual):
                    singular_values_t = result_gpu.singular_values
                    singular_values = (
                        singular_values_t.detach().cpu().numpy().astype(np.float64, copy=False)
                        if singular_values_t is not None
                        else np.asarray([], dtype=np.float64)
                    )
                    rank = _extract_lstsq_rank(result_gpu.rank)
                    solution_real = solution_real_t.detach().cpu().numpy().astype(np.float64, copy=False)
                    backend_name = "torch_linalg_lstsq_real_mixed_cuda"
                else:
                    raise RuntimeError("CUDA lstsq returned non-finite values.")
            except RuntimeError:
                result = torch.linalg.lstsq(matrix_cpu, rhs_cpu, driver="gelsd")
                solution_real_t = result.solution
                residual_t = matrix_cpu @ solution_real_t - rhs_cpu
                singular_values_t = result.singular_values
                singular_values = (
                    singular_values_t.detach().cpu().numpy().astype(np.float64, copy=False)
                    if singular_values_t is not None
                    else np.asarray([], dtype=np.float64)
                )
                rank = _extract_lstsq_rank(result.rank)
                solution_real = solution_real_t.detach().cpu().numpy().astype(np.float64, copy=False)
                residual_norm = float(torch.linalg.vector_norm(residual_t).detach().cpu())
                backend_name = "torch_linalg_lstsq_real_mixed_cpu_gelsd"
        else:
            result = torch.linalg.lstsq(matrix_cpu, rhs_cpu, driver="gelsd")
            solution_real_t = result.solution
            residual_t = matrix_cpu @ solution_real_t - rhs_cpu
            singular_values_t = result.singular_values
            singular_values = (
                singular_values_t.detach().cpu().numpy().astype(np.float64, copy=False)
                if singular_values_t is not None
                else np.asarray([], dtype=np.float64)
            )
            rank = _extract_lstsq_rank(result.rank)
            solution_real = solution_real_t.detach().cpu().numpy().astype(np.float64, copy=False)
            residual_norm = float(torch.linalg.vector_norm(residual_t).detach().cpu())
            backend_name = "torch_linalg_lstsq_real_mixed_cpu_gelsd"
    elif backend == "numpy":
        matrix_np = matrix_cpu.detach().cpu().numpy().astype(np.float64, copy=False)
        rhs_np = rhs_cpu.detach().cpu().numpy().astype(np.float64, copy=False)
        solution_real, residuals, rank, singular_values = np.linalg.lstsq(
            matrix_np,
            rhs_np,
            rcond=None,
        )
        residual_norm = (
            float(np.sqrt(float(residuals[0])))
            if np.size(residuals)
            else float(np.linalg.norm(matrix_np @ solution_real - rhs_np))
        )
        backend_name = "numpy_linalg_lstsq_real_mixed"
    else:
        raise ValueError(f"Unsupported dense least-squares backend: {backend}")

    half = solution_real.size // 2
    if 2 * half != solution_real.size:
        raise ValueError("real pressure solution must contain equal real and imaginary halves")
    solution_complex = solution_real[:half] + 1j * solution_real[half:]

    singular_values = np.asarray(singular_values, dtype=np.float64)
    sigma_max = float(np.max(singular_values)) if np.size(singular_values) else float("nan")
    sigma_min = float(np.min(singular_values)) if np.size(singular_values) else float("nan")
    nonzero_threshold = (
        float(np.finfo(np.float64).eps) * max(matrix_cpu.shape) * sigma_max
        if np.size(singular_values) and math.isfinite(sigma_max)
        else float("nan")
    )
    finite_nonzero = (
        singular_values[np.isfinite(singular_values) & (singular_values > max(nonzero_threshold, 0.0))]
        if np.size(singular_values)
        else np.asarray([], dtype=np.float64)
    )
    info: dict[str, float | int | bool | str] = {
        "backend": backend_name,
        "istop": float("nan"),
        "iterations": float("nan"),
        "r1norm": residual_norm,
        "r2norm": residual_norm,
        "anorm": sigma_max,
        "acond": (
            float(sigma_max / sigma_min)
            if sigma_min > 0.0 and math.isfinite(sigma_max) and math.isfinite(sigma_min)
            else float("nan")
        ),
        "arnorm": float("nan"),
        "xnorm": float(np.linalg.norm(solution_real)),
        "rank": int(rank),
        "success": True,
        "singular_values_desc": singular_values.tolist(),
        "smallest_singular_values": singular_values[-10:].tolist() if np.size(singular_values) else [],
        "largest_singular_values": singular_values[:10].tolist() if np.size(singular_values) else [],
        "smallest_nonzero_singular_value": (
            float(finite_nonzero[-1]) if finite_nonzero.size else float("nan")
        ),
        "nonzero_singular_value_count": int(finite_nonzero.size),
        "relative_residual": residual_norm / max(
            float(torch.linalg.vector_norm(real_rhs).detach().cpu()),
            1.0e-30,
        ),
    }
    return solution_complex.astype(np.complex128, copy=False), info


def summarize_magnitude_stats(values: np.ndarray, *, nonzero_only: bool = False) -> dict[str, float]:
    mags = np.abs(np.asarray(values))
    finite = mags[np.isfinite(mags)]
    if nonzero_only:
        finite = finite[finite > 0.0]
    if finite.size == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def block_diagnostics(matrix: np.ndarray, rhs: np.ndarray, *, weight_sqrt: float, normalization_scale: float) -> dict[str, object]:
    matrix_np = np.asarray(matrix)
    rhs_np = np.asarray(rhs)
    matrix_abs = np.abs(matrix_np)
    rhs_abs = np.abs(rhs_np)
    return {
        "shape": [int(dim) for dim in matrix_np.shape],
        "fro_norm": float(np.linalg.norm(matrix_np)),
        "max_abs_entry": float(np.max(matrix_abs)) if matrix_abs.size else 0.0,
        "rhs_norm": float(np.linalg.norm(rhs_np)),
        "rhs_max_abs_entry": float(np.max(rhs_abs)) if rhs_abs.size else 0.0,
        "weight_sqrt": float(weight_sqrt),
        "normalization_scale": float(normalization_scale),
    }


def common_phase_rad(pressures: np.ndarray, arterial_idx: np.ndarray) -> float:
    """Circular mean of arterial pressure phases, independent of amplitudes."""
    if arterial_idx.size == 0:
        return 0.0
    arterial = np.asarray(pressures[arterial_idx], dtype=np.complex128)
    finite = np.isfinite(arterial.real) & np.isfinite(arterial.imag) & (np.abs(arterial) > 0.0)
    if not np.any(finite):
        return 0.0
    unit = arterial[finite] / np.abs(arterial[finite])
    mean_unit = np.mean(unit)
    if abs(mean_unit) < 1.0e-15:
        # Degenerate circular mean; use the first finite arterial phase.
        return float(np.angle(unit[0]))
    return float(np.angle(mean_unit))


def phase_only_constraint_rows(
    pressures: np.ndarray,
    arterial_idx: np.ndarray,
    num_nodes: int,
    device: torch.device,
    *,
    amplitude_floor_fraction: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Build normalized real rows enforcing a common arterial phase.

    For fixed common phase phi, each row penalizes

        Im(exp(-i phi) P_Ak) / a_scale,k,

    which changes phase but does not directly penalize the in-phase pressure
    amplitude. The rows act on [Re(P), Im(P)].
    """
    arterial_idx = np.asarray(arterial_idx, dtype=np.int64)
    if arterial_idx.size == 0:
        return (
            torch.zeros((0, 2 * int(num_nodes)), dtype=torch.float64, device=device),
            torch.zeros((0,), dtype=torch.float64, device=device),
            0.0,
            1.0,
        )

    phi = common_phase_rad(pressures, arterial_idx)
    arterial_amp = np.abs(np.asarray(pressures[arterial_idx], dtype=np.complex128))
    finite_positive = arterial_amp[np.isfinite(arterial_amp) & (arterial_amp > 0.0)]
    reference_amp = float(np.median(finite_positive)) if finite_positive.size else 1.0
    floor = max(float(amplitude_floor_fraction) * reference_amp, 1.0e-12)
    scales = np.maximum(
        np.where(np.isfinite(arterial_amp), arterial_amp, 0.0),
        floor,
    )

    rows = torch.zeros(
        (arterial_idx.size, 2 * int(num_nodes)),
        dtype=torch.float64,
        device=device,
    )
    row_ids = torch.arange(arterial_idx.size, dtype=torch.long, device=device)
    node_ids = torch.as_tensor(arterial_idx, dtype=torch.long, device=device)
    inv_scale = torch.as_tensor(1.0 / scales, dtype=torch.float64, device=device)

    # Im(exp(-i phi) P) = -sin(phi) Re(P) + cos(phi) Im(P)
    rows[row_ids, node_ids] = -math.sin(phi) * inv_scale
    rows[row_ids, int(num_nodes) + node_ids] = math.cos(phi) * inv_scale
    rhs = torch.zeros((arterial_idx.size,), dtype=torch.float64, device=device)
    return rows, rhs, phi, floor


def _build_source_driven_base_blocks(
    laplacian: torch.Tensor,
    source_t: torch.Tensor,
    flow_matrix: torch.Tensor,
    q_obs_t: torch.Tensor | None,
    flow_rows: np.ndarray | None,
    flow_row_weights_t: torch.Tensor | None,
    lambda_k: float,
    lambda_q: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
    """Build normalized complex Kirchhoff and optional observed-flow blocks."""
    blocks: list[torch.Tensor] = []
    rhs_blocks: list[torch.Tensor] = []
    laplacian_scale_value = 1.0
    flow_scale_value = 1.0

    if lambda_k > 0.0:
        nonzero = torch.abs(laplacian) > 0.0
        laplacian_scale = (
            torch.median(torch.abs(laplacian[nonzero])).clamp_min(1.0e-30)
            if bool(torch.any(nonzero))
            else torch.tensor(1.0, dtype=torch.float64, device=device)
        )
        laplacian_scale_value = float(laplacian_scale.detach().cpu())
        blocks.append(math.sqrt(lambda_k) * (laplacian / laplacian_scale.to(dtype)))
        rhs_blocks.append(math.sqrt(lambda_k) * (source_t / laplacian_scale.to(dtype)))

    if (
        lambda_q > 0.0
        and q_obs_t is not None
        and flow_rows is not None
        and flow_rows.size > 0
    ):
        flow_rows_t = torch.as_tensor(flow_rows, dtype=torch.long, device=device)
        flow_block = flow_matrix.index_select(0, flow_rows_t)
        nonzero = torch.abs(flow_block) > 0.0
        flow_scale = (
            torch.median(torch.abs(flow_block[nonzero])).clamp_min(1.0e-30)
            if bool(torch.any(nonzero))
            else torch.tensor(1.0, dtype=torch.float64, device=device)
        )
        flow_scale_value = float(flow_scale.detach().cpu())
        weighted_flow_block = flow_block / flow_scale.to(dtype)
        weighted_flow_rhs = q_obs_t.index_select(0, flow_rows_t) / flow_scale.to(dtype)
        if flow_row_weights_t is not None:
            row_weights = flow_row_weights_t.index_select(0, flow_rows_t).to(dtype=torch.float64, device=device)
            row_sqrt = torch.sqrt(row_weights.clamp_min(1.0e-12)).to(dtype).unsqueeze(1)
            weighted_flow_block = weighted_flow_block * row_sqrt
            weighted_flow_rhs = weighted_flow_rhs * row_sqrt.squeeze(1)
        blocks.append(math.sqrt(lambda_q) * weighted_flow_block)
        rhs_blocks.append(
            math.sqrt(lambda_q) * weighted_flow_rhs
        )

    return blocks, rhs_blocks, laplacian_scale_value, flow_scale_value


def _solve_source_driven_with_phase_only_constraint(
    *,
    laplacian: torch.Tensor,
    flow_matrix: torch.Tensor,
    source_t: torch.Tensor,
    q_obs_t: torch.Tensor | None,
    flow_rows: np.ndarray | None,
    flow_row_weights_t: torch.Tensor | None,
    arterial_idx: np.ndarray,
    lambda_k: float,
    lambda_q: float,
    lambda_b: float,
    max_iterations: int,
    tol: float,
    device: torch.device,
    lstsq_backend: str,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Solve the active source-driven harmonic least-squares formulation.

    The pressure is initialized from the unconstrained Kirchhoff/flow problem.
    The common arterial phase is then enforced with normalized real phase-only
    rows, avoiding direct pressure-amplitude shrinkage.
    """
    dtype = torch.complex128
    num_nodes = int(laplacian.shape[0])
    arterial_idx = np.asarray(arterial_idx, dtype=np.int64)

    base_blocks, base_rhs_blocks, laplacian_scale, flow_scale = (
        _build_source_driven_base_blocks(
            laplacian=laplacian,
            source_t=source_t,
            flow_matrix=flow_matrix,
            q_obs_t=q_obs_t,
            flow_rows=flow_rows,
            flow_row_weights_t=flow_row_weights_t,
            lambda_k=float(lambda_k),
            lambda_q=float(lambda_q),
            dtype=dtype,
            device=device,
        )
    )
    if not base_blocks:
        raise ValueError(
            "At least one of lambda_k or lambda_q must be positive; "
            "a phase-only constraint cannot determine pressure amplitudes."
        )

    complex_matrix = torch.cat(base_blocks, dim=0)
    complex_rhs = torch.cat(base_rhs_blocks, dim=0)
    base_real_matrix, base_real_rhs = complex_to_real_system_torch(
        complex_matrix,
        complex_rhs,
    )

    # Unconstrained initialization: prevents the first boundary update from
    # pulling arterial pressures toward zero.
    pressure, init_info = solve_real_dense_lstsq_system(
        base_real_matrix,
        base_real_rhs,
        backend=str(lstsq_backend),
    )

    solver_info: dict[str, object] = {
        "backend": str(init_info.get("backend", "")),
        "device": str(device),
        "istop": float(init_info.get("istop", float("nan"))),
        "iterations": float(init_info.get("iterations", float("nan"))),
        "r1norm": float(init_info.get("r1norm", float("nan"))),
        "r2norm": float(init_info.get("r2norm", float("nan"))),
        "anorm": float(init_info.get("anorm", float("nan"))),
        "acond": float(init_info.get("acond", float("nan"))),
        "arnorm": float(init_info.get("arnorm", float("nan"))),
        "xnorm": float(init_info.get("xnorm", float("nan"))),
        "matrix_rank": float(init_info.get("rank", float("nan"))),
        "phase_iterations_used": 0,
        "phase_iteration_relative_change": float("nan"),
        "converged": lambda_b <= 0.0 or arterial_idx.size == 0,
        "success": bool(init_info.get("success", True)),
        "message": "unconstrained source-driven initialization completed",
        "phase_constraint_kind": "normalized_phase_only_real_rows",
        "laplacian_scale": laplacian_scale,
        "flow_scale": flow_scale,
        "initialization": "unconstrained_kirchhoff_flow_lstsq",
        "block_diagnostics": {},
    }
    block_info = dict(solver_info["block_diagnostics"])
    if lambda_k > 0.0:
        kirchhoff_matrix = base_blocks[0].detach().cpu().numpy().astype(np.complex128, copy=False)
        kirchhoff_rhs = base_rhs_blocks[0].detach().cpu().numpy().astype(np.complex128, copy=False)
        block_info["kirchhoff_source_block"] = block_diagnostics(
            kirchhoff_matrix,
            kirchhoff_rhs,
            weight_sqrt=math.sqrt(lambda_k),
            normalization_scale=laplacian_scale,
        )
    if lambda_q > 0.0 and q_obs_t is not None and flow_rows is not None and flow_rows.size > 0:
        flow_matrix_block = base_blocks[-1].detach().cpu().numpy().astype(np.complex128, copy=False)
        flow_rhs_block = base_rhs_blocks[-1].detach().cpu().numpy().astype(np.complex128, copy=False)
        block_info["edge_flow_fit_block"] = block_diagnostics(
            flow_matrix_block,
            flow_rhs_block,
            weight_sqrt=math.sqrt(lambda_q),
            normalization_scale=flow_scale,
        )
    solver_info["block_diagnostics"] = block_info
    final_real_matrix = base_real_matrix
    final_real_rhs = base_real_rhs

    if lambda_b > 0.0 and arterial_idx.size > 0:
        converged = False
        for iteration in range(max(1, int(max_iterations))):
            phase_rows, phase_rhs, phi, amp_floor = phase_only_constraint_rows(
                pressures=pressure,
                arterial_idx=arterial_idx,
                num_nodes=num_nodes,
                device=device,
            )
            real_matrix = torch.cat(
                [base_real_matrix, math.sqrt(lambda_b) * phase_rows],
                dim=0,
            )
            real_rhs = torch.cat(
                [base_real_rhs, math.sqrt(lambda_b) * phase_rhs],
                dim=0,
            )
            new_pressure, ls_info = solve_real_dense_lstsq_system(
                real_matrix,
                real_rhs,
                backend=str(lstsq_backend),
            )
            denom = max(float(np.linalg.norm(new_pressure)), 1.0e-30)
            delta = float(np.linalg.norm(new_pressure - pressure) / denom)
            pressure = new_pressure
            final_real_matrix = real_matrix
            final_real_rhs = real_rhs
            solver_info.update(
                {
                    "backend": str(ls_info.get("backend", solver_info["backend"])),
                    "istop": float(ls_info.get("istop", float("nan"))),
                    "iterations": float(ls_info.get("iterations", float("nan"))),
                    "r1norm": float(ls_info.get("r1norm", float("nan"))),
                    "r2norm": float(ls_info.get("r2norm", float("nan"))),
                    "anorm": float(ls_info.get("anorm", float("nan"))),
                    "acond": float(ls_info.get("acond", float("nan"))),
                    "arnorm": float(ls_info.get("arnorm", float("nan"))),
                    "xnorm": float(ls_info.get("xnorm", float("nan"))),
                    "matrix_rank": float(ls_info.get("rank", float("nan"))),
                    "phase_iterations_used": iteration + 1,
                    "phase_iteration_relative_change": delta,
                    "common_arterial_phase_rad": phi,
                    "common_arterial_phase_deg": phi * DEG_PER_RAD,
                    "phase_constraint_amplitude_floor_pa": amp_floor,
                }
            )
            phase_matrix_np = (math.sqrt(lambda_b) * phase_rows).detach().cpu().numpy().astype(np.float64, copy=False)
            phase_rhs_np = (math.sqrt(lambda_b) * phase_rhs).detach().cpu().numpy().astype(np.float64, copy=False)
            solver_info["block_diagnostics"]["arterial_phase_block"] = block_diagnostics(
                phase_matrix_np,
                phase_rhs_np,
                weight_sqrt=math.sqrt(lambda_b),
                normalization_scale=1.0,
            )
            if delta <= float(tol):
                converged = True
                solver_info["message"] = "normalized phase-only iteration converged"
                break

        if not converged:
            solver_info["message"] = (
                "normalized phase-only iteration reached max iterations "
                "without meeting tolerance"
            )
        solver_info["converged"] = converged
        solver_info["success"] = converged

    final_matrix_np = (
        final_real_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    final_rhs_np = final_real_rhs.detach().cpu().numpy().astype(np.float64, copy=False)
    ones_complex = np.ones(num_nodes, dtype=np.complex128)
    ones_real = np.concatenate([np.ones(num_nodes, dtype=np.float64), np.zeros(num_nodes, dtype=np.float64)])
    solver_info["matrix_diagnostics"] = full_complex_matrix_diagnostics(
        final_matrix_np.astype(np.complex128),
        precomputed_rank=(
            int(solver_info["matrix_rank"])
            if np.isfinite(float(solver_info.get("matrix_rank", float("nan"))))
            else None
        ),
    )
    solver_info["final_real_matrix_diagnostics"] = {
        "shape": [int(dim) for dim in final_matrix_np.shape],
        "fro_norm": float(np.linalg.norm(final_matrix_np)),
        "max_abs_entry": float(np.max(np.abs(final_matrix_np))) if final_matrix_np.size else 0.0,
        "rhs_norm": float(np.linalg.norm(final_rhs_np)),
        "rhs_max_abs_entry": float(np.max(np.abs(final_rhs_np))) if final_rhs_np.size else 0.0,
        "ones_response_norm": float(np.linalg.norm(final_matrix_np @ ones_real)),
        "ones_response_max_abs": float(np.max(np.abs(final_matrix_np @ ones_real))) if final_matrix_np.size else 0.0,
    }
    solver_info["nullspace_diagnostics"] = {
        "laplacian_times_ones_norm": float(
            np.linalg.norm(
                laplacian.detach().cpu().numpy().astype(np.complex128, copy=False) @ ones_complex
            )
        ),
        "laplacian_times_ones_max_abs": float(
            np.max(
                np.abs(
                    laplacian.detach().cpu().numpy().astype(np.complex128, copy=False) @ ones_complex
                )
            )
        ),
        "flow_matrix_times_ones_norm": float(
            np.linalg.norm(
                flow_matrix.detach().cpu().numpy().astype(np.complex128, copy=False) @ ones_complex
            )
        ),
        "flow_matrix_times_ones_max_abs": float(
            np.max(
                np.abs(
                    flow_matrix.detach().cpu().numpy().astype(np.complex128, copy=False) @ ones_complex
                )
            )
        ),
    }

    arterial = pressure[arterial_idx] if arterial_idx.size else np.zeros((0,), dtype=np.complex128)
    if arterial.size:
        phi = common_phase_rad(pressure, arterial_idx)
        projected = np.real(arterial * np.exp(-1j * phi))
        solver_info["arterial_antiphase_count"] = int(np.count_nonzero(projected < 0.0))
        solver_info["arterial_min_in_phase_projection_pa"] = float(np.min(projected))
    else:
        solver_info["arterial_antiphase_count"] = 0
        solver_info["arterial_min_in_phase_projection_pa"] = float("nan")

    return pressure, solver_info, final_real_matrix.detach().cpu().numpy()

def pressure_targets_common_phase(
    pressures: np.ndarray,
    arterial_idx: np.ndarray,
    phase_offset: float,
) -> np.ndarray:
    """Diagnostic target phasors for a common arterial phase.

    The active solver uses phase-only real constraint rows. These targets are
    retained for output diagnostics. Nonzero phase_offset is rejected because
    an absolute phase offset is not part of the equal-phase boundary condition.
    """
    if abs(float(phase_offset)) > 1.0e-15:
        raise ValueError(
            "phase_offset must be zero for the common-phase constraint. "
            "The previous implementation canceled this value algebraically."
        )
    if arterial_idx.size == 0:
        return np.zeros((0,), dtype=np.complex128)
    arterial_pressures = np.asarray(pressures[arterial_idx], dtype=np.complex128)
    phi = common_phase_rad(pressures, np.asarray(arterial_idx, dtype=np.int64))
    return np.abs(arterial_pressures) * np.exp(1j * phi)


def pressure_targets(
    pressures: np.ndarray,
    arterial_idx: np.ndarray,
    pressure_constraint_type: str,
    phase_offset: float,
) -> np.ndarray:
    if pressure_constraint_type in {"none", "pure_direct", ""}:
        return np.zeros((0,), dtype=np.complex128)
    if pressure_constraint_type in {"common_arterial_phase", "equal_phase"}:
        return pressure_targets_common_phase(pressures, arterial_idx, phase_offset)
    if pressure_constraint_type in {"equal_arterial_pressure_phasors", "equal_phasor"}:
        if arterial_idx.size == 0:
            return np.zeros((0,), dtype=np.complex128)
        arterial_pressures = pressures[arterial_idx]
        finite = np.isfinite(arterial_pressures.real) & np.isfinite(arterial_pressures.imag)
        if not np.any(finite):
            return np.zeros(arterial_idx.size, dtype=np.complex128)
        common_phasor = complex(np.mean(arterial_pressures[finite]))
        targets = np.zeros(arterial_idx.size, dtype=np.complex128)
        targets[:] = common_phasor
        return targets
    raise ValueError(f"Unsupported pressure_constraint_type: {pressure_constraint_type}")


def build_complex_admittance_matrices(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = int(np.max(edge_index)) + 1 if edge_index.size else 0
    dtype = torch.complex128
    source_t = torch.as_tensor(edge_index[0], dtype=torch.long, device=device)
    target_t = torch.as_tensor(edge_index[1], dtype=torch.long, device=device)
    rows_t = torch.arange(int(source_t.numel()), dtype=torch.long, device=device)
    admittance_diag_t = torch.as_tensor(admittance_diag, dtype=dtype, device=device)
    admittance_off_t = torch.as_tensor(admittance_off, dtype=dtype, device=device)

    flow_matrix = torch.zeros((len(admittance_diag), num_nodes), dtype=dtype, device=device)
    flow_matrix.index_put_((rows_t, source_t), admittance_diag_t, accumulate=True)
    flow_matrix.index_put_((rows_t, target_t), admittance_off_t, accumulate=True)

    laplacian = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    laplacian.index_put_((source_t, source_t), admittance_diag_t, accumulate=True)
    laplacian.index_put_((target_t, target_t), admittance_diag_t, accumulate=True)
    laplacian.index_put_((source_t, target_t), admittance_off_t, accumulate=True)
    laplacian.index_put_((target_t, source_t), admittance_off_t, accumulate=True)
    return flow_matrix, laplacian


def _finalize_source_driven_solution(
    pressure: np.ndarray,
    flow_matrix: torch.Tensor,
    laplacian: torch.Tensor,
    source_vector_m3_s: np.ndarray,
    arterial_idx: np.ndarray,
    pressure_constraint_type: str,
    phase_offset: float,
    solver_info: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    # Internal source-driven solve units:
    # admittances: m^3 / (s Pa)
    # source injections: m^3 / s
    # pressure: Pa
    # predicted flows: m^3 / s
    # nodal residuals: m^3 / s
    dtype = torch.complex128
    device = flow_matrix.device
    pressure_t = torch.as_tensor(pressure, dtype=dtype, device=device)
    flow_pred_m3_s = (flow_matrix @ pressure_t).detach().cpu().numpy().astype(np.complex128)
    nodal_balance_m3_s = (laplacian @ pressure_t).detach().cpu().numpy().astype(np.complex128)
    nodal_residual_m3_s = nodal_balance_m3_s - np.asarray(source_vector_m3_s, dtype=np.complex128)
    final_targets = (
        pressure_targets(
            pressure,
            arterial_idx,
            pressure_constraint_type=str(pressure_constraint_type),
            phase_offset=float(phase_offset),
        )
        if arterial_idx.size
        else np.zeros((0,), dtype=np.complex128)
    )
    finite_pressure = bool(np.all(np.isfinite(pressure.real)) and np.all(np.isfinite(pressure.imag)))
    finite_flow = bool(np.all(np.isfinite(flow_pred_m3_s.real)) and np.all(np.isfinite(flow_pred_m3_s.imag)))
    solver_info["success"] = finite_pressure and finite_flow and bool(solver_info.get("success", True))
    boundary_residual_deg = phase_target_residual_deg(pressure, final_targets, arterial_idx)
    diagnostics = {
        "flow_pred_m3_s": flow_pred_m3_s,
        "nodal_balance_m3_s": nodal_balance_m3_s,
        "nodal_residual_m3_s": nodal_residual_m3_s,
        "laplacian_m3_s_per_pa": laplacian.detach().cpu().numpy().astype(np.complex128),
        "flow_matrix_m3_s_per_pa": flow_matrix.detach().cpu().numpy().astype(np.complex128),
        "source_vector_m3_s": np.asarray(source_vector_m3_s, dtype=np.complex128),
        "boundary_targets_pa": final_targets,
        "boundary_phase_target_residual_deg": boundary_residual_deg,
        "lsqr_info": solver_info,
    }
    return pressure, diagnostics


def solve_complex_pressure_with_nodal_injections(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    source_vector_m3_s: np.ndarray,
    arterial_idx: np.ndarray,
    lambda_k: float,
    lambda_b: float,
    pressure_constraint_type: str,
    phase_offset: float,
    max_iterations: int,
    tol: float,
    device: torch.device,
    lstsq_backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, object]]:
    if pressure_constraint_type not in {
        "none",
        "pure_direct",
        "",
        "common_arterial_phase",
        "equal_phase",
    }:
        raise ValueError(
            "The corrected source-driven solver supports only no pressure "
            "constraint or a common arterial phase constraint."
        )
    if abs(float(phase_offset)) > 1.0e-15:
        raise ValueError(
            "phase_offset must be zero; an absolute phase offset is not part "
            "of the equal-phase boundary condition."
        )

    dtype = torch.complex128
    flow_matrix, laplacian = build_complex_admittance_matrices(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        device=device,
    )
    source_t = torch.as_tensor(source_vector_m3_s, dtype=dtype, device=device)
    use_phase_constraint = pressure_constraint_type in {
        "common_arterial_phase",
        "equal_phase",
    }

    pressure, solver_info, _ = _solve_source_driven_with_phase_only_constraint(
        laplacian=laplacian,
        flow_matrix=flow_matrix,
        source_t=source_t,
        q_obs_t=None,
        flow_rows=None,
        arterial_idx=np.asarray(arterial_idx, dtype=np.int64),
        lambda_k=float(lambda_k),
        lambda_q=0.0,
        lambda_b=float(lambda_b) if use_phase_constraint else 0.0,
        max_iterations=int(max_iterations),
        tol=float(tol),
        device=device,
        lstsq_backend=str(lstsq_backend),
    )
    return _finalize_source_driven_solution(
        pressure=pressure,
        flow_matrix=flow_matrix,
        laplacian=laplacian,
        source_vector_m3_s=source_vector_m3_s,
        arterial_idx=np.asarray(arterial_idx, dtype=np.int64),
        pressure_constraint_type=pressure_constraint_type,
        phase_offset=0.0,
        solver_info=solver_info,
    )


def solve_complex_pressure_direct(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    source_vector_m3_s: np.ndarray,
    device: torch.device,
    *,
    reference_node: int | None = None,
    singular_rcond: float = 1.0e-12,
    compare_with_numpy: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    dtype = torch.complex128
    flow_matrix, laplacian = build_complex_admittance_matrices(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        device=device,
    )
    num_nodes = int(laplacian.shape[0])
    if num_nodes == 0:
        raise ValueError("Direct harmonic solve received an empty graph.")
    if reference_node is None:
        reference_node = 0
    if not (0 <= int(reference_node) < num_nodes):
        raise ValueError(f"reference_node={reference_node} is outside [0, {num_nodes}).")

    arterial_idx = np.zeros((0,), dtype=np.int64)
    rhs_t = torch.as_tensor(source_vector_m3_s, dtype=dtype, device=device).clone()
    system_np = laplacian.detach().cpu().numpy().astype(np.complex128)
    rhs_np = rhs_t.detach().cpu().numpy().astype(np.complex128)
    if not (
        np.all(np.isfinite(system_np.real))
        and np.all(np.isfinite(system_np.imag))
        and np.all(np.isfinite(rhs_np.real))
        and np.all(np.isfinite(rhs_np.imag))
    ):
        raise ValueError("non-finite entries detected in direct harmonic solve")

    matrix_diag = full_complex_matrix_diagnostics(system_np, rcond=singular_rcond)
    ones_residual_norm = float(np.linalg.norm(system_np @ np.ones(num_nodes, dtype=np.complex128)))
    harmonic_matrix_full_rank = bool(matrix_diag.get("is_full_column_rank", False))
    gauge_applied = False
    gauge_reason = ""
    backend = "torch_linalg_solve_direct"

    if harmonic_matrix_full_rank:
        try:
            pressure_t = torch.linalg.solve(laplacian, rhs_t)
            pressure = pressure_t.detach().cpu().numpy().astype(np.complex128)
        except RuntimeError:
            harmonic_matrix_full_rank = False
            gauge_applied = True
            gauge_reason = "rank_deficient_harmonic_matrix"
            backend = "numpy_linalg_solve_reference_gauge_fallback"
    if not harmonic_matrix_full_rank:
        gauge_applied = True
        if not gauge_reason:
            gauge_reason = "rank_deficient_harmonic_matrix"
        backend = "numpy_linalg_solve_reference_gauge_fallback"
        gauged_system = system_np.copy()
        gauged_rhs = rhs_np.copy()
        gauged_system[int(reference_node), :] = 0.0
        gauged_system[int(reference_node), int(reference_node)] = 1.0
        gauged_rhs[int(reference_node)] = 0.0
        try:
            pressure = np.linalg.solve(gauged_system, gauged_rhs).astype(np.complex128, copy=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Direct harmonic solve found a rank-deficient matrix and the documented reference-gauge "
                f"fallback also failed: {exc}"
            ) from exc

    residual = system_np @ pressure - rhs_np
    direct_system_residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs_np))
    relative_direct_residual = float(direct_system_residual_norm / max(rhs_norm, 1.0e-30))
    max_abs_kirchhoff_residual_m3_s = float(np.max(np.abs(residual))) if residual.size else 0.0
    rms_kirchhoff_residual_m3_s = complex_rmse(residual)

    relative_pressure_difference = float("nan")
    if compare_with_numpy and harmonic_matrix_full_rank:
        pressure_numpy = np.linalg.solve(system_np, rhs_np)
        relative_pressure_difference = float(
            np.linalg.norm(pressure - pressure_numpy) / max(float(np.linalg.norm(pressure_numpy)), 1.0e-30)
        )

    solver_info: dict[str, object] = {
        "backend": backend,
        "device": str(device),
        "success": bool(np.all(np.isfinite(pressure.real)) and np.all(np.isfinite(pressure.imag))),
        "converged": True,
        "message": "direct harmonic solve completed",
        "phase_iterations_used": float("nan"),
        "phase_iteration_relative_change": float("nan"),
        "istop": float("nan"),
        "iterations": float("nan"),
        "r1norm": direct_system_residual_norm,
        "r2norm": direct_system_residual_norm,
        "matrix_rank": float(matrix_diag.get("matrix_rank", float("nan"))),
        "matrix_diagnostics": matrix_diag,
        "reference_node": int(reference_node),
        "harmonic_matrix_full_rank": harmonic_matrix_full_rank,
        "harmonic_matrix_rank": int(matrix_diag.get("matrix_rank", 0)),
        "harmonic_matrix_condition_number": float(matrix_diag.get("condition_number", float("nan"))),
        "harmonic_matrix_ones_residual_norm": ones_residual_norm,
        "gauge_applied": gauge_applied,
        "gauge_reason": gauge_reason,
        "direct_system_residual_norm": direct_system_residual_norm,
        "max_abs_kirchhoff_residual_m3_s": max_abs_kirchhoff_residual_m3_s,
        "rms_kirchhoff_residual_m3_s": rms_kirchhoff_residual_m3_s,
        "relative_direct_residual": relative_direct_residual,
        "relative_pressure_difference": relative_pressure_difference,
    }
    return _finalize_source_driven_solution(
        pressure=pressure,
        flow_matrix=flow_matrix,
        laplacian=laplacian,
        source_vector_m3_s=source_vector_m3_s,
        arterial_idx=arterial_idx,
        pressure_constraint_type="none",
        phase_offset=0.0,
        solver_info=solver_info,
    )


def solve_complex_pressure_with_nodal_injections_and_flow(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    q_obs_m3_s: np.ndarray,
    valid_edge_mask: np.ndarray,
    flow_row_weights: np.ndarray | None,
    source_vector_m3_s: np.ndarray,
    arterial_idx: np.ndarray,
    lambda_q: float,
    lambda_k: float,
    lambda_b: float,
    pressure_constraint_type: str,
    phase_offset: float,
    max_iterations: int,
    tol: float,
    device: torch.device,
    lstsq_backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, object]]:
    if pressure_constraint_type not in {
        "none",
        "pure_direct",
        "",
        "common_arterial_phase",
        "equal_phase",
    }:
        raise ValueError(
            "The corrected source-driven flow solver supports only no pressure "
            "constraint or a common arterial phase constraint."
        )
    if abs(float(phase_offset)) > 1.0e-15:
        raise ValueError(
            "phase_offset must be zero; an absolute phase offset is not part "
            "of the equal-phase boundary condition."
        )

    dtype = torch.complex128
    flow_matrix, laplacian = build_complex_admittance_matrices(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        device=device,
    )
    source_t = torch.as_tensor(source_vector_m3_s, dtype=dtype, device=device)
    q_obs_t = torch.as_tensor(q_obs_m3_s, dtype=dtype, device=device)
    flow_row_weights_t = (
        torch.as_tensor(flow_row_weights, dtype=torch.float64, device=device)
        if flow_row_weights is not None
        else None
    )
    flow_rows = np.flatnonzero(np.asarray(valid_edge_mask, dtype=bool))
    use_phase_constraint = pressure_constraint_type in {
        "common_arterial_phase",
        "equal_phase",
    }

    pressure, solver_info, _ = _solve_source_driven_with_phase_only_constraint(
        laplacian=laplacian,
        flow_matrix=flow_matrix,
        source_t=source_t,
        q_obs_t=q_obs_t,
        flow_rows=flow_rows,
        flow_row_weights_t=flow_row_weights_t,
        arterial_idx=np.asarray(arterial_idx, dtype=np.int64),
        lambda_k=float(lambda_k),
        lambda_q=float(lambda_q),
        lambda_b=float(lambda_b) if use_phase_constraint else 0.0,
        max_iterations=int(max_iterations),
        tol=float(tol),
        device=device,
        lstsq_backend=str(lstsq_backend),
    )
    return _finalize_source_driven_solution(
        pressure=pressure,
        flow_matrix=flow_matrix,
        laplacian=laplacian,
        source_vector_m3_s=source_vector_m3_s,
        arterial_idx=np.asarray(arterial_idx, dtype=np.int64),
        pressure_constraint_type=pressure_constraint_type,
        phase_offset=0.0,
        solver_info=solver_info,
    )


def _build_reduced_complex_system(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    q_obs_nl_s: np.ndarray,
    valid_edge_mask: np.ndarray,
    arterial_idx: np.ndarray,
    venous_idx: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    num_nodes = int(np.max(edge_index)) + 1 if edge_index.size else 0
    dtype = torch.complex128
    flow_matrix, laplacian = build_complex_admittance_matrices(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        device=device,
    )
    source_t = torch.as_tensor(edge_index[0], dtype=torch.long, device=device)
    target_t = torch.as_tensor(edge_index[1], dtype=torch.long, device=device)
    rows_t = torch.arange(int(source_t.numel()), dtype=torch.long, device=device)
    q_obs_t = torch.as_tensor(q_obs_nl_s, dtype=dtype, device=device)

    gauge_node = int(venous_idx[0]) if venous_idx.size else 0
    unknown_nodes = torch.nonzero(torch.arange(num_nodes, device=device) != gauge_node, as_tuple=False).flatten()
    node_to_col = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    node_to_col[unknown_nodes] = torch.arange(int(unknown_nodes.numel()), dtype=torch.long, device=device)

    boundary_mask = np.zeros(num_nodes, dtype=bool)
    boundary_mask[arterial_idx] = True
    boundary_mask[venous_idx] = True
    internal_idx = np.flatnonzero(~boundary_mask)
    internal_idx_t = torch.as_tensor(internal_idx, dtype=torch.long, device=device)
    internal_matrix = (
        laplacian.index_select(0, internal_idx_t).index_select(1, unknown_nodes)
        if internal_idx_t.numel()
        else torch.zeros((0, int(unknown_nodes.numel())), dtype=dtype, device=device)
    )

    reduced_flow_matrix = torch.zeros((int(source_t.numel()), int(unknown_nodes.numel())), dtype=dtype, device=device)
    cols_src = node_to_col.index_select(0, source_t)
    cols_dst = node_to_col.index_select(0, target_t)
    src_keep = cols_src >= 0
    dst_keep = cols_dst >= 0
    admittance_diag_t = torch.as_tensor(admittance_diag, dtype=dtype, device=device)
    admittance_off_t = torch.as_tensor(admittance_off, dtype=dtype, device=device)
    if bool(torch.any(src_keep)):
        reduced_flow_matrix.index_put_((rows_t[src_keep], cols_src[src_keep]), admittance_diag_t[src_keep], accumulate=True)
    if bool(torch.any(dst_keep)):
        reduced_flow_matrix.index_put_((rows_t[dst_keep], cols_dst[dst_keep]), admittance_off_t[dst_keep], accumulate=True)

    boundary_matrix = torch.zeros((len(arterial_idx), int(unknown_nodes.numel())), dtype=dtype, device=device)
    if arterial_idx.size:
        arterial_idx_t = torch.as_tensor(arterial_idx, dtype=torch.long, device=device)
        arterial_cols = node_to_col.index_select(0, arterial_idx_t)
        keep = arterial_cols >= 0
        if bool(torch.any(keep)):
            boundary_matrix[
                torch.arange(len(arterial_idx), dtype=torch.long, device=device)[keep],
                arterial_cols[keep],
            ] = torch.ones(int(keep.sum().item()), dtype=dtype, device=device)

    flow_rows = np.flatnonzero(valid_edge_mask)
    return {
        "num_nodes": num_nodes,
        "dtype": dtype,
        "flow_matrix": flow_matrix,
        "laplacian": laplacian,
        "q_obs_t": q_obs_t,
        "gauge_node": gauge_node,
        "unknown_nodes": unknown_nodes,
        "internal_idx": internal_idx,
        "internal_matrix": internal_matrix,
        "reduced_flow_matrix": reduced_flow_matrix,
        "boundary_matrix": boundary_matrix,
        "flow_rows": flow_rows,
    }


def _finalize_complex_solution(
    pressure: np.ndarray,
    system: dict[str, object],
    arterial_idx: np.ndarray,
    pressure_constraint_type: str,
    phase_offset: float,
    solver_info: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    dtype = torch.complex128
    device = cast(torch.Tensor, system["flow_matrix"]).device
    pressure_t = torch.as_tensor(pressure, dtype=dtype, device=device)
    flow_matrix = cast(torch.Tensor, system["flow_matrix"])
    laplacian = cast(torch.Tensor, system["laplacian"])
    flow_pred = (flow_matrix @ pressure_t).detach().cpu().numpy().astype(np.complex128)
    nodal_residual = (laplacian @ pressure_t).detach().cpu().numpy().astype(np.complex128)
    final_targets = (
        pressure_targets(
            pressure,
            arterial_idx,
            pressure_constraint_type=str(pressure_constraint_type),
            phase_offset=float(phase_offset),
        )
        if arterial_idx.size
        else np.zeros((0,), dtype=np.complex128)
    )
    finite_pressure = bool(np.all(np.isfinite(pressure.real)) and np.all(np.isfinite(pressure.imag)))
    finite_flow = bool(np.all(np.isfinite(flow_pred.real)) and np.all(np.isfinite(flow_pred.imag)))
    solver_info["success"] = finite_pressure and finite_flow and bool(solver_info.get("success", True))
    diagnostics = {
        "flow_pred_nl_s": flow_pred,
        "nodal_residual_nl_s": nodal_residual,
        "laplacian_nl_s_per_pa": laplacian.detach().cpu().numpy().astype(np.complex128),
        "flow_matrix_nl_s_per_pa": flow_matrix.detach().cpu().numpy().astype(np.complex128),
        "boundary_targets_pa": final_targets,
        "boundary_phase_target_residual_deg": phase_target_residual_deg(pressure, final_targets, arterial_idx),
        "lsqr_info": solver_info,
        "internal_idx": np.asarray(system["internal_idx"], dtype=np.int64),
    }
    return pressure, diagnostics


def solve_complex_pressure(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    q_obs_nl_s: np.ndarray,
    valid_edge_mask: np.ndarray,
    arterial_idx: np.ndarray,
    venous_idx: np.ndarray,
    lambda_q: float,
    lambda_k: float,
    lambda_b: float,
    pressure_constraint_type: str,
    phase_offset: float,
    max_iterations: int,
    tol: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    system = _build_reduced_complex_system(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        q_obs_nl_s=q_obs_nl_s,
        valid_edge_mask=valid_edge_mask,
        arterial_idx=arterial_idx,
        venous_idx=venous_idx,
        device=device,
    )
    num_nodes = int(system["num_nodes"])
    dtype = cast(torch.dtype, system["dtype"])
    internal_matrix = cast(torch.Tensor, system["internal_matrix"])
    reduced_flow_matrix = cast(torch.Tensor, system["reduced_flow_matrix"])
    boundary_matrix = cast(torch.Tensor, system["boundary_matrix"])
    q_obs_t = cast(torch.Tensor, system["q_obs_t"])
    unknown_nodes = cast(torch.Tensor, system["unknown_nodes"])
    flow_rows = cast(np.ndarray, system["flow_rows"])
    pressure = np.zeros(num_nodes, dtype=np.complex128)
    targets = np.zeros(len(arterial_idx), dtype=np.complex128)

    solver_info: dict[str, object] = {
        "backend": "torch_linalg_lstsq",
        "device": str(device),
        "istop": float("nan"),
        "iterations": float("nan"),
        "r1norm": float("nan"),
        "r2norm": float("nan"),
        "phase_iterations_used": float("nan"),
        "phase_iteration_relative_change": float("nan"),
        "converged": False,
        "success": False,
        "message": "solver did not run",
    }

    delta = float("nan")
    lstsq_exception: Exception | None = None
    for iteration in range(max(1, int(max_iterations))):
        if arterial_idx.size:
            targets = pressure_targets(
                pressure,
                arterial_idx,
                pressure_constraint_type=str(pressure_constraint_type),
                phase_offset=float(phase_offset),
            )
        blocks: list[torch.Tensor] = []
        rhs_blocks: list[torch.Tensor] = []
        if lambda_k > 0.0 and internal_matrix.shape[0] > 0:
            nonzero = torch.abs(internal_matrix) > 0.0
            laplacian_scale = (
                torch.median(torch.abs(internal_matrix[nonzero])).clamp_min(1.0e-30)
                if bool(torch.any(nonzero))
                else torch.tensor(1.0, dtype=torch.float64, device=device)
            )
            blocks.append(math.sqrt(lambda_k) * (internal_matrix / laplacian_scale.to(dtype)))
            rhs_blocks.append(torch.zeros(internal_matrix.shape[0], dtype=dtype, device=device))
        if lambda_q > 0.0 and flow_rows.size > 0:
            flow_rows_t = torch.as_tensor(flow_rows, dtype=torch.long, device=device)
            flow_block = reduced_flow_matrix.index_select(0, flow_rows_t)
            nonzero = torch.abs(flow_block) > 0.0
            flow_scale = (
                torch.median(torch.abs(flow_block[nonzero])).clamp_min(1.0e-30)
                if bool(torch.any(nonzero))
                else torch.tensor(1.0, dtype=torch.float64, device=device)
            )
            blocks.append(math.sqrt(lambda_q) * (flow_block / flow_scale.to(dtype)))
            rhs_blocks.append(math.sqrt(lambda_q) * (q_obs_t.index_select(0, flow_rows_t) / flow_scale.to(dtype)))
        if lambda_b > 0.0 and arterial_idx.size > 0:
            blocks.append(math.sqrt(lambda_b) * boundary_matrix)
            rhs_blocks.append(math.sqrt(lambda_b) * torch.as_tensor(targets, dtype=dtype, device=device))
        if not blocks:
            raise ValueError("At least one solver block must be active.")

        stacked_matrix = torch.cat(blocks, dim=0)
        stacked_rhs = torch.cat(rhs_blocks, dim=0)
        try:
            lstsq_result = torch.linalg.lstsq(stacked_matrix, stacked_rhs)
        except RuntimeError as exc:
            lstsq_exception = exc
            solver_info.update(
                {
                    "phase_iterations_used": iteration + 1,
                    "converged": False,
                    "success": False,
                    "message": f"torch.linalg.lstsq failed: {exc}",
                }
            )
            break
        reduced_pressure = lstsq_result.solution
        residual = stacked_matrix @ reduced_pressure - stacked_rhs
        solver_info.update(
            {
                "r1norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
                "r2norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
            }
        )
        new_pressure_t = torch.zeros(num_nodes, dtype=dtype, device=device)
        new_pressure_t.index_copy_(0, unknown_nodes, reduced_pressure)
        new_pressure = new_pressure_t.detach().cpu().numpy().astype(np.complex128)
        denom = max(float(np.linalg.norm(new_pressure)), 1.0e-30)
        delta = float(np.linalg.norm(new_pressure - pressure) / denom)
        pressure = new_pressure
        solver_info["phase_iterations_used"] = iteration + 1
        solver_info["phase_iteration_relative_change"] = delta
        if delta <= float(tol):
            solver_info["converged"] = True
            solver_info["message"] = "phase iteration converged"
            break

        if not bool(solver_info.get("converged", False)) and lstsq_exception is None:
            solver_info["message"] = "phase iteration reached max iterations without meeting tolerance"
    solver_info["success"] = bool(solver_info.get("converged", False)) and lstsq_exception is None
    return _finalize_complex_solution(
        pressure=pressure,
        system=system,
        arterial_idx=arterial_idx,
        pressure_constraint_type=pressure_constraint_type,
        phase_offset=phase_offset,
        solver_info=solver_info,
    )


def solve_complex_pressure_hard(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    q_obs_nl_s: np.ndarray,
    valid_edge_mask: np.ndarray,
    arterial_idx: np.ndarray,
    venous_idx: np.ndarray,
    lambda_q: float,
    lambda_k: float,
    pressure_constraint_type: str,
    phase_offset: float,
    max_iterations: int,
    tol: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    system = _build_reduced_complex_system(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        q_obs_nl_s=q_obs_nl_s,
        valid_edge_mask=valid_edge_mask,
        arterial_idx=arterial_idx,
        venous_idx=venous_idx,
        device=device,
    )
    num_nodes = int(system["num_nodes"])
    dtype = cast(torch.dtype, system["dtype"])
    internal_matrix = cast(torch.Tensor, system["internal_matrix"])
    reduced_flow_matrix = cast(torch.Tensor, system["reduced_flow_matrix"])
    boundary_matrix = cast(torch.Tensor, system["boundary_matrix"])
    q_obs_t = cast(torch.Tensor, system["q_obs_t"])
    unknown_nodes = cast(torch.Tensor, system["unknown_nodes"])
    flow_rows = cast(np.ndarray, system["flow_rows"])

    pressure = np.zeros(num_nodes, dtype=np.complex128)
    solver_info: dict[str, object] = {
        "backend": "torch_linalg_lstsq_kkt",
        "device": str(device),
        "istop": float("nan"),
        "iterations": float("nan"),
        "r1norm": float("nan"),
        "r2norm": float("nan"),
        "phase_iterations_used": float("nan"),
        "phase_iteration_relative_change": float("nan"),
        "converged": False,
        "success": False,
        "message": "solver did not run",
    }

    delta = float("nan")
    for iteration in range(max(1, int(max_iterations))):
        targets = pressure_targets(
            pressure,
            arterial_idx,
            pressure_constraint_type=str(pressure_constraint_type),
            phase_offset=float(phase_offset),
        ) if arterial_idx.size else np.zeros((0,), dtype=np.complex128)

        lhs_blocks: list[torch.Tensor] = []
        rhs_blocks: list[torch.Tensor] = []
        if lambda_k > 0.0 and internal_matrix.shape[0] > 0:
            nonzero = torch.abs(internal_matrix) > 0.0
            laplacian_scale = (
                torch.median(torch.abs(internal_matrix[nonzero])).clamp_min(1.0e-30)
                if bool(torch.any(nonzero))
                else torch.tensor(1.0, dtype=torch.float64, device=device)
            )
            lhs_blocks.append(math.sqrt(lambda_k) * (internal_matrix / laplacian_scale.to(dtype)))
            rhs_blocks.append(torch.zeros(internal_matrix.shape[0], dtype=dtype, device=device))
        if lambda_q > 0.0 and flow_rows.size > 0:
            flow_rows_t = torch.as_tensor(flow_rows, dtype=torch.long, device=device)
            flow_block = reduced_flow_matrix.index_select(0, flow_rows_t)
            nonzero = torch.abs(flow_block) > 0.0
            flow_scale = (
                torch.median(torch.abs(flow_block[nonzero])).clamp_min(1.0e-30)
                if bool(torch.any(nonzero))
                else torch.tensor(1.0, dtype=torch.float64, device=device)
            )
            lhs_blocks.append(math.sqrt(lambda_q) * (flow_block / flow_scale.to(dtype)))
            rhs_blocks.append(math.sqrt(lambda_q) * (q_obs_t.index_select(0, flow_rows_t) / flow_scale.to(dtype)))
        if not lhs_blocks:
            raise ValueError("At least one of lambda_q or lambda_k must be positive for the hard solver.")

        design = torch.cat(lhs_blocks, dim=0)
        rhs = torch.cat(rhs_blocks, dim=0)
        constraints = boundary_matrix
        constraint_rhs = torch.as_tensor(targets, dtype=dtype, device=device)
        n_unknowns = int(unknown_nodes.numel())
        n_constraints = int(constraints.shape[0])
        normal = design.conj().transpose(0, 1) @ design
        kkt = torch.zeros((n_unknowns + n_constraints, n_unknowns + n_constraints), dtype=dtype, device=device)
        kkt[:n_unknowns, :n_unknowns] = normal
        if n_constraints:
            kkt[:n_unknowns, n_unknowns:] = constraints.conj().transpose(0, 1)
            kkt[n_unknowns:, :n_unknowns] = constraints
        kkt_rhs = torch.zeros((n_unknowns + n_constraints,), dtype=dtype, device=device)
        kkt_rhs[:n_unknowns] = design.conj().transpose(0, 1) @ rhs
        if n_constraints:
            kkt_rhs[n_unknowns:] = constraint_rhs

        lstsq_result = torch.linalg.lstsq(kkt, kkt_rhs)
        reduced_pressure = lstsq_result.solution[:n_unknowns]
        residual = design @ reduced_pressure - rhs
        new_pressure_t = torch.zeros(num_nodes, dtype=dtype, device=device)
        new_pressure_t.index_copy_(0, unknown_nodes, reduced_pressure)
        new_pressure = new_pressure_t.detach().cpu().numpy().astype(np.complex128)
        denom = max(float(np.linalg.norm(new_pressure)), 1.0e-30)
        delta = float(np.linalg.norm(new_pressure - pressure) / denom)
        pressure = new_pressure
        solver_info.update(
            {
                "r1norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
                "r2norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
                "phase_iterations_used": iteration + 1,
                "phase_iteration_relative_change": delta,
            }
        )
        if pressure_constraint_type in {"equal_arterial_pressure_phasors", "equal_phasor"} or delta <= float(tol):
            solver_info["converged"] = True
            solver_info["message"] = "hard constraint solve converged"
            break
    if not bool(solver_info.get("converged", False)):
        solver_info["message"] = "hard constraint phase iteration reached max iterations"
    solver_info["success"] = bool(solver_info.get("converged", False))
    return _finalize_complex_solution(
        pressure=pressure,
        system=system,
        arterial_idx=arterial_idx,
        pressure_constraint_type=pressure_constraint_type,
        phase_offset=phase_offset,
        solver_info=solver_info,
    )


def solve_complex_pressure_exact(
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    edge_index: np.ndarray,
    q_obs_nl_s: np.ndarray,
    valid_edge_mask: np.ndarray,
    arterial_idx: np.ndarray,
    venous_idx: np.ndarray,
    lambda_q: float,
    lambda_k: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    system = _build_reduced_complex_system(
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        edge_index=edge_index,
        q_obs_nl_s=q_obs_nl_s,
        valid_edge_mask=valid_edge_mask,
        arterial_idx=arterial_idx,
        venous_idx=venous_idx,
        device=device,
    )
    num_nodes = int(system["num_nodes"])
    dtype = cast(torch.dtype, system["dtype"])
    internal_matrix = cast(torch.Tensor, system["internal_matrix"])
    reduced_flow_matrix = cast(torch.Tensor, system["reduced_flow_matrix"])
    q_obs_t = cast(torch.Tensor, system["q_obs_t"])
    unknown_nodes = cast(torch.Tensor, system["unknown_nodes"])
    flow_rows = cast(np.ndarray, system["flow_rows"])

    blocks: list[torch.Tensor] = []
    rhs_blocks: list[torch.Tensor] = []
    if lambda_k > 0.0 and internal_matrix.shape[0] > 0:
        nonzero = torch.abs(internal_matrix) > 0.0
        laplacian_scale = (
            torch.median(torch.abs(internal_matrix[nonzero])).clamp_min(1.0e-30)
            if bool(torch.any(nonzero))
            else torch.tensor(1.0, dtype=torch.float64, device=device)
        )
        blocks.append(math.sqrt(lambda_k) * (internal_matrix / laplacian_scale.to(dtype)))
        rhs_blocks.append(torch.zeros(internal_matrix.shape[0], dtype=dtype, device=device))
    if lambda_q > 0.0 and flow_rows.size > 0:
        flow_rows_t = torch.as_tensor(flow_rows, dtype=torch.long, device=device)
        flow_block = reduced_flow_matrix.index_select(0, flow_rows_t)
        nonzero = torch.abs(flow_block) > 0.0
        flow_scale = (
            torch.median(torch.abs(flow_block[nonzero])).clamp_min(1.0e-30)
            if bool(torch.any(nonzero))
            else torch.tensor(1.0, dtype=torch.float64, device=device)
        )
        blocks.append(math.sqrt(lambda_q) * (flow_block / flow_scale.to(dtype)))
        rhs_blocks.append(math.sqrt(lambda_q) * (q_obs_t.index_select(0, flow_rows_t) / flow_scale.to(dtype)))
    if not blocks:
        raise ValueError("At least one of lambda_q or lambda_k must be positive for the exact solver.")
    stacked_matrix = torch.cat(blocks, dim=0)
    stacked_rhs = torch.cat(rhs_blocks, dim=0)
    lstsq_result = torch.linalg.lstsq(stacked_matrix, stacked_rhs)
    reduced_pressure = lstsq_result.solution
    residual = stacked_matrix @ reduced_pressure - stacked_rhs
    pressure_t = torch.zeros(num_nodes, dtype=dtype, device=device)
    pressure_t.index_copy_(0, unknown_nodes, reduced_pressure)
    pressure = pressure_t.detach().cpu().numpy().astype(np.complex128)
    solver_info = {
        "backend": "torch_linalg_lstsq",
        "device": str(device),
        "istop": float("nan"),
        "iterations": float("nan"),
        "r1norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
        "r2norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
        "phase_iterations_used": float("nan"),
        "phase_iteration_relative_change": float("nan"),
        "converged": True,
        "success": True,
        "message": "unconstrained exact benchmark",
    }
    return _finalize_complex_solution(
        pressure=pressure,
        system=system,
        arterial_idx=arterial_idx,
        pressure_constraint_type="equal_phase",
        phase_offset=0.0,
        solver_info=solver_info,
    )


def fixed_transmission_line_admittance(
    radius_m: float,
    length_m: float,
    omega_n: float,
    viscosity_pa_s: float,
    distensibility_d: float,
) -> tuple[complex, complex, complex]:
    if (
        not math.isfinite(float(radius_m))
        or not math.isfinite(float(length_m))
        or float(radius_m) <= 0.0
        or float(length_m) <= 0.0
    ):
        return 0.0j, 0.0j, 0.0j
    r_e = 8.0 * float(viscosity_pa_s) / (math.pi * float(radius_m) ** 4)
    c_e = math.pi * float(radius_m) ** 2 * float(distensibility_d)
    if r_e <= 0.0 or not math.isfinite(r_e) or c_e < 0.0 or not math.isfinite(c_e):
        return 0.0j, 0.0j, 0.0j
    if abs(omega_n) < 1.0e-12:
        conductance = 1.0 / max(r_e * float(length_m), 1.0e-30)
        return complex(conductance), complex(-conductance), complex(0.0)

    k_e = np.sqrt(1j * float(omega_n) * r_e * c_e)
    kL = k_e * float(length_m)
    prefactor = k_e / r_e
    if not np.isfinite(kL.real) or not np.isfinite(kL.imag) or not np.isfinite(prefactor.real) or not np.isfinite(prefactor.imag):
        return 0.0j, 0.0j, 0.0j
    if abs(kL) < 1.0e-6:
        coth_kL = 1.0 / kL + kL / 3.0
        csch_kL = 1.0 / kL - kL / 6.0
    elif abs(np.real(kL)) > 500.0:
        coth_kL = 1.0 + 0.0j
        csch_kL = 0.0 + 0.0j
    else:
        sinh_kL = np.sinh(kL)
        cosh_kL = np.cosh(kL)
        if (
            not np.isfinite(sinh_kL.real)
            or not np.isfinite(sinh_kL.imag)
            or not np.isfinite(cosh_kL.real)
            or not np.isfinite(cosh_kL.imag)
            or abs(sinh_kL) < 1.0e-30
        ):
            coth_kL = 1.0 + 0.0j
            csch_kL = 0.0 + 0.0j
        else:
            coth_kL = cosh_kL / sinh_kL
            csch_kL = 1.0 / sinh_kL
    y_s = prefactor * coth_kL
    y_t = prefactor * csch_kL
    return complex(y_s), complex(-y_t), complex(kL)


def taylor_transmission_line_admittance(
    radius_m: float,
    length_m: float,
    omega_n: float,
    viscosity_pa_s: float,
    distensibility_d: float,
    conductance_scale: float = 1.0,
) -> tuple[complex, complex, complex, float, float]:
    """First-order short-edge admittance with optional resistance correction.

    conductance_scale = exp(delta) implies r_star = r / conductance_scale.
    The first-order compliance terms are independent of r_star, while the
    reported kL must use r_star.
    """
    if (
        not math.isfinite(float(radius_m))
        or not math.isfinite(float(length_m))
        or not math.isfinite(float(conductance_scale))
        or float(radius_m) <= 0.0
        or float(length_m) <= 0.0
        or float(conductance_scale) <= 0.0
    ):
        return 0.0j, 0.0j, 0.0j, float("nan"), float("nan")

    r_e = 8.0 * float(viscosity_pa_s) / (math.pi * float(radius_m) ** 4)
    c_e = math.pi * float(radius_m) ** 2 * float(distensibility_d)
    if r_e <= 0.0 or not math.isfinite(r_e) or c_e < 0.0 or not math.isfinite(c_e):
        return 0.0j, 0.0j, 0.0j, float("nan"), float("nan")

    conductance = 1.0 / max(r_e * float(length_m), 1.0e-30)
    conductance_scale = float(conductance_scale)
    g_e_star = conductance_scale * conductance
    r_e_star = r_e / conductance_scale

    if abs(omega_n) < 1.0e-12:
        return (
            complex(g_e_star),
            complex(-g_e_star),
            complex(0.0),
            float(conductance),
            float(c_e),
        )

    k_e_star = np.sqrt(1j * float(omega_n) * r_e_star * c_e)
    kL_star = k_e_star * float(length_m)

    # Since g_star * (k_star L)^2 = i omega c L, the first-order compliance
    # corrections do not depend on the resistance correction.
    imag_term = 1j * float(omega_n) * c_e * float(length_m)
    y_s = complex(g_e_star) + imag_term / 3.0
    negative_y_t = complex(-g_e_star) + imag_term / 6.0

    return (
        complex(y_s),
        complex(negative_y_t),
        complex(kL_star),
        float(conductance),
        float(c_e),
    )
