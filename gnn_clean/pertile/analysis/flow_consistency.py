"""
Flow consistency analysis using Kirchhoff's law.

At each junction node, conservation of mass requires:
    Σ Q_in = Σ Q_out

This module computes residuals at each node and provides metrics
for how well the measured flows satisfy conservation.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gammaln
from typing import Dict, Tuple, Optional, Set, List, Any
from dataclasses import dataclass

from .config import MIN_PATH_LENGTH_PX, PX_SIZE_UM


def compute_variance_features(
    length: np.ndarray,
    chi2: np.ndarray,
    snr: np.ndarray,
    L_min: float = 30.0,
    snr_min: float = 0.1
) -> np.ndarray:
    """
    Compute basis functions for learned variance model with theoretical justification.

    Model: log(σᵢ) = β₀ + β₁f₁(length) + β₂f₂(χ²) + β₃f₃(SNR)

    With the 0.5 scaling baked into all basis functions, expected coefficients:
        β₁ ≈ 1.0  (from 1/√N averaging, with 0.5 factor in basis)
        β₂ ≈ 1.0  (from χ² variance scaling, with 0.5 factor in basis)
        β₃ ≈ 1.0  (from SNR definition, with 0.5 factor in basis)

    Parameters
    ----------
    length : array
        Vessel lengths in pixels
    chi2 : array
        Reduced chi-squared from sinusoidal fit
    snr : array
        Signal-to-noise ratio (linear, not dB)
    L_min : float
        Minimum length for clipping (below this, physics changes - entrance effects)
    snr_min : float
        Minimum SNR for clipping (avoid log(0))

    Returns
    -------
    features : array of shape (n, 3)
        Columns: [f_length, f_chi2, f_snr]
    """
    # Length: averaging over N ∝ L samples → σ ∝ 1/√L → log(σ) ∝ -0.5 log(L)
    # Include 0.5 factor so expected β₁ ≈ 1.0
    # Clipped to avoid log(0) and entrance region effects
    f_length = -0.5 * np.log(np.maximum(length, L_min))

    # Chi2: variance scaling when σ_noise was mis-estimated
    # σ_true = σ_estimated × √χ² → log(σ) ∝ 0.5 log(χ²)
    # NO CLIPPING AT 1: χ² < 1 means σ_noise was overestimated (trust more)
    #                   χ² > 1 means σ_noise was underestimated (trust less)
    # χ² = 1.0 → f_chi2 = 0 (baseline, σ_noise was correct)
    # χ² = 0.25 → f_chi2 = -0.69 → σ scaled down (was overestimated)
    # χ² = 4.0 → f_chi2 = +0.69 → σ scaled up (was underestimated)
    chi2_safe = np.maximum(chi2, 1e-6)  # Only avoid log(0), allow χ² < 1
    f_chi2 = 0.5 * np.log(chi2_safe)

    # SNR: relative uncertainty scales as 1/√SNR
    # (SNR here is R²/(1-R²), so variance ∝ 1/SNR, std ∝ 1/√SNR)
    # log(σ) ∝ -0.5 log(SNR)
    # NO CLIPPING AT 1: SNR < 1 means poor fit (higher uncertainty)
    snr_safe = np.maximum(snr, snr_min)  # Only avoid log(0)
    f_snr = -0.5 * np.log(snr_safe)

    return np.column_stack([f_length, f_chi2, f_snr])


def student_t_nll(residuals: np.ndarray, sigma: np.ndarray, nu: float) -> float:
    """
    Negative log-likelihood for Student-t distribution.

    L = Π Student-t(rᵢ | 0, σᵢ, ν)

    Parameters
    ----------
    residuals : array
        r = Q_measured - Q_predicted
    sigma : array
        Per-observation standard deviations
    nu : float
        Degrees of freedom (smaller = heavier tails, more robust)

    Returns
    -------
    nll : float
        Negative log-likelihood (to minimize)
    """
    n = len(residuals)
    z = residuals / sigma  # Standardized residuals

    # Student-t log-pdf:
    # log p(z) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - (ν+1)/2 log(1 + z²/ν)
    # Plus - log(σ) for the scale
    nll = (
        -n * gammaln((nu + 1) / 2)
        + n * gammaln(nu / 2)
        + n * 0.5 * np.log(nu * np.pi)
        + np.sum(np.log(sigma))
        + (nu + 1) / 2 * np.sum(np.log(1 + z**2 / nu))
    )
    return nll


def student_t_weights(residuals: np.ndarray, sigma: np.ndarray, nu: float) -> np.ndarray:
    """
    Compute weights for weighted least squares from Student-t likelihood.

    For Student-t, the weight is: w = (ν + 1) / (ν + z²)
    where z = r/σ is the standardized residual.

    This downweights outliers more aggressively than Huber for small ν.

    Parameters
    ----------
    residuals : array
        r = Q_measured - Q_predicted
    sigma : array
        Per-observation standard deviations
    nu : float
        Degrees of freedom

    Returns
    -------
    weights : array
        Weights for weighted least squares
    """
    z = residuals / sigma
    weights = (nu + 1) / (nu + z**2)
    return weights


@dataclass
class FlowConsistencyResult:
    """Results from flow consistency analysis."""
    # Per-node residuals (flow imbalance)
    node_residuals: Dict[int, float]

    # Classification
    internal_nodes: Set[int]
    boundary_nodes: Set[int]

    # Metrics
    chi_squared: float  # Σ(residual² / variance)
    r_squared: float    # 1 - (SS_res / SS_tot)
    rmse: float         # Root mean squared residual

    # Per-node details
    node_inflow: Dict[int, float]
    node_outflow: Dict[int, float]

    # Inferred boundary flows (for boundary nodes)
    boundary_flows: Dict[int, float]  # positive = source, negative = sink

    # Predicted flows (after optimization)
    predicted_Q: Optional[Dict[Tuple[int, int], float]] = None


def compute_node_flows(
    G: nx.Graph,
    flow_attr: str = 'mean_Q_nL_s',
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Compute inflow, outflow, and residual at each node.

    Flow sign convention:
    - Q > 0: flow from u to v (edge source to edge sink)
    - Q < 0: flow from v to u

    Parameters
    ----------
    G : nx.Graph
        Graph with flow attributes on edges
    flow_attr : str
        Edge attribute containing signed flow (positive = u→v)

    Returns
    -------
    inflow, outflow, residual : dict
    """
    inflow = {n: 0.0 for n in G.nodes()}
    outflow = {n: 0.0 for n in G.nodes()}

    for u, v, data in G.edges(data=True):
        Q = data.get(flow_attr)
        if Q is None or not np.isfinite(Q):
            continue

        # Q is signed: positive means u→v, negative means v→u
        Q_mag = abs(Q)

        if Q >= 0:
            # Flow from u to v
            source, sink = u, v
        else:
            # Flow from v to u
            source, sink = v, u

        outflow[source] = outflow.get(source, 0.0) + Q_mag
        inflow[sink] = inflow.get(sink, 0.0) + Q_mag

    # Compute residuals
    residual = {}
    for n in G.nodes():
        residual[n] = inflow.get(n, 0.0) - outflow.get(n, 0.0)

    return inflow, outflow, residual


def identify_boundary_nodes(
    G: nx.Graph,
    residual: Dict[int, float],
    threshold_fraction: float = 0.1,
    manual_boundary: Optional[Set[int]] = None,
) -> Tuple[Set[int], Set[int]]:
    """
    Identify boundary vs internal nodes.

    Parameters
    ----------
    G : nx.Graph
        The vessel graph
    residual : dict
        Flow residual at each node
    threshold_fraction : float
        Nodes with |residual| > threshold_fraction * max_flow are boundary
    manual_boundary : set, optional
        Manually specified boundary nodes

    Returns
    -------
    internal_nodes, boundary_nodes : set
    """
    if manual_boundary is not None:
        boundary_nodes = set(manual_boundary)
        internal_nodes = set(G.nodes()) - boundary_nodes
        return internal_nodes, boundary_nodes

    # Auto-detect: degree-1 nodes are always boundary
    boundary_nodes = set()
    for n in G.nodes():
        if G.degree(n) == 1:
            boundary_nodes.add(n)

    # Also add nodes with large residuals
    if residual:
        max_residual = max(abs(r) for r in residual.values()) if residual else 0
        if max_residual > 0:
            threshold = threshold_fraction * max_residual
            for n, r in residual.items():
                if abs(r) > threshold:
                    boundary_nodes.add(n)

    internal_nodes = set(G.nodes()) - boundary_nodes
    return internal_nodes, boundary_nodes


def compute_flow_consistency(
    G: nx.Graph,
    flow_attr: str = 'mean_Q_nL_s',
    manual_boundary: Optional[Set[int]] = None,
) -> FlowConsistencyResult:
    """
    Compute flow consistency metrics using Kirchhoff's law.

    Parameters
    ----------
    G : nx.Graph
        Graph with flow attributes on edges
    flow_attr : str
        Edge attribute containing flow magnitude
    manual_boundary : set, optional
        Manually specified boundary nodes

    Returns
    -------
    FlowConsistencyResult
    """
    # Compute flows at each node
    inflow, outflow, residual = compute_node_flows(G, flow_attr)

    # Identify boundary vs internal nodes
    internal_nodes, boundary_nodes = identify_boundary_nodes(
        G, residual, manual_boundary=manual_boundary
    )

    # Compute metrics on internal nodes only
    internal_residuals = [residual[n] for n in internal_nodes if n in residual]

    if not internal_residuals:
        return FlowConsistencyResult(
            node_residuals=residual,
            internal_nodes=internal_nodes,
            boundary_nodes=boundary_nodes,
            chi_squared=np.nan,
            r_squared=np.nan,
            rmse=np.nan,
            node_inflow=inflow,
            node_outflow=outflow,
            boundary_flows={n: -residual.get(n, 0.0) for n in boundary_nodes},
        )

    internal_residuals = np.array(internal_residuals)

    # RMSE
    rmse = np.sqrt(np.mean(internal_residuals**2))

    # Chi-squared
    mad = np.median(np.abs(internal_residuals))
    sigma = mad * 1.4826 if mad > 0 else 1.0
    chi_squared = np.sum((internal_residuals / sigma)**2)

    # R-squared
    total_flows = [inflow.get(n, 0) + outflow.get(n, 0) for n in internal_nodes]
    ss_tot = np.var(total_flows) * len(total_flows) if total_flows else 1.0
    ss_res = np.sum(internal_residuals**2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Boundary flows
    boundary_flows = {n: -residual.get(n, 0.0) for n in boundary_nodes}

    return FlowConsistencyResult(
        node_residuals=residual,
        internal_nodes=internal_nodes,
        boundary_nodes=boundary_nodes,
        chi_squared=chi_squared,
        r_squared=r_squared,
        rmse=rmse,
        node_inflow=inflow,
        node_outflow=outflow,
        boundary_flows=boundary_flows,
    )


def print_consistency_report(result: FlowConsistencyResult, top_n: int = 10):
    """Print a summary report of flow consistency analysis."""
    print("\n" + "=" * 70)
    print("FLOW CONSISTENCY ANALYSIS (Kirchhoff's Law)")
    print("=" * 70)

    print(f"\nNodes: {len(result.internal_nodes)} internal, {len(result.boundary_nodes)} boundary")

    print(f"\nMetrics (internal nodes):")
    print(f"  RMSE:        {result.rmse:.4f} nL/s")
    print(f"  χ²:          {result.chi_squared:.2f}")
    print(f"  R²:          {result.r_squared:.4f}")

    # Worst residuals
    sorted_residuals = sorted(
        result.node_residuals.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\nTop {top_n} nodes by |residual|:")
    print(f"  {'Node':>6}  {'Residual':>10}  {'Inflow':>10}  {'Outflow':>10}  {'Type':>10}")
    for node, res in sorted_residuals[:top_n]:
        node_type = "boundary" if node in result.boundary_nodes else "internal"
        infl = result.node_inflow.get(node, 0)
        outfl = result.node_outflow.get(node, 0)
        print(f"  {node:>6}  {res:>10.4f}  {infl:>10.4f}  {outfl:>10.4f}  {node_type:>10}")

    if result.boundary_flows:
        print(f"\nBoundary flows (+ = source, - = sink):")
        sorted_boundary = sorted(
            result.boundary_flows.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for node, flow in sorted_boundary[:top_n]:
            flow_type = "SOURCE" if flow > 0 else "SINK"
            print(f"  Node {node}: {flow:+.4f} nL/s ({flow_type})")

    print("=" * 70)


def _plot_svd_diagnostic(
    s: np.ndarray,
    col_norms: np.ndarray,
    free_boundary: List[int],
    A: np.ndarray,
    cond: float,
    effective_rank: int
) -> None:
    """
    Plot SVD diagnostic figures for understanding ill-conditioning.

    Parameters
    ----------
    s : array
        Singular values from SVD of A
    col_norms : array
        Column norms of A (one per free boundary node)
    free_boundary : list
        Node IDs for free boundary nodes (columns of A)
    A : array
        The design matrix (m equations × n unknowns)
    cond : float
        Condition number of A
    effective_rank : int
        Number of singular values > 1% of max
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'SVD Diagnostic (cond = {cond:.1e}, effective rank = {effective_rank}/{len(s)})',
                 fontsize=12, fontweight='bold')

    # Plot 1: Singular value spectrum (log scale)
    ax1 = axes[0, 0]
    ax1.semilogy(s, 'b.-', linewidth=1, markersize=4)
    ax1.axhline(y=0.01 * s[0], color='r', linestyle='--', alpha=0.7,
                label=f'1% of max (effective rank threshold)')
    ax1.axhline(y=1e-10 * s[0], color='orange', linestyle=':', alpha=0.7,
                label=f'1e-10 of max (numerical zero)')
    ax1.set_xlabel('Singular value index')
    ax1.set_ylabel('Singular value (log scale)')
    ax1.set_title('Singular Value Spectrum')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Column norms (influence of each boundary node)
    ax2 = axes[0, 1]
    invisible_threshold = 0.01 * np.max(col_norms)
    colors = ['red' if cn < invisible_threshold else 'steelblue' for cn in col_norms]
    ax2.bar(range(len(col_norms)), col_norms, color=colors, alpha=0.7)
    ax2.axhline(y=invisible_threshold, color='r', linestyle='--', alpha=0.7,
                label=f'Invisible threshold (1% of max)')
    ax2.set_xlabel('Boundary node index')
    ax2.set_ylabel('Column norm (influence on measured flows)')
    ax2.set_title('Boundary Node Influence')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add node labels if not too many
    if len(free_boundary) <= 30:
        ax2.set_xticks(range(len(free_boundary)))
        ax2.set_xticklabels([str(n) for n in free_boundary], rotation=45, fontsize=7)

    # Plot 3: Cumulative explained variance
    ax3 = axes[1, 0]
    cumulative_var = np.cumsum(s**2) / np.sum(s**2)
    ax3.plot(cumulative_var, 'g.-', linewidth=1, markersize=4)
    ax3.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='99% variance')
    ax3.axhline(y=0.999, color='orange', linestyle=':', alpha=0.7, label='99.9% variance')
    # Mark effective rank
    ax3.axvline(x=effective_rank, color='purple', linestyle='--', alpha=0.7,
                label=f'Effective rank = {effective_rank}')
    ax3.set_xlabel('Number of components')
    ax3.set_ylabel('Cumulative explained variance')
    ax3.set_title('Cumulative Variance')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.9, 1.005)

    # Plot 4: Matrix structure heatmap (if not too large)
    ax4 = axes[1, 1]
    m, n = A.shape
    if m <= 100 and n <= 100:
        im = ax4.imshow(np.abs(A), aspect='auto', cmap='Blues', interpolation='nearest')
        ax4.set_xlabel('Boundary node index')
        ax4.set_ylabel('Edge index')
        ax4.set_title(f'|A| Matrix Structure ({m}×{n})')
        plt.colorbar(im, ax=ax4, shrink=0.8)
    else:
        # Show summary statistics instead
        row_nnz = np.sum(np.abs(A) > 1e-10, axis=1)
        ax4.hist(row_nnz, bins=min(20, len(np.unique(row_nnz))),
                 color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Non-zeros per row')
        ax4.set_ylabel('Number of rows (edges)')
        ax4.set_title(f'Matrix Sparsity ({m}×{n}, too large for heatmap)')
        ax4.axvline(x=np.mean(row_nnz), color='r', linestyle='--',
                    label=f'Mean = {np.mean(row_nnz):.1f}')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def _plot_variance_model_diagnostic(
    residuals: np.ndarray,
    sigma: np.ndarray,
    beta: np.ndarray,
    lengths_px: np.ndarray,
    chi2_vals: np.ndarray,
    snr_vals: np.ndarray,
    nu: float
) -> None:
    """
    Plot diagnostics for the learned variance model.

    Parameters
    ----------
    residuals : array
        Q_measured - Q_predicted
    sigma : array
        Learned per-observation standard deviations
    beta : array
        Learned coefficients [β₀, β₁, β₂, β₃]
    lengths_px : array
        Vessel lengths in pixels
    chi2_vals : array
        Reduced chi-squared values
    snr_vals : array
        SNR values (linear scale)
    nu : float
        Student-t degrees of freedom
    """
    from scipy.stats import t as student_t

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f'Variance Model Diagnostics (β=[{beta[0]:.2f}, {beta[1]:.2f}, {beta[2]:.2f}, {beta[3]:.2f}])',
                 fontsize=12, fontweight='bold')

    standardized = residuals / sigma
    abs_residuals = np.abs(residuals)

    # Row 1: Raw feature distributions
    # Plot 1: χ² histogram
    ax1 = axes[0, 0]
    ax1.hist(chi2_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='χ²=1 (ideal)')
    ax1.axvline(x=np.median(chi2_vals), color='orange', linestyle=':', linewidth=2,
                label=f'median={np.median(chi2_vals):.2f}')
    ax1.set_xlabel('χ² (reduced)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'χ² Distribution ({100*np.mean(chi2_vals < 1):.0f}% < 1)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: SNR histogram (in dB)
    ax2 = axes[0, 1]
    snr_db = 10 * np.log10(snr_vals + 1e-10)  # Convert to dB for display
    ax2.hist(snr_db, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='0 dB (R²=0.5)')
    ax2.axvline(x=np.median(snr_db), color='orange', linestyle=':', linewidth=2,
                label=f'median={np.median(snr_db):.1f} dB')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'SNR Distribution ({100*np.mean(snr_db < 0):.0f}% < 0 dB)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Length histogram
    ax3 = axes[0, 2]
    ax3.hist(lengths_px, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=30, color='r', linestyle='--', linewidth=2, label='L_min=30')
    ax3.axvline(x=np.median(lengths_px), color='orange', linestyle=':', linewidth=2,
                label=f'median={np.median(lengths_px):.0f}')
    ax3.set_xlabel('Length (px)')
    ax3.set_ylabel('Count')
    ax3.set_title('Length Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Row 2: Model calibration
    # Plot 4: Predicted σ vs |residuals|
    ax4 = axes[1, 0]
    ax4.scatter(sigma, abs_residuals, alpha=0.5, s=20, c='steelblue')
    max_val = max(sigma.max(), abs_residuals.max()) if sigma.max() > 0 else 1
    ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='y = x (ideal)')
    ax4.set_xlabel('Predicted σ')
    ax4.set_ylabel('|Residual|')
    ax4.set_title('Calibration: σ vs |r|')
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, None)
    ax4.set_ylim(0, None)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Standardized residuals distribution
    ax5 = axes[1, 1]
    # Clip for visualization (handle σ→0 case)
    std_clipped = np.clip(standardized, -10, 10)
    ax5.hist(std_clipped, bins=50, density=True, alpha=0.7, color='steelblue',
             edgecolor='black', label='Observed')
    x = np.linspace(-5, 5, 100)
    ax5.plot(x, student_t.pdf(x, nu), 'r-', linewidth=2, label=f'Student-t (ν={nu})')
    ax5.set_xlabel('Standardized residual (r/σ)')
    ax5.set_ylabel('Density')
    ax5.set_title('Standardized Residuals (clipped)')
    ax5.legend(fontsize=8)
    ax5.set_xlim(-10, 10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: QQ plot
    ax6 = axes[1, 2]
    sorted_std = np.sort(std_clipped)
    n = len(sorted_std)
    theoretical_q = student_t.ppf(np.linspace(0.01, 0.99, n), nu)
    ax6.scatter(theoretical_q, sorted_std, alpha=0.5, s=15, c='steelblue')
    lim = 5
    ax6.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.7)
    ax6.set_xlabel(f'Theoretical quantiles (Student-t, ν={nu})')
    ax6.set_ylabel('Observed quantiles (clipped)')
    ax6.set_title('Q-Q Plot')
    ax6.set_xlim(-lim, lim)
    ax6.set_ylim(-lim, lim)
    ax6.grid(True, alpha=0.3)

    # Row 3: Partial dependence plots
    # Use same basis function formulas as compute_variance_features (0.5× scaling)
    # Plot 7: σ vs Length
    ax7 = axes[2, 0]
    ax7.scatter(lengths_px, sigma, alpha=0.5, s=20, c='steelblue')
    L_range = np.linspace(lengths_px.min(), lengths_px.max(), 100)
    f1_range = -0.5 * np.log(np.maximum(L_range, 30.0))
    f2_med = np.median(0.5 * np.log(np.maximum(chi2_vals, 1e-6)))
    f3_med = np.median(-0.5 * np.log(np.maximum(snr_vals, 0.1)))
    sigma_pred = np.exp(beta[0] + beta[1] * f1_range + beta[2] * f2_med + beta[3] * f3_med)
    ax7.plot(L_range, sigma_pred, 'r-', linewidth=2, label=f'Model (β₁={beta[1]:.2f})')
    ax7.set_xlabel('Length (px)')
    ax7.set_ylabel('σ')
    ax7.set_title(f'σ vs Length (expect β₁≈1.0)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # Plot 8: σ vs chi2
    ax8 = axes[2, 1]
    ax8.scatter(chi2_vals, sigma, alpha=0.5, s=20, c='steelblue')
    chi2_max = min(np.percentile(chi2_vals, 99), 10.0)
    chi2_range = np.linspace(0.01, chi2_max, 100)
    f1_med = np.median(-0.5 * np.log(np.maximum(lengths_px, 30.0)))
    f2_range = 0.5 * np.log(np.maximum(chi2_range, 1e-6))  # No clipping at 1
    sigma_pred = np.exp(beta[0] + beta[1] * f1_med + beta[2] * f2_range + beta[3] * f3_med)
    ax8.plot(chi2_range, sigma_pred, 'r-', linewidth=2, label=f'Model (β₂={beta[2]:.2f})')
    ax8.set_xlabel('χ² (reduced)')
    ax8.set_ylabel('σ')
    ax8.set_title(f'σ vs χ² (expect β₂≈1.0)')
    ax8.legend(fontsize=8)
    ax8.set_xlim(0, chi2_max)
    ax8.grid(True, alpha=0.3)

    # Plot 9: σ vs SNR
    ax9 = axes[2, 2]
    ax9.scatter(snr_vals, sigma, alpha=0.5, s=20, c='steelblue')
    snr_max = np.percentile(snr_vals, 99)
    snr_range = np.linspace(max(0.1, snr_vals.min()), snr_max, 100)
    f3_range = -0.5 * np.log(np.maximum(snr_range, 0.1))
    sigma_pred = np.exp(beta[0] + beta[1] * f1_med + beta[2] * f2_med + beta[3] * f3_range)
    ax9.plot(snr_range, sigma_pred, 'r-', linewidth=2, label=f'Model (β₃={beta[3]:.2f})')
    ax9.set_xlabel('SNR (linear)')
    ax9.set_ylabel('σ')
    ax9.set_title(f'σ vs SNR (expect β₃≈1.0)')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Poiseuille-based Flow Simulation
# =============================================================================

@dataclass
class PoiseuilleSimulationResult:
    """Results from Poiseuille flow simulation."""
    # Inferred parameters
    mu_cP: float  # Viscosity in centipoise

    # Node pressures (arbitrary units, relative)
    node_pressures: Dict[int, float]

    # Predicted flows
    predicted_Q: Dict[Tuple[int, int], float]  # (u, v) -> Q in nL/s

    # Measured vs predicted comparison
    measured_Q: Dict[Tuple[int, int], float]

    # Fit quality
    r_squared: float
    rmse: float  # nL/s

    # Boundary info
    boundary_nodes: Set[int]
    boundary_pressures: Dict[int, float]

    # Robust regression results (if robust_method was used)
    outlier_edges: Optional[List[Dict]] = None  # List of outlier edge info
    outlier_weights: Optional[np.ndarray] = None  # Per-edge weights (1.0 = inlier)

    # Per-vessel diagnostics
    z_scores: Optional[Dict[Tuple[int, int], float]] = None  # (u,v) -> z = (Q_meas - Q_pred) / σ_Q
    sigma_Q_used: Optional[Dict[Tuple[int, int], float]] = None  # (u,v) -> σ_Q used for weighting


def compute_conductance(
    radius_um: float,
    length_um: float,
    mu_cP: float = 3.0,
) -> float:
    """
    Compute Poiseuille conductance G = πR⁴/(8μL).

    Parameters
    ----------
    radius_um : float
        Vessel radius in micrometers
    length_um : float
        Vessel length in micrometers
    mu_cP : float
        Dynamic viscosity in centipoise (1 cP = 1 mPa·s)

    Returns
    -------
    G : float
        Conductance in nL/s per Pa (flow per unit pressure)
    """
    # Convert units:
    # R: um -> m (1e-6)
    # L: um -> m (1e-6)
    # mu: cP -> Pa·s (1e-3)
    # Result in m³/s/Pa, then convert to nL/s/Pa (1e12 nL/m³)

    R_m = radius_um * 1e-6
    L_m = length_um * 1e-6
    mu_Pa_s = mu_cP * 1e-3

    G_m3_per_s_per_Pa = np.pi * R_m**4 / (8 * mu_Pa_s * L_m)
    G_nL_per_s_per_Pa = G_m3_per_s_per_Pa * 1e12  # m³ → nL (1 m³ = 1e12 nL)

    return G_nL_per_s_per_Pa


# ---------------------------------------------------------------------------
# Spatial outlier filtering
# ---------------------------------------------------------------------------

def _wrap_angle(a):
    """Wrap angle(s) to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _circular_median(angles):
    """Circular median: the candidate angle that minimises total arc distance."""
    angles = np.asarray(angles)
    best_cost = np.inf
    best = angles[0]
    for candidate in angles:
        cost = np.sum(np.abs(_wrap_angle(angles - candidate)))
        if cost < best_cost:
            best_cost = cost
            best = candidate
    return float(best)


def _get_neighbor_edges(G, u, v):
    """Return graph-adjacent edges of (u, v) — edges sharing node u or v."""
    neighbors = []
    for w in G.neighbors(u):
        if w != v and G.has_edge(u, w):
            neighbors.append((u, w))
    for w in G.neighbors(v):
        if w != u and G.has_edge(v, w):
            neighbors.append((v, w))
    return neighbors


def _mad_z(value, neighbor_values):
    """MAD-based z-score of *value* relative to the local pool (self + neighbors)."""
    pool = np.array([value] + list(neighbor_values))
    med = np.median(pool)
    mad = np.median(np.abs(pool - med))
    sigma = 1.4826 * mad
    if sigma < 1e-12:
        return 0.0
    return abs(value - med) / sigma


def _circular_mad_z(angle, neighbor_angles):
    """Circular MAD z-score for angle data in radians."""
    pool = np.array([angle] + list(neighbor_angles))
    circ_med = _circular_median(pool)
    deviations = np.abs(_wrap_angle(pool - circ_med))
    circ_mad = float(np.median(deviations))
    sigma = 1.4826 * circ_mad
    if sigma < 1e-12:
        return 0.0
    return float(np.abs(_wrap_angle(angle - circ_med))) / sigma


def _best_tile_id(edge_data: dict) -> Optional[int]:
    """Return the tile_id of the best measurement on an edge.

    The best measurement is the one whose top-level ``mean_Q`` matches
    (priority tile / highest SNR logic).  Falls back to the last
    measurement if no match is found.
    """
    measurements = edge_data.get('measurements', [])
    if not measurements:
        return None
    if len(measurements) == 1:
        return measurements[0].get('tile_id')
    # Match by snr_db (same logic as _pick_best_measurement)
    top_snr = edge_data.get('snr_db')
    if top_snr is not None and np.isfinite(top_snr):
        for m in measurements:
            if m.get('snr_db') == top_snr:
                return m.get('tile_id')
    return measurements[-1].get('tile_id')


def filter_spatial_outliers(
    G: nx.Graph,
    px_size_um: float = 1.7,
    mu_cP: float = 3.5,
    z_thresh_PI: float = 3.5,
    z_thresh_phase: float = 3.5,
    z_thresh_dP: float = 4.0,
    min_neighbors: int = 2,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Dict[str, bool]]:
    """Flag spatial outliers on a mosaic vessel graph, per tile.

    For each edge, compare its PI, phase, and mean_Q to graph-adjacent
    edges **from the same tile**.  PI and phase use MAD-based z-scores
    (circular for phase).  mean_Q is normalised by Hagen-Poiseuille
    conductance to get implied |ΔP|, then compared locally.

    Filtering is done per-tile so that cross-tile measurement
    discrepancies do not cause false positives.

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with edge attributes from ``_RESULT_FIELDS``.
    px_size_um : float
        Pixel size in µm for converting radius_px / path_length_px.
    mu_cP : float
        Blood viscosity in centipoise.
    z_thresh_PI, z_thresh_phase, z_thresh_dP : float
        MAD z-score thresholds for flagging outliers.
    min_neighbors : int
        Minimum number of same-tile adjacent edges with valid data
        required to test.
    verbose : bool
        Print summary.

    Returns
    -------
    dict
        ``{(u, v): {'PI': bool, 'phase': bool, 'mean_Q': bool}}``
        for every edge.  ``True`` = flagged as outlier.
    """
    # --- Determine source tile for each edge ---
    edge_tile: Dict[Tuple[int, int], int] = {}
    for u, v, data in G.edges(data=True):
        tid = _best_tile_id(data)
        if tid is not None:
            edge_tile[(u, v)] = tid

    # --- Pre-compute implied |ΔP| for mean_Q filter ---
    edge_dP: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        Q = data.get('mean_Q')
        R = data.get('radius_px')
        L = data.get('path_length_px')
        if (Q is None or R is None or L is None
                or not np.isfinite(Q) or not np.isfinite(R) or not np.isfinite(L)
                or R <= 0 or L <= 0):
            continue
        G_cond = compute_conductance(R * px_size_um, L * px_size_um, mu_cP)
        if G_cond <= 0:
            continue
        edge_dP[(u, v)] = abs(Q) / G_cond

    # --- Main loop ---
    results: Dict[Tuple[int, int], Dict[str, bool]] = {}
    n_flagged = {'PI': 0, 'phase': 0, 'mean_Q': 0}
    n_tested = {'PI': 0, 'phase': 0, 'mean_Q': 0}

    for u, v, data in G.edges(data=True):
        flags = {'PI': False, 'phase': False, 'mean_Q': False}
        my_tile = edge_tile.get((u, v))

        # Only compare to neighbors from the same tile
        neighbor_edges = _get_neighbor_edges(G, u, v)
        if my_tile is not None:
            same_tile_neighbors = [
                e for e in neighbor_edges
                if edge_tile.get(e, edge_tile.get((e[1], e[0]))) == my_tile
            ]
        else:
            same_tile_neighbors = neighbor_edges

        # --- PI ---
        PI_val = data.get('PI')
        if PI_val is not None and np.isfinite(PI_val):
            nbr_PIs = [G.edges[e]['PI'] for e in same_tile_neighbors
                       if G.edges[e].get('PI') is not None
                       and np.isfinite(G.edges[e]['PI'])]
            if len(nbr_PIs) >= min_neighbors:
                z = _mad_z(PI_val, nbr_PIs)
                flags['PI'] = z > z_thresh_PI
                n_tested['PI'] += 1
                if flags['PI']:
                    n_flagged['PI'] += 1

        # --- Phase (circular) ---
        phase_val = data.get('phase')
        if phase_val is not None and np.isfinite(phase_val):
            nbr_phases = [G.edges[e]['phase'] for e in same_tile_neighbors
                          if G.edges[e].get('phase') is not None
                          and np.isfinite(G.edges[e]['phase'])]
            if len(nbr_phases) >= min_neighbors:
                z = _circular_mad_z(phase_val, nbr_phases)
                flags['phase'] = z > z_thresh_phase
                n_tested['phase'] += 1
                if flags['phase']:
                    n_flagged['phase'] += 1

        # --- mean_Q via implied ΔP ---
        dP = edge_dP.get((u, v))
        if dP is not None:
            nbr_dPs = []
            for e in same_tile_neighbors:
                d = edge_dP.get(e) or edge_dP.get((e[1], e[0]))
                if d is not None:
                    nbr_dPs.append(d)
            if len(nbr_dPs) >= min_neighbors:
                z = _mad_z(dP, nbr_dPs)
                flags['mean_Q'] = z > z_thresh_dP
                n_tested['mean_Q'] += 1
                if flags['mean_Q']:
                    n_flagged['mean_Q'] += 1

        results[(u, v)] = flags

    if verbose:
        # Per-tile breakdown
        tile_ids = sorted(set(edge_tile.values()))
        print(f"Spatial outlier filtering ({len(tile_ids)} tiles):")
        for field, thresh in [('PI', z_thresh_PI),
                              ('phase', z_thresh_phase),
                              ('mean_Q', z_thresh_dP)]:
            print(f"  {field:8s}: {n_flagged[field]:3d}/{n_tested[field]:3d} flagged "
                  f"(z > {thresh})")

    return results


def apply_spatial_outlier_flags(
    G: nx.Graph,
    outlier_flags: Dict[Tuple[int, int], Dict[str, bool]],
) -> int:
    """Apply spatial outlier flags: backup originals, NaN flagged fields.

    For each flagged edge/field, the original value is saved in a
    ``_pre_spatial_filter`` dict on the edge before being set to NaN.
    A ``spatial_outlier`` dict is also stored on the edge.

    Returns the number of edges that had at least one field flagged.
    """
    _FLOW_EXTRAS = ('amp_Q', 'v_max', 'mean_Q_nL_s')
    _FIELD_MAP = {'PI': 'PI', 'phase': 'phase', 'mean_Q': 'mean_Q'}

    n_modified = 0
    for (u, v), flags in outlier_flags.items():
        if not any(flags.values()):
            continue

        edge_data = G.edges[u, v]

        # Create or get backup dict (don't overwrite existing backup)
        if '_pre_spatial_filter' not in edge_data:
            edge_data['_pre_spatial_filter'] = {}
        backup = edge_data['_pre_spatial_filter']

        for flag_key, attr_name in _FIELD_MAP.items():
            if flags[flag_key]:
                val = edge_data.get(attr_name)
                if val is not None and np.isfinite(val):
                    backup[attr_name] = val
                    edge_data[attr_name] = np.nan

        # If mean_Q flagged, also NaN related flow fields
        if flags.get('mean_Q', False):
            for extra in _FLOW_EXTRAS:
                val = edge_data.get(extra)
                if val is not None and np.isfinite(val):
                    backup[extra] = val
                    edge_data[extra] = np.nan

        edge_data['spatial_outlier'] = dict(flags)
        n_modified += 1

    return n_modified


# ------------------------------------------------------------------
# Graph-regularised field smoothing (PI, etc.)
# ------------------------------------------------------------------

def _build_edge_adjacency_laplacian(G, edge_list, edge_idx):
    """Build the edge-adjacency graph Laplacian as a sparse matrix.

    Two edges are adjacent if they share a junction node in *G*.

    Parameters
    ----------
    G : nx.Graph
    edge_list : list of (u, v)
        Ordered list of edges.
    edge_idx : dict
        Maps ``(u, v)`` → index (both orderings).

    Returns
    -------
    scipy.sparse.csr_matrix
        Edge-adjacency Laplacian (n_edges × n_edges).
    """
    from scipy.sparse import lil_matrix
    n = len(edge_list)
    L = lil_matrix((n, n))
    for i, (u, v) in enumerate(edge_list):
        for w in G.neighbors(u):
            if w != v:
                j = edge_idx.get((u, w))
                if j is not None and j != i:
                    L[i, j] -= 1.0
                    L[i, i] += 1.0
        for w in G.neighbors(v):
            if w != u:
                j = edge_idx.get((v, w))
                if j is not None and j != i:
                    L[i, j] -= 1.0
                    L[i, i] += 1.0
    return L.tocsr()


def normalize_pi_by_f0(
    G: nx.Graph,
    f_ref: float = 2.5,
    verbose: bool = True,
) -> int:
    """Compute PI_f0 = PI * (f0_hz / f_ref) on each edge.

    In the embryonic tubular heart PI ∝ 1/f0: faster hearts have lower PI
    for the same network attenuation because mean flow scales with f0 while
    pulse amplitude (set by stroke volume × waveform shape) does not.
    Multiplying by f0/f_ref normalises to a reference rate, isolating
    the network's pulse-wave attenuation.

    f_ref = 2.5 Hz is the nominal HH14-16 heart rate at 37 °C.
    A tile at 2.0 Hz gets PI_f0 = PI × 0.8 (raw PI inflated by slow rate).

    Also updates PI_f0 on each measurement dict for downstream consensus.

    Returns number of edges updated.
    """
    n = 0
    for u, v, data in G.edges(data=True):
        pi = data.get('PI')
        f0 = data.get('f0_hz')
        if pi is not None and f0 is not None and np.isfinite(pi) and np.isfinite(f0) and f0 > 0:
            data['PI_f0'] = float(pi * f0 / f_ref)
            n += 1

        # Also tag each measurement so consensus can use PI_f0
        for m in data.get('measurements', []):
            m_pi = m.get('PI')
            m_f0 = m.get('f0_hz')
            if (m_pi is not None and m_f0 is not None
                    and np.isfinite(m_pi) and np.isfinite(m_f0) and m_f0 > 0):
                m['PI_f0'] = float(m_pi * m_f0 / f_ref)

    if verbose:
        print(f"PI_f0 normalization: {n} edges computed "
              f"(PI × f0/{f_ref:.2f} Hz)")
    return n


def filter_tile_field_outliers(
    G: nx.Graph,
    field: str = 'PI',
    k: float = 3.0,
    verbose: bool = True,
) -> int:
    """Per-tile outlier rejection using median absolute deviation.

    For each tile, computes the median and MAD of *field* values.
    Edges with values beyond ``median + k * 1.4826 * MAD`` are set to
    NaN so that downstream smoothing can interpolate them.

    This catches low-flow vessels where noise dominates and PI blows up.

    Parameters
    ----------
    G : nx.Graph
    field : str
        Edge attribute to filter.
    k : float
        Number of robust-sigma to use as threshold (default 3.0).
    verbose : bool

    Returns
    -------
    int
        Number of edges set to NaN.
    """
    from collections import defaultdict

    # Collect field values per tile
    tile_edges: Dict[int, list] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        val = data.get(field)
        if val is None or not np.isfinite(val):
            continue
        tid = _best_tile_id(data)
        if tid is not None:
            tile_edges[tid].append((u, v, val))

    n_removed = 0
    for tid in sorted(tile_edges):
        entries = tile_edges[tid]
        vals = np.array([v for _, _, v in entries])
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        # 1.4826 converts MAD to Gaussian sigma equivalent
        sigma_est = 1.4826 * mad if mad > 1e-12 else float(np.std(vals))
        if sigma_est < 1e-12:
            continue
        hi = med + k * sigma_est

        tile_removed = 0
        for u, v, val in entries:
            if val > hi:
                G.edges[u, v][field] = float('nan')
                tile_removed += 1

        n_removed += tile_removed
        if verbose and tile_removed > 0:
            print(f"  Tile {tid:3d}: {tile_removed}/{len(entries)} outliers "
                  f"(>{hi:.2f}, median={med:.2f}, sigma={sigma_est:.2f})")

    if verbose:
        total = sum(len(e) for e in tile_edges.values())
        print(f"Tile outlier filter ({field}): {n_removed}/{total} edges removed")
    return n_removed


def compute_edge_consensus(
    G: nx.Graph,
    field: str = 'PI',
    verbose: bool = True,
) -> int:
    """Set top-level *field* to a weighted median of all tile measurements.

    For edges with multiple measurements, uses the inverse relative
    uncertainty (``snr_db``) as weight.  Replaces the current top-level
    value (which is just the "best" single measurement).

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with ``measurements`` on edges.
    field : str
        Field name (must exist in measurements).
    verbose : bool

    Returns
    -------
    int
        Number of edges updated (had ≥ 2 valid measurements).
    """
    n_updated = 0
    for u, v, data in G.edges(data=True):
        measurements = data.get('measurements', [])
        vals, wts = [], []
        for m in measurements:
            if not m.get('fit_success', False):
                continue
            val = m.get(field)
            if val is None or not np.isfinite(val):
                continue
            snr = m.get('snr_db', 0.0)
            if snr is None or not np.isfinite(snr):
                snr = 0.0
            wts.append(max(snr, 0.1))
            vals.append(val)

        if len(vals) < 2:
            continue

        # Weighted median: sort by value, find where cumulative weight
        # crosses 50%
        vals = np.array(vals)
        wts = np.array(wts)
        order = np.argsort(vals)
        vals_s = vals[order]
        wts_s = wts[order]
        cum = np.cumsum(wts_s)
        half = cum[-1] / 2.0
        idx = np.searchsorted(cum, half)
        idx = min(idx, len(vals_s) - 1)
        data[field] = float(vals_s[idx])
        n_updated += 1

    if verbose:
        print(f"Edge consensus ({field}): {n_updated} edges updated "
              f"from multiple measurements")
    return n_updated


def estimate_tile_field_corrections(
    G: nx.Graph,
    field: str = 'PI',
    min_vessels: int = 3,
    max_pair_ratio: float = 3.0,
    max_factor: float = 2.0,
    snr_min: float = 3.0,
    verbose: bool = True,
) -> Dict[int, float]:
    """Estimate per-tile multiplicative corrections for *field*.

    For each tile pair sharing edges, computes the median ratio of
    field values.  Solves a weighted least-squares system on the
    log-ratios (like flat-field correction) to get per-tile
    multiplicative factors.

    Parameters
    ----------
    G : nx.Graph
    field : str
    min_vessels : int
        Minimum number of overlapping vessels per tile pair to include
        that pair in the estimation (default 3).
    max_pair_ratio : float
        Reject tile pairs whose median ratio exceeds this or falls
        below 1/this (default 3.0).  Extreme ratios indicate bad fits.
    max_factor : float
        Clamp final correction factors to [1/max_factor, max_factor]
        (default 2.0).
    snr_min : float
        Minimum SNR (dB) for a measurement to contribute (default 3.0).
    verbose : bool

    Returns
    -------
    dict
        ``{vid: factor}`` — multiply ``field`` values from this tile
        by ``factor`` to bring them to the reference frame.
        Empty if no tile systematics detected.
    """
    from collections import defaultdict
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr

    # Collect per-tile-pair ratios from overlap edges
    pair_ratios: Dict[Tuple[int, int], list] = defaultdict(list)

    for u, v, data in G.edges(data=True):
        measurements = data.get('measurements', [])
        valid = []
        for m in measurements:
            if not m.get('fit_success', False):
                continue
            val = m.get(field)
            if val is None or not np.isfinite(val) or val <= 0:
                continue
            snr = m.get('snr_db', 0.0)
            if snr is None or not np.isfinite(snr):
                snr = 0.0
            if snr < snr_min:
                continue
            valid.append(m)
        if len(valid) < 2:
            continue

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                tid_a = valid[i]['tile_id']
                tid_b = valid[j]['tile_id']
                val_a = valid[i][field]
                val_b = valid[j][field]
                key = (min(tid_a, tid_b), max(tid_a, tid_b))
                ratio = val_a / val_b if tid_a == key[0] else val_b / val_a
                pair_ratios[key].append(ratio)

    if not pair_ratios:
        if verbose:
            print(f"Tile corrections ({field}): no overlap data")
        return {}

    # Median ratio per pair — filter by vessel count and ratio sanity
    pair_log_offsets: Dict[Tuple[int, int], float] = {}
    pair_weights: Dict[Tuple[int, int], float] = {}
    max_ratio = 1.0
    n_rejected = 0
    for pair, ratios in pair_ratios.items():
        ratios = np.array(ratios)
        n_v = len(ratios)

        # Require minimum vessel count
        if n_v < min_vessels:
            if verbose:
                print(f"  Tiles {pair[0]:3d}-{pair[1]:<3d}: "
                      f"{n_v:3d} vessels (< {min_vessels}) — skipped")
            n_rejected += 1
            continue

        med = float(np.median(ratios))

        # Reject extreme ratios (bad fits driving garbage)
        if med > max_pair_ratio or med < 1.0 / max_pair_ratio:
            if verbose:
                print(f"  Tiles {pair[0]:3d}-{pair[1]:<3d}: "
                      f"{n_v:3d} vessels, "
                      f"median ratio = {med:.3f} (extreme, > {max_pair_ratio}) — skipped")
            n_rejected += 1
            continue

        pair_log_offsets[pair] = np.log(med)
        # Weight by number of vessels / spread
        if n_v > 1:
            pair_weights[pair] = n_v / max(np.std(np.log(ratios)), 0.01)
        else:
            pair_weights[pair] = 1.0
        max_ratio = max(max_ratio, med, 1.0 / med)

        if verbose:
            print(f"  Tiles {pair[0]:3d}-{pair[1]:<3d}: "
                  f"{n_v:3d} vessels, "
                  f"median ratio = {med:.3f}")

    if not pair_log_offsets:
        if verbose:
            print(f"  No valid tile pairs after filtering "
                  f"({n_rejected} rejected)")
        return {}

    # If all ratios are close to 1, no correction needed
    if max_ratio < 1.05:
        if verbose:
            print(f"  Max ratio {max_ratio:.3f} < 1.05 — no tile correction needed")
        return {}

    # Solve least-squares on log-ratios: log(α_i) - log(α_j) = log(ratio_ij)
    all_tiles: set = set()
    for a, b in pair_log_offsets:
        all_tiles.add(a)
        all_tiles.add(b)
    tile_list = sorted(all_tiles)
    tile_idx = {t: i for i, t in enumerate(tile_list)}
    n_tiles = len(tile_list)

    n_constraints = len(pair_log_offsets) + 1  # +1 to pin reference
    A = lil_matrix((n_constraints, n_tiles))
    b_vec = np.zeros(n_constraints)
    w_vec = np.zeros(n_constraints)

    row = 0
    for (ti, tj), log_off in pair_log_offsets.items():
        A[row, tile_idx[ti]] = 1.0
        A[row, tile_idx[tj]] = -1.0
        b_vec[row] = log_off
        w_vec[row] = pair_weights[(ti, tj)]
        row += 1

    # Pin: product of all factors = 1 → sum of logs = 0
    # Simplest: pin the most-connected tile to log(α) = 0
    tile_conn: Dict[int, int] = defaultdict(int)
    for a, b in pair_log_offsets:
        tile_conn[a] += 1
        tile_conn[b] += 1
    ref = max(tile_conn, key=tile_conn.get)
    A[row, tile_idx[ref]] = 1.0
    b_vec[row] = 0.0
    w_vec[row] = 1e6
    row += 1

    from scipy.sparse import diags
    sw = np.sqrt(w_vec[:row])
    A_w = diags(sw) @ A[:row].tocsr()
    b_w = sw * b_vec[:row]

    result = lsqr(A_w, b_w)
    log_factors = result[0]

    # Clamp correction factors to [1/max_factor, max_factor]
    log_max = np.log(max_factor)
    tile_factors: Dict[int, float] = {}
    n_clamped = 0
    for t, idx in tile_idx.items():
        lf = log_factors[idx]
        if abs(lf) > log_max:
            lf = np.clip(lf, -log_max, log_max)
            n_clamped += 1
        tile_factors[t] = float(np.exp(lf))

    if verbose:
        print(f"  Tile correction factors ({field}, ref = tile {ref}):")
        for tid in sorted(tile_factors):
            print(f"    Tile {tid:3d}: x{tile_factors[tid]:.4f}")
        if n_clamped > 0:
            print(f"  ({n_clamped} factors clamped to "
                  f"[{1/max_factor:.2f}, {max_factor:.2f}])")
        if n_rejected > 0:
            print(f"  ({n_rejected} tile pairs rejected)")

    return tile_factors


def apply_tile_field_corrections(
    G: nx.Graph,
    tile_factors: Dict[int, float],
    field: str = 'PI',
) -> int:
    """Multiply *field* on each edge by the tile correction factor.

    Returns number of edges updated.
    """
    n = 0
    for u, v, data in G.edges(data=True):
        val = data.get(field)
        if val is None or not np.isfinite(val):
            continue
        tid = _best_tile_id(data)
        if tid is None or tid not in tile_factors:
            continue
        data[field] = float(val * tile_factors[tid])
        n += 1
    return n


def smooth_graph_field(
    G: nx.Graph,
    field: str = 'PI',
    output_field: Optional[str] = None,
    lambda_reg: float = 1.0,
    sigma_field: Optional[str] = None,
    default_sigma_rel: float = 0.3,
    verbose: bool = True,
) -> int:
    """Graph-regularised smoothing of an edge field.

    Solves ``(D + λ L) s = D p`` where *D* = diag(1/σ²) is the
    data-fidelity weight, *L* is the edge-adjacency graph Laplacian,
    and *p* are the measured values.

    Edges with missing / NaN data get *D_ii = 0*, so they are
    interpolated purely from their neighbors.

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph.
    field : str
        Input field on edges.
    output_field : str, optional
        Output field name (default: ``field + '_smooth'``).
    lambda_reg : float
        Regularisation strength.  Higher = smoother.
    sigma_field : str, optional
        Edge attribute containing measurement uncertainty.  If *None*,
        uses ``default_sigma_rel × |value|`` as the uncertainty.
    default_sigma_rel : float
        Relative uncertainty when ``sigma_field`` is not available.
    verbose : bool

    Returns
    -------
    int
        Number of edges that received a smoothed value.
    """
    from scipy.sparse import diags as sp_diags
    from scipy.sparse.linalg import spsolve

    if output_field is None:
        output_field = field + '_smooth'

    # Build ordered edge list and index
    edge_list = list(G.edges())
    edge_idx: Dict[Tuple[int, int], int] = {}
    for i, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    n_edges = len(edge_list)

    # Extract data vector and weights
    p = np.full(n_edges, np.nan)
    sigma = np.full(n_edges, np.inf)

    for i, (u, v) in enumerate(edge_list):
        data = G.edges[u, v]
        val = data.get(field)
        if val is not None and np.isfinite(val):
            p[i] = val
            # Get uncertainty
            sig = None
            if sigma_field is not None:
                sig = data.get(sigma_field)
            if sig is None or not np.isfinite(sig) or sig <= 0:
                sig = default_sigma_rel * abs(val) if abs(val) > 1e-12 else 1.0
            sigma[i] = sig

    valid = np.isfinite(p)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        if verbose:
            print(f"Smooth ({field}): no valid data")
        return 0

    # Build D = diag(1/σ²), with 0 for missing data
    d_diag = np.zeros(n_edges)
    d_diag[valid] = 1.0 / sigma[valid]**2
    D = sp_diags(d_diag)

    # Build edge-adjacency Laplacian
    L = _build_edge_adjacency_laplacian(G, edge_list, edge_idx)

    # Solve (D + λL) s = D p
    # For missing data: p[i] is NaN, D[i,i] = 0, so RHS = 0, and
    # the Laplacian term interpolates from neighbors.
    rhs = d_diag * np.where(valid, p, 0.0)
    A_mat = D + lambda_reg * L

    s = spsolve(A_mat, rhs)

    # Write results
    n_written = 0
    for i, (u, v) in enumerate(edge_list):
        if np.isfinite(s[i]):
            G.edges[u, v][output_field] = float(s[i])
            n_written += 1

    if verbose:
        # Report how much the smoothing changed things
        residuals = s[valid] - p[valid]
        rms_change = np.sqrt(np.mean(residuals**2))
        mean_val = np.mean(np.abs(p[valid]))
        print(f"Smooth ({field} → {output_field}): "
              f"{n_valid} measured + {n_edges - n_valid} interpolated, "
              f"λ={lambda_reg:.1f}, "
              f"RMS change = {rms_change:.4f} "
              f"({100*rms_change/mean_val:.1f}% of mean |{field}|)")

    return n_written


def filter_pi_per_tile(
    G: nx.Graph,
    k_outlier: float = 3.0,
    min_mean_Q: float = 0.1,
    tile_id: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Per-tile PI outlier rejection and low-flow filtering.

    Writes ``PI_filt`` into each measurement dict — never overwrites
    the raw ``PI``.  For each tile independently:

    1. Reject measurements where ``|mean_Q| < min_mean_Q`` (low-flow
       vessels where PI = 2*amp/|mean| blows up from noise).
    2. Reject outliers via median + k×MAD on remaining PI values.
    3. Rejected measurements get ``PI_filt = NaN``; survivors get
       ``PI_filt = PI``.

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with ``measurements`` on edges.
    k_outlier : float
        Outlier rejection threshold in robust-sigma (default 3.0).
    min_mean_Q : float
        Minimum |mean_Q| for a measurement to keep its PI (default 0.1).
    tile_id : int, optional
        If given, only filter this tile.  Otherwise filter all tiles.
    verbose : bool

    Returns
    -------
    int
        Number of measurements rejected.
    """
    from collections import defaultdict

    # Group measurements by tile_id
    tile_data: Dict[int, list] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        for m in data.get('measurements', []):
            if not m.get('fit_success', False):
                continue
            tid = m.get('tile_id')
            if tid is None:
                continue
            if tile_id is not None and tid != tile_id:
                continue
            pi = m.get('PI')
            if hasattr(pi, 'item'):
                pi = pi.item()
            if pi is None or not np.isfinite(pi):
                m['PI_filt'] = float('nan')
                continue
            tile_data[tid].append((u, v, m, pi))

    if not tile_data:
        if verbose:
            print("PI filter: no tile measurements found")
        return 0

    total_rejected = 0

    for tid in sorted(tile_data):
        entries = tile_data[tid]
        n_tile = len(entries)

        # Step 1: low-flow filter (|mean_Q| < threshold)
        n_low_flow = 0
        surviving = []
        for u, v, m, pi in entries:
            mean_q = m.get('mean_Q', 0.0)
            if hasattr(mean_q, 'item'):
                mean_q = mean_q.item()
            if mean_q is None or not np.isfinite(mean_q):
                mean_q = 0.0
            if abs(mean_q) < min_mean_Q:
                m['PI_filt'] = float('nan')
                n_low_flow += 1
            else:
                surviving.append((u, v, m, pi))

        # Step 2: MAD outlier rejection on surviving values
        n_mad = 0
        if len(surviving) >= 3:
            vals = np.array([e[3] for e in surviving])
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            sigma_est = 1.4826 * mad if mad > 1e-12 else float(np.std(vals))
            if sigma_est > 1e-12:
                hi = med + k_outlier * sigma_est
                lo = med - k_outlier * sigma_est
                for u, v, m, pi in surviving:
                    if pi > hi or pi < lo:
                        m['PI_filt'] = float('nan')
                        n_mad += 1
                    else:
                        m['PI_filt'] = float(pi)
            else:
                for u, v, m, pi in surviving:
                    m['PI_filt'] = float(pi)
        else:
            # Too few to do MAD — keep all survivors
            for u, v, m, pi in surviving:
                m['PI_filt'] = float(pi)

        tile_rejected = n_low_flow + n_mad
        total_rejected += tile_rejected

        if verbose and tile_rejected > 0:
            print(f"  Tile {tid:3d}: {tile_rejected}/{n_tile} rejected "
                  f"({n_low_flow} low-flow, {n_mad} MAD outliers)")

    # Propagate best tile's filtered value to top-level PI_filt
    for u, v, data in G.edges(data=True):
        best_val = None
        best_snr = -np.inf
        for m in data.get('measurements', []):
            pf = m.get('PI_filt')
            if pf is None or not np.isfinite(pf):
                continue
            snr = m.get('snr_db', 0.0)
            if snr is None or not np.isfinite(snr):
                snr = 0.0
            if snr > best_snr:
                best_snr = snr
                best_val = pf
        if best_val is not None:
            data['PI_filt'] = float(best_val)

    if verbose:
        total_meas = sum(len(e) for e in tile_data.values())
        print(f"PI filter: {total_rejected}/{total_meas} rejected "
              f"across {len(tile_data)} tiles")

    return total_rejected


# ------------------------------------------------------------------
# Phase stitching across tiles
# ------------------------------------------------------------------

def _circular_mean_weighted(angles, weights=None):
    """Weighted circular mean of angles (radians).

    Returns the angle whose direction is the weighted vector sum.
    """
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.asarray(weights, dtype=float)
    S = np.sum(weights * np.sin(angles))
    C = np.sum(weights * np.cos(angles))
    return float(np.arctan2(S, C))


def _canonicalize_phase(phase, mean_Q):
    """Canonicalize phase to 'peak speed time'.

    Q(t) = mean_Q + amp_Q·cos(ωt + φ).  Peak |Q| occurs at cos = +1
    when mean_Q > 0 and at cos = -1 when mean_Q < 0.  Shifting by π
    in the negative case makes the phase represent time-of-peak-speed
    regardless of vessel tracing direction.
    """
    if mean_Q < 0:
        return phase + np.pi
    return phase


def stitch_tile_phases(
    G: nx.Graph,
    reference_tile: Optional[int] = None,
    snr_min: float = 3.0,
    outlier_sigma: float = 2.0,
    verbose: bool = True,
) -> Dict[int, float]:
    """Compute per-tile phase offsets for globally consistent phases.

    Uses the **doubled-angle trick** to handle the π-ambiguity from
    vessel tracing direction without relying on the sign of ``mean_Q``.

    For each overlap vessel with raw phases φ_A, φ_B from tiles A, B:
    - Δφ = wrap(φ_A − φ_B) may be off by ±π (direction ambiguity)
    - But 2Δφ has NO ambiguity (since ±π → ±2π → wraps to 0)

    So we solve in doubled-angle space, then halve to get offsets
    (mod π).  A branch-resolution step picks the correct π-branch
    per tile pair using majority vote of the raw deltas.

    Tile offsets are solved via weighted least squares (Laplacian).

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with ``measurements`` lists on edges.
    reference_tile : int, optional
        Tile whose phase frame defines the global reference (offset = 0).
    snr_min : float
        Minimum ``snr_db`` for a measurement to contribute.
    outlier_sigma : float
        Per tile-pair outlier rejection threshold in doubled-angle
        space (MAD z-score).
    verbose : bool
        Print summary.

    Returns
    -------
    dict
        ``{vid: offset}`` where ``phase_global = wrap(canon_phase - offset)``.
    """
    from collections import defaultdict
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr

    # --- Step 1: collect RAW pairwise phase differences (no canonicalization) ---
    pair_deltas: Dict[Tuple[int, int], list] = defaultdict(list)
    n_skipped_snr = 0

    for u, v, data in G.edges(data=True):
        measurements = data.get('measurements', [])
        valid = []
        for m in measurements:
            if not m.get('fit_success', False):
                continue
            phase_m = m.get('phase')
            if phase_m is None or not np.isfinite(phase_m):
                continue
            snr = m.get('snr_db', 0.0)
            if snr is None or not np.isfinite(snr):
                snr = 0.0
            if snr < snr_min:
                n_skipped_snr += 1
                continue
            valid.append(m)
        if len(valid) < 2:
            continue

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                tid_a = valid[i]['tile_id']
                tid_b = valid[j]['tile_id']
                # Raw phases — NO canonicalization
                phi_a = valid[i]['phase']
                phi_b = valid[j]['phase']
                delta = _wrap_angle(phi_a - phi_b)

                snr_a = valid[i].get('snr_db', 0.0) or 0.0
                snr_b = valid[j].get('snr_db', 0.0) or 0.0
                w = min(snr_a, snr_b)

                key_ab = (min(tid_a, tid_b), max(tid_a, tid_b))
                sign = 1.0 if tid_a == key_ab[0] else -1.0
                pair_deltas[key_ab].append((sign * delta, w))

    if verbose:
        skip_str = f" ({n_skipped_snr} low-SNR skipped)" if n_skipped_snr else ""
        print(f"Phase stitching: {len(pair_deltas)} tile pairs{skip_str}")

    if not pair_deltas:
        if verbose:
            print("  No overlapping phase measurements — nothing to stitch")
        return {}

    # --- Step 2: Per-pair offset via doubled-angle trick ---
    # 2Δφ eliminates the ±π ambiguity.  We work in doubled space,
    # then halve.  The result is the correct offset mod π.
    # A majority-vote step resolves the ±π branch.
    pair_offsets: Dict[Tuple[int, int], float] = {}
    pair_weights: Dict[Tuple[int, int], float] = {}

    for pair, dw_list in pair_deltas.items():
        deltas = np.array([d for d, w in dw_list])
        weights = np.array([w for d, w in dw_list])
        n_total = len(deltas)

        # Doubled-angle: 2Δφ removes ±π ambiguity
        doubled = 2.0 * deltas

        # Outlier rejection in doubled space
        med2 = _circular_median(doubled)
        residuals2 = np.abs(_wrap_angle(doubled - med2))
        circ_mad2 = float(np.median(residuals2))
        if circ_mad2 > 1e-6:
            inlier_mask = residuals2 < outlier_sigma * 1.4826 * circ_mad2
        else:
            inlier_mask = np.ones(n_total, dtype=bool)
        n_inlier = int(np.sum(inlier_mask))
        if n_inlier == 0:
            inlier_mask = np.ones(n_total, dtype=bool)
            n_inlier = n_total

        # Weighted circular mean of doubled inlier angles
        mean2 = _circular_mean_weighted(
            doubled[inlier_mask], weights[inlier_mask])

        # Halve to get offset mod π (in [-π/2, π/2])
        offset_mod_pi = mean2 / 2.0

        # Resolve π-branch: which of offset_mod_pi or offset_mod_pi + π
        # is closer to the majority of raw inlier deltas?
        inlier_deltas = deltas[inlier_mask]
        inlier_wts = weights[inlier_mask]
        dist_a = np.abs(_wrap_angle(inlier_deltas - offset_mod_pi))
        dist_b = np.abs(_wrap_angle(inlier_deltas - (offset_mod_pi + np.pi)))
        # Weighted vote
        vote_a = float(np.sum(inlier_wts[dist_a <= dist_b]))
        vote_b = float(np.sum(inlier_wts[dist_a > dist_b]))
        if vote_b > vote_a:
            final = _wrap_angle(offset_mod_pi + np.pi)
        else:
            final = offset_mod_pi

        pair_offsets[pair] = float(final)

        # Confidence: how well do inlier deltas agree with chosen offset?
        inlier_res = _wrap_angle(inlier_deltas - final)
        circ_var = float(np.var(inlier_res)) if n_inlier > 1 else 1.0
        pair_weights[pair] = n_inlier / max(circ_var, 0.01)

        if verbose:
            spread = np.degrees(np.std(inlier_res)) if n_inlier > 1 else 0.0
            n_rej = n_total - n_inlier
            rej_str = f", {n_rej} rejected" if n_rej else ""
            branch = "A" if vote_b <= vote_a else "B"
            print(f"  Tiles {pair[0]:3d}-{pair[1]:<3d}: "
                  f"{n_inlier:3d}/{n_total} vessels{rej_str}, "
                  f"Δφ = {np.degrees(final):+7.1f}° "
                  f"(spread ±{spread:.0f}°, "
                  f"branch {branch}: {vote_a:.0f}/{vote_b:.0f})")

    # --- Step 3: Weighted least squares on tile overlap graph ---
    all_tiles_set: set = set()
    for a, b in pair_offsets:
        all_tiles_set.add(a)
        all_tiles_set.add(b)
    tile_list = sorted(all_tiles_set)
    tile_idx = {t: i for i, t in enumerate(tile_list)}
    n_tiles = len(tile_list)

    if reference_tile is not None and reference_tile in all_tiles_set:
        ref = reference_tile
    else:
        tile_conn: Dict[int, int] = defaultdict(int)
        for a, b in pair_offsets:
            tile_conn[a] += 1
            tile_conn[b] += 1
        ref = max(tile_conn, key=tile_conn.get)
        if verbose and reference_tile is not None:
            print(f"  Warning: reference_tile {reference_tile} not in "
                  f"overlap graph; using tile {ref} instead")
    ref_idx = tile_idx[ref]

    n_constraints = len(pair_offsets) + 1
    A = lil_matrix((n_constraints, n_tiles))
    b_vec = np.zeros(n_constraints)
    w_vec = np.zeros(n_constraints)

    row = 0
    for (ti, tj), delta in pair_offsets.items():
        A[row, tile_idx[ti]] = 1.0
        A[row, tile_idx[tj]] = -1.0
        b_vec[row] = delta
        w_vec[row] = pair_weights[(ti, tj)]
        row += 1

    A[row, ref_idx] = 1.0
    b_vec[row] = 0.0
    w_vec[row] = 1e6
    row += 1

    sw = np.sqrt(w_vec[:row])
    A_csr = A[:row].tocsr()
    from scipy.sparse import diags
    A_w = diags(sw) @ A_csr
    b_w = sw * b_vec[:row]

    result = lsqr(A_w, b_w)
    theta = result[0]

    tile_offsets: Dict[int, float] = {}
    for t, idx in tile_idx.items():
        tile_offsets[t] = float(_wrap_angle(theta[idx]))

    if verbose:
        residual_norm = result[3]
        print(f"\n  Least-squares (ref = tile {ref}, "
              f"residual = {residual_norm:.3f}):")
        for tid in sorted(tile_offsets):
            print(f"    Tile {tid:3d}: {np.degrees(tile_offsets[tid]):+7.1f}°")

    return tile_offsets


def apply_global_phase(
    G: nx.Graph,
    tile_offsets: Dict[int, float],
) -> int:
    """Write ``phase_global = wrap(canonical_phase - offset)`` to each edge.

    The raw phase is canonicalized to "peak speed time" (shifted by π
    when ``mean_Q < 0``) before the tile offset is subtracted, matching
    the convention used in :func:`stitch_tile_phases`.

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph.
    tile_offsets : dict
        ``{vid: offset}`` from :func:`stitch_tile_phases`.

    Returns
    -------
    int
        Number of edges updated.
    """
    n_updated = 0
    for u, v, data in G.edges(data=True):
        phase = data.get('phase')
        if phase is None or not np.isfinite(phase):
            continue
        mean_Q = data.get('mean_Q', 0.0)
        tid = _best_tile_id(data)
        if tid is None or tid not in tile_offsets:
            continue
        # Canonicalize to peak-speed phase (same convention as stitch)
        if not np.isfinite(mean_Q):
            mean_Q = 0.0
        canon_phase = _canonicalize_phase(phase, mean_Q)
        data['phase_global'] = float(_wrap_angle(canon_phase - tile_offsets[tid]))
        n_updated += 1
    return n_updated


def smooth_phase_pi_flips(
    G: nx.Graph,
    field: str = 'phase_global',
    max_iterations: int = 10,
    verbose: bool = True,
) -> int:
    """Fix residual π-flip artefacts in a phase field.

    Two stages:

    1. **Tile-level**: for each tile, check if flipping ALL of its
       edges by π improves agreement at tile boundaries.  This fixes
       whole-tile π errors from the branch-resolution step.
    2. **Edge-level**: iteratively flip individual edges that are ~π
       away from their graph neighbors.

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with ``field`` attribute on edges.
    field : str
        Edge attribute to smooth.
    max_iterations : int
        Maximum edge-level sweeps.
    verbose : bool
        Print summary.

    Returns
    -------
    int
        Total number of edges flipped.
    """
    total_flipped = 0

    # --- Stage 1: Tile-level π-flip ---
    # Group edges by source tile
    tile_edges: Dict[int, list] = {}
    for u, v, data in G.edges(data=True):
        if data.get(field) is None or not np.isfinite(data.get(field, np.nan)):
            continue
        tid = _best_tile_id(data)
        if tid is not None:
            tile_edges.setdefault(tid, []).append((u, v))

    # For each tile, check if flipping by π improves cross-tile agreement
    tile_flips = 0
    for tid, edges in tile_edges.items():
        # Find boundary edges: those with neighbors from OTHER tiles
        cost_current = 0.0
        cost_flipped = 0.0
        n_boundary = 0

        for u, v in edges:
            phi = G.edges[u, v][field]
            for nb_edge in _get_neighbor_edges(G, u, v):
                nb_data = G.edges[nb_edge]
                nb_tid = _best_tile_id(nb_data)
                if nb_tid == tid:
                    continue  # same tile — skip
                nb_phi = nb_data.get(field)
                if nb_phi is None or not np.isfinite(nb_phi):
                    continue
                # Cross-tile pair: compare distances
                cost_current += abs(_wrap_angle(phi - nb_phi))
                cost_flipped += abs(_wrap_angle(phi + np.pi - nb_phi))
                n_boundary += 1

        if n_boundary >= 2 and cost_flipped < cost_current - 0.3 * n_boundary:
            # Flip all edges in this tile
            for u, v in edges:
                phi = G.edges[u, v].get(field)
                if phi is not None and np.isfinite(phi):
                    G.edges[u, v][field] = float(_wrap_angle(phi + np.pi))
                    total_flipped += 1
            tile_flips += 1
            if verbose:
                improvement = np.degrees((cost_current - cost_flipped) / n_boundary)
                print(f"  Tile {tid}: flipped {len(edges)} edges "
                      f"(boundary improvement: {improvement:.0f}°/edge)")

    if verbose and tile_flips > 0:
        print(f"  Stage 1: {tile_flips} tiles flipped")

    # --- Stage 2: Edge-level π-flip ---
    for iteration in range(max_iterations):
        n_flipped = 0
        for u, v, data in G.edges(data=True):
            phi = data.get(field)
            if phi is None or not np.isfinite(phi):
                continue

            nbr_phases = []
            for nb_edge in _get_neighbor_edges(G, u, v):
                nb_phi = G.edges[nb_edge].get(field)
                if nb_phi is not None and np.isfinite(nb_phi):
                    nbr_phases.append(nb_phi)

            if len(nbr_phases) < 2:
                continue

            nbr_arr = np.array(nbr_phases)
            nbr_mean = np.arctan2(
                np.mean(np.sin(nbr_arr)),
                np.mean(np.cos(nbr_arr)),
            )

            dist_current = abs(_wrap_angle(phi - nbr_mean))
            dist_flipped = abs(_wrap_angle(phi + np.pi - nbr_mean))

            if dist_flipped < dist_current - 0.5:  # ~30° margin
                data[field] = float(_wrap_angle(phi + np.pi))
                n_flipped += 1

        total_flipped += n_flipped
        if verbose and n_flipped > 0:
            print(f"  Stage 2, pass {iteration + 1}: {n_flipped} edges flipped")
        if n_flipped == 0:
            break

    if verbose:
        print(f"  π-flip total: {total_flipped} flips")
    return total_flipped


def run_poiseuille_simulation(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: Optional[float] = None,
    radius_attr: str = 'radius_um',
    length_attr: str = 'length_um',
    flow_attr: str = 'mean_Q',
    ridge_alpha: Optional[float] = None,
    verbose: bool = True,
) -> PoiseuilleSimulationResult:
    """
    Run Poiseuille flow simulation to infer viscosity and pressures.

    Uses σ_Q-weighted regression with Student-t robust loss.
    All vessels with finite σ_Q are included (no hard filters).
    Vessels with large uncertainty naturally get downweighted by 1/σ_Q².
    Outliers are further downweighted by Student-t loss.

    Algorithm:
    1. Include all vessels with finite σ_Q and geometry
    2. Eliminate interior pressures: P_interior = M × P_boundary
    3. Build linear system: Q_model = A × P_boundary
    4. Solve via 1/σ_Q²-weighted IRLS with Student-t weights
    5. Compute z-scores: z = (Q_meas - Q_pred) / σ_Q for outlier detection

    Poiseuille's Law: Q = G × ΔP where G = πR⁴/(8μL)

    Parameters
    ----------
    G : nx.Graph
        Vessel network with geometry and flow attributes.
        Must have σ_Q (or sigma_Q) on edges for proper weighting.
    boundary_nodes : set, optional
        Nodes where pressure is unknown (sources/sinks).
        If None, auto-detect from degree-1 nodes.
    mu_cP : float, optional
        If provided, use this viscosity. Otherwise, use default 3.0 cP.
    radius_attr, length_attr, flow_attr : str
        Edge attribute names
    ridge_alpha : float, optional
        Ridge regularization strength. Auto-determined if None.
    verbose : bool
        Print progress

    Returns
    -------
    PoiseuilleSimulationResult
        Includes z_scores dict for outlier identification.
    """
    from scipy.linalg import lstsq

    # Get list of nodes and edges with valid data
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Collect edge data - include all vessels with valid geometry and σ_Q
    edges_data = []
    n_no_sigma = 0
    n_fit_failed = 0
    n_high_rel_unc = 0
    for u, v, data in G.edges(data=True):
        # Skip vessels where NLLS profile fit failed - these have unreliable Q estimates
        if data.get('fit_success') is False:
            n_fit_failed += 1
            continue

        # Skip vessels with rel_uncertainty > 100% - measurement is meaningless
        rel_unc = data.get('rel_uncertainty', 0)
        if rel_unc is not None and rel_unc > 1.0:
            n_high_rel_unc += 1
            continue

        R = data.get(radius_attr)
        L = data.get(length_attr)
        Q = data.get(flow_attr)

        # Try alternate attribute names
        if R is None:
            R = data.get('radius')
            if R is not None:
                R = R * PX_SIZE_UM  # Convert px to um if needed
        if L is None:
            L = data.get('length')
            if L is not None:
                L = L * PX_SIZE_UM

        if R is None or L is None or not np.isfinite(R) or not np.isfinite(L):
            continue
        if R <= 0 or L <= 0:
            continue

        # Get σ_Q (measurement uncertainty) - this is the key for weighting
        # Check multiple attribute names for compatibility
        sigma_Q = data.get('sigma_Q')
        if sigma_Q is None or not np.isfinite(sigma_Q) or sigma_Q <= 0:
            sigma_Q = data.get('sigma_Q_nL_s')  # Reduced graph convention
        if sigma_Q is None or not np.isfinite(sigma_Q) or sigma_Q <= 0:
            sigma_Q = data.get('sigma_Q_total')  # Legacy
        if sigma_Q is None or not np.isfinite(sigma_Q) or sigma_Q <= 0:
            n_no_sigma += 1
            sigma_Q = None  # Will use MAD fallback

        # Get start_junction and end_junction for consistent sign convention
        start_j = data.get('start_junction')
        end_j = data.get('end_junction')

        # If junctions not available, fall back to (u, v) but this may cause sign issues
        if start_j is None or end_j is None:
            start_j, end_j = u, v

        # Reconstruct signed flow from magnitude and direction
        if Q is not None and np.isfinite(Q):
            flow_source = data.get('flow_source')
            flow_sink = data.get('flow_sink')

            if flow_source is not None and flow_sink is not None:
                # Determine sign from flow_source/flow_sink vs start_j/end_j
                if flow_source == start_j and flow_sink == end_j:
                    Q_signed = abs(Q)
                elif flow_source == end_j and flow_sink == start_j:
                    Q_signed = -abs(Q)
                else:
                    Q_signed = abs(Q)
            else:
                Q_signed = Q

            # Filter: skip vessels where |Q| < σ_Q (rel_uncertainty > 100%)
            # These measurements are too noisy to contribute meaningful information
            if sigma_Q is not None and abs(Q_signed) < sigma_Q:
                Q_signed = None  # Treat as unmeasured
        else:
            Q_signed = None

        # Compute conductance factor (without μ): g_0 = πR⁴/(8L)
        # Full conductance is g = g_0 / μ
        R_m = R * 1e-6  # μm → m
        L_m = L * 1e-6  # μm → m
        g0 = np.pi * R_m**4 / (8 * L_m)  # m³/s/Pa (at μ=1 Pa·s)
        g0_nL = g0 * 1e12  # m³ → nL (1 m³ = 1e12 nL)

        edges_data.append({
            'u': u, 'v': v,  # Original graph edge for lookups
            'start_j': start_j, 'end_j': end_j,  # Consistent with Q_signed convention
            'start_idx': node_to_idx.get(start_j),
            'end_idx': node_to_idx.get(end_j),
            'R': R, 'L': L,
            'g0': g0_nL,  # Conductance factor (g = g0/μ)
            'Q_signed': Q_signed,  # Signed: positive = flow from start_j to end_j
            'sigma_Q': sigma_Q,  # Per-vessel uncertainty from profile fit (nL/s)
        })

    if not edges_data:
        raise ValueError("No edges with valid radius/length data")

    n_with_Q = sum(1 for e in edges_data if e['Q_signed'] is not None)
    n_with_sigma = sum(1 for e in edges_data if e['Q_signed'] is not None and e['sigma_Q'] is not None)
    if verbose:
        print(f"  {len(edges_data)} edges with geometry")
        print(f"  {n_with_Q} edges with measured flow")
        print(f"  {n_with_sigma}/{n_with_Q} with σ_Q uncertainty ({100*n_with_sigma/max(1,n_with_Q):.0f}%)")
        if n_no_sigma > 0:
            print(f"  ({n_no_sigma} vessels without σ_Q will use MAD-estimated uncertainty)")
        if n_fit_failed > 0:
            print(f"  ({n_fit_failed} vessels with fit_success=False excluded from regression)")
        if n_high_rel_unc > 0:
            print(f"  ({n_high_rel_unc} vessels with σ_Q > |Q| excluded from regression)")

    # Auto-detect boundary nodes if not provided
    # First try explicitly marked nodes, fall back to degree-1 if none found
    if boundary_nodes is None:
        boundary_nodes = set()
        for n in nodes:
            # First: include nodes explicitly marked as source/sink
            if G.nodes[n].get('boundary_type') is not None:
                boundary_nodes.add(n)

        if len(boundary_nodes) >= 2:
            if verbose:
                print(f"  Found {len(boundary_nodes)} boundary nodes (from boundary_type attribute)")
        else:
            # Fallback: use degree-1 nodes if no explicit boundaries
            boundary_nodes = set(n for n in nodes if G.degree(n) == 1)
            if verbose:
                print(f"  No explicit boundaries marked, using {len(boundary_nodes)} degree-1 nodes as boundaries")
                print(f"  (Note: some may be segmentation artifacts - mark boundaries manually for best results)")

    # Separate internal and boundary nodes
    boundary_list = list(boundary_nodes)
    internal_nodes = [n for n in nodes if n not in boundary_nodes]

    if len(boundary_list) < 2:
        raise ValueError(f"Need at least 2 boundary nodes, got {len(boundary_list)}")

    # Reference node: first boundary node, P = 0
    ref_node = boundary_list[0]
    free_boundary = boundary_list[1:]  # Pressures to solve for

    if verbose:
        print(f"  Reference node: {ref_node} (P=0)")
        print(f"  {len(free_boundary)} free boundary pressures")
        print(f"  {len(internal_nodes)} internal nodes (eliminated)")

    # Build node index mappings
    internal_idx = {n: i for i, n in enumerate(internal_nodes)}
    boundary_idx = {n: i for i, n in enumerate(free_boundary)}
    n_internal = len(internal_nodes)
    n_free_boundary = len(free_boundary)

    # Build Laplacian submatrices (at μ=1)
    # L_ii: internal-internal, L_ib: internal-boundary
    L_ii = np.zeros((n_internal, n_internal))
    L_ib = np.zeros((n_internal, n_free_boundary))

    for e in edges_data:
        u, v, g0 = e['u'], e['v'], e['g0']

        # For each node pair, add conductance contributions
        for node_a, node_b in [(u, v), (v, u)]:
            if node_a in internal_nodes:
                i = internal_idx[node_a]
                L_ii[i, i] += g0  # Diagonal

                if node_b in internal_nodes:
                    j = internal_idx[node_b]
                    L_ii[i, j] -= g0
                elif node_b in free_boundary:
                    j = boundary_idx[node_b]
                    L_ib[i, j] -= g0
                # If node_b is ref_node, contribution is to RHS (but we set P_ref=0)

    # Solve for interior pressures: P_interior = M × P_boundary
    # From: L_ii × P_i + L_ib × P_b = 0
    # So: P_i = -L_ii^(-1) × L_ib × P_b = M × P_b
    if n_internal > 0:
        try:
            M = -np.linalg.solve(L_ii, L_ib)
        except np.linalg.LinAlgError:
            if verbose:
                print("  Warning: Singular Laplacian, using pseudoinverse")
            M = -np.linalg.pinv(L_ii) @ L_ib
    else:
        M = np.zeros((0, n_free_boundary))

    # Build A matrix: Q_model = A × P_boundary (at μ=1)
    # For each measured edge: Q_signed = g0 × (P_start - P_end)
    # Using start_j/end_j ensures sign convention matches Q_signed
    measured_edges = [e for e in edges_data if e['Q_signed'] is not None]
    n_meas = len(measured_edges)
    A = np.zeros((n_meas, n_free_boundary))
    Q_meas_vec = np.zeros(n_meas)

    for idx, e in enumerate(measured_edges):
        start_j, end_j, g0 = e['start_j'], e['end_j'], e['g0']
        Q_meas_vec[idx] = e['Q_signed']  # Use signed value (matches start→end convention)

        # Q_signed = g0 × (P_start - P_end)
        # Express P_start and P_end in terms of P_boundary

        for node, sign in [(start_j, +1), (end_j, -1)]:
            if node == ref_node:
                pass  # P = 0, no contribution
            elif node in free_boundary:
                j = boundary_idx[node]
                A[idx, j] += sign * g0
            elif node in internal_nodes:
                i = internal_idx[node]
                A[idx, :] += sign * g0 * M[i, :]

    # Solve using robust IRLS (Iteratively Reweighted Least Squares) with Huber weights
    # This downweights outliers automatically while handling ill-conditioned systems
    if verbose:
        print(f"  Solving {n_meas} × {n_free_boundary} robust IRLS regression...")

    # SVD analysis - diagnose ill-conditioning
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    cond = s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf

    # Column norms - how much each boundary affects measurements
    col_norms = np.linalg.norm(A, axis=0)
    invisible_threshold = 0.01 * np.max(col_norms)
    invisible_mask = col_norms < invisible_threshold
    n_invisible = np.sum(invisible_mask)
    effective_rank = np.sum(s > 0.01 * s[0])

    if verbose:
        print(f"\n  === SVD DIAGNOSTIC ===")
        print(f"  Condition number: {cond:.1e}")
        print(f"  Singular values (top 5): {s[:5]}")
        print(f"  Singular values (bottom 5): {s[-5:]}")

        # How many singular values are "effectively zero"?
        threshold = 1e-10 * s[0]
        n_small = np.sum(s < threshold)
        print(f"  Near-zero singular values (< 1e-10 × max): {n_small} / {len(s)}")

        if n_invisible > 0:
            print(f"  'Invisible' boundary nodes (weak effect on flows): {n_invisible}")
            invisible_nodes = [free_boundary[j] for j in np.where(invisible_mask)[0]]
            print(f"    Nodes: {invisible_nodes[:10]}{'...' if len(invisible_nodes) > 10 else ''}")

        print(f"  Effective rank (s > 1% of max): {effective_rank} / {len(s)}")
        print(f"  ======================\n")

        # Note: SVD diagnostic plot disabled for interactive use
        # Call _plot_svd_diagnostic(s, col_norms, free_boundary, A, cond, effective_rank) manually if needed

    # Determine ridge regularization
    ATA = A.T @ A
    ATQ = A.T @ Q_meas_vec
    trace_scale = np.trace(ATA) / n_free_boundary

    if ridge_alpha is not None:
        ridge_lambda = ridge_alpha * trace_scale
        if verbose:
            print(f"  Using ridge α = {ridge_alpha:.2e} (λ = {ridge_lambda:.2e})")
    elif cond > 1e10:
        ridge_lambda = 1e-4 * trace_scale
        if verbose:
            print(f"  Using strong ridge regularization (λ = {ridge_lambda:.2e})")
    elif cond > 1e6:
        ridge_lambda = 1e-6 * trace_scale
        if verbose:
            print(f"  Using moderate ridge regularization (λ = {ridge_lambda:.2e})")
    else:
        ridge_lambda = 1e-10 * trace_scale
        if verbose:
            print(f"  Well-conditioned, minimal regularization (λ = {ridge_lambda:.2e})")

    # ==========================================================================
    # Student-t robust regression with per-vessel σ_Q when available
    # Falls back to MAD-estimated σ for vessels without uncertainty estimates
    # ==========================================================================

    # Student-t degrees of freedom (ν=4 gives heavier tails than Gaussian)
    nu = 4.0

    # Build per-vessel sigma array from profile-fit uncertainties
    # Use sigma_Q from covariance propagation when available
    sigma_Q_vec = np.zeros(n_meas)
    n_with_sigma_Q = 0
    for idx, e in enumerate(measured_edges):
        sigma_Q = e.get('sigma_Q')
        if sigma_Q is not None and np.isfinite(sigma_Q) and sigma_Q > 0:
            sigma_Q_vec[idx] = sigma_Q
            n_with_sigma_Q += 1
        else:
            sigma_Q_vec[idx] = 0  # Mark for MAD fallback

    # Apply floors to sigma_Q:
    # 1. Relative floor (20% of |Q|) - prevents vessels from claiming unrealistically low relative uncertainty
    #    Calibration: Kirchhoff z-scores suggest raw uncertainties underestimated by ~2.4×
    # 2. Absolute floor (0.01 nL/s) - prevents tiny-flow vessels from having σ_Q → 0
    # sigma_Q_total = sqrt(sigma_Q_raw² + max(0.20*|Q|, 0.01)²)
    REL_SIGMA_FLOOR = 0.20  # 20% minimum relative uncertainty
    ABS_SIGMA_FLOOR = 0.01  # Absolute minimum σ_Q in nL/s
    n_floored = 0
    n_abs_floored = 0
    for idx, e in enumerate(measured_edges):
        Q_signed = e['Q_signed']
        if Q_signed is not None and sigma_Q_vec[idx] > 0:
            sigma_floor_rel = REL_SIGMA_FLOOR * abs(Q_signed)
            sigma_floor = max(sigma_floor_rel, ABS_SIGMA_FLOOR)  # Use larger of relative or absolute
            sigma_raw = sigma_Q_vec[idx]
            sigma_Q_vec[idx] = np.sqrt(sigma_raw**2 + sigma_floor**2)
            # Count how many were floored
            if sigma_floor > sigma_raw:
                n_floored += 1
            if sigma_floor_rel < ABS_SIGMA_FLOOR:
                n_abs_floored += 1

    # For vessels without sigma_Q, we'll use MAD-based estimate
    use_per_vessel_sigma = n_with_sigma_Q > 0.5 * n_meas  # Use per-vessel if >50% have it

    if verbose:
        print(f"  Per-vessel σ_Q available: {n_with_sigma_Q}/{n_meas} ({100*n_with_sigma_Q/n_meas:.0f}%)")
        if use_per_vessel_sigma:
            valid_sigmas = sigma_Q_vec[sigma_Q_vec > 0]
            print(f"  σ_Q range (with floor): {np.min(valid_sigmas):.4f} - {np.max(valid_sigmas):.3f} nL/s "
                  f"(median: {np.median(valid_sigmas):.3f})")
            if n_floored > 0:
                print(f"  Floor applied to {n_floored} vessels (floor > raw σ_Q)")
            if n_abs_floored > 0:
                print(f"  Absolute floor ({ABS_SIGMA_FLOOR} nL/s) dominated for {n_abs_floored} vessels (tiny flows)")
        else:
            print(f"  Using MAD-based σ (too few vessels with σ_Q)")

    # Start with OLS solution for P
    P_boundary_scaled = np.linalg.solve(ATA + ridge_lambda * np.eye(n_free_boundary), ATQ)

    # IRLS with Student-t weights
    max_irls_iter = 30
    tol = 1e-5

    for irls_iter in range(max_irls_iter):
        # Compute residuals
        Q_pred_scaled = A @ P_boundary_scaled
        residuals = Q_meas_vec - Q_pred_scaled

        # Build sigma vector
        if use_per_vessel_sigma:
            # Use per-vessel sigma_Q where available
            sigma = sigma_Q_vec.copy()
            # For vessels without sigma_Q, use MAD-based estimate from residuals
            missing_mask = sigma <= 0
            if np.any(missing_mask):
                mad = np.median(np.abs(residuals - np.median(residuals)))
                sigma_fallback = mad / 0.6745
                sigma_fallback = max(sigma_fallback, 1e-6)
                sigma[missing_mask] = sigma_fallback
        else:
            # Use constant MAD-based estimate
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma_scalar = mad / 0.6745
            sigma_scalar = max(sigma_scalar, 1e-6)
            sigma = np.full(n_meas, sigma_scalar)

        # Student-t weights: w = (ν + 1) / (ν + z²) where z = r/σ
        student_weights = student_t_weights(residuals, sigma, nu)

        # Weighted least squares for P
        # Full weight = student_weight / σ² (inverse variance × Student-t downweighting)
        if use_per_vessel_sigma:
            full_weights = student_weights / (sigma**2)
        else:
            # With constant σ, σ² cancels in the weighted problem
            full_weights = student_weights

        W = np.diag(full_weights)
        ATWA = A.T @ W @ A
        ATWQ = A.T @ W @ Q_meas_vec
        P_new = np.linalg.solve(ATWA + ridge_lambda * np.eye(n_free_boundary), ATWQ)

        # Check convergence
        if np.max(np.abs(P_new - P_boundary_scaled)) < tol * (np.max(np.abs(P_boundary_scaled)) + 1e-10):
            if verbose:
                print(f"  Student-t IRLS converged in {irls_iter + 1} iterations")
            P_boundary_scaled = P_new
            break
        P_boundary_scaled = P_new
    else:
        if verbose:
            print(f"  Student-t IRLS reached max iterations ({max_irls_iter})")

    # Final residuals and weights
    Q_pred_scaled = A @ P_boundary_scaled
    residuals = Q_meas_vec - Q_pred_scaled
    student_weights_final = student_t_weights(residuals, sigma, nu)

    # Compute z-scores: z = (Q_meas - Q_pred) / σ_Q for outlier detection
    z_scores_vec = residuals / sigma  # Per-vessel standardized residual

    if verbose:
        print(f"\n  === STUDENT-T ROBUST REGRESSION ===")
        if use_per_vessel_sigma:
            valid_sigmas = sigma_Q_vec[sigma_Q_vec > 0]
            print(f"  Weighting: per-vessel 1/σ_Q² ({n_with_sigma_Q} vessels with σ_Q)")
            print(f"  σ_Q median: {np.median(valid_sigmas):.3f} nL/s")
        else:
            mad = np.median(np.abs(residuals - np.median(residuals)))
            print(f"  Weighting: MAD-based σ = {mad / 0.6745:.4f} nL/s")
        print(f"  Student-t ν = {nu:.1f}")
        print(f"  |z| > 2: {np.sum(np.abs(z_scores_vec) > 2)} vessels")
        print(f"  |z| > 3: {np.sum(np.abs(z_scores_vec) > 3)} vessels")
        print(f"  ====================================\n")

    rmse_scaled = np.sqrt(np.mean(residuals ** 2))

    # Store weights for result
    outlier_weights = student_weights_final

    # Identify outliers: Student-t weight < 0.3 (heavily downweighted)
    outlier_mask = student_weights_final < 0.3
    n_outliers = np.sum(outlier_mask)

    if verbose:
        print(f"  Identified {n_outliers} outliers (Student-t weight < 0.3)")

    # Record outlier edges
    outlier_edges = []
    for idx in np.where(outlier_mask)[0]:
        e = measured_edges[idx]
        outlier_edges.append({
            'edge': (e['u'], e['v']),
            'Q_meas': e['Q_signed'],
            'Q_pred': Q_pred_scaled[idx],
            'residual': residuals[idx],
            'sigma': sigma[idx],
            'sigma_Q_from_fit': e.get('sigma_Q'),  # Per-vessel uncertainty if available
            'sigma_source': 'profile_fit' if (e.get('sigma_Q') is not None and
                                               e.get('sigma_Q') > 0) else 'MAD_fallback',
            'standardized_residual': np.abs(residuals[idx]) / sigma[idx],
            'student_weight': student_weights_final[idx],
            'snr_db': e.get('snr_db'),
            'chi2_reduced': e.get('chi2_reduced'),
            'length_px': e.get('length_px'),
        })

    if verbose:
        print(f"  Residual RMSE (at μ=1): {rmse_scaled:.4f}")

    # Predict flows at μ=1
    Q_pred_mu1 = A @ P_boundary_scaled

    # Use fixed μ for chick embryo blood (typical value ~3 cP for 50-100 μm vessels)
    if mu_cP is None:
        mu_fit = 3.0  # cP - established value for chick embryo blood
        if verbose:
            print(f"  Using default μ = {mu_fit:.3f} cP (chick embryo blood)")
    else:
        mu_fit = mu_cP
        if verbose:
            print(f"  Using fixed μ = {mu_fit:.3f} cP")

    # Scale pressures by μ (in Pa·s = cP/1000)
    mu_Pa_s = mu_fit * 1e-3
    P_boundary_final = P_boundary_scaled * mu_Pa_s

    # Compute interior pressures
    if n_internal > 0:
        P_interior = M @ P_boundary_final
    else:
        P_interior = np.array([])

    # Build full pressure dict
    P_all = np.zeros(len(nodes))
    P_all[node_to_idx[ref_node]] = 0.0
    for i, n in enumerate(free_boundary):
        P_all[node_to_idx[n]] = P_boundary_final[i]
    for i, n in enumerate(internal_nodes):
        P_all[node_to_idx[n]] = P_interior[i]

    # Compute direction agreement (diagnostic - with signed regression, should be high)
    n_agree, n_disagree = 0, 0
    for e in edges_data:
        if e['Q_signed'] is None:
            continue
        start_j, end_j = e['start_j'], e['end_j']
        if start_j not in node_to_idx or end_j not in node_to_idx:
            continue

        P_start = P_all[node_to_idx[start_j]]
        P_end = P_all[node_to_idx[end_j]]
        mean_Q = e['Q_signed']

        # Measured: mean_Q > 0 means flow from start to end
        # Pressure: P_start > P_end means flow from start to end
        measured_dir = np.sign(mean_Q)
        pressure_dir = np.sign(P_start - P_end)

        if measured_dir == pressure_dir:
            n_agree += 1
        else:
            n_disagree += 1

    if n_agree + n_disagree > 0:
        agreement = n_agree / (n_agree + n_disagree)
        if verbose:
            print(f"  Direction agreement: {agreement:.1%} ({n_agree}/{n_agree + n_disagree})")

    # Predict all flows with final μ using start_j/end_j convention
    # Q_predicted has same sign convention as Q_signed: positive = start→end
    predicted_Q = {}
    for e in edges_data:
        start_j, end_j = e['start_j'], e['end_j']
        if start_j not in node_to_idx or end_j not in node_to_idx:
            continue
        g = e['g0'] / mu_Pa_s  # g = g0 / μ
        dP = P_all[node_to_idx[start_j]] - P_all[node_to_idx[end_j]]
        predicted_Q[(e['u'], e['v'])] = g * dP  # Stored with (u,v) key for compatibility

    # Compute fit metrics using signed values (since we did signed regression)
    measured_Q = {}
    Q_meas_list, Q_pred_list = [], []
    for e in edges_data:
        if e['Q_signed'] is not None:
            measured_Q[(e['u'], e['v'])] = e['Q_signed']  # Signed
            Q_meas_list.append(e['Q_signed'])
            Q_pred_list.append(predicted_Q.get((e['u'], e['v']), 0))

    Q_meas_arr = np.array(Q_meas_list)
    Q_pred_arr = np.array(Q_pred_list)

    if len(Q_meas_arr) > 0:
        ss_res = np.sum((Q_meas_arr - Q_pred_arr)**2)
        ss_tot = np.sum((Q_meas_arr - np.mean(Q_meas_arr))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((Q_meas_arr - Q_pred_arr)**2))
    else:
        r_squared = np.nan
        rmse = np.nan

    # Build result
    node_pressures = {n: P_all[node_to_idx[n]] for n in nodes}
    boundary_pressures = {n: P_all[node_to_idx[n]] for n in boundary_nodes}

    # Build z-scores and sigma_Q_used dictionaries
    z_scores_dict = {}
    sigma_Q_used_dict = {}
    for idx, e in enumerate(measured_edges):
        edge_key = (e['u'], e['v'])
        z_scores_dict[edge_key] = z_scores_vec[idx]
        sigma_Q_used_dict[edge_key] = sigma[idx]

    return PoiseuilleSimulationResult(
        mu_cP=mu_fit,
        node_pressures=node_pressures,
        predicted_Q=predicted_Q,
        measured_Q=measured_Q,
        r_squared=r_squared,
        rmse=rmse,
        boundary_nodes=boundary_nodes,
        boundary_pressures=boundary_pressures,
        outlier_edges=outlier_edges if outlier_edges else None,
        outlier_weights=outlier_weights,
        z_scores=z_scores_dict,
        sigma_Q_used=sigma_Q_used_dict,
    )


def print_simulation_report(result: PoiseuilleSimulationResult, top_n: int = 10):
    """Print Poiseuille simulation results."""
    print("\n" + "=" * 70)
    print("POISEUILLE FLOW SIMULATION")
    print("=" * 70)

    print(f"\nInferred viscosity: μ = {result.mu_cP:.3f} cP")
    print(f"  (Blood viscosity typically 3-4 cP)")

    print(f"\nFit quality (signed flows):")
    print(f"  R² = {result.r_squared:.4f}")
    print(f"  RMSE = {result.rmse:.4f} nL/s")

    print(f"\nBoundary pressures ({len(result.boundary_nodes)} nodes):")
    sorted_bp = sorted(result.boundary_pressures.items(), key=lambda x: x[1], reverse=True)
    for node, P in sorted_bp[:top_n]:
        print(f"  Node {node}: P = {P:.2f} Pa")

    # Show worst predictions (using signed values)
    if result.measured_Q and result.predicted_Q:
        errors = []
        for edge, Q_meas in result.measured_Q.items():
            Q_pred = result.predicted_Q.get(edge, 0)  # Signed
            errors.append((edge, Q_meas, Q_pred, abs(Q_pred - Q_meas)))

        errors.sort(key=lambda x: x[3], reverse=True)

        print(f"\nTop {top_n} prediction errors (signed, nL/s, + = start→end):")
        print(f"  {'Edge':>12}  {'Measured':>10}  {'Predicted':>10}  {'Error':>10}")
        for edge, Q_m, Q_p, err in errors[:top_n]:
            print(f"  {str(edge):>12}  {Q_m:>+10.4f}  {Q_p:>+10.4f}  {err:>10.4f}")

    # Z-score summary
    if result.z_scores:
        z_arr = np.array(list(result.z_scores.values()))
        z_abs = np.abs(z_arr)
        z_std = np.std(z_arr)
        print(f"\nZ-score summary (z = (Q_meas - Q_pred) / sigma_Q):")
        print(f"  std(z): {z_std:.2f}  (should be ~1.0 if σ_Q calibrated)")
        print(f"  Median |z|: {np.median(z_abs):.2f}  (should be ~0.67 if σ_Q calibrated)")
        print(f"  |z| > 1: {np.sum(z_abs > 1)} vessels ({100*np.sum(z_abs > 1)/len(z_abs):.0f}%)  (expect ~32%)")
        print(f"  |z| > 2: {np.sum(z_abs > 2)} vessels ({100*np.sum(z_abs > 2)/len(z_abs):.0f}%)  (expect ~5%)")
        print(f"  |z| > 3: {np.sum(z_abs > 3)} vessels ({100*np.sum(z_abs > 3)/len(z_abs):.0f}%)  (expect ~0.3%)")

    # Report outliers from robust regression (based on z-scores)
    if result.outlier_edges:
        print(f"\nOutliers (Student-t weight < 0.3): {len(result.outlier_edges)} vessels")
        print(f"  {'Edge':>12}  {'Measured':>10}  {'Predicted':>10}  {'z-score':>8}  {'sigma_Q':>10}  {'src':>10}")
        for o in sorted(result.outlier_edges, key=lambda x: abs(x.get('standardized_residual', 0)), reverse=True)[:top_n]:
            z = o.get('standardized_residual', 0)
            sigma = o.get('sigma', np.nan)
            src = o.get('sigma_source', 'unknown')[:8]
            print(f"  {str(o['edge']):>12}  {o['Q_meas']:>+10.4f}  {o['Q_pred']:>+10.4f}  {z:>+8.2f}  {sigma:>10.4f}  {src:>10}")

    print("=" * 70)


def analyze_direction_mismatch(result: PoiseuilleSimulationResult):
    """
    Detailed analysis of direction mismatch between measured and predicted flows.
    """
    print("\n" + "=" * 70)
    print("DIRECTION MISMATCH ANALYSIS")
    print("=" * 70)

    if not result.measured_Q or not result.predicted_Q:
        print("No flow data to analyze")
        return

    # Collect paired data
    Q_meas_list = []
    Q_pred_list = []
    edges = []

    for edge, Q_meas in result.measured_Q.items():
        Q_pred = result.predicted_Q.get(edge)
        if Q_pred is not None:
            Q_meas_list.append(Q_meas)
            Q_pred_list.append(Q_pred)
            edges.append(edge)

    Q_meas = np.array(Q_meas_list)
    Q_pred = np.array(Q_pred_list)
    n_edges = len(Q_meas)

    if n_edges == 0:
        print("No matching edges found")
        return

    # Direction agreement
    same_sign = np.sign(Q_meas) == np.sign(Q_pred)
    n_same = np.sum(same_sign)
    n_opposite = n_edges - n_same

    print(f"\n1. Direction Agreement:")
    print(f"   Same direction:     {n_same:4d} ({100*n_same/n_edges:.1f}%)")
    print(f"   Opposite direction: {n_opposite:4d} ({100*n_opposite/n_edges:.1f}%)")

    # Correlation analysis
    corr = np.corrcoef(Q_meas, Q_pred)[0, 1]
    corr_flipped = np.corrcoef(Q_meas, -Q_pred)[0, 1]

    print(f"\n2. Correlation Analysis:")
    print(f"   Correlation(Q_meas, Q_pred):  {corr:+.4f}")
    print(f"   Correlation(Q_meas, -Q_pred): {corr_flipped:+.4f}")

    if corr_flipped > corr:
        print(f"   → Flipping predicted signs would IMPROVE correlation!")
        print(f"   → This suggests a global sign inversion in the pressure field")

    # R² analysis
    ss_res = np.sum((Q_meas - Q_pred)**2)
    ss_res_flipped = np.sum((Q_meas - (-Q_pred))**2)
    ss_tot = np.sum((Q_meas - np.mean(Q_meas))**2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_flipped = 1 - ss_res_flipped / ss_tot if ss_tot > 0 else 0

    print(f"\n3. R² Analysis:")
    print(f"   R² (as-is):   {r2:+.4f}")
    print(f"   R² (flipped): {r2_flipped:+.4f}")

    if r2_flipped > r2:
        print(f"   → Flipping would improve R² by {r2_flipped - r2:.4f}")

    # Magnitude analysis (ignoring sign)
    Q_meas_mag = np.abs(Q_meas)
    Q_pred_mag = np.abs(Q_pred)

    ss_res_mag = np.sum((Q_meas_mag - Q_pred_mag)**2)
    ss_tot_mag = np.sum((Q_meas_mag - np.mean(Q_meas_mag))**2)
    r2_mag = 1 - ss_res_mag / ss_tot_mag if ss_tot_mag > 0 else 0

    corr_mag = np.corrcoef(Q_meas_mag, Q_pred_mag)[0, 1]

    print(f"\n4. Magnitude Analysis (ignoring direction):")
    print(f"   Correlation(|Q_meas|, |Q_pred|): {corr_mag:+.4f}")
    print(f"   R² on magnitudes: {r2_mag:+.4f}")

    # Show worst direction mismatches (large flows in opposite directions)
    direction_errors = []
    for i, edge in enumerate(edges):
        if np.sign(Q_meas[i]) != np.sign(Q_pred[i]):
            # Opposite direction - magnitude indicates severity
            direction_errors.append((edge, Q_meas[i], Q_pred[i], abs(Q_meas[i])))

    direction_errors.sort(key=lambda x: x[3], reverse=True)

    print(f"\n5. Worst Direction Mismatches (opposite sign, sorted by |Q_meas|):")
    print(f"   {'Edge':>12}  {'Q_meas':>10}  {'Q_pred':>10}  {'Comment'}")
    for edge, qm, qp, _ in direction_errors[:10]:
        comment = "LARGE" if abs(qm) > 10 else ""
        print(f"   {str(edge):>12}  {qm:>+10.2f}  {qp:>+10.2f}  {comment}")

    # Summary recommendation
    print(f"\n6. Recommendation:")
    if corr_flipped > 0.5 and corr < 0:
        print("   ⚠️  Strong negative correlation suggests GLOBAL SIGN ERROR")
        print("   The pressure field appears to be inverted (P should be -P)")
        print("   Check: is the reference node at high or low pressure?")
    elif n_opposite > n_same:
        print("   ⚠️  Majority of flows have opposite direction")
        print("   Possible causes:")
        print("   - Sign convention mismatch in start_junction/end_junction")
        print("   - Pressure reference node selection")
    else:
        print("   Direction agreement is reasonable")
        print("   Mismatches may be due to:")
        print("   - Local deviations from Poiseuille flow")
        print("   - Measurement noise")
        print("   - Network topology constraints")

    print("=" * 70)


def compute_kirchhoff_residuals(
    G: nx.Graph,
    min_rel_uncertainty: float = 1.0,
    merge_unmeasured: bool = True,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Compute Kirchhoff residuals at each junction node.

    For each node, computes:
        residual = Σ Q_in - Σ Q_out
        σ_residual = √(Σ σ_Q²)
        z = residual / σ_residual

    Only includes vessels where |Q| ≥ σ_Q (rel_uncertainty < 100%).

    Parameters
    ----------
    G : nx.Graph
        Graph with flow measurements (mean_Q, sigma_Q, start_junction, end_junction)
    min_rel_uncertainty : float
        Exclude vessels with rel_uncertainty > this value from Kirchhoff sums (default 1.0 = 100%)
    merge_unmeasured : bool
        If True, merge junction nodes connected by short vessels (path_length_px < MIN_PATH_LENGTH_PX)
        into "super-nodes" for Kirchhoff checking (default True). Short vessels cannot produce
        reliable kymograph measurements, so the endpoints are treated as a single node.
    verbose : bool
        Print summary statistics

    Returns
    -------
    dict : {node_id: {'residual': float, 'sigma': float, 'z_score': float,
                      'n_edges': int, 'n_measured': int, 'class': str,
                      'super_node': tuple or None, 'is_representative': bool}}
        - class: 'A' (all measured), 'B' (partial), 'C' (cannot check)
        - super_node: tuple of merged node IDs, or None if not merged
        - is_representative: True if this node represents a super-node group
    """
    results = {}

    # Identify boundary nodes: any node with boundary_type set (source, sink, edge, pending)
    # These are network endpoints where we don't know the external flow
    boundary_nodes = set(n for n in G.nodes()
                         if G.nodes[n].get('boundary_type') is not None)

    # Get all junction nodes (degree >= 3), excluding boundary nodes
    # Kirchhoff's law only applies to interior junctions
    junction_nodes = [n for n in G.nodes() if G.degree(n) >= 3
                      and n not in boundary_nodes]
    junction_set = set(junction_nodes)

    # ========== Super-node merging ==========
    # Find grey edges (unmeasured or filtered) between junction nodes
    # and merge those junction pairs into super-nodes

    # Union-Find data structure for merging
    parent = {n: n for n in junction_nodes}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Track nodes connected to cyan (>100% uncertainty) vessels for exclusion
    cyan_connected_nodes = set()

    # Track nodes connected to boundary nodes (source/sink) - exclude from class A
    boundary_connected_nodes = set()

    if merge_unmeasured:
        # Find grey vessels (trace chains through degree-2 nodes)
        # Track which edges we've already processed as part of a vessel
        processed_edges = set()
        n_grey_merged = 0
        n_cyan_vessels = 0

        for u, v, data in G.edges(data=True):
            edge_key = tuple(sorted([u, v]))
            if edge_key in processed_edges:
                continue

            # Trace the full vessel chain from this edge
            # Walk in both directions through degree-2 nodes to find junction endpoints
            def walk_to_junction(start, other):
                """Walk from start away from other until hitting a junction (degree != 2)."""
                curr, prev = start, other
                while G.degree(curr) == 2:
                    nbrs = [n for n in G.neighbors(curr) if n != prev]
                    if len(nbrs) != 1:
                        break
                    prev, curr = curr, nbrs[0]
                return curr

            # Get vessel endpoints (junction nodes at each end)
            endpoint1 = walk_to_junction(u, v)
            endpoint2 = walk_to_junction(v, u)

            # Mark all edges in this vessel as processed
            # Collect edges by walking from endpoint1 to endpoint2
            curr = endpoint1
            path = [curr]
            while curr != endpoint2:
                found = False
                for nbr in G.neighbors(curr):
                    if nbr not in path:
                        edge_key = tuple(sorted([curr, nbr]))
                        processed_edges.add(edge_key)
                        curr = nbr
                        path.append(curr)
                        found = True
                        break
                if not found:
                    break  # Safety: shouldn't happen

            # Skip if endpoints are the same (single-node vessel or loop)
            if endpoint1 == endpoint2:
                continue

            # Get actual centerline path length from edge attributes (computed during analysis)
            # This is the resampled path length, not the node-to-node distance
            path_length_px = data.get('path_length_px')
            if path_length_px is None or not np.isfinite(path_length_px):
                # Fall back to node-to-node distance if path_length_px not available
                path_length_px = 0.0
                for i in range(len(path) - 1):
                    n1, n2 = path[i], path[i+1]
                    x1, y1 = G.nodes[n1].get('x', 0), G.nodes[n1].get('y', 0)
                    x2, y2 = G.nodes[n2].get('x', 0), G.nodes[n2].get('y', 0)
                    path_length_px += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Check if this vessel is grey/unmeasured (check any edge in chain)
            mean_Q = data.get('mean_Q')
            sigma_Q = data.get('sigma_Q')

            # Check if vessel is too short or couldn't be properly trimmed
            is_short = path_length_px < MIN_PATH_LENGTH_PX  # Too short for reliable kymograph
            was_trimmed = data.get('was_trimmed', True)  # Assume trimmed if not set
            is_untrimmed = not was_trimmed  # Couldn't be trimmed (junction contamination)

            is_cyan = False  # >100% uncertainty

            # Check for cyan vessels (>100% uncertainty) - these exclude nodes from Kirchhoff
            if mean_Q is not None and np.isfinite(mean_Q) and abs(mean_Q) > 0:
                if sigma_Q is not None and np.isfinite(sigma_Q):
                    rel_unc = sigma_Q / abs(mean_Q)
                    if rel_unc > 1.0:  # >100% uncertainty = cyan
                        is_cyan = True

            # Short/untrimmed vessels ALWAYS trigger super-node merging (even if also cyan)
            # This must happen BEFORE cyan handling since untrimmed vessels have junction
            # contamination and their flow values are unreliable regardless of uncertainty
            if is_short or is_untrimmed:
                # Short vessels OR untrimmed vessels trigger super-node merging
                # - Short: path_length < MIN_PATH_LENGTH_PX (too short even after trimming)
                # - Untrimmed: couldn't be trimmed due to short arc length (junction contamination)
                if endpoint1 in junction_set and endpoint2 in junction_set:
                    union(endpoint1, endpoint2)
                    n_grey_merged += 1

            if is_cyan:
                # Mark endpoint junctions for exclusion from Kirchhoff
                if endpoint1 in junction_set:
                    cyan_connected_nodes.add(endpoint1)
                if endpoint2 in junction_set:
                    cyan_connected_nodes.add(endpoint2)
                n_cyan_vessels += 1

            # Check if vessel connects to boundary node - exclude the junction endpoint from class A
            if endpoint1 in boundary_nodes and endpoint2 in junction_set:
                boundary_connected_nodes.add(endpoint2)
            if endpoint2 in boundary_nodes and endpoint1 in junction_set:
                boundary_connected_nodes.add(endpoint1)

        if verbose:
            print(f"  Boundary nodes (degree-1 or source/sink): {len(boundary_nodes)}")
            print(f"  Interior junctions (degree >= 3): {len(junction_nodes)}")
            if n_grey_merged > 0:
                print(f"  Merged {n_grey_merged} short/untrimmed vessels between junction nodes")
            if n_cyan_vessels > 0:
                print(f"  Found {n_cyan_vessels} cyan vessels (>100% uncertainty)")
                print(f"  Excluding {len(cyan_connected_nodes)} junction nodes connected to cyan vessels")
            if boundary_connected_nodes:
                print(f"  Junctions connected to boundaries: {len(boundary_connected_nodes)} (excluded from class A)")

    # Build super-node groups
    super_node_groups = {}  # representative -> list of members
    for node in junction_nodes:
        rep = find(node)
        if rep not in super_node_groups:
            super_node_groups[rep] = []
        super_node_groups[rep].append(node)

    # Sort members within each group for consistency
    for rep in super_node_groups:
        super_node_groups[rep] = tuple(sorted(super_node_groups[rep]))

    n_class_a = 0
    n_class_b = 0
    n_class_c = 0
    n_class_x = 0  # Excluded due to cyan vessel connection
    n_super_nodes = 0

    # Process each super-node group
    for representative, members in super_node_groups.items():
        is_super = len(members) > 1
        if is_super:
            n_super_nodes += 1

        members_set = set(members)

        # Track if this group has cyan connections (for reporting, but don't exclude)
        has_cyan_connection = any(m in cyan_connected_nodes for m in members)

        # Track if this group is connected to boundary nodes - exclude from class A
        has_boundary_connection = any(m in boundary_connected_nodes for m in members)

        # Collect all external edges (not between super-node members)
        external_edges = []
        internal_edges = []
        for node in members:
            for u, v, data in G.edges(node, data=True):
                other = v if u == node else u
                edge_key = tuple(sorted([u, v]))
                if other in members_set:
                    # Internal edge - skip for Kirchhoff but track
                    if edge_key not in [tuple(sorted([e[0], e[1]])) for e in internal_edges]:
                        internal_edges.append((u, v, data, node))
                else:
                    # External edge - include in Kirchhoff
                    external_edges.append((u, v, data, node))

        n_edges = len(external_edges)
        if n_edges == 0:
            # No external edges (isolated super-node)
            for node in members:
                results[node] = {
                    'residual': np.nan,
                    'sigma': np.nan,
                    'z_score': np.nan,
                    'n_edges': 0,
                    'n_measured': 0,
                    'n_filtered': 0,
                    'n_cyan_filtered': 0,
                    'class': 'C',
                    'super_node': members if is_super else None,
                    'is_representative': node == representative,
                    'has_cyan_connection': has_cyan_connection,
                    'has_boundary_connection': has_boundary_connection,
                }
            n_class_c += 1
            continue

        # Collect flows: positive = into super-node, negative = out
        flows = []
        sigmas = []
        flow_details = []
        n_measured = 0
        n_filtered = 0
        n_cyan_filtered = 0  # Track cyan (>100% uncertainty) separately

        for u, v, data, from_node in external_edges:
            mean_Q = data.get('mean_Q')
            sigma_Q = data.get('sigma_Q')

            if mean_Q is None or not np.isfinite(mean_Q):
                continue

            # Filter by relative uncertainty
            if sigma_Q is not None and np.isfinite(sigma_Q) and sigma_Q > 0:
                rel_unc = sigma_Q / abs(mean_Q) if abs(mean_Q) > 0 else np.inf
                if rel_unc > min_rel_uncertainty:
                    n_filtered += 1
                    if rel_unc > 1.0:  # Cyan = >100% uncertainty
                        n_cyan_filtered += 1
                    continue
            else:
                continue

            # Determine flow direction relative to this node (which is in the super-node)
            start_j = data.get('start_junction')
            end_j = data.get('end_junction')

            if start_j is not None and end_j is not None:
                # mean_Q > 0 means flow from start_j to end_j
                if mean_Q > 0:
                    # Flow goes start_j → end_j
                    if end_j in members_set:
                        Q_relative = abs(mean_Q)  # INTO super-node
                    else:
                        Q_relative = -abs(mean_Q)  # OUT OF super-node
                else:
                    # Flow goes end_j → start_j
                    if start_j in members_set:
                        Q_relative = abs(mean_Q)  # INTO super-node
                    else:
                        Q_relative = -abs(mean_Q)  # OUT OF super-node
            else:
                # Fallback
                if mean_Q > 0:
                    if v in members_set:
                        Q_relative = abs(mean_Q)
                    else:
                        Q_relative = -abs(mean_Q)
                else:
                    if u in members_set:
                        Q_relative = abs(mean_Q)
                    else:
                        Q_relative = -abs(mean_Q)

            flows.append(Q_relative)
            sigmas.append(sigma_Q)
            flow_details.append((mean_Q, sigma_Q, Q_relative, start_j, end_j))
            n_measured += 1

        # Determine class and compute residual
        if n_measured == 0:
            node_class = 'C'
            n_class_c += 1
            for node in members:
                results[node] = {
                    'residual': np.nan,
                    'sigma': np.nan,
                    'z_score': np.nan,
                    'n_edges': n_edges,
                    'n_measured': 0,
                    'n_filtered': n_filtered,
                    'n_cyan_filtered': n_cyan_filtered,
                    'class': node_class,
                    'super_node': members if is_super else None,
                    'is_representative': node == representative,
                    'has_cyan_connection': has_cyan_connection,
                    'has_boundary_connection': has_boundary_connection,
                }
        else:
            # Boundary-connected nodes cannot be class A (unknown boundary flow)
            if has_boundary_connection:
                node_class = 'C'
                n_class_c += 1
            elif n_measured == n_edges:
                node_class = 'A'
                n_class_a += 1
            else:
                node_class = 'B'
                n_class_b += 1

            residual = sum(flows)
            sigma_residual = np.sqrt(sum(s**2 for s in sigmas))
            z_score = residual / sigma_residual if sigma_residual > 0 else np.nan

            # Debug: print details for high z-score nodes (only class A)
            if verbose and node_class == 'A' and abs(z_score) > 2:
                label = f"Super-node {members}" if is_super else f"Node {representative}"
                print(f"\n  {label} |z|={abs(z_score):.2f} (class {node_class}):")
                print(f"    {n_edges} external edges, {n_measured} measured, {n_filtered} filtered")
                if is_super:
                    print(f"    {len(internal_edges)} internal (grey) edges ignored")
                for i, (mQ, sQ, Qrel, sj, ej) in enumerate(flow_details):
                    direction = "IN" if Qrel > 0 else "OUT"
                    print(f"    [{i}] Q={mQ:+.1f}±{sQ:.1f} → {direction} {abs(Qrel):.1f} (start={sj}, end={ej})")
                print(f"    Σflows={residual:.2f}, σ={sigma_residual:.2f}, z={z_score:.2f}")

            for node in members:
                results[node] = {
                    'residual': residual,
                    'sigma': sigma_residual,
                    'z_score': z_score,
                    'n_edges': n_edges,
                    'n_measured': n_measured,
                    'n_filtered': n_filtered,
                    'n_cyan_filtered': n_cyan_filtered,
                    'class': node_class,
                    'super_node': members if is_super else None,
                    'is_representative': node == representative,
                    'has_cyan_connection': has_cyan_connection,
                    'has_boundary_connection': has_boundary_connection,
                }

    # Store on graph nodes
    for node, data in results.items():
        for key, val in data.items():
            G.nodes[node][f'kirchhoff_{key}'] = val

    if verbose:
        # Summary statistics - only count unique super-node groups (via representative)
        # Only include class A nodes for z-score summary (all vessels measured, interior nodes)
        z_vals_A = [r['z_score'] for r in results.values()
                    if r.get('class') == 'A' and np.isfinite(r['z_score'])
                    and r.get('is_representative', True)]

        # Count groups with cyan vessels filtered
        n_with_cyan = sum(1 for r in results.values()
                         if r.get('is_representative', True) and r.get('n_cyan_filtered', 0) > 0)

        # Count boundary-connected groups
        n_boundary = sum(1 for r in results.values()
                        if r.get('is_representative', True) and r.get('has_boundary_connection', False))

        n_groups = len(super_node_groups)
        print(f"\nKirchhoff Check ({len(junction_nodes)} junction nodes → {n_groups} groups):")
        if n_super_nodes > 0:
            print(f"  Super-nodes merged: {n_super_nodes} (grey vessels between junctions)")
        print(f"  Class A (all measured, interior): {n_class_a}")
        print(f"  Class B (partial): {n_class_b}")
        print(f"  Class C (cannot check): {n_class_c}")
        if n_boundary > 0:
            print(f"    (includes {n_boundary} boundary-connected nodes)")
        if n_with_cyan > 0:
            print(f"  Groups with cyan vessels filtered: {n_with_cyan}")
        if z_vals_A:
            z_arr = np.abs(np.array(z_vals_A))
            print(f"  Class A |z| summary (n={len(z_vals_A)}): median={np.median(z_arr):.2f}, "
                  f"|z|>1: {np.sum(z_arr > 1)}, |z|>2: {np.sum(z_arr > 2)}, |z|>3: {np.sum(z_arr > 3)}")

    return results


def _get_flow_data(edge_data: dict) -> Tuple[float, float]:
    """Get mean_Q and sigma_Q from edge data, handling both naming conventions.

    Returns (mean_Q, sigma_Q) or (nan, nan) if not available.
    """
    # Try standard names first (full graph)
    mean_Q = edge_data.get('mean_Q')
    sigma_Q = edge_data.get('sigma_Q')

    # Try reduced graph names
    if mean_Q is None:
        mean_Q = edge_data.get('mean_Q_nL_s')
    if sigma_Q is None:
        sigma_Q = edge_data.get('sigma_Q_nL_s')

    # Convert to float, handling None
    if mean_Q is None:
        mean_Q = np.nan
    if sigma_Q is None:
        sigma_Q = np.nan

    return float(mean_Q), float(sigma_Q)


def check_two_end_consistency(
    G: nx.Graph,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Check two-end consistency for unmeasured (cyan) vessels.

    This is the ONLY non-circular validation for unmeasured vessels:
    - For each cyan vessel, check if BOTH endpoints have all other edges measured
    - If so, each endpoint can independently infer what Q must be from Kirchhoff
    - If the two inferred values agree (|z| < 2), the vessel is consistent

    This avoids the circularity of one-sided inference (just tells you what Q "must be"
    if Kirchhoff holds - tautological) and doesn't assume Poiseuille physics.

    Parameters
    ----------
    G : nx.Graph
        Graph with flow data (should be reduced graph for efficiency)
    verbose : bool
        Print summary

    Returns
    -------
    dict with:
        'vessels': list of dicts with two-end consistency info
        'n_checkable': count of vessels where both ends can infer
        'n_consistent': count where both-end inferences agree (|z| < 2)
        'n_inconsistent': count where both-end inferences disagree (|z| >= 2)
    """
    # Detect if this is a reduced graph (all edges connect junctions directly)
    # On reduced graph: edges have 'mean_Q_nL_s' attribute (from export_simulation)
    # On full graph: edges have 'mean_Q' attribute
    sample_edge = next(iter(G.edges(data=True)), (None, None, {}))
    is_reduced_graph = 'mean_Q_nL_s' in sample_edge[2] or 'flow_from_node' in sample_edge[2]

    # Find junction nodes (non-boundary)
    # On reduced graph: all nodes are junctions
    # On full graph: junctions have degree != 2
    if is_reduced_graph:
        junction_nodes = {n for n in G.nodes()
                          if G.nodes[n].get('boundary_type') not in ('source', 'sink')}
    else:
        junction_nodes = {n for n in G.nodes() if G.degree(n) != 2
                          and G.nodes[n].get('boundary_type') not in ('source', 'sink')}

    # Helper to trace vessel to junction endpoint (only needed on full graph)
    def walk_to_junction(start, other):
        if is_reduced_graph:
            return start  # On reduced graph, all nodes are already junctions
        curr, prev = start, other
        while G.degree(curr) == 2:
            nbrs = [n for n in G.neighbors(curr) if n != prev]
            if len(nbrs) != 1:
                break
            prev, curr = curr, nbrs[0]
        return curr

    # Track processed vessels
    processed_edges = set()
    results = []

    for u, v, data in G.edges(data=True):
        edge_key = tuple(sorted([u, v]))
        if edge_key in processed_edges:
            continue

        # Find vessel endpoints (junctions)
        endpoint1 = walk_to_junction(u, v)
        endpoint2 = walk_to_junction(v, u)

        # Mark all edges in this vessel as processed
        curr = endpoint1
        path = [curr]
        while curr != endpoint2:
            found = False
            for nbr in G.neighbors(curr):
                if nbr not in path:
                    processed_edges.add(tuple(sorted([curr, nbr])))
                    curr = nbr
                    path.append(curr)
                    found = True
                    break
            if not found:
                break

        if endpoint1 == endpoint2:
            continue
        if endpoint1 not in junction_nodes or endpoint2 not in junction_nodes:
            continue

        # Check if this vessel is unmeasured (cyan)
        mean_Q, sigma_Q = _get_flow_data(data)

        is_measured = False
        if np.isfinite(mean_Q) and abs(mean_Q) > 0:
            if np.isfinite(sigma_Q):
                rel_unc = sigma_Q / abs(mean_Q)
                if rel_unc <= 1.0:  # Not cyan
                    is_measured = True

        if is_measured:
            continue  # Skip measured vessels

        # Try to infer from BOTH endpoints
        inferences = {}

        for junction, other_junction in [(endpoint1, endpoint2), (endpoint2, endpoint1)]:
            if junction not in junction_nodes:
                continue

            # Count measured edges at this junction (excluding this vessel)
            n_other_edges = 0
            n_other_measured = 0
            sum_Q = 0.0
            sum_sigma_sq = 0.0

            for nbr in G.neighbors(junction):
                # Find other endpoint of this edge's vessel
                # On reduced graph: neighbor IS the other junction
                # On full graph: may need to walk through degree-2 nodes
                if is_reduced_graph:
                    other_end = nbr
                elif G.degree(nbr) == 2:
                    other_end = walk_to_junction(nbr, junction)
                else:
                    other_end = nbr

                if other_end == other_junction:
                    # This is the vessel we're checking - skip
                    continue

                n_other_edges += 1

                edge_data = G.edges[junction, nbr]
                other_Q, other_sigma = _get_flow_data(edge_data)

                if np.isfinite(other_Q) and abs(other_Q) > 0:
                    if np.isfinite(other_sigma):
                        other_rel_unc = other_sigma / abs(other_Q)
                        if other_rel_unc <= 1.0:
                            n_other_measured += 1

                            # Determine flow direction relative to junction
                            # Try both naming conventions for junction info
                            start_j = edge_data.get('start_junction') or edge_data.get('flow_from_node')
                            end_j = edge_data.get('end_junction') or edge_data.get('flow_to_node')

                            if start_j is not None and end_j is not None:
                                if other_Q > 0:
                                    Q_relative = abs(other_Q) if end_j == junction else -abs(other_Q)
                                else:
                                    Q_relative = abs(other_Q) if start_j == junction else -abs(other_Q)
                            else:
                                Q_relative = other_Q  # Fallback

                            sum_Q += Q_relative
                            sum_sigma_sq += other_sigma ** 2

            # Can we infer? All OTHER edges must be measured
            if n_other_measured == n_other_edges and n_other_edges > 0:
                # Infer: Q_vessel = -sum_Q (to satisfy Kirchhoff)
                Q_inferred = -sum_Q
                sigma_inferred = np.sqrt(sum_sigma_sq)

                inferences[junction] = {
                    'Q_inferred': Q_inferred,
                    'sigma_inferred': sigma_inferred,
                    'n_other_edges': n_other_edges,
                }

        # Only report if BOTH ends can infer (the non-circular case)
        if len(inferences) != 2:
            continue

        Q1 = inferences[endpoint1]['Q_inferred']
        Q2 = inferences[endpoint2]['Q_inferred']
        s1 = inferences[endpoint1]['sigma_inferred']
        s2 = inferences[endpoint2]['sigma_inferred']

        # Consistency check: Q1 + Q2 ≈ 0
        # (flow out of A should equal flow into B, so opposite signs)
        diff = Q1 + Q2
        sigma_diff = np.sqrt(s1**2 + s2**2)
        z_consistency = diff / sigma_diff if sigma_diff > 0 else np.nan

        is_consistent = abs(z_consistency) < 2.0 if np.isfinite(z_consistency) else False

        # Compute weighted average estimate
        Q_combined = None
        sigma_combined = None
        if s1 > 0 and s2 > 0:
            w1 = 1 / s1**2
            w2 = 1 / s2**2
            Q_combined = (w1 * Q1 - w2 * Q2) / (w1 + w2)  # Note minus for Q2 (opposite direction)
            sigma_combined = np.sqrt(1 / (w1 + w2))

        results.append({
            'endpoints': (endpoint1, endpoint2),
            'edge': (u, v),
            'Q1': Q1,
            'Q2': Q2,
            'sigma1': s1,
            'sigma2': s2,
            'z_consistency': z_consistency,
            'is_consistent': is_consistent,
            'Q_combined': Q_combined,
            'sigma_combined': sigma_combined,
        })

    # Summary statistics
    n_checkable = len(results)
    n_consistent = sum(1 for r in results if r['is_consistent'])
    n_inconsistent = n_checkable - n_consistent

    if verbose:
        print(f"\nTwo-End Consistency Check for Cyan Vessels:")
        graph_type = "reduced" if is_reduced_graph else "full"
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({graph_type})")
        print(f"  Junction nodes: {len(junction_nodes)}")

        # Count measured vs cyan
        n_measured = 0
        n_cyan = 0
        n_no_data = 0
        for u, v, data in G.edges(data=True):
            mean_Q, sigma_Q = _get_flow_data(data)
            if np.isfinite(mean_Q) and abs(mean_Q) > 0 and np.isfinite(sigma_Q):
                rel_unc = sigma_Q / abs(mean_Q)
                if rel_unc <= 1.0:
                    n_measured += 1
                else:
                    n_cyan += 1
            else:
                n_no_data += 1
        print(f"  Vessels: {n_measured} measured, {n_cyan} cyan (>100% unc), {n_no_data} no data")

        print(f"  Vessels checkable (both ends can infer): {n_checkable}")
        if n_checkable > 0:
            print(f"  Consistent (|z| < 2): {n_consistent}")
            print(f"  Inconsistent (|z| >= 2): {n_inconsistent}")

            # Show details
            for r in results:
                z = r['z_consistency']
                status = "OK" if r['is_consistent'] else "FAIL"
                Q1, Q2 = r['Q1'], r['Q2']
                s1, s2 = r['sigma1'], r['sigma2']
                Q_est = r['Q_combined']
                print(f"    {r['endpoints']}: Q_A={Q1:+.1f}±{s1:.1f}, Q_B={Q2:+.1f}±{s2:.1f} → z={z:.2f} [{status}]")
                if Q_est is not None:
                    print(f"      Combined estimate: Q={Q_est:+.2f} ± {r['sigma_combined']:.2f} nL/s")

    return {
        'vessels': results,
        'n_checkable': n_checkable,
        'n_consistent': n_consistent,
        'n_inconsistent': n_inconsistent,
    }


def sweep_ridge_regularization(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    flow_attr: str = 'mean_Q',
    alpha_range: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Sweep ridge regularization values and analyze effect on fit quality.

    Parameters
    ----------
    G : nx.Graph
        Graph with flow data
    boundary_nodes : set, optional
        Boundary nodes (auto-detect if None)
    flow_attr : str
        Edge attribute for measured flow
    alpha_range : list of float, optional
        Ridge alpha values to try. Default: logspace from 1e-10 to 1e-1
    verbose : bool
        Print progress

    Returns
    -------
    dict with keys: 'alpha', 'r_squared', 'slope', 'rmse', 'direction_match'
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    if alpha_range is None:
        alpha_range = np.logspace(-10, -1, 20)

    results = {
        'alpha': [],
        'r_squared': [],
        'slope': [],
        'intercept': [],
        'rmse': [],
        'mae': [],
        'direction_match': [],
    }

    print("\n" + "=" * 70)
    print("RIDGE REGULARIZATION SWEEP")
    print("=" * 70)

    for alpha in alpha_range:
        try:
            result = run_poiseuille_simulation(
                G,
                boundary_nodes=boundary_nodes,
                flow_attr=flow_attr,
                ridge_alpha=alpha,
                verbose=False,
            )

            # Collect paired measurements
            Q_meas = []
            Q_pred = []
            for edge, qm in result.measured_Q.items():
                qp = result.predicted_Q.get(edge)
                if qp is not None and np.isfinite(qm) and np.isfinite(qp):
                    Q_meas.append(qm)
                    Q_pred.append(qp)

            if len(Q_meas) < 2:
                continue

            Q_meas = np.array(Q_meas)
            Q_pred = np.array(Q_pred)

            # Statistics
            slope, intercept = np.polyfit(Q_meas, Q_pred, 1)
            r_value, _ = stats.pearsonr(Q_meas, Q_pred)
            r_squared = r_value ** 2
            rmse = np.sqrt(np.mean((Q_meas - Q_pred) ** 2))
            mae = np.mean(np.abs(Q_meas - Q_pred))
            same_sign = np.sum(np.sign(Q_meas) == np.sign(Q_pred))
            direction_match = 100 * same_sign / len(Q_meas)

            results['alpha'].append(alpha)
            results['r_squared'].append(r_squared)
            results['slope'].append(slope)
            results['intercept'].append(intercept)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['direction_match'].append(direction_match)

            if verbose:
                print(f"  α={alpha:.2e}: R²={r_squared:.3f}, slope={slope:.3f}, "
                      f"RMSE={rmse:.3f}, dir_match={direction_match:.1f}%")

        except Exception as e:
            if verbose:
                print(f"  α={alpha:.2e}: FAILED - {e}")

    if not results['alpha']:
        print("No successful runs!")
        return results

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # R² vs alpha
    ax = axes[0, 0]
    ax.semilogx(results['alpha'], results['r_squared'], 'b.-')
    ax.set_xlabel('Ridge α')
    ax.set_ylabel('R²')
    ax.set_title('Fit Quality vs Regularization')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=max(results['r_squared']), color='r', linestyle='--', alpha=0.5)

    # Slope vs alpha
    ax = axes[0, 1]
    ax.semilogx(results['alpha'], results['slope'], 'g.-')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect (slope=1)')
    ax.set_xlabel('Ridge α')
    ax.set_ylabel('Slope')
    ax.set_title('Prediction Bias vs Regularization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE vs alpha
    ax = axes[1, 0]
    ax.semilogx(results['alpha'], results['rmse'], 'r.-')
    ax.set_xlabel('Ridge α')
    ax.set_ylabel('RMSE (nL/s)')
    ax.set_title('Error vs Regularization')
    ax.grid(True, alpha=0.3)

    # Direction match vs alpha
    ax = axes[1, 1]
    ax.semilogx(results['alpha'], results['direction_match'], 'm.-')
    ax.set_xlabel('Ridge α')
    ax.set_ylabel('Direction Match (%)')
    ax.set_title('Direction Agreement vs Regularization')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.show(block=False)

    # Find optimal alpha (maximize R² while keeping slope close to 1)
    best_idx = np.argmax(results['r_squared'])
    print(f"\nBest R² = {results['r_squared'][best_idx]:.4f} at α = {results['alpha'][best_idx]:.2e}")
    print(f"  Slope = {results['slope'][best_idx]:.3f}")
    print(f"  Direction match = {results['direction_match'][best_idx]:.1f}%")

    # Find alpha with slope closest to 1
    slope_deviation = np.abs(np.array(results['slope']) - 1.0)
    best_slope_idx = np.argmin(slope_deviation)
    print(f"\nSlope closest to 1.0 at α = {results['alpha'][best_slope_idx]:.2e}")
    print(f"  Slope = {results['slope'][best_slope_idx]:.3f}")
    print(f"  R² = {results['r_squared'][best_slope_idx]:.4f}")

    print("=" * 70)

    return results


def write_simulation_to_graph(
    G: nx.Graph,
    result: PoiseuilleSimulationResult,
    verbose: bool = True,
) -> int:
    """
    Write Poiseuille simulation results back to graph nodes and edges.

    Writes to nodes:
        - pressure_Pa: Simulated pressure at node
        - is_boundary_sim: True if node is a boundary in simulation

    Writes to edges:
        - Q_predicted: Predicted flow from simulation (nL/s, signed)
        - Q_residual: measured - predicted (nL/s)
        - Q_residual_rel: relative residual (Q_residual / |Q_measured|)
        - z_score: standardized residual (Q_residual / σ_Q)

    Also stores simulation metadata in graph.graph:
        - sim_mu_cP: Inferred viscosity
        - sim_r_squared: R² of fit
        - sim_rmse: RMSE in nL/s

    Args:
        G: Graph to write results to (modified in place)
        result: PoiseuilleSimulationResult from run_poiseuille_simulation
        verbose: Print summary

    Returns:
        Number of edges with predictions written
    """
    # Write node pressures
    for node, pressure in result.node_pressures.items():
        if G.has_node(node):
            G.nodes[node]['pressure_Pa'] = pressure
            G.nodes[node]['is_boundary_sim'] = node in result.boundary_nodes

    # Write edge predictions and residuals
    # Q_predicted sign determined by pressure gradient at junctions:
    # positive = flow from start_junction to end_junction (high P to low P)
    n_written = 0
    for (u, v), Q_pred_raw in result.predicted_Q.items():
        # Try both edge directions (undirected graph)
        if G.has_edge(u, v):
            edge_key = (u, v)
        elif G.has_edge(v, u):
            edge_key = (v, u)
        else:
            continue

        edge_data = G.edges[edge_key]

        # Determine Q_predicted sign from pressure gradient (not from internal u,v ordering)
        start_j = edge_data.get('start_junction')
        end_j = edge_data.get('end_junction')

        # Get magnitude from simulation
        Q_mag = abs(Q_pred_raw)

        # Determine sign from pressure gradient at vessel junctions
        if start_j is not None and end_j is not None:
            P_start = result.node_pressures.get(start_j, 0.0)
            P_end = result.node_pressures.get(end_j, 0.0)
            # Flow goes from high pressure to low pressure
            if P_start > P_end:
                # Flow from start to end: positive Q
                Q_pred = Q_mag
            else:
                # Flow from end to start: negative Q
                Q_pred = -Q_mag
        else:
            # No junction info, use raw value
            Q_pred = Q_pred_raw

        edge_data['Q_predicted'] = Q_pred

        # Compute residual vs measured flow
        # Try mean_Q first (full graph), then mean_Q_nL_s with sign reconstruction (reduced graph)
        Q_meas = edge_data.get('mean_Q')

        if Q_meas is None or not np.isfinite(Q_meas):
            # Try reduced graph convention: mean_Q_nL_s (magnitude) + flow_source/flow_sink (direction)
            Q_mag = edge_data.get('mean_Q_nL_s')
            if Q_mag is not None and np.isfinite(Q_mag):
                flow_source = edge_data.get('flow_source')
                flow_sink = edge_data.get('flow_sink')
                if flow_source is not None and flow_sink is not None and start_j is not None and end_j is not None:
                    # Reconstruct signed flow
                    if flow_source == start_j and flow_sink == end_j:
                        Q_meas = abs(Q_mag)  # Flow goes start→end: positive
                    elif flow_source == end_j and flow_sink == start_j:
                        Q_meas = -abs(Q_mag)  # Flow goes end→start: negative
                    else:
                        Q_meas = abs(Q_mag)  # Default to positive

        if Q_meas is not None and np.isfinite(Q_meas):
            residual = Q_meas - Q_pred
            edge_data['Q_residual'] = residual
            if abs(Q_meas) > 1e-6:
                edge_data['Q_residual_rel'] = residual / abs(Q_meas)
            else:
                edge_data['Q_residual_rel'] = np.nan

        # Write z_score if available from simulation result
        if result.z_scores is not None:
            z = result.z_scores.get((u, v))
            if z is not None:
                edge_data['z_score'] = z
            elif result.z_scores.get((v, u)) is not None:
                edge_data['z_score'] = result.z_scores.get((v, u))

        # Write sigma_Q_used if available
        if result.sigma_Q_used is not None:
            sigma = result.sigma_Q_used.get((u, v))
            if sigma is not None:
                edge_data['sigma_Q_sim'] = sigma
            elif result.sigma_Q_used.get((v, u)) is not None:
                edge_data['sigma_Q_sim'] = result.sigma_Q_used.get((v, u))

        n_written += 1

    # Store simulation metadata on graph
    G.graph['sim_mu_cP'] = result.mu_cP
    G.graph['sim_r_squared'] = result.r_squared
    G.graph['sim_rmse'] = result.rmse
    G.graph['sim_n_boundary'] = len(result.boundary_nodes)

    if verbose:
        print(f"Wrote simulation results to graph:")
        print(f"  Node pressures: {len(result.node_pressures)}")
        print(f"  Edge predictions: {n_written}")
        print(f"  μ = {result.mu_cP:.3f} cP, R² = {result.r_squared:.4f}")

    return n_written


def run_and_write_simulation(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: Optional[float] = None,
    flow_attr: str = 'mean_Q',
    verbose: bool = True,
) -> PoiseuilleSimulationResult:
    """
    Run Poiseuille simulation and write results to graph.

    Convenience function that combines run_poiseuille_simulation and
    write_simulation_to_graph. Uses σ_Q-weighted robust IRLS.

    Args:
        G: Graph with geometry and measured flows
        boundary_nodes: Boundary nodes (auto-detect if None)
        mu_cP: Fixed viscosity, or None to use default 3.0 cP
        flow_attr: Edge attribute containing measured flow
        verbose: Print progress

    Returns:
        PoiseuilleSimulationResult with z_scores for outlier identification
    """
    if verbose:
        print("\nRunning Poiseuille flow simulation (sigma_Q-weighted IRLS)...")

    result = run_poiseuille_simulation(
        G,
        boundary_nodes=boundary_nodes,
        mu_cP=mu_cP,
        flow_attr=flow_attr,
        verbose=verbose,
    )

    write_simulation_to_graph(G, result, verbose=verbose)

    return result


# =============================================================================
# Bayesian Inference for Network Flow (Level 1: Steady-State)
# =============================================================================

@dataclass
class BayesianSimulationResult:
    """Results from Bayesian Poiseuille flow simulation with uncertainty."""
    # Point estimates (MAP)
    mu_cP: float
    node_pressures: Dict[int, float]  # MAP estimates
    predicted_Q: Dict[Tuple[int, int], float]
    measured_Q: Dict[Tuple[int, int], float]

    # Uncertainties (from Laplace approximation)
    P_boundary_std: Dict[int, float]  # Std dev of boundary pressures
    P_boundary_cov: np.ndarray  # Full covariance matrix
    Q_pred_std: Dict[Tuple[int, int], float]  # Uncertainty in predictions
    mu_std: Optional[float]  # Std dev of viscosity (if inferred)

    # Inferred measurement noise (if infer_sigma=True)
    sigma_Q_inferred: Optional[float]  # Global noise scale or σ_base (nL/s)
    sigma_Q_std: Optional[float]  # Uncertainty in noise scale
    sigma_rel_inferred: Optional[float]  # Relative noise coefficient (if per-vessel model)
    sigma_rel_std: Optional[float]  # Uncertainty in relative noise
    sigma_Q_per_vessel: Optional[Dict[Tuple[int, int], float]]  # Per-vessel σ values

    # 95% credible intervals
    P_boundary_ci95: Dict[int, Tuple[float, float]]
    Q_pred_ci95: Dict[Tuple[int, int], Tuple[float, float]]

    # Fit quality
    r_squared: float
    rmse: float
    chi2_reduced: float
    log_posterior: float

    # Model info
    boundary_nodes: Set[int]
    ref_node: int
    n_parameters: int
    n_measurements: int


def _prepare_bayesian_data(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    radius_attr: str = 'radius_um',
    length_attr: str = 'length_um',
    flow_attr: str = 'mean_Q_nL_s',
    sigma_attr: str = 'sigma_Q_nL_s',
    max_chi2_reduced: Optional[float] = None,
    min_snr_percentile: float = 50.0,
    min_vessel_length_px: float = 20.0,
    max_Q_nL_s: Optional[float] = None,
    default_sigma_fraction: float = 0.5,
    infer_sigma: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Prepare data structures for Bayesian inference.

    Returns dict with all arrays needed for forward model and likelihood.

    Parameters
    ----------
    max_chi2_reduced : float, optional
        Maximum chi2_reduced for profile fit quality filter.
    min_snr_percentile : float
        Only include vessels with SNR above this percentile (default 50 = median).
        Set to 0 to disable SNR filtering.
    min_vessel_length_px : float
        Minimum vessel length in pixels (default 20). Short vessels are excluded.
        Set to 0 to disable length filtering.
    max_Q_nL_s : float, optional
        Maximum |Q| in nL/s. Vessels with larger flows are excluded as outliers.
        Set to None to disable (default).
    default_sigma_fraction : float
        If sigma_Q not available, use this fraction of |Q| as uncertainty.
        Default 0.5 (50% uncertainty). Ignored if infer_sigma=True.
    infer_sigma : bool
        If True, sigma_Q will be inferred from the data as a global parameter.
        In this mode, sigma_Q values from the data are ignored.
    """
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Compute SNR threshold from percentile
    snr_threshold = -np.inf
    if min_snr_percentile > 0:
        snr_values = []
        for _, _, edata in G.edges(data=True):
            snr = edata.get('snr_db')
            if snr is not None and isinstance(snr, (int, float)) and np.isfinite(snr):
                snr_values.append(float(snr))
        if snr_values:
            snr_threshold = float(np.percentile(snr_values, min_snr_percentile))
            if verbose:
                print(f"  SNR filter: >= {snr_threshold:.1f} dB ({min_snr_percentile:.0f}th percentile)")

    # Collect edge data
    n_snr_filtered = 0
    n_length_filtered = 0
    n_Q_outlier_filtered = 0
    n_fit_failed = 0
    n_high_rel_unc = 0
    edges_data = []
    for u, v, data in G.edges(data=True):
        # Skip vessels where NLLS profile fit failed - these have unreliable Q estimates
        if data.get('fit_success') is False:
            n_fit_failed += 1
            continue

        # Skip vessels with rel_uncertainty > 100% - measurement is meaningless
        rel_unc = data.get('rel_uncertainty', 0)
        if rel_unc is not None and rel_unc > 1.0:
            n_high_rel_unc += 1
            continue

        R = data.get(radius_attr)
        L = data.get(length_attr)
        Q = data.get(flow_attr)
        sigma_Q = data.get(sigma_attr, None)

        # Try alternate attribute names for radius/length
        if R is None:
            R = data.get('radius')
            if R is not None:
                R = R * PX_SIZE_UM
        if L is None:
            L = data.get('length')
            if L is not None:
                L = L * PX_SIZE_UM

        # Try alternate attribute names for flow (handles both naming conventions)
        if Q is None:
            Q = data.get('mean_Q_nL_s')
        if Q is None:
            Q = data.get('mean_Q')
        if sigma_Q is None:
            sigma_Q = data.get('sigma_Q_nL_s')
        if sigma_Q is None:
            sigma_Q = data.get('sigma_Q')

        if R is None or L is None or not np.isfinite(R) or not np.isfinite(L):
            continue
        if R <= 0 or L <= 0:
            continue

        # Get start_junction and end_junction for consistent sign convention
        # These define the reference direction: positive Q = flow from start_j to end_j
        start_j = data.get('start_junction')
        end_j = data.get('end_junction')

        # Fall back to (u, v) if junctions not available
        if start_j is None or end_j is None:
            start_j, end_j = u, v

        # Reconstruct signed flow from magnitude and direction
        # On reduced graph: flow_attr is typically POSITIVE (magnitude only)
        # flow_source/flow_sink encode the actual measured flow direction
        Q_signed = None
        if Q is not None and np.isfinite(Q):
            flow_source = data.get('flow_source')
            flow_sink = data.get('flow_sink')

            if flow_source is not None and flow_sink is not None:
                # Determine sign from flow_source/flow_sink vs start_j/end_j
                if flow_source == start_j and flow_sink == end_j:
                    # Flow goes start→end: positive
                    Q_signed = abs(Q)
                elif flow_source == end_j and flow_sink == start_j:
                    # Flow goes end→start: negative
                    Q_signed = -abs(Q)
                else:
                    # flow_source/flow_sink don't match junctions - use magnitude
                    Q_signed = abs(Q)
            else:
                # No flow direction info - assume Q is already signed correctly
                Q_signed = Q

        # Filter by chi2_reduced if specified
        if max_chi2_reduced is not None and Q_signed is not None:
            chi2 = data.get('chi2_reduced', np.nan)
            if chi2 is None or not np.isfinite(chi2) or chi2 > max_chi2_reduced:
                Q_signed = None

        # Filter by SNR percentile (matching viewer display threshold)
        if Q_signed is not None and snr_threshold > -np.inf:
            snr_db = data.get('snr_db')
            # Handle None or non-numeric values - don't filter if no SNR data
            if snr_db is not None and isinstance(snr_db, (int, float)):
                if snr_db < snr_threshold:
                    Q_signed = None
                    n_snr_filtered += 1

        # Filter by vessel length (matching viewer short vessel filter)
        if Q_signed is not None and min_vessel_length_px > 0:
            # Try 'length' (pixels) first, then 'length_um' (convert to pixels)
            vessel_length = data.get('length')
            if vessel_length is None or not isinstance(vessel_length, (int, float)):
                # Try length_um and convert to pixels
                length_um = data.get('length_um')
                if length_um is not None and isinstance(length_um, (int, float)):
                    vessel_length = length_um / PX_SIZE_UM  # Convert um to pixels
                else:
                    vessel_length = np.inf  # No length info - don't filter
            if vessel_length < min_vessel_length_px:
                Q_signed = None
                n_length_filtered += 1

        # Filter by maximum |Q| (outlier filter)
        if Q_signed is not None and max_Q_nL_s is not None:
            if abs(Q_signed) > max_Q_nL_s:
                Q_signed = None
                n_Q_outlier_filtered += 1

        # Filter by relative uncertainty: exclude if sigma_Q > |Q| (rel_unc > 100%)
        # This is the primary quality filter based on error-propagated uncertainty
        if Q_signed is not None:
            sigma_Q_raw = data.get('sigma_Q_nL_s') or data.get('sigma_Q') or data.get('sigma_mean_Q')
            if sigma_Q_raw is not None and np.isfinite(sigma_Q_raw) and sigma_Q_raw > 0:
                if sigma_Q_raw > abs(Q_signed):
                    Q_signed = None
                    n_high_rel_unc += 1

        # Handle sigma_Q based on mode
        if infer_sigma:
            # When inferring sigma globally, set placeholder (will be multiplied by inferred σ)
            # Use 1.0 as placeholder - the actual σ_Q will be inferred
            sigma_Q = 1.0
        else:
            # Try sigma_mean_Q first (total = random + systematic, calibrated values)
            # Fall back to sigma_attr for backwards compatibility with older graphs
            sigma_Q_val = data.get('sigma_mean_Q')
            if sigma_Q_val is None or not np.isfinite(sigma_Q_val) or sigma_Q_val <= 0:
                sigma_Q_val = sigma_Q  # Use value from sigma_attr

            if sigma_Q_val is None or not np.isfinite(sigma_Q_val) or sigma_Q_val <= 0:
                # Fall back to default fraction of |Q|
                if Q_signed is not None:
                    sigma_Q_val = max(default_sigma_fraction * abs(Q_signed), 0.1)
                else:
                    sigma_Q_val = 1.0
            else:
                # Ensure minimum uncertainty floor (at least 10% of |Q| or 0.1 nL/s)
                # This prevents σ=0 from dominating the likelihood
                min_sigma = max(0.1 * abs(Q_signed), 0.1) if Q_signed else 0.1
                sigma_Q_val = max(sigma_Q_val, min_sigma)

            sigma_Q = sigma_Q_val

        # Conductance factor (at μ=1 Pa·s)
        R_m = R * 1e-6  # μm → m
        L_m = L * 1e-6  # μm → m
        g0 = np.pi * R_m**4 / (8 * L_m) * 1e12  # m³ → nL (1 m³ = 1e12 nL)

        # Extract chi2_reduced for extended heteroscedastic model
        chi2_red = data.get('chi2_reduced', 1.0)
        if chi2_red is None or not np.isfinite(chi2_red):
            chi2_red = 1.0  # Default to 1.0 (good fit) if not available

        edges_data.append({
            'u': u, 'v': v,  # Original edge endpoints for graph lookups
            'start_j': start_j, 'end_j': end_j,  # Reference direction for Q_signed
            'start_idx': node_to_idx.get(start_j),
            'end_idx': node_to_idx.get(end_j),
            'R': R, 'L': L,
            'g0': g0,
            'Q_measured': Q_signed,  # Signed: positive = flow from start_j to end_j
            'sigma_Q': sigma_Q,
            'chi2_reduced': chi2_red,  # For extended heteroscedastic noise model
        })

    if not edges_data:
        raise ValueError("No edges with valid radius/length data")

    # Auto-detect boundary nodes if not provided
    # First try explicitly marked nodes, fall back to degree-1 if none found
    if boundary_nodes is None:
        boundary_nodes = set()
        for n in nodes:
            # First: include nodes explicitly marked as source/sink
            if G.nodes[n].get('boundary_type') is not None:
                boundary_nodes.add(n)

        if len(boundary_nodes) >= 2:
            if verbose:
                print(f"  Found {len(boundary_nodes)} boundary nodes (from boundary_type attribute)")
        else:
            # Fallback: use degree-1 nodes if no explicit boundaries
            boundary_nodes = set(n for n in nodes if G.degree(n) == 1)
            if verbose:
                print(f"  No explicit boundaries, using {len(boundary_nodes)} degree-1 nodes")

    boundary_list = list(boundary_nodes)
    internal_nodes = [n for n in nodes if n not in boundary_nodes]

    if len(boundary_list) < 2:
        raise ValueError(f"Need at least 2 boundary nodes, got {len(boundary_list)}")

    ref_node = boundary_list[0]
    free_boundary = boundary_list[1:]

    internal_idx = {n: i for i, n in enumerate(internal_nodes)}
    boundary_idx = {n: i for i, n in enumerate(free_boundary)}
    n_internal = len(internal_nodes)
    n_free_boundary = len(free_boundary)

    # Build Laplacian submatrices
    L_ii = np.zeros((n_internal, n_internal))
    L_ib = np.zeros((n_internal, n_free_boundary))

    for e in edges_data:
        u, v, g0 = e['u'], e['v'], e['g0']
        for node_a, node_b in [(u, v), (v, u)]:
            if node_a in internal_nodes:
                i = internal_idx[node_a]
                L_ii[i, i] += g0
                if node_b in internal_nodes:
                    j = internal_idx[node_b]
                    L_ii[i, j] -= g0
                elif node_b in free_boundary:
                    j = boundary_idx[node_b]
                    L_ib[i, j] -= g0

    # Precompute M matrix: P_internal = M @ P_boundary
    if n_internal > 0:
        try:
            M = -np.linalg.solve(L_ii, L_ib)
        except np.linalg.LinAlgError:
            M = -np.linalg.pinv(L_ii) @ L_ib
    else:
        M = np.zeros((0, n_free_boundary))

    # Build measurement arrays
    measured_edges = [e for e in edges_data if e['Q_measured'] is not None]
    n_meas = len(measured_edges)

    Q_meas = np.array([e['Q_measured'] for e in measured_edges])
    sigma_Q = np.array([e['sigma_Q'] for e in measured_edges])
    chi2_meas = np.array([e['chi2_reduced'] for e in measured_edges])

    if verbose:
        print(f"  {len(edges_data)} edges with geometry")
        print(f"  {n_meas} edges with measured flow")
        if n_snr_filtered > 0:
            print(f"    ({n_snr_filtered} excluded by SNR < {snr_threshold:.1f} dB)")
        if n_length_filtered > 0:
            print(f"    ({n_length_filtered} excluded by length < {min_vessel_length_px:.0f} px)")
        if n_Q_outlier_filtered > 0:
            print(f"    ({n_Q_outlier_filtered} excluded by |Q| > {max_Q_nL_s:.0f} nL/s)")
        if n_fit_failed > 0:
            print(f"    ({n_fit_failed} excluded by fit_success=False)")
        if n_high_rel_unc > 0:
            print(f"    ({n_high_rel_unc} excluded by σ_Q > |Q|)")
        print(f"  {len(boundary_nodes)} boundary nodes, {n_internal} internal")
        print(f"  {n_free_boundary} free pressure parameters")
        if n_meas > 0:
            if infer_sigma:
                print(f"  σ_Q will be inferred from data")
            else:
                print(f"  σ_Q range: [{sigma_Q.min():.2f}, {sigma_Q.max():.2f}] nL/s (mean {sigma_Q.mean():.2f})")

    return {
        'nodes': nodes,
        'node_to_idx': node_to_idx,
        'edges_data': edges_data,
        'measured_edges': measured_edges,
        'boundary_nodes': boundary_nodes,
        'boundary_list': boundary_list,
        'internal_nodes': internal_nodes,
        'ref_node': ref_node,
        'free_boundary': free_boundary,
        'internal_idx': internal_idx,
        'boundary_idx': boundary_idx,
        'n_internal': n_internal,
        'n_free_boundary': n_free_boundary,
        'L_ii': L_ii,
        'L_ib': L_ib,
        'M': M,
        'Q_meas': Q_meas,
        'sigma_Q': sigma_Q,
        'chi2_meas': chi2_meas,  # chi2_reduced for each measured vessel
        'n_meas': n_meas,
    }


def _forward_model(
    P_boundary: np.ndarray,
    mu_Pa_s: float,
    data: dict,
) -> np.ndarray:
    """
    Forward model: boundary pressures → predicted flows.

    Args:
        P_boundary: (n_free_boundary,) array of boundary pressures (Pa)
        mu_Pa_s: Viscosity in Pa·s
        data: Dict from _prepare_bayesian_data

    Returns:
        Q_pred: (n_meas,) predicted flows for measured edges (nL/s)
    """
    M = data['M']
    measured_edges = data['measured_edges']
    ref_node = data['ref_node']
    free_boundary = data['free_boundary']
    internal_nodes = data['internal_nodes']
    boundary_idx = data['boundary_idx']
    internal_idx = data['internal_idx']

    # Compute internal pressures
    if data['n_internal'] > 0:
        P_internal = M @ P_boundary
    else:
        P_internal = np.array([])

    # Build full pressure array
    def get_pressure(node):
        if node == ref_node:
            return 0.0
        elif node in free_boundary:
            return P_boundary[boundary_idx[node]]
        elif node in internal_nodes:
            return P_internal[internal_idx[node]]
        return 0.0

    # Compute predicted flows
    # Q_pred positive when flow goes from start_j to end_j (consistent with Q_measured sign)
    Q_pred = np.zeros(len(measured_edges))
    for idx, e in enumerate(measured_edges):
        g = e['g0'] / mu_Pa_s
        dP = get_pressure(e['start_j']) - get_pressure(e['end_j'])
        Q_pred[idx] = g * dP

    return Q_pred


def _log_likelihood(
    Q_pred: np.ndarray,
    Q_meas: np.ndarray,
    sigma_Q: np.ndarray,
    loss_type: str = 'gaussian',
    huber_delta: float = 1.5,
) -> float:
    """
    Log-likelihood with choice of loss function.

    Parameters
    ----------
    Q_pred, Q_meas, sigma_Q : arrays
        Predicted, measured flows and uncertainties
    loss_type : str
        'gaussian' (default) - standard L2 loss, sensitive to outliers
        'huber' - robust loss, downweights large residuals
    huber_delta : float
        Threshold for Huber loss (in units of sigma). Residuals below this
        are treated as Gaussian, above are treated linearly. Default 1.5.

    Returns
    -------
    float : log-likelihood (negative loss)
    """
    residuals = (Q_meas - Q_pred) / sigma_Q  # Standardized residuals

    if loss_type == 'gaussian':
        return -0.5 * np.sum(residuals**2)

    elif loss_type == 'huber':
        # Huber loss: quadratic for small residuals, linear for large
        # This downweights outliers, making the fit more robust
        abs_r = np.abs(residuals)
        losses = np.where(
            abs_r <= huber_delta,
            0.5 * residuals**2,  # Quadratic (like Gaussian)
            huber_delta * (abs_r - 0.5 * huber_delta)  # Linear
        )
        return -np.sum(losses)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def _log_prior_pressure(
    P_boundary: np.ndarray,
    P_std: float = 30000.0,  # 30 kPa - weak prior
) -> float:
    """
    Weak Gaussian prior on boundary pressures, centered at 0.

    Since only pressure differences matter physically, this is just
    a weak regularization to prevent extreme values. With σ = 30 kPa,
    pressures in the typical range (0-50 kPa) are essentially unpenalized.
    """
    # Gaussian log-pdf (up to constant): -0.5 * sum((P / σ)²)
    return -0.5 * np.sum((P_boundary / P_std)**2)


def _log_prior_mu(
    mu_cP: float,
    mu_mean: float = 3.5,
    mu_std: float = 0.5,
) -> float:
    """Normal prior on viscosity."""
    return -0.5 * ((mu_cP - mu_mean) / mu_std)**2


def bayesian_poiseuille_simulation(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: Optional[float] = None,
    infer_mu: bool = True,
    infer_sigma: bool = False,
    sigma_model: str = 'per_vessel',
    loss_type: str = 'gaussian',
    huber_delta: float = 1.5,
    radius_attr: str = 'radius_um',
    length_attr: str = 'length_um',
    flow_attr: str = 'mean_Q_nL_s',
    sigma_attr: str = 'sigma_Q_nL_s',
    max_chi2_reduced: Optional[float] = 5.0,
    min_snr_percentile: float = 50.0,
    min_vessel_length_px: float = 20.0,
    default_sigma_fraction: float = 0.5,
    prior_P_std: float = 30000.0,
    prior_mu_mean: float = 3.5,
    prior_mu_std: float = 0.5,
    prior_sigma_scale: float = 5.0,
    verbose: bool = True,
) -> BayesianSimulationResult:
    """
    Bayesian Poiseuille simulation with MAP + Laplace approximation.

    Uses optimization to find MAP estimate, then computes Hessian for
    uncertainty quantification via Laplace approximation.

    By default, uses calibrated per-vessel uncertainties (sigma_mean_Q) from
    the profile fitting pipeline. These combine random (harmonic fit) and
    systematic (profile parameters) uncertainties in quadrature, with a 20%
    velocity floor based on Kirchhoff z-score calibration.

    Optionally, measurement noise can be inferred from the data following
    Rasmussen et al. (2018), but this is no longer the default.

    Parameters
    ----------
    G : nx.Graph
        Vessel network with geometry and flow attributes
    boundary_nodes : set, optional
        Boundary nodes (auto-detect if None)
    mu_cP : float, optional
        Fixed viscosity. If None and infer_mu=True, infer from data.
    infer_mu : bool
        Whether to infer viscosity as a parameter
    infer_sigma : bool
        Whether to infer measurement noise σ_Q from the data.
        If False (default), uses sigma_mean_Q from edges (calibrated total
        uncertainty), falling back to sigma_attr or default_sigma_fraction.
        If True, noise parameters are learned from residuals.
    sigma_model : str
        Noise model when infer_sigma=True:
        - 'global': Single σ for all vessels
        - 'per_vessel': σ_i = σ_base + σ_rel * |Q_i| (like Rasmussen et al.)
    loss_type : str
        Loss function for fitting:
        - 'gaussian' (default): Standard L2 loss, sensitive to outliers
        - 'huber': Robust loss, downweights large residuals (outliers)
    huber_delta : float
        Threshold for Huber loss in units of σ. Residuals |z| < delta are
        treated quadratically, |z| > delta are treated linearly. Default 1.5.
    radius_attr, length_attr, flow_attr, sigma_attr : str
        Edge attribute names
    max_chi2_reduced : float, optional
        Filter threshold for profile fit quality (chi2_reduced < threshold).
    min_snr_percentile : float
        Only include vessels with SNR above this percentile (default 50 = median).
        Matches viewer display threshold. Set to 0 to disable.
    min_vessel_length_px : float
        Minimum vessel length in pixels (default 20). Short vessels excluded.
        Matches viewer display threshold. Set to 0 to disable.
    default_sigma_fraction : float
        If sigma_Q not available and infer_sigma=False, use this fraction
        of |Q| as uncertainty. Default 0.5 (50% uncertainty).
    prior_P_std : float
        Std dev for weak Gaussian prior on pressures (Pa). Default 30 kPa.
    prior_mu_mean, prior_mu_std : float
        Prior on viscosity (cP)
    prior_sigma_scale : float
        Scale parameter for half-Cauchy prior on σ parameters (nL/s).
        Default 5.0 nL/s is weakly informative for typical flow magnitudes.
    verbose : bool
        Print progress

    Returns
    -------
    BayesianSimulationResult
    """
    from scipy.optimize import minimize

    if verbose:
        print("\n" + "=" * 70)
        print("BAYESIAN POISEUILLE SIMULATION (MAP + Laplace)")
        print("=" * 70)

    # Run regular Poiseuille simulation first if not already done
    # This provides a good initial guess for the Bayesian optimization
    if 'sim_mu_cP' not in G.graph:
        if verbose:
            print("\n  Running standard Poiseuille simulation first for initialization...")
        run_and_write_simulation(
            G,
            boundary_nodes=boundary_nodes,
            mu_cP=None,  # Infer
            flow_attr=flow_attr,
            verbose=verbose,
        )

    # Prepare data
    data = _prepare_bayesian_data(
        G,
        boundary_nodes=boundary_nodes,
        radius_attr=radius_attr,
        length_attr=length_attr,
        flow_attr=flow_attr,
        sigma_attr=sigma_attr,
        max_chi2_reduced=max_chi2_reduced,
        min_snr_percentile=min_snr_percentile,
        min_vessel_length_px=min_vessel_length_px,
        max_Q_nL_s=None,  # No outlier filter - rely on sigma_Q
        default_sigma_fraction=default_sigma_fraction,
        infer_sigma=infer_sigma,
        verbose=verbose,
    )

    n_free_boundary = data['n_free_boundary']
    Q_meas = data['Q_meas']
    sigma_Q_data = data['sigma_Q']  # May be placeholder 1.0 if infer_sigma=True
    n_meas = data['n_meas']
    abs_Q_meas = np.abs(Q_meas)  # For per-vessel sigma model

    # Determine if we're inferring mu
    if mu_cP is not None:
        infer_mu = False
        mu_fixed = mu_cP * 1e-3  # Convert to Pa·s

    # Validate sigma_model
    use_per_vessel = infer_sigma and sigma_model == 'per_vessel'

    # Parameter vector: [P_boundary_1, ..., P_boundary_{n-1}, (log_mu), (log_sigma_base), (log_sigma_rel)]
    # Index positions for optional parameters
    mu_idx = n_free_boundary if infer_mu else None
    if infer_sigma:
        sigma_base_idx = n_free_boundary + (1 if infer_mu else 0)
        sigma_rel_idx = sigma_base_idx + 1 if use_per_vessel else None
    else:
        sigma_base_idx = None
        sigma_rel_idx = None

    n_params = n_free_boundary + (1 if infer_mu else 0)
    if infer_sigma:
        n_params += 2 if use_per_vessel else 1

    def neg_log_posterior(theta):
        """Negative log posterior (for minimization)."""
        P_boundary = theta[:n_free_boundary]

        if infer_mu:
            log_mu = theta[mu_idx]
            mu_Pa_s = np.exp(log_mu)
            mu_cP_val = mu_Pa_s * 1000
        else:
            mu_Pa_s = mu_fixed
            mu_cP_val = mu_cP

        if infer_sigma:
            log_sigma_base = theta[sigma_base_idx]
            sigma_base = np.exp(log_sigma_base)

            if use_per_vessel:
                # Per-vessel model: σ_i = σ_base + σ_rel * |Q_i|
                log_sigma_rel = theta[sigma_rel_idx]
                sigma_rel = np.exp(log_sigma_rel)
                sigma_Q = sigma_base + sigma_rel * abs_Q_meas
            else:
                # Global model: single σ for all
                sigma_Q = sigma_base
                sigma_rel = None
        else:
            sigma_Q = sigma_Q_data  # Use per-edge values from data

        # Forward model
        Q_pred = _forward_model(P_boundary, mu_Pa_s, data)

        # Log likelihood (Gaussian with per-vessel sigma)
        # -0.5 * sum((Q_meas - Q_pred)^2 / sigma_i^2) - sum(log(sigma_i))
        residuals = Q_meas - Q_pred
        if infer_sigma:
            # Per-vessel normalization
            if use_per_vessel:
                ll = -0.5 * np.sum((residuals / sigma_Q)**2) - np.sum(np.log(sigma_Q))
            else:
                ll = -0.5 * np.sum((residuals / sigma_Q)**2) - n_meas * np.log(sigma_Q)
        else:
            ll = _log_likelihood(Q_pred, Q_meas, sigma_Q)

        # Log prior on pressures (weak Gaussian)
        lp_P = _log_prior_pressure(P_boundary, prior_P_std)

        # Log prior on mu (if inferring)
        lp_mu = _log_prior_mu(mu_cP_val, prior_mu_mean, prior_mu_std) if infer_mu else 0.0

        # Log prior on sigma parameters (half-Cauchy)
        lp_sigma = 0.0
        if infer_sigma:
            # Half-Cauchy prior on sigma_base: p(σ) ∝ 1/(1 + (σ/scale)²)
            lp_sigma += -np.log(1 + (sigma_base / prior_sigma_scale)**2)
            if use_per_vessel:
                # Half-Cauchy prior on sigma_rel (use scale of 0.5 for relative noise)
                lp_sigma += -np.log(1 + (sigma_rel / 0.5)**2)

        return -(ll + lp_P + lp_mu + lp_sigma)

    # Initial guess: try to use existing simulation results from graph
    free_boundary = data['free_boundary']
    use_graph_init = False

    # Initial sigma estimates from RMSE of standard simulation
    if 'sim_rmse' in G.graph:
        sigma_base_init = G.graph['sim_rmse'] * 0.5  # Half as base
    else:
        sigma_base_init = 2.0  # Default base noise
    sigma_base_init = max(sigma_base_init, 0.1)

    # Initial relative sigma: start with ~10% relative error
    sigma_rel_init = 0.1

    if 'sim_mu_cP' in G.graph:
        # Check if boundary nodes have pressure values
        existing_pressures = {}
        for n in free_boundary:
            if 'pressure_Pa' in G.nodes[n]:
                existing_pressures[n] = G.nodes[n]['pressure_Pa']

        if len(existing_pressures) == len(free_boundary):
            use_graph_init = True
            P_init = np.array([existing_pressures[n] for n in free_boundary])
            mu_init_cP = G.graph['sim_mu_cP']
            mu_init_Pa_s = mu_init_cP * 1e-3

            # Build theta_init based on which parameters are being inferred
            theta_init = P_init.copy()
            if infer_mu:
                theta_init = np.concatenate([theta_init, [np.log(mu_init_Pa_s)]])
            if infer_sigma:
                theta_init = np.concatenate([theta_init, [np.log(sigma_base_init)]])
                if use_per_vessel:
                    theta_init = np.concatenate([theta_init, [np.log(sigma_rel_init)]])

            if verbose:
                print(f"\n  Using existing simulation as initial guess:")
                print(f"  Initial μ = {mu_init_cP:.3f} cP (from previous simulation)")
                if infer_sigma:
                    if use_per_vessel:
                        print(f"  Initial σ_base = {sigma_base_init:.3f} nL/s, σ_rel = {sigma_rel_init:.3f}")
                    else:
                        print(f"  Initial σ_Q = {sigma_base_init:.3f} nL/s")

    if not use_graph_init:
        # Fall back to least squares initialization
        if verbose:
            print("\n  Finding initial estimate via least squares...")

        from scipy.linalg import lstsq as scipy_lstsq

        # Build A matrix for measured edges (use start_j/end_j for consistency)
        A = np.zeros((data['n_meas'], n_free_boundary))
        for idx, e in enumerate(data['measured_edges']):
            g0 = e['g0']
            # Use start_j/end_j to match forward model sign convention
            for node, sign in [(e['start_j'], +1), (e['end_j'], -1)]:
                if node == data['ref_node']:
                    pass
                elif node in data['free_boundary']:
                    j = data['boundary_idx'][node]
                    A[idx, j] += sign * g0
                elif node in data['internal_nodes']:
                    i = data['internal_idx'][node]
                    A[idx, :] += sign * g0 * data['M'][i, :]

        P_init, _, _, _ = scipy_lstsq(A, Q_meas)

        # Infer initial mu (use ratio method, clip to reasonable range)
        if infer_mu:
            Q_pred_init = A @ P_init
            # mu = sum(Q_pred * Q_meas) / sum(Q_meas^2) if we want Q_pred/mu ≈ Q_meas
            mu_init_Pa_s = np.sum(np.abs(Q_pred_init) * np.abs(Q_meas)) / np.sum(Q_meas**2)
            mu_init_Pa_s = np.clip(mu_init_Pa_s, 1e-3, 0.01)  # 1-10 cP range
            P_init = P_init * mu_init_Pa_s  # Scale pressures
            theta_init = P_init.copy()
            theta_init = np.concatenate([theta_init, [np.log(mu_init_Pa_s)]])
        else:
            P_init = P_init * mu_fixed
            theta_init = P_init.copy()

        # Add sigma initialization
        if infer_sigma:
            theta_init = np.concatenate([theta_init, [np.log(sigma_base_init)]])
            if use_per_vessel:
                theta_init = np.concatenate([theta_init, [np.log(sigma_rel_init)]])

        if verbose:
            if infer_mu:
                print(f"  Initial μ = {np.exp(theta_init[mu_idx]) * 1000:.3f} cP")
            else:
                print(f"  Initial μ = {mu_cP:.3f} cP (fixed)")
            if infer_sigma:
                if use_per_vessel:
                    print(f"  Initial σ_base = {sigma_base_init:.3f} nL/s, σ_rel = {sigma_rel_init:.3f}")
                else:
                    print(f"  Initial σ_Q = {sigma_base_init:.3f} nL/s")

    # MAP optimization
    if verbose:
        print("  Running MAP optimization...")
        if use_per_vessel:
            print("  Using per-vessel noise model: σ_i = σ_base + σ_rel * |Q_i|")

    # Bounds: pressures can be positive or negative, mu and sigma in reasonable ranges
    bounds = [(None, None)] * n_free_boundary
    if infer_mu:
        bounds.append((np.log(1e-3), np.log(0.01)))  # mu in [1, 10] cP
    if infer_sigma:
        bounds.append((np.log(0.1), np.log(100.0)))  # sigma_base in [0.1, 100] nL/s
        if use_per_vessel:
            bounds.append((np.log(0.001), np.log(1.0)))  # sigma_rel in [0.1%, 100%]

    result = minimize(
        neg_log_posterior,
        theta_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if not result.success and verbose:
        print(f"  Warning: Optimization did not converge: {result.message}")

    theta_MAP = result.x
    P_MAP = theta_MAP[:n_free_boundary]

    if infer_mu:
        mu_MAP_Pa_s = np.exp(theta_MAP[mu_idx])
        mu_MAP_cP = mu_MAP_Pa_s * 1000
    else:
        mu_MAP_Pa_s = mu_fixed
        mu_MAP_cP = mu_cP

    # Extract sigma parameters
    if infer_sigma:
        sigma_base_MAP = np.exp(theta_MAP[sigma_base_idx])
        if use_per_vessel:
            sigma_rel_MAP = np.exp(theta_MAP[sigma_rel_idx])
            # Compute per-vessel sigma values
            sigma_Q_per_vessel_arr = sigma_base_MAP + sigma_rel_MAP * abs_Q_meas
        else:
            sigma_rel_MAP = None
            sigma_Q_per_vessel_arr = np.full(n_meas, sigma_base_MAP)
    else:
        sigma_base_MAP = None
        sigma_rel_MAP = None
        sigma_Q_per_vessel_arr = sigma_Q_data

    log_posterior_MAP = -result.fun

    if verbose:
        print(f"  MAP estimate: μ = {mu_MAP_cP:.3f} cP")
        if infer_sigma:
            if use_per_vessel:
                print(f"  MAP estimate: σ_base = {sigma_base_MAP:.3f} nL/s, σ_rel = {sigma_rel_MAP:.3f}")
                print(f"  Per-vessel σ range: [{sigma_Q_per_vessel_arr.min():.2f}, {sigma_Q_per_vessel_arr.max():.2f}] nL/s")
            else:
                print(f"  MAP estimate: σ_Q = {sigma_base_MAP:.3f} nL/s")
        print(f"  Log posterior = {log_posterior_MAP:.2f}")

    # Laplace approximation: compute Hessian
    if verbose:
        print("  Computing Hessian for uncertainty...")

    # Numerical Hessian
    eps = 1e-5
    H = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            theta_pp = theta_MAP.copy()
            theta_pp[i] += eps
            theta_pp[j] += eps

            theta_pm = theta_MAP.copy()
            theta_pm[i] += eps
            theta_pm[j] -= eps

            theta_mp = theta_MAP.copy()
            theta_mp[i] -= eps
            theta_mp[j] += eps

            theta_mm = theta_MAP.copy()
            theta_mm[i] -= eps
            theta_mm[j] -= eps

            H[i, j] = (neg_log_posterior(theta_pp) - neg_log_posterior(theta_pm)
                      - neg_log_posterior(theta_mp) + neg_log_posterior(theta_mm)) / (4 * eps**2)
            H[j, i] = H[i, j]

    # Covariance = inverse Hessian
    try:
        Cov = np.linalg.inv(H)
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(Cov)
        if np.any(eigvals < 0):
            if verbose:
                print("  Warning: Covariance not positive definite, using pseudoinverse")
            Cov = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        if verbose:
            print("  Warning: Hessian singular, using pseudoinverse")
        Cov = np.linalg.pinv(H)

    # Extract standard deviations
    P_std = np.sqrt(np.maximum(np.diag(Cov[:n_free_boundary, :n_free_boundary]), 0))

    if infer_mu:
        # Transform from log(mu) to mu: σ_mu ≈ mu * σ_log_mu
        sigma_log_mu = np.sqrt(max(Cov[mu_idx, mu_idx], 0))
        mu_std_cP = mu_MAP_cP * sigma_log_mu
    else:
        mu_std_cP = None

    if infer_sigma:
        # Transform from log(sigma_base) to sigma_base: σ ≈ sigma * σ_log_sigma
        sigma_log_sigma_base = np.sqrt(max(Cov[sigma_base_idx, sigma_base_idx], 0))
        sigma_base_std = sigma_base_MAP * sigma_log_sigma_base
        if use_per_vessel:
            sigma_log_sigma_rel = np.sqrt(max(Cov[sigma_rel_idx, sigma_rel_idx], 0))
            sigma_rel_std = sigma_rel_MAP * sigma_log_sigma_rel
        else:
            sigma_rel_std = None
    else:
        sigma_base_std = None
        sigma_rel_std = None

    # Build full pressure dict
    P_internal_MAP = data['M'] @ P_MAP if data['n_internal'] > 0 else np.array([])

    node_pressures = {data['ref_node']: 0.0}
    for i, n in enumerate(data['free_boundary']):
        node_pressures[n] = P_MAP[i]
    for i, n in enumerate(data['internal_nodes']):
        node_pressures[n] = P_internal_MAP[i]

    # Boundary pressure uncertainties and CIs
    P_boundary_std = {data['ref_node']: 0.0}
    P_boundary_ci95 = {data['ref_node']: (0.0, 0.0)}
    for i, n in enumerate(data['free_boundary']):
        P_boundary_std[n] = P_std[i]
        P_boundary_ci95[n] = (P_MAP[i] - 1.96 * P_std[i], P_MAP[i] + 1.96 * P_std[i])

    # Predicted flows and uncertainties
    Q_pred_MAP = _forward_model(P_MAP, mu_MAP_Pa_s, data)

    predicted_Q = {}
    measured_Q_dict = {}
    for idx, e in enumerate(data['measured_edges']):
        predicted_Q[(e['u'], e['v'])] = Q_pred_MAP[idx]
        measured_Q_dict[(e['u'], e['v'])] = e['Q_measured']

    # Also predict for unmeasured edges
    for e in data['edges_data']:
        if e['Q_measured'] is None:
            P_u = node_pressures.get(e['u'], 0)
            P_v = node_pressures.get(e['v'], 0)
            g = e['g0'] / mu_MAP_Pa_s
            predicted_Q[(e['u'], e['v'])] = g * (P_u - P_v)

    # Propagate uncertainty to Q predictions
    # ∂Q/∂P is linear, so we can propagate covariance
    # For now, use numerical gradient
    Q_pred_std = {}
    Q_pred_ci95 = {}

    for idx, e in enumerate(data['measured_edges']):
        # Numerical gradient of Q[idx] w.r.t. theta
        dQ_dtheta = np.zeros(n_params)
        for k in range(n_params):
            theta_plus = theta_MAP.copy()
            theta_plus[k] += eps
            theta_minus = theta_MAP.copy()
            theta_minus[k] -= eps

            # Extract mu from the correct index
            if infer_mu:
                mu_plus = np.exp(theta_plus[mu_idx])
                mu_minus = np.exp(theta_minus[mu_idx])
            else:
                mu_plus = mu_MAP_Pa_s
                mu_minus = mu_MAP_Pa_s

            Q_plus = _forward_model(theta_plus[:n_free_boundary], mu_plus, data)
            Q_minus = _forward_model(theta_minus[:n_free_boundary], mu_minus, data)

            dQ_dtheta[k] = (Q_plus[idx] - Q_minus[idx]) / (2 * eps)

        # Propagate: Var(Q) = dQ/dθ' @ Cov @ dQ/dθ
        var_Q = dQ_dtheta @ Cov @ dQ_dtheta
        std_Q = np.sqrt(max(var_Q, 0))

        Q_pred_std[(e['u'], e['v'])] = std_Q
        Q_pred_ci95[(e['u'], e['v'])] = (
            Q_pred_MAP[idx] - 1.96 * std_Q,
            Q_pred_MAP[idx] + 1.96 * std_Q
        )

    # Fit metrics
    residuals = Q_meas - Q_pred_MAP
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Q_meas - np.mean(Q_meas))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residuals**2))

    # Chi-squared using the appropriate sigma (per-vessel array in all cases)
    chi2 = np.sum((residuals / sigma_Q_per_vessel_arr)**2)
    dof = data['n_meas'] - n_params
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    if verbose:
        print(f"\n  Results:")
        print(f"    μ = {mu_MAP_cP:.3f} ± {mu_std_cP:.3f} cP" if mu_std_cP else f"    μ = {mu_MAP_cP:.3f} cP (fixed)")
        if infer_sigma:
            if use_per_vessel:
                print(f"    σ_base = {sigma_base_MAP:.3f} ± {sigma_base_std:.3f} nL/s")
                print(f"    σ_rel = {sigma_rel_MAP:.3f} ± {sigma_rel_std:.3f}")
            else:
                print(f"    σ_Q = {sigma_base_MAP:.3f} ± {sigma_base_std:.3f} nL/s (inferred)")
        print(f"    R² = {r_squared:.4f}")
        print(f"    RMSE = {rmse:.4f} nL/s")
        print(f"    χ²_reduced = {chi2_reduced:.2f} (should be ~1.0 if σ correct)")
        print(f"    {n_params} parameters, {data['n_meas']} measurements")
        print("=" * 70)

    # Build per-vessel sigma dict
    sigma_Q_per_vessel_dict = None
    if infer_sigma:
        sigma_Q_per_vessel_dict = {}
        for idx, e in enumerate(data['measured_edges']):
            sigma_Q_per_vessel_dict[(e['u'], e['v'])] = sigma_Q_per_vessel_arr[idx]

    return BayesianSimulationResult(
        mu_cP=mu_MAP_cP,
        node_pressures=node_pressures,
        predicted_Q=predicted_Q,
        measured_Q=measured_Q_dict,
        P_boundary_std=P_boundary_std,
        P_boundary_cov=Cov[:n_free_boundary, :n_free_boundary],
        Q_pred_std=Q_pred_std,
        mu_std=mu_std_cP,
        sigma_Q_inferred=sigma_base_MAP,
        sigma_Q_std=sigma_base_std,
        sigma_rel_inferred=sigma_rel_MAP if infer_sigma else None,
        sigma_rel_std=sigma_rel_std if infer_sigma else None,
        sigma_Q_per_vessel=sigma_Q_per_vessel_dict,
        P_boundary_ci95=P_boundary_ci95,
        Q_pred_ci95=Q_pred_ci95,
        r_squared=r_squared,
        rmse=rmse,
        chi2_reduced=chi2_reduced,
        log_posterior=log_posterior_MAP,
        boundary_nodes=data['boundary_nodes'],
        ref_node=data['ref_node'],
        n_parameters=n_params,
        n_measurements=data['n_meas'],
    )


def write_bayesian_to_graph(
    G: nx.Graph,
    result: BayesianSimulationResult,
    verbose: bool = True,
) -> int:
    """
    Write Bayesian simulation results back to graph.

    Writes point estimates plus uncertainties:
    - pressure_Pa, pressure_std: Node pressure and uncertainty
    - Q_predicted, Q_pred_std, Q_pred_ci95_lo, Q_pred_ci95_hi: Edge flow predictions

    Also stores metadata in graph.graph:
    - bayes_mu_cP, bayes_mu_std
    - bayes_r_squared, bayes_chi2_reduced
    """
    # Write node pressures
    for node, pressure in result.node_pressures.items():
        if G.has_node(node):
            G.nodes[node]['pressure_Pa'] = pressure
            G.nodes[node]['pressure_std'] = result.P_boundary_std.get(node, 0.0)
            G.nodes[node]['is_boundary_sim'] = node in result.boundary_nodes

            if node in result.P_boundary_ci95:
                ci = result.P_boundary_ci95[node]
                G.nodes[node]['pressure_ci95_lo'] = ci[0]
                G.nodes[node]['pressure_ci95_hi'] = ci[1]

    # Write edge predictions
    n_written = 0
    for (u, v), Q_pred in result.predicted_Q.items():
        if G.has_edge(u, v):
            edge_key = (u, v)
        elif G.has_edge(v, u):
            edge_key = (v, u)
            Q_pred = -Q_pred
        else:
            continue

        G.edges[edge_key]['Q_predicted'] = Q_pred

        # Uncertainties
        if (u, v) in result.Q_pred_std:
            G.edges[edge_key]['Q_pred_std'] = result.Q_pred_std[(u, v)]
        if (u, v) in result.Q_pred_ci95:
            ci = result.Q_pred_ci95[(u, v)]
            if edge_key == (v, u):
                G.edges[edge_key]['Q_pred_ci95_lo'] = -ci[1]
                G.edges[edge_key]['Q_pred_ci95_hi'] = -ci[0]
            else:
                G.edges[edge_key]['Q_pred_ci95_lo'] = ci[0]
                G.edges[edge_key]['Q_pred_ci95_hi'] = ci[1]

        # Residual
        Q_meas = result.measured_Q.get((u, v))
        if Q_meas is None:
            Q_meas = result.measured_Q.get((v, u))
            if Q_meas is not None:
                Q_meas = -Q_meas

        if Q_meas is not None:
            residual = Q_meas - G.edges[edge_key]['Q_predicted']
            G.edges[edge_key]['Q_residual'] = residual
            if abs(Q_meas) > 1e-6:
                G.edges[edge_key]['Q_residual_rel'] = residual / abs(Q_meas)

        # Per-vessel sigma (from heteroscedastic model)
        if result.sigma_Q_per_vessel is not None:
            sigma_val = result.sigma_Q_per_vessel.get((u, v))
            if sigma_val is not None:
                G.edges[edge_key]['sigma_Q_per_vessel'] = sigma_val

        n_written += 1

    # Store metadata
    G.graph['bayes_mu_cP'] = result.mu_cP
    G.graph['bayes_mu_std'] = result.mu_std
    G.graph['bayes_sigma_Q'] = result.sigma_Q_inferred
    G.graph['bayes_sigma_Q_std'] = result.sigma_Q_std
    G.graph['bayes_r_squared'] = result.r_squared
    G.graph['bayes_chi2_reduced'] = result.chi2_reduced
    G.graph['bayes_n_params'] = result.n_parameters
    G.graph['bayes_n_meas'] = result.n_measurements

    if verbose:
        print(f"Wrote Bayesian results to graph:")
        print(f"  Node pressures: {len(result.node_pressures)} (with uncertainties)")
        print(f"  Edge predictions: {n_written}")
        print(f"  μ = {result.mu_cP:.3f} ± {result.mu_std:.3f} cP" if result.mu_std else f"  μ = {result.mu_cP:.3f} cP")
        if result.sigma_Q_inferred is not None:
            print(f"  σ_Q = {result.sigma_Q_inferred:.3f} ± {result.sigma_Q_std:.3f} nL/s (inferred)" if result.sigma_Q_std else f"  σ_Q = {result.sigma_Q_inferred:.3f} nL/s")
        print(f"  χ²_reduced = {result.chi2_reduced:.2f}")

    return n_written


def run_bayesian_simulation(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: Optional[float] = None,
    infer_mu: bool = True,
    infer_sigma: bool = True,
    sigma_model: str = 'per_vessel',
    max_chi2_reduced: float = 5.0,
    min_snr_percentile: float = 50.0,
    min_vessel_length_px: float = 20.0,
    default_sigma_fraction: float = 0.5,
    flow_attr: str = 'mean_Q_nL_s',
    sigma_attr: str = 'sigma_Q_nL_s',
    verbose: bool = True,
) -> BayesianSimulationResult:
    """
    Run Bayesian Poiseuille simulation and write results to graph.

    Convenience function combining bayesian_poiseuille_simulation and
    write_bayesian_to_graph.

    Parameters
    ----------
    infer_sigma : bool
        If True (default), infer measurement noise σ_Q from the data.
        This follows Rasmussen et al. (2018) approach for more robust estimates.
    sigma_model : str
        Noise model when infer_sigma=True:
        - 'global': Single σ for all vessels
        - 'per_vessel': σ_i = σ_base + σ_rel × |Q_i| (default)
    max_chi2_reduced : float
        Filter threshold for profile fit quality.
    min_snr_percentile : float
        Only include vessels with SNR above this percentile (default 50 = median).
        Matches viewer display threshold. Set to 0 to disable.
    min_vessel_length_px : float
        Minimum vessel length in pixels (default 20). Short vessels excluded.
        Matches viewer display threshold. Set to 0 to disable.
    default_sigma_fraction : float
        If sigma_Q not available and infer_sigma=False, use this fraction
        of |Q| as uncertainty. Default 0.5 (50% uncertainty).
    """
    result = bayesian_poiseuille_simulation(
        G,
        boundary_nodes=boundary_nodes,
        mu_cP=mu_cP,
        infer_mu=infer_mu,
        infer_sigma=infer_sigma,
        sigma_model=sigma_model,
        flow_attr=flow_attr,
        sigma_attr=sigma_attr,
        max_chi2_reduced=max_chi2_reduced,
        min_snr_percentile=min_snr_percentile,
        min_vessel_length_px=min_vessel_length_px,
        default_sigma_fraction=default_sigma_fraction,
        verbose=verbose,
    )

    write_bayesian_to_graph(G, result, verbose=verbose)

    return result


def print_bayesian_report(result: BayesianSimulationResult, top_n: int = 10):
    """Print Bayesian simulation results with uncertainties."""
    print("\n" + "=" * 70)
    print("BAYESIAN POISEUILLE FLOW SIMULATION")
    print("=" * 70)

    # Viscosity with uncertainty
    if result.mu_std is not None:
        ci_lo = result.mu_cP - 1.96 * result.mu_std
        ci_hi = result.mu_cP + 1.96 * result.mu_std
        print(f"\nInferred viscosity: μ = {result.mu_cP:.3f} ± {result.mu_std:.3f} cP")
        print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] cP")
    else:
        print(f"\nFixed viscosity: μ = {result.mu_cP:.3f} cP")

    # Inferred measurement noise
    if result.sigma_Q_inferred is not None:
        if result.sigma_Q_std is not None:
            ci_lo = result.sigma_Q_inferred - 1.96 * result.sigma_Q_std
            ci_hi = result.sigma_Q_inferred + 1.96 * result.sigma_Q_std
            print(f"\nInferred noise: σ_Q = {result.sigma_Q_inferred:.3f} ± {result.sigma_Q_std:.3f} nL/s")
            print(f"  95% CI: [{max(0, ci_lo):.3f}, {ci_hi:.3f}] nL/s")
        else:
            print(f"\nInferred noise: σ_Q = {result.sigma_Q_inferred:.3f} nL/s")

    print(f"\nFit quality:")
    print(f"  R² = {result.r_squared:.4f}")
    print(f"  RMSE = {result.rmse:.4f} nL/s")
    print(f"  χ²_reduced = {result.chi2_reduced:.3f} (should be ~1.0 if σ is correct)")
    print(f"  log(posterior) = {result.log_posterior:.2f}")

    print(f"\nModel info:")
    print(f"  Parameters: {result.n_parameters}")
    print(f"  Measurements: {result.n_measurements}")
    print(f"  Reference node: {result.ref_node}")

    # Boundary pressures with uncertainties
    print(f"\nBoundary pressures ({len(result.boundary_nodes)} nodes):")
    sorted_bp = sorted(
        [(n, result.node_pressures.get(n, 0)) for n in result.boundary_nodes],
        key=lambda x: x[1], reverse=True
    )
    for node, P in sorted_bp[:top_n]:
        if node in result.P_boundary_std:
            std = result.P_boundary_std[node]
            ci = result.P_boundary_ci95.get(node, (P - 1.96*std, P + 1.96*std))
            print(f"  Node {node}: P = {P:.1f} ± {std:.1f} Pa  [95% CI: {ci[0]:.1f}, {ci[1]:.1f}]")
        else:
            # Reference node
            print(f"  Node {node}: P = {P:.1f} Pa (reference)")

    # Largest flow uncertainties
    if result.Q_pred_std:
        print(f"\nFlows with largest uncertainty:")
        sorted_q = sorted(result.Q_pred_std.items(), key=lambda x: x[1], reverse=True)
        print(f"  {'Edge':>12}  {'Q_pred':>10}  {'σ_Q':>8}  {'95% CI':>20}")
        for edge, std in sorted_q[:top_n]:
            Q = result.predicted_Q.get(edge, 0)
            ci = result.Q_pred_ci95.get(edge, (Q - 1.96*std, Q + 1.96*std))
            print(f"  {str(edge):>12}  {Q:>+10.4f}  {std:>8.4f}  [{ci[0]:>+8.4f}, {ci[1]:>+8.4f}]")

    print("=" * 70)


def plot_bayesian_diagnostics(
    result: BayesianSimulationResult,
    figsize: Tuple[float, float] = (12, 10),
    show: bool = True,
):
    """
    Generate diagnostic plots for Bayesian simulation results.

    Creates a 2x2 figure with:
    1. Measured vs Predicted scatter plot with 1:1 line and error bars
    2. Residual histogram with fitted Gaussian
    3. Residuals vs Predicted (check for heteroscedasticity)
    4. Q-Q plot of standardized residuals

    Similar to diagnostic figures in Rasmussen et al. (2018).

    Parameters
    ----------
    result : BayesianSimulationResult
        Results from bayesian_poiseuille_simulation
    figsize : tuple
        Figure size (width, height) in inches
    show : bool
        Whether to call plt.show()

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Extract paired measurements
    Q_meas = []
    Q_pred = []
    Q_pred_std = []
    edges = []

    for edge, qm in result.measured_Q.items():
        qp = result.predicted_Q.get(edge)
        std = result.Q_pred_std.get(edge, 0)
        if qp is not None and np.isfinite(qm) and np.isfinite(qp):
            Q_meas.append(qm)
            Q_pred.append(qp)
            Q_pred_std.append(std)
            edges.append(edge)

    Q_meas = np.array(Q_meas)
    Q_pred = np.array(Q_pred)
    Q_pred_std = np.array(Q_pred_std)
    residuals = Q_meas - Q_pred

    # Use inferred sigma if available, otherwise estimate from residuals
    if result.sigma_Q_inferred is not None:
        sigma = result.sigma_Q_inferred
    else:
        sigma = np.std(residuals)

    standardized_residuals = residuals / sigma

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ===== Panel 1: Measured vs Predicted =====
    ax = axes[0, 0]

    # Scatter plot
    ax.scatter(Q_meas, Q_pred, alpha=0.5, s=20, c='steelblue', edgecolors='none')

    # 1:1 line
    lims = [min(Q_meas.min(), Q_pred.min()), max(Q_meas.max(), Q_pred.max())]
    margin = 0.1 * (lims[1] - lims[0])
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1 line')

    # ±σ bands
    ax.fill_between(lims, [l - sigma for l in lims], [l + sigma for l in lims],
                    alpha=0.2, color='gray', label=f'±σ = ±{sigma:.1f} nL/s')
    ax.fill_between(lims, [l - 2*sigma for l in lims], [l + 2*sigma for l in lims],
                    alpha=0.1, color='gray', label=f'±2σ')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Measured Q (nL/s)')
    ax.set_ylabel('Predicted Q (nL/s)')
    ax.set_title(f'Measured vs Predicted Flow\nR² = {result.r_squared:.3f}, n = {len(Q_meas)}')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ===== Panel 2: Residual Histogram =====
    ax = axes[0, 1]

    # Histogram
    n_bins = max(5, min(50, len(residuals) // 5))  # At least 5 bins
    counts, bins, _ = ax.hist(residuals, bins=n_bins, density=True, alpha=0.7,
                               color='steelblue', edgecolor='white', label='Residuals')

    # Fitted Gaussian
    x_gauss = np.linspace(residuals.min(), residuals.max(), 100)
    y_gauss = stats.norm.pdf(x_gauss, 0, sigma)
    ax.plot(x_gauss, y_gauss, 'r-', lw=2, label=f'N(0, σ={sigma:.1f})')

    # Actual std for comparison
    actual_std = np.std(residuals)
    y_actual = stats.norm.pdf(x_gauss, np.mean(residuals), actual_std)
    ax.plot(x_gauss, y_actual, 'g--', lw=1.5, label=f'Empirical (σ={actual_std:.1f})')

    ax.set_xlabel('Residual (Q_meas - Q_pred) [nL/s]')
    ax.set_ylabel('Density')
    ax.set_title(f'Residual Distribution\nχ²_red = {result.chi2_reduced:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ===== Panel 3: Residuals vs Predicted =====
    ax = axes[1, 0]

    ax.scatter(Q_pred, residuals, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.axhline(sigma, color='r', linestyle=':', lw=1, alpha=0.7)
    ax.axhline(-sigma, color='r', linestyle=':', lw=1, alpha=0.7, label=f'±σ')
    ax.axhline(2*sigma, color='r', linestyle=':', lw=0.5, alpha=0.5)
    ax.axhline(-2*sigma, color='r', linestyle=':', lw=0.5, alpha=0.5, label=f'±2σ')

    ax.set_xlabel('Predicted Q (nL/s)')
    ax.set_ylabel('Residual (nL/s)')
    ax.set_title('Residuals vs Predicted\n(check for heteroscedasticity)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ===== Panel 4: Q-Q Plot =====
    ax = axes[1, 1]

    # Q-Q plot of standardized residuals
    sorted_resid = np.sort(standardized_residuals)
    n = len(sorted_resid)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    ax.scatter(theoretical_quantiles, sorted_resid, alpha=0.5, s=20, c='steelblue', edgecolors='none')

    # Reference line
    q_lims = [theoretical_quantiles.min(), theoretical_quantiles.max()]
    ax.plot(q_lims, q_lims, 'r--', lw=1.5, label='Normal reference')

    ax.set_xlabel('Theoretical Quantiles (Standard Normal)')
    ax.set_ylabel('Standardized Residuals')
    ax.set_title('Q-Q Plot\n(points should follow diagonal if Gaussian)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add overall title with key parameters
    mu_str = f"μ = {result.mu_cP:.2f}"
    if result.mu_std:
        mu_str += f" ± {result.mu_std:.2f}"
    mu_str += " cP"

    sigma_str = ""
    if result.sigma_Q_inferred:
        sigma_str = f", σ_Q = {result.sigma_Q_inferred:.1f}"
        if result.sigma_Q_std:
            sigma_str += f" ± {result.sigma_Q_std:.1f}"
        sigma_str += " nL/s (inferred)"

    fig.suptitle(f'Bayesian Poiseuille Simulation Diagnostics\n{mu_str}{sigma_str}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    if show:
        plt.show(block=False)

    return fig


def plot_parameter_posteriors(
    result: BayesianSimulationResult,
    figsize: Tuple[float, float] = (12, 5),
    show: bool = True,
):
    """
    Plot posterior distributions for inferred parameters (μ, σ_Q).

    Uses Laplace approximation (Gaussian) posteriors. Similar to Figure 3
    in Rasmussen et al. (2018).

    Parameters
    ----------
    result : BayesianSimulationResult
        Results from bayesian_poiseuille_simulation
    figsize : tuple
        Figure size (width, height) in inches
    show : bool
        Whether to call plt.show()

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Count how many parameters we have posteriors for
    n_panels = 0
    if result.mu_std is not None and result.mu_std > 0:
        n_panels += 1
    if result.sigma_Q_inferred is not None and result.sigma_Q_std is not None and result.sigma_Q_std > 0:
        n_panels += 1

    if n_panels == 0:
        print("No parameter uncertainties available to plot.")
        return None

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # ===== Viscosity posterior =====
    if result.mu_std is not None and result.mu_std > 0:
        ax = axes[panel_idx]
        panel_idx += 1

        mu = result.mu_cP
        sigma = result.mu_std

        # Plot range: ±4σ
        x_min = max(0, mu - 4 * sigma)
        x_max = mu + 4 * sigma
        x = np.linspace(x_min, x_max, 200)
        y = stats.norm.pdf(x, mu, sigma)

        # Fill under curve
        ax.fill_between(x, y, alpha=0.3, color='steelblue')
        ax.plot(x, y, 'b-', lw=2)

        # Mark MAP estimate
        ax.axvline(mu, color='red', linestyle='--', lw=1.5, label=f'MAP = {mu:.3f} cP')

        # Mark 95% CI
        ci_lo = mu - 1.96 * sigma
        ci_hi = mu + 1.96 * sigma
        ax.axvline(ci_lo, color='gray', linestyle=':', lw=1)
        ax.axvline(ci_hi, color='gray', linestyle=':', lw=1)
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color='gray', label=f'95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')

        # Reference: typical blood viscosity
        ax.axvline(3.0, color='green', linestyle='--', lw=1, alpha=0.7, label='Typical blood (3 cP)')

        ax.set_xlabel('Viscosity μ (cP)')
        ax.set_ylabel('Posterior Density')
        ax.set_title(f'Viscosity Posterior\nμ = {mu:.3f} ± {sigma:.3f} cP')
        ax.legend(fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    # ===== Noise scale posterior =====
    if result.sigma_Q_inferred is not None and result.sigma_Q_std is not None and result.sigma_Q_std > 0:
        ax = axes[panel_idx]
        panel_idx += 1

        sigma_Q = result.sigma_Q_inferred
        sigma_Q_std = result.sigma_Q_std

        # Plot range: ±4σ (but not below 0)
        x_min = max(0, sigma_Q - 4 * sigma_Q_std)
        x_max = sigma_Q + 4 * sigma_Q_std
        x = np.linspace(x_min, x_max, 200)
        y = stats.norm.pdf(x, sigma_Q, sigma_Q_std)

        # Fill under curve
        ax.fill_between(x, y, alpha=0.3, color='coral')
        ax.plot(x, y, color='orangered', lw=2)

        # Mark MAP estimate
        ax.axvline(sigma_Q, color='red', linestyle='--', lw=1.5, label=f'MAP = {sigma_Q:.2f} nL/s')

        # Mark 95% CI
        ci_lo = max(0, sigma_Q - 1.96 * sigma_Q_std)
        ci_hi = sigma_Q + 1.96 * sigma_Q_std
        ax.axvline(ci_lo, color='gray', linestyle=':', lw=1)
        ax.axvline(ci_hi, color='gray', linestyle=':', lw=1)
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color='gray', label=f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]')

        # Mark RMSE for comparison
        ax.axvline(result.rmse, color='green', linestyle='--', lw=1, alpha=0.7,
                   label=f'RMSE = {result.rmse:.2f} nL/s')

        ax.set_xlabel('Measurement Noise σ_Q (nL/s)')
        ax.set_ylabel('Posterior Density')
        ax.set_title(f'Noise Scale Posterior\nσ_Q = {sigma_Q:.2f} ± {sigma_Q_std:.2f} nL/s')
        ax.legend(fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Parameter Posterior Distributions (Laplace Approximation)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show(block=False)

    return fig


def plot_boundary_pressure_posteriors(
    result: BayesianSimulationResult,
    top_n: int = 8,
    figsize: Tuple[float, float] = (14, 8),
    show: bool = True,
):
    """
    Plot posterior distributions for boundary node pressures.

    Parameters
    ----------
    result : BayesianSimulationResult
        Results from bayesian_poiseuille_simulation
    top_n : int
        Number of boundary nodes to show (sorted by pressure)
    figsize : tuple
        Figure size
    show : bool
        Whether to call plt.show()

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Get boundary nodes with uncertainties (exclude reference node with σ=0)
    boundary_data = []
    for node in result.boundary_nodes:
        P = result.node_pressures.get(node, 0)
        std = result.P_boundary_std.get(node, 0)
        if std > 0:  # Exclude reference node
            boundary_data.append((node, P, std))

    if not boundary_data:
        print("No boundary pressure uncertainties available.")
        return None

    # Sort by pressure (descending) and take top_n
    boundary_data.sort(key=lambda x: x[1], reverse=True)
    boundary_data = boundary_data[:top_n]

    n_nodes = len(boundary_data)
    n_cols = min(4, n_nodes)
    n_rows = (n_nodes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_nodes == 1:
        axes = np.array([[axes]])
    axes = axes.flatten() if n_nodes > 1 else [axes]

    for idx, (node, P, std) in enumerate(boundary_data):
        ax = axes[idx]

        # Plot range
        x_min = P - 4 * std
        x_max = P + 4 * std
        x = np.linspace(x_min, x_max, 200)
        y = stats.norm.pdf(x, P, std)

        # Fill under curve
        ax.fill_between(x, y, alpha=0.3, color='steelblue')
        ax.plot(x, y, 'b-', lw=2)

        # Mark MAP
        ax.axvline(P, color='red', linestyle='--', lw=1.5)

        # 95% CI
        ci_lo = P - 1.96 * std
        ci_hi = P + 1.96 * std
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color='gray')

        ax.set_xlabel('Pressure (Pa)')
        ax.set_ylabel('Density')
        ax.set_title(f'Node {node}\nP = {P/1000:.1f} ± {std/1000:.2f} kPa')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_nodes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Boundary Pressure Posteriors (Laplace Approximation)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show(block=False)

    return fig


def plot_all_bayesian(
    result: BayesianSimulationResult,
    show: bool = True,
):
    """
    Plot all Bayesian diagnostics and posteriors.

    Convenience function that calls:
    - plot_bayesian_diagnostics (measured vs predicted, residuals, Q-Q)
    - plot_parameter_posteriors (μ and σ_Q posteriors)
    - plot_boundary_pressure_posteriors (boundary pressure posteriors)

    Parameters
    ----------
    result : BayesianSimulationResult
        Results from bayesian_poiseuille_simulation
    show : bool
        Whether to call plt.show()

    Returns
    -------
    figs : tuple of matplotlib Figures (diagnostics, parameters, pressures)
    """
    fig1 = plot_bayesian_diagnostics(result, show=show)
    fig2 = plot_parameter_posteriors(result, show=show)
    fig3 = plot_boundary_pressure_posteriors(result, show=show)
    return fig1, fig2, fig3


def plot_pressure_network(
    result: BayesianSimulationResult,
    G: Optional[nx.Graph] = None,
    figsize: Tuple[float, float] = (10, 10),
    show: bool = True,
):
    """
    Plot the vessel network colored by pressure.

    Parameters
    ----------
    result : BayesianSimulationResult
        Results from bayesian_poiseuille_simulation
    G : nx.Graph, optional
        Graph with node positions. If None, uses spring layout.
    figsize : tuple
        Figure size
    show : bool
        Whether to call plt.show()

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    # Build graph from result if not provided
    if G is None:
        G = nx.Graph()
        for edge in result.predicted_Q.keys():
            G.add_edge(edge[0], edge[1])

    # Get positions
    if 'pos' in G.nodes[list(G.nodes())[0]]:
        pos = {n: (G.nodes[n].get('pos_x', G.nodes[n].get('pos', (0, 0))[0]),
                   G.nodes[n].get('pos_y', G.nodes[n].get('pos', (0, 0))[1]))
               for n in G.nodes()}
    else:
        pos = nx.spring_layout(G)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ===== Panel 1: Pressure field =====
    ax = axes[0]

    # Get pressure values
    pressures = result.node_pressures
    P_vals = np.array([pressures.get(n, 0) for n in G.nodes()])
    P_min, P_max = P_vals.min(), P_vals.max()

    # Draw edges
    edge_pos = [(pos[u], pos[v]) for u, v in G.edges() if u in pos and v in pos]
    lc = LineCollection(edge_pos, colors='gray', linewidths=0.5, alpha=0.5)
    ax.add_collection(lc)

    # Draw nodes colored by pressure
    node_list = list(G.nodes())
    node_pos = np.array([pos[n] for n in node_list])
    node_P = np.array([pressures.get(n, 0) for n in node_list])

    scatter = ax.scatter(node_pos[:, 0], node_pos[:, 1], c=node_P,
                         cmap='coolwarm', s=30, edgecolors='k', linewidths=0.5)

    # Mark boundary nodes
    boundary_pos = np.array([pos[n] for n in result.boundary_nodes if n in pos])
    if len(boundary_pos) > 0:
        ax.scatter(boundary_pos[:, 0], boundary_pos[:, 1],
                   s=100, facecolors='none', edgecolors='black', linewidths=2,
                   label='Boundary nodes')

    # Mark reference node
    if result.ref_node in pos:
        ax.scatter([pos[result.ref_node][0]], [pos[result.ref_node][1]],
                   s=150, marker='*', c='gold', edgecolors='black', linewidths=1,
                   label=f'Reference (P=0)', zorder=10)

    plt.colorbar(scatter, ax=ax, label='Pressure (Pa)')
    ax.set_title('Inferred Pressure Field')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ===== Panel 2: Flow magnitude =====
    ax = axes[1]

    # Draw edges colored by flow
    edge_colors = []
    edge_widths = []
    edge_segments = []

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        Q = result.predicted_Q.get((u, v)) or result.predicted_Q.get((v, u)) or 0
        edge_segments.append([pos[u], pos[v]])
        edge_colors.append(abs(Q))
        edge_widths.append(0.5 + 2 * abs(Q) / max(abs(q) for q in result.predicted_Q.values()))

    if edge_segments:
        edge_colors = np.array(edge_colors)
        lc = LineCollection(edge_segments, cmap='viridis', linewidths=edge_widths)
        lc.set_array(edge_colors)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='|Q| (nL/s)')

    # Draw nodes
    ax.scatter(node_pos[:, 0], node_pos[:, 1], c='lightgray', s=10, zorder=1)

    ax.set_title('Predicted Flow Magnitude')
    ax.set_aspect('equal')
    ax.axis('off')

    fig.suptitle(f'Network Visualization (μ = {result.mu_cP:.2f} cP)', fontsize=12)
    plt.tight_layout()

    if show:
        plt.show(block=False)

    return fig


def compare_sigma_empirical_vs_inferred(
    G: nx.Graph,
    result: BayesianSimulationResult,
    f0: Optional[float] = None,
    n_harmonics: int = 3,
    figsize: Tuple[float, float] = (14, 10),
    show: bool = True,
) -> Dict[str, Any]:
    """
    Compare inferred σ_vessel with empirical σ from Q(t) harmonic residuals.

    This diagnostic helps determine if model inadequacy is driving the large
    inferred noise, or if it's genuine measurement noise.

    Parameters
    ----------
    G : nx.Graph
        Graph with Q_t time series attached to edges
    result : BayesianSimulationResult
        Results from bayesian_poiseuille_simulation
    f0 : float, optional
        Heart rate frequency (Hz). If None, uses G.graph['f0_consensus']
    n_harmonics : int
        Number of harmonics for fitting Q(t)
    figsize : tuple
        Figure size
    show : bool
        Whether to call plt.show()

    Returns
    -------
    dict with:
        - sigma_empirical: array of empirical σ per vessel
        - sigma_inferred: array of inferred σ per vessel
        - edges: list of (u,v) tuples
        - slope: linear fit slope (σ_inferred = slope * σ_empirical)
        - r_squared: correlation R²
        - interpretation: string describing what the results mean
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Get f0
    if f0 is None:
        f0 = G.graph.get('f0_consensus', G.graph.get('f0', 1.0))

    # Get frame dt
    frame_dt = G.graph.get('frame_dt', 1.0 / 100)  # Default 100 fps

    # Collect data
    edges = []
    sigma_empirical = []
    sigma_inferred = []
    Q_meas_list = []
    Q_pred_list = []
    residuals_list = []

    for u, v, data in G.edges(data=True):
        # Need Q_t time series
        Q_t = data.get('Q_t')
        if Q_t is None or len(Q_t) < 10:
            continue

        # Need inferred sigma
        sigma_inf = data.get('sigma_Q_per_vessel')
        if sigma_inf is None or not np.isfinite(sigma_inf):
            continue

        # Need predicted and measured Q
        Q_pred = data.get('Q_predicted')
        Q_meas = data.get('mean_Q_nL_s', data.get('mean_Q'))
        if Q_pred is None or Q_meas is None:
            continue
        if not np.isfinite(Q_pred) or not np.isfinite(Q_meas):
            continue

        # Fit harmonic model to Q(t)
        T = len(Q_t)
        t = np.arange(T) * frame_dt

        # Build design matrix for harmonic regression
        # Q(t) = a0 + sum_k [a_k * cos(2πkf0t) + b_k * sin(2πkf0t)]
        X = np.ones((T, 1 + 2 * n_harmonics))
        X[:, 0] = 1  # DC component
        for k in range(1, n_harmonics + 1):
            X[:, 2*k - 1] = np.cos(2 * np.pi * k * f0 * t)
            X[:, 2*k] = np.sin(2 * np.pi * k * f0 * t)

        # Fit
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, Q_t, rcond=None)
            Q_harmonic = X @ coeffs
            resid = Q_t - Q_harmonic

            # Empirical sigma: std of residuals / sqrt(N_eff)
            # N_eff = T / (samples per heartbeat) for uncertainty in mean
            samples_per_beat = int(1.0 / (f0 * frame_dt)) if f0 > 0 else T
            N_heartbeats = T / samples_per_beat if samples_per_beat > 0 else 1

            sigma_emp = np.std(resid) / np.sqrt(max(N_heartbeats, 1))

            if sigma_emp > 0 and np.isfinite(sigma_emp):
                edges.append((u, v))
                sigma_empirical.append(sigma_emp)
                sigma_inferred.append(sigma_inf)
                Q_meas_list.append(Q_meas)
                Q_pred_list.append(Q_pred)
                residuals_list.append(Q_meas - Q_pred)

        except Exception:
            continue

    if len(edges) < 5:
        print("Not enough edges with Q_t time series for comparison")
        return None

    sigma_empirical = np.array(sigma_empirical)
    sigma_inferred = np.array(sigma_inferred)
    Q_meas_arr = np.array(Q_meas_list)
    Q_pred_arr = np.array(Q_pred_list)
    residuals = np.array(residuals_list)

    # Compute ratio for diagnostic
    ratio = sigma_inferred / sigma_empirical

    # Linear regression: σ_inferred = slope * σ_empirical + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        sigma_empirical, sigma_inferred
    )
    r_squared = r_value ** 2

    # Standardized residuals (compute first, needed for interpretation)
    z_empirical = residuals / sigma_empirical
    z_inferred = residuals / sigma_inferred

    # Interpretation based on median ratio and std(z_empirical)
    # The median ratio is the key metric, not the slope (which can be dominated by outliers)
    median_ratio = np.median(ratio)
    z_emp_std = np.std(z_empirical)

    # Model error fraction: if σ_inferred = √(σ²_meas + σ²_model), then
    # ratio = σ_inferred/σ_empirical ≈ √(1 + σ²_model/σ²_meas)
    # For ratio >> 1: σ²_model ≈ ratio² × σ²_meas
    # Model error fraction = σ²_model / σ²_total ≈ (ratio² - 1) / ratio²
    model_error_fraction = (median_ratio**2 - 1) / median_ratio**2 if median_ratio > 1 else 0
    model_error_pct = model_error_fraction * 100

    if median_ratio < 1.5 and z_emp_std < 2.0:
        interpretation = (
            f"σ_inferred ≈ σ_empirical (ratio={median_ratio:.1f}×, std(z)={z_emp_std:.1f})\n"
            "→ Model is ADEQUATE: residuals explained by measurement noise"
        )
    elif median_ratio < 3.0 and z_emp_std < 5.0:
        interpretation = (
            f"σ_inferred > σ_empirical (ratio={median_ratio:.1f}×, std(z)={z_emp_std:.1f})\n"
            f"→ Model is MARGINAL: ~{model_error_pct:.0f}% of variance from model error"
        )
    else:
        interpretation = (
            f"σ_inferred >> σ_empirical (ratio={median_ratio:.1f}×, std(z)={z_emp_std:.1f})\n"
            f"→ Model is INADEQUATE: ~{model_error_pct:.0f}% of variance from model error\n"
            "→ σ_inferred is correctly calibrated, but includes large model error"
        )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. σ_inferred vs σ_empirical scatter
    ax = axes[0, 0]
    ax.scatter(sigma_empirical, sigma_inferred, alpha=0.5, s=20)

    # Fit line
    x_fit = np.array([0, sigma_empirical.max()])
    ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
            label=f'Fit: slope={slope:.2f}, R²={r_squared:.2f}')
    ax.plot(x_fit, x_fit, 'k--', lw=1, alpha=0.5, label='1:1 line')

    ax.set_xlabel('σ_empirical (from Q(t) residuals) [nL/s]')
    ax.set_ylabel('σ_inferred (from Bayesian model) [nL/s]')
    ax.set_title('Noise Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Ratio histogram
    ax = axes[0, 1]
    ax.hist(ratio, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(ratio), color='r', linestyle='--', lw=2,
               label=f'Median = {np.median(ratio):.2f}')
    ax.axvline(1.0, color='k', linestyle=':', lw=1, label='Ratio = 1')
    ax.set_xlabel('σ_inferred / σ_empirical')
    ax.set_ylabel('Count')
    ax.set_title('Noise Ratio Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Standardized residuals histogram (empirical)
    ax = axes[0, 2]
    ax.hist(z_empirical, bins=30, density=True, edgecolor='black', alpha=0.7,
            label='z = (Q_meas - Q_pred) / σ_emp')
    x_norm = np.linspace(-5, 5, 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm), 'r-', lw=2, label='N(0,1)')
    ax.set_xlabel('Standardized Residual (empirical σ)')
    ax.set_ylabel('Density')
    ax.set_title(f'Residuals / σ_empirical\nstd = {np.std(z_empirical):.2f}')
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3)

    # 4. Standardized residuals histogram (inferred)
    ax = axes[1, 0]
    ax.hist(z_inferred, bins=30, density=True, edgecolor='black', alpha=0.7,
            label='z = (Q_meas - Q_pred) / σ_inf')
    ax.plot(x_norm, stats.norm.pdf(x_norm), 'r-', lw=2, label='N(0,1)')
    ax.set_xlabel('Standardized Residual (inferred σ)')
    ax.set_ylabel('Density')
    ax.set_title(f'Residuals / σ_inferred\nstd = {np.std(z_inferred):.2f}')
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3)

    # 5. Q-Q plot comparison
    ax = axes[1, 1]
    # Q-Q for empirical
    sorted_z_emp = np.sort(z_empirical)
    theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_z_emp)))
    ax.scatter(theoretical_q, sorted_z_emp[:len(theoretical_q)], alpha=0.5, s=10,
               label=f'Using σ_emp (std={np.std(z_empirical):.2f})')
    # Q-Q for inferred
    sorted_z_inf = np.sort(z_inferred)
    ax.scatter(theoretical_q, sorted_z_inf[:len(theoretical_q)], alpha=0.5, s=10,
               label=f'Using σ_inf (std={np.std(z_inferred):.2f})')
    ax.plot([-4, 4], [-4, 4], 'k--', lw=1)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-6, 6)

    # 6. Residual vs |Q| (check heteroscedasticity)
    ax = axes[1, 2]
    ax.scatter(np.abs(Q_meas_arr), np.abs(residuals), alpha=0.5, s=20, label='|Residual|')
    ax.scatter(np.abs(Q_meas_arr), sigma_empirical, alpha=0.5, s=20, label='σ_empirical')
    ax.scatter(np.abs(Q_meas_arr), sigma_inferred, alpha=0.5, s=20, label='σ_inferred')
    ax.set_xlabel('|Q_measured| (nL/s)')
    ax.set_ylabel('nL/s')
    ax.set_title('Heteroscedasticity Check')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Model Adequacy Diagnostic\n{interpretation}', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show(block=False)

    # Print summary
    print("\n" + "=" * 70)
    print("MODEL ADEQUACY DIAGNOSTIC")
    print("=" * 70)
    print(f"\nCompared {len(edges)} edges with Q(t) time series")
    print(f"\nσ_empirical (from harmonic residuals = measurement noise):")
    print(f"  Range: [{sigma_empirical.min():.2f}, {sigma_empirical.max():.2f}] nL/s")
    print(f"  Median: {np.median(sigma_empirical):.2f} nL/s")
    print(f"\nσ_inferred (from Bayesian model = measurement + model error):")
    print(f"  Range: [{sigma_inferred.min():.2f}, {sigma_inferred.max():.2f}] nL/s")
    print(f"  Median: {np.median(sigma_inferred):.2f} nL/s")
    print(f"\nKey diagnostic - Ratio σ_inferred / σ_empirical:")
    print(f"  Median: {median_ratio:.1f}× (if ~1 → adequate, if >>1 → inadequate)")
    print(f"  Mean: {np.mean(ratio):.1f}×")
    print(f"\nModel error breakdown:")
    print(f"  ~{model_error_pct:.0f}% of variance from MODEL ERROR")
    print(f"  ~{100-model_error_pct:.0f}% of variance from measurement noise")
    print(f"\nStandardized residuals std:")
    print(f"  Using σ_empirical: {z_emp_std:.2f} (should be ~1 if model adequate)")
    print(f"  Using σ_inferred:  {np.std(z_inferred):.2f} (should be ~1 by construction)")
    print(f"\n{interpretation}")
    print("=" * 70)

    return {
        'sigma_empirical': sigma_empirical,
        'sigma_inferred': sigma_inferred,
        'edges': edges,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'ratio_median': median_ratio,
        'z_empirical_std': z_emp_std,
        'z_inferred_std': np.std(z_inferred),
        'model_error_pct': model_error_pct,
        'interpretation': interpretation,
        'figure': fig,
    }


def analyze_radius_error_contribution(
    G: nx.Graph,
    diagnostic_result: Dict[str, Any],
    sigma_R_rel: float = 0.10,
    figsize: Tuple[float, float] = (14, 10),
    show: bool = True,
) -> Dict[str, Any]:
    """
    Analyze whether radius measurement error explains the model inadequacy.

    Since Q ∝ R⁴ (Poiseuille), a relative error in R produces 4× relative error in Q:
        σ_Q / Q ≈ 4 × σ_R / R

    This function compares:
    - σ_model = √(σ²_inferred - σ²_empirical)  [observed model error]
    - σ_R_expected = 4 × σ_R_rel × |Q|  [expected error from radius uncertainty]

    If they match, radius uncertainty is the dominant error source.

    Parameters
    ----------
    G : nx.Graph
        Graph with flow and geometry attributes
    diagnostic_result : dict
        Output from compare_sigma_empirical_vs_inferred()
    sigma_R_rel : float
        Assumed relative uncertainty in radius (default 0.10 = 10%)
    figsize : tuple
        Figure size
    show : bool
        Whether to call plt.show()

    Returns
    -------
    dict with analysis results
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Extract data from diagnostic result
    sigma_empirical = diagnostic_result['sigma_empirical']
    sigma_inferred = diagnostic_result['sigma_inferred']
    edges = diagnostic_result['edges']

    # Get Q values for each edge
    Q_measured = []
    Q_predicted = []
    radii = []
    lengths = []

    for (u, v), sig_emp, sig_inf in zip(edges, sigma_empirical, sigma_inferred):
        data = G.edges[u, v]
        Q_meas = data.get('mean_Q_nL_s', data.get('mean_Q', 0))
        Q_pred = data.get('Q_predicted', Q_meas)
        R = data.get('radius_um', data.get('radius', 0))
        L = data.get('length_um', data.get('length', 0))

        Q_measured.append(abs(Q_meas) if Q_meas else 0)
        Q_predicted.append(abs(Q_pred) if Q_pred else 0)
        radii.append(R if R else 0)
        lengths.append(L if L else 0)

    Q_measured = np.array(Q_measured)
    Q_predicted = np.array(Q_predicted)
    radii = np.array(radii)
    lengths = np.array(lengths)

    # Compute model error: σ_model = √(σ²_inferred - σ²_empirical)
    # Clamp to avoid negative under sqrt
    sigma_model_sq = np.maximum(sigma_inferred**2 - sigma_empirical**2, 0)
    sigma_model = np.sqrt(sigma_model_sq)

    # Expected error from radius uncertainty: σ_R = 4 × σ_R_rel × |Q|
    # Use Q_predicted as reference (model-based)
    sigma_from_R = 4 * sigma_R_rel * Q_predicted

    # Compute residuals
    residuals = np.array([
        G.edges[u, v].get('mean_Q_nL_s', G.edges[u, v].get('mean_Q', 0)) -
        G.edges[u, v].get('Q_predicted', 0)
        for u, v in edges
    ])

    # Statistics
    valid_mask = (sigma_model > 0) & (sigma_from_R > 0)
    if valid_mask.sum() < 5:
        print("Not enough valid data points for analysis")
        return None

    # Linear regression: σ_model vs σ_from_R
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        sigma_from_R[valid_mask], sigma_model[valid_mask]
    )
    r_squared = r_value ** 2

    # Ratio of observed to expected
    ratio = sigma_model[valid_mask] / sigma_from_R[valid_mask]
    median_ratio = np.median(ratio)

    # Explained variance
    # If σ_model ≈ σ_from_R, then radius error explains most model error
    ss_model = np.sum(sigma_model[valid_mask]**2)
    ss_from_R = np.sum(sigma_from_R[valid_mask]**2)
    explained_fraction = min(ss_from_R / ss_model, 1.0) if ss_model > 0 else 0

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. σ_model vs σ_from_R scatter
    ax = axes[0, 0]
    ax.scatter(sigma_from_R, sigma_model, alpha=0.5, s=20)
    max_val = max(sigma_from_R.max(), sigma_model.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label='1:1 line')
    ax.plot([0, max_val], [intercept, intercept + slope * max_val], 'r-', lw=2,
            label=f'Fit: slope={slope:.2f}, R²={r_squared:.2f}')
    ax.set_xlabel(f'Expected σ from R error (4×{sigma_R_rel:.0%}×|Q|) [nL/s]')
    ax.set_ylabel('Observed σ_model [nL/s]')
    ax.set_title('Does Radius Error Explain Model Error?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # 2. Ratio histogram
    ax = axes[0, 1]
    ax.hist(ratio, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='k', linestyle='--', lw=2, label='Ratio = 1 (R error explains all)')
    ax.axvline(median_ratio, color='r', linestyle='-', lw=2, label=f'Median = {median_ratio:.2f}')
    ax.set_xlabel('σ_model / σ_from_R')
    ax.set_ylabel('Count')
    ax.set_title('Ratio: Observed / Expected from R Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Residuals vs Radius
    ax = axes[0, 2]
    ax.scatter(radii, np.abs(residuals), alpha=0.5, s=20, label='|Residual|')
    ax.scatter(radii, sigma_model, alpha=0.5, s=20, label='σ_model')
    ax.set_xlabel('Vessel Radius [μm]')
    ax.set_ylabel('nL/s')
    ax.set_title('Residuals vs Radius\n(look for trends)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Residuals vs |Q|
    ax = axes[1, 0]
    ax.scatter(Q_predicted, np.abs(residuals), alpha=0.5, s=20, c='blue', label='|Residual|')
    ax.scatter(Q_predicted, sigma_from_R, alpha=0.5, s=20, c='orange', label=f'4×{sigma_R_rel:.0%}×|Q|')
    ax.scatter(Q_predicted, sigma_model, alpha=0.5, s=20, c='green', label='σ_model')
    ax.set_xlabel('|Q_predicted| [nL/s]')
    ax.set_ylabel('nL/s')
    ax.set_title('Error Scaling with Flow Magnitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Residuals vs Length
    ax = axes[1, 1]
    ax.scatter(lengths, np.abs(residuals), alpha=0.5, s=20, label='|Residual|')
    ax.scatter(lengths, sigma_model, alpha=0.5, s=20, label='σ_model')
    ax.set_xlabel('Vessel Length [μm]')
    ax.set_ylabel('nL/s')
    ax.set_title('Residuals vs Length\n(look for trends)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Component breakdown
    ax = axes[1, 2]
    components = ['σ_empirical\n(measurement)', 'σ_from_R\n(radius error)', 'σ_model\n(observed)', 'σ_inferred\n(total)']
    values = [
        np.median(sigma_empirical),
        np.median(sigma_from_R),
        np.median(sigma_model),
        np.median(sigma_inferred)
    ]
    colors = ['steelblue', 'orange', 'green', 'red']
    bars = ax.bar(components, values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Median σ [nL/s]')
    ax.set_title('Error Component Breakdown')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Interpretation
    if median_ratio < 0.5:
        interp = (
            f"Radius error ({sigma_R_rel:.0%}) OVER-EXPLAINS model error\n"
            f"→ Actual radius uncertainty may be < {sigma_R_rel:.0%}"
        )
    elif median_ratio < 1.5:
        interp = (
            f"Radius error ({sigma_R_rel:.0%}) EXPLAINS model error well\n"
            f"→ R⁴ sensitivity is the dominant error source"
        )
    elif median_ratio < 3.0:
        interp = (
            f"Radius error ({sigma_R_rel:.0%}) explains ~{1/median_ratio:.0%} of model error\n"
            f"→ Other sources contribute (topology, model violations)"
        )
    else:
        interp = (
            f"Radius error ({sigma_R_rel:.0%}) explains only ~{1/median_ratio:.0%} of model error\n"
            f"→ Major non-radius error sources present"
        )

    fig.suptitle(
        f'Radius Error Analysis (assuming σ_R/R = {sigma_R_rel:.0%})\n'
        f'Median(σ_model / σ_from_R) = {median_ratio:.2f} | '
        f'R² = {r_squared:.2f} | '
        f'Explained: ~{explained_fraction:.0%}\n'
        f'{interp}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    if show:
        plt.show(block=False)

    # Print summary
    print("\n" + "=" * 70)
    print("RADIUS ERROR ANALYSIS")
    print("=" * 70)
    print(f"\nAssumed radius uncertainty: σ_R/R = {sigma_R_rel:.0%}")
    print(f"  → Expected flow error: σ_Q/Q = 4 × {sigma_R_rel:.0%} = {4*sigma_R_rel:.0%}")
    print(f"\nMedian values:")
    print(f"  σ_empirical (measurement noise): {np.median(sigma_empirical):.2f} nL/s")
    print(f"  σ_from_R (expected from R error): {np.median(sigma_from_R):.2f} nL/s")
    print(f"  σ_model (observed model error):   {np.median(sigma_model):.2f} nL/s")
    print(f"  σ_inferred (total):               {np.median(sigma_inferred):.2f} nL/s")
    print(f"\nRatio σ_model / σ_from_R:")
    print(f"  Median: {median_ratio:.2f}")
    print(f"  (1.0 = R error explains all, >1 = other sources, <1 = R error over-explains)")
    print(f"\nCorrelation: R² = {r_squared:.2f}")
    print(f"Variance explained by R error: ~{explained_fraction:.0%}")
    print(f"\n{interp}")
    print("=" * 70)

    return {
        'sigma_model': sigma_model,
        'sigma_from_R': sigma_from_R,
        'sigma_empirical': sigma_empirical,
        'sigma_inferred': sigma_inferred,
        'Q_predicted': Q_predicted,
        'radii': radii,
        'lengths': lengths,
        'residuals': residuals,
        'ratio_median': median_ratio,
        'r_squared': r_squared,
        'explained_fraction': explained_fraction,
        'interpretation': interp,
        'figure': fig,
    }
