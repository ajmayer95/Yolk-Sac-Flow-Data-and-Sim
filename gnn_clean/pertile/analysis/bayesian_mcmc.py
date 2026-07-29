"""
Bayesian pressure inference using MCMC sampling.

This provides proper posterior distributions over boundary pressures,
as opposed to the Laplace approximation which assumes Gaussian posteriors.

Key advantages over Laplace:
- Robust (Student-t) likelihood for handling outlier measurements
- Full posterior samples for rigorous uncertainty quantification
- Model diagnostics (R-hat, ESS, divergences)

Supports two backends:
- NumPyro (JAX-based, preferred): pip install numpyro jax jaxlib
- PyMC (Theano-based): pip install pymc arviz
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional, List
import networkx as nx

# Check for NumPyro availability (preferred)
try:
    # Enable parallel CPU chains before importing JAX
    import os
    os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    # Enable 4 CPU devices for parallel MCMC chains
    numpyro.set_host_device_count(4)

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

# Check for PyMC availability (fallback)
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

# At least one backend must be available
MCMC_AVAILABLE = NUMPYRO_AVAILABLE or PYMC_AVAILABLE


@dataclass
class MCMCResult:
    """Results from MCMC pressure inference."""
    # Point estimates (posterior mean)
    mu_cP: float
    node_pressures: Dict[int, float]  # Posterior mean
    predicted_Q: Dict[Tuple[int, int], float]
    measured_Q: Dict[Tuple[int, int], float]

    # Uncertainties (from posterior)
    P_std: Dict[int, float]  # Posterior std dev
    P_ci95: Dict[int, Tuple[float, float]]  # 95% credible intervals
    Q_pred_std: Dict[Tuple[int, int], float]
    Q_pred_ci95: Dict[Tuple[int, int], Tuple[float, float]]

    # Fit quality
    r_squared: float
    rmse: float
    chi2_reduced: float

    # Diagnostics
    converged: bool
    r_hat_max: float
    ess_min: float
    n_divergences: int

    # Model info
    boundary_nodes: Set[int]
    ref_node: int
    n_parameters: int
    n_measurements: int

    # Inferred noise scale (if infer_sigma=True)
    sigma_scale: Optional[float] = None  # Multiplier on input sigma_Q values (uniform model)

    # Heteroscedastic noise model (if heteroscedastic=True)
    # Basic: σ_i = σ_base + σ_rel × |Q_i|
    # Extended (with chi2): σ_i = (σ_base + σ_rel × |Q_i|) × max(1, χ²_i)^α
    sigma_base: Optional[float] = None  # Base uncertainty (nL/s)
    sigma_rel: Optional[float] = None   # Flow-proportional uncertainty (dimensionless)
    chi2_alpha: Optional[float] = None  # Exponent for chi2 amplification factor

    # Full posterior samples (for custom analyses)
    trace: Optional[object] = None  # ArviZ InferenceData

    # All edge predictions (measured AND unmeasured)
    # Maps (u, v) -> {'Q_mean', 'Q_std', 'Q_ci95_lo', 'Q_ci95_hi', 'direction_confidence', 'is_measured'}
    all_edge_predictions: Optional[Dict[Tuple[int, int], dict]] = None

    # Shrinkage diagnostics (for identifying prior-dominated nodes)
    # shrinkage[node] = 1 - (posterior_std / prior_std)^2
    # shrinkage ≈ 0: posterior ≈ prior (no data influence)
    # shrinkage ≈ 1: posterior << prior (data-dominated)
    P_shrinkage: Optional[Dict[int, float]] = None
    P_prior_std: Optional[Dict[int, float]] = None  # Prior std for each node (σ_k^IRLS from IRLS covariance)
    P_prior_mean: Optional[Dict[int, float]] = None  # Prior mean (P_k^IRLS)

    # Data influence scores for edges
    # Maps (u, v) -> fraction of variance from well-constrained (shrinkage > 0.2) boundary nodes
    edge_data_influence: Optional[Dict[Tuple[int, int], float]] = None


def build_design_matrix(
    G: nx.Graph,
    measured_edges: List[dict],
    boundary_list: List[int],
    internal_nodes: List[int],
    ref_node: int,
    M: np.ndarray,
) -> np.ndarray:
    """
    Build design matrix A such that Q_pred = A @ P_boundary.

    Args:
        G: Graph
        measured_edges: List of edge dicts from _prepare_bayesian_data
        boundary_list: All boundary nodes
        internal_nodes: Internal (non-boundary) nodes
        ref_node: Reference node (P=0)
        M: Matrix mapping boundary pressures to internal pressures

    Returns:
        A: (n_edges, n_free_boundary) design matrix
    """
    # Free boundary nodes (excluding reference)
    free_boundary = [n for n in boundary_list if n != ref_node]
    n_free = len(free_boundary)
    n_edges = len(measured_edges)

    boundary_idx = {n: i for i, n in enumerate(free_boundary)}
    internal_idx = {n: i for i, n in enumerate(internal_nodes)}

    A = np.zeros((n_edges, n_free))

    for idx, e in enumerate(measured_edges):
        g0 = e['g0']  # Conductance at μ=1 Pa·s

        # For each node in the edge, compute contribution to A
        # Q = g * (P_start - P_end)
        # Positive Q means flow from start to end, so P_start > P_end
        for node, sign in [(e['start_j'], +1), (e['end_j'], -1)]:
            if node == ref_node:
                # Reference node has P=0, no contribution
                pass
            elif node in free_boundary:
                # Direct boundary node
                j = boundary_idx[node]
                A[idx, j] += sign * g0
            elif node in internal_nodes:
                # Internal node: P_internal = M @ P_boundary
                i = internal_idx[node]
                A[idx, :] += sign * g0 * M[i, :]

    return A


def compute_irls_covariance(
    A: np.ndarray,
    Q_obs: np.ndarray,
    sigma_Q: np.ndarray,
    mu_Pa_s: float = 3.5e-3,
    nu: float = 4.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    lambda_ridge: float = 1e-6,  # Ridge regularization for numerical stability
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute IRLS (Iteratively Reweighted Least Squares) solution with covariance.

    Uses Student-t weights for robustness. The covariance matrix encodes
    which boundary nodes are well-constrained vs poorly-constrained by the data.

    Ridge regularization (lambda_ridge) is added to stabilize ill-conditioned systems.

    Args:
        A: Design matrix (n_edges, n_free_boundary)
        Q_obs: Observed flows (n_edges,)
        sigma_Q: Flow uncertainties (n_edges,)
        mu_Pa_s: Viscosity in Pa·s
        nu: Degrees of freedom for Student-t weights
        max_iter: Maximum IRLS iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        P_irls: IRLS solution (n_free,)
        P_std: Standard deviations from covariance (n_free,)
        Cov: Full covariance matrix (n_free, n_free)
    """
    n_edges, n_free = A.shape

    # Scale A by viscosity
    A_scaled = A / mu_Pa_s

    # Initial weights (inverse variance)
    w = 1.0 / sigma_Q**2

    # IRLS with Student-t weights and ridge regularization
    P = np.zeros(n_free)
    for iteration in range(max_iter):
        # Weighted least squares with ridge: (A'WA + λI)^{-1} A'W Q
        W = np.diag(w)
        AtWA = A_scaled.T @ W @ A_scaled
        AtWQ = A_scaled.T @ W @ Q_obs

        # Add ridge regularization for numerical stability
        AtWA_reg = AtWA + lambda_ridge * np.eye(n_free)

        try:
            P_new = np.linalg.solve(AtWA_reg, AtWQ)
        except np.linalg.LinAlgError:
            # Still singular - use pseudoinverse
            P_new = np.linalg.lstsq(AtWA_reg, AtWQ, rcond=None)[0]

        # Check convergence
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            if verbose:
                print(f"  IRLS converged in {iteration + 1} iterations")
            break
        P = P_new

        # Update weights using Student-t reweighting
        residuals = Q_obs - A_scaled @ P
        z2 = (residuals / sigma_Q)**2
        # Student-t weight: (nu + 1) / (nu + z^2)
        w = (nu + 1) / (nu + z2) / sigma_Q**2

    # Compute covariance from final weights (with regularization)
    W = np.diag(w)
    AtWA = A_scaled.T @ W @ A_scaled
    AtWA_reg = AtWA + lambda_ridge * np.eye(n_free)

    try:
        Cov = np.linalg.inv(AtWA_reg)
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(Cov)
        if np.any(eigvals < 0):
            if verbose:
                print("  Warning: Covariance not positive definite, using pseudoinverse")
            Cov = np.linalg.pinv(AtWA_reg)
    except np.linalg.LinAlgError:
        if verbose:
            print("  Warning: Hessian singular, using pseudoinverse")
        Cov = np.linalg.pinv(AtWA_reg)

    P_std = np.sqrt(np.maximum(np.diag(Cov), 0))

    if verbose:
        print(f"  IRLS solution range: [{P.min():.1f}, {P.max():.1f}] Pa")
        print(f"  IRLS std range: [{P_std.min():.1f}, {P_std.max():.1f}] Pa")
        # Identify poorly-constrained nodes
        high_std_mask = P_std > 1000
        if np.any(high_std_mask):
            n_high = np.sum(high_std_mask)
            print(f"  ⚠ {n_high} boundary nodes with σ > 1000 Pa (poorly constrained)")

    # Check for pathological IRLS solution
    if not np.isfinite(P).all() or np.max(np.abs(P)) > 1e6:
        raise ValueError(
            f"IRLS solution is pathological: range [{P.min():.2e}, {P.max():.2e}] Pa\n"
            f"This usually indicates:\n"
            f"  - Conductance values (g0) are wrong (check units)\n"
            f"  - Some vessel lengths are zero or near-zero\n"
            f"  - Disconnected boundary nodes with no measured vessels\n"
            f"  - Design matrix A is rank-deficient\n"
            f"Check the diagnostic output above for clues."
        )

    return P, P_std, Cov


def solve_wls_with_chi2_correction(
    A: np.ndarray,
    Q_obs: np.ndarray,
    sigma_Q: np.ndarray,
    mu_Pa_s: float = 3.5e-3,
    use_studentt_weights: bool = True,
    nu: float = 4.0,
    verbose: bool = True,
) -> dict:
    """
    Solve for boundary pressures using Weighted Least Squares with χ² correction.

    This is a simpler alternative to full MCMC that gives equivalent results
    for the linear forward model. The χ²_red correction calibrates uncertainties:

        σ_corrected = σ × √χ²_red  (if χ²_red > 1)

    Student-t reweighting provides outlier robustness without full Bayesian machinery.

    Args:
        A: Design matrix (n_edges, n_free_boundary)
        Q_obs: Observed flows (n_edges,)
        sigma_Q: Flow uncertainties (n_edges,)
        mu_Pa_s: Viscosity in Pa·s
        use_studentt_weights: If True, use IRLS with Student-t weights for outlier robustness
        nu: Degrees of freedom for Student-t (lower = more robust, typically 4)
        verbose: Print diagnostics

    Returns:
        dict with:
            P: Boundary pressures (n_free,)
            P_std: Pressure uncertainties (n_free,)
            P_std_corrected: Uncertainties scaled by √χ²_red (n_free,)
            Q_pred: Predicted flows (n_edges,)
            Q_pred_std: Flow prediction uncertainties (n_edges,)
            chi2: Raw chi-squared
            chi2_reduced: χ²/(n-p)
            sigma_scale: √χ²_red (uncertainty correction factor)
            n_measurements: Number of measurements (n)
            n_parameters: Number of parameters (p)
            r_squared: Coefficient of determination
            rmse: Root mean squared error
    """
    n_edges, n_free = A.shape
    A_scaled = A / mu_Pa_s

    if verbose:
        print(f"\n  WLS Setup: {n_edges} measurements, {n_free} boundary pressures")
        print(f"  Degrees of freedom: {n_edges - n_free}")

    # Solve using IRLS (handles outliers if use_studentt_weights=True)
    if use_studentt_weights:
        P, P_std, Cov = compute_irls_covariance(
            A, Q_obs, sigma_Q, mu_Pa_s, nu=nu, verbose=verbose
        )
        if verbose:
            print(f"  Using Student-t weights (ν={nu}) for outlier robustness")
    else:
        # Pure WLS (no outlier robustness)
        w = 1.0 / sigma_Q**2
        W = np.diag(w)
        AtWA = A_scaled.T @ W @ A_scaled
        AtWQ = A_scaled.T @ W @ Q_obs

        # Add small ridge for numerical stability
        AtWA_reg = AtWA + 1e-6 * np.eye(n_free)
        P = np.linalg.solve(AtWA_reg, AtWQ)
        Cov = np.linalg.inv(AtWA_reg)
        P_std = np.sqrt(np.maximum(np.diag(Cov), 0))

        if verbose:
            print(f"  Using standard WLS (no outlier robustness)")

    # Compute predicted flows
    Q_pred = A_scaled @ P
    residuals = Q_obs - Q_pred

    # Chi-squared statistics
    chi2 = np.sum((residuals / sigma_Q)**2)
    dof = n_edges - n_free
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    # Uncertainty correction factor
    # If χ²_red > 1, uncertainties were underestimated
    # If χ²_red < 1, uncertainties were overestimated (but we don't shrink below 1)
    sigma_scale = np.sqrt(max(1.0, chi2_reduced)) if np.isfinite(chi2_reduced) else 1.0

    # Corrected uncertainties
    P_std_corrected = P_std * sigma_scale
    Cov_corrected = Cov * sigma_scale**2

    # Propagate uncertainty to predicted flows
    # Var(Q_pred) = A @ Cov @ A.T (diagonal elements)
    Q_pred_var = np.sum((A_scaled @ Cov_corrected) * A_scaled, axis=1)
    Q_pred_std = np.sqrt(np.maximum(Q_pred_var, 0))

    # Fit metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Q_obs - np.mean(Q_obs))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))

    if verbose:
        print(f"\n  Results:")
        print(f"    χ² = {chi2:.1f}")
        print(f"    χ²_red = {chi2_reduced:.3f}  (n={n_edges}, p={n_free}, dof={dof})")
        if chi2_reduced > 1:
            print(f"    → Uncertainties underestimated, scaling by √χ²_red = {sigma_scale:.3f}")
        elif chi2_reduced < 1:
            print(f"    → Uncertainties overestimated (χ²_red < 1), no correction applied")
        else:
            print(f"    → Uncertainties well-calibrated")
        print(f"    R² = {r_squared:.4f}")
        print(f"    RMSE = {rmse:.3f} nL/s")
        print(f"    Pressure range: [{P.min():.1f}, {P.max():.1f}] Pa")

    return {
        'P': P,
        'P_std': P_std,
        'P_std_corrected': P_std_corrected,
        'Cov': Cov,
        'Cov_corrected': Cov_corrected,
        'Q_pred': Q_pred,
        'Q_pred_std': Q_pred_std,
        'residuals': residuals,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced,
        'sigma_scale': sigma_scale,
        'n_measurements': n_edges,
        'n_parameters': n_free,
        'dof': dof,
        'r_squared': r_squared,
        'rmse': rmse,
    }


def compute_shrinkage_diagnostics(
    posterior_std: Dict[int, float],
    prior_std: Dict[int, float],
    free_boundary: List[int],
    sigma_scale: Optional[float] = None,
) -> Dict[int, float]:
    """
    Compute shrinkage diagnostic for each boundary node.

    We compare posterior_std to the "effective prior std" which accounts
    for sigma_scale inflation of measurement uncertainties.

    If sigma_scale > 1, the data told us uncertainties were underestimated,
    so we expect posteriors to be wider. The effective prior std is:
        effective_prior_std = prior_std * sigma_scale

    Shrinkage = 1 - (posterior_std / effective_prior_std)^2

    Interpretation:
    - shrinkage ≈ 0: posterior ≈ effective prior (weakly constrained by network)
    - shrinkage ≈ 1: posterior << effective prior (strongly constrained by network)
    - shrinkage < 0: posterior > effective prior (clamped to 0)

    Note: When sigma_scale > 1, ALL posteriors widen, but some may widen less
    than others depending on network connectivity. Those that widen less are
    more constrained by the network topology.

    Args:
        posterior_std: Posterior std from MCMC
        prior_std: Prior std used in MCMC (from IRLS at sigma_scale=1)
        free_boundary: List of free boundary nodes
        sigma_scale: Inferred noise multiplier (if available)

    Returns:
        Dict mapping node -> shrinkage value (0-1, higher = more constrained)
    """
    # Scale prior std by sigma_scale to get "effective" prior
    scale = sigma_scale if sigma_scale is not None else 1.0

    shrinkage = {}
    for node in free_boundary:
        post_s = posterior_std.get(node, 0.0)
        prior_s = prior_std.get(node, 1.0) * scale

        if prior_s > 0:
            ratio = post_s / prior_s
            shrinkage[node] = max(0.0, 1.0 - ratio**2)
        else:
            shrinkage[node] = 1.0  # If prior_std = 0, it's fully constrained

    return shrinkage


def compute_edge_data_influence(
    G: nx.Graph,
    shrinkage: Dict[int, float],
    free_boundary: List[int],
    ref_node: int,
    internal_nodes: List[int],
    M: np.ndarray,
    threshold: float = 0.2,
) -> Dict[Tuple[int, int], float]:
    """
    Compute data influence score for each edge's predicted flow.

    For each edge (u, v), we compute what fraction of the flow prediction
    uncertainty comes from well-constrained (data-dominated) boundary nodes
    vs poorly-constrained (prior-dominated) boundary nodes.

    This helps identify which predicted flows are trustworthy (data-driven)
    vs which are essentially prior-driven guesses.

    Args:
        G: NetworkX graph
        shrinkage: Dict mapping boundary node -> shrinkage value
        free_boundary: List of free boundary nodes
        ref_node: Reference node (P=0)
        internal_nodes: List of internal nodes
        M: Matrix mapping boundary -> internal pressures
        threshold: Shrinkage threshold for "data-dominated" (default 0.2)

    Returns:
        Dict mapping (u, v) -> data influence score (0-1)
        0 = entirely prior-driven, 1 = entirely data-driven
    """
    boundary_idx = {n: i for i, n in enumerate(free_boundary)}
    internal_idx = {n: i for i, n in enumerate(internal_nodes)}

    # For each internal node, compute its dependence on boundary nodes
    # P_internal[i] = sum_j M[i,j] * P_boundary[j]
    # So internal node i's "data influence" is weighted avg of boundary shrinkages

    internal_data_influence = {}
    for i, node in enumerate(internal_nodes):
        weights = np.abs(M[i, :])
        total_weight = np.sum(weights)
        if total_weight > 0:
            weighted_shrinkage = 0.0
            for j, b_node in enumerate(free_boundary):
                weighted_shrinkage += weights[j] * shrinkage.get(b_node, 0.0)
            internal_data_influence[node] = weighted_shrinkage / total_weight
        else:
            internal_data_influence[node] = 0.0

    # For each edge, compute data influence based on its endpoints
    edge_influence = {}

    for u, v in G.edges():
        # Get data influence for each endpoint
        influences = []

        for node in [u, v]:
            if node == ref_node:
                # Reference node is perfectly constrained (P=0 exactly)
                influences.append(1.0)
            elif node in shrinkage:
                # Direct boundary node
                influences.append(shrinkage[node])
            elif node in internal_data_influence:
                # Internal node
                influences.append(internal_data_influence[node])
            else:
                # Unknown node
                influences.append(0.0)

        # Edge influence is the minimum of its endpoints
        # (weakest link determines reliability)
        edge_influence[(u, v)] = min(influences)

    return edge_influence


def run_mcmc_inference_numpyro(
    A: np.ndarray,
    Q_obs: np.ndarray,
    sigma_Q: np.ndarray,
    mu_Pa_s: float = 3.5e-3,
    likelihood: str = 'gaussian',
    nu: float = 4.0,
    P_prior_std: Optional[np.ndarray] = None,  # Per-node prior std (from IRLS covariance)
    P_prior_std_default: float = 500.0,  # Default if no IRLS covariance available
    P_prior_std_scale: float = 2.0,  # Scale factor on IRLS std
    n_warmup: int = 1000,
    n_samples: int = 2000,
    n_chains: int = 4,
    init_P: Optional[np.ndarray] = None,
    infer_sigma: bool = False,
    heteroscedastic: bool = False,
    chi2_reduced: Optional[np.ndarray] = None,  # Per-edge chi2_reduced for extended model
    verbose: bool = True,
) -> Tuple[object, np.ndarray, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Run MCMC inference using NumPyro (JAX backend).

    Args:
        A: Design matrix (n_edges, n_free_boundary)
        Q_obs: Observed flows (n_edges,)
        sigma_Q: Flow uncertainties (n_edges,)
        mu_Pa_s: Viscosity in Pa·s
        likelihood: 'gaussian' or 'studentt'
        nu: Degrees of freedom for Student-t
        P_prior_std: Per-node prior std from IRLS covariance (n_free,). If None, uses default.
        P_prior_std_default: Default prior std if P_prior_std not provided (Pa)
        P_prior_std_scale: Scale factor applied to IRLS-derived prior std
        n_warmup: Warmup iterations
        n_samples: Sampling iterations per chain
        n_chains: Number of chains
        init_P: Initial pressure guess (IRLS solution as prior mean)
        infer_sigma: If True, infer a global noise scale multiplier (uniform model)
        heteroscedastic: If True, use heteroscedastic noise model.
                        If chi2_reduced is provided: σ_i = σ_base + σ_rel × |Q_i| + σ_chi × max(0, χ²_i - 1)
                        Otherwise: σ_i = σ_base + σ_rel × |Q_i|
                        This overrides infer_sigma.
        chi2_reduced: Per-edge chi2_reduced values (n_edges,) for extended heteroscedastic model.
                     Vessels with poor profile fits (high χ²) get larger uncertainty.
        verbose: Print progress

    Returns:
        mcmc: NumPyro MCMC object
        P_samples: (n_samples * n_chains, n_free) pressure samples
        sigma_scale: Inferred sigma multiplier (uniform model), or None
        sigma_base: Inferred base uncertainty (heteroscedastic model), or None
        sigma_rel: Inferred flow-proportional uncertainty (heteroscedastic model), or None
        sigma_chi: Inferred chi2 uncertainty coefficient (extended model), or None
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro not available. Install with: pip install numpyro jax jaxlib")

    n_edges, n_free = A.shape

    # Scale A by viscosity
    A_scaled = A / mu_Pa_s

    # Convert to JAX arrays
    A_jax = jnp.array(A_scaled)
    Q_obs_jax = jnp.array(Q_obs)
    sigma_Q_jax = jnp.array(sigma_Q)
    abs_Q_obs_jax = jnp.abs(Q_obs_jax)

    # Check if we should use chi2 in the extended heteroscedastic model
    use_chi2 = heteroscedastic and chi2_reduced is not None

    # Use init_P as prior mean if available, otherwise use 0
    if init_P is not None:
        P_prior_mean = jnp.array(init_P)
    else:
        P_prior_mean = jnp.zeros(n_free)

    # Set up per-node prior std
    if P_prior_std is not None:
        # Use IRLS covariance-derived prior, scaled
        P_prior_std_arr = jnp.array(P_prior_std * P_prior_std_scale)
        # Ensure minimum prior std to avoid numerical issues
        P_prior_std_arr = jnp.maximum(P_prior_std_arr, 10.0)
        prior_type = "IRLS-adaptive"
    else:
        # Use uniform prior std
        P_prior_std_arr = jnp.full(n_free, P_prior_std_default)
        prior_type = "uniform"

    if verbose:
        print(f"  MCMC setup (NumPyro): {n_free} boundary pressures, {n_edges} measurements")
        print(f"  Likelihood: {likelihood}" + (f" (ν={nu})" if likelihood == 'studentt' else ""))
        if heteroscedastic:
            if use_chi2:
                print(f"  Noise model: EXTENDED HETEROSCEDASTIC (σ_i = (σ_base + σ_rel × |Q_i|) × max(1, χ²_i)^α)")
                chi2_arr = np.array(chi2_reduced)
                print(f"    χ² range: [{chi2_arr.min():.2f}, {chi2_arr.max():.2f}]")
            else:
                print(f"  Noise model: HETEROSCEDASTIC (σ_i = σ_base + σ_rel × |Q_i|)")
        elif infer_sigma:
            print(f"  Noise model: Uniform scale (σ_i = σ_scale × σ_Q_i)")
        if prior_type == "IRLS-adaptive":
            print(f"  Prior: N(IRLS, {P_prior_std_scale:.1f}×σ_IRLS) - adaptive per-node")
            print(f"    Prior std range: [{float(P_prior_std_arr.min()):.1f}, {float(P_prior_std_arr.max()):.1f}] Pa")
        else:
            print(f"  Prior: N(0, {P_prior_std_default:.0f} Pa) - uniform")
        print(f"  Chains: {n_chains}, samples: {n_samples}, warmup: {n_warmup}")

    # Define NumPyro model
    def model():
        # Prior on boundary pressures: Gaussian centered on IRLS solution
        # with per-node std from IRLS covariance (or uniform default)
        P_free = numpyro.sample('P_free', dist.Normal(P_prior_mean, P_prior_std_arr))

        # Forward model: predicted flows
        Q_pred = jnp.dot(A_jax, P_free)

        # Noise model
        if heteroscedastic:
            # Heteroscedastic model with optional chi2 term
            # Priors based on analysis: σ_base ~ 1-2 nL/s, σ_rel ~ 0.3-0.5
            sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(5.0))
            sigma_rel = numpyro.sample('sigma_rel', dist.HalfNormal(1.0))
            base_sigma = sigma_base + sigma_rel * abs_Q_obs_jax

            if use_chi2:
                # Extended model: multiply by max(1, χ²)^α
                # This amplifies ALL noise for vessels with poor fits
                # α prior: expect ~0.3-0.7 (sqrt-ish scaling)
                chi2_alpha = numpyro.sample('chi2_alpha', dist.HalfNormal(0.5))
                # chi2_factor = max(1, χ²)^α - vessels with χ² <= 1 get factor = 1
                chi2_factor = jnp.power(jnp.maximum(1.0, jnp.array(chi2_reduced)), chi2_alpha)
                effective_sigma = base_sigma * chi2_factor
            else:
                effective_sigma = base_sigma

            # Ensure minimum sigma to avoid numerical issues
            effective_sigma = jnp.maximum(effective_sigma, 0.1)
        elif infer_sigma:
            # Uniform scaling model: σ_i = σ_scale × σ_Q_i
            sigma_scale = numpyro.sample('sigma_scale', dist.HalfNormal(10.0))
            effective_sigma = sigma_scale * sigma_Q_jax
        else:
            effective_sigma = sigma_Q_jax

        # Likelihood
        if likelihood == 'gaussian':
            numpyro.sample('Q_obs', dist.Normal(Q_pred, effective_sigma), obs=Q_obs_jax)
        elif likelihood == 'studentt':
            numpyro.sample('Q_obs', dist.StudentT(nu, Q_pred, effective_sigma), obs=Q_obs_jax)
        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")

    # Set up MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains,
                progress_bar=verbose)

    # Run sampling
    if verbose:
        print("  Sampling...")

    # Use random key
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    # Extract samples
    samples = mcmc.get_samples()
    P_samples = np.array(samples['P_free'])  # (n_samples * n_chains, n_free)

    # Extract inferred noise parameters
    inferred_sigma_scale = None
    inferred_sigma_base = None
    inferred_sigma_rel = None
    inferred_chi2_alpha = None

    if heteroscedastic:
        if 'sigma_base' in samples:
            sigma_base_samples = np.array(samples['sigma_base'])
            inferred_sigma_base = float(np.mean(sigma_base_samples))
            if verbose:
                print(f"  Inferred σ_base: {inferred_sigma_base:.3f} nL/s "
                      f"(95% CI: [{np.percentile(sigma_base_samples, 2.5):.3f}, "
                      f"{np.percentile(sigma_base_samples, 97.5):.3f}])")
        if 'sigma_rel' in samples:
            sigma_rel_samples = np.array(samples['sigma_rel'])
            inferred_sigma_rel = float(np.mean(sigma_rel_samples))
            if verbose:
                print(f"  Inferred σ_rel: {inferred_sigma_rel:.3f} "
                      f"(95% CI: [{np.percentile(sigma_rel_samples, 2.5):.3f}, "
                      f"{np.percentile(sigma_rel_samples, 97.5):.3f}])")
        if 'chi2_alpha' in samples:
            chi2_alpha_samples = np.array(samples['chi2_alpha'])
            inferred_chi2_alpha = float(np.mean(chi2_alpha_samples))
            if verbose:
                print(f"  Inferred χ² exponent α: {inferred_chi2_alpha:.3f} "
                      f"(95% CI: [{np.percentile(chi2_alpha_samples, 2.5):.3f}, "
                      f"{np.percentile(chi2_alpha_samples, 97.5):.3f}])")

        # Show effective sigma examples
        if verbose and inferred_sigma_base is not None and inferred_sigma_rel is not None:
            print(f"\n  Effective σ examples:")
            base_1 = inferred_sigma_base + inferred_sigma_rel
            base_10 = inferred_sigma_base + 10 * inferred_sigma_rel
            print(f"    |Q|=1,  χ²=1.0: {base_1:.2f} nL/s")
            print(f"    |Q|=10, χ²=1.0: {base_10:.2f} nL/s")
            if inferred_chi2_alpha is not None:
                factor_5 = 5.0 ** inferred_chi2_alpha  # max(1, 5)^α = 5^α
                print(f"    |Q|=1,  χ²=5.0: {base_1 * factor_5:.2f} nL/s (×{factor_5:.2f})")
                print(f"    |Q|=10, χ²=5.0: {base_10 * factor_5:.2f} nL/s (×{factor_5:.2f})")

    elif infer_sigma and 'sigma_scale' in samples:
        sigma_samples = np.array(samples['sigma_scale'])
        inferred_sigma_scale = float(np.mean(sigma_samples))
        if verbose:
            print(f"  Inferred sigma_scale: {inferred_sigma_scale:.2f} "
                  f"(95% CI: [{np.percentile(sigma_samples, 2.5):.2f}, "
                  f"{np.percentile(sigma_samples, 97.5):.2f}])")

    return mcmc, P_samples, inferred_sigma_scale, inferred_sigma_base, inferred_sigma_rel, inferred_chi2_alpha


def check_convergence_numpyro(mcmc, verbose: bool = True, param_names: Optional[List[str]] = None) -> Tuple[bool, float, float, int]:
    """
    Check MCMC convergence diagnostics for NumPyro.

    Args:
        mcmc: NumPyro MCMC object
        verbose: Print diagnostics
        param_names: List of parameter names to check. Defaults to ['P_free'].
                    For pulsatile model, use ['P_mean', 'P_re', 'P_im'].

    Returns:
        converged: True if all diagnostics pass
        r_hat_max: Maximum R-hat across parameters
        ess_min: Minimum effective sample size
        n_divergences: Number of divergent transitions
    """
    if not NUMPYRO_AVAILABLE:
        return True, 1.0, 1000, 0

    if param_names is None:
        param_names = ['P_free']

    from numpyro.diagnostics import summary, print_summary

    samples = mcmc.get_samples(group_by_chain=True)

    # Use numpyro diagnostics
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    # Compute R-hat and ESS for all specified parameters
    all_r_hat = []
    all_ess = []
    for param_name in param_names:
        if param_name in samples:
            P_samples = samples[param_name]  # (n_chains, n_samples, n_params)
            r_hat_values = split_gelman_rubin(P_samples)
            ess_values = effective_sample_size(P_samples)
            all_r_hat.append(np.nanmax(r_hat_values))
            all_ess.append(np.nanmin(ess_values))

    r_hat_max = float(np.max(all_r_hat)) if all_r_hat else 1.0
    ess_min = float(np.min(all_ess)) if all_ess else 1000

    # Count divergences
    n_divergences = int(np.sum(mcmc.get_extra_fields()['diverging']))

    # Check thresholds
    converged = (r_hat_max < 1.01) and (ess_min > 400) and (n_divergences < 10)

    if verbose:
        status = "✓ CONVERGED" if converged else "⚠ WARNING"
        print(f"  Diagnostics: {status}")
        print(f"    R-hat max: {r_hat_max:.3f} (should be < 1.01)")
        print(f"    ESS min: {ess_min:.0f} (should be > 400)")
        print(f"    Divergences: {n_divergences} (should be < 10)")

    return converged, r_hat_max, ess_min, n_divergences


def run_mcmc_inference(
    A: np.ndarray,
    Q_obs: np.ndarray,
    sigma_Q: np.ndarray,
    mu_Pa_s: float = 3.5e-3,
    likelihood: str = 'gaussian',
    nu: float = 4.0,
    P_prior_upper: float = 16000.0,  # ~120 mmHg in Pa
    n_tune: int = 1000,
    n_draws: int = 2000,
    n_chains: int = 4,
    init_P: Optional[np.ndarray] = None,
    infer_sigma: bool = False,
    verbose: bool = True,
) -> Tuple[object, np.ndarray, Optional[float]]:
    """
    Run MCMC inference for boundary pressures.

    Args:
        A: Design matrix (n_edges, n_free_boundary)
        Q_obs: Observed flows (n_edges,)
        sigma_Q: Flow uncertainties (n_edges,) - used as relative weights if infer_sigma=True
        mu_Pa_s: Viscosity in Pa·s (fixed, not inferred here)
        likelihood: 'gaussian' or 'studentt'
        nu: Degrees of freedom for Student-t (lower = more robust)
        P_prior_upper: Upper bound for uniform pressure prior (Pa)
        n_tune: Tuning iterations
        n_draws: Sampling iterations per chain
        n_chains: Number of chains
        init_P: Initial pressure guess (for warm start)
        infer_sigma: If True, infer a global noise scale multiplier from data
        verbose: Print progress

    Returns:
        trace: ArviZ InferenceData object
        P_samples: (n_samples, n_free) pressure samples
        sigma_scale: Inferred sigma multiplier (posterior mean), or None if not inferred
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC not available. Install with: pip install pymc arviz")

    n_edges, n_free = A.shape

    # Scale A by viscosity (A was computed at μ=1 Pa·s)
    A_scaled = A / mu_Pa_s

    if verbose:
        print(f"  MCMC setup: {n_free} boundary pressures, {n_edges} measurements")
        print(f"  Likelihood: {likelihood}" + (f" (ν={nu})" if likelihood == 'studentt' else ""))
        if infer_sigma:
            print(f"  Inferring global noise scale (sigma_scale)")
        print(f"  Chains: {n_chains}, draws: {n_draws}, tune: {n_tune}")

    with pm.Model() as model:
        # Prior on boundary pressures: Uniform(0, P_max)
        # This is uninformative within physiological range
        P_free = pm.Uniform('P_free', lower=0, upper=P_prior_upper, shape=n_free)

        # Forward model: predicted flows
        Q_pred = pm.math.dot(A_scaled, P_free)

        # Noise model
        if infer_sigma:
            # Infer a global noise scale multiplier
            # Prior: HalfNormal(10) allows scale factors from ~1 to ~30
            # This multiplies the input sigma_Q values
            sigma_scale = pm.HalfNormal('sigma_scale', sigma=10.0)
            effective_sigma = sigma_scale * sigma_Q
        else:
            effective_sigma = sigma_Q

        # Likelihood
        if likelihood == 'gaussian':
            pm.Normal('Q_obs', mu=Q_pred, sigma=effective_sigma, observed=Q_obs)
        elif likelihood == 'studentt':
            pm.StudentT('Q_obs', nu=nu, mu=Q_pred, sigma=effective_sigma, observed=Q_obs)
        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")

        # Initial values (warm start from IRLS if available)
        init_vals = None
        if init_P is not None:
            init_vals = {'P_free': init_P}
            if infer_sigma:
                # Start sigma_scale at a reasonable value
                init_vals['sigma_scale'] = 5.0

        # Sample
        if verbose:
            print("  Sampling...")

        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            cores=min(n_chains, 4),
            initvals=init_vals,
            return_inferencedata=True,
            progressbar=verbose,
        )

    # Extract samples
    P_samples = trace.posterior['P_free'].values
    # Reshape from (chains, draws, n_free) to (n_samples, n_free)
    P_samples = P_samples.reshape(-1, n_free)

    # Extract inferred sigma_scale if applicable
    inferred_sigma_scale = None
    if infer_sigma:
        sigma_samples = trace.posterior['sigma_scale'].values.flatten()
        inferred_sigma_scale = float(np.mean(sigma_samples))
        if verbose:
            print(f"  Inferred sigma_scale: {inferred_sigma_scale:.2f} "
                  f"(95% CI: [{np.percentile(sigma_samples, 2.5):.2f}, "
                  f"{np.percentile(sigma_samples, 97.5):.2f}])")

    return trace, P_samples, inferred_sigma_scale


def check_convergence(trace, verbose: bool = True) -> Tuple[bool, float, float, int]:
    """
    Check MCMC convergence diagnostics.

    Returns:
        converged: True if all diagnostics pass
        r_hat_max: Maximum R-hat across parameters
        ess_min: Minimum effective sample size
        n_divergences: Number of divergent transitions
    """
    if not PYMC_AVAILABLE:
        return True, 1.0, 1000, 0

    # Compute summary statistics
    summary = az.summary(trace, var_names=['P_free'])

    r_hat_values = summary['r_hat'].values
    ess_values = summary['ess_bulk'].values

    r_hat_max = float(np.nanmax(r_hat_values))
    ess_min = float(np.nanmin(ess_values))

    # Count divergences
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        n_divergences = int(trace.sample_stats['diverging'].sum())
    else:
        n_divergences = 0

    # Check thresholds
    converged = (r_hat_max < 1.01) and (ess_min > 400) and (n_divergences < 10)

    if verbose:
        status = "✓ CONVERGED" if converged else "⚠ WARNING"
        print(f"  Diagnostics: {status}")
        print(f"    R-hat max: {r_hat_max:.3f} (should be < 1.01)")
        print(f"    ESS min: {ess_min:.0f} (should be > 400)")
        print(f"    Divergences: {n_divergences} (should be < 10)")

    return converged, r_hat_max, ess_min, n_divergences


def summarize_posterior(
    P_samples: np.ndarray,
    A: np.ndarray,
    Q_obs: np.ndarray,
    sigma_Q: np.ndarray,
    mu_Pa_s: float,
    free_boundary: List[int],
    ref_node: int,
    all_boundary: List[int],
    internal_nodes: List[int],
    M: np.ndarray,
) -> dict:
    """
    Compute posterior summaries for pressures and predicted flows.

    Returns dict with:
        P_mean, P_std, P_ci95: Pressure summaries
        Q_pred_mean, Q_pred_std, Q_pred_ci95: Flow prediction summaries
        r_squared, rmse, chi2_reduced: Fit metrics
    """
    n_samples = P_samples.shape[0]
    n_free = len(free_boundary)

    # Build full pressure array (including ref_node = 0)
    boundary_idx = {n: i for i, n in enumerate(free_boundary)}

    # Pressure summaries for boundary nodes
    P_mean = {ref_node: 0.0}
    P_std = {ref_node: 0.0}
    P_ci95 = {ref_node: (0.0, 0.0)}

    for i, node in enumerate(free_boundary):
        samples = P_samples[:, i]
        P_mean[node] = float(np.mean(samples))
        P_std[node] = float(np.std(samples))
        P_ci95[node] = (float(np.percentile(samples, 2.5)),
                        float(np.percentile(samples, 97.5)))

    # Internal node pressures (derived from boundary)
    if len(internal_nodes) > 0 and M.shape[0] > 0:
        # P_internal = M @ P_boundary for each sample
        P_internal_samples = P_samples @ M.T  # (n_samples, n_internal)
        for i, node in enumerate(internal_nodes):
            samples = P_internal_samples[:, i]
            P_mean[node] = float(np.mean(samples))
            P_std[node] = float(np.std(samples))
            P_ci95[node] = (float(np.percentile(samples, 2.5)),
                           float(np.percentile(samples, 97.5)))

    # Predicted flows: Q = A @ P / mu
    A_scaled = A / mu_Pa_s
    Q_pred_samples = P_samples @ A_scaled.T  # (n_samples, n_edges)

    Q_pred_mean = np.mean(Q_pred_samples, axis=0)
    Q_pred_std = np.std(Q_pred_samples, axis=0)
    Q_pred_ci95_lo = np.percentile(Q_pred_samples, 2.5, axis=0)
    Q_pred_ci95_hi = np.percentile(Q_pred_samples, 97.5, axis=0)

    # Fit metrics (using posterior mean)
    residuals = Q_obs - Q_pred_mean
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Q_obs - np.mean(Q_obs))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2)))

    chi2 = np.sum((residuals / sigma_Q)**2)
    dof = len(Q_obs) - n_free
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    return {
        'P_mean': P_mean,
        'P_std': P_std,
        'P_ci95': P_ci95,
        'Q_pred_mean': Q_pred_mean,
        'Q_pred_std': Q_pred_std,
        'Q_pred_ci95_lo': Q_pred_ci95_lo,
        'Q_pred_ci95_hi': Q_pred_ci95_hi,
        'r_squared': r_squared,
        'rmse': rmse,
        'chi2_reduced': chi2_reduced,
    }


def compute_all_edge_predictions(
    G: nx.Graph,
    P_samples: np.ndarray,
    free_boundary: List[int],
    ref_node: int,
    internal_nodes: List[int],
    M: np.ndarray,
    mu_Pa_s: float,
    measured_edges: Set[Tuple[int, int]],
    radius_attr: str = 'radius_um',
    length_attr: str = 'length_um',
    verbose: bool = True,
) -> Dict[Tuple[int, int], dict]:
    """
    Compute flow predictions for ALL edges in the graph (measured and unmeasured).

    For each edge, propagates pressure posterior samples through the forward model
    to get flow predictions with full uncertainty quantification.

    Args:
        G: NetworkX graph with geometry attributes
        P_samples: (n_samples, n_free) pressure samples from MCMC
        free_boundary: List of free boundary nodes (excluding ref_node)
        ref_node: Reference node (P=0)
        internal_nodes: List of internal (non-boundary) nodes
        M: Matrix mapping boundary pressures to internal pressures
        mu_Pa_s: Viscosity in Pa·s
        measured_edges: Set of (u, v) tuples that have measurements
        radius_attr: Edge attribute for radius (um)
        length_attr: Edge attribute for length (um)
        verbose: Print progress

    Returns:
        Dict mapping (u, v) -> {
            'Q_mean': float,           # Posterior mean flow (nL/s)
            'Q_std': float,            # Posterior std
            'Q_ci95_lo': float,        # 2.5th percentile
            'Q_ci95_hi': float,        # 97.5th percentile
            'direction_confidence': float,  # Fraction of samples agreeing on sign
            'is_measured': bool,       # Whether this edge had a measurement
        }
    """
    n_samples = P_samples.shape[0]

    # Build index mappings
    boundary_idx = {n: i for i, n in enumerate(free_boundary)}
    internal_idx = {n: i for i, n in enumerate(internal_nodes)}

    # Compute internal node pressures: P_internal = M @ P_boundary
    if len(internal_nodes) > 0 and M.shape[0] > 0:
        P_internal_samples = P_samples @ M.T  # (n_samples, n_internal)
    else:
        P_internal_samples = np.zeros((n_samples, 0))

    results = {}
    n_unmeasured = 0

    for u, v, data in G.edges(data=True):
        edge_key = (u, v)

        # Get geometry
        R_um = data.get(radius_attr)
        L_um = data.get(length_attr)

        if R_um is None or L_um is None or L_um <= 0:
            # Can't compute prediction without geometry
            continue

        # Compute conductance: g0 = π R^4 / (8 L) at μ=1 Pa·s
        # R in um, L in um -> g0 in um^3/Pa·s
        # Convert to nL/s per Pa: 1 um^3 = 1e-6 nL
        R_um = float(R_um)
        L_um = float(L_um)
        g0 = np.pi * R_um**4 / (8 * L_um) * 1e-6  # nL/s per Pa at μ=1 Pa·s

        # Get pressure samples for each endpoint
        P_u_samples = np.zeros(n_samples)
        P_v_samples = np.zeros(n_samples)

        for node, P_arr in [(u, P_u_samples), (v, P_v_samples)]:
            if node == ref_node:
                P_arr[:] = 0.0
            elif node in boundary_idx:
                P_arr[:] = P_samples[:, boundary_idx[node]]
            elif node in internal_idx:
                P_arr[:] = P_internal_samples[:, internal_idx[node]]
            else:
                # Node not in model - skip this edge
                break
        else:
            # Compute flow samples: Q = g0/μ × (P_u - P_v)
            Q_samples = (g0 / mu_Pa_s) * (P_u_samples - P_v_samples)

            # Compute summaries
            Q_mean = float(np.mean(Q_samples))
            Q_std = float(np.std(Q_samples))
            Q_ci95_lo = float(np.percentile(Q_samples, 2.5))
            Q_ci95_hi = float(np.percentile(Q_samples, 97.5))

            # Direction confidence: fraction of samples agreeing on sign
            n_positive = np.sum(Q_samples > 0)
            n_negative = np.sum(Q_samples < 0)
            direction_confidence = float(max(n_positive, n_negative) / n_samples)

            is_measured = edge_key in measured_edges or (v, u) in measured_edges

            results[edge_key] = {
                'Q_mean': Q_mean,
                'Q_std': Q_std,
                'Q_ci95_lo': Q_ci95_lo,
                'Q_ci95_hi': Q_ci95_hi,
                'direction_confidence': direction_confidence,
                'is_measured': is_measured,
            }

            if not is_measured:
                n_unmeasured += 1

    if verbose:
        n_total = len(results)
        n_measured = n_total - n_unmeasured
        print(f"  Flow predictions: {n_total} edges ({n_measured} measured, {n_unmeasured} unmeasured)")

        # Summarize direction confidence for unmeasured
        if n_unmeasured > 0:
            unmeasured_conf = [r['direction_confidence'] for r in results.values() if not r['is_measured']]
            n_high_conf = sum(1 for c in unmeasured_conf if c >= 0.95)
            n_med_conf = sum(1 for c in unmeasured_conf if 0.75 <= c < 0.95)
            n_low_conf = sum(1 for c in unmeasured_conf if c < 0.75)
            print(f"  Unmeasured direction confidence: "
                  f"{n_high_conf} high (≥95%), {n_med_conf} medium (75-95%), {n_low_conf} low (<75%)")

    return results


def bayesian_pressure_mcmc(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: float = 3.5,
    likelihood: str = 'studentt',
    nu: float = 4.0,
    radius_attr: str = 'radius_um',
    length_attr: str = 'length_um',
    flow_attr: str = 'mean_Q',  # Use mean_Q (handles both mean_Q and mean_Q_nL_s)
    sigma_attr: str = 'sigma_Q',  # Use sigma_Q (handles both conventions)
    max_chi2_reduced: Optional[float] = 5.0,
    min_snr_percentile: float = 50.0,
    min_vessel_length_px: float = 20.0,
    max_Q_nL_s: Optional[float] = None,
    infer_sigma: bool = True,
    heteroscedastic: bool = False,
    n_tune: int = 1000,
    n_draws: int = 2000,
    n_chains: int = 4,
    verbose: bool = True,
) -> MCMCResult:
    """
    Bayesian pressure inference using MCMC sampling.

    This is the main entry point. Uses the existing data preparation
    from flow_consistency.py, then runs PyMC MCMC sampling.

    Parameters
    ----------
    G : nx.Graph
        Vessel network with geometry and flow attributes
    boundary_nodes : set, optional
        Boundary nodes (auto-detect if None)
    mu_cP : float
        Viscosity in cP (default 3.5)
    likelihood : str
        'gaussian' for standard likelihood, 'studentt' for robust
    nu : float
        Degrees of freedom for Student-t (default 4, lower = more robust)
    radius_attr, length_attr, flow_attr, sigma_attr : str
        Edge attribute names
    max_chi2_reduced : float, optional
        Filter threshold for profile fit quality
    min_snr_percentile : float
        SNR percentile threshold (default 50 = median)
    min_vessel_length_px : float
        Minimum vessel length in pixels
    max_Q_nL_s : float, optional
        Maximum |Q| in nL/s for outlier filtering. Default None (no filter).
    infer_sigma : bool
        If True (default), infer a global noise scale multiplier from the data.
        This corrects for systematically under/over-estimated uncertainties.
        The inferred scale is stored in result.sigma_scale.
    heteroscedastic : bool
        If True, use heteroscedastic noise model: σ_i = σ_base + σ_rel × |Q_i|
        This models flow-dependent uncertainty where larger flows have larger errors.
        Overrides infer_sigma. Results stored in result.sigma_base and result.sigma_rel.
    n_tune, n_draws, n_chains : int
        MCMC sampling parameters
    verbose : bool
        Print progress

    Returns
    -------
    MCMCResult
    """
    if not MCMC_AVAILABLE:
        raise ImportError(
            "No MCMC backend available.\n"
            "Install NumPyro (recommended): pip install numpyro jax jaxlib\n"
            "Or install PyMC: pip install pymc arviz\n"
            "Falling back to Laplace approximation is possible via "
            "bayesian_poiseuille_simulation() instead."
        )

    # Determine which backend to use
    use_numpyro = NUMPYRO_AVAILABLE
    backend_name = "NumPyro" if use_numpyro else "PyMC"

    # Import data preparation from flow_consistency
    from .flow_consistency import _prepare_bayesian_data

    if verbose:
        print("\n" + "=" * 70)
        print(f"BAYESIAN PRESSURE INFERENCE ({backend_name} MCMC)")
        print("=" * 70)

    # Prepare data (same as Laplace version)
    data = _prepare_bayesian_data(
        G, boundary_nodes, radius_attr, length_attr, flow_attr, sigma_attr,
        max_chi2_reduced, min_snr_percentile, min_vessel_length_px,
        max_Q_nL_s=max_Q_nL_s,
        default_sigma_fraction=0.5,
        infer_sigma=False,
        verbose=verbose,
    )

    # Extract what we need
    measured_edges = data['measured_edges']
    Q_meas = data['Q_meas']
    sigma_Q = data['sigma_Q']
    chi2_meas = data['chi2_meas']  # chi2_reduced for each measured vessel
    boundary_list = data['boundary_list']
    internal_nodes = data['internal_nodes']
    ref_node = data['ref_node']
    free_boundary = data['free_boundary']
    M = data['M']
    n_meas = data['n_meas']

    if n_meas < 2:
        raise ValueError(f"Need at least 2 measurements, got {n_meas}")

    # Build design matrix
    A = build_design_matrix(
        G, measured_edges, boundary_list, internal_nodes, ref_node, M
    )

    if verbose:
        print(f"\n  Design matrix A: {A.shape}")
        # Diagnostic: check for problems
        g0_values = [e['g0'] for e in measured_edges]
        print(f"  Conductance g0 range: [{min(g0_values):.2e}, {max(g0_values):.2e}] nL/s/Pa")
        print(f"  A matrix range: [{A.min():.2e}, {A.max():.2e}]")
        print(f"  A condition number: {np.linalg.cond(A):.2e}")
        # Check for zero columns (disconnected nodes)
        zero_cols = np.sum(np.abs(A).sum(axis=0) < 1e-10)
        if zero_cols > 0:
            print(f"  ⚠ WARNING: {zero_cols} boundary nodes have no measured vessels!")
        # Check for extreme rows
        row_norms = np.abs(A).sum(axis=1)
        if row_norms.max() / row_norms.min() > 1e6:
            print(f"  ⚠ WARNING: Huge variation in row magnitudes (ratio {row_norms.max()/row_norms.min():.1e})")

    # Check for underdetermined system
    n_free = len(free_boundary)
    if n_meas < n_free:
        if verbose:
            print(f"\n  ⚠ WARNING: Underdetermined system!")
            print(f"    {n_meas} measurements < {n_free} parameters")
            print(f"    Solution will be weakly constrained. Consider:")
            print(f"    - Lowering SNR filter (min_snr_percentile)")
            print(f"    - Including more vessels")
            print(f"    - Using stronger priors")

    # Run MCMC with appropriate backend
    mu_Pa_s = mu_cP * 1e-3

    # Compute IRLS solution and covariance for adaptive priors
    # This gives us per-node prior std based on how well each node is constrained
    if verbose:
        print("\n  Computing IRLS solution for adaptive priors...")

    # Adaptive ridge regularization based on condition number
    cond_A = np.linalg.cond(A)
    if cond_A > 1e10:
        # Severely ill-conditioned: use strong regularization
        lambda_ridge = 1e-3
        if verbose:
            print(f"  ⚠ Matrix severely ill-conditioned (cond={cond_A:.1e})")
            print(f"    Using strong ridge regularization (λ={lambda_ridge:.1e})")
    elif cond_A > 1e6:
        # Moderately ill-conditioned
        lambda_ridge = 1e-5
    else:
        # Well-conditioned
        lambda_ridge = 1e-8

    P_irls, P_irls_std, Cov_irls = compute_irls_covariance(
        A, Q_meas, sigma_Q, mu_Pa_s=mu_Pa_s, nu=nu,
        lambda_ridge=lambda_ridge, verbose=verbose
    )

    # Use IRLS solution as prior mean, IRLS std as per-node prior std
    init_P = P_irls
    P_prior_std_arr = P_irls_std

    # Store prior std and mean for each node (for shrinkage computation and diagnostics)
    prior_std_dict = {ref_node: 0.0}
    prior_mean_dict = {ref_node: 0.0}
    for i, node in enumerate(free_boundary):
        prior_std_dict[node] = float(P_prior_std_arr[i])
        prior_mean_dict[node] = float(P_irls[i])

    if verbose:
        # Summarize prior informativeness
        n_tight = np.sum(P_prior_std_arr < 100)
        n_loose = np.sum(P_prior_std_arr > 1000)
        n_medium = n_free - n_tight - n_loose
        print(f"  Prior informativeness: {n_tight} tight (<100 Pa), "
              f"{n_medium} medium, {n_loose} loose (>1000 Pa)")

    # Initialize sigma parameters
    inferred_sigma_scale = None
    inferred_sigma_base = None
    inferred_sigma_rel = None
    inferred_chi2_alpha = None

    if use_numpyro:
        # Use NumPyro backend with adaptive per-node priors
        mcmc_obj, P_samples, inferred_sigma_scale, inferred_sigma_base, inferred_sigma_rel, inferred_chi2_alpha = run_mcmc_inference_numpyro(
            A, Q_meas, sigma_Q,
            mu_Pa_s=mu_Pa_s,
            likelihood=likelihood,
            nu=nu,
            P_prior_std=P_prior_std_arr,  # Per-node prior from IRLS covariance
            n_warmup=n_tune,
            n_samples=n_draws,
            n_chains=n_chains,
            init_P=init_P,
            infer_sigma=infer_sigma,
            heteroscedastic=heteroscedastic,
            chi2_reduced=None,  # Disabled: chi2 inflation handled separately
            verbose=verbose,
        )
        # Check convergence
        converged, r_hat_max, ess_min, n_divergences = check_convergence_numpyro(mcmc_obj, verbose)
        trace = mcmc_obj  # Store MCMC object as trace for compatibility
    else:
        # Use PyMC backend (doesn't support heteroscedastic yet, falls back to uniform sigma_scale)
        if heteroscedastic:
            print("  ⚠ WARNING: Heteroscedastic model not supported in PyMC backend, using uniform sigma_scale")
        trace, P_samples, inferred_sigma_scale = run_mcmc_inference(
            A, Q_meas, sigma_Q,
            mu_Pa_s=mu_Pa_s,
            likelihood=likelihood,
            nu=nu,
            n_tune=n_tune,
            n_draws=n_draws,
            n_chains=n_chains,
            init_P=init_P,
            infer_sigma=infer_sigma,
            verbose=verbose,
        )
        # Check convergence
        converged, r_hat_max, ess_min, n_divergences = check_convergence(trace, verbose)

    # Summarize posterior
    summary = summarize_posterior(
        P_samples, A, Q_meas, sigma_Q, mu_Pa_s,
        free_boundary, ref_node, boundary_list, internal_nodes, M
    )

    if verbose:
        print(f"\n  Results:")
        print(f"    R² = {summary['r_squared']:.4f}")
        print(f"    RMSE = {summary['rmse']:.4f} nL/s")
        print(f"    χ²_reduced = {summary['chi2_reduced']:.2f}")

    # Build result
    # Map predictions back to edge tuples (for measured edges only - legacy)
    predicted_Q = {}
    measured_Q_dict = {}
    Q_pred_std = {}
    Q_pred_ci95 = {}

    for idx, e in enumerate(measured_edges):
        edge_key = (e['u'], e['v'])
        predicted_Q[edge_key] = float(summary['Q_pred_mean'][idx])
        measured_Q_dict[edge_key] = e['Q_measured']
        Q_pred_std[edge_key] = float(summary['Q_pred_std'][idx])
        Q_pred_ci95[edge_key] = (
            float(summary['Q_pred_ci95_lo'][idx]),
            float(summary['Q_pred_ci95_hi'][idx])
        )

    # Compute predictions for ALL edges (measured and unmeasured)
    measured_edge_set = {(e['u'], e['v']) for e in measured_edges}
    all_edge_predictions = compute_all_edge_predictions(
        G, P_samples, free_boundary, ref_node, internal_nodes, M, mu_Pa_s,
        measured_edge_set, radius_attr, length_attr, verbose=verbose,
    )

    # Compute shrinkage diagnostics
    # For uniform sigma_scale: shrinkage = 1 - (posterior_std / (prior_std * sigma_scale))^2
    # For heteroscedastic: use sigma_scale=1 (shrinkage compares posterior to prior directly)
    # shrinkage ≈ 0: weakly constrained by network topology
    # shrinkage ≈ 1: strongly constrained by network topology
    effective_scale = inferred_sigma_scale if not heteroscedastic else None
    shrinkage = compute_shrinkage_diagnostics(
        summary['P_std'], prior_std_dict, free_boundary,
        sigma_scale=effective_scale
    )

    if verbose:
        # Report shrinkage statistics
        # Shrinkage measures how much the network topology constrains each node
        # relative to what we'd expect from the scaled measurement uncertainties
        shrinkage_vals = [shrinkage[n] for n in free_boundary]
        n_weakly_constrained = sum(1 for s in shrinkage_vals if s < 0.2)
        n_strongly_constrained = sum(1 for s in shrinkage_vals if s > 0.8)
        n_mixed = len(shrinkage_vals) - n_weakly_constrained - n_strongly_constrained

        print(f"\n  Shrinkage diagnostics (network constraint on pressures):")
        if heteroscedastic:
            print(f"    (heteroscedastic model: σ = {inferred_sigma_base:.2f} + {inferred_sigma_rel:.3f} × |Q|)")
        else:
            scale = inferred_sigma_scale if inferred_sigma_scale is not None else 1.0
            print(f"    (comparing posterior_std to prior_std × σ_scale = prior_std × {scale:.2f})")
        print(f"    Strongly constrained (shrinkage > 0.8): {n_strongly_constrained} nodes")
        print(f"    Moderately constrained (0.2-0.8): {n_mixed} nodes")
        print(f"    Weakly constrained (shrinkage < 0.2): {n_weakly_constrained} nodes")

        if n_weakly_constrained > 0 and n_weakly_constrained < len(free_boundary):
            print(f"    ⚠ {n_weakly_constrained} boundary nodes are weakly constrained by network")
            # List the worst ones
            worst_nodes = sorted(free_boundary, key=lambda n: shrinkage[n])[:5]
            for node in worst_nodes:
                if shrinkage[node] < 0.2:
                    if heteroscedastic:
                        print(f"      Node {node}: shrinkage = {shrinkage[node]:.2f}, "
                              f"prior_std = {prior_std_dict[node]:.0f} Pa, "
                              f"post_std = {summary['P_std'][node]:.0f} Pa")
                    else:
                        scale = inferred_sigma_scale if inferred_sigma_scale is not None else 1.0
                        eff_prior = prior_std_dict[node] * scale
                        print(f"      Node {node}: shrinkage = {shrinkage[node]:.2f}, "
                              f"eff_prior_std = {eff_prior:.0f} Pa, "
                              f"post_std = {summary['P_std'][node]:.0f} Pa")

    # Compute edge data influence scores
    edge_influence = compute_edge_data_influence(
        G, shrinkage, free_boundary, ref_node, internal_nodes, M
    )

    if verbose:
        # Report edge influence statistics
        influence_vals = list(edge_influence.values())
        n_reliable = sum(1 for v in influence_vals if v > 0.5)
        n_mixed_edges = sum(1 for v in influence_vals if 0.2 <= v <= 0.5)
        n_unreliable = sum(1 for v in influence_vals if v < 0.2)

        print(f"\n  Edge data influence (reliability of predicted flows):")
        print(f"    Reliable (influence > 0.5): {n_reliable} edges")
        print(f"    Mixed (0.2 < influence < 0.5): {n_mixed_edges} edges")
        print(f"    Prior-driven (influence < 0.2): {n_unreliable} edges")

    if verbose:
        print("=" * 70)

    return MCMCResult(
        mu_cP=mu_cP,
        node_pressures=summary['P_mean'],
        predicted_Q=predicted_Q,
        measured_Q=measured_Q_dict,
        P_std=summary['P_std'],
        P_ci95=summary['P_ci95'],
        Q_pred_std=Q_pred_std,
        Q_pred_ci95=Q_pred_ci95,
        r_squared=summary['r_squared'],
        rmse=summary['rmse'],
        chi2_reduced=summary['chi2_reduced'],
        converged=converged,
        r_hat_max=r_hat_max,
        ess_min=ess_min,
        n_divergences=n_divergences,
        boundary_nodes=data['boundary_nodes'],
        ref_node=ref_node,
        n_parameters=len(free_boundary),
        n_measurements=n_meas,
        sigma_scale=inferred_sigma_scale,
        sigma_base=inferred_sigma_base,
        sigma_rel=inferred_sigma_rel,
        chi2_alpha=inferred_chi2_alpha,
        trace=trace,
        all_edge_predictions=all_edge_predictions,
        P_shrinkage=shrinkage,
        P_prior_std=prior_std_dict,
        P_prior_mean=prior_mean_dict,
        edge_data_influence=edge_influence,
    )


@dataclass
class WLSResult:
    """Results from Weighted Least Squares pressure inference."""
    # Point estimates
    mu_cP: float
    node_pressures: Dict[int, float]
    predicted_Q: Dict[Tuple[int, int], float]
    measured_Q: Dict[Tuple[int, int], float]

    # Uncertainties (χ²-corrected)
    P_std: Dict[int, float]
    Q_pred_std: Dict[Tuple[int, int], float]

    # Fit quality
    r_squared: float
    rmse: float
    chi2: float
    chi2_reduced: float
    sigma_scale: float  # √χ²_red correction factor

    # Model info
    boundary_nodes: Set[int]
    ref_node: int
    n_parameters: int
    n_measurements: int
    dof: int


def wls_pressure_inference(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: float = 3.5,
    use_studentt_weights: bool = True,
    nu: float = 4.0,
    radius_attr: str = 'radius_um',
    length_attr: str = 'length_um',
    flow_attr: str = 'mean_Q',
    sigma_attr: str = 'sigma_Q',
    max_chi2_reduced: Optional[float] = 5.0,
    min_snr_percentile: float = 50.0,
    min_vessel_length_px: float = 20.0,
    max_Q_nL_s: Optional[float] = None,
    verbose: bool = True,
) -> WLSResult:
    """
    Pressure inference using Weighted Least Squares with χ² correction.

    This is a simpler alternative to full MCMC that gives equivalent point estimates.
    Uncertainties are calibrated using reduced chi-squared:

        σ_corrected = σ × √χ²_red  (if χ²_red > 1)

    Student-t reweighting (IRLS) provides outlier robustness.

    Parameters
    ----------
    G : nx.Graph
        Vessel network with geometry and flow attributes
    boundary_nodes : set, optional
        Boundary nodes (auto-detect if None)
    mu_cP : float
        Viscosity in cP (default 3.5)
    use_studentt_weights : bool
        If True, use IRLS with Student-t weights for outlier robustness
    nu : float
        Degrees of freedom for Student-t (default 4, lower = more robust)
    radius_attr, length_attr, flow_attr, sigma_attr : str
        Edge attribute names
    max_chi2_reduced : float, optional
        Filter threshold for profile fit quality
    min_snr_percentile : float
        SNR percentile threshold (default 50 = median)
    min_vessel_length_px : float
        Minimum vessel length in pixels
    max_Q_nL_s : float, optional
        Maximum |Q| for outlier filtering
    verbose : bool
        Print progress

    Returns
    -------
    WLSResult
    """
    from .flow_consistency import _prepare_bayesian_data

    if verbose:
        print("\n" + "=" * 70)
        print("WEIGHTED LEAST SQUARES PRESSURE INFERENCE")
        print("=" * 70)

    # Prepare data (same as MCMC version)
    data = _prepare_bayesian_data(
        G, boundary_nodes, radius_attr, length_attr, flow_attr, sigma_attr,
        max_chi2_reduced, min_snr_percentile, min_vessel_length_px,
        max_Q_nL_s=max_Q_nL_s,
        default_sigma_fraction=0.5,
        infer_sigma=False,
        verbose=verbose,
    )

    # Extract what we need
    measured_edges = data['measured_edges']
    Q_meas = data['Q_meas']
    sigma_Q = data['sigma_Q']
    boundary_list = data['boundary_list']
    internal_nodes = data['internal_nodes']
    ref_node = data['ref_node']
    free_boundary = data['free_boundary']
    M = data['M']
    n_meas = data['n_meas']

    if n_meas < 2:
        raise ValueError(f"Need at least 2 measurements, got {n_meas}")

    # Build design matrix
    A = build_design_matrix(
        G, measured_edges, boundary_list, internal_nodes, ref_node, M
    )

    if verbose:
        print(f"\n  Design matrix A: {A.shape}")
        n_free = len(free_boundary)
        print(f"  Measurements: {n_meas}, Parameters: {n_free}, DoF: {n_meas - n_free}")

    # Solve using WLS with chi2 correction
    mu_Pa_s = mu_cP * 1e-3
    wls_result = solve_wls_with_chi2_correction(
        A, Q_meas, sigma_Q, mu_Pa_s,
        use_studentt_weights=use_studentt_weights,
        nu=nu,
        verbose=verbose,
    )

    # Build pressure dict for all nodes
    node_pressures = {ref_node: 0.0}
    P_std = {ref_node: 0.0}

    for i, node in enumerate(free_boundary):
        node_pressures[node] = float(wls_result['P'][i])
        P_std[node] = float(wls_result['P_std_corrected'][i])

    # Internal node pressures (derived from boundary)
    if len(internal_nodes) > 0 and M.shape[0] > 0:
        P_internal = M @ wls_result['P']
        # Propagate uncertainty: Var(P_int) = M @ Cov @ M.T
        P_int_var = np.sum((M @ wls_result['Cov_corrected']) * M, axis=1)
        P_int_std = np.sqrt(np.maximum(P_int_var, 0))
        for i, node in enumerate(internal_nodes):
            node_pressures[node] = float(P_internal[i])
            P_std[node] = float(P_int_std[i])

    # Build flow dicts
    predicted_Q = {}
    measured_Q_dict = {}
    Q_pred_std = {}

    for idx, e in enumerate(measured_edges):
        edge_key = (e['u'], e['v'])
        predicted_Q[edge_key] = float(wls_result['Q_pred'][idx])
        measured_Q_dict[edge_key] = e['Q_measured']
        Q_pred_std[edge_key] = float(wls_result['Q_pred_std'][idx])

    if verbose:
        print("\n" + "=" * 70)
        print("WLS INFERENCE COMPLETE")
        print(f"  χ²_red = {wls_result['chi2_reduced']:.3f}")
        if wls_result['chi2_reduced'] > 1:
            print(f"  Uncertainty correction: ×{wls_result['sigma_scale']:.3f}")
        print(f"  R² = {wls_result['r_squared']:.4f}")
        print("=" * 70)

    return WLSResult(
        mu_cP=mu_cP,
        node_pressures=node_pressures,
        predicted_Q=predicted_Q,
        measured_Q=measured_Q_dict,
        P_std=P_std,
        Q_pred_std=Q_pred_std,
        r_squared=wls_result['r_squared'],
        rmse=wls_result['rmse'],
        chi2=wls_result['chi2'],
        chi2_reduced=wls_result['chi2_reduced'],
        sigma_scale=wls_result['sigma_scale'],
        boundary_nodes=data['boundary_nodes'],
        ref_node=ref_node,
        n_parameters=len(free_boundary),
        n_measurements=n_meas,
        dof=wls_result['dof'],
    )


def verify_mcmc_reproducibility(
    G: nx.Graph,
    n_runs: int = 2,
    **mcmc_kwargs,
) -> Tuple[List[MCMCResult], dict]:
    """
    Run MCMC multiple times with different seeds to verify reproducibility.

    This is a sanity check: if the posterior is well-sampled, independent runs
    should give nearly identical results. Large differences indicate insufficient
    samples or convergence issues.

    Parameters
    ----------
    G : nx.Graph
        Vessel network (will NOT be modified)
    n_runs : int
        Number of independent MCMC runs (default 2)
    **mcmc_kwargs
        Arguments passed to bayesian_pressure_mcmc()

    Returns
    -------
    results : list of MCMCResult
        Results from each run
    comparison : dict
        Comparison statistics:
        - 'max_P_diff': Maximum absolute difference in pressure means
        - 'max_P_rel_diff': Maximum relative difference (vs posterior std)
        - 'max_Q_diff': Maximum difference in predicted Q
        - 'r_squared_diff': Difference in R² between runs
        - 'reproducible': True if results are consistent
    """
    import copy

    print("\n" + "=" * 70)
    print("MCMC REPRODUCIBILITY CHECK")
    print("=" * 70)
    print(f"Running {n_runs} independent MCMC chains with different seeds...")

    results = []
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")

        # Make a copy of the graph so we don't accumulate results
        G_copy = copy.deepcopy(G)

        # Set different random seed for each run
        if NUMPYRO_AVAILABLE:
            import jax
            key = jax.random.PRNGKey(i * 12345)
            # NumPyro uses the key internally, we just ensure different initialization

        # Run MCMC
        result = bayesian_pressure_mcmc(G_copy, **mcmc_kwargs)
        results.append(result)

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON OF RUNS")
    print("=" * 70)

    r1, r2 = results[0], results[1]

    # Compare pressure posteriors
    common_nodes = set(r1.node_pressures.keys()) & set(r2.node_pressures.keys())

    P_diffs = []
    P_rel_diffs = []
    for node in common_nodes:
        p1, p2 = r1.node_pressures[node], r2.node_pressures[node]
        std1, std2 = r1.P_std.get(node, 1.0), r2.P_std.get(node, 1.0)
        avg_std = (std1 + std2) / 2

        diff = abs(p1 - p2)
        rel_diff = diff / avg_std if avg_std > 0 else 0

        P_diffs.append(diff)
        P_rel_diffs.append(rel_diff)

    max_P_diff = max(P_diffs) if P_diffs else 0
    max_P_rel_diff = max(P_rel_diffs) if P_rel_diffs else 0
    mean_P_rel_diff = np.mean(P_rel_diffs) if P_rel_diffs else 0

    # Compare predicted flows
    common_edges = set(r1.predicted_Q.keys()) & set(r2.predicted_Q.keys())
    Q_diffs = []
    for edge in common_edges:
        q1, q2 = r1.predicted_Q[edge], r2.predicted_Q[edge]
        Q_diffs.append(abs(q1 - q2))

    max_Q_diff = max(Q_diffs) if Q_diffs else 0
    mean_Q_diff = np.mean(Q_diffs) if Q_diffs else 0

    # Compare fit quality
    r2_diff = abs(r1.r_squared - r2.r_squared)

    # Determine if reproducible
    # Criterion: relative differences should be < 10% of posterior std
    reproducible = max_P_rel_diff < 0.1 and r2_diff < 0.01

    print(f"\nPressure comparison ({len(common_nodes)} nodes):")
    print(f"  Max |ΔP|: {max_P_diff:.3f} Pa")
    print(f"  Max |ΔP|/σ_P: {max_P_rel_diff:.3f} (should be < 0.1)")
    print(f"  Mean |ΔP|/σ_P: {mean_P_rel_diff:.3f}")

    print(f"\nFlow comparison ({len(common_edges)} edges):")
    print(f"  Max |ΔQ|: {max_Q_diff:.3f} nL/s")
    print(f"  Mean |ΔQ|: {mean_Q_diff:.3f} nL/s")

    print(f"\nFit quality:")
    print(f"  R² run 1: {r1.r_squared:.4f}")
    print(f"  R² run 2: {r2.r_squared:.4f}")
    print(f"  |ΔR²|: {r2_diff:.4f} (should be < 0.01)")

    print("\n" + "-" * 70)
    if reproducible:
        print("✓ RESULTS ARE REPRODUCIBLE")
        print("  Posterior is well-sampled. Current settings are sufficient.")
    else:
        print("⚠ RESULTS VARY BETWEEN RUNS")
        print("  Consider increasing n_draws (e.g., 5000-10000)")
        if max_P_rel_diff >= 0.1:
            print(f"  Pressure means differ by {max_P_rel_diff:.1%} of posterior std")
        if r2_diff >= 0.01:
            print(f"  R² differs by {r2_diff:.4f} between runs")
    print("-" * 70)

    comparison = {
        'max_P_diff': max_P_diff,
        'max_P_rel_diff': max_P_rel_diff,
        'mean_P_rel_diff': mean_P_rel_diff,
        'max_Q_diff': max_Q_diff,
        'mean_Q_diff': mean_Q_diff,
        'r_squared_diff': r2_diff,
        'reproducible': reproducible,
    }

    return results, comparison


def write_mcmc_to_graph(
    G: nx.Graph,
    result: MCMCResult,
    verbose: bool = True,
) -> int:
    """
    Write MCMC results to graph for ALL edges (measured and unmeasured).

    Writes:
    - Node: pressure_Pa, pressure_std, pressure_ci95_lo/hi
    - Edge: Q_predicted, Q_pred_std, Q_pred_ci95_lo/hi, direction_confidence,
            is_measured, z_score (for measured only)
    - Graph: mcmc_* metadata
    """
    # Write node pressures and shrinkage
    for node, pressure in result.node_pressures.items():
        if G.has_node(node):
            G.nodes[node]['pressure_Pa'] = pressure
            G.nodes[node]['pressure_std'] = result.P_std.get(node, 0.0)
            G.nodes[node]['is_boundary_sim'] = node in result.boundary_nodes

            if node in result.P_ci95:
                ci = result.P_ci95[node]
                G.nodes[node]['pressure_ci95_lo'] = ci[0]
                G.nodes[node]['pressure_ci95_hi'] = ci[1]

            # Write shrinkage diagnostic (how much data influenced this node's posterior)
            if result.P_shrinkage is not None and node in result.P_shrinkage:
                G.nodes[node]['pressure_shrinkage'] = result.P_shrinkage[node]

            # Write prior mean and std (for diagnostics)
            if result.P_prior_mean is not None and node in result.P_prior_mean:
                G.nodes[node]['pressure_prior_mean'] = result.P_prior_mean[node]
            if result.P_prior_std is not None and node in result.P_prior_std:
                G.nodes[node]['pressure_prior_std'] = result.P_prior_std[node]

    # Write edge predictions for ALL edges (using new all_edge_predictions)
    n_written = 0
    n_measured = 0
    n_unmeasured = 0

    if result.all_edge_predictions is not None:
        for (u, v), pred in result.all_edge_predictions.items():
            if G.has_edge(u, v):
                edge_key = (u, v)
                sign = 1.0
            elif G.has_edge(v, u):
                edge_key = (v, u)
                sign = -1.0
            else:
                continue

            # Write predictions
            G.edges[edge_key]['Q_predicted'] = sign * pred['Q_mean']
            G.edges[edge_key]['Q_pred_std'] = pred['Q_std']
            G.edges[edge_key]['direction_confidence'] = pred['direction_confidence']
            G.edges[edge_key]['is_measured'] = pred['is_measured']

            # CI (flip if edge direction reversed)
            if sign > 0:
                G.edges[edge_key]['Q_pred_ci95_lo'] = pred['Q_ci95_lo']
                G.edges[edge_key]['Q_pred_ci95_hi'] = pred['Q_ci95_hi']
            else:
                G.edges[edge_key]['Q_pred_ci95_lo'] = -pred['Q_ci95_hi']
                G.edges[edge_key]['Q_pred_ci95_hi'] = -pred['Q_ci95_lo']

            # Write data influence score
            if result.edge_data_influence is not None:
                # Check both (u,v) and (v,u) orderings
                infl = result.edge_data_influence.get((u, v))
                if infl is None:
                    infl = result.edge_data_influence.get((v, u))
                if infl is not None:
                    G.edges[edge_key]['data_influence'] = infl

            n_written += 1
            if pred['is_measured']:
                n_measured += 1
            else:
                n_unmeasured += 1

    # Add z-scores for measured edges (using measured_Q from result)
    for (u, v), Q_meas in result.measured_Q.items():
        if G.has_edge(u, v):
            edge_key = (u, v)
        elif G.has_edge(v, u):
            edge_key = (v, u)
            Q_meas = -Q_meas
        else:
            continue

        Q_pred = G.edges[edge_key].get('Q_predicted', 0.0)
        Q_std = G.edges[edge_key].get('Q_pred_std', 1.0)
        if Q_std > 0 and Q_meas is not None:
            z = (Q_meas - Q_pred) / Q_std
            G.edges[edge_key]['z_score'] = z

    # Store metadata
    G.graph['mcmc_mu_cP'] = result.mu_cP
    G.graph['mcmc_r_squared'] = result.r_squared
    G.graph['mcmc_rmse'] = result.rmse
    G.graph['mcmc_chi2_reduced'] = result.chi2_reduced
    G.graph['mcmc_converged'] = result.converged
    G.graph['mcmc_r_hat_max'] = result.r_hat_max
    G.graph['mcmc_ess_min'] = result.ess_min
    G.graph['mcmc_n_divergences'] = result.n_divergences
    if result.sigma_scale is not None:
        G.graph['mcmc_sigma_scale'] = result.sigma_scale
    # Heteroscedastic model parameters
    if result.sigma_base is not None:
        G.graph['mcmc_sigma_base'] = result.sigma_base
    if result.sigma_rel is not None:
        G.graph['mcmc_sigma_rel'] = result.sigma_rel

    # Write AC (pulsatile) results if this is a PulsatileMCMCResult
    n_ac_written = 0
    if isinstance(result, PulsatileMCMCResult):
        # Write AC node pressures (amplitude and phase)
        if result.P_amp is not None:
            for node, amp in result.P_amp.items():
                if G.has_node(node):
                    G.nodes[node]['P_amp_Pa'] = amp
                    if result.P_phase is not None:
                        G.nodes[node]['P_phase_rad'] = result.P_phase.get(node, 0.0)
                    if result.P_re is not None:
                        G.nodes[node]['P_re_Pa'] = result.P_re.get(node, 0.0)
                    if result.P_im is not None:
                        G.nodes[node]['P_im_Pa'] = result.P_im.get(node, 0.0)

        # Write AC edge predictions
        if result.predicted_Q_amp is not None:
            for (u, v), amp_pred in result.predicted_Q_amp.items():
                if G.has_edge(u, v):
                    edge_key = (u, v)
                elif G.has_edge(v, u):
                    edge_key = (v, u)
                else:
                    continue

                G.edges[edge_key]['amp_Q_predicted'] = amp_pred
                if result.predicted_Q_phase is not None:
                    G.edges[edge_key]['phase_predicted'] = result.predicted_Q_phase.get((u, v), 0.0)

                # Write measured values and residuals
                if result.measured_Q_amp is not None:
                    amp_meas = result.measured_Q_amp.get((u, v))
                    if amp_meas is not None:
                        G.edges[edge_key]['amp_Q_measured'] = amp_meas
                        G.edges[edge_key]['amp_Q_residual'] = amp_meas - amp_pred

                if result.measured_Q_phase is not None:
                    phase_meas = result.measured_Q_phase.get((u, v))
                    if phase_meas is not None:
                        G.edges[edge_key]['phase_measured'] = phase_meas
                        phase_pred = result.predicted_Q_phase.get((u, v), 0.0) if result.predicted_Q_phase else 0.0
                        # Phase residual (wrapped to [-π, π])
                        phase_res = phase_meas - phase_pred
                        phase_res = np.arctan2(np.sin(phase_res), np.cos(phase_res))
                        G.edges[edge_key]['phase_residual'] = phase_res

                n_ac_written += 1

        # Store AC metadata
        if result.r_squared_ac is not None:
            G.graph['mcmc_r_squared_ac'] = result.r_squared_ac
        if result.rmse_amp is not None:
            G.graph['mcmc_rmse_amp'] = result.rmse_amp

    if verbose:
        print(f"  Wrote MCMC results: {n_written} edges "
              f"({n_measured} measured, {n_unmeasured} unmeasured), "
              f"{len(result.node_pressures)} nodes")
        if n_ac_written > 0:
            print(f"  Wrote AC results: {n_ac_written} edges with amplitude/phase")
        if not result.converged:
            print(f"  ⚠ WARNING: MCMC may not have converged!")

    return n_written


def run_and_write_mcmc(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: float = 3.5,
    likelihood: str = 'studentt',
    flow_attr: str = 'mean_Q_nL_s',
    verbose: bool = True,
) -> MCMCResult:
    """
    Convenience function: run MCMC and write results to graph.

    This is the recommended entry point for the "y" analysis.
    """
    result = bayesian_pressure_mcmc(
        G,
        boundary_nodes=boundary_nodes,
        mu_cP=mu_cP,
        likelihood=likelihood,
        flow_attr=flow_attr,
        verbose=verbose,
    )

    write_mcmc_to_graph(G, result, verbose=verbose)

    return result


# ========== Diagnostic Plots ==========

def plot_mcmc_diagnostics(result: MCMCResult, show: bool = True):
    """
    Plot MCMC convergence diagnostics: trace plots and R-hat summary.

    Args:
        result: MCMCResult from bayesian_pressure_mcmc
        show: Whether to plt.show()

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt

    if result.trace is None:
        print("No trace data available for diagnostics")
        return None

    # Create figure with trace plots for first few parameters
    n_params = min(6, result.n_parameters)  # Show up to 6 parameters
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 2.5 * n_params))

    if n_params == 1:
        axes = axes.reshape(1, 2)

    # Get samples - handle both NumPyro MCMC and PyMC/ArviZ formats
    if NUMPYRO_AVAILABLE and hasattr(result.trace, 'get_samples'):
        # NumPyro MCMC object
        all_samples = result.trace.get_samples()
        P_samples = np.array(all_samples['P_free'])  # (total_samples, n_params)
        # Reshape to (n_chains, n_draws, n_params) for consistent plotting
        n_chains = 4  # Default assumption
        n_total = P_samples.shape[0]
        n_draws = n_total // n_chains
        samples = P_samples.reshape(n_chains, n_draws, -1)
    elif hasattr(result.trace, 'posterior'):
        # ArviZ InferenceData (from PyMC)
        samples = result.trace.posterior['P_free'].values  # (chains, draws, params)
    else:
        print("Unknown trace format for diagnostic plots")
        return None
    n_chains = samples.shape[0]

    # Get parameter names (boundary node IDs)
    boundary_list = list(result.boundary_nodes - {result.ref_node})

    for i in range(n_params):
        # Trace plot (left column)
        ax_trace = axes[i, 0]
        for c in range(n_chains):
            ax_trace.plot(samples[c, :, i], alpha=0.7, linewidth=0.5)
        ax_trace.set_ylabel(f'P[{boundary_list[i] if i < len(boundary_list) else i}]')
        if i == 0:
            ax_trace.set_title('Trace plots')
        if i == n_params - 1:
            ax_trace.set_xlabel('Iteration')

        # Density plot (right column)
        ax_dens = axes[i, 1]
        for c in range(n_chains):
            ax_dens.hist(samples[c, :, i], bins=50, alpha=0.5, density=True)
        ax_dens.set_ylabel('Density')
        if i == 0:
            ax_dens.set_title('Posterior distributions')
        if i == n_params - 1:
            ax_dens.set_xlabel('Pressure (Pa)')

    # Add convergence summary as text
    fig.suptitle(
        f'MCMC Diagnostics: R-hat={result.r_hat_max:.3f}, ESS={result.ess_min:.0f}, '
        f'Divergences={result.n_divergences}',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_mcmc_residuals(result: MCMCResult, show: bool = True):
    """
    Plot MCMC residual analysis: observed vs predicted, residual histogram.

    Args:
        result: MCMCResult from bayesian_pressure_mcmc
        show: Whether to plt.show()

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt

    # Extract observed and predicted
    edges = list(result.measured_Q.keys())
    Q_obs = np.array([result.measured_Q[e] for e in edges])
    Q_pred = np.array([result.predicted_Q[e] for e in edges])
    Q_std = np.array([result.Q_pred_std.get(e, 1.0) for e in edges])

    # Compute residuals and z-scores
    residuals = Q_obs - Q_pred
    z_scores = residuals / Q_std

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Observed vs Predicted scatter
    ax1 = axes[0, 0]
    ax1.errorbar(Q_pred, Q_obs, yerr=Q_std, fmt='o', alpha=0.5, markersize=4, capsize=2)
    lims = [min(Q_obs.min(), Q_pred.min()), max(Q_obs.max(), Q_pred.max())]
    ax1.plot(lims, lims, 'k--', linewidth=1, label='y=x')
    ax1.set_xlabel('Predicted Q (nL/s)')
    ax1.set_ylabel('Observed Q (nL/s)')
    ax1.set_title(f'Observed vs Predicted (R²={result.r_squared:.3f})')
    ax1.legend()
    ax1.set_aspect('equal')

    # 2. Residual histogram
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (Q_obs - Q_pred) (nL/s)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Residuals (RMSE={result.rmse:.3f} nL/s)')

    # Add normal fit for comparison
    mu_res, std_res = residuals.mean(), residuals.std()
    x_fit = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    y_fit = norm.pdf(x_fit, mu_res, std_res) * len(residuals) * (residuals.max() - residuals.min()) / 30
    ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Normal(μ={mu_res:.2f}, σ={std_res:.2f})')
    ax2.legend()

    # 3. Z-score histogram
    ax3 = axes[1, 0]
    ax3.hist(z_scores, bins=30, edgecolor='black', alpha=0.7, density=True)
    x_z = np.linspace(-4, 4, 100)
    ax3.plot(x_z, norm.pdf(x_z, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    ax3.set_xlabel('Z-score (residual/σ)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Z-scores (χ²_red={result.chi2_reduced:.2f})')
    ax3.legend()

    # 4. Residuals vs predicted (check for heteroscedasticity)
    ax4 = axes[1, 1]
    ax4.scatter(np.abs(Q_pred), np.abs(residuals), alpha=0.5, s=20)
    ax4.set_xlabel('|Predicted Q| (nL/s)')
    ax4.set_ylabel('|Residual| (nL/s)')
    ax4.set_title('Residual magnitude vs predicted (check heteroscedasticity)')

    # Add trend line
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(np.abs(Q_pred), np.abs(residuals))
    x_fit = np.linspace(0, np.abs(Q_pred).max(), 100)
    ax4.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2,
             label=f'Trend: r={r_value:.2f}')
    ax4.legend()

    plt.suptitle('MCMC Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_mcmc_pressures(result: MCMCResult, show: bool = True):
    """
    Plot boundary pressure posterior distributions.

    Args:
        result: MCMCResult from bayesian_pressure_mcmc
        show: Whether to plt.show()

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt

    # Get boundary node pressures and uncertainties
    boundary_nodes = list(result.boundary_nodes)
    pressures = [result.node_pressures.get(n, 0) for n in boundary_nodes]
    stds = [result.P_std.get(n, 0) for n in boundary_nodes]

    # Sort by pressure
    sorted_idx = np.argsort(pressures)[::-1]
    boundary_nodes = [boundary_nodes[i] for i in sorted_idx]
    pressures = [pressures[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Pressure bar chart with error bars
    ax1 = axes[0]
    y_pos = np.arange(len(boundary_nodes))
    ax1.barh(y_pos, pressures, xerr=stds, alpha=0.7, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([str(n) for n in boundary_nodes], fontsize=8)
    ax1.set_xlabel('Pressure (Pa)')
    ax1.set_ylabel('Boundary Node')
    ax1.set_title('Boundary Pressure Estimates (±1σ)')
    ax1.invert_yaxis()  # Highest pressure at top

    # Add mmHg scale on top
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim()[0]/133.3, ax1.get_xlim()[1]/133.3)
    ax1_top.set_xlabel('Pressure (mmHg)')

    # 2. Pressure uncertainty (CV = std/mean)
    ax2 = axes[1]
    cv = [s/p if p > 0 else 0 for p, s in zip(pressures, stds)]
    colors = ['green' if c < 0.1 else 'orange' if c < 0.3 else 'red' for c in cv]
    ax2.barh(y_pos, [c*100 for c in cv], color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([str(n) for n in boundary_nodes], fontsize=8)
    ax2.set_xlabel('Coefficient of Variation (%)')
    ax2.set_ylabel('Boundary Node')
    ax2.set_title('Pressure Uncertainty (CV = σ/μ)')
    ax2.axvline(10, color='green', linestyle='--', alpha=0.5, label='10% (good)')
    ax2.axvline(30, color='red', linestyle='--', alpha=0.5, label='30% (poor)')
    ax2.legend()
    ax2.invert_yaxis()

    plt.suptitle(f'Boundary Pressure Posteriors ({len(boundary_nodes)} nodes)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_mcmc_summary(result: MCMCResult, show: bool = True):
    """
    Plot summary panel with key MCMC results.

    Args:
        result: MCMCResult from bayesian_pressure_mcmc
        show: Whether to plt.show()

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Build summary text
    conv_status = "✓ CONVERGED" if result.converged else "⚠ NOT CONVERGED"
    conv_color = "green" if result.converged else "red"

    # Build noise model section
    sigma_section = ""
    if result.sigma_base is not None and result.sigma_rel is not None:
        # Heteroscedastic model
        sigma_section = f"""
INFERRED NOISE MODEL (heteroscedastic):
  σ_i = σ_base + σ_rel × |Q_i|
  σ_base:        {result.sigma_base:.3f} nL/s  (base uncertainty)
  σ_rel:         {result.sigma_rel:.3f}        (flow-proportional)
  Example: σ at |Q|=10 nL/s: {result.sigma_base + 10*result.sigma_rel:.2f} nL/s
           σ at |Q|=50 nL/s: {result.sigma_base + 50*result.sigma_rel:.2f} nL/s"""
    elif result.sigma_scale is not None:
        # Uniform scale model
        sigma_section = f"""
INFERRED NOISE SCALE (uniform):
  σ_scale:       {result.sigma_scale:.2f}×  (multiplier on input σ_Q)
  Interpretation: Input uncertainties were ~{result.sigma_scale:.1f}× too small"""

    summary = f"""
MCMC Bayesian Pressure Inference Summary
{'='*50}

CONVERGENCE: {conv_status}
  R-hat max:     {result.r_hat_max:.4f}  (should be < 1.01)
  ESS min:       {result.ess_min:.0f}  (should be > 400)
  Divergences:   {result.n_divergences}  (should be 0)

FIT QUALITY:
  R²:            {result.r_squared:.4f}
  RMSE:          {result.rmse:.4f} nL/s
  χ² reduced:    {result.chi2_reduced:.2f}
{sigma_section}

MODEL:
  Viscosity:     {result.mu_cP:.3f} cP (fixed)
  Parameters:    {result.n_parameters} boundary pressures
  Measurements:  {result.n_measurements} flow values
  Reference:     Node {result.ref_node} (P=0)

PRESSURE RANGE:
  Min:           {min(result.node_pressures.values()):.0f} Pa ({min(result.node_pressures.values())/133.3:.1f} mmHg)
  Max:           {max(result.node_pressures.values()):.0f} Pa ({max(result.node_pressures.values())/133.3:.1f} mmHg)
  Mean std:      {np.mean(list(result.P_std.values())):.0f} Pa

INTERPRETATION:
  • Heteroscedastic: σ depends on |Q|, larger flows have larger errors
  • σ_scale >> 1 (uniform): flow uncertainties were underestimated
  • χ²_red ≈ 1 indicates good fit with proper uncertainties
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Add convergence status with color
    ax.text(0.62, 0.92, conv_status, transform=ax.transAxes, fontsize=14,
            fontweight='bold', color=conv_color)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_all_mcmc(
    result: MCMCResult,
    G: Optional[nx.Graph] = None,
    background: Optional[np.ndarray] = None,
    show: bool = True,
):
    """
    Generate all MCMC diagnostic plots.

    This is the main entry point for visualizing MCMC results.

    Args:
        result: MCMCResult from bayesian_pressure_mcmc
        G: Optional graph for sigma comparison plot
        background: Optional video/image for boundary priors overlay
        show: Whether to plt.show()

    Returns:
        dict of figures
    """
    print("\nGenerating MCMC diagnostic plots...")

    figs = {}

    # 1. Summary
    figs['summary'] = plot_mcmc_summary(result, show=False)
    print("  - Summary panel")

    # 2. Residuals
    figs['residuals'] = plot_mcmc_residuals(result, show=False)
    print("  - Residual analysis")

    # 3. Pressure posteriors
    figs['pressures'] = plot_mcmc_pressures(result, show=False)
    print("  - Pressure posteriors")

    # 4. Convergence diagnostics (if trace available)
    if result.trace is not None:
        figs['diagnostics'] = plot_mcmc_diagnostics(result, show=False)
        print("  - Trace diagnostics")

    # 5. Sigma comparison (if graph provided)
    if G is not None:
        fig_sigma = plot_sigma_comparison(G, show=False)
        if fig_sigma is not None:
            figs['sigma_comparison'] = fig_sigma
            print("  - Sigma comparison (σ_pred vs σ_meas)")

    # 6. Network boundary priors (if graph provided)
    if G is not None:
        fig_boundary = plot_network_boundary_priors(G, result=result, background=background, show=False)
        if fig_boundary is not None:
            figs['boundary_priors'] = fig_boundary
            print("  - Network boundary priors")

    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return figs


def plot_sigma_comparison(G: nx.Graph, show: bool = True):
    """
    Plot comparison of σ_pred (pressure posterior) vs σ_meas (measurement noise).

    This helps diagnose which uncertainty source dominates the total prediction
    interval: the model uncertainty from pressure inference, or the original
    measurement uncertainty.

    Args:
        G: NetworkX graph with MCMC results written to edges
        show: Whether to plt.show()

    Returns:
        fig: matplotlib figure, or None if insufficient data
    """
    import matplotlib.pyplot as plt

    # Get sigma_scale if available
    sigma_scale = G.graph.get('mcmc_sigma_scale', 1.0)
    if sigma_scale is None:
        sigma_scale = 1.0

    # Collect data from edges
    sigma_preds = []
    sigma_meas_raw = []
    sigma_meas_scaled = []
    Q_preds = []

    for u, v, data in G.edges(data=True):
        sigma_pred = data.get('Q_pred_std')
        sigma_m = data.get('sigma_Q_nL_s') or data.get('sigma_Q')
        Q_pred = data.get('Q_predicted')

        if sigma_pred is None or sigma_m is None or Q_pred is None:
            continue
        if not np.isfinite(sigma_pred) or not np.isfinite(sigma_m):
            continue

        sigma_preds.append(sigma_pred)
        sigma_meas_raw.append(sigma_m)
        sigma_meas_scaled.append(sigma_scale * sigma_m)
        Q_preds.append(abs(Q_pred))

    if len(sigma_preds) < 3:
        print("  Insufficient data for sigma comparison plot")
        return None

    sigma_preds = np.array(sigma_preds)
    sigma_meas_raw = np.array(sigma_meas_raw)
    sigma_meas_scaled = np.array(sigma_meas_scaled)
    Q_preds = np.array(Q_preds)

    # Compute statistics
    ratio = sigma_preds / sigma_meas_scaled
    sigma_total = np.sqrt(sigma_preds**2 + sigma_meas_scaled**2)
    frac_from_pred = (sigma_preds**2 / sigma_total**2).mean()
    pred_dominates = np.sum(sigma_preds > sigma_meas_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter: σ_pred vs σ_meas_scaled
    ax1 = axes[0, 0]
    ax1.scatter(sigma_meas_scaled, sigma_preds, alpha=0.5, s=20)
    lims = [0, max(sigma_preds.max(), sigma_meas_scaled.max()) * 1.05]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='y=x (equal)')
    ax1.set_xlabel('σ_meas × σ_scale (nL/s)')
    ax1.set_ylabel('σ_pred (nL/s)')
    ax1.set_title('σ_pred vs σ_meas (scaled)')
    ax1.legend()
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)

    # 2. Histogram of log ratio
    ax2 = axes[0, 1]
    log_ratio = np.log10(ratio)
    ax2.hist(log_ratio, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Equal (ratio=1)')
    ax2.set_xlabel('log₁₀(σ_pred / σ_meas_scaled)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Ratio distribution (median={np.median(ratio):.2f})')
    ax2.legend()

    # 3. Both vs |Q_pred|
    ax3 = axes[1, 0]
    ax3.scatter(Q_preds, sigma_preds, alpha=0.5, s=20, label='σ_pred (model)')
    ax3.scatter(Q_preds, sigma_meas_scaled, alpha=0.5, s=20, label=f'σ_meas × {sigma_scale:.1f}')
    ax3.set_xlabel('|Q_pred| (nL/s)')
    ax3.set_ylabel('Uncertainty (nL/s)')
    ax3.set_title('Uncertainties vs flow magnitude')
    ax3.legend()
    ax3.set_xlim(0, None)
    ax3.set_ylim(0, None)

    # 4. Summary text panel
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
σ_pred vs σ_meas Comparison
{'='*40}

DATA:
  Edges analyzed:  {len(sigma_preds)}
  σ_scale (MCMC):  {sigma_scale:.2f}×

σ_pred (pressure posterior propagated):
  Range:   [{sigma_preds.min():.3f}, {sigma_preds.max():.3f}] nL/s
  Median:  {np.median(sigma_preds):.3f} nL/s
  Mean:    {sigma_preds.mean():.3f} nL/s

σ_meas × σ_scale (scaled measurement noise):
  Range:   [{sigma_meas_scaled.min():.3f}, {sigma_meas_scaled.max():.3f}] nL/s
  Median:  {np.median(sigma_meas_scaled):.3f} nL/s
  Mean:    {sigma_meas_scaled.mean():.3f} nL/s

RATIO (σ_pred / σ_meas_scaled):
  Range:   [{ratio.min():.2f}, {ratio.max():.2f}]
  Median:  {np.median(ratio):.2f}

DOMINANCE:
  σ_pred > σ_meas: {pred_dominates}/{len(ratio)} ({100*pred_dominates/len(ratio):.0f}%)
  σ_meas > σ_pred: {len(ratio)-pred_dominates}/{len(ratio)} ({100*(1-pred_dominates/len(ratio)):.0f}%)

VARIANCE CONTRIBUTION to σ_total:
  From σ_pred²:  {100*frac_from_pred:.0f}%
  From σ_meas²:  {100*(1-frac_from_pred):.0f}%

INTERPRETATION:
  • σ_pred reflects uncertainty from pressure inference
  • σ_meas reflects original measurement uncertainty
  • Prediction interval uses σ_total = √(σ_pred² + σ_meas²)
"""

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Uncertainty Decomposition: σ_pred vs σ_meas (σ_scale={sigma_scale:.2f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_network_boundary_priors(
    G: nx.Graph,
    result: Optional[MCMCResult] = None,
    background: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot network graph with boundary nodes colored by IRLS prior pressure.

    Visualizes the network showing:
    - Boundary nodes: colored by point estimate, sized by uncertainty
    - Edges: colored differently for measured vs unmeasured
    - Optional background image (e.g., max intensity projection)

    Args:
        G: NetworkX graph with MCMC results (or with pressure_prior_mean on nodes)
        result: Optional MCMCResult for additional data
        background: Optional 2D or 3D array. If 3D, max projection is computed.
        output_path: Optional path to save figure
        show: Whether to plt.show()

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Process background image
    bg_image = None
    if background is not None:
        if background.ndim == 3:
            # Compute max intensity projection along time axis
            bg_image = np.max(background, axis=0)
        elif background.ndim == 2:
            bg_image = background
        else:
            print(f"  Warning: background has {background.ndim} dimensions, expected 2 or 3")

    # Collect boundary node data
    boundary_nodes = []
    node_x = []
    node_y = []
    prior_means = []
    prior_stds = []

    for node, data in G.nodes(data=True):
        # Check if boundary node (either from result or from graph attribute)
        is_boundary = data.get('is_boundary_sim', False)
        if result is not None:
            is_boundary = is_boundary or (node in result.boundary_nodes)

        if not is_boundary:
            continue

        # Get coordinates
        x = data.get('x')
        y = data.get('y')
        if x is None or y is None:
            continue

        # Get prior mean (P_k^IRLS)
        prior_mean = data.get('pressure_prior_mean')
        if prior_mean is None and result is not None and result.P_prior_mean is not None:
            prior_mean = result.P_prior_mean.get(node)
        if prior_mean is None:
            prior_mean = data.get('pressure_Pa', 0.0)  # Fall back to posterior

        # Get prior std (σ_k^IRLS from IRLS covariance diagonal)
        prior_std = data.get('pressure_prior_std')
        if prior_std is None and result is not None and result.P_prior_std is not None:
            prior_std = result.P_prior_std.get(node)
        if prior_std is None:
            prior_std = 100.0  # Default

        boundary_nodes.append(node)
        node_x.append(x)
        node_y.append(y)
        prior_means.append(prior_mean)
        prior_stds.append(prior_std)  # Already σ_k^IRLS

    if len(boundary_nodes) == 0:
        print("  No boundary nodes found with coordinates")
        return None

    node_x = np.array(node_x)
    node_y = np.array(node_y)
    prior_means = np.array(prior_means)
    prior_stds = np.array(prior_stds)

    # Debug: show range of prior_stds
    print(f"  Boundary priors: {len(boundary_nodes)} nodes")
    print(f"    P_k^IRLS range: [{prior_means.min():.1f}, {prior_means.max():.1f}] Pa")
    print(f"    σ_k^IRLS range: [{prior_stds.min():.1f}, {prior_stds.max():.1f}] Pa")

    # Determine if we need to flip y-coordinates for image overlay
    # Graph uses Cartesian (y up), images use (y down)
    if bg_image is not None:
        img_height, img_width = bg_image.shape[:2]
        # Transform y: image_y = height - cartesian_y
        node_y = img_height - node_y
        use_image_coords = True
        # Figure size based on image aspect ratio
        aspect = img_width / img_height
        fig_height = 10
        fig_width = fig_height * aspect + 2  # Extra space for colorbar
    else:
        use_image_coords = False
        img_height = None
        fig_width, fig_height = 14, 10

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Display background image if provided
    if bg_image is not None:
        ax.imshow(bg_image, cmap='gray', origin='upper', aspect='equal')

    # Draw edges first (underneath nodes)
    # Collect measured and unmeasured edges
    meas_segments = []
    unmeas_segments = []

    for u, v, data in G.edges(data=True):
        # Get node positions
        u_data = G.nodes.get(u, {})
        v_data = G.nodes.get(v, {})
        x1, y1 = u_data.get('x'), u_data.get('y')
        x2, y2 = v_data.get('x'), v_data.get('y')

        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        # Transform to image coordinates if needed
        if use_image_coords:
            y1 = img_height - y1
            y2 = img_height - y2

        is_measured = data.get('is_measured', False)
        # Also check for flow measurement attributes
        if not is_measured:
            Q = data.get('mean_Q_nL_s') or data.get('Q_measured')
            is_measured = Q is not None and np.isfinite(Q)

        if is_measured:
            meas_segments.append([(x1, y1), (x2, y2)])
        else:
            unmeas_segments.append([(x1, y1), (x2, y2)])

    # Draw edges
    from matplotlib.collections import LineCollection

    # Adjust edge colors for visibility over image
    if bg_image is not None:
        meas_color = 'cyan'
        unmeas_color = 'white'
        meas_alpha = 0.8
        unmeas_alpha = 0.4
    else:
        meas_color = 'steelblue'
        unmeas_color = 'lightgray'
        meas_alpha = 0.7
        unmeas_alpha = 0.5

    if meas_segments:
        lc_meas = LineCollection(meas_segments, colors=meas_color, linewidths=1.5,
                                  alpha=meas_alpha, label='Measured')
        ax.add_collection(lc_meas)

    if unmeas_segments:
        lc_unmeas = LineCollection(unmeas_segments, colors=unmeas_color, linewidths=1.0,
                                    alpha=unmeas_alpha, linestyles='--', label='Unmeasured')
        ax.add_collection(lc_unmeas)

    # Node colors: by P_k^IRLS (pressure)
    # Use symmetric colormap centered at 0 (or median if all positive)
    if prior_means.min() >= 0:
        norm = Normalize(vmin=0, vmax=prior_means.max() * 1.1)
        cmap = plt.get_cmap('Reds')
    else:
        vmax = max(abs(prior_means.min()), abs(prior_means.max()))
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')

    node_colors = cmap(norm(prior_means))

    # Node sizes: by σ_k^IRLS (larger std = larger node = less well constrained)
    # Use sqrt scaling for better visual discrimination across the range
    # Well constrained (small σ) = small node, poorly constrained (large σ) = large node
    std_min, std_max = prior_stds.min(), prior_stds.max()
    if std_max > std_min:
        # Sqrt scaling helps spread out the visual range
        std_sqrt = np.sqrt(prior_stds)
        sqrt_min, sqrt_max = np.sqrt(std_min), np.sqrt(std_max)
        std_norm = (std_sqrt - sqrt_min) / (sqrt_max - sqrt_min)
    else:
        std_norm = np.ones_like(prior_stds) * 0.5
    # Wide size range: 30 (well constrained) to 800 (poorly constrained)
    node_sizes = 30 + 770 * std_norm

    # Draw nodes - use white edges for visibility on dark backgrounds
    node_edge_color = 'white' if bg_image is not None else 'black'
    scatter = ax.scatter(node_x, node_y, c=node_colors, s=node_sizes,
                         edgecolors=node_edge_color, linewidths=1.5, zorder=5)

    # Add uncertainty labels on nodes (σ_k^IRLS values)
    for i, node in enumerate(boundary_nodes):
        # Format uncertainty value - use integer if large, otherwise 1 decimal
        sigma_val = prior_stds[i]
        if sigma_val >= 10:
            label = f"{sigma_val:.0f}"
        else:
            label = f"{sigma_val:.1f}"

        # Compute font size based on node size (larger nodes get larger text)
        # sqrt(s) gives radius in points; scale font to fit inside
        node_radius_pts = np.sqrt(node_sizes[i])
        fontsize = max(5, min(10, node_radius_pts * 0.35))

        # Text color: black for light nodes, white for dark nodes
        # Use luminance of node color to decide
        rgb = node_colors[i][:3]
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = 'black' if luminance > 0.5 else 'white'

        ax.text(node_x[i], node_y[i], label,
                fontsize=fontsize, ha='center', va='center',
                color=text_color, fontweight='bold', zorder=6)

    # Colorbar for pressure
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Point Pressure Estimate (Pa)', fontsize=11)

    # Axis settings
    ax.set_aspect('equal')
    if bg_image is not None:
        # Calculate padding to prevent nodes from being cut off
        # Convert largest node size from points^2 to approximate pixels
        # (rough approximation: 1 point ≈ 1.33 pixels at 100 dpi figure)
        max_node_radius = np.sqrt(node_sizes.max()) * 0.7  # points -> pixels approx
        padding = max_node_radius + 5  # Extra margin for safety

        # Set axis limits with padding
        ax.set_xlim(-padding, img_width + padding)
        ax.set_ylim(img_height + padding, -padding)  # Inverted for image coords

        # Clean look for image overlay - no axis labels
        ax.axis('off')
        ax.set_title(f'Boundary Priors ({len(boundary_nodes)} nodes)\n'
                     f'Color = Point Estimate, Size = Uncertainty', fontsize=12)
    else:
        ax.autoscale()
        # Add padding for nodes at edges
        x_margin = (node_x.max() - node_x.min()) * 0.05 + np.sqrt(node_sizes.max()) * 0.5
        y_margin = (node_y.max() - node_y.min()) * 0.05 + np.sqrt(node_sizes.max()) * 0.5
        ax.set_xlim(node_x.min() - x_margin, node_x.max() + x_margin)
        ax.set_ylim(node_y.min() - y_margin, node_y.max() + y_margin)
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.set_title(f'Network Boundary Priors: {len(boundary_nodes)} boundary nodes\n'
                     f'Color = Point Estimate, Size = Uncertainty (σ)', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


def save_mcmc_figures(
    result: MCMCResult,
    G: nx.Graph,
    output_dir: str,
    prefix: str = "mcmc",
    background: Optional[np.ndarray] = None,
    show: bool = False,
):
    """
    Save all MCMC diagnostic figures to a directory.

    Args:
        result: MCMCResult from bayesian_pressure_mcmc
        G: NetworkX graph with MCMC results written
        output_dir: Directory to save figures
        prefix: Filename prefix (default: "mcmc")
        background: Optional video/image for boundary priors overlay
        show: Whether to also display figures

    Returns:
        dict: Mapping of figure names to file paths
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    print(f"\nSaving MCMC figures to: {output_dir}")

    # 1. Summary
    fig = plot_mcmc_summary(result, show=False)
    if fig is not None:
        path = output_dir / f"{prefix}_summary.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['summary'] = str(path)
        print(f"  - {path.name}")
        plt.close(fig)

    # 2. Residuals
    fig = plot_mcmc_residuals(result, show=False)
    if fig is not None:
        path = output_dir / f"{prefix}_residuals.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['residuals'] = str(path)
        print(f"  - {path.name}")
        plt.close(fig)

    # 3. Pressure posteriors
    fig = plot_mcmc_pressures(result, show=False)
    if fig is not None:
        path = output_dir / f"{prefix}_pressures.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['pressures'] = str(path)
        print(f"  - {path.name}")
        plt.close(fig)

    # 4. Convergence diagnostics
    if result.trace is not None:
        fig = plot_mcmc_diagnostics(result, show=False)
        if fig is not None:
            path = output_dir / f"{prefix}_diagnostics.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['diagnostics'] = str(path)
            print(f"  - {path.name}")
            plt.close(fig)

    # 5. Sigma comparison
    fig = plot_sigma_comparison(G, show=False)
    if fig is not None:
        path = output_dir / f"{prefix}_sigma_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files['sigma_comparison'] = str(path)
        print(f"  - {path.name}")
        plt.close(fig)

    # 6. Network boundary priors (with optional background image)
    path = output_dir / f"{prefix}_boundary_priors.png"
    fig = plot_network_boundary_priors(G, result, background=background, output_path=str(path), show=False)
    if fig is not None:
        saved_files['boundary_priors'] = str(path)
        plt.close(fig)

    # 7. MCMC sampling schematic (for presentations)
    path = output_dir / f"{prefix}_sampling_schematic.png"
    fig = plot_mcmc_sampling_schematic(result, use_synthetic=True, output_path=str(path), show=False)
    if fig is not None:
        saved_files['sampling_schematic'] = str(path)
        plt.close(fig)

    # 8. Noise model motivation (σ vs |Q| and χ² histogram)
    path = output_dir / f"{prefix}_noise_motivation.png"
    fig = plot_noise_model_motivation(G, output_path=str(path), show=False)
    if fig is not None:
        saved_files['noise_motivation'] = str(path)
        plt.close(fig)

    print(f"\nSaved {len(saved_files)} figures to {output_dir}")

    if show:
        # Re-open and show all figures
        for name, path in saved_files.items():
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(name)
        plt.show()

    return saved_files


def plot_noise_model_motivation(
    G: nx.Graph,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot empirical evidence motivating the heteroscedastic noise model.

    Creates a single-panel scatter plot:
    - X-axis: |Q_obs| (flow magnitude)
    - Y-axis: σ_Q^meas (measurement uncertainty)
    - Color: χ²_red (fit quality) - shows these provide independent information

    The scatter of colors throughout the plot demonstrates that χ² and σ_meas
    are weakly correlated, justifying both σ_rel and α terms in the noise model.

    Args:
        G: NetworkX graph with flow measurements on edges
        output_path: Optional path to save figure
        show: Whether to display the figure

    Returns:
        fig: matplotlib figure, or None if insufficient data
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Collect matched data from edges (need all three values for each point)
    Q_obs = []
    sigma_meas = []
    chi2_vals = []

    for u, v, data in G.edges(data=True):
        # Get flow magnitude
        Q = data.get('mean_Q_nL_s') or data.get('Q') or data.get('mean_Q')
        if Q is None or not np.isfinite(Q):
            continue

        # Get chi-squared (fit quality)
        chi2 = data.get('chi2_red') or data.get('chi2_reduced')
        if chi2 is None or not np.isfinite(chi2) or chi2 <= 0:
            continue

        # Get measurement uncertainty - prefer raw (before any model-based scaling)
        sigma_raw = data.get('sigma_Q_raw')
        if sigma_raw is not None and np.isfinite(sigma_raw) and sigma_raw > 0:
            sigma = sigma_raw
        else:
            sigma_orig = data.get('sigma_Q_original')
            if sigma_orig is not None and np.isfinite(sigma_orig):
                sigma = sigma_orig
            else:
                sigma = data.get('sigma_Q_nL_s') or data.get('sigma_Q')

        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            continue

        Q_obs.append(abs(Q))
        sigma_meas.append(sigma)
        chi2_vals.append(chi2)

    Q_obs = np.array(Q_obs)
    sigma_meas = np.array(sigma_meas)
    chi2_vals = np.array(chi2_vals)

    n_points = len(Q_obs)
    if n_points < 10:
        print(f"  Insufficient matched data for noise motivation plot (n={n_points})")
        return None

    # Filter outliers for visualization (use 95th percentile as upper bound for σ)
    sigma_threshold = np.percentile(sigma_meas, 95)
    chi2_cap = 5.0  # Cap χ² color at 5 for better visualization
    inlier_mask = sigma_meas <= sigma_threshold
    n_outliers = np.sum(~inlier_mask)

    Q_plot = Q_obs[inlier_mask]
    sigma_plot = sigma_meas[inlier_mask]
    chi2_plot = np.clip(chi2_vals[inlier_mask], 0.1, chi2_cap)  # Clip for color scale

    # Create figure - single panel
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter plot with χ² as color (log scale for better contrast)
    scatter = ax.scatter(Q_plot, sigma_plot, c=chi2_plot, cmap='plasma',
                         norm=LogNorm(vmin=0.1, vmax=chi2_cap),
                         alpha=0.7, s=30, edgecolors='white', linewidths=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r'$\chi^2_{\mathrm{red}}$ (profile fit quality)', fontsize=10)

    # Add trend line using robust regression
    if len(Q_plot) > 10:
        from scipy import stats
        res = stats.theilslopes(sigma_plot, Q_plot)
        slope, intercept = res[0], res[1]
        Q_line = np.linspace(0, Q_plot.max() * 1.05, 100)
        ax.plot(Q_line, intercept + slope * Q_line, '--', c='white', lw=2.5)
        ax.plot(Q_line, intercept + slope * Q_line, '--', c='#e53e3e', lw=2,
                label=f'Trend: σ ≈ {intercept:.2f} + {slope:.2f}|Q|')
        ax.legend(frameon=True, fontsize=9, loc='upper left',
                  facecolor='white', edgecolor='none', framealpha=0.8)

    ax.set_xlabel(r'$|Q_e^{\mathrm{obs}}|$ (nL/s)', fontsize=11)
    ax.set_ylabel(r'$\sigma_Q^{\mathrm{meas}}$ (nL/s)', fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title
    title = 'Measurement uncertainty vs flow'
    subtitle = f'Color = fit quality (n={n_points}'
    if n_outliers > 0:
        subtitle += f', {n_outliers} σ outliers hidden'
    subtitle += ')'
    ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_mcmc_sampling_schematic(
    result: Optional[MCMCResult] = None,
    param1: str = "P_0",
    param2: str = "P_1",
    n_warmup: int = 50,
    output_path: Optional[str] = None,
    show: bool = True,
    use_synthetic: bool = False,
):
    """
    Create an MCMC sampling schematic figure for presentations.

    Shows 2D posterior contours with the sampler's trace overlaid,
    distinguishing warmup (adaptation) from actual sampling.

    Args:
        result: Optional MCMCResult with trace. If None or use_synthetic=True,
                generates a synthetic example.
        param1: First parameter name (x-axis). For pressures, use "P_0", "P_1", etc.
                or "sigma_base", "sigma_rel", "alpha" for noise hyperparameters.
        param2: Second parameter name (y-axis).
        n_warmup: Number of warmup samples to distinguish (gray).
        output_path: Optional path to save figure.
        show: Whether to display the figure.
        use_synthetic: Force use of synthetic data (for clean illustrations).

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Extract samples from result or generate synthetic
    samples_x = None
    samples_y = None
    xlabel = param1
    ylabel = param2

    if result is not None and result.trace is not None and not use_synthetic:
        try:
            import arviz as az
            # Try to extract the requested parameters
            trace = result.trace

            # Get parameter names from trace
            if hasattr(trace, 'posterior'):
                posterior = trace.posterior
                var_names = list(posterior.data_vars)

                # Map friendly names to trace variable names
                param_map = {}
                for vn in var_names:
                    if vn.startswith('P_'):
                        # Pressure parameters
                        param_map[vn] = vn
                    elif vn in ['sigma_base', 'sigma_rel', 'alpha', 'mu']:
                        param_map[vn] = vn

                # Also map P_0, P_1 to actual pressure variable names
                if 'P' in var_names:
                    # Multi-dimensional P array
                    n_pressures = posterior['P'].shape[-1]
                    for i in range(n_pressures):
                        param_map[f'P_{i}'] = ('P', i)

                # Extract samples
                def get_samples(param_name):
                    if param_name in param_map:
                        mapped = param_map[param_name]
                        if isinstance(mapped, tuple):
                            var_name, idx = mapped
                            return posterior[var_name].values.flatten()[:, idx] if posterior[var_name].ndim > 2 else posterior[var_name].values[:, :, idx].flatten()
                        else:
                            return posterior[mapped].values.flatten()
                    return None

                samples_x = get_samples(param1)
                samples_y = get_samples(param2)

                # Update labels with units if known
                if param1.startswith('P_'):
                    # Convert P_0 -> $P_{0}$ for LaTeX
                    latex1 = '$' + param1.replace('_', '_{') + '}$ (Pa)'
                    xlabel = latex1
                elif param1 == 'sigma_base':
                    xlabel = r'$\sigma_{\mathrm{base}}$ (nL/s)'
                elif param1 == 'sigma_rel':
                    xlabel = r'$\sigma_{\mathrm{rel}}$'
                elif param1 == 'alpha':
                    xlabel = r'$\alpha$'

                if param2.startswith('P_'):
                    latex2 = '$' + param2.replace('_', '_{') + '}$ (Pa)'
                    ylabel = latex2
                elif param2 == 'sigma_base':
                    ylabel = r'$\sigma_{\mathrm{base}}$ (nL/s)'
                elif param2 == 'sigma_rel':
                    ylabel = r'$\sigma_{\mathrm{rel}}$'
                elif param2 == 'alpha':
                    ylabel = r'$\alpha$'

        except Exception as e:
            print(f"  Could not extract samples from trace: {e}")
            samples_x = None
            samples_y = None

    # Generate synthetic data if needed
    if samples_x is None or samples_y is None:
        print("  Using synthetic data for illustration")
        np.random.seed(42)

        # Create correlated 2D Gaussian posterior
        mean = [15, 20]
        cov = [[25, 12], [12, 30]]

        # Generate "warmup" samples: random walk toward mode
        n_total = 200
        warmup_x = [mean[0] - 25]
        warmup_y = [mean[1] + 15]
        for i in range(n_warmup - 1):
            # Drift toward mode with decreasing step size (simulating adaptation)
            drift_x = (mean[0] - warmup_x[-1]) * 0.08
            drift_y = (mean[1] - warmup_y[-1]) * 0.08
            step_scale = 3.0 * (1 - i / n_warmup)  # Decreasing step size
            warmup_x.append(warmup_x[-1] + drift_x + np.random.randn() * step_scale)
            warmup_y.append(warmup_y[-1] + drift_y + np.random.randn() * step_scale)

        # Generate post-warmup samples from the actual posterior
        post_samples = np.random.multivariate_normal(mean, cov, n_total - n_warmup)

        samples_x = np.concatenate([warmup_x, post_samples[:, 0]])
        samples_y = np.concatenate([warmup_y, post_samples[:, 1]])

        xlabel = r'$P_1$ (Pa)'
        ylabel = r'$P_2$ (Pa)'

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Compute and plot posterior contours from post-warmup samples
    post_x = samples_x[n_warmup:]
    post_y = samples_y[n_warmup:]

    # Create 2D histogram for contours
    x_range = [post_x.min() - 0.15 * (post_x.max() - post_x.min()),
               post_x.max() + 0.15 * (post_x.max() - post_x.min())]
    y_range = [post_y.min() - 0.15 * (post_y.max() - post_y.min()),
               post_y.max() + 0.15 * (post_y.max() - post_y.min())]

    # Kernel density estimation for smooth contours
    try:
        # Combine post-warmup samples for KDE
        xy = np.vstack([post_x, post_y])
        kde = stats.gaussian_kde(xy)

        # Evaluate on grid
        xx, yy = np.mgrid[x_range[0]:x_range[1]:100j, y_range[0]:y_range[1]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = np.reshape(kde(positions).T, xx.shape)

        # Plot filled contours (posterior density)
        levels = np.percentile(zz[zz > 0], [5, 25, 50, 75, 90, 95])
        ax.contourf(xx, yy, zz, levels=levels, cmap='Blues', alpha=0.4)
        ax.contour(xx, yy, zz, levels=levels, colors='#2c5282', alpha=0.6, linewidths=0.8)
    except Exception:
        # Fallback to simple scatter density
        pass

    # Plot warmup trace (gray, with connecting lines to show adaptation path)
    warmup_x = samples_x[:n_warmup]
    warmup_y = samples_y[:n_warmup]
    ax.plot(warmup_x, warmup_y, '-', c='gray', lw=0.8, alpha=0.5, zorder=2)
    ax.scatter(warmup_x, warmup_y, c='gray', s=18, alpha=0.7, edgecolors='white',
               linewidths=0.3, label='Warmup (adaptation)', zorder=3)

    # Plot post-warmup samples (scatter only - cleaner for presentations)
    ax.scatter(post_x, post_y, c='#e53e3e', s=25, alpha=0.7, edgecolors='white',
               linewidths=0.4, label='Posterior samples', zorder=5)

    # Mark starting point
    ax.scatter([samples_x[0]], [samples_y[0]], c='black', s=100, marker='*',
               edgecolors='white', linewidths=1, label='Start (IRLS)', zorder=6)

    # Mark posterior mean
    ax.scatter([post_x.mean()], [post_y.mean()], c='#2c5282', s=80, marker='x',
               linewidths=2.5, label='Posterior mean', zorder=6)

    # Labels and styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('MCMC Sampling: From Prior to Posterior', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# PULSATILE EXTENSION: Joint DC + AC (n=0 and n=1 harmonics) inference
# =============================================================================

@dataclass
class PulsatileMCMCResult(MCMCResult):
    """
    Extended results from pulsatile MCMC inference.

    Inherits all DC fields from MCMCResult, adds AC (n=1 harmonic) fields.

    Convention:
    - Pressure phasor: P̃_i = P_amp_i × exp(i × P_phase_i) = P_re_i + i × P_im_i
    - Flow phasor: Q̃ = g × (P̃_start - P̃_end)
    - Positive phase = leading (peaks earlier in time)
    """
    # AC pressure posteriors (n=1 harmonic)
    # Complex representation: P̃ = P_re + i × P_im
    P_re: Optional[Dict[int, float]] = None  # Real part of pressure phasor (Pa)
    P_im: Optional[Dict[int, float]] = None  # Imaginary part (Pa)
    P_re_std: Optional[Dict[int, float]] = None
    P_im_std: Optional[Dict[int, float]] = None

    # Amplitude/phase representation (derived from P_re, P_im)
    P_amp: Optional[Dict[int, float]] = None  # Pressure amplitude (Pa)
    P_phase: Optional[Dict[int, float]] = None  # Pressure phase (radians)
    P_amp_std: Optional[Dict[int, float]] = None
    P_phase_std: Optional[Dict[int, float]] = None

    # Predicted AC flows
    predicted_Q_amp: Optional[Dict[Tuple[int, int], float]] = None  # nL/s
    predicted_Q_phase: Optional[Dict[Tuple[int, int], float]] = None  # radians

    # Measured AC flows (for comparison)
    measured_Q_amp: Optional[Dict[Tuple[int, int], float]] = None
    measured_Q_phase: Optional[Dict[Tuple[int, int], float]] = None

    # AC fit quality
    r_squared_ac: Optional[float] = None
    rmse_amp: Optional[float] = None  # nL/s
    rmse_phase: Optional[float] = None  # radians

    # Phase reference node (P_im = 0 at this node)
    phase_ref_node: Optional[int] = None


def _prepare_pulsatile_data(
    G: nx.Graph,
    measured_edges: List[dict],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract AC measurements (amp_Q, phase) from edges.

    Returns:
        amp_Q: (n_edges,) flow amplitudes in nL/s
        phase_Q: (n_edges,) flow phases in radians
        Q_re: (n_edges,) real part = amp_Q × cos(phase)
        Q_im: (n_edges,) imaginary part = amp_Q × sin(phase)
    """
    n_edges = len(measured_edges)
    amp_Q = np.full(n_edges, np.nan)
    phase_Q = np.full(n_edges, np.nan)

    n_valid = 0
    for i, edge_info in enumerate(measured_edges):
        u, v = edge_info['u'], edge_info['v']
        edge_data = G[u][v]

        # Get amplitude and phase from edge data
        # Try both naming conventions: full graph uses amp_Q/phase,
        # reduced graph uses amp_Q_nL_s/phase_rad
        amp = edge_data.get('amp_Q')
        if amp is None:
            amp = edge_data.get('amp_Q_nL_s')

        phase = edge_data.get('phase')
        if phase is None:
            phase = edge_data.get('phase_rad')

        if amp is not None and phase is not None and np.isfinite(amp) and np.isfinite(phase):
            amp_Q[i] = amp
            phase_Q[i] = phase
            n_valid += 1

    if verbose:
        print(f"  AC data: {n_valid}/{n_edges} edges have valid amp_Q and phase")

    # Convert to Cartesian (handles NaN gracefully)
    Q_re = amp_Q * np.cos(phase_Q)
    Q_im = amp_Q * np.sin(phase_Q)

    return amp_Q, phase_Q, Q_re, Q_im


def run_mcmc_pulsatile_numpyro(
    A: np.ndarray,
    Q_mean_obs: np.ndarray,
    amp_Q_obs: np.ndarray,
    Q_re_obs: np.ndarray,
    Q_im_obs: np.ndarray,
    mu_Pa_s: float = 3.5e-3,
    likelihood: str = 'studentt',
    nu: float = 4.0,
    P_mean_prior_mean: Optional[np.ndarray] = None,
    P_mean_prior_std: Optional[np.ndarray] = None,
    P_ac_prior_std: float = 50.0,  # Prior std for AC pressure components (Pa)
    n_warmup: int = 1000,
    n_samples: int = 2000,
    n_chains: int = 4,
    verbose: bool = True,
) -> Tuple[object, dict]:
    """
    Run joint DC + AC MCMC inference using NumPyro.

    Model:
        DC: Q̄_pred = A @ P̄ / μ
        AC: Q̃_pred = A @ P̃ / μ  (same A matrix!)

    Noise model (shared):
        σ_dc = σ_base + σ_rel × |Q̄_obs|
        σ_ac = σ_base + σ_rel × amp_Q_obs

    Parameters
    ----------
    A : np.ndarray
        Design matrix (n_edges, n_free_boundary) - same for DC and AC
    Q_mean_obs : np.ndarray
        Observed mean flows (n_edges,)
    amp_Q_obs : np.ndarray
        Observed flow amplitudes (n_edges,), NaN for missing
    Q_re_obs, Q_im_obs : np.ndarray
        Observed flow phasors in Cartesian form (n_edges,)
    mu_Pa_s : float
        Viscosity in Pa·s
    likelihood : str
        'gaussian' or 'studentt'
    nu : float
        Degrees of freedom for Student-t
    P_mean_prior_mean, P_mean_prior_std : np.ndarray
        Prior for DC pressures (from IRLS)
    P_ac_prior_std : float
        Prior std for AC pressure components

    Returns
    -------
    mcmc : NumPyro MCMC object
    results : dict with posterior samples and summaries
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro not available")

    n_edges, n_free = A.shape

    # Scale A by viscosity
    A_scaled = jnp.array(A / mu_Pa_s)

    # Mask for valid AC observations (compute before converting to JAX)
    ac_valid_np = np.isfinite(amp_Q_obs) & (amp_Q_obs > 0)
    n_ac_valid = int(np.sum(ac_valid_np))

    # Convert observations to JAX arrays
    # Replace NaN with 0 for AC observations (they'll be ignored via large sigma)
    Q_mean_jax = jnp.array(Q_mean_obs)
    amp_Q_jax = jnp.array(np.nan_to_num(amp_Q_obs, nan=0.0))
    Q_re_jax = jnp.array(np.nan_to_num(Q_re_obs, nan=0.0))
    Q_im_jax = jnp.array(np.nan_to_num(Q_im_obs, nan=0.0))
    ac_valid = jnp.array(ac_valid_np)

    # Priors for DC
    if P_mean_prior_mean is not None:
        P_mean_prior_mean_jax = jnp.array(P_mean_prior_mean)
    else:
        P_mean_prior_mean_jax = jnp.zeros(n_free)

    if P_mean_prior_std is not None:
        P_mean_prior_std_jax = jnp.array(P_mean_prior_std)
    else:
        P_mean_prior_std_jax = jnp.full(n_free, 500.0)

    if verbose:
        print(f"\n  Pulsatile MCMC setup (NumPyro):")
        print(f"    Boundary pressures: {n_free}")
        print(f"    DC measurements: {n_edges}")
        print(f"    AC measurements: {n_ac_valid}")
        print(f"    Parameters: {n_free} (DC) + {2*n_free} (AC) = {3*n_free}")
        print(f"    Likelihood: {likelihood}" + (f" (ν={nu})" if likelihood == 'studentt' else ""))

    def model():
        # ===== Shared noise parameters =====
        # Heteroscedastic: σ = σ_base + σ_rel × |Q|
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(5.0))
        sigma_rel = numpyro.sample('sigma_rel', dist.HalfNormal(1.0))

        # ===== DC pressures (n=0 harmonic) =====
        P_mean = numpyro.sample('P_mean',
                                dist.Normal(P_mean_prior_mean_jax, P_mean_prior_std_jax))

        # ===== AC pressures (n=1 harmonic) =====
        # Real and imaginary parts, centered at 0
        # Note: One node's P_im could be fixed to 0 as phase reference,
        # but letting it float is fine - phases are relative anyway
        P_re = numpyro.sample('P_re', dist.Normal(0, P_ac_prior_std).expand([n_free]))
        P_im = numpyro.sample('P_im', dist.Normal(0, P_ac_prior_std).expand([n_free]))

        # ===== Forward model =====
        # DC: Q̄ = A @ P̄
        Q_mean_pred = jnp.dot(A_scaled, P_mean)

        # AC: Q̃ = A @ P̃ (same matrix!)
        Q_re_pred = jnp.dot(A_scaled, P_re)
        Q_im_pred = jnp.dot(A_scaled, P_im)

        # ===== Noise model =====
        # DC noise: based on observed |Q̄|
        sigma_dc = sigma_base + sigma_rel * jnp.abs(Q_mean_jax)
        sigma_dc = jnp.maximum(sigma_dc, 0.1)

        # AC noise: based on observed amplitude
        # Use same noise model structure for consistency
        sigma_ac = sigma_base + sigma_rel * amp_Q_jax
        sigma_ac = jnp.maximum(sigma_ac, 0.1)
        # Set large sigma for invalid AC observations (effectively ignore them)
        sigma_ac = jnp.where(ac_valid, sigma_ac, 1e6)

        # ===== Likelihoods =====
        if likelihood == 'gaussian':
            # DC likelihood (all edges)
            numpyro.sample('Q_mean_obs', dist.Normal(Q_mean_pred, sigma_dc),
                          obs=Q_mean_jax)
            # AC likelihood (valid edges only, via large sigma masking)
            numpyro.sample('Q_re_obs', dist.Normal(Q_re_pred, sigma_ac),
                          obs=Q_re_jax)
            numpyro.sample('Q_im_obs', dist.Normal(Q_im_pred, sigma_ac),
                          obs=Q_im_jax)
        else:
            # Student-t for robustness
            numpyro.sample('Q_mean_obs', dist.StudentT(nu, Q_mean_pred, sigma_dc),
                          obs=Q_mean_jax)
            numpyro.sample('Q_re_obs', dist.StudentT(nu, Q_re_pred, sigma_ac),
                          obs=Q_re_jax)
            numpyro.sample('Q_im_obs', dist.StudentT(nu, Q_im_pred, sigma_ac),
                          obs=Q_im_jax)

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, progress_bar=verbose)

    if verbose:
        print("  Sampling...")

    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    # Extract samples
    samples = mcmc.get_samples()

    results = {
        'P_mean_samples': np.array(samples['P_mean']),
        'P_re_samples': np.array(samples['P_re']),
        'P_im_samples': np.array(samples['P_im']),
        'sigma_base_samples': np.array(samples['sigma_base']),
        'sigma_rel_samples': np.array(samples['sigma_rel']),
    }

    # Compute posterior summaries
    results['P_mean'] = np.mean(results['P_mean_samples'], axis=0)
    results['P_mean_std'] = np.std(results['P_mean_samples'], axis=0)
    results['P_re'] = np.mean(results['P_re_samples'], axis=0)
    results['P_re_std'] = np.std(results['P_re_samples'], axis=0)
    results['P_im'] = np.mean(results['P_im_samples'], axis=0)
    results['P_im_std'] = np.std(results['P_im_samples'], axis=0)

    # Derived: amplitude and phase from Cartesian
    # P_amp = sqrt(P_re² + P_im²), P_phase = atan2(P_im, P_re)
    results['P_amp'] = np.sqrt(results['P_re']**2 + results['P_im']**2)
    results['P_phase'] = np.arctan2(results['P_im'], results['P_re'])

    # Noise parameters
    results['sigma_base'] = float(np.mean(results['sigma_base_samples']))
    results['sigma_rel'] = float(np.mean(results['sigma_rel_samples']))

    if verbose:
        print(f"\n  Inferred noise model:")
        print(f"    σ_base = {results['sigma_base']:.3f} nL/s")
        print(f"    σ_rel = {results['sigma_rel']:.3f}")
        print(f"    Example: |Q|=1 → σ={results['sigma_base'] + results['sigma_rel']:.2f}, "
              f"|Q|=10 → σ={results['sigma_base'] + 10*results['sigma_rel']:.2f}")

    return mcmc, results


def bayesian_pressure_mcmc_pulsatile(
    G: nx.Graph,
    boundary_nodes: Optional[Set[int]] = None,
    mu_cP: float = 3.5,
    likelihood: str = 'studentt',
    nu: float = 4.0,
    P_ac_prior_std: float = 50.0,
    n_tune: int = 1000,
    n_draws: int = 2000,
    n_chains: int = 4,
    verbose: bool = True,
) -> PulsatileMCMCResult:
    """
    Joint Bayesian inference of DC (mean) and AC (pulsatile) boundary pressures.

    This extends bayesian_pressure_mcmc() to include the n=1 harmonic,
    inferring pressure amplitude and phase at each boundary node.

    Physics:
        P_i(t) = P̄_i + P_amp_i × cos(ωt + φ_i)

        With Wo << 1 (quasi-steady), same conductance g applies:
        Q̄ = g × (P̄_start - P̄_end)           [DC, n=0]
        Q̃ = g × (P̃_start - P̃_end)           [AC, n=1]

    Parameters
    ----------
    G : nx.Graph
        Vessel network with flow attributes:
        - mean_Q: mean flow (required)
        - amp_Q: flow amplitude (optional, for AC)
        - phase: flow phase in radians (optional, for AC)
    boundary_nodes : set, optional
        Boundary nodes (auto-detect if None)
    mu_cP : float
        Viscosity in centipoise (default 3.5)
    likelihood : str
        'gaussian' or 'studentt' (default)
    nu : float
        Student-t degrees of freedom (default 4)
    P_ac_prior_std : float
        Prior std for AC pressure components in Pa (default 50)
    n_tune, n_draws, n_chains : int
        MCMC parameters
    verbose : bool
        Print progress

    Returns
    -------
    PulsatileMCMCResult
        Extended result with both DC and AC pressure posteriors
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro required for pulsatile inference")

    from .flow_consistency import _prepare_bayesian_data

    if verbose:
        print("\n" + "=" * 70)
        print("PULSATILE BAYESIAN PRESSURE INFERENCE (DC + AC)")
        print("=" * 70)

    # Prepare DC data (reuse existing infrastructure)
    data = _prepare_bayesian_data(
        G, boundary_nodes,
        verbose=verbose,
    )

    measured_edges = data['measured_edges']
    Q_meas = data['Q_meas']
    boundary_list = data['boundary_list']
    internal_nodes = data['internal_nodes']
    ref_node = data['ref_node']
    free_boundary = data['free_boundary']
    M = data['M']

    # Build design matrix (same for DC and AC!)
    A = build_design_matrix(
        G, measured_edges, boundary_list, internal_nodes, ref_node, M
    )

    # Prepare AC data
    amp_Q, phase_Q, Q_re, Q_im = _prepare_pulsatile_data(G, measured_edges, verbose)

    # Run joint inference
    mu_Pa_s = mu_cP * 1e-3

    mcmc, results = run_mcmc_pulsatile_numpyro(
        A=A,
        Q_mean_obs=Q_meas,
        amp_Q_obs=amp_Q,
        Q_re_obs=Q_re,
        Q_im_obs=Q_im,
        mu_Pa_s=mu_Pa_s,
        likelihood=likelihood,
        nu=nu,
        P_ac_prior_std=P_ac_prior_std,
        n_warmup=n_tune,
        n_samples=n_draws,
        n_chains=n_chains,
        verbose=verbose,
    )

    # Build result dictionaries keyed by node ID
    node_pressures = {}
    P_std = {}
    P_re_dict = {}
    P_im_dict = {}
    P_re_std_dict = {}
    P_im_std_dict = {}
    P_amp_dict = {}
    P_phase_dict = {}

    for i, node in enumerate(free_boundary):
        node_pressures[node] = float(results['P_mean'][i])
        P_std[node] = float(results['P_mean_std'][i])
        P_re_dict[node] = float(results['P_re'][i])
        P_im_dict[node] = float(results['P_im'][i])
        P_re_std_dict[node] = float(results['P_re_std'][i])
        P_im_std_dict[node] = float(results['P_im_std'][i])
        P_amp_dict[node] = float(results['P_amp'][i])
        P_phase_dict[node] = float(results['P_phase'][i])

    # Reference node has P = 0 for both DC and AC
    node_pressures[ref_node] = 0.0
    P_std[ref_node] = 0.0
    P_re_dict[ref_node] = 0.0
    P_im_dict[ref_node] = 0.0
    P_re_std_dict[ref_node] = 0.0
    P_im_std_dict[ref_node] = 0.0
    P_amp_dict[ref_node] = 0.0
    P_phase_dict[ref_node] = 0.0

    # Compute predicted flows
    A_scaled = A / mu_Pa_s
    Q_mean_pred = A_scaled @ results['P_mean']
    Q_re_pred = A_scaled @ results['P_re']
    Q_im_pred = A_scaled @ results['P_im']
    Q_amp_pred = np.sqrt(Q_re_pred**2 + Q_im_pred**2)
    Q_phase_pred = np.arctan2(Q_im_pred, Q_re_pred)

    predicted_Q = {}
    predicted_Q_amp = {}
    predicted_Q_phase = {}
    measured_Q = {}
    measured_Q_amp = {}
    measured_Q_phase = {}

    for i, edge_info in enumerate(measured_edges):
        key = (edge_info['start_j'], edge_info['end_j'])
        predicted_Q[key] = float(Q_mean_pred[i])
        predicted_Q_amp[key] = float(Q_amp_pred[i])
        predicted_Q_phase[key] = float(Q_phase_pred[i])
        measured_Q[key] = float(Q_meas[i])
        if np.isfinite(amp_Q[i]):
            measured_Q_amp[key] = float(amp_Q[i])
            measured_Q_phase[key] = float(phase_Q[i])

    # Fit quality metrics
    residuals_dc = Q_meas - Q_mean_pred
    ss_res_dc = np.sum(residuals_dc**2)
    ss_tot_dc = np.sum((Q_meas - np.mean(Q_meas))**2)
    r_squared_dc = 1 - ss_res_dc / ss_tot_dc if ss_tot_dc > 0 else 0.0
    rmse_dc = np.sqrt(np.mean(residuals_dc**2))

    # AC fit quality (only for valid measurements)
    ac_valid_mask = np.isfinite(amp_Q) & (amp_Q > 0)
    if np.any(ac_valid_mask):
        residuals_re = (Q_re - Q_re_pred)[ac_valid_mask]
        residuals_im = (Q_im - Q_im_pred)[ac_valid_mask]
        ss_res_ac = np.sum(residuals_re**2 + residuals_im**2)
        ss_tot_ac = np.sum(Q_re[ac_valid_mask]**2 + Q_im[ac_valid_mask]**2)
        r_squared_ac = 1 - ss_res_ac / ss_tot_ac if ss_tot_ac > 0 else 0.0
        rmse_amp = np.sqrt(np.mean((amp_Q[ac_valid_mask] - Q_amp_pred[ac_valid_mask])**2))
    else:
        r_squared_ac = np.nan
        rmse_amp = np.nan

    # Check convergence (pulsatile model uses different parameter names)
    converged, r_hat_max, ess_min, n_divergences = check_convergence_numpyro(
        mcmc, verbose, param_names=['P_mean', 'P_re', 'P_im']
    )

    if verbose:
        print(f"\n  DC fit: R² = {r_squared_dc:.3f}, RMSE = {rmse_dc:.3f} nL/s")
        if not np.isnan(r_squared_ac):
            print(f"  AC fit: R² = {r_squared_ac:.3f}, RMSE(amp) = {rmse_amp:.3f} nL/s")
        print(f"\n  Pressure amplitudes at boundaries:")
        for node in sorted(P_amp_dict.keys()):
            if node != ref_node:
                print(f"    Node {node}: P̄={node_pressures[node]:.1f} Pa, "
                      f"amp={P_amp_dict[node]:.1f} Pa, phase={np.degrees(P_phase_dict[node]):.0f}°")

    return PulsatileMCMCResult(
        # DC fields (inherited)
        mu_cP=mu_cP,
        node_pressures=node_pressures,
        predicted_Q=predicted_Q,
        measured_Q=measured_Q,
        P_std=P_std,
        P_ci95={},  # TODO: compute from samples
        Q_pred_std={},
        Q_pred_ci95={},
        r_squared=r_squared_dc,
        rmse=rmse_dc,
        chi2_reduced=np.nan,  # TODO
        converged=converged,
        r_hat_max=r_hat_max,
        ess_min=ess_min,
        n_divergences=n_divergences,
        boundary_nodes=set(boundary_list),
        ref_node=ref_node,
        n_parameters=3 * len(free_boundary),  # DC + AC_re + AC_im
        n_measurements=len(Q_meas) + 2 * np.sum(ac_valid_mask),
        sigma_base=results['sigma_base'],
        sigma_rel=results['sigma_rel'],
        # AC fields (new)
        P_re=P_re_dict,
        P_im=P_im_dict,
        P_re_std=P_re_std_dict,
        P_im_std=P_im_std_dict,
        P_amp=P_amp_dict,
        P_phase=P_phase_dict,
        predicted_Q_amp=predicted_Q_amp,
        predicted_Q_phase=predicted_Q_phase,
        measured_Q_amp=measured_Q_amp,
        measured_Q_phase=measured_Q_phase,
        r_squared_ac=r_squared_ac,
        rmse_amp=rmse_amp,
        phase_ref_node=ref_node,
    )
