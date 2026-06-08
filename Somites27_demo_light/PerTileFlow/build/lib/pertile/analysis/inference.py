"""Unified inference engine for transmission-line model parameters.

Replaces the parallel implementations in `_per_tile_alpha`,
`_per_tile_alpha_D_mle`, `_optimize_D_mu_BCs`, and the greyzone
optimizer with a single dispatcher driven by an `InferenceSpec`
parameter mask.

Status (2026-05): Step 1 scaffold.  The dataclasses, dispatcher, and
core helper signatures are in place.  Closed-form solver routines and
the `_per_tile_alpha_D_mle` migration land in Step 2 (Week 2).  Until
then the existing buttons in `viewer/mosaic/_simulation.py` continue to
work as before; nothing currently calls into this module.

Forward model (linear-after-reparam):

    Q_e(α, β_D, μ', τ) = α · b^(0)_e
                       + β_D · b^(D)_e            (β_D := α·(D − D₀))
                       + μ' · b^(μ)_e             (μ' := α·(1/μ − 1/μ₀))
                       (then rotate Q̂_meas at H_n by e^{−jnω₀τ})

For each parameter listed in InferenceSpec (`fit_alpha`, `fit_D`,
`fit_mu`, `fit_tau`), a sensitivity basis vector b^(p) is computed by
finite-differencing the forward solver:

    b^(p)_e = (Q_sim(p₀ + ε) − Q_sim(p₀)) / ε

where ε is set by `eps_<p>` on the spec.  τ is treated specially —
it doesn't perturb the solver but rotates Q̂_meas; see _solve_tau in
the closed-form router for the closed-form update.

DC observations have b^(D)_DC ≡ 0 (capacitive admittance vanishes at
n=0); we hard-zero it to remove finite-difference noise.

The `run_inference` dispatcher routes to one of:

    * _solve_closed_form_linear        — α / αD / αDμ (linear after reparam)
    * _solve_closed_form_with_tau      — adds bilinear τ-alternation
    * _solve_closed_form_hierarchical  — per-tile α + global pooling prior
    * _solve_profile_likelihood        — 1-D scan over a single nonlinear param
    * _solve_newton                    — Levenberg-Marquardt for general nonlinear

(Routing rules are implemented in `_pick_optimizer` below.)

The data layer (`_build_design_matrix`) handles:

    * τ alignment of Q̂_meas
    * sign_uv flip from PIV's positive-flow convention to (u,v) tuples
    * Q(t) refit fallback when harmonic lists were dropped
    * harmonic stacking (DC, H1, optionally H2/H3)

Synthetic-recovery mode bypasses real measurements: `run_inference`
samples ground-truth params from the spec, runs the forward sim, adds
noise per the per-tile noise model, then recovers and reports bias.

References
----------
Reviewer feedback (2026-05) and design discussion (Step A/B/C/D/E/F):
see project memory `memory/inference_refactor_plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Parameter registry (extensibility foundation)
# ──────────────────────────────────────────────────────────────────
#
# To add a new inference parameter (e.g. peripheral resistance R,
# a per-edge ζ greyzone share, a global heart-rate offset), define a
# `ParameterDef` and register it via `register_parameter(...)`.  The
# rest of the engine never branches on parameter name — it iterates
# the registry, builds basis vectors via each definition's
# `build_basis` callback, stacks them into a design matrix, and the
# closed-form/Newton solvers handle them uniformly.
#
# Each parameter is one of:
#   • LINEAR      — Q depends linearly on the param (after reparam).
#                    The param contributes one basis column built by
#                    finite-differencing the forward solver.  Closed
#                    form: column appears in the k×k WLS.
#   • PHASE       — Param enters as e^{-jnω₀·param} on Q̂_meas (a
#                    rotation, not a basis column).  Closed form via
#                    cross-correlation over included harmonics.  Used
#                    for τ (cardiac-phase offset).
#   • PER_EDGE    — Param is a vector with one entry per edge (e.g.
#                    ζ_e greyzone shares).  Routes to Newton.
#   • NONLINEAR   — Param has nonlinear dependence the linearization
#                    can't capture (e.g. D in cosh/sinh admittance for
#                    large κL).  Routes to profile or Newton.

ParamKind = Literal['LINEAR', 'PHASE', 'PER_EDGE', 'NONLINEAR']


@dataclass
class ParameterDef:
    """Plug-in description of one inference parameter.

    Attributes
    ----------
    name
        Short identifier used as dict key in `InferenceResult.params`.
    kind
        One of LINEAR / PHASE / PER_EDGE / NONLINEAR (see module docstring).
    spec_flag
        Name of the boolean attribute on `InferenceSpec` that toggles
        this parameter on (e.g. ``'fit_alpha'``, ``'fit_D'``).
    eps_attr
        For LINEAR/NONLINEAR: name of the float attribute on
        `InferenceSpec` that gives the finite-difference step (e.g.
        ``'eps_D'``).  None for parameters that don't perturb the solver
        (PHASE) or have no scalar perturbation (PER_EDGE).
    sim_state_key
        For LINEAR/NONLINEAR: viewer-state attribute the forward solver
        reads (e.g. ``'_sim_D'``).  Used by the design-matrix builder
        to set up the perturbation runs.
    default_value
        Anchor point p₀ for the linearization.  None ⇒ read from
        viewer state at fit time.
    reparam_label
        Optional tex/string for the reparameterized regression
        coefficient (e.g. for D the regression solves for β = α(D−D₀)
        and recovers D as D₀ + β/α).  Pure documentation — used only
        by reporting code.
    recover
        Callable ``recover(theta_dict, p0_dict) → param_value`` that
        converts the reparameterized regression coefficients back to
        the natural parameter.  For α this is identity; for D it is
        ``p0_dict['D'] + theta_dict['beta_D'] / theta_dict['alpha']``.
        None ⇒ engine treats the regression coefficient as the
        parameter directly (e.g. α).
    propagate_sigma
        Callable ``propagate_sigma(theta_dict, sigma_dict, cov, p0_dict)
        → float`` that returns σ for the recovered parameter via the
        δ-method.  Mirrors `recover`.
    build_basis
        Callable executed during `_build_design_matrix` to compute the
        per-row complex basis column for this parameter.  Signature:
        ``build_basis(graph, edge, ref_meas, sim_phasors_at_p0,
                       sim_phasors_at_p1, eps, harmonic) → complex``.
        Engine provides the two snapshots; the callable just picks
        the right entries.  None ⇒ engine uses the default
        finite-difference rule.
    """

    name: str
    kind: ParamKind
    spec_flag: str
    eps_attr: Optional[str] = None
    sim_state_key: Optional[str] = None
    default_value: Optional[float] = None
    reparam_label: Optional[str] = None
    recover: Optional[Callable[..., float]] = None
    propagate_sigma: Optional[Callable[..., float]] = None
    build_basis: Optional[Callable[..., complex]] = None


_REGISTRY: dict[str, ParameterDef] = {}


def register_parameter(p: ParameterDef) -> None:
    """Register a new inference parameter.  Idempotent on `name`."""
    _REGISTRY[p.name] = p


def get_parameter(name: str) -> ParameterDef:
    return _REGISTRY[name]


def all_parameters() -> list[ParameterDef]:
    return list(_REGISTRY.values())


def active_parameters(spec: 'InferenceSpec') -> list[ParameterDef]:
    """Return the parameters this spec asks to fit, in registration order."""
    out = []
    for p in _REGISTRY.values():
        if getattr(spec, p.spec_flag, False):
            out.append(p)
    return out


# ── Built-in parameter definitions ──────────────────────────────────
#
# These wire the existing α / D / τ / μ / ζ machinery into the
# registry.  To add a new parameter (e.g. peripheral R), call
# register_parameter(ParameterDef(name='R', kind='LINEAR', ...)) at
# the bottom of this module or from any caller before run_inference.


def _recover_alpha(theta, p0):
    return float(theta['alpha'])


def _recover_D(theta, p0):
    alpha = float(theta['alpha'])
    if abs(alpha) < 1e-30:
        return float('nan')
    return float(p0['D']) + float(theta['beta_D']) / alpha


def _recover_mu(theta, p0):
    """μ enters resistively as Q ∝ 1/μ, so we solve for β_μ = α·(1/μ − 1/μ₀)
    in the linear regression and recover μ as 1 / (1/μ₀ + β_μ/α)."""
    alpha = float(theta['alpha'])
    if abs(alpha) < 1e-30:
        return float('nan')
    inv_mu0 = 1.0 / float(p0['mu'])
    inv_mu = inv_mu0 + float(theta['beta_mu']) / alpha
    return 1.0 / inv_mu if abs(inv_mu) > 1e-30 else float('nan')


def _propagate_sigma_alpha(theta, sigma, cov, p0):
    return float(sigma['alpha'])


def _propagate_sigma_D(theta, sigma, cov, p0):
    """δ-method: D = D₀ + β_D/α
        Var(D) = Var(β)/α² + (β²/α⁴)·Var(α) − 2·(β/α³)·Cov(α,β)
    """
    alpha = float(theta['alpha'])
    beta = float(theta['beta_D'])
    if abs(alpha) < 1e-30:
        return float('nan')
    var_alpha = float(sigma['alpha']) ** 2
    var_beta = float(sigma['beta_D']) ** 2
    cov_ab = float(cov.get(('alpha', 'beta_D'), 0.0))
    a2 = alpha ** 2
    a4 = a2 ** 2
    var_D = (var_beta / a2
             + (beta ** 2 / a4) * var_alpha
             - 2.0 * (beta / (alpha ** 3)) * cov_ab)
    return float(np.sqrt(max(var_D, 0.0)))


def _propagate_sigma_mu(theta, sigma, cov, p0):
    """δ-method: μ = 1 / (1/μ₀ + β_μ/α)
       Let g = 1/μ₀ + β_μ/α, then μ = 1/g, ∂μ/∂α = β_μ/(α²·g²),
       ∂μ/∂β_μ = −1/(α·g²)."""
    alpha = float(theta['alpha'])
    beta = float(theta['beta_mu'])
    inv_mu0 = 1.0 / float(p0['mu'])
    g = inv_mu0 + beta / alpha if abs(alpha) > 1e-30 else float('nan')
    if not np.isfinite(g) or abs(g) < 1e-30 or abs(alpha) < 1e-30:
        return float('nan')
    var_alpha = float(sigma['alpha']) ** 2
    var_beta = float(sigma['beta_mu']) ** 2
    cov_ab = float(cov.get(('alpha', 'beta_mu'), 0.0))
    da = beta / (alpha ** 2 * g ** 2)
    db = -1.0 / (alpha * g ** 2)
    var_mu = (da ** 2) * var_alpha + (db ** 2) * var_beta + 2 * da * db * cov_ab
    return float(np.sqrt(max(var_mu, 0.0)))


# Register built-ins.  Order matters for column layout in M:
#   alpha (col 0), beta_D, beta_mu, ...
register_parameter(ParameterDef(
    name='alpha', kind='LINEAR', spec_flag='fit_alpha',
    default_value=1.0, reparam_label='α',
    recover=_recover_alpha,
    propagate_sigma=_propagate_sigma_alpha,
))
register_parameter(ParameterDef(
    name='D', kind='LINEAR', spec_flag='fit_D',
    eps_attr='eps_D', sim_state_key='_sim_D',
    reparam_label='β_D = α·(D − D₀)',
    recover=_recover_D,
    propagate_sigma=_propagate_sigma_D,
))
register_parameter(ParameterDef(
    name='mu', kind='LINEAR', spec_flag='fit_mu',
    eps_attr='eps_mu', sim_state_key='_sim_mu',
    reparam_label='β_μ = α·(1/μ − 1/μ₀)',
    recover=_recover_mu,
    propagate_sigma=_propagate_sigma_mu,
))
register_parameter(ParameterDef(
    name='tau', kind='PHASE', spec_flag='fit_tau',
    default_value=0.0, reparam_label='τ (per-tile cardiac offset)',
))
register_parameter(ParameterDef(
    name='zeta', kind='PER_EDGE', spec_flag='fit_zeta',
    reparam_label='ζ_e (greyzone share)',
))


# ──────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────


@dataclass
class InferenceSpec:
    """Declarative description of a single inference run.

    Every existing per-tile / global / greyzone fit reduces to a
    particular setting of this struct.  Examples:

        # Current α-only per-tile fit (greyzone L2)
        InferenceSpec(fit_alpha=True, scope='per_tile',
                       use_dc=True, harmonics=())

        # Current (α, D, τ) MLE — what `_per_tile_alpha_D_mle` does
        InferenceSpec(fit_alpha=True, fit_D=True, fit_tau=True,
                       scope='single_tile', focus_tile=26,
                       harmonics=(1,))

        # Joint global D inference across all tiles, no τ
        InferenceSpec(fit_alpha=True, fit_D=True,
                       scope='global', harmonics=(1,))

        # Hierarchical α with global pooling prior, fit D too
        InferenceSpec(fit_alpha=True, fit_D=True, fit_tau=True,
                       scope='hierarchical', pool_alpha=True,
                       harmonics=(1,))
    """

    # ── Parameter mask ──
    fit_alpha: bool = True
    fit_D: bool = False
    fit_tau: bool = False           # per-tile cardiac-phase nuisance
    fit_mu: bool = False
    fit_zeta: bool = False          # per-edge greyzone fractions

    # ── Forward-sim sensitivity steps ──
    eps_D: float = 0.10             # relative; D₁ = D₀·(1 + eps_D)
    eps_mu: float = 0.05

    # ── Data scope ──
    scope: Literal['single_tile', 'per_tile', 'global',
                    'hierarchical', 'synthetic_recovery'] = 'single_tile'
    focus_tile: Optional[int] = None
    pool_alpha: bool = False
    pool_alpha_sigma: Optional[float] = None  # None ⇒ infer via empirical Bayes

    # ── Noise model ──
    noise: Literal['heteroscedastic', 'homoscedastic',
                    'identity'] = 'heteroscedastic'
    noise_form: Literal['linear', 'quadrature', 'powerlaw',
                         'variance_linear'] = 'variance_linear'

    # ── Optimizer ──
    optimizer: Literal['closed_form', 'profile', 'newton',
                        'auto'] = 'auto'

    # ── Harmonics included ──
    use_dc: bool = True
    harmonics: Sequence[int] = (1,)

    # ── τ warm-starts ──
    tau_warm_starts: Sequence[float] = (0.0, np.pi)
    n_alt_iter: int = 3
    alt_tol: float = 1e-5

    # ── Outer FGLS iteration (refit noise model from latest residuals,
    #     then re-solve for the linear params + τ).  Each outer step
    #     wraps the bilinear (θ, τ) closed-form alternation.  Default
    #     2 — empirically enough for noise-form='variance_linear'.
    n_outer_iter: int = 2
    outer_tol: float = 1e-3      # relative change in θ to declare converged

    # ── Bounded-influence cap on the per-row σ_e (robust FGLS).
    # Without this, when Stage 1 misfits the data, |r|² ≈ |Q|² and the
    # noise model fits b ≈ 1 in σ² = a + b|Q|² → weights ∝ 1/|Q|² →
    # high-|Q| points crushed → WLS migrates toward α=0.  Capping the
    # weight dynamic range stops the runaway.  Set to None to disable.
    # Cap = max(σ_e) / min(σ_e) ≤ sigma_dynamic_range_cap.
    sigma_dynamic_range_cap: Optional[float] = 10.0

    # ── Synthetic-recovery params ──
    synth_truth: Optional[dict] = None       # {'alpha': ..., 'D': ..., 'tau': ...}
    synth_seed: Optional[int] = None

    # ── Output knobs ──
    save_figure: bool = True
    save_to_graph: bool = True
    verbose: bool = True


@dataclass
class InferenceResult:
    """Result of a single `run_inference` call.

    `params` and `sigma` are dicts keyed by parameter name (subset of
    {'alpha', 'D', 'tau', 'mu', 'zeta', 'beta'}).  For per-tile or
    hierarchical fits, values are np.ndarrays indexed by tile.

    `noise_model` carries the per-tile, per-harmonic σ(|Q|) coefficients
    used during the fit:

        noise_model[tile_id] = {
            'a_dc': ..., 'b_dc': ...,
            'a_h1': ..., 'b_h1': ...,
            'form': 'linear' | 'quadrature' | 'powerlaw',
        }

    `convergence` carries iteration counts, residual norms, and
    warm-start selection (which τ basin won, etc).

    `design_matrix` carries the (Qm, basis, harmonic_idx, sigma_e,
    Qm_rotated) needed to plot diagnostics without re-running the
    forward solver.  Only populated when `spec.save_figure` is True
    or the caller asks for it explicitly.
    """

    params: dict
    sigma: dict
    cov: Optional[np.ndarray]
    chi2_red: float
    n_obs_complex: int
    n_obs_real: int
    n_params_fit: int
    dof: int
    noise_model: dict
    convergence: dict

    # Diagnostic data — keep with the result so callers don't have to
    # rebuild the design matrix to plot.
    design_matrix: Optional['DesignMatrix'] = None
    sigma_e: Optional[np.ndarray] = None        # (M,) per-row σ used in WLS
    Qm_rotated: Optional[np.ndarray] = None     # τ-aligned measured phasors
    residuals: Optional[np.ndarray] = None      # complex (M,) Qm_rot − pred

    # Two-stage GLS diagnostic data (populated by closed-form solvers
    # so plotters can show Stage 1 → noise fit → Stage 2 progression):
    stage1_theta: Optional[dict] = None         # {coef_name: float} unweighted
    stage1_residuals: Optional[np.ndarray] = None   # complex (M,) from Stage 1
    stage1_Qm_rotated: Optional[np.ndarray] = None  # Stage-1 τ-aligned Qm

    # Per-tile diagnostics (when scope='per_tile' or 'hierarchical')
    per_tile: Optional[dict] = None

    # Synthetic-recovery diagnostics (when scope='synthetic_recovery')
    synth_truth: Optional[dict] = None
    synth_bias: Optional[dict] = None         # truth − estimate
    synth_z: Optional[dict] = None             # (truth − estimate) / σ


# ──────────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────────


def run_inference(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[[float, float], None],
    *,
    sim_state_get: Callable[[], dict] | None = None,
    sim_state_set: Callable[[dict], None] | None = None,
) -> InferenceResult:
    """Top-level entry point.

    Parameters
    ----------
    graph
        NetworkX graph carrying ``measurements_piv`` per edge and
        ``mean_Q_sim`` / ``amp_Q_sim`` / ``phase_sim`` after a sim run.
    spec
        See :class:`InferenceSpec`.
    forward_solver
        Callable ``f(D, mu)`` that runs the transmission-line solve at
        the supplied parameters and writes ``mean_Q_sim``, ``amp_Q_sim``,
        ``phase_sim`` onto the graph edges.  In the mosaic viewer this
        is a thin wrapper around ``_run_transmission_line``.
    sim_state_get / sim_state_set
        Optional callbacks for snapshotting/restoring parameters that
        live on the viewer (e.g. ``self._sim_D``) so the inference does
        not pollute viewer state.

    Returns
    -------
    InferenceResult
    """
    if spec.scope == 'synthetic_recovery':
        return _run_synthetic_recovery(graph, spec, forward_solver,
                                        sim_state_get=sim_state_get,
                                        sim_state_set=sim_state_set)

    optimizer = _pick_optimizer(spec)
    if optimizer == 'closed_form_linear':
        return _solve_closed_form_linear(graph, spec, forward_solver,
                                          sim_state_get=sim_state_get,
                                          sim_state_set=sim_state_set)
    if optimizer == 'closed_form_with_tau':
        return _solve_closed_form_with_tau(graph, spec, forward_solver,
                                            sim_state_get=sim_state_get,
                                            sim_state_set=sim_state_set)
    if optimizer == 'closed_form_hierarchical':
        return _solve_closed_form_hierarchical(
            graph, spec, forward_solver,
            sim_state_get=sim_state_get,
            sim_state_set=sim_state_set,
        )
    if optimizer == 'profile':
        return _solve_profile_likelihood(graph, spec, forward_solver,
                                           sim_state_get=sim_state_get,
                                           sim_state_set=sim_state_set)
    if optimizer == 'newton':
        return _solve_newton(graph, spec, forward_solver,
                              sim_state_get=sim_state_get,
                              sim_state_set=sim_state_set)
    raise NotImplementedError(f'optimizer route {optimizer!r}')


# ──────────────────────────────────────────────────────────────────
# Internal: optimizer routing
# ──────────────────────────────────────────────────────────────────


def _pick_optimizer(spec: InferenceSpec) -> str:
    """Choose an optimizer route based on the parameter mask + scope.

    Rules:
      * ``optimizer='closed_form'`` is honored only when the joint
        problem actually has a closed form.  Otherwise raises.
      * ``optimizer='auto'`` picks the cheapest optimizer that can
        handle the spec:
          * α only / αD / αDμ + no τ + no ζ ⇒ closed_form_linear
          * + τ                                ⇒ closed_form_with_tau
          * scope=hierarchical                ⇒ closed_form_hierarchical
          * fit_zeta=True                      ⇒ newton
          * any scope with no closed form     ⇒ newton
    """
    if spec.scope == 'hierarchical':
        if spec.fit_zeta:
            return 'newton'
        return 'closed_form_hierarchical'

    nonlinear_params = spec.fit_zeta
    if nonlinear_params:
        return 'newton'

    if spec.optimizer == 'profile':
        return 'profile'
    if spec.optimizer == 'newton':
        return 'newton'

    if spec.fit_tau:
        return 'closed_form_with_tau'
    return 'closed_form_linear'


# ──────────────────────────────────────────────────────────────────
# Internal: design-matrix construction (shared)
# ──────────────────────────────────────────────────────────────────


def _snapshot_phasors(graph, harmonics: Sequence[int]) -> dict:
    """Capture per-edge sim phasors for DC + each requested harmonic.

    Convention fix (mirrors the mosaic viewer): the TL solver's
    ``_extract_boundary_harmonics`` negates Q at sink BCs, propagating a
    systematic π offset on every interior phasor relative to the
    (u,v)-tuple measured phasor.  We undo this by reading ``phase_sim``
    as a lag (``e^{-jωt}``) rather than an advance (``e^{+jωt}``).

    Returns
    -------
    snapshot : dict
        snapshot[edge] = {0: Q_dc_real, 1: Q_h1_complex, 2: Q_h2_complex, ...}

    NOTE: only DC and H1 are populated by the current TL solver
    (it stores `amp_Q_sim`/`phase_sim` for the dominant harmonic only).
    H2/H3 will need the solver to expose per-harmonic amp/phase before
    multi-harmonic inference works — flagged as a known limitation.
    """
    out = {}
    for u, v, d in graph.edges(data=True):
        Q_dc = d.get('mean_Q_sim')
        if Q_dc is None:
            continue
        per_n = {0: float(Q_dc) if np.isfinite(Q_dc) else float('nan')}
        for n in harmonics:
            if n == 1:
                amp = d.get('amp_Q_sim')
                phase = d.get('phase_sim')
                if (amp is not None and phase is not None
                        and np.isfinite(amp) and np.isfinite(phase)):
                    # exp(-j·phase) lag convention — see docstring.
                    per_n[n] = (complex(float(amp))
                                * np.exp(-1j * float(phase)))
                else:
                    per_n[n] = complex(float('nan'))
            else:
                # H2/H3 not yet exposed per-edge by the solver.
                per_n[n] = complex(float('nan'))
        out[(u, v)] = per_n
    return out


def _meas_phasors_for_edge(
    edge: tuple,
    m_ref: dict,
    harmonics: Sequence[int],
) -> tuple[float, dict[int, complex], float]:
    """Recover per-edge complex measured phasors in (u,v)-tuple convention.

    Returns
    -------
    Q_meas_dc : float          (signed in (u,v) convention; NaN if missing)
    Q_meas_hn : dict[int, complex]    keyed by n ≥ 1
    sign_uv   : float          +1 or -1
    """
    u, v = edge
    ff = m_ref.get('flow_from')
    ft = m_ref.get('flow_to')
    if ff == u and ft == v:
        sign_uv = +1.0
    elif ff == v and ft == u:
        sign_uv = -1.0
    else:
        sign_uv = +1.0  # caller can re-derive from sim agreement if needed

    Q_dc = m_ref.get('mean_Q', np.nan)
    Q_dc_uv = float(Q_dc) * sign_uv if np.isfinite(Q_dc) else float('nan')

    out_h = {}
    for n in harmonics:
        if n == 0:
            continue
        # 1) harmonics list direct
        A = B = None
        harm_list = m_ref.get('harmonics', []) or []
        if harm_list:
            kn = next((h for h in harm_list if h.get('k') == n), None)
            if kn is not None:
                A = kn.get('A')
                B = kn.get('B')
                if A is None or B is None or not (np.isfinite(A)
                                                    and np.isfinite(B)):
                    amp_m = kn.get('amp')
                    phi_m = kn.get('phi')
                    if (amp_m is not None and phi_m is not None
                            and np.isfinite(amp_m)
                            and np.isfinite(phi_m)):
                        A = float(amp_m) * np.cos(float(phi_m))
                        B = -float(amp_m) * np.sin(float(phi_m))
        # 2) Q(t) refit fallback
        if A is None or B is None or not (np.isfinite(A)
                                            and np.isfinite(B)):
            Qt = m_ref.get('Q_t')
            if Qt is None:
                Qt = m_ref.get('Q_t_plug')
            f0 = m_ref.get('f0_hz') or m_ref.get('f0')
            if (Qt is not None and f0 is not None and float(f0) > 0):
                try:
                    from .harmonic import fit_harmonics
                    from .config import FRAME_DT_S
                    Qt_arr = np.asarray(Qt, dtype=float)
                    if Qt_arr.size >= 16 and np.isfinite(Qt_arr).any():
                        hr = fit_harmonics(Qt_arr, FRAME_DT_S,
                                            float(f0),
                                            K=max(harmonics),
                                            include_dc=True)
                        hl = hr.get('harmonics', []) or []
                        knr = next((h for h in hl
                                     if h.get('k') == n), None)
                        if knr is not None:
                            Ar = knr.get('A')
                            Br = knr.get('B')
                            if (Ar is not None and Br is not None
                                    and np.isfinite(Ar)
                                    and np.isfinite(Br)):
                                A, B = float(Ar), float(Br)
                except Exception:
                    pass
        # 3) top-level amp_Q + phase fallback (for older PIV records
        # that only stored H1)
        if (A is None or B is None or not (np.isfinite(A)
                                            and np.isfinite(B))) and n == 1:
            amp_top = m_ref.get('amp_Q')
            phase_top = m_ref.get('phase')   # degrees
            if (amp_top is not None and phase_top is not None
                    and np.isfinite(amp_top)
                    and np.isfinite(phase_top)):
                phi_rad = np.radians(float(phase_top))
                A = float(amp_top) * np.cos(phi_rad)
                B = -float(amp_top) * np.sin(phi_rad)
        if (A is not None and B is not None
                and np.isfinite(A) and np.isfinite(B)):
            out_h[n] = sign_uv * complex(float(A), -float(B))
        else:
            out_h[n] = complex(float('nan'))

    return Q_dc_uv, out_h, sign_uv


@dataclass
class DesignMatrix:
    """Output of `_build_design_matrix`.

    Attributes
    ----------
    Qm
        Complex (M,) — measured phasors in (u,v) convention.
    basis
        Dict mapping regression-coefficient name → (M,) complex column.
        Keys come from each ParameterDef's reparam slot:
          - 'alpha'   for α
          - 'beta_D'  for D (β_D = α·(D-D₀))
          - 'beta_mu' for μ (β_μ = α·(1/μ-1/μ₀))
    harmonic_idx
        (M,) int — 0=DC, n=Hn.  Used by the τ-rotation phasor and the
        per-harmonic noise model.
    edge_keys
        List of (u, v) per row.
    tile_ids
        (M,) int — tile id per row (useful for per-tile/hierarchical scopes).
    p0
        Anchor parameter values around which we linearized
        (e.g. {'D': 1e-4, 'mu': 2.5e-3}).
    f0_tile
        Median per-edge f₀ for the tile; defines ω₀ for τ rotation.
    n_dc, n_h1, ...
        Counts per harmonic for diagnostics.
    """
    Qm: np.ndarray
    basis: dict[str, np.ndarray]
    harmonic_idx: np.ndarray
    edge_keys: list
    tile_ids: np.ndarray
    p0: dict[str, float]
    f0_tile: float
    n_per_harmonic: dict[int, int]


def _build_design_matrix(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[..., None],
    *,
    sim_state_get: Optional[Callable[[], dict]] = None,
    sim_state_set: Optional[Callable[[dict], None]] = None,
    cached_snapshots: Optional[dict] = None,
) -> DesignMatrix:
    """Run forward sim at each LINEAR-param's perturbation point and
    assemble the complex (M, K) design matrix.

    Adding a new LINEAR parameter only requires (a) registering its
    `ParameterDef` with `eps_attr` and `sim_state_key`, and (b) the
    forward_solver accepting the corresponding kwarg.  The builder
    discovers everything else from the registry — there are no
    parameter-specific branches in this function.
    """
    if forward_solver is None:
        raise ValueError('forward_solver required to build design matrix')

    active = active_parameters(spec)
    linear_params = [p for p in active if p.kind == 'LINEAR']
    # α is always implicitly fit (it's the linear scale on b^(0)).  If
    # the spec has only β_D etc. without α, we still need b^(0).
    have_alpha = any(p.name == 'alpha' for p in linear_params)
    if not have_alpha:
        # Guard: a non-α LINEAR fit needs α as the global scale; auto-add.
        linear_params = [get_parameter('alpha')] + linear_params

    # Snapshot baseline viewer state so we can restore.
    state_orig = sim_state_get() if sim_state_get is not None else {}

    # Anchor values p0 for each LINEAR param: prefer current viewer
    # state (sim_state_get), else ParameterDef.default_value.
    p0: dict[str, float] = {}
    for p in linear_params:
        if p.name == 'alpha':
            p0['alpha'] = 1.0   # α anchors at 1.0 (identity scale)
            continue
        if p.sim_state_key and p.sim_state_key in state_orig:
            p0[p.name] = float(state_orig[p.sim_state_key])
        elif p.default_value is not None:
            p0[p.name] = float(p.default_value)
        else:
            raise ValueError(f'cannot anchor parameter {p.name!r}: '
                              f'no sim_state_key value and no default')

    # Run the forward solver at the baseline (run A) and at each
    # perturbation (run B per perturbed parameter).
    snapshot_at_p0 = None
    snapshots_at_p1: dict[str, dict] = {}
    eps_actual: dict[str, float] = {}

    if cached_snapshots is not None:
        # Reuse pre-computed snapshots (multi-tile sweep optimization —
        # the same forward sims serve every tile).  Format:
        #   {'p0': snapshot_dict,
        #    'p1': {param_name: snapshot_dict},
        #    'eps_actual': {param_name: float}}
        snapshot_at_p0 = cached_snapshots['p0']
        snapshots_at_p1 = dict(cached_snapshots.get('p1', {}))
        eps_actual = dict(cached_snapshots.get('eps_actual', {}))
    else:
        # Run A
        if sim_state_set is not None and state_orig:
            sim_state_set(state_orig)
        forward_solver()
        snapshot_at_p0 = _snapshot_phasors(graph, spec.harmonics)

        # Per-LINEAR-param run at p0 + ε·p0
        for p in linear_params:
            if p.name == 'alpha':
                continue   # α doesn't perturb the solver — pure scale.
            if not p.eps_attr or not p.sim_state_key:
                continue
            eps_rel = float(getattr(spec, p.eps_attr))
            p1_val = p0[p.name] * (1.0 + eps_rel)
            eps_actual[p.name] = p1_val - p0[p.name]
            if sim_state_set is not None:
                new_state = dict(state_orig)
                new_state[p.sim_state_key] = p1_val
                sim_state_set(new_state)
            forward_solver()
            snapshots_at_p1[p.name] = _snapshot_phasors(
                graph, spec.harmonics)

        # Restore baseline state
        if sim_state_set is not None and state_orig:
            sim_state_set(state_orig)

    # ── Walk edges, extract measurements, assemble rows ──
    Qm_rows: list[complex] = []
    basis_rows: dict[str, list[complex]] = {}
    # Reparam name per LINEAR param: α → 'alpha', D → 'beta_D', μ → 'beta_mu'
    coef_names = {p.name: ('alpha' if p.name == 'alpha'
                            else f'beta_{p.name}')
                   for p in linear_params}
    for cn in coef_names.values():
        basis_rows[cn] = []
    harmonic_idx: list[int] = []
    edge_keys: list = []
    tile_ids_list: list[int] = []
    f0_seen: list[float] = []

    ref_vid = spec.focus_tile
    if spec.scope == 'single_tile' and ref_vid is None:
        raise ValueError("scope='single_tile' requires spec.focus_tile")

    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv', []) or []
        # Iterate matching tile measurements.  For single_tile, only ref_vid;
        # for per_tile/global, all tiles.
        for m_ref in piv:
            tile_id = m_ref.get('tile_id')
            if tile_id is None:
                continue
            if (spec.scope == 'single_tile'
                    and int(tile_id) != int(ref_vid)):
                continue
            f0_e = m_ref.get('f0_hz') or m_ref.get('f0')
            if f0_e and float(f0_e) > 0:
                f0_seen.append(float(f0_e))

            sim0 = snapshot_at_p0.get((u, v))
            if sim0 is None:
                continue
            Q_dc_meas, Q_hn_meas, sign_uv = _meas_phasors_for_edge(
                (u, v), m_ref, spec.harmonics)

            # ── DC row ──
            if spec.use_dc and np.isfinite(Q_dc_meas) and abs(Q_dc_meas) > 1e-30:
                qA_dc = sim0.get(0, np.nan)
                if np.isfinite(qA_dc):
                    Qm_rows.append(complex(Q_dc_meas, 0.0))
                    for p in linear_params:
                        cn = coef_names[p.name]
                        if p.name == 'alpha':
                            basis_rows[cn].append(complex(float(qA_dc), 0.0))
                        else:
                            # D, μ both have b^(p)_DC ≡ 0 at DC
                            # (capacitive admittance + resistive scaling
                            # cancel; exactly zero for both via
                            # construction — hard-zero to remove FD noise).
                            basis_rows[cn].append(complex(0.0, 0.0))
                    harmonic_idx.append(0)
                    edge_keys.append((u, v))
                    tile_ids_list.append(int(tile_id))

            # ── Hn rows ──
            for n in spec.harmonics:
                if n == 0:
                    continue
                Qmn = Q_hn_meas.get(n)
                if Qmn is None or not np.isfinite(Qmn):
                    continue
                if abs(Qmn) <= 1e-30:
                    continue
                qA_hn = sim0.get(n, complex(np.nan))
                if not np.isfinite(qA_hn) or abs(qA_hn) < 1e-30:
                    continue
                Qm_rows.append(complex(Qmn))
                for p in linear_params:
                    cn = coef_names[p.name]
                    if p.name == 'alpha':
                        basis_rows[cn].append(complex(qA_hn))
                    else:
                        snap_p1 = snapshots_at_p1.get(p.name, {})
                        sim1 = snap_p1.get((u, v))
                        qB_hn = sim1.get(n, complex(np.nan)) \
                            if sim1 is not None else complex(np.nan)
                        eps_p = eps_actual.get(p.name, 1.0)
                        if (np.isfinite(qB_hn) and np.isfinite(qA_hn)
                                and abs(eps_p) > 1e-30):
                            basis_rows[cn].append(
                                (complex(qB_hn) - complex(qA_hn)) / eps_p)
                        else:
                            basis_rows[cn].append(complex(np.nan))
                harmonic_idx.append(n)
                edge_keys.append((u, v))
                tile_ids_list.append(int(tile_id))

    Qm_arr = np.asarray(Qm_rows, dtype=np.complex128)
    basis_arr = {cn: np.asarray(col, dtype=np.complex128)
                 for cn, col in basis_rows.items()}
    harmonic_idx_arr = np.asarray(harmonic_idx, dtype=int)
    tile_ids_arr = np.asarray(tile_ids_list, dtype=int)

    # Drop rows with any NaN in Qm or any basis column
    valid = np.isfinite(Qm_arr.real) & np.isfinite(Qm_arr.imag)
    for col in basis_arr.values():
        valid &= np.isfinite(col.real) & np.isfinite(col.imag)
    if not valid.all():
        Qm_arr = Qm_arr[valid]
        for cn in basis_arr:
            basis_arr[cn] = basis_arr[cn][valid]
        harmonic_idx_arr = harmonic_idx_arr[valid]
        edge_keys = [edge_keys[i] for i, ok in enumerate(valid) if ok]
        tile_ids_arr = tile_ids_arr[valid]

    # Per-harmonic counts
    n_per_h = {int(n): int(np.sum(harmonic_idx_arr == int(n)))
               for n in [0] + list(spec.harmonics)}

    f0_tile = float(np.median(f0_seen)) if f0_seen else 2.5
    return DesignMatrix(
        Qm=Qm_arr,
        basis=basis_arr,
        harmonic_idx=harmonic_idx_arr,
        edge_keys=edge_keys,
        tile_ids=tile_ids_arr,
        p0=p0,
        f0_tile=f0_tile,
        n_per_harmonic=n_per_h,
    )


# ──────────────────────────────────────────────────────────────────
# Internal: closed-form linear solver (no τ)
# ──────────────────────────────────────────────────────────────────


def precompute_snapshots(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[..., None],
    *,
    sim_state_get: Optional[Callable[[], dict]] = None,
    sim_state_set: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run the forward solver once at p0 and once per perturbed LINEAR
    param, snapshot per-edge complex phasors, and return a
    `cached_snapshots` dict consumable by `_build_design_matrix`.

    For a multi-tile sweep the forward sims are tile-independent, so
    calling this once and reusing the result across N tiles cuts solver
    runs from 2N to 2.
    """
    state_orig = sim_state_get() if sim_state_get is not None else {}
    active = active_parameters(spec)
    linear_params = [p for p in active if p.kind == 'LINEAR']
    if not any(p.name == 'alpha' for p in linear_params):
        linear_params = [get_parameter('alpha')] + linear_params

    p0: dict[str, float] = {}
    for p in linear_params:
        if p.name == 'alpha':
            p0['alpha'] = 1.0
            continue
        if p.sim_state_key and p.sim_state_key in state_orig:
            p0[p.name] = float(state_orig[p.sim_state_key])
        elif p.default_value is not None:
            p0[p.name] = float(p.default_value)
        else:
            raise ValueError(f'cannot anchor {p.name!r}: '
                              'no sim_state_key value and no default')

    # Run A
    if sim_state_set is not None and state_orig:
        sim_state_set(state_orig)
    forward_solver()
    snap_p0 = _snapshot_phasors(graph, spec.harmonics)

    # Run B per perturbed param
    snaps_p1: dict[str, dict] = {}
    eps_actual: dict[str, float] = {}
    for p in linear_params:
        if p.name == 'alpha':
            continue
        if not p.eps_attr or not p.sim_state_key:
            continue
        eps_rel = float(getattr(spec, p.eps_attr))
        p1_val = p0[p.name] * (1.0 + eps_rel)
        eps_actual[p.name] = p1_val - p0[p.name]
        if sim_state_set is not None:
            new_state = dict(state_orig)
            new_state[p.sim_state_key] = p1_val
            sim_state_set(new_state)
        forward_solver()
        snaps_p1[p.name] = _snapshot_phasors(graph, spec.harmonics)

    if sim_state_set is not None and state_orig:
        sim_state_set(state_orig)

    return {'p0': snap_p0, 'p1': snaps_p1, 'eps_actual': eps_actual,
            'p0_values': p0}


def run_inference_multi_tile(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[..., None],
    *,
    tiles: Optional[Sequence[int]] = None,
    sim_state_get: Optional[Callable[[], dict]] = None,
    sim_state_set: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run the per-tile (α, D, τ) MLE on every tile and aggregate.

    Optimization: forward solver runs once at p0 and once per perturbed
    LINEAR param (typically D₁), then snapshots are reused across all
    tiles.  Total solver runs = 2 instead of 2N.

    Returns
    -------
    dict with keys:
      'tiles'          : list[int]    tile ids that succeeded
      'results'        : dict[int, InferenceResult]
      'failures'       : dict[int, str]    tile_id → error message
      'snapshots'      : dict          (the cached snapshots, for reuse)
    """
    # Discover tiles if not given
    if tiles is None:
        seen = set()
        for _, _, d in graph.edges(data=True):
            for m in d.get('measurements_piv', []) or []:
                tid = m.get('tile_id')
                if tid is not None and int(tid) > 0:
                    seen.add(int(tid))
        tiles = sorted(seen)
    tiles = list(tiles)

    if spec.verbose:
        print(f'  [multi-tile] precomputing forward-sim snapshots '
              f'(2 solver runs, will be reused across {len(tiles)} tiles)...')
    snaps = precompute_snapshots(graph, spec, forward_solver,
                                    sim_state_get=sim_state_get,
                                    sim_state_set=sim_state_set)

    results: dict[int, 'InferenceResult'] = {}
    failures: dict[int, str] = {}

    for tid in tiles:
        spec_t = _replace_spec(
            spec,
            scope='single_tile',
            focus_tile=int(tid),
            verbose=False,            # keep per-tile output terse
            save_figure=False,        # skip figure per-tile
            save_to_graph=spec.save_to_graph,
        )
        try:
            if spec_t.fit_tau:
                r = _solve_closed_form_with_tau(
                    graph, spec_t, forward_solver,
                    sim_state_get=sim_state_get,
                    sim_state_set=sim_state_set,
                    cached_snapshots=snaps)
            else:
                r = _solve_closed_form_linear(
                    graph, spec_t, forward_solver,
                    sim_state_get=sim_state_get,
                    sim_state_set=sim_state_set,
                    cached_snapshots=snaps)
            results[int(tid)] = r
            if spec.verbose:
                a = r.params.get('alpha', float('nan'))
                D = r.params.get('D', float('nan'))
                t = r.params.get('tau', 0.0)
                f0 = r.convergence.get('f0_tile', 0.0)
                tau_deg = (np.degrees(2 * np.pi * f0 * t)
                            if f0 > 0 else float('nan'))
                print(f'    tile {tid:>3d}: '
                      f'α̂={a:6.3f}±{r.sigma.get("alpha", 0.0):.3f}, '
                      f'D̂={D:.3e}±{r.sigma.get("D", 0.0):.2e}, '
                      f'τ̂={t * 1e3:+6.1f} ms ({tau_deg:+5.1f}°), '
                      f'χ²/dof={r.chi2_red:.2f}, '
                      f'N={r.n_obs_complex}')
        except Exception as e:
            failures[int(tid)] = str(e)
            if spec.verbose:
                print(f'    tile {tid:>3d}: FAILED — {e}')

    return {'tiles': sorted(results.keys()),
            'results': results,
            'failures': failures,
            'snapshots': snaps}


def _solve_kxk_wls(
    Qm: np.ndarray,
    basis: dict[str, np.ndarray],
    weights: np.ndarray,
    coef_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form weighted LSQ for real coefficients with complex data.

    Solves min_θ Σ w_i · |Q_meas_i − Σ_k θ_k · b^(k)_i|²
    via the real-projection normal equations:

        M[i,j] = Σ w · Re(b_i^* · b_j)
        v[i]   = Σ w · Re(b_i^* · Q_meas)
        θ      = M⁻¹ · v   ∈ ℝ^k

    Returns
    -------
    theta : np.ndarray (k,)        regression coefficients
    M     : np.ndarray (k, k)      normal-equations matrix
    Minv  : np.ndarray (k, k)      its inverse (for covariance)

    Raises
    ------
    np.linalg.LinAlgError if M is singular.
    """
    k = len(coef_order)
    cols = [basis[name] for name in coef_order]
    w = np.asarray(weights, dtype=float)
    M = np.empty((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            M[i, j] = float(np.sum(
                w * np.real(np.conj(cols[i]) * cols[j])))
    v = np.empty(k, dtype=float)
    for i in range(k):
        v[i] = float(np.sum(w * np.real(np.conj(cols[i]) * Qm)))
    theta = np.linalg.solve(M, v)
    Minv = np.linalg.inv(M)
    return theta, M, Minv


def _clip_sigma_e(
    sigma_e: np.ndarray,
    harmonic_idx: np.ndarray,
    cap: Optional[float],
) -> np.ndarray:
    """Bounded-influence clip on per-row σ_e.

    Per harmonic, rescale so that
        max(σ_e) / min(σ_e) ≤ cap

    This prevents the FGLS runaway where, after a Stage-1 misfit,
    the noise model fits b ≈ 1 in σ² = a + b|Q|² (because |r|² ≈
    |Q|² when α is off), giving weights ∝ 1/|Q|² that crush all
    high-|Q| points and pull α toward zero.

    Implementation: anchor on the median σ_e per harmonic, then
    clip to ±√cap relative to it, so the geometric range is `cap`.

    Pass `cap=None` to disable.
    """
    if cap is None or cap <= 1.0:
        return sigma_e
    out = sigma_e.copy()
    for n in np.unique(harmonic_idx):
        mask = harmonic_idx == n
        if mask.sum() < 2:
            continue
        med = float(np.median(out[mask]))
        if med <= 0 or not np.isfinite(med):
            continue
        ratio = float(np.sqrt(cap))
        lo = med / ratio
        hi = med * ratio
        out[mask] = np.clip(out[mask], lo, hi)
    return out


def _coef_order_from_spec(spec: InferenceSpec) -> list[str]:
    """Order of regression coefficients in θ — matches design-matrix
    column order in `_build_design_matrix`."""
    out = ['alpha']  # always first
    for p in active_parameters(spec):
        if p.name == 'alpha' or p.kind != 'LINEAR':
            continue
        out.append(f'beta_{p.name}')
    return out


def _solve_closed_form_linear(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[..., None],
    *,
    sim_state_get: Optional[Callable[[], dict]] = None,
    sim_state_set: Optional[Callable[[dict], None]] = None,
    cached_snapshots: Optional[dict] = None,
) -> InferenceResult:
    """k×k WLS for arbitrary linear parameters (no τ alignment).

    Generalizes the legacy 2×2 (α, β_D) fit to any subset of LINEAR
    params from the registry.  Adding a new LINEAR parameter requires
    only registering its `ParameterDef` — this function is unchanged.
    """
    dm = _build_design_matrix(graph, spec, forward_solver,
                                sim_state_get=sim_state_get,
                                sim_state_set=sim_state_set,
                                cached_snapshots=cached_snapshots)
    Qm = dm.Qm
    N = len(Qm)
    if N == 0:
        raise ValueError('No usable observations after sign-correction '
                          'and finite-finiteness filtering.')

    coef_order = _coef_order_from_spec(spec)
    Qm_mag = np.abs(Qm)
    h_idx = dm.harmonic_idx

    # ── Pass 1: identity weights, get residuals for noise fit ──
    theta1, _, _ = _solve_kxk_wls(Qm, dm.basis, np.ones(N), coef_order)
    pred1 = sum(theta1[i] * dm.basis[c] for i, c in enumerate(coef_order))
    resid1 = Qm - pred1
    resid1_mag = np.abs(resid1)
    stage1_theta_dict = {c: float(theta1[i]) for i, c in enumerate(coef_order)}

    # ── Outer FGLS loop (refit noise → re-weight → re-solve θ) ────
    cur_resid_mag = resid1_mag
    cur_Qm_mag = Qm_mag
    theta2 = theta1.copy()
    M = Minv = None
    noise_models: dict[int, dict] = {}
    sigma_e = np.full(N, 1.0)
    weights = np.ones(N)
    outer_history: list[dict] = []

    for outer_it in range(max(1, int(spec.n_outer_iter))):
        noise_models = {}
        sigma_e = np.full(N, 1.0)
        for n in [0] + list(spec.harmonics):
            mask = h_idx == int(n)
            if not mask.any():
                continue
            if spec.noise == 'identity':
                noise_models[n] = {'form': 'linear', 'a': 1.0, 'b': 0.0}
            elif spec.noise == 'homoscedastic':
                sig = (float(np.std(cur_resid_mag[mask], ddof=1))
                        if mask.sum() >= 2 else 1.0)
                noise_models[n] = {'form': 'linear',
                                     'a': max(sig, 1e-6), 'b': 0.0}
            else:
                noise_models[n] = fit_noise_model(
                    cur_resid_mag[mask], cur_Qm_mag[mask],
                    form=spec.noise_form)
            sigma_e[mask] = np.maximum(
                evaluate_noise_model(noise_models[n],
                                       cur_Qm_mag[mask]), 1e-12)
        # Bounded-influence cap: prevent runaway down-weighting when
        # the noise fit absorbs model misfit (|r|² ≈ |Q|² when α is
        # off → b ≈ 1 → weights ∝ 1/|Q|² → α collapses to 0).
        sigma_e = _clip_sigma_e(sigma_e, h_idx,
                                 spec.sigma_dynamic_range_cap)
        weights = 1.0 / sigma_e ** 2

        theta_prev = theta2.copy()
        theta2, M, Minv = _solve_kxk_wls(Qm, dm.basis, weights, coef_order)
        pred2 = sum(theta2[i] * dm.basis[c]
                     for i, c in enumerate(coef_order))
        resid2 = Qm - pred2
        cur_resid_mag = np.abs(resid2)
        # Qm_mag is unchanged across outer iterations (no τ rotation here)

        denom = max(float(np.linalg.norm(theta_prev)), 1e-30)
        rel_dtheta = float(np.linalg.norm(theta2 - theta_prev) / denom)
        outer_history.append({
            'iter': outer_it + 1,
            'theta': {c: float(theta2[i])
                       for i, c in enumerate(coef_order)},
            'rel_dtheta': rel_dtheta,
            'noise': dict(noise_models),
        })
        if spec.verbose:
            print(f'  [FGLS outer {outer_it + 1}] θ='
                  + ', '.join(f'{c}={theta2[i]:.4g}'
                                for i, c in enumerate(coef_order))
                  + f', ‖Δθ‖/‖θ‖ = {rel_dtheta:.3g}')
        if outer_it > 0 and rel_dtheta < spec.outer_tol:
            break

    # ── χ²/dof ──
    n_dc = int(np.sum(h_idx == 0))
    n_h = sum(int(np.sum(h_idx == int(nn))) for nn in spec.harmonics)
    n_obs_real = n_dc + 2 * n_h
    n_params_fit = len(coef_order)   # τ not fit here
    chi2 = float(np.sum(np.abs(resid2 / sigma_e) ** 2))
    dof = max(n_obs_real - n_params_fit, 1)
    chi2_red = chi2 / dof

    cov_scaled = Minv * chi2_red
    sigma_theta = np.sqrt(np.maximum(np.diag(cov_scaled), 0.0))

    # Pack regression θ + σ
    theta_dict = {c: float(theta2[i]) for i, c in enumerate(coef_order)}
    sigma_dict = {c: float(sigma_theta[i]) for i, c in enumerate(coef_order)}
    cov_pairs = {(coef_order[i], coef_order[j]): float(cov_scaled[i, j])
                 for i in range(len(coef_order))
                 for j in range(len(coef_order))}

    # Recover natural params via each ParameterDef.recover
    params_out = {}
    sigma_out = {}
    for p in active_parameters(spec):
        if p.kind != 'LINEAR' or p.recover is None:
            continue
        params_out[p.name] = p.recover(theta_dict, dm.p0)
        if p.propagate_sigma is not None:
            sigma_out[p.name] = p.propagate_sigma(
                theta_dict, sigma_dict, cov_pairs, dm.p0)
    # Also expose regression coefficients
    for c in coef_order:
        params_out.setdefault(c, theta_dict[c])
        sigma_out.setdefault(c, sigma_dict[c])

    return InferenceResult(
        params=params_out,
        sigma=sigma_out,
        cov=cov_scaled,
        chi2_red=chi2_red,
        n_obs_complex=N,
        n_obs_real=n_obs_real,
        n_params_fit=n_params_fit,
        dof=dof,
        noise_model={spec.focus_tile or 0: noise_models},
        convergence={'iterations': 1, 'method': 'closed_form_linear',
                      'coef_order': coef_order, 'p0': dm.p0,
                      'f0_tile': dm.f0_tile,
                      'n_per_harmonic': dm.n_per_harmonic,
                      'outer_iterations': len(outer_history),
                      'outer_history': outer_history},
        design_matrix=dm,
        sigma_e=sigma_e,
        Qm_rotated=Qm,            # no τ-rotation in this route
        residuals=resid2,
        stage1_theta=stage1_theta_dict,
        stage1_residuals=resid1,
        stage1_Qm_rotated=Qm,
    )


# ──────────────────────────────────────────────────────────────────
# Internal: closed-form alternation with τ (current production path)
# ──────────────────────────────────────────────────────────────────


def _phase_rotate(Qm: np.ndarray, h_idx: np.ndarray,
                    omega0: float, tau: float) -> np.ndarray:
    """Apply e^{−j·n·ω₀·τ} to each row's Q̂_meas (DC unchanged)."""
    phasor = np.where(h_idx == 0,
                       1.0 + 0j,
                       np.exp(-1j * h_idx.astype(float) * omega0 * tau))
    return Qm * phasor


def _solve_tau_closed_form(
    Qm: np.ndarray,
    basis: dict[str, np.ndarray],
    theta: np.ndarray,
    coef_order: Sequence[str],
    weights: np.ndarray,
    h_idx: np.ndarray,
    omega0: float,
) -> float:
    """Closed-form τ update given current linear-coef estimate θ.

    Each H_n contributes its own cross-correlation; the global τ
    that minimizes the WLS residual satisfies

        Σ_n  n · w_n · Q̂_n · conj(μ_n) · exp(-j·n·ω₀·τ) = real

    Stationary point: arg(Σ_n n · …) / ω₀.  For single H1 this reduces
    to `arg(Σ_h1 …) / ω₀`.  Generalizes to multi-harmonic with the
    n-weighting that arises from the chain rule on exp(-j·n·ω₀·τ).
    """
    h = h_idx.astype(float)
    mask = h > 0
    if not mask.any():
        return 0.0
    mu = sum(theta[i] * basis[c] for i, c in enumerate(coef_order))
    w = np.asarray(weights, dtype=float)
    # Stationary point of d/dτ [Σ w |Q − e^{jnω₀τ}·μ|²] = 0
    # gives  Σ n·w·Q·conj(μ)·exp(-jnω₀·τ) ∈ ℝ
    # ⇒  τ* such that arg(Σ n·w·Q·conj(μ)) = n·ω₀·τ*  for the dominant n.
    # When only one harmonic contributes (n=1), this is exactly the
    # legacy formula.  When multiple n are present, take the
    # argmax-likelihood τ from the H1 cross-correlation (still
    # correct when H1 dominates the AC content).
    C = np.sum(w[mask] * Qm[mask] * np.conj(mu[mask]) * h[mask])
    if abs(C) < 1e-30:
        return 0.0
    return float(np.angle(C)) / omega0


def _solve_closed_form_with_tau(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[..., None],
    *,
    sim_state_get: Optional[Callable[[], dict]] = None,
    sim_state_set: Optional[Callable[[dict], None]] = None,
    cached_snapshots: Optional[dict] = None,
) -> InferenceResult:
    """Bilinear closed-form alternation between (linear params) and τ.

    Both subproblems are convex and closed-form:
      * (linear | τ): k×k WLS via Re-projection (`_solve_kxk_wls`)
      * (τ | linear): cross-correlation argmax (`_solve_tau_closed_form`)

    Pass 1 tries τ warm-starts {0, π/ω₀} (configurable via
    `spec.tau_warm_starts`) and keeps the lower-SSR fixed point —
    guards against τ landing in the wrong basin when the true τ is
    near ±π.

    Pass 2 alternates with the WLS weights from the per-harmonic
    noise model.
    """
    dm = _build_design_matrix(graph, spec, forward_solver,
                                sim_state_get=sim_state_get,
                                sim_state_set=sim_state_set,
                                cached_snapshots=cached_snapshots)
    Qm = dm.Qm
    N = len(Qm)
    if N == 0:
        raise ValueError('No usable observations.')

    coef_order = _coef_order_from_spec(spec)
    h_idx = dm.harmonic_idx
    omega0 = 2.0 * np.pi * dm.f0_tile

    n_dc = int(np.sum(h_idx == 0))
    n_h = sum(int(np.sum(h_idx == int(nn))) for nn in spec.harmonics)
    if n_h < 2:
        # Fall back to no-τ path if AC is too thin to identify τ.
        if spec.verbose:
            print('  Insufficient AC observations for τ alignment '
                  f'(n_h={n_h}); falling back to closed_form_linear.')
        spec_no_tau = _replace_spec(spec, fit_tau=False)
        return _solve_closed_form_linear(graph, spec_no_tau,
                                            forward_solver,
                                            sim_state_get=sim_state_get,
                                            sim_state_set=sim_state_set)

    # ── Pass 1: try each τ warm-start, pick lower-SSR fixed point ──
    def _pass1_from(tau_init: float):
        tau = float(tau_init)
        Qm_rot = _phase_rotate(Qm, h_idx, omega0, tau)
        try:
            theta, _, _ = _solve_kxk_wls(Qm_rot, dm.basis,
                                            np.ones(N), coef_order)
        except np.linalg.LinAlgError:
            return None
        tau = _solve_tau_closed_form(Qm, dm.basis, theta, coef_order,
                                       np.ones(N), h_idx, omega0)
        Qm_rot = _phase_rotate(Qm, h_idx, omega0, tau)
        pred = sum(theta[i] * dm.basis[c]
                    for i, c in enumerate(coef_order))
        r = Qm_rot - pred
        ssr = float(np.sum(np.abs(r) ** 2))
        return (tau, theta, Qm_rot, r, ssr)

    cands = []
    for t0 in spec.tau_warm_starts:
        c = _pass1_from(float(t0))
        if c is not None:
            cands.append(c)
    if not cands:
        raise np.linalg.LinAlgError('Pass 1 singular for all warm-starts.')
    cands.sort(key=lambda c: c[4])
    tau_hat, theta1, Qm_rot, resid1, ssr_best = cands[0]
    Qm_mag = np.abs(Qm_rot)
    resid1_mag = np.abs(resid1)
    # Stage-1 captures for two-stage-GLS plotting
    stage1_theta_dict = {c: float(theta1[i]) for i, c in enumerate(coef_order)}
    stage1_Qm_rot = Qm_rot.copy()
    stage1_resid_complex = resid1.copy()

    # ── Outer FGLS loop ────────────────────────────────────────────
    # Each outer step:
    #   (a) Fit per-harmonic noise model on the *current* residual
    #       magnitudes (Stage 1's residuals on the first pass; Stage 2
    #       residuals on subsequent passes).
    #   (b) Build weights = 1/σ²(|Q|).
    #   (c) Run the bilinear (θ, τ) closed-form alternation under
    #       these weights.
    # Converges when ‖Δθ‖ / ‖θ‖ < spec.outer_tol.
    #
    # Why this matters: the first noise-model fit is biased high on
    # whichever harmonic Stage 1 fits poorly (a misfit α₀ inflates
    # those residuals → b_n looks huge → Stage 2 down-weights that
    # harmonic to near-zero → α collapses).  One or two outer
    # iterations recover the true α/β by re-fitting σ² with residuals
    # from the *better* Stage-2 estimate.
    theta = theta1
    M = Minv = None
    converged_iter = 0
    noise_models: dict[int, dict] = {}
    sigma_e = np.ones(N)
    weights = np.ones(N)
    cur_resid_mag = resid1_mag
    cur_Qm_mag = Qm_mag
    cur_Qm_rot = Qm_rot
    outer_history: list[dict] = []

    for outer_it in range(max(1, int(spec.n_outer_iter))):
        # (a) Per-harmonic noise model on current residuals
        noise_models = {}
        sigma_e = np.ones(N)
        for n in [0] + list(spec.harmonics):
            mask = h_idx == int(n)
            if not mask.any():
                continue
            if spec.noise == 'identity':
                noise_models[n] = {'form': 'linear', 'a': 1.0, 'b': 0.0}
            elif spec.noise == 'homoscedastic':
                sig = (float(np.std(cur_resid_mag[mask], ddof=1))
                        if mask.sum() >= 2 else 1.0)
                noise_models[n] = {'form': 'linear',
                                     'a': max(sig, 1e-6), 'b': 0.0}
            else:
                noise_models[n] = fit_noise_model(
                    cur_resid_mag[mask], cur_Qm_mag[mask],
                    form=spec.noise_form)
            sigma_e[mask] = np.maximum(
                evaluate_noise_model(noise_models[n],
                                       cur_Qm_mag[mask]), 1e-12)
        # Bounded-influence cap on σ_e per harmonic (see _clip_sigma_e).
        sigma_e = _clip_sigma_e(sigma_e, h_idx,
                                 spec.sigma_dynamic_range_cap)
        weights = 1.0 / sigma_e ** 2

        # (b/c) Bilinear (θ, τ) closed-form alternation
        theta_prev = theta.copy()
        tau_prev = tau_hat
        converged_iter = 0
        for it in range(spec.n_alt_iter):
            Qm_rot = _phase_rotate(Qm, h_idx, omega0, tau_hat)
            try:
                theta, M, Minv = _solve_kxk_wls(Qm_rot, dm.basis,
                                                  weights, coef_order)
            except np.linalg.LinAlgError:
                break
            tau_new = _solve_tau_closed_form(
                Qm, dm.basis, theta, coef_order, weights,
                h_idx, omega0)
            T_period = 2.0 * np.pi / omega0 if omega0 > 0 else 1.0
            d_tau = abs(((tau_new - tau_hat + T_period / 2)
                          % T_period) - T_period / 2)
            tau_hat = tau_new
            converged_iter = it + 1
            if d_tau < spec.alt_tol:
                break

        # Update residuals at the current θ for the next outer pass
        Qm_rot_now = _phase_rotate(Qm, h_idx, omega0, tau_hat)
        pred_now = sum(theta[i] * dm.basis[c]
                        for i, c in enumerate(coef_order))
        resid_now = Qm_rot_now - pred_now
        cur_resid_mag = np.abs(resid_now)
        cur_Qm_mag = np.abs(Qm_rot_now)
        cur_Qm_rot = Qm_rot_now

        # Outer convergence check
        denom = max(float(np.linalg.norm(theta_prev)), 1e-30)
        rel_dtheta = float(np.linalg.norm(theta - theta_prev) / denom)
        outer_history.append({
            'iter': outer_it + 1,
            'theta': {c: float(theta[i])
                       for i, c in enumerate(coef_order)},
            'tau': float(tau_hat),
            'rel_dtheta': rel_dtheta,
            'noise': dict(noise_models),
        })
        if spec.verbose:
            print(f'  [FGLS outer {outer_it + 1}] θ='
                  + ', '.join(f'{c}={theta[i]:.4g}'
                                for i, c in enumerate(coef_order))
                  + f', τ={tau_hat * 1e3:+.2f} ms, '
                  f'‖Δθ‖/‖θ‖ = {rel_dtheta:.3g}')
            for n_lbl, n_key in (('DC', 0), ('H1', 1)):
                nmm = noise_models.get(n_key)
                if nmm and nmm.get('form') == 'variance_linear':
                    print(f'    σ²_{n_lbl} = {nmm["a"]:.3g} + '
                          f'{nmm["b"]:.3g}·|Q|²')
        if outer_it > 0 and rel_dtheta < spec.outer_tol:
            break

    # ── Final residuals + χ²/dof ──
    Qm_rot_final = cur_Qm_rot
    pred = sum(theta[i] * dm.basis[c] for i, c in enumerate(coef_order))
    resid = Qm_rot_final - pred
    n_obs_real = n_dc + 2 * n_h
    n_params_fit = len(coef_order) + 1   # +1 for τ
    chi2 = float(np.sum(np.abs(resid / sigma_e) ** 2))
    dof = max(n_obs_real - n_params_fit, 1)
    chi2_red = chi2 / dof

    if Minv is not None:
        cov_scaled = Minv * chi2_red
        sigma_theta = np.sqrt(np.maximum(np.diag(cov_scaled), 0.0))
    else:
        cov_scaled = np.full((len(coef_order), len(coef_order)), np.nan)
        sigma_theta = np.full(len(coef_order), np.nan)

    theta_dict = {c: float(theta[i]) for i, c in enumerate(coef_order)}
    sigma_dict = {c: float(sigma_theta[i]) for i, c in enumerate(coef_order)}
    cov_pairs = {(coef_order[i], coef_order[j]): float(cov_scaled[i, j])
                 for i in range(len(coef_order))
                 for j in range(len(coef_order))}

    params_out = {}
    sigma_out = {}
    for p in active_parameters(spec):
        if p.kind == 'LINEAR' and p.recover is not None:
            params_out[p.name] = p.recover(theta_dict, dm.p0)
            if p.propagate_sigma is not None:
                sigma_out[p.name] = p.propagate_sigma(
                    theta_dict, sigma_dict, cov_pairs, dm.p0)
    params_out['tau'] = float(tau_hat)
    sigma_out['tau'] = float('nan')   # τ uncertainty needs a separate
                                        # Hessian element; skip for now.
    for c in coef_order:
        params_out.setdefault(c, theta_dict[c])
        sigma_out.setdefault(c, sigma_dict[c])

    convergence = {
        'iterations': converged_iter,
        'method': 'closed_form_with_tau',
        'coef_order': coef_order,
        'p0': dm.p0,
        'f0_tile': dm.f0_tile,
        'n_per_harmonic': dm.n_per_harmonic,
        'tau_warm_start_ssrs': [c[4] for c in cands],
        'tau_warm_start_picked': float(tau_hat),
        'outer_iterations': len(outer_history),
        'outer_history': outer_history,
    }

    return InferenceResult(
        params=params_out,
        sigma=sigma_out,
        cov=cov_scaled,
        chi2_red=chi2_red,
        n_obs_complex=N,
        n_obs_real=n_obs_real,
        n_params_fit=n_params_fit,
        dof=dof,
        noise_model={spec.focus_tile or 0: noise_models},
        convergence=convergence,
        design_matrix=dm,
        sigma_e=sigma_e,
        Qm_rotated=Qm_rot_final,
        residuals=resid,
        stage1_theta=stage1_theta_dict,
        stage1_residuals=stage1_resid_complex,
        stage1_Qm_rotated=stage1_Qm_rot,
    )


def _replace_spec(spec: InferenceSpec, **kwargs) -> InferenceSpec:
    """Return a copy of `spec` with the named fields overridden."""
    from dataclasses import replace
    return replace(spec, **kwargs)


# ──────────────────────────────────────────────────────────────────
# Per-tile state on the graph
# ──────────────────────────────────────────────────────────────────
#
# `G.graph['tile_models'][tile_id]` is a per-tile dict with keys:
#   'noise'       : { harmonic_idx → noise_model dict }
#   'tau'         : float            (cardiac-phase offset, seconds)
#   'last_chi2'   : float            (chi2/dof from last fit)
#   'last_fit_method' : str          (e.g. 'closed_form_with_tau')
#   'last_fit_at'     : ISO-8601 timestamp string
#
# Use cases:
#   * Inference reuses a previously-fit noise model rather than refitting
#     (set spec.noise='cached' — TODO Step 4).
#   * Visualization layer: σ(|Q|) heatmap, τ-by-tile bar chart, etc.
#   * Outlier flagging: tiles whose `last_chi2` ≫ 1 are unreliable.
#
# Helpers below are the only intended way to read/write this dict;
# callers should never poke `G.graph['tile_models']` directly.


def get_tile_model(graph, tile_id: int) -> dict:
    """Return the per-tile model dict, creating an empty one if missing."""
    tm = graph.graph.setdefault('tile_models', {})
    return tm.setdefault(int(tile_id), {})


def set_tile_noise_model(
    graph, tile_id: int, noise_by_harmonic: dict[int, dict],
) -> None:
    """Persist per-harmonic noise models for one tile.

    `noise_by_harmonic` maps harmonic n → dict from `fit_noise_model`,
    e.g. {0: {'form': 'linear', 'a': ..., 'b': ...},
          1: {'form': 'linear', 'a': ..., 'b': ...}}.
    """
    tm = get_tile_model(graph, tile_id)
    tm['noise'] = {int(n): dict(m) for n, m in noise_by_harmonic.items()}


def set_tile_tau(graph, tile_id: int, tau: float) -> None:
    """Persist the cardiac-phase offset (seconds) for one tile."""
    tm = get_tile_model(graph, tile_id)
    tm['tau'] = float(tau)


def set_tile_fit_meta(
    graph, tile_id: int,
    *, chi2_red: float, method: str,
) -> None:
    """Persist last-fit metadata for one tile."""
    import datetime as _dt
    tm = get_tile_model(graph, tile_id)
    tm['last_chi2'] = float(chi2_red)
    tm['last_fit_method'] = str(method)
    tm['last_fit_at'] = _dt.datetime.now(
        _dt.timezone.utc).isoformat(timespec='seconds')


def persist_result_to_graph(
    graph, spec: InferenceSpec, result: InferenceResult,
) -> None:
    """Write the result of `run_inference` onto the graph's per-tile state.

    Only persists when `spec.save_to_graph` is True.  Routing:

      * single_tile / per_tile: per-tile noise + τ + chi2 keyed by tile_id
      * global / hierarchical:  populated under tile_id=0 as a sentinel
    """
    if not spec.save_to_graph:
        return
    if spec.scope in ('single_tile', 'per_tile'):
        if spec.scope == 'single_tile':
            tids = [spec.focus_tile] if spec.focus_tile is not None else []
        else:
            tids = list(result.noise_model.keys())
        for tid in tids:
            if tid is None:
                continue
            nm = result.noise_model.get(tid)
            if nm:
                set_tile_noise_model(graph, int(tid), nm)
            if 'tau' in result.params:
                set_tile_tau(graph, int(tid), result.params['tau'])
            set_tile_fit_meta(
                graph, int(tid),
                chi2_red=result.chi2_red,
                method=result.convergence.get('method', '?'))
    else:
        # Global / hierarchical — store under tile_id=0 sentinel
        nm = result.noise_model.get(0) or next(iter(result.noise_model.values()),
                                                  None)
        if nm:
            set_tile_noise_model(graph, 0, nm)


def cached_noise_model(
    graph, tile_id: int, harmonic: int = 1,
) -> Optional[dict]:
    """Look up a previously-fit noise model on the graph; None if absent."""
    tm = graph.graph.get('tile_models', {}).get(int(tile_id))
    if not tm:
        return None
    return tm.get('noise', {}).get(int(harmonic))


# ──────────────────────────────────────────────────────────────────
# Internal: hierarchical α + pooling
# ──────────────────────────────────────────────────────────────────


def _solve_closed_form_hierarchical(graph, spec, forward_solver, **kw):
    """Per-tile α with a global pooling prior α_t ~ 𝒩(α_g, σ_pool²).

    Block-diagonal normal equations:

        [M_g + Σ_t λ      −λ_1  −λ_2   …]   [α_g  ]   [v_g  ]
        [−λ_1     M_t,1+λ_1                ]   [α_t,1] = [v_t,1]
        [−λ_2              M_t,2+λ_2       ]   [α_t,2]   [v_t,2]
        …

    where λ_t = 1/σ_pool² and σ_pool is either fixed (spec.pool_alpha_sigma)
    or chosen by empirical Bayes on the marginal likelihood.

    When fit_D is also True, extends to per-tile (α_t, D_t) with a
    multivariate pooling prior.

    Step 5 milestone (week 4).
    """
    raise NotImplementedError(
        "Step 5 (week 4): empirical-Bayes hierarchical α."
    )


# ──────────────────────────────────────────────────────────────────
# Internal: profile likelihood (1-D scan)
# ──────────────────────────────────────────────────────────────────


def _solve_profile_likelihood(graph, spec, forward_solver, **kw):
    """1-D profile over a single nonlinear parameter.

    Useful when D is in a regime where the linear approximation around
    D₀ is poor (κL² varies appreciably across the scan range).  Sweeps
    D over a grid, runs the forward solver at each point, fits the
    remaining (linear) params at each, returns the maximum-likelihood
    D̂ with a parabolic σ from the curvature of the log-likelihood.

    Also serves as a brute-force comparison for the closed-form
    alternation (reviewer's suggestion).

    Step 3-4 milestone.
    """
    raise NotImplementedError(
        "Step 3-4: profile likelihood scan."
    )


# ──────────────────────────────────────────────────────────────────
# Internal: Newton / Levenberg-Marquardt
# ──────────────────────────────────────────────────────────────────


def _solve_newton(graph, spec, forward_solver, **kw):
    """General nonlinear optimizer for cases that don't have a closed
    form: full (α, D, μ, τ, ζ_e) joint inference, or non-linear D
    dependence in cosh/sinh admittance.

    Backed by ``scipy.optimize.least_squares`` with finite-difference
    Jacobian.  Slower than closed-form but unconstrained.

    Step 4 milestone.
    """
    raise NotImplementedError(
        "Step 4: scipy.optimize.least_squares LM solver."
    )


# ──────────────────────────────────────────────────────────────────
# Internal: synthetic recovery
# ──────────────────────────────────────────────────────────────────


def _run_synthetic_recovery(
    graph,
    spec: InferenceSpec,
    forward_solver: Callable[..., None],
    *,
    sim_state_get: Optional[Callable[[], dict]] = None,
    sim_state_set: Optional[Callable[[dict], None]] = None,
) -> InferenceResult:
    """Synthetic-data validation harness.

    Strategy:
      1. Choose ground-truth params from `spec.synth_truth` (defaults
         pulled from current solver state for D, μ).
      2. Run the forward solver at (D*, μ*) and snapshot per-edge sim
         phasors → these are the "true" Q_e^*.
      3. Build synthetic per-edge measurement records:
            Q_meas_dc = α* · Q_dc^* + ε_dc
            Q̂_meas_h1 = α* · Q̂_h1^* · e^{+jω₀τ*} + ε_h1
         where ε ~ 𝒩(0, σ²) with σ from `spec.synth_truth['noise']`
         (falls back to defaults if missing).  Records are injected
         into edge `measurements_piv` under tile id 999998 (a synthetic
         sentinel) so the rest of the pipeline doesn't have to know.
      4. Run inference on the synthetic data using `spec` rerouted as
         a normal `single_tile` fit at the synthetic tile id.
      5. Compute bias and z-scores per parameter.

    Reviewer uses:
      * σ_D/D scaling — sweep `synth_truth['noise']['scale']`.
      * Sanity at D=D₀ — set `synth_truth['D'] = current_D`; β should
        be 0 within σ_β.
      * Brute-force comparison — call `_solve_newton` on the same data
        (Step 5).

    The synthetic measurements are removed from the graph before
    return so the user's real data is untouched.
    """
    if not spec.synth_truth:
        raise ValueError(
            'synthetic_recovery requires `spec.synth_truth = {"alpha": ..., '
            '"D": ..., "tau": ..., "noise": {"a_dc": ..., "b_dc": ..., '
            '"a_h1": ..., "b_h1": ...}}`')
    truth = dict(spec.synth_truth)
    rng = np.random.default_rng(spec.synth_seed)

    # Sentinel tile id for synthetic measurements
    SYNTH_TID = 999998

    # ── 1. Establish ground-truth solver state ──
    # Use viewer state for D₀ if not specified; otherwise force the
    # solver to run at synth_truth['D'].
    state_orig = sim_state_get() if sim_state_get is not None else {}
    truth_state = dict(state_orig)
    if 'D' in truth and '_sim_D' in state_orig:
        truth_state['_sim_D'] = float(truth['D'])
    if 'mu' in truth and '_sim_mu' in state_orig:
        truth_state['_sim_mu'] = float(truth['mu'])
    if sim_state_set is not None:
        sim_state_set(truth_state)
    forward_solver()

    # ── 2. Snapshot ground-truth phasors ──
    snap_truth = _snapshot_phasors(graph, spec.harmonics)
    if sim_state_set is not None:
        sim_state_set(state_orig)

    # ── 3. Inject synthetic measurements ──
    alpha_t = float(truth.get('alpha', 1.0))
    tau_t = float(truth.get('tau', 0.0))
    f0_t = float(truth.get('f0',
                            graph.graph.get('reference_f0_hz', 2.5)))
    omega0_t = 2.0 * np.pi * f0_t
    nm = truth.get('noise', {})
    a_dc = float(nm.get('a_dc', 0.02))
    b_dc = float(nm.get('b_dc', 0.05))
    a_h1 = float(nm.get('a_h1', 0.02))
    b_h1 = float(nm.get('b_h1', 0.10))

    saved_records: list[tuple[tuple, dict]] = []
    n_injected = 0
    for u, v, d in graph.edges(data=True):
        truth_phasors = snap_truth.get((u, v))
        if truth_phasors is None:
            continue
        Q_dc_truth = truth_phasors.get(0, np.nan)
        if not np.isfinite(Q_dc_truth):
            continue

        # DC measurement: α* · Q_dc + N(0, σ_dc(|Q|))
        sigma_dc_e = a_dc + b_dc * abs(Q_dc_truth) * abs(alpha_t)
        Q_dc_meas = (alpha_t * float(Q_dc_truth)
                      + float(rng.normal(0, sigma_dc_e)))

        # H1 measurement: α* · Q̂_h1 · exp(+jω₀τ*) + complex N(0, σ_h1)
        # (positive sign in the phasor — measurement is "delayed by τ*"
        # vs ground truth, matching the engine's convention.)
        Q_h1_truth = truth_phasors.get(1, complex(np.nan))
        Q_h1_meas = None
        if np.isfinite(Q_h1_truth) and abs(Q_h1_truth) > 1e-30:
            Q_h1_aligned = (alpha_t * Q_h1_truth
                              * np.exp(1j * omega0_t * tau_t))
            sigma_h1_e = a_h1 + b_h1 * abs(Q_h1_aligned)
            # σ scales BOTH Re and Im components independently
            noise_h1 = (rng.normal(0, sigma_h1_e)
                          + 1j * rng.normal(0, sigma_h1_e))
            Q_h1_meas = Q_h1_aligned + noise_h1

        # Write synthetic record to edge.  Save originals so we can
        # restore on exit.
        rec = {
            'tile_id': SYNTH_TID,
            'mean_Q': float(Q_dc_meas),
            'amp_Q': float(abs(Q_h1_meas))
                      if Q_h1_meas is not None
                      and np.isfinite(Q_h1_meas) else 0.0,
            'phase': float(np.degrees(np.angle(Q_h1_meas)))
                      if Q_h1_meas is not None
                      and np.isfinite(Q_h1_meas) else 0.0,
            'f0_hz': f0_t,
            'flow_from': u,
            'flow_to': v,
            'harmonics': [],
        }
        if Q_h1_meas is not None and np.isfinite(Q_h1_meas):
            # Q̂_h1 = A − jB ⇒ A = Re(Q̂), B = −Im(Q̂)
            rec['harmonics'].append({
                'k': 1,
                'A': float(Q_h1_meas.real),
                'B': float(-Q_h1_meas.imag),
                'amp': float(abs(Q_h1_meas)),
                'phi': float(np.angle(Q_h1_meas)),
            })
        # Save existing list and inject
        existing = list(d.get('measurements_piv', []) or [])
        saved_records.append(((u, v), existing))
        d['measurements_piv'] = existing + [rec]
        n_injected += 1

    if spec.verbose:
        print(f'  [synthetic_recovery] injected {n_injected} records, '
              f'truth: α={alpha_t}, D={truth.get("D")}, τ={tau_t * 1e3:.1f} ms')

    try:
        # ── 4. Run the engine on the synthetic data ──
        spec_run = _replace_spec(
            spec,
            scope='single_tile',
            focus_tile=SYNTH_TID,
            save_to_graph=False,
            save_figure=False,
            synth_truth=None,         # avoid recursion
        )
        # Pick the right closed-form route (with τ if requested).
        if spec_run.fit_tau:
            result = _solve_closed_form_with_tau(
                graph, spec_run, forward_solver,
                sim_state_get=sim_state_get,
                sim_state_set=sim_state_set)
        else:
            result = _solve_closed_form_linear(
                graph, spec_run, forward_solver,
                sim_state_get=sim_state_get,
                sim_state_set=sim_state_set)
    finally:
        # ── 6. Remove injected synthetic records ──
        for (u, v), original in saved_records:
            graph.edges[u, v]['measurements_piv'] = original

    # ── 5. Compute bias and z-scores ──
    bias = {}
    z = {}
    for k in ('alpha', 'D', 'tau', 'mu'):
        if k in truth and k in result.params:
            est = float(result.params[k])
            sig = float(result.sigma.get(k, float('nan')))
            t = float(truth[k])
            bias[k] = est - t
            z[k] = (est - t) / sig if (sig > 0 and np.isfinite(sig)) \
                else float('nan')

    if spec.verbose:
        print(f'  [synthetic_recovery] bias  = {bias}')
        print(f'  [synthetic_recovery] z     = {z}')

    result.synth_truth = truth
    result.synth_bias = bias
    result.synth_z = z
    return result


# ──────────────────────────────────────────────────────────────────
# Internal: noise-model fit (linear / quadrature / powerlaw)
# ──────────────────────────────────────────────────────────────────


_HALF_NORMAL_STD_TO_SIGMA = 1.0 / np.sqrt(1.0 - 2.0 / np.pi)
"""Correction factor: if X ~ N(0, σ), then std(|X|) = σ·√(1 − 2/π).

When `fit_noise_model` is given magnitudes of zero-mean residuals,
`std(|r|)` underestimates σ by 1/_HALF_NORMAL_STD_TO_SIGMA ≈ 0.603.
Multiply by this factor to recover the true σ.  Off by default to
match legacy `_fit_noise_one` behavior — flip on once the synthetic-
recovery harness validates the corrected χ²/dof distribution."""


def fit_noise_model(
    residuals_mag: np.ndarray,
    Q_mag: np.ndarray,
    *,
    form: Literal['linear', 'quadrature', 'powerlaw',
                    'variance_linear'] = 'variance_linear',
    n_bins: Optional[int] = None,
    floor: float = 1e-6,
    correct_half_normal: bool = False,
) -> dict:
    """Fit a noise model σ(|Q|) via percentile-binned WLS.

    Steps:
      1. Bin |Q| into `n_bins` percentile bins (auto: max(3, min(6, N//4))).
      2. In each bin compute (mean_Q, std_residual, sqrt(N_bin)) as
         (x, y, weight).
      3. Fit the form-specific regression with weights.

    Returns a dict that is consumable by `evaluate_noise_model`:

        linear          → {'form': 'linear',          'a': float, 'b': float}
        quadrature      → {'form': 'quadrature',      'a': float, 'b': float}
        powerlaw        → {'form': 'powerlaw',        'c': float, 'p': float}
        variance_linear → {'form': 'variance_linear', 'a': float, 'b': float}

    Heuristic notes:
      * `variance_linear` (DEFAULT) — σ² = a + b·|Q|².  Fit by direct
        linear regression of |residual|² against |Q|² — no log, no
        sqrt-of-noisy-stat.  Most robust default: it captures both a
        constant floor (a) and a flow-proportional variance term (b)
        without the half-normal bias of the std-of-magnitude estimators.
      * `linear` (legacy) — σ = a + b·|Q|.  Original PerTileFlow form.
      * `quadrature` — σ = √(a² + (b·|Q|)²).
      * `powerlaw` — σ = c·|Q|^p.

    Falls back to a constant σ when there are too few bins.
    """
    r = np.asarray(residuals_mag, dtype=float)
    q = np.asarray(Q_mag, dtype=float)
    mask = np.isfinite(r) & np.isfinite(q) & (q >= 0)
    r = r[mask]
    q = q[mask]
    n_pts = len(r)
    if n_pts < 4:
        # Too few — return a constant σ that matches `form` schema.
        sigma_const = (max(float(np.std(r, ddof=1))
                            if n_pts >= 2 else float(np.std(r)),
                            floor)
                       if n_pts > 0 else floor)
        if form == 'linear':
            return {'form': 'linear', 'a': sigma_const, 'b': 0.0}
        if form == 'quadrature':
            return {'form': 'quadrature', 'a': sigma_const, 'b': 0.0}
        if form == 'powerlaw':
            return {'form': 'powerlaw', 'c': sigma_const, 'p': 0.0}
        if form == 'variance_linear':
            return {'form': 'variance_linear',
                    'a': sigma_const ** 2, 'b': 0.0}
        raise ValueError(f'unknown noise form {form!r}')

    # ── Direct regression for variance_linear (no binning required) ──
    # σ² = a + b·|Q|²  ⇒  E[|r|²] = a + b·|Q|²  (since E[|r|²] = σ²
    # for zero-mean noise, regardless of distribution shape).
    # Reviewer's one-line spec: regress |r|² on |Q|² with two params.
    # No half-normal bias issue since we're not taking std-of-magnitudes.
    if form == 'variance_linear':
        r2 = r ** 2
        q2 = q ** 2
        A = np.column_stack([np.ones_like(q2), q2])
        sol, *_ = np.linalg.lstsq(A, r2, rcond=None)
        a_v = max(float(sol[0]), floor ** 2)
        b_v = max(float(sol[1]), 0.0)
        return {'form': 'variance_linear', 'a': a_v, 'b': b_v}

    if n_bins is None:
        n_bins = max(3, min(6, n_pts // 4))
    edges = np.percentile(q, np.linspace(0, 100, n_bins + 1))
    cx, cy, cw = [], [], []
    bias = _HALF_NORMAL_STD_TO_SIGMA if correct_half_normal else 1.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mb = (q >= edges[i]) & (q <= edges[i + 1])
        else:
            mb = (q >= edges[i]) & (q < edges[i + 1])
        if mb.sum() >= 2:
            cx.append(float(np.mean(q[mb])))
            cy.append(float(np.std(r[mb], ddof=1)) * bias)
            cw.append(float(np.sqrt(mb.sum())))
    if len(cx) < 2:
        sigma_const = max(float(np.std(r, ddof=1)), floor)
        if form == 'linear':
            return {'form': 'linear', 'a': sigma_const, 'b': 0.0}
        if form == 'quadrature':
            return {'form': 'quadrature', 'a': sigma_const, 'b': 0.0}
        if form == 'powerlaw':
            return {'form': 'powerlaw', 'c': sigma_const, 'p': 0.0}
    cx = np.asarray(cx); cy = np.asarray(cy); cw = np.asarray(cw)

    if form == 'linear':
        # σ = a + b·Q  ⇒  weighted lstsq on [1, Q] vs cy
        A = np.column_stack([np.ones_like(cx), cx]) * cw[:, None]
        sol, *_ = np.linalg.lstsq(A, cy * cw, rcond=None)
        return {'form': 'linear',
                'a': max(float(sol[0]), floor),
                'b': max(float(sol[1]), 0.0)}

    if form == 'quadrature':
        # σ² = a² + (b·Q)²  ⇒  weighted lstsq on [1, Q²] vs cy².
        # Variance-space weights: w² (since y = σ² has variance ∝ σ⁴).
        A = np.column_stack([np.ones_like(cx), cx ** 2]) * (cw ** 2)[:, None]
        sol, *_ = np.linalg.lstsq(A, (cy ** 2) * (cw ** 2), rcond=None)
        a2 = max(float(sol[0]), floor ** 2)
        b2 = max(float(sol[1]), 0.0)
        return {'form': 'quadrature',
                'a': float(np.sqrt(a2)), 'b': float(np.sqrt(b2))}

    if form == 'powerlaw':
        # σ = c·Q^p  ⇒  log σ = log c + p·log Q
        # Drop bins where Q=0 (log undefined).
        keep = cx > 0
        if keep.sum() < 2:
            sigma_const = max(float(np.std(r, ddof=1)), floor)
            return {'form': 'powerlaw', 'c': sigma_const, 'p': 0.0}
        lx = np.log(cx[keep])
        ly = np.log(np.maximum(cy[keep], floor))
        lw = cw[keep]
        A = np.column_stack([np.ones_like(lx), lx]) * lw[:, None]
        sol, *_ = np.linalg.lstsq(A, ly * lw, rcond=None)
        return {'form': 'powerlaw',
                'c': max(float(np.exp(sol[0])), floor),
                'p': float(sol[1])}

    raise ValueError(f'unknown noise form {form!r}')


def evaluate_noise_model(model: dict, Q_mag: np.ndarray) -> np.ndarray:
    """Evaluate σ(|Q|) under the fitted model dict from `fit_noise_model`."""
    form = model.get('form', 'linear')
    if form == 'linear':
        return float(model['a']) + float(model['b']) * Q_mag
    if form == 'quadrature':
        a = float(model['a'])
        b = float(model['b'])
        return np.sqrt(a * a + (b * Q_mag) ** 2)
    if form == 'powerlaw':
        c = float(model['c'])
        p = float(model['p'])
        # Guard |Q|=0
        return c * np.power(np.maximum(Q_mag, 1e-30), p)
    if form == 'variance_linear':
        # σ² = a + b·|Q|²   ⇒   σ = √(a + b·|Q|²)
        a = float(model['a'])
        b = float(model['b'])
        return np.sqrt(np.maximum(a + b * Q_mag ** 2, 0.0))
    raise ValueError(f'unknown noise form {form!r}')
