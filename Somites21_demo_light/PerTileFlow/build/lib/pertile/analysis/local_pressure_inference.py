"""Per-tile local pressure inference.

A separate workflow from the global (α, D, τ) MLE in `inference.py`.

The global MLE asks: "what global flow scale α and distensibility D
best match measurements assuming the simulator's boundary conditions
are right?"

This module asks instead: "given a tile's measured edge flows alone,
what boundary pressures + D explain them?"  Each tile is an
independent inverse problem — no global α, no DA/SV reference,
no inter-tile coupling.

Forward model (per-tile, per-harmonic n):

    Q^sim_e(n)  =  Σ_k  T_e^k(n, D) · P_k(n)

where
    P_k(n)        = pressure at tile-perimeter boundary node k, harmonic n
                    (real for n=0, complex for n≥1)
    T_e^k(n, D)   = transfer-matrix element: edge e's flow when unit
                    pressure is applied at boundary k (and zero at
                    other boundaries), at harmonic n, with distensibility D

Inference: alternating least squares, both subproblems closed-form

    Step 1 (linear, P given D):
        complex WLS on the T·P = Q_meas system, with one boundary
        pressure pinned to 0 for gauge

    Step 2 (1-D linear, D given P):
        linearize T(D) ≈ T(D₀) + (D − D₀)·∂T/∂D  via finite difference
        regress (Q_meas − T(D₀)·P) onto (∂T/∂D · P)
        update D ← D + ε

    iterate until |ΔD|/D < tol_rel (typically 5–10 iterations).

Pi-model lumped admittances (yolk-sac is in the lumped regime
|κL| ≪ 1 so this is accurate):
    G_e   =  πR⁴ / (8μL)            [m³/(Pa·s)]   series conductance
    C_e   =  2πR² · D · L           [m³/Pa]        compliance
    Y_AC  =  G_e + jωC_e/2 (shunts to ground at each endpoint)

Outputs per tile:
    D̂_i, σ_D_i
    P_k^DC, σ_P_k^DC          per boundary node k
    P_k^H1 (complex), σ_P_k^H1
    χ²/dof, condition numbers (DC, H1), iteration history

Persisted on graph at G.graph['per_tile_local_inference'][tile_id].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────


@dataclass
class LocalInferenceSpec:
    """Knobs for `infer_local`.

    Defaults are tuned for chick yolk-sac vasculature (D ≈ 1e-4 1/Pa,
    μ = 2.5 mPa·s, f₀ ≈ 2.5 Hz).
    """

    D_init: float = 1.0e-4              # 1/Pa, initial guess
    eps_D: float = 0.10                 # relative finite-diff step for ∂T/∂D
    lambda_reg: float = 0.0             # ridge regularization on P (per-harmonic)
    max_iter: int = 10                  # outer alternation iterations
    tol_rel: float = 1.0e-3             # convergence: |ΔD|/D
    harmonics: Sequence[int] = (1,)     # AC harmonics included beyond DC
    use_dc: bool = True
    mu: float = 2.5e-3                  # Pa·s
    f0_hz: Optional[float] = None       # None ⇒ tile median from PIV records
    pin_node: Optional[int] = None      # None ⇒ auto-pick by G_attach (highest-Fisher boundary node)
    px_size_m: Optional[float] = None   # None ⇒ load from analysis.config
    verbose: bool = True
    save_to_graph: bool = True
    # When True, include unmeasured anatomical edges in the per-tile
    # network model (conduction-only — they contribute to L/B/T but
    # carry no residual term).  Closes connectivity gaps caused by the
    # "edge has PIV for this tile" filter.  See `extract_tile_subgraph`.
    include_unmeasured_anatomy: bool = False
    # Carve geometry knobs.  Defaults reproduce the long-standing
    # production carve (full bbox, all edges that fit).  Setting both
    # to notebook values (inset=0.05, restrict=True) reproduces the
    # notebook's `_carve_tile + select_subgraph` carve — useful for
    # comparison work, but on real data this smaller carve has *worse*
    # H1 conditioning than the full bbox (cond jumps by ~6 orders of
    # magnitude and D̂ rolls to the floor).  Investigate before re-
    # enabling as default.
    carve_inset_frac: float = 0.0
    carve_restrict_to_tile_piv: bool = False
    # Drop boundary nodes whose only carve-edges go to OTHER boundary
    # nodes (no interior neighbor).  Such "dangling" boundaries carry
    # no information about interior dynamics but still add 1 DC + 2 AC
    # parameters each to the joint Hessian, inflating σ_D and σ_P_b
    # without rescuing identifiability.  Default off for back-compat.
    carve_drop_dangling_boundaries: bool = False
    # FGLS lock-b diagnostic.  When None (default), the noise model
    # σ²(Q) = a + b·|Q|² is fit freely.  When set to a float, b is
    # locked to that value and only a is re-fit (a = mean(|r|² − b·|Q|²)).
    # The natural diagnostic value is 0.0: forces a constant noise
    # floor and reveals whether FGLS's relative-noise term was real
    # PIV heteroscedasticity or absorbed model misspecification.
    # Tiles whose D̂ shifts under fgls_lock_b_to=0.0 had the latter.
    fgls_lock_b_to: Optional[float] = None
    # Empirical-Bayes prior on boundary pressure magnitude (Pa).  When
    # set, adds a Tikhonov term λ‖P‖² with
    #     λ  =  percentile_90(|diag(T†WT)|)  /  P_scale_Pa²
    # so the prior is comparable to a robust upper estimate of per-mode
    # data Fisher information when |P| = P_scale_Pa.  The 90th-percentile
    # makes λ outlier-resistant: a single artery boundary node with
    # extreme Fisher info doesn't recalibrate the prior strength for
    # the bulk of the network (this was the tile-27 fix).
    #
    # **Empirical-Bayes caveat:** because λ is set from data via
    # diag(T†WT), the reported σ_P from the inverse Hessian is
    # *conditional on* the data-chosen regularization, not on a fixed
    # physiological prior.  σ_P should be read as "uncertainty given
    # the worst-conditioned boundary direction has been shrunk to
    # comparable size with the bulk", NOT as "physiologically calibrated
    # uncertainty".  If you have a hard physiological scale to anchor
    # the prior in absolute Pa², set `P_scale_Pa_fixed` instead.
    #
    # None ⇒ no scale-aware prior; only the explicit `lambda_reg` is
    # applied.
    P_scale_Pa: Optional[float] = None
    # Fixed Tikhonov strength in *physical* units: λ = 1/P_scale_Pa_fixed².
    # No data-adaptive scaling.  Cleaner physical interpretation of σ_P
    # at the cost of sensitivity to the choice (a too-tight σ_P here
    # will bias well-determined boundary pressures toward zero).  Only
    # one of P_scale_Pa / P_scale_Pa_fixed should be set; the empirical-
    # Bayes form takes precedence if both are non-None.
    P_scale_Pa_fixed: Optional[float] = None
    # Hard prune (drop edges below threshold) is OFF by default.  The
    # P_scale_Pa prior already regularises weakly-coupled boundary
    # directions continuously — set these only when you've confirmed
    # the offending edges are topology errors (segmentation
    # hallucinations), not thin-but-real vessels.  See the boundary-
    # coupling diagnostic for the report that decides which case
    # applies.
    min_G_factor: Optional[float] = None
    min_R_px: Optional[float] = None
    # Solver mode.  True (default): joint Levenberg-Marquardt — single
    # stacked Gauss-Newton step per iter over (D, P_DC, P_H1), giving
    # quadratic convergence and a unified Hessian for joint σ.
    # False: alternating LS — fix P then 1-D step on D, alternating.
    # Linearly convergent (geometric tail) but historically the default
    # and useful for cross-check.
    use_joint_lm: bool = True
    lm_mu0: float = 1.0e-3
    lm_factor: float = 3.0
    # FGLS outer iterations: after each inner solve, refit
    # σ²(|Q|) = a + b·|Q|² (variance_linear form, from
    # `analysis.inference.fit_noise_model`) and re-run the inner LM
    # with the new per-edge weights.  Typically settles in 2-3 passes.
    # Set to 1 to disable (homoscedastic σ from initial residuals,
    # equivalent to legacy behaviour).
    n_outer_iter: int = 3
    # Prior structure on boundary pressures:
    #   'magnitude' (default, legacy)  — penalises ‖P_b‖² so all
    #     boundary entries are shrunk toward 0.  Biases D when truth
    #     has substantial P_b magnitude (Phase 0 diagnostic confirmed
    #     this on tile 22 + tile 17).
    #   'smooth_h1'                    — keeps the magnitude prior on
    #     DC, but replaces the H1 prior with a SMOOTHNESS penalty
    #     λ·‖L_b·P_b_H1‖² where L_b is a 1D Laplacian over the
    #     angle-ordered boundary perimeter.  Penalises zig-zag
    #     oscillations between adjacent boundary nodes without
    #     forcing the bulk H1 scale toward zero.  Better-suited to
    #     fields where the global solution is smooth around the
    #     carve perimeter (which is the case in all tested tiles).
    prior_mode: str = 'magnitude'
    # Drift bail-out.  In strong-prior regimes (e.g. smooth_h1 with a
    # tight P_scale_Pa) the LM step is dominated by the prior block: D
    # is "free" to drift along a flat χ² ridge while the boundary
    # pressures keep adjusting, producing |ΔD|/D ≈ tol_rel for hundreds
    # of accepted iterations without ever crossing the rel_step
    # threshold.  When `n_drift_window` consecutive ACCEPTED iterations
    # have rel_dD < tol_rel, declare "D non-identifiable at this prior
    # strength" and break.  Counter resets on a step with rel_dD ≥
    # tol_rel.  Result is returned with converged=False so callers
    # know to treat D̂ with caution.  Set to 0 to disable.
    n_drift_window: int = 8


@dataclass
class LocalInferenceResult:
    """Output of `infer_local` for one tile."""

    tile_id: int
    D_hat: float
    sigma_D: float
    P_DC: dict                           # {node_id: float}
    P_H1: dict                           # {node_id: complex}
    sigma_P_DC: dict                     # {node_id: float}  (gauge node = 0)
    sigma_P_H1: dict                     # {node_id: float}  (magnitude of σ)
    chi2_red: float
    n_obs_real: int
    n_params: int
    dof: int
    iterations: int
    converged: bool
    boundary_nodes: list
    interior_edges: list                 # list of (u, v) tuples
    pin_node: int
    cond_DC: float
    cond_H1: float
    f0_hz: float
    # Diagnostic arrays (per interior edge)
    Q_meas_DC: np.ndarray                # real
    Q_meas_H1: np.ndarray                # complex
    Q_pred_DC: np.ndarray
    Q_pred_H1: np.ndarray
    valid_dc: np.ndarray
    valid_h1: np.ndarray
    convergence_history: list = field(default_factory=list)
    # FGLS-fitted per-edge noise model σ²(|Q|) = a + b·|Q|², if used.
    # None when n_outer_iter == 1 or noise-fit machinery unavailable.
    noise_model_dc: Optional[dict] = None
    noise_model_h1: Optional[dict] = None
    # Which single AC harmonic this fit used (1 or 2 typically).  The
    # joint LM fits one AC harmonic at a time; this lets the persistence
    # layer write harmonic-specific graph fields (`amp_Q_h1_local`,
    # `amp_Q_h2_local`, etc.) so multiple harmonics' fits can coexist.
    ac_harmonic: int = 1
    # ──────────────────────────────────────────────────────────────
    # Identifiability diagnostics (added 2026-05-13)
    # Populated at the end of the final solve (no extra forward/adjoint
    # solves; just SVDs on the normal-equation matrices already built).
    # See notebooks/local_inference_identifiability.ipynb for usage.
    # ──────────────────────────────────────────────────────────────
    # Singular value spectra of the per-harmonic normal-equation
    # matrices M_n = T_n^H W T_n (gauge-pinned column removed).  The
    # ratio sv[0]/sv[-1] is exactly `cond_DC` / `cond_H1` above; the
    # FULL spectrum lets you count near-null directions.
    sv_DC: Optional[np.ndarray] = None
    sv_H1: Optional[np.ndarray] = None
    # Count of singular values below max(sv) × tolerance.
    # n_null_*: < 1e-12·max  (essentially-zero, machine precision)
    # n_marginal_*: < 1e-8·max  (numerically unstable but not strictly zero)
    n_null_DC: int = 0
    n_null_H1: int = 0
    n_marginal_DC: int = 0
    n_marginal_H1: int = 0
    # Per-boundary-node coupling strength to the interior network.
    # G_attach[b] = Σ_{e=(b,i): i∈interior} G_e  (DC conductance only).
    # Nodes with G_attach below ~max/100 are weakly coupled and their
    # pressures are essentially unidentifiable — candidates for
    # reclassification as interior nodes.  Units: m³/(Pa·s).
    G_attach_by_node: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Subgraph extraction
# ──────────────────────────────────────────────────────────────────


def extract_tile_subgraph_spatial(graph, tile_id: int,
                                     *, padding_frac: float = 0.0,
                                     inset_frac: float = 0.0,
                                     restrict_to_tile_piv_nodes: bool = False,
                                     drop_dangling_boundaries: bool = False):
    """Spatial rectangle carve of a tile's subgraph.

    Defines the tile region as the (axis-aligned) bbox of all nodes
    touching a PIV record for this ``tile_id``, optionally padded or
    inset.  Then walks every graph edge:

      • both endpoints inside the rectangle  ⇒  interior edge,
                                                both nodes interior.
      • one inside, one outside              ⇒  edge kept; the outside
                                                node becomes a boundary
                                                node (its pressure gets
                                                fitted by the inverse).
      • both outside                          ⇒  edge dropped.

    Parameters
    ----------
    padding_frac : float
        Fractional outward pad of the bbox (positive grows it).
    inset_frac : float
        Fractional inward shrink of the bbox.  Mutually exclusive with
        padding_frac in practice; if both are nonzero, both are applied
        (net = padding − inset).  The notebook uses 0.05 (5% inset).
    restrict_to_tile_piv_nodes : bool
        If True, only edges between nodes that touch a tile-N PIV record
        are eligible.  This excludes anatomy edges that pass through the
        bbox between non-tile vessels (the source of "data-less boundary
        nodes" in production carves).  The notebook uses True.
    """
    # 1. Compute bbox from PIV-touching nodes for this tile_id.
    piv_nodes = set()
    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv', []) or []
        if any(m.get('tile_id') == tile_id for m in piv):
            piv_nodes.add(u); piv_nodes.add(v)
    if not piv_nodes:
        raise ValueError(f"No PIV edges for tile_id={tile_id}")

    xs = [float(graph.nodes[n].get('x', 0.0)) for n in piv_nodes]
    ys = [float(graph.nodes[n].get('y', 0.0)) for n in piv_nodes]
    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))
    span_x = x_max - x_min
    span_y = y_max - y_min
    pad_x = float(padding_frac) * span_x - float(inset_frac) * span_x
    pad_y = float(padding_frac) * span_y - float(inset_frac) * span_y
    x_lo = x_min - pad_x; x_hi = x_max + pad_x
    y_lo = y_min - pad_y; y_hi = y_max + pad_y

    def _inside(n):
        x = graph.nodes[n].get('x', float('nan'))
        y = graph.nodes[n].get('y', float('nan'))
        try:
            x = float(x); y = float(y)
        except (TypeError, ValueError):
            return False
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return x_lo <= x <= x_hi and y_lo <= y <= y_hi

    # 2. Walk edges; classify each one's endpoints.  When
    # restrict_to_tile_piv_nodes is set, only edges between two PIV-
    # touching nodes are eligible (matches the notebook's carve).
    edges_in = []
    interior = set()
    boundary = set()
    for u, v in graph.edges():
        if restrict_to_tile_piv_nodes and not (u in piv_nodes and v in piv_nodes):
            continue
        u_in = _inside(u)
        v_in = _inside(v)
        if u_in and v_in:
            edges_in.append((u, v))
            interior.add(u); interior.add(v)
        elif u_in and not v_in:
            edges_in.append((u, v))
            interior.add(u); boundary.add(v)
        elif v_in and not u_in:
            edges_in.append((u, v))
            interior.add(v); boundary.add(u)
        # else: both outside — skip

    # A node should never be classified as both interior and boundary
    # (interior wins by construction since we only add to boundary
    # when the OTHER endpoint is inside).  Defensive cleanup just in
    # case of weird coordinate ties.
    boundary -= interior

    # Global heart sources/sinks that fall inside the rectangle stay
    # as interior unknowns; if they happen to fall on the perimeter
    # via being an outside-endpoint of a crossing edge they're already
    # in `boundary`.  Either way the inference handles them fine.

    # ── demote_leaky (added 2026-05-18) ──
    # An interior node X with a crossing edge to a boundary node Y
    # carries compliance through that edge.  Production-pre-fix kept X
    # interior, which means KCL is enforced at X: ΣQ(X) = 0.  But the
    # global TL forward also has compliance OUTSIDE the carve that
    # propagates flux into Y, and the relationship between Y's
    # pressure and that external flux is not in the local model.  This
    # forces KCL at X to absorb the model error → systematic bias in
    # D̂ (~7% per-tile on noiseless synthetic, validated 2026-05-18).
    #
    # Fix: promote any interior node with a graph-neighbour OUTSIDE
    # the bbox to boundary.  This relaxes KCL at X and gives it a free
    # pressure that can absorb the local-model misspecification.
    # Equivalent to the notebook's `demote_leaky` step in
    # `select_subgraph`.
    leaky = {n for n in interior
             if any((not _inside(nb))
                    for nb in graph.neighbors(n))}
    interior -= leaky
    boundary |= leaky

    # Optionally drop "dangling" boundary nodes: nodes whose only
    # carve-edges go to other boundary nodes (no interior neighbor in
    # the carve edge set).  These nodes carry no information about
    # interior dynamics — their P_b is a free parameter constrained
    # only by edges to other free P_b parameters.  Pure plumbing that
    # inflates the Hessian without constraining D.  Mirrors the
    # "dangling-boundary drop" in `extract_tile_subgraph` (topology
    # variant).  Runs after demote_leaky so newly-promoted nodes are
    # also subject to the filter.
    if drop_dangling_boundaries and boundary:
        edges_set = {(u, v) for u, v in edges_in}
        edges_set |= {(v, u) for u, v in edges_in}
        keep_bnd = set()
        for n in boundary:
            for nb in graph.neighbors(n):
                if nb in interior and ((n, nb) in edges_set):
                    keep_bnd.add(n)
                    break
        dropped = boundary - keep_bnd
        if dropped:
            edges_in = [(u, v) for u, v in edges_in
                         if u not in dropped and v not in dropped]
            boundary = keep_bnd

    all_nodes = sorted(interior | boundary)
    return edges_in, all_nodes, sorted(boundary), sorted(interior)


def extract_tile_subgraph(graph, tile_id: int):
    """Identify the tile's subnetwork.

    Returns
    -------
    edges_in_tile : list[(u, v)]
        Edges with at least one PIV measurement matching `tile_id`.
    all_nodes : list
        Every node touched by an edge in `edges_in_tile`.
    boundary_nodes : list
        Nodes that are either:
          (a) connected to an edge OUTSIDE the tile, or
          (b) globally a source/sink (boundary_type ∈ {'source', 'sink'})
    interior_nodes : list
        Everything else in `all_nodes`.
    """
    edges_in = []
    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv', []) or []
        if any(m.get('tile_id') == tile_id for m in piv):
            edges_in.append((u, v))

    nodes_in = set()
    for u, v in edges_in:
        nodes_in.add(u)
        nodes_in.add(v)

    edges_set = set()
    for u, v in edges_in:
        edges_set.add((u, v))
        edges_set.add((v, u))

    # Step 1: candidate boundary set — nodes connecting outside the tile
    # OR globally source/sink.
    cand_bdry = set()
    for n in nodes_in:
        is_bdry = False
        for nb in graph.neighbors(n):
            if (n, nb) not in edges_set:
                is_bdry = True
                break
        if not is_bdry:
            try:
                bt = graph.nodes[n].get('boundary_type', None)
            except Exception:
                bt = None
            if bt in ('source', 'sink'):
                is_bdry = True
        if is_bdry:
            cand_bdry.add(n)
    cand_int = nodes_in - cand_bdry

    # Step 2: keep only candidate boundary nodes that are adjacent (via a
    # tile edge) to at least one interior node.  A "dangling" boundary
    # node — connected only to other boundaries — carries no information
    # about the interior pressure solve and should be dropped from the
    # inference set entirely (along with its incident edges).
    boundary_filt = set()
    for n in cand_bdry:
        for nb in graph.neighbors(n):
            if nb in cand_int and ((n, nb) in edges_set):
                boundary_filt.add(n)
                break

    dropped = cand_bdry - boundary_filt   # dangling boundaries to discard
    if dropped:
        edges_in = [(u, v) for u, v in edges_in
                     if u not in dropped and v not in dropped]
        # Recompute the surviving node set
        nodes_in = set()
        for u, v in edges_in:
            nodes_in.add(u)
            nodes_in.add(v)

    # Final classification on the surviving nodes
    boundary_nodes = sorted(boundary_filt & nodes_in)
    interior_nodes = sorted(cand_int & nodes_in)
    return edges_in, sorted(nodes_in), boundary_nodes, interior_nodes


# ──────────────────────────────────────────────────────────────────
# Admittance + transfer matrices
# ──────────────────────────────────────────────────────────────────


def _edge_geometry(data, px_size_m: float):
    """Return (R_m, L_m) for a graph edge data dict.

    Local inference uses **measured anatomy**, not adaptation-simulation
    output.  `radius_adapted_m` is the post-Chatterjee-Katifori flow-
    adapted radius from the global sim — it grows heart-boundary stub
    edges to absurd values (e.g. R≈60 px on a vessel the segmentation
    says is 3 px wide) because that's where the sim dumps inflow
    current.  Using it for local inference puts those synthetic radii
    into G = πR⁴/(8μL) and the artery-sized G values then dominate
    cond(L_int), corrupt the boundary regression, and ruin DC fit.

    Preference order:
      1. `radius_px_true` (anisotropy-corrected real anatomy)
      2. `radius` (raw centerline radius from distance transform)
      3. `radius_adapted_m` (last resort — sim output, may be wrong)

    Same heuristic: anything <1e-3 is assumed already-meters; otherwise
    pixels × px_size_m.
    """
    R_raw = data.get('radius_px_true')
    if R_raw is None:
        R_raw = data.get('radius')
    if R_raw is None:
        R_raw = data.get('radius_adapted_m', 1.0)
    if hasattr(R_raw, 'item'):
        R_raw = R_raw.item()
    R_v = float(R_raw)
    R_m = R_v if R_v < 1e-3 else R_v * px_size_m

    L_raw = data.get('length_true') or data.get('length', 1.0)
    if hasattr(L_raw, 'item'):
        L_raw = L_raw.item()
    L_v = float(L_raw)
    L_m = L_v if L_v < 1e-3 else L_v * px_size_m

    return R_m, L_m


def _build_admittance_system(
    graph, edges_in, boundary_nodes, interior_nodes,
    D, mu, f0_hz, harmonics, px_size_m,
):
    """For each harmonic n ∈ {0} ∪ harmonics, build:

        L[n_int × n_int]  : interior Laplacian (complex)
        B[n_int × n_bnd]  : boundary forcing
        edge_G            : per-edge series conductance G_e (real, n-indep)

    such that interior pressures satisfy   L · P_int = −B · P_bnd
    and per-edge flow is   Q_e = G_e · (P_u − P_v).

    Pi-model: series G plus shunt jωC/2 to ground at each endpoint.

    Returns dict harmonic → (L, B, edge_G, ω).
    """
    n_int = len(interior_nodes)
    n_bnd = len(boundary_nodes)
    interior_idx = {n: i for i, n in enumerate(interior_nodes)}
    boundary_idx = {n: i for i, n in enumerate(boundary_nodes)}

    edge_data = {}
    for u, v in edges_in:
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0 or not np.isfinite(R_m) or not np.isfinite(L_m):
            edge_data[(u, v)] = (0.0, 0.0)
            continue
        G = float(np.pi * R_m ** 4 / (8.0 * mu * L_m))
        # Areal-distensibility convention: c = πR²D so total edge
        # compliance C = c·L = πR²·D·L.  (Pre-2026-05-18: 2πR²·D·L.)
        C = float(np.pi * R_m ** 2 * D * L_m)
        edge_data[(u, v)] = (G, C)

    out = {}
    harm_set = sorted(set([0] + list(harmonics)))
    for n_harm in harm_set:
        omega = 2.0 * np.pi * float(n_harm) * f0_hz
        L_mat = np.zeros((n_int, n_int), dtype=complex)
        B_mat = np.zeros((n_int, n_bnd), dtype=complex)

        for u, v in edges_in:
            G, C = edge_data[(u, v)]
            if G == 0:
                continue
            u_int = u in interior_idx
            v_int = v in interior_idx
            # Series contribution
            if u_int and v_int:
                iu, iv = interior_idx[u], interior_idx[v]
                L_mat[iu, iu] += G
                L_mat[iv, iv] += G
                L_mat[iu, iv] -= G
                L_mat[iv, iu] -= G
            elif u_int and not v_int:
                iu, jb = interior_idx[u], boundary_idx[v]
                L_mat[iu, iu] += G
                B_mat[iu, jb] -= G
            elif v_int and not u_int:
                iv, jb = interior_idx[v], boundary_idx[u]
                L_mat[iv, iv] += G
                B_mat[iv, jb] -= G
            # both boundary: no contribution to interior solve
            # (the edge's flow is determined directly by P_u, P_v)

            # Shunt at each endpoint (AC only)
            if n_harm > 0 and omega > 0 and C > 0:
                Y_shunt = 1j * omega * C / 2.0
                for endpt in (u, v):
                    if endpt in interior_idx:
                        L_mat[interior_idx[endpt],
                               interior_idx[endpt]] += Y_shunt
                    # boundary endpoints: shunt absorbed into BC

        edge_G = {e: edge_data[e][0] for e in edges_in}
        out[n_harm] = (L_mat, B_mat, edge_G, omega)
    return out


def _compute_transfer_matrices(L_B_dict, edges_in, boundary_nodes,
                                  interior_nodes, *, verbose=False):
    """Solve L · P_int = −B for each unit boundary input → assemble
    transfer matrix T per harmonic.

    T[edge_idx, k] = G_e · (P_u(when P_bnd = e_k) − P_v(when P_bnd = e_k))

    Returns dict harmonic → T (n_edges × n_bnd, complex).

    Numerical stability: at SI flow scales, L's diagonal entries are
    ~G_e ≈ 1e-15 m³/(Pa·s).  numpy.linalg.solve on such tiny absolute
    values can underflow or trigger denormal-float performance issues.
    We rescale rows of [L | -B] by `1/median(|diag(L)|)` before solving
    — algebraically identical (each row equation is scaled by the same
    constant on both sides) but moves the system into O(1) magnitudes
    where the LAPACK solver is well-behaved.

    Rank deficiency: at DC there is no shunt admittance, so the interior
    Laplacian block can be singular if some interior subcomponent has
    no edge to any boundary node (floating island).  We use lstsq with
    an rcond cutoff, which returns the minimum-norm solution: floating
    islands get P=0 (physically correct — no DC drive ⇒ no DC flow),
    while well-connected interiors get the unique correct answer.

    Set `verbose=True` to print per-harmonic diagnostics (L conditioning,
    P_int_basis range, T row-norm distribution).
    """
    n_bnd = len(boundary_nodes)
    n_edges = len(edges_in)
    interior_idx = {n: i for i, n in enumerate(interior_nodes)}
    boundary_idx = {n: i for i, n in enumerate(boundary_nodes)}

    out = {}
    for n_harm, (L_mat, B_mat, edge_G, omega) in L_B_dict.items():
        if len(interior_nodes) == 0:
            P_int_basis = np.zeros((0, n_bnd), dtype=complex)
        else:
            # Row-rescale L and B so the linear solve operates on O(1)
            # magnitudes regardless of the absolute G scale.
            diag_abs = np.abs(np.diag(L_mat))
            scale = np.median(diag_abs[diag_abs > 0]) if np.any(diag_abs > 0) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            L_scaled = L_mat / scale
            B_scaled = B_mat / scale
            # lstsq with rcond cutoff: handles rank-deficient L (DC with
            # floating interior islands) by returning min-norm solution.
            try:
                P_int_basis, _res, rank, _sv = np.linalg.lstsq(
                    L_scaled, -B_scaled, rcond=1e-12)
            except np.linalg.LinAlgError:
                P_int_basis = np.zeros(
                    (len(interior_nodes), n_bnd), dtype=complex)
                rank = 0
            if verbose:
                cond = float(np.linalg.cond(L_scaled))
                p_norm_per_col = np.linalg.norm(P_int_basis, axis=0)
                n_zero_cols = int(np.sum(p_norm_per_col < 1e-30))
                print(f"    harmonic n={n_harm}:  scale={scale:.3e}, "
                      f"cond(L_scaled)={cond:.2e}, "
                      f"rank={rank}/{L_scaled.shape[0]}, "
                      f"|P_int_basis| max={np.abs(P_int_basis).max():.3e}, "
                      f"zero columns={n_zero_cols}/{n_bnd}")

        T = np.zeros((n_edges, n_bnd), dtype=complex)
        for ei, (u, v) in enumerate(edges_in):
            G = edge_G[(u, v)]
            if G == 0:
                continue
            for k in range(n_bnd):
                if u in interior_idx:
                    P_u = P_int_basis[interior_idx[u], k]
                else:
                    P_u = 1.0 if boundary_idx[u] == k else 0.0
                if v in interior_idx:
                    P_v = P_int_basis[interior_idx[v], k]
                else:
                    P_v = 1.0 if boundary_idx[v] == k else 0.0
                T[ei, k] = G * (P_u - P_v)
        if verbose:
            row_norms = np.linalg.norm(T, axis=1)
            n_zero_rows = int(np.sum(row_norms < 1e-30))
            interior_only = sum(1 for u, v in edges_in
                                 if u in interior_idx and v in interior_idx)
            print(f"      T (n_edges×n_bnd={n_edges}×{n_bnd}): "
                  f"|T| range = [{row_norms[row_norms > 0].min():.3e}, "
                  f"{row_norms.max():.3e}], "
                  f"zero rows = {n_zero_rows}/{n_edges} "
                  f"(interior-only edges: {interior_only})")
        out[n_harm] = T
    return out


# ──────────────────────────────────────────────────────────────────
# Measurement extraction
# ──────────────────────────────────────────────────────────────────


_NL_PER_S_TO_M3_PER_S = 1.0e-12   # 1 nL/s = 10⁻¹² m³/s


def _extract_measured_flows(graph, edges_in, tile_id, harmonics):
    """For each edge in the tile, return measured (DC, H1, ...) flows
    in (u,v)-tuple convention, **converted to SI (m³/s)**.

    PIV records store Q in nL/s; the admittance system uses SI throughout
    (G ≈ 10⁻¹⁵ m³/(Pa·s)).  Without this conversion, the WLS fit
    minimises `|Q_meas (nL/s) − T·P (m³/s)|²` in mismatched units and
    absorbs the 10¹² factor into P, giving meaningless ~10¹⁶ Pa
    boundary pressures.  Converting Q to SI here keeps both sides
    consistent so the recovered P is in physical Pa.

    Reuses inference._meas_phasors_for_edge for harmonic refit + sign
    correction (with edge-level flow_from override authoritative)."""
    from .inference import _meas_phasors_for_edge

    n_edges = len(edges_in)
    Q_dc = np.full(n_edges, np.nan, dtype=float)
    Q_hn = {n: np.full(n_edges, np.nan, dtype=complex)
            for n in harmonics}

    for ei, (u, v) in enumerate(edges_in):
        d = graph.edges[u, v]
        piv = d.get('measurements_piv', []) or []
        m_ref = next((m for m in piv
                       if m.get('tile_id') == tile_id), None)
        if m_ref is None:
            continue
        try:
            Q_dc_uv, Q_hn_dict, sign_uv_meas = _meas_phasors_for_edge(
                (u, v), m_ref, harmonics=tuple(harmonics))
        except Exception:
            continue

        # Edge-level direction takes priority over per-measurement
        edge_ff = d.get('flow_from')
        edge_ft = d.get('flow_to')
        if edge_ff == u and edge_ft == v:
            sign_uv = +1.0
        elif edge_ff == v and edge_ft == u:
            sign_uv = -1.0
        else:
            sign_uv = sign_uv_meas
        corr = sign_uv * (sign_uv_meas if sign_uv_meas != 0 else 1.0)

        if np.isfinite(Q_dc_uv):
            Q_dc[ei] = corr * Q_dc_uv * _NL_PER_S_TO_M3_PER_S
        for n in harmonics:
            qhn = Q_hn_dict.get(n)
            if qhn is not None and np.isfinite(qhn):
                Q_hn[n][ei] = corr * qhn * _NL_PER_S_TO_M3_PER_S

    return Q_dc, Q_hn


# ──────────────────────────────────────────────────────────────────
# Linear solve for boundary pressures given D
# ──────────────────────────────────────────────────────────────────


def _joint_lm_inner_loop(
    graph, edges_in, boundary_nodes, interior_nodes,
    pin_idx, pin_node,
    Q_dc, Q_hn, valid_dc, valid_h1, n_dc, n_h1,
    spec, f0_hz, px_size_m, verbose=False,
    sigma_dc_e: Optional[np.ndarray] = None,
    sigma_h1_e: Optional[np.ndarray] = None,
):
    """Joint Levenberg-Marquardt over (D, P_DC, P_H1).

    Replaces the alternating LS inner loop.  Single stacked normal
    equation per iteration; quadratic convergence near the minimum.

    Parameter vector θ ∈ ℝ^N:
        θ[0]                              = D
        θ[1 : 1+n_p_dc]                    = P_DC at unpinned boundaries
        θ[1+n_p_dc : 1+n_p_dc+n_bnd]       = Re(P_H1)  at ALL boundaries
        θ[1+n_p_dc+n_bnd : 1+n_p_dc+2n_bnd] = Im(P_H1) at ALL boundaries
    where n_p_dc = n_bnd - 1.

    GAUGE CONVENTION (corrected 2026-05-18 — was the AC-pin gauge bug):
      - DC: pin one boundary phasor to 0 for the gauge degree of
        freedom (constant-shift invariance of the Kirchhoff Laplacian).
      - AC: NO pin.  The shunt admittance jωc breaks the constant-
        shift symmetry — there is no AC gauge.  All n_bnd boundary
        AC phasors are free parameters.
      Pre-fix code shared a single keep_idx between DC and AC, dropping
      the pin column from the AC Jacobian.  This forced the AC fit into
      an (n_bnd-1)-dimensional subspace that can't span the actual H1
      patterns produced by the global TL forward, causing systematic
      D̂ bias and |Q_pred|/|Q_meas| < 1 ratios across tiles.

    Returns the same outputs `infer_local`'s alternating loop returns:
        (D, P_DC, P_H1, cov_DC, cov_H1, cond_DC, cond_H1,
         history, converged, sigma_D_proxy)
    """
    n_bnd = len(boundary_nodes)
    n_p_dc = max(n_bnd - 1, 0)        # DC: drop pin column for gauge
    n_p_ac = n_bnd                     # AC: no pin (shunt admittance breaks gauge)
    keep_idx = np.array([k for k in range(n_bnd) if k != pin_idx])
    # Backwards compat: many local refs below still call this `n_p`.
    # That variable now means specifically the DC column count.
    n_p = n_p_dc

    # Pick the single AC harmonic in spec.harmonics.  Currently the
    # joint LM only supports one AC harmonic at a time (multi-harmonic
    # joint fit would need a wider θ vector — path B in the H2 cross-
    # check discussion).  For the dual-fit comparison test, we just
    # need to support any single harmonic, so we extract it here
    # instead of hard-coding to 1.
    ac_harmonics = [h for h in spec.harmonics if h > 0]
    ac_n = ac_harmonics[0] if ac_harmonics else None
    has_h1 = ac_n is not None and n_h1 > 0
    has_dc = n_dc > 0

    # Pack θ_init.  D init from spec; P at zero (gauge).
    # Layout: [D, P_DC_keep (n_p_dc), Re(P_H1_full) (n_p_ac), Im(P_H1_full) (n_p_ac)]
    D = float(spec.D_init)
    theta_size = 1 + n_p_dc + (2 * n_p_ac if has_h1 else 0)
    theta = np.zeros(theta_size, dtype=float)
    theta[0] = D

    # ── Warm-start P_b via profile-LS at D_init (Bug fix 4, 2026-05-18).
    # Previously θ started with P = 0, which means the first LM Newton
    # step is computed at a wildly wrong (D, P_b) point — Jacobian
    # directions point in P_b-correcting directions of length O(|Q|),
    # which dominate the joint step and can drag D away from truth even
    # when D_init = D_true.  Pre-solving for P_b at the current D in
    # closed-form weighted LS gives the LM a reasonable starting point
    # where residuals are already small in the P_b direction; the LM
    # then just refines D.
    if sigma_dc_e is not None:
        _sig_dc_for_ws = np.asarray(sigma_dc_e, dtype=float)
    else:
        _sig_dc_for_ws = np.ones(len(edges_in), dtype=float)
    if sigma_h1_e is not None:
        _sig_h1_for_ws = np.asarray(sigma_h1_e, dtype=float)
    else:
        _sig_h1_for_ws = np.ones(len(edges_in), dtype=float)
    try:
        _T_init = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            float(D), spec.mu, f0_hz, spec.harmonics, px_size_m)
        _T_init = _compute_transfer_matrices(
            _T_init, edges_in, boundary_nodes, interior_nodes,
            verbose=False)
        if has_dc and n_p_dc > 0 and int(valid_dc.sum()) >= n_p_dc:
            _w = 1.0 / np.where(_sig_dc_for_ws[valid_dc] > 0,
                                 _sig_dc_for_ws[valid_dc], 1.0)
            _A = (_T_init[0][:, keep_idx][valid_dc].real
                   * _w[:, None])
            _b = Q_dc[valid_dc].real * _w
            _P_dc, *_ = np.linalg.lstsq(_A, _b, rcond=1e-10)
            theta[1:1 + n_p_dc] = _P_dc
        if has_h1 and n_p_ac > 0 and int(valid_h1.sum()) >= n_p_ac:
            _w = 1.0 / np.where(_sig_h1_for_ws[valid_h1] > 0,
                                 _sig_h1_for_ws[valid_h1], 1.0)
            _A = _T_init[ac_n][valid_h1] * _w[:, None]
            _b = Q_hn[ac_n][valid_h1] * _w
            _P_h, *_ = np.linalg.lstsq(_A, _b, rcond=1e-10)
            theta[1 + n_p_dc:1 + n_p_dc + n_p_ac] = _P_h.real
            theta[1 + n_p_dc + n_p_ac:1 + n_p_dc + 2 * n_p_ac] = _P_h.imag
    except Exception:
        # If warm-start fails (e.g. rank-deficient T at D_init), fall
        # back silently to P=0 init — same as pre-fix behaviour.
        pass

    # ── Boundary smoothness operator L_b (built once, used by the
    # smooth_h1 prior path inside the LM loop) ──
    # Order boundary nodes by angle around the centroid; connect each
    # to its cyclic next neighbour.  Each row of L_b is a unit
    # difference operator (+1 at node i, −1 at next node i+1 mod n_b).
    # ‖L_b · P_b‖² penalises zig-zag oscillations between adjacent
    # nodes without forcing the bulk scale toward zero.  L_b_reduced
    # is L_b with the pin column removed (since pin is fixed at 0,
    # the pin entry of P_b never enters θ; the pin's row in L_b
    # naturally references it as 0).
    prior_mode = getattr(spec, 'prior_mode', 'magnitude')
    L_b_reduced_HtH = None
    if prior_mode == 'smooth_h1' and n_p > 0:
        xs_b_arr = np.array([float(graph.nodes[n].get('x', 0.0))
                             for n in boundary_nodes])
        ys_b_arr = np.array([float(graph.nodes[n].get('y', 0.0))
                             for n in boundary_nodes])
        cx_b_ = float(np.nanmean(xs_b_arr))
        cy_b_ = float(np.nanmean(ys_b_arr))
        ang_b = np.arctan2(ys_b_arr - cy_b_, xs_b_arr - cx_b_)
        order_b = np.argsort(ang_b)
        n_bnd_full = len(boundary_nodes)
        L_b_full = np.zeros((n_bnd_full, n_bnd_full), dtype=float)
        for ii in range(n_bnd_full):
            jj = order_b[ii]
            kk = order_b[(ii + 1) % n_bnd_full]
            L_b_full[ii, jj] = +1.0
            L_b_full[ii, kk] = -1.0
        # Drop pin column → P_b_full[keep_idx] is the reduced vector
        # we actually optimise.  Pin's contribution is 0 (P_pin = 0).
        L_b_reduced = L_b_full[:, keep_idx]      # (n_bnd_full, n_p)
        L_b_reduced_HtH = L_b_reduced.T @ L_b_reduced   # (n_p, n_p)

    # LM state
    mu = float(spec.lm_mu0)
    mu_factor = float(spec.lm_factor)

    history: list = []
    converged = False
    chi2_prev = float('inf')
    # Per-edge noise σ.  If the caller passed arrays we use them as-is
    # (the FGLS outer loop in infer_local does this).  Otherwise we'll
    # set them after the initial forward eval below using scalar
    # std(r) (back-compat).  σ shapes match the full edges_in length;
    # only entries at valid_dc / valid_h1 are read.
    if sigma_dc_e is None:
        sigma_dc = np.ones(len(edges_in)) * 1.0
    else:
        sigma_dc = np.asarray(sigma_dc_e, dtype=float).copy()
    if sigma_h1_e is None:
        sigma_h1 = np.ones(len(edges_in)) * 1.0
    else:
        sigma_h1 = np.asarray(sigma_h1_e, dtype=float).copy()

    # ── Pre-compute per-iter weights & store last accepted Ts ──
    def build_T(D_val):
        ab = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            float(D_val), spec.mu, f0_hz, spec.harmonics, px_size_m)
        return _compute_transfer_matrices(
            ab, edges_in, boundary_nodes, interior_nodes,
            verbose=False)

    def unpack(theta):
        Dv = float(theta[0])
        P_DC_full = np.zeros(n_bnd, dtype=complex)
        P_H1_full = np.zeros(n_bnd, dtype=complex)
        P_DC_full[keep_idx] = theta[1:1 + n_p_dc]
        if has_h1:
            # AC is fully unpinned — load into ALL n_bnd boundary slots.
            re = theta[1 + n_p_dc:1 + n_p_dc + n_p_ac]
            im = theta[1 + n_p_dc + n_p_ac:1 + n_p_dc + 2 * n_p_ac]
            P_H1_full[:] = re + 1j * im
        return Dv, P_DC_full, P_H1_full

    def forward_residual(theta):
        """Return (r_packed, r_dc_for_chi2, r_h1_for_chi2, T_all)."""
        Dv, P_DC_full, P_H1_full = unpack(theta)
        T_all = build_T(Dv)
        r_parts = []
        T_DC = T_all[0]
        if has_dc:
            r_DC = (Q_dc - (T_DC @ P_DC_full).real)
            r_parts.append(r_DC[valid_dc])
        else:
            r_DC = None
        if has_h1:
            T_H1 = T_all[ac_n]
            r_H1 = Q_hn[ac_n] - (T_H1 @ P_H1_full)
            r_parts.append(r_H1[valid_h1].real)
            r_parts.append(r_H1[valid_h1].imag)
        else:
            r_H1 = None
        r_packed = np.concatenate(r_parts) if r_parts else np.zeros(0)
        return r_packed, r_DC, r_H1, T_all

    def chi2(r_DC, r_H1, sigma_dc_v, sigma_h1_v):
        s = 0.0
        if has_dc and r_DC is not None:
            sd = sigma_dc_v[valid_dc]
            s += float(np.sum((r_DC[valid_dc] / sd) ** 2))
        if has_h1 and r_H1 is not None:
            sh = sigma_h1_v[valid_h1]
            s += float(np.sum(np.abs(r_H1[valid_h1] / sh) ** 2))
        return s

    # Initial forward.  σ_*_e is either caller-provided (FGLS outer
    # loop) or constructed here from initial-residual std as a scalar
    # repeated over edges (back-compat / homoscedastic init).
    r_packed, r_DC, r_H1, T_all = forward_residual(theta)
    if sigma_dc_e is None and has_dc and r_DC is not None \
            and r_DC[valid_dc].size > 1:
        sigma_dc[:] = max(float(np.std(r_DC[valid_dc])), 1e-30)
    if sigma_h1_e is None and has_h1 and r_H1 is not None \
            and r_H1[valid_h1].size > 1:
        sigma_h1[:] = max(float(np.std(np.abs(r_H1[valid_h1]))), 1e-30)
    # Floor σ to avoid divide-by-near-zero on quiet edges.
    sigma_dc = np.maximum(sigma_dc, 1e-30)
    sigma_h1 = np.maximum(sigma_h1, 1e-30)
    chi2_prev = chi2(r_DC, r_H1, sigma_dc, sigma_h1)

    n_p_total = theta.size
    consec_rejects = 0
    consec_drift = 0
    n_drift_window = int(getattr(spec, 'n_drift_window', 0) or 0)
    drift_bail = False
    for it in range(int(spec.max_iter)):
        D, P_DC_full, P_H1_full = unpack(theta)

        # ── Build Jacobian J at current θ ──
        # ∂T/∂D via finite difference, shared across the iteration
        D1 = D * (1.0 + spec.eps_D)
        T_pert = build_T(D1)
        dD = D1 - D
        dT = {n: (T_pert[n] - T_all[n]) / dD for n in T_all}

        # Stack rows: DC valid + H1 re + H1 im
        rows_dc = int(valid_dc.sum()) if has_dc else 0
        rows_h1 = int(valid_h1.sum()) if has_h1 else 0
        m_total = rows_dc + 2 * rows_h1
        J = np.zeros((m_total, n_p_total), dtype=float)
        r_vec = np.zeros(m_total, dtype=float)

        # Per-edge inverse-σ weights for the valid rows.
        w_dc = (1.0 / sigma_dc[valid_dc]) if has_dc else None
        w_h1 = (1.0 / sigma_h1[valid_h1]) if has_h1 else None

        row = 0
        if has_dc:
            T_DC = T_all[0]
            T_DC_re = T_DC.real
            dT_DC = dT[0]
            jac_D_dc = -(dT_DC @ P_DC_full).real
            J[row:row + rows_dc, 0] = jac_D_dc[valid_dc] * w_dc
            # DC: keep_idx (drop pin column for gauge)
            for k_local, k in enumerate(keep_idx):
                J[row:row + rows_dc, 1 + k_local] = (
                    -T_DC_re[valid_dc, k] * w_dc)
            r_vec[row:row + rows_dc] = (
                (Q_dc - (T_DC @ P_DC_full).real)[valid_dc] * w_dc)
            row += rows_dc

        if has_h1:
            T_H1 = T_all[ac_n]
            dT_H1 = dT[ac_n]
            jac_D_h1 = -(dT_H1 @ P_H1_full)  # complex
            J[row:row + rows_h1, 0] = jac_D_h1[valid_h1].real * w_h1
            J[row + rows_h1:row + 2 * rows_h1, 0] = (
                jac_D_h1[valid_h1].imag * w_h1)
            T_H1_re = T_H1.real
            T_H1_im = T_H1.imag
            # AC: iterate over ALL n_bnd boundary columns (no gauge pin).
            # Column layout: re cols start at 1+n_p_dc, im cols at
            # 1+n_p_dc+n_p_ac.
            ac_re_off = 1 + n_p_dc
            ac_im_off = 1 + n_p_dc + n_p_ac
            for k in range(n_p_ac):
                col_re = ac_re_off + k
                col_im = ac_im_off + k
                J[row:row + rows_h1, col_re] = (
                    -T_H1_re[valid_h1, k] * w_h1)
                J[row:row + rows_h1, col_im] = (
                    T_H1_im[valid_h1, k] * w_h1)
                J[row + rows_h1:row + 2 * rows_h1, col_re] = (
                    -T_H1_im[valid_h1, k] * w_h1)
                J[row + rows_h1:row + 2 * rows_h1, col_im] = (
                    -T_H1_re[valid_h1, k] * w_h1)
            r_H1_full = (Q_hn[ac_n] - (T_H1 @ P_H1_full))[valid_h1]
            r_vec[row:row + rows_h1] = r_H1_full.real * w_h1
            r_vec[row + rows_h1:row + 2 * rows_h1] = (
                r_H1_full.imag * w_h1)
            row += 2 * rows_h1

        # ── Normal equations with Tikhonov prior on P + LM damping ──
        # Gauss-Newton step minimises ‖r + J δ‖² ⇒ δ = -(J^T J)^(-1) J^T r.
        # With J = ∂r/∂θ = -T (data − model convention), -J^T r is the
        # descent direction.  We negate g here so solve(H, g) gives the
        # correct step direction directly.
        H = J.T @ J
        g = -(J.T @ r_vec)

        # Tikhonov: prior on P_DC and P_H1 separately.  Apply via λ added
        # to the corresponding diagonal entries of H.  This is the same
        # empirical-Bayes calibration as _solve_pressures_complex_wls,
        # done on the appropriate sub-block.
        if spec.P_scale_Pa is not None and float(spec.P_scale_Pa) > 0:
            # Per-block reference scale (90th percentile of diag(H) over
            # P_DC columns, separately for P_H1).  AC block now spans
            # n_p_ac columns each for Re/Im (was n_p under the buggy
            # gauge — corrected 2026-05-18).
            ac_re_lo = 1 + n_p_dc
            ac_re_hi = 1 + n_p_dc + n_p_ac
            ac_im_lo = ac_re_hi
            ac_im_hi = ac_im_lo + n_p_ac
            if has_dc:
                diag_dc = np.abs(np.diag(H)[1:1 + n_p_dc])
                ref_dc = (float(np.percentile(diag_dc, 90))
                          if diag_dc.size else 0.0)
                lam_dc = ref_dc / (float(spec.P_scale_Pa) ** 2)
                H[1:1 + n_p_dc, 1:1 + n_p_dc] += lam_dc * np.eye(n_p_dc)
            if has_h1:
                diag_h1_re = np.abs(np.diag(H)[ac_re_lo:ac_re_hi])
                diag_h1_im = np.abs(np.diag(H)[ac_im_lo:ac_im_hi])
                diag_h1 = np.concatenate([diag_h1_re, diag_h1_im])
                ref_h1 = (float(np.percentile(diag_h1, 90))
                          if diag_h1.size else 0.0)
                lam_h1 = ref_h1 / (float(spec.P_scale_Pa) ** 2)
                if (prior_mode == 'smooth_h1'
                        and L_b_reduced_HtH is not None):
                    # Smoothness prior on H1 (Re and Im separately).
                    # L_b_reduced was built for the n_p (=keep_idx)
                    # dimensional unpinned-DC subspace.  Under the
                    # corrected gauge AC has n_bnd columns, not n_p,
                    # so the prior dimensions mismatch.  Fall back to
                    # the magnitude prior for AC under smooth_h1 mode
                    # (DC keeps smoothing if desired; AC uses
                    # magnitude).
                    H[ac_re_lo:ac_im_hi, ac_re_lo:ac_im_hi] += \
                        lam_h1 * np.eye(2 * n_p_ac)
                else:
                    # Default magnitude prior on H1
                    H[ac_re_lo:ac_im_hi, ac_re_lo:ac_im_hi] += \
                        lam_h1 * np.eye(2 * n_p_ac)
        if spec.P_scale_Pa_fixed is not None \
                and float(spec.P_scale_Pa_fixed) > 0:
            lam_fixed = 1.0 / (float(spec.P_scale_Pa_fixed) ** 2)
            if n_p > 0:
                H[1:, 1:] += lam_fixed * np.eye(H.shape[0] - 1)
        if spec.lambda_reg > 0:
            H[1:, 1:] += float(spec.lambda_reg) \
                * np.eye(H.shape[0] - 1)

        # LM damping: H' = H + μ·diag(H)
        diag_H = np.diag(H).copy()
        diag_H = np.where(diag_H > 0, diag_H, 1.0)
        H_lm = H + mu * np.diag(diag_H)

        # Solve δ
        try:
            delta = np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H_lm, g, rcond=1e-10)[0]

        # Trial step
        theta_trial = theta + delta
        # D-step cap (Bug fix 3, 2026-05-18): |ΔD|/D ≤ 0.5 per LM iter.
        # Without this, when the LM's Newton step suggests a huge D
        # change (e.g. because the Jacobian sees a flat direction at
        # the current point), the trial D can overshoot through D≈0
        # and land at the lower positivity floor, then get stuck.
        # Capping per-step keeps the LM exploring smoothly near the
        # current scale.
        D_cap = 0.5 * abs(theta[0])
        if abs(theta_trial[0] - theta[0]) > D_cap:
            theta_trial[0] = theta[0] + np.sign(
                theta_trial[0] - theta[0]) * D_cap
        # D positivity floor
        if theta_trial[0] < 1e-12:
            theta_trial[0] = 1e-12
        if theta_trial[0] > 1.0:
            theta_trial[0] = 1.0

        r_packed_t, r_DC_t, r_H1_t, T_all_t = forward_residual(
            theta_trial)
        chi2_trial = chi2(r_DC_t, r_H1_t, sigma_dc, sigma_h1)

        # Always compute the step's relative size; we use it for both
        # the acceptance display and the convergence test.
        delta_norm = float(np.linalg.norm(delta))
        theta_norm_pre = float(np.linalg.norm(theta))
        rel_step_pre = delta_norm / max(theta_norm_pre, 1e-30)

        # Acceptance.  σ_dc and σ_h1 are NOT updated within the loop:
        # they represent noise scale (set once from initial residuals).
        # Updating them per-iter would self-normalise χ² and make
        # acceptance comparisons apples-to-oranges across iterations.
        if chi2_trial <= chi2_prev:
            accepted = True
            rel_dD = abs(theta_trial[0] - theta[0]) / max(
                abs(theta[0]), 1e-30)
            theta = theta_trial
            T_all = T_all_t
            chi2_prev = chi2_trial
            mu = max(mu / mu_factor, 1e-12)
            consec_rejects = 0
        else:
            accepted = False
            rel_dD = 0.0
            mu *= mu_factor
            consec_rejects += 1
            if mu > 1e12:   # runaway damping
                break

        # Diagnostics
        D_curr, P_DC_curr_full, P_H1_curr_full = unpack(theta)
        qratio_dc = qratio_h1 = float('nan')
        if has_dc:
            qm = float(np.sqrt(np.mean(
                Q_dc[valid_dc] ** 2))) or 1e-30
            qp = float(np.sqrt(np.mean(
                (T_all[0] @ P_DC_curr_full).real[valid_dc] ** 2)))
            qratio_dc = qp / qm
        if has_h1:
            qm = float(np.sqrt(np.mean(np.abs(
                Q_hn[ac_n][valid_h1]) ** 2))) or 1e-30
            qp = float(np.sqrt(np.mean(np.abs(
                (T_all[ac_n] @ P_H1_curr_full)[valid_h1]) ** 2)))
            qratio_h1 = qp / qm

        # Convergence metric: relative step norm.  We use the PROPOSED
        # step size (not just accepted steps), so rejection-storms near
        # the minimum still terminate cleanly.  Pure |ΔD|/D would
        # terminate prematurely on iter 1 (D column of J = 0 when P=0).
        rel_step = rel_step_pre

        history.append({
            'iter': it + 1,
            'D_before': float(D), 'D_after': float(D_curr),
            'rel_dD': float(rel_dD),
            'rel_step': float(rel_step),
            'mu': float(mu), 'accepted': bool(accepted),
            'chi2': float(chi2_prev),
            'qratio_dc': float(qratio_dc),
            'qratio_h1': float(qratio_h1),
        })
        if verbose:
            print(f"    [LM {it + 1}] D = {D:.3e} → {D_curr:.3e}  "
                  f"(|ΔD|/D = {rel_dD:.3g})  "
                  f"||δ||/||θ||={rel_step:.2e}  μ={mu:.2e} "
                  f"{'ACC' if accepted else 'REJ'}  "
                  f"χ²={chi2_prev:.3g}  "
                  f"|Q_p|/|Q_m| DC={qratio_dc:.3f} H1={qratio_h1:.3f}")
        # Convergence: any of three signals.
        # (a) proposed step is small relative to current θ (rel_step
        #     < tol_rel), whether or not it was accepted;
        # (b) consecutive rejections — LM can't make further progress
        #     because we're at a local minimum and trial χ² rises;
        # (c) tiny absolute step regardless of θ scale (numerical floor).
        # `it >= 1` guards against the iter-0 corner case where δ=0
        # would falsely trigger.
        stuck_at_min = consec_rejects >= 3
        if it >= 1 and (rel_step < spec.tol_rel or stuck_at_min):
            converged = True
            if verbose and stuck_at_min:
                print(f"    [LM {it + 1}] {consec_rejects} consecutive "
                      f"rejections → at local minimum.")
            break

        # Drift bail: ACCEPTED step but D barely moved.  Indicates the
        # prior is dominating along the D direction (χ² ridge in P_b
        # absorbs the residual; D drifts slowly but never converges).
        if n_drift_window > 0 and accepted and it >= 1:
            if rel_dD < spec.tol_rel:
                consec_drift += 1
            else:
                consec_drift = 0
            if consec_drift >= n_drift_window:
                drift_bail = True
                converged = False
                if verbose:
                    print(f"    [LM {it + 1}] D drift bail: "
                          f"{consec_drift} consecutive accepted iters "
                          f"with |ΔD|/D < tol_rel ({spec.tol_rel:.1e}). "
                          f"Prior likely dominates D direction; D̂ is "
                          f"non-identifiable at this prior strength.")
                break

    # ── Final outputs ──
    D_final, P_DC_full, P_H1_full = unpack(theta)
    # cov from posterior Hessian (no LM damping) on unpinned P columns.
    # AC blocks now span n_p_ac columns each (re, im) instead of n_p.
    P_DC_dict = None  # filled by caller from full arrays
    ac_re_lo = 1 + n_p_dc
    ac_re_hi = 1 + n_p_dc + n_p_ac
    ac_im_lo = ac_re_hi
    ac_im_hi = ac_im_lo + n_p_ac

    def _safe_cond(M):
        """Condition number robust to rank deficiency.

        Returns σ_max / σ_min over the *non-trivial* singular values
        (using a relative threshold = max·N·eps as numpy's matrix_rank
        does).  Returns inf only if the matrix is exactly zero.

        Production's bbox carve can include boundary nodes whose
        incident edges are all anatomy-only (no PIV data).  Those
        nodes contribute zero Jacobian columns to the DC/H1 block of
        H, making it rank-deficient.  np.linalg.cond on such a matrix
        returns inf.  But the rank-deficient direction is just an
        unobserved parameter — the conditioning of the OBSERVED
        subspace is still informative, so we report cond over
        nonzero singular values.
        """
        if M.size == 0:
            return float('inf')
        try:
            s = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            return float('inf')
        if len(s) == 0 or float(s[0]) <= 0:
            return float('inf')
        s_max = float(s[0])
        thr = s_max * max(M.shape) * np.finfo(M.dtype).eps
        s_pos = s[s > thr]
        if len(s_pos) == 0:
            return float('inf')
        return s_max / float(s_pos[-1])

    cond_DC = (_safe_cond(H[1:1 + n_p_dc, 1:1 + n_p_dc])
               if has_dc and n_p_dc > 0 else float('inf'))
    cond_H1 = (_safe_cond(H[ac_re_lo:ac_im_hi, ac_re_lo:ac_im_hi])
               if has_h1 and n_p_ac > 0 else float('inf'))
    # Full singular-value spectra of the per-harmonic normal-equation
    # blocks — gives the dimension of the null/marginal subspaces.
    if has_dc and n_p_dc > 0:
        try:
            sv_DC = np.linalg.svd(
                H[1:1 + n_p_dc, 1:1 + n_p_dc], compute_uv=False)
        except np.linalg.LinAlgError:
            sv_DC = np.array([])
    else:
        sv_DC = np.array([])
    if has_h1 and n_p_ac > 0:
        try:
            sv_H1 = np.linalg.svd(
                H[ac_re_lo:ac_im_hi, ac_re_lo:ac_im_hi],
                compute_uv=False)
        except np.linalg.LinAlgError:
            sv_H1 = np.array([])
    else:
        sv_H1 = np.array([])
    # σ_D from inverse Hessian element (0,0)
    try:
        H_inv = np.linalg.inv(H + 1e-20 * np.eye(H.shape[0]))
        sigma_D_proxy = float(np.sqrt(max(H_inv[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sigma_D_proxy = float('inf')

    # Build cov_DC and cov_H1 (complex).  Downstream code (
    # persist_result_to_graph et al.) expects shape (n_bnd, n_bnd) for
    # cov_H1 since AC now has all n_bnd boundary slots; cov_DC stays at
    # (n_p_dc, n_p_dc) since DC drops the pin.
    cov_DC = np.zeros((n_p_dc, n_p_dc), dtype=complex)
    cov_H1 = np.zeros((n_p_ac, n_p_ac), dtype=complex)
    try:
        if has_dc and n_p_dc > 0:
            cov_DC = H_inv[1:1 + n_p_dc, 1:1 + n_p_dc].astype(complex)
        if has_h1 and n_p_ac > 0:
            re_block = H_inv[ac_re_lo:ac_re_hi, ac_re_lo:ac_re_hi]
            im_block = H_inv[ac_im_lo:ac_im_hi, ac_im_lo:ac_im_hi]
            cov_H1 = (re_block + im_block).astype(complex)
    except Exception:
        pass

    return (float(D_final), P_DC_full, P_H1_full,
            cov_DC, cov_H1, cond_DC, cond_H1,
            history, converged, sigma_D_proxy,
            sv_DC, sv_H1)


def _solve_pressures_complex_wls(
    T: np.ndarray,
    Q_meas: np.ndarray,
    weights: np.ndarray,
    *,
    lambda_reg: float = 0.0,
    p_scale: Optional[float] = None,
    p_scale_fixed: Optional[float] = None,
    pin_idx: int = 0,
):
    """Closed-form complex weighted LSQ:

        min_P  Σ w_e |Q_e − Σ_k T_e^k P_k|² + λ ‖P‖²

    with P[pin_idx] pinned to 0 (gauge fix).  Solve for the remaining
    K−1 columns then reinsert P[pin_idx] = 0 in the output.

    Rank deficiency: a single pin is enough to fix the global gauge,
    but T can have ADDITIONAL near-null directions when some interior
    subcomponent is structurally disconnected from the bulk (one extra
    null mode per floating island propagates through the boundary
    nodes that touch only that island).  These directions are simply
    unidentifiable from flow data — no algorithm can pin them down.
    We solve via SVD pseudoinverse with an rcond cutoff, which returns
    the minimum-norm P estimate (unidentified boundary pressures get
    P=0) and a regularized covariance.  The reported `cond` is the
    full condition number of TtWT before truncation, so it correctly
    flags the underlying ill-conditioning.

    Returns (P, cov, cond):
      P    : (K,) complex; P[pin_idx] is exactly 0
      cov  : (K−1, K−1) complex pseudo-covariance of the unpinned params
      cond : condition number of the normal-equations matrix
    """
    K = T.shape[1]
    if K <= 1:
        return (np.zeros(K, dtype=complex),
                np.zeros((max(K - 1, 0), max(K - 1, 0)), dtype=complex),
                float('inf'))
    keep = np.array([k for k in range(K) if k != pin_idx])
    Tk = T[:, keep]
    W = np.asarray(weights, dtype=float)

    # Normal equations
    TtW = np.conj(Tk).T * W
    TtWT = TtW @ Tk

    # Scale-aware Tikhonov: λ = ref(|diag(T†WT)|) / P_scale².  Earlier
    # versions used max(diag) but that is corrupted by outlier modes —
    # tiles whose carve includes one or two major vessels (R ≈ 100 µm)
    # have G_attach ~10⁵ above the capillary bulk, which inflates
    # max(diag) by the same factor and over-regularises every other
    # boundary node by 5 orders of magnitude.  Median fails the
    # opposite way (under-regularises the null directions in
    # well-conditioned systems).  The 90th-percentile is a robust
    # compromise: outlier-resistant on the high end, while still
    # picking a reference well above the noise floor in low-rank
    # directions.  See `boundary_coupling_report` to see whether a
    # tile has the outlier pattern.
    eff_lambda = float(lambda_reg)
    if p_scale is not None and float(p_scale) > 0 and TtWT.size > 0:
        diag_abs = np.abs(np.diag(TtWT))
        if diag_abs.size:
            ref_eig = float(np.percentile(diag_abs, 90))
        else:
            ref_eig = 0.0
        if ref_eig > 0:
            eff_lambda += ref_eig / (float(p_scale) ** 2)
    # Fixed-physical-units Tikhonov is additive on top of any
    # empirical-Bayes contribution.  Typically only one is set, but
    # combining them is well-defined: stronger of the two dominates.
    if p_scale_fixed is not None and float(p_scale_fixed) > 0:
        eff_lambda += 1.0 / (float(p_scale_fixed) ** 2)

    if eff_lambda > 0:
        TtWT = TtWT + eff_lambda * np.eye(TtWT.shape[0], dtype=complex)
    TtWQ = TtW @ Q_meas

    try:
        cond = float(np.linalg.cond(TtWT))
    except np.linalg.LinAlgError:
        cond = float('inf')

    # SVD pseudoinverse with rcond cutoff handles rank-deficient TtWT
    # (floating-island null modes beyond the explicit gauge pin).  The
    # min-norm solution leaves unidentified components at zero.
    try:
        cov = np.linalg.pinv(TtWT, rcond=1e-10, hermitian=True)
        P_keep = cov @ TtWQ
    except np.linalg.LinAlgError:
        P_keep = np.full(len(keep), np.nan, dtype=complex)
        cov = np.full((len(keep), len(keep)), np.nan, dtype=complex)

    P = np.zeros(K, dtype=complex)
    P[keep] = P_keep
    return P, cov, cond


# ──────────────────────────────────────────────────────────────────
# Top-level: per-tile local inference
# ──────────────────────────────────────────────────────────────────


def infer_local(
    graph, tile_id: int, spec: LocalInferenceSpec,
) -> LocalInferenceResult:
    """Run the alternating-LS local inference on one tile.

    Pseudocode:

        1.  Extract subgraph (edges + boundary/interior nodes) for tile_id.
        2.  Auto-pick gauge pin = boundary node with highest interior degree.
        3.  Extract measured flows (sign-corrected to (u,v) convention).
        4.  D ← spec.D_init.  P ← 0.
        5.  for it in range(max_iter):
              T = build_transfer_matrices(D)
              P_DC = complex_WLS(T_DC, Q_meas_DC)              [Step 1 DC]
              P_H1 = complex_WLS(T_H1, Q_meas_H1)              [Step 1 H1]
              dT/dD ≈ (T(D·1.1) − T(D)) / (D·0.1)              [finite-diff]
              ε* = WLS(  Q_meas − T·P,   dT/dD · P  )          [Step 2]
              D ← D + ε*
              if |ΔD|/D < tol_rel: break
        6.  Compute final residuals, χ²/dof, σ on D and P.
    """
    # ── Resolve pixel size ──
    if spec.px_size_m is None:
        try:
            from .config import PX_SIZE_UM
            px_size_m = float(PX_SIZE_UM) * 1e-6
        except Exception:
            px_size_m = 1.7e-6
    else:
        px_size_m = float(spec.px_size_m)

    # ── 1. Subgraph ──
    # When include_unmeasured_anatomy is on, use the **spatial
    # rectangle carve**: the tile region is the bbox of PIV-touching
    # nodes, and any graph edge crossing that rectangle keeps its
    # outside-endpoint as a boundary node.  This produces a clean
    # "interior + perimeter" subgraph where boundary nodes literally
    # sit on the rectangle edge — no PIV-coverage holes, no graph-
    # topology gymnastics, no bbox-padding spurious-perimeter paths.
    #
    # When the flag is off, fall back to the original graph-topology
    # carve (boundary = nodes with neighbours outside the tile-PIV
    # edge set).  Useful for legacy comparisons.
    if spec.include_unmeasured_anatomy:
        edges_in, all_nodes, boundary_nodes, interior_nodes = \
            extract_tile_subgraph_spatial(
                graph, tile_id, padding_frac=0.0,
                inset_frac=float(spec.carve_inset_frac),
                restrict_to_tile_piv_nodes=bool(spec.carve_restrict_to_tile_piv),
                drop_dangling_boundaries=bool(spec.carve_drop_dangling_boundaries))
        n_data_edges = sum(
            1 for u, v in edges_in
            if any(m.get('tile_id') == tile_id
                   for m in graph.edges[u, v].get('measurements_piv', []) or []))
        if spec.verbose:
            print(f"  Spatial-rectangle carve: "
                  f"{len(edges_in)} edges total ({n_data_edges} carry PIV), "
                  f"{len(boundary_nodes)} boundary nodes, "
                  f"{len(interior_nodes)} interior nodes.")
    else:
        edges_in, all_nodes, boundary_nodes, interior_nodes = \
            extract_tile_subgraph(graph, tile_id)
        n_data_edges = len(edges_in)

    # ── 1b. Bottleneck pruning ──
    # Two combined criteria for dropping carved edges before building
    # the admittance system:
    #   (a) R_px < min_R_px       — unphysical radii (segmentation ghosts)
    #   (b) G < G_median/min_G_factor — relative conductance bottleneck
    # Both are useful: (a) catches sub-pixel artefacts that (b) might
    # miss when G_median itself is small; (b) catches long+thin paths
    # whose individual radius is OK but whose conductance still
    # dominates cond(L).
    if (spec.min_G_factor is not None or spec.min_R_px is not None) \
            and len(edges_in) > 0:
        G_arr = np.zeros(len(edges_in))
        R_px_arr = np.zeros(len(edges_in))
        for i, (u, v) in enumerate(edges_in):
            R_m, L_m = _edge_geometry(graph.edges[u, v], px_size_m)
            G_arr[i] = (np.pi * R_m ** 4 / (8.0 * spec.mu * L_m)
                         if (R_m > 0 and L_m > 0) else 0.0)
            R_px_arr[i] = R_m / px_size_m if R_m > 0 else 0.0
        keep_mask = G_arr > 0
        # (a) R-based prune
        if spec.min_R_px is not None:
            keep_mask &= R_px_arr >= float(spec.min_R_px)
        # (b) Relative-G prune
        G_thr = None
        if spec.min_G_factor is not None:
            Gv = G_arr[G_arr > 0]
            if len(Gv) > 0:
                G_thr = float(np.median(Gv)) / float(spec.min_G_factor)
                keep_mask &= G_arr >= G_thr
        n_drop = int(np.sum(~keep_mask))
        if n_drop > 0:
            if spec.verbose:
                criteria = []
                if spec.min_R_px is not None:
                    criteria.append(f"R<{spec.min_R_px:.1f}px")
                if G_thr is not None:
                    criteria.append(
                        f"G<G_med/{spec.min_G_factor:.0e}={G_thr:.2e}")
                print(f"  Pruning {n_drop} edge(s)  "
                      f"[{' OR '.join(criteria)}]:")
                for i in range(len(edges_in)):
                    if keep_mask[i]:
                        continue
                    u, v = edges_in[i]
                    d = graph.edges[u, v]
                    R_m, L_m = _edge_geometry(d, px_size_m)
                    L_px = L_m / px_size_m
                    ux = float(graph.nodes[u].get('x', 0.0))
                    uy = float(graph.nodes[u].get('y', 0.0))
                    vx = float(graph.nodes[v].get('x', 0.0))
                    vy = float(graph.nodes[v].get('y', 0.0))
                    print(f"    edge ({u},{v}): G={G_arr[i]:.2e}  "
                          f"R={R_px_arr[i]:.2f}px  L={L_px:.1f}px  "
                          f"@ ({(ux+vx)/2:.0f}, {(uy+vy)/2:.0f})")
            edges_in = [edges_in[i] for i in range(len(edges_in))
                         if keep_mask[i]]
            used = set()
            for u, v in edges_in:
                used.add(u); used.add(v)
            boundary_nodes = [n for n in boundary_nodes if n in used]
            interior_nodes = [n for n in interior_nodes if n in used]

    n_edges = len(edges_in)
    n_bnd = len(boundary_nodes)
    if n_edges < 5:
        raise ValueError(
            f"Tile {tile_id}: only {n_edges} edges with PIV "
            "measurements; need ≥ 5 for inference.")
    if n_bnd < 2:
        raise ValueError(
            f"Tile {tile_id}: only {n_bnd} boundary nodes; need ≥ 2 "
            "(gauge pin consumes 1).")
    if spec.verbose:
        print(f"  Tile {tile_id}: {n_edges} edges, "
              f"{n_bnd} boundary nodes, "
              f"{len(interior_nodes)} interior nodes.")

    # ── 2. Gauge pin: boundary node with largest G_attach ──
    # Earlier versions picked by interior-degree (count of incident
    # edges).  That conflates connectivity count with admittance scale,
    # so a weakly-coupled node with several thin attachments could
    # become the pin.  Pinning a low-Fisher boundary node forces all the
    # well-determined pressures to absorb its noise through the gauge
    # constraint.  Picking by G_attach = Σ G_e for incident interior
    # edges puts the gauge at the most data-determined boundary node,
    # minimising gauge-induced uncertainty in the rest of P.
    interior_set = set(interior_nodes)
    # Always compute G_attach — used for pin selection AND retained as a
    # per-tile identifiability diagnostic on LocalInferenceResult.
    g_attach = {n: 0.0 for n in boundary_nodes}
    if n_bnd > 0:
        for u, v in edges_in:
            d = graph.edges[u, v]
            R_m, L_m = _edge_geometry(d, px_size_m)
            if R_m <= 0 or L_m <= 0:
                continue
            Ge = float(np.pi * R_m ** 4 / (8.0 * spec.mu * L_m))
            if u in g_attach and v in interior_set:
                g_attach[u] += Ge
            if v in g_attach and u in interior_set:
                g_attach[v] += Ge
    if spec.pin_node is None and n_bnd > 0:
        pin_node = max(g_attach, key=g_attach.get)
    else:
        pin_node = spec.pin_node if spec.pin_node is not None \
            else boundary_nodes[0]
    pin_idx = boundary_nodes.index(pin_node)
    if spec.verbose:
        if spec.pin_node is None and n_bnd > 0:
            print(f"  Gauge: pin node {pin_node} (idx {pin_idx}) → P=0  "
                  f"(G_attach={g_attach[pin_node]:.2e}, "
                  f"max in carve).")
        else:
            print(f"  Gauge: pin node {pin_node} (idx {pin_idx}) → P=0.")

    # ── 3. Measured flows ──
    Q_dc, Q_hn = _extract_measured_flows(
        graph, edges_in, tile_id, list(spec.harmonics))
    valid_dc = (np.isfinite(Q_dc) & (np.abs(Q_dc) > 1e-30)) \
        if spec.use_dc else np.zeros(n_edges, bool)
    # Select the single AC harmonic in spec.harmonics (the joint LM
    # currently supports one).  Picking from spec lets the dual-fit
    # test run with harmonics=(2,) just by re-calling infer_local.
    ac_harmonics = [h for h in spec.harmonics if h > 0]
    ac_n = ac_harmonics[0] if ac_harmonics else None
    if ac_n is not None and ac_n in Q_hn:
        valid_h1 = (np.isfinite(Q_hn[ac_n].real)
                    & np.isfinite(Q_hn[ac_n].imag)
                    & (np.abs(Q_hn[ac_n]) > 1e-30))
    else:
        valid_h1 = np.zeros(n_edges, bool)
    n_dc = int(valid_dc.sum())
    n_h1 = int(valid_h1.sum())
    if spec.verbose:
        h_label = f"H{ac_n}" if ac_n is not None else "AC"
        print(f"  Valid observations: DC={n_dc}, {h_label}={n_h1}")

    # ── 4. f₀ from tile measurements if not specified ──
    if spec.f0_hz is None:
        f0_seen = []
        for u, v in edges_in:
            d = graph.edges[u, v]
            piv = d.get('measurements_piv', []) or []
            m_ref = next((m for m in piv
                           if m.get('tile_id') == tile_id), None)
            if m_ref is None:
                continue
            f0c = m_ref.get('f0_hz') or m_ref.get('f0')
            if f0c is not None and float(f0c) > 0:
                f0_seen.append(float(f0c))
        f0_hz = float(np.median(f0_seen)) if f0_seen else 2.5
    else:
        f0_hz = float(spec.f0_hz)
    if spec.verbose:
        print(f"  f₀ = {f0_hz:.3f} Hz (median from PIV records)")

    # ── 5. Solve: joint LM (default) or alternating LS ──
    # FGLS outer loop: refit the per-edge noise model σ²(|Q|) = a + b|Q|²
    # between inner solver passes.  Pass 0 uses scalar σ from initial
    # residuals (caller passes sigma_*_e=None → inner sets it).  Subsequent
    # passes use the fitted noise model.  Settles in 2-3 outer passes.
    noise_model_dc: Optional[dict] = None
    noise_model_h1: Optional[dict] = None
    sigma_dc_arr = None
    sigma_h1_arr = None
    # Identifiability diagnostics — populated by joint LM; empty for the
    # alternating-LS path (which doesn't track full SV spectra).
    sv_DC = np.array([])
    sv_H1 = np.array([])
    if spec.use_joint_lm:
        # Import noise-model helpers from the global inference module.
        # Same form, same caller convention, single source of truth.
        try:
            from .inference import fit_noise_model, evaluate_noise_model
            _have_noise_fit = True
        except Exception:
            _have_noise_fit = False
        n_outer = max(1, int(getattr(spec, 'n_outer_iter', 3)))
        history_all = []
        D = P_DC = P_H1 = cov_DC = cov_H1 = None
        cond_DC = cond_H1 = float('inf')
        sigma_D_proxy = float('inf')
        for outer in range(n_outer):
            if spec.verbose and n_outer > 1:
                print(f"  [FGLS outer {outer + 1}/{n_outer}]")
            (D, P_DC, P_H1,
             cov_DC, cov_H1, cond_DC, cond_H1,
             history_o, converged, sigma_D_proxy,
             sv_DC, sv_H1) = _joint_lm_inner_loop(
                graph, edges_in, boundary_nodes, interior_nodes,
                pin_idx, pin_node,
                Q_dc, Q_hn, valid_dc, valid_h1, n_dc, n_h1,
                spec, f0_hz, px_size_m, verbose=spec.verbose,
                sigma_dc_e=sigma_dc_arr, sigma_h1_e=sigma_h1_arr)
            history_all.extend(history_o)
            # If the inner bailed on D drift (converged=False due to a
            # prior-dominated D direction), additional FGLS passes won't
            # rescue identifiability — bail the outer loop too.
            if (not converged) and history_o and \
                    history_o[-1].get('rel_dD', 1.0) < spec.tol_rel:
                if spec.verbose:
                    print(f"    [FGLS outer {outer + 1}] inner drift-"
                          f"bailed; skipping remaining outer passes.")
                break
            # Non-convergence bailout (added 2026-05-19, loosened later
            # the same day).  When the inner LM ran out of iterations
            # AND the AC Jacobian is significantly ill-conditioned, the
            # next FGLS pass won't rescue it — re-fitting σ from
            # residuals doesn't fix a geometry-driven near-null
            # direction.  Trigger conditions (any qualifies):
            #   (a) D collapsed to floor AND cond very large
            #   (b) cond_H1 > 1e7 (moderate ill-conditioning)
            # cond_H1 > 1e7 is conservative — clean tiles run with
            # cond ~ 1e3-1e4; the cap is well above well-behaved
            # tiles' worst case (~1e6).  Cuts non-convergent tiles
            # from ~3-7 min wall to ~30 s.
            if (not converged) and (
                    (float(D) < 1.0e-7 and float(cond_H1) > 1.0e10)
                    or (float(cond_H1) > 1.0e7)):
                if spec.verbose:
                    print(f"    [FGLS outer {outer + 1}] non-converged "
                          f"+ ill-conditioned (D={D:.2e}, "
                          f"cond_H1={cond_H1:.1e}); "
                          f"skipping remaining outer passes.")
                break
            # Refit noise model from current residuals for the NEXT outer
            # pass.  Skip on the last pass (no benefit, costs a fit).
            if outer < n_outer - 1 and _have_noise_fit:
                ab = _build_admittance_system(
                    graph, edges_in, boundary_nodes, interior_nodes,
                    float(D), spec.mu, f0_hz, spec.harmonics, px_size_m)
                T_all_cur = _compute_transfer_matrices(
                    ab, edges_in, boundary_nodes, interior_nodes)
                # Noise model fit is done in nL/s (where the helper's
                # default floor=1e-6 is calibrated).  Internal Q's are
                # in SI (m³/s), so multiply by 1e12 going in.  σ returned
                # by evaluate_noise_model is in nL/s — divide by 1e12 to
                # get back to SI for the inner LM weights.
                _NL = 1.0e12  # m³/s → nL/s scale
                # DC noise model
                if n_dc > 0:
                    Q_pred_dc = (T_all_cur[0] @ P_DC).real
                    r_dc = (Q_dc - Q_pred_dc)
                    noise_model_dc = fit_noise_model(
                        np.abs(r_dc[valid_dc]) * _NL,
                        np.abs(Q_dc[valid_dc]) * _NL,
                        form='variance_linear')
                    if spec.fgls_lock_b_to is not None:
                        b_lock = float(spec.fgls_lock_b_to)
                        r2 = (np.abs(r_dc[valid_dc]) * _NL) ** 2
                        q2 = (np.abs(Q_dc[valid_dc]) * _NL) ** 2
                        a_locked = max(float(np.mean(r2 - b_lock * q2)),
                                        1e-12)
                        noise_model_dc = {'form': 'variance_linear',
                                          'a': a_locked, 'b': b_lock}
                    sigma_dc_nl = evaluate_noise_model(
                        noise_model_dc, np.abs(Q_dc) * _NL)
                    sigma_dc_arr = np.maximum(sigma_dc_nl / _NL, 1e-30)
                    if spec.verbose:
                        a, b = (noise_model_dc.get('a', 0.0),
                                 noise_model_dc.get('b', 0.0))
                        print(f"    DC noise model: σ²(nL/s) = "
                              f"{a:.3e} + {b:.3e}·|Q(nL/s)|²  "
                              f"→ σ(nL/s) range "
                              f"[{sigma_dc_nl[valid_dc].min():.3g}, "
                              f"{sigma_dc_nl[valid_dc].max():.3g}]")
                # H1 noise model (on magnitudes)
                if n_h1 > 0 and ac_n is not None:
                    Q_pred_h1 = T_all_cur[ac_n] @ P_H1
                    r_h1 = Q_hn[ac_n] - Q_pred_h1
                    noise_model_h1 = fit_noise_model(
                        np.abs(r_h1[valid_h1]) * _NL,
                        np.abs(Q_hn[ac_n][valid_h1]) * _NL,
                        form='variance_linear')
                    if spec.fgls_lock_b_to is not None:
                        b_lock = float(spec.fgls_lock_b_to)
                        r2 = (np.abs(r_h1[valid_h1]) * _NL) ** 2
                        q2 = (np.abs(Q_hn[ac_n][valid_h1]) * _NL) ** 2
                        a_locked = max(float(np.mean(r2 - b_lock * q2)),
                                        1e-12)
                        noise_model_h1 = {'form': 'variance_linear',
                                          'a': a_locked, 'b': b_lock}
                    sigma_h1_nl = evaluate_noise_model(
                        noise_model_h1, np.abs(Q_hn[ac_n]) * _NL)
                    sigma_h1_arr = np.maximum(sigma_h1_nl / _NL, 1e-30)
                    if spec.verbose:
                        a, b = (noise_model_h1.get('a', 0.0),
                                 noise_model_h1.get('b', 0.0))
                        print(f"    H1 noise model: σ²(nL/s) = "
                              f"{a:.3e} + {b:.3e}·|Q(nL/s)|²  "
                              f"→ σ(nL/s) range "
                              f"[{sigma_h1_nl[valid_h1].min():.3g}, "
                              f"{sigma_h1_nl[valid_h1].max():.3g}]")
        history = history_all
        # Skip alternating-LS fallback path.
        den_eps = (1.0 / sigma_D_proxy ** 2
                   if np.isfinite(sigma_D_proxy)
                   and sigma_D_proxy > 0 else 1.0)
        _skip_alternating = True
    else:
        _skip_alternating = False

    if not _skip_alternating:
        D = float(spec.D_init)
        P_DC = np.zeros(n_bnd, dtype=complex)
        P_H1 = np.zeros(n_bnd, dtype=complex)
        cov_DC = np.zeros((max(n_bnd - 1, 0),
                            max(n_bnd - 1, 0)), dtype=complex)
        cov_H1 = np.zeros((max(n_bnd - 1, 0),
                            max(n_bnd - 1, 0)), dtype=complex)
        cond_DC = cond_H1 = float('inf')
        den_eps = 1.0
        history: list = []
        converged = False

    for it in range(int(spec.max_iter) if not _skip_alternating else 0):
        # Build T(D) and T(D + dD).  Verbose diagnostics on the FIRST
        # iteration only — checks L conditioning, P_int_basis range,
        # T row-norm distribution.  See _compute_transfer_matrices.
        ab = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            D, spec.mu, f0_hz, spec.harmonics, px_size_m)
        T_all = _compute_transfer_matrices(
            ab, edges_in, boundary_nodes, interior_nodes,
            verbose=(spec.verbose and it == 0))

        # ── Step 1: solve for P given D ──
        T_DC = T_all[0]
        if n_dc > 0:
            # Simple homoscedastic σ_DC from current residuals
            Q_DC_full = np.where(valid_dc, Q_dc, 0.0).astype(complex)
            w_DC = valid_dc.astype(float)
            P_DC, cov_DC, cond_DC = _solve_pressures_complex_wls(
                T_DC, Q_DC_full, w_DC,
                lambda_reg=spec.lambda_reg,
                p_scale=spec.P_scale_Pa,
                p_scale_fixed=spec.P_scale_Pa_fixed,
                pin_idx=pin_idx)
        if ac_n is not None and n_h1 > 0:
            T_H1 = T_all[ac_n]
            Q_H1_full = np.where(valid_h1, Q_hn[ac_n], 0.0)
            w_H1 = valid_h1.astype(float)
            P_H1, cov_H1, cond_H1 = _solve_pressures_complex_wls(
                T_H1, Q_H1_full, w_H1,
                lambda_reg=spec.lambda_reg,
                p_scale=spec.P_scale_Pa,
                p_scale_fixed=spec.P_scale_Pa_fixed,
                pin_idx=pin_idx)

        # ── Step 2: linearize T(D), solve 1-D for ε ──
        D1 = D * (1.0 + spec.eps_D)
        ab_p = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            D1, spec.mu, f0_hz, spec.harmonics, px_size_m)
        T_pert = _compute_transfer_matrices(
            ab_p, edges_in, boundary_nodes, interior_nodes)
        dD = D1 - D
        dT = {n: (T_pert[n] - T_all[n]) / dD for n in T_all}

        num_eps = 0.0
        den_eps = 0.0
        if n_dc > 0:
            Q_pred_DC_curr = (T_DC @ P_DC).real
            r_DC = Q_dc[valid_dc] - Q_pred_DC_curr[valid_dc]
            b_DC = (dT[0] @ P_DC)[valid_dc].real
            sigma_dc_e = max(float(np.std(r_DC)), 1e-12) \
                if r_DC.size > 1 else 1.0
            w_dc_eps = 1.0 / sigma_dc_e ** 2
            num_eps += w_dc_eps * float(np.sum(r_DC * b_DC))
            den_eps += w_dc_eps * float(np.sum(b_DC ** 2))
        if ac_n is not None and n_h1 > 0:
            Q_pred_H1_curr = T_all[ac_n] @ P_H1
            r_H1 = Q_hn[ac_n][valid_h1] - Q_pred_H1_curr[valid_h1]
            b_H1 = (dT[ac_n] @ P_H1)[valid_h1]
            sigma_h1_e = max(float(np.std(np.abs(r_H1))), 1e-12) \
                if r_H1.size > 1 else 1.0
            w_h1_eps = 1.0 / sigma_h1_e ** 2
            num_eps += w_h1_eps * float(np.sum(np.real(
                np.conj(b_H1) * r_H1)))
            den_eps += w_h1_eps * float(np.sum(np.abs(b_H1) ** 2))

        if abs(den_eps) > 1e-30:
            eps = num_eps / den_eps
            D_new = float(np.clip(D + eps, 1e-12, 1.0))
        else:
            D_new = D

        rel_dD = abs(D_new - D) / max(abs(D), 1e-30)

        # Identifiability diagnostics: per-iter χ² (DC and H1 separately),
        # |Q_pred|/|Q_meas| ratios, and ||dT/dD|| Frobenius norm. A flat χ²
        # across iterations while D drifts means D is unconstrained by the
        # data on this tile — not a solver bug.
        chi2_dc_iter = float('nan')
        chi2_h1_iter = float('nan')
        qratio_dc = float('nan')
        qratio_h1 = float('nan')
        if n_dc > 0:
            r_DC_i = Q_dc[valid_dc] - Q_pred_DC_curr[valid_dc]
            sd = max(float(np.std(r_DC_i)) if r_DC_i.size > 1 else 1.0, 1e-12)
            chi2_dc_iter = float(np.sum((r_DC_i / sd) ** 2))
            qm = float(np.sqrt(np.mean(Q_dc[valid_dc] ** 2))) or 1e-30
            qp = float(np.sqrt(np.mean(Q_pred_DC_curr[valid_dc] ** 2)))
            qratio_dc = qp / qm
        if ac_n is not None and n_h1 > 0:
            r_H1_i = Q_hn[ac_n][valid_h1] - Q_pred_H1_curr[valid_h1]
            sh = max(float(np.std(np.abs(r_H1_i)))
                     if r_H1_i.size > 1 else 1.0, 1e-12)
            chi2_h1_iter = float(np.sum(np.abs(r_H1_i / sh) ** 2))
            qm = float(np.sqrt(np.mean(np.abs(Q_hn[ac_n][valid_h1]) ** 2))) \
                or 1e-30
            qp = float(np.sqrt(np.mean(np.abs(Q_pred_H1_curr[valid_h1])
                                       ** 2)))
            qratio_h1 = qp / qm
        dT_norm_dc = float(np.linalg.norm(dT.get(0, np.zeros((1, 1))))) \
            if 0 in dT else float('nan')
        dT_norm_h1 = float(np.linalg.norm(
            dT.get(ac_n, np.zeros((1, 1))))) \
            if ac_n is not None and ac_n in dT else float('nan')

        history.append({
            'iter': it + 1,
            'D_before': D, 'D_after': D_new,
            'rel_dD': rel_dD,
            'cond_DC': cond_DC, 'cond_H1': cond_H1,
            'chi2_dc': chi2_dc_iter, 'chi2_h1': chi2_h1_iter,
            'qratio_dc': qratio_dc, 'qratio_h1': qratio_h1,
            'dT_norm_dc': dT_norm_dc, 'dT_norm_h1': dT_norm_h1,
            'num_eps': num_eps, 'den_eps': den_eps,
        })
        if spec.verbose:
            print(f"    [iter {it + 1}] D = {D:.3e} → {D_new:.3e}  "
                  f"(|ΔD|/D = {rel_dD:.3g})  "
                  f"χ²_DC={chi2_dc_iter:.3g} χ²_H1={chi2_h1_iter:.3g}  "
                  f"|Q_p|/|Q_m| DC={qratio_dc:.3g} H1={qratio_h1:.3g}  "
                  f"||dT/dD|| DC={dT_norm_dc:.2e} H1={dT_norm_h1:.2e}  "
                  f"num/den={num_eps:.2e}/{den_eps:.2e}")
        D = D_new
        if rel_dD < spec.tol_rel:
            converged = True
            break

    # ── 6. Final fit + uncertainties ──
    ab = _build_admittance_system(
        graph, edges_in, boundary_nodes, interior_nodes,
        D, spec.mu, f0_hz, spec.harmonics, px_size_m)
    T_all = _compute_transfer_matrices(
        ab, edges_in, boundary_nodes, interior_nodes)
    T_DC = T_all[0]
    Q_pred_DC = (T_DC @ P_DC).real
    if ac_n is not None and ac_n in T_all:
        T_H1 = T_all[ac_n]
        Q_pred_H1 = T_H1 @ P_H1
    else:
        Q_pred_H1 = np.zeros(n_edges, dtype=complex)

    # Residuals & χ².  Prefer the FGLS-fitted per-edge σ if available
    # (joint LM path with n_outer_iter > 1); fall back to the legacy
    # self-normalising std(residuals) otherwise.  Using the FGLS σ
    # makes χ²/dof a proper goodness-of-fit metric (≈ 1 when noise model
    # is correct) and makes the post-hoc σ_D scaling honest.
    chi2 = 0.0
    sigma_dc_final = 1.0
    sigma_h1_final = 1.0
    if n_dc > 0:
        r_DC = Q_dc[valid_dc] - Q_pred_DC[valid_dc]
        if sigma_dc_arr is not None:
            sd = sigma_dc_arr[valid_dc]
            chi2 += float(np.sum((r_DC / sd) ** 2))
            sigma_dc_final = float(np.median(sd))
        else:
            sigma_dc_final = max(float(np.std(r_DC, ddof=1))
                                  if r_DC.size > 1 else 1.0, 1e-12)
            chi2 += float(np.sum((r_DC / sigma_dc_final) ** 2))
    if n_h1 > 0 and ac_n is not None:
        r_H1 = Q_hn[ac_n][valid_h1] - Q_pred_H1[valid_h1]
        if sigma_h1_arr is not None:
            sh = sigma_h1_arr[valid_h1]
            chi2 += float(np.sum(np.abs(r_H1 / sh) ** 2))
            sigma_h1_final = float(np.median(sh))
        else:
            sigma_h1_final = max(float(np.std(np.abs(r_H1), ddof=1))
                                  if r_H1.size > 1 else 1.0, 1e-12)
            chi2 += float(np.sum(np.abs(r_H1 / sigma_h1_final) ** 2))

    n_obs_real = n_dc + 2 * n_h1
    # Free params: (n_bnd-1) DC + 2(n_bnd-1) H1 (real+imag) + 1 D
    n_params_dc = (n_bnd - 1) if n_dc > 0 else 0
    n_params_h1 = 2 * (n_bnd - 1) if n_h1 > 0 else 0
    n_params = n_params_dc + n_params_h1 + 1
    dof = max(n_obs_real - n_params, 1)
    chi2_red = chi2 / dof

    cov_DC_scaled = cov_DC * chi2_red
    cov_H1_scaled = cov_H1 * chi2_red

    # Pack pressures + uncertainties as dicts keyed by node id
    P_DC_dict = {}
    P_H1_dict = {}
    sig_P_DC = {}
    sig_P_H1 = {}
    keep_cols = [k for k in range(n_bnd) if k != pin_idx]
    for k in range(n_bnd):
        node = boundary_nodes[k]
        P_DC_dict[node] = float(P_DC[k].real)
        P_H1_dict[node] = complex(P_H1[k])
    for i, k in enumerate(keep_cols):
        node = boundary_nodes[k]
        sig_P_DC[node] = float(np.sqrt(
            max(cov_DC_scaled[i, i].real, 0.0)))
        sig_P_H1[node] = float(np.sqrt(
            max(cov_H1_scaled[i, i].real, 0.0)))
    sig_P_DC[pin_node] = 0.0
    sig_P_H1[pin_node] = 0.0

    # σ_D from Step 2's WLS (Var(ε) = chi2_red / den_eps)
    if abs(den_eps) > 1e-30:
        sigma_D = float(np.sqrt(chi2_red / abs(den_eps)))
    else:
        sigma_D = float('nan')

    if spec.verbose:
        print(f"  Result for tile {tile_id}:")
        print(f"    D̂ = {D:.3e} 1/Pa  ± {sigma_D:.3e}  "
              f"(D₀ = {spec.D_init:.3e})")
        print(f"    χ²/dof = {chi2_red:.3f}  "
              f"(N_obs_real = {n_obs_real}, dof = {dof}, "
              f"params = {n_params})")
        # sigma_dc_final / sigma_h1_final are in SI (m³/s); convert to
        # nL/s for display so the printed value matches the noise-model
        # nL/s coefficients above.
        print(f"    σ_DC = {sigma_dc_final * 1e12:.3g} nL/s,  "
              f"σ_H1 = {sigma_h1_final * 1e12:.3g} nL/s")
        print(f"    cond(M_DC) = {cond_DC:.2g}, "
              f"cond(M_H1) = {cond_H1:.2g}")
        print(f"    iterations = {len(history)}, "
              f"converged = {converged}")

        # Per-edge σ_Q distribution and SNR — sanity check the
        # heteroscedastic noise model lands in physically sensible
        # territory (σ_Q ~ pL/s–nL/s, SNR > 1 on edges that carry data).
        if sigma_dc_arr is not None and n_dc > 0:
            sd_nl = sigma_dc_arr[valid_dc] * 1e12  # m³/s → nL/s
            qm_nl = np.abs(Q_dc[valid_dc]) * 1e12
            snr = qm_nl / np.maximum(sd_nl, 1e-30)
            print(f"    σ_Q_DC per-edge (nL/s): median={np.median(sd_nl):.3g}, "
                  f"5-95%=[{np.percentile(sd_nl, 5):.3g}, "
                  f"{np.percentile(sd_nl, 95):.3g}]")
            print(f"    SNR_DC per-edge: median={np.median(snr):.2f}, "
                  f"5-95%=[{np.percentile(snr, 5):.2f}, "
                  f"{np.percentile(snr, 95):.2f}], "
                  f"n_SNR<1: {int(np.sum(snr < 1))}/{n_dc}")
        if sigma_h1_arr is not None and n_h1 > 0 and ac_n is not None:
            sh_nl = sigma_h1_arr[valid_h1] * 1e12
            qm_nl = np.abs(Q_hn[ac_n][valid_h1]) * 1e12
            snr = qm_nl / np.maximum(sh_nl, 1e-30)
            print(f"    σ_Q_H1 per-edge (nL/s): median={np.median(sh_nl):.3g}, "
                  f"5-95%=[{np.percentile(sh_nl, 5):.3g}, "
                  f"{np.percentile(sh_nl, 95):.3g}]")
            print(f"    SNR_H1 per-edge: median={np.median(snr):.2f}, "
                  f"5-95%=[{np.percentile(snr, 5):.2f}, "
                  f"{np.percentile(snr, 95):.2f}], "
                  f"n_SNR<1: {int(np.sum(snr < 1))}/{n_h1}")

        # Bottom-k weakly-coupled boundary nodes (lowest G_attach).
        # These are where gauge-pin choice and prior strength matter
        # most; flagging them up-front replaces "stare at cond_DC and
        # guess".  Only print if there are weak nodes worth flagging.
        try:
            g_att = {n: 0.0 for n in boundary_nodes}
            for u, v in edges_in:
                d = graph.edges[u, v]
                R_m, L_m = _edge_geometry(d, px_size_m)
                if R_m <= 0 or L_m <= 0:
                    continue
                Ge = float(np.pi * R_m ** 4 / (8.0 * spec.mu * L_m))
                if u in g_att and v in interior_set:
                    g_att[u] += Ge
                if v in g_att and u in interior_set:
                    g_att[v] += Ge
            sorted_b = sorted(g_att.items(), key=lambda kv: kv[1])
            G_med_b = float(np.median([g for _, g in sorted_b
                                         if g > 0])) \
                if any(g > 0 for _, g in sorted_b) else 0.0
            weak_thr = G_med_b / 100.0 if G_med_b > 0 else 0.0
            n_weak = sum(1 for _, g in sorted_b if g < weak_thr)
            if n_weak > 0:
                print(f"    Weak boundary nodes (G_attach < "
                      f"G_med/100, n={n_weak}/{n_bnd}):")
                for nb, gv in sorted_b[:min(3, n_weak)]:
                    print(f"      node {nb}: G_attach={gv:.2e}  "
                          f"(P̂_DC={float(P_DC_dict.get(nb, 0.0)):.2f} Pa)")
        except Exception:
            pass

    # Q's were converted to SI (m³/s) at extraction so the WLS units
    # match T's (G in m³/(Pa·s)) → P comes out in physical Pa.  Convert
    # back to nL/s for the user-facing result fields, since that's the
    # convention every consumer (plots, the viewer tab, downstream
    # storage) already speaks.
    nL_per_m3 = 1.0e12
    return LocalInferenceResult(
        tile_id=tile_id, D_hat=D, sigma_D=sigma_D,
        P_DC=P_DC_dict, P_H1=P_H1_dict,
        sigma_P_DC=sig_P_DC, sigma_P_H1=sig_P_H1,
        chi2_red=chi2_red,
        n_obs_real=n_obs_real, n_params=n_params, dof=dof,
        iterations=len(history), converged=converged,
        boundary_nodes=boundary_nodes, interior_edges=edges_in,
        pin_node=pin_node, cond_DC=cond_DC, cond_H1=cond_H1,
        f0_hz=f0_hz,
        Q_meas_DC=Q_dc * nL_per_m3,
        Q_meas_H1=(Q_hn[ac_n] * nL_per_m3
                    if ac_n is not None and ac_n in Q_hn
                    else np.zeros(n_edges, complex)),
        Q_pred_DC=Q_pred_DC * nL_per_m3,
        Q_pred_H1=Q_pred_H1 * nL_per_m3,
        valid_dc=valid_dc, valid_h1=valid_h1,
        convergence_history=history,
        noise_model_dc=noise_model_dc,
        noise_model_h1=noise_model_h1,
        ac_harmonic=int(ac_n) if ac_n is not None else 1,
        # Identifiability diagnostics (always populated; cost is O(n_p³) once).
        sv_DC=np.asarray(sv_DC, dtype=float),
        sv_H1=np.asarray(sv_H1, dtype=float),
        n_null_DC=int((np.asarray(sv_DC) < (sv_DC[0] * 1e-12)).sum())
            if len(sv_DC) > 0 and sv_DC[0] > 0 else 0,
        n_null_H1=int((np.asarray(sv_H1) < (sv_H1[0] * 1e-12)).sum())
            if len(sv_H1) > 0 and sv_H1[0] > 0 else 0,
        n_marginal_DC=int((np.asarray(sv_DC) < (sv_DC[0] * 1e-8)).sum())
            if len(sv_DC) > 0 and sv_DC[0] > 0 else 0,
        n_marginal_H1=int((np.asarray(sv_H1) < (sv_H1[0] * 1e-8)).sum())
            if len(sv_H1) > 0 and sv_H1[0] > 0 else 0,
        G_attach_by_node=dict(g_attach),
    )


def persist_result_to_graph(graph, result: LocalInferenceResult) -> None:
    """Save a LocalInferenceResult onto the mosaic graph.

    Writes two layers:

      (1) Per-tile metadata at ``graph.graph['per_tile_local_inference'][tile_id]``
          (D̂, σ_D, χ²/dof, conditioning, boundary pressures, etc.).

          Fit-quality summary fields:

            dc_ratio  =  RMS(Q_pred_DC) / RMS(Q_meas_DC)   over valid_dc
            h1_ratio  =  RMS(|Q_pred_H1|) / RMS(|Q_meas_H1|)  over valid_h1

          Both should be ~1 for a well-fit tile.  They complement χ²_red
          (which can look fine when σ noise is over-estimated): the
          ratios catch the "predicted ≈ 0 on most edges" failure mode
          that arises when boundary pressures are over-shrunk by the
          prior or pinv truncates significant transfer-matrix
          directions.  Tile 27 originally hit χ²_red = 0.02 with
          dc_ratio = 0.27 — χ²_red said the fit was great, the ratios
          revealed it was 73% under-predicted.

      (2) Per-edge predicted fields with the ``*_local`` suffix
          (mirrors the global-sim ``*_sim`` convention so the viewer's
          source toggle can pick them up).  Stored as a dict keyed by
          tile_id so multiple tiles can coexist on the same graph.

          ``edge['local_sim'] = { tile_id: { 'mean_Q': ..., 'amp_Q': ...,
                                               'phase': ..., 'PI': ... } }``

          Plus convenience flat fields ``mean_Q_local``, ``phase_local``,
          etc. that resolve to whichever tile most recently wrote to
          that edge — this is what the viewer's `_resolve_field` will
          read for the simple case.

    Both layers persist automatically via the existing pickle save.
    """
    # ── (1) Per-tile summary container ──
    tile_id = int(result.tile_id)
    container = graph.graph.setdefault('per_tile_local_inference', {})

    # Fit-quality summary: |Q_pred|/|Q_meas| RMS ratios.  Cheap to
    # compute now, saves the user from re-running inference to
    # produce diagnostic plots.
    dc_ratio = float('nan')
    h1_ratio = float('nan')
    if result.valid_dc.any():
        qm = float(np.sqrt(np.mean(
            result.Q_meas_DC[result.valid_dc] ** 2))) or 1e-30
        qp = float(np.sqrt(np.mean(
            result.Q_pred_DC[result.valid_dc] ** 2)))
        dc_ratio = qp / qm
    if result.valid_h1.any():
        qm = float(np.sqrt(np.mean(np.abs(
            result.Q_meas_H1[result.valid_h1]) ** 2))) or 1e-30
        qp = float(np.sqrt(np.mean(np.abs(
            result.Q_pred_H1[result.valid_h1]) ** 2)))
        h1_ratio = qp / qm

    container[tile_id] = {
        'D': result.D_hat,
        'sigma_D': result.sigma_D,
        'P_DC': dict(result.P_DC),
        'P_H1': {n: complex(v) for n, v in result.P_H1.items()},
        'sigma_P_DC': dict(result.sigma_P_DC),
        'sigma_P_H1': dict(result.sigma_P_H1),
        'chi2_red': result.chi2_red,
        'dc_ratio': dc_ratio,
        'h1_ratio': h1_ratio,
        'iterations': result.iterations,
        'converged': result.converged,
        'cond_DC': result.cond_DC,
        'cond_H1': result.cond_H1,
        'f0_hz': result.f0_hz,
        'pin_node': int(result.pin_node),
        'n_obs_real': result.n_obs_real,
        'dof': result.dof,
        'n_bnd': len(result.boundary_nodes),
        'n_edges': len(result.interior_edges),
        'noise_model_dc': (dict(result.noise_model_dc)
                           if result.noise_model_dc is not None
                           else None),
        'noise_model_h1': (dict(result.noise_model_h1)
                           if result.noise_model_h1 is not None
                           else None),
    }

    # ── (2) Per-edge predicted fields ──
    # Q_pred_DC / Q_pred_H1 are aligned to result.interior_edges and
    # already in nL/s (the user-facing unit; see infer_local).
    Q_dc = np.asarray(result.Q_pred_DC, dtype=float)
    Q_h1 = np.asarray(result.Q_pred_H1, dtype=complex)
    Q_meas_dc_e = np.asarray(result.Q_meas_DC, dtype=float)
    Q_meas_h1_e = np.asarray(result.Q_meas_H1, dtype=complex)

    # Per-edge σ_Q from the fitted heteroscedastic noise model (nL/s).
    # Stored on each edge so downstream code can quote per-measurement
    # uncertainty without re-evaluating the model.
    sigma_dc_e_nl = None
    sigma_h1_e_nl = None
    try:
        from .inference import evaluate_noise_model as _eval_noise
        if result.noise_model_dc is not None:
            sigma_dc_e_nl = _eval_noise(
                result.noise_model_dc, np.abs(Q_meas_dc_e))
        if result.noise_model_h1 is not None:
            sigma_h1_e_nl = _eval_noise(
                result.noise_model_h1, np.abs(Q_meas_h1_e))
    except Exception:
        pass

    for ei, (u, v) in enumerate(result.interior_edges):
        if not graph.has_edge(u, v):
            continue
        d = graph.edges[u, v]
        q_dc = float(Q_dc[ei]) if ei < len(Q_dc) else float('nan')
        q_h1 = complex(Q_h1[ei]) if ei < len(Q_h1) else complex('nan')

        amp_h1 = float(abs(q_h1)) if np.isfinite(q_h1) else float('nan')
        phase_h1 = (float(np.angle(q_h1))
                     if np.isfinite(q_h1) and abs(q_h1) > 0
                     else float('nan'))
        # Pulsatility index PI = 2|H1|/|DC|.  Falls back to nan when
        # |DC| ≈ 0 (avoid runaway divisions).
        if np.isfinite(q_dc) and abs(q_dc) > 1e-30:
            pi_local = 2 * amp_h1 / abs(q_dc)
        else:
            pi_local = float('nan')

        # Per-tile sub-record (keyed by tile_id so multiple tiles can
        # coexist on the same edge — the same edge can be visible
        # from neighbouring tiles' inference runs).
        # Per-edge noise σ from the FGLS-fitted model (nL/s).
        sigma_Q_dc = (float(sigma_dc_e_nl[ei])
                       if sigma_dc_e_nl is not None
                       and ei < len(sigma_dc_e_nl)
                       else float('nan'))
        sigma_Q_h1 = (float(sigma_h1_e_nl[ei])
                       if sigma_h1_e_nl is not None
                       and ei < len(sigma_h1_e_nl)
                       else float('nan'))

        tile_record = {
            'tile_id': tile_id,
            'mean_Q': q_dc,
            'amp_Q': amp_h1,
            'phase': phase_h1,
            'PI': pi_local,
            # Legacy names (always reflect whatever AC harmonic was
            # fit; downstream callers using these names assumed H1).
            'Q_H1_re': float(q_h1.real) if np.isfinite(q_h1) else float('nan'),
            'Q_H1_im': float(q_h1.imag) if np.isfinite(q_h1) else float('nan'),
            # Harmonic-specific names so multiple-harmonic fits coexist:
            # H1 fit writes Q_H1_*, H2 fit writes Q_H2_*, etc.  The
            # phase-gradient code reads these to render ∇φ at the
            # selected harmonic.
            f'Q_H{int(result.ac_harmonic)}_re': (
                float(q_h1.real) if np.isfinite(q_h1)
                else float('nan')),
            f'Q_H{int(result.ac_harmonic)}_im': (
                float(q_h1.imag) if np.isfinite(q_h1)
                else float('nan')),
            'D': float(result.D_hat),
            'chi2_red': float(result.chi2_red),
            'sigma_Q_dc': sigma_Q_dc,
            'sigma_Q_h1': sigma_Q_h1,
        }
        local_sims = d.setdefault('local_sim', {})
        local_sims[tile_id] = tile_record

        # Flat convenience fields — last-writer-wins.  These are what
        # the viewer's source dispatcher reads via the `_local` suffix
        # convention.  When multiple tiles cover the same edge, the
        # most recently fitted tile shows up; the per-tile record
        # above keeps the full history.
        d['mean_Q_local'] = q_dc
        d['amp_Q_local'] = amp_h1
        d['phase_local'] = phase_h1
        d['PI_local'] = pi_local
        d['D_local'] = float(result.D_hat)
        d['chi2_local'] = float(result.chi2_red)
        d['sigma_Q_dc_local'] = sigma_Q_dc
        d['sigma_Q_h1_local'] = sigma_Q_h1
        # Harmonic-specific local fields.  This fit was for harmonic
        # `result.ac_harmonic` (1 or 2).  Writing under suffixes lets
        # H1- and H2-fits coexist on the same edge, so the viewer can
        # plot e.g. amp_Q_h1_local from the H1 fit AND amp_Q_h2_local
        # from a later H2 fit.
        h = int(result.ac_harmonic)
        d[f'amp_Q_h{h}_local'] = amp_h1
        d[f'phase_h{h}_local'] = phase_h1
        d[f'sigma_Q_h{h}_local'] = sigma_Q_h1
        # If both H1 and H2 local fits exist on this edge, also compute
        # the local-predicted ratio and phase offset (model's prediction
        # for the same diagnostic the PIV-side fields carry).
        a1 = d.get('amp_Q_h1_local')
        a2 = d.get('amp_Q_h2_local')
        p1 = d.get('phase_h1_local')
        p2 = d.get('phase_h2_local')
        if (a1 is not None and a2 is not None
                and np.isfinite(a1) and np.isfinite(a2) and a1 > 0):
            d['h2_h1_ratio_local'] = a2 / a1
            if (p1 is not None and p2 is not None
                    and np.isfinite(p1) and np.isfinite(p2)):
                z = np.exp(1j * (p2 - 2 * p1))
                d['h2_phase_offset_local'] = float(np.angle(z))
        # SNR per edge per harmonic — handy for quality gating in plots.
        if np.isfinite(sigma_Q_dc) and sigma_Q_dc > 0 \
                and ei < len(Q_meas_dc_e):
            d['snr_dc_local'] = (float(np.abs(Q_meas_dc_e[ei]))
                                  / sigma_Q_dc)
        if np.isfinite(sigma_Q_h1) and sigma_Q_h1 > 0 \
                and ei < len(Q_meas_h1_e):
            d[f'snr_h{h}_local'] = (float(np.abs(Q_meas_h1_e[ei]))
                                     / sigma_Q_h1)
            # Back-compat alias for the H1-specific SNR field.
            if h == 1:
                d['snr_h1_local'] = d['snr_h1_local']


def clear_local_inference_for_tile(graph, tile_id: int) -> None:
    """Remove all local-inference traces for one tile from the graph.

    Useful before re-running on a tile to avoid stale metadata.  Removes
    the per-tile metadata entry and the per-tile sub-record on every
    edge.  The flat ``*_local`` convenience fields are re-derived from
    whichever tile-record remains (or cleared if none).
    """
    tid = int(tile_id)
    container = graph.graph.get('per_tile_local_inference', {})
    container.pop(tid, None)

    for u, v, d in graph.edges(data=True):
        local_sims = d.get('local_sim') or {}
        if tid in local_sims:
            local_sims.pop(tid, None)
        # Reset flat fields based on whatever tiles remain
        if local_sims:
            # Pick the tile with the most recent (lexicographically
            # latest) id as the "current" — arbitrary but deterministic.
            latest_tid = max(local_sims.keys())
            rec = local_sims[latest_tid]
            d['mean_Q_local'] = rec['mean_Q']
            d['amp_Q_local'] = rec['amp_Q']
            d['phase_local'] = rec['phase']
            d['PI_local'] = rec['PI']
            d['D_local'] = rec['D']
            d['chi2_local'] = rec['chi2_red']
        else:
            for k in ('mean_Q_local', 'amp_Q_local', 'phase_local',
                       'PI_local', 'D_local', 'chi2_local', 'local_sim'):
                d.pop(k, None)


# ──────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────


def synthetic_identifiability_scan(
    graph,
    tile_id: int,
    *,
    base_spec: Optional['LocalInferenceSpec'] = None,
    D_truth_grid: Optional[Sequence[float]] = None,
    P_truth_amp_Pa: float = 100.0,
    P_truth_amp_H1_Pa: Optional[float] = None,
    sigma_Q_nL_per_s: float = 0.1,
    sigma_Q_H1_nL_per_s: Optional[float] = None,
    sigma_rel: float = 0.0,
    sign_clip_dc: bool = True,
    n_reps: int = 5,
    rng_seed: Optional[int] = 0,
):
    """Forward-simulate Q with known (D_true, P_true) on this tile's
    carve, then re-fit via `infer_local` and compare D̂ to D_true.

    Maps the identifiability floor for this specific geometry: the
    smallest σ_D the data can possibly support given σ_Q noise and the
    tile's transfer matrix.  If on real data σ_D/D̂ is much larger than
    the synthetic floor, the inference is noise-limited and the
    per-tile D estimate isn't trustworthy.

    Procedure per (D_truth, rep):
      1. Build T(D_truth) on this tile's carve.
      2. Draw P_true_DC ~ N(0, P_truth_amp²) and P_true_H1 ~ CN(0, P²)
         on the boundary nodes (pin node fixed at 0).
      3. Forward-simulate Q = T(D_truth) · P_true.
      4. Add iid Gaussian noise σ_Q to each edge measurement (DC real,
         H1 complex with per-component σ_Q/√2 so |Q| noise ≈ σ_Q).
      5. Re-fit via `infer_local` (with this synthetic Q stuffed into
         the measurement extraction path).
      6. Record D_hat, σ_D, dc_ratio, h1_ratio.

    Parameters
    ----------
    D_truth_grid : list of D values to test.  Default: logspace from
                   1e-6 to 1e-2 (7 points).
    P_truth_amp_Pa : Std-dev of synthetic DC boundary pressures.  Should
                     reflect physiological scale; 100 Pa is reasonable
                     for embryonic vasculature.
    P_truth_amp_H1_Pa : Std-dev of synthetic H1 boundary phasor (Re/Im
                       each drawn independently).  Defaults to
                       P_truth_amp_Pa (legacy behaviour).  Set lower for
                       physiologically realistic AC amplitudes (e.g. 5
                       Pa with 50 Pa DC matches the observed pulsatile
                       fraction in plexus PIV).
    sigma_Q_nL_per_s : Edge-flow measurement noise applied to the DC
                       channel (and to the AC channel if
                       ``sigma_Q_H1_nL_per_s`` is None).  In nL/s.
    sigma_Q_H1_nL_per_s : Optional separate σ for the H1 channel.  If
                          None (default), the DC σ is used for AC as
                          well (legacy behaviour).  Set independently
                          to reproduce real-PIV heteroscedasticity
                          across channels (σ_DC ≈ 5× σ_H1 in plexus
                          PIV after FGLS).
    n_reps : Independent noise realizations per D_truth value.

    Returns
    -------
    dict keyed by D_truth → list of per-rep dicts with keys
        {'D_hat', 'sigma_D', 'dc_ratio', 'h1_ratio', 'iters',
         'converged'}.
    Also prints a summary table.
    """
    if base_spec is None:
        base_spec = LocalInferenceSpec()
    if D_truth_grid is None:
        D_truth_grid = list(np.logspace(-6, -2, 7))
    rng = np.random.default_rng(rng_seed)

    # We need to mirror infer_local's pre-fit setup so that the
    # synthetic Q sits at exactly the indices the refitter expects.
    px_size_m = float(base_spec.px_size_m) \
        if base_spec.px_size_m is not None else 1.7e-6
    if base_spec.include_unmeasured_anatomy:
        edges_in, _, boundary_nodes, interior_nodes = \
            extract_tile_subgraph_spatial(
                graph, int(tile_id),
                inset_frac=float(base_spec.carve_inset_frac),
                restrict_to_tile_piv_nodes=bool(base_spec.carve_restrict_to_tile_piv),
                drop_dangling_boundaries=bool(base_spec.carve_drop_dangling_boundaries))
    else:
        edges_in, _, boundary_nodes, interior_nodes = \
            extract_tile_subgraph(graph, int(tile_id))
    n_edges = len(edges_in)
    n_bnd = len(boundary_nodes)
    if n_edges < 5 or n_bnd < 2:
        raise ValueError(
            f"Tile {tile_id}: carve too small "
            f"(edges={n_edges}, bnd={n_bnd}).")

    # Pin = G_attach max (same as infer_local).
    interior_set = set(interior_nodes)
    g_attach = {n: 0.0 for n in boundary_nodes}
    for u, v in edges_in:
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0:
            continue
        Ge = float(np.pi * R_m ** 4 / (8.0 * base_spec.mu * L_m))
        if u in g_attach and v in interior_set:
            g_attach[u] += Ge
        if v in g_attach and u in interior_set:
            g_attach[v] += Ge
    pin_node = max(g_attach, key=g_attach.get)
    pin_idx = boundary_nodes.index(pin_node)

    # f0 from PIV records on this tile (same as infer_local).
    f0_hz = base_spec.f0_hz
    if f0_hz is None:
        f0s = []
        for u, v in edges_in:
            for m in graph.edges[u, v].get('measurements_piv', []) or []:
                if m.get('tile_id') == int(tile_id):
                    f0 = m.get('f0_hz') or m.get('f0')
                    if f0 is not None and np.isfinite(f0) and f0 > 0:
                        f0s.append(float(f0))
        f0_hz = float(np.median(f0s)) if f0s else 2.5

    nL_per_m3 = 1.0e12   # Q (nL/s) → Q (m³/s) is divide by this
    sigma_Q_si = float(sigma_Q_nL_per_s) / nL_per_m3
    sigma_Q_h1_si = (float(sigma_Q_H1_nL_per_s) / nL_per_m3
                      if sigma_Q_H1_nL_per_s is not None
                      else sigma_Q_si)

    print(f"\n[Synthetic scan] tile {tile_id}  "
          f"({n_edges} edges, {n_bnd} bnd, pin={pin_node})")
    print(f"  σ_Q = {sigma_Q_nL_per_s:.3g} nL/s, "
          f"P_truth_amp = {P_truth_amp_Pa:.0f} Pa, "
          f"n_reps = {n_reps}, "
          f"f0 = {f0_hz:.3f} Hz")
    print(f"  {'D_true':>10}  {'mean D̂':>12}  "
          f"{'std D̂':>11}  {'median σ_D':>11}  "
          f"{'mean DC':>8}  {'mean H1':>8}  {'iters':>5}")

    results = {}
    for D_truth in D_truth_grid:
        # Forward T(D_truth) once (T's only depend on D, not on Q).
        ab = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            float(D_truth), base_spec.mu, f0_hz,
            base_spec.harmonics, px_size_m)
        T_all = _compute_transfer_matrices(
            ab, edges_in, boundary_nodes, interior_nodes)
        T_DC = T_all[0].real
        T_H1 = T_all.get(1) if 1 in base_spec.harmonics else None

        rep_records = []
        for rep in range(int(n_reps)):
            # Synthetic P_true: zero at pin, Gaussian elsewhere.
            P_DC_true = rng.normal(
                0.0, P_truth_amp_Pa, size=n_bnd)
            P_DC_true[pin_idx] = 0.0
            Q_DC_clean = T_DC @ P_DC_true
            # Per-edge σ for DC: floor + sigma_rel·|Q|.  σ_rel = 0
            # recovers the legacy homoscedastic injection.
            sigma_dc_per = np.sqrt(sigma_Q_si ** 2
                                    + (float(sigma_rel)
                                       * np.abs(Q_DC_clean)) ** 2)
            Q_DC_syn = Q_DC_clean + rng.normal(
                0.0, sigma_dc_per, size=n_edges)
            # Sign-clip DC: PIV records preserve flow direction; large
            # negative noise that flips Q's sign is a measurement
            # artifact, not real signal.  Clip to ±ε of zero so the
            # sign is consistent with Q_DC_clean.
            if sign_clip_dc:
                eps_si = 1e-15  # in m³/s; well below any meaningful Q
                pos = Q_DC_clean > 0
                neg = Q_DC_clean < 0
                Q_DC_syn[pos] = np.maximum(Q_DC_syn[pos], eps_si)
                Q_DC_syn[neg] = np.minimum(Q_DC_syn[neg], -eps_si)
            if T_H1 is not None:
                _amp_h1 = (P_truth_amp_H1_Pa
                            if P_truth_amp_H1_Pa is not None
                            else P_truth_amp_Pa)
                P_H1_true = (rng.normal(0.0, _amp_h1, n_bnd)
                              + 1j * rng.normal(0.0, _amp_h1, n_bnd))
                P_H1_true[pin_idx] = 0.0 + 0.0j
                Q_H1_clean = T_H1 @ P_H1_true
                sigma_h1_per = np.sqrt(sigma_Q_h1_si ** 2
                                        + (float(sigma_rel)
                                           * np.abs(Q_H1_clean)) ** 2)
                Q_H1_syn = Q_H1_clean + (
                    rng.normal(0.0, sigma_h1_per / np.sqrt(2.0), n_edges)
                    + 1j * rng.normal(0.0, sigma_h1_per / np.sqrt(2.0),
                                       n_edges))
            else:
                Q_H1_syn = None

            # Refit via a one-shot solve mirroring infer_local's loop.
            # Use a wrapper spec that points at a synthetic measurement
            # provider.  Easier: temporarily monkey-patch
            # graph.edges with synthetic PIV records, run infer_local,
            # restore.  But that's invasive; instead inline the same
            # alternating LS here using the existing solver primitives.
            D_hat, sigma_D_hat, dc_r, h1_r, n_iter, conv = \
                _synthetic_refit(
                    graph, edges_in, boundary_nodes, interior_nodes,
                    pin_idx, pin_node,
                    Q_DC_syn, Q_H1_syn, f0_hz, base_spec, px_size_m,
                    sigma_Q_si)
            rep_records.append({
                'D_hat': D_hat, 'sigma_D': sigma_D_hat,
                'dc_ratio': dc_r, 'h1_ratio': h1_r,
                'iters': n_iter, 'converged': conv,
            })
        Dhs = np.array([r['D_hat'] for r in rep_records])
        sDs = np.array([r['sigma_D'] for r in rep_records])
        dcrs = np.array([r['dc_ratio'] for r in rep_records])
        h1rs = np.array([r['h1_ratio'] for r in rep_records])
        itrs = np.array([r['iters'] for r in rep_records])
        print(f"  {D_truth:>10.2e}  {Dhs.mean():>12.3e}  "
              f"{Dhs.std():>11.2e}  {np.median(sDs):>11.2e}  "
              f"{dcrs.mean():>8.3f}  {h1rs.mean():>8.3f}  "
              f"{int(np.mean(itrs)):>5d}")
        results[float(D_truth)] = rep_records

    # Verdict
    medians = {D: float(np.median([r['D_hat'] for r in recs]))
               for D, recs in results.items()}
    print(f"\n  Identifiability summary:")
    print(f"    D_truth values where mean(D̂) is within 50%: ", end='')
    in_band = [D for D, m in medians.items()
                if abs(np.log10(m / D)) < np.log10(1.5)
                and np.isfinite(m) and m > 0]
    if in_band:
        print(f"{min(in_band):.1e} – {max(in_band):.1e}")
    else:
        print("none (per-tile D unidentifiable on this carve)")
    return results


def _synthetic_refit(
    graph, edges_in, boundary_nodes, interior_nodes,
    pin_idx, pin_node,
    Q_DC_syn, Q_H1_syn, f0_hz, spec, px_size_m, sigma_Q_si,
):
    """Internal helper: run the alternating LS loop on synthetic Q's
    without going through the PIV-record extraction path.  Returns
    (D_hat, sigma_D, dc_ratio, h1_ratio, n_iter, converged).
    """
    n_edges = len(edges_in)
    n_bnd = len(boundary_nodes)
    valid_dc = np.ones(n_edges, dtype=bool) if Q_DC_syn is not None \
        else np.zeros(n_edges, dtype=bool)
    valid_h1 = np.ones(n_edges, dtype=bool) if Q_H1_syn is not None \
        else np.zeros(n_edges, dtype=bool)
    n_dc = int(valid_dc.sum())
    n_h1 = int(valid_h1.sum())

    D = float(spec.D_init)
    P_DC = np.zeros(n_bnd, dtype=complex)
    P_H1 = np.zeros(n_bnd, dtype=complex)
    converged = False
    nL_per_m3 = 1.0e12

    for it in range(int(spec.max_iter)):
        ab = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            D, spec.mu, f0_hz, spec.harmonics, px_size_m)
        T_all = _compute_transfer_matrices(
            ab, edges_in, boundary_nodes, interior_nodes)
        T_DC = T_all[0]
        if Q_DC_syn is not None:
            Q_DC_full = Q_DC_syn.astype(complex)
            w_DC = valid_dc.astype(float)
            P_DC, _, _ = _solve_pressures_complex_wls(
                T_DC, Q_DC_full, w_DC,
                lambda_reg=spec.lambda_reg,
                p_scale=spec.P_scale_Pa,
                p_scale_fixed=spec.P_scale_Pa_fixed,
                pin_idx=pin_idx)
        if Q_H1_syn is not None and (1 in spec.harmonics):
            T_H1 = T_all[1]
            w_H1 = valid_h1.astype(float)
            P_H1, _, _ = _solve_pressures_complex_wls(
                T_H1, Q_H1_syn, w_H1,
                lambda_reg=spec.lambda_reg,
                p_scale=spec.P_scale_Pa,
                p_scale_fixed=spec.P_scale_Pa_fixed,
                pin_idx=pin_idx)

        D1 = D * (1.0 + spec.eps_D)
        ab_p = _build_admittance_system(
            graph, edges_in, boundary_nodes, interior_nodes,
            D1, spec.mu, f0_hz, spec.harmonics, px_size_m)
        T_pert = _compute_transfer_matrices(
            ab_p, edges_in, boundary_nodes, interior_nodes)
        dD = D1 - D
        dT = {n: (T_pert[n] - T_all[n]) / dD for n in T_all}

        num_eps = 0.0
        den_eps = 0.0
        if Q_DC_syn is not None:
            Q_pred_DC_curr = (T_DC @ P_DC).real
            r_DC = Q_DC_syn - Q_pred_DC_curr
            b_DC = (dT[0] @ P_DC).real
            sd = max(float(np.std(r_DC)), 1e-30) \
                if r_DC.size > 1 else 1.0
            w = 1.0 / sd ** 2
            num_eps += w * float(np.sum(r_DC * b_DC))
            den_eps += w * float(np.sum(b_DC ** 2))
        if Q_H1_syn is not None and (1 in spec.harmonics):
            Q_pred_H1_curr = T_all[1] @ P_H1
            r_H1 = Q_H1_syn - Q_pred_H1_curr
            b_H1 = (dT[1] @ P_H1)
            sh = max(float(np.std(np.abs(r_H1))), 1e-30) \
                if r_H1.size > 1 else 1.0
            w = 1.0 / sh ** 2
            num_eps += w * float(np.sum(np.real(
                np.conj(b_H1) * r_H1)))
            den_eps += w * float(np.sum(np.abs(b_H1) ** 2))

        if abs(den_eps) > 1e-30:
            eps = num_eps / den_eps
            D_new = float(np.clip(D + eps, 1e-12, 1.0))
        else:
            D_new = D
        rel_dD = abs(D_new - D) / max(abs(D), 1e-30)
        D = D_new
        if rel_dD < spec.tol_rel:
            converged = True
            break

    # σ_D from den_eps (Fisher info on D at convergence).
    sigma_D = float(1.0 / np.sqrt(den_eps)) if den_eps > 0 else float('inf')

    # Forward-compute fit ratios
    ab = _build_admittance_system(
        graph, edges_in, boundary_nodes, interior_nodes,
        D, spec.mu, f0_hz, spec.harmonics, px_size_m)
    T_all = _compute_transfer_matrices(
        ab, edges_in, boundary_nodes, interior_nodes)
    dc_r = h1_r = float('nan')
    if Q_DC_syn is not None:
        Q_pred_DC = (T_all[0] @ P_DC).real
        qm = float(np.sqrt(np.mean(Q_DC_syn ** 2))) or 1e-30
        dc_r = float(np.sqrt(np.mean(Q_pred_DC ** 2))) / qm
    if Q_H1_syn is not None and (1 in spec.harmonics):
        Q_pred_H1 = T_all[1] @ P_H1
        qm = float(np.sqrt(np.mean(np.abs(Q_H1_syn) ** 2))) or 1e-30
        h1_r = float(np.sqrt(np.mean(np.abs(Q_pred_H1) ** 2))) / qm

    return D, sigma_D, dc_r, h1_r, it + 1, converged


def well_constrained_subset_analysis(
    graph,
    *,
    sigma_over_D_cutoff: float = 0.3,
):
    """Restrict the dual-harmonic sweep to tiles where both H1 and H2
    fits are well-constrained (σ_D/D̂ < cutoff for each harmonic).
    Compute the median log-ratio, 95% CI, and translate to a bound on
    the SLS viscoelastic stiffness ratio E_inst/E_relax for relaxations
    centered in the H1–H2 band.  Also produces the regression-to-prior
    sanity check: D̂ distribution of constrained subset vs full pop.

    Reads `graph.graph['dual_harmonic_sweep']` populated by
    `dual_harmonic_sweep_all_tiles`.
    """
    import matplotlib.pyplot as plt
    sweep = graph.graph.get('dual_harmonic_sweep')
    if sweep is None or not sweep.get('rows'):
        raise RuntimeError(
            "No saved sweep data on graph — run "
            "`dual_harmonic_sweep_all_tiles` first.")
    rows = sweep['rows']
    Ds_H1_full = np.array([r['D_H1'] for r in rows])
    Ds_H2_full = np.array([r['D_H2'] for r in rows])
    s1_full = np.array([r['sigma_D_H1'] for r in rows])
    s2_full = np.array([r['sigma_D_H2'] for r in rows])

    # Filter: well-constrained requires σ/D < cutoff at both harmonics
    # AND neither D̂ at the lower clamp (1e-12), which signals
    # data-uninformative fits.
    mask = (
        (s1_full / np.maximum(np.abs(Ds_H1_full), 1e-30)
         < sigma_over_D_cutoff) &
        (s2_full / np.maximum(np.abs(Ds_H2_full), 1e-30)
         < sigma_over_D_cutoff) &
        (Ds_H1_full > 1e-10) & (Ds_H2_full > 1e-10)
    )
    n_full = len(rows)
    n_sub = int(mask.sum())
    print(f"\n[Well-constrained subset]  σ/D < {sigma_over_D_cutoff} "
          f"at both harmonics, D > 1e-10  → {n_sub}/{n_full} tiles")
    if n_sub < 3:
        print("  Too few well-constrained tiles for a meaningful "
              "bound — try a looser cutoff (e.g. 0.5).")
        return {'n_sub': n_sub, 'n_full': n_full}

    Ds_H1 = Ds_H1_full[mask]
    Ds_H2 = Ds_H2_full[mask]
    s1 = s1_full[mask]
    s2 = s2_full[mask]
    log_ratio = np.log(Ds_H1 / Ds_H2)
    # Per-tile σ on log-ratio (propagation):
    #   σ_log(D̂_H1/D̂_H2)² = (σ_H1/D̂_H1)² + (σ_H2/D̂_H2)²
    sig_log = np.sqrt((s1 / Ds_H1) ** 2 + (s2 / Ds_H2) ** 2)

    # Inverse-variance-weighted mean + t-based CI.  z-based intervals
    # over-claim significance at small N — at n=5, t_{4, 0.025} = 2.776
    # widens the CI by ~40% over a z-based one and converts a nominal
    # 3.5σ to ~p=0.025.  Use the more conservative form.
    weights = 1.0 / np.maximum(sig_log ** 2, 1e-30)
    mean_log = float(np.sum(weights * log_ratio) / np.sum(weights))
    se_mean = float(1.0 / np.sqrt(np.sum(weights)))
    n = len(log_ratio)
    try:
        from scipy.stats import t as _t_dist
        t_crit = float(_t_dist.ppf(0.975, df=max(n - 1, 1)))
    except Exception:
        t_crit = 1.96  # z fallback
    lo = mean_log - t_crit * se_mean
    hi = mean_log + t_crit * se_mean
    median_log = float(np.median(log_ratio))

    ratio_mean = float(np.exp(mean_log))
    ratio_med = float(np.exp(median_log))
    band_lo = float(np.exp(lo))
    band_hi = float(np.exp(hi))

    print(f"  log(D̂_H1/D̂_H2):  "
          f"weighted-mean = {mean_log:+.3f} ± {se_mean:.3f}  "
          f"(SE)")
    print(f"  median ratio D̂_H1/D̂_H2 = {ratio_med:.3f}  "
          f"95% CI [{band_lo:.3f}, {band_hi:.3f}]")
    print(f"  weighted-mean ratio   = {ratio_mean:.3f}  "
          f"±{(ratio_mean * se_mean):.3f} (1σ on linear scale)")

    # SLS-bound translation: max stiffness ratio across the octave
    # for a Debye-like relaxation centered in-band is ~E_inst/E_relax.
    # Our null bound says |log ratio| < max(|log(band_lo)|,
    # |log(band_hi)|).  Express that as a viscoelastic constraint.
    bound = max(abs(np.log(band_lo)), abs(np.log(band_hi)))
    print(f"\n  Viscoelastic constraint (Debye / SLS, band-centered "
          f"relaxation):")
    print(f"    |log(E_inst/E_relax)| ≤ {bound:.3f}  →  "
          f"E_inst/E_relax ≤ {np.exp(bound):.3f}")
    print(f"    (Slow relaxations below H1 or fast above H2 are not "
          f"probed; bound applies only to ωτ ~ 1 in [H1, H2].)")
    print(f"  Lumped-regime check: the same null rules out wave-term "
          f"contamination at H2 to the same level.")

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              num='Well-constrained subset')

    # (a) Subset scatter with error bars + CI band
    axes[0].errorbar(Ds_H1, Ds_H2, xerr=s1, yerr=s2,
                     fmt='o', ms=6, alpha=0.9, color='#5A4FCF',
                     ecolor='gray', capsize=2,
                     label=f'subset (n={n_sub})')
    rng_x = (min(Ds_H1.min(), Ds_H2.min()) * 0.5,
             max(Ds_H1.max(), Ds_H2.max()) * 2.0)
    x_ref = np.array([rng_x[0], rng_x[1]])
    axes[0].plot(x_ref, x_ref, 'k--', lw=1.2, alpha=0.6,
                 label='D̂_H1 = D̂_H2')
    axes[0].fill_between(x_ref, x_ref / band_lo, x_ref / band_hi,
                         color='red', alpha=0.12,
                         label=f'95% CI on ratio  '
                                f'[{band_lo:.2f}, {band_hi:.2f}]')
    axes[0].plot(x_ref, x_ref / ratio_med, 'r--', lw=1.2,
                 label=f'median ratio = {ratio_med:.2f}')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('D̂_H1 (1/Pa)')
    axes[0].set_ylabel('D̂_H2 (1/Pa)')
    axes[0].set_title(
        f'Well-constrained subset  σ/D̂ < {sigma_over_D_cutoff}\n'
        f'E_inst/E_relax ≤ {np.exp(bound):.2f} '
        f'(SLS / Debye-band centered)')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].grid(alpha=0.3, which='both')

    # (b) Sanity check: D̂ distribution constrained vs full
    # If the constrained subset is at one end of the full D range,
    # the H1/H2 agreement there might be regression-to-prior, not
    # independent agreement.  Plot histograms of log-D̂ for both.
    finite_full_h1 = Ds_H1_full[(Ds_H1_full > 1e-10)
                                  & np.isfinite(Ds_H1_full)]
    finite_sub_h1 = Ds_H1
    bins = np.logspace(
        np.log10(min(finite_full_h1.min(), finite_sub_h1.min())),
        np.log10(max(finite_full_h1.max(), finite_sub_h1.max())),
        20)
    axes[1].hist(finite_full_h1, bins=bins, alpha=0.5, color='gray',
                 edgecolor='black', linewidth=0.5,
                 label=f'full population (n={len(finite_full_h1)})')
    axes[1].hist(finite_sub_h1, bins=bins, alpha=0.85,
                 color='#5A4FCF', edgecolor='black', linewidth=0.5,
                 label=f'subset (n={n_sub})')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('D̂_H1 (1/Pa)')
    axes[1].set_ylabel('# tiles')
    axes[1].set_title(
        'Sanity check: D̂_H1 distribution\n'
        '(subset should overlap the bulk; otherwise the agreement\n'
        'is regression-to-prior, not independent physical agreement)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    try:
        plt.tight_layout()
    except Exception:
        pass
    try:
        import os, datetime as _dt
        out_dir = os.path.expanduser(
            '~/Downloads/pertile_diagnostics')
        os.makedirs(out_dir, exist_ok=True)
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(
            out_dir, f'well_constrained_subset_{ts}.png')
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        print(f"\n  saved figure to {out_path}")
    except Exception as _e:
        print(f"  save failed: {_e}")
    plt.show(block=False)
    try:
        fig.canvas.manager.window.raise_()
        fig.canvas.manager.window.activateWindow()
    except Exception:
        pass

    return {
        'n_sub': n_sub, 'n_full': n_full,
        'log_ratio': log_ratio.tolist(),
        'sig_log': sig_log.tolist(),
        'mean_log': mean_log, 'se_mean': se_mean,
        'median_log': median_log,
        'ci_log': (lo, hi),
        'median_ratio': ratio_med,
        'ratio_95CI': (band_lo, band_hi),
        'E_ratio_bound': float(np.exp(bound)),
    }


def replot_dual_harmonic_sweep(graph):
    """Regenerate the dual-harmonic sweep scatter plots from saved data
    in ``graph.graph['dual_harmonic_sweep']`` (populated by a previous
    `dual_harmonic_sweep_all_tiles` run).  Use when the figure window
    got closed or hidden behind napari and you don't want to rerun the
    full sweep.
    """
    import matplotlib.pyplot as plt
    sweep = graph.graph.get('dual_harmonic_sweep')
    if sweep is None or not sweep.get('rows'):
        raise RuntimeError(
            "No saved dual_harmonic_sweep on graph — run the sweep "
            "button first.")
    rows = sweep['rows']
    ratio_med = float(sweep.get('ratio_median', 1.0))
    Ds_H1 = np.array([r['D_H1'] for r in rows])
    Ds_H2 = np.array([r['D_H2'] for r in rows])
    s1 = np.array([r['sigma_D_H1'] for r in rows])
    s2 = np.array([r['sigma_D_H2'] for r in rows])
    zs = np.array([r['z'] for r in rows])
    snr_h2 = np.array([r['snr_H2_median'] for r in rows])
    n_fail = int(np.sum(zs > 3))
    n_marg = int(np.sum((zs >= 2) & (zs <= 3)))
    n_ok = int(np.sum(zs < 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              num='Dual-harmonic sweep')
    axes[0].errorbar(Ds_H1, Ds_H2, xerr=s1, yerr=s2,
                     fmt='o', ms=5, alpha=0.7, color='#5A4FCF',
                     ecolor='gray', capsize=2)
    rng = (min(Ds_H1.min(), Ds_H2.min()) * 0.5,
           max(Ds_H1.max(), Ds_H2.max()) * 2.0)
    x_ref = np.array([rng[0], rng[1]])
    axes[0].plot(x_ref, x_ref, 'k--', lw=1, alpha=0.5,
                 label='D̂_H1 = D̂_H2  (no mismatch)')
    if ratio_med > 0:
        axes[0].plot(x_ref, x_ref / ratio_med, 'r--', lw=1, alpha=0.7,
                     label=f'slope {ratio_med:.2f}:1  (median ratio)')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('D̂_H1 (1/Pa)')
    axes[0].set_ylabel('D̂_H2 (1/Pa)')
    axes[0].set_title(
        f'Per-tile D̂ at H1 vs H2  '
        f'({n_ok}/{n_marg}/{n_fail} OK/marg/fail)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which='both')

    D_ratio = Ds_H1 / np.maximum(Ds_H2, 1e-30)
    mask = np.isfinite(snr_h2) & np.isfinite(D_ratio)
    if mask.any():
        sc = axes[1].scatter(snr_h2[mask], D_ratio[mask],
                              c=zs[mask], cmap='RdYlGn_r',
                              vmin=0, vmax=5,
                              alpha=0.8, s=40,
                              edgecolor='black', linewidth=0.5)
        plt.colorbar(sc, ax=axes[1], label='|ΔD|/σ')
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(snr_h2[mask], D_ratio[mask])
            title_extra = f'(Spearman ρ = {rho:.2f}, p = {pval:.3f})'
        except Exception:
            title_extra = ''
        axes[1].axhline(1.0, color='gray', ls='--', alpha=0.5)
        if ratio_med > 0:
            axes[1].axhline(ratio_med, color='red', ls='--',
                            alpha=0.5,
                            label=f'median ratio = {ratio_med:.2f}')
        axes[1].set_xlabel('median SNR_H2 per tile')
        axes[1].set_ylabel('D̂_H1 / D̂_H2')
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
        axes[1].set_title(f'SNR-vs-ratio  {title_extra}\n'
                           f'anti-correlation ⇒ prior dominates H2 '
                           f'at low SNR')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which='both')
    try:
        plt.tight_layout()
    except Exception:
        pass
    # Save to disk in addition to showing — matplotlib + Qt sometimes
    # leaves the figure hidden behind napari and the user can't find
    # it.  Saving gives a guaranteed-reachable copy.
    try:
        import os, datetime as _dt
        out_dir = os.path.expanduser(
            '~/Downloads/pertile_diagnostics')
        os.makedirs(out_dir, exist_ok=True)
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir,
                                  f'dual_harmonic_sweep_{ts}.png')
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        print(f"  [replot] saved figure to {out_path}")
    except Exception as _e:
        print(f"  [replot] save failed: {_e}")
    plt.show(block=False)
    try:
        fig.canvas.manager.window.raise_()
        fig.canvas.manager.window.activateWindow()
    except Exception:
        pass
    return fig


def dual_harmonic_sweep_all_tiles(
    graph,
    tile_ids,
    *,
    base_spec_factory,
    plot: bool = True,
):
    """Run the dual-harmonic (H1, H2) inference on every tile and
    produce the two diagnostic plots from the artifact-check battery:

      (1) D̂_H1 vs D̂_H2 scatter with error bars (per tile).
          Real physical effect ⇒ tight cluster on a line of slope ≠ 1.
          Prior leakage ⇒ wide scatter, only the mean is offset.

      (2) SNR-vs-D-ratio correlation.
          For each tile, median per-edge SNR at H1 and H2 (from the
          fitted noise models).  Scatter (D̂_H1 / D̂_H2) vs
          (median SNR_H2).  Anti-correlation ⇒ prior dominates the
          H2 fit at low SNR; invariance ⇒ structural.

    `base_spec_factory(harmonic)` returns a fresh `LocalInferenceSpec`
    for the given harmonic.  Caller passes the rest of the
    inference knobs (P_scale, mu, prune flags, etc.) baked in.

    Returns dict per tile with both fits' summaries and per-tile SNR
    medians.
    """
    import time
    rows = []
    failures = []
    print(f"\n[Dual-harmonic sweep] {len(tile_ids)} tiles, "
          f"both H1 and H2…")
    print(f"  {'tile':>4}  {'D̂_H1':>11} ±{'σ_H1':>10}   "
          f"{'D̂_H2':>11} ±{'σ_H2':>10}   "
          f"{'|ΔD|/σ':>7}  "
          f"{'SNR_H1':>7} {'SNR_H2':>7}  {'cond':>5}  "
          f"{'time':>6}")
    t_start = time.time()
    for i_tile, tid in enumerate(tile_ids):
        t0 = time.time()
        # Live progress so a slow/stuck tile is visible immediately.
        print(f"  tile {tid:>3} ({i_tile + 1}/{len(tile_ids)}) "
              f"H1…", end='', flush=True)
        try:
            res1 = infer_local(graph, int(tid),
                                base_spec_factory(1))
            t1 = time.time() - t0
            print(f" done ({res1.iterations} iter, {t1:.1f}s); "
                  f"H2…", end='', flush=True)
            t_h2 = time.time()
            res2 = infer_local(graph, int(tid),
                                base_spec_factory(2))
            t2 = time.time() - t_h2
            print(f" done ({res2.iterations} iter, {t2:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f" FAILED: {str(e).split(chr(10))[0][:60]}",
                  flush=True)
            failures.append((tid, str(e).split('\n')[0][:60]))
            continue
        # Per-tile median SNR from the FGLS noise models.
        from .inference import evaluate_noise_model as _en
        snr_h1 = float('nan')
        snr_h2 = float('nan')
        try:
            if res1.noise_model_h1 is not None and res1.valid_h1.any():
                sig = _en(res1.noise_model_h1,
                          np.abs(res1.Q_meas_H1[res1.valid_h1]))
                snr_h1 = float(np.median(
                    np.abs(res1.Q_meas_H1[res1.valid_h1])
                    / np.maximum(sig, 1e-30)))
            if res2.noise_model_h1 is not None and res2.valid_h1.any():
                sig = _en(res2.noise_model_h1,
                          np.abs(res2.Q_meas_H1[res2.valid_h1]))
                snr_h2 = float(np.median(
                    np.abs(res2.Q_meas_H1[res2.valid_h1])
                    / np.maximum(sig, 1e-30)))
        except Exception:
            pass

        D1, s1 = res1.D_hat, res1.sigma_D
        D2, s2 = res2.D_hat, res2.sigma_D
        s_comb = np.sqrt(s1 ** 2 + s2 ** 2)
        z = abs(D1 - D2) / max(s_comb, 1e-30)
        verdict_short = ("OK" if z < 2 else
                         "?" if z < 3 else "FAIL")
        print(f"  {tid:>4}  {D1:>11.3e} ±{s1:>10.2e}   "
              f"{D2:>11.3e} ±{s2:>10.2e}   "
              f"{z:>6.2f}σ  "
              f"{snr_h1:>7.2f} {snr_h2:>7.2f}  {verdict_short:>5}")
        rows.append({
            'tile_id': int(tid),
            'D_H1': float(D1), 'sigma_D_H1': float(s1),
            'D_H2': float(D2), 'sigma_D_H2': float(s2),
            'z': float(z),
            'snr_H1_median': snr_h1,
            'snr_H2_median': snr_h2,
            'chi2_H1': float(res1.chi2_red),
            'chi2_H2': float(res2.chi2_red),
        })

    if not rows:
        print("  → no successful fits.")
        return {'rows': [], 'failures': failures}

    # Aggregate verdict
    Ds_H1 = np.array([r['D_H1'] for r in rows])
    Ds_H2 = np.array([r['D_H2'] for r in rows])
    zs = np.array([r['z'] for r in rows])
    n_fail = int(np.sum(zs > 3))
    n_marg = int(np.sum((zs >= 2) & (zs <= 3)))
    n_ok = int(np.sum(zs < 2))
    ratio_med = float(np.median(Ds_H1 / np.maximum(Ds_H2, 1e-30)))
    print(f"\n  Summary: {n_ok} consistent / {n_marg} marginal "
          f"/ {n_fail} falsified  (of {len(rows)} fits).")
    print(f"  median D̂_H1/D̂_H2 = {ratio_med:.2f}")
    if ratio_med > 3.0:
        print(f"  → If artifact checks come up clean, this is a "
              f"super-Debye stiffening signature.")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # (a) D̂_H1 vs D̂_H2
        s1_arr = np.array([r['sigma_D_H1'] for r in rows])
        s2_arr = np.array([r['sigma_D_H2'] for r in rows])
        axes[0].errorbar(Ds_H1, Ds_H2, xerr=s1_arr, yerr=s2_arr,
                         fmt='o', ms=5, alpha=0.7, color='#5A4FCF',
                         ecolor='gray', capsize=2)
        rng = (min(Ds_H1.min(), Ds_H2.min()) * 0.5,
               max(Ds_H1.max(), Ds_H2.max()) * 2.0)
        x_ref = np.array([rng[0], rng[1]])
        axes[0].plot(x_ref, x_ref, 'k--', lw=1, alpha=0.5,
                     label='D̂_H1 = D̂_H2  (no mismatch)')
        axes[0].plot(x_ref, x_ref / ratio_med, 'r--', lw=1, alpha=0.7,
                     label=f'slope {ratio_med:.1f}:1  '
                            f'(median ratio)')
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
        axes[0].set_xlabel('D̂_H1 (1/Pa)')
        axes[0].set_ylabel('D̂_H2 (1/Pa)')
        axes[0].set_title(
            f'Per-tile D̂ at H1 vs H2  '
            f'({n_ok}/{n_marg}/{n_fail} OK/marg/fail)')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3, which='both')

        # (b) D ratio vs H2 SNR
        snr_h2 = np.array([r['snr_H2_median'] for r in rows])
        D_ratio = Ds_H1 / np.maximum(Ds_H2, 1e-30)
        mask = np.isfinite(snr_h2) & np.isfinite(D_ratio)
        if mask.any():
            sc = axes[1].scatter(snr_h2[mask], D_ratio[mask],
                                 c=zs[mask], cmap='RdYlGn_r',
                                 vmin=0, vmax=5,
                                 alpha=0.8, s=40,
                                 edgecolor='black', linewidth=0.5)
            plt.colorbar(sc, ax=axes[1], label='|ΔD|/σ')
            # Correlation (Spearman, robust)
            from scipy.stats import spearmanr
            rho, pval = spearmanr(snr_h2[mask], D_ratio[mask])
            axes[1].axhline(1.0, color='gray', ls='--', alpha=0.5)
            axes[1].axhline(ratio_med, color='red', ls='--',
                            alpha=0.5,
                            label=f'median ratio = {ratio_med:.1f}')
            axes[1].set_xlabel('median SNR_H2 per tile')
            axes[1].set_ylabel('D̂_H1 / D̂_H2')
            axes[1].set_yscale('log')
            axes[1].set_xscale('log')
            axes[1].set_title(
                f'SNR-vs-ratio correlation  '
                f'(Spearman ρ = {rho:.2f}, p = {pval:.3f})\n'
                f'anti-correlation ⇒ prior dominates H2 at low SNR')
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3, which='both')
        plt.tight_layout()
        plt.show(block=False)
        try:
            fig.canvas.manager.window.raise_()
        except Exception:
            pass

    # Persist on graph for later reference
    graph.graph['dual_harmonic_sweep'] = {
        'rows': rows, 'failures': failures,
        'ratio_median': ratio_med,
    }
    return {'rows': rows, 'failures': failures,
            'ratio_median': ratio_med}


def kappa_L_check(
    graph,
    tile_id: int,
    *,
    D: float = 8.6e-4,
    mu: float = 3.5e-3,
    px_size_m: float = 1.7e-6,
    harmonic: int = 2,
    f0_hz: Optional[float] = None,
    plot: bool = True,
):
    """Lumped-regime check |κL| per edge at the requested harmonic.

    For a transmission line with series resistance R = 1/G (Ω·s/m³)
    and shunt capacitance C (m³/Pa) on a length-L edge::

        κL  =  √(jωC / G)            (dimensionless propagation phase)
        |κL|²  =  ωτ                  with τ = C/G

    Substituting G = πR⁴/(8μL) and C = πR²DL (areal distensibility)
    gives

        τ  =  8 μ L² D / R²           per edge
        |κL|  =  2√2 · L √(ω μ D) / R

    Lumped-π assumes |κL| ≪ 1.  At H1 this should hold network-wide
    (you verified ωτ ≲ 10⁻³).  At H2, ω doubles → |κL| grows by √2,
    which on the smallest-R edges can push it past 0.3 even when H1
    is comfortably lumped.  Returns the per-edge distribution; values
    > 0.3 indicate wave-propagation terms entering and the lumped-π
    model leaking on those edges (an alternative explanation for any
    H1/H2 D mismatch).

    NOTE: pre-2026-05-18 used C = 2πR²DL (radius convention) which
    gave coefficient 4 (not 2√2) in the |κL| formula.

    Returns dict with keys: 'kL' (array per edge), 'R_px', 'L_px',
    'median', 'p95', 'max', 'frac_above_03', 'frac_above_01'.
    """
    edges_in, _, _, _ = extract_tile_subgraph_spatial(graph, int(tile_id))
    if f0_hz is None:
        f0s = []
        for u, v in edges_in:
            for m in (graph.edges[u, v].get('measurements_piv', [])
                       or []):
                if m.get('tile_id') == int(tile_id):
                    f0 = m.get('f0_hz') or m.get('f0')
                    if f0 is not None and np.isfinite(f0) and f0 > 0:
                        f0s.append(float(f0))
        f0_hz = float(np.median(f0s)) if f0s else 2.5
    omega = 2.0 * np.pi * int(harmonic) * float(f0_hz)
    kL = []
    R_px_arr = []
    L_px_arr = []
    for u, v in edges_in:
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0:
            continue
        # Areal-distensibility convention (C = πR²DL): coefficient
        # is 2√2 (was 4 under the old radius convention).
        kL_e = 2.0 * np.sqrt(2.0) * L_m * np.sqrt(omega * mu * D) / R_m
        kL.append(float(kL_e))
        R_px_arr.append(float(R_m / px_size_m))
        L_px_arr.append(float(L_m / px_size_m))
    kL = np.array(kL)
    R_px_arr = np.array(R_px_arr)
    L_px_arr = np.array(L_px_arr)

    med = float(np.median(kL))
    p95 = float(np.percentile(kL, 95))
    mx = float(np.max(kL))
    f03 = float(np.mean(kL > 0.3))
    f01 = float(np.mean(kL > 0.1))

    if mx < 0.1:
        verdict = (f"lumped-π safe at H{harmonic} — max |κL| < 0.1")
    elif mx < 0.3:
        verdict = (f"marginal at H{harmonic} — max |κL| = {mx:.2f}, "
                   f"a few edges approaching wave regime")
    else:
        verdict = (f"LUMPED-π LEAKING at H{harmonic} — max |κL| = {mx:.2f}, "
                   f"{100*f03:.1f}% of edges > 0.3.  This alone could "
                   f"explain an H1/H2 D mismatch.")

    print(f"\n  [κL check] tile {tile_id}  H{harmonic}  "
          f"(D={D:.2e}, f₀={f0_hz:.3f} Hz, ω={omega:.2f} rad/s)")
    print(f"    |κL| per edge: median={med:.3f}, 95%-ile={p95:.3f}, "
          f"max={mx:.3f}")
    print(f"    fraction > 0.30: {100*f03:.1f}%")
    print(f"    fraction > 0.10: {100*f01:.1f}%")
    print(f"    → {verdict}")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(kL, bins=50, color='#5A4FCF',
                     edgecolor='black', alpha=0.85)
        axes[0].axvline(0.1, color='orange', ls='--',
                         label='|κL|=0.1 (marginal)')
        axes[0].axvline(0.3, color='red', ls='--',
                         label='|κL|=0.3 (wave regime)')
        axes[0].axvline(med, color='black', ls='-', lw=1.5,
                         label=f'median = {med:.2f}')
        axes[0].set_xlabel(f'|κL| per edge at H{harmonic}')
        axes[0].set_ylabel('# edges')
        axes[0].set_title(
            f'Tile {tile_id}: lumped-π check at H{harmonic}\n{verdict}')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # |κL| vs R (the dependence is 1/R → small vessels dominate)
        axes[1].loglog(R_px_arr, kL, '.', ms=4, color='#5A4FCF',
                       alpha=0.5)
        axes[1].axhline(0.1, color='orange', ls='--', alpha=0.6)
        axes[1].axhline(0.3, color='red', ls='--', alpha=0.6)
        axes[1].set_xlabel('R (pixels)')
        axes[1].set_ylabel(f'|κL| at H{harmonic}')
        axes[1].set_title(
            'Wave-regime risk scales as 1/R '
            f'(slope -1 in log-log)')
        axes[1].grid(alpha=0.3, which='both')
        plt.tight_layout()
        plt.show()

    return {
        'kL': kL, 'R_px': R_px_arr, 'L_px': L_px_arr,
        'median': med, 'p95': p95, 'max': mx,
        'frac_above_03': f03, 'frac_above_01': f01,
        'omega': float(omega), 'D': float(D), 'f0_hz': float(f0_hz),
        'verdict': verdict,
    }


def compute_h2_fields_on_graph(graph) -> int:
    """Walk every edge with PIV records and write per-edge H1+H2 fields:

      Per-harmonic:
        ``amp_Q_h1_piv``, ``amp_Q_h2_piv``      — |Q_Hn| in nL/s
        ``phase_h1_piv``, ``phase_h2_piv``      — arg(Q_Hn) in radians

      Derived:
        ``h2_h1_ratio_piv``                     — |Q_H2|/|Q_H1|
        ``h2_phase_offset_piv``                 — arg(Q_H2) − 2·arg(Q_H1)

    These become plottable via the viewer's source dispatch — fields
    with the ``_piv`` suffix are resolved by `_resolve_field` when
    Source = PIV.  For Local Sim, see `persist_result_to_graph` which
    writes the equivalent ``_local`` fields per harmonic.

    Returns the number of edges updated.  Skips edges with no PIV
    records, or with only H1 (no H2 phasor).
    """
    from .inference import _meas_phasors_for_edge
    n_updated = 0
    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv', []) or []
        if not piv:
            continue
        amps_h1 = []
        amps_h2 = []
        z_h1_unit = []
        z_h2_unit = []
        offsets = []
        for m_ref in piv:
            try:
                _, Q_hn_dict, _ = _meas_phasors_for_edge(
                    (u, v), m_ref, harmonics=(1, 2))
            except Exception:
                continue
            q1 = Q_hn_dict.get(1)
            q2 = Q_hn_dict.get(2)
            if q1 is None or q2 is None:
                continue
            a1 = abs(complex(q1)) if np.isfinite(q1) else 0.0
            a2 = abs(complex(q2)) if np.isfinite(q2) else 0.0
            if a1 > 0:
                amps_h1.append(a1)
                z_h1_unit.append(complex(q1) / a1)
            if a2 > 0:
                amps_h2.append(a2)
                z_h2_unit.append(complex(q2) / a2)
            # Phase offset: arg(Q_H2) − 2·arg(Q_H1), via complex math
            # to avoid manual wrap.  exp(i·(φ2 − 2·φ1)) = Q_H2 · conj(Q_H1)²
            # / (|Q_H2| · |Q_H1|²).
            if a1 > 0 and a2 > 0:
                num = complex(q2) * np.conj(complex(q1)) ** 2
                den = a2 * a1 ** 2
                if den > 0:
                    offsets.append(num / den)
        if not amps_h1 and not amps_h2:
            continue
        if amps_h1:
            d['amp_Q_h1_piv'] = float(np.median(amps_h1))
            z_mean1 = np.mean(z_h1_unit) if z_h1_unit else 0
            d['phase_h1_piv'] = (float(np.angle(z_mean1))
                                  if abs(z_mean1) > 0 else float('nan'))
        if amps_h2:
            d['amp_Q_h2_piv'] = float(np.median(amps_h2))
            z_mean2 = np.mean(z_h2_unit) if z_h2_unit else 0
            d['phase_h2_piv'] = (float(np.angle(z_mean2))
                                  if abs(z_mean2) > 0 else float('nan'))
        # `h2_h1_ratio` and `phase_h2_rel` are already populated lazily
        # by the viewer's `_ensure_derived_field` (legacy, no suffix);
        # don't write them here to avoid two writers with different
        # units (legacy is degrees; we'd be tempted to write radians).
        n_updated += 1
    graph.graph['h2_fields_computed'] = True
    return n_updated


def h1_h2_amplitude_check(
    graph,
    tile_id: int,
    *,
    plot: bool = True,
):
    """Pre-flight check for whether H2 carries useful D-information.

    The model predicts H2 is 2× more sensitive to D than H1 in the
    Jacobian, but only if H2 actually has comparable per-edge SNR.
    For a near-sinusoidal heart pulse |Q_H2|/|Q_H1| can be ≪ 1, in
    which case the H2-fit's effective Fisher info on D is killed by
    the (|Q_H2|/|Q_H1|)² penalty regardless of the frequency factor.

    Heuristics:
      median ratio > 0.3 → H2 is genuinely discriminating
      0.1–0.3          → H2 useful but tight
      < 0.05           → H2 won't tighten σ_D; only resolves gross
                          model failures

    Returns
    -------
    dict with keys: 'ratios' (per-edge |Q_H2|/|Q_H1|), 'median',
    'p05', 'p95', 'frac_above_03', 'frac_above_01', 'n_edges'.
    Prints a verdict line.  If `plot`, opens a matplotlib histogram.
    """
    from .inference import _meas_phasors_for_edge
    edges_in, _, _, _ = extract_tile_subgraph_spatial(graph, int(tile_id))
    ratios = []
    Q_h1_amp = []
    Q_h2_amp = []
    for u, v in edges_in:
        d = graph.edges[u, v]
        piv = d.get('measurements_piv', []) or []
        m_ref = next((m for m in piv
                       if m.get('tile_id') == int(tile_id)), None)
        if m_ref is None:
            continue
        try:
            _, Q_hn_dict, _ = _meas_phasors_for_edge(
                (u, v), m_ref, harmonics=(1, 2))
        except Exception:
            continue
        q1 = Q_hn_dict.get(1)
        q2 = Q_hn_dict.get(2)
        if q1 is None or q2 is None:
            continue
        a1 = abs(complex(q1)) if np.isfinite(q1) else 0.0
        a2 = abs(complex(q2)) if np.isfinite(q2) else 0.0
        if a1 <= 0:
            continue
        Q_h1_amp.append(a1)
        Q_h2_amp.append(a2)
        ratios.append(a2 / a1)
    if not ratios:
        print(f"  [H1/H2 check] tile {tile_id}: no PIV records with "
              f"both H1 and H2 phasors.")
        return {'ratios': [], 'median': float('nan'),
                'n_edges': 0}
    ratios = np.array(ratios, dtype=float)
    med = float(np.median(ratios))
    p05 = float(np.percentile(ratios, 5))
    p95 = float(np.percentile(ratios, 95))
    f30 = float((ratios > 0.30).mean())
    f10 = float((ratios > 0.10).mean())

    if med > 0.30:
        verdict = "discriminating — H2 will tighten σ_D significantly"
    elif med > 0.10:
        verdict = ("useful but tight — H2 will add some info, "
                   "joint H1+H2 worth building")
    elif med > 0.05:
        verdict = "marginal — H2 mostly catches gross model failures"
    else:
        verdict = ("formality — H2 amplitudes too small to "
                   "constrain D; skip joint fit")

    print(f"\n  [H1/H2 amplitude check] tile {tile_id}  "
          f"({len(ratios)} edges)")
    print(f"    |Q_H2|/|Q_H1|: median={med:.3f}, "
          f"5-95%=[{p05:.3f}, {p95:.3f}]")
    print(f"    fraction with ratio > 0.30: {100*f30:.1f}%")
    print(f"    fraction with ratio > 0.10: {100*f10:.1f}%")
    print(f"    → {verdict}")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(ratios, bins=40, color='#5A4FCF',
                     edgecolor='black', alpha=0.85)
        axes[0].axvline(med, color='black', ls='-',
                         label=f'median = {med:.3f}')
        axes[0].axvline(0.30, color='green', ls='--',
                         label='discriminating threshold (0.30)')
        axes[0].axvline(0.05, color='red', ls='--',
                         label='formality threshold (0.05)')
        axes[0].set_xlabel('|Q_H2| / |Q_H1| per edge')
        axes[0].set_ylabel('# edges')
        axes[0].set_title(
            f'Tile {tile_id}: H2/H1 amplitude ratio  ({verdict})')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].loglog(Q_h1_amp, Q_h2_amp, '.', ms=4,
                       color='#5A4FCF', alpha=0.6)
        qmax = max(max(Q_h1_amp), max(Q_h2_amp))
        qmin = min(min(Q_h1_amp), min(Q_h2_amp))
        # Reference lines: ratios 1, 0.3, 0.1, 0.05
        x_ref = np.array([qmin, qmax])
        for r, lbl, ls in [(1.0, '1×', '-'),
                           (0.3, '0.3', '--'),
                           (0.1, '0.1', ':'),
                           (0.05, '0.05', ':')]:
            axes[1].plot(x_ref, r * x_ref, ls=ls, color='gray',
                         alpha=0.5, label=lbl)
        axes[1].set_xlabel('|Q_H1| (nL/s)')
        axes[1].set_ylabel('|Q_H2| (nL/s)')
        axes[1].set_title('Per-edge H2 vs H1 amplitude')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which='both')
        plt.tight_layout()
        plt.show()

    return {
        'ratios': ratios,
        'Q_h1_amp': np.array(Q_h1_amp),
        'Q_h2_amp': np.array(Q_h2_amp),
        'median': med,
        'p05': p05, 'p95': p95,
        'frac_above_03': f30,
        'frac_above_01': f10,
        'n_edges': len(ratios),
        'verdict': verdict,
    }


def interior_identifiability_map(
    graph,
    tile_id: int,
    *,
    mu: float = 3.5e-3,
    px_size_m: float = 1.7e-6,
    k_modes: int = 8,
):
    """Per-interior-node identifiability via null-space analysis.

    Builds the interior DC Laplacian L_int (with boundary contributions
    on the diagonal as Dirichlet, same as the solver) and SVDs it.  The
    `k_modes` smallest singular vectors span the part of P_int that is
    NOT pinned by any boundary measurement — pinv min-norm sets those
    components to zero regardless of prior, so any interior nodes with
    large amplitude in those modes get P=0 in the inversion, and their
    incident edges under-predict Q.

    Returns
    -------
    interior_nodes : list of node ids
    weights        : (n_int,) array; weights[i] = Σ_k V_k[i]² over the
                     k smallest singular vectors.  Large value ⇒ this
                     interior node sits in the unidentifiable subspace.
    smallest_svals : (k,) array of singular values, smallest first.
    cond_L         : full cond(L_int) for reference.
    """
    edges_in, _, boundary, interior = \
        extract_tile_subgraph_spatial(graph, int(tile_id))
    n_int = len(interior)
    if n_int == 0:
        raise RuntimeError("No interior nodes in carve.")
    node_to_idx = {n: k for k, n in enumerate(interior)}

    L = np.zeros((n_int, n_int))
    for u, v in edges_in:
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0:
            continue
        G = float(np.pi * R_m ** 4 / (8.0 * mu * L_m))
        iu = node_to_idx.get(u)
        iv = node_to_idx.get(v)
        if iu is not None and iv is not None:
            L[iu, iu] += G; L[iv, iv] += G
            L[iu, iv] -= G; L[iv, iu] -= G
        elif iu is not None:
            L[iu, iu] += G       # boundary attachment (Dirichlet)
        elif iv is not None:
            L[iv, iv] += G

    U, s, Vt = np.linalg.svd(L, hermitian=True)
    k = int(min(k_modes, n_int))
    bottom_vecs = Vt[-k:, :]      # rows are right-singular vectors
    bottom_svals = s[-k:]          # smallest k singular values
    weights = np.sum(np.abs(bottom_vecs) ** 2, axis=0)
    cond_L = float(s[0] / max(s[-1], 1e-30))

    # Sort interior nodes by weight, descending.
    order = np.argsort(-weights)
    print(f"\n  Interior identifiability map — tile {tile_id}")
    print(f"  L_int: {n_int}×{n_int},  cond={cond_L:.2e}")
    print(f"  Smallest {k} singular values: "
          + ", ".join(f"{sv:.2e}" for sv in bottom_svals))
    print(f"  Bottom-{k}-mode weight per node (top 15 worst):")
    print(f"    {'rank':>4} {'node':>8} {'weight':>10}  "
          f"(x, y)")
    for rank, i in enumerate(order[:15]):
        n = interior[i]
        x = float(graph.nodes[n].get('x', 0.0))
        y = float(graph.nodes[n].get('y', 0.0))
        print(f"    {rank + 1:>4} {n:>8} {weights[i]:>10.3e}  "
              f"({x:.0f}, {y:.0f})")
    # How much of total null-space "mass" is in the top-1% of nodes?
    sorted_w = np.sort(weights)[::-1]
    n_top1 = max(1, n_int // 100)
    frac_top1 = float(sorted_w[:n_top1].sum() / max(weights.sum(), 1e-30))
    print(f"  → top-1% of nodes ({n_top1}) carry "
          f"{100 * frac_top1:.1f}% of bottom-{k}-mode mass.")
    return interior, weights, bottom_svals, cond_L


def boundary_coupling_report(
    graph,
    tile_id: int,
    *,
    mu: float = 3.5e-3,
    px_size_m: float = 1.7e-6,
    sigma_Q_nL_per_s: float = 0.1,
    P_scale_Pa: float = 100.0,
):
    """Per-boundary-node identifiability report.

    For each boundary node b in the spatial-rectangle carve, computes:

      G_attach    = Σ G_e for edges (b, interior)         [m³/(Pa·s)]
      n_edges     = number of incident interior edges
      τ_data      = Σ G_e² / σ_Q²                        (Fisher info on P_b)
      τ_prior     = 1 / σ_P²                              (prior precision)
      τ_total     = τ_data + τ_prior
      data_frac   = τ_data / τ_total                      ∈ [0, 1]

    `data_frac` ≈ 1 means the boundary node is well-determined by
    measurements (prior invisible).  `data_frac` ≈ 0 means the prior
    fully determines P_b — no useful info comes from this node.
    Intermediate values are exactly the regime where regularisation
    matters most.

    Prints a table sorted by data_frac (weakest first) so the
    problematic boundary nodes show up at the top.
    """
    edges_in, _, boundary, interior = \
        extract_tile_subgraph_spatial(graph, int(tile_id))
    interior_set = set(interior)
    sigma_Q_si = float(sigma_Q_nL_per_s) * 1e-12  # m³/s
    tau_prior = 1.0 / (float(P_scale_Pa) ** 2)

    # Compute G_e for all edges, and group incident edges per boundary.
    bnd_edges: dict[int, list] = {b: [] for b in boundary}
    for u, v in edges_in:
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0:
            continue
        Ge = float(np.pi * R_m ** 4 / (8.0 * mu * L_m))
        if u in bnd_edges and v in interior_set:
            bnd_edges[u].append((v, Ge, R_m / px_size_m, L_m / px_size_m))
        elif v in bnd_edges and u in interior_set:
            bnd_edges[v].append((u, Ge, R_m / px_size_m, L_m / px_size_m))

    rows = []
    for b, eds in bnd_edges.items():
        if not eds:
            G_attach = 0.0
            tau_data = 0.0
        else:
            G_attach = sum(ge for _, ge, _, _ in eds)
            tau_data = sum(ge ** 2 for _, ge, _, _ in eds) \
                / (sigma_Q_si ** 2)
        tau_total = tau_data + tau_prior
        data_frac = tau_data / tau_total if tau_total > 0 else 0.0
        rows.append((b, len(eds), G_attach, tau_data, data_frac, eds))

    rows.sort(key=lambda r: r[4])  # weakest first

    print(f"\n  Boundary coupling report — tile {tile_id} "
          f"(σ_Q={sigma_Q_nL_per_s} nL/s, σ_P={P_scale_Pa} Pa)")
    print(f"  {'node':>8} {'n_e':>4} {'G_attach':>12} "
          f"{'τ_data':>11} {'τ_prior':>11} {'data_frac':>10}")
    weak = 0
    for b, ne, G_attach, tau_data, data_frac, eds in rows:
        flag = " ← weak" if data_frac < 0.5 else ""
        if data_frac < 0.5:
            weak += 1
        print(f"  {b:>8} {ne:>4} {G_attach:>12.2e} "
              f"{tau_data:>11.2e} {tau_prior:>11.2e} "
              f"{data_frac:>9.3f}{flag}")
    print(f"  → {weak}/{len(rows)} boundary nodes are prior-dominated "
          f"(data_frac < 0.5).")
    if weak > 0:
        print(f"  Examining the weakest one's edges:")
        weakest = rows[0]
        for nbr, Ge, R_p, L_p in weakest[5][:5]:
            print(f"    edge → interior {nbr}: "
                  f"G={Ge:.2e}  R={R_p:.2f}px  L={L_p:.1f}px")
    # Also dump edges of the 3 STRONGEST boundary nodes — these are
    # the candidates for "artery" outliers that inflate cond(L_int).
    strongest = rows[-3:][::-1]
    if strongest:
        print(f"  Strongest boundary nodes (suspected arteries):")
        for b, ne, G_attach, tau_data, data_frac, eds in strongest:
            print(f"    node {b}  n_e={ne}  G_attach={G_attach:.2e}:")
            for nbr, Ge, R_p, L_p in eds:
                print(f"      edge → interior {nbr}: "
                      f"G={Ge:.2e}  R={R_p:.2f}px  L={L_p:.1f}px")
    return rows


def plot_carve_diagnostic(
    graph,
    tile_id: int,
    *,
    mu: float = 3.5e-3,
    px_size_m: float = 1.7e-6,
    bottleneck_factor: float = 100.0,
    figsize: tuple = (15, 7),
):
    """Diagnostic figure for a tile's spatial-rectangle carve.

    Visualises the carved subgraph **before** any inference runs, so
    you can see whether the carve produced a well-conditioned linear
    system or whether thin/short edges create bottleneck modes.

    Left panel: edges drawn at their geometric path positions, coloured
    by log10(G_e/G_median) where G_e = πR⁴/(8μL).  Bottleneck edges
    (G < G_median / bottleneck_factor) are drawn as red dashed lines so
    they pop visually.  Boundary nodes red, interior small grey,
    PIV-touching edges with white halo.  Tile bbox dashed.

    Right panel: histogram of log10(G_e) with median + bottleneck
    threshold annotated.  Identifies edges that are 100× weaker than
    typical — those are usually responsible for cond(L) blowups.

    Parameters mirror the solver: same `mu`, `px_size_m`, same
    geometry helper.  Result is just a figure — nothing persisted.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import numpy as np

    print(f"  [plot_carve] Extracting carve for tile {tile_id}...")
    edges_in, all_nodes, boundary, interior = \
        extract_tile_subgraph_spatial(graph, int(tile_id))
    n_b = len(boundary)
    n_i = len(interior)
    print(f"  [plot_carve] {len(edges_in)} edges, {n_b} bnd, {n_i} int")

    # Bbox of PIV-touching nodes (same as the carve uses)
    piv_nodes = set()
    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv', []) or []
        if any(m.get('tile_id') == int(tile_id) for m in piv):
            piv_nodes.add(u); piv_nodes.add(v)
    xs = [float(graph.nodes[n].get('x', 0.0)) for n in piv_nodes]
    ys = [float(graph.nodes[n].get('y', 0.0)) for n in piv_nodes]
    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))

    # Per-edge conductance G = πR⁴/(8μL).  Also collect path geometry.
    G = np.zeros(len(edges_in))
    R_px = np.zeros(len(edges_in))
    has_piv = np.zeros(len(edges_in), dtype=bool)
    paths = []  # list of (xs, ys) arrays in pixel coords
    for i, (u, v) in enumerate(edges_in):
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0:
            G[i] = np.nan
        else:
            G[i] = np.pi * (R_m ** 4) / (8.0 * mu * L_m)
        # Use the same radius source as the inference (see `_edge_geometry`).
        R_raw = d.get('radius_px_true')
        if R_raw is None:
            R_raw = d.get('radius')
        if R_raw is None:
            R_raw = d.get('radius_adapted_m', 1.0)
        if hasattr(R_raw, 'item'):
            R_raw = R_raw.item()
        R_px[i] = float(R_raw) if float(R_raw) > 1e-3 else \
            float(R_raw) / px_size_m
        piv = d.get('measurements_piv', []) or []
        has_piv[i] = any(m.get('tile_id') == int(tile_id) for m in piv)
        # Path geometry
        path = d.get('path', None)
        if path is not None and len(path) >= 2:
            arr = np.asarray(path, dtype=float)
            if arr.shape[1] >= 2:
                paths.append((arr[:, 0], arr[:, 1]))
            else:
                paths.append(None)
        else:
            ux = float(graph.nodes[u].get('x', 0.0))
            uy = float(graph.nodes[u].get('y', 0.0))
            vx = float(graph.nodes[v].get('x', 0.0))
            vy = float(graph.nodes[v].get('y', 0.0))
            paths.append((np.array([ux, vx]), np.array([uy, vy])))

    Gv = G[np.isfinite(G) & (G > 0)]
    if len(Gv) == 0:
        raise RuntimeError("No finite-conductance edges in carve.")
    G_med = float(np.median(Gv))
    G_min = float(np.min(Gv))
    G_max = float(np.max(Gv))
    log_ratio = np.where(np.isfinite(G) & (G > 0),
                         np.log10(G / G_med), np.nan)

    bottleneck_thr = G_med / float(bottleneck_factor)
    is_bottleneck = (G > 0) & (G < bottleneck_thr)
    n_bottleneck = int(np.sum(is_bottleneck))
    if n_bottleneck > 0:
        print(f"  [plot_carve] bottleneck edges (G < G_med/"
              f"{bottleneck_factor:.0f}):")
        for i in np.where(is_bottleneck)[0]:
            u, v = edges_in[i]
            d = graph.edges[u, v]
            R_m, L_m = _edge_geometry(d, px_size_m)
            R_p = R_m / px_size_m
            L_p = L_m / px_size_m
            ux = float(graph.nodes[u].get('x', 0.0))
            uy = float(graph.nodes[u].get('y', 0.0))
            vx = float(graph.nodes[v].get('x', 0.0))
            vy = float(graph.nodes[v].get('y', 0.0))
            print(f"    edge ({u},{v}): G={G[i]:.2e}  "
                  f"R={R_p:.2f}px  L={L_p:.1f}px  "
                  f"midpoint=({(ux+vx)/2:.0f},{(uy+vy)/2:.0f})  "
                  f"PIV={'yes' if has_piv[i] else 'no'}")

    # Compute cond(L_DC) on the interior block alone (boundary as
    # Dirichlet) — same conditioning the solver sees.
    try:
        node_to_idx = {n: k for k, n in enumerate(interior)}
        L_int = np.zeros((n_i, n_i))
        for (u, v), Ge in zip(edges_in, G):
            if not (np.isfinite(Ge) and Ge > 0):
                continue
            iu = node_to_idx.get(u)
            iv = node_to_idx.get(v)
            if iu is not None and iv is not None:
                L_int[iu, iu] += Ge
                L_int[iv, iv] += Ge
                L_int[iu, iv] -= Ge
                L_int[iv, iu] -= Ge
            elif iu is not None:
                L_int[iu, iu] += Ge
            elif iv is not None:
                L_int[iv, iv] += Ge
        if n_i > 0:
            sv = np.linalg.svd(L_int, compute_uv=False)
            cond_L = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
        else:
            cond_L = np.nan
    except Exception:
        cond_L = np.nan

    # ── Build figure ──
    fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [2, 1]})

    # ── Left: spatial graph ──
    cmap = cm.get_cmap('viridis')
    abs_max = float(np.nanmax(np.abs(log_ratio))) if np.any(
        np.isfinite(log_ratio)) else 1.0
    norm = Normalize(vmin=-abs_max, vmax=+abs_max)

    for i, ((u, v), pth) in enumerate(zip(edges_in, paths)):
        if pth is None:
            continue
        xs_p, ys_p = pth
        if is_bottleneck[i]:
            ax_g.plot(xs_p, ys_p, color='red', lw=2.0, alpha=0.95,
                      ls='--', zorder=4)
        else:
            lr = log_ratio[i]
            color = cmap(norm(lr)) if np.isfinite(lr) else (0.5, 0.5, 0.5, 1)
            lw = 0.8 + 1.2 * (R_px[i] / max(R_px.max(), 1.0))
            if has_piv[i]:
                ax_g.plot(xs_p, ys_p, color='white', lw=lw + 1.4,
                          alpha=0.7, zorder=2)
            ax_g.plot(xs_p, ys_p, color=color, lw=lw, alpha=0.95,
                      zorder=3)

    # Nodes
    if interior:
        ix = [graph.nodes[n].get('x', 0.0) for n in interior]
        iy = [graph.nodes[n].get('y', 0.0) for n in interior]
        ax_g.scatter(ix, iy, s=5, c='lightgray', alpha=0.6, zorder=5,
                     edgecolors='none')
    if boundary:
        bx = [graph.nodes[n].get('x', 0.0) for n in boundary]
        by = [graph.nodes[n].get('y', 0.0) for n in boundary]
        ax_g.scatter(bx, by, s=42, c='red', alpha=0.95, zorder=6,
                     edgecolors='black', linewidths=0.6,
                     label=f'boundary ({n_b})')

    # Bbox
    ax_g.plot([x_min, x_max, x_max, x_min, x_min],
              [y_min, y_min, y_max, y_max, y_min],
              ls='--', color='black', lw=1.0, alpha=0.6, zorder=1)

    ax_g.set_aspect('equal')
    ax_g.invert_yaxis()  # image coordinates: y down
    ax_g.set_title(
        f"Tile {tile_id} carve  |  "
        f"edges={len(edges_in)}  bnd={n_b}  int={n_i}  "
        f"PIV-edges={int(has_piv.sum())}\n"
        f"cond(L_DC)={cond_L:.2e}  "
        f"G_max/G_min={G_max/max(G_min, 1e-30):.2e}  "
        f"bottleneck<G_med/{bottleneck_factor:.0f}: {n_bottleneck}")
    ax_g.set_xlabel('x [px]')
    ax_g.set_ylabel('y [px]')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_g, fraction=0.04, pad=0.02)
    cb.set_label('log10(G / G_median)')
    ax_g.legend(loc='upper right', fontsize=8)

    # ── Right: histogram ──
    log_G = np.log10(Gv)
    ax_h.hist(log_G, bins=40, color='#5A4FCF', alpha=0.8,
              edgecolor='black', linewidth=0.4)
    ax_h.axvline(np.log10(G_med), color='black', ls='-', lw=1.4,
                 label=f'median = {G_med:.2e}')
    ax_h.axvline(np.log10(bottleneck_thr), color='red', ls='--', lw=1.4,
                 label=f'G_med/{bottleneck_factor:.0f}')
    ax_h.set_xlabel('log10(G)  [m³/(Pa·s)]')
    ax_h.set_ylabel('# edges')
    ax_h.set_title(
        f"Edge conductance distribution\n"
        f"G_min={G_min:.2e}  G_max={G_max:.2e}  "
        f"R_px range=[{R_px[R_px>0].min():.2f}, {R_px.max():.2f}]")
    ax_h.legend(fontsize=8)
    ax_h.grid(alpha=0.3)

    plt.tight_layout()
    print(f"  [plot_carve] cond(L_DC)={cond_L:.2e}  "
          f"G_max/G_min={G_max/max(G_min, 1e-30):.2e}  "
          f"bottleneck={n_bottleneck}")
    return fig


def plot_local_inference_result(
    result: LocalInferenceResult,
    *,
    title: Optional[str] = None,
    figsize: tuple = (13, 9),
):
    """Diagnostic 2×2 figure for a LocalInferenceResult.

    (0,0) DC scatter: Q^pred vs Q^meas, identity line, R²
    (0,1) H1 scatter: Re(Q^pred) vs Re(Q^meas) and Im vs Im on same axes,
                       diagonal y=x, both colored by edge index
    (1,0) summary text: D̂ ± σ_D, χ²/dof, σ per harmonic, conditioning,
                        boundary count, gauge pin, convergence
    (1,1) bar chart of boundary pressures: |P_DC| as bars (signed),
          and |P_H1| as a separate set of bars (magnitudes), per node
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    color_dc = '#5A4FCF'
    color_h1 = '#FF7F0E'

    # ── (0,0) DC scatter ──
    ax = axes[0, 0]
    if result.valid_dc.any():
        x = result.Q_pred_DC[result.valid_dc]
        y = result.Q_meas_DC[result.valid_dc]
        # Drop any NaN/Inf entries (can appear if edge geometry is zero
        # or the linear solve produced singular columns)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        ax.scatter(x, y, color=color_dc, s=36,
                    edgecolor='black', lw=0.5, alpha=0.75,
                    label=f'DC (N={int(ok.sum())})')
        if x.size and y.size:
            lim = float(np.max(np.abs(np.concatenate([x, y])))) * 1.1
            if not np.isfinite(lim) or lim == 0:
                lim = 1.0
            lim = max(lim, 1e-6)
            ax.plot([-lim, lim], [-lim, lim], color='gray',
                     ls='--', lw=1.0, label='y = x')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        # R²
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        ax.text(0.03, 0.97, f'R² = {r2:.3f}',
                 transform=ax.transAxes, va='top', ha='left',
                 fontsize=11, family='monospace',
                 bbox=dict(boxstyle='round', fc='white',
                            ec='gray', alpha=0.85))
    else:
        ax.text(0.5, 0.5, 'No DC data', transform=ax.transAxes,
                 ha='center', va='center', color='gray')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$Q^{\mathrm{pred}}$  [nL/s]', fontsize=12)
    ax.set_ylabel(r'$Q^{\mathrm{meas}}$  [nL/s]', fontsize=12)
    ax.set_title('DC: predicted vs measured', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    ax.tick_params(labelsize=11)

    # ── (0,1) H1 scatter (Re + Im) ──
    ax = axes[0, 1]
    if result.valid_h1.any():
        x_re = result.Q_pred_H1[result.valid_h1].real
        y_re = result.Q_meas_H1[result.valid_h1].real
        x_im = result.Q_pred_H1[result.valid_h1].imag
        y_im = result.Q_meas_H1[result.valid_h1].imag
        ok = (np.isfinite(x_re) & np.isfinite(y_re)
              & np.isfinite(x_im) & np.isfinite(y_im))
        x_re, y_re, x_im, y_im = (
            x_re[ok], y_re[ok], x_im[ok], y_im[ok])
        ax.scatter(x_re, y_re, color=color_h1, s=36,
                    edgecolor='black', lw=0.5, alpha=0.7,
                    marker='o', label=f'Re (N={int(ok.sum())})')
        ax.scatter(x_im, y_im, color=color_h1, s=36,
                    edgecolor='black', lw=0.5, alpha=0.4,
                    marker='^', label=f'Im (N={int(ok.sum())})')
        if ok.any():
            all_xy = np.concatenate([x_re, y_re, x_im, y_im])
            lim = float(np.max(np.abs(all_xy))) * 1.1
            if not np.isfinite(lim) or lim == 0:
                lim = 1.0
            lim = max(lim, 1e-6)
            ax.plot([-lim, lim], [-lim, lim], color='gray',
                     ls='--', lw=1.0, label='y = x')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ss_res = (float(np.sum((y_re - x_re) ** 2))
                   + float(np.sum((y_im - x_im) ** 2)))
        ss_tot = (float(np.sum((y_re - np.mean(y_re)) ** 2))
                   + float(np.sum((y_im - np.mean(y_im)) ** 2)))
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        ax.text(0.03, 0.97, f'R² (Re+Im) = {r2:.3f}',
                 transform=ax.transAxes, va='top', ha='left',
                 fontsize=11, family='monospace',
                 bbox=dict(boxstyle='round', fc='white',
                            ec='gray', alpha=0.85))
    else:
        ax.text(0.5, 0.5, 'No H1 data', transform=ax.transAxes,
                 ha='center', va='center', color='gray')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\hat Q^{\mathrm{pred}}$  [nL/s]', fontsize=12)
    ax.set_ylabel(r'$\hat Q^{\mathrm{meas}}$  [nL/s]', fontsize=12)
    ax.set_title('H1: predicted vs measured (Re + Im)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    ax.tick_params(labelsize=11)

    # ── (1,0) summary text ──
    ax = axes[1, 0]
    ax.axis('off')
    lines = []
    lines.append(f'Tile {result.tile_id} — local inference')
    lines.append('')
    lines.append(f'  D̂  = {result.D_hat:.3e} 1/Pa')
    lines.append(f'         ± {result.sigma_D:.3e}')
    lines.append(f'  f₀  = {result.f0_hz:.3f} Hz')
    lines.append('')
    lines.append(f'  χ²/dof    = {result.chi2_red:.3f}')
    lines.append(f'  N_obs_real= {result.n_obs_real}')
    lines.append(f'  N_params  = {result.n_params}')
    lines.append(f'  dof       = {result.dof}')
    lines.append('')
    lines.append(f'  K_boundary = {len(result.boundary_nodes)}')
    lines.append(f'  N_edges    = {len(result.interior_edges)}')
    lines.append(f'  pin node   = {result.pin_node}')
    lines.append('')
    lines.append(f'  cond(M_DC) = {result.cond_DC:.2g}')
    lines.append(f'  cond(M_H1) = {result.cond_H1:.2g}')
    lines.append('')
    lines.append(f'  iterations = {result.iterations}  '
                  f'({"converged" if result.converged else "MAX_ITER"})')
    if result.convergence_history:
        last = result.convergence_history[-1]
        lines.append(f'  last |ΔD|/D = {last.get("rel_dD", 0):.3g}')
    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes,
             ha='left', va='top', fontsize=11, family='monospace')

    # ── (1,1) boundary-pressure bar chart ──
    ax = axes[1, 1]
    nodes = result.boundary_nodes
    labels = [str(n) for n in nodes]
    P_DC_vals = np.array([result.P_DC.get(n, 0.0) for n in nodes])
    P_H1_mag = np.array([abs(result.P_H1.get(n, 0.0)) for n in nodes])
    P_H1_phase = np.array([np.degrees(np.angle(
        result.P_H1.get(n, 0.0 + 0j))) for n in nodes])
    sig_DC = np.array([result.sigma_P_DC.get(n, 0.0) for n in nodes])
    sig_H1 = np.array([result.sigma_P_H1.get(n, 0.0) for n in nodes])
    x = np.arange(len(nodes))
    width = 0.4
    ax.bar(x - width / 2, P_DC_vals, width, color=color_dc,
            yerr=sig_DC, capsize=2, alpha=0.85,
            edgecolor='black', lw=0.4, label='P_DC')
    ax.bar(x + width / 2, P_H1_mag, width, color=color_h1,
            yerr=sig_H1, capsize=2, alpha=0.85,
            edgecolor='black', lw=0.4, label='|P_H1|')
    # Mark the gauge pin
    if result.pin_node in nodes:
        pi = nodes.index(result.pin_node)
        ax.axvspan(pi - 0.5, pi + 0.5, color='gray',
                    alpha=0.15, zorder=0)
        ax.text(pi, ax.get_ylim()[1] * 0.95, 'pin',
                 ha='center', va='top', fontsize=9, color='gray')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Boundary node id', fontsize=12)
    ax.set_ylabel('Pressure  [Pa]', fontsize=12)
    ax.set_title('Inferred boundary pressures '
                  '(DC + |H1|, ±σ; pin node shaded)',
                  fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(labelsize=10)

    suptitle = title or f'Per-tile local inference — tile {result.tile_id}'
    fig.suptitle(suptitle, fontweight='bold', fontsize=13)
    return fig


def plot_local_flow_diagnostics(
    result: LocalInferenceResult,
    *,
    title: Optional[str] = None,
    figsize: tuple = (15, 12),
    sort_by: str = 'edge_index',
):
    """Detailed measured-vs-simulated edge flow diagnostics.

    This complements `plot_local_inference_result` by showing edge-level
    flow, direction, amplitude attenuation, and phase diagnostics.

    Parameters
    ----------
    result : LocalInferenceResult
        Output from `infer_local`.
    title : str, optional
        Figure title.
    figsize : tuple
        Matplotlib figure size.
    sort_by : {'edge_index', 'dc_measured_abs', 'h1_measured_abs'}
        Ordering used along the edge axis.
    """
    import matplotlib.pyplot as plt

    def _wrap_phase_deg(phi_deg):
        return (phi_deg + 180.0) % 360.0 - 180.0

    n_edges = len(result.interior_edges)
    edge_idx = np.arange(n_edges)

    valid_dc = np.asarray(result.valid_dc, dtype=bool)
    valid_h1 = np.asarray(result.valid_h1, dtype=bool)

    qdc_meas = np.asarray(result.Q_meas_DC, dtype=float)
    qdc_sim = np.asarray(result.Q_pred_DC, dtype=float)
    qh1_meas = np.asarray(result.Q_meas_H1, dtype=complex)
    qh1_sim = np.asarray(result.Q_pred_H1, dtype=complex)

    # Ignore essentially-zero H1 amplitudes when evaluating phase.
    # Phase is not meaningful where the harmonic amplitude is tiny.
    h1_amp_meas = np.abs(qh1_meas)
    phase_mask = valid_h1 & np.isfinite(h1_amp_meas)
    if np.any(phase_mask):
        phase_thresh = 0.05 * np.nanmax(h1_amp_meas[phase_mask])
        phase_mask &= (h1_amp_meas >= phase_thresh)
    else:
        phase_thresh = np.nan

    if sort_by == 'dc_measured_abs':
        order_key = np.where(valid_dc, np.abs(qdc_meas), -np.inf)
        order = np.argsort(order_key)[::-1]
    elif sort_by == 'h1_measured_abs':
        order_key = np.where(valid_h1, np.abs(qh1_meas), -np.inf)
        order = np.argsort(order_key)[::-1]
    elif sort_by == 'edge_index':
        order = edge_idx
    else:
        raise ValueError(
            "sort_by must be one of {'edge_index', "
            "'dc_measured_abs', 'h1_measured_abs'}")

    x = np.arange(n_edges)
    labels = [f'{i}: {u}-{v}' for i, (u, v) in enumerate(result.interior_edges)]
    labels = [labels[i] for i in order]

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    width = 0.42

    # ── (0,0) Signed DC flow per edge ──
    ax = axes[0, 0]
    if valid_dc.any():
        y_meas = qdc_meas[order]
        y_sim = qdc_sim[order]
        valid = valid_dc[order]
        ax.bar(x[valid] - width / 2, y_meas[valid], width,
               alpha=0.8, edgecolor='black', lw=0.3,
               label='measured')
        ax.bar(x[valid] + width / 2, y_sim[valid], width,
               alpha=0.8, edgecolor='black', lw=0.3,
               label='simulated')
        ax.axhline(0, color='black', lw=0.7)
        ax.set_ylabel('Signed DC flow [nL/s]')
        ax.set_title('DC flow by edge')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No valid DC data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3, axis='y')

    # ── (0,1) DC direction agreement ──
    ax = axes[0, 1]
    if valid_dc.any():
        meas_sign = np.sign(qdc_meas[order])
        sim_sign = np.sign(qdc_sim[order])
        valid = valid_dc[order] & np.isfinite(meas_sign) & np.isfinite(sim_sign)
        agree = valid & (meas_sign == sim_sign) & (meas_sign != 0)
        disagree = valid & (meas_sign != sim_sign) & (meas_sign != 0) & (sim_sign != 0)
        sim_zero = valid & (sim_sign == 0)
        ax.scatter(x[agree], np.zeros(int(agree.sum())), s=55,
                   marker='o', label='same direction')
        ax.scatter(x[disagree], np.zeros(int(disagree.sum())), s=70,
                   marker='x', label='opposite direction')
        ax.scatter(x[sim_zero], np.zeros(int(sim_zero.sum())), s=45,
                   marker='|', label='sim ≈ 0')
        n_valid = int(valid.sum())
        n_agree = int(agree.sum())
        frac = n_agree / n_valid if n_valid else np.nan
        ax.text(0.03, 0.95,
                f'agreement = {n_agree}/{n_valid} ({frac:.1%})',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_title('DC flow-direction agreement')
        ax.legend(fontsize=9, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No valid DC data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3, axis='x')

    # ── (1,0) H1 amplitude per edge ──
    ax = axes[1, 0]
    if valid_h1.any():
        amp_meas = np.abs(qh1_meas[order])
        amp_sim = np.abs(qh1_sim[order])
        valid = valid_h1[order]
        ax.bar(x[valid] - width / 2, amp_meas[valid], width,
               alpha=0.8, edgecolor='black', lw=0.3,
               label='measured')
        ax.bar(x[valid] + width / 2, amp_sim[valid], width,
               alpha=0.8, edgecolor='black', lw=0.3,
               label='simulated')
        ax.set_ylabel(r'$|Q_{H1}|$ [nL/s]')
        ax.set_title('H1 amplitude by edge')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No valid H1 data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3, axis='y')

    # ── (1,1) H1 attenuation / gain ──
    ax = axes[1, 1]
    if valid_h1.any():
        amp_meas = np.abs(qh1_meas[order])
        amp_sim = np.abs(qh1_sim[order])
        valid = (valid_h1[order]
                 & np.isfinite(amp_meas)
                 & np.isfinite(amp_sim)
                 & (amp_meas > 0))
        ratio = np.full(n_edges, np.nan, dtype=float)
        ratio[valid] = amp_sim[valid] / amp_meas[valid]
        ax.plot(x[valid], ratio[valid], marker='o', lw=1.2,
                label=r'$|Q_{sim}|/|Q_{meas}|$')
        ax.axhline(1.0, color='black', ls='--', lw=0.9,
                   label='no attenuation/gain')
        ax.set_ylabel('Amplitude ratio')
        ax.set_title('H1 simulated/measured amplitude ratio')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No valid H1 data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3)

    # ── (2,0) H1 phase per edge ──
    ax = axes[2, 0]
    if valid_h1.any():
        phase_meas = np.degrees(np.angle(qh1_meas[order]))
        phase_sim = np.degrees(np.angle(qh1_sim[order]))
        valid = phase_mask[order] & np.isfinite(phase_meas) & np.isfinite(phase_sim)
        ax.plot(x[valid], phase_meas[valid], marker='o', lw=1.2,
                label='measured')
        ax.plot(x[valid], phase_sim[valid], marker='s', lw=1.2,
                label='simulated')
        ax.axhline(0.0, color='black', lw=0.6)
        ax.set_ylabel('H1 phase [deg]')
        ax.set_title('H1 phase by edge')
        ax.set_ylim(-190, 190)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No valid H1 data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3)

    # ── (2,1) H1 phase residual ──
    ax = axes[2, 1]
    if valid_h1.any():
        phase_meas = np.degrees(np.angle(qh1_meas[order]))
        phase_sim = np.degrees(np.angle(qh1_sim[order]))
        valid = phase_mask[order] & np.isfinite(phase_meas) & np.isfinite(phase_sim)
        dphi = _wrap_phase_deg(phase_sim - phase_meas)
        ax.bar(x[valid], dphi[valid], width=0.75,
               alpha=0.85, edgecolor='black', lw=0.3)
        ax.axhline(0.0, color='black', lw=0.7)
        ax.set_ylabel('Wrapped phase error [deg]')
        ax.set_title('H1 phase error: simulated − measured')
        ax.set_ylim(-190, 190)
    else:
        ax.text(0.5, 0.5, 'No valid H1 data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3, axis='y')

    for ax in axes.ravel():
        ax.set_xlabel('Edge')
        ax.set_xlim(-0.75, max(n_edges - 0.25, 0.75))
        if n_edges <= 35:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
        else:
            ax.tick_params(labelsize=9)

    suptitle = title or f'Per-tile local flow diagnostics — tile {result.tile_id}'
    fig.suptitle(suptitle, fontweight='bold', fontsize=14)

    print('\n' + '=' * 72)
    print('LOCAL FLOW DIAGNOSTICS')
    print('=' * 72)

    dc_valid = valid_dc & np.isfinite(qdc_meas) & np.isfinite(qdc_sim)
    dc_same = dc_valid & (np.sign(qdc_meas) == np.sign(qdc_sim))
    n_valid_dc = int(dc_valid.sum())
    n_same_dc = int(dc_same.sum())
    if n_valid_dc > 0:
        print(f'DC direction agreement : {n_same_dc}/{n_valid_dc} '
              f'({n_same_dc / n_valid_dc:.1%})')

    amp_meas = np.abs(qh1_meas)
    amp_sim = np.abs(qh1_sim)
    amp_valid = (
        valid_h1
        & np.isfinite(amp_meas)
        & np.isfinite(amp_sim)
        & (amp_meas > 0)
    )

    if np.any(amp_valid):
        amp_ratio = amp_sim[amp_valid] / amp_meas[amp_valid]
        print(f'Median H1 amplitude ratio : {np.median(amp_ratio):.3f}')
        print(f'Mean H1 amplitude ratio   : {np.mean(amp_ratio):.3f}')
        print(f'95th percentile ratio     : {np.percentile(amp_ratio, 95):.3f}')

    if np.any(phase_mask):
        phase_meas_raw = np.degrees(np.angle(qh1_meas[phase_mask]))
        phase_sim_raw = np.degrees(np.angle(qh1_sim[phase_mask]))
        dphi = _wrap_phase_deg(phase_sim_raw - phase_meas_raw)
        weights = np.abs(qh1_meas[phase_mask])
        weighted_rmse = np.sqrt(np.average(dphi ** 2, weights=weights))

        print(f'Phase threshold            : {phase_thresh:.3e} nL/s')
        print(f'Phase edges retained       : {int(phase_mask.sum())}/{int(valid_h1.sum())}')
        print(f'Weighted phase RMSE        : {weighted_rmse:.2f} deg')
        print(f'Median |phase error|       : {np.median(np.abs(dphi)):.2f} deg')

    if np.any(amp_valid):
        print('\nWorst amplitude mismatches:')
        full_ratio = np.full(len(amp_meas), np.nan)
        full_ratio[amp_valid] = amp_sim[amp_valid] / amp_meas[amp_valid]
        amp_edges = np.where(amp_valid)[0]
        mismatch = np.abs(np.log10(full_ratio[amp_valid]))
        worst_edges = amp_edges[np.argsort(mismatch)[::-1][:5]]
        for i in worst_edges:
            u, v = result.interior_edges[i]
            print(
                f'  edge {i:3d} ({u}-{v}) : '
                f'ratio={full_ratio[i]:.2f}, '
                f'|Qm|={amp_meas[i]:.3e}, '
                f'|Qs|={amp_sim[i]:.3e}'
            )

    print('=' * 72 + '\n')
    return fig


# ──────────────────────────────────────────────────────────────────
# Graph-level diagnostics and synthetic boundary-pressure helpers
# ──────────────────────────────────────────────────────────────────
def plot_local_flow_graphs(
    result: LocalInferenceResult,
    pos: dict,
    *,
    figsize=(14, 10),
):
    """Plot measured vs simulated graph-level flow diagnostics.

    Parameters
    ----------
    result : LocalInferenceResult
        Inference result.
    pos : dict
        Node position dictionary: {node: (x, y)}.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(result.interior_edges)

    qdc_meas = np.asarray(result.Q_meas_DC, dtype=float)
    qdc_sim = np.asarray(result.Q_pred_DC, dtype=float)
    qh1_meas = np.asarray(result.Q_meas_H1, dtype=complex)
    qh1_sim = np.asarray(result.Q_pred_H1, dtype=complex)

    edge_list = list(result.interior_edges)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    panels = [
        ('Measured DC flow', qdc_meas, True),
        ('Simulated DC flow', qdc_sim, True),
        ('Measured H1 amplitude', np.abs(qh1_meas), False),
        ('Simulated H1 amplitude', np.abs(qh1_sim), False),
    ]

    for ax, (panel_title, vals, signed) in zip(axes.ravel(), panels):
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals)
        if finite.any():
            vmax = float(np.nanmax(np.abs(vals[finite])))
        else:
            vmax = 1.0
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1.0

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=12,
            node_color='black',
            ax=ax,
        )

        if signed:
            edges = nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edge_list,
                edge_color=vals,
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=-vmax,
                edge_vmax=vmax,
                width=2.0,
                ax=ax,
            )
        else:
            edges = nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edge_list,
                edge_color=vals,
                edge_cmap=plt.cm.viridis,
                edge_vmin=0.0,
                edge_vmax=vmax,
                width=2.0,
                ax=ax,
            )

        plt.colorbar(edges, ax=ax, shrink=0.8)
        ax.set_title(panel_title)
        ax.set_aspect('equal')
        ax.axis('off')

    return fig


def make_physics_guided_boundary_pressures(
    boundary_nodes,
    pos: dict,
    *,
    flow_axis=(1.0, 0.0),
    D: float = 1.0e-3,
    f0_hz: float = 2.5,
    P_dc_drop: float = 20.0,
    P_dc_offset: float = 0.0,
    P_h1_upstream_amp: float = 1.0,
    phase_sign: float = -1.0,
    verbose: bool = True,
):
    """Create simple physics-guided boundary pressures for a tile.

    This is useful for synthetic tests where random or weak boundary
    conditions produce nearly stagnant interior flow.

    The coordinate along `flow_axis` is treated as the upstream-to-downstream
    direction. DC pressure decreases linearly along this axis. H1 pressure
    amplitude decays and phase shifts according to the diffusion-like
    harmonic pressure scale

        alpha = sqrt(omega / (2D)).

    Parameters
    ----------
    boundary_nodes : sequence
        Boundary node ids.
    pos : dict
        Node position dictionary `{node: (x, y)}`. Coordinates must use the
        same length units as D. If D is in pixel^2/s, pos should be pixels; if
        D is in mm^2/s, pos should be mm.
    flow_axis : tuple
        Direction of mean flow, e.g. `(1, 0)` for left-to-right.
    D : float
        Effective pressure diffusion coefficient in coordinate units^2 / s.
    f0_hz : float
        Fundamental frequency in Hz.
    P_dc_drop : float
        Upstream minus downstream DC pressure drop, in Pa.
    P_dc_offset : float
        Downstream DC pressure offset, in Pa.
    P_h1_upstream_amp : float
        Upstream H1 pressure amplitude, in Pa.
    phase_sign : float
        Use `-1` for downstream phase lag under exp(-i alpha x), `+1` for
        downstream phase lead.
    verbose : bool
        Print phase/attenuation diagnostics.

    Returns
    -------
    P_DC : dict
        `{node: pressure_dc}` in Pa.
    P_H1 : dict
        `{node: pressure_h1_complex}` in Pa.
    info : dict
        Summary values: alpha, L_tile, total_phase_rad, total_phase_deg,
        total_attenuation, s_norm.
    """
    nodes = list(boundary_nodes)
    if len(nodes) < 2:
        raise ValueError('Need at least two boundary nodes.')

    axis = np.asarray(flow_axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0 or not np.isfinite(axis_norm):
        raise ValueError('flow_axis must be a nonzero finite vector.')
    axis = axis / axis_norm

    missing = [n for n in nodes if n not in pos]
    if missing:
        raise KeyError(
            f'pos is missing {len(missing)} boundary nodes, e.g. {missing[:5]}')

    xy = np.array([pos[n] for n in nodes], dtype=float)
    s = xy @ axis
    L_tile = float(np.nanmax(s) - np.nanmin(s))
    if L_tile <= 0 or not np.isfinite(L_tile):
        raise ValueError('Boundary nodes have zero projected length along flow_axis.')

    s_norm = (s - np.nanmin(s)) / L_tile

    omega = 2.0 * np.pi * float(f0_hz)
    alpha = float(np.sqrt(omega / (2.0 * float(D))))
    total_phase_rad = alpha * L_tile
    total_phase_deg = float(np.degrees(total_phase_rad))
    total_attenuation = float(np.exp(-total_phase_rad))

    P_dc_vals = P_dc_offset + P_dc_drop * (1.0 - s_norm)

    # Upstream has amplitude P_h1_upstream_amp. Downstream is attenuated
    # and phase shifted by alpha * L_tile.
    attenuation = np.exp(-total_phase_rad * s_norm)
    phase = phase_sign * total_phase_rad * s_norm
    P_h1_vals = P_h1_upstream_amp * attenuation * np.exp(1j * phase)

    P_DC = {n: float(p) for n, p in zip(nodes, P_dc_vals)}
    P_H1 = {n: complex(p) for n, p in zip(nodes, P_h1_vals)}

    info = {
        'alpha': alpha,
        'L_tile': L_tile,
        'total_phase_rad': total_phase_rad,
        'total_phase_deg': total_phase_deg,
        'total_attenuation': total_attenuation,
        's': s,
        's_norm': s_norm,
        'flow_axis': tuple(axis),
        'D': float(D),
        'f0_hz': float(f0_hz),
    }

    if verbose:
        print('\n' + '=' * 72)
        print('PHYSICS-GUIDED BOUNDARY PRESSURES')
        print('=' * 72)
        print(f'D                 : {float(D):.3e} length^2/s')
        print(f'f0                : {float(f0_hz):.3f} Hz')
        print(f'omega             : {omega:.3e} rad/s')
        print(f'alpha             : {alpha:.3e} 1/length')
        print(f'projected L_tile  : {L_tile:.3e} length')
        print(f'total phase shift : {total_phase_deg:.2f} deg')
        print(f'total attenuation : {total_attenuation:.3e}')
        print(f'DC pressure drop  : {float(P_dc_drop):.3e} Pa')
        print(f'H1 upstream amp   : {float(P_h1_upstream_amp):.3e} Pa')
        print('=' * 72 + '\n')

    return P_DC, P_H1, info


def plot_physics_guided_boundary_pressures(
    boundary_nodes,
    pos: dict,
    P_DC: dict,
    P_H1: dict,
    *,
    figsize=(13, 4),
):
    """Visualize physics-guided boundary pressure initialization."""
    import matplotlib.pyplot as plt

    nodes = list(boundary_nodes)
    x = np.arange(len(nodes))
    labels = [str(n) for n in nodes]

    pdc = np.array([P_DC[n] for n in nodes], dtype=float)
    ph1 = np.array([P_H1[n] for n in nodes], dtype=complex)
    amp = np.abs(ph1)
    phase = np.degrees(np.angle(ph1))

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    axes[0].bar(x, pdc, edgecolor='black', lw=0.4)
    axes[0].set_title('Boundary DC pressure')
    axes[0].set_ylabel('P_DC [Pa]')

    axes[1].bar(x, amp, edgecolor='black', lw=0.4)
    axes[1].set_title('Boundary H1 amplitude')
    axes[1].set_ylabel('|P_H1| [Pa]')

    axes[2].plot(x, phase, marker='o', lw=1.2)
    axes[2].set_title('Boundary H1 phase')
    axes[2].set_ylabel('phase(P_H1) [deg]')
    axes[2].set_ylim(-190, 190)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
        ax.set_xlabel('Boundary node')
        ax.grid(alpha=0.3)

    return fig


# =====================================================================
# Global → Local recovery test
# =====================================================================
#
# End-to-end synthetic check that goes BEYOND the single-tile
# identifiability scan: it asks whether per-tile local inference can
# recover a *known global* D when given synthetic interior-edge Q's
# generated by the same global transmission-line solver that the
# real-data forward sim uses.
#
# Procedure:
#   1. Run `solve_transmission_line(graph, D=D_true, ...)` → produces
#      complex edge flows {(u,v): [Q_dc, Q_h1, Q_h2, ...]} at the
#      `D_true` of choice, with boundary conditions either pulled from
#      the graph's PIV measurements (default — gives realistic wave
#      structure) or supplied by the caller.
#   2. For each interior edge of every tile in `tile_ids`, build a
#      synthetic Q(t) time-series from the harmonic phasors and add
#      iid Gaussian noise.  Stuff into the matching PIV record
#      (`m_ref['Q_t']` and `m_ref['mean_Q']`) so that the existing
#      inference path reads identical data.
#   3. Run `infer_local(graph, tile, spec)` on each tile.  The carve
#      and boundary identification follow the EXACT same code path as
#      real-data inference (because the per-edge `tile_id` tags in
#      `measurements_piv` are unchanged) — only Q values are replaced.
#   4. Restore the original PIV records from a snapshot taken before
#      step 2 (try/finally guarantees this even on exception).
#   5. Plot D̂ vs D_true scatter with σ_D errorbars + a summary line
#      reporting median D̂/D_true and fraction of tiles within 1σ
#      and 2σ of truth.
# =====================================================================

def run_synthetic_simulation(
    graph,
    D: float = 1e-4,
    sigma_Q_nL_per_s: float = 0.05,
    sigma_Q_base_nL_per_s: Optional[float] = None,
    sigma_Q_rel: Optional[float] = None,
    n_harmonics: int = 2,
    bc_mode: str = 'all_q',                  # 'all_q' | 'sink_p0' | 'merged'
    bc_harmonics_override: Optional[Dict[int, np.ndarray]] = None,
    rng_seed: int = 0,
    verbose: bool = True,
):
    """Run the global TL solver at a known D + write noisy synthetic
    edge-flow phasors as new ``*_syn`` fields on the graph.

    This is a STANDALONE forward simulation — distinct from the
    existing real-data forward sim (``mean_Q_sim`` etc.) and from
    measured PIV (``mean_Q_piv``).  It produces:

      Per edge:
        - ``Q_DC_syn``, ``Q_H1_re_syn``, ``Q_H1_im_syn``,
          ``Q_H2_re_syn``, ``Q_H2_im_syn``  (raw phasors, nL/s)
        - ``mean_Q_syn`` (signed DC in edge's flow_from/flow_to convention)
        - ``amp_Q_syn`` = |Q_H1_syn|, ``phase_syn`` (deg)
        - ``amp_Q_h1_syn``, ``phase_h1_syn``
        - ``amp_Q_h2_syn``, ``phase_h2_syn``
        - ``PI_syn`` = 2·|Q_H1|/|Q_DC|

      Per graph:
        ``graph.graph['synthetic_sim_meta']`` = dict with D, sigma_Q,
        bc_mode, f0_hz, n_harmonics, n_edges_written, timestamp.

    Source dispatch: setting ``_active_field_source = 'synthetic'`` in
    the viewer pulls the ``_syn`` variants of source-aware base fields.

    Parameters
    ----------
    D : distensibility used in the forward solve, 1/Pa.
    sigma_Q_nL_per_s : per-phasor iid Gaussian noise std added directly
        in the complex phasor domain.  For DC: real noise only.  For
        each AC harmonic: independent real + imag noise components
        with std ``sigma_Q_nL_per_s / √2`` so the resulting |noise|
        has std = ``sigma_Q_nL_per_s`` nL/s.
    bc_mode : how to handle venous sinks:
        'all_q'   — Q BCs at all boundary nodes (DA + SV), as solved
                    by transmission_line's default mode.
        'sink_p0' — sink_pressure_bc=0 (Dirichlet P=0 at SV).
        'merged'  — merged_boundary=True (all boundaries lumped into
                    one node with shared pressure).
    bc_harmonics_override : optional explicit per-node Q phasors
        (bypasses extraction from measurements).

    Returns
    -------
    dict: {n_edges_written, f0_hz, edge_sample (a few example edges)}.
    """
    import copy
    from .transmission_line import solve_transmission_line
    import datetime as _dt

    if bc_mode not in ('all_q', 'sink_p0', 'merged'):
        raise ValueError(f"bc_mode must be 'all_q'|'sink_p0'|'merged', "
                          f"got {bc_mode!r}")

    rng = np.random.default_rng(rng_seed)
    sink_p_kw = 0.0 if bc_mode == 'sink_p0' else None
    merged_kw = (bc_mode == 'merged')

    if verbose:
        print(f"\n[run_synthetic_simulation] D={D:.2e} 1/Pa, "
              f"σ_Q={sigma_Q_nL_per_s} nL/s, n_harm={n_harmonics}, "
              f"bc={bc_mode}")

    tl = solve_transmission_line(
        graph,
        D=float(D),
        n_harmonics=int(n_harmonics),
        bc_harmonics_override=bc_harmonics_override,
        sink_pressure_bc=sink_p_kw,
        merged_boundary=merged_kw,
        verbose=verbose,
    )
    edge_Q = tl.edge_flows
    f0_hz = float(tl.f0_hz)

    # ── Write per-edge synthetic fields ──
    n_written = 0
    sample_edges = []
    nh = int(n_harmonics)
    for u, v, d in graph.edges(data=True):
        Qharm = edge_Q.get((u, v))
        if Qharm is None:
            Qharm = edge_Q.get((v, u))
            sgn_uv = -1.0
        else:
            sgn_uv = +1.0
        if Qharm is None or len(Qharm) == 0:
            continue
        # Apply edge-direction sign so the stored phasors are in
        # the edge's "flow_from → flow_to" convention (matches
        # mean_Q_piv / mean_Q_sim conventions on this graph).
        ff = d.get('flow_from'); ft = d.get('flow_to')
        if ff == v and ft == u:
            sign_edge = -1.0 * sgn_uv
        else:
            sign_edge = +1.0 * sgn_uv
        # Per-edge noise σ.  Heteroscedastic if base+rel provided:
        # σ_e = sqrt(base² + (rel · |Q_clean|)²)  per harmonic.
        # Otherwise constant σ_Q across all edges (legacy).
        # DC
        q_dc_clean = float(np.real(Qharm[0])) * sign_edge
        if (sigma_Q_base_nL_per_s is not None
                or sigma_Q_rel is not None):
            sb = float(sigma_Q_base_nL_per_s or 0.0)
            sr = float(sigma_Q_rel or 0.0)
            sigma_dc_e = float(np.sqrt(
                sb ** 2 + (sr * abs(q_dc_clean)) ** 2))
        else:
            sigma_dc_e = float(sigma_Q_nL_per_s)
        q_dc = q_dc_clean + float(rng.normal(0.0, sigma_dc_e))
        d['Q_DC_syn'] = q_dc
        d['mean_Q_syn'] = q_dc
        # AC harmonics — each with its own per-edge σ based on
        # clean phasor magnitude
        h1_complex = None
        h2_complex = None
        for n_h in range(1, min(nh, len(Qharm) - 1) + 1):
            phasor = complex(Qharm[n_h]) * sign_edge
            if (sigma_Q_base_nL_per_s is not None
                    or sigma_Q_rel is not None):
                sigma_ac_e = float(np.sqrt(
                    sb ** 2 + (sr * abs(phasor)) ** 2))
            else:
                sigma_ac_e = float(sigma_Q_nL_per_s)
            sigma_each_n = sigma_ac_e / np.sqrt(2.0)
            phasor_n = (phasor
                         + complex(rng.normal(0, sigma_each_n),
                                    rng.normal(0, sigma_each_n)))
            d[f'Q_H{n_h}_re_syn'] = float(phasor_n.real)
            d[f'Q_H{n_h}_im_syn'] = float(phasor_n.imag)
            d[f'amp_Q_h{n_h}_syn'] = float(abs(phasor_n))
            d[f'phase_h{n_h}_syn'] = float(
                np.degrees(np.angle(phasor_n)))
            if n_h == 1:
                h1_complex = phasor_n
            if n_h == 2:
                h2_complex = phasor_n
        # Legacy aliases for H1 (so generic 'amp_Q' / 'phase' under
        # source='synthetic' resolves to H1 amplitude / phase).
        if h1_complex is not None:
            d['amp_Q_syn'] = float(abs(h1_complex))
            d['phase_syn'] = float(
                np.degrees(np.angle(h1_complex)))
            # Pulsatility index
            if abs(q_dc) > 1e-30:
                d['PI_syn'] = float(
                    2.0 * abs(h1_complex) / abs(q_dc))
            else:
                d['PI_syn'] = float('nan')
        n_written += 1
        if len(sample_edges) < 5:
            sample_edges.append({
                'edge': (u, v),
                'Q_DC': q_dc, 'Q_DC_clean': q_dc_clean,
                '|Q_H1|': float(abs(h1_complex))
                            if h1_complex is not None else None,
                '|Q_H2|': float(abs(h2_complex))
                            if h2_complex is not None else None,
            })

    # ── Persist metadata ──
    is_hetero = (sigma_Q_base_nL_per_s is not None
                 or sigma_Q_rel is not None)
    graph.graph['synthetic_sim_meta'] = {
        'D': float(D),
        'sigma_Q_nL_per_s': float(sigma_Q_nL_per_s),
        'sigma_Q_base_nL_per_s': (float(sigma_Q_base_nL_per_s)
                                   if sigma_Q_base_nL_per_s is not None
                                   else None),
        'sigma_Q_rel': (float(sigma_Q_rel)
                         if sigma_Q_rel is not None else None),
        'noise_mode': ('heteroscedastic' if is_hetero
                        else 'constant'),
        'bc_mode': bc_mode,
        'f0_hz': f0_hz,
        'n_harmonics': int(n_harmonics),
        'n_edges_written': int(n_written),
        'timestamp': _dt.datetime.now().isoformat(),
    }

    if verbose:
        print(f"  wrote *_syn fields on {n_written} edges  "
              f"(f0 = {f0_hz:.3f} Hz)")
        for s in sample_edges:
            print(f"   sample {s['edge']}: Q_DC = "
                  f"{s['Q_DC']:.4f} nL/s  "
                  f"|Q_H1| = {s['|Q_H1|']!s}  "
                  f"|Q_H2| = {s['|Q_H2|']!s}")
        print("  View by setting Source = Synthetic in Tile View "
              "(or pick any *_syn field).")

    return dict(n_edges_written=n_written, f0_hz=f0_hz,
                bc_mode=bc_mode, sample_edges=sample_edges)


def run_local_inference_on_synthetic(
    graph,
    tile_id: int,
    *,
    base_spec: Optional['LocalInferenceSpec'] = None,
    sigma_Q_nL_per_s: Optional[float] = None,
    save_to_graph: bool = False,
    rng_seed: int = 1,
    verbose: bool = True,
):
    """Run `infer_local` on one tile using the previously-stored
    synthetic forward (from ``run_synthetic_simulation``).

    Temporarily overwrites the tile's PIV records with synthetic Q(t)
    reconstructed from the stored ``*_syn`` phasors, runs
    `infer_local`, restores originals via try/finally.

    Returns the `LocalInferenceResult` dataclass.  If the graph has
    no ``synthetic_sim_meta`` (forward never run), raises.
    """
    import copy
    meta = graph.graph.get('synthetic_sim_meta')
    if meta is None:
        raise RuntimeError(
            "No synthetic forward on the graph.  Click "
            "'Run synthetic forward' first.")
    if base_spec is None:
        base_spec = LocalInferenceSpec(D_init=float(meta['D']))
    f0_hz = float(meta['f0_hz'])
    omega0 = 2.0 * np.pi * f0_hz
    if sigma_Q_nL_per_s is None:
        sigma_Q_nL_per_s = float(meta['sigma_Q_nL_per_s'])
    rng = np.random.default_rng(rng_seed)

    # Snapshot PIV records touching this tile
    snapshot: Dict[Tuple[int, int], List[Tuple[int, dict]]] = {}
    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv') or []
        recs = []
        for i, m in enumerate(piv):
            if m.get('tile_id') == int(tile_id):
                recs.append((i, copy.deepcopy(m)))
        if recs:
            snapshot[(u, v)] = recs

    n_t_default = None
    dt_default = None
    for (u, v), recs in snapshot.items():
        for _i, m in recs:
            Qt = m.get('Q_t')
            if Qt is not None and len(Qt) > 4:
                n_t_default = len(Qt)
                dt_default = float(m.get('frame_dt_s', 1.0 / 250.0))
                break
        if n_t_default:
            break
    if n_t_default is None:
        n_t_default = 250; dt_default = 1.0 / 250.0
    t_arr = np.arange(n_t_default) * dt_default

    try:
        for (u, v), recs in snapshot.items():
            d = graph.edges[u, v]
            # Pull stored synthetic phasors written by
            # run_synthetic_simulation (already in edge convention).
            q_dc = d.get('Q_DC_syn')
            if q_dc is None or not np.isfinite(q_dc):
                continue
            q_dc = float(q_dc)
            Q_t_syn = np.full_like(t_arr, q_dc, dtype=float)
            for n_h in (1, 2, 3):
                qre = d.get(f'Q_H{n_h}_re_syn')
                qim = d.get(f'Q_H{n_h}_im_syn')
                if (qre is None or qim is None
                        or not np.isfinite(qre)
                        or not np.isfinite(qim)):
                    continue
                Q_t_syn = Q_t_syn + (
                    float(qre) * np.cos(n_h * omega0 * t_arr)
                    - float(qim) * np.sin(n_h * omega0 * t_arr))
            # Add fresh time-domain noise.  SKIP in heteroscedastic
            # mode — the phasor noise is already baked into *_syn,
            # adding time-domain noise here would double-count and
            # also flatten the per-edge heteroscedastic structure.
            if meta.get('noise_mode') != 'heteroscedastic':
                Q_t_syn = Q_t_syn + rng.normal(
                    0.0, sigma_Q_nL_per_s, size=n_t_default)
            # Write into every PIV record covering this tile, with
            # per-record sign correction.
            for idx, _orig in recs:
                m_live = graph.edges[u, v]['measurements_piv'][idx]
                ff = m_live.get('flow_from')
                ft = m_live.get('flow_to')
                edge_ff = d.get('flow_from'); edge_ft = d.get('flow_to')
                # mean_Q_syn was stored in edge's flow_from→flow_to
                # convention.  If the record uses the opposite
                # orientation, flip on write.
                if (ff is not None and ft is not None
                        and edge_ff is not None and edge_ft is not None
                        and ff == edge_ft and ft == edge_ff):
                    sign_rec = -1.0
                else:
                    sign_rec = +1.0
                m_live['Q_t'] = (sign_rec * Q_t_syn).tolist()
                m_live['mean_Q'] = float(sign_rec * q_dc)
                m_live.pop('harmonics', None)
                m_live['f0_hz'] = f0_hz
                m_live['frame_dt_s'] = dt_default

        # Run inference
        spec = LocalInferenceSpec(
            **{k: v for k, v in base_spec.__dict__.items()})
        spec.save_to_graph = bool(save_to_graph)
        spec.verbose = verbose
        result = infer_local(graph, int(tile_id), spec)

    finally:
        for (u, v), recs in snapshot.items():
            for idx, orig in recs:
                graph.edges[u, v]['measurements_piv'][idx] = orig

    if verbose:
        print(f"\n[infer-on-synthetic] tile {tile_id}: "
              f"D_true = {meta['D']:.3e},  "
              f"D̂ = {result.D_hat:.3e} ± "
              f"{result.sigma_D:.2e}  "
              f"(χ²/dof = {result.chi2_red:.2f}, "
              f"iter = {result.iterations})")
    return result


def _landscape_chi2_over_D(graph, tile_id, D_grid, spec):
    """Profile likelihood: χ²(D) after eliminating P_b in closed form.

    For each D in `D_grid`:
      1. Build T(0, D), T(ω₀, D) on the tile's carve.
      2. Solve P_b in unweighted complex LS (no prior).
      3. Return residual χ² = Σ |Q^obs − T·P_b|² (DC + H1 stacked).

    Returns dict with keys: D_grid, chi2, chi2_dc, chi2_h1, dof.
    No prior, no FGLS, no LM — just the raw cost surface so the user
    can see how identifiable D is given the data + carve.
    """
    px_size_m = (float(spec.px_size_m) if spec.px_size_m is not None
                 else 1.7e-6)
    # Same carve as infer_local
    if spec.include_unmeasured_anatomy:
        edges_in, _all, boundary_nodes, interior_nodes = \
            extract_tile_subgraph_spatial(graph, int(tile_id),
                                            padding_frac=0.0)
    else:
        edges_in, _all, boundary_nodes, interior_nodes = \
            extract_tile_subgraph(graph, int(tile_id))
    n_edges = len(edges_in)
    n_bnd = len(boundary_nodes)
    if n_edges < 3 or n_bnd < 2:
        raise ValueError(f"Carve too small for landscape: "
                          f"{n_edges} edges, {n_bnd} boundary")

    # f0 — median of tile PIV records
    f0_list = []
    for u, v in edges_in:
        for m in (graph.edges[u, v].get('measurements_piv') or []):
            if m.get('tile_id') == int(tile_id):
                f0 = m.get('f0_hz')
                if f0 and np.isfinite(f0) and f0 > 0:
                    f0_list.append(float(f0))
    f0_hz = (float(np.median(f0_list)) if f0_list
             else float(spec.f0_hz) if spec.f0_hz else 2.5)

    # Pin = highest G_attach (matches infer_local).
    interior_set = set(interior_nodes)
    g_attach = {n: 0.0 for n in boundary_nodes}
    for u, v in edges_in:
        d = graph.edges[u, v]
        R_m, L_m = _edge_geometry(d, px_size_m)
        if R_m <= 0 or L_m <= 0:
            continue
        Ge = float(np.pi * R_m ** 4 / (8.0 * spec.mu * L_m))
        if u in g_attach and v in interior_set:
            g_attach[u] += Ge
        if v in g_attach and u in interior_set:
            g_attach[v] += Ge
    pin_node = max(g_attach, key=g_attach.get)
    pin_idx = boundary_nodes.index(pin_node)
    keep_idx = [i for i in range(n_bnd) if i != pin_idx]

    # Measured Q's (SI units) — same path as infer_local.
    # `_extract_measured_flows` returns only (Q_dc, Q_hn); compute
    # valid masks here.
    Q_dc, Q_hn = _extract_measured_flows(
        graph, edges_in, int(tile_id), list(spec.harmonics))
    ac_harmonics = [h for h in spec.harmonics if h > 0]
    ac_n = ac_harmonics[0] if ac_harmonics else None

    valid_dc = np.isfinite(Q_dc)
    if ac_n is not None:
        valid_h1 = (np.isfinite(Q_hn[ac_n].real)
                    & np.isfinite(Q_hn[ac_n].imag)
                    & (np.abs(Q_hn[ac_n]) > 1e-30))
    else:
        valid_h1 = np.zeros(n_edges, dtype=bool)

    n_dc_obs = int(valid_dc.sum())
    n_h1_obs = int(valid_h1.sum()) if ac_n is not None else 0
    if n_dc_obs + n_h1_obs == 0:
        raise ValueError("No valid observations.")

    # Sweep
    D_grid = np.asarray(D_grid, dtype=float)
    chi2 = np.full(D_grid.size, np.nan)
    chi2_dc = np.full(D_grid.size, np.nan)
    chi2_h1 = np.full(D_grid.size, np.nan)
    for i_D, D_val in enumerate(D_grid):
        try:
            ab = _build_admittance_system(
                graph, edges_in, boundary_nodes, interior_nodes,
                float(D_val), spec.mu, f0_hz, spec.harmonics,
                px_size_m)
            T_all = _compute_transfer_matrices(
                ab, edges_in, boundary_nodes, interior_nodes,
                verbose=False)
            # ── Solve P_b closed-form (unweighted, no prior) ──
            # DC
            T_DC = T_all[0]
            T_DC_keep = T_DC[:, keep_idx][valid_dc]
            if T_DC_keep.shape[0] >= T_DC_keep.shape[1] and n_dc_obs > 0:
                P_DC_keep, *_ = np.linalg.lstsq(
                    T_DC_keep, Q_dc[valid_dc], rcond=None)
                P_DC_full = np.zeros(n_bnd, dtype=complex)
                P_DC_full[keep_idx] = P_DC_keep
                r_dc = (T_DC @ P_DC_full).real - Q_dc.real
                c_dc = float(np.sum(r_dc[valid_dc] ** 2))
            else:
                c_dc = 0.0
            # H1
            if ac_n is not None and n_h1_obs > 0:
                T_H1 = T_all[ac_n]
                T_H1_keep = T_H1[:, keep_idx][valid_h1]
                Q_H1 = Q_hn[ac_n]
                if T_H1_keep.shape[0] >= T_H1_keep.shape[1]:
                    P_H1_keep, *_ = np.linalg.lstsq(
                        T_H1_keep, Q_H1[valid_h1], rcond=None)
                    P_H1_full = np.zeros(n_bnd, dtype=complex)
                    P_H1_full[keep_idx] = P_H1_keep
                    r_h1 = (T_H1 @ P_H1_full) - Q_H1
                    c_h1 = float(np.sum(np.abs(r_h1[valid_h1]) ** 2))
                else:
                    c_h1 = 0.0
            else:
                c_h1 = 0.0
            chi2_dc[i_D] = c_dc
            chi2_h1[i_D] = c_h1
            chi2[i_D] = c_dc + c_h1
        except Exception:
            pass

    dof = max(n_dc_obs + 2 * n_h1_obs - (3 * (n_bnd - 1) + 1), 1)
    return dict(
        D_grid=D_grid, chi2=chi2,
        chi2_dc=chi2_dc, chi2_h1=chi2_h1,
        dof=dof,
        n_dc=n_dc_obs, n_h1=n_h1_obs,
        n_bnd=n_bnd, n_int=len(interior_nodes),
    )


def global_to_local_recovery_test(
    graph,
    D_true: float = 1e-4,
    tile_ids=None,                      # int (single tile) | sequence | None
    *,
    base_spec: Optional['LocalInferenceSpec'] = None,
    sigma_Q_nL_per_s: float = 0.1,
    n_harmonics: int = 2,
    bc_harmonics_override: Optional[Dict[int, np.ndarray]] = None,
    sink_pressure_bc: Optional[float] = None,
    rng_seed: int = 0,
    verbose: bool = True,
    save_fig: bool = True,
    landscape_D: Optional[Sequence[float]] = None,
):
    """End-to-end synthetic test: global forward with known D → per-tile
    local inference → recovered D̂.

    Parameters
    ----------
    graph : nx.Graph
        The mosaic graph — the same one local inference reads from.
    D_true : float
        Uniform distensibility used in the global forward solve.  All
        tiles should recover this same value if the per-tile model is
        a clean abstraction.
    tile_ids : sequence of int, optional
        Tiles to test.  Default: every tile id that appears in at
        least one edge's `measurements_piv`.
    base_spec : LocalInferenceSpec, optional
        Spec passed to infer_local for each tile.  Default: stock
        LocalInferenceSpec with `D_init=D_true` so iteration starts
        at truth and the test isolates noise + carve effects (not
        the initial-guess basin).  Overrideable.
    sigma_Q_nL_per_s : float
        iid Gaussian noise std added to each synthetic Q(t) sample
        in nL/s.  Typical PIV noise is ~0.01-0.1 nL/s.
    n_harmonics : int
        Harmonic count to forward-simulate and to fit (DC + N AC).
    bc_harmonics_override : dict, optional
        Optional {node_id: complex_array[Q_dc, Q_1, ...]} prescribed
        Q at source nodes.  If None, `solve_transmission_line`
        extracts BCs from the graph's measurements (realistic wave
        shape; depends on real boundary data).
    sink_pressure_bc : float, optional
        Sink (venous) pressure BC in Pa (default 0).  Pass None to
        keep all-Q BCs.
    rng_seed : int
        Noise reproducibility.
    save_fig : bool
        Save matplotlib figure to Mosaic/renders/.

    Returns
    -------
    dict with keys:
        D_true, per_tile (dict tile_id → {D_hat, sigma_D, chi2_red,
        converged, n_obs}), summary {median_ratio, frac_1sigma,
        frac_2sigma}, figure path (if saved).
    """
    import copy
    from .transmission_line import solve_transmission_line
    if base_spec is None:
        base_spec = LocalInferenceSpec(D_init=float(D_true))
    rng = np.random.default_rng(rng_seed)
    # `edge_flows` from solve_transmission_line is in **nL/s** (line
    # 1087 multiplies SI Q by 1e12), and PIV records store Q_t in
    # nL/s.  So both Q_t synthesis and the noise live in nL/s; no SI
    # conversion needed here.
    sigma_Q = float(sigma_Q_nL_per_s)

    # ── 1. Global forward solve ──
    if verbose:
        print(f"\n[global→local recovery] forward-solving "
              f"transmission line at D={D_true:.2e} 1/Pa")
    tl_result = solve_transmission_line(
        graph,
        D=float(D_true),
        n_harmonics=int(n_harmonics),
        bc_harmonics_override=bc_harmonics_override,
        sink_pressure_bc=sink_pressure_bc,
        verbose=verbose,
    )
    edge_Q = tl_result.edge_flows     # (u,v) → [Q_dc, Q_h1, ..., Q_hN]
    f0_hz = float(tl_result.f0_hz)
    omega0 = 2.0 * np.pi * f0_hz

    if tile_ids is None:
        raise ValueError(
            "tile_ids must be specified (int or sequence). "
            "Pass the currently loaded tile.")
    if isinstance(tile_ids, (int, np.integer)):
        tile_ids = [int(tile_ids)]
    tile_ids = [int(t) for t in tile_ids]
    single_tile_mode = (len(tile_ids) == 1)

    # ── 2. Snapshot all PIV records that will be touched ──
    # Touch = any edge with a PIV record for any tile in tile_ids.
    # Save (u,v) → list of deep-copied original records, plus their
    # indices in the live list so we can replace in place and
    # restore by index.
    snapshot: Dict[Tuple[int, int], List[Tuple[int, dict]]] = {}
    affected_tiles = set(tile_ids)
    for u, v, d in graph.edges(data=True):
        piv = d.get('measurements_piv') or []
        records_to_save = []
        for i, m in enumerate(piv):
            t = m.get('tile_id')
            if t is not None and int(t) in affected_tiles:
                records_to_save.append((i, copy.deepcopy(m)))
        if records_to_save:
            snapshot[(u, v)] = records_to_save

    # ── 3. Synthetic Q(t) length: read from first existing record ──
    n_t_default = None
    dt_default = None
    for (u, v), recs in snapshot.items():
        for _i, m in recs:
            Qt = m.get('Q_t')
            if Qt is not None and len(Qt) > 4:
                n_t_default = len(Qt)
                dt_default = float(m.get('frame_dt_s', 1.0 / 250.0))
                break
        if n_t_default:
            break
    if n_t_default is None:
        n_t_default = 250                  # ~1 s at 250 fps fallback
        dt_default = 1.0 / 250.0
    t_arr = np.arange(n_t_default) * dt_default

    # ── 4. Overwrite Q's per edge per tile ──
    # For each edge with a snapshot, reconstruct Q(t) from edge_Q
    # phasors and write into every covered PIV record.
    n_edges_written = 0
    try:
        for (u, v), recs in snapshot.items():
            Qharm = edge_Q.get((u, v))
            if Qharm is None:
                Qharm = edge_Q.get((v, u))
                sgn_uv = -1.0
            else:
                sgn_uv = +1.0
            if Qharm is None:
                continue
            Q_dc = float(np.real(Qharm[0])) * sgn_uv
            Q_t_syn = np.full_like(t_arr, Q_dc, dtype=float)
            for n_h in range(1, len(Qharm)):
                phasor = complex(Qharm[n_h]) * sgn_uv
                # Forward convention: Q(t) = Q_dc + Σ_n Re[Q̂_n
                # · exp(j·n·ω0·t)] — **single-sided** (no factor of 2),
                # matching solve_transmission_line lines 106 / 218 /
                # 1117 and `_meas_phasors_for_edge` which returns
                # Q̂ = A − jB for Q(t) = A cos + B sin.
                Q_t_syn = Q_t_syn + (
                    phasor.real * np.cos(n_h * omega0 * t_arr)
                    - phasor.imag * np.sin(n_h * omega0 * t_arr))
            # iid Gaussian noise (in nL/s — matches Q_t / mean_Q units)
            noise = rng.normal(0.0, sigma_Q, size=n_t_default)
            Q_t_syn_noisy = Q_t_syn + noise
            # Sign convention: record's flow_from/flow_to defines the
            # storage orientation in m_ref.  We've already signed Q
            # in (u,v) convention; if a particular record has the
            # opposite orientation, flip on write so the inference
            # reads the right sign.
            for idx, _orig in recs:
                m_live = graph.edges[u, v]['measurements_piv'][idx]
                ff = m_live.get('flow_from')
                ft = m_live.get('flow_to')
                if ff == v and ft == u:
                    sign_rec = -1.0
                else:
                    sign_rec = +1.0
                m_live['Q_t'] = (sign_rec * Q_t_syn_noisy).tolist()
                m_live['mean_Q'] = float(sign_rec * Q_dc)
                # Wipe pre-cached harmonics so _meas_phasors_for_edge
                # falls back to Q_t FFT (always uses fresh synthetic).
                m_live.pop('harmonics', None)
                m_live['f0_hz'] = f0_hz
                m_live['frame_dt_s'] = dt_default
            n_edges_written += 1

        if verbose:
            print(f"[global→local recovery] wrote synthetic Q(t) on "
                  f"{n_edges_written} edges (covering "
                  f"{len(tile_ids)} tiles); σ_Q = "
                  f"{sigma_Q_nL_per_s} nL/s, "
                  f"n_harmonics = {n_harmonics}")

        # ── 5. Per-tile local inference + (single-tile) landscape ──
        per_tile = {}
        landscape = None
        for i_t, tid in enumerate(tile_ids):
            if verbose:
                print(f"  tile {tid} ({i_t+1}/{len(tile_ids)}) …",
                      end=' ', flush=True)
            spec = LocalInferenceSpec(
                **{k: v for k, v in base_spec.__dict__.items()})
            # Start D_init AWAY from truth so the landscape & solver
            # are an honest test of basin-of-convergence, not just
            # noise reaction near the optimum.
            spec.D_init = float(base_spec.D_init)
            spec.save_to_graph = False
            spec.verbose = False
            try:
                res = infer_local(graph, int(tid), spec)
                per_tile[int(tid)] = {
                    'D_hat': float(res.D_hat),
                    'sigma_D': float(res.sigma_D),
                    'chi2_red': float(res.chi2_red),
                    'converged': bool(res.converged),
                    'n_obs': int(res.n_obs_real),
                    'iters': int(res.iterations),
                }
                if verbose:
                    print(f"D̂ = {res.D_hat:.3e} ± "
                          f"{res.sigma_D:.2e}  "
                          f"(χ²/dof = {res.chi2_red:.2f})")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                per_tile[int(tid)] = {
                    'D_hat': float('nan'),
                    'sigma_D': float('nan'),
                    'chi2_red': float('nan'),
                    'converged': False, 'n_obs': 0, 'iters': 0,
                    'error': str(e)}

            # χ²(D) landscape — only in single-tile mode.  At each D
            # in a log grid: build T(D), solve P_b in closed form
            # (no prior), compute residual χ²/dof.  This is the
            # profile likelihood over D after eliminating P_b.
            if single_tile_mode and i_t == 0:
                try:
                    if landscape_D is None:
                        D_grid = np.logspace(
                            np.log10(D_true) - 3,
                            np.log10(D_true) + 3,
                            61)
                    else:
                        D_grid = np.asarray(landscape_D, dtype=float)
                    landscape = _landscape_chi2_over_D(
                        graph, int(tid), D_grid, spec)
                    if verbose:
                        idx_min = int(np.argmin(landscape['chi2']))
                        print(f"  landscape: min χ² at D = "
                              f"{landscape['D_grid'][idx_min]:.3e}, "
                              f"χ²/dof = "
                              f"{landscape['chi2'][idx_min] / max(landscape['dof'], 1):.2f}")
                except Exception as _e:
                    if verbose:
                        print(f"  landscape failed: {_e}")

    finally:
        # ── 6. Restore originals (ALWAYS) ──
        for (u, v), recs in snapshot.items():
            for idx, orig in recs:
                graph.edges[u, v]['measurements_piv'][idx] = orig
        if verbose:
            print(f"[global→local recovery] restored original PIV "
                  f"records on {len(snapshot)} edges.")

    # ── 7. Summary stats + plot ──
    D_hats = np.array([per_tile[t]['D_hat'] for t in tile_ids
                        if np.isfinite(per_tile[t]['D_hat'])])
    sigs = np.array([per_tile[t]['sigma_D'] for t in tile_ids
                       if np.isfinite(per_tile[t]['D_hat'])])
    valid_tids = [t for t in tile_ids
                    if np.isfinite(per_tile[t]['D_hat'])]

    if D_hats.size == 0:
        return dict(D_true=D_true, per_tile=per_tile,
                    summary={'median_ratio': float('nan'),
                              'frac_1sigma': 0.0,
                              'frac_2sigma': 0.0},
                    figure_path=None)

    ratios = D_hats / float(D_true)
    z = (D_hats - float(D_true)) / np.where(sigs > 0, sigs, np.inf)
    frac_1s = float(np.mean(np.abs(z) <= 1.0))
    frac_2s = float(np.mean(np.abs(z) <= 2.0))
    median_ratio = float(np.median(ratios))

    if verbose:
        print(f"\n[global→local recovery] summary:")
        print(f"  D_true = {D_true:.3e} 1/Pa")
        print(f"  n_tiles fit successfully = {D_hats.size}/"
              f"{len(tile_ids)}")
        print(f"  median D̂ / D_true = {median_ratio:.3f}")
        print(f"  fraction within 1σ of truth = {frac_1s:.2f}")
        print(f"  fraction within 2σ of truth = {frac_2s:.2f}")

    figure_path = None
    if save_fig:
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            if single_tile_mode and landscape is not None:
                # ── Single-tile landscape figure ──
                tid = tile_ids[0]
                rec = per_tile[tid]
                D_grid = landscape['D_grid']
                # Normalize χ² to (nL/s)² — multiplies raw SI χ² by
                # 1e24 so y-axis values are O(1) instead of O(10⁻²⁵).
                # All three (total, DC, H1) get the same factor.
                _NL2 = 1.0e24
                chi2 = landscape['chi2'] * _NL2
                chi2_dc = landscape['chi2_dc'] * _NL2
                chi2_h1 = landscape['chi2_h1'] * _NL2
                dof = landscape['dof']
                # Per-obs noise floor (expected χ² if residuals were
                # pure σ_Q noise).  σ_phasor ≈ σ_t/√(N/2) for an
                # ~N-sample Q(t) → ~250 samples → factor √125.
                _N_samples_typ = 250.0
                _sigma_dc_nL = float(sigma_Q_nL_per_s) / np.sqrt(
                    _N_samples_typ)
                _sigma_h1_nL = float(sigma_Q_nL_per_s) / np.sqrt(
                    _N_samples_typ / 2.0)
                _n_dc = int(landscape.get('n_dc') or 284)
                _n_h1 = int(landscape.get('n_h1') or 284)
                # χ²_noise expectation in (nL/s)²
                noise_floor_total = (
                    _n_dc * _sigma_dc_nL ** 2
                    + 2.0 * _n_h1 * _sigma_h1_nL ** 2)
                # Avoid log(0); chi² values are non-neg
                ok = np.isfinite(chi2) & (chi2 > 0)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
                ax1.plot(D_grid[ok], chi2[ok], color='#1f77b4',
                          lw=2.0, label='total χ²')
                ax1.plot(D_grid[ok], chi2_dc[ok], color='#ff7f0e',
                          lw=1.0, ls=':', label='DC χ²')
                ax1.plot(D_grid[ok], chi2_h1[ok], color='#2ca02c',
                          lw=1.0, ls='--', label='H1 χ²')
                ax1.axvline(D_true, color='red', lw=1.5, ls='--',
                              label=f'D_true = {D_true:.2e}')
                if np.isfinite(rec['D_hat']):
                    ax1.axvline(rec['D_hat'], color='#444',
                                  lw=1.5, ls=':',
                                  label=f"D̂_solver = "
                                        f"{rec['D_hat']:.2e}")
                idx_min = int(np.nanargmin(chi2))
                ax1.axvline(D_grid[idx_min], color='purple',
                              lw=1.0, ls='-.', alpha=0.6,
                              label=f'argmin χ² = '
                                     f'{D_grid[idx_min]:.2e}')
                # Reference: expected χ² noise floor (per-obs σ²
                # summed).  Below this line, the fit is doing better
                # than pure noise alone would explain — typical for
                # synthetic data where the noise was injected at the
                # phasor level but the fit can absorb some of it.
                if np.isfinite(noise_floor_total) and noise_floor_total > 0:
                    ax1.axhline(noise_floor_total, color='gray',
                                  lw=1.0, ls=':',
                                  label=f'σ_Q noise floor = '
                                         f'{noise_floor_total:.2e}')
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.set_xlabel('D (1/Pa)')
                ax1.set_ylabel('χ²  [(nL/s)², unweighted]')
                ax1.set_title(
                    f'Tile {tid}  •  χ²(D) profile  •  '
                    f'σ_Q = {sigma_Q_nL_per_s:g} nL/s  •  '
                    f'{n_harmonics} harmonics')
                ax1.legend(fontsize=8, loc='best')
                ax1.grid(alpha=0.3, which='both')

                # Right panel: zoom to a window around the actual χ²
                # minimum (NOT around D_true), to keep the parabola
                # fit honest when the curve is asymmetric.
                idx_min_zoom = int(np.nanargmin(chi2))
                D_at_min = D_grid[idx_min_zoom]
                # Tight zoom ± 1 decade around the min
                D_show_lo = max(D_at_min / 10, D_grid[ok].min())
                D_show_hi = min(D_at_min * 10, D_grid[ok].max())
                m = ok & (D_grid >= D_show_lo) & (D_grid <= D_show_hi)
                ax2.plot(D_grid[m], chi2[m], color='#1f77b4',
                          lw=2.0)
                ax2.axvline(D_true, color='red', lw=1.5, ls='--',
                              label=f'D_true')
                if np.isfinite(rec['D_hat']):
                    ax2.axvline(rec['D_hat'], color='#444',
                                  lw=1.5, ls=':', label='D̂_solver')
                # Parabola fit: restrict to points within Δχ² ≤
                # 4·(χ²_min depth) of the actual minimum, so the fit
                # captures local curvature only (not the steep climb).
                try:
                    Cm_all = chi2
                    chi2_min = float(np.nanmin(Cm_all))
                    chi2_max_window = float(np.nanmax(Cm_all[m]))
                    depth = max(chi2_max_window - chi2_min, 1e-30)
                    # Keep points within 25% of the depth above min
                    near = m & (Cm_all <= chi2_min + 0.25 * depth)
                    if int(near.sum()) >= 5:
                        log_D = np.log(D_grid[near])
                        Cm = chi2[near]
                        coefs = np.polyfit(log_D, Cm, 2)
                        a_p = coefs[0]
                        if a_p > 0:
                            D_min_parab = float(np.exp(
                                -coefs[1] / (2 * a_p)))
                            # σ_D via Δχ² = noise_floor_total
                            # criterion: the basin width corresponding
                            # to one "noise-floor's worth" of χ²
                            # increase = 1σ on D.  Replaces the
                            # plain Δχ²=1 criterion (which is only
                            # valid when χ² is noise-normalized).
                            target_dchi2 = max(noise_floor_total, 1.0)
                            log_sigma_logD = np.sqrt(
                                target_dchi2 / a_p)
                            sigma_D_parab = (D_min_parab
                                              * log_sigma_logD)
                            # Sanity-cap σ to ±5× D_min so the label
                            # stays legible when the basin is shallow.
                            sigma_D_parab = min(
                                sigma_D_parab, 5.0 * D_min_parab)
                            ax2.axvspan(
                                max(D_min_parab - sigma_D_parab,
                                    D_grid[ok].min()),
                                D_min_parab + sigma_D_parab,
                                color='purple', alpha=0.12,
                                label=f'parabola: D = '
                                       f'{D_min_parab:.2e} ± '
                                       f'{sigma_D_parab:.1e}')
                            ax2.axvline(D_min_parab,
                                          color='purple',
                                          lw=1.0, ls='-.',
                                          alpha=0.7)
                except Exception:
                    pass
                ax2.set_xscale('log')
                ax2.set_xlabel('D (1/Pa)')
                ax2.set_ylabel('χ²  [(nL/s)²]')
                ax2.set_title(
                    f'Zoom near truth  •  carve: '
                    f"{landscape['n_int']} interior, "
                    f"{landscape['n_bnd']} boundary nodes  •  "
                    f"obs: DC×{landscape['n_dc']}, "
                    f"H1×{landscape['n_h1']}")
                ax2.legend(fontsize=8, loc='best')
                ax2.grid(alpha=0.3)

                plt.tight_layout()
            else:
                # ── Multi-tile scatter + histogram (original) ──
                fig, (ax1, ax2) = plt.subplots(1, 2,
                                                  figsize=(12, 4.5))
                x = np.arange(len(valid_tids))
                ax1.errorbar(x, D_hats, yerr=sigs, fmt='o',
                              color='#1f77b4', ms=5, capsize=2,
                              ecolor='#888')
                ax1.axhline(D_true, color='red', lw=1.5, ls='--',
                              label=f'D_true = {D_true:.2e}')
                ax1.set_yscale('log')
                ax1.set_xticks(x)
                ax1.set_xticklabels([str(t) for t in valid_tids],
                                      rotation=60, fontsize=7)
                ax1.set_xlabel('tile id')
                ax1.set_ylabel('D̂ (1/Pa)')
                ax1.set_title(f'Per-tile D̂ vs D_true  '
                                f'(σ_Q = {sigma_Q_nL_per_s:g} '
                                f'nL/s, {n_harmonics} harmonics)')
                ax1.legend(fontsize=8)
                ax1.grid(alpha=0.3, which='both')

                ax2.hist(np.log10(ratios), bins=15, color='#1f77b4',
                           alpha=0.75, edgecolor='black')
                ax2.axvline(0, color='red', lw=1.5, ls='--',
                              label='ratio = 1')
                ax2.axvline(np.log10(median_ratio), color='#444',
                              lw=1.5, ls=':', label=f'median = '
                              f'{median_ratio:.2f}')
                ax2.set_xlabel('log10( D̂ / D_true )')
                ax2.set_ylabel('# tiles')
                ax2.set_title(
                    f'Recovery distribution  •  '
                    f'within 1σ: {frac_1s:.0%}  •  '
                    f'within 2σ: {frac_2s:.0%}')
                ax2.legend(fontsize=8)
                ax2.grid(alpha=0.3, axis='y')

                plt.tight_layout()
            mosaic_path = getattr(graph, 'graph', {}).get(
                'mosaic_path', None)
            if mosaic_path:
                out_dir = (Path(mosaic_path).parent.parent
                           / 'renders' / 'recovery_tests')
            else:
                out_dir = Path.home() / 'Downloads' \
                    / 'pertile_diagnostics'
            out_dir.mkdir(parents=True, exist_ok=True)
            import datetime as _dt
            ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = out_dir / (
                f'global_to_local_recovery_D{D_true:.0e}_'
                f'sigQ{sigma_Q_nL_per_s:g}_{ts}.png')
            fig.savefig(str(out_path), dpi=140, bbox_inches='tight')
            figure_path = str(out_path)
            if verbose:
                print(f"  saved figure → {out_path}")
            plt.show()
        except Exception as _e:
            if verbose:
                print(f"  (Could not save figure: {_e})")

    return dict(
        D_true=float(D_true),
        per_tile=per_tile,
        summary=dict(
            median_ratio=median_ratio,
            frac_1sigma=frac_1s,
            frac_2sigma=frac_2s,
            n_valid=int(D_hats.size),
        ),
        landscape=landscape,
        figure_path=figure_path,
    )


def recovery_sweep_across_tiles(
    graph,
    tile_ids: Optional[Sequence[int]] = None,
    D_grid: Optional[Sequence[float]] = None,
    *,
    sigma_Q_nL_per_s: float = 0.05,
    sigma_Q_base_nL_per_s: Optional[float] = None,
    sigma_Q_rel: Optional[float] = None,
    n_harmonics: int = 2,
    bc_mode: str = 'all_q',
    base_spec: Optional['LocalInferenceSpec'] = None,
    bias_tol: tuple = (0.5, 2.0),
    rng_seed: int = 0,
    mode: str = 'fast',                # 'fast' | 'full'
    n_grid_per_tile: int = 21,
    verbose: bool = True,
    save_fig: bool = True,
):
    """Batch recovery sweep: every tile × every D_true.

    Produces a single heatmap that tells you the operating envelope
    of local inference: which (tile, D_true) combinations recover
    within the user's bias tolerance, and which are problematic.

    For each D_true in `D_grid`:
      1. Run the global TL solve at D_true with chosen BC mode.
      2. For each tile in `tile_ids`:
         - Snapshot the tile's PIV records
         - Write synthetic Q(t) from *_syn fields
         - Run `infer_local` with D_init=D_true
         - Record D̂, σ_D, ratio = D̂/D_true
         - Restore PIV records via try/finally
      3. Move to next D_true.

    Outputs:
      - 2D heatmap (D_true × tile) of log10(D̂/D_true).
      - Per-D_true histogram of bias factor.
      - Reliability table: fraction of tiles within bias_tol per D_true.
      - Saves figure to Mosaic/renders/recovery_tests/ + CSV alongside.

    Parameters
    ----------
    bias_tol : (low, high) tuple
        Bias factors considered "acceptable".  Default (0.5, 2.0) =
        within a factor of 2 of truth.  Cells outside this range
        are flagged in the heatmap.
    """
    import copy
    import datetime as _dt
    from pathlib import Path
    from .transmission_line import solve_transmission_line

    if base_spec is None:
        base_spec = LocalInferenceSpec()

    # Default tile list: all tiles with PIV
    if tile_ids is None:
        tids = set()
        for _u, _v, d in graph.edges(data=True):
            for m in (d.get('measurements_piv') or []):
                t = m.get('tile_id')
                if t is not None:
                    tids.add(int(t))
        tile_ids = sorted(tids)
    tile_ids = [int(t) for t in tile_ids]

    # Default D grid: covers the realistic embryonic range
    # (D ~ 1e-4 from memory, with margin into stiffer regime).
    if D_grid is None:
        D_grid = [1e-4, 1.78e-4, 3.16e-4, 5.62e-4, 1e-3]
    D_grid = list(D_grid)

    if verbose:
        print(f"\n[recovery sweep] {len(tile_ids)} tiles × "
              f"{len(D_grid)} D values = "
              f"{len(tile_ids) * len(D_grid)} inferences")
        print(f"  σ_Q = {sigma_Q_nL_per_s} nL/s, "
              f"n_harmonics = {n_harmonics}, "
              f"BC = {bc_mode}, "
              f"bias tol = [{bias_tol[0]}, {bias_tol[1]}]")

    # results[(tile_id, D_true)] → dict
    results = {}
    t0_total = _dt.datetime.now()

    for i_D, D_true in enumerate(D_grid):
        if verbose:
            print(f"\n  D_true = {D_true:.2e}  ({i_D+1}/{len(D_grid)})")
        # ── Run synthetic forward once at this D ──
        try:
            run_synthetic_simulation(
                graph, D=float(D_true),
                sigma_Q_nL_per_s=sigma_Q_nL_per_s,
                sigma_Q_base_nL_per_s=sigma_Q_base_nL_per_s,
                sigma_Q_rel=sigma_Q_rel,
                n_harmonics=n_harmonics,
                bc_mode=bc_mode,
                rng_seed=rng_seed,
                verbose=False)
        except Exception as e:
            if verbose:
                print(f"    [synthetic forward failed: {e}]")
            for tid in tile_ids:
                results[(int(tid), float(D_true))] = dict(
                    D_hat=float('nan'), sigma_D=float('nan'),
                    ratio=float('nan'), chi2_red=float('nan'),
                    converged=False, error=str(e))
            continue

        # ── Per-tile inference ──
        for i_t, tid in enumerate(tile_ids):
            try:
                # Build per-tile spec.  HARD-cap max_iter and
                # n_outer_iter for the sweep regardless of what the
                # caller's base_spec says — the sweep is a smoke
                # test, not a production fit, and the LM's
                # asymptotic crawling can otherwise drag a single
                # tile to 300+ iterations.
                spec_kw = {k: v for k, v in base_spec.__dict__.items()}
                spec_kw['D_init'] = float(D_true)
                spec_kw['save_to_graph'] = False
                spec_kw['verbose'] = False
                spec_kw['max_iter'] = min(
                    int(spec_kw.get('max_iter', 50)), 30)
                spec_kw['n_outer_iter'] = min(
                    int(spec_kw.get('n_outer_iter', 2)), 2)
                spec_kw['tol_rel'] = max(
                    float(spec_kw.get('tol_rel', 1e-3)), 1e-3)
                spec = LocalInferenceSpec(**spec_kw)
                t0 = _dt.datetime.now()

                if mode == 'fast':
                    # χ²(D) landscape with no LM, no FGLS.  Snapshot
                    # the tile's PIV records, write synthetic Q(t)
                    # from saved *_syn phasors, run landscape, restore.
                    # This is the "what does the data say" answer
                    # without LM convergence-basin artefacts.
                    import copy
                    snapshot_t: Dict[Tuple[int, int],
                                      List[Tuple[int, dict]]] = {}
                    for u, v, d in graph.edges(data=True):
                        piv = d.get('measurements_piv') or []
                        recs_t = [(i, copy.deepcopy(m))
                                  for i, m in enumerate(piv)
                                  if m.get('tile_id') == int(tid)]
                        if recs_t:
                            snapshot_t[(u, v)] = recs_t
                    # Need f0 + dt for Q_t reconstruction
                    meta = graph.graph.get(
                        'synthetic_sim_meta') or {}
                    f0_hz = float(meta.get('f0_hz', 2.5))
                    omega0 = 2.0 * np.pi * f0_hz
                    # Determine record length from one snapshot record
                    n_t_def = 250
                    dt_def = 1.0 / 250.0
                    for _e, _r in snapshot_t.items():
                        for _i, _m in _r:
                            Qt = _m.get('Q_t')
                            if Qt is not None and len(Qt) > 4:
                                n_t_def = len(Qt)
                                dt_def = float(
                                    _m.get('frame_dt_s', 1.0/250.0))
                                break
                        break
                    t_arr = np.arange(n_t_def) * dt_def
                    rng_t = np.random.default_rng(
                        rng_seed + i_t)
                    try:
                        # Overwrite PIV records with synthetic
                        for (u, v), recs in snapshot_t.items():
                            d = graph.edges[u, v]
                            q_dc_s = d.get('Q_DC_syn')
                            if q_dc_s is None or not np.isfinite(q_dc_s):
                                continue
                            q_dc_s = float(q_dc_s)
                            Q_t_syn = np.full_like(
                                t_arr, q_dc_s, dtype=float)
                            for n_h in (1, 2, 3):
                                qre = d.get(f'Q_H{n_h}_re_syn')
                                qim = d.get(f'Q_H{n_h}_im_syn')
                                if (qre is None or qim is None
                                        or not np.isfinite(qre)
                                        or not np.isfinite(qim)):
                                    continue
                                Q_t_syn = Q_t_syn + (
                                    float(qre)*np.cos(n_h*omega0*t_arr)
                                    - float(qim)*np.sin(n_h*omega0*t_arr))
                            Q_t_syn = Q_t_syn + rng_t.normal(
                                0.0, sigma_Q_nL_per_s, size=n_t_def)
                            for idx, _orig in recs:
                                m_live = graph.edges[u, v][
                                    'measurements_piv'][idx]
                                edge_ff = d.get('flow_from')
                                edge_ft = d.get('flow_to')
                                ff = m_live.get('flow_from')
                                ft = m_live.get('flow_to')
                                if (ff is not None and ft is not None
                                        and edge_ff is not None
                                        and edge_ft is not None
                                        and ff == edge_ft
                                        and ft == edge_ff):
                                    sign_r = -1.0
                                else:
                                    sign_r = +1.0
                                m_live['Q_t'] = (
                                    sign_r * Q_t_syn).tolist()
                                m_live['mean_Q'] = float(
                                    sign_r * q_dc_s)
                                m_live.pop('harmonics', None)
                                m_live['f0_hz'] = f0_hz
                                m_live['frame_dt_s'] = dt_def

                        # Run landscape
                        D_grid_local = np.logspace(
                            np.log10(D_true) - 1.5,
                            np.log10(D_true) + 1.5,
                            int(n_grid_per_tile))
                        lscape = _landscape_chi2_over_D(
                            graph, int(tid), D_grid_local, spec)
                        chi2 = lscape['chi2']
                        ok = np.isfinite(chi2) & (chi2 > 0)
                        if not np.any(ok):
                            raise ValueError("landscape all-NaN")
                        idx_min = int(np.nanargmin(chi2))
                        D_hat_l = float(lscape['D_grid'][idx_min])
                        # σ_D via parabola in log-D near min
                        chi2_min = float(chi2[idx_min])
                        depth = (float(np.nanmax(chi2)) - chi2_min)
                        near = ok & (chi2 <= chi2_min + 0.25 * depth)
                        sigma_D_l = float('nan')
                        if int(near.sum()) >= 5:
                            log_D_n = np.log(lscape['D_grid'][near])
                            Cm_n = chi2[near]
                            try:
                                coefs = np.polyfit(log_D_n, Cm_n, 2)
                                a_p = coefs[0]
                                if a_p > 0:
                                    D_min_p = float(np.exp(
                                        -coefs[1] / (2 * a_p)))
                                    log_sig = (1.0
                                                / np.sqrt(max(a_p, 1e-30)))
                                    sigma_D_l = float(
                                        D_min_p * log_sig)
                            except Exception:
                                pass
                        dt = (_dt.datetime.now()
                              - t0).total_seconds()
                        ratio = D_hat_l / float(D_true)
                        results[(int(tid), float(D_true))] = dict(
                            D_hat=D_hat_l, sigma_D=sigma_D_l,
                            ratio=float(ratio),
                            chi2_red=float('nan'),
                            converged=True, iters=0,
                            secs=float(dt))
                        if verbose:
                            flag = ('✓'
                                    if bias_tol[0] <= ratio <= bias_tol[1]
                                    else '✗')
                            print(f"    tile {tid:>3} "
                                  f"({i_t+1}/{len(tile_ids)}): "
                                  f"D̂_landscape = {D_hat_l:.2e}, "
                                  f"ratio = {ratio:.2f} {flag}  "
                                  f"({dt:.1f}s)")
                    finally:
                        for (u, v), recs in snapshot_t.items():
                            for idx, orig in recs:
                                graph.edges[u, v][
                                    'measurements_piv'][idx] = orig
                else:
                    # 'full' mode: original behaviour (LM + FGLS)
                    res = run_local_inference_on_synthetic(
                        graph, int(tid),
                        base_spec=spec,
                        sigma_Q_nL_per_s=sigma_Q_nL_per_s,
                        rng_seed=rng_seed + i_t,
                        verbose=False)
                    dt = (_dt.datetime.now() - t0).total_seconds()
                    ratio = (float(res.D_hat) / float(D_true)
                             if D_true > 0 else float('nan'))
                    results[(int(tid), float(D_true))] = dict(
                        D_hat=float(res.D_hat),
                        sigma_D=float(res.sigma_D),
                        ratio=float(ratio),
                        chi2_red=float(res.chi2_red),
                        converged=bool(res.converged),
                        iters=int(res.iterations),
                        secs=float(dt),
                    )
                    if verbose:
                        flag = ('✓'
                                if bias_tol[0] <= ratio <= bias_tol[1]
                                else '✗')
                        print(f"    tile {tid:>3} "
                              f"({i_t+1}/{len(tile_ids)}): "
                              f"D̂ = {res.D_hat:.2e}, ratio = "
                              f"{ratio:.2f} {flag}  "
                              f"χ²/dof = {res.chi2_red:.2f}  "
                              f"({dt:.1f}s)")
            except Exception as e:
                results[(int(tid), float(D_true))] = dict(
                    D_hat=float('nan'), sigma_D=float('nan'),
                    ratio=float('nan'), chi2_red=float('nan'),
                    converged=False, error=str(e))
                if verbose:
                    print(f"    tile {tid:>3}: FAIL ({e})")

    total_secs = (_dt.datetime.now() - t0_total).total_seconds()
    if verbose:
        print(f"\n[recovery sweep] total time: {total_secs:.0f}s")

    # ── Per-D_true summary ──
    summary_per_D = {}
    for D in D_grid:
        ratios = [results[(t, D)]['ratio'] for t in tile_ids
                  if np.isfinite(results[(t, D)]['ratio'])]
        if ratios:
            ratios = np.array(ratios)
            inside = (ratios >= bias_tol[0]) & (ratios <= bias_tol[1])
            summary_per_D[D] = dict(
                n_tiles=int(len(ratios)),
                median_ratio=float(np.median(ratios)),
                frac_in_tol=float(inside.mean()),
                geomean_ratio=float(np.exp(np.mean(np.log(ratios)))),
            )
        else:
            summary_per_D[D] = dict(
                n_tiles=0, median_ratio=float('nan'),
                frac_in_tol=0.0, geomean_ratio=float('nan'))

    if verbose:
        print(f"\n[recovery sweep] reliability summary "
              f"(bias tol [{bias_tol[0]}, {bias_tol[1]}]):")
        for D in D_grid:
            s = summary_per_D[D]
            print(f"  D = {D:.2e}: median {s['median_ratio']:.2f}, "
                  f"geomean {s['geomean_ratio']:.2f}, "
                  f"{s['frac_in_tol']*100:.0f}% of {s['n_tiles']} "
                  f"tiles within tol")

    # ── Heatmap + histograms ──
    figure_path = None
    if save_fig:
        try:
            import matplotlib.pyplot as plt
            ratio_matrix = np.full(
                (len(tile_ids), len(D_grid)), np.nan, dtype=float)
            for ti, t in enumerate(tile_ids):
                for di, D in enumerate(D_grid):
                    r = results[(t, D)]['ratio']
                    if np.isfinite(r) and r > 0:
                        ratio_matrix[ti, di] = np.log10(r)

            fig, axes = plt.subplots(
                1, 2, figsize=(14, max(6, 0.18 * len(tile_ids))),
                gridspec_kw={'width_ratios': [2.2, 1.0]})

            ax_heat, ax_hist = axes
            vmax = float(np.log10(bias_tol[1]))
            vmin = float(np.log10(bias_tol[0]))
            # Symmetrize about 0 so red = over, blue = under, white = on truth
            vsym = max(abs(vmin), abs(vmax)) * 2
            im = ax_heat.imshow(
                ratio_matrix, aspect='auto', cmap='RdBu_r',
                vmin=-vsym, vmax=vsym, interpolation='nearest')
            ax_heat.set_xticks(np.arange(len(D_grid)))
            ax_heat.set_xticklabels(
                [f"{D:.0e}" for D in D_grid], rotation=0)
            ax_heat.set_xlabel("D_true (1/Pa)")
            ax_heat.set_yticks(np.arange(len(tile_ids)))
            ax_heat.set_yticklabels([str(t) for t in tile_ids],
                                       fontsize=6)
            ax_heat.set_ylabel("tile id")
            ax_heat.set_title(
                f"log₁₀(D̂ / D_true)  •  "
                f"σ_Q = {sigma_Q_nL_per_s} nL/s, BC={bc_mode}")
            cb = plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
            cb.set_label("log₁₀(D̂/D_true)")
            # Mark out-of-tolerance cells with hatching
            for ti in range(len(tile_ids)):
                for di in range(len(D_grid)):
                    v = ratio_matrix[ti, di]
                    if np.isfinite(v):
                        if v < vmin or v > vmax:
                            ax_heat.add_patch(plt.Rectangle(
                                (di - 0.5, ti - 0.5), 1, 1,
                                fill=False, edgecolor='black',
                                lw=0.5, hatch='///'))

            # Histogram per D
            for di, D in enumerate(D_grid):
                col = ratio_matrix[:, di]
                col = col[np.isfinite(col)]
                if col.size == 0: continue
                ax_hist.hist(col, bins=20, alpha=0.4,
                              label=f"D={D:.0e} "
                                    f"(med {summary_per_D[D]['median_ratio']:.2f})")
            ax_hist.axvline(0, color='red', ls='--', lw=1.5,
                              label='unbiased')
            ax_hist.axvline(np.log10(bias_tol[0]), color='gray',
                              ls=':', alpha=0.7)
            ax_hist.axvline(np.log10(bias_tol[1]), color='gray',
                              ls=':', alpha=0.7,
                              label=f'tol [{bias_tol[0]}, '
                                     f'{bias_tol[1]}]')
            ax_hist.set_xlabel("log₁₀(D̂/D_true)")
            ax_hist.set_ylabel("# tiles")
            ax_hist.set_title("Bias distribution per D_true")
            ax_hist.legend(fontsize=8)
            ax_hist.grid(alpha=0.3, axis='y')

            plt.tight_layout()

            # Save
            mosaic_path = getattr(graph, 'graph', {}).get(
                'mosaic_path', None)
            if mosaic_path:
                out_dir = (Path(mosaic_path).parent.parent
                           / 'renders' / 'recovery_tests')
            else:
                out_dir = Path.home() / 'Downloads' \
                    / 'pertile_diagnostics'
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = out_dir / (
                f'recovery_sweep_sigQ{sigma_Q_nL_per_s:g}_'
                f'{bc_mode}_{ts}.png')
            fig.savefig(str(out_path), dpi=140, bbox_inches='tight')

            # CSV
            csv_path = out_dir / (
                f'recovery_sweep_sigQ{sigma_Q_nL_per_s:g}_'
                f'{bc_mode}_{ts}.csv')
            with open(csv_path, 'w') as f:
                f.write("tile_id,D_true,D_hat,sigma_D,ratio,"
                        "chi2_red,converged\n")
                for t in tile_ids:
                    for D in D_grid:
                        r = results[(t, D)]
                        f.write(f"{t},{D},{r['D_hat']:.6e},"
                                f"{r['sigma_D']:.6e},"
                                f"{r['ratio']:.4f},"
                                f"{r['chi2_red']:.4f},"
                                f"{int(r['converged'])}\n")
            figure_path = str(out_path)
            if verbose:
                print(f"  saved figure → {out_path}")
                print(f"  saved CSV → {csv_path}")
            plt.show()
        except Exception as _e:
            if verbose:
                print(f"  (Could not save sweep figure: {_e})")

    return dict(
        tile_ids=tile_ids,
        D_grid=D_grid,
        results=results,
        summary_per_D=summary_per_D,
        bias_tol=tuple(bias_tol),
        figure_path=figure_path,
    )


# =====================================================================
# Phase-gradient D̂ estimator (Dijkstra many-edges-vs-reference)
# =====================================================================
#
# Reads per-edge oriented-phase data along a flow-direction-respecting
# graph and fits  Δφ = a + slope · cum_LR  by amplitude-weighted LS
# plus a ±π disambiguation pass and an optional Huber variant.
#
#     D̂ = slope² / (8 · n · ω₀ · μ)
#
# Wave-equation derivation: Re(k) = 2√2 √(ωμD)/R, so along any path
#     dφ/d(cum_LR) = −2√2 √(n ω₀ μ D)
# regardless of how R varies along the path (cum_LR absorbs it).
#
# Source modes:
#   'piv' — read per-record PIV data for the requested tile_id (each
#           (edge, tile) pair has its own Q_t time series; H1 phasor is
#           computed by direct FFT projection at the record's f0_hz).
#           This is the production path on real data.
#   'syn' — read the *_syn fields written by run_synthetic_simulation.
#           Used for validation against a known D_true.
# =====================================================================

def per_tile_D_flow_dijkstra(
    graph,
    tile_id: int,
    *,
    source: str = 'piv',
    mu: float = 2.5e-3,
    px_size_m: Optional[float] = None,
    harmonic: int = 1,
    min_edges: int = 20,
    min_quality: str = 'C',
    frame_dt_s_default: float = 1.0 / 250.0,
) -> Optional[dict]:
    """Phase-gradient D̂ on one tile.

    Returns dict with:
        D_hat        — primary estimate from the LS+π-disambig fit
        slope, intercept, residual_std, f0_hz, n_edges_used, n_total,
        coverage_pct, source_node, e_ref,
        df           — per-edge rows (cum_LR_rel, dphi, dphi_corrected,
                       amp, R_um, L_um, Q_DC, u, v, reachable)
        fit_variants — {LS, LS+π-disambig, Huber}: each
                         {slope, intercept, resid_std, D_hat}
        edge_records — list of per-edge dicts with positions for plot

    Returns None if fewer than `min_edges` reachable edges remain.
    """
    import numpy as _np
    import networkx as _nx

    if px_size_m is None:
        try:
            from .config import get_px_size_m as _gp
            px_size_m = float(_gp())
        except Exception:
            px_size_m = 1.7e-6

    quality_order = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    min_q = quality_order.get(str(min_quality).upper(), 1)

    def _h1_from_qt(Q_t, f0, dt):
        Q = _np.asarray(Q_t, dtype=float)
        N = len(Q)
        if N < 30:
            return None
        t = _np.arange(N) * float(dt)
        mean = float(_np.mean(Q))
        Q_dm = Q - mean
        omega = 2.0 * _np.pi * float(f0) * harmonic
        c = (2.0 / N) * _np.sum(Q_dm * _np.exp(-1j * omega * t))
        return mean, c

    # ── Collect tile-specific per-edge oriented H1 phasors ──
    edges = []
    for u, v, d in graph.edges(data=True):
        if source == 'syn':
            suffix = '_syn'
            dc = d.get(f'Q_DC{suffix}')
            qre = d.get(f'Q_H{harmonic}_re{suffix}')
            qim = d.get(f'Q_H{harmonic}_im{suffix}')
            piv = d.get('measurements_piv') or []
            if not any(m.get('tile_id') == int(tile_id) for m in piv):
                continue
            if dc is None or qre is None or qim is None:
                continue
            if not (_np.isfinite(dc) and _np.isfinite(qre)
                    and _np.isfinite(qim)):
                continue
            mean_Q = float(dc)
            q_h1 = complex(qre, qim)
            ff = d.get('flow_from'); ft = d.get('flow_to')
            f0_e = float(graph.graph.get(
                'synthetic_sim_meta', {}).get('f0_hz', 2.5))
        else:
            piv = d.get('measurements_piv') or []
            rec = None
            for m in piv:
                if m.get('tile_id') == int(tile_id):
                    rec = m
                    break
            if rec is None:
                continue
            qt = rec.get('quality_tier', 'D')
            if quality_order.get(str(qt).upper(), 0) < min_q:
                continue
            Q_t = rec.get('Q_t')
            f0_e = rec.get('f0_hz')
            ff = rec.get('flow_from')
            ft = rec.get('flow_to')
            if (Q_t is None or f0_e is None
                    or ff is None or ft is None):
                continue
            if not _np.isfinite(f0_e):
                continue
            dt = rec.get('frame_dt_s') or frame_dt_s_default
            h1 = _h1_from_qt(Q_t, f0_e, dt)
            if h1 is None:
                continue
            mean_Q, q_h1 = h1
        amp = abs(q_h1)
        if amp <= 0 or not _np.isfinite(amp):
            continue
        R_m, L_m = _edge_geometry(d, float(px_size_m))
        if R_m <= 0 or L_m <= 0:
            continue
        if mean_Q >= 0:
            us, ds = ff, ft
        else:
            us, ds = ft, ff
        edges.append({
            'u': u, 'v': v, 'R_m': R_m, 'L_m': L_m,
            'mean_Q': float(mean_Q), 'q_h1': q_h1,
            'L_over_R': L_m / R_m,
            'us': us, 'ds': ds, 'f0_hz': float(f0_e),
        })
    if len(edges) < min_edges:
        return None

    # ── Directed flow graph + max-reachable-subtree source ──
    D = _nx.DiGraph()
    for e in edges:
        D.add_edge(e['us'], e['ds'], weight=e['L_over_R'])
    candidates = [n for n in D.nodes() if D.in_degree(n) == 0]
    if not candidates:
        candidates = list(D.nodes())
    best = None
    for src in candidates:
        try:
            dl = _nx.single_source_dijkstra_path_length(
                D, src, weight='weight')
        except Exception:
            continue
        if best is None or len(dl) > best[1]:
            best = (src, len(dl), dl)
    if best is None:
        return None
    src_node, _, dist_node = best

    reachable = [e for e in edges if e['us'] in dist_node]
    if len(reachable) < min_edges:
        return None
    e_ref = max(reachable, key=lambda e: abs(e['mean_Q']))
    if e_ref['mean_Q'] == 0:
        return None
    sign_ref = 1.0 if e_ref['mean_Q'] >= 0 else -1.0
    phi_ref = float(_np.angle(sign_ref * e_ref['q_h1']))
    cum_LR_ref = dist_node[e_ref['us']] + 0.5 * e_ref['L_over_R']

    rows = []
    for e in edges:
        d_us = dist_node.get(e['us'])
        if d_us is None:
            e['reachable'] = False
            continue
        e['reachable'] = True
        x = (d_us + 0.5 * e['L_over_R']) - cum_LR_ref
        sign_e = 1.0 if e['mean_Q'] >= 0 else -1.0
        q_or = sign_e * e['q_h1']
        amp = abs(q_or)
        if amp == 0:
            continue
        y = float(_np.angle(q_or)) - phi_ref
        y -= 2.0 * _np.pi * round(y / (2.0 * _np.pi))
        e['cum_LR_rel'] = x; e['dphi'] = y; e['amp_or'] = amp
        rows.append({
            'cum_LR_rel': x, 'dphi': y, 'amp': amp,
            'R_um': e['R_m'] * 1e6, 'L_um': e['L_m'] * 1e6,
            'mean_Q': e['mean_Q'], 'u': e['u'], 'v': e['v'],
        })
    if len(rows) < min_edges:
        return None

    x_arr = _np.array([r['cum_LR_rel'] for r in rows])
    y_arr = _np.array([r['dphi'] for r in rows])
    amps_arr = _np.array([r['amp'] for r in rows])
    # Per-observation LS weight = amp² (inverse-variance optimal for
    # phase noise: σ_φ ∝ 1/|Q_H1|).  Matrix-scaling weight w = amp
    # (since lstsq squares it implicitly via Aw = A·w).
    w = amps_arr

    def _ls(y_use, weights):
        A = _np.column_stack([_np.ones(len(x_arr)), x_arr])
        Aw = A * weights[:, None]
        coefs, *_x = _np.linalg.lstsq(Aw, y_use * weights, rcond=None)
        return float(coefs[0]), float(coefs[1])

    def _se_slope(intercept, slope, y_use, w_matrix):
        """SE_slope from weighted-LS residuals.

        w_matrix is the weight used in matrix scaling (Aw = A · w),
        so the per-observation LS weight is w_matrix².
        Uses sandwich-ish formula: Var(slope) = σ² / Σ w² (x − x̄_w)²
        with σ² = Σ w² r² / (n − 2).
        """
        w2 = _np.asarray(w_matrix) ** 2
        r = y_use - (intercept + slope * x_arr)
        sw2 = float(_np.sum(w2))
        if sw2 == 0:
            return float('inf')
        xw = float(_np.sum(w2 * x_arr) / sw2)
        Sxx_w = float(_np.sum(w2 * (x_arr - xw) ** 2))
        if Sxx_w == 0:
            return float('inf')
        n = len(y_use)
        if n <= 2:
            return float('inf')
        sigma2 = float(_np.sum(w2 * r ** 2) / (n - 2))
        return float(_np.sqrt(sigma2 / Sxx_w))

    # LS baseline
    i_ls, s_ls = _ls(y_arr, w)
    se_ls = _se_slope(i_ls, s_ls, y_arr, w)

    # LS with π-disambig (3 iters, snap residual to nearest multiple of π)
    y_dis = y_arr.copy()
    i_dis, s_dis = i_ls, s_ls
    for _it in range(3):
        y_pred = i_dis + s_dis * x_arr
        r = y_dis - y_pred
        y_dis = y_pred + (r - _np.pi * _np.round(r / _np.pi))
        i_dis, s_dis = _ls(y_dis, w)
    se_dis = _se_slope(i_dis, s_dis, y_dis, w)

    # Huber IRLS
    i_hub, s_hub = i_ls, s_ls
    w_hub = w.copy()
    for _it in range(8):
        r = y_arr - (i_hub + s_hub * x_arr)
        sigma = (1.4826 * _np.median(_np.abs(r - _np.median(r)))
                  or 1e-12)
        c = 1.345 * sigma
        hub_w = _np.where(_np.abs(r) <= c, 1.0,
                           c / _np.maximum(_np.abs(r), 1e-12))
        w_hub = w * _np.sqrt(hub_w)
        ni, ns = _ls(y_arr, w_hub)
        if abs(ns - s_hub) < 1e-7 and abs(ni - i_hub) < 1e-7:
            i_hub, s_hub = ni, ns
            break
        i_hub, s_hub = ni, ns
    se_hub = _se_slope(i_hub, s_hub, y_arr, w_hub)

    def _rstd(intercept, slope, y_use):
        return float(_np.std(y_use - intercept - slope * x_arr))

    rstd_ls  = _rstd(i_ls,  s_ls,  y_arr)
    rstd_dis = _rstd(i_dis, s_dis, y_dis)
    rstd_hub = _rstd(i_hub, s_hub, y_arr)

    f0_hz = float(_np.median([e['f0_hz'] for e in edges]))
    omega = 2.0 * _np.pi * f0_hz * harmonic

    def _slope_to_D(slope):
        return (slope ** 2) / (8.0 * harmonic * omega * mu)

    def _se_D(slope, se_s):
        # Delta method: D = slope²/k → SE_D = |2·slope/k| · SE_slope
        k = 8.0 * harmonic * omega * mu
        if not _np.isfinite(se_s):
            return float('inf')
        return float(abs(2.0 * slope / k) * se_s)

    # Primary = disambig
    intercept, slope, resid_std = i_dis, s_dis, rstd_dis
    D_hat = _slope_to_D(slope)
    se_slope = se_dis
    se_D     = _se_D(slope, se_slope)

    # Build dataframe-like rows for export
    df_rows = []
    for k, r in enumerate(rows):
        df_rows.append(dict(
            cum_LR_rel=r['cum_LR_rel'],
            dphi=r['dphi'],
            dphi_corrected=float(y_dis[k]),
            amp=r['amp'],
            R_um=r['R_um'],
            L_um=r['L_um'],
            mean_Q=r['mean_Q'],
            u=int(r['u']), v=int(r['v']),
        ))

    return dict(
        D_hat=float(D_hat),
        slope=float(slope), intercept=float(intercept),
        SE_slope=float(se_slope), SE_D=float(se_D),
        residual_std=float(resid_std), f0_hz=float(f0_hz),
        n_edges_used=int(len(rows)),
        n_tile_edges=int(len(edges)),
        coverage_pct=float(100.0 * len(rows) / max(len(edges), 1)),
        source_node=int(src_node),
        e_ref=dict(
            u=int(e_ref['u']), v=int(e_ref['v']),
            mean_Q=float(e_ref['mean_Q']),
            q_h1=complex(e_ref['q_h1']),
            R_m=float(e_ref['R_m']), L_m=float(e_ref['L_m']),
            us=int(e_ref['us']), ds=int(e_ref['ds']),
        ),
        df=df_rows,
        fit_variants={
            'LS':            dict(slope=s_ls,  intercept=i_ls,
                                   SE_slope=se_ls,
                                   resid_std=rstd_ls,
                                   D_hat=_slope_to_D(s_ls),
                                   SE_D=_se_D(s_ls, se_ls)),
            'LS+π-disambig': dict(slope=s_dis, intercept=i_dis,
                                   SE_slope=se_dis,
                                   resid_std=rstd_dis,
                                   D_hat=_slope_to_D(s_dis),
                                   SE_D=_se_D(s_dis, se_dis)),
            'Huber':         dict(slope=s_hub, intercept=i_hub,
                                   SE_slope=se_hub,
                                   resid_std=rstd_hub,
                                   D_hat=_slope_to_D(s_hub),
                                   SE_D=_se_D(s_hub, se_hub)),
        },
        edge_records=[
            dict(u=int(e['u']), v=int(e['v']),
                 us=int(e['us']), ds=int(e['ds']),
                 R_m=e['R_m'], L_m=e['L_m'],
                 mean_Q=float(e['mean_Q']),
                 q_h1=complex(e['q_h1']),
                 reachable=bool(e.get('reachable', False)),
                 cum_LR_rel=float(e.get('cum_LR_rel', _np.nan))
                            if e.get('reachable') else float('nan'),
                 dphi=float(e.get('dphi', _np.nan))
                       if e.get('reachable') else float('nan'),
                 )
            for e in edges
        ],
    )


# =====================================================================
# Multi-subtree (fixed-effects) phase-gradient D̂
# =====================================================================
# Extends per_tile_D_flow_dijkstra to use ALL DAG-roots (not just the
# largest), pooling their reachable subtrees into a single fit with
# per-subtree intercepts and a shared slope.
#
# Each subtree has its own phase reference (the intercept a_k absorbs
# the arbitrary phase shift inherent to that subtree's data) but the
# physics says the slope is the same everywhere:
#     y_{i,k} = a_k + slope · cum_LR_{i,k} + ε
# Solved as a single weighted-LS regression with K+1 parameters.
#
# Coverage typically rises from ~30% (single-subtree) to ~70-90%
# (multi-subtree) and the SE on slope tightens.  Each edge is assigned
# to the DAG-root whose Dijkstra distance is smallest, so subtrees
# don't overlap.
# =====================================================================

def per_tile_D_multi_subtree(
    graph,
    tile_id: int,
    *,
    source: str = 'piv',
    mu: float = 2.5e-3,
    px_size_m: Optional[float] = None,
    harmonic: int = 1,
    min_quality: str = 'C',
    min_edges: int = 20,
    min_subtree_edges: int = 5,
    frame_dt_s_default: float = 1.0 / 250.0,
) -> Optional[dict]:
    """Multi-subtree fixed-effects phase-gradient D̂.

    Returns dict with the same keys as per_tile_D_flow_dijkstra plus:
        n_subtrees      — number of subtrees with ≥ min_subtree_edges
        intercepts      — dict {root_node: a_k}
        edge_records    — each record now has 'root' (subtree assignment)
        df              — per-edge rows with 'subtree_root' label
    """
    import numpy as _np
    import networkx as _nx

    if px_size_m is None:
        try:
            from .config import get_px_size_m as _gp
            px_size_m = float(_gp())
        except Exception:
            px_size_m = 1.7e-6

    quality_order = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    min_q = quality_order.get(str(min_quality).upper(), 1)

    def _h1_from_qt(Q_t, f0, dt):
        Q = _np.asarray(Q_t, dtype=float)
        N = len(Q)
        if N < 30:
            return None
        t = _np.arange(N) * float(dt)
        mean = float(_np.mean(Q))
        Q_dm = Q - mean
        omega = 2.0 * _np.pi * float(f0) * harmonic
        c = (2.0 / N) * _np.sum(Q_dm * _np.exp(-1j * omega * t))
        return mean, c

    # ── Collect tile edges (same as single-subtree variant) ──
    edges = []
    for u, v, d in graph.edges(data=True):
        if source == 'syn':
            suffix = '_syn'
            dc = d.get(f'Q_DC{suffix}')
            qre = d.get(f'Q_H{harmonic}_re{suffix}')
            qim = d.get(f'Q_H{harmonic}_im{suffix}')
            piv = d.get('measurements_piv') or []
            if not any(m.get('tile_id') == int(tile_id) for m in piv):
                continue
            if dc is None or qre is None or qim is None:
                continue
            if not (_np.isfinite(dc) and _np.isfinite(qre)
                    and _np.isfinite(qim)):
                continue
            mean_Q = float(dc)
            q_h1 = complex(qre, qim)
            ff = d.get('flow_from'); ft = d.get('flow_to')
            f0_e = float(graph.graph.get(
                'synthetic_sim_meta', {}).get('f0_hz', 2.5))
        else:
            piv = d.get('measurements_piv') or []
            rec = None
            for m in piv:
                if m.get('tile_id') == int(tile_id):
                    rec = m
                    break
            if rec is None:
                continue
            qt = rec.get('quality_tier', 'D')
            if quality_order.get(str(qt).upper(), 0) < min_q:
                continue
            Q_t = rec.get('Q_t')
            f0_e = rec.get('f0_hz')
            ff = rec.get('flow_from')
            ft = rec.get('flow_to')
            if (Q_t is None or f0_e is None
                    or ff is None or ft is None):
                continue
            if not _np.isfinite(f0_e):
                continue
            dt = rec.get('frame_dt_s') or frame_dt_s_default
            h1 = _h1_from_qt(Q_t, f0_e, dt)
            if h1 is None:
                continue
            mean_Q, q_h1 = h1
        amp = abs(q_h1)
        if amp <= 0 or not _np.isfinite(amp):
            continue
        R_m, L_m = _edge_geometry(d, float(px_size_m))
        if R_m <= 0 or L_m <= 0:
            continue
        us, ds = (ff, ft) if mean_Q >= 0 else (ft, ff)
        edges.append({
            'u': u, 'v': v, 'R_m': R_m, 'L_m': L_m,
            'mean_Q': float(mean_Q), 'q_h1': q_h1,
            'L_over_R': L_m / R_m,
            'us': us, 'ds': ds, 'f0_hz': float(f0_e),
        })
    if len(edges) < min_edges:
        return None

    # ── Directed flow graph + all DAG-roots ──
    D = _nx.DiGraph()
    for e in edges:
        D.add_edge(e['us'], e['ds'], weight=e['L_over_R'])
    dag_roots = [n for n in D.nodes() if D.in_degree(n) == 0]
    if not dag_roots:
        # No DAG roots (cycles): treat every node as a candidate source
        dag_roots = list(D.nodes())

    # Dijkstra from each root
    dists_per_root = {}
    for root in dag_roots:
        try:
            dl = _nx.single_source_dijkstra_path_length(
                D, root, weight='weight')
            if len(dl) > 0:
                dists_per_root[root] = dl
        except (_nx.NodeNotFound, _nx.NetworkXNoPath):
            continue
    if not dists_per_root:
        return None

    # Assign each edge to nearest-root subtree (by L/R distance to its
    # upstream endpoint).  Edges whose upstream node is unreachable
    # from any root are dropped.
    for e in edges:
        best = None
        for root, dd in dists_per_root.items():
            d_us = dd.get(e['us'])
            if d_us is None:
                continue
            if best is None or d_us < best[1]:
                best = (root, d_us)
        if best is None:
            e['root'] = None
            e['cum_LR_rel'] = float('nan')
            e['reachable'] = False
        else:
            root, d_us = best
            e['root'] = int(root)
            e['cum_LR_rel'] = float(d_us + 0.5 * e['L_over_R'])
            e['reachable'] = True

    reachable = [e for e in edges if e['reachable']]
    if len(reachable) < min_edges:
        return None

    # Drop subtrees with too few edges (won't anchor an intercept)
    from collections import Counter
    counts = Counter(e['root'] for e in reachable)
    valid_roots = [r for r, c in counts.items() if c >= min_subtree_edges]
    if not valid_roots:
        return None
    valid_eds = [e for e in reachable if e['root'] in valid_roots]
    if len(valid_eds) < min_edges:
        return None

    # ── Build design matrix M and response y ──
    K = len(valid_roots)
    root_idx = {r: i for i, r in enumerate(valid_roots)}
    n = len(valid_eds)

    M = _np.zeros((n, K + 1))
    x_arr = _np.empty(n)
    y_raw = _np.empty(n)
    amps_arr = _np.empty(n)
    for i, e in enumerate(valid_eds):
        M[i, root_idx[e['root']]] = 1.0
        M[i, K] = e['cum_LR_rel']
        x_arr[i] = e['cum_LR_rel']
        sign_e = 1.0 if e['mean_Q'] >= 0 else -1.0
        q_or = sign_e * e['q_h1']
        y_raw[i] = float(_np.angle(q_or))
        amps_arr[i] = abs(q_or)
    w = amps_arr  # matrix-scaling weight; per-obs LS weight = amp²

    def _fit(y_use):
        Mw = M * w[:, None]
        coefs, *_x = _np.linalg.lstsq(Mw, y_use * w, rcond=None)
        intercepts = coefs[:K]
        slope = float(coefs[K])
        return intercepts, slope

    def _predict(intercepts, slope):
        return M @ _np.concatenate([intercepts, [slope]])

    def _se_slope(intercepts, slope, y_use, w_use):
        """SE of slope from weighted-LS sandwich approx."""
        Mw = M * w_use[:, None]
        # Cov(β) ≈ σ² (M^T W^2 M)^{-1}; take (K, K) element for slope.
        A = Mw.T @ Mw
        r = y_use - _predict(intercepts, slope)
        dof = max(n - K - 1, 1)
        sigma2 = float(_np.sum(w_use ** 2 * r ** 2) / dof)
        try:
            cov = _np.linalg.pinv(A) * sigma2
            return float(_np.sqrt(max(cov[K, K], 0.0)))
        except _np.linalg.LinAlgError:
            return float('inf')

    # Initial wrap and LS
    y0 = y_raw - 2.0 * _np.pi * _np.round(y_raw / (2.0 * _np.pi))
    int_ls, s_ls = _fit(y0)
    se_ls = _se_slope(int_ls, s_ls, y0, w)

    # LS with π-disambig (3 iters; snap residual to nearest multiple of π)
    y_dis = y0.copy()
    int_dis, s_dis = int_ls, s_ls
    for _it in range(3):
        y_pred = _predict(int_dis, s_dis)
        r = y_dis - y_pred
        y_dis = y_pred + (r - _np.pi * _np.round(r / _np.pi))
        int_dis, s_dis = _fit(y_dis)
    se_dis = _se_slope(int_dis, s_dis, y_dis, w)

    # Huber IRLS
    int_hub, s_hub = int_ls, s_ls
    w_hub = w.copy()
    for _it in range(8):
        r = y0 - _predict(int_hub, s_hub)
        sigma = (1.4826 * _np.median(_np.abs(r - _np.median(r)))
                  or 1e-12)
        c = 1.345 * sigma
        hub_w = _np.where(_np.abs(r) <= c, 1.0,
                           c / _np.maximum(_np.abs(r), 1e-12))
        w_hub = w * _np.sqrt(hub_w)
        ni, ns = _fit(y0)  # not quite right: should use w_hub
        # Re-fit with Huber weights
        Mw_h = M * w_hub[:, None]
        coefs_h, *_x = _np.linalg.lstsq(Mw_h, y0 * w_hub, rcond=None)
        ni = coefs_h[:K]; ns = float(coefs_h[K])
        if abs(ns - s_hub) < 1e-7 and float(_np.linalg.norm(ni - int_hub)) < 1e-7:
            int_hub, s_hub = ni, ns
            break
        int_hub, s_hub = ni, ns
    se_hub = _se_slope(int_hub, s_hub, y0, w_hub)

    def _rstd(intercepts, slope, y_use):
        return float(_np.std(y_use - _predict(intercepts, slope)))

    rstd_ls  = _rstd(int_ls,  s_ls,  y0)
    rstd_dis = _rstd(int_dis, s_dis, y_dis)
    rstd_hub = _rstd(int_hub, s_hub, y0)

    f0_hz = float(_np.median([e['f0_hz'] for e in edges]))
    omega = 2.0 * _np.pi * f0_hz * harmonic

    def _slope_to_D(slope):
        return (slope ** 2) / (8.0 * harmonic * omega * mu)

    def _se_D(slope, se_s):
        k = 8.0 * harmonic * omega * mu
        if not _np.isfinite(se_s):
            return float('inf')
        return float(abs(2.0 * slope / k) * se_s)

    # Primary = disambig
    intercepts, slope, resid_std = int_dis, s_dis, rstd_dis
    D_hat = _slope_to_D(slope)
    se_slope_p = se_dis
    se_D_p     = _se_D(slope, se_slope_p)

    df_rows = []
    for i, e in enumerate(valid_eds):
        df_rows.append(dict(
            cum_LR_rel=float(x_arr[i]),
            dphi=float(y0[i]),
            dphi_corrected=float(y_dis[i]),
            amp=float(amps_arr[i]),
            R_um=float(e['R_m'] * 1e6),
            L_um=float(e['L_m'] * 1e6),
            mean_Q=float(e['mean_Q']),
            u=int(e['u']), v=int(e['v']),
            subtree_root=int(e['root']),
        ))

    return dict(
        D_hat=float(D_hat),
        slope=float(slope), intercept=float(0.0),  # intercept is per-subtree
        SE_slope=float(se_slope_p), SE_D=float(se_D_p),
        residual_std=float(resid_std), f0_hz=float(f0_hz),
        n_edges_used=int(n), n_tile_edges=int(len(edges)),
        coverage_pct=float(100.0 * n / max(len(edges), 1)),
        n_subtrees=int(K),
        intercepts={int(r): float(int_dis[i]) for r, i in root_idx.items()},
        df=df_rows,
        fit_variants={
            'LS':            dict(slope=s_ls,  intercepts=dict(zip(
                                       [int(r) for r in valid_roots],
                                       [float(v) for v in int_ls])),
                                   SE_slope=float(se_ls),
                                   resid_std=float(rstd_ls),
                                   D_hat=_slope_to_D(s_ls),
                                   SE_D=_se_D(s_ls, se_ls)),
            'LS+π-disambig': dict(slope=s_dis, intercepts=dict(zip(
                                       [int(r) for r in valid_roots],
                                       [float(v) for v in int_dis])),
                                   SE_slope=float(se_dis),
                                   resid_std=float(rstd_dis),
                                   D_hat=_slope_to_D(s_dis),
                                   SE_D=_se_D(s_dis, se_dis)),
            'Huber':         dict(slope=s_hub, intercepts=dict(zip(
                                       [int(r) for r in valid_roots],
                                       [float(v) for v in int_hub])),
                                   SE_slope=float(se_hub),
                                   resid_std=float(rstd_hub),
                                   D_hat=_slope_to_D(s_hub),
                                   SE_D=_se_D(s_hub, se_hub)),
        },
        edge_records=[
            dict(u=int(e['u']), v=int(e['v']),
                 us=int(e['us']), ds=int(e['ds']),
                 R_m=e['R_m'], L_m=e['L_m'],
                 mean_Q=float(e['mean_Q']),
                 q_h1=complex(e['q_h1']),
                 reachable=bool(e.get('reachable', False)),
                 root=int(e['root']) if e.get('root') is not None else -1,
                 cum_LR_rel=float(e.get('cum_LR_rel', _np.nan))
                            if e.get('reachable') else float('nan'),
                 )
            for e in edges
        ],
    )
