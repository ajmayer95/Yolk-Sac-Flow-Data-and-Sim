"""
Update flow measurement uncertainties using MCMC-inferred heteroscedastic model.

After running Bayesian MCMC, use the inferred noise parameters to correct
the stored sigma_Q values based on empirical self-consistency.
"""

import numpy as np
import networkx as nx


def update_sigma_Q_from_mcmc(
    G: nx.Graph,
    sigma_base: float,
    sigma_rel: float,
    flow_attr: str = 'mean_Q',
    sigma_attr: str = 'sigma_Q',
    verbose: bool = True,
):
    """
    Update edge uncertainties using MCMC heteroscedastic model.

    The MCMC infers: σ_i = σ_base + σ_rel × |Q_i|

    This represents the "true" measurement noise inferred from network self-consistency,
    as opposed to the original uncertainty estimates from individual vessel analysis.

    Args:
        G: NetworkX graph with flow measurements
        sigma_base: MCMC-inferred base uncertainty (nL/s)
        sigma_rel: MCMC-inferred flow-proportional factor (dimensionless)
        flow_attr: Edge attribute for flow (default 'mean_Q')
        sigma_attr: Edge attribute for uncertainty (default 'sigma_Q')
        verbose: Print statistics

    Updates:
        For each edge with flow measurement:
        - Adds 'sigma_Q_original': original uncertainty from flow analysis
        - Adds 'sigma_Q_mcmc': MCMC-calibrated uncertainty
        - Updates 'sigma_Q' to MCMC-calibrated value
        - Adds 'sigma_Q_correction_factor': ratio of new/old
    """

    n_updated = 0
    correction_factors = []

    # Detect alternative attribute names
    alt_flow_attrs = ['mean_Q_nL_s', 'Q_mean', 'flow_nL_s']
    alt_sigma_attrs = ['sigma_Q_nL_s', 'Q_std', 'flow_std']

    for u, v, data in G.edges(data=True):
        # Try to find flow value
        Q = data.get(flow_attr)
        if Q is None:
            for alt in alt_flow_attrs:
                Q = data.get(alt)
                if Q is not None:
                    break

        if Q is None:
            continue

        # Try to find original sigma
        sigma_original = data.get(sigma_attr)
        if sigma_original is None:
            for alt in alt_sigma_attrs:
                sigma_original = data.get(alt)
                if sigma_original is not None:
                    break

        if sigma_original is None or sigma_original == 0:
            # No original uncertainty - skip
            continue

        # Compute MCMC-calibrated uncertainty
        sigma_mcmc = sigma_base + sigma_rel * abs(Q)

        # Store all versions
        data['sigma_Q_original'] = sigma_original
        data['sigma_Q_mcmc'] = sigma_mcmc
        data['sigma_Q'] = sigma_mcmc  # Update default to calibrated

        # Also update alternative names if they exist
        if 'sigma_Q_nL_s' in data:
            data['sigma_Q_nL_s'] = sigma_mcmc

        # Store correction factor for diagnostics
        correction = sigma_mcmc / sigma_original if sigma_original > 0 else 1.0
        data['sigma_Q_correction_factor'] = correction
        correction_factors.append(correction)

        n_updated += 1

    # Store MCMC parameters in graph metadata
    G.graph['mcmc_sigma_base'] = sigma_base
    G.graph['mcmc_sigma_rel'] = sigma_rel
    G.graph['sigma_Q_calibration_applied'] = True

    if verbose:
        print(f"\nUpdated {n_updated} edge uncertainties using MCMC heteroscedastic model:")
        print(f"  σ_Q = {sigma_base:.2f} + {sigma_rel:.3f} × |Q| nL/s")

        if correction_factors:
            cf = np.array(correction_factors)
            print(f"\nCorrection factor statistics:")
            print(f"  Mean:   {cf.mean():.2f}×")
            print(f"  Median: {np.median(cf):.2f}×")
            print(f"  Range:  [{cf.min():.2f}, {cf.max():.2f}]×")

            # Histogram
            n_decreased = np.sum(cf < 0.9)
            n_similar = np.sum((cf >= 0.9) & (cf <= 1.1))
            n_increased = np.sum(cf > 1.1)

            print(f"\nCorrection distribution:")
            print(f"  Decreased (< 0.9×): {n_decreased} edges")
            print(f"  Similar (0.9-1.1×):  {n_similar} edges")
            print(f"  Increased (> 1.1×): {n_increased} edges")

    return n_updated


def plot_sigma_Q_comparison(
    G: nx.Graph,
    flow_attr: str = 'mean_Q',
    show: bool = True,
    save_path: str = None,
):
    """
    Plot comparison of original vs MCMC-calibrated uncertainties.
    """
    import matplotlib.pyplot as plt

    # Collect data
    Q_vals = []
    sigma_original = []
    sigma_mcmc = []

    for u, v, data in G.edges(data=True):
        Q = data.get(flow_attr)
        sigma_orig = data.get('sigma_Q_original')
        sigma_cal = data.get('sigma_Q_mcmc')

        if Q is not None and sigma_orig is not None and sigma_cal is not None:
            Q_vals.append(abs(Q))
            sigma_original.append(sigma_orig)
            sigma_mcmc.append(sigma_cal)

    if len(Q_vals) == 0:
        print("No data to plot - run update_sigma_Q_from_mcmc() first")
        return None

    Q_vals = np.array(Q_vals)
    sigma_original = np.array(sigma_original)
    sigma_mcmc = np.array(sigma_mcmc)

    # Get MCMC parameters
    sigma_base = G.graph.get('mcmc_sigma_base', 0)
    sigma_rel = G.graph.get('mcmc_sigma_rel', 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter: Original vs Calibrated
    ax1 = axes[0, 0]
    ax1.scatter(Q_vals, sigma_original, alpha=0.5, s=20, label='Original σ_Q', color='blue')
    ax1.scatter(Q_vals, sigma_mcmc, alpha=0.5, s=20, label='MCMC-calibrated', color='red')

    # Plot MCMC model line
    Q_line = np.linspace(0, Q_vals.max(), 100)
    sigma_line = sigma_base + sigma_rel * Q_line
    ax1.plot(Q_line, sigma_line, 'k--', linewidth=2,
             label=f'MCMC: σ = {sigma_base:.2f} + {sigma_rel:.3f}|Q|')

    ax1.set_xlabel('|Q| (nL/s)')
    ax1.set_ylabel('σ_Q (nL/s)')
    ax1.legend()
    ax1.set_title('Original vs MCMC-Calibrated Uncertainties')
    ax1.grid(alpha=0.3)

    # 2. Correction factor vs Q
    ax2 = axes[0, 1]
    correction = sigma_mcmc / sigma_original
    ax2.scatter(Q_vals, correction, alpha=0.5, s=20)
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=1, label='No correction')
    ax2.axhline(correction.mean(), color='r', linestyle='-', linewidth=2,
                label=f'Mean = {correction.mean():.2f}×')
    ax2.set_xlabel('|Q| (nL/s)')
    ax2.set_ylabel('Correction Factor (σ_mcmc / σ_original)')
    ax2.legend()
    ax2.set_title('Correction Factor vs Flow')
    ax2.grid(alpha=0.3)

    # 3. Histogram of corrections
    ax3 = axes[1, 0]
    ax3.hist(correction, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(1.0, color='k', linestyle='--', linewidth=2, label='No correction')
    ax3.axvline(correction.mean(), color='r', linestyle='-', linewidth=2,
                label=f'Mean = {correction.mean():.2f}×')
    ax3.set_xlabel('Correction Factor')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.set_title('Distribution of Correction Factors')

    # 4. Relative uncertainty comparison
    ax4 = axes[1, 1]
    rel_unc_original = sigma_original / Q_vals
    rel_unc_mcmc = sigma_mcmc / Q_vals

    ax4.scatter(Q_vals, rel_unc_original * 100, alpha=0.5, s=20,
                label='Original', color='blue')
    ax4.scatter(Q_vals, rel_unc_mcmc * 100, alpha=0.5, s=20,
                label='MCMC-calibrated', color='red')
    ax4.set_xlabel('|Q| (nL/s)')
    ax4.set_ylabel('Relative Uncertainty (%)')
    ax4.legend()
    ax4.set_title('Relative Uncertainty vs Flow')
    ax4.grid(alpha=0.3)
    ax4.set_ylim(0, 100)

    plt.suptitle(f'Uncertainty Calibration from MCMC (σ = {sigma_base:.2f} + {sigma_rel:.3f}|Q| nL/s)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig


# Example usage:
if __name__ == '__main__':
    import pickle

    # Load your graph
    graph_path = '/path/to/your/graph.gpickle'
    G = nx.read_gpickle(graph_path)

    # After running MCMC and getting result:
    # result = bayesian_pressure_mcmc(G, heteroscedastic=True, ...)

    # Use the inferred parameters
    sigma_base = 0.79  # From your MCMC result
    sigma_rel = 0.34

    # Update uncertainties
    update_sigma_Q_from_mcmc(
        G,
        sigma_base=sigma_base,
        sigma_rel=sigma_rel,
        verbose=True
    )

    # Plot comparison
    plot_sigma_Q_comparison(G, show=True, save_path='sigma_Q_calibration.png')

    # Save updated graph
    nx.write_gpickle(G, graph_path.replace('.gpickle', '_calibrated.gpickle'))
    print(f"\nSaved calibrated graph")
