"""
granger.py
==========
Granger causality analysis with F-test for directed edge validation.

Method:
  For each directed pair (j -> i), fit two MVAR(order) models:
  1. Full model: X_i[t] = sum_k A_ik @ X_k[t-lag] + noise (all nodes)
  2. Restricted model: X_i[t] = sum_{k!=j} A_ik @ X_k[t-lag] + noise (exclude j)

  F-statistic: F = ((RSS_r - RSS_f) / order) / (RSS_f / df2)
    where df2 = T - order*N - 1 (residual degrees of freedom)

  G[i,j] = 1 if p_value(F) < alpha, else 0 (directed: j causes i)
  G diagonal = 0 (no self-causality tested)

Why Granger causality:
  - Pearson correlation is symmetric by construction (FC = FC.T)
  - Granger causality is inherently asymmetric (G != G.T generally)
  - This asymmetry proves FC cannot recover directed connectivity

Known limitations:
  - Assumes stationarity of timeseries
  - Requires T >> order*N for reliable estimation
  - Does not account for latent confounders
  - VAR(1) may miss higher-order temporal dependencies

Parameters:
  X: (T, N) timeseries matrix
  order: MVAR model order (default=1)
  alpha: significance threshold (default=0.05)

Returns:
  G: binary (N,N) causality matrix
  F_matrix: raw F-statistics (N,N)
  p_matrix: p-values (N,N)
"""

import numpy as np
from scipy import stats
import warnings


def _fit_mvar_ols(X, order, exclude_node=None):
    """
    Fit MVAR(order) model via OLS, optionally excluding one node from predictors.

    Parameters
    ----------
    X : ndarray (T, N)
        Timeseries matrix
    order : int
        Model order (number of lags)
    exclude_node : int or None
        If not None, exclude this node from predictors

    Returns
    -------
    RSS : ndarray (N,)
        Residual sum of squares for each target node
    """
    T, N = X.shape

    # Build lagged design matrix
    # X_lagged[t] = [X[t-1], X[t-2], ..., X[t-order]] flattened
    effective_T = T - order

    if exclude_node is None:
        n_predictors = N * order
    else:
        n_predictors = (N - 1) * order

    X_design = np.zeros((effective_T, n_predictors))
    X_target = X[order:]  # (effective_T, N)

    for t in range(order, T):
        idx = 0
        for lag in range(1, order + 1):
            for node in range(N):
                if exclude_node is not None and node == exclude_node:
                    continue
                X_design[t - order, idx] = X[t - lag, node]
                idx += 1

    # Fit OLS for each target node
    RSS = np.zeros(N)
    for i in range(N):
        y = X_target[:, i]  # (effective_T,)
        # OLS: beta = (X'X)^{-1} X'y, residuals = y - X @ beta
        # Use lstsq for numerical stability
        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
        if len(residuals) > 0:
            RSS[i] = residuals[0]
        else:
            # If underdetermined, compute RSS manually
            y_pred = X_design @ beta
            RSS[i] = np.sum((y - y_pred) ** 2)

    return RSS


def granger_causality_matrix(X, order=1, alpha=0.05):
    """
    Compute Granger causality matrix with F-test significance.

    For each directed pair (j -> i), tests whether including lagged values
    of node j significantly improves prediction of node i.

    Parameters
    ----------
    X : ndarray (T, N)
        Timeseries matrix with T timepoints and N nodes
    order : int, default=1
        MVAR model order (number of lags to include)
    alpha : float, default=0.05
        Significance threshold for Granger causality

    Returns
    -------
    G : ndarray (N, N), dtype=int
        Binary causality matrix. G[i,j] = 1 if j Granger-causes i
    F_matrix : ndarray (N, N)
        Raw F-statistics for each directed pair
    p_matrix : ndarray (N, N)
        P-values for each directed pair

    Notes
    -----
    - G is asymmetric in general (this is the key property)
    - Diagonal is always 0 (no self-causality tested)
    - F-test: F = ((RSS_r - RSS_f) / order) / (RSS_f / df2)
      where df2 = T - order*N - 1
    - Uses scipy.stats.f.sf for p-value computation
    """
    T, N = X.shape
    df2 = T - order * N - 1  # residual degrees of freedom

    if df2 <= 0:
        raise ValueError(
            f"Timeseries too short for MVAR({order}): "
            f"T={T}, N={N}, df2={df2}. Need T > order*N + 1."
        )

    G = np.zeros((N, N), dtype=int)
    F_matrix = np.zeros((N, N))
    p_matrix = np.zeros((N, N))

    # Test each directed pair (j -> i)
    for i in range(N):  # target node
        for j in range(N):  # potential cause
            if i == j:
                # No self-causality
                G[i, j] = 0
                F_matrix[i, j] = 0.0
                p_matrix[i, j] = 1.0
                continue

            # Full model: all nodes predict i
            # Build design matrix for full model
            effective_T = T - order
            X_design_full = np.zeros((effective_T, N * order))
            X_target = X[order:, i]

            for t in range(order, T):
                for lag in range(1, order + 1):
                    for node in range(N):
                        X_design_full[t - order, lag_idx(N, lag - 1, node)] = X[t - lag, node]

            # Restricted model: all nodes except j predict i
            X_design_restricted = np.zeros((effective_T, (N - 1) * order))
            for t in range(order, T):
                idx = 0
                for lag in range(1, order + 1):
                    for node in range(N):
                        if node == j:
                            continue
                        X_design_restricted[t - order, idx] = X[t - lag, node]
                        idx += 1

            # Fit both models
            _, res_full, _, _ = np.linalg.lstsq(X_design_full, X_target, rcond=None)
            _, res_restricted, _, _ = np.linalg.lstsq(X_design_restricted, X_target, rcond=None)

            RSS_full = res_full[0] if len(res_full) > 0 else np.sum((X_target - X_design_full @ np.linalg.lstsq(X_design_full, X_target, rcond=None)[0]) ** 2)
            RSS_restricted = res_restricted[0] if len(res_restricted) > 0 else np.sum((X_target - X_design_restricted @ np.linalg.lstsq(X_design_restricted, X_target, rcond=None)[0]) ** 2)

            # F-statistic
            # F = ((RSS_r - RSS_f) / df1) / (RSS_f / df2)
            # df1 = order (number of restrictions = coefficients for node j)
            df1 = order

            if RSS_full <= 0:
                # Perfect fit - shouldn't happen with real data
                F_stat = np.inf
            else:
                F_stat = ((RSS_restricted - RSS_full) / df1) / (RSS_full / df2)

            # P-value (one-tailed: larger F = more evidence against null)
            p_val = stats.f.sf(F_stat, df1, df2)

            F_matrix[i, j] = F_stat
            p_matrix[i, j] = p_val

            if p_val < alpha:
                G[i, j] = 1
            else:
                G[i, j] = 0

    # Ensure diagonal is zero
    np.fill_diagonal(G, 0)

    return G, F_matrix, p_matrix


def lag_idx(N, lag_idx_zero, node):
    """Helper to compute index in flattened lag design matrix."""
    return lag_idx_zero * N + node


def causality_vs_correlation_summary(A_var1, X, order=1, alpha=0.05):
    """
    Compare VAR(1) effective connectivity against Granger causality and FC.

    This function demonstrates the core methodological argument:
    - FC (Pearson correlation) is symmetric by construction
    - VAR(1) A matrix is asymmetric (captures directed relationships)
    - Granger causality G matrix is also asymmetric
    - High FC with no Granger edge = spurious correlation
    - Significant Granger with low FC = hidden causal edge

    Parameters
    ----------
    A_var1 : ndarray (N, N)
        Directed effective connectivity from VAR(1) OLS estimation
    X : ndarray (T, N)
        Timeseries matrix used for Granger causality computation
    order : int, default=1
        MVAR order for Granger causality
    alpha : float, default=0.05
        Significance threshold for Granger test

    Returns
    -------
    summary : dict
        Dictionary with keys:
        - 'n_spurious': int, count of FC[i,j] > 0.3 AND G[i,j] == 0
        - 'n_hidden': int, count of G[i,j] == 1 AND |FC[i,j]| < 0.1
        - 'fc_asymmetry': float, mean(|FC - FC.T|), should be ~0
        - 'var1_asymmetry': float, mean(|A_var1 - A_var1.T|)
        - 'G': ndarray (N,N), Granger causality matrix
        - 'F_matrix': ndarray (N,N), F-statistics
        - 'p_matrix': ndarray (N,N), p-values
        - 'FC': ndarray (N,N), functional connectivity (Pearson)

    Notes
    -----
    - FC diagonal is set to 0 for fair comparison
    - Spurious correlations indicate indirect effects or common drivers
    - Hidden edges indicate causal relationships masked by low correlation
    """
    N = A_var1.shape[0]

    # Compute FC (Pearson correlation)
    FC = np.corrcoef(X.T)  # (N, N)
    np.fill_diagonal(FC, 0)  # no self-correlation for comparison

    # Compute Granger causality
    G, F_matrix, p_matrix = granger_causality_matrix(X, order=order, alpha=alpha)

    # Count spurious correlations: high FC, no causal edge
    # Check upper triangle only to avoid double counting
    spurious_mask = (np.abs(FC) > 0.3) & (G == 1)  # G[i,j]=1 means j->i exists
    # Actually: spurious = FC high but NO Granger edge
    spurious_mask = (np.abs(FC) > 0.3) & (G == 0)
    n_spurious = int(np.sum(spurious_mask) / 2)  # divide by 2 for symmetric counting

    # Count hidden causal edges: Granger significant, low FC
    hidden_mask = (G == 1) & (np.abs(FC) < 0.1)
    n_hidden = int(np.sum(hidden_mask))  # don't divide - G is asymmetric

    # Asymmetry measures
    fc_asymmetry = float(np.mean(np.abs(FC - FC.T)))  # should be ~0
    var1_asymmetry = float(np.mean(np.abs(A_var1 - A_var1.T)))

    return {
        'n_spurious': n_spurious,
        'n_hidden': n_hidden,
        'fc_asymmetry': fc_asymmetry,
        'var1_asymmetry': var1_asymmetry,
        'G': G,
        'F_matrix': F_matrix,
        'p_matrix': p_matrix,
        'FC': FC,
    }


def generate_random_stable_system(N=10, rho=0.7, T=300, seed=None):
    """
    Generate a random stable VAR(1) system for testing.

    Parameters
    ----------
    N : int, default=10
        Number of nodes
    rho : float, default=0.7
        Target spectral radius (must be < 1 for stability)
    T : int, default=300
        Number of timepoints
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    A : ndarray (N, N)
        Stable connectivity matrix
    X : ndarray (T, N)
        Timeseries from VAR(1) process
    """
    if seed is not None:
        np.random.seed(seed)

    # Random asymmetric connectivity
    A = np.random.randn(N, N) * 0.3
    np.fill_diagonal(A, 0)

    # Scale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    current_rho = np.max(np.abs(eigvals))
    if current_rho > 1e-8:
        A = A * (rho / current_rho)

    # Generate timeseries
    X = np.zeros((T, N))
    X[0] = np.random.randn(N) * 0.1

    for t in range(1, T):
        X[t] = A @ X[t-1] + np.random.randn(N) * 0.1

    return A, X


if __name__ == '__main__':
    """
    Demonstration: FC is always symmetric, A_var1 is always asymmetric,
    and spurious/hidden edges exist.
    """
    print("=" * 60)
    print("  Granger Causality vs Functional Connectivity")
    print("=" * 60)

    np.random.seed(42)

    n_systems = 5
    N = 10
    T = 300
    rho = 0.7

    print(f"\nTesting on {n_systems} random stable systems (N={N}, T={T}, rho={rho})\n")

    fc_asymmetries = []
    var1_asymmetries = []
    total_spurious = 0
    total_hidden = 0

    for i in range(n_systems):
        # Generate random stable system
        A_true, X = generate_random_stable_system(N=N, rho=rho, T=T, seed=i)

        # Fit VAR(1) to recover A
        X_past = X[:-1]
        X_future = X[1:]
        A_est_T, _, _, _ = np.linalg.lstsq(X_past, X_future, rcond=None)
        A_var1 = A_est_T.T
        np.fill_diagonal(A_var1, 0)

        # Run causality vs correlation summary
        summary = causality_vs_correlation_summary(A_var1, X, order=1, alpha=0.05)

        fc_asymmetries.append(summary['fc_asymmetry'])
        var1_asymmetries.append(summary['var1_asymmetry'])
        total_spurious += summary['n_spurious']
        total_hidden += summary['n_hidden']

        print(f"System {i+1}:")
        print(f"  FC asymmetry: {summary['fc_asymmetry']:.2e} (should be ~0)")
        print(f"  VAR(1) asymmetry: {summary['var1_asymmetry']:.4f} (should be > 0)")
        print(f"  Spurious correlations: {summary['n_spurious']}")
        print(f"  Hidden causal edges: {summary['n_hidden']}")
        print()

    print("-" * 60)
    print("SUMMARY:")
    print(f"  Mean FC asymmetry: {np.mean(fc_asymmetries):.2e} (confirming FC is symmetric)")
    print(f"  Mean VAR(1) asymmetry: {np.mean(var1_asymmetries):.4f} (confirming A is asymmetric)")
    print(f"  Total spurious correlations: {total_spurious}")
    print(f"  Total hidden causal edges: {total_hidden}")
    print()
    print("Conclusion: FC symmetry cannot encode directed dynamics.")
    print("            Granger causality recovers asymmetric relationships.")
    print("=" * 60)
