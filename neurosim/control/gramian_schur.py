"""
gramian_schur.py
================
Gramian precision benchmarking for large-scale systems.

Answers Dr. Agarwal's Q2: "Does the Lyapunov solver maintain precision
for large N and high spectral radius (rho -> 1)?"

Method:
  Solve the discrete Lyapunov equation:
    A @ Wc @ A.T - Wc + I = 0
  using scipy.linalg.solve_discrete_lyapunov (Schur method).

  The Schur method is backward stable and O(n^3), making it suitable
  for large-scale systems where finite-horizon summation would require
  prohibitively large T for convergence.

Why Lyapunov vs finite-horizon:
  Finite-horizon: Wc_T = sum_{k=0}^{T-1} A^k @ (A^k).T
  Convergence rate: ||Wc_inf - Wc_T|| ~ rho^(2T)

  For rho=0.90, T=20: residual ~ 0.90^40 = 0.015 (1.5% error)
  For rho=0.97, T=20: residual ~ 0.97^40 = 0.30 (30% error!)

  Lyapunov solve is exact regardless of rho (as long as rho < 1).

Known limitations:
  - Requires spectral radius < 1 for existence/uniqueness
  - O(n^3) complexity may be slow for N > 1000
  - Condition number grows as rho -> 1

Parameters:
  A: (N, N) system matrix with spectral radius < 1

Returns:
  Wc: (N, N) controllability Gramian
  precision_report: dict with residual, condition number, eigenvalue info
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import pandas as pd
import time
import warnings


def compute_gramian_large_scale(A):
    """
    Compute controllability Gramian for large-scale systems.

    Solves the discrete Lyapunov equation A @ Wc @ A.T - Wc + I = 0
    using the Schur decomposition method (scipy.linalg.solve_discrete_lyapunov).

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix. Must have spectral radius < 1 for stability.

    Returns
    -------
    Wc : ndarray (N, N)
        Controllability Gramian (symmetric positive definite)
    precision_report : dict
        Dictionary with keys:
        - 'lyapunov_residual': float, ||A @ Wc @ A.T - Wc + I||_F
        - 'condition_number': float, cond(Wc)
        - 'min_eigenvalue': float, lambda_min(Wc)
        - 'effective_rank': int, # eigenvalues > 1e-6 * lambda_max
        - 'is_valid': bool, True if residual < 1e-8 AND min_eig > 0

    Raises
    ------
    RuntimeError
        If spectral radius of A >= 1.0 (system unstable, Gramian undefined)

    Notes
    -----
    - The Schur method is backward stable: computed solution is exact
      for a slightly perturbed problem
    - Residual < 1e-8 indicates machine precision accuracy
    - Effective rank measures numerical rank deficiency
    """
    N = A.shape[0]

    # Check stability
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))

    if rho >= 1.0:
        raise RuntimeError(
            f"Spectral radius rho={rho:.4f} >= 1.0. "
            "Gramian is undefined for unstable systems."
        )

    # Solve Lyapunov equation
    Q = np.eye(N)
    Wc = solve_discrete_lyapunov(A, Q)

    # Compute precision metrics
    residual = np.linalg.norm(A @ Wc @ A.T - Wc + np.eye(N), 'fro')

    # Eigenvalue analysis
    eigvals_Wc = np.linalg.eigvalsh(Wc)  # Wc is symmetric, use eigvalsh
    min_eig = float(eigvals_Wc.min())
    max_eig = float(eigvals_Wc.max())

    # Condition number
    if min_eig > 1e-15:
        cond = max_eig / min_eig
    else:
        cond = np.inf

    # Effective rank: count eigenvalues > 1e-6 * max_eig
    threshold = 1e-6 * max_eig
    effective_rank = int(np.sum(eigvals_Wc > threshold))

    # Validity check
    is_valid = (residual < 1e-8) and (min_eig > 0)

    precision_report = {
        'lyapunov_residual': float(residual),
        'condition_number': float(cond),
        'min_eigenvalue': min_eig,
        'max_eigenvalue': max_eig,
        'effective_rank': effective_rank,
        'is_valid': is_valid,
    }

    return Wc, precision_report


def generate_random_stable_A(n, rho=0.7, seed=None):
    """
    Generate a random stable asymmetric matrix with target spectral radius.

    Parameters
    ----------
    n : int
        Matrix dimension
    rho : float, default=0.7
        Target spectral radius (must be < 1)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    A : ndarray (n, n)
        Stable connectivity matrix with spectral radius ≈ rho
    """
    if seed is not None:
        np.random.seed(seed)

    # Random asymmetric matrix with sparse connectivity
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and np.random.rand() > 0.6:
                A[i, j] = np.random.randn() * 0.1

    # Scale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    current_rho = np.max(np.abs(eigvals))
    if current_rho > 1e-8:
        A = A * (rho / current_rho)

    return A


def gramian_precision_benchmark(ns=[50, 100, 200], rho_values=[0.7, 0.85, 0.95],
                                 n_trials=3):
    """
    Benchmark Gramian computation precision across scales and spectral radii.

    For each combination of (n, rho, trial):
    1. Generate random stable A matrix
    2. Compute Gramian via Lyapunov solve
    3. Record precision metrics and timing

    Parameters
    ----------
    ns : list of int, default=[50, 100, 200]
        System sizes to benchmark
    rho_values : list of float, default=[0.7, 0.85, 0.95]
        Spectral radii to test (approaching instability)
    n_trials : int, default=3
        Number of random trials per (n, rho) combination

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
        - n: system size
        - rho: spectral radius
        - trial: trial number
        - lyapunov_residual: float
        - condition_number: float
        - min_eigenvalue: float
        - effective_rank: int
        - compute_time_ms: float

    Notes
    -----
    - Expected result: residual < 1e-8 for all (n, rho) combinations
    - Condition number increases as rho -> 1
    - Effective rank should equal n for well-conditioned systems
    """
    results = []

    for n in ns:
        for rho in rho_values:
            for trial in range(n_trials):
                seed = trial  # deterministic seed for reproducibility

                # Generate stable A
                A = generate_random_stable_A(n, rho=rho, seed=seed)

                # Verify stability
                actual_rho = np.max(np.abs(np.linalg.eigvals(A)))
                if actual_rho >= 1.0:
                    warnings.warn(
                        f"Generated unstable A for n={n}, rho={rho}, trial={trial}. "
                        f"Actual rho={actual_rho:.4f}. Skipping."
                    )
                    continue

                # Time the computation
                start = time.perf_counter()
                Wc, report = compute_gramian_large_scale(A)
                elapsed_ms = (time.perf_counter() - start) * 1000

                results.append({
                    'n': n,
                    'rho': rho,
                    'trial': trial,
                    'lyapunov_residual': report['lyapunov_residual'],
                    'condition_number': report['condition_number'],
                    'min_eigenvalue': report['min_eigenvalue'],
                    'effective_rank': report['effective_rank'],
                    'compute_time_ms': elapsed_ms,
                })

    df = pd.DataFrame(results)
    return df


def print_benchmark_summary(df):
    """
    Print a formatted summary table of benchmark results.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from gramian_precision_benchmark()
    """
    print("\n" + "=" * 80)
    print("  GRAMIAN PRECISION BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by (n, rho) and compute statistics
    grouped = df.groupby(['n', 'rho'])

    for (n, rho), group in grouped:
        print(f"\nn={n}, rho={rho} ({len(group)} trials):")
        print(f"  Residual:       mean={group['lyapunov_residual'].mean():.2e}, "
              f"max={group['lyapunov_residual'].max():.2e}")
        print(f"  Condition #:    mean={group['condition_number'].mean():.2e}, "
              f"max={group['condition_number'].max():.2e}")
        print(f"  Min eigenvalue: mean={group['min_eigenvalue'].mean():.2e}, "
              f"min={group['min_eigenvalue'].min():.2e}")
        print(f"  Effective rank: mean={group['effective_rank'].mean():.1f} / {n}")
        print(f"  Compute time:   mean={group['compute_time_ms'].mean():.2f} ms")

        # Check validity
        all_valid = group['lyapunov_residual'].max() < 1e-8
        status = "PASS" if all_valid else "FAIL"
        print(f"  Status: {status} (residual < 1e-8)")

    print("\n" + "=" * 80)

    # Overall summary
    print("\nOverall:")
    print(f"  Max residual across all trials: {df['lyapunov_residual'].max():.2e}")
    print(f"  All residuals < 1e-8: {(df['lyapunov_residual'] < 1e-8).all()}")
    print(f"  All min_eigenvalues > 0: {(df['min_eigenvalue'] > 0).all()}")
    print("=" * 80)


if __name__ == '__main__':
    """
    Run Gramian precision benchmark and print results.

    Expected: residual < 1e-8 holds across all (n, rho) combinations
    including rho=0.95 (near-instability).
    """
    print("=" * 60)
    print("  Gramian Precision Benchmark")
    print("=" * 60)
    print("\nTesting Lyapunov solver precision for large-scale systems...")
    print("System sizes: n = [50, 100, 200]")
    print("Spectral radii: rho = [0.7, 0.85, 0.95]")
    print("Trials per configuration: 3")

    # Run benchmark
    df = gramian_precision_benchmark(
        ns=[50, 100, 200],
        rho_values=[0.7, 0.85, 0.95],
        n_trials=3
    )

    # Print summary
    print_benchmark_summary(df)

    # Save results
    df.to_csv('outputs/gramian_benchmark_results.csv', index=False)
    print(f"\nResults saved to: outputs/gramian_benchmark_results.csv")
