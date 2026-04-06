"""
compute_EC.py
=============
Estimates Effective Connectivity (EC) from ROI timeseries using VAR(1).

Method: OLS regression (numpy lstsq) — no statsmodels needed.
VAR(1): X[t] = A @ X[t-1] + noise
  → Rearrange: X[1:] = X[:-1] @ A.T + noise
  → A.T = lstsq(X[:-1], X[1:])
  → A = result.T

Why OLS and not statsmodels.VAR?
  statsmodels.VAR fits one model with lags. Our OLS approach:
  - Is identical mathematically for lag=1
  - Requires only numpy (runs anywhere)
  - Is fully transparent (easy to explain in paper)

Post-processing (CRITICAL for NCT):
  1. Zero the diagonal (no self-loops in connectivity model)
  2. Normalize so spectral radius < 1 (required for Gramian convergence)
  3. Assert asymmetry (if A ≈ A.T something went badly wrong)
"""

import numpy as np
import pandas as pd
import os


def load_timeseries(path):
    """Load T x N timeseries CSV. Demean each column."""
    X = np.loadtxt(path, delimiter=',')
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"
    X = X - X.mean(axis=0)   # demean per ROI (stationarity assumption)
    return X


def fit_var1_ols(X):
    """
    Fit VAR(1) via OLS.

    X has shape (T, N).
    We want A such that X[t] ≈ A @ X[t-1].
    Rewrite as: X[1:] ≈ X[:-1] @ A.T
    Solve: A.T = lstsq(X[:-1], X[1:])

    Returns N x N asymmetric coefficient matrix A.
    """
    X_past = X[:-1]   # (T-1, N)  — predictor
    X_future = X[1:]  # (T-1, N)  — target

    # OLS: A.T = pinv(X_past) @ X_future
    # lstsq is numerically safer than direct inversion
    A_T, residuals, rank, sv = np.linalg.lstsq(X_past, X_future, rcond=None)
    A = A_T.T  # (N, N)

    print(f"  VAR(1) fit rank: {rank} / {X_past.shape[1]}  "
          f"(full rank = {X_past.shape[1] == rank})")
    return A


def postprocess_A(A, subject_id=""):
    """
    Apply mandatory post-processing for NCT validity:
    1. Zero diagonal
    2. Normalize spectral radius to < 1
    3. Verify asymmetry
    """
    N = A.shape[0]

    # 1. Zero diagonal
    np.fill_diagonal(A, 0)

    # 2. Spectral radius normalization
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))
    print(f"  Spectral radius before normalization: {rho:.4f}")

    if rho >= 1.0:
        A = A / (rho + 1e-6)
        rho_new = np.max(np.abs(np.linalg.eigvals(A)))
        print(f"  Normalized → spectral radius: {rho_new:.4f}")
    else:
        print(f"  Spectral radius already < 1 — no normalization needed.")

    # 3. Asymmetry check
    sym_error = np.mean(np.abs(A - A.T))
    print(f"  Asymmetry check: mean|A - A.T| = {sym_error:.6f}")
    if sym_error < 1e-8:
        raise ValueError(
            f"EC matrix for {subject_id} is (near-)symmetric. "
            "This means VAR(1) failed to capture directed relationships. "
            "Check that your timeseries has enough variance across ROIs."
        )

    # 4. Final stability check
    rho_final = np.max(np.abs(np.linalg.eigvals(A)))
    assert rho_final < 1.0, f"Spectral radius {rho_final:.4f} ≥ 1 after normalization!"

    return A


def process_all_subjects(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(files)} subjects: {files}\n")

    for fname in files:
        subj = fname.replace('.csv', '')
        print(f"── {subj} ──────────────────────────────")
        X = load_timeseries(os.path.join(data_dir, fname))
        T, N = X.shape
        print(f"  Timeseries shape: T={T}, N={N}")

        if T < 5 * N:
            print(f"  ⚠ WARNING: T={T} < 5*N={5*N}. "
                  "VAR(1) may be poorly conditioned. "
                  "Consider shorter ROI set or longer scan.")

        A = fit_var1_ols(X)
        A = postprocess_A(A, subject_id=subj)

        out_path = os.path.join(output_dir, f"{subj}_EC.csv")
        np.savetxt(out_path, A, delimiter=',', fmt='%.8f')
        print(f"  ✓ Saved: {out_path}\n")

    print(f"EC computation complete. {len(files)} files written to {output_dir}/")


if __name__ == '__main__':
    process_all_subjects('data/roi_timeseries', 'outputs/EC')
