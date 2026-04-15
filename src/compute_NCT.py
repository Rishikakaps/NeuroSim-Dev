"""
compute_NCT.py
==============
Computes Network Control Theory metrics from EC matrices.

Metrics:
  1. Average Controllability (AC)
     AC = trace(Wc) / N
     → Measures a node's ability to drive the brain into easily-reachable states.
     → Higher AC = node can influence many nearby brain states efficiently.

  2. Modal Controllability (MC)
     MC_i = Σ_j (1 - λ_j²) * |v_ij|²
     where λ_j = real eigenvalues of A, v_ij = i-th row of eigenvector matrix
     → High MC = node can push brain into difficult, high-energy modes.
     → From Gu et al. (2015) Nature Communications.

  3. Minimum Control Energy (E*)
     E*_i = x_T^T Wc^{-1} x_T
     where x_T = one-hot vector (target state = activate ROI i from rest)
     → Energy cost to drive the network from rest to activate each ROI.
     → High E* = expensive/hard to reach = potential biomarker for pathology.

The Gramian:
  We solve the discrete Lyapunov equation: A @ Wc @ A.T - Wc + I = 0
  using scipy.linalg.solve_discrete_lyapunov, which is exact and O(n³).

  Why Lyapunov instead of finite-horizon sum?
  The finite-horizon sum Wc = Σ_{k=0}^{T-1} A^k (A^k)^T converges at rate ρ^(2T),
  where ρ = spectral_radius(A). At ρ=0.90, T=20 gives ρ^40 ≈ 0.015 residual error
  — acceptable for healthy controls. But at ρ=0.97 (AUD subjects with rigid
  attractors), ρ^40 ≈ 0.30 — clinically unacceptable. The Lyapunov solve is
  exact regardless of ρ, as long as ρ < 1.

Numerical safety:
  - We verify the Lyapunov residual ||A @ Wc @ A.T - Wc + I||_F < 1e-8.
  - Gramian inversion uses regularization (+ eps*I) to handle ill-conditioning.
  - MC uses absolute values of complex eigenvalues/eigenvectors.
"""

import numpy as np
import pandas as pd
import os
from scipy.linalg import solve_discrete_lyapunov


def compute_gramian(A):
    """
    Solve the discrete Lyapunov equation: A @ Wc @ A.T - Wc + I = 0

    This is mathematically equivalent to the infinite-horizon sum
    Wc = Σ_{k=0}^{∞} A^k (A^k)^T, but computed exactly in O(n³) time.

    The finite-horizon approximation with T=20 has residual error scaling
    as ρ^(2T). For ρ=0.90, this gives ~1.5% error; for ρ=0.97 (rigid
    attractors in AUD), ~30% error. The Lyapunov solution is exact.

    Returns:
        Wc: n×n positive definite controllability Gramian

    Raises:
        RuntimeError: if residual ||A @ Wc @ A.T - Wc + I||_F >= 1e-8
    """
    n = A.shape[0]
    Q = np.eye(n)  # Identity matrix for the Lyapunov equation

    # Solve: A @ Wc @ A.T - Wc + Q = 0
    Wc = solve_discrete_lyapunov(A, Q)

    # Verify solution accuracy
    residual = np.linalg.norm(A @ Wc @ A.T - Wc + np.eye(n), 'fro')
    if residual >= 1e-8:
        raise RuntimeError(
            f"Lyapunov solution failed: residual = {residual:.2e} >= 1e-8. "
            "This indicates A is not stable (spectral radius >= 1)."
        )

    frob = np.linalg.norm(Wc, 'fro')
    print(f"  Gramian Frobenius norm: {frob:.4f}  (should be finite and positive)")
    print(f"  Lyapunov residual: {residual:.2e}  (should be < 1e-8)")
    return Wc


def average_controllability(A):
    """
    AC_i = Σ_{k=0}^{∞} ||col_i(A^k)||^2

    This measures how much influence node i has across all k-step paths.
    The squared column norm of A^k gives the energy contribution from node i
    at step k; summing over all k gives total controllability.

    We compute this efficiently using the Lyapunov Gramian:
    AC = diag(Wc) where Wc solves A @ Wc @ A.T - Wc + I = 0

    Returns: array of shape (N,) with per-node AC values.
    """
    N = A.shape[0]
    Wc = compute_gramian(A)
    return np.diag(Wc)



def modal_controllability(A):
    """
    MC_i = Σ_j (1 - |λ_j|²) * |v_ij|²

    Uses absolute values of complex eigenvalues/eigenvectors.
    Complex eigenvalues arise naturally from asymmetric A (not numerical artifacts).

    Sign convention: eigenvalues of our normalized A satisfy |λ| < 1,
    so (1 - |λ|²) > 0 always → MC is guaranteed non-negative.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Use absolute values for complex eigenvalues/eigenvectors
    weights = 1 - np.abs(eigenvalues) ** 2      # (N,)
    V_sq = np.abs(eigenvectors) ** 2            # (N, N) — element-wise squared magnitude

    # Check: all weights should be positive since |λ| < 1
    if np.any(weights < 0):
        n_neg = np.sum(weights < 0)
        print(f"  ⚠ WARNING: {n_neg} eigenvalues have |λ| > 1 → negative MC weights. "
              "This means the EC normalization didn't fully work. Clipping to 0.")
        weights = np.clip(weights, 0, None)

    # MC_i = Σ_j weights[j] * |V[i,j]|²
    mc = V_sq @ weights   # (N,) — matrix multiply

    print(f"  MC: min={mc.min():.4f}  max={mc.max():.4f}  mean={mc.mean():.4f}")
    return mc


def min_control_energy(W, A, x0, N):
    """
    E*_i = x_T^T Wc^{-1} x_T  where x_T = e_i (one-hot, target = ROI i active)
    = Wc^{-1}[i, i]  (diagonal of the inverse Gramian)

    This is the energy cost to drive the network from rest (x0=0) to
    a state where only ROI i is active (x_T = e_i).

    NOTE: This formula assumes x0 = 0 (rest state).
    For real patient data, correct formula is:
    E*_i = (x_T - A^T x_0)^T Wc^{-1} (x_T - A^T x_0)
    This is a known limitation of this scaled-down implementation.

    Regularization: we add eps*I to Wc before inverting to handle near-singular cases.
    """
    eps = 1e-6
    W_reg = W + eps * np.eye(N)
    cond = np.linalg.cond(W_reg)
    print(f"  Gramian condition number: {cond:.2e}")
    try:
        W_inv = np.linalg.inv(W_reg)
    except np.linalg.LinAlgError:
        W_inv = np.linalg.pinv(W_reg)

    Ax0 = A.T @ x0
    energy = np.zeros(N)
    for i in range(N):
        x_T = np.zeros(N)
        x_T[i] = 1.0
        delta = x_T - Ax0      # correct: subtract propagated initial state
        energy[i] = delta @ W_inv @ delta

    energy = np.clip(energy, 0, None)
    print(f"  E*: min={energy.min():.4f}  max={energy.max():.4f}  mean={energy.mean():.4f}")
    return energy


def process_all_subjects(ec_dir, output_dir, ts_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(ec_dir) if f.endswith('_EC.csv')])
    
    for fname in files:
        subj = fname.replace('_EC.csv', '')
        A = np.loadtxt(os.path.join(ec_dir, fname), delimiter=',')
        N = A.shape[0]

        # Load timeseries to get real x0
        ts_path = os.path.join(ts_dir, f"{subj}.csv")
        X_raw = np.loadtxt(ts_path, delimiter=',')
        x0 = X_raw.mean(axis=0)   # actual mean brain state, not zeros

        rho = np.max(np.abs(np.linalg.eigvals(A)))
        if rho >= 1.0:
            raise RuntimeError(f"EC for {subj} has rho={rho:.4f} >= 1.")

        W = compute_gramian(A)
        ac = average_controllability(A)
        mc = modal_controllability(A)
        energy = min_control_energy(W, A, x0, N)   # pass A and x0

        results = pd.DataFrame({
            'ROI': np.arange(N),
            'AverageControllability': ac,
            'ModalControllability': mc,
            'MinControlEnergy': energy
        })
        out_path = os.path.join(output_dir, f"{subj}_NCT.csv")
        results.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}\n")

if __name__ == '__main__':
    process_all_subjects('outputs/EC', 'outputs/NCT', 'data/roi_timeseries')