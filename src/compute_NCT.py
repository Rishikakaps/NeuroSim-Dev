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
  Finite-horizon: Wc = Σ_{k=0}^{T-1} A^k (A^k)^T
  We use B = I (all nodes can be controlled — standard in the literature).
  T=20 is enough for spectral-radius-normalized A (contributions decay fast).

Numerical safety:
  - We check Gramian Frobenius norm is finite at each step.
  - Gramian inversion uses regularization (+ eps*I) to handle ill-conditioning.
  - MC uses real parts of eigenvalues only (imaginary parts are numerical noise).
"""

import numpy as np
import pandas as pd
import os


GRAMIAN_T = 20  # finite-horizon timesteps


def compute_gramian(A, T=GRAMIAN_T):
    """
    Wc = Σ_{k=0}^{T-1} A^k (A^k)^T
    B = I (identity — all nodes are inputs).
    """
    n = A.shape[0]
    W = np.zeros((n, n))
    Ak = np.eye(n)  # A^0 = I

    for k in range(T):
        contrib = Ak @ Ak.T
        W += contrib

        # Numerical guard: if any value explodes, the EC was not properly normalized
        if not np.isfinite(W).all():
            raise RuntimeError(
                f"Gramian became non-finite at k={k}. "
                "Spectral radius of A must be < 1 before calling this. "
                "Check compute_EC.py postprocess_A()."
            )

        Ak = A @ Ak

    frob = np.linalg.norm(W, 'fro')
    print(f"  Gramian Frobenius norm: {frob:.4f}  (should be finite and positive)")
    return W


def average_controllability(A, T=GRAMIAN_T):
    """
    AC_i = sum_{k=0}^{T-1} ||col_i(A^k)||^2

    This measures how much influence node i has across all k-step paths.
    The squared column norm of A^k gives the energy contribution from node i
    at step k; summing over all k gives total controllability.

    Returns: array of shape (N,) with per-node AC values.
    """
    N = A.shape[0]
    ac = np.zeros(N)
    Ak = A.copy()          # ← start at A^1, not np.eye(N)
    for k in range(1, T): # ← k starts at 1
        # col_i(A^k) is Ak[:, i]
        # ||col_i(A^k)||^2 = sum_j (Ak[j, i])^2
        # np.sum(Ak**2, axis=0) computes this for all i simultaneously
        ac += np.sum(Ak**2, axis=0)
        Ak = A @ Ak
    return ac



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


def process_all_subjects(ec_dir, output_dir, ts_dir, T=GRAMIAN_T):
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

        W = compute_gramian(A, T=T)
        ac = average_controllability(A, T=T)
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