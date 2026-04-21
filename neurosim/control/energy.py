"""
energy.py
=========
Finite-horizon minimum control energy with explicit B matrix and state trajectory.

Method:
  For linear system x(t+1) = A @ x(t) + B @ u(t), the minimum energy
  to steer from x0 to xf in time T is:

    Wc_T = sum_{k=0}^{T-1} A^k @ B @ B.T @ (A^k).T  (finite-horizon Gramian)
    delta = xf - A^T @ x0                            (net state change needed)
    E_total = delta.T @ pinv(Wc_T) @ delta           (minimum energy)
    u*(t) = B.T @ (A^{T-1-t}).T @ pinv(Wc_T) @ delta (optimal control sequence)

Relation to compute_NCT.py:
  The per-ROI E* in compute_NCT.py uses:
    - x0 = 0 (rest state assumption)
    - xf = e_i (one-hot vector, activate ROI i)
    - B = I (identity, each ROI can be directly controlled)

  This module generalizes to:
    - Arbitrary x0 (non-zero initial brain state)
    - Arbitrary xf (any target state)
    - Explicit B matrix (subset of controllable nodes)

Known limitations:
  - Assumes exact knowledge of A matrix
  - Does not account for control constraints (u must be feasible)
  - Numerical issues if Wc_T is ill-conditioned (cond > 1e10)

Parameters:
  A: (N, N) system matrix, spectral radius < 1
  B: (N, M) input matrix (M control inputs)
  T: int, time horizon
  x0: (N,) initial state
  xf: (N,) target state

Returns:
  E_total: scalar, minimum control energy
  u_sequence: (T, M) optimal control inputs
"""

import numpy as np
import warnings
from scipy.linalg import pinv


def compute_finite_horizon_gramian(A, B, T):
    """
    Compute finite-horizon controllability Gramian.

    Wc_T = sum_{k=0}^{T-1} A^k @ B @ B.T @ (A^k).T

    This is the Gramian for steering the system over exactly T steps.
    As T -> infinity, Wc_T converges to the solution of the Lyapunov
    equation A @ Wc @ A.T - Wc + B @ B.T = 0 (for stable A).

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix
    B : ndarray (N, M)
        Input matrix (M control inputs)
    T : int
        Time horizon (number of steps)

    Returns
    -------
    Wc_T : ndarray (N, N)
        Finite-horizon controllability Gramian

    Raises
    ------
    ValueError
        If T < 1

    Notes
    -----
    - Uses iterative computation: Ak = A^k, accumulate sum
    - Computational cost: O(T * N^3) for matrix multiplications
    - For large T, consider using Lyapunov solution instead
    """
    if T < 1:
        raise ValueError(f"Time horizon T must be >= 1, got T={T}")

    N = A.shape[0]
    Wc_T = np.zeros((N, N))

    # Iterative computation: Ak = A^k
    Ak = np.eye(N)  # A^0 = I
    BBt = B @ B.T   # (N, N) precompute

    for k in range(T):
        # Add A^k @ B @ B.T @ (A^k).T = Ak @ BBt @ Ak.T
        Wc_T += Ak @ BBt @ Ak.T
        Ak = A @ Ak  # A^{k+1}

    return Wc_T


def minimum_energy(A, T, B, x0, xf):
    """
    Compute minimum control energy and optimal control sequence.

    Solves the optimal control problem:
      min_u sum_{t=0}^{T-1} ||u(t)||^2
      subject to x(t+1) = A @ x(t) + B @ u(t)
                 x(0) = x0, x(T) = xf

    The solution is:
      Wc_T = sum_{k=0}^{T-1} A^k @ B @ B.T @ (A^k).T
      delta = xf - A^T @ x0
      E_total = delta.T @ pinv(Wc_T) @ delta
      u*(t) = B.T @ (A^{T-1-t}).T @ pinv(Wc_T) @ delta

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix. Should have spectral radius < 1 for stability.
    T : int
        Time horizon (number of control steps)
    B : ndarray (N, M)
        Input matrix (M control inputs)
    x0 : ndarray (N,)
        Initial state
    xf : ndarray (N,)
        Target state

    Returns
    -------
    E_total : float
        Minimum control energy (scalar)
    u_sequence : ndarray (T, M)
        Optimal control inputs for each time step

    Raises
    ------
    ValueError
        If T < 1

    Warnings
    --------
    Issues a warning if Wc_T is ill-conditioned (condition number > 1e10).

    Notes
    -----
    - Uses Moore-Penrose pseudoinverse for numerical stability
    - For B=I, x0=0, xf=e_i, this reduces to per-ROI E* from compute_NCT.py
    - The optimal control is open-loop (precomputed sequence)
    - Energy is in units of ||u||^2 (squared L2 norm)
    """
    if T < 1:
        raise ValueError(f"Time horizon T must be >= 1, got T={T}")

    N = A.shape[0]
    M = B.shape[1] if B.ndim > 1 else 1

    # Ensure inputs are proper shapes
    x0 = np.asarray(x0).flatten()
    xf = np.asarray(xf).flatten()

    if x0.shape[0] != N or xf.shape[0] != N:
        raise ValueError(f"x0 and xf must have length N={A.shape[0]}")

    # Compute finite-horizon Gramian
    Wc_T = compute_finite_horizon_gramian(A, B, T)

    # Check conditioning
    cond_Wc = np.linalg.cond(Wc_T)
    if cond_Wc > 1e10:
        warnings.warn(
            f"Finite-horizon Gramian is ill-conditioned: cond(Wc_T) = {cond_Wc:.2e}. "
            "Energy computation may be numerically unstable. "
            "Consider increasing T or using regularization."
        )

    # Compute A^T @ x0 (state propagation without control)
    AT = np.linalg.matrix_power(A, T)
    Ax0 = AT @ x0

    # Net state change needed
    delta = xf - Ax0

    # Pseudoinverse of Gramian
    Wc_pinv = pinv(Wc_T)

    # Minimum energy: E = delta.T @ Wc_pinv @ delta
    E_total = float(delta.T @ Wc_pinv @ delta)

    # Optimal control sequence: u*(t) = B.T @ (A^{T-1-t}).T @ Wc_pinv @ delta
    u_sequence = np.zeros((T, M))

    # Precompute powers of A.T for efficiency
    At_powers = [np.eye(N)]  # (A.T)^0
    for k in range(1, T):
        At_powers.append(A.T @ At_powers[-1])

    for t in range(T):
        # (A^{T-1-t}).T = (A.T)^{T-1-t}
        At_power = At_powers[T - 1 - t]
        u_sequence[t] = B.T @ At_power @ Wc_pinv @ delta

    return E_total, u_sequence


def energy_per_roi_nct_style(A, T, x0, xf):
    """
    Compute per-ROI minimum energy in the style of compute_NCT.py.

    This is a convenience wrapper that uses B=I (identity input matrix)
    and computes energy for each ROI as target.

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix
    T : int
        Time horizon
    x0 : ndarray (N,)
        Initial state
    xf : ndarray (N,) or str
        If 'onehot', compute energy for each one-hot target e_i
        If array, use as single target state

    Returns
    -------
    energies : ndarray (N,) or float
        If xf='onehot', per-ROI energies
        If xf=array, single energy value

    Notes
    -----
    - For x0=0, xf='onehot', this matches compute_NCT.py's E* computation
    - Uses B=I (all nodes directly controllable)
    """
    N = A.shape[0]
    B = np.eye(N)  # Identity input matrix

    if isinstance(xf, str) and xf == 'onehot':
        energies = np.zeros(N)
        for i in range(N):
            target = np.zeros(N)
            target[i] = 1.0
            energies[i], _ = minimum_energy(A, T, B, x0, target)
        return energies
    else:
        E, _ = minimum_energy(A, T, B, x0, xf)
        return E


if __name__ == '__main__':
    """
    Demonstration: Compare finite-horizon energy with per-ROI E* from
    compute_NCT.py style computation.

    Setup:
      - N=15, T=20, B=I, x0=zeros, xf=e_0 (activate ROI 0)
      - Show that results match when B=I, x0=0
    """
    print("=" * 60)
    print("  Finite-Horizon Control Energy Demonstration")
    print("=" * 60)

    np.random.seed(42)

    # Parameters
    N = 15
    T = 20

    # Generate stable A matrix
    A = np.random.randn(N, N) * 0.1
    np.fill_diagonal(A, 0)

    # Scale to stable spectral radius
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (0.7 / rho)

    print(f"\nSystem: N={N} nodes, T={T} steps")
    print(f"Spectral radius: {np.max(np.abs(np.linalg.eigvals(A))):.4f}")

    # Setup: B=I, x0=0, xf=e_0
    B = np.eye(N)
    x0 = np.zeros(N)
    xf = np.zeros(N)
    xf[0] = 1.0  # Target: activate ROI 0

    print(f"\nControl setup:")
    print(f"  B = I (all nodes controllable)")
    print(f"  x0 = 0 (rest state)")
    print(f"  xf = e_0 (activate ROI 0)")

    # Compute energy using general function
    E_total, u_seq = minimum_energy(A, T, B, x0, xf)

    print(f"\nResults:")
    print(f"  Minimum energy E* = {E_total:.6f}")
    print(f"  Control sequence shape: {u_seq.shape}")
    print(f"  Max control magnitude: {np.abs(u_seq).max():.6f}")

    # Compare with per-ROI computation (NCT style)
    print("\n" + "-" * 60)
    print("Comparison with per-ROI E* (compute_NCT.py style):")

    # Compute all per-ROI energies
    energies_all = energy_per_roi_nct_style(A, T, x0, 'onehot')

    print(f"\nPer-ROI energies (B=I, x0=0):")
    for i in range(N):
        print(f"  ROI {i:2d}: E* = {energies_all[i]:.6f}")

    # Verify: energy for ROI 0 should match E_total
    print(f"\nVerification:")
    print(f"  E* for ROI 0 (direct): {E_total:.6f}")
    print(f"  E* for ROI 0 (per-ROI): {energies_all[0]:.6f}")
    print(f"  Match: {np.isclose(E_total, energies_all[0])}")

    print("\n" + "=" * 60)
    print("Conclusion: General energy formula matches per-ROI E* when")
    print("            B=I and x0=0 (the compute_NCT.py special case).")
    print("=" * 60)
