"""
compute_NCT.py
==============
Network Control Theory metrics computation.

Functions:
  - compute_gramian(A, T=20): Finite-horizon controllability Gramian
  - average_controllability(A, T=20): Per-node average controllability
  - modal_controllability(A): Per-node modal controllability
  - min_control_energy(W, N): Per-node minimum control energy

Constants:
  - GRAMIAN_T = 20: Default time horizon for Gramian computation
"""

import numpy as np


GRAMIAN_T = 20


def compute_gramian(A, T=20):
    """
    Compute finite-horizon controllability Gramian.

    Wc_T = sum_{k=0}^{T-1} A^k @ (A^k).T

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix (must be stable: spectral radius < 1)
    T : int, default=20
        Time horizon (number of terms in the sum)

    Returns
    -------
    Wc : ndarray (N, N)
        Controllability Gramian (symmetric positive definite)

    Raises
    ------
    RuntimeError
        If spectral radius >= 1 (unstable system) or if non-finite values
    """
    n = A.shape[0]

    # Check stability first
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))

    if rho >= 1.0:
        raise RuntimeError(
            f"Spectral radius rho={rho:.4f} >= 1.0. "
            "Gramian undefined for unstable systems."
        )

    # Finite-horizon sum: Wc = sum_{k=0}^{T-1} A^k @ (A^k).T
    Wc = np.zeros((n, n))
    Ak = np.eye(n)
    for k in range(T):
        Wc += Ak @ Ak.T
        Ak = A @ Ak

    # Check for non-finite values
    if not np.isfinite(Wc).all():
        raise RuntimeError(
            "Gramian contains non-finite values. "
            "This may indicate spectral radius too close to 1."
        )

    return Wc


def average_controllability(A, T=20):
    """
    Compute average controllability for each node.

    AC_i = sum_{k=0}^{T-1} ||col_i(A^k)||^2

    This measures how much each node's input propagates through the network
    over T time steps.

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix (must be stable)
    T : int, default=20
        Time horizon for finite sum

    Returns
    -------
    ac : ndarray (N,)
        Per-node average controllability values

    Notes
    -----
    - Computes sum of squared column norms: AC_i = sum_k ||col_i(A^k)||^2
    - This differs from diag(Wc) which gives row norms for asymmetric A
    """
    N = A.shape[0]
    ac = np.zeros(N)
    Ak = np.eye(N)

    for k in range(T):
        ac += np.sum(Ak**2, axis=0)  # ||col_i(A^k)||^2 for each i
        Ak = A @ Ak

    return ac


def modal_controllability(A):
    """
    Compute modal controllability for each node.

    MC_i = sum_j (1 - |lambda_j|^2) * |v_ij|^2

    where lambda_j = eigenvalues, v_ij = eigenvector components.
    Uses absolute values for complex eigenvalues/eigenvectors.

    Parameters
    ----------
    A : ndarray (N, N)
        System matrix

    Returns
    -------
    mc : ndarray (N,)
        Per-node modal controllability values
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Use absolute values for complex eigenvalues/eigenvectors
    weights = 1 - np.abs(eigenvalues) ** 2
    V_sq = np.abs(eigenvectors) ** 2

    # MC_i = sum_j weights[j] * |V[i,j]|^2
    mc = V_sq @ weights

    return mc


def min_control_energy(W, N=10):
    """
    Compute minimum control energy for each node.

    E*_i = W_inv[i,i] (diagonal of inverse Gramian)

    This assumes x0=0 (rest state) and xf=e_i (one-hot target).

    Parameters
    ----------
    W : ndarray (N, N)
        Controllability Gramian (positive definite)
    N : int, default=10
        Number of nodes (dimension of system)

    Returns
    -------
    energy : ndarray (N,)
        Per-node minimum control energy values
    """
    # Regularize before inversion
    eps = 1e-6
    W_reg = W + eps * np.eye(N)

    W_inv = np.linalg.inv(W_reg)

    # Diagonal of inverse gives E* for each node
    energy = np.diag(W_inv)

    return energy
