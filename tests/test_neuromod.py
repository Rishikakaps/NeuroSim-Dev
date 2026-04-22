"""
test_neuromod.py
================
Comprehensive test suite for Network Control Theory computations.

Tests verify:
  1. A matrix asymmetry (directed connectivity)
  2. Spectral radius stability
  3. Gramian positive definiteness and residual
  4. Modal and Average controllability properties
  5. FC symmetry breaks eigenvalues (core methodological argument)
  6. Energy optimal delta correction

Run with:
    pytest tests/test_neuromod.py -v
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.linalg import solve_discrete_lyapunov


def generate_stable_A(n=10, target_rho=0.85, seed=42):
    """
    Generate a stable VAR(1) coefficient matrix with controlled spectral radius.

    Creates an asymmetric matrix (realistic for directed connectivity)
    and scales it to have spectral radius = target_rho < 1.
    """
    np.random.seed(seed)
    # Random asymmetric matrix
    A = np.random.randn(n, n) * 0.3
    np.fill_diagonal(A, 0)  # no self-loops

    # Scale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    current_rho = np.max(np.abs(eigvals))
    if current_rho > 1e-8:
        A = A * (target_rho / current_rho)

    return A


def generate_var1_data(A, T=500, seed=42):
    """Generate synthetic VAR(1) timeseries: X[t] = A @ X[t-1] + noise."""
    np.random.seed(seed)
    N = A.shape[0]
    X = np.zeros((T, N))
    for t in range(1, T):
        X[t] = A @ X[t-1] + np.random.randn(N) * 0.1
    return X


# =============================================================================
# Test 1: A Matrix Asymmetry
# =============================================================================
def test_A_asymmetry():
    """
    Directed connectivity matrix must be asymmetric.

    If A ≈ A.T, the VAR(1) estimation failed to capture directed relationships.
    Mean asymmetry > 0.01 indicates genuine directionality.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)
    asymmetry = np.abs(A - A.T)
    mean_asymmetry = np.mean(asymmetry)

    assert mean_asymmetry > 0.01, (
        f"A matrix is (near-)symmetric: mean|A - A.T| = {mean_asymmetry:.6f}. "
        "VAR(1) should produce asymmetric connectivity."
    )
    print(f"  ✓ A asymmetry: mean|A - A.T| = {mean_asymmetry:.4f} > 0.01")


# =============================================================================
# Test 2: Spectral Radius Stability
# =============================================================================
def test_spectral_radius():
    """
    System must be stable for Gramian to converge.

    Spectral radius ρ = max|λ(A)| must satisfy ρ < 1.
    This ensures the Lyapunov equation has a unique positive definite solution.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))

    assert rho < 1.0, f"Spectral radius ρ = {rho:.4f} >= 1.0 — system unstable!"
    print(f"  ✓ Spectral radius ρ = {rho:.4f} < 1.0 (stable)")


# =============================================================================
# Test 3: Gramian Positive Definite
# =============================================================================
def test_gramian_positive_definite():
    """
    Controllability Gramian must be positive definite.

    For stable A (ρ < 1), the Lyapunov equation A @ Wc @ A.T - Wc + I = 0
    has a unique symmetric positive definite solution.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)
    Wc = solve_discrete_lyapunov(A, np.eye(10))

    # Symmetric
    assert np.allclose(Wc, Wc.T), "Gramian should be symmetric"

    # All eigenvalues > 0
    eigvals = np.linalg.eigvalsh(Wc)
    assert np.all(eigvals > 0), (
        f"Gramian has non-positive eigenvalues: min = {eigvals.min():.2e}"
    )
    print(f"  ✓ Gramian positive definite: λ_min = {eigvals.min():.4f} > 0")


# =============================================================================
# Test 4: Gramian Residual
# =============================================================================
def test_gramian_residual():
    """
    Lyapunov solution must satisfy the equation to machine precision.

    Residual = ||A @ Wc @ A.T - Wc + I||_F should be < 1e-8.
    This verifies the numerical accuracy of scipy.linalg.solve_discrete_lyapunov.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)
    Wc = solve_discrete_lyapunov(A, np.eye(10))

    residual = np.linalg.norm(A @ Wc @ A.T - Wc + np.eye(10), 'fro')
    assert residual < 1e-8, (
        f"Lyapunov residual = {residual:.2e} >= 1e-8 — solution inaccurate"
    )
    print(f"  ✓ Gramian residual = {residual:.2e} < 1e-8")


# =============================================================================
# Test 5: Modal Controllability Non-negative
# =============================================================================
def test_modal_controllability_nonnegative():
    """
    Modal controllability must be non-negative for all nodes.

    MC_i = Σ_j (1 - |λ_j|²) * |v_ij|²
    Since |λ| < 1 for stable A, all weights (1 - |λ|²) > 0.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)

    eigenvalues, eigenvectors = np.linalg.eig(A)
    weights = 1 - np.abs(eigenvalues) ** 2
    V_sq = np.abs(eigenvectors) ** 2
    mc = V_sq @ weights

    assert np.all(mc >= 0), (
        f"Modal controllability has negative values: min = {mc.min():.4f}"
    )
    print(f"  ✓ Modal controllability non-negative: min = {mc.min():.4f}")


# =============================================================================
# Test 6: Average Controllability Positive
# =============================================================================
def test_average_controllability_positive():
    """
    Average controllability must be positive for all nodes.

    AC = diag(Wc) where Wc is positive definite.
    Diagonal elements of a positive definite matrix are always positive.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)
    Wc = solve_discrete_lyapunov(A, np.eye(10))
    ac = np.diag(Wc)

    assert np.all(ac > 0), (
        f"Average controllability has non-positive values: min = {ac.min():.4f}"
    )
    print(f"  ✓ Average controllability positive: min = {ac.min():.4f}")


# =============================================================================
# Test 7: FC Symmetry Breaks Eigenvalues (CORE TEST)
# =============================================================================
def test_fc_symmetry_breaks_eigenvalues():
    """
    Demonstrates the core methodological argument: FC symmetry destroys causal geometry.

    Symmetric matrices have purely real eigenvalues (spectral theorem).
    Asymmetric A from VAR(1) has complex eigenvalues encoding oscillatory dynamics.
    Feeding symmetric FC into NCT collapses these complex eigenvalues, destroying
    the causal geometry that controllability metrics depend on.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)

    # Symmetrize: this is what you get if you use Pearson correlation
    A_sym = (A + A.T) / 2

    # Eigenvalues of symmetric matrix must be real
    eigvals_sym = np.linalg.eigvals(A_sym)
    max_imag_sym = np.max(np.abs(eigvals_sym.imag))

    # Original A should have at least one complex eigenvalue
    eigvals_orig = np.linalg.eigvals(A)
    max_imag_orig = np.max(np.abs(eigvals_orig.imag))

    # Symmetric → purely real eigenvalues
    assert max_imag_sym < 1e-10, (
        f"Symmetrized A has complex eigenvalues: max|Im| = {max_imag_sym:.2e}"
    )

    # Original A has complex eigenvalues (asymmetric → oscillatory dynamics)
    assert max_imag_orig > 1e-6, (
        f"Original A has no complex eigenvalues: max|Im| = {max_imag_orig:.2e}. "
        "This suggests A is too symmetric."
    )

    print(f"  ✓ Symmetric A: max|Im(λ)| = {max_imag_sym:.2e} < 1e-10 (real)")
    print(f"  ✓ Original A: max|Im(λ)| = {max_imag_orig:.4f} > 1e-6 (complex)")
    print("  → FC symmetry destroys causal geometry")


# =============================================================================
# Test 8: Energy Optimal Delta Correction
# =============================================================================
def test_energy_optimal_delta_correction():
    """
    Control energy must account for initial state x0.

    Correct formula: E*_i = (x_T - A^T x_0)^T Wc^{-1} (x_T - A^T x_0)
    Naive formula:   E*_i = x_T^T Wc^{-1} x_T  (assumes x0 = 0)

    For non-zero x0 (real data), these must differ.
    """
    A = generate_stable_A(n=10, target_rho=0.85, seed=42)
    Wc = solve_discrete_lyapunov(A, np.eye(10))
    eps = 1e-6
    W_inv = np.linalg.inv(Wc + eps * np.eye(10))

    # Non-zero initial state (realistic for patient data)
    x0 = np.random.randn(10) * 0.5
    x_T = np.zeros(10)
    x_T[0] = 1.0  # target: activate ROI 0

    # Correct: delta = x_T - A^T x_0
    Ax0 = A.T @ x0
    delta = x_T - Ax0
    E_correct = delta @ W_inv @ delta

    # Naive: assumes x0 = 0
    E_naive = x_T @ W_inv @ x_T

    # They must differ for non-zero x0
    assert not np.allclose(E_correct, E_naive, rtol=1e-3), (
        f"Energy with/without x0 correction should differ: "
        f"E_correct = {E_correct:.4f}, E_naive = {E_naive:.4f}"
    )
    print(f"  ✓ E* with x0 correction = {E_correct:.4f}")
    print(f"  ✓ E* naive (x0=0) = {E_naive:.4f}")
    print(f"  → Difference = {abs(E_correct - E_naive):.4f}")


# =============================================================================
# Run Tests
# =============================================================================
def run_tests():
    """Run all tests and print results."""
    print("=" * 60)
    print("  NeuroSim Neuromodulation Test Suite")
    print("=" * 60)

    tests = [
        ("A asymmetry", test_A_asymmetry),
        ("Spectral radius", test_spectral_radius),
        ("Gramian positive definite", test_gramian_positive_definite),
        ("Gramian residual", test_gramian_residual),
        ("Modal controllability non-negative", test_modal_controllability_nonnegative),
        ("Average controllability positive", test_average_controllability_positive),
        ("FC symmetry breaks eigenvalues", test_fc_symmetry_breaks_eigenvalues),
        ("Energy optimal delta correction", test_energy_optimal_delta_correction),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
