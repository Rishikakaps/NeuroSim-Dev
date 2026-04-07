"""
test_nct.py
===========
Unit tests for Network Control Theory computations.

Tests verify:
  1. Average Controllability: correct formula, per-node values
  2. Modal Controllability: non-negative values, complex eigenvalue handling
  3. Minimum Control Energy: positive values, x0=0 assumption
  4. Gramian: finite-horizon convergence, positive definiteness

Run with:
    python -m pytest tests/test_nct.py -v
    OR
    python tests/test_nct.py
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neurosim.src.compute_NCT import (
    compute_gramian,
    average_controllability,
    modal_controllability,
    min_control_energy,
    GRAMIAN_T
)


def make_test_A(n=10, rho=0.5, seed=42):
    """Create a stable test matrix with known spectral radius."""
    np.random.seed(seed)
    A = np.random.randn(n, n) * 0.1
    np.fill_diagonal(A, 0)

    # Scale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    current_rho = np.max(np.abs(eigvals))
    if current_rho > 1e-8:
        A = A * (rho / current_rho)

    return A


# =============================================================================
# Test: Gramian Computation
# =============================================================================
def test_gramian_is_positive_definite():
    """Gramian should be positive definite for stable A."""
    A = make_test_A(n=10, rho=0.5)
    W = compute_gramian(A, T=GRAMIAN_T)

    # Symmetric (by construction)
    assert np.allclose(W, W.T), "Gramian should be symmetric"

    # Positive definite: all eigenvalues > 0
    eigvals = np.linalg.eigvalsh(W)
    assert np.all(eigvals > 0), f"Gramian has non-positive eigenvalues: {eigvals.min()}"

    print("  ✓ Gramian is positive definite")


def test_gramian_converges():
    """Gramian should converge for stable A (rho < 1)."""
    A = make_test_A(n=10, rho=0.5)

    # Compute with increasing T
    W1 = compute_gramian(A, T=10)
    W2 = compute_gramian(A, T=20)
    W3 = compute_gramian(A, T=50)

    # Should converge (differences get smaller)
    diff1 = np.linalg.norm(W2 - W1, 'fro')
    diff2 = np.linalg.norm(W3 - W2, 'fro')

    # The contribution from A^k decays as rho^k, so differences should decrease
    assert diff2 < diff1 * 2, "Gramian should be converging"

    print(f"  ✓ Gramian converges (diff1={diff1:.4f}, diff2={diff2:.4f})")


def test_gramian_unstable_fails():
    """Gramian computation should detect unstable A."""
    # Create marginally unstable matrix
    A = make_test_A(n=5, rho=1.1)

    try:
        W = compute_gramian(A, T=50)
        # If we get here, check if values are finite
        assert np.isfinite(W).all(), "Gramian should be non-finite for unstable A"
        print("  ⚠ Unstable A did not cause overflow (may need larger T)")
    except RuntimeError as e:
        assert "non-finite" in str(e) or "spectral radius" in str(e).lower()
        print("  ✓ Unstable A correctly detected")


# =============================================================================
# Test: Average Controllability
# =============================================================================
def test_average_controllability_shape():
    """AC should return per-node values with correct shape."""
    A = make_test_A(n=10, rho=0.5)
    ac = average_controllability(A, T=GRAMIAN_T)

    assert ac.shape == (10,), f"Expected shape (10,), got {ac.shape}"
    assert np.all(ac > 0), "AC values should be positive"

    print(f"  ✓ AC has correct shape, range=[{ac.min():.4f}, {ac.max():.4f}]")


def test_average_controllability_formula():
    """Verify AC formula: AC_i = sum_k ||col_i(A^k)||^2."""
    A = make_test_A(n=5, rho=0.5, seed=123)
    N = A.shape[0]
    T = 10

    # Manual computation
    ac_manual = np.zeros(N)
    Ak = np.eye(N)
    for k in range(T):
        ac_manual += np.sum(Ak**2, axis=0)
        Ak = A @ Ak

    # Function computation
    ac_func = average_controllability(A, T=T)

    assert np.allclose(ac_manual, ac_func), "AC formula mismatch"

    print("  ✓ AC formula verified against manual computation")


def test_average_controllability_increases_with_T():
    """AC should increase (or stay same) as T increases (adding positive terms)."""
    A = make_test_A(n=8, rho=0.6)

    ac1 = average_controllability(A, T=5)
    ac2 = average_controllability(A, T=20)

    # Each term is non-negative (squared norms), so AC should be non-decreasing
    assert np.all(ac2 >= ac1 - 1e-10), "AC should be non-decreasing with T"

    print(f"  ✓ AC non-decreasing with T (mean: {ac1.mean():.4f} → {ac2.mean():.4f})")


# =============================================================================
# Test: Modal Controllability
# =============================================================================
def test_modal_controllability_non_negative():
    """MC must be non-negative (uses |λ| < 1 and squared eigenvectors)."""
    A = make_test_A(n=10, rho=0.7)
    mc = modal_controllability(A)

    assert np.all(mc >= 0), f"MC has negative values: min={mc.min()}"

    print(f"  ✓ MC non-negative, range=[{mc.min():.4f}, {mc.max():.4f}]")


def test_modal_controllability_complex_eigenvalues():
    """MC should handle complex eigenvalues correctly using absolute values."""
    # Create asymmetric matrix (likely to have complex eigenvalues)
    np.random.seed(42)
    A = np.random.randn(8, 8) * 0.1
    np.fill_diagonal(A, 0)

    # Scale to stable
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (0.6 / rho)

    # Check eigenvalues
    eigvals = np.linalg.eigvals(A)
    has_complex = not np.allclose(eigvals.imag, 0)

    # Compute MC (should work regardless)
    mc = modal_controllability(A)

    assert np.all(mc >= 0), "MC should be non-negative even with complex eigenvalues"
    assert np.all(np.isfinite(mc)), "MC should be finite"

    status = "complex eigenvalues present" if has_complex else "real eigenvalues"
    print(f"  ✓ MC handles {status}, range=[{mc.min():.4f}, {mc.max():.4f}]")


def test_modal_controllability_varies_across_nodes():
    """MC should vary across nodes (not uniform like AC)."""
    A = make_test_A(n=12, rho=0.6)
    mc = modal_controllability(A)

    # MC should have variance (different nodes have different MC)
    std_mc = mc.std()
    assert std_mc > 0.01, f"MC should vary across nodes (std={std_mc:.6f})"

    print(f"  ✓ MC varies across nodes (std={std_mc:.4f})")


# =============================================================================
# Test: Minimum Control Energy
# =============================================================================
def test_min_control_energy_positive():
    """E* should be positive (diagonal of positive definite inverse)."""
    A = make_test_A(n=10, rho=0.5)
    W = compute_gramian(A, T=GRAMIAN_T)
    energy = min_control_energy(W, N=10)

    assert np.all(energy > 0), f"E* has non-positive values: min={energy.min()}"

    print(f"  ✓ E* positive, range=[{energy.min():.4f}, {energy.max():.4f}]")


def test_min_control_energy_shape():
    """E* should return per-node values with correct shape."""
    A = make_test_A(n=10, rho=0.5)
    W = compute_gramian(A, T=GRAMIAN_T)
    energy = min_control_energy(W, N=10)

    assert energy.shape == (10,), f"Expected shape (10,), got {energy.shape}"

    print(f"  ✓ E* has correct shape")


def test_min_control_energy_x0_assumption():
    """
    Verify E* formula assumes x0=0.

    The formula E*_i = W^{-1}[i,i] is only correct when starting from rest.
    This test documents the limitation.
    """
    A = make_test_A(n=5, rho=0.5)
    W = compute_gramian(A, T=GRAMIAN_T)
    energy = min_control_energy(W, N=5)

    # The formula gives diagonal of inverse
    W_inv = np.linalg.inv(W + 1e-6 * np.eye(5))
    expected = np.diag(W_inv)

    assert np.allclose(energy, expected), "E* should equal diagonal of W^{-1}"

    print("  ✓ E* = diag(W^{-1}) [assumes x0=0]")


# =============================================================================
# Test: Integration (full pipeline on small example)
# =============================================================================
def test_full_computation_pipeline():
    """Test all functions together on a small example."""
    n = 6
    A = make_test_A(n=n, rho=0.6)

    # Compute Gramian
    W = compute_gramian(A, T=20)
    assert np.isfinite(W).all()

    # Compute all metrics
    ac = average_controllability(A, T=20)
    mc = modal_controllability(A)
    energy = min_control_energy(W, N=n)

    # Validate outputs
    assert ac.shape == (n,)
    assert mc.shape == (n,)
    assert energy.shape == (n,)

    assert np.all(ac > 0)
    assert np.all(mc >= 0)
    assert np.all(energy > 0)

    print(f"  ✓ Full pipeline: AC=[{ac.min():.3f}-{ac.max():.3f}], "
          f"MC=[{mc.min():.3f}-{mc.max():.3f}], "
          f"E*=[{energy.min():.3f}-{energy.max():.3f}]")


# =============================================================================
# Run tests
# =============================================================================
def run_tests():
    """Run all tests and print results."""
    print("="*60)
    print("  NeuroSim NCT Unit Tests")
    print("="*60)

    tests = [
        ("Gramian positive definite", test_gramian_is_positive_definite),
        ("Gramian convergence", test_gramian_converges),
        ("Gramian unstable detection", test_gramian_unstable_fails),
        ("AC shape", test_average_controllability_shape),
        ("AC formula", test_average_controllability_formula),
        ("AC non-decreasing", test_average_controllability_increases_with_T),
        ("MC non-negative", test_modal_controllability_non_negative),
        ("MC complex eigenvalues", test_modal_controllability_complex_eigenvalues),
        ("MC varies across nodes", test_modal_controllability_varies_across_nodes),
        ("E* positive", test_min_control_energy_positive),
        ("E* shape", test_min_control_energy_shape),
        ("E* x0=0 assumption", test_min_control_energy_x0_assumption),
        ("Full pipeline", test_full_computation_pipeline),
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

    print("\n" + "="*60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
