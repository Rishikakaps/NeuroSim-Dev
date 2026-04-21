"""
test_granger.py
===============
Unit tests for the Granger causality module.

Tests verify:
  1. Granger matrix asymmetry (G != G.T)
  2. Granger diagonal is zero
  3. FC is always symmetric (corrcoef property)
  4. Spurious edges exist (FC non-zero where true A = 0)
  5. Causality summary returns all required keys
  6. Granger recovers true edges with sufficient signal

Run with:
    python -m pytest tests/test_granger.py -v
    OR
    python tests/test_granger.py
"""

import numpy as np
import sys
import os

# Add neurosim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosim.connectivity.granger import (
    granger_causality_matrix,
    causality_vs_correlation_summary,
    generate_random_stable_system,
)


def generate_var1_timeseries(A, T=300, seed=None):
    """Generate VAR(1) timeseries: X[t] = A @ X[t-1] + noise."""
    if seed is not None:
        np.random.seed(seed)

    N = A.shape[0]
    X = np.zeros((T, N))
    X[0] = np.random.randn(N) * 0.1

    for t in range(1, T):
        X[t] = A @ X[t-1] + np.random.randn(N) * 0.1

    return X


# =============================================================================
# Test 1: Granger Matrix Asymmetry
# =============================================================================
def test_granger_matrix_asymmetric():
    """
    Granger causality matrix should NOT be symmetric.

    G[i,j] = 1 means j -> i (j causes i)
    G[j,i] = 1 means i -> j (i causes j)
    These are independent relationships, so G should not equal G.T.
    """
    np.random.seed(42)

    # Generate random stable system
    A, X = generate_random_stable_system(N=10, rho=0.7, T=300, seed=42)

    # Compute Granger causality
    G, F_matrix, p_matrix = granger_causality_matrix(X, order=1, alpha=0.05)

    # G should NOT be symmetric
    is_symmetric = np.array_equal(G, G.T)

    assert not is_symmetric, (
        "Granger matrix G is symmetric - this should not happen! "
        "G[i,j] and G[j,i] represent different causal relationships."
    )

    # Quantify asymmetry
    asymmetry = np.sum(G != G.T) / 2  # count asymmetric pairs
    print(f"  ✓ G is asymmetric: {asymmetry} asymmetric pairs out of {G.size} total")


# =============================================================================
# Test 2: Granger Diagonal Zero
# =============================================================================
def test_granger_diagonal_zero():
    """
    Granger causality diagonal must always be zero.

    We do NOT test self-causality (i -> i), so G[i,i] = 0 for all i.
    """
    np.random.seed(42)

    A, X = generate_random_stable_system(N=10, rho=0.7, T=300, seed=42)
    G, F_matrix, p_matrix = granger_causality_matrix(X, order=1, alpha=0.05)

    # Diagonal should be exactly zero
    diagonal = np.diag(G)

    assert np.all(diagonal == 0), (
        f"Granger diagonal should be all zeros, got: {diagonal}"
    )

    # Also check F_matrix and p_matrix diagonals
    # F[i,i] should be 0 (no test performed)
    # p[i,i] should be 1.0 (null hypothesis: no causality)
    assert np.all(np.diag(F_matrix) == 0), "F_matrix diagonal should be 0"
    assert np.all(np.diag(p_matrix) == 1.0), "p_matrix diagonal should be 1.0"

    print("  ✓ G diagonal is zero (no self-causality tested)")


# =============================================================================
# Test 3: FC Is Always Symmetric
# =============================================================================
def test_fc_is_always_symmetric():
    """
    Functional connectivity (Pearson correlation) must satisfy FC == FC.T exactly.

    This is a mathematical property of correlation - corr(X,Y) = corr(Y,X).
    This test verifies that np.corrcoef produces symmetric output.
    """
    np.random.seed(42)

    # Generate random timeseries (doesn't need to be VAR(1))
    T, N = 300, 10
    X = np.random.randn(T, N)

    # Compute FC
    FC = np.corrcoef(X.T)
    np.fill_diagonal(FC, 0)

    # Check exact symmetry
    is_symmetric = np.allclose(FC, FC.T)
    max_asymmetry = np.max(np.abs(FC - FC.T))

    assert is_symmetric, (
        f"FC matrix should be symmetric! Max |FC - FC.T| = {max_asymmetry}"
    )

    print(f"  ✓ FC is symmetric: max|FC - FC.T| = {max_asymmetry:.2e}")


# =============================================================================
# Test 4: Spurious Edges Exist
# =============================================================================
def test_spurious_edges_exist():
    """
    On synthetic data with known zero edges, FC should have non-zero entries
    where true A[i,j] = 0.

    This demonstrates that correlation can arise from:
    - Common drivers (A->B and A->C creates FC[B,C])
    - Indirect paths (A->B->C creates FC[A,C])
    """
    np.random.seed(42)

    # Create sparse A with known zero edges
    N = 10
    A = np.zeros((N, N))

    # Only add a few specific edges
    A[1, 0] = 0.5  # 0 -> 1
    A[2, 1] = 0.5  # 1 -> 2
    # Note: A[2, 0] = 0 (no direct edge 0 -> 2)

    # Scale to stable
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (0.7 / rho)

    # Generate timeseries
    X = generate_var1_timeseries(A, T=500, seed=42)

    # Compute FC
    FC = np.corrcoef(X.T)
    np.fill_diagonal(FC, 0)

    # Check: FC[0, 2] should be non-zero due to indirect path 0->1->2
    fc_indirect = np.abs(FC[2, 0])

    # We expect some correlation even though A[2, 0] = 0
    # (exact value depends on noise, but should be > 0.1 for strong indirect path)
    assert fc_indirect > 0.05, (
        f"Expected FC[2,0] > 0.05 due to indirect path, got {fc_indirect:.4f}. "
        "Spurious correlations should exist from indirect effects."
    )

    print(f"  ✓ Spurious correlation exists: |FC[2,0]| = {fc_indirect:.4f} (true A[2,0]=0)")


# =============================================================================
# Test 5: Causality Summary Keys
# =============================================================================
def test_causality_summary_keys():
    """
    causality_vs_correlation_summary must return all required keys.
    """
    np.random.seed(42)

    A, X = generate_random_stable_system(N=10, rho=0.7, T=300, seed=42)

    # Fit VAR(1) to get A_var1
    X_past = X[:-1]
    X_future = X[1:]
    A_est_T, _, _, _ = np.linalg.lstsq(X_past, X_future, rcond=None)
    A_var1 = A_est_T.T
    np.fill_diagonal(A_var1, 0)

    # Run summary
    summary = causality_vs_correlation_summary(A_var1, X, order=1, alpha=0.05)

    # Check required keys
    required_keys = [
        'n_spurious',
        'n_hidden',
        'fc_asymmetry',
        'var1_asymmetry',
        'G',
        'F_matrix',
        'p_matrix',
        'FC',
    ]

    for key in required_keys:
        assert key in summary, f"Missing required key: {key}"

    # Check types
    assert isinstance(summary['n_spurious'], int)
    assert isinstance(summary['n_hidden'], int)
    assert isinstance(summary['fc_asymmetry'], float)
    assert isinstance(summary['var1_asymmetry'], float)
    assert summary['G'].shape == (10, 10)
    assert summary['F_matrix'].shape == (10, 10)
    assert summary['p_matrix'].shape == (10, 10)
    assert summary['FC'].shape == (10, 10)

    print("  ✓ causality_vs_correlation_summary returns all required keys")


# =============================================================================
# Test 6: Granger Recovers True Edge
# =============================================================================
def test_granger_recovers_true_edge():
    """
    Inject a strong edge A[0,1]=0.4 in synthetic data and verify
    granger_causality_matrix detects it with G[0,1]=1 and p<0.05.
    """
    np.random.seed(42)

    # Create A with a known strong edge
    N = 10
    A = np.zeros((N, N))

    # Inject strong edge: 1 -> 0 (A[0,1] = 0.4)
    A[0, 1] = 0.4

    # Add some background connectivity
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] == 0 and np.random.rand() > 0.7:
                A[i, j] = np.random.randn() * 0.1

    # Scale to stable
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 1.0:
        A = A * (0.7 / rho)

    # Generate long timeseries for better detection
    X = generate_var1_timeseries(A, T=500, seed=42)

    # Compute Granger causality
    G, F_matrix, p_matrix = granger_causality_matrix(X, order=1, alpha=0.05)

    # Check: G[0, 1] should be 1 (edge detected)
    # p_matrix[0, 1] should be < 0.05 (significant)
    edge_detected = G[0, 1] == 1
    p_significant = p_matrix[0, 1] < 0.05

    assert edge_detected, (
        f"Granger failed to detect strong edge A[0,1]=0.4: G[0,1]={G[0,1]}"
    )

    assert p_significant, (
        f"Edge A[0,1]=0.4 should be significant: p={p_matrix[0,1]:.6f}"
    )

    print(f"  ✓ Granger recovers true edge A[0,1]=0.4: G[0,1]={G[0,1]}, p={p_matrix[0,1]:.4f}")


# =============================================================================
# Run Tests
# =============================================================================
def run_tests():
    """Run all tests and print results."""
    print("=" * 60)
    print("  NeuroSim Granger Causality Test Suite")
    print("=" * 60)

    tests = [
        ("Granger matrix asymmetric", test_granger_matrix_asymmetric),
        ("Granger diagonal zero", test_granger_diagonal_zero),
        ("FC always symmetric", test_fc_is_always_symmetric),
        ("Spurious edges exist", test_spurious_edges_exist),
        ("Causality summary keys", test_causality_summary_keys),
        ("Granger recovers true edge", test_granger_recovers_true_edge),
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
