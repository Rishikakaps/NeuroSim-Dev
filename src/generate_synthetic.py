"""
generate_synthetic.py
=====================
Generates synthetic ROI timeseries from a known VAR(1) generative model.
- 2 control subjects (A_control)
- 3 patient subjects  (A_patient = A_control + perturbation)
- Shape: T x N (300 timepoints, 15 ROIs)

Why this is legitimate for the paper:
  We can validate the VAR(1) recovery by comparing estimated A vs A_true.
  This is a genuine numerical validation step.
"""

import numpy as np
import os

np.random.seed(42)

N = 15   # ROIs (≤20 per ai_rules.md)
T = 300  # timepoints (must be >> 5*N = 75 for VAR(1) to be estimable)

# -------------------------------------------------------------------------
# Build a stable ground-truth A for controls
# Stable means spectral radius < 1 so the VAR(1) process doesn't explode.
# -------------------------------------------------------------------------
def make_stable_A(n, noise_scale=0.05, target_rho=0.7):
    """
    Create a random asymmetric matrix with spectral radius ≈ target_rho.
    The sparse structure gives it biological plausibility.
    """
    A = np.random.randn(n, n) * noise_scale
    np.fill_diagonal(A, 0)  # no self-loops

    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho > 1e-8:
        A = A * (target_rho / rho)  # scale to desired spectral radius

    # Verify
    rho_final = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"  A spectral radius: {rho_final:.4f} (target {target_rho})")
    assert rho_final < 1.0, "Ground-truth A is not stable!"
    return A


def generate_timeseries(A, T, noise_std=0.1):
    """
    Simulate X[t] = A @ X[t-1] + noise
    Returns T x N matrix.
    """
    n = A.shape[0]
    X = np.zeros((T, n))
    X[0] = np.random.randn(n) * 0.1
    for t in range(1, T):
        X[t] = A @ X[t - 1] + np.random.randn(n) * noise_std
    return X


def main():
    os.makedirs('data/roi_timeseries', exist_ok=True)

    print("Building control ground-truth A...")
    A_control = make_stable_A(N, noise_scale=0.05, target_rho=0.7)

    # Patient A = control A with structured DMN degradation
    # This simulates AD-like hub weakening in the Default Mode Network.
    print("Building patient ground-truth A (DMN degradation)...")
    A_patient = A_control.copy()
    # ROIs 0-4 represent DMN hubs (posterior cingulate, mPFC, angular gyrus)
    dmn_nodes = [0, 1, 2, 3, 4]
    A_patient[np.ix_(dmn_nodes, dmn_nodes)] *= 0.6  # 40% reduction in DMN coupling
    # Re-stabilize patient A (perturbation might push rho over 1)
    rho_p = np.max(np.abs(np.linalg.eigvals(A_patient)))
    if rho_p >= 1.0:
        A_patient = A_patient * (0.75 / rho_p)
    print(f"  Patient A spectral radius: {np.max(np.abs(np.linalg.eigvals(A_patient))):.4f}")

    # Save ground-truth matrices for later validation
    np.savetxt('data/A_control_true.csv', A_control, delimiter=',')
    np.savetxt('data/A_patient_true.csv', A_patient, delimiter=',')
    print("Saved ground-truth A matrices.")

    # ---- Generate control subjects ----
    for i in range(1, 3):
        X = generate_timeseries(A_control, T, noise_std=0.1)
        path = f'data/roi_timeseries/control_{i}.csv'
        np.savetxt(path, X, delimiter=',')
        print(f"Saved {path}  shape={X.shape}  mean={X.mean():.4f}  std={X.std():.4f}")

    # ---- Generate patient subjects ----
    for i in range(1, 4):
        X = generate_timeseries(A_patient, T, noise_std=0.1)
        path = f'data/roi_timeseries/patient_{i}.csv'
        np.savetxt(path, X, delimiter=',')
        print(f"Saved {path}  shape={X.shape}  mean={X.mean():.4f}  std={X.std():.4f}")

    print("\nDone. Run sanity check: each file should be 300 rows x 15 cols.")


if __name__ == '__main__':
    main()
