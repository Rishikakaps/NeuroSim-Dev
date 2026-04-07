"""
generate_synthetic.py
=====================
Generates realistic synthetic ROI timeseries for four groups:

1. Healthy Controls (n=5)
   - Stable VAR(1) system with balanced connectivity
   - Spectral radius ~0.7
   - No pathological features

2. Alcohol Use Disorder (n=5)
   - Reduced prefrontal control (ROIs 0-4: dorsolateral PFC, vmPFC)
   - Increased noise/variance (reflecting neural instability)
   - Slightly elevated spectral radius (but still stable)
   - Higher energy cost for state transitions

3. Epilepsy (n=5)
   - Hyper-excitable seizure focus (ROIs 5-7: temporal/limbic nodes)
   - Strong localized outgoing connectivity from focus
   - Burst-like dynamics in timeseries
   - Elevated modal controllability in focus nodes

4. Alzheimer's Disease (n=5)
   - 40% reduction in outgoing connectivity from DMN nodes 0-4
   - Progressive hub degradation
   - Lowered spectral radius (~0.65)
   - Reduced network integration capacity

Each dataset outputs:
  - ROI timeseries (T x N CSV files)
  - Ground-truth A matrix (saved for validation)

Why synthetic data is legitimate for this paper:
  - Validates VAR(1) recovery by comparing estimated A vs A_true
  - Demonstrates pipeline can detect simulated pathological patterns
  - Provides proof-of-principle before applying to real clinical data
"""

import numpy as np
import os

np.random.seed(42)

# -----------------------------------------------------------------------------
# Network topology
# -----------------------------------------------------------------------------
N = 15   # ROIs (organized into functional groups)
T = 300  # timepoints (must be >> 5*N = 75 for VAR(1) to be estimable)

# Functional network assignments (simplified for simulation)
DMN_NODES = [0, 1, 2, 3, 4]       # Default Mode Network (includes PFC)
SEIZURE_FOCUS = [5, 6, 7]         # Temporal/limbic nodes (epilepsy focus)
SENSORIMOTOR = [8, 9, 10]         # Sensorimotor cortex
VISUAL = [11, 12, 13, 14]         # Visual cortex

# Subject counts
N_CONTROLS = 5
N_AUD = 5
N_EPILEPSY = 5
N_AD = 5


def make_stable_A(n, noise_scale=0.05, target_rho=0.7, seed=None):
    """
    Create a random asymmetric matrix with spectral radius ≈ target_rho.

    The sparse structure gives biological plausibility (not every node
    connects to every other node with equal strength).

    Parameters
    ----------
    n : int
        Number of nodes
    noise_scale : float
        Base standard deviation of connection weights
    target_rho : float
        Target spectral radius (must be < 1 for stability)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    A : ndarray (n, n)
        Stable connectivity matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Create sparse-ish connectivity (biological: not all-to-all)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and np.random.rand() > 0.6:  # 40% connection density
                A[i, j] = np.random.randn() * noise_scale

    # Scale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))
    if rho > 1e-8:
        A = A * (target_rho / rho)

    # Verify stability
    rho_final = np.max(np.abs(np.linalg.eigvals(A)))
    assert rho_final < 1.0, f"Ground-truth A is not stable! rho={rho_final:.4f}"

    return A


def generate_timeseries(A, T, noise_std=0.1, x0=None):
    """
    Simulate VAR(1) process: X[t] = A @ X[t-1] + noise

    Parameters
    ----------
    A : ndarray (n, n)
        Connectivity matrix
    T : int
        Number of timepoints
    noise_std : float
        Standard deviation of Gaussian noise
    x0 : ndarray (n,), optional
        Initial state (default: small random)

    Returns
    -------
    X : ndarray (T, n)
        Timeseries matrix
    """
    n = A.shape[0]
    X = np.zeros((T, n))
    X[0] = x0 if x0 is not None else np.random.randn(n) * 0.1

    for t in range(1, T):
        X[t] = A @ X[t - 1] + np.random.randn(n) * noise_std

    return X


def make_control_A(seed):
    """
    Generate healthy control connectivity matrix.

    Features:
    - Balanced connectivity across networks
    - Moderate spectral radius (~0.7)
    - No pathological hyper/hypo-connectivity
    """
    np.random.seed(seed)
    A = make_stable_A(N, noise_scale=0.05, target_rho=0.70, seed=seed)
    return A


def make_AUD_A(control_A, seed):
    """
    Generate Alcohol Use Disorder connectivity matrix.

    Simulated pathology:
    - 35% reduction in prefrontal outgoing connections (DMN nodes 0-4)
      → Reduced top-down control
    - 50% increase in noise sensitivity (implemented in timeseries)
    - Slightly elevated spectral radius (0.75 vs 0.70)
      → Network operates closer to instability

    References:
    - Reduced PFC function in AUD (Goldstein & Volkow, 2011)
    - Increased neural variability in addiction (Luijten et al., 2017)
    """
    np.random.seed(seed)
    A = control_A.copy()

    # Reduce prefrontal outgoing connectivity (reduced top-down control)
    for pfc_node in DMN_NODES:
        A[pfc_node, :] *= 0.65  # 35% reduction

    # Slightly increase overall connectivity (compensatory?)
    # But keep spectral radius < 1
    A = A * 1.05

    # Ensure stability
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (0.90 / rho)

    return A


def make_Epilepsy_A(control_A, seed):
    """
    Generate Epilepsy connectivity matrix.

    Simulated pathology:
    - 2.5x increase in outgoing connectivity from seizure focus
      → Hyper-excitable nodes that strongly drive others
    - Slight increase in local recurrent connections within focus
    - Higher spectral radius (closer to instability threshold)

    This creates:
    - Elevated modal controllability in focus nodes
    - Burst-like propagation patterns in timeseries

    References:
    - Hyper-excitable focus in temporal lobe epilepsy (Spencer & Spencer, 1991)
    - Increased network controllability in epilepsy (Taylor et al., 2015)
    """
    np.random.seed(seed)
    A = control_A.copy()

    # Hyper-excite seizure focus (strong outgoing connections)
    for focus_node in SEIZURE_FOCUS:
        A[focus_node, :] *= 2.5  # 150% increase in outgoing strength

    # Increase local recurrent connections within focus
    for i in SEIZURE_FOCUS:
        for j in SEIZURE_FOCUS:
            if i != j:
                A[i, j] = np.sign(A[i, j]) * min(abs(A[i, j]) * 1.5, 0.3)

    # Re-stabilize (epileptic networks are still stable inter-ictally)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 1.0:
        A = A * (0.80 / rho)

    return A


def make_AD_A(control_A, seed):
    """
    Generate Alzheimer's Disease connectivity matrix.

    Simulated pathology:
    - 40% reduction in outgoing connectivity from DMN nodes 0-4
      → Progressive hub degradation in default mode network
    - Lowered spectral radius (~0.65 vs 0.70)
      → Reduced network integration capacity

    References:
    - DMN connectivity loss in AD (Buckner et al., 2005)
    - Hub degradation in Alzheimer's (Stam et al., 2009)
    """
    np.random.seed(seed)
    A = control_A.copy()

    # Reduce DMN outgoing connectivity by 40% (hub degradation)
    for dmn_node in DMN_NODES:
        A[dmn_node, :] *= 0.60  # 40% reduction

    # Scale to target spectral radius ~0.65
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (0.65 / rho)

    return A


def main():
    """Generate all synthetic datasets."""

    os.makedirs('data/roi_timeseries', exist_ok=True)
    os.makedirs('data/ground_truth', exist_ok=True)

    print("=" * 60)
    print("  NeuroSim Synthetic Data Generation")
    print("=" * 60)
    print(f"\nNetwork: {N} ROIs, {T} timepoints")
    print(f"  - DMN/PFC nodes: {DMN_NODES}")
    print(f"  - Seizure focus: {SEIZURE_FOCUS}")
    print(f"\nGenerating {N_CONTROLS} controls, {N_AUD} AUD, {N_EPILEPSY} epilepsy subjects\n")

    # -------------------------------------------------------------------------
    # Generate a shared "template" control matrix as base
    # -------------------------------------------------------------------------
    print("Building control ground-truth A...")
    A_control_template = make_control_A(seed=100)
    rho_c = np.max(np.abs(np.linalg.eigvals(A_control_template)))
    print(f"  Control A spectral radius: {rho_c:.4f}")

    # Save template
    np.savetxt('data/ground_truth/A_control_template.csv',
               A_control_template, delimiter=',')

    # -------------------------------------------------------------------------
    # Generate control subjects
    # -------------------------------------------------------------------------
    print(f"\n── Controls (n={N_CONTROLS}) ─────────────────────────────────")
    for i in range(1, N_CONTROLS + 1):
        # Each control has slight individual variation
        np.random.seed(200 + i)
        A_control = A_control_template + np.random.randn(N, N) * 0.005
        np.fill_diagonal(A_control, 0)

        # Ensure stability
        rho = np.max(np.abs(np.linalg.eigvals(A_control)))
        if rho >= 1.0:
            A_control = A_control * (0.70 / rho)

        X = generate_timeseries(A_control, T, noise_std=0.10)
        path = f'data/roi_timeseries/control_{i}.csv'
        np.savetxt(path, X, delimiter=',')

        # Save individual ground truth
        np.savetxt(f'data/ground_truth/A_control_{i}.csv', A_control, delimiter=',')

        print(f"  control_{i}: rho={np.max(np.abs(np.linalg.eigvals(A_control))):.4f}, "
              f"timeseries mean={X.mean():.4f}, std={X.std():.4f}")

    # -------------------------------------------------------------------------
    # Generate AUD subjects
    # -------------------------------------------------------------------------
    print(f"\n── Alcohol Use Disorder (n={N_AUD}) ─────────────────────────")
    for i in range(1, N_AUD + 1):
        A_AUD = make_AUD_A(A_control_template, seed=300 + i)

        # AUD: increased noise (neural instability)
        X = generate_timeseries(A_AUD, T, noise_std=0.15)  # 50% more noise
        path = f'data/roi_timeseries/aud_{i}.csv'
        np.savetxt(path, X, delimiter=',')

        np.savetxt(f'data/ground_truth/A_aud_{i}.csv', A_AUD, delimiter=',')

        print(f"  aud_{i}: rho={np.max(np.abs(np.linalg.eigvals(A_AUD))):.4f}, "
              f"timeseries mean={X.mean():.4f}, std={X.std():.4f}")

    # -------------------------------------------------------------------------
    # Generate Epilepsy subjects
    # -------------------------------------------------------------------------
    print(f"\n── Epilepsy (n={N_EPILEPSY}) ──────────────────────────────────")
    for i in range(1, N_EPILEPSY + 1):
        A_epilepsy = make_Epilepsy_A(A_control_template, seed=400 + i)

        # Epilepsy: occasional "burst" events (simulate interictal spikes)
        X = generate_timeseries(A_epilepsy, T, noise_std=0.10)

        # Add burst events in seizure focus nodes
        burst_times = np.random.choice(range(50, T-50), size=3, replace=False)
        for bt in burst_times:
            for t in range(bt, min(bt + 5, T)):
                for node in SEIZURE_FOCUS:
                    X[t, node] += np.random.randn() * 0.5  # burst amplitude

        path = f'data/roi_timeseries/epilepsy_{i}.csv'
        np.savetxt(path, X, delimiter=',')

        np.savetxt(f'data/ground_truth/A_epilepsy_{i}.csv', A_epilepsy, delimiter=',')

        print(f"  epilepsy_{i}: rho={np.max(np.abs(np.linalg.eigvals(A_epilepsy))):.4f}, "
              f"timeseries mean={X.mean():.4f}, std={X.std():.4f}")

    # -------------------------------------------------------------------------
    # Generate Alzheimer's Disease subjects
    # -------------------------------------------------------------------------
    print(f"\n── Alzheimer's Disease (n={N_AD}) ───────────────────────────────")
    for i in range(1, N_AD + 1):
        A_AD = make_AD_A(A_control_template, seed=500 + i)

        X = generate_timeseries(A_AD, T, noise_std=0.10)
        path = f'data/roi_timeseries/ad_{i}.csv'
        np.savetxt(path, X, delimiter=',')

        np.savetxt(f'data/ground_truth/A_ad_{i}.csv', A_AD, delimiter=',')

        print(f"  ad_{i}: rho={np.max(np.abs(np.linalg.eigvals(A_AD))):.4f}, "
              f"timeseries mean={X.mean():.4f}, std={X.std():.4f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Timeseries: data/roi_timeseries/ ({N_CONTROLS + N_AUD + N_EPILEPSY + N_AD} files)")
    print(f"  Ground truth A: data/ground_truth/ ({N_CONTROLS + N_AUD + N_EPILEPSY + N_AD + 1} files)")
    print(f"\nExpected findings:")
    print(f"  AUD: Reduced AC in PFC (nodes {DMN_NODES}), higher E*")
    print(f"  Epilepsy: Elevated MC in seizure focus (nodes {SEIZURE_FOCUS})")
    print(f"  AD: Reduced AC in DMN (nodes {DMN_NODES}), reduced spectral radius")


if __name__ == '__main__':
    main()
