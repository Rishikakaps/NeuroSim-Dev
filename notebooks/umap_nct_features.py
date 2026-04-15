"""
umap_nct_features.py
====================
UMAP visualization of Network Control Theory feature vectors.

Generates synthetic subjects from three cohorts (HC, AUD, AD) with
different spectral radius distributions, computes per-node controllability
metrics, and embeds into 2D using UMAP.

Cohorts:
  - HC (Healthy Control): ρ ~ N(0.75, 0.05) — flexible dynamics
  - AUD (Alcohol Use Disorder): ρ ~ N(0.88, 0.04) — rigid attractor
  - AD (Alzheimer's Disease): ρ ~ N(0.70, 0.06) — degraded connectivity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov

try:
    import umap
except ImportError:
    print("Install umap-learn: pip install umap-learn")
    exit(1)


def generate_stable_A(n=200, target_rho=0.85, seed=None):
    """
    Generate a stable asymmetric connectivity matrix.

    Parameters:
        n: number of ROIs (default 200 for Schaefer atlas)
        target_rho: target spectral radius (< 1 for stability)
        seed: random seed for reproducibility

    Returns:
        A: n×n asymmetric connectivity matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Random asymmetric matrix (no self-loops)
    A = np.random.randn(n, n) * 0.3
    np.fill_diagonal(A, 0)

    # Scale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    current_rho = np.max(np.abs(eigvals))
    if current_rho > 1e-8:
        A = A * (target_rho / current_rho)

    return A


def compute_nct_features(A):
    """
    Compute per-node NCT metrics for a given connectivity matrix.

    Parameters:
        A: n×n connectivity matrix

    Returns:
        features: 2n-dimensional feature vector [AC_1...AC_n, MC_1...MC_n]
    """
    n = A.shape[0]

    # Solve Lyapunov for Gramian
    Wc = solve_discrete_lyapunov(A, np.eye(n))

    # Average Controllability: diagonal of Gramian
    AC = np.diag(Wc)

    # Modal Controllability: Σ_j (1 - |λ_j|²) * |v_ij|²
    eigenvalues, eigenvectors = np.linalg.eig(A)
    weights = 1 - np.abs(eigenvalues) ** 2
    V_sq = np.abs(eigenvectors) ** 2
    MC = V_sq @ weights

    # Concatenate into feature vector
    return np.concatenate([AC, MC])


def generate_cohort(cohort_name, n_subjects=20, n_rois=200, rho_mean=0.75, rho_std=0.05, seed_offset=0):
    """
    Generate synthetic subjects for a cohort.

    Parameters:
        cohort_name: 'HC', 'AUD', or 'AD'
        n_subjects: number of subjects to generate
        n_rois: number of ROIs in the atlas
        rho_mean: mean spectral radius for this cohort
        rho_std: standard deviation of spectral radius
        seed_offset: offset for random seed to ensure different subjects

    Returns:
        feature_matrix: (n_subjects × 2*n_rois) array
        spectral_radii: (n_subjects,) array of actual spectral radii
    """
    features = []
    radii = []

    for i in range(n_subjects):
        # Draw spectral radius from cohort distribution
        rho = np.random.normal(rho_mean, rho_std)
        rho = np.clip(rho, 0.5, 0.95)  # ensure stability

        # Generate connectivity matrix
        seed = seed_offset * 1000 + i
        A = generate_stable_A(n=n_rois, target_rho=rho, seed=seed)

        # Compute features
        feat = compute_nct_features(A)
        features.append(feat)

        # Record actual spectral radius
        actual_rho = np.max(np.abs(np.linalg.eigvals(A)))
        radii.append(actual_rho)

    return np.array(features), np.array(radii)


def main():
    print("=" * 60)
    print("  UMAP Visualization of NCT Features (Synthetic Data)")
    print("=" * 60)

    # Generate cohorts
    print("\nGenerating synthetic subjects...")

    hc_features, hc_radii = generate_cohort(
        'HC', n_subjects=20, rho_mean=0.75, rho_std=0.05, seed_offset=1
    )
    print(f"  HC: 20 subjects, ρ = {hc_radii.mean():.3f} ± {hc_radii.std():.3f}")

    aud_features, aud_radii = generate_cohort(
        'AUD', n_subjects=20, rho_mean=0.88, rho_std=0.04, seed_offset=2
    )
    print(f"  AUD: 20 subjects, ρ = {aud_radii.mean():.3f} ± {aud_radii.std():.3f}")

    ad_features, ad_radii = generate_cohort(
        'AD', n_subjects=20, rho_mean=0.70, rho_std=0.06, seed_offset=3
    )
    print(f"  AD: 20 subjects, ρ = {ad_radii.mean():.3f} ± {ad_radii.std():.3f}")

    # Concatenate all features
    X = np.vstack([hc_features, aud_features, ad_features])
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)  # 0=HC, 1=AUD, 2=AD

    print(f"\nFeature matrix shape: {X.shape}")
    print("  Features: [AC_1...AC_200, MC_1...MC_200]")

    # Run UMAP
    print("\nRunning UMAP (n_components=2, n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)

    print(f"  Embedding shape: {embedding.shape}")


    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels_str = {0: 'HC (Healthy)', 1: 'AUD (Alcohol Use Disorder)', 2: 'AD (Alzheimer\'s)'}

    for cohort_id in [0, 1, 2]:
        mask = labels == cohort_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[cohort_id],
            label=labels_str[cohort_id],
            alpha=0.7,
            s=80,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP of NCT Feature Vectors (Synthetic Data, N=60)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig('figures/umap_synthetic.png', dpi=150, bbox_inches='tight')
    print("\n  ✓ Saved: figures/umap_synthetic.png")
    plt.close()

    print("\n" + "=" * 60)
    print("  UMAP visualization complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
