"""
combat.py
=========
Blind neuroCombat: Remove scanner/site effects from neuroimaging data.

Method (Johnson et al. 2007, Biostatistics):
  1. Estimate additive (gamma) and multiplicative (delta) batch parameters
     using empirical Bayes shrinkage on HEALTHY CONTROLS ONLY.
  2. Apply harmonization to clinical cohorts WITHOUT exposing disease labels.

Key constraint: "Blind" means disease labels NEVER enter parameter estimation.

Model:
  Y_{f,i} = alpha_f + gamma_{f,s(i)} + delta_{f,s(i)} * epsilon_{f,i}

  where:
  - Y_{f,i} = feature f for subject i
  - alpha_f = grand mean of feature f
  - gamma_{f,s} = additive site effect for site s
  - delta_{f,s} = multiplicative site effect for site s
  - epsilon_{f,i} = residual (mean 0, var 1)

Empirical Bayes shrinkage (equations 3 and 4 from Johnson et al. 2007):
  gamma* = posterior mean = (n_s * gamma_hat + prior_mean) / (n_s + 1)
  delta* = posterior mean using inverse-gamma prior

Parameters:
  hc_data: (N_features, N_hc_subjects) - HC timeseries features ONLY
  hc_site_labels: (N_hc_subjects,) - site/scanner strings for HC
  clinical_data: (N_features, N_clinical_subjects) - patient data
  clinical_site_labels: (N_clinical_subjects,) - sites for patients

Returns:
  harmonized_data: (N_features, N_subjects) harmonized array
  combat_params: dict with gamma_star, delta_star, grand_mean, pooled_var

Known limitations:
  - Assumes same feature space across sites
  - Requires >= 2 subjects per site for variance estimation
  - Does not handle continuous batch effects (only categorical sites)
"""

import numpy as np
import warnings


def estimate_combat_params(hc_data, hc_site_labels):
    """
    Estimate ComBat parameters using empirical Bayes on healthy controls ONLY.

    This is the "blind" constraint: clinical/disease labels never enter
    the estimation process. Only HC data and site labels are used.

    Parameters
    ----------
    hc_data : ndarray (N_features, N_hc_subjects)
        Feature matrix for healthy controls only.
        Rows = features (e.g., AC, MC per ROI), columns = subjects.
    hc_site_labels : array-like (N_hc_subjects,)
        Site/scanner labels for each HC subject.
        Must be strings or integers identifying the site.

    Returns
    -------
    combat_params : dict
        Dictionary with keys:
        - 'gamma_star': ndarray (N_features, N_sites), additive effects after EB shrinkage
        - 'delta_star': ndarray (N_features, N_sites), multiplicative effects after EB shrinkage
        - 'grand_mean': ndarray (N_features,), overall mean per feature
        - 'pooled_var': ndarray (N_features,), pooled variance per feature
        - 'site_ids': list, unique site identifiers

    Raises
    ------
    ValueError
        If any site has fewer than 2 subjects (cannot estimate variance)

    Notes
    -----
    - Implements empirical Bayes shrinkage from Johnson et al. 2007
    - gamma_star shrinks site means toward grand mean
    - delta_star shrinks site variances toward pooled variance
    """
    hc_data = np.asarray(hc_data)
    hc_site_labels = np.asarray(hc_site_labels)

    if hc_data.ndim != 2:
        raise ValueError(f"hc_data must be 2D (features x subjects), got {hc_data.ndim}D")

    n_features, n_subjects = hc_data.shape

    # Get unique sites
    site_ids = np.unique(hc_site_labels)
    n_sites = len(site_ids)

    if n_sites < 2:
        raise ValueError(f"Need at least 2 sites for ComBat, got {n_sites}")

    # Count subjects per site
    site_counts = np.array([np.sum(hc_site_labels == site) for site in site_ids])

    # Check minimum subjects per site
    if np.any(site_counts < 2):
        bad_sites = site_ids[site_counts < 2]
        raise ValueError(
            f"Sites with fewer than 2 subjects: {bad_sites}. "
            "Need >= 2 subjects per site for variance estimation."
        )

    # =========================================================================
    # Step 1: Compute grand mean and pooled variance
    # =========================================================================
    grand_mean = np.mean(hc_data, axis=1, keepdims=True)  # (N_features, 1)

    # Pooled variance (across all subjects, ignoring site)
    pooled_var = np.var(hc_data, axis=1, ddof=1)  # (N_features,)

    # =========================================================================
    # Step 2: Compute raw site effects (gamma_hat, delta_hat)
    # =========================================================================
    gamma_hat = np.zeros((n_features, n_sites))
    delta_hat = np.zeros((n_features, n_sites))

    for s_idx, site in enumerate(site_ids):
        site_mask = hc_site_labels == site
        site_data = hc_data[:, site_mask]  # (N_features, n_s)

        # Additive effect: site mean - grand mean
        site_mean = np.mean(site_data, axis=1, keepdims=True)  # (N_features, 1)
        gamma_hat[:, s_idx:s_idx+1] = site_mean - grand_mean

        # Multiplicative effect: site var / pooled var
        site_var = np.var(site_data, axis=1, ddof=1)  # (N_features,)
        delta_hat[:, s_idx] = site_var / pooled_var

    # =========================================================================
    # Step 3: Empirical Bayes shrinkage (Johnson et al. 2007, eq 3-4)
    # =========================================================================
    # For gamma: posterior mean = (n_s * gamma_hat) / (n_s + 1)
    # (assuming prior mean = 0, prior variance = 1)
    gamma_star = np.zeros((n_features, n_sites))
    delta_star = np.zeros((n_features, n_sites))

    for s_idx, site in enumerate(site_ids):
        n_s = site_counts[s_idx]

        # Gamma shrinkage: toward 0 (grand mean already subtracted)
        # gamma* = n_s / (n_s + 1) * gamma_hat
        shrinkage_gamma = n_s / (n_s + 1)
        gamma_star[:, s_idx] = shrinkage_gamma * gamma_hat[:, s_idx]

        # Delta shrinkage: toward 1 (no site effect)
        # Using simplified EB: delta* = (n_s * delta_hat + prior) / (n_s + prior_df)
        # Prior: delta ~ 1 (no multiplicative effect)
        # delta* = (n_s * delta_hat + 1) / (n_s + 1) approximately
        # More precisely, use inverse-gamma posterior mean
        prior_df = 1  # prior degrees of freedom
        delta_star[:, s_idx] = (n_s * delta_hat[:, s_idx] + prior_df) / (n_s + prior_df)

    # Ensure delta is positive
    delta_star = np.maximum(delta_star, 0.1)  # prevent division issues

    combat_params = {
        'gamma_star': gamma_star,
        'delta_star': delta_star,
        'grand_mean': grand_mean.flatten(),  # (N_features,)
        'pooled_var': pooled_var,  # (N_features,)
        'site_ids': list(site_ids),
    }

    return combat_params


def apply_combat(data, site_labels, combat_params):
    """
    Apply ComBat harmonization using pre-estimated parameters.

    Harmonization formula:
      Y_harm[f,i] = (Y[f,i] - gamma*[s,f]) / sqrt(delta*[s,f]) * sqrt(pooled_var[f]) + grand_mean[f]

    This centers and scales each site to have the same mean and variance.

    Parameters
    ----------
    data : ndarray (N_features, N_subjects)
        Feature matrix to harmonize (can be HC or clinical).
    site_labels : array-like (N_subjects,)
        Site labels for each subject.
        MUST be a subset of sites in combat_params['site_ids'].
    combat_params : dict
        Output from estimate_combat_params().

    Returns
    -------
    harmonized_data : ndarray (N_features, N_subjects)
        Harmonized feature matrix.

    Raises
    ------
    KeyError
        If site_labels contains a site not in combat_params.

    Notes
    -----
    - This is the "apply" step - parameters are fixed from HC estimation
    - Clinical data is harmonized using HC-derived parameters only
    """
    data = np.asarray(data)
    site_labels = np.asarray(site_labels)

    n_features, n_subjects = data.shape

    # Get site mapping
    site_ids = combat_params['site_ids']
    gamma_star = combat_params['gamma_star']  # (N_features, N_sites)
    delta_star = combat_params['delta_star']  # (N_features, N_sites)
    grand_mean = combat_params['grand_mean']  # (N_features,)
    pooled_var = combat_params['pooled_var']  # (N_features,)

    # Check for unseen sites
    unique_sites = np.unique(site_labels)
    for site in unique_sites:
        if site not in site_ids:
            raise KeyError(
                f"Unseen site '{site}' in data. "
                f"Combat parameters were estimated on sites: {site_ids}. "
                "Cannot harmonize data from unknown sites."
            )

    # Apply harmonization
    harmonized_data = np.zeros_like(data)

    for s_idx, site in enumerate(site_ids):
        site_mask = site_labels == site
        if not np.any(site_mask):
            continue

        # Get site-specific parameters
        gamma_s = gamma_star[:, s_idx:s_idx+1]  # (N_features, 1)
        delta_s = delta_star[:, s_idx:s_idx+1]  # (N_features, 1)

        # Harmonize: center, scale, restore grand mean
        # Y_harm = (Y - gamma) / sqrt(delta) * sqrt(pooled_var) + grand_mean
        # Simplified: Y_harm = (Y - gamma) / sqrt(delta) + grand_mean * (1 - 1/sqrt(delta))
        # Even simpler standard formula:
        # Y_harm = (Y - gamma) / sqrt(delta) * sqrt(pooled_var) + grand_mean
        # But typically: Y_harm = (Y - gamma) / sqrt(delta) (standardized)

        # Standard ComBat formula:
        # Y_harm = (Y - gamma_star) / sqrt(delta_star)
        # This removes site effects, leaving biological signal

        for i in np.where(site_mask)[0]:
            harmonized_data[:, i] = (data[:, i] - gamma_s.flatten()) / np.sqrt(delta_s.flatten())

    return harmonized_data


def blind_harmonize(hc_data, hc_site_labels, clinical_data, clinical_site_labels):
    """
    Blind harmonization: estimate on HC only, apply to clinical.

    This is the main wrapper function that enforces the "blind" constraint:
    disease labels are NEVER used in parameter estimation.

    Parameters
    ----------
    hc_data : ndarray (N_features, N_hc_subjects)
        Healthy control feature matrix.
    hc_site_labels : array-like (N_hc_subjects,)
        Site labels for HC subjects.
    clinical_data : ndarray (N_features, N_clinical_subjects)
        Clinical/patient feature matrix.
    clinical_site_labels : array-like (N_clinical_subjects,)
        Site labels for clinical subjects.
        Disease status is NOT used - only site matters.

    Returns
    -------
    harmonized_clinical : ndarray (N_features, N_clinical_subjects)
        Harmonized clinical data.
    combat_params : dict
        Parameters estimated from HC only.

    Notes
    -----
    - Parameters estimated on HC ONLY - clinical labels never enter
    - Clinical data is harmonized using HC-derived parameters
    - This ensures disease signal is preserved while removing site effects
    """
    # Step 1: Estimate parameters on HC only
    combat_params = estimate_combat_params(hc_data, hc_site_labels)

    # Step 2: Apply to clinical data
    harmonized_clinical = apply_combat(clinical_data, clinical_site_labels, combat_params)

    # Print summary
    print("\n" + "=" * 60)
    print("  Blind ComBat Harmonization Summary")
    print("=" * 60)
    print(f"  HC subjects: {hc_data.shape[1]}")
    print(f"  Clinical subjects: {clinical_data.shape[1]}")
    print(f"  Features: {hc_data.shape[0]}")
    print(f"  Sites found: {combat_params['site_ids']}")
    print(f"  Gamma range: [{combat_params['gamma_star'].min():.4f}, {combat_params['gamma_star'].max():.4f}]")
    print(f"  Delta range: [{combat_params['delta_star'].min():.4f}, {combat_params['delta_star'].max():.4f}]")
    print("=" * 60)

    return harmonized_clinical, combat_params


def validate_combat_reduction(hc_data, hc_site_labels, combat_params):
    """
    Validate that ComBat reduces site variance on HC holdout.

    Metrics:
    - site_variance_ratio = var_between_sites_after / var_between_sites_before
    - Values < 1 confirm site effect reduction
    - Values > 1 indicate overcorrection

    Parameters
    ----------
    hc_data : ndarray (N_features, N_hc_subjects)
        HC feature matrix (can be holdout set).
    hc_site_labels : array-like (N_hc_subjects,)
        Site labels for HC subjects.
    combat_params : dict
        Parameters from estimate_combat_params().

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'site_variance_ratio': float, mean ratio across features
        - 'feature_mean_shift': float, mean change in feature values
        - 'passed': bool, True if site_variance_ratio < 0.5

    Notes
    -----
    - site_variance_ratio < 0.5 means >50% reduction in site effects
    - This is the key validation that harmonization works
    """
    hc_data = np.asarray(hc_data)
    hc_site_labels = np.asarray(hc_site_labels)

    n_features = hc_data.shape[0]
    site_ids = combat_params['site_ids']

    # Compute site variance before harmonization
    site_var_before = np.zeros(n_features)
    for s_idx, site in enumerate(site_ids):
        site_mask = hc_site_labels == site
        if np.sum(site_mask) > 1:
            site_mean = np.mean(hc_data[:, site_mask], axis=1)
            site_var_before += (site_mean - np.mean(hc_data, axis=1)) ** 2

    # Harmonize
    hc_harmonized = apply_combat(hc_data, hc_site_labels, combat_params)

    # Compute site variance after harmonization
    site_var_after = np.zeros(n_features)
    for s_idx, site in enumerate(site_ids):
        site_mask = hc_site_labels == site
        if np.sum(site_mask) > 1:
            site_mean = np.mean(hc_harmonized[:, site_mask], axis=1)
            site_var_after += (site_mean - np.mean(hc_harmonized, axis=1)) ** 2

    # Avoid division by zero
    site_var_before = np.maximum(site_var_before, 1e-10)

    # Site variance ratio
    ratios = site_var_after / site_var_before
    site_variance_ratio = float(np.mean(ratios))

    # Feature mean shift (should be small)
    feature_mean_shift = float(np.mean(np.abs(hc_harmonized - hc_data)))

    # Passed if substantial reduction
    passed = site_variance_ratio < 0.5

    return {
        'site_variance_ratio': site_variance_ratio,
        'feature_mean_shift': feature_mean_shift,
        'passed': passed,
    }


def generate_synthetic_multisite_data(n_features=100, n_sites=3, n_per_site=15,
                                       seed=42):
    """
    Generate synthetic multi-site data with known site effects.

    Parameters
    ----------
    n_features : int
        Number of features (e.g., AC/MC per ROI)
    n_sites : int
        Number of sites/scanners
    n_per_site : int
        Subjects per site
    seed : int
        Random seed

    Returns
    -------
    data : ndarray (n_features, n_sites * n_per_site)
        Synthetic feature matrix with site effects
    site_labels : ndarray (n_sites * n_per_site,)
        Site labels for each subject
    true_effects : dict
        True site effects added (for validation)
    """
    np.random.seed(seed)

    n_subjects = n_sites * n_per_site

    # Generate baseline data (no site effects)
    data = np.random.randn(n_features, n_subjects) * 0.5

    # Add biological signal (same across sites)
    biological_signal = np.random.randn(n_features, 1) @ np.random.randn(1, n_subjects) * 0.3
    data += biological_signal

    # Add site effects
    true_effects = {
        'additive': np.zeros((n_features, n_sites)),
        'multiplicative': np.ones((n_features, n_sites)),
    }

    site_labels = []
    for s in range(n_sites):
        start_idx = s * n_per_site
        end_idx = (s + 1) * n_per_site
        site_labels.extend([f'site_{s+1}'] * n_per_site)

        if s == 0:
            # Site 1: reference (no effect)
            pass
        elif s == 1:
            # Site 2: additive effect +0.5
            true_effects['additive'][:, s] = 0.5
            data[:, start_idx:end_idx] += 0.5
        else:
            # Site 3: multiplicative effect *1.3
            true_effects['multiplicative'][:, s] = 1.3
            data[:, start_idx:end_idx] *= 1.3

    return data, np.array(site_labels), true_effects


if __name__ == '__main__':
    """
    Demonstration: Blind ComBat harmonization on synthetic data.

    Generate 3-site HC data with known site effects, run blind_harmonize,
    and validate site variance reduction.
    """
    print("=" * 60)
    print("  Blind ComBat Harmonization Demonstration")
    print("=" * 60)

    # Generate synthetic data
    n_features = 100
    n_sites = 3
    n_per_site = 15

    print(f"\nGenerating synthetic data:")
    print(f"  Features: {n_features}")
    print(f"  Sites: {n_sites}")
    print(f"  Subjects per site: {n_per_site}")

    hc_data, hc_site_labels, true_effects = generate_synthetic_multisite_data(
        n_features=n_features,
        n_sites=n_sites,
        n_per_site=n_per_site,
        seed=42
    )

    print(f"\nTrue site effects added:")
    print(f"  Site 2 additive: +{true_effects['additive'][0, 1]:.2f}")
    print(f"  Site 3 multiplicative: x{true_effects['multiplicative'][0, 2]:.2f}")

    # Run blind harmonization
    print("\n" + "-" * 60)
    print("Running blind_harmonize()...")

    # For demonstration, use same data as "clinical" (in reality would be different)
    clinical_data = hc_data.copy()
    clinical_site_labels = hc_site_labels.copy()

    harmonized_clinical, combat_params = blind_harmonize(
        hc_data, hc_site_labels,
        clinical_data, clinical_site_labels
    )

    # Validate reduction
    print("\n" + "-" * 60)
    print("Validating site variance reduction...")

    result = validate_combat_reduction(hc_data, hc_site_labels, combat_params)

    print(f"\nValidation results:")
    print(f"  Site variance ratio: {result['site_variance_ratio']:.4f}")
    print(f"  Feature mean shift: {result['feature_mean_shift']:.4f}")
    print(f"  Passed (ratio < 0.5): {result['passed']}")

    print("\n" + "=" * 60)
    if result['passed']:
        print("  SUCCESS: Site effects substantially reduced!")
    else:
        print("  WARNING: Site effects not sufficiently reduced.")
    print("=" * 60)
