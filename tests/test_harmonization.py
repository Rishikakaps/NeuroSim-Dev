"""
test_harmonization.py
=====================
Unit tests for the Blind neuroCombat harmonization module.

Tests verify:
  1. Blind harmonization does not use clinical labels
     (combat_params invariant to clinical_site_labels shuffling)
  2. apply_combat raises KeyError for unseen sites
  3. Site variance reduces after harmonization (validate_combat_reduction passes)

Run with:
    python -m pytest tests/test_harmonization.py -v
    OR
    python tests/test_harmonization.py
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurosim.harmonization.combat import (
    estimate_combat_params,
    apply_combat,
    blind_harmonize,
    validate_combat_reduction,
    generate_synthetic_multisite_data,
)


# =============================================================================
# Test 1: Blind Does Not Use Clinical Labels
# =============================================================================
def test_blind_does_not_use_clinical_labels():
    """
    Calling blind_harmonize with randomized clinical_site_labels must produce
    identical combat_params. Parameters must be invariant to clinical labels.

    This verifies the "blind" constraint: disease labels never enter estimation.
    """
    np.random.seed(42)

    # Generate HC data
    hc_data, hc_site_labels, _ = generate_synthetic_multisite_data(
        n_features=50,
        n_sites=3,
        n_per_site=10,
        seed=42
    )

    # Generate clinical data (same structure, different values)
    np.random.seed(123)
    clinical_data = np.random.randn(50, 30) * 0.5
    clinical_site_labels_original = np.array(['site_1'] * 10 + ['site_2'] * 10 + ['site_3'] * 10)

    # Shuffle clinical site labels randomly
    np.random.seed(999)
    clinical_site_labels_shuffled = clinical_site_labels_original.copy()
    np.random.shuffle(clinical_site_labels_shuffled)

    # Run blind harmonization with original labels
    _, params_original = blind_harmonize(
        hc_data, hc_site_labels,
        clinical_data, clinical_site_labels_original
    )

    # Run blind harmonization with shuffled labels
    _, params_shuffled = blind_harmonize(
        hc_data, hc_site_labels,
        clinical_data, clinical_site_labels_shuffled
    )

    # Combat params should be IDENTICAL (estimated from HC only)
    assert np.allclose(params_original['gamma_star'], params_shuffled['gamma_star']), (
        "combat_params['gamma_star'] changed when clinical labels were shuffled! "
        "This violates the 'blind' constraint."
    )

    assert np.allclose(params_original['delta_star'], params_shuffled['delta_star']), (
        "combat_params['delta_star'] changed when clinical labels were shuffled! "
        "This violates the 'blind' constraint."
    )

    assert np.allclose(params_original['grand_mean'], params_shuffled['grand_mean']), (
        "combat_params['grand_mean'] changed when clinical labels were shuffled! "
        "This violates the 'blind' constraint."
    )

    assert np.allclose(params_original['pooled_var'], params_shuffled['pooled_var']), (
        "combat_params['pooled_var'] changed when clinical labels were shuffled! "
        "This violates the 'blind' constraint."
    )

    assert params_original['site_ids'] == params_shuffled['site_ids'], (
        "combat_params['site_ids'] changed when clinical labels were shuffled! "
        "This violates the 'blind' constraint."
    )

    print("  ✓ blind_harmonize produces identical params regardless of clinical labels")


# =============================================================================
# Test 2: Apply ComBat Unseen Site Raises
# =============================================================================
def test_apply_combat_unseen_site_raises():
    """
    apply_combat with an unseen site must raise KeyError.

    This ensures we don't silently harmonize data from unknown scanners/sites.
    """
    np.random.seed(42)

    # Generate HC data with 3 sites
    hc_data, hc_site_labels, _ = generate_synthetic_multisite_data(
        n_features=20,
        n_sites=3,
        n_per_site=5,
        seed=42
    )

    # Estimate parameters
    combat_params = estimate_combat_params(hc_data, hc_site_labels)

    # Create data with unseen site
    new_data = np.random.randn(20, 5)
    new_site_labels = np.array(['unknown_site'] * 5)

    # Should raise KeyError
    try:
        harmonized = apply_combat(new_data, new_site_labels, combat_params)
        assert False, (
            "apply_combat should have raised KeyError for unseen site 'unknown_site'"
        )
    except KeyError as e:
        assert 'unknown_site' in str(e), (
            f"KeyError should mention the unseen site, got: {e}"
        )
        print(f"  ✓ apply_combat correctly raises KeyError for unseen site: {e}")


# =============================================================================
# Test 3: Site Variance Reduces
# =============================================================================
def test_site_variance_reduces():
    """
    validate_combat_reduction must return passed=True on synthetic data
    with known site effects.

    Site variance ratio < 0.5 means >50% reduction in site effects.
    """
    np.random.seed(42)

    # Generate data with strong site effects
    hc_data, hc_site_labels, true_effects = generate_synthetic_multisite_data(
        n_features=100,
        n_sites=3,
        n_per_site=15,
        seed=42
    )

    # Estimate parameters
    combat_params = estimate_combat_params(hc_data, hc_site_labels)

    # Validate reduction
    result = validate_combat_reduction(hc_data, hc_site_labels, combat_params)

    assert result['passed'], (
        f"validate_combat_reduction should pass for data with known site effects. "
        f"Got site_variance_ratio = {result['site_variance_ratio']:.4f} (need < 0.5)"
    )

    assert result['site_variance_ratio'] < 0.5, (
        f"Site variance ratio should be < 0.5, got {result['site_variance_ratio']:.4f}"
    )

    print(f"  ✓ Site variance reduced: ratio = {result['site_variance_ratio']:.4f} < 0.5")


# =============================================================================
# Additional Tests
# =============================================================================
def test_estimate_combat_params_minimum_subjects():
    """
    estimate_combat_params should raise ValueError if any site has < 2 subjects.
    """
    np.random.seed(42)

    # Create data with one site having only 1 subject
    hc_data = np.random.randn(10, 5)
    hc_site_labels = np.array(['site_a', 'site_a', 'site_a', 'site_a', 'site_b'])

    try:
        params = estimate_combat_params(hc_data, hc_site_labels)
        assert False, (
            "estimate_combat_params should raise ValueError for site with < 2 subjects"
        )
    except ValueError as e:
        assert 'site_b' in str(e) or '2 subjects' in str(e), (
            f"ValueError should mention the minimum subject requirement, got: {e}"
        )
        print(f"  ✓ estimate_combat_params correctly rejects site with < 2 subjects")


def test_harmonization_preserves_biological_signal():
    """
    Harmonization should remove site effects while preserving biological signal.
    """
    np.random.seed(42)

    n_features = 50
    n_sites = 3
    n_per_site = 20
    n_subjects = n_sites * n_per_site

    # Generate data with biological signal and site effects
    np.random.seed(42)

    # Biological signal (same across all subjects)
    biological_signal = np.random.randn(n_features, 1) @ np.random.randn(1, n_subjects) * 0.5

    # Site effects
    site_labels = np.array(['site_1'] * n_per_site + ['site_2'] * n_per_site + ['site_3'] * n_per_site)
    data = biological_signal.copy()

    for s in range(n_sites):
        start_idx = s * n_per_site
        end_idx = (s + 1) * n_per_site
        if s == 1:
            data[:, start_idx:end_idx] += 0.5  # additive effect
        elif s == 2:
            data[:, start_idx:end_idx] *= 1.2  # multiplicative effect

    # Estimate and apply
    combat_params = estimate_combat_params(data, site_labels)
    harmonized = apply_combat(data, site_labels, combat_params)

    # Check that biological signal correlation is preserved
    # (harmonized data should still correlate with original biological signal)
    for i in range(min(5, n_subjects)):
        corr_before = np.corrcoef(biological_signal[:, 0], data[:, i])[0, 1]
        corr_after = np.corrcoef(biological_signal[:, 0], harmonized[:, i])[0, 1]

        # Correlation should be maintained (not necessarily identical, but not destroyed)
        # Strong correlation (positive or negative) means signal is preserved
        # Check that |correlation| is maintained, not that it's positive
        assert abs(corr_after) > 0.3, (
            f"Biological signal correlation destroyed: before={corr_before:.3f}, after={corr_after:.3f}"
        )

    print("  ✓ Harmonization preserves biological signal structure")


# =============================================================================
# Run Tests
# =============================================================================
def run_tests():
    """Run all tests and print results."""
    print("=" * 60)
    print("  NeuroSim Blind neuroCombat Test Suite")
    print("=" * 60)

    tests = [
        ("Blind does not use clinical labels", test_blind_does_not_use_clinical_labels),
        ("Apply ComBat unseen site raises", test_apply_combat_unseen_site_raises),
        ("Site variance reduces", test_site_variance_reduces),
        ("Minimum subjects per site", test_estimate_combat_params_minimum_subjects),
        ("Preserves biological signal", test_harmonization_preserves_biological_signal),
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
