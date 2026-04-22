"""
validate_ec_recovery.py
=======================
Validate EC (Effective Connectivity) estimation recovery against ground-truth.

Method:
  For each subject in each cohort:
    A_true = load ground-truth VAR(1) connectivity matrix
    A_est  = load estimated EC matrix from VAR(1) OLS

  Compute normalized Frobenius error:
    normalized_frobenius_error = ||A_est - A_true||_F / ||A_true||_F

  This metric quantifies how well the VAR(1) OLS estimation recovers
  the true directed connectivity pattern.

Reference:
  Gilson et al. (2016), "From Time Series to Network Structure:
  A Comparative Study of Methods for Estimating Effective Connectivity",
  PLOS Computational Biology. This paper establishes normalized Frobenius
  error as the standard metric for validating EC recovery.

Known limitations:
  - Normalization by ||A_true||_F assumes A_true is non-zero.
    For near-zero ground-truth matrices, the metric is ill-defined.
  - Does not account for edge-wise false positive/negative rates.
  - Spectral radius comparison provides additional stability check.

Parameters:
  ec_dir: str, path to directory containing estimated EC CSV files
  gt_dir: str, path to directory containing ground-truth A CSV files

Returns:
  pd.DataFrame with columns:
    - subject: str, subject identifier
    - cohort: str, group label (control, aud, epilepsy, ad)
    - frobenius_error: float, normalized recovery error
    - spectral_radius_true: float, rho(A_true)
    - spectral_radius_est: float, rho(A_est)
    - rho_error: float, |rho(A_est) - rho(A_true)|
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


COHORTS = ['control', 'aud', 'epilepsy', 'ad']
N_SUBJECTS = 5


def compute_ec_recovery(ec_dir, gt_dir) -> pd.DataFrame:
    """
    Compute EC recovery metrics for all subjects across all cohorts.

    For each subject, loads the ground-truth A matrix and estimated EC
    matrix, then computes:
      1. Normalized Frobenius error: ||A_est - A_true||_F / ||A_true||_F
      2. Spectral radius of true and estimated matrices
      3. Spectral radius error: |rho_est - rho_true|

    Parameters
    ----------
    ec_dir : str
        Path to directory containing estimated EC CSV files.
        Files should be named: {cohort}_{i}_EC.csv
    gt_dir : str
        Path to directory containing ground-truth A CSV files.
        Files should be named: A_{cohort}_{i}.csv

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
        - subject: str, subject identifier (e.g., 'control_1')
        - cohort: str, group label (control, aud, epilepsy, ad)
        - frobenius_error: float, normalized recovery error
        - spectral_radius_true: float, spectral radius of A_true
        - spectral_radius_est: float, spectral radius of A_est
        - rho_error: float, absolute difference in spectral radii

    Raises
    ------
    FileNotFoundError
        If expected EC or ground-truth files are missing

    Notes
    -----
    - Frobenius norm: ||A||_F = sqrt(sum_{i,j} |a_ij|^2)
    - Spectral radius: rho(A) = max|lambda(A)|
    - Normalization by ||A_true||_F makes error comparable across scales
    """
    results = []

    for cohort in COHORTS:
        for i in range(1, N_SUBJECTS + 1):
            subject = f'{cohort}_{i}'

            # Load ground-truth matrix
            gt_path = os.path.join(gt_dir, f'A_{subject}.csv')
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
            A_true = np.loadtxt(gt_path, delimiter=',')

            # Load estimated EC matrix
            ec_path = os.path.join(ec_dir, f'{subject}_EC.csv')
            if not os.path.exists(ec_path):
                raise FileNotFoundError(f"EC file not found: {ec_path}")
            A_est = np.loadtxt(ec_path, delimiter=',')

            # Compute normalized Frobenius error
            frobenius_norm_true = np.linalg.norm(A_true, 'fro')
            frobenius_norm_diff = np.linalg.norm(A_est - A_true, 'fro')

            if frobenius_norm_true < 1e-10:
                # Near-zero ground truth: metric ill-defined
                frobenius_error = np.nan
            else:
                frobenius_error = frobenius_norm_diff / frobenius_norm_true

            # Compute spectral radii
            eigvals_true = np.linalg.eigvals(A_true)
            eigvals_est = np.linalg.eigvals(A_est)
            rho_true = float(np.max(np.abs(eigvals_true)))
            rho_est = float(np.max(np.abs(eigvals_est)))
            rho_error = abs(rho_est - rho_true)

            results.append({
                'subject': subject,
                'cohort': cohort,
                'frobenius_error': frobenius_error,
                'spectral_radius_true': rho_true,
                'spectral_radius_est': rho_est,
                'rho_error': rho_error,
            })

    df = pd.DataFrame(results)

    # Print per-cohort summary
    print("\n" + "=" * 70)
    print("  EC RECOVERY VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Cohort':<12} | {'Mean Frob Err':<14} | {'Std Frob Err':<13} | {'Mean Rho Err':<12}")
    print("  " + "-" * 60)

    for cohort in COHORTS:
        cohort_data = df[df['cohort'] == cohort]
        mean_frob = cohort_data['frobenius_error'].mean()
        std_frob = cohort_data['frobenius_error'].std()
        mean_rho = cohort_data['rho_error'].mean()

        # Handle NaN values in summary
        mean_frob_str = f"{mean_frob:.4f}" if not np.isnan(mean_frob) else "NaN"
        std_frob_str = f"{std_frob:.4f}" if not np.isnan(std_frob) else "NaN"
        mean_rho_str = f"{mean_rho:.4f}" if not np.isnan(mean_rho) else "NaN"

        print(f"  {cohort:<12} | {mean_frob_str:<14} | {std_frob_str:<13} | {mean_rho_str:<12}")

    print("  " + "=" * 66)

    # Overall summary
    overall_mean_frob = df['frobenius_error'].mean()
    overall_mean_rho = df['rho_error'].mean()
    print(f"\n  Overall mean Frobenius error: {overall_mean_frob:.4f}")
    print(f"  Overall mean spectral radius error: {overall_mean_rho:.4f}")
    print("=" * 70)

    return df


def main():
    """
    Run EC recovery validation.

    If ground-truth data is missing, runs generate_synthetic first.
    Saves results to outputs/ec_recovery_validation.csv.
    """
    gt_dir = 'data/ground_truth'
    ec_dir = 'outputs/EC'

    # Check if ground-truth data exists
    if not os.path.exists(gt_dir) or len(os.listdir(gt_dir)) == 0:
        print("Ground-truth directory empty or missing.")
        print("Running generate_synthetic.py to create synthetic data...")

        from generate_synthetic import main as generate_main
        generate_main()

    # Check if EC estimates exist
    if not os.path.exists(ec_dir) or len(os.listdir(ec_dir)) == 0:
        print("\nEC output directory empty or missing.")
        print("Run the pipeline first: python src/run_pipeline.py")
        print("\n" + "=" * 70)
        return

    # Run validation
    print("=" * 70)
    print("  EC RECOVERY VALIDATION")
    print("=" * 70)
    print(f"\nGround-truth directory: {gt_dir}")
    print(f"EC estimates directory: {ec_dir}")

    df = compute_ec_recovery(ec_dir, gt_dir)

    # Save results
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/ec_recovery_validation.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Verify no NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"\nWARNING: {nan_count} NaN values detected in results!")
    else:
        print(f"\nValidation complete: {len(df)} subjects, 0 NaN values")


if __name__ == '__main__':
    main()
