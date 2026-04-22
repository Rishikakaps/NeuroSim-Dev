"""
metrics.py
==========
Clinical biomarker ranking functions for network control theory analysis.

This module provides tools for identifying potential therapeutic targets
and quantifying disease-related changes in brain network dynamics.

Functions:
  1. rank_facilitator_nodes: Identify "gateway" nodes for seizure propagation
     or craving circuits using modal controllability ranking.

  2. compute_attractor_rigidity: Quantify how close a patient's network
     operates to the instability threshold compared to healthy controls.

  3. group_biomarker_summary: Aggregate NCT metrics across subject groups
     with normative z-scoring relative to healthy controls.

Clinical motivation:
  - High modal controllability nodes are "facilitators" that can push
    the brain into high-energy, difficult-to-reach states
  - In epilepsy, these may be seizure propagation hubs
  - In AUD, these may be craving circuit hubs
  - Attractor rigidity quantifies reduced network flexibility in disease

Known limitations:
  - Assumes A matrix is accurately estimated from data
  - Z-scores depend on control group size and homogeneity
  - Does not account for individual anatomical variability

Parameters:
  A: (N, N) effective connectivity matrix (asymmetric, stable)
  control_A: (N, N) healthy control template or mean matrix

Returns:
  Varies by function - see individual docstrings
"""

import numpy as np
import pandas as pd
import os
from scipy.linalg import solve_discrete_lyapunov
import warnings

# Import from sibling module
from neurosim.control.energy import minimum_energy


def rank_facilitator_nodes(A, top_k=10):
    """
    Rank nodes by modal controllability to identify facilitator hubs.

    Modal controllability measures a node's ability to push the brain
    into difficult, high-energy states. High MC nodes are "facilitators"
    or "gateways" for pathological dynamics:
      - In epilepsy: seizure propagation hubs
      - In AUD: craving circuit hubs
      - In depression: rumination network hubs

    Formula:
      MC_i = sum_j (1 - |lambda_j|^2) * |v_ij|^2
      where lambda_j = eigenvalues of A, v_ij = eigenvector components

    Parameters
    ----------
    A : ndarray (N, N)
        Effective connectivity matrix (asymmetric, spectral radius < 1)
    top_k : int, default=10
        Number of top facilitator nodes to return

    Returns
    -------
    top_node_indices : ndarray (top_k,)
        Indices of top-k facilitator nodes (sorted descending by MC)
    mc_scores : ndarray (top_k,)
        Modal controllability scores for top-k nodes

    Notes
    -----
    - MC weights are (1 - |lambda|^2), so eigenvalues with |lambda| < 1
      contribute positively (stable systems)
    - High MC = node can access high-energy modes = potential therapeutic target
    """
    N = A.shape[0]

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Modal controllability weights: (1 - |lambda_j|^2)
    # For stable A, |lambda| < 1, so weights are positive
    weights = 1 - np.abs(eigenvalues) ** 2

    # Check for instability warning
    if np.any(weights < 0):
        n_neg = np.sum(weights < 0)
        warnings.warn(
            f"{n_neg} eigenvalues have |lambda| > 1 (unstable modes). "
            "Clipping weights to 0. Check that A is properly normalized."
        )
        weights = np.clip(weights, 0, None)

    # MC_i = sum_j weights[j] * |V[i,j]|^2
    V_sq = np.abs(eigenvectors) ** 2  # element-wise squared magnitude
    mc = V_sq @ weights  # (N,)

    # Sort descending and get top-k
    sorted_idx = np.argsort(mc)[::-1]  # descending
    top_indices = sorted_idx[:top_k]
    top_scores = mc[top_indices]

    return top_indices, top_scores


def compute_attractor_rigidity(A_patient, A_control_mean):
    """
    Quantify attractor rigidity in patient vs control networks.

    Attractor rigidity has two components:

    1. Spectral rigidity: How close does the network operate to instability?
       rigidity_spectral = rho(A_patient) - rho(A_control)
       Positive = patient network is closer to instability threshold

    2. Energy rigidity: How much more energy is needed for state transitions?
       delta_E = E_patient - E_control
       Positive = patient needs more energy = more rigid attractor state

    Clinical interpretation:
      - In AUD: rigid attractors = craving states are hard to escape
      - In depression: rigid attractors = negative mood states persist
      - In epilepsy: rigid attractors = seizure states easily triggered

    Parameters
    ----------
    A_patient : ndarray (N, N)
        Patient effective connectivity matrix
    A_control_mean : ndarray (N, N)
        Healthy control mean/template connectivity matrix

    Returns
    -------
    rigidity_score : float
        spectral_radius(A_patient) - spectral_radius(A_control_mean)
        Positive = patient operates closer to instability
    delta_E : float
        E_patient - E_control for B=I, T=20, x0=0, xf=ones/N
        Positive = patient needs more energy for same transition

    Notes
    -----
    - Uses uniform target state xf = ones/N (activate all ROIs equally)
    - Energy computed with B=I (all nodes controllable), T=20
    - rigidity_score > 0 suggests reduced network stability margin
    """
    N = A_patient.shape[0]

    # Spectral radii
    rho_patient = np.max(np.abs(np.linalg.eigvals(A_patient)))
    rho_control = np.max(np.abs(np.linalg.eigvals(A_control_mean)))

    rigidity_score = rho_patient - rho_control

    # Energy comparison: B=I, T=20, x0=0, xf=ones/N
    B = np.eye(N)
    T = 20
    x0 = np.zeros(N)
    xf = np.ones(N) / N  # Uniform activation target

    try:
        E_patient, _ = minimum_energy(A_patient, T, B, x0, xf)
    except Exception as e:
        warnings.warn(f"Energy computation failed for patient: {e}")
        E_patient = np.nan

    try:
        E_control, _ = minimum_energy(A_control_mean, T, B, x0, xf)
    except Exception as e:
        warnings.warn(f"Energy computation failed for control: {e}")
        E_control = np.nan

    delta_E = E_patient - E_control

    return rigidity_score, delta_E


def group_biomarker_summary(nct_dir, group_ids, control_ids, focus_nodes=None):
    """
    Compute group-level biomarker summary with normative z-scores.

    For each subject in group_ids:
    1. Load NCT CSV (ROI x metrics)
    2. Compute mean AC and MC at focus_nodes (or all ROIs if None)
    3. Compute z-scores relative to control group

    Parameters
    ----------
    nct_dir : str
        Path to directory containing NCT CSV files
    group_ids : list of str
        Subject IDs for the patient group (e.g., ['aud_1', 'aud_2', ...])
    control_ids : list of str
        Subject IDs for healthy control group
    focus_nodes : list of int or None
        ROI indices to focus on (e.g., DMN nodes, seizure focus)
        If None, use all ROIs

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
        - subject: str, subject ID
        - group: str, 'patient' or 'control'
        - AC_mean: float, mean average controllability at focus_nodes
        - MC_mean: float, mean modal controllability at focus_nodes
        - AC_zscore: float, z-score relative to controls
        - MC_zscore: float, z-score relative to controls
        - is_facilitator: bool, True if MC_zscore > 2 (high MC outlier)

    Notes
    -----
    - Z-scores computed as (X - mean_control) / std_control
    - is_facilitator flags subjects with MC_zscore > 2 (97.5th percentile)
    - focus_nodes should be chosen based on hypothesis (e.g., DMN for AD)
    """
    results = []

    # Load control data first for z-score computation
    control_ac_means = []
    control_mc_means = []

    for ctrl_id in control_ids:
        nct_path = os.path.join(nct_dir, f"{ctrl_id}_NCT.csv")
        if not os.path.exists(nct_path):
            warnings.warn(f"NCT file not found: {nct_path}")
            continue

        df_ctrl = pd.read_csv(nct_path)

        if focus_nodes is not None:
            ac_mean = df_ctrl['AverageControllability'].iloc[focus_nodes].mean()
            mc_mean = df_ctrl['ModalControllability'].iloc[focus_nodes].mean()
        else:
            ac_mean = df_ctrl['AverageControllability'].mean()
            mc_mean = df_ctrl['ModalControllability'].mean()

        control_ac_means.append(ac_mean)
        control_mc_means.append(mc_mean)

        results.append({
            'subject': ctrl_id,
            'group': 'control',
            'AC_mean': ac_mean,
            'MC_mean': mc_mean,
        })

    # Compute control statistics for z-scores
    if len(control_ac_means) > 1:
        ac_mean_ctrl = np.mean(control_ac_means)
        ac_std_ctrl = np.std(control_ac_means, ddof=1)
        mc_mean_ctrl = np.mean(control_mc_means)
        mc_std_ctrl = np.std(control_mc_means, ddof=1)
    else:
        warnings.warn("Only one control subject - z-scores will be unreliable")
        ac_mean_ctrl = control_ac_means[0] if control_ac_means else 0
        ac_std_ctrl = 1.0  # Default to prevent division by zero
        mc_mean_ctrl = control_mc_means[0] if control_mc_means else 0
        mc_std_ctrl = 1.0

    # Load patient data
    for pat_id in group_ids:
        nct_path = os.path.join(nct_dir, f"{pat_id}_NCT.csv")
        if not os.path.exists(nct_path):
            warnings.warn(f"NCT file not found: {nct_path}")
            continue

        df_pat = pd.read_csv(nct_path)

        if focus_nodes is not None:
            ac_mean = df_pat['AverageControllability'].iloc[focus_nodes].mean()
            mc_mean = df_pat['ModalControllability'].iloc[focus_nodes].mean()
        else:
            ac_mean = df_pat['AverageControllability'].mean()
            mc_mean = df_pat['ModalControllability'].mean()

        # Compute z-scores
        ac_zscore = (ac_mean - ac_mean_ctrl) / ac_std_ctrl if ac_std_ctrl > 0 else 0
        mc_zscore = (mc_mean - mc_mean_ctrl) / mc_std_ctrl if mc_std_ctrl > 0 else 0

        results.append({
            'subject': pat_id,
            'group': 'patient',
            'AC_mean': ac_mean,
            'MC_mean': mc_mean,
            'AC_zscore': ac_zscore,
            'MC_zscore': mc_zscore,
            'is_facilitator': mc_zscore > 2,
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    """
    Demonstration: Run biomarker analysis on NCT outputs from pipeline.

    This requires that the pipeline has been run and NCT CSVs exist.
    """
    print("=" * 60)
    print("  Clinical Biomarker Summary")
    print("=" * 60)

    # Check if NCT outputs exist
    nct_dir = 'outputs/NCT'

    if not os.path.exists(nct_dir):
        print(f"\nNCT directory not found: {nct_dir}")
        print("Run the pipeline first: python src/run_pipeline.py")
        print("\n" + "=" * 60)
    else:
        # List available NCT files
        nct_files = [f for f in os.listdir(nct_dir) if f.endswith('_NCT.csv')]
        print(f"\nFound {len(nct_files)} NCT files in {nct_dir}")

        # Example: synthetic data groups
        control_ids = [f.replace('_NCT.csv', '') for f in nct_files if 'control' in f]
        aud_ids = [f.replace('_NCT.csv', '') for f in nct_files if 'aud' in f]
        epilepsy_ids = [f.replace('_NCT.csv', '') for f in nct_files if 'epilepsy' in f]
        ad_ids = [f.replace('_NCT.csv', '') for f in nct_files if 'ad' in f]

        print(f"  Controls: {len(control_ids)}")
        print(f"  AUD: {len(aud_ids)}")
        print(f"  Epilepsy: {len(epilepsy_ids)}")
        print(f"  AD: {len(ad_ids)}")

        if len(control_ids) > 0:
            # Example: AUD analysis with DMN focus (nodes 0-4)
            print("\n" + "-" * 60)
            print("AUD Biomarker Summary (DMN focus: nodes 0-4):")

            dmn_nodes = [0, 1, 2, 3, 4]
            df_aud = group_biomarker_summary(nct_dir, aud_ids, control_ids, dmn_nodes)

            if len(df_aud) > 0:
                print(df_aud.to_string(index=False))

                # Summary statistics
                print("\nAUD vs Control differences:")
                aud_mc_z = df_aud[df_aud['group'] == 'patient']['MC_zscore'].mean()
                print(f"  Mean AUD MC z-score: {aud_mc_z:.2f}")
                print(f"  Facilitator subjects: {df_aud['is_facilitator'].sum()}")

        if len(epilepsy_ids) > 0:
            # Example: Epilepsy analysis with seizure focus (nodes 5-7)
            print("\n" + "-" * 60)
            print("Epilepsy Biomarker Summary (Seizure focus: nodes 5-7):")

            seizure_focus = [5, 6, 7]
            df_epi = group_biomarker_summary(nct_dir, epilepsy_ids, control_ids, seizure_focus)

            if len(df_epi) > 0:
                print(df_epi.to_string(index=False))

                # Summary statistics
                print("\nEpilepsy vs Control differences:")
                epi_mc_z = df_epi[df_epi['group'] == 'patient']['MC_zscore'].mean()
                print(f"  Mean Epilepsy MC z-score: {epi_mc_z:.2f}")
                print(f"  Facilitator subjects: {df_epi['is_facilitator'].sum()}")

        print("\n" + "=" * 60)
