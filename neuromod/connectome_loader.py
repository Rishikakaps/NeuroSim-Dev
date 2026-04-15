"""
connectome_loader.py
====================
Utilities for loading and harmonizing neuroimaging data.

Functions:
  1. load_bids_timeseries: Extract ROI timeseries from fMRIPrep output
  2. run_neurocombat: Cross-site harmonization using neuroCombat
  3. qc_filter_subjects: Quality control by framewise displacement
"""

import numpy as np
import pandas as pd
import os

# Optional imports with graceful fallback
try:
    from nilearn.maskers import NiftiLabelsMasker
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    print("Install nilearn to use load_bids_timeseries: pip install nilearn")

try:
    from neurocombat import neuroCombat
    HAS_NEUROCOMBAT = True
except ImportError:
    HAS_NEUROCOMBAT = False
    print("Install neurocombat to use run_neurocombat: pip install neurocombat")


def load_bids_timeseries(bids_root, subject_list, atlas='schaefer200'):
    """
    Extract ROI timeseries from fMRIPrep preprocessed BOLD data.

    Expects fMRIPrep output naming convention:
        {bids_root}/sub-{sid}/func/sub-{sid}_task-{task}_desc-preproc_bold.nii.gz

    Parameters:
        bids_root: path to BIDS dataset root
        subject_list: list of subject IDs (e.g., ['001', '002'])
        atlas: atlas name for nilearn (default 'schaefer200')

    Returns:
        dict: {subject_id: np.ndarray of shape (T, N)}

    Raises:
        ImportError: if nilearn is not installed
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required. Install with: pip install nilearn")

    # Load atlas
    try:
        from nilearn import datasets
        if atlas == 'schaefer200':
            # Use Schaefer 2018 atlas (200 parcels, 7 networks)
            atlas_obj = datasets.fetch_schaefer_atlas_2018()
            labels_img = atlas_obj['maps']
        else:
            raise ValueError(f"Unknown atlas: {atlas}")
    except Exception as e:
        raise RuntimeError(f"Failed to load atlas '{atlas}': {e}")

    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=True, detrend=True)

    timeseries_dict = {}

    for subj in subject_list:
        # Find preproc BOLD file
        subj_dir = os.path.join(bids_root, f'sub-{subj}')
        func_dir = os.path.join(subj_dir, 'func')

        if not os.path.exists(func_dir):
            print(f"  ⚠ No func directory for sub-{subj}, skipping")
            continue

        # Find desc-preproc_bold file
        bold_files = [f for f in os.listdir(func_dir) if 'desc-preproc_bold.nii.gz' in f]
        if not bold_files:
            print(f"  ⚠ No preproc BOLD found for sub-{subj}, skipping")
            continue

        # Take first task (or could iterate over tasks)
        bold_path = os.path.join(func_dir, bold_files[0])
        print(f"  Loading sub-{subj}: {bold_files[0]}")

        # Extract timeseries
        ts = masker.fit_transform(bold_path)
        timeseries_dict[subj] = ts

    print(f"  Loaded {len(timeseries_dict)} subjects")
    return timeseries_dict


def run_neurocombat(data_matrix, batch_vector):
    """
    Harmonize multi-site data using neuroCombat.

    Removes batch effects while preserving biological variance.

    Parameters:
        data_matrix: np.ndarray of shape (N_features × N_subjects)
                     Note: neuroCombat expects features × samples
        batch_vector: array-like of shape (N_subjects,) with site labels

    Returns:
        harmonized_matrix: np.ndarray of same shape as data_matrix

    Raises:
        ImportError: if neurocombat is not installed
    """
    if not HAS_NEUROCOMBAT:
        raise ImportError("neurocombat required. Install with: pip install neurocombat")

    # Ensure 2D array
    if data_matrix.ndim == 1:
        data_matrix = data_matrix.reshape(1, -1)

    # neuroCombat expects dict with 'batch' key
    model_matrix = {'batch': np.asarray(batch_vector)}

    # Run harmonization
    combat_model = neuroCombat(data_matrix, model_matrix)
    harmonized = combat_model['data']

    print(f"  NeuroCombat: harmonized {harmonized.shape[1]} subjects across "
          f"{len(np.unique(batch_vector))} sites")

    return harmonized


def qc_filter_subjects(timeseries_dict, fd_threshold=0.5, min_volumes=200):
    """
    Filter subjects by motion and scan length quality metrics.

    Parameters:
        timeseries_dict: dict {subject_id: np.ndarray of shape (T, N)}
        fd_threshold: maximum mean framewise displacement (default 0.5 mm)
        min_volumes: minimum number of volumes after scrubbing (default 200)

    Returns:
        (filtered_dict, qc_report_df):
            - filtered_dict: subjects passing QC
            - qc_report_df: DataFrame with QC metrics for all subjects
    """
    qc_records = []

    for subj, ts in timeseries_dict.items():
        T, N = ts.shape

        # Estimate framewise displacement from timeseries
        # FD = sum of absolute differences across consecutive volumes
        # This is a simplified FD estimate (true FD requires motion parameters)
        fd = np.abs(np.diff(ts, axis=0)).sum(axis=1)
        mean_fd = fd.mean()

        # Count volumes with FD < threshold
        good_volumes = np.sum(fd < fd_threshold)

        qc_records.append({
            'subject': subj,
            'n_volumes': T,
            'n_rois': N,
            'mean_fd': mean_fd,
            'good_volumes': good_volumes,
            'passed_fd': mean_fd < fd_threshold,
            'passed_length': good_volumes >= min_volumes
        })

    qc_df = pd.DataFrame(qc_records)
    qc_df['passed_qc'] = qc_df['passed_fd'] & qc_df['passed_length']

    # Filter to passing subjects
    passed_subjects = qc_df[qc_df['passed_qc']]['subject'].tolist()
    filtered_dict = {subj: timeseries_dict[subj] for subj in passed_subjects}

    print(f"  QC: {len(passed_subjects)} / {len(timeseries_dict)} subjects passed")
    print(f"    - FD threshold: {fd_threshold} mm ({qc_df['passed_fd'].sum()} passed)")
    print(f"    - Min volumes: {min_volumes} ({qc_df['passed_length'].sum()} passed)")

    return filtered_dict, qc_df
