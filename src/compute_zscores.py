"""
compute_zscores.py
==================
Computes normative z-scores for patient subjects relative to control reference.

z = (patient_value - mean_controls) / std_controls

This is the simplest valid normative model (as specified in ai_rules.md).
The limitation (small control N) is acknowledged in the paper's limitations.

Output per patient: ROI-level z-scores for each NCT metric.
Also saves the normative reference (mean, std per ROI per metric) as CSV.
"""

import numpy as np
import pandas as pd
import os


METRICS = ['AverageControllability', 'ModalControllability', 'MinControlEnergy']


def load_nct(nct_dir, subject_id):
    """Load NCT CSV for one subject. Returns dict of metric → np.array(N)."""
    path = os.path.join(nct_dir, f"{subject_id}_NCT.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NCT file not found: {path}")
    df = pd.read_csv(path)
    return {m: df[m].values for m in METRICS}


def compute_normative_reference(control_ids, nct_dir):
    all_data = {m: [] for m in METRICS}
    for cid in control_ids:
        data = load_nct(nct_dir, cid)
        for m in METRICS:
            all_data[m].append(data[m])

    norms = {}
    for m in METRICS:
        arr = np.array(all_data[m])          # (n_controls, N)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)        # Bessel-corrected, unbiased

        # Flag unreliable ROIs (std too small relative to mean)
        flag = std < 0.02 * np.abs(mean) + 1e-8
        std_safe = std.copy()
        std_safe[flag] = 0.02 * np.abs(mean[flag]) + 1e-8

        n_flagged = flag.sum()
        if n_flagged > 0:
            print(f"  ⚠ {m}: {n_flagged} ROIs flagged as unreliable (std too small).")

        norms[m] = {'mean': mean, 'std': std_safe, 'flag': flag}
        print(f"  {m}: mean range=[{mean.min():.4f}, {mean.max():.4f}]  "
              f"std range=[{std_safe.min():.4f}, {std_safe.max():.4f}]")
    return norms


def save_normative_reference(norms, output_dir):
    """Save normative reference (mean, std per ROI per metric) to CSV."""
    n_rois = len(norms[METRICS[0]]['mean'])
    df = pd.DataFrame({'ROI': np.arange(n_rois)})
    for m in METRICS:
        df[f'{m}_mean'] = norms[m]['mean']
        df[f'{m}_std'] = norms[m]['std']
    path = os.path.join(output_dir, 'normative_reference.csv')
    df.to_csv(path, index=False)
    print(f"  Normative reference saved: {path}")


def compute_patient_zscores(patient_ids, control_ids, nct_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    norms = compute_normative_reference(control_ids, nct_dir)
    save_normative_reference(norms, output_dir)

    summary_rows = []
    for pid in patient_ids:
        data = load_nct(nct_dir, pid)
        n_rois = len(data[METRICS[0]])
        z_df = pd.DataFrame({'ROI': np.arange(n_rois)})

        for m in METRICS:
            z = (data[m] - norms[m]['mean']) / norms[m]['std']
            flag = norms[m]['flag']
            z[flag] = np.nan               # mark unreliable as NaN

            z_df[f'{m}_zscore'] = z
            z_df[f'{m}_flag'] = flag.astype(int)

            reliable = ~flag
            n_anomalous = np.sum(np.abs(z[reliable]) > 2)
            mean_abs_z = np.nanmean(np.abs(z))
            print(f"  {m}: mean|z|={mean_abs_z:.4f}  ROIs with |z|>2 (reliable only): {n_anomalous}/{reliable.sum()}")

            summary_rows.append({
                'subject': pid, 'metric': m,
                'mean_abs_z': mean_abs_z,
                'n_anomalous_rois': n_anomalous
            })

        out_path = os.path.join(output_dir, f"{pid}_zscores.csv")
        z_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'zscore_summary.csv'), index=False)
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    control_ids = ['control_1', 'control_2']
    patient_ids = ['patient_1', 'patient_2', 'patient_3']
    compute_patient_zscores(patient_ids, control_ids, 'outputs/NCT', 'outputs/zscores')
