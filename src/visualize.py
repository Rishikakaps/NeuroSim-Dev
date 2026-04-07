"""
visualize.py
============
Generates all figures needed for the TICSR paper's Results section.

Figures produced:
  1. EC heatmaps (control + patient) — shows asymmetric directed connectivity
  2. Modal Controllability bar chart — controls vs patients per ROI
  3. Minimum Control Energy bar chart — controls vs patients per ROI
  4. Z-score deviation heatmap — patients x ROIs, colored by |z|

Each figure has a clear title and is saved as high-DPI PNG.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --------------------------------------------------------------------------
# Style config — clean, publication-ready
# --------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'font.family': 'sans-serif',
})

COLORS = {
    'control': '#2196F3',   # blue
    'patient': '#F44336',   # red
    'neutral': '#90A4AE',
}

METRICS_LABELS = {
    'ModalControllability': 'Modal Controllability (MC)',
    'MinControlEnergy': 'Minimum Control Energy (E*)',
    'AverageControllability': 'Average Controllability (AC)',
}


# --------------------------------------------------------------------------
# Figure 1: EC Heatmap
# --------------------------------------------------------------------------
def plot_ec_heatmap(ec_path, title, out_path):
    """
    Asymmetric EC heatmap. The asymmetry is the key methodological point
    justifying EC over FC for NCT.
    """
    A = np.loadtxt(ec_path, delimiter=',')
    vmax = np.abs(A).max()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('EC weight (VAR(1) coefficient)', fontsize=10)

    ax.set_title(f'Effective Connectivity Matrix\n{title}', fontweight='bold')
    ax.set_xlabel('Target ROI')
    ax.set_ylabel('Source ROI')

    # Quantify asymmetry in the plot
    sym_err = np.mean(np.abs(A - A.T))
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    ax.text(0.02, 0.02,
            f'Asymmetry: {sym_err:.4f}\nSpectral radius: {rho:.3f}',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# Figure 2 & 3: Controllability metric bar charts (controls vs patients)
# --------------------------------------------------------------------------
def plot_metric_comparison(nct_dir, output_dir, control_ids, patient_ids, metric, patient_label=None):
    """
    Grouped bar chart: controls (blue) vs patients (red) for each ROI.
    Shows where the groups diverge — these ROIs are your candidate biomarkers.

    Parameters
    ----------
    patient_label : str, optional
        Label for patient group (e.g., 'AUD', 'Epilepsy').
        If None, defaults to 'Patients'. Used in title and filename.
    """
    def load(subj_ids):
        arrays = []
        for sid in subj_ids:
            df = pd.read_csv(os.path.join(nct_dir, f"{sid}_NCT.csv"))
            arrays.append(df[metric].values)
        return np.array(arrays)   # (n_subjects, n_rois)

    ctrl = load(control_ids)     # (n_controls, N)
    pat = load(patient_ids)      # (n_patients, N)
    N = ctrl.shape[1]
    x = np.arange(N)
    width = 0.35

    # Determine label for title and filename
    group_label = patient_label if patient_label else 'Patients'
    fig_label = patient_label.replace(' ', '_') if patient_label else 'patients'

    fig, ax = plt.subplots(figsize=(12, 4))

    bars_c = ax.bar(x - width/2, ctrl.mean(axis=0), width,
                    label=f'Controls (n={len(control_ids)})',
                    color=COLORS['control'], alpha=0.85)
    bars_p = ax.bar(x + width/2, pat.mean(axis=0), width,
                    label=f'{group_label} (n={len(patient_ids)})',
                    color=COLORS['patient'], alpha=0.85)

    # Error bars (std across subjects)
    ax.errorbar(x - width/2, ctrl.mean(axis=0), ctrl.std(axis=0),
                fmt='none', color='#1565C0', capsize=3, linewidth=1.2)
    ax.errorbar(x + width/2, pat.mean(axis=0), pat.std(axis=0),
                fmt='none', color='#B71C1C', capsize=3, linewidth=1.2)

    ax.set_xlabel('ROI Index')
    ax.set_ylabel(METRICS_LABELS.get(metric, metric))
    ax.set_title(f'{METRICS_LABELS.get(metric, metric)}: Controls vs {group_label}',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{metric}_{fig_label}_comparison.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# Figure 4: Z-score deviation heatmap
# --------------------------------------------------------------------------
def plot_zscore_heatmap(zscore_dir, output_dir, patient_ids, metric):
    """
    Heatmap: patients (rows) x ROIs (columns), colored by z-score.
    Red = positive deviation, Blue = negative deviation.
    White dashed lines mark the ±2 threshold (clinical convention).
    """
    z_matrix = []
    for pid in patient_ids:
        path = os.path.join(zscore_dir, f"{pid}_zscores.csv")
        df = pd.read_csv(path)
        col = f'{metric}_zscore'
        z_matrix.append(df[col].values)
    z_matrix = np.array(z_matrix)  # (n_patients, N)

    fig, ax = plt.subplots(figsize=(12, 2.5 + 0.6 * len(patient_ids)))
    im = ax.imshow(z_matrix, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('z-score', fontsize=10)

    # Mark |z| > 2 cells with an asterisk
    for i in range(z_matrix.shape[0]):
        for j in range(z_matrix.shape[1]):
            if abs(z_matrix[i, j]) > 2:
                ax.text(j, i, '*', ha='center', va='center',
                        color='black', fontsize=8, fontweight='bold')

    ax.set_yticks(range(len(patient_ids)))
    ax.set_yticklabels(patient_ids)
    ax.set_xlabel('ROI Index')
    ax.set_title(
        f'Normative Deviation: {METRICS_LABELS.get(metric, metric)}\n'
        f'(* = |z| > 2, clinically anomalous)',
        fontweight='bold'
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'zscore_{metric}_heatmap.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# Figure 5: Summary — mean |z| per patient per metric (bar chart)
# --------------------------------------------------------------------------
def plot_zscore_summary(zscore_dir, output_dir, patient_ids):
    """
    One bar per patient per metric showing mean |z| across ROIs.
    Quickly shows which patient deviates most and on which metric.
    Useful for the Results narrative.
    """
    metrics_znames = [f'{m}_zscore' for m in
                      ['ModalControllability', 'MinControlEnergy', 'AverageControllability']]
    labels = ['Modal Controllability', 'Min Control Energy', 'Average Controllability']

    n_patients = len(patient_ids)
    n_metrics = len(metrics_znames)
    x = np.arange(n_patients)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ['#7B1FA2', '#F57F17', '#1B5E20']

    for mi, (zname, label) in enumerate(zip(metrics_znames, labels)):
        means = []
        for pid in patient_ids:
            df = pd.read_csv(os.path.join(zscore_dir, f"{pid}_zscores.csv"))
            means.append(np.mean(np.abs(df[zname].values)))
        offset = (mi - 1) * width
        ax.bar(x + offset, means, width, label=label,
               color=palette[mi], alpha=0.85)

    ax.axhline(2, color='gray', linestyle='--', linewidth=1, label='|z|=2 threshold')
    ax.set_xlabel('Patient')
    ax.set_ylabel('Mean |z-score| across ROIs')
    ax.set_title('Normative Deviation Summary per Patient', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patient_ids)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'zscore_summary_barplot.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('outputs/figures', exist_ok=True)

    control_ids = ['control_1', 'control_2']
    patient_ids = ['patient_1', 'patient_2', 'patient_3']

    print("── EC Heatmaps ─────────────────────────────────")
    plot_ec_heatmap(
        'outputs/EC/control_1_EC.csv',
        'Representative Control (Subject 1)',
        'outputs/figures/EC_heatmap_control_1.png'
    )
    plot_ec_heatmap(
        'outputs/EC/patient_1_EC.csv',
        'Representative Patient (Subject 1)',
        'outputs/figures/EC_heatmap_patient_1.png'
    )

    print("\n── NCT Metric Comparisons ──────────────────────")
    for metric in ['ModalControllability', 'MinControlEnergy', 'AverageControllability']:
        plot_metric_comparison(
            'outputs/NCT', 'outputs/figures',
            control_ids, patient_ids, metric
        )

    print("\n── Z-Score Heatmaps ────────────────────────────")
    for metric in ['ModalControllability', 'MinControlEnergy']:
        plot_zscore_heatmap(
            'outputs/zscores', 'outputs/figures',
            patient_ids, metric
        )

    print("\n── Z-Score Summary ─────────────────────────────")
    plot_zscore_summary('outputs/zscores', 'outputs/figures', patient_ids)

    print("\n✓ All figures saved to outputs/figures/")
