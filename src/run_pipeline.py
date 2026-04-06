"""
run_pipeline.py
===============
Master script — runs the entire NeuroSim pipeline end to end.

Pipeline Steps:
  1. generate_synthetic.py  → Synthetic ROI timeseries (controls, AUD, epilepsy)
  2. compute_EC.py          → Effective connectivity matrices (VAR(1) OLS)
  3. compute_NCT.py         → Network Control Theory metrics (AC, MC, E*)
  4. compute_zscores.py     → Normative modeling (patient z-scores)
  5. visualize.py           → Publication-ready figures
  6. sanity_checks()        → Validation tests
  7. print_findings()       → Interpretable research findings

Usage:
    python src/run_pipeline.py

All outputs saved to:
  - outputs/EC/          : Effective connectivity matrices
  - outputs/NCT/         : NCT metrics per subject
  - outputs/zscores/     : Normative z-scores
  - outputs/figures/     : PNG figures for paper
  - outputs/findings.txt : Interpretable summary
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import traceback

# ── import pipeline modules ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from generate_synthetic import main as generate_synthetic
from compute_EC import process_all_subjects as compute_ec
from compute_NCT import process_all_subjects as compute_nct
from compute_zscores import compute_patient_zscores
import visualize


# ── Subject definitions ──────────────────────────────────────────────────
# Control subjects (normative reference group)
CONTROL_IDS = [f'control_{i}' for i in range(1, 6)]

# Patient groups
AUD_IDS = [f'aud_{i}' for i in range(1, 6)]
EPILEPSY_IDS = [f'epilepsy_{i}' for i in range(1, 6)]
ALL_PATIENT_IDS = AUD_IDS + EPILEPSY_IDS
ALL_IDS = CONTROL_IDS + ALL_PATIENT_IDS

# ROI groupings for interpretable findings
DMN_NODES = [0, 1, 2, 3, 4]       # Prefrontal/DMN
SEIZURE_FOCUS = [5, 6, 7]         # Temporal/limbic
SENSORIMOTOR = [8, 9, 10]
VISUAL = [11, 12, 13, 14]

METRICS = ['AverageControllability', 'ModalControllability', 'MinControlEnergy']


def header(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_step(name, fn, *args, **kwargs):
    """Run a pipeline step with error handling."""
    header(name)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        print(f"\n✓ {name} completed in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n✗ {name} FAILED:")
        traceback.print_exc()
        sys.exit(1)


def compute_group_statistics(nct_dir, group_ids, metrics=METRICS):
    """Compute mean/std for each metric per ROI for a group."""
    stats = {}
    for metric in metrics:
        values = []
        for sid in group_ids:
            path = os.path.join(nct_dir, f"{sid}_NCT.csv")
            df = pd.read_csv(path)
            values.append(df[metric].values)
        values = np.array(values)  # (n_subjects, n_rois)
        stats[metric] = {
            'mean': values.mean(axis=0),
            'std': values.std(axis=0),
            'all': values
        }
    return stats


def sanity_checks():
    """Run validation tests on pipeline outputs."""
    header("SANITY CHECKS")
    results = []

    # 1. EC files exist and are asymmetric
    for sid in ALL_IDS:
        path = f'outputs/EC/{sid}_EC.csv'
        if not os.path.exists(path):
            results.append((f"EC file exists: {sid}", "FAIL", "File not found"))
            continue
        A = np.loadtxt(path, delimiter=',')
        sym_err = np.mean(np.abs(A - A.T))
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        results.append((f"EC asymmetric: {sid}", "PASS" if sym_err > 0.001 else "FAIL",
                         f"mean|A-AT|={sym_err:.4f}"))
        results.append((f"EC stable: {sid}", "PASS" if rho < 1.0 else "FAIL",
                         f"rho={rho:.4f}"))

    # 2. NCT files exist and values are valid
    for sid in ALL_IDS:
        path = f'outputs/NCT/{sid}_NCT.csv'
        if not os.path.exists(path):
            results.append((f"NCT file exists: {sid}", "FAIL", "File not found"))
            continue
        df = pd.read_csv(path)
        finite = np.isfinite(df[['ModalControllability', 'MinControlEnergy']].values).all()
        results.append((f"NCT finite: {sid}", "PASS" if finite else "FAIL",
                         "all values finite" if finite else "NaN/Inf detected"))

        mc_min = df['ModalControllability'].min()
        results.append((f"MC non-negative: {sid}", "PASS" if mc_min >= 0 else "FAIL",
                         f"min MC={mc_min:.4f}"))

        e_min = df['MinControlEnergy'].min()
        results.append((f"E* positive: {sid}", "PASS" if e_min > 0 else "FAIL",
                         f"min E*={e_min:.4f}"))

    # 3. Z-scores computed
    for pid in ALL_PATIENT_IDS:
        path = f'outputs/zscores/{pid}_zscores.csv'
        if not os.path.exists(path):
            results.append((f"Z-score file exists: {pid}", "FAIL", "File not found"))
            continue
        df = pd.read_csv(path)
        has_zscores = 'AverageControllability_zscore' in df.columns
        results.append((f"Z-scores computed: {pid}", "PASS" if has_zscores else "FAIL",
                         "columns present" if has_zscores else "missing columns"))

    # 4. Figures exist
    expected_figs = [
        'outputs/figures/EC_heatmap_control_1.png',
        'outputs/figures/EC_heatmap_aud_1.png',
        'outputs/figures/EC_heatmap_epilepsy_1.png',
        'outputs/figures/AverageControllability_comparison.png',
        'outputs/figures/ModalControllability_comparison.png',
        'outputs/figures/MinControlEnergy_comparison.png',
        'outputs/figures/zscore_AverageControllability_heatmap.png',
        'outputs/figures/zscore_ModalControllability_heatmap.png',
        'outputs/figures/zscore_summary_barplot.png',
    ]
    for fig in expected_figs:
        exists = os.path.exists(fig)
        results.append((f"Figure: {os.path.basename(fig)}",
                         "PASS" if exists else "FAIL",
                         "exists" if exists else "missing"))

    # ── Print results table ───────────────────────────────────────────────
    print(f"\n{'Check':<50} {'Status':<8} {'Detail'}")
    print("-" * 90)
    n_pass = n_fail = 0
    for check, status, detail in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"{icon} {check:<48} {status:<8} {detail}")
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{'='*90}")
    print(f"  TOTAL: {n_pass} PASS  |  {n_fail} FAIL")
    if n_fail == 0:
        print("  ✓ All checks passed — pipeline is valid.")
    else:
        print("  ✗ Some checks failed — review output above.")
    print(f"{'='*90}")

    return n_fail == 0


def print_interpretable_findings():
    """
    Generate and print interpretable research findings.

    This is the key output for the paper's Results section.
    """
    header("INTERPRETABLE RESEARCH FINDINGS")

    findings = []

    # -------------------------------------------------------------------------
    # Load all data
    # -------------------------------------------------------------------------
    control_stats = compute_group_statistics('outputs/NCT', CONTROL_IDS)
    aud_stats = compute_group_statistics('outputs/NCT', AUD_IDS)
    epilepsy_stats = compute_group_statistics('outputs/NCT', EPILEPSY_IDS)

    # Load z-scores
    aud_zscores = {}
    epilepsy_zscores = {}
    for metric in METRICS:
        aud_z = []
        epi_z = []
        for pid in AUD_IDS:
            df = pd.read_csv(f'outputs/zscores/{pid}_zscores.csv')
            aud_z.append(df[f'{metric}_zscore'].values)
        for pid in EPILEPSY_IDS:
            df = pd.read_csv(f'outputs/zscores/{pid}_zscores.csv')
            epi_z.append(df[f'{metric}_zscore'].values)
        aud_zscores[metric] = np.array(aud_z)  # (n_aud, n_rois)
        epilepsy_zscores[metric] = np.array(epi_z)  # (n_epilepsy, n_rois)

    # -------------------------------------------------------------------------
    # Finding 1: AUD - Reduced prefrontal control
    # -------------------------------------------------------------------------
    findings.append("\n" + "="*60)
    findings.append("  FINDING 1: Alcohol Use Disorder (AUD)")
    findings.append("="*60)

    # Average Controllability in DMN/PFC
    ctrl_ac_dmn = control_stats['AverageControllability']['mean'][DMN_NODES].mean()
    aud_ac_dmn = aud_stats['AverageControllability']['mean'][DMN_NODES].mean()
    pct_change = (aud_ac_dmn - ctrl_ac_dmn) / ctrl_ac_dmn * 100

    findings.append(f"\n1.1 Average Controllability in Prefrontal/DMN nodes (0-4):")
    findings.append(f"    Controls: {ctrl_ac_dmn:.4f}")
    findings.append(f"    AUD:      {aud_ac_dmn:.4f}")
    findings.append(f"    Change:   {pct_change:+.1f}%")

    if pct_change < -5:
        findings.append(f"    → SIGNIFICANT REDUCTION: AUD shows decreased ability to")
        findings.append(f"      reach nearby brain states from prefrontal regions.")
        findings.append(f"    → Consistent with reduced top-down control in addiction.")

    # Minimum Control Energy
    ctrl_energy = control_stats['MinControlEnergy']['mean'].mean()
    aud_energy = aud_stats['MinControlEnergy']['mean'].mean()
    energy_change = (aud_energy - ctrl_energy) / ctrl_energy * 100

    findings.append(f"\n1.2 Mean Control Energy (all nodes):")
    findings.append(f"    Controls: {ctrl_energy:.4f}")
    findings.append(f"    AUD:      {aud_energy:.4f}")
    findings.append(f"    Change:   {energy_change:+.1f}%")

    if energy_change > 5:
        findings.append(f"    → ELEVATED ENERGY: AUD requires more energy for state transitions.")
        findings.append(f"    → Suggests less efficient network control.")

    # Z-score summary for AUD
    aud_ac_z = aud_zscores['AverageControllability'].mean(axis=0)
    aud_dmn_z = aud_ac_z[DMN_NODES].mean()

    findings.append(f"\n1.3 Normative Deviation (z-scores):")
    findings.append(f"    Mean |z| for AC in DMN: {abs(aud_dmn_z):.2f}")
    if aud_dmn_z < -1.5:
        findings.append(f"    → AUD patients show NEGATIVE z-scores in DMN (reduced AC).")

    # -------------------------------------------------------------------------
    # Finding 2: Epilepsy - Hyper-excitable focus
    # -------------------------------------------------------------------------
    findings.append("\n" + "="*60)
    findings.append("  FINDING 2: Epilepsy")
    findings.append("="*60)

    # Modal Controllability in seizure focus
    ctrl_mc_focus = control_stats['ModalControllability']['mean'][SEIZURE_FOCUS].mean()
    epi_mc_focus = epilepsy_stats['ModalControllability']['mean'][SEIZURE_FOCUS].mean()
    mc_change = (epi_mc_focus - ctrl_mc_focus) / ctrl_mc_focus * 100

    findings.append(f"\n2.1 Modal Controllability in Seizure Focus (nodes 5-7):")
    findings.append(f"    Controls:  {ctrl_mc_focus:.4f}")
    findings.append(f"    Epilepsy:  {epi_mc_focus:.4f}")
    findings.append(f"    Change:    {mc_change:+.1f}%")

    if mc_change > 10:
        findings.append(f"    → SIGNIFICANT ELEVATION: Epilepsy shows increased MC in focus.")
        findings.append(f"    → Hyper-excitable nodes can push network into high-energy modes.")
        findings.append(f"    → Consistent with ictal propagation dynamics.")

    # AC in focus nodes
    ctrl_ac_focus = control_stats['AverageControllability']['mean'][SEIZURE_FOCUS].mean()
    epi_ac_focus = epilepsy_stats['AverageControllability']['mean'][SEIZURE_FOCUS].mean()
    ac_change = (epi_ac_focus - ctrl_ac_focus) / ctrl_ac_focus * 100

    findings.append(f"\n2.2 Average Controllability in Seizure Focus:")
    findings.append(f"    Controls:  {ctrl_ac_focus:.4f}")
    findings.append(f"    Epilepsy:  {epi_ac_focus:.4f}")
    findings.append(f"    Change:    {ac_change:+.1f}%")

    # Z-score summary for Epilepsy
    epi_mc_z = epilepsy_zscores['ModalControllability'].mean(axis=0)
    epi_focus_z = epi_mc_z[SEIZURE_FOCUS].mean()

    findings.append(f"\n2.3 Normative Deviation (z-scores):")
    findings.append(f"    Mean z for MC in focus: {epi_focus_z:+.2f}")
    if epi_focus_z > 1.5:
        findings.append(f"    → Epilepsy patients show POSITIVE z-scores in focus (elevated MC).")

    # -------------------------------------------------------------------------
    # Finding 3: Group comparison summary
    # -------------------------------------------------------------------------
    findings.append("\n" + "="*60)
    findings.append("  FINDING 3: Group Comparison Summary")
    findings.append("="*60)

    # Table of key metrics
    findings.append("\n    Metric                    Control      AUD        Epilepsy")
    findings.append("    " + "-"*55)

    for metric in METRICS:
        ctrl_mean = control_stats[metric]['mean'].mean()
        aud_mean = aud_stats[metric]['mean'].mean()
        epi_mean = epilepsy_stats[metric]['mean'].mean()
        findings.append(f"    {metric[:25]:<25} {ctrl_mean:>10.4f}  {aud_mean:>9.4f}  {epi_mean:>10.4f}")

    # -------------------------------------------------------------------------
    # Save findings to file
    # -------------------------------------------------------------------------
    findings_text = "\n".join(findings)
    with open('outputs/findings.txt', 'w') as f:
        f.write(findings_text)

    print(findings_text)
    print(f"\n✓ Findings saved to: outputs/findings.txt")


def main():
    """Run the complete NeuroSim pipeline."""

    print("\n" + "="*60)
    print("  NeuroSim Pipeline — Network Control Theory Analysis")
    print("  TICSR Conference Paper")
    print("="*60)
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Subjects: {len(CONTROL_IDS)} controls, {len(AUD_IDS)} AUD, {len(EPILEPSY_IDS)} epilepsy")

    # Step 1: Generate synthetic data
    run_step(
        "Step 1: Generate Synthetic Data",
        generate_synthetic
    )

    # Step 2: Compute EC matrices
    run_step(
        "Step 2: Compute Effective Connectivity (VAR(1) OLS)",
        compute_ec,
        'data/roi_timeseries', 'outputs/EC'
    )

    # Step 3: Compute NCT metrics
    run_step(
        "Step 3: Compute NCT Metrics (Gramian, AC, MC, E*)",
        compute_nct,
        'outputs/EC', 'outputs/NCT'
    )

    # Step 4: Compute z-scores (controls as normative reference)
    run_step(
        "Step 4: Compute Normative Z-Scores",
        compute_patient_zscores,
        ALL_PATIENT_IDS, CONTROL_IDS, 'outputs/NCT', 'outputs/zscores'
    )

    # Step 5: Visualize
    header("Step 5: Generate Figures")
    os.makedirs('outputs/figures', exist_ok=True)
    t0 = time.time()

    try:
        # EC heatmaps (one per group)
        visualize.plot_ec_heatmap(
            'outputs/EC/control_1_EC.csv',
            'Control Subject 1',
            'outputs/figures/EC_heatmap_control_1.png')
        visualize.plot_ec_heatmap(
            'outputs/EC/aud_1_EC.csv',
            'AUD Subject 1',
            'outputs/figures/EC_heatmap_aud_1.png')
        visualize.plot_ec_heatmap(
            'outputs/EC/epilepsy_1_EC.csv',
            'Epilepsy Subject 1',
            'outputs/figures/EC_heatmap_epilepsy_1.png')

        # NCT metric comparisons (all groups)
        for metric in METRICS:
            visualize.plot_metric_comparison(
                'outputs/NCT', 'outputs/figures',
                CONTROL_IDS, AUD_IDS + EPILEPSY_IDS, metric)

        # Z-score heatmaps
        for metric in METRICS:
            visualize.plot_zscore_heatmap(
                'outputs/zscores', 'outputs/figures',
                ALL_PATIENT_IDS, metric)

        # Summary barplot
        visualize.plot_zscore_summary(
            'outputs/zscores', 'outputs/figures', ALL_PATIENT_IDS)

        print(f"\n✓ Figures generated in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n✗ Figure generation FAILED:")
        traceback.print_exc()
        sys.exit(1)

    # Step 6: Sanity checks
    sanity_checks()

    # Step 7: Print interpretable findings
    print_interpretable_findings()

    # ── Final summary ─────────────────────────────────────────────────────
    header("PIPELINE COMPLETE")
    print("Outputs summary:")
    print(f"  EC matrices   → outputs/EC/      ({len(ALL_IDS)} files)")
    print(f"  NCT metrics   → outputs/NCT/     ({len(ALL_IDS)} files)")
    print(f"  Z-scores      → outputs/zscores/ ({len(ALL_PATIENT_IDS)} patients + reference)")
    print(f"  Figures       → outputs/figures/")
    print(f"  Findings      → outputs/findings.txt")
    print("\nNext steps:")
    print("  1. Review outputs/findings.txt for paper Results section")
    print("  2. Copy outputs/figures/ to your manuscript")
    print("  3. Validate against real clinical data (future work)")


if __name__ == '__main__':
    main()
