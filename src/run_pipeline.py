"""
run_pipeline.py
===============
Master script — runs the entire NeuroSim scaled-down pipeline end to end.

Usage:
    python run_pipeline.py

Steps:
    1. generate_synthetic.py  → data/roi_timeseries/*.csv
    2. compute_EC.py          → outputs/EC/*_EC.csv
    3. compute_NCT.py         → outputs/NCT/*_NCT.csv
    4. compute_zscores.py     → outputs/zscores/*_zscores.csv
    5. visualize.py           → outputs/figures/*.png
    6. sanity_checks()        → prints pass/fail table

If any step fails, the script stops immediately and tells you which step and why.
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import traceback

# ── import our pipeline modules ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import neurosim.src.generate_synthetic as generate_synthetic
import neurosim.src.compute_EC as compute_EC
import neurosim.src.compute_NCT as compute_NCT
import neurosim.src.compute_zscores as compute_zscores
import neurosim.src.visualize as visualize


CONTROL_IDS = ['control_1', 'control_2']
PATIENT_IDS = ['patient_1', 'patient_2', 'patient_3']
ALL_IDS = CONTROL_IDS + PATIENT_IDS


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_step(name, fn, *args, **kwargs):
    header(name)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        print(f"\n✓ {name} completed in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n✗ {name} FAILED:")
        traceback.print_exc()
        sys.exit(1)


def sanity_checks():
    """
    Run the full sanity check table from plan.md.
    Prints PASS / FAIL for each check.
    """
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

    # 2. Gramian finite (check via NCT files existing and values being finite)
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

    # 3. Z-scores vary across patients
    for pid in PATIENT_IDS:
        path = f'outputs/zscores/{pid}_zscores.csv'
        if not os.path.exists(path):
            results.append((f"Z-score file exists: {pid}", "FAIL", "File not found"))
            continue
        df = pd.read_csv(path)
        z_std = df['ModalControllability_zscore'].std()
        results.append((f"Z-scores vary: {pid}", "PASS" if z_std > 0.1 else "FAIL",
                         f"MC z-score std={z_std:.4f}"))

    # 4. Figures exist
    expected_figs = [
        'outputs/figures/EC_heatmap_control_1.png',
        'outputs/figures/EC_heatmap_patient_1.png',
        'outputs/figures/ModalControllability_comparison.png',
        'outputs/figures/MinControlEnergy_comparison.png',
        'outputs/figures/zscore_ModalControllability_heatmap.png',
        'outputs/figures/zscore_summary_barplot.png',
    ]
    for fig in expected_figs:
        exists = os.path.exists(fig)
        results.append((f"Figure: {os.path.basename(fig)}",
                         "PASS" if exists else "FAIL",
                         "exists" if exists else "missing"))

    # ── Print results table ───────────────────────────────────────────────
    print(f"\n{'Check':<45} {'Status':<8} {'Detail'}")
    print("-" * 80)
    n_pass = n_fail = 0
    for check, status, detail in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"{icon} {check:<43} {status:<8} {detail}")
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{'='*80}")
    print(f"  TOTAL: {n_pass} PASS  |  {n_fail} FAIL")
    if n_fail == 0:
        print("  ✓ All checks passed — pipeline is valid.")
    else:
        print("  ✗ Some checks failed — review output above.")
    print(f"{'='*80}")


def main():
    print("NeuroSim Scaled-Down Pipeline (15% version)")
    print("For TICSR Conference Paper")
    print(f"Working directory: {os.getcwd()}")

    # Step 1: Generate synthetic data
    run_step(
        "Step 1: Generate Synthetic Data",
        generate_synthetic.main
    )

    # Step 2: Compute EC matrices
    run_step(
        "Step 2: Compute Effective Connectivity (VAR(1) OLS)",
        compute_EC.process_all_subjects,
        'data/roi_timeseries', 'outputs/EC'
    )

    # Step 3: Compute NCT metrics
    run_step(
        "Step 3: Compute NCT Metrics (Gramian, AC, MC, E*)",
        compute_NCT.process_all_subjects,
        'outputs/EC', 'outputs/NCT'
    )

    # Step 4: Compute z-scores
    run_step(
        "Step 4: Compute Normative Z-Scores",
        compute_zscores.compute_patient_zscores,
        PATIENT_IDS, CONTROL_IDS, 'outputs/NCT', 'outputs/zscores'
    )

    # Step 5: Visualize
    run_step(
        "Step 5: Generate Figures",
        lambda: (
            os.makedirs('outputs/figures', exist_ok=True),
            visualize.plot_ec_heatmap(
                'outputs/EC/control_1_EC.csv',
                'Representative Control (Subject 1)',
                'outputs/figures/EC_heatmap_control_1.png'),
            visualize.plot_ec_heatmap(
                'outputs/EC/patient_1_EC.csv',
                'Representative Patient (Subject 1)',
                'outputs/figures/EC_heatmap_patient_1.png'),
            [visualize.plot_metric_comparison(
                'outputs/NCT', 'outputs/figures',
                CONTROL_IDS, PATIENT_IDS, m)
             for m in ['ModalControllability', 'MinControlEnergy', 'AverageControllability']],
            [visualize.plot_zscore_heatmap(
                'outputs/zscores', 'outputs/figures',
                PATIENT_IDS, m)
             for m in ['ModalControllability', 'MinControlEnergy']],
            visualize.plot_zscore_summary(
                'outputs/zscores', 'outputs/figures', PATIENT_IDS)
        )
    )

    # Final sanity check
    sanity_checks()

    header("PIPELINE COMPLETE")
    print("Outputs summary:")
    print(f"  EC matrices   → outputs/EC/      ({len(ALL_IDS)} files)")
    print(f"  NCT metrics   → outputs/NCT/     ({len(ALL_IDS)} files)")
    print(f"  Z-scores      → outputs/zscores/ ({len(PATIENT_IDS)} patients + reference)")
    print(f"  Figures       → outputs/figures/")
    print("\nNext step: copy outputs/ to your paper's figures/ folder.")


if __name__ == '__main__':
    main()
