# NeuroSim — EC + NCT Pipeline for fMRI

GSoC 2026 — NBRC / EBRAINS Project #39

## What this is

End-to-end pipeline for estimating directed effective connectivity (EC) from fMRI timeseries and computing Network Control Theory (NCT) metrics. Targets AUD, AD, and epilepsy cohorts to identify controllability biomarkers.

## Why OLS not correlation

OLS regression on VAR(1) residuals estimates A such that x(t) ≈ A x(t-1), recovering temporal ordering and directed relationships. Pearson correlation is symmetric by construction and cannot encode directionality. Feeding symmetric FC into NCT collapses complex eigenvalues to purely real ones (spectral theorem), destroying the causal geometry that controllability metrics depend on. See Karrer et al. (2020) for methodological considerations.

## Gramian precision

We use `scipy.linalg.solve_discrete_lyapunov` to solve A @ Wc @ A.T - Wc + I = 0 exactly. The finite-horizon sum converges at rate ρ^(2T). At ρ=0.90, T=20 gives ~1.5% error; at ρ=0.97 (AUD rigid attractors), ~30% error. Lyapunov is exact regardless of ρ. We verify residual < 1e-8.

## Modules

- `neuromod/connectome_loader.py` — BIDS timeseries extraction, neuroCombat harmonization, QC filtering
- `src/compute_EC.py` — VAR(1) OLS estimation, causality_vs_fc_audit
- `src/compute_NCT.py` — Lyapunov Gramian, AC, MC, minimum control energy

## Run the tests

```bash
pytest tests/test_neuromod.py -v
```

## Install

```bash
git clone https://github.com/Rishikakaps/NeuroSim.git
cd NeuroSim
pip install -e .
```

Requirements:
- numpy
- scipy
- pandas
- matplotlib
- pytest (testing)
- umap-learn (visualization)
- nilearn (BIDS loading, optional)
- neurocombat (harmonization, optional)
