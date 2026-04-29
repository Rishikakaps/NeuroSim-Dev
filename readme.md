# NeuroSim — EC + NCT Pipeline for fMRI

![GSoC 2026](https://img.shields.io/badge/GSoC-2026-orange?style=for-the-badge&logo=google)
![INCF](https://img.shields.io/badge/INCF-%23005596?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen?style=for-the-badge)


**GSoC 2026 Project #39 — INCF / EBRAINS**
*Automating In-Silico Stimulation for Non-Invasive Biomarker Discovery*

**Mentor:** Dr. Khushbu Agarwal | **Author:** Rishika Kapil | **Timezone:** UTC+5:30 (India)

---

This repository is a working proof-of-concept for the NeuroSim proposal. It directly
addresses both physics-constrained benchmark challenges raised during the INCF mentor
review and demonstrates them with quantitative, reproducible results.

---

## Addressing the Approximation Crisis

Dr. Khushbu Agarwal posed two benchmark questions during the Neurostars review (April 2026).
Both are answered by specific, tested implementations.

### Q1 — How does the engine distinguish directed causality from functional correlation?

The answer is algebraic. `np.corrcoef()` produces a symmetric matrix by construction.
By the spectral theorem, symmetric matrices have purely real eigenvalues. Feeding this
into any inversion solver collapses the complex eigenvalue structure that VAR(1) OLS
preserves — and modal controllability depends on that structure directly.

`test_fc_symmetry_breaks_eigenvalues()` in `tests/test_neuromod.py` demonstrates this:

```python
# Symmetrized A (what you get from FC-based inversion):
max|Im(λ)| = 3.2e-16   # purely real — causal geometry destroyed

# VAR(1) OLS A (this pipeline):
max|Im(λ)| = 0.31      # complex conjugate pairs preserved
```

The primary EC estimator is VAR(1) OLS:

```python
from src.compute_EC import fit_var1_ols, postprocess_A

X = np.loadtxt('data/roi_timeseries/control_1.csv', delimiter=',')

# A such that X[t] ≈ A @ X[t-1]
# Temporal asymmetry preserved by construction
A = fit_var1_ols(X)
A = postprocess_A(A)  # zero diagonal, normalize ρ < 1, assert asymmetry

print(f"Asymmetry: mean|A - A.T| = {np.mean(np.abs(A - A.T)):.4f}")  # > 0
print(f"Spectral radius: {np.max(np.abs(np.linalg.eigvals(A))):.4f}") # < 1
```

Granger causality provides formal statistical validation of directed edges:

```python
from neurosim.connectivity.granger import (
    granger_causality_matrix,
    causality_vs_correlation_summary,
)

G, F_matrix, p_matrix = granger_causality_matrix(X, order=1, alpha=0.05)
# G[i,j] = 1 means j → i is a statistically validated causal edge
# F = ((RSS_restricted - RSS_full) / order) / (RSS_full / (T - N·order - 1))

summary = causality_vs_correlation_summary(A, X)
print(f"FC asymmetry:   {summary['fc_asymmetry']:.2e}")   # ~0 — FC is symmetric
print(f"A asymmetry:    {summary['var1_asymmetry']:.4f}")  # > 0 — A is directed
print(f"Spurious pairs: {summary['n_spurious']}")          # high FC, no causal edge
print(f"Hidden edges:   {summary['n_hidden']}")            # significant Granger, low FC
```

### Q2 — How does the Controllability Gramian scale without losing numerical precision?

`compute_gramian_large_scale()` uses `scipy.linalg.solve_discrete_lyapunov`
(Bartels-Stewart, O(N³)) to solve A Wc Aᵀ − Wc + I = 0 exactly. The finite-horizon
sum alternative converges at rate ρ^(2T): at ρ=0.97 (AUD rigid attractors) with T=20,
this gives ~30% residual error — clinically unacceptable. Lyapunov is exact regardless
of ρ as long as ρ < 1.

```python
from neurosim.control.gramian_schur import (
    compute_gramian_large_scale,
    gramian_precision_benchmark,
)

Wc, report = compute_gramian_large_scale(A)

print(f"Lyapunov residual: {report['lyapunov_residual']:.2e}")  # < 1e-8
print(f"Condition number:  {report['condition_number']:.2e}")
print(f"Min eigenvalue:    {report['min_eigenvalue']:.6f}")      # > 0
print(f"Effective rank:    {report['effective_rank']} / {A.shape[0]}")
print(f"Valid:             {report['is_valid']}")                 # True

# Validate precision scales to clinical dataset size
df = gramian_precision_benchmark(ns=[50, 100, 200], rho_values=[0.7, 0.85, 0.95])
# All residuals < 1e-8 across all (N, ρ) combinations including ρ=0.95
```

---

## EC Recovery Validation

Because the pipeline generates synthetic data with known ground-truth A matrices,
EC recovery is benchmarked using normalized Frobenius error
‖A_est − A_true‖_F / ‖A_true‖_F per subject per cohort (Gilson et al. 2016):

```python
from src.validate_ec_recovery import compute_ec_recovery

df = compute_ec_recovery('outputs/EC', 'data/ground_truth')
```

| Cohort   | Mean Frob Error | Std   | Interpretation |
|----------|----------------|-------|----------------|
| Control  | 0.335          | 0.023 | Baseline OLS noise floor |
| AUD      | 0.220          | 0.010 | Lower due to ρ≈0.90 — SNR artifact, not biology |
| Epilepsy | 0.316          | 0.017 | Renormalization after 2.5× focus amplification |
| AD       | 0.350          | 0.015 | Weakest entries → lowest SNR → highest error |

The AUD error is lowest despite being a pathological group — a statistical artifact
of higher spectral radius improving OLS conditioning, documented as a limitation.
A symmetric FC-inversion approach cannot produce this validation at all: there is no
well-defined directed ground truth to recover from a correlation matrix.

---

## Quick Start

### 1. Estimate directed connectivity

```python
import numpy as np
from src.compute_EC import fit_var1_ols, postprocess_A

X = np.loadtxt('data/roi_timeseries/control_1.csv', delimiter=',')
A = postprocess_A(fit_var1_ols(X))
```

### 2. Compute NCT metrics

```python
from neurosim.src.compute_NCT import (
    compute_gramian, average_controllability,
    modal_controllability, min_control_energy, GRAMIAN_T,
)

Wc = compute_gramian(A, T=GRAMIAN_T)
ac = average_controllability(A, T=GRAMIAN_T)  # (N,) per-node
mc = modal_controllability(A)                  # (N,) per-node
e  = min_control_energy(Wc, N=A.shape[0])      # (N,) per-node
```

### 3. Identify facilitator nodes

```python
from neurosim.control.metrics import rank_facilitator_nodes

top_nodes, mc_scores = rank_facilitator_nodes(A, top_k=5)
print(f"Seizure propagation / craving circuit hubs: {top_nodes}")
```

### 4. Harmonize multi-site data (blind)

```python
from neurosim.harmonization.combat import blind_harmonize

# Parameters estimated on HC only — disease labels never enter estimation
harmonized, params = blind_harmonize(
    hc_data, hc_site_labels,
    clinical_data, clinical_labels,
)
```

### 5. Run the full pipeline

```bash
python src/run_pipeline.py
```

Outputs saved to `outputs/EC/`, `outputs/NCT/`, `outputs/zscores/`,
`outputs/figures/`, `outputs/ec_recovery_validation.csv`.

---

## Package Structure

```
neurosim/
├── connectivity/
│   └── granger.py          Granger F-test, causality_vs_correlation_summary()
├── control/
│   ├── energy.py           Finite-horizon minimum energy, arbitrary x0/xf/B
│   ├── gramian_schur.py    Lyapunov Gramian + precision report + benchmark
│   └── metrics.py          rank_facilitator_nodes(), compute_attractor_rigidity()
├── harmonization/
│   └── combat.py           Blind neuroCombat (HC-only parameter estimation)
└── NCT/
    └── compute_NCT.py      AC, MC, E* — Lyapunov Gramian, GRAMIAN_T=20

src/
├── compute_EC.py           VAR(1) OLS, postprocess_A(), causality_vs_fc_audit()
├── compute_zscores.py      Normative z-score deviation scoring
├── generate_synthetic.py   Synthetic cohorts with known ground-truth A matrices
├── run_pipeline.py         Master pipeline (EC → NCT → z-scores → figures)
├── validate_ec_recovery.py Frobenius error vs ground-truth A (Gilson et al. 2016)
└── visualize.py            Publication figures

tests/
├── test_neuromod.py        8 tests  — A asymmetry, Gramian, MC, AC, FC eigenvalue collapse
├── test_nct.py             13 tests — NCT signatures, convergence, formula verification
├── test_granger.py         6 tests  — asymmetry, diagonal, FC symmetry, edge recovery
└── test_harmonization.py   5 tests  — blind constraint, unseen site, variance reduction
```

---

## Install

```bash
git clone https://github.com/Rishikakaps/NeuroSim-Dev.git
cd NeuroSim-Dev
pip install -e ".[dev]"
```

## Run the Tests

```bash
pytest tests/ -v
```

Expected: **32 passed** across all four test modules.

---

## Deliverables Status

| Deliverable | Status | Location |
|---|---|---|
| VAR(1) OLS EC estimation | ✅ Complete | `src/compute_EC.py` |
| Granger causality F-test + causality summary | ✅ Complete | `neurosim/connectivity/granger.py` |
| FC eigenvalue collapse unit test | ✅ Complete | `tests/test_neuromod.py` |
| Lyapunov Gramian (exact, O(N³)) | ✅ Complete | `neurosim/control/gramian_schur.py` |
| Gramian precision benchmark N=[50,100,200] | ✅ Complete | `neurosim/control/gramian_schur.py` |
| NCT metrics: AC, MC, E* | ✅ Complete | `neurosim/src/compute_NCT.py` |
| Finite-horizon energy, arbitrary x0/xf/B | ✅ Complete | `neurosim/control/energy.py` |
| Facilitator node ranking | ✅ Complete | `neurosim/control/metrics.py` |
| Blind neuroCombat (HC-only) | ✅ Complete | `neurosim/harmonization/combat.py` |
| Normative z-score deviation scoring | ✅ Complete | `src/compute_zscores.py` |
| EC recovery validation (Frobenius) | ✅ Complete | `src/validate_ec_recovery.py` |
| Full test suite | ✅ 32 passed | `tests/` |
| Master pipeline | ✅ Complete | `src/run_pipeline.py` |

---

## Key Findings (Synthetic Validation)

- **Epilepsy**: Modal controllability elevated at seizure focus nodes 5–7 (mean z = +5.51),
  consistent with hyper-excitable ictal propagation hubs (Gu et al. 2015).
- **AUD**: E* reduction at prefrontal nodes is a spectral radius artifact (ρ=0.90 vs 0.70);
  documented as a limitation, not reported as a biological finding.
- **AD**: AC reduction in DMN nodes 0–4; effect size limited by normative model noise floor
  at small n — PCNtoolkit BLR is the correct approach for real demographic data.

---

## References

- Karrer et al. (2020). *A practical guide to methodological considerations in the
  controllability of structural brain networks.* J Neural Eng.
- Gu et al. (2015). *Controllability of structural brain networks.* Nat Commun.
- Gilson et al. (2016). *Estimation of directed effective connectivity from fMRI
  functional connectivity hints at asymmetries of cortical connectome.* PLOS Comp Bio.
- Cornblath et al. (2020). *Temporal sequences of brain activity at rest are constrained
  by white matter structure.* Nat Commun.
- Johnson et al. (2007). *Adjusting batch effects in microarray expression data using
  empirical Bayes methods.* Biostatistics.
- Bartels & Stewart (1972). *Algorithm 432: Solution of the matrix equation AX + XB = C.*
  CACM.
