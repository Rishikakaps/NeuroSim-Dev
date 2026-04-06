# NeuroSim — Brain Network Control Pipeline

**GSoC 2026 — NBRC / EBRAINS Project #39**

A minimal, end-to-end Python pipeline to:
- estimate **directed brain connectivity (A matrix)** from fMRI timeseries
- compute **Network Control Theory (NCT) metrics**
- generate **per-subject biomarkers and deviations from controls**

---

## 🚀 What This Repo Does

This is a **working prototype pipeline**, not just theory.

**Input:**
- ROI BOLD timeseries (T × N)

**Output:**
- Directed connectivity matrices (A)
- Controllability metrics (AC, MC, Energy)
- Z-scored deviations vs control group
- Visualizations

---

## 🧠 Pipeline Overview

1. data/roi_timeseries/
2. generate_synthetic.py # (optional) create test data
3. compute_EC.py # timeseries → directed A matrix
4. compute_NCT.py # A → controllability metrics
5. compute_zscores.py # compare vs controls
6. visualize.py # plots + heatmaps


Run everything:
```bash
python src/run_pipeline.py
```
---

## ⚙️ Core Modules
|Module|	What it does|	Output|
|---|---|---|
|compute_EC.py |Fits VAR(1) model → directed A matrix	|outputs/EC/|
|compute_NCT.py	|Computes Gramian, AC, MC, Energy	|outputs/NCT/|
|compute_zscores.py	|Normalizes vs controls	|outputs/zscores/|
|visualize.py	|Generates plots |outputs/figures/|

---

## 🧪 How It Works

```python
1. Estimate Directed Connectivity
from src.compute_EC import load_timeseries, fit_var1_ols

X = load_timeseries("data/roi_timeseries/patient_1.csv")
A = fit_var1_ols(X)
``` 
- Learns causal relationships between brain regions
- Produces asymmetric A matrix
- Automatically normalized for stability

```python 
2. Compute Control Metrics
from src.compute_NCT import compute_gramian, modal_controllability

W = compute_gramian(A)
mc = modal_controllability(A)
```
Outputs:
- AC → easy-to-reach states
- MC → hard-to-reach states
- Energy → cost of transitions

```python 
3. Compare Against Controls
from src.compute_zscores import compute_zscores

zscores = compute_zscores(patient_data, control_data)
```
- Highlights abnormal regions per subject
- Produces per-ROI deviation maps

---

## 📁 Outputs
``` bash
outputs/
├── EC/          # directed A matrices
├── NCT/         # AC, MC, Energy
├── zscores/     # patient vs control deviations
└── figures/     # heatmaps + plots
```
---

## 🧪 Testing
```bash 
pytest tests/ -v
```
--- 

## 🔬 Validation (Synthetic)
- Generates synthetic VAR(1) data
- Simulates network degradation (e.g. reduced coupling)
Verifies:
- A matrix recovery
- sensitivity of NCT metrics

---

## ⚡ Key Design Choices
- Uses directed connectivity (A) instead of symmetric FC
- Ensures system stability (spectral radius < 1)
- Keeps pipeline modular and reproducible

---

## 📦 Installation
```bash 
git clone https://github.com/Rishikakaps/NeuroSim.git
cd NeuroSim
pip install -r requirements.txt
```

---

## Dependencies:
1. numpy
2. scipy
3. pandas
4. matplotlib

---

## 🧭 Roadmap
- BIDS dataset ingestion
- Multi-site harmonization
- Real clinical datasets (AD, AUD, Epilepsy)
- Improved normative modeling

---
## References 
1. Gu, S., et al. (2015). Controllability of structural brain networks. *Nature Communications*, 6, 8414. 
2. Gilson, M., et al. (2016). Estimation of directed effective connectivity from fMRI. *PLOS Computational Biology*, 12(3).
3. Karrer, T.M., et al. (2020). A practical guide to methodological considerations in the controllability of structural brain networks. *Journal of Neural Engineering*, 17(2).
4. Parkes, L., et al. (2024). A network control theory pipeline for studying the dynamics of the structural connectome. *Nature Protocols*.

--- 

## 👩‍💻 Author
Rishika Kapil
