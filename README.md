# crossScale — ML-Accelerated Multi-Scale Biomolecular Simulation

> **Pre-experiment MVP**: a modular framework bridging all-atom MD, coarse-grained ML potentials, and continuum kinetics — built in 3 days, extensible by design.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

---

## What is crossScale?

Simulating biomolecular interfaces (peptides on membranes, protein aggregation) is bottlenecked by timescale: all-atom MD captures atomic detail but runs microseconds at best, while biological processes span milliseconds to seconds.

**crossScale** bridges this gap with a 4-module pipeline:

```
Module A (core)      Module B           Module C           Module D
AA MD                Path               RL                 BO
  │                  Exploration        Optimization       Prediction
  ▼                     │                   │                  ▲
ML Bridge               │                   │                  │
(AA → CG)               ▼                   ▼                  │
  │                  Müller-Brown        HP Lattice         XGBoost
  ▼                  adaptive            PPO-based          + Bayesian
CG Simulation        path sampling       sequence opt       optimization
  │
  ▼
Continuum ODE
(Smoluchowski)
```

---

## Architecture

### Module A — Tri-Scale Pipeline (core)

| Component | What it does | Output |
|-----------|-------------|--------|
| `aa_md/run_md.py` | All-atom MD with OpenMM (GBSA implicit solvent) | DCD trajectories |
| `aa_md/extract_features.py` | Contact probability, H-bond lifetimes, secondary structure | `aa_features.csv` |
| `cg_bridge/train_ml.py` | MLP maps AA features → CG bead parameters (ε, σ) | Trained MLP model |
| `cg_bridge/run_cg.py` | Coarse-grained MD with learned potentials | CG trajectories |
| `cg_bridge/extract_kinetics.py` | Diffusion coefficients, aggregation rates | `cg_kinetics.csv` |
| `continuum/solve_ode.py` | Smoluchowski population balance ODE | Cluster size distributions |
| `validate.py` | Cross-scale validation: continuum→CG→AA | Validation figures |

### Module B — Adaptive Path Exploration

Müller-Brown potential with adaptive sampling — finds minimum-energy paths and transition barriers without prior knowledge of the energy landscape.

### Module C — RL Sequence Optimization

HP lattice model + PPO — learns peptide sequences that fold into target structures. Demonstrates reinforcement learning for biomolecular design.

### Module D — Bayesian Optimization

XGBoost surrogate + BO — predicts material properties (loading, stability, release) from structural parameters. Multi-objective: maximize loading × stability while minimizing release rate.

---

## Quick Start

### 1. Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate crossScale

# Or pip
pip install -r requirements.txt
```

### 2. Run Module A (tri-scale pipeline)

```bash
python -m modules.A_triScale.aa_md.run_md        # AA MD (~1-2 hrs on GPU)
python -m modules.A_triScale.aa_md.extract_features
python -m modules.A_triScale.cg_bridge.train_ml
python -m modules.A_triScale.cg_bridge.run_cg
python -m modules.A_triScale.cg_bridge.extract_kinetics
python -m modules.A_triScale.continuum.solve_ode
python -m modules.A_triScale.validate
```

### 3. Run other modules

```bash
python -m modules.B_pathExploration.run_sampling   # Path exploration
python -m modules.C_rlOptimization.train_rl        # RL optimization
python -m modules.D_boPrediction.run_bo            # BO prediction
```

### 4. Generate all figures

```bash
python scripts/plot_all.py
```

---

## Data Flow (Module A)

```
AA MD trajectories
  → contact probabilities, H-bond lifetimes, secondary structure
  → ML bridge (3-layer MLP: 7→64→32→16→2)
  → CG bead LJ ε, σ + bond constraints
  → CG simulation → D, k_on, k_off
  → Smoluchowski ODE → cluster size distribution c_i(t)
  → Reverse validation: continuum → CG → AA
```

---

## Requirements

- Python 3.10+
- OpenMM (for AA MD)
- PyTorch (for ML bridge)
- See `environment.yml` for full dependencies

---

## Project Structure

```
crossScale/
├── modules/
│   ├── A_triScale/          # Core: AA → CG → Continuum
│   │   ├── aa_md/           #   All-atom MD (OpenMM)
│   │   ├── cg_bridge/       #   ML bridge + CG simulation
│   │   └── continuum/       #   Smoluchowski ODE solver
│   ├── B_pathExploration/   # Adaptive path sampling
│   ├── C_rlOptimization/    # RL sequence optimization
│   └── D_boPrediction/      # Bayesian optimization
├── scripts/                 # Figure generation
├── guides/                  # Step-by-step walkthroughs
├── data/                    # Generated data (gitignored)
├── outputs/                 # Figures and reports (gitignored)
├── PLAN.md                  # Implementation plan
├── PPT_design.md            # 2-page presentation design spec
└── environment.yml          # Conda environment
```

---

## Status

**🏗️ Pre-experiment / MVP** — Framework architecture complete. All modules runnable on synthetic/mock data. Next steps: real GPU trajectories → real parameter extraction → method paper.

---

## Citation

If you use crossScale in your research, please cite:

```bibtex
@software{crossScale2026,
  author = {Yiwei Xu},
  title = {crossScale: ML-Accelerated Multi-Scale Biomolecular Simulation},
  year = {2026},
  url = {https://github.com/SpencerRaw/crossScale}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
