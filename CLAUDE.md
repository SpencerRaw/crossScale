# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pre-experiment (MVP) for a research project on ML-accelerated multi-scale simulation of biomolecular interfaces. Four modules spanning AA→CG→continuum coupling, adaptive path exploration, RL optimization, and BO prediction.

Target hardware: RTX 5090 (32GB VRAM), 3-day timeline on rented server.

## Environment

```bash
conda env create -f environment.yml
conda activate crossScale
```

## Module structure

```
modules/
├── A_triScale/          AA→CG→continuum three-scale pipeline (core)
│   ├── aa_md/           All-atom MD with OpenMM
│   ├── cg_bridge/       ML bridge AA→CG + CG simulation
│   └── continuum/       Smoluchowski population balance ODE
├── B_pathExploration/   Müller-Brown adaptive path sampling
├── C_rlOptimization/    HP lattice model RL sequence optimization
└── D_boPrediction/      XGBoost + Bayesian optimization
```

## Conventions

- Each module is self-contained and runnable via `python -m modules.X.submodule`
- Shared utilities go in `scripts/`, not duplicated across modules
- All output figures write to `outputs/figures/`
- Intermediate data (trajectories, trained models, extracted parameters) write to `data/`
- Figure scripts are named `plot_*.py`, analysis scripts `analyze_*.py`
- Simulation input files (PDB, topology) live inside the module that uses them
- MD parameter files (.xml, .prmtop) are generated programmatically, not committed

## Key commands

```bash
# Module A: Three-scale pipeline
python -m modules.A_triScale.aa_md.run_md        # AA MD production
python -m modules.A_triScale.cg_bridge.train_ml   # Train ML bridge
python -m modules.A_triScale.cg_bridge.run_cg     # CG simulation
python -m modules.A_triScale.continuum.solve_ode  # Smoluchowski ODE

# Module B: Path exploration
python -m modules.B_pathExploration.run_sampling

# Module C: RL optimization
python -m modules.C_rlOptimization.train_rl

# Module D: BO prediction
python -m modules.D_boPrediction.run_bo
```

## Parameter passing chain (Module A)

```
AA MD trajectories
  → residue contact probabilities, H-bond lifetimes, secondary structure
  → ML bridge (3-layer MLP) → CG bead LJ ε,σ + bond constraints
  → CG simulation → diffusion coefficients D, aggregation rates k_on/k_off
  → Smoluchowski ODE → cluster size distribution c_i(t)
  → reverse validation: continuum→CG, CG→AA
```
