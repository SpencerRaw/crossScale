# ARCHITECTURE.md

This file documents the project structure, conventions, and key commands for contributors.

## Project overview

Pre-experiment (MVP) for a research project on ML-accelerated multi-scale simulation of biomolecular interfaces. Four modules spanning AA→CG→continuum coupling, adaptive path exploration, RL optimization, and BO prediction.

## Module structure

```
modules/
├── A_triScale/          AA→CG→continuum three-scale pipeline (core)
│   ├── aa_md/           All-atom MD with OpenMM (GBSA implicit solvent)
│   ├── cg_bridge/       ML bridge (MLP: AA features → CG parameters)
│   └── continuum/       Smoluchowski population balance ODE solver
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
python -m modules.A_triScale.aa_md.run_md          # AA MD production
python -m modules.A_triScale.aa_md.extract_features # Feature extraction
python -m modules.A_triScale.cg_bridge.train_ml     # Train ML bridge
python -m modules.A_triScale.cg_bridge.run_cg       # CG simulation
python -m modules.A_triScale.cg_bridge.extract_kinetics
python -m modules.A_triScale.continuum.solve_ode    # Smoluchowski ODE
python -m modules.A_triScale.validate               # Cross-scale validation

# Module B: Path exploration
python -m modules.B_pathExploration.run_sampling

# Module C: RL optimization
python -m modules.C_rlOptimization.train_rl

# Module D: BO prediction
python -m modules.D_boPrediction.run_bo
```

## Data flow (Module A)

```
AA MD trajectories
  → contact probabilities, H-bond lifetimes, secondary structure
  → ML bridge (3-layer MLP: 7→64→32→16→2)
  → CG bead LJ ε, σ + bond constraints
  → CG simulation → diffusion coefficients D, aggregation rates k_on/k_off
  → Smoluchowski ODE → cluster size distribution c_i(t)
  → Reverse validation: continuum → CG → AA
```

## Data contracts (Module A)

**aa_md → cg_bridge:**
- File: `data/aa_features.csv`
- Columns: residue_i, residue_j, contact_prob, avg_distance, hbond_lifetime

**cg_bridge → continuum:**
- File: `data/cg_kinetics.csv`
- Columns: condition (conc/salt), D_diffusion, k_on_dimer, k_off_dimer

**continuum → validation:**
- File: `data/cluster_distribution.csv`
- Columns: time, cluster_size, concentration

## Adding a new module

1. Create `modules/E_yourModule/__init__.py` and main script
2. Follow the self-contained convention — the module should be runnable with a single `python -m` command
3. Add data contracts if your module consumes/produces data for other modules
4. Add a figure script if the module produces visualizations
5. Update `scripts/plot_all.py` to include the new figure
