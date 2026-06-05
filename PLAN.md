# Implementation Plan

## Module dependency graph

```
Day 1                         Day 2                         Day 3
──────                        ──────                        ──────
A.aa_md ──┐                   A.continuum ──┐               All figures
           ├─ A.cg_bridge ──┘                ├─ A validation               
           │                                │                              
B (独立) ──────────────────────────────────┘               B figure
C (独立) ──────────────────────────────────┘               C figure
D ─────── (needs A completed parameter set) ─┘             D figure
```

## Module interfaces (data contracts)

### A: Three-scale pipeline

**A.aa_md → A.cg_bridge:**
File: `data/aa_features.csv`
Columns: residue_i, residue_j, contact_prob, avg_distance, hbond_lifetime
Generating code: `modules/A_triScale/aa_md/extract_features.py`

**A.cg_bridge → A.continuum:**
File: `data/cg_kinetics.csv`
Columns: condition (conc/salt), D_diffusion, k_on_dimer, k_off_dimer
Generating code: `modules/A_triScale/cg_bridge/extract_kinetics.py`

**A.continuum → validation:**
File: `data/cluster_distribution.csv`
Columns: time, cluster_size, concentration

### B: Adaptive path exploration

Input: none (self-contained Müller-Brown potential)
Output: `outputs/figures/fig4_paths.png`

### C: RL optimization

Input: none (self-contained HP lattice)
Output: `outputs/figures/fig5_rl_bo_comparison.png`

### D: BO prediction

Input: `data/aa_features.csv` + `data/cg_kinetics.csv` (from Module A)
Output: `outputs/figures/fig6_bo.png`

## Implementation order

### Day 1 (hours 0-10)
1. `modules/A_triScale/aa_md/run_md.py` — OpenMM peptide simulation
2. `modules/A_triScale/aa_md/extract_features.py` — AA trajectory analysis
3. `modules/A_triScale/cg_bridge/train_ml.py` — MLP for AA→CG mapping
4. `modules/A_triScale/cg_bridge/run_cg.py` — CG simulation with learned params

### Day 2 (hours 10-20)
5. `modules/A_triScale/continuum/solve_ode.py` — Smoluchowski solver
6. `modules/A_triScale/validate.py` — Cross-scale validation
7. `modules/B_pathExploration/run_sampling.py` — Müller-Brown path sampling
8. `modules/C_rlOptimization/train_rl.py` — HP lattice RL
9. `modules/D_boPrediction/run_bo.py` — XGBoost + BO

### Day 3 (hours 20-28)
10. All figure scripts in `outputs/figures/`
11. Report writing

## Figure output mapping

| Figure | Script | Module | PPT mapping |
|--------|--------|--------|-------------|
| fig1_triScale_overview.png | plot_triScale.py | A | Multi-scale coupling diagram |
| fig2_fes_slowVars.png | plot_fes.py | A | ML enhanced sampling + FES |
| fig3_parameter_validation.png | plot_validation.py | A | Parameter transfer validation |
| fig4_adaptive_paths.png | plot_paths.py | B | Adaptive path exploration |
| fig5_rl_convergence.png | plot_rl.py | C | RL vs BO convergence |
| fig6_bo_optimization.png | plot_bo.py | D | BO optimization + feature importance |
