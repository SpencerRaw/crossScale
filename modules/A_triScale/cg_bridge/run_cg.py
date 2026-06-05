"""
Run CG simulation with ML-learned parameters.

Each residue → 1 CG bead. Interactions: harmonic bonds (consecutive
beads) + LJ nonbonded (all pairs with ML-predicted ε, σ).

Usage:
    python -m modules.A_triScale.cg_bridge.run_cg

Output:
    data/cg_trajectories/   — CG trajectory data
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CG_TRAJ_DIR = DATA_DIR / "cg_trajectories"
CG_TRAJ_DIR.mkdir(parents=True, exist_ok=True)

# CG simulation parameters
CG_TEMP = 300.0  # K
CG_DT = 0.01     # dimensionless timestep (≈ 10 fs)
CG_N_STEPS = 500000   # total steps (≈ 5 ns effective)
CG_N_CONDITIONS = 5   # different concentrations/conditions


def load_cg_params():
    """Load CG parameters from ML bridge output."""
    params_path = DATA_DIR / "cg_params.csv"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found. Run train_ml.py first."
        )
    return pd.read_csv(params_path)


def build_cg_topology(params_df):
    """
    Build CG topology from parameter dataframe.

    Each unique residue index → 1 CG bead.
    Extract bead types and interaction matrix.
    """
    # Get unique residue indices
    residues = set(params_df["residue_i"].unique()) | set(
        params_df["residue_j"].unique()
    )
    n_beads = max(residues) + 1
    print(f"CG system: {n_beads} beads")

    # Build interaction matrices
    eps_matrix = np.zeros((n_beads, n_beads))
    sig_matrix = np.zeros((n_beads, n_beads))

    for _, row in params_df.iterrows():
        i = int(row.residue_i)
        j = int(row.residue_j)
        eps_matrix[i, j] = row.cg_epsilon
        eps_matrix[j, i] = row.cg_epsilon
        sig_matrix[i, j] = row.cg_sigma
        sig_matrix[j, i] = row.cg_sigma

    # Determine bonds: consecutive residues in same chain
    bonds = []
    for _, row in params_df.iterrows():
        if row.same_chain and abs(row.residue_j - row.residue_i) == 1:
            bonds.append((int(row.residue_i), int(row.residue_j)))

    # Chain info: find contiguous blocks
    chain_ends = []
    for _, row in params_df.iterrows():
        if row.same_chain:
            i = int(row.residue_i)
            j = int(row.residue_j)
            if j - i > 3:  # non-adjacent in same chain → different chains
                pass
    # Simple heuristic: bonds connect consecutive residues
    # Each contiguous bond block = one chain

    return eps_matrix, sig_matrix, bonds, n_beads


def lj_potential(r, eps, sig):
    """Lennard-Jones potential: U = 4ε[(σ/r)^12 − (σ/r)^6]"""
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = sig / np.where(r < 1e-10, 1e-10, r)
        sr6 = sr ** 6
        sr12 = sr6 * sr6
        U = 4.0 * eps * (sr12 - sr6)
        U = np.where(r < 1e-10, 1e6, U)  # hard core
    return U


def lj_force(r, eps, sig):
    """LJ force magnitude: F = -dU/dr = 24ε/r * [2(σ/r)^12 − (σ/r)^6]"""
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = sig / np.where(r < 1e-10, 1e-10, r)
        sr6 = sr ** 6
        sr12 = sr6 * sr6
        F = 24.0 * eps / np.where(r < 1e-10, 1e-10, r) * (2.0 * sr12 - sr6)
        F = np.where(r < 1e-10, 0.0, F)
    return F


def run_cg_simulation(
    eps_matrix, sig_matrix, bonds, n_beads,
    box_size, concentration_label,
):
    """
    Run Langevin dynamics CG simulation.

    Parameters:
        box_size: simulation box edge length (nm)
        concentration_label: str label for output files
    """
    rng = np.random.RandomState(42)

    # Initialize positions randomly in box
    positions = rng.uniform(0, box_size, (n_beads, 3))

    # Ensure bonded beads start close
    for _ in range(100):
        for i, j in bonds:
            rij = positions[i] - positions[j]
            # Minimum image convention
            rij = rij - box_size * np.round(rij / box_size)
            dist = np.linalg.norm(rij)
            if dist > 0.5:  # overly far
                target_dist = 0.38  # nm, typical Cα–Cα distance
                if dist > 1e-10:
                    rij = rij / dist * target_dist
                positions[j] = positions[i] - rij

    velocities = rng.normal(0, np.sqrt(CG_TEMP / 1000.0), (n_beads, 3))

    # Store trajectory
    save_interval = 1000
    n_save = CG_N_STEPS // save_interval
    traj_positions = np.zeros((n_save, n_beads, 3))

    gamma = 1.0  # friction coefficient
    k_B = 0.008314  # kJ/(mol·K)
    k_spring = 5000.0  # kJ/(mol·nm²) harmonic bond constant
    r0 = 0.38  # nm equilibrium bond length

    print(f"  [{concentration_label}] Running {CG_N_STEPS} steps, "
          f"box={box_size:.1f} nm …")

    for step in range(CG_N_STEPS):
        forces = np.zeros((n_beads, 3))

        # Bond forces
        for i, j in bonds:
            rij = positions[i] - positions[j]
            rij = rij - box_size * np.round(rij / box_size)
            dist = np.linalg.norm(rij)
            if dist > 1e-10:
                f_mag = -k_spring * (dist - r0)
                f_ij = f_mag * rij / dist
                forces[i] += f_ij
                forces[j] -= f_ij

        # Nonbonded LJ forces
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                if ((i, j) in bonds) or ((j, i) in bonds):
                    continue
                rij = positions[i] - positions[j]
                rij = rij - box_size * np.round(rij / box_size)
                dist = np.linalg.norm(rij)
                eps = eps_matrix[i, j]
                sig = sig_matrix[i, j]
                if eps > 0 and dist < 2.5 * sig:
                    f_mag = lj_force(dist, eps, sig)
                    if dist > 1e-10:
                        f_ij = f_mag * rij / dist
                        forces[i] += f_ij
                        forces[j] -= f_ij

        # Langevin: F − γv + √(2γk_B T) η
        noise_scale = np.sqrt(2.0 * gamma * k_B * CG_TEMP / CG_DT)
        noise = rng.normal(0, noise_scale, (n_beads, 3))

        velocities = (
            velocities
            + (forces / 1.0 - gamma * velocities + noise) * CG_DT
        )
        positions = positions + velocities * CG_DT

        # Periodic boundary
        positions = positions % box_size

        if step % save_interval == 0:
            idx = step // save_interval
            traj_positions[idx] = positions.copy()

    # Save trajectory
    np.save(
        CG_TRAJ_DIR / f"positions_{concentration_label}.npy",
        traj_positions,
    )

    # Save metadata
    metadata = {
        "n_beads": n_beads,
        "box_size": box_size,
        "n_steps": CG_N_STEPS,
        "dt": CG_DT,
        "temperature": CG_TEMP,
        "concentration": concentration_label,
        "bonds": bonds,
    }
    import json
    with open(
        CG_TRAJ_DIR / f"meta_{concentration_label}.json", "w"
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"    → saved {n_save} frames")
    return traj_positions


def run_all():
    """Run CG simulations at multiple concentrations."""
    print("=" * 60)
    print("Module A — CG Simulation with ML Parameters")
    print("=" * 60)

    params_df = load_cg_params()
    eps_matrix, sig_matrix, bonds, n_beads = build_cg_topology(params_df)

    print(f"\nBeads: {n_beads}")
    print(f"Bonds: {len(bonds)}")
    print(f"Non-zero ε pairs: {(eps_matrix > 0.01).sum() // 2}")

    # Run at different box sizes (effective concentrations)
    # box_size → effective concentration c ∝ N/V = N/box³
    box_sizes = [8.0, 6.0, 5.0, 4.0, 3.5]  # nm
    labels = ["low", "med_low", "medium", "med_high", "high"]

    all_trajs = {}
    for box, label in zip(box_sizes, labels):
        traj = run_cg_simulation(
            eps_matrix, sig_matrix, bonds, n_beads, box, label
        )
        all_trajs[label] = traj

    print(f"\nAll CG trajectories saved to {CG_TRAJ_DIR}")
    return all_trajs


if __name__ == "__main__":
    run_all()
