"""
Smoluchowski coagulation-fragmentation population balance model.

Solves the ODE system:
    dc_k/dt = ½ Σ_{i+j=k} K_{ij} c_i c_j
              − c_k Σ_j K_{kj} c_j
              − F_k c_k
              + Σ_{j>k} F_{jk} c_j

where K_{ij} = 4π(D_i + D_j)(R_i + R_j) is the diffusion-limited
coagulation kernel, and F_k is the fragmentation rate.

Parameters (D_i, k_on, k_off) are imported from CG kinetics.

Usage:
    python -m modules.A_triScale.continuum.solve_ode

Output:
    data/cluster_distribution.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

# Continuum model parameters
MAX_CLUSTER_SIZE = 50  # maximum aggregate size to track
T_MAX = 1.0  # simulation time (seconds)
N_POINTS = 500  # time points for output


def load_kinetics():
    """Load kinetic parameters from CG extraction."""
    kinetics_path = DATA_DIR / "cg_kinetics.csv"
    if not kinetics_path.exists():
        raise FileNotFoundError(
            f"{kinetics_path} not found. Run extract_kinetics.py first."
        )
    return pd.read_csv(kinetics_path)


def build_kernel(D0, R0, max_size):
    """
    Build diffusion-limited coagulation kernel.

    K_{ij} = 4π (D_i + D_j) (R_i + R_j)

    D_i = D0 / i^{1/3} (Stokes-Einstein scaling for compact clusters)
    R_i = R0 * i^{1/3}
    """
    sizes = np.arange(1, max_size + 1)
    D = D0 / (sizes ** (1 / 3))
    R = R0 * (sizes ** (1 / 3))

    K = np.zeros((max_size, max_size))
    for i in range(max_size):
        for j in range(max_size):
            K[i, j] = 4.0 * np.pi * (D[i] + D[j]) * (R[i] + R[j])

    return K


def fragmentation_kernel(max_size, k_off):
    """
    Build fragmentation kernel.

    F_k = k_off * (k - 1)  for k ≥ 2
    (each bond breaks with rate k_off)

    Fragments distribute mass equally: F_{jk} → j fragments
    """
    F = np.zeros(max_size)
    F[1:] = k_off * np.arange(1, max_size)  # k≥2: F_k = k_off*(k-1)
    return F


def smoluchowski_rhs(t, c, K, F, max_size):
    """
    Right-hand side of Smoluchowski equations.

    dc_k/dt = ½ Σ_{i+j=k} K_{ij} c_i c_j
              − c_k Σ_j K_{kj} c_j
              − F_k c_k
              + Σ_{j>k} F_j * c_j * (2/(j-1))  # uniform fragment distribution
    """
    dcdt = np.zeros(max_size)

    for k in range(max_size):
        # Coagulation gain: ½ Σ_{i+j=k} K_{ij} c_i c_j
        gain = 0.0
        for i in range(k):
            j = k - i - 1  # i + j + 2 = k + 1 → j = k - i - 1
            if j >= 0 and j < max_size:
                gain += K[i, j] * c[i] * c[j]
        dcdt[k] += 0.5 * gain

        # Coagulation loss: −c_k Σ_j K_{kj} c_j
        loss = c[k] * np.sum(K[k, :] * c)
        dcdt[k] -= loss

        # Fragmentation loss: −F_k c_k
        dcdt[k] -= F[k] * c[k]

        # Fragmentation gain: from larger clusters
        # Clusters of size j (> k+1) break into k+1 + rest
        for j in range(k + 1, max_size):
            # Fragment of size j+1 breaking → produces (k+1)-mer with probability
            # Binary fragmentation: j+1 → (k+1) + (j−k)
            if j > k:
                dcdt[k] += F[j] * c[j] * (2.0 / j)

    return dcdt


def solve_system(kinetics_row, max_size=MAX_CLUSTER_SIZE):
    """
    Solve Smoluchowski ODE for one set of kinetic parameters.
    """
    D0 = kinetics_row["D_nm2_per_step"]
    k_on = kinetics_row["k_on"]
    k_off = kinetics_row["k_off"]
    n_beads = int(kinetics_row["n_beads"])
    box_size = kinetics_row["box_size_nm"]

    # Convert CG units to continuum
    # D0 in nm²/timestep → physical D (m²/s)
    # For MVP: use D0 directly with model scaling
    R0 = 0.3  # nm, bead radius

    # Scale D0 so that aggregation happens on observable timescale
    D_scaled = D0 * 1e6  # empirical scaling to get ms–s aggregation

    K = build_kernel(D_scaled, R0, max_size)
    F = fragmentation_kernel(max_size, k_off)

    # Initial condition: all monomers
    # Total mass M = Σ k * c_k
    total_mass = float(n_beads) / (box_size ** 3)  # number density
    c0 = np.zeros(max_size)
    c0[0] = total_mass  # all mass as monomers

    print(f"    Total mass = {total_mass:.4f} beads/nm³")
    print(f"    D0 = {D_scaled:.2e}, k_off = {k_off:.4f}")

    # Solve
    t_span = (0, T_MAX)
    t_eval = np.logspace(-3, np.log10(T_MAX), N_POINTS)

    sol = solve_ivp(
        smoluchowski_rhs,
        t_span,
        c0,
        args=(K, F, max_size),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-12,
    )

    return sol


def compute_moments(sol, max_size):
    """Compute moments of cluster size distribution."""
    sizes = np.arange(1, max_size + 1)
    c = sol.y  # (max_size, n_times)

    # M0 = Σ c_k (total number of clusters)
    M0 = c.sum(axis=0)

    # M1 = Σ k * c_k (total mass, conserved)
    M1 = (c * sizes[:, None]).sum(axis=0)

    # Mean cluster size = M1 / M0
    mean_size = np.where(M0 > 0, M1 / M0, 1.0)

    # Weight-averaged cluster size = Σ k² c_k / Σ k c_k
    M2 = (c * (sizes ** 2)[:, None]).sum(axis=0)
    mean_size_w = np.where(M1 > 0, M2 / M1, 1.0)

    return M0, M1, mean_size, mean_size_w


def run_all():
    """Solve Smoluchowski for all CG conditions."""
    print("=" * 60)
    print("Module A — Continuum: Smoluchowski Population Balance")
    print("=" * 60)

    kinetics = load_kinetics()
    print(f"Loaded {len(kinetics)} conditions\n")

    all_results = []

    for idx, row in kinetics.iterrows():
        label = row["condition"]
        print(f"[{label}] box={row.box_size_nm:.1f} nm")

        sol = solve_system(row)
        M0, M1, mean_size, mean_size_w = compute_moments(
            sol, MAX_CLUSTER_SIZE
        )

        # Save cluster distribution at key times
        key_indices = [0, len(sol.t) // 4, len(sol.t) // 2, -1]
        for ki in key_indices:
            ti = sol.t[ki]
            ci = sol.y[:, ki]
            for k in range(min(10, MAX_CLUSTER_SIZE)):
                if ci[k] > 1e-10:
                    all_results.append({
                        "condition": label,
                        "time_s": round(float(ti), 6),
                        "cluster_size": k + 1,
                        "concentration": round(float(ci[k]), 10),
                        "mean_cluster_size": round(float(mean_size[ki]), 3),
                    })

        # Summary
        final_mean = mean_size[-1]
        agg_time = sol.t[np.argmax(mean_size > 1.5)] if np.any(mean_size > 1.5) else T_MAX
        print(f"    Final mean size: {final_mean:.2f}")
        print(f"    Aggregation onset: {agg_time:.3f} s")

    # Save
    df = pd.DataFrame(all_results)
    output_path = DATA_DIR / "cluster_distribution.csv"
    df.to_csv(output_path, index=False)
    print(f"\nCluster distribution saved → {output_path}")

    return df


if __name__ == "__main__":
    run_all()
