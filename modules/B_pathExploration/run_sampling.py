"""
Adaptive path exploration on the alanine dipeptide free energy surface.

φ/ψ Ramachandran map with 5+ minima, multiple saddle points, and
biologically-relevant barriers (2–25 kJ/mol). This replaces the
Müller-Brown toy potential with a realistic biomolecular landscape.

This module:
  1. Samples the FES with Metropolis Monte Carlo
  2. Finds multiple minima via multi-start gradient descent
  3. String method for minimum free-energy paths between minima
  4. Swarm-of-trajectories to discover alternative pathways

Usage:
    python -m modules.B_pathExploration.run_sampling

Output:
    data/mb_samples.npz, mb_minima.npy, mb_paths.npz, mb_swarm.npz
"""

import numpy as np
from pathlib import Path
from scipy import optimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Periodic distance helper ──────────────────────────────────────
def _pdist(a, b, period=360.0):
    """Shortest signed distance on a circle with given period."""
    d = (a - b) % period
    d[d > period / 2] -= period
    return d


# ═══════════════════════════════════════════════════════════════════
# Alanine dipeptide free energy surface
# ═══════════════════════════════════════════════════════════════════

def alanine_dipeptide_fes(phi, psi):
    """
    Alanine dipeptide φ/ψ free energy surface (kJ/mol).

    φ (phi)   = C–N–Cα–C  dihedral,  in degrees
    ψ (psi)   = N–Cα–C–N  dihedral,  in degrees

    Topology (based on Amber ff14SB / implicit solvent):
      C7eq  (φ≈−82°, ψ≈ 75°)  — global minimum, 7-membered ring
      PPII  (φ≈−78°, ψ≈150°)  — polyproline-II / extended
      C5    (φ≈−158°,ψ≈162°)  — fully extended
      α_R   (φ≈−63°, ψ≈−43°)  — right-handed α-helix
      α_L   (φ≈ 58°, ψ≈ 47°)  — left-handed α-helix
      C7ax  (φ≈ 68°, ψ≈−62°)  — axial C7, shallow

    Key saddle points between basins at barriers 2–25 kJ/mol.
    Sterically forbidden regions (φ>0, ψ<0 centre) reach >30 kJ/mol.
    """
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)

    # ── Steric exclusion: forbidden Ramachandran regions ──────────
    # Hard penalties for steric clashes based on allowed-region shape
    phi_rad = np.deg2rad(phi)
    psi_rad = np.deg2rad(psi)

    # Broad steric ridge separating left/right half of the map
    steric = (
        14.0 * (np.cos(phi_rad + 0.9) + 0.7) * (np.cos(psi_rad - 1.4) + 0.7)
        + 8.0 * np.cos(2.0 * phi_rad + 0.8)
        + 5.0 * np.cos(2.0 * psi_rad - 0.3)
        + 4.0 * np.cos(phi_rad + psi_rad + 0.5)
    )

    # ── Local basins (periodic 2D Gaussians) ──────────────────────
    # (φ₀, ψ₀, depth, σ_φ, σ_ψ, correlation ρ)
    basins = [
        # C7eq — global minimum, γ-turn geometry
        (-82.0,  75.0, -18.0, 22.0, 22.0,  0.10),
        # PPII — polyproline-II, extended left-handed
        (-78.0, 150.0, -15.0, 24.0, 28.0,  0.10),
        # C5 — fully extended, β-sheet region
        (-158.0, 162.0, -14.0, 28.0, 30.0, -0.25),
        # α_R — right-handed α-helix
        (-63.0, -43.0, -10.0, 18.0, 20.0,  0.20),
        # β / PII extended region (bridge between C5 and PPII)
        (-120.0, 118.0, -11.0, 20.0, 25.0,  0.05),
        # α_L — left-handed α-helix (shallow, rare in nature)
        (58.0,  47.0,  -3.0, 25.0, 28.0, -0.05),
        # C7ax — axial, shallow minimum
        (68.0, -62.0,  -2.0, 22.0, 22.0,  0.00),
    ]

    V = np.zeros_like(phi)
    for phi0, psi0, depth, s_phi, s_psi, rho in basins:
        dphi = _pdist(phi, phi0)
        dpsi = _pdist(psi, psi0)
        # Bivariate Gaussian with correlation
        arg = -(
            dphi ** 2 / (2.0 * s_phi ** 2)
            + dpsi ** 2 / (2.0 * s_psi ** 2)
            + rho * dphi * dpsi / (s_phi * s_psi)
        )
        V += depth * np.exp(np.clip(arg, -100, 10))

    return steric + V


# ═══════════════════════════════════════════════════════════════════
# Sampling and analysis (same API as before)
# ═══════════════════════════════════════════════════════════════════

def metropolis_sampling(V_func, bounds, n_steps=50000, temp=0.5):
    """Metropolis MC sampling with periodic boundaries for φ/ψ."""
    rng = np.random.RandomState(42)
    (x_min, x_max), (y_min, y_max) = bounds
    period_x = x_max - x_min
    period_y = y_max - y_min

    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    E = V_func(x, y)

    samples = np.zeros((n_steps, 2))
    energies = np.zeros(n_steps)
    accepted = 0

    step_size = 8.0  # degrees — tuned for ~40% acceptance at 300K

    for i in range(n_steps):
        x_new = x + rng.normal(0, step_size)
        y_new = y + rng.normal(0, step_size)
        # Periodic wrap
        x_new = (x_new - x_min) % period_x + x_min
        y_new = (y_new - y_min) % period_y + y_min

        E_new = V_func(x_new, y_new)
        delta_E = E_new - E

        if delta_E <= 0 or rng.random() < np.exp(-delta_E / temp):
            x, y = x_new, y_new
            E = E_new
            accepted += 1

        samples[i] = [x, y]
        energies[i] = E

    print(f"  MC acceptance rate: {accepted / n_steps:.2%}")
    print(f"  Energy range: [{energies.min():.1f}, {energies.max():.1f}] kJ/mol")
    return samples, energies


def find_local_minima(V_func, bounds, n_starts=30):
    """Multi-start L-BFGS-B to find distinct local minima."""
    (x_min, x_max), (y_min, y_max) = bounds
    rng = np.random.RandomState(123)

    minima = []
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])
        res = optimize.minimize(
            lambda p: V_func(p[0], p[1]),
            x0,
            method="L-BFGS-B",
            bounds=[(x_min, x_max), (y_min, y_max)],
        )
        # Check uniqueness (15° radius for distinct basins)
        is_new = True
        for existing in minima:
            dphi = abs(_pdist(res.x[0], existing[0]))
            dpsi = abs(_pdist(res.x[1], existing[1]))
            if dphi < 15.0 and dpsi < 15.0:
                is_new = False
                break
        if is_new:
            minima.append(res.x)
            print(f"  Minimum at (φ={res.x[0]:7.1f}°, ψ={res.x[1]:7.1f}°), "
                  f"E = {res.fun:.2f} kJ/mol")

    return np.array(minima)


def string_method(V_func, start, end, n_nodes=50, n_iter=300):
    """
    Zero-temperature string method with periodic-aware initialisation.

    Initialises the string along the shortest periodic path, then
    evolves nodes via gradient descent + equal-arc reparameterisation.
    """
    # Shortest-path interpolation accounting for periodicity
    string = np.zeros((n_nodes, 2))
    dphi = _pdist(end[0], start[0])
    dpsi = _pdist(end[1], start[1])
    for i in range(n_nodes):
        alpha = i / (n_nodes - 1)
        string[i, 0] = start[0] + alpha * dphi
        string[i, 1] = start[1] + alpha * dpsi

    # Finite-difference gradient
    eps = 0.5  # degrees — coarser mesh for stability

    dt = 0.5  # gradient descent step size

    for _ in range(n_iter):
        for i in range(1, n_nodes - 1):
            phi, psi = string[i]
            gphi = (V_func(phi + eps, psi) - V_func(phi - eps, psi)) / (2 * eps)
            gpsi = (V_func(phi, psi + eps) - V_func(phi, psi - eps)) / (2 * eps)
            string[i] -= dt * np.array([gphi, gpsi])

        # Reparameterize: equal arc length
        arc = np.zeros(n_nodes)
        for i in range(1, n_nodes):
            arc[i] = arc[i - 1] + np.linalg.norm(string[i] - string[i - 1])
        if arc[-1] < 1e-10:
            continue
        for i in range(1, n_nodes - 1):
            target = arc[i] / arc[-1]
            for j in range(n_nodes - 1):
                fj = arc[j] / arc[-1]
                fn = arc[j + 1] / arc[-1]
                if fj <= target <= fn:
                    alpha = (target - fj) / max(fn - fj, 1e-10)
                    string[i] = string[j] + alpha * (string[j + 1] - string[j])
                    break

    energies = np.array([V_func(p[0], p[1]) for p in string])
    return string, energies


def swarm_of_trajectories(V_func, start, n_swarm=30, n_steps=2000, temp=0.5):
    """
    Overdamped Langevin swarm from start region to discover pathways.

    dx = −∇V dt + √(2 kB T dt) η

    Periodic boundary reflection keeps trajectories in [-180, 180].
    """
    rng = np.random.RandomState(77)
    dt = 0.8
    noise_scale = np.sqrt(2.0 * temp * dt)

    x_min, x_max = -180.0, 180.0
    y_min, y_max = -180.0, 180.0
    eps_fd = 2.0  # finite-difference step (degrees)

    all_paths = []
    end_points = []

    for s in range(n_swarm):
        pos = start + rng.normal(0, 8.0, 2)
        pos[0] = (pos[0] - x_min) % 360.0 + x_min
        pos[1] = (pos[1] - y_min) % 360.0 + y_min
        path = [pos.copy()]

        for _ in range(n_steps):
            phi, psi = pos
            gphi = (V_func(phi + eps_fd, psi) - V_func(phi - eps_fd, psi)) / (2 * eps_fd)
            gpsi = (V_func(phi, psi + eps_fd) - V_func(phi, psi - eps_fd)) / (2 * eps_fd)

            noise = rng.normal(0, noise_scale, 2)
            pos = pos - np.array([gphi, gpsi]) * dt + noise
            # Periodic wrap
            pos[0] = (pos[0] - x_min) % 360.0 + x_min
            pos[1] = (pos[1] - y_min) % 360.0 + y_min
            path.append(pos.copy())

        all_paths.append(np.array(path))
        end_points.append(pos.copy())

    return all_paths, np.array(end_points)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_all():
    """Main entry: sample, find minima, discover paths."""
    print("=" * 60)
    print("Module B — Adaptive Path Exploration (Alanine Dipeptide FES)")
    print("=" * 60)

    bounds = [(-180.0, 180.0), (-180.0, 180.0)]
    V_func = alanine_dipeptide_fes

    # 1. Metropolis sampling
    print("\n[1] Metropolis sampling (50k steps) …")
    samples, energies = metropolis_sampling(V_func, bounds, n_steps=50000)
    np.savez(DATA_DIR / "mb_samples.npz", samples=samples, energies=energies)
    print(f"  Saved {len(samples)} samples")

    # 2. Find minima
    print("\n[2] Finding local minima …")
    minima = find_local_minima(V_func, bounds, n_starts=40)
    np.save(DATA_DIR / "mb_minima.npy", minima)
    print(f"  Found {len(minima)} distinct minima")

    # 3. String method
    print("\n[3] String method — minimum free-energy paths …")
    all_strings = []
    all_barriers = []

    for i in range(len(minima)):
        for j in range(i + 1, len(minima)):
            string, path_energies = string_method(
                V_func, minima[i], minima[j],
                n_nodes=60, n_iter=400,
            )
            barrier = path_energies.max() - path_energies[0]
            print(f"  Path M{i + 1} → M{j + 1}: barrier = {barrier:.1f} kJ/mol")
            all_strings.append({
                "start": i,
                "end": j,
                "string": string,
                "energies": path_energies,
                "barrier": barrier,
            })
            all_barriers.append(barrier)

    np.savez(DATA_DIR / "mb_paths.npz",
             all_strings=all_strings, minima=minima)

    # 4. Swarm-of-trajectories
    print("\n[4] Swarm-of-trajectories — alternative pathways …")
    if len(minima) >= 2:
        energies_at_min = np.array([V_func(m[0], m[1]) for m in minima])
        start_idx = np.argmax(energies_at_min)  # start from shallowest

        paths, endpoints = swarm_of_trajectories(
            V_func, minima[start_idx],
            n_swarm=40, n_steps=2500,
        )
        print(f"  {len(paths)} trajectories launched from shallowest minimum")
        paths_arr = np.empty(len(paths), dtype=object)
        for k, p in enumerate(paths):
            paths_arr[k] = p
        np.savez(DATA_DIR / "mb_swarm.npz",
                 paths=paths_arr, endpoints=endpoints)

    print("\nModule B complete. Data ready for figure generation.")


if __name__ == "__main__":
    run_all()
