"""
Adaptive path exploration on the alanine dipeptide free energy surface.

φ/ψ Ramachandran map with 5+ minima, multiple saddle points, and
biologically-relevant barriers (2–35 kJ/mol).

The FES is built as a sum of periodic Gaussians anchored at
literature-verified stationary points from Amber ff14SB / implicit
solvent.  C7eq (φ≈−82°, ψ≈75°) is the global minimum.

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

# ── Periodic helper ───────────────────────────────────────────────
def _pwrap(d, period=360.0):
    """Wrap angular difference to [−period/2, period/2]."""
    return (np.asarray(d) + period / 2.0) % period - period / 2.0


# ═══════════════════════════════════════════════════════════════════
# Alanine dipeptide FES — sum of periodic Gaussians
# ═══════════════════════════════════════════════════════════════════

def alanine_dipeptide_fes(phi, psi):
    """
    Alanine dipeptide φ/ψ free energy surface (kJ/mol).

    Built from literature reference points.  C7eq is the global
    minimum (set to ~0 kJ/mol).  Steric clash zone (φ>0, ψ<0)
    reaches 30–40 kJ/mol.

    Parameters
    ----------
    phi, psi : float or ndarray
        Backbone dihedral angles in degrees.

    Returns
    -------
    V : float or ndarray
        Free energy in kJ/mol.
    """
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)

    # ── Attractive basins (deep, narrow Gaussians) ────────────────
    # fmt: off
    basins = [
        # (φ₀,  ψ₀,   depth,  σ_φ, σ_ψ)          kJ/mol
        (-82,   75,   -24.0,  16,  18),   # C7eq  — γ-turn, GLOBAL MIN
        (-78,  150,   -19.0,  18,  20),   # PPII  — polyproline-II
        (-160, 165,   -17.0,  20,  22),   # C5    — fully extended / β
        (-63,  -43,   -13.0,  14,  16),   # α_R   — right-handed α-helix
        ( 58,   47,    -5.5,  18,  20),   # α_L   — left-handed α-helix
        ( 68,  -65,    -3.0,  16,  16),   # C7ax  — axial (shallow)
    ]
    # fmt: on

    V = np.zeros_like(phi)
    for φ0, ψ0, depth, sφ, sψ in basins:
        dφ = _pwrap(phi - φ0)
        dψ = _pwrap(psi - ψ0)
        V += depth * np.exp(-(dφ ** 2 / (2 * sφ ** 2) + dψ ** 2 / (2 * sψ ** 2)))

    # ── Repulsive barriers (broad, positive Gaussians) ────────────
    # fmt: off
    barriers = [
        # (φ₀,   ψ₀,  height, σ_φ, σ_ψ)
        (  0,    0,   30.0,  30,  35),   # central α_R ↔ α_L barrier
        ( 90,  -80,   38.0,  35,  30),   # steric clash core (φ>0, ψ<0)
        (130,  -40,   32.0,  30,  25),   # deep steric ridge
        ( 50,   90,   24.0,  30,  35),   # α_L isolation
        (  0,  -120,  30.0,  25,  20),   # lower steric
        (-40,   10,   19.0,  20,  15),   # C7eq ↔ α_R saddle
        ( 20,   15,   26.0,  25,  30),   # α_R → α_L ridge
        (  0,   55,   16.0,  40,  25),   # upper central plateau
    ]
    # fmt: on

    for φ0, ψ0, height, sφ, sψ in barriers:
        dφ = _pwrap(phi - φ0)
        dψ = _pwrap(psi - ψ0)
        V += height * np.exp(-(dφ ** 2 / (2 * sφ ** 2) + dψ ** 2 / (2 * sψ ** 2)))

    return V + 2.0  # small baseline shift


# ═══════════════════════════════════════════════════════════════════
# Sampling & analysis
# ═══════════════════════════════════════════════════════════════════

def metropolis_sampling(V_func, bounds, n_steps=50000, temp=0.5):
    """Metropolis MC with periodic wrapping for φ/ψ."""
    rng = np.random.RandomState(42)
    (x_min, x_max), (y_min, y_max) = bounds
    px, py = x_max - x_min, y_max - y_min

    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    E = V_func(x, y)

    samples = np.zeros((n_steps, 2))
    energies = np.zeros(n_steps)
    accepted = 0
    step = 12.0  # degrees

    for i in range(n_steps):
        xn = (x + rng.normal(0, step) - x_min) % px + x_min
        yn = (y + rng.normal(0, step) - y_min) % py + y_min
        En = V_func(xn, yn)
        if En <= E or rng.random() < np.exp(-(En - E) / temp):
            x, y, E = xn, yn, En
            accepted += 1
        samples[i], energies[i] = [x, y], E

    print(f"  MC acceptance: {accepted / n_steps:.1%}")
    print(f"  Energy range:  [{energies.min():.1f}, {energies.max():.1f}] kJ/mol")
    return samples, energies


def find_local_minima(V_func, bounds, n_starts=40):
    """Multi-start L-BFGS-B with periodic distance deduplication."""
    (x_min, x_max), (y_min, y_max) = bounds
    rng = np.random.RandomState(123)
    minima = []

    for _ in range(n_starts):
        x0 = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])
        res = optimize.minimize(
            lambda p: V_func(p[0], p[1]),
            x0, method="L-BFGS-B",
            bounds=[(x_min, x_max), (y_min, y_max)],
        )
        dup = False
        for m in minima:
            if all(abs(_pwrap(res.x - m)) < 12.0):
                dup = True
                break
        if not dup:
            minima.append(res.x)
            print(f"  Min  φ={res.x[0]:7.1f}°  ψ={res.x[1]:7.1f}°  "
                  f"E={res.fun:.1f} kJ/mol")

    return np.array(minima)


def string_method(V_func, start, end, n_nodes=60, n_iter=400):
    """Zero-temperature string method with periodic-aware init."""
    string = np.zeros((n_nodes, 2))
    dφ = _pwrap(end[0] - start[0])
    dψ = _pwrap(end[1] - start[1])
    for i in range(n_nodes):
        α = i / (n_nodes - 1)
        string[i] = [start[0] + α * dφ, start[1] + α * dψ]

    eps, dt = 0.5, 0.8

    for _ in range(n_iter):
        for i in range(1, n_nodes - 1):
            φ, ψ = string[i]
            gφ = (V_func(φ + eps, ψ) - V_func(φ - eps, ψ)) / (2 * eps)
            gψ = (V_func(φ, ψ + eps) - V_func(φ, ψ - eps)) / (2 * eps)
            string[i] -= dt * np.array([gφ, gψ])

        # Reparameterize: equal arc length
        arc = np.zeros(n_nodes)
        for i in range(1, n_nodes):
            arc[i] = arc[i - 1] + np.linalg.norm(string[i] - string[i - 1])
        if arc[-1] < 1e-10:
            continue
        for i in range(1, n_nodes - 1):
            t = arc[i] / arc[-1]
            for j in range(n_nodes - 1):
                fj, fn = arc[j] / arc[-1], arc[j + 1] / arc[-1]
                if fj <= t <= fn:
                    α = (t - fj) / max(fn - fj, 1e-10)
                    string[i] = string[j] + α * (string[j + 1] - string[j])
                    break

    energies = np.array([V_func(p[0], p[1]) for p in string])
    return string, energies


def swarm_of_trajectories(V_func, start, n_swarm=40, n_steps=2500, temp=0.5):
    """Overdamped Langevin swarm with periodic boundaries."""
    rng = np.random.RandomState(77)
    dt, eps_fd = 0.8, 2.0
    noise_scale = np.sqrt(2.0 * temp * dt)
    x_min, x_max = -180.0, 180.0
    y_min, y_max = -180.0, 180.0

    all_paths, end_points = [], []
    for _ in range(n_swarm):
        pos = start + rng.normal(0, 8.0, 2)
        path = [pos.copy()]
        for _ in range(n_steps):
            φ, ψ = pos
            gφ = (V_func(φ + eps_fd, ψ) - V_func(φ - eps_fd, ψ)) / (2 * eps_fd)
            gψ = (V_func(φ, ψ + eps_fd) - V_func(φ, ψ - eps_fd)) / (2 * eps_fd)
            pos = pos - np.array([gφ, gψ]) * dt + rng.normal(0, noise_scale, 2)
            pos[0] = (pos[0] - x_min) % 360 + x_min
            pos[1] = (pos[1] - y_min) % 360 + y_min
            path.append(pos.copy())
        all_paths.append(np.array(path))
        end_points.append(pos.copy())

    return all_paths, np.array(end_points)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_all():
    print("=" * 60)
    print("Module B — Adaptive Path Exploration (Alanine Dipeptide FES)")
    print("=" * 60)

    bounds = [(-180.0, 180.0), (-180.0, 180.0)]
    V = alanine_dipeptide_fes

    # 1. Metropolis
    print("\n[1] Metropolis sampling (50k steps) …")
    samples, energies = metropolis_sampling(V, bounds, n_steps=50000)
    np.savez(DATA_DIR / "mb_samples.npz", samples=samples, energies=energies)
    print(f"  {len(samples)} samples saved")

    # 2. Minima
    print("\n[2] Finding local minima …")
    minima = find_local_minima(V, bounds, n_starts=40)
    np.save(DATA_DIR / "mb_minima.npy", minima)
    print(f"  {len(minima)} distinct minima found")

    # 3. String method
    print("\n[3] String method — minimum free-energy paths …")
    all_strings, all_barriers = [], []
    for i in range(len(minima)):
        for j in range(i + 1, len(minima)):
            string, e_path = string_method(V, minima[i], minima[j],
                                           n_nodes=60, n_iter=400)
            barrier = e_path.max() - e_path[0]
            print(f"  Path M{i + 1} → M{j + 1}: barrier = {barrier:.1f} kJ/mol")
            all_strings.append(dict(start=i, end=j, string=string,
                                    energies=e_path, barrier=barrier))
            all_barriers.append(barrier)
    np.savez(DATA_DIR / "mb_paths.npz",
             all_strings=all_strings, minima=minima)

    # 4. Swarm
    print("\n[4] Swarm-of-trajectories …")
    if len(minima) >= 2:
        idx = np.argmax([V(m[0], m[1]) for m in minima])
        paths, endpoints = swarm_of_trajectories(V, minima[idx],
                                                 n_swarm=40, n_steps=2500)
        print(f"  {len(paths)} trajectories from shallowest minimum")
        arr = np.empty(len(paths), dtype=object)
        for k, p in enumerate(paths):
            arr[k] = p
        np.savez(DATA_DIR / "mb_swarm.npz", paths=arr, endpoints=endpoints)

    print("\nModule B complete.")


if __name__ == "__main__":
    run_all()
