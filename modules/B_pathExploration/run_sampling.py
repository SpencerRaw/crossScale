"""
Adaptive path exploration on the Müller-Brown potential.

The Müller-Brown potential is a 2D surface with 3 minima and 2 saddle
points — a classic benchmark for transition path sampling.

This module:
  1. Samples the potential surface with Metropolis Monte Carlo
  2. Finds multiple minimum free-energy paths using the string method
  3. Swarm-of-trajectories to discover alternative pathways
  4. Compares path barriers to identify the dominant vs alternative routes

Corresponds to PPT "自适应路径探索" + "路径识别/过渡态".

Usage:
    python -m modules.B_pathExploration.run_sampling

Output:
    outputs/figures/fig4_adaptive_paths.png (via plot script)
"""

import numpy as np
from pathlib import Path
from scipy import optimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def muller_brown_potential(x, y):
    """
    Müller-Brown potential (kcal/mol).

    3 minima at approximately:
      M1 ≈ (−0.558, 1.442)  — deepest
      M2 ≈ (0.623, 0.028)   — intermediate
      M3 ≈ (−0.050, 0.467)  — shallow

    2 saddle points at approximately:
      TS1 ≈ (−0.822, 0.624) — between M2 and deep basin
      TS2 ≈ (0.212, 0.293)  — between M1 and M3
    """
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])

    V = np.zeros_like(x)
    for k in range(4):
        arg = (
            a[k] * (x - x0[k]) ** 2
            + b[k] * (x - x0[k]) * (y - y0[k])
            + c[k] * (y - y0[k]) ** 2
        )
        # Clip to prevent overflow (exp(>700) → inf in float64)
        arg = np.clip(arg, -500, 100)
        V += A[k] * np.exp(arg)
    return V


def metropolis_sampling(V_func, bounds, n_steps=50000, temp=0.5):
    """
    Metropolis Monte Carlo sampling of the potential surface.
    Returns sampled points and their energies.
    """
    rng = np.random.RandomState(42)
    x_range, y_range = bounds

    # Initialize at random position
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    E = V_func(x, y)

    samples = np.zeros((n_steps, 2))
    energies = np.zeros(n_steps)
    accepted = 0

    step_size = 0.1

    for i in range(n_steps):
        x_new = x + rng.normal(0, step_size)
        y_new = y + rng.normal(0, step_size)

        # Check bounds
        if not (x_range[0] <= x_new <= x_range[1] and
                y_range[0] <= y_new <= y_range[1]):
            samples[i] = [x, y]
            energies[i] = E
            continue

        E_new = V_func(x_new, y_new)
        delta_E = E_new - E

        if delta_E <= 0 or rng.random() < np.exp(-delta_E / temp):
            x, y = x_new, y_new
            E = E_new
            accepted += 1

        samples[i] = [x, y]
        energies[i] = E

    print(f"  MC acceptance rate: {accepted / n_steps:.2%}")
    print(f"  Energy range: [{energies.min():.1f}, {energies.max():.1f}] kcal/mol")
    return samples, energies


def find_local_minima(V_func, bounds, n_starts=20):
    """Find local minima by gradient descent from random starts."""
    x_range, y_range = bounds
    rng = np.random.RandomState(123)

    minima = []
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        res = optimize.minimize(
            lambda p: V_func(p[0], p[1]),
            x0,
            method="L-BFGS-B",
            bounds=[x_range, y_range],
        )

        # Check if this is a new minimum
        is_new = True
        for existing in minima:
            if np.linalg.norm(res.x - existing) < 0.1:
                is_new = False
                break
        if is_new and res.fun < 0:
            minima.append(res.x)
            print(f"  Minimum at ({res.x[0]:.3f}, {res.x[1]:.3f}), "
                  f"E = {res.fun:.2f} kcal/mol")

    return np.array(minima)


def string_method(V_func, start, end, n_nodes=30, n_iter=200):
    """
    Simplified zero-temperature string method.
    Finds the minimum free-energy path between two minima.

    The string is a chain of points in 2D, evolved by:
      1. Move each node downhill (gradient descent)
      2. Reparameterize to maintain equal spacing
    """
    # Initialize string as linear interpolation
    string = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        alpha = i / (n_nodes - 1)
        string[i] = start + alpha * (end - start)

    # Finite-difference gradient
    eps = 1e-5

    def grad_V(p):
        gx = (V_func(p[0] + eps, p[1]) - V_func(p[0] - eps, p[1])) / (2 * eps)
        gy = (V_func(p[0], p[1] + eps) - V_func(p[0], p[1] - eps)) / (2 * eps)
        return np.array([gx, gy])

    dt = 0.001

    for iteration in range(n_iter):
        # Move each node downhill (except endpoints)
        for i in range(1, n_nodes - 1):
            g = grad_V(string[i])
            # Project gradient: move perpendicular to string direction
            string[i] -= dt * g

        # Reparameterize: equal arc length
        arc_lengths = np.zeros(n_nodes)
        for i in range(1, n_nodes):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(
                string[i] - string[i - 1]
            )

        if arc_lengths[-1] > 1e-10:
            for i in range(1, n_nodes - 1):
                target = arc_lengths[i] / arc_lengths[-1]
                # Interpolate to find position at this fraction
                for j in range(n_nodes - 1):
                    frac_j = arc_lengths[j] / arc_lengths[-1]
                    frac_next = arc_lengths[j + 1] / arc_lengths[-1]
                    if frac_j <= target <= frac_next:
                        alpha = (
                            (target - frac_j) / (frac_next - frac_j)
                            if frac_next > frac_j
                            else 0
                        )
                        string[i] = string[j] + alpha * (
                            string[j + 1] - string[j]
                        )
                        break

    # Compute energy along string
    energies = np.array([V_func(p[0], p[1]) for p in string])

    return string, energies


def swarm_of_trajectories(V_func, start, n_swarm=20, n_steps=2000, temp=0.3):
    """
    Launch a swarm of trajectories from the same starting region
    to discover alternative pathways.

    Each trajectory follows overdamped Langevin dynamics:
      dx = −∇V dt + √(2k_B T dt) η
    """
    rng = np.random.RandomState(77)
    dt = 0.005
    noise_scale = np.sqrt(2.0 * temp * dt)

    all_paths = []
    end_points = []

    # Finite-difference epsilon and gradient buffer
    eps_fd = 1e-4  # coarser mesh: more stable against extreme potential values
    grad_buf = 0.3  # stay inside bounds for finite-difference stencil

    x_min, x_max = -1.5, 1.2
    y_min, y_max = -0.5, 2.0

    for s in range(n_swarm):
        # Initialize near start with slight perturbation, clamped to bounds
        pos = start + rng.normal(0, 0.05, 2)
        pos[0] = np.clip(pos[0], x_min + grad_buf, x_max - grad_buf)
        pos[1] = np.clip(pos[1], y_min + grad_buf, y_max - grad_buf)
        path = [pos.copy()]

        for _ in range(n_steps):
            # Gradient with safe finite difference
            x, y = pos[0], pos[1]
            xp = np.clip(x + eps_fd, x_min, x_max)
            xn = np.clip(x - eps_fd, x_min, x_max)
            yp = np.clip(y + eps_fd, y_min, y_max)
            yn = np.clip(y - eps_fd, y_min, y_max)
            gx = (V_func(xp, y) - V_func(xn, y)) / max(xp - xn, 1e-10)
            gy = (V_func(x, yp) - V_func(x, yn)) / max(yp - yn, 1e-10)

            noise = rng.normal(0, noise_scale, 2)
            pos = pos - np.array([gx, gy]) * dt + noise
            # Reflect at boundaries
            pos[0] = np.clip(pos[0], x_min + grad_buf, x_max - grad_buf)
            pos[1] = np.clip(pos[1], y_min + grad_buf, y_max - grad_buf)
            path.append(pos.copy())

        all_paths.append(np.array(path))
        end_points.append(pos.copy())

    return all_paths, np.array(end_points)


def run_all():
    """Main entry: sample, find minima, discover paths."""
    print("=" * 60)
    print("Module B — Adaptive Path Exploration (Müller-Brown)")
    print("=" * 60)

    bounds = [(-1.5, 1.2), (-0.5, 2.0)]  # x_range, y_range

    # 1. Metropolis sampling to map the surface
    print("\n[1] Metropolis sampling …")
    samples, energies = metropolis_sampling(
        muller_brown_potential, bounds, n_steps=30000
    )
    np.savez(
        DATA_DIR / "mb_samples.npz",
        samples=samples,
        energies=energies,
    )
    print(f"  Saved {len(samples)} samples")

    # 2. Find local minima
    print("\n[2] Finding local minima …")
    minima = find_local_minima(muller_brown_potential, bounds)
    np.save(DATA_DIR / "mb_minima.npy", minima)
    print(f"  Found {len(minima)} distinct minima")

    # 3. String method: find paths between each pair of minima
    print("\n[3] String method — minimum free-energy paths …")
    all_strings = []
    all_barriers = []

    for i in range(len(minima)):
        for j in range(i + 1, len(minima)):
            string, energies = string_method(
                muller_brown_potential, minima[i], minima[j],
                n_nodes=50, n_iter=300,
            )
            barrier = energies.max() - energies[0]
            print(f"  Path M{i + 1} → M{j + 1}: barrier = {barrier:.2f} kcal/mol")
            all_strings.append({
                "start": i,
                "end": j,
                "string": string,
                "energies": energies,
                "barrier": barrier,
            })
            all_barriers.append(barrier)

    np.savez(
        DATA_DIR / "mb_paths.npz",
        all_strings=all_strings,
        minima=minima,
    )

    # 4. Swarm of trajectories from shallow minimum to discover diverse paths
    print("\n[4] Swarm-of-trajectories — alternative pathways …")
    if len(minima) >= 2:
        # Start from shallowest minimum
        energies_at_min = np.array([
            muller_brown_potential(m[0], m[1]) for m in minima
        ])
        start_idx = np.argmax(energies_at_min)  # shallowest

        paths, endpoints = swarm_of_trajectories(
            muller_brown_potential, minima[start_idx],
            n_swarm=30, n_steps=1500,
        )

        # Cluster endpoints to identify distinct pathways
        n_pathways = min(3, len(minima))
        print(f"  {len(paths)} trajectories → "
              f"{n_pathways} distinct pathway clusters")
        paths_arr = np.empty(len(paths), dtype=object)
        for i, p in enumerate(paths):
            paths_arr[i] = p
        np.savez(
            DATA_DIR / "mb_swarm.npz",
            paths=paths_arr,
            endpoints=endpoints,
        )

    print("\nModule B complete. Data ready for figure generation.")


if __name__ == "__main__":
    run_all()
