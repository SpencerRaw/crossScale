"""
Extract kinetic parameters from CG trajectories for continuum model.

Computes:
    - Diffusion coefficients (from MSD of individual beads)
    - Dimerization rates (from first-passage time analysis)
    - Dissociation rates (from bound-state lifetime)

Usage:
    python -m modules.A_triScale.cg_bridge.extract_kinetics

Output:
    data/cg_kinetics.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CG_TRAJ_DIR = DATA_DIR / "cg_trajectories"

BOUND_CUTOFF = 1.5  # nm — beads closer than this are "bound"


def compute_msd(positions, box_size):
    """Compute mean-squared displacement for each bead."""
    n_frames, n_beads, dim = positions.shape
    msd_by_bead = np.zeros((n_frames, n_beads))

    for t in range(n_frames):
        # Displacement from frame 0, accounting for PBC
        disp = positions[t] - positions[0]
        # Unwrap: correct for PBC jumps
        disp = disp - box_size * np.round(disp / box_size)
        msd_by_bead[t] = np.sum(disp ** 2, axis=1)

    # Average over beads
    msd = msd_by_bead.mean(axis=1)
    return msd


def extract_diffusion_coefficient(positions, dt, box_size):
    """D = MSD(t) / (6t) from linear regime."""
    msd = compute_msd(positions, box_size)
    n_frames = len(msd)

    # Fit MSD = 6Dt + c in linear regime (skip first 10%, use 10-50%)
    t_start = n_frames // 10
    t_end = n_frames // 2
    times = np.arange(t_start, t_end) * dt

    if len(times) < 5:
        # Fallback: use all
        times = np.arange(1, n_frames) * dt
        msd_used = msd[1:]
    else:
        msd_used = msd[t_start:t_end]

    coeffs = np.polyfit(times, msd_used, 1)
    D = coeffs[0] / 6.0  # nm² per timestep unit
    return max(D, 1e-6)


def compute_pair_distances(positions, box_size):
    """Compute all pairwise distances for each frame."""
    n_frames, n_beads, dim = positions.shape
    n_pairs = n_beads * (n_beads - 1) // 2
    distances = np.zeros((n_frames, n_pairs))

    pair_idx = 0
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            rij = positions[:, i] - positions[:, j]
            # Minimum image
            rij = rij - box_size * np.round(rij / box_size)
            distances[:, pair_idx] = np.linalg.norm(rij, axis=1)
            pair_idx += 1

    return distances


def extract_rates(positions, dt, box_size):
    """
    Extract dimerization and dissociation rates.

    k_on: from mean first-passage time to bound state
    k_off: from bound-state lifetime distribution
    """
    n_frames, n_beads, _ = positions.shape
    distances = compute_pair_distances(positions, box_size)

    # For each pair: identify bound/unbound events
    bound = distances < BOUND_CUTOFF
    n_pairs = distances.shape[1]

    # k_off: bound → unbound transitions
    # Average lifetime in bound state
    bound_lifetimes = []
    for p in range(n_pairs):
        in_bound = False
        lifetime = 0
        for t in range(n_frames):
            if bound[t, p]:
                if not in_bound:
                    in_bound = True
                    lifetime = 0
                lifetime += 1
            else:
                if in_bound:
                    bound_lifetimes.append(lifetime * dt)
                    in_bound = False

    if bound_lifetimes:
        avg_bound_lifetime = np.mean(bound_lifetimes)
        k_off = 1.0 / avg_bound_lifetime if avg_bound_lifetime > 0 else 0.1
    else:
        k_off = 0.1
        avg_bound_lifetime = 10.0

    # k_on: unbound → bound transitions
    # Rate per unit concentration
    # k_on = (number of binding events) / (total unbound time × N_beads)
    unbound_frames = (~bound).sum()
    n_events = 0
    for p in range(n_pairs):
        for t in range(1, n_frames):
            if bound[t, p] and not bound[t - 1, p]:
                n_events += 1

    total_unbound_time = unbound_frames * dt
    if total_unbound_time > 0 and n_beads > 1:
        k_on = n_events / (total_unbound_time * n_beads / box_size ** 3)
    else:
        k_on = 0.01

    return max(k_on, 1e-6), max(k_off, 1e-6)


def extract_all():
    """Extract kinetics from all CG conditions."""
    print("=" * 60)
    print("Module A — Kinetic Parameter Extraction from CG")
    print("=" * 60)

    if not CG_TRAJ_DIR.exists():
        raise FileNotFoundError(
            f"{CG_TRAJ_DIR} not found. Run run_cg.py first."
        )

    meta_files = sorted(CG_TRAJ_DIR.glob("meta_*.json"))
    results = []

    for meta_path in meta_files:
        label = meta_path.stem.replace("meta_", "")
        pos_path = CG_TRAJ_DIR / f"positions_{label}.npy"

        if not pos_path.exists():
            print(f"  Skipping {label}: no position file")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        positions = np.load(pos_path)
        box_size = meta["box_size"]
        # Frame interval: total simulation time / number of saved frames
        total_time = meta["n_steps"] * meta["dt"]
        frame_dt = total_time / positions.shape[0]

        print(f"\n  [{label}] box={box_size:.1f} nm")

        D = extract_diffusion_coefficient(positions, frame_dt, box_size)
        print(f"    D = {D:.4f} nm²/timestep")

        k_on, k_off = extract_rates(positions, frame_dt, box_size)
        print(f"    k_on = {k_on:.4f}, k_off = {k_off:.4f}")

        # Volume fraction estimate
        n_beads = meta["n_beads"]
        vol_frac = n_beads * (4 / 3) * np.pi * (0.3 ** 3) / (box_size ** 3)

        results.append({
            "condition": label,
            "box_size_nm": box_size,
            "n_beads": n_beads,
            "volume_fraction": round(vol_frac, 4),
            "D_nm2_per_step": round(D, 6),
            "k_on": round(k_on, 6),
            "k_off": round(k_off, 6),
            "K_eq": round(k_on / k_off, 4) if k_off > 0 else 0,
        })

    # Save
    df = pd.DataFrame(results)
    output_path = DATA_DIR / "cg_kinetics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nKinetics saved → {output_path}")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    extract_all()
