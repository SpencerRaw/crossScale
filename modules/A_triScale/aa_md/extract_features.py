"""
Extract features from AA MD trajectories for ML bridge training.

Output: data/aa_features.csv
    Columns: residue_i, residue_j, contact_prob, avg_distance_nm,
             min_distance_nm, res_i_type, res_j_type,
             hbond_frac, same_chain
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
TRAJ_DIR = DATA_DIR / "trajectories"
CONTACT_CUTOFF_NM = 0.6  # Cα–Cα distance for residue contact


def load_trajectories():
    """Load all DCD trajectories with MDTraj."""
    import mdtraj as md

    topo_path = str(DATA_DIR / "topology.pdb")
    if not os.path.exists(topo_path):
        raise FileNotFoundError(
            f"Topology not found: {topo_path}. Run run_md.py first."
        )

    dcd_files = sorted(TRAJ_DIR.glob("traj_*.dcd"))
    if not dcd_files:
        raise FileNotFoundError(
            f"No DCD files found in {TRAJ_DIR}. Run run_md.py first."
        )

    print(f"Loading {len(dcd_files)} trajectories …")
    trajectories = []
    for dcd in dcd_files:
        traj = md.load(str(dcd), top=str(topo_path))
        # Strip solvent (water, ions) — keep only protein/peptide
        protein = traj.atom_slice(
            traj.topology.select("protein or resname ACE or resname NME")
        )
        trajectories.append(protein)
        print(f"  {dcd.name}: {traj.n_frames} frames, "
              f"{protein.n_atoms} protein atoms")

    return trajectories


def get_residue_pairs(traj):
    """Get list of (res_idx_i, res_idx_j, same_chain) pairs."""
    residues = list(traj.topology.residues)
    pairs = []
    for i, ri in enumerate(residues):
        for j, rj in enumerate(residues):
            if j <= i:
                continue
            same_chain = (ri.chain == rj.chain)
            # Skip adjacent residues in same chain (1–2, 1–3)
            if same_chain and abs(j - i) < 3:
                continue
            pairs.append((i, j, ri.name, rj.name, same_chain))
    return pairs


def compute_contacts(traj, pairs):
    """Compute contact probability for all residue pairs."""
    print("  Computing residue–residue contacts …")
    n_residues = traj.n_residues

    # Get Cα atom indices per residue
    ca_indices = []
    for ri in range(n_residues):
        atoms = [
            a.index
            for a in traj.topology.residue(ri).atoms
            if a.name == "CA"
        ]
        if atoms:
            ca_indices.append(atoms[0])
        else:
            # Fallback: first atom of residue
            atoms = [
                a.index
                for a in traj.topology.residue(ri).atoms
            ]
            ca_indices.append(atoms[0] if atoms else 0)

    xyz = traj.xyz
    n_frames = len(xyz)

    results = []
    for (ri, rj, name_i, name_j, same_chain) in pairs:
        ci = ca_indices[ri]
        cj = ca_indices[rj]

        distances = np.linalg.norm(
            xyz[:, ci, :] - xyz[:, cj, :], axis=1
        )
        contact_prob = np.mean(distances < CONTACT_CUTOFF_NM)
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)

        results.append({
            "residue_i": ri,
            "residue_j": rj,
            "res_i_name": name_i,
            "res_j_name": name_j,
            "same_chain": int(same_chain),
            "contact_prob": round(contact_prob, 6),
            "avg_distance_nm": round(float(avg_dist), 4),
            "min_distance_nm": round(float(min_dist), 4),
        })

    return pd.DataFrame(results)


def compute_hbond_lifetimes(traj):
    """Compute H-bond lifetimes using Baker-Hubbard criterion."""
    import mdtraj as md

    print("  Computing H-bond lifetimes …")
    # Find backbone H-bonds
    hbonds = md.baker_hubbard(
        traj, periodic=False, freq=0.0,
    )
    # hbonds: array of (donor_idx, hydrogen_idx, acceptor_idx) per frame

    # Group into residue pairs and compute occupancy
    residue_of_atom = {}
    for ri, res in enumerate(traj.topology.residues):
        for atom in res.atoms:
            residue_of_atom[atom.index] = ri

    pair_lifetimes = defaultdict(float)
    n_frames = len(hbonds)

    for frame_hbonds in hbonds:
        frame_pairs = set()
        for d, h, a in frame_hbonds:
            ri = residue_of_atom.get(d, -1)
            rj = residue_of_atom.get(a, -1)
            if ri != rj and ri >= 0 and rj >= 0:
                key = (min(ri, rj), max(ri, rj))
                frame_pairs.add(key)
        for key in frame_pairs:
            pair_lifetimes[key] += 1.0

    # Normalize to fraction of trajectory
    hbond_fracs = {k: v / n_frames for k, v in pair_lifetimes.items()}
    return hbond_fracs


def extract_all():
    """Main entry: load trajectories and extract all features."""
    print("=" * 60)
    print("Module A — Feature Extraction from AA MD")
    print("=" * 60)

    trajectories = load_trajectories()

    # Get residue pairs from first trajectory
    pairs = get_residue_pairs(trajectories[0])
    print(f"Residue pairs: {len(pairs)} (non-adjacent)")

    # Compute contacts for each trajectory and average
    all_contacts = []
    for ti, traj in enumerate(trajectories):
        print(f"\nTrajectory {ti + 1}/{len(trajectories)}:")
        df_contacts = compute_contacts(traj, pairs)
        df_contacts["traj_id"] = ti
        all_contacts.append(df_contacts)

    # Average across trajectories
    full_df = pd.concat(all_contacts, ignore_index=True)
    avg_df = (
        full_df
        .groupby(["residue_i", "residue_j", "res_i_name", "res_j_name",
                   "same_chain"])
        .agg({
            "contact_prob": "mean",
            "avg_distance_nm": "mean",
            "min_distance_nm": "min",
        })
        .reset_index()
    )

    # Add H-bond fractions from first trajectory (representative)
    hbond_fracs = compute_hbond_lifetimes(trajectories[0])
    avg_df["hbond_frac"] = avg_df.apply(
        lambda row: hbond_fracs.get(
            (int(row.residue_i), int(row.residue_j)), 0.0
        ),
        axis=1,
    )

    # Add residue type encoding for ML
    res_type_map = {
        "ACE": 0, "NME": 1,
        "ALA": 2, "PHE": 3, "LYS": 4,
    }
    avg_df["res_type_i"] = avg_df["res_i_name"].map(
        lambda x: res_type_map.get(x, 5)
    )
    avg_df["res_type_j"] = avg_df["res_j_name"].map(
        lambda x: res_type_map.get(x, 5)
    )

    # Save
    output_path = DATA_DIR / "aa_features.csv"
    avg_df.to_csv(output_path, index=False)
    print(f"\nFeatures saved → {output_path}")
    print(f"  {len(avg_df)} residue pairs")
    print(f"  Contact range: [{avg_df.contact_prob.min():.3f}, "
          f"{avg_df.contact_prob.max():.3f}]")

    # Run TICA for slow variable identification (PPT: 慢变量/反应坐标学习)
    print()
    compute_tica(trajectories)

    return avg_df


def compute_tica(trajectories, lag_time_ps=50):
    """
    Time-lagged Independent Component Analysis (TICA) to identify
    slow collective variables from MD trajectories.

    Featurization: all pairwise Cα distances (upper triangle).
    TICA finds the linear combinations that decorrelate slowest,
    giving the dominant reaction coordinates.

    Saves:
        data/tica_timescales.npy  — implied timescales per component
        data/tica_eigenvectors.npy — TICA eigenvectors
        data/tica_projections.npy  — trajectory projected onto top 2 TICs
    """
    print("\n" + "—" * 40)
    print("TICA: Learning slow collective variables …")

    # Featurize each trajectory: pairwise Cα distances
    all_features = []

    for ti, traj in enumerate(trajectories):
        # Get Cα atom indices
        ca_indices = traj.topology.select("name CA")
        if len(ca_indices) < 3:
            print(f"  Traj {ti}: too few Cα atoms ({len(ca_indices)}), skip")
            continue

        # Pairwise distances: for each frame, compute distance matrix
        # and take upper triangle
        xyz = traj.xyz[:, ca_indices, :]
        n_frames = xyz.shape[0]
        n_atoms = xyz.shape[1]

        features = np.zeros((n_frames, n_atoms * (n_atoms - 1) // 2))
        for f in range(n_frames):
            dist_mat = np.linalg.norm(
                xyz[f, :, None, :] - xyz[f, None, :, :], axis=2
            )
            features[f] = dist_mat[np.triu_indices(n_atoms, k=1)]

        all_features.append(features)
        print(f"  Traj {ti}: {n_frames} frames, {features.shape[1]} features")

    if len(all_features) == 0:
        print("  No valid trajectories for TICA.")
        return None

    # Concatenate all trajectories
    X = np.concatenate(all_features, axis=0)
    print(f"  Total: {X.shape[0]} frames × {X.shape[1]} features")

    # Run TICA using deeptime
    try:
        from deeptime.decomposition import TICA

        # Determine lag time in frames
        # Estimate: each frame is REPORT_INTERVAL_PS apart
        # lag_time_ps / frame_spacing = lag in frames
        frame_spacing_ps = 10.0  # matches REPORT_INTERVAL_PS in run_md.py
        lag_frames = max(1, int(lag_time_ps / frame_spacing_ps))

        tica = TICA(dim=min(10, X.shape[1] - 1), lagtime=lag_frames)
        tica_result = tica.fit(X).fetch_model()

        # Get implied timescales
        timescales = tica_result.timescales
        # Convert from frames to ps
        timescales_ps = timescales * frame_spacing_ps

        print(f"\n  TICA lag time: {lag_frames} frames ({lag_time_ps} ps)")
        print(f"  Top implied timescales (ps):")
        for i, ts in enumerate(timescales_ps[:8]):
            marker = " ← slow" if ts > lag_time_ps * 2 else ""
            print(f"    IC {i + 1}: {ts:.1f} ps{marker}")

        # Project trajectories onto top 2 TICs for visualization
        projections = tica_result.transform(X)

        # Save
        np.save(DATA_DIR / "tica_timescales.npy", timescales_ps)
        np.save(DATA_DIR / "tica_eigenvectors.npy", tica_result.eigenvectors)
        np.save(DATA_DIR / "tica_projections.npy", projections)
        print(f"\n  TICA results saved to {DATA_DIR}/")

        return timescales_ps, projections

    except ImportError:
        print("  deeptime not available — using manual TICA implementation.")
        return _manual_tica(X)

    except Exception as e:
        print(f"  TICA failed: {e}")
        print("  This is non-fatal — skipping slow variable analysis.")
        return None


def _manual_tica(X):
    """
    Manual TICA implementation as fallback when deeptime is unavailable.

    TICA solves: C(τ) v = λ C(0) v
    where C(τ) is the time-lagged correlation matrix.
    """
    import numpy as np
    from scipy.linalg import eigh

    n_frames = X.shape[0]
    lag = max(1, n_frames // 50)

    # Mean-free data
    X_mean = X.mean(axis=0)
    X0 = X - X_mean

    # Instantaneous covariance C(0)
    C0 = (X0.T @ X0) / (n_frames - 1)
    C0 += 1e-6 * np.eye(C0.shape[0])  # regularization

    # Time-lagged covariance C(τ)
    X_lag = X0[lag:]
    X_ref = X0[:-lag]
    Ctau = (X_ref.T @ X_lag) / (len(X_ref) - 1)

    # Symmetrize
    Ctau_sym = 0.5 * (Ctau + Ctau.T)

    # Solve generalized eigenvalue problem: Ctau v = λ C0 v
    eigenvalues, eigenvectors = eigh(Ctau_sym, C0)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Implied timescales: t_i = -τ / ln(λ_i)
    frame_spacing_ps = 10.0
    with np.errstate(divide="ignore", invalid="ignore"):
        timescales_ps = -lag * frame_spacing_ps / np.log(np.abs(eigenvalues))
        timescales_ps = np.where(
            np.abs(eigenvalues) < 0.999,
            timescales_ps,
            np.inf,
        )

    print(f"\n  Lag: {lag} frames ({lag * frame_spacing_ps} ps)")
    print(f"  Top implied timescales (ps):")
    for i in range(min(8, len(timescales_ps))):
        ts = timescales_ps[i]
        ts_str = f"{ts:.1f}" if ts < 1e6 else "∞"
        marker = " ← slow" if ts > lag * frame_spacing_ps * 2 else ""
        print(f"    IC {i + 1}: {ts_str} ps{marker}")

    # Project onto top 2 ICs
    projections = X0 @ eigenvectors[:, :2]

    np.save(DATA_DIR / "tica_timescales.npy", timescales_ps)
    np.save(DATA_DIR / "tica_eigenvectors.npy", eigenvectors)
    np.save(DATA_DIR / "tica_projections.npy", projections)
    print(f"\n  TICA results saved to {DATA_DIR}/")

    return timescales_ps, projections


if __name__ == "__main__":
    extract_all()
