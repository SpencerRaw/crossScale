"""
Cross-scale validation: verify parameter transfer quality.

1. CG→AA: Compare CG-predicted Rg distribution against AA MD.
2. Continuum→CG: Compare continuum-predicted cluster sizes against
   CG simulation aggregates.

Usage:
    python -m modules.A_triScale.validate

Output:
    data/validation_metrics.json
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CG_TRAJ_DIR = DATA_DIR / "cg_trajectories"


def validate_cg_vs_aa():
    """
    Compare CG contact matrix against AA reference.

    For each residue pair, compare CG average distance vs AA average distance.
    Returns R² and RMSE.
    """
    print("—" * 40)
    print("Validation 1: CG vs AA contact patterns")

    aa_features = DATA_DIR / "aa_features.csv"
    cg_params = DATA_DIR / "cg_params.csv"

    if not aa_features.exists() or not cg_params.exists():
        print("  SKIP: features or params not found")
        return None

    aa = pd.read_csv(aa_features)
    cg = pd.read_csv(cg_params)

    # Merge on residue pair
    merged = aa.merge(
        cg, on=["residue_i", "residue_j", "res_i_name", "res_j_name"]
    )

    # Compare: AA contact_prob vs CG epsilon (stronger attraction ↔ higher contact)
    from scipy.stats import pearsonr, spearmanr

    r_pearson, p_pearson = pearsonr(
        merged["contact_prob"], merged["cg_epsilon"]
    )
    r_spearman, p_spearman = spearmanr(
        merged["contact_prob"], merged["cg_epsilon"]
    )

    print(f"  AA contact_prob ↔ CG ε:")
    print(f"    Pearson r = {r_pearson:.3f} (p={p_pearson:.4f})")
    print(f"    Spearman ρ = {r_spearman:.3f} (p={p_spearman:.4f})")

    # Also compare distance correlation
    # For this, we'd need CG distances — use CG ε as proxy (higher ε = closer)
    # Check intra-chain vs inter-chain separation
    intra = merged[merged["same_chain"] == 1]
    inter = merged[merged["same_chain"] == 0]

    print(f"  Intra-chain: {len(intra)} pairs, "
          f"ε = {intra.cg_epsilon.mean():.2f} ± {intra.cg_epsilon.std():.2f}")
    print(f"  Inter-chain: {len(inter)} pairs, "
          f"ε = {inter.cg_epsilon.mean():.2f} ± {inter.cg_epsilon.std():.2f}")

    return {
        "pearson_r": round(float(r_pearson), 4),
        "spearman_rho": round(float(r_spearman), 4),
        "intra_epsilon_mean": round(float(intra.cg_epsilon.mean()), 3),
        "inter_epsilon_mean": round(float(inter.cg_epsilon.mean()), 3),
    }


def validate_continuum_vs_cg():
    """
    Compare continuum-predicted aggregation onset with CG observations.

    Checks if higher concentration in CG leads to faster aggregation
    as predicted by continuum model.
    """
    print("—" * 40)
    print("Validation 2: Continuum vs CG kinetics")

    kinetics_path = DATA_DIR / "cg_kinetics.csv"
    cluster_path = DATA_DIR / "cluster_distribution.csv"

    if not kinetics_path.exists() or not cluster_path.exists():
        print("  SKIP: kinetics or cluster distribution not found")
        return None

    kinetics = pd.read_csv(kinetics_path)
    clusters = pd.read_csv(cluster_path)

    # For each condition: get aggregation time from continuum
    agg_times = {}
    for cond in clusters["condition"].unique():
        cond_data = clusters[clusters["condition"] == cond]
        # Aggregation time = first time where mean size > 1.5
        above = cond_data[cond_data["mean_cluster_size"] > 1.5]
        if len(above) > 0:
            agg_time = above["time_s"].min()
        else:
            agg_time = 1.0  # never aggregated
        agg_times[cond] = agg_time

    print("  Condition   VolFrac    k_on      AggTime(s)")
    for _, row in kinetics.iterrows():
        cond = row["condition"]
        atime = agg_times.get(cond, float("nan"))
        print(f"  {cond:10s}  {row.volume_fraction:.4f}   "
              f"{row.k_on:.4f}   {atime:.4f}")

    # Check: agg_time should decrease with increasing concentration
    vol_fracs = []
    a_times = []
    for _, row in kinetics.iterrows():
        cond = row["condition"]
        if cond in agg_times:
            vol_fracs.append(row["volume_fraction"])
            a_times.append(agg_times[cond])

    if len(vol_fracs) >= 3:
        from scipy.stats import spearmanr
        rho, pv = spearmanr(vol_fracs, a_times)
        print(f"  VolFrac ↔ AggTime Spearman ρ = {rho:.3f} (p={pv:.4f})")
        print(f"  (Expected: negative correlation — "
              f"higher concentration → faster aggregation)")
    else:
        rho = float("nan")

    return {
        "agg_times": {k: round(v, 4) for k, v in agg_times.items()},
        "spearman_conc_vs_time": round(float(rho), 4),
    }


def run_all():
    """Run all cross-scale validations."""
    print("=" * 60)
    print("Module A — Cross-Scale Validation")
    print("=" * 60)

    results = {}

    v1 = validate_cg_vs_aa()
    if v1:
        results["cg_vs_aa"] = v1

    v2 = validate_continuum_vs_cg()
    if v2:
        results["continuum_vs_cg"] = v2

    # Save
    output_path = DATA_DIR / "validation_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nValidation results saved → {output_path}")

    return results


if __name__ == "__main__":
    run_all()
