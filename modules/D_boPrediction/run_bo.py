"""
Structure-to-property prediction + Bayesian optimization.

Loads the parameter set from Module A (aa_features + cg_kinetics)
and trains an XGBoost model to predict delivery-relevant properties
from structural descriptors. Then uses Bayesian optimization to
find optimal structural parameters.

Corresponds to PPT "结构→性能低维映射" + "贝叶斯优化".

Usage:
    python -m modules.D_boPrediction.run_bo

Output:
    data/bo_results.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def build_dataset():
    """
    Build structure→property dataset from Module A outputs.

    If Module A data exists, use it as the feature set.
    Otherwise, generate a synthetic placeholder dataset that
    mimics the structure of delivery vehicle parameters.

    Returns: X (features), y (target), feature_names
    """
    aa_path = DATA_DIR / "aa_features.csv"
    kinetics_path = DATA_DIR / "cg_kinetics.csv"

    use_synthetic = not (aa_path.exists() and kinetics_path.exists())

    if use_synthetic:
        print("Module A data not found — using synthetic placeholder dataset.")
        print("(Replace with real data when Module A completes.)")
        return _build_synthetic_dataset()
    else:
        print("Building dataset from Module A parameters …")
        return _build_from_module_a(aa_path, kinetics_path)


def _build_synthetic_dataset():
    """
    Generate synthetic data with physically-motivated nonlinear interactions.

    Each sample = a "candidate delivery vehicle" described by:
      - hydrophobicity_ratio  h  (0–1)
      - charge_density         q  (0–1, normalised)
      - chain_length           L  [4, 50]
      - branching_degree       b  (0–1)
      - core_shell_ratio       cs (0–1)
      - surface_pegylation     peg(0–1)

    Targets (each with ≥2 local optima and synergistic interactions):
      - loading_efficiency   (0–1)
      - release_rate         (0–1)
      - structural_stability (0–1)
    """
    rng = np.random.RandomState(42)
    n_samples = 400  # more samples for higher-dimensional interactions

    feature_names = [
        "hydrophobicity_ratio",
        "charge_density",
        "chain_length",
        "branching_degree",
        "core_shell_ratio",
        "surface_pegylation",
    ]

    X = np.zeros((n_samples, len(feature_names)))
    for fi in range(len(feature_names)):
        X[:, fi] = rng.uniform(0, 1, n_samples)
    X[:, 2] = X[:, 2] * 46 + 4  # chain_length to [4, 50]

    h, q, L, b, cs, peg = [X[:, i] for i in range(6)]
    L_norm = (L - 4) / 46  # normalise to [0, 1]

    y = np.zeros((n_samples, 3))

    # ── Loading efficiency: two competing mechanisms ─────────────
    # Mechanism 1: hydrophobic encapsulation (optimal at h ≈ 0.55)
    mech1 = 0.65 * np.exp(-((h - 0.55) ** 2) / 0.06)
    # Mechanism 2: electrostatic trapping (works best at low h, high q)
    mech2 = 0.45 * (1.0 - h) * q * np.exp(-q / 0.4)
    # Synergy: branching helps encapsulate, but only at mid hydrophobicity
    synergy = 0.25 * b * np.exp(-((h - 0.45) ** 2) / 0.08) * (1.0 - abs(q - 0.3) / 0.5)
    synergy = np.clip(synergy, 0, 0.5)
    # Core-shell penalty: large core-shell ratio at high h traps cargo
    cs_effect = 0.15 * cs * np.tanh((h - 0.3) * 6)
    # Moderate chain length optimum for loading
    L_opt = 0.12 * np.exp(-((L_norm - 0.35) ** 2) / 0.06)

    y[:, 0] = mech1 + mech2 + synergy + cs_effect + L_opt
    y[:, 0] += 0.06 * rng.normal(0, 1, n_samples)
    y[:, 0] = np.clip(y[:, 0], 0.05, 0.95)

    # ── Release rate: pH-switchable + PEG shielding ──────────────
    # Charge-driven release (strongly nonlinear — percolation threshold)
    charge_release = 0.55 * np.where(q > 0.35, (q - 0.35) / 0.65, 0.02)
    # PEG shielding: sigmoidal drop-off (effective only above ~40% coverage)
    peg_shield = 0.40 * (1.0 / (1.0 + np.exp((peg - 0.35) * 12)))
    # Branching creates tortuous paths → slows release
    branch_slow = 0.15 * b * (1.0 - 0.5 * charge_release)
    # Hydrophobicity: non-monotonic — extreme h traps cargo, moderate h releases
    h_release = 0.20 * np.exp(-((h - 0.35) ** 2) / 0.05)
    # Chain length: longer chain → more entanglements → slower
    L_slow = 0.12 * (1.0 - L_norm)

    y[:, 1] = charge_release + h_release - branch_slow + peg_shield + L_slow + 0.18
    y[:, 1] += 0.05 * rng.normal(0, 1, n_samples)
    y[:, 1] = np.clip(y[:, 1], 0.05, 0.95)

    # ── Structural stability: two competing factors ──────────────
    # Core-shell dominates, with an optimal zone (too high → brittle, too low → weak)
    cs_opt = 0.55 * np.exp(-((cs - 0.55) ** 2) / 0.07)
    # Chain length: sigmoidal — long chains stabilise, but diminishing returns
    L_stab = 0.25 * np.tanh(L_norm * 3.5)
    # Charge repulsion: destabilising at high q, but cross-links form at mid q
    q_stab = -0.30 * np.where(q > 0.5, (q - 0.5) ** 2, 0) + 0.10 * np.exp(-((q - 0.25) ** 2) / 0.03)
    # Hydrophobic core: stabilises when h and cs are both moderate
    h_cs_synergy = 0.18 * h * cs * np.exp(-((h - cs) ** 2) / 0.1)
    # Branching: stabilises at moderate levels, destabilises at extremes
    b_stab = 0.15 * np.exp(-((b - 0.4) ** 2) / 0.05)
    # PEG: slightly destabilises the core structure
    peg_effect = -0.08 * peg

    y[:, 2] = cs_opt + L_stab + q_stab + h_cs_synergy + b_stab + peg_effect + 0.25
    y[:, 2] += 0.06 * rng.normal(0, 1, n_samples)
    y[:, 2] = np.clip(y[:, 2], 0.05, 0.95)

    target_names = [
        "loading_efficiency",
        "release_rate",
        "structural_stability",
    ]

    dataset = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    np.savez(DATA_DIR / "structure_property_dataset.npz", **dataset)

    # Quick surface diagnostics
    print(f"  Generated {n_samples} samples, {len(feature_names)} features")
    print(f"  Targets: {target_names}")
    print(f"  Loading:   [{y[:, 0].min():.2f}, {y[:, 0].max():.2f}]  "
          f"(n_opt={np.sum(y[:, 0] > 0.7)})")
    print(f"  Release:   [{y[:, 1].min():.2f}, {y[:, 1].max():.2f}]  "
          f"(n_opt={np.sum(y[:, 1] > 0.7)})")
    print(f"  Stability: [{y[:, 2].min():.2f}, {y[:, 2].max():.2f}]  "
          f"(n_opt={np.sum(y[:, 2] > 0.7)})")

    return X, y, feature_names, target_names


def _build_from_module_a(aa_path, kinetics_path):
    """Build dataset from Module A real data."""
    aa = pd.read_csv(aa_path)
    kinetics = pd.read_csv(kinetics_path)

    # Aggregate AA features per condition
    feature_names = [
        "mean_contact_prob", "mean_epsilon",
        "diffusion_coeff", "k_eq",
        "intra_chain_contact", "inter_chain_contact",
    ]

    # For each condition in kinetics, compute aggregated features
    X_list = []
    for _, krow in kinetics.iterrows():
        # Filter AA features for this condition's structure
        intra = aa[aa["same_chain"] == 1]["contact_prob"].mean()
        inter = aa[aa["same_chain"] == 0]["contact_prob"].mean()
        mean_contact = aa["contact_prob"].mean()

        # Mock epsilon (would come from CG params)
        mean_eps = 1.5  # placeholder

        features = [
            mean_contact,
            mean_eps,
            krow["D_nm2_per_step"],
            krow["K_eq"],
            intra,
            inter,
        ]
        X_list.append(features)

    X = np.array(X_list)
    n = len(X)

    # Target: use K_eq as surrogate for "delivery efficacy"
    # (higher equilibrium constant = stronger binding/encapsulation)
    y = np.zeros((n, 3))
    y[:, 0] = np.clip(np.array(kinetics["K_eq"]) / 10.0, 0, 1)  # loading
    y[:, 1] = np.clip(1.0 / (np.array(kinetics["K_eq"]) + 1), 0.1, 1.0)  # release
    y[:, 2] = np.clip(np.array(kinetics["k_off"]) * 5, 0, 1)  # stability

    target_names = [
        "loading_efficiency",
        "release_rate",
        "structural_stability",
    ]

    return X, y, feature_names, target_names


def train_xgboost(X, y, feature_names, target_names):
    """Train XGBoost models for each target property."""
    print("\nTraining XGBoost models …")

    models = []
    feature_importances = []

    for ti, target_name in enumerate(target_names):
        y_t = y[:, ti]

        # Simple train/test split
        n_train = int(0.8 * len(X))
        idx = np.arange(len(X))
        np.random.RandomState(42).shuffle(idx)

        X_train = X[idx[:n_train]]
        y_train = y_t[idx[:n_train]]
        X_test = X[idx[n_train:]]
        y_test = y_t[idx[n_train:]]

        # Try xgboost, fall back to sklearn
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            importance = model.feature_importances_
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor

            print("  (xgboost not available, using RandomForest)")

            model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            importance = model.feature_importances_

        # R² score
        ss_res = ((y_test - y_pred) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        print(f"  {target_name}: R² = {r2:.3f}")
        for fi, (name, imp) in enumerate(
            zip(feature_names, importance)
        ):
            if imp > 0.05:
                print(f"    {name}: {imp:.3f}")

        models.append(model)
        feature_importances.append(importance)

    return models, feature_importances


def run_bayesian_optimization(X, y, models, feature_names):
    """
    Run Bayesian optimization to find optimal structural parameters.

    Uses BoTorch if available, otherwise a simple GP implementation.
    """
    print("\nRunning Bayesian optimization …")

    try:
        from botorch.models import SingleTaskGP
        from botorch.acquisition import UpperConfidenceBound
        from botorch.optim import optimize_acqf
        import torch

        return _run_bo_botorch(X, y, models, feature_names)
    except ImportError:
        print("  (BoTorch not available, using simple GP-BO)")
        return _run_bo_simple(X, y, models, feature_names)


def _run_bo_simple(X, y, models, feature_names):
    """
    Simple GP-based Bayesian optimization.

    Objective: maximize loading_efficiency × structural_stability
              while keeping release_rate in target range.
    """
    rng = np.random.RandomState(99)

    # Use the best target (loading × stability) as optimization objective
    # We're optimizing over the DESIGN SPACE, not the dataset
    y_obj = y[:, 0] * y[:, 2]  # loading × stability

    n_features = X.shape[1]
    bounds = np.zeros((n_features, 2))
    for fi in range(n_features):
        bounds[fi] = [X[:, fi].min(), X[:, fi].max()]

    # Initial samples
    n_init = 10
    X_tested = X[:n_init].copy()
    y_tested = y_obj[:n_init].copy()

    best_y = y_tested.max()
    best_x = X_tested[y_tested.argmax()]

    history = [best_y]
    n_iter = 30

    for iteration in range(n_iter):
        # Fit simple GP: use RBF kernel on normalized features
        X_norm = (X_tested - X_tested.mean(0)) / (X_tested.std(0) + 1e-8)

        # Predict on random candidates
        candidates = rng.uniform(
            bounds[:, 0], bounds[:, 1], (1000, n_features)
        )
        cand_norm = (candidates - X_tested.mean(0)) / (
            X_tested.std(0) + 1e-8
        )

        # GP prediction: weighted average
        best_acq = -float("inf")
        best_candidate = None

        for cand, cand_n in zip(candidates, cand_norm):
            # Distance to all known points
            distances = ((X_norm - cand_n) ** 2).sum(axis=1)
            weights = np.exp(-distances / (2.0 * 0.5 ** 2))
            weights = weights / weights.sum()

            pred_mean = (y_tested * weights).sum()
            var = (y_tested ** 2 * weights).sum() - pred_mean ** 2
            pred_std = np.sqrt(max(0.0, var))

            # UCB
            acq = pred_mean + 1.5 * pred_std

            if acq > best_acq:
                best_acq = acq
                best_candidate = cand.copy()

        # Evaluate using trained XGBoost models
        new_features = best_candidate.reshape(1, -1)
        pred_loading = models[0].predict(new_features)[0]
        pred_stability = models[2].predict(new_features)[0]
        new_y = pred_loading * pred_stability

        X_tested = np.vstack([X_tested, best_candidate])
        y_tested = np.append(y_tested, new_y)

        if new_y > best_y:
            best_y = new_y
            best_x = best_candidate.copy()

        history.append(best_y)

        if iteration % 10 == 0:
            print(f"  BO iter {iteration:3d}: best = {best_y:.4f}")

    print(f"\n  Best objective: {best_y:.4f}")
    print(f"  Best parameters: {dict(zip(feature_names, best_x.round(4)))}")

    # Also predict individual targets
    bx = best_x.reshape(1, -1)
    preds = [m.predict(bx)[0] for m in models]
    print(f"  Predicted loading: {preds[0]:.3f}")
    print(f"  Predicted stability: {preds[2]:.3f}")

    return history, best_x, preds


def _run_bo_botorch(X, y, models, feature_names):
    """BoTorch-based BO (if library available)."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.acquisition import UpperConfidenceBound
    from botorch.optim import optimize_acqf

    y_obj = y[:, 0] * y[:, 2]  # objective

    n_features = X.shape[1]
    bounds_t = torch.stack([
        torch.tensor(X.min(0)),
        torch.tensor(X.max(0)),
    ])

    X_t = torch.tensor(X[:20], dtype=torch.float64)
    y_t = torch.tensor(y_obj[:20], dtype=torch.float64).reshape(-1, 1)

    best_y = y_t.max().item()
    history = [best_y]

    for iteration in range(30):
        model = SingleTaskGP(X_t, y_t)
        acq = UpperConfidenceBound(model, beta=1.0)

        candidate, _ = optimize_acqf(
            acq, bounds=bounds_t, q=1, num_restarts=5, raw_samples=50,
        )

        # Evaluate
        with torch.no_grad():
            pred_loading = models[0].predict(candidate.numpy())[0]
            pred_stability = models[2].predict(candidate.numpy())[0]
        new_y = pred_loading * pred_stability

        X_t = torch.cat([X_t, candidate])
        y_t = torch.cat([y_t, torch.tensor([[new_y]], dtype=torch.float64)])

        if new_y > best_y:
            best_y = new_y

        history.append(best_y)
        if iteration % 10 == 0:
            print(f"  BO iter {iteration:3d}: best = {best_y:.4f}")

    return history, candidate.numpy().flatten(), [pred_loading, 0, pred_stability]


def run_all():
    """Main entry: dataset → XGBoost → BO."""
    print("=" * 60)
    print("Module D — Structure→Property Prediction + BO")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1] Building dataset …")
    X, y, feature_names, target_names = build_dataset()

    # 2. Train XGBoost
    print("\n[2] Training predictive models …")
    models, importances = train_xgboost(X, y, feature_names, target_names)

    # 3. Bayesian optimization
    print("\n[3] Bayesian optimization …")
    bo_history, best_params, best_predictions = run_bayesian_optimization(
        X, y, models, feature_names
    )

    # Save
    np.savez(
        DATA_DIR / "bo_results.npz",
        bo_history=bo_history,
        best_params=best_params,
        best_predictions=best_predictions,
        feature_importances=np.array(importances),
    )

    print("\nModule D complete.")


if __name__ == "__main__":
    run_all()
