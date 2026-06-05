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
    Generate synthetic data mimicking a structure→property mapping.

    Each sample = a "candidate delivery vehicle" described by:
      - hydrophobicity_ratio (0–1): fraction of hydrophobic residues
      - charge_density (+e/nm³): positive charge per volume
      - chain_length: number of residues
      - branching_degree (0–1): how branched the structure is
      - core_shell_ratio: size of core / total size
      - surface_pegylation (0–1): PEG coverage

    Target properties:
      - loading_efficiency (0–1)
      - release_rate (h⁻¹)
      - structural_stability (k_B T units)
    """
    rng = np.random.RandomState(42)
    n_samples = 200

    feature_names = [
        "hydrophobicity_ratio",
        "charge_density",
        "chain_length",
        "branching_degree",
        "core_shell_ratio",
        "surface_pegylation",
    ]

    X = np.zeros((n_samples, len(feature_names)))

    # Sample design space with Latin Hypercube-like coverage
    for fi in range(len(feature_names)):
        X[:, fi] = rng.uniform(0, 1, n_samples)

    # Scale chain_length to [4, 50]
    X[:, 2] = X[:, 2] * 46 + 4

    # Generate target properties with physically-motivated nonlinearities
    y = np.zeros((n_samples, 3))

    # Loading efficiency: high for moderate hydrophobicity + low charge
    y[:, 0] = (
        0.7 * np.exp(-((X[:, 0] - 0.5) ** 2) / 0.1)  # optimal at 0.5
        + 0.3 * (1.0 - X[:, 1])  # low charge helps
        + 0.1 * rng.normal(0, 0.05, n_samples)
    )
    y[:, 0] = np.clip(y[:, 0], 0, 1)

    # Release rate: high charge → faster release; PEG slows release
    y[:, 1] = (
        0.5 * X[:, 1]  # charge-driven release
        + 0.2 * X[:, 0]  # slight hydrophobic effect
        - 0.3 * X[:, 5]  # PEG reduces release
        + 0.2
        + 0.05 * rng.normal(0, 0.05, n_samples)
    )
    y[:, 1] = np.clip(y[:, 1], 0.05, 1.0)

    # Structural stability: core-shell + chain length matters
    y[:, 2] = (
        0.4 * X[:, 4]  # core-shell ratio
        + 0.3 * np.tanh(X[:, 2] / 15)  # longer chain = more stable
        + 0.2 * (1.0 - X[:, 1])  # low charge = more stable
        + 0.3
        + 0.05 * rng.normal(0, 0.05, n_samples)
    )
    y[:, 2] = np.clip(y[:, 2], 0, 1)

    target_names = [
        "loading_efficiency",
        "release_rate",
        "structural_stability",
    ]

    # Save
    dataset = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    np.savez(DATA_DIR / "structure_property_dataset.npz", **dataset)

    print(f"  Generated {n_samples} samples, {len(feature_names)} features")
    print(f"  Targets: {target_names}")

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

    y_obj = y[:, 0] * y[:, 2]  # objective

    n_features = X.shape[1]
    bounds_t = torch.stack([
        torch.tensor(X.min(0)),
        torch.tensor(X.max(0)),
    ])

    X_t = torch.tensor(X[:20], dtype=torch.float32)
    y_t = torch.tensor(y_obj[:20], dtype=torch.float32).reshape(-1, 1)

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
        y_t = torch.cat([y_t, torch.tensor([[new_y]], dtype=torch.float32)])

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
