"""
Train ML bridge: AA contact features → CG LJ parameters.

The target CG LJ epsilon is derived from AA contact probability
via a physical approximation: ε_ij ≈ -k_B T ln(P_contact,ij)

NOTE (MVP limitation): The MLP learns to fit this predefined physical
mapping, not experimental CG data. In production, replace the target
with actual CG parameterization data (e.g., from iterative Boltzmann
inversion or force-matching). For this MVP, the physical approximation
is sufficient to demonstrate the parameter transfer pipeline.

Usage:
    python -m modules.A_triScale.cg_bridge.train_ml

Output:
    data/ml_bridge.pt    — trained PyTorch model
    data/cg_params.csv   — predicted CG parameters for all pairs
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "aa_features.csv"

# Physical constants
K_B_T = 2.479  # kJ/mol at 300K
EPSILON_MIN = 0.1   # min LJ epsilon (kJ/mol)
EPSILON_MAX = 5.0   # max LJ epsilon (kJ/mol)
SIGMA_DEFAULT = 0.6  # nm (approximate CG bead radius)
SIGMA_RANGE = (0.4, 0.8)  # nm


def contact_to_epsilon(contact_prob):
    """
    Physical mapping: higher contact → stronger attraction.

    ε = -k_B T * ln(max(P_contact, P_min))
    Clamped to [EPSILON_MIN, EPSILON_MAX]
    """
    p_min = 0.001  # floor to avoid log(0)
    p = np.clip(np.asarray(contact_prob, dtype=float), p_min, 1.0)
    # Map: P→0 → ε→0, P→1 → ε large
    eps = -K_B_T * np.log(p)
    # Scale: P=0.001 → eps ≈ 17 kJ/mol (too high), rescale
    eps = eps * 0.3  # empirical scaling factor
    return np.clip(eps, EPSILON_MIN, EPSILON_MAX)


def prepare_training_data():
    """Load AA features and create training targets."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run extract_features.py first."
        )

    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} residue pairs from {FEATURES_PATH}")

    # Features: contact features + residue type encoding
    feature_cols = [
        "contact_prob", "avg_distance_nm", "min_distance_nm",
        "hbond_frac", "same_chain", "res_type_i", "res_type_j",
    ]

    X = df[feature_cols].values.astype(np.float32)

    # Targets: epsilon derived from contact prob, sigma = default
    eps_target = contact_to_epsilon(df["contact_prob"].values)
    # Sigma: slightly modulate based on residue sizes
    # PHE (res_type=3) → slightly larger sigma
    avg_res_type = (df["res_type_i"] + df["res_type_j"]) / 2.0
    sigma_target = SIGMA_DEFAULT + 0.02 * (avg_res_type == 3)
    sigma_target = np.clip(sigma_target, *SIGMA_RANGE)

    y = np.column_stack([eps_target, sigma_target]).astype(np.float32)

    print(f"Feature shape: {X.shape}")
    print(f"Target epsilon range: [{eps_target.min():.2f}, {eps_target.max():.2f}]")
    print(f"Target sigma range: [{sigma_target.min():.3f}, {sigma_target.max():.3f}]")

    return X, y, feature_cols, df


def train_mlp(X, y):
    """Train a 3-layer MLP to predict CG parameters."""
    import torch
    import torch.nn as nn

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on {device}")

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # Normalize features and targets
    x_mean = X_t.mean(dim=0)
    x_std = X_t.std(dim=0).clamp(min=1e-6)
    y_mean = y_t.mean(dim=0)
    y_std = y_t.std(dim=0).clamp(min=1e-6)

    X_norm = (X_t - x_mean) / x_std
    y_norm = (y_t - y_mean) / y_std

    # Simple MLP: 7→64→32→16→2
    model = nn.Sequential(
        nn.Linear(7, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5
    )
    loss_fn = nn.MSELoss()

    n_epochs = 2000
    batch_size = min(32, len(X))

    print(f"Training {n_epochs} epochs, batch size {batch_size} …")
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_norm), device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_norm), batch_size):
            idx = perm[i: i + batch_size]
            pred = model(X_norm[idx])
            loss = loss_fn(pred, y_norm[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}: loss = {avg_loss:.6f}")

    # Evaluate
    with torch.no_grad():
        pred_norm = model(X_norm)
        pred = pred_norm * y_std + y_mean
        mae = (pred - y_t).abs().mean(dim=0)
        print(f"  Final MAE: ε={mae[0]:.3f} kJ/mol, σ={mae[1]:.4f} nm")

    # Save model and normalization params
    model_path = DATA_DIR / "ml_bridge.pt"
    torch.save({
        "model_state": model.state_dict(),
        "x_mean": x_mean.cpu(),
        "x_std": x_std.cpu(),
        "y_mean": y_mean.cpu(),
        "y_std": y_std.cpu(),
        "feature_cols": [
            "contact_prob", "avg_distance_nm", "min_distance_nm",
            "hbond_frac", "same_chain", "res_type_i", "res_type_j",
        ],
    }, model_path)
    print(f"Model saved → {model_path}")

    return model, x_mean, x_std, y_mean, y_std


def predict_cg_params(model, x_mean, x_std, y_mean, y_std, X, df):
    """Predict CG parameters for all residue pairs and save CSV."""
    import torch

    device = next(model.parameters()).device
    X_t = torch.tensor(X, device=device)
    X_norm = (X_t - x_mean.to(device)) / x_std.to(device)

    with torch.no_grad():
        pred_norm = model(X_norm)
        pred = pred_norm * y_std.to(device) + y_mean.to(device)

    pred_np = pred.cpu().numpy()

    # Build output dataframe
    out = df[[
        "residue_i", "residue_j", "res_i_name", "res_j_name",
        "same_chain", "contact_prob",
    ]].copy()
    out["cg_epsilon"] = pred_np[:, 0].round(4)
    out["cg_sigma"] = pred_np[:, 1].round(4)

    output_path = DATA_DIR / "cg_params.csv"
    out.to_csv(output_path, index=False)
    print(f"CG parameters saved → {output_path}")

    return out


def main():
    print("=" * 60)
    print("Module A — ML Bridge: AA Features → CG Parameters")
    print("=" * 60)

    X, y, feature_cols, df = prepare_training_data()
    model, x_mean, x_std, y_mean, y_std = train_mlp(X, y)
    cg_params = predict_cg_params(
        model, x_mean, x_std, y_mean, y_std, X, df
    )

    # Print summary
    print(f"\nCG parameter summary:")
    print(f"  ε range: [{cg_params.cg_epsilon.min():.2f}, "
          f"{cg_params.cg_epsilon.max():.2f}] kJ/mol")
    print(f"  σ range: [{cg_params.cg_sigma.min():.3f}, "
          f"{cg_params.cg_sigma.max():.3f}] nm")
    print(f"  Bonded pairs (same chain): "
          f"{cg_params.same_chain.sum()}")

    return cg_params


if __name__ == "__main__":
    main()
