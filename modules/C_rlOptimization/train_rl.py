"""
Reinforcement learning for sequence optimization on the HP lattice model.

The HP (Hydrophobic-Polar) model is a minimalist protein folding model:
  - 2D square lattice
  - Two residue types: H (hydrophobic, wants to be buried) and P (polar)
  - Energy = −(number of H–H contacts not adjacent in sequence)
  - Goal: find sequence that folds to lowest-energy structure

This directly analogizes "sequence determines structure determines function"
— the same logic as "carrier structure determines delivery performance".

We train a PPO agent to select residue types along a chain,
then compare convergence with random search and Bayesian optimization.

Corresponds to PPT "强化学习" + "结合贝叶斯优化与强化学习".

Usage:
    python -m modules.C_rlOptimization.train_rl

Output:
    data/rl_results.npz
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# HP model parameters
CHAIN_LENGTH = 16  # length of HP sequence
LATTICE_SIZE = 6   # 2D grid size
# Residue encoding: 0=H (hydrophobic), 1=P (polar)
N_ACTIONS = 2


def fold_hp_sequence(sequence, max_attempts=500):
    """
    Fold a given HP sequence on a 2D lattice using Monte Carlo growth.

    Returns the lowest energy structure found.
    Energy = −(# H–H contacts between non-sequence-adjacent residues)

    A structure is a list of (x, y) positions for each residue.
    """
    n = len(sequence)
    rng = np.random.RandomState(hash(str(sequence)) % (2 ** 31))

    best_energy = float("inf")
    best_structure = None

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for _ in range(max_attempts):
        structure = [(0, 0)]
        occupied = {(0, 0)}

        success = True
        for i in range(1, n):
            rng.shuffle(dirs)
            placed = False
            for dx, dy in dirs:
                nx, ny = structure[-1][0] + dx, structure[-1][1] + dy
                if (nx, ny) not in occupied:
                    structure.append((nx, ny))
                    occupied.add((nx, ny))
                    placed = True
                    break
            if not placed:
                success = False
                break

        if not success:
            continue

        # Compute energy
        energy = 0
        for i in range(n):
            if sequence[i] != 0:  # only H contributes
                continue
            for j in range(i + 2, n):  # skip sequence-adjacent
                if sequence[j] != 0:
                    continue
                if abs(structure[i][0] - structure[j][0]) + abs(
                    structure[i][1] - structure[j][1]
                ) == 1:
                    energy -= 1

        if energy < best_energy:
            best_energy = energy
            best_structure = structure

    if best_structure is None:
        return float("inf"), None
    return best_energy, best_structure


def compute_fold_energy_fast(sequence, n_attempts=200):
    """Fast wrapper for batch evaluation."""
    energy, _ = fold_hp_sequence(sequence, max_attempts=n_attempts)
    return energy


class HPEnvironment:
    """
    Gym-like environment for HP sequence optimization.

    State: current sequence (partial or full)
    Action: choose next residue type (0=H or 1=P)
    Reward: −(fold energy) after full sequence is built
    """

    def __init__(self, chain_length=CHAIN_LENGTH):
        self.chain_length = chain_length
        self.sequence = []
        self.done = False

    def reset(self):
        self.sequence = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        """State: fraction of H in current sequence + length ratio."""
        if len(self.sequence) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        h_frac = 1.0 - np.mean(self.sequence)
        len_frac = len(self.sequence) / self.chain_length
        return np.array([h_frac, len_frac], dtype=np.float32)

    def step(self, action):
        self.sequence.append(int(action))

        if len(self.sequence) == self.chain_length:
            self.done = True
            energy = compute_fold_energy_fast(
                np.array(self.sequence), n_attempts=100
            )
            # Reward: negative energy (lower = better folded = more contacts)
            # Scale: typical best energy ≈ −5 to −9 for length 16
            reward = float(-energy) / self.chain_length
        else:
            reward = 0.0

        return self._get_state(), reward, self.done


class SimplePPO:
    """
    Minimal PPO implementation for discrete actions.
    Uses a simple actor-critic network.

    This avoids the heavy gymnasium + stable-baselines3 dependency
    for an MVP that runs in < 5 minutes.
    """

    def __init__(self, state_dim=2, n_actions=N_ACTIONS, lr=0.01):
        self.n_actions = n_actions

        # Tiny neural network: 2 → 16 → 2 (policy) + 1 (value)
        # Weights stored as numpy arrays for simplicity (no torch needed)
        rng = np.random.RandomState(42)
        scale = np.sqrt(2.0 / state_dim)
        self.W1 = rng.normal(0, scale, (state_dim, 16))
        self.b1 = np.zeros(16)
        self.W2_pi = rng.normal(0, 0.1, (16, n_actions))
        self.b2_pi = np.zeros(n_actions)
        self.W2_v = rng.normal(0, 0.1, (16, 1))
        self.b2_v = np.zeros(1)

        self.lr = lr

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def forward(self, state):
        h = self._relu(state @ self.W1 + self.b1)
        logits = h @ self.W2_pi + self.b2_pi
        probs = self._softmax(logits)
        value = float((h @ self.W2_v + self.b2_v)[0])
        return probs, value

    def sample_action(self, state):
        probs, value = self.forward(state)
        action = np.random.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob, value

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Simple policy gradient update (no clipping for MVP)."""
        batch_size = len(states)
        total_pi_loss = 0.0
        total_v_loss = 0.0

        for i in range(batch_size):
            s = states[i]
            a = actions[i]
            old_lp = old_log_probs[i]
            ret = returns[i]
            adv = advantages[i]

            probs, value = self.forward(s)
            new_lp = np.log(probs[a] + 1e-10)

            # Policy gradient with importance sampling ratio
            ratio = np.exp(new_lp - old_lp)
            pi_loss = -ratio * adv

            # Value loss
            v_loss = (ret - value) ** 2

            # Manual gradient for policy (simple case)
            # Update output layer
            # (This is a rough approximation — enough for the MVP)
            h = self._relu(s @ self.W1 + self.b1)

            # Policy gradient
            grad_logits = probs.copy()
            grad_logits[a] -= 1.0
            grad_logits *= adv * ratio

            self.W2_pi -= self.lr * np.outer(h, grad_logits)
            self.b2_pi -= self.lr * grad_logits

            # Value gradient
            v_err = value - ret
            self.W2_v -= self.lr * v_err * h.reshape(-1, 1)
            self.b2_v -= self.lr * v_err

            # Shared layer gradient (from value only for simplicity)
            grad_h = (
                self.W2_pi @ grad_logits
                + self.W2_v.flatten() * v_err
            )
            grad_h[h <= 0] = 0  # ReLU gradient
            self.W1 -= self.lr * np.outer(s, grad_h)
            self.b1 -= self.lr * grad_h

            total_pi_loss += abs(pi_loss)
            total_v_loss += v_loss

        return total_pi_loss / batch_size, total_v_loss / batch_size


def train_ppo(n_episodes=2000):
    """Train PPO agent on HP sequence optimization."""
    print("Training PPO agent …")
    env = HPEnvironment(CHAIN_LENGTH)
    agent = SimplePPO(state_dim=2, n_actions=N_ACTIONS, lr=0.005)

    episode_rewards = []
    best_reward = -float("inf")
    best_sequence = None

    gamma = 0.95  # discount factor

    for episode in range(n_episodes):
        states, actions, log_probs, rewards, values = [], [], [], [], []

        state = env.reset()
        done = False

        while not done:
            action, log_prob, value = agent.sample_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = next_state

        # Compute returns and advantages
        T = len(rewards)
        returns = np.zeros(T)
        advantages = np.zeros(T)

        running_return = 0
        for t in range(T - 1, -1, -1):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        for t in range(T):
            advantages[t] = returns[t] - values[t]

        # Normalize advantages
        if advantages.std() > 1e-10:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # Update
        pi_loss, v_loss = agent.update(
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            returns,
            advantages,
        )

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_sequence = env.sequence.copy()

        if episode % 500 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode:4d}: avg_reward = {avg_r:.3f}, "
                  f"best = {best_reward:.3f}")

    # Compute fold energy of best sequence
    if best_sequence is not None:
        best_energy, best_structure = fold_hp_sequence(
            np.array(best_sequence), max_attempts=1000
        )
        print(f"\n  Best sequence: {''.join('H' if s == 0 else 'P' for s in best_sequence)}")
        print(f"  Best energy: {best_energy}")
    else:
        best_energy = 0

    return episode_rewards, best_sequence, best_energy


def random_search(n_iterations=2000):
    """Random search baseline for HP sequence optimization."""
    print("Running random search baseline …")
    rng = np.random.RandomState(123)
    best_energy = float("inf")
    best_seq = None
    history = []

    for _ in range(n_iterations):
        seq = rng.randint(0, 2, CHAIN_LENGTH)
        energy = compute_fold_energy_fast(seq, n_attempts=100)
        history.append(-energy / CHAIN_LENGTH)

        if energy < best_energy:
            best_energy = energy
            best_seq = seq

    print(f"  Best RS energy: {best_energy}")
    return history, best_seq, best_energy


def bayesian_optimization(n_iterations=200):
    """
    Bayesian optimization baseline for HP sequence optimization.

    Uses Gaussian process to model sequence→energy mapping.
    Since sequence space is discrete (2^CHAIN_LENGTH),
    we use a simple GP with Hamming kernel.
    """
    print("Running Bayesian optimization baseline …")
    rng = np.random.RandomState(456)

    # Initial random samples
    n_init = 20
    X_init = rng.randint(0, 2, (n_init, CHAIN_LENGTH))
    y_init = np.array([
        compute_fold_energy_fast(seq, n_attempts=100)
        for seq in X_init
    ])

    X_known = X_init.copy()
    y_known = y_init.copy()
    best_energy = y_init.min()
    best_seq = X_init[y_init.argmin()]

    history = [-best_energy / CHAIN_LENGTH]

    for iteration in range(n_iterations):
        # Simple GP: use RBF kernel on sequence features
        # For the MVP, use a crude approximation:
        # Predict mean = weighted average of known energies
        # Uncertainty ∝ 1/distance to nearest known point

        best_candidate = None
        best_acq = -float("inf")

        # Sample candidates (random for MVP speed)
        candidates = rng.randint(0, 2, (500, CHAIN_LENGTH))

        for cand in candidates:
            # Compute distance to nearest known point (Hamming)
            distances = (X_known != cand).sum(axis=1)
            nearest_idx = distances.argmin()

            # Mean prediction
            # Use weighted average of 3 nearest neighbors
            nearest_3 = distances.argsort()[:3]
            weights = 1.0 / (distances[nearest_3] + 1.0)
            weights = weights / weights.sum()
            pred_mean = (y_known[nearest_3] * weights).sum()

            # Uncertainty = min_distance (higher = more uncertain)
            uncertainty = distances[nearest_idx]

            # UCB acquisition
            kappa = 2.0
            acq_value = -pred_mean + kappa * uncertainty

            if acq_value > best_acq:
                best_acq = acq_value
                best_candidate = cand.copy()

        # Evaluate
        new_y = compute_fold_energy_fast(best_candidate, n_attempts=100)
        X_known = np.vstack([X_known, best_candidate])
        y_known = np.append(y_known, new_y)

        if new_y < best_energy:
            best_energy = new_y
            best_seq = best_candidate.copy()

        history.append(-best_energy / CHAIN_LENGTH)

        if iteration % 50 == 0:
            print(f"  BO iter {iteration}: best energy = {best_energy}")

    # Pad history to match PPO iterations for plotting
    padded = list(history)
    while len(padded) < 2000:
        padded.append(padded[-1])

    print(f"  Best BO energy: {best_energy}")
    return padded, best_seq, best_energy


def run_all():
    """Run RL, RS, and BO; compare results."""
    print("=" * 60)
    print("Module C — RL Sequence Optimization (HP Lattice)")
    print("=" * 60)

    # Train PPO
    print("\n[1] PPO Training")
    ppo_rewards, ppo_seq, ppo_energy = train_ppo(n_episodes=2000)

    # Random search
    print("\n[2] Random Search")
    rs_rewards, rs_seq, rs_energy = random_search(n_iterations=2000)

    # Bayesian optimization
    print("\n[3] Bayesian Optimization")
    bo_rewards, bo_seq, bo_energy = bayesian_optimization(n_iterations=200)

    # Save results
    np.savez(
        DATA_DIR / "rl_results.npz",
        ppo_rewards=ppo_rewards,
        rs_rewards=rs_rewards,
        bo_rewards=bo_rewards,
        ppo_sequence=ppo_seq,
        rs_sequence=rs_seq,
        bo_sequence=bo_seq,
        ppo_energy=ppo_energy,
        rs_energy=rs_energy,
        bo_energy=bo_energy,
    )

    print(f"\n{'=' * 40}")
    print(f"Final comparison:")
    print(f"  PPO:  energy={ppo_energy}, seq={ppo_seq}")
    print(f"  RS:   energy={rs_energy}, seq={rs_seq}")
    print(f"  BO:   energy={bo_energy}, seq={bo_seq}")

    print("\nModule C complete.")


if __name__ == "__main__":
    run_all()
