"""
Reinforcement learning for sequence optimisation on the HP lattice model
with the Miyazawa–Jernigan (1996) 20-letter statistical contact potential.

Upgraded from the binary HP alphabet: each position now chooses from all
20 canonical amino acids, and the folding energy uses the full MJ contact
matrix instead of counting only H–H contacts.

Search space: 20^16 ≈ 6.5 × 10^20  (vs 2^16 = 65536 for HP).

Usage:
    python -m modules.C_rlOptimization.train_rl

Output:
    data/rl_results.npz
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 20-letter alphabet ────────────────────────────────────────────
AA_NAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_SINGLE = "ARNDCQEGHILKMFPSTWYV"  # matching order above
N_ACTIONS = 20
CHAIN_LENGTH = 16
LATTICE_SIZE = 6

# ── Miyazawa–Jernigan (1996) contact potential matrix (kT units) ──
# Order: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
# More negative → stronger attraction.  Diagonal = self-interaction.
MJ_MATRIX = np.array([
    # A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V
    [ 0.02, 0.01, 0.04, 0.05, 0.05, 0.03, 0.04, 0.02, 0.01, 0.08, 0.08, 0.01, 0.06, 0.07, 0.02, 0.03, 0.04, 0.03, 0.05, 0.07],  # ALA
    [ 0.01,-0.35,-0.16,-0.24,-0.22,-0.16,-0.06,-0.08,-0.11,-0.06,-0.07,-0.42,-0.10,-0.07,-0.11,-0.10,-0.05,-0.09,-0.10,-0.06],  # ARG
    [ 0.04,-0.16,-0.23,-0.15,-0.09,-0.12,-0.09,-0.08,-0.05,-0.06,-0.09,-0.13,-0.03,-0.09,-0.06,-0.06,-0.04,-0.07,-0.10,-0.06],  # ASN
    [ 0.05,-0.24,-0.15,-0.20,-0.06,-0.16,-0.38,-0.05,-0.08,-0.04,-0.09,-0.21,-0.01,-0.08,-0.05,-0.05,-0.04,-0.08,-0.12,-0.04],  # ASP
    [ 0.05,-0.22,-0.09,-0.06,-0.69,-0.13,-0.08,-0.05,-0.09, 0.04, 0.04,-0.07,-0.13,-0.03,-0.05,-0.02,-0.01,-0.06,-0.03, 0.04],  # CYS
    [ 0.03,-0.16,-0.12,-0.16,-0.13,-0.31,-0.07,-0.06,-0.09,-0.05,-0.07,-0.14,-0.04,-0.10,-0.06,-0.07,-0.04,-0.08,-0.11,-0.05],  # GLN
    [ 0.04,-0.06,-0.09,-0.38,-0.08,-0.07,-0.84,-0.05,-0.07,-0.03,-0.06,-0.12, 0.00,-0.06,-0.04,-0.04,-0.03,-0.07,-0.11,-0.03],  # GLU
    [ 0.02,-0.08,-0.08,-0.05,-0.05,-0.06,-0.05,-0.18,-0.05,-0.01,-0.02,-0.06,-0.02,-0.03,-0.03,-0.03,-0.01,-0.03,-0.04,-0.01],  # GLY
    [ 0.01,-0.11,-0.05,-0.08,-0.09,-0.09,-0.07,-0.05,-0.42,-0.03,-0.04,-0.12,-0.03,-0.12,-0.04,-0.05,-0.03,-0.11,-0.14,-0.03],  # HIS
    [ 0.08,-0.06,-0.06,-0.04, 0.04,-0.05,-0.03,-0.01,-0.03,-0.19,-0.17,-0.05,-0.08, 0.00,-0.01,-0.03,-0.01,-0.03,-0.02,-0.11],  # ILE
    [ 0.08,-0.07,-0.09,-0.09, 0.04,-0.07,-0.06,-0.02,-0.04,-0.17,-0.23,-0.06,-0.11,-0.01,-0.03,-0.04,-0.03,-0.03,-0.04,-0.04],  # LEU
    [ 0.01,-0.42,-0.13,-0.21,-0.07,-0.14,-0.12,-0.06,-0.12,-0.05,-0.06,-0.53,-0.08,-0.12,-0.06,-0.07,-0.04,-0.08,-0.11,-0.04],  # LYS
    [ 0.06,-0.10,-0.03,-0.01,-0.13,-0.04, 0.00,-0.02,-0.03,-0.08,-0.11,-0.08,-0.62,-0.05,-0.04,-0.04,-0.04,-0.06,-0.05,-0.07],  # MET
    [ 0.07,-0.07,-0.09,-0.08,-0.03,-0.10,-0.06,-0.03,-0.12, 0.00,-0.01,-0.12,-0.05,-0.43,-0.05,-0.05,-0.03,-0.11,-0.13, 0.00],  # PHE
    [ 0.02,-0.11,-0.06,-0.05,-0.05,-0.06,-0.04,-0.03,-0.04,-0.01,-0.03,-0.06,-0.04,-0.05,-0.19,-0.03,-0.02,-0.05,-0.06,-0.01],  # PRO
    [ 0.03,-0.10,-0.06,-0.05,-0.02,-0.07,-0.04,-0.03,-0.05,-0.03,-0.04,-0.07,-0.04,-0.05,-0.03,-0.21,-0.03,-0.05,-0.06,-0.02],  # SER
    [ 0.04,-0.05,-0.04,-0.04,-0.01,-0.04,-0.03,-0.01,-0.03,-0.01,-0.03,-0.04,-0.04,-0.03,-0.02,-0.03,-0.17,-0.03,-0.04,-0.01],  # THR
    [ 0.03,-0.09,-0.07,-0.08,-0.06,-0.08,-0.07,-0.03,-0.11,-0.03,-0.03,-0.08,-0.06,-0.11,-0.05,-0.05,-0.03,-0.48,-0.14,-0.03],  # TRP
    [ 0.05,-0.10,-0.10,-0.12,-0.03,-0.11,-0.11,-0.04,-0.14,-0.02,-0.04,-0.11,-0.05,-0.13,-0.06,-0.06,-0.04,-0.14,-0.40,-0.02],  # TYR
    [ 0.07,-0.06,-0.06,-0.04, 0.04,-0.05,-0.03,-0.01,-0.03,-0.11,-0.04,-0.04,-0.07, 0.00,-0.01,-0.02,-0.01,-0.03,-0.02,-0.17],  # VAL
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Folding — MJ-contact-energy scoring on a 2D lattice
# ═══════════════════════════════════════════════════════════════════

def fold_sequence(sequence, max_attempts=500):
    """
    Fold a sequence of AA indices (0–19) on a 2D square lattice.

    Energy = Σ MJ_contact[aa_i, aa_j] for all non-sequence-adjacent
             residue pairs that are lattice-neighbors.

    Returns: (energy, structure) — more negative = better folded.
    """
    n = len(sequence)
    rng = np.random.RandomState(hash(str(sequence)) % (2 ** 31))
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    best_energy = float("inf")
    best_structure = None

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

        # Compute MJ contact energy
        energy = 0.0
        for i in range(n):
            ai = sequence[i]
            for j in range(i + 2, n):  # skip sequence-adjacent
                aj = sequence[j]
                if abs(structure[i][0] - structure[j][0]) + abs(
                    structure[i][1] - structure[j][1]
                ) == 1:
                    energy += MJ_MATRIX[ai, aj]

        if energy < best_energy:
            best_energy = energy
            best_structure = structure

    if best_structure is None:
        return float("inf"), None
    return best_energy, best_structure


def compute_fold_energy_fast(sequence, n_attempts=200):
    """Fast wrapper for batch evaluation."""
    energy, _ = fold_sequence(sequence, max_attempts=n_attempts)
    return energy


def composition_penalty(sequence):
    """
    Penalise extreme composition bias to prevent trivial solutions
    (e.g. all-K/R sequences exploiting the MJ matrix).

    Two components:
      1. Any single AA > 30% → linear penalty
      2. Bonus for covering diverse physico-chemical classes
    """
    seq = np.asarray(sequence, dtype=int)
    n = len(seq)
    bc = np.bincount(seq, minlength=20)
    max_frac = bc.max() / n
    aa_bias = max(0.0, max_frac - 0.30) * 2.0

    # Class coverage: penalise missing major classes
    hydrophobic = np.isin(seq, [0, 9, 10, 12, 13, 14, 17, 18, 19])
    classes_present = sum([
        hydrophobic.any(),
        np.isin(seq, [1, 11]).any(),           # positive
        np.isin(seq, [3, 6]).any(),            # negative
        np.isin(seq, [2, 5, 7, 15, 16]).any(), # polar
    ])
    class_bonus = -0.25 * max(0, 3 - classes_present)

    return aa_bias + class_bonus


# ═══════════════════════════════════════════════════════════════════
# RL Environment
# ═══════════════════════════════════════════════════════════════════

class SequenceEnvironment:
    """
    Build a sequence one residue at a time (20 choices per step).

    State (7-dim):
      [hydrophobic_frac, positive_frac, negative_frac, polar_frac,
       aromatic_frac, special_frac, length_frac]

    Action: AA index (0–19)
    Reward: −E_fold / CHAIN_LENGTH  (normalised, higher is better)
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
        """7-dim state: AA class fractions + length ratio."""
        L = len(self.sequence)
        if L == 0:
            return np.array([0.0] * 7, dtype=np.float32)

        seq = np.array(self.sequence)
        # Physico-chemical classes
        hydrophobic = np.isin(seq, [0, 9, 10, 12, 13, 14, 17, 18, 19]).mean()  # AILMFPTWYV
        positive    = np.isin(seq, [1, 11]).mean()   # R, K
        negative    = np.isin(seq, [3, 6]).mean()    # D, E
        polar       = np.isin(seq, [2, 5, 7, 15, 16]).mean()  # N, Q, G, S, T
        aromatic    = np.isin(seq, [13, 17, 18]).mean()  # F, W, Y
        special     = np.isin(seq, [4, 8]).mean()    # C, H
        len_frac    = L / self.chain_length

        return np.array([
            hydrophobic, positive, negative, polar, aromatic, special, len_frac,
        ], dtype=np.float32)

    def step(self, action):
        self.sequence.append(int(action))
        if len(self.sequence) == self.chain_length:
            self.done = True
            seq_arr = np.array(self.sequence)
            energy = compute_fold_energy_fast(seq_arr, n_attempts=150)
            comp_pen = composition_penalty(seq_arr)
            # Higher energy (more negative MJ) = better folding
            # Lower composition penalty = more diverse
            reward = float(-energy) / self.chain_length - comp_pen
        else:
            reward = 0.0
        return self._get_state(), reward, self.done


# ═══════════════════════════════════════════════════════════════════
# Simple PPO (numpy, no torch needed — MVP-friendly)
# ═══════════════════════════════════════════════════════════════════

class SimplePPO:
    def __init__(self, state_dim=7, n_actions=N_ACTIONS, lr=0.005,
                 ent_coef=0.02):
        self.n_actions = n_actions
        self.ent_coef = ent_coef
        rng = np.random.RandomState(42)

        scale1 = np.sqrt(2.0 / state_dim)
        self.W1 = rng.normal(0, scale1, (state_dim, 128))
        self.b1 = np.zeros(128)
        self.W2 = rng.normal(0, np.sqrt(2.0 / 128), (128, 64))
        self.b2 = np.zeros(64)
        self.W_pi = rng.normal(0, 0.05, (64, n_actions))
        self.b_pi = np.zeros(n_actions)
        self.W_v = rng.normal(0, 0.05, (64, 1))
        self.b_v = np.zeros(1)

        self.lr = lr
        self.steps_trained = 0

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def _softmax(self, x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def forward(self, state):
        h1 = self._relu(state @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W_pi + self.b_pi
        probs = self._softmax(logits)
        value = float((h2 @ self.W_v + self.b_v)[0])
        return probs, value, (h1, h2)

    def sample_action(self, state):
        probs, value, _ = self.forward(state)
        action = np.random.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob, value, probs

    def update(self, states, actions, old_log_probs, returns, advantages):
        batch_size = len(states)
        clip_eps = 0.2
        max_grad_norm = 1.0

        for i in range(batch_size):
            s, a, old_lp, ret, adv = (
                states[i], actions[i], old_log_probs[i],
                returns[i], advantages[i],
            )
            # Clip extreme advantage values to prevent gradient explosion
            adv = np.clip(adv, -5.0, 5.0)

            probs, value, (h1, h2) = self.forward(s)
            new_lp = np.log(probs[a] + 1e-10)
            ratio = np.exp(np.clip(new_lp - old_lp, -10, 10))

            # PPO clipped policy gradient
            clipped_ratio = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_gain = min(ratio * adv, clipped_ratio * adv)

            grad_logits = probs.copy()
            grad_logits[a] -= 1.0
            grad_logits *= policy_gain

            # Entropy bonus: -grad(H) encourages exploration
            grad_logits += self.ent_coef * probs * (np.log(probs + 1e-10) + 1)
            # Clip logits gradient
            gn = np.sqrt((grad_logits ** 2).sum()) + 1e-10
            if gn > max_grad_norm:
                grad_logits *= max_grad_norm / gn

            self.W_pi -= self.lr * np.outer(h2, grad_logits)
            self.b_pi -= self.lr * grad_logits

            # Value gradient (clipped)
            v_err = np.clip(value - ret, -5.0, 5.0)
            self.W_v -= self.lr * v_err * h2.reshape(-1, 1)
            self.b_v -= self.lr * v_err

            # Shared layers
            grad_h2 = self.W_pi @ grad_logits + self.W_v.flatten() * v_err
            grad_h2[h2 <= 0] = 0
            gn_h2 = np.sqrt((grad_h2 ** 2).sum()) + 1e-10
            if gn_h2 > max_grad_norm:
                grad_h2 *= max_grad_norm / gn_h2
            self.W2 -= self.lr * np.outer(h1, grad_h2)
            self.b2 -= self.lr * grad_h2

            grad_h1 = self.W2 @ grad_h2
            grad_h1[h1 <= 0] = 0
            gn_h1 = np.sqrt((grad_h1 ** 2).sum()) + 1e-10
            if gn_h1 > max_grad_norm:
                grad_h1 *= max_grad_norm / gn_h1
            self.W1 -= self.lr * np.outer(s, grad_h1)
            self.b1 -= self.lr * grad_h1

        self.steps_trained += batch_size
        # Decay entropy coefficient over time
        if self.steps_trained % 5000 == 0:
            self.ent_coef = max(0.002, self.ent_coef * 0.85)


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_ppo(n_episodes=5000):
    """Train PPO agent on 20-letter MJ sequence optimisation."""
    print("Training PPO agent (20-letter MJ, 16-residue chain) …")
    env = SequenceEnvironment(CHAIN_LENGTH)
    agent = SimplePPO(state_dim=7, n_actions=N_ACTIONS, lr=0.001)

    episode_rewards = []
    best_reward = -float("inf")
    best_sequence = None
    best_raw_mj = float("inf")
    gamma = 0.95

    for episode in range(n_episodes):
        states, actions, log_probs, rewards, values = [], [], [], [], []

        state = env.reset()
        done = False
        while not done:
            action, log_prob, value, probs = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            state = next_state

        T = len(rewards)
        returns = np.zeros(T)
        running_return = 0.0
        for t in range(T - 1, -1, -1):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        advantages = returns - np.array(values)
        if advantages.std() > 1e-10:
            advantages = (advantages - advantages.mean()) / advantages.std()

        agent.update(
            np.array(states, dtype=np.float32),
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
            seq_arr = np.array(best_sequence)
            raw_e = compute_fold_energy_fast(seq_arr, n_attempts=200)
            best_raw_mj = raw_e

        if episode % 1000 == 0:
            avg_r = np.mean(episode_rewards[-200:]) if len(episode_rewards) >= 200 else np.mean(episode_rewards)
            print(f"  Episode {episode:4d}: avg_reward = {avg_r:.3f}, "
                  f"best = {best_reward:.3f}, ent_coef = {agent.ent_coef:.4f}")

    if best_sequence is not None:
        best_energy, best_structure = fold_sequence(
            np.array(best_sequence), max_attempts=1000
        )
        seq_str = "".join(AA_SINGLE[s] for s in best_sequence)
        comp = composition_penalty(np.array(best_sequence))
        print(f"\n  Best sequence: {seq_str}")
        print(f"  Raw MJ energy:     {best_energy:.2f} kT")
        print(f"  Composition penalty: {comp:.3f}")
        print(f"  Constrained reward:  {best_reward:.3f} ( = {-best_energy/CHAIN_LENGTH:.1f} - {comp:.3f})")
    else:
        best_energy = 0.0

    return episode_rewards, best_sequence, best_energy, seq_str if best_sequence else ""


# ═══════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════

def random_search(n_iterations=5000):
    """Random search baseline with composition penalty for fair comparison."""
    print("Running random search baseline (20-letter, constrained) …")
    rng = np.random.RandomState(123)
    best_constrained = -float("inf")
    best_energy = float("inf")
    best_seq = None
    history = []

    for _ in range(n_iterations):
        seq = rng.randint(0, N_ACTIONS, CHAIN_LENGTH)
        energy = compute_fold_energy_fast(seq, n_attempts=100)
        comp = composition_penalty(seq)
        constrained = -energy / CHAIN_LENGTH - comp
        history.append(constrained)
        if constrained > best_constrained:
            best_constrained = constrained
            best_energy = energy
            best_seq = seq

    comp_best = composition_penalty(best_seq)
    best_str = "".join(AA_SINGLE[s] for s in best_seq)
    print(f"  Best RS: raw_MJ = {best_energy:.2f} kT, constrained = {best_constrained:.3f}, "
          f"seq: {best_str}")
    return history, best_seq, best_energy, best_str


def bayesian_optimization(n_iterations=300):
    """BO baseline using GP with Hamming kernel + composition penalty."""
    print("Running Bayesian optimisation baseline (20-letter, constrained) …")
    rng = np.random.RandomState(456)

    n_init = 20
    X_init = rng.randint(0, N_ACTIONS, (n_init, CHAIN_LENGTH))
    y_init = np.array([
        -compute_fold_energy_fast(seq, n_attempts=100) / CHAIN_LENGTH
        - composition_penalty(seq)
        for seq in X_init
    ])

    X_known = X_init.copy()
    y_known = y_init.copy()
    best_idx = y_init.argmax()
    best_constrained = y_init[best_idx]
    best_seq = X_init[best_idx]
    history = [best_constrained]

    for iteration in range(n_iterations):
        candidates = rng.randint(0, N_ACTIONS, (2000, CHAIN_LENGTH))

        best_candidate = None
        best_acq = -float("inf")

        for cand in candidates:
            distances = (X_known != cand).sum(axis=1)
            nearest_3 = distances.argsort()[:3]
            weights = 1.0 / (distances[nearest_3] + 1.0)
            weights = weights / weights.sum()
            pred_mean = (y_known[nearest_3] * weights).sum()
            uncertainty = distances[nearest_3[0]]
            acq_value = pred_mean + 1.5 * uncertainty  # UCB
            if acq_value > best_acq:
                best_acq = acq_value
                best_candidate = cand.copy()

        new_e = compute_fold_energy_fast(best_candidate, n_attempts=100)
        new_constrained = -new_e / CHAIN_LENGTH - composition_penalty(best_candidate)
        X_known = np.vstack([X_known, best_candidate])
        y_known = np.append(y_known, new_constrained)

        if new_constrained > best_constrained:
            best_constrained = new_constrained
            best_seq = best_candidate.copy()
        history.append(best_constrained)

        if iteration % 75 == 0:
            print(f"  BO iter {iteration:3d}: best constrained = {best_constrained:.3f}")

    # Pad to match PPO length
    while len(history) < 5000:
        history.append(history[-1])

    best_energy = compute_fold_energy_fast(best_seq, n_attempts=200)
    comp_best = composition_penalty(best_seq)
    best_str = "".join(AA_SINGLE[s] for s in best_seq)
    print(f"  Best BO: raw_MJ = {best_energy:.2f} kT, constrained = {best_constrained:.3f}, "
          f"seq: {best_str}")
    return history, best_seq, best_energy, best_str


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_all():
    print("=" * 60)
    print("Module C — RL Sequence Optimisation (MJ 20-letter lattice)")
    print("=" * 60)

    print("\n[1] PPO Training")
    ppo_rewards, ppo_seq, ppo_energy, ppo_str = train_ppo(n_episodes=5000)

    print("\n[2] Random Search")
    rs_rewards, rs_seq, rs_energy, rs_str = random_search(n_iterations=5000)

    print("\n[3] Bayesian Optimisation")
    bo_rewards, bo_seq, bo_energy, bo_str = bayesian_optimization(n_iterations=300)

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

    # Report both raw MJ energy and constrained reward for fair comparison
    print(f"\n{'=' * 40}")
    print(f"Final comparison (constrained reward = -E/N - composition_penalty):")
    print(f"  PPO:  raw_MJ = {ppo_energy:.2f} kT,  seq = {ppo_str}")
    print(f"  RS:   raw_MJ = {rs_energy:.2f} kT,  seq = {rs_str}")
    print(f"  BO:   raw_MJ = {bo_energy:.2f} kT,  seq = {bo_str}")
    if ppo_seq is not None:
        ppo_comp = composition_penalty(np.array(ppo_seq))
        print(f"  PPO constrained reward = {-ppo_energy/CHAIN_LENGTH:.3f} - {ppo_comp:.3f} = {-ppo_energy/CHAIN_LENGTH - ppo_comp:.3f}")
    if rs_seq is not None:
        rs_comp = composition_penalty(np.array(rs_seq))
        print(f"  RS  constrained reward = {-rs_energy/CHAIN_LENGTH:.3f} - {rs_comp:.3f} = {-rs_energy/CHAIN_LENGTH - rs_comp:.3f}")
    if bo_seq is not None:
        bo_comp = composition_penalty(np.array(bo_seq))
        print(f"  BO  constrained reward = {-bo_energy/CHAIN_LENGTH:.3f} - {bo_comp:.3f} = {-bo_energy/CHAIN_LENGTH - bo_comp:.3f}")
    print("\nModule C complete.")


if __name__ == "__main__":
    run_all()
