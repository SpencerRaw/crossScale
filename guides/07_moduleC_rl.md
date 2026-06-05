# Step 6: 强化学习序列优化（HP 格子模型）

**目标**: 用 PPO 训练一个 agent 在 HP 格子模型上选择最优序列，并与 Random Search 和 BO 对比。

**预计时间**: 15 分钟
**GPU**: 否
**依赖**: 无

---

## Cell 6.1: 运行 RL + RS + BO 对比

```python
import sys
sys.path.insert(0, '.')

from modules.C_rlOptimization.train_rl import run_all

run_all()
```

## Cell 6.2: 可视化收敛对比

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/rl_results.npz", allow_pickle=True)

ppo_r = data["ppo_rewards"]
rs_r = data["rs_rewards"]
bo_r = data["bo_rewards"]
ppo_seq = data["ppo_sequence"]
bo_seq = data["bo_sequence"]
rs_seq = data["rs_sequence"]

ppo_energy = data["ppo_energy"]
rs_energy = data["rs_energy"]
bo_energy = data["bo_energy"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 左: 收敛曲线
ax = axes[0]
ax.plot(np.arange(len(ppo_r)), ppo_r, "#1565C0", lw=1.5, label="PPO (RL)")
ax.plot(np.arange(len(rs_r)), rs_r, "#757575", lw=1.5, label="Random Search")
ax.plot(np.arange(len(bo_r)), bo_r, "#E65100", lw=1.5, label="Bayesian Opt.")
ax.set_xlabel("Evaluation step")
ax.set_ylabel("Best reward (−E/N)")
ax.set_title("Convergence Comparison")
ax.legend(fontsize=9)

# 中: 最终能量对比
ax = axes[1]
methods = ["PPO (RL)", "Random\nSearch", "Bayesian\nOpt."]
energies = [ppo_energy, rs_energy, bo_energy]
colors = ["#1565C0", "#757575", "#E65100"]
bars = ax.bar(methods, energies, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, energies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val}", ha="center", fontweight="bold", fontsize=12)
ax.set_ylabel("Fold Energy (−#HH contacts)")
ax.set_title("Best Energy Found")

# 右: 序列对比
ax = axes[2]
ax.axis("off")

def format_seq(seq):
    return "".join("H" if s == 0 else "P" for s in seq)

methods_seq = ["PPO", "RS", "BO"]
seqs = [ppo_seq, rs_seq, bo_seq]
seq_text = "\n\n".join(
    f"{m}:  {format_seq(s)}  (E={e})"
    for m, s, e in zip(methods_seq, seqs, energies)
)
ax.text(0.5, 0.5, seq_text, ha="center", va="center", fontsize=11,
        fontfamily="monospace", transform=ax.transAxes)
ax.set_title("Optimized Sequences")

plt.suptitle("Module C: RL Sequence Optimization Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\n结果汇总:")
print(f"  PPO 最优能量: {ppo_energy}  ({format_seq(ppo_seq)})")
print(f"  RS  最优能量: {rs_energy}  ({format_seq(rs_seq)})")
print(f"  BO  最优能量: {bo_energy}  ({format_seq(bo_seq)})")
```

期望看到：
- PPO 收敛速度快于 Random Search
- BO 在少迭代时表现最优（样本效率高）
- RL 最终找到的能量与BO接近或更好

---

## ✓ 检查点 6

```
[ ] data/rl_results.npz 存在
[ ] PPO 最终 reward ≥ Random Search
[ ] 三条收敛曲线图出现
[ ] 最优序列中 H 倾向于聚集（形成疏水核心）
```

全部 ✓ 后，模块C完成。
