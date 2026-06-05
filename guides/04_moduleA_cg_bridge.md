# Step 3: ML 桥 AA→CG + CG 粗粒化模拟

**目标**: 训练 MLP 将 AA 接触特征映射为 CG 有效势参数，并用学到的参数跑 CG 模拟。

**预计时间**: ML训练 ~5分钟 + CG模拟 ~20分钟
**GPU**: ML训练用GPU (30s)，CG模拟用CPU

---

## Cell 3.1: 训练 ML 桥

```python
import sys
sys.path.insert(0, '.')

from modules.A_triScale.cg_bridge.train_ml import main as train_ml

cg_params = train_ml()
```

输出应显示：
- 训练集大小
- Target ε/σ 范围
- 训练过程中的 loss 下降
- 最终 MAE
- 输出文件路径

## Cell 3.2: 检查 ML 桥质量

```python
import pandas as pd
import matplotlib.pyplot as plt

cg = pd.read_csv("data/cg_params.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ε vs contact_prob
ax = axes[0]
ax.scatter(cg.contact_prob, cg.cg_epsilon, c=cg.same_chain.map({0: "blue", 1: "red"}), alpha=0.6, s=30)
ax.set_xlabel("AA Contact Probability")
ax.set_ylabel("CG ε (kJ/mol)")
ax.set_title("ML Bridge: AA→CG Parameter Mapping")

# ε distribution
ax = axes[1]
ax.hist(cg[cg.same_chain == 1].cg_epsilon, bins=20, alpha=0.5, label="Intra-chain", color="red")
ax.hist(cg[cg.same_chain == 0].cg_epsilon, bins=20, alpha=0.5, label="Inter-chain", color="blue")
ax.set_xlabel("CG ε (kJ/mol)")
ax.set_ylabel("Count")
ax.set_title("CG ε Distribution")
ax.legend()

plt.tight_layout()
plt.show()
```

期望看到：
- ε 和 contact_prob 正相关（接触越频繁，吸引力越强）
- 链内 ε 高于链间 ε（合理，同链残基天然更近）

## Cell 3.3: 运行 CG 模拟

```python
from modules.A_triScale.cg_bridge.run_cg import run_all

all_trajs = run_all()
```

这会跑5个不同浓度（不同box size）的CG模拟。观察输出：
- 每个浓度打印 bead 数、键数、非零ε对
- 进度信息和保存帧数

## Cell 3.4: 提取 CG 动力学参数

```python
from modules.A_triScale.cg_bridge.extract_kinetics import extract_all

kinetics_df = extract_all()
```

输出应显示每个浓度的扩散系数D、k_on、k_off。

## Cell 3.5: 快速可视化 CG 轨迹

```python
import numpy as np
import matplotlib.pyplot as plt

# 载入一个CG轨迹
positions = np.load("data/cg_trajectories/positions_medium.npy")
n_frames, n_beads, _ = positions.shape

# 画第一帧和最后一帧的bead位置
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, frame_idx, title in [(axes[0], 0, "Frame 1"), (axes[1], -1, f"Frame {n_frames}")]:
    pos = positions[frame_idx]
    ax.scatter(pos[:, 0], pos[:, 1], s=50, c=range(n_beads), cmap="tab20")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.suptitle("CG Bead Positions (top view)")
plt.tight_layout()
plt.show()
```

---

## ✓ 检查点 3

```
[ ] data/cg_params.csv 存在
[ ] ε 和 contact_prob 正相关（散点图趋势向上）
[ ] data/cg_trajectories/ 下有 5 个 conditions 的 .npy 文件
[ ] data/cg_kinetics.csv 存在且包含 5 行
[ ] k_on 随浓度增加而增加（趋势合理）
```

全部 ✓ 后进入下一步（连续介质）。
