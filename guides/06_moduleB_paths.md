# Step 5: 自适应路径探索（Müller-Brown）

**目标**: 在 Müller-Brown 势能面上演示自适应路径发现——找到多条过渡路径并识别能垒。

**预计时间**: 30 分钟（大部分是 MC 采样）
**GPU**: 否
**依赖**: 无（独立模块，可与 Module A 并行运行）

---

## Cell 5.1: 运行路径探索

```python
import sys
sys.path.insert(0, '.')

from modules.B_pathExploration.run_sampling import run_all

run_all()
```

## Cell 5.2: 可视化势能面 + 发现的路径

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成 MB 表面
def muller_brown(x, y):
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])
    V = np.zeros_like(x)
    for k in range(4):
        V += A[k] * np.exp(a[k]*(x-x0[k])**2 + b[k]*(x-x0[k])*(y-y0[k]) + c[k]*(y-y0[k])**2)
    return V

x = np.linspace(-1.5, 1.2, 150)
y = np.linspace(-0.5, 2.0, 150)
X, Y = np.meshgrid(x, y)
Z = muller_brown(X, Y)
Z = np.clip(Z - Z.min(), 0, 12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左: MC采样分布
ax = axes[0]
samples = np.load("data/mb_samples.npz")
ax.contourf(X, Y, Z, levels=20, cmap="YlOrRd", alpha=0.7)
ax.scatter(samples["samples"][::20, 0], samples["samples"][::20, 1],
           c="blue", s=1, alpha=0.5)
ax.set_title("Metropolis Sampling of Müller-Brown Surface")
ax.set_xlabel("x"); ax.set_ylabel("y")

# 标注3个极小值
minima_pos = [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]
for i, (mx, my) in enumerate(minima_pos):
    ax.plot(mx, my, "o", color="darkred", ms=10, markeredgecolor="white")
    ax.text(mx+0.06, my+0.06, f"M{i+1}", fontsize=10, fontweight="bold", color="darkred")

# 右: 识别出的路径
ax = axes[1]
ax.contourf(X, Y, Z, levels=20, cmap="YlOrRd", alpha=0.7)

# 加载识别出的路径
paths_data = np.load("data/mb_paths.npz", allow_pickle=True)
minima = paths_data["minima"]
strings = paths_data["all_strings"]

colors = ["blue", "cyan", "magenta"]
for si, sdata in enumerate(strings[:3]):
    s = sdata.item() if hasattr(sdata, "item") else sdata
    ax.plot(s["string"][:, 0], s["string"][:, 1], color=colors[si], lw=2.5,
            label=f"Path {si+1}: ΔE={s['barrier']:.1f} kcal/mol")
    # 标记能垒最高点（过渡态）
    barrier_idx = np.argmax(s["energies"])
    ax.plot(s["string"][barrier_idx, 0], s["string"][barrier_idx, 1],
            "x", color=colors[si], ms=10, mew=2)

for i, m in enumerate(minima):
    ax.plot(m[0], m[1], "o", color="darkred", ms=10, markeredgecolor="white")
    ax.text(m[0]+0.06, m[1]+0.06, f"M{i+1}", fontsize=10, fontweight="bold")

ax.set_title("Identified Minimum Free-Energy Paths")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8)

plt.suptitle("Module B: Adaptive Path Exploration Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

期望看到：
- 3个极小值(M1–M3)被正确识别
- 2-3条不同的过渡路径连接极小值
- 'x'标记过渡态（鞍点）位置
- 从M1出发的swarm轨迹分叉进入不同通道

---

## ✓ 检查点 5

```
[ ] data/mb_samples.npz 存在
[ ] data/mb_paths.npz 存在
[ ] data/mb_swarm.npz 存在
[ ] 图中出现3个极小值和2+条路径
```

全部 ✓ 后，模块B完成。
