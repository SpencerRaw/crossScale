# Step 4: 连续介质模型 + 跨尺度验证

**目标**: 用 CG 提取的动力学参数求解 Smoluchowski 聚集方程，完成三尺度全链验证。

**预计时间**: 10 分钟
**GPU**: 否（CPU ODE 求解）

---

## Cell 4.1: 求解 Smoluchowski 方程

```python
import sys
sys.path.insert(0, '.')

from modules.A_triScale.continuum.solve_ode import run_all

cluster_df = run_all()
```

输出应显示每个CG条件（low→high浓度）的：
- 最终平均团簇尺寸
- 聚集开始时间

期望趋势：浓度越高 → 聚集越快 → 最终团簇越大。

## Cell 4.2: 可视化团簇分布演化

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/cluster_distribution.csv")

# 选一个条件，画团簇分布随时间演化
conditions = df.condition.unique()

fig, axes = plt.subplots(1, len(conditions), figsize=(14, 3.5))
if len(conditions) == 1:
    axes = [axes]

for ax, cond in zip(axes, conditions):
    cond_data = df[df.condition == cond]
    times = sorted(cond_data.time_s.unique())

    # 选4个关键时刻
    key_times = [times[0], times[len(times) // 4], times[len(times) // 2], times[-1]]
    colors = plt.cm.Oranges(np.linspace(0.3, 1, 4))

    for ti, (t_val, color) in enumerate(zip(key_times, colors)):
        t_data = cond_data[cond_data.time_s == t_val]
        sizes = t_data.cluster_size.values[:15]
        concs = t_data.concentration.values[:15]
        ax.plot(sizes, concs, "o-", color=color, ms=4, lw=1,
                label=f"t={t_val:.3f}s")

    ax.set_xlabel("Cluster size k")
    ax.set_ylabel("Concentration c_k")
    ax.set_title(f"[{cond}]")
    ax.set_yscale("log")
    if cond == conditions[-1]:
        ax.legend(fontsize=7)

plt.suptitle("Cluster Size Distribution Evolution", fontsize=13)
plt.tight_layout()
plt.show()

# 对比不同条件下的聚集时间
print("\n聚集动力学汇总:")
for cond in conditions:
    cond_data = df[df.condition == cond]
    # 找mean_cluster_size首次超过1.5的时间
    above = cond_data[cond_data.mean_cluster_size > 1.5]
    agg_time = above.time_s.min() if len(above) > 0 else "未聚集"
    final_mean = cond_data[cond_data.time_s == cond_data.time_s.max()].mean_cluster_size.values[0]
    print(f"  {cond:10s}: agg_time = {agg_time}, final_mean_size = {final_mean:.2f}")
```

## Cell 4.3: 跨尺度验证

```python
from modules.A_triScale.validate import run_all

validation_results = run_all()
```

输出应验证：
1. AA接触概率 ↔ CG ε 的Spearman相关性
2. 连续介质预测 ↔ CG观察的聚集趋势一致性

---

## ✓ 检查点 4

```
[ ] data/cluster_distribution.csv 存在
[ ] 团簇分布演化图出现（浓度越高聚集越快）
[ ] Spearman ρ < 0（更高浓度→更快聚集，负相关合理）
[ ] data/validation_metrics.json 存在
```

全部 ✓ 后，Module A 完整管线跑通。
