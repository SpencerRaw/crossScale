# Step 7: 结构→性能预测 + 贝叶斯优化

**目标**: 建立从结构参数到递送性能的预测模型，用BO搜索最优结构。

**预计时间**: 15 分钟
**GPU**: 否
**依赖**: Module A 输出（如果Module A未完成，会用合成数据）

---

## Cell 7.1: 运行 BO 管线

```python
import sys
sys.path.insert(0, '.')

from modules.D_boPrediction.run_bo import run_all

run_all()
```

## Cell 7.2: 可视化 BO 结果

```python
import numpy as np
import matplotlib.pyplot as plt
import json

data = np.load("data/bo_results.npz", allow_pickle=True)
bo_history = data["bo_history"]
importances = data["feature_importances"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 左: BO 收敛
ax = axes[0]
ax.plot(range(len(bo_history)), bo_history, "#E65100", lw=2, marker="o", ms=4)
ax.set_xlabel("BO iteration")
ax.set_ylabel("Objective (loading × stability)")
ax.set_title("Bayesian Optimization Convergence")
ax.axhline(y=max(bo_history), color="gray", ls="--", alpha=0.4)
ax.text(0.5, 0.1, f"Best = {max(bo_history):.3f}",
        transform=ax.transAxes, fontsize=11, fontweight="bold")

# 中: 特征重要性（3个target）
ax = axes[1]
feature_names_short = ["Hydro-\nphobicity", "Charge\ndensity", "Chain\nlength",
                       "Branching", "Core-shell\nratio", "PEGylation"]
target_names = ["Loading", "Release", "Stability"]
x = np.arange(len(feature_names_short))
width = 0.25
colors = ["#1565C0", "#2E7D32", "#E65100"]
for ti in range(3):
    offset = (ti - 1) * width
    ax.bar(x + offset, importances[ti], width, label=target_names[ti],
           color=colors[ti], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(feature_names_short, fontsize=8)
ax.set_ylabel("Feature importance")
ax.set_title("XGBoost Feature Importance")
ax.legend(fontsize=8)

# 右: 最优参数雷达图
ax = axes[2]
ax.set_aspect("equal")
best_params = data["best_params"]
angles = np.linspace(0, 2 * np.pi, len(best_params), endpoint=False).tolist()
values = best_params.tolist()
values += values[:1]  # 闭合
angles += angles[:1]

ax.fill(angles, values, color="#E65100", alpha=0.3)
ax.plot(angles, values, color="#E65100", lw=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([n.replace("_", "\n") for n in [
    "hydrophobicity\nratio", "charge\ndensity", "chain\nlength",
    "branching\ndegree", "core-shell\nratio", "surface\npegylation"
]], fontsize=7)
ax.set_title("BO-Optimized Parameters", fontsize=11)

plt.suptitle("Module D: Structure→Property BO Optimization", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("BO最优参数:")
for name, val in zip(
    ["Hydrophobicity ratio", "Charge density", "Chain length",
     "Branching degree", "Core-shell ratio", "Surface PEGylation"],
    best_params,
):
    print(f"  {name:25s}: {val:.4f}")
```

---

## ✓ 检查点 7

```
[ ] data/bo_results.npz 存在
[ ] BO 收敛曲线上升趋势
[ ] 特征重要性图三个target各有主要驱动特征
[ ] 最优参数雷达图显示合理分布
```

全部 ✓ 后，模块D完成。
