# Step 8: 生成全部6张图 + 报告

**目标**: 生成预实验报告所需的全部图表。

**预计时间**: 10 分钟
**GPU**: 否

---

## Cell 8.1: 生成所有图

```python
import sys
sys.path.insert(0, '.')

from scripts.plot_all import plot_all

plot_all()
```

## Cell 8.2: 确认所有图

```python
from pathlib import Path

fig_dir = Path("outputs/figures")
expected_figs = [
    "fig1_triScale_overview.png",
    "fig2_fes_slowVars.png",
    "fig3_parameter_validation.png",
    "fig4_adaptive_paths.png",
    "fig5_rl_convergence.png",
    "fig6_bo_optimization.png",
]

print("生成的图表:")
for figname in expected_figs:
    path = fig_dir / figname
    size_kb = path.stat().st_size / 1024 if path.exists() else 0
    status = f"✓ ({size_kb:.0f} KB)" if path.exists() else "✗ 缺失"
    print(f"  {status} — {figname}")
```

## Cell 8.3: 图表清单

| 图 | 内容 | 对应PPT概念 | 数据来源 |
|----|------|------------|---------|
| fig1 | 三尺度参数传递全景 (AA→CG→Continuum) | 多尺度耦合模拟模型 | Module A |
| fig2 | ML增强采样 + 慢变量 + 自由能景观 | 6步流程图 (a→f) | Module A |
| fig3 | 跨尺度参数传递验证 | 参数更新/同步分析 | Module A |
| fig4 | Müller-Brown自适应路径 | 自适应路径探索 | Module B |
| fig5 | RL vs RS vs BO 收敛对比 | 强化学习 | Module C |
| fig6 | BO优化 + 特征重要性 | 贝叶斯优化+低维映射 | Module D |

---

## 完成！

所有模块跑完后的文件清单：

```
data/
├── topology.pdb                    ← AA拓扑
├── trajectories/traj_*.dcd         ← 8条AA轨迹
├── aa_features.csv                 ← AA→CG特征
├── ml_bridge.pt                    ← ML桥模型
├── cg_params.csv                   ← CG参数
├── cg_trajectories/                ← CG模拟
├── cg_kinetics.csv                 ← CG动力学
├── cluster_distribution.csv        ← 连续介质
├── validation_metrics.json         ← 验证指标
├── mb_samples.npz, mb_paths.npz    ← 路径探索
├── rl_results.npz                  ← RL优化
├── bo_results.npz                  ← BO优化
└── structure_property_dataset.npz  ← 结构-性能数据集

outputs/figures/
├── fig1_triScale_overview.png
├── fig2_fes_slowVars.png
├── fig3_parameter_validation.png
├── fig4_adaptive_paths.png
├── fig5_rl_convergence.png
└── fig6_bo_optimization.png
```

下一步：在这6张图的基础上撰写预实验报告。
