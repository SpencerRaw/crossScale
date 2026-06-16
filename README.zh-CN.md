# crossScale — ML加速多尺度生物分子模拟

> **预实验 MVP**：一个模块化框架，桥接全原子MD、粗粒化ML势函数和连续介质动力学——3天搭建，可扩展架构。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

---

## 为什么需要 crossScale？

模拟生物分子界面（膜上多肽、蛋白质聚集）受限于时间尺度：全原子MD捕捉原子级细节但最多跑微秒，而生物过程跨越毫秒到秒。跨尺度模拟是解决这个问题的标准方法，但参数传递（AA→CG→连续介质）依赖人工标定，费时且不可推广。

**crossScale** 用 ML 自动学习跨尺度参数映射，将方法泛化到任意肽序列和膜组成。

```
Module A (主干)      Module B           Module C           Module D
全原子MD             路径探索           RL优化             BO预测
  │                     │                  │                  ▲
  ▼                     ▼                  ▼                  │
ML桥接                                           ──────────────┘
(AA → CG)           自适应采样          PPO序列优化        XGBoost
  │                                                      + 贝叶斯优化
  ▼
粗粒化MD
  │
  ▼
连续介质ODE
```

---

## 四模块架构

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| **A — 三尺度管道** | AA MD → ML桥接 → CG MD → 连续介质 | OpenMM + PyTorch + SciPy |
| **B — 路径探索** | 自适应采样找最小能量路径和过渡态 | Müller-Brown 势能面 |
| **C — RL优化** | 强化学习设计折叠肽序列 | PPO + HP格点模型 |
| **D — BO预测** | 多目标贝叶斯优化预测材料性能 | XGBoost + BoTorch |

---

## 快速开始

```bash
# Conda 安装
conda env create -f environment.yml
conda activate crossScale

# 运行主干管道 (Module A)
python -m modules.A_triScale.aa_md.run_md
python -m modules.A_triScale.aa_md.extract_features
python -m modules.A_triScale.cg_bridge.train_ml
python -m modules.A_triScale.cg_bridge.run_cg
python -m modules.A_triScale.continuum.solve_ode

# 生成所有图
python scripts/plot_all.py
```

---

## 论文路线图

- [x] MVP 框架搭建（4模块 + 数据合约 + 图生成）
- [ ] GPU 实跑（真实肽/膜体系）
- [ ] 参数传递验证（AA→CG→连续介质闭环）
- [ ] 方法论文（JCTC/JCP 目标）
- [ ] 扩展到全原子等变架构（Nature Comp Sci 目标）

---

## 引用

```bibtex
@software{crossScale2026,
  author = {Yiwei Xu},
  title = {crossScale: ML-Accelerated Multi-Scale Biomolecular Simulation},
  year = {2026},
  url = {https://github.com/SpencerRaw/crossScale}
}
```

---

## License

MIT
