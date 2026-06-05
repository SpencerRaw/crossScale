# 预实验运行指南

本文档是**3天预实验的完整操作手册**，在类似Colab的Jupyter Notebook环境中逐步执行。

## 总览

```
模块        案例                      GPU需求    预计时间     执行时机
──────────────────────────────────────────────────────────────────
A.aa_md     短肽AA MD                高 (80%)   2-3 h        Day 1 第一个
A.cg_bridge  ML桥+CG模拟             中 (40%)   1-2 h        Day 1 第二个
A.continuum  Smoluchowski ODE        无 (CPU)   10 min       Day 2 第一个
A.validate   跨尺度验证               无 (CPU)   5 min        Day 2 第二个
B            Müller-Brown路径        无 (CPU)   30 min       Day 2 (可与A并行)
C            HP格子RL                无 (CPU)   15 min       Day 2 (可与A并行)
D            XGBoost+BO              无 (CPU)   15 min       Day 2 第三个
Figures      生成6张图                无 (CPU)   10 min       Day 3
```

## 运行环境

所有模块在一个 conda 环境中运行。GPU 仅模块 A 需要，B/C/D/Figures 纯 CPU。

## 每步的 Notebook Cell

下面每一步都是一个独立的 Jupyter Notebook cell。将代码复制到 Colab/Jupyter 的单元格中，按顺序执行。

## 验证点

每个模块末尾有 **✓ 检查点**，确认这一步产生了预期输出再继续。
