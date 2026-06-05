# Step 1: AA 全原子分子动力学模拟

**目标**: 运行短肽在隐式溶剂中的全原子 MD，生成轨迹用于特征提取。

**预计时间**: 2-4 小时（取决于 GPU，5090 约 2h）
**GPU**: 是（OpenMM CUDA）

---

## 背景

模拟体系：
- **肽序列**: Ac-Ala-Phe-Ala-Phe-Ala-Phe-Ala-Lys-NMe（8残基，两亲性）
- **拷贝数**: 2条肽链（观察链间接触）
- **溶剂**: 隐式溶剂 GBSA-OBC2（加速采样）
- **温度**: 300K
- **轨迹**: 8条独立轨迹 × 10ns = 共80ns
- **输出频率**: 每10ps保存一帧 → 每条轨迹1000帧

## Cell 1.1: 运行 MD 模拟

```python
# 将此 cell 粘贴到 Jupyter Notebook 中运行
# ⚠️ 预计运行时间：2-4 小时

import sys
sys.path.insert(0, '.')

from modules.A_triScale.aa_md.run_md import run_all

run_all()
```

如果报错 `No module named 'openmm'`：回到 Cell 0.2 确认 openmm 安装正确。

**运行时可以观察的日志**：
- `[CUDA]` — 使用 GPU 加速，速度应 >50 ns/day
- `[CPU — slow!]` — GPU 未正确配置，速度仅 ~5 ns/day，需排查

## Cell 1.2: 等待期间，检查第一条轨迹

```python
# 在第一条轨迹跑完后（约15分钟），可以检查输出
from pathlib import Path
import os

data_dir = Path("data/trajectories")
if data_dir.exists():
    dcd_files = sorted(data_dir.glob("traj_*.dcd"))
    print(f"已生成 {len(dcd_files)} 条轨迹文件:")
    for f in dcd_files:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")

    # 检查能量文件
    energy_files = sorted(data_dir.glob("energy_*.csv"))
    if energy_files:
        import pandas as pd
        df = pd.read_csv(energy_files[0])
        print(f"\n能量文件列: {list(df.columns)}")
        print(f"温度范围: {df['Temperature (K)'].min():.1f} – {df['Temperature (K)'].max():.1f} K")
```

---

## Cell 1.3: 确认所有轨迹完成

```python
# 等所有轨迹跑完后运行这个 cell
from pathlib import Path

data_dir = Path("data/trajectories")
dcd_files = sorted(data_dir.glob("traj_*.dcd"))
print(f"轨迹文件: {len(dcd_files)}/8")

topo = Path("data/topology.pdb")
print(f"拓扑文件: {'✓' if topo.exists() else '✗'} — {topo}")

# 每个 DCD 应该 > 1MB
import os
for f in dcd_files:
    size_mb = os.path.getsize(f) / 1e6
    status = "✓" if size_mb > 1 else "✗ 文件太小！"
    print(f"  {status} {f.name}: {size_mb:.1f} MB")
```

---

## ✓ 检查点 1

```
[ ] data/trajectories/ 下有 8 个 traj_0*.dcd 文件
[ ] data/topology.pdb 存在
[ ] 每个 DCD 文件 > 1MB
[ ] 能量文件中温度在 290–310K 范围内
```

全部 ✓ 后进入下一步（特征提取）。
