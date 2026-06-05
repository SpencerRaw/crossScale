# Step 0: 环境搭建

**目标**: 在服务器上安装所有依赖，验证GPU可用。

**预计时间**: 15-30 分钟

---

## Cell 0.1: 创建 conda 环境

```bash
# 在终端中运行（非 notebook cell）
# 如果服务器上还没有 conda，先安装 Miniconda:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

conda env create -f environment.yml
conda activate crossScale
```

如果 `environment.yml` 中的包版本冲突，分批安装：

```bash
conda create -n crossScale python=3.10 -y
conda activate crossScale
conda install -c conda-forge numpy scipy matplotlib pandas -y
conda install -c conda-forge openmm -y
conda install -c conda-forge mdtraj mdanalysis -y
conda install -c conda-forge deeptime -y
pip install torch botorch ax-platform xgboost
```

## Cell 0.2: 验证环境

```python
# 将此 cell 粘贴到 Jupyter Notebook 中运行
import sys
print(f"Python: {sys.version}")

# 核心科学计算
import numpy as np; print(f"numpy: {np.__version__}")
import scipy; print(f"scipy: {scipy.__version__}")
import pandas as pd; print(f"pandas: {pd.__version__}")

# MD 引擎
try:
    import openmm; print(f"openmm: {openmm.__version__}")
    # 检查 GPU
    print(f"  CUDA available: {openmm.Platform.getPlatformByName('CUDA')}")
except Exception as e:
    print(f"openmm: ERROR — {e}")

# 轨迹分析
try:
    import mdtraj; print(f"mdtraj: {mdtraj.__version__}")
except Exception as e:
    print(f"mdtraj: {e} (will install later if needed)")

# ML
try:
    import torch; print(f"torch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"torch: {e}")

# 绘图
import matplotlib; print(f"matplotlib: {matplotlib.__version__}")
```

## Cell 0.3: 确认工作目录

```python
import os
from pathlib import Path

# 切换到项目根目录
project_root = Path(os.getcwd())
print(f"Current directory: {project_root}")

# 确认目录结构存在
required_dirs = [
    "modules/A_triScale/aa_md",
    "modules/A_triScale/cg_bridge",
    "modules/A_triScale/continuum",
    "modules/B_pathExploration",
    "modules/C_rlOptimization",
    "modules/D_boPrediction",
    "data",
    "outputs/figures",
    "scripts",
]
for d in required_dirs:
    path = project_root / d
    exists = path.exists()
    print(f"  {'✓' if exists else '✗'} {d}")

# 确保 data 目录存在
(project_root / "data").mkdir(parents=True, exist_ok=True)
(project_root / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
```

---

## ✓ 检查点 0

```
[ ] conda activate crossScale 成功
[ ] import openmm 无报错
[ ] torch.cuda.is_available() == True
[ ] 所有 required_dirs 有 ✓
```

全部 ✓ 后进入下一步。
