# Step 2: AA 轨迹特征提取

**目标**: 从 AA MD 轨迹中提取残基接触概率、氢键、距离分布，为 ML 桥提供训练数据。

**预计时间**: 5-10 分钟
**GPU**: 否（CPU）

---

## Cell 2.1: 提取特征

```python
import sys
sys.path.insert(0, '.')

from modules.A_triScale.aa_md.extract_features import extract_all

features_df = extract_all()
```

输出应显示：
- 加载了多少条轨迹
- 残基对数量
- 接触概率范围
- 输出文件路径

## Cell 2.2: 检查特征质量

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/aa_features.csv")
print(f"特征表: {len(df)} 行 × {len(df.columns)} 列")
print(f"列名: {list(df.columns)}")
print()

# 接触概率分布
print("接触概率分布:")
print(f"  Mean: {df.contact_prob.mean():.3f}")
print(f"  Median: {df.contact_prob.median():.3f}")
print(f"  >0.1 的比例: {(df.contact_prob > 0.1).mean():.1%}")
print()

# 链内 vs 链间
intra = df[df.same_chain == 1]
inter = df[df.same_chain == 0]
print(f"链内残基对: {len(intra)} (平均接触={intra.contact_prob.mean():.3f})")
print(f"链间残基对: {len(inter)} (平均接触={inter.contact_prob.mean():.3f})")

# 可视化：接触概率矩阵
import matplotlib.pyplot as plt

n_res = max(df.residue_i.max(), df.residue_j.max()) + 1
contact_mat = np.zeros((n_res, n_res))
for _, row in df.iterrows():
    i, j = int(row.residue_i), int(row.residue_j)
    contact_mat[i, j] = row.contact_prob
    contact_mat[j, i] = row.contact_prob

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(contact_mat, cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xlabel("Residue j")
ax.set_ylabel("Residue i")
ax.set_title("AA Residue Contact Probability Matrix")
plt.colorbar(im, ax=ax, label="Contact probability")
plt.tight_layout()
plt.show()
```

期望看到：
- 链内相邻残基有较高接触概率（对角线附近亮）
- 链间残基接触概率较低（非对角块暗）
- 如果链间也有明显接触，说明肽之间有相互作用，CG层面会更有趣

---

## ✓ 检查点 2

```
[ ] data/aa_features.csv 存在且 >100 行
[ ] contact_prob 范围合理（0–1之间）
[ ] 接触矩阵热图出现（非全黑/全白）
```

全部 ✓ 后进入下一步（ML 桥）。
