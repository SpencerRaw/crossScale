# 2-Page PPT Design Spec

## Page 1 — ML-Accelerated Multi-Scale Pipeline

**Slide title (顶部):** ML Bridge from All-Atom to Continuum — A 3-Scale Parameter Transfer Pipeline

**Layout:** 上中下三带。上带标题，中带主体（左→右三尺度流程），下带关键数字。

### Zone map (16:9 landscape)

```
┌──────────────────────────────────────────────────────────────┐
│  Page 1: ML-Accelerated Multi-Scale Pipeline                 │ ← 24pt, bold
│  ACE-A-F-A-F-A-F-A-K-NME  ·  2 copies  ·  GBSA implicit     │ ← 11pt, gray
├──────────────┬───────────────────┬───────────────────────────┤
│   AA MD      │   ML Bridge       │   CG → Continuum         │
│   (blue)     │   (purple)        │   (green → orange)       │
│              │                   │                           │
│   [contact   │   7D→64→32→16→2D  │   5 concentrations        │
│    matrix]   │   MLP predicts    │   Langevin MD             │
│              │   ε, σ from       │   → D, k_on, k_off        │
│   8 traj.    │   contact feat.   │                           │
│   ×10ns      │   train <1min     │   Smoluchowski ODE        │
│              │                   │   → cluster size(t)       │
│              │   [ε scatter]     │                           │
│              │                   │   [aggregation plot]      │
├──────────────┴───────────────────┴───────────────────────────┤
│  Key: P(contact) → ε prediction r > 0.7  |  D ≈ measured    │ ← 12pt
│  Intra-chain ε > Inter-chain ε  ✓  physics-aware            │
└──────────────────────────────────────────────────────────────┘
```

### Visual elements

**Zone 1 (左, ~30% width) — AA MD**
- Fig 2(a): Contact probability matrix heatmap, cropped square
- 下方 annotation: "200+ residue pairs"
- 色条C_AA (#2471A3)

**Zone 2 (中, ~30% width) — ML Bridge**
- Fig 2(b): ε vs contact_prob scatter with fit line + r value
- 上方 annotation: "MLP(7→64→32→16→2)"
- 色条 C_ML (#8E44AD)
- 关键标注: Pearson r 值用大字号突出

**Zone 3 (右, ~40% width) — CG→Continuum**
- 上方小图: Fig 2(c) ε distribution (intra vs inter chain histogram)
- 下方小图: Fig 4(c) aggregation time vs concentration scatter
- 色条: C_CG (#229954) / C_CONT (#D35400)

**底部信息带**
- 两行: (1) 核心数字 (2) 物理一致性确认
- 深色背景底条，白字

### Implementation notes
- 三个zone之间用 C_ML 色箭头连接 (→)
- Zone 1→Zone 2 箭头标签: "特征提取"
- Zone 2→Zone 3 箭头标签: "参数传递 + 动力学外推"

---

## Page 2 — Validation & Four-Module Framework

**Slide title (顶部):** Cross-Scale Validation & Extensible Framework — 4 Modules in 3 Days

**Layout:** 左半（验证证据） + 右半（四模块体系 + 展望）

### Zone map

```
┌──────────────────────────────────────────────────────────────┐
│  Page 2: Validation & Modular Framework                      │ ← 24pt, bold
│  4 modules · 7 figures · 3 days · single GPU                 │ ← 11pt, gray
├─────────────────────────────┬────────────────────────────────┤
│  Cross-Scale Validation     │  Four-Module Architecture      │
│                             │                                │
│  [ε vs contact scatter]     │       ┌─────────┐             │
│   Pearson r = ___           │       │ Module A│ ← 纵        │
│   Spearman ρ = ___          │       │ 3-scale │   向        │
│                             │       │ pipeline│   主        │
│  Intra-chain ε = ___ mean   │       └────┬────┘   干        │
│  Inter-chain ε = ___ mean   │            │                  │
│  p = ___ (Mann-Whitney)     │    ┌───────┼───────┐         │
│                             │    │       │       │         │
│  [intra/inter boxplot]      │  Module B Module C Module D   │
│                             │  path RL BO 横                │
│                             │  explor. opt. predict. 向     │
│                             │            │   分              │
│                             │            │   枝              │
│                             │    ┌───────┴───────┐         │
│                             │    │   Unified     │         │
│                             │    │ ML×Simulation │         │
│                             │    │  Framework    │         │
│                             │    └───────────────┘         │
├─────────────────────────────┴────────────────────────────────┤
│  Done: pipeline code · data contracts · figure generation    │ ← 12pt
│  Next: GPU run → real data → method paper → platformize     │
└──────────────────────────────────────────────────────────────┘
```

### Visual elements

**Zone 1 (左, ~45% width) — Validation**
- 上图: Fig 4(a) ε-correlation scatter。这是整两页PPT的"证据锚点"
  - r 值用 C_ML 大字号标注
  - 拟合线虚线
- 下图: Fig 4(b) intra/inter boxplot + swarm overlay
  - 两列对比: intra-chain (C_CG绿) vs inter-chain (C_CONT橙)
  - 标注p值

**Zone 2 (右, ~55% width) — Four-Module Map**
- 核心是一个自绘的模块关系图 (见下方)
- 颜色编码:
  - Module A → C_AA 蓝 (主干, 三尺度参数传递)
  - Module B → C_ML 紫 (路径探索, 独立于A)
  - Module C → C_CG 绿 (RL优化, 独立于A)
  - Module D → C_CONT 橙 (BO预测, 依赖A)
- 每个模块旁边标注: 输入 → 输出
  - A: AA traj. → D, k_on, k_off, cluster dist.
  - B: Potential surface → Minima + paths + barriers
  - C: HP sequence space → PPO-optimized sequence + fold
  - D: Structure params → Loading × Stability × Release
- 模块间连线:
  - A→D (实线箭头, 标注 "features + kinetics")
  - A←B/C/D (虚线箭头, 标注 "future feedback loop")

**底部信息带**
- 左: "Built in 3 days" — 代码量, 模块数, 图数
- 右: "Next →" — GPU运行 → 真实数据 → 论文 → 平台化
- 深色背景底条

### 模块关系图绘制说明

用 FancyBboxPatch 或 PPT 原生圆角矩形:

```
宽矩形:  ┌─────────────────────────────────┐
         │  Module A: Tri-Scale Pipeline   │ ← C_AA fill
         │  AA MD → CG ML → Continuum ODE │
         └───────────────┬─────────────────┘
                         │ features + kinetics
         ┌───────────────┼─────────────────┐
         │               │                 │
    ┌────┴────┐    ┌────┴────┐    ┌───────┴──────┐
    │Module B │    │Module C │    │   Module D   │
    │MB paths │    │RL optim │    │BO structure→ │
    │(indep.) │    │(indep.) │    │property (dep)│
    └─────────┘    └─────────┘    └──────────────┘
```

### Implementation notes
- 左上角放一枚小徽章: "Pre-experiment | MVP"
- Zone 1 的 Pearson r 是整个 PPT 最重要的单个数字 — 如果真实数据 < 0.5 则需要重新设计这个位置的叙事

---

## Unified style rules

| Element | Spec |
|---------|------|
| Background | White (#FFFFFF) |
| Title font | DejaVu Sans, 24pt, bold, #2C3E50 |
| Body font | DejaVu Sans, 11pt, #34495E |
| Annotation font | DejaVu Sans, 9pt, #7F8C8D |
| Box corner radius | 8pt |
| Arrow color/thickness | #8E44AD / 2pt |
| Color palette | AA=#2471A3, CG=#229954, Cont=#D35400, ML=#8E44AD, Neutral=#7F8C8D |
| Figure DPI | 200 (embedded PNG) |
| Slide aspect | 16:9 (width:height) |

---

## Figure placement summary

| Figure | Page | Zone | Role |
|--------|------|------|------|
| Fig 2(a) contact matrix | 1 | left | Input: what AA sees |
| Fig 2(b) ε scatter | 1 | center | Core: ML bridge evidence |
| Fig 2(c) ε distribution | 1 | right-top | Output: CG parameter space |
| Fig 4(c) aggregation | 1 | right-bottom | Output: continuum prediction |
| Fig 4(a) validation scatter | 2 | left-top | ***Anchor proof*** |
| Fig 4(b) intra/inter boxplot | 2 | left-bottom | Physical sanity check |
| Module map (自绘) | 2 | right | Architecture narrative |

**不直接出现在PPT中但支撑叙事:** Fig 1 (太宽), Fig 3 (太技术), Fig 5/6/7 (放到backup或口头提及)
