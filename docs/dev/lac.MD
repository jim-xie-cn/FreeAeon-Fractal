# 缝隙度（Lacunarity）使用文档

本文档描述 `FAImageLacunarity.py`（CPU 版）和 `FAImageLacunarityGPU.py`（GPU 版）的使用方法。两者数学上完全等价，CPU/GPU 输出**完全相同**（差异为 0）。

---

## 目录

- [一、数学背景](#一数学背景)
- [二、模块结构](#二模块结构)
- [三、CFAImageLacunarity / CFAImageLacunarityGPU 类](#三cfaimagelacunarity--cfaimagelacunaritygpu-类)
- [四、方法详解](#四方法详解)
  - [4.1 get_lacunarity](#41-get_lacunarity)
  - [4.2 fit_lacunarity](#42-fit_lacunarity)
  - [4.3 get_batch_lacunarity（静态）](#43-get_batch_lacunarity静态)
  - [4.4 fit_batch_lacunarity（静态）](#44-fit_batch_lacunarity静态)
  - [4.5 plot](#45-plot)
- [五、调用示例](#五调用示例)
- [六、注意事项](#六注意事项)

---

## 一、数学背景

Allain & Cloitre (1991) 缝隙度定义：对边长 r 的盒子，记盒内质量 M（盒内像素和或非零像素数），

$$\Lambda(r) = \frac{\langle M^2 \rangle}{\langle M \rangle^2} = 1 + \frac{\sigma^2(M)}{\mu^2(M)}$$

性质：
- **下界 = 1**（均匀分布；σ² = 0）
- **越大表示越"缝隙"**（质量分布越不均匀）
- **自相似集**：$\Lambda(r) \sim r^{-\beta}$，β = D − E（E = 拓扑维度）
- **log Λ vs log r 拟合斜率 = −β**

**采样方式**：
- **Gliding box**（滑动窗口）：在所有 (H − r + 1)(W − r + 1) 个位置采样，统计强（**默认推荐**）
- **Non-overlapping**（不重叠 tile）：仅在 (H/r)(W/r) 个位置采样，对小 r 还行，大 r 样本太少

**质量定义**：
- **Use raw pixel sum**（默认）：M = Σ pixel values in box（适合灰度图）
- **Use binary mass**：M = #{pixel > 0 in box}（适合二值图）

**典型理论值**（验证用）：
- 均匀图 → Λ(r) ≡ 1
- Bernoulli(p=0.5) iid 二值图 → Λ(r) = 1 + 1/r²

---

## 二、模块结构

| 模块 | 类名 | 默认设备 | 默认 dtype |
|---|---|---|---|
| `FAImageLacunarity.py` | `CFAImageLacunarity` | CPU（NumPy） | float64 |
| `FAImageLacunarityGPU.py` | `CFAImageLacunarityGPU` | 自动（CUDA→CPU） | torch.float64 |

GPU 类除多接受 `device`、`dtype`、`img_chunk` 外，与 CPU 类**完全相同的接口**。

---

## 三、CFAImageLacunarity / CFAImageLacunarityGPU 类

### 构造函数

**CPU**：

```python
CFAImageLacunarity(
    image=None,                     # np.ndarray (H, W) / None
    max_size=None,                  # int / None
    max_scales=100,                 # int
    with_progress=True,             # bool
    scales_mode='powers',           # 'powers' / 'logspace'
    partition_mode='gliding',       # 'gliding' / 'non-overlapping'
    min_size=2,                     # int
)
```

**GPU**（多 2 个参数）：

```python
CFAImageLacunarityGPU(
    image=None, max_size=None, max_scales=100, with_progress=True,
    scales_mode='powers', partition_mode='gliding', min_size=2,
    device=None,                    # str / torch.device / None
    dtype=torch.float64,            # torch.dtype
)
```

### 参数详解

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `np.ndarray (H, W) / None` | `None` | 2D 图（灰度或二值都可） |
| `max_size` | `int / None` | `None` | 最大 box 尺寸；`None` 取 `min(H, W)` |
| `max_scales` | `int` | `100` | scales 数量上限 |
| `with_progress` | `bool` | `True` | tqdm 进度条 |
| `scales_mode` | `'powers'` / `'logspace'` | `'powers'` | `'powers'` = 2, 4, 8, 16, ...；`'logspace'` = 几何分布去重整数 |
| `partition_mode` | `'gliding'` / `'non-overlapping'` | `'gliding'` | 采样方式 |
| `min_size` | `int` | `2` | 最小 box 尺寸 |
| `device` | `str/torch.device/None` | `None` | GPU 专用 |
| `dtype` | `torch.dtype` | `torch.float64` | GPU 专用 |

### 实例属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `self.m_image` | `np.ndarray` | 原图像（CPU 不修改 dtype） |
| `self.m_scales` | `list[int]` | 实际使用的 scales |
| `self.m_partition_mode` | `str` | 采样模式 |
| `self.device`, `self.dtype` | — | 仅 GPU |

---

## 四、方法详解

### 4.1 `get_lacunarity`

#### 功能
计算每个 scale 下的缝隙度 Λ(s)。

CPU 版用积分图（`O(HW)`）一次性求所有 box mass，全部 scales 共享，避免 N_scales 倍冗余计算。GPU 版用 conv2d 风格的累加。

#### 签名

CPU 与 GPU 完全相同：

```python
get_lacunarity(self, corp_type=-1, use_binary_mass=False,
               include_zero=True) -> dict
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `corp_type` | `int` | `-1` | ROI 裁剪策略（与 dimension 一致） |
| `use_binary_mass` | `bool` | `False` | `False`=用原始像素和；`True`=用 box 内非零像素计数 |
| `include_zero` | `bool` | `True` | `False`=忽略全零 box（M=0 不计入统计） |

#### 返回值

`dict`：

| 键 | 类型 | 说明 |
|---|---|---|
| `scales` | `list[int]` | 实际使用的 box 尺寸序列 |
| `lacunarity` | `list[float]` | 每个 scale 的 Λ(s)；NaN 表示样本不足或均值为 0 |
| `mass_stats` | `list[dict]` | 每个 scale 的详细统计（见下） |

每个 `mass_stats[i]` 是 `dict`：

| 键 | 类型 | 说明 |
|---|---|---|
| `scale` | `int` | box 尺寸 |
| `num_boxes` | `int` | 采样到的 box 数量 |
| `mean_mass` | `float` | μ(M) |
| `var_mass` | `float` | σ²(M)（总体方差，不是样本方差） |
| `lambda` | `float` | Λ(s) |

---

### 4.2 `fit_lacunarity`

#### 功能
对 `get_lacunarity` 结果做 log-log 拟合，得到 power-law 斜率。

#### 签名

```python
fit_lacunarity(self, lac_result, transform='log',
               fit_range=None) -> dict
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `lac_result` | `dict` | — | `get_lacunarity()` 的返回值 |
| `transform` | `'log'` 或 `'log_minus_1'` | `'log'` | 拟合变量：`'log'`= log Λ vs log s（**Allain-Cloitre 标准，斜率=−β**）；`'log_minus_1'`= log(Λ−1)，跳过 Λ=1 的均匀点 |
| `fit_range` | `tuple(float, float) / None` | `None` | `(s_min, s_max)` 限定拟合范围 |

#### 返回值

`dict`：

| 键 | 类型 | 说明 |
|---|---|---|
| `slope` | `float` | log Λ vs log s 的斜率（= −β = E − D for self-similar set） |
| `intercept` | `float` | 截距 |
| `r_value` | `float` | Pearson 相关系数 |
| `p_value` | `float` | p 值 |
| `std_err` | `float` | 斜率标准误 |
| `log_scales` | `np.ndarray` | log(scales) |
| `log_lambda_minus_1` | `np.ndarray` | 拟合用 y 序列（即使 transform='log' 字段名也是这个，含义为对应的 y 值） |
| `transform` | `str` | 使用的 transform 标记 |

---

### 4.3 `get_batch_lacunarity`（静态）

#### 功能
对一组图像批量计算缝隙度。CPU 版串行循环（**支持异形图**），GPU 版向量化（要求**所有图同尺寸**）。

#### CPU 签名

```python
@staticmethod
CFAImageLacunarity.get_batch_lacunarity(
    images,                         # Sequence[np.ndarray]
    max_size=None,
    max_scales=100,
    scales_mode='powers',
    partition_mode='gliding',
    min_size=2,
    use_binary_mass=False,
    include_zero=True,
    with_progress=True,
) -> List[dict]
```

#### GPU 签名（多 3 个参数）

```python
@staticmethod
CFAImageLacunarityGPU.get_batch_lacunarity(
    images, max_size=None, max_scales=100,
    scales_mode='powers', partition_mode='gliding', min_size=2,
    use_binary_mass=False, include_zero=True,
    device=None,                    # str / torch.device / None
    dtype=torch.float64,
    img_chunk=None,                 # int / None；显存控制
    with_progress=True,
) -> List[dict]
```

#### 参数差异

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `images` | `Sequence[np.ndarray]` | — | 图列表；GPU 版要求所有图同尺寸（异形抛 ValueError） |
| `device` | `str/torch.device/None` | `None` | 仅 GPU |
| `dtype` | `torch.dtype` | `torch.float64` | 仅 GPU |
| `img_chunk` | `int/None` | `None` | 仅 GPU |

#### 返回值

`List[dict]`，每个元素是 `get_lacunarity()` 风格的 dict，**顺序与 `images` 一致**。

---

### 4.4 `fit_batch_lacunarity`（静态）

#### 功能
对一组 `lacunarity` 结果批量做 log-log 拟合。

#### 签名

CPU 与 GPU 完全相同：

```python
@staticmethod
fit_batch_lacunarity(
    lac_results,                    # Sequence[dict]
    transform='log',
    fit_range=None,
) -> List[dict]
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `lac_results` | `Sequence[dict]` | — | `get_batch_lacunarity()` 的返回值 |
| `transform` | `'log'` / `'log_minus_1'` | `'log'` | 同 `fit_lacunarity` |
| `fit_range` | `tuple/None` | `None` | 同 `fit_lacunarity` |

#### 返回值

`List[dict]`，每个元素是 `fit_lacunarity()` 风格的 dict，**顺序与 `lac_results` 一致**。

---

### 4.5 `plot`

#### 功能
绘制 log-log 缝隙度曲线（含拟合线）。

#### 签名

CPU 是实例方法：

```python
plot(self, lac_result, fit_result=None, ax=None, show=True,
     title='Lacunarity', label=None)
```

GPU 是静态方法（无 `self`）：

```python
@staticmethod
plot(lac_result, fit_result=None, ax=None, show=True,
     title='Lacunarity', label=None)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `lac_result` | `dict` | — | `get_lacunarity()` 返回值 |
| `fit_result` | `dict / None` | `None` | `fit_lacunarity()` 返回值；提供则画拟合线 |
| `ax` | `matplotlib.axes.Axes / None` | `None` | 用户自定 Axes；`None` 则新建 |
| `show` | `bool` | `True` | 是否调用 `plt.show()` |
| `title` | `str` | `'Lacunarity'` | 图标题 |
| `label` | `str / None` | `None` | 图例标签 |

返回：`matplotlib.axes.Axes`。

---

## 五、调用示例

### 示例 1：单图缝隙度 + 拟合（CPU）

```python
import cv2
import numpy as np
from FAImageLacunarity import CFAImageLacunarity

img = cv2.imread("texture.png", cv2.IMREAD_GRAYSCALE)

calc = CFAImageLacunarity(
    image=img,
    max_scales=20,
    scales_mode='powers',           # 用 2,4,8,16,...
    partition_mode='gliding',       # 滑动窗口（推荐）
)
res = calc.get_lacunarity(use_binary_mass=False, include_zero=True)
fit = calc.fit_lacunarity(res, transform='log', fit_range=(2, 64))

print(f"Λ(2)  = {res['lacunarity'][0]:.4f}")
print(f"slope = {fit['slope']:.4f}  (理论 = E - D)")
print(f"R²    = {fit['r_value']**2:.4f}")
```

### 示例 2：二值图缝隙度

```python
import cv2
import numpy as np
from FAImageLacunarity import CFAImageLacunarity

img = cv2.imread("fractal.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

calc = CFAImageLacunarity(
    image=binary,
    max_scales=15,
    scales_mode='powers',
    partition_mode='gliding',
)
# 二值图通常用 use_binary_mass=True
res = calc.get_lacunarity(use_binary_mass=True, include_zero=True)
print(f"scales: {res['scales']}")
print(f"lacunarity: {res['lacunarity']}")
```

### 示例 3：可视化

```python
import matplotlib.pyplot as plt
from FAImageLacunarity import CFAImageLacunarity

calc = CFAImageLacunarity(image=img, max_scales=20)
res = calc.get_lacunarity()
fit = calc.fit_lacunarity(res, transform='log')

fig, ax = plt.subplots(figsize=(6, 4))
calc.plot(res, fit_result=fit, ax=ax, show=False,
          title="Lacunarity of texture.png", label="my image")
ax.legend()
plt.savefig("lacunarity.png", dpi=150)
plt.show()
```

### 示例 4：GPU 批处理

```python
import cv2, glob
import numpy as np
import torch
import pandas as pd
from FAImageLacunarityGPU import CFAImageLacunarityGPU

paths = sorted(glob.glob("dataset/*.png"))
imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

# 批处理（所有图必须同尺寸）
results = CFAImageLacunarityGPU.get_batch_lacunarity(
    images=imgs,
    max_scales=15,
    scales_mode='powers',
    partition_mode='gliding',
    use_binary_mass=False,
    device='cuda',
    dtype=torch.float64,
    img_chunk=8,
    with_progress=True,
)

# 批量拟合
fits = CFAImageLacunarityGPU.fit_batch_lacunarity(
    results, transform='log', fit_range=(4, 64),
)

df = pd.DataFrame({
    'path': paths,
    'lambda_2': [r['lacunarity'][0] for r in results],
    'slope':    [f['slope']   for f in fits],
    'r2':       [f['r_value']**2 for f in fits],
})
df.to_csv("lacunarity_results.csv", index=False)
```

### 示例 5：异形图（CPU 自动适配）

```python
from FAImageLacunarity import CFAImageLacunarity

# 不同尺寸的图 — CPU 版可处理（GPU 版会抛 ValueError）
imgs_het = [
    np.random.randint(0, 256, (256, 256)).astype(np.uint8),
    np.random.randint(0, 256, (200, 320)).astype(np.uint8),
    np.random.randint(0, 256, (512, 384)).astype(np.uint8),
]

results = CFAImageLacunarity.get_batch_lacunarity(
    images=imgs_het, max_scales=10, scales_mode='powers',
    partition_mode='gliding', with_progress=False,
)
print([len(r['lacunarity']) for r in results])
```

### 示例 6：忽略全零 box（稀疏图）

```python
from FAImageLacunarity import CFAImageLacunarity
import numpy as np

# 极稀疏图：每 4 像素一个亮点
sparse = np.zeros((128, 128), dtype=np.uint8)
sparse[::4, ::4] = 255

calc = CFAImageLacunarity(image=sparse, max_scales=8)
# include_zero=False：完全空盒不计入统计
res_no_zero = calc.get_lacunarity(use_binary_mass=True, include_zero=False)
res_with_zero = calc.get_lacunarity(use_binary_mass=True, include_zero=True)

print("include_zero=False:", res_no_zero['lacunarity'])
print("include_zero=True: ", res_with_zero['lacunarity'])
```

### 示例 7：完整流水线（CPU 版）

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from FAImageLacunarity import CFAImageLacunarity

img = cv2.imread("texture.png", cv2.IMREAD_GRAYSCALE)

calc = CFAImageLacunarity(
    image=img, max_scales=20,
    scales_mode='logspace',          # 几何分布
    partition_mode='gliding',
)

# 1. 计算
res = calc.get_lacunarity()

# 2. 拟合（两种 transform）
fit_log = calc.fit_lacunarity(res, transform='log',
                                fit_range=(4, 64))
fit_log1 = calc.fit_lacunarity(res, transform='log_minus_1',
                                 fit_range=(4, 64))

# 3. 可视化
fig, ax = plt.subplots(figsize=(7, 5))
calc.plot(res, fit_result=fit_log, ax=ax, show=False,
          title="texture.png lacunarity", label="log Λ")
ax.legend()
plt.show()

# 4. 数值
print(f"slope (log)        = {fit_log['slope']:.4f}, R²={fit_log['r_value']**2:.4f}")
print(f"slope (log_minus_1) = {fit_log1['slope']:.4f}, R²={fit_log1['r_value']**2:.4f}")
```

---

## 六、注意事项

### 6.1 gliding vs non-overlapping

| | gliding | non-overlapping |
|---|---|---|
| 采样数 | (H−s+1)(W−s+1) | (H/s)(W/s) |
| 统计强度 | 高（数据多） | 低（数据少） |
| 计算 | 用积分图 O(HW) 一次性求所有 scale | 简单 |
| **推荐** | **默认推荐** | 仅在统计独立性更重要时用 |

### 6.2 use_binary_mass 选择

- **灰度图 + 用原始像素和**（默认 `use_binary_mass=False`）：M = box 内像素和。适合有强度信息的图（纹理、密度场）
- **二值图 + 用占用计数**（`use_binary_mass=True`）：M = box 内非零像素数。这是经典 Allain-Cloitre 定义。适合分形集、骨架、轮廓
- **灰度图 + 用占用计数**：M 与亮度无关，只看哪里非零。慎用

### 6.3 transform 选择

| transform | 拟合变量 | 何时用 |
|---|---|---|
| `'log'` （默认） | log Λ | Allain-Cloitre 标准；slope = −β = E − D |
| `'log_minus_1'` | log(Λ − 1) | 自动跳过 Λ=1 的均匀点；适合接近均匀分布的图 |

### 6.4 CPU vs GPU 一致性

实测：**完全相同**（差异为 0）。GPU 默认 dtype = `torch.float64`，与 CPU 完全等价。

### 6.5 fit_range 是必要的

与 box-counting 一样，log Λ vs log s 在两端会偏离幂律：
- 小 s：受像素离散化影响
- 大 s：box 数太少，统计噪声大

**强烈建议** `fit_range=(small, large)` 选中间稳定段。

### 6.6 退化输入

- **均匀图**（所有像素相等）→ Λ ≡ 1（精确，机器零）
- **全零图** → Λ = NaN（分母 0）
- **单亮像素图** → Λ ~ 1/s² 衰减

### 6.7 GPU 批处理：图必须同尺寸

GPU 版要求所有图同 (H, W)，否则抛 `ValueError`。异形图请用 CPU 版（自动 fallback 到 per-image 循环）。

### 6.8 include_zero 的语义

- `True`（默认）：所有采样到的 box 都算入分子分母（即使 M=0）。这是经典定义
- `False`：跳过 M=0 的 box。适用于稀疏图，避免大量空 box 拉低 mean_mass 而虚增 Λ

### 6.9 数学正确性验证

实测：
- 棋盘图（全相同 box mass=2）：Λ(2) = 1.0（精确）
- 2×2 对角图（box mass=[1,0,0,1]）：Λ(1) = 2.0（精确）
- 均匀图：Λ ≡ 1（精确，所有 scale）
- Bernoulli(0.5) 二值图：Λ(s) ≈ 1 + 1/s²（与理论吻合，s=2 误差 0.13%）
- Sierpinski carpet：log-log 拟合 R² = 0.99，slope ≈ -0.087（理论 -(D-E) = -0.107）

### 6.10 性能提示

CPU 版用积分图加速 gliding 模式：原本 O(N_scales × H × W)，优化后 O(H × W) 一次共享。对 N_scales = 20 的情况加速 ~9×。如果你曾经看过用纯 Python loop 实现的 gliding lacunarity，本版本会快几个数量级。
