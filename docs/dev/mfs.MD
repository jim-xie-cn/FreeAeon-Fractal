# 多重分形谱（MFS）使用文档

本文档描述 `FA2DMFS.py`（CPU 版）和 `FA2DMFSGPU.py`（GPU 版）的使用方法。两者数学上完全等价，在 fp64 下输出差异 ≤ 1.2e-14（机器精度）。

---

## 目录

- [一、数学背景](#一数学背景)
- [二、模块结构](#二模块结构)
- [三、CFA2DMFS / CFA2DMFSGPU 类](#三cfa2dmfs--cfa2dmfsgpu-类)
- [四、方法详解](#四方法详解)
  - [4.1 get_mass_table](#41-get_mass_table)
  - [4.2 fit_tau_and_D1](#42-fit_tau_and_d1)
  - [4.3 alpha_falpha_from_tau](#43-alpha_falpha_from_tau)
  - [4.4 get_mfs](#44-get_mfs-完整流水线)
  - [4.5 get_batch_mfs（静态）](#45-get_batch_mfs静态--多图批处理)
  - [4.6 compute_alpha_map](#46-compute_alpha_map--逐像素奇异度)
  - [4.7 compute_alpha_map_batch（静态）](#47-compute_alpha_map_batch静态)
  - [4.8 plot](#48-plot)
- [五、调用示例](#五调用示例)
- [六、注意事项](#六注意事项)

---

## 一、数学背景

将归一化后的图像视为概率测度 μ。用边长 ε 的盒子覆盖图像，第 i 个盒子的概率为 μᵢ(ε)。多重分形谱定义如下：

| 量 | 定义 | 物理意义 |
|---|---|---|
| 配分函数 Z(q, ε) | $\sum_i \mu_i(\varepsilon)^q$ | q 阶矩 |
| 质量指数 τ(q) | $\lim_{\varepsilon \to 0} \log Z(q,\varepsilon) / \log \varepsilon$ | log-log 斜率 |
| 广义维度 D(q) | $\tau(q) / (q - 1)$ | D₀ 容量维, D₁ 信息维, D₂ 关联维 |
| 多重分形谱 α(q) | $d\tau / dq$ | 奇异度 |
| f(α) | $q\alpha - \tau$ | 奇异度分布的分形维数 |
| 局部奇异度 α(x, y) | $\mu_{B_\varepsilon(x)}(\varepsilon) \sim \varepsilon^{\alpha(x)}$ | 像素级局部 scaling |

特殊处理：
- **q = 0**：D₀ = 容量维度（box-counting），用 N(ε) = 非空盒计数代替 Z
- **q = 1**：D₁ = 信息维度，用 S(ε) = −Σᵢ μᵢ log μᵢ 代替 log Z

---

## 二、模块结构

| 模块 | 类名 | 默认设备 | 默认 dtype |
|---|---|---|---|
| `FA2DMFS.py` | `CFA2DMFS` | CPU（NumPy） | float64 |
| `FA2DMFSGPU.py` | `CFA2DMFSGPU` | 自动（CUDA→CPU） | torch.float64 |

GPU 类除多接受 `device`、`dtype`、`img_chunk`、`q_chunk` 外，与 CPU 类**完全相同的接口、相同的方法名**，可以无缝替换。

---

## 三、CFA2DMFS / CFA2DMFSGPU 类

### 构造函数

**CPU**：

```python
CFA2DMFS(
    image,                          # np.ndarray (H, W)
    corp_type=-1,                   # int
    q_list=np.linspace(-5, 5, 51),  # np.ndarray
    with_progress=True,             # bool
    bg_threshold=0.01,              # float
    bg_reverse=False,               # bool
    bg_otsu=False,                  # bool
    mu_floor=1e-12,                 # float
)
```

**GPU**（多 2 个参数）：

```python
CFA2DMFSGPU(
    image, corp_type=-1, q_list=...,
    with_progress=True, bg_threshold=0.01, bg_reverse=False,
    bg_otsu=False, mu_floor=1e-12,
    device=None,                    # str / torch.device / None
    dtype=torch.float64,            # torch.dtype
)
```

### 参数详解

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `np.ndarray (H, W)` | — | 必填。2D 灰度图，支持 uint8/uint16/int32/float32/float64/bool |
| `corp_type` | `int` | `-1` | ROI 裁剪策略；`-1`=裁到整数倍 box，`0`=要求整除否则报错，`1`=零填充 |
| `q_list` | `np.ndarray` 或 list | `linspace(-5, 5, 51)` | q 取值列表；不论是否含 q=0/q=1，输出 df_fit 总会含这两行 |
| `with_progress` | `bool` | `True` | 是否显示 tqdm 进度条 |
| `bg_threshold` | `float` | `0.01` | 像素归一化到 [0,1] 后小于该值视为背景置零；为 0 表示不阈值化 |
| `bg_reverse` | `bool` | `False` | 反转前景/背景（像素 ← 1 - 像素） |
| `bg_otsu` | `bool` | `False` | 用 Otsu 自动求二值化阈值（覆盖 `bg_threshold`） |
| `mu_floor` | `float` | `1e-12` | μ 的下界（保留参数，当前实现用 logsumexp 不需要 flooring） |
| `device` | `str/torch.device/None` | `None` | GPU 专用；`None` 自动选 cuda（不可用则 cpu）；可写 `'cuda'`、`'cuda:0'`、`'cpu'` |
| `dtype` | `torch.dtype` | `torch.float64` | GPU 专用；CPU/GPU 完全一致请用 `torch.float64`，性能优先用 `torch.float32` |

### 实例属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `self.m_image` | `np.ndarray` | 预处理（归一化+背景阈值）后的图像，dtype=float64 |
| `self.m_q_list` | `np.ndarray` | 用户传入的 q_list 副本 |
| `self.device` | `torch.device` | 仅 GPU 类 |
| `self.dtype` | `torch.dtype` | 仅 GPU 类 |

---

## 四、方法详解

### 4.1 `get_mass_table`

#### 功能
对图像在多个 box 尺寸 s 下计算质量表（mass table），含每个 (s, q) 处的 N、S 或 logMq 值。这是 MFS 的最底层数值表。

#### 签名

```python
get_mass_table(self, max_size=None, max_scales=80,
               min_box=2, roi_mode='center')
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_size` | `int / None` | `None` | 最大 box 尺寸；`None` 取 `min(H, W)` |
| `max_scales` | `int` | `80` | scales 数量上限（整数化去重后通常远少于此） |
| `min_box` | `int` | `2` | 最小 box 尺寸 |
| `roi_mode` | `'center'` 或 `'topleft'` | `'center'` | ROI 裁剪方式 |

#### 返回值

`pandas.DataFrame`，列：

| 列 | 类型 | 说明 |
|---|---|---|
| `scale` | int | box 尺寸 s（像素） |
| `eps` | float | 归一化尺度 ε = s/L |
| `q` | float | q 值 |
| `value` | float | 内容因 kind 而异：N=非空盒数、S=−Σμlog μ、logMq=log Σμᵍ |
| `kind` | str | `'N'`（q=0）、`'S'`（q=1）、`'logMq'`（其它 q） |

> **关键**：N 行（kind='N'）和 S 行（kind='S'）**总是存在**，与用户传入的 q_list 无关。这是 D₀、D₁ 标准多重分形量。

---

### 4.2 `fit_tau_and_D1`

#### 功能
对 mass_table 做 log-log 线性回归，得到 τ(q)、Dq、D1。

#### 签名

```python
fit_tau_and_D1(self, df_mass, min_points=6,
               require_common_scales=True,
               use_middle_scales=True,
               fit_scale_frac=(0.2, 0.8),
               if_auto_line_fit=False,
               auto_fit_min_len_ratio=0.5,
               cap_d0_at_2=False)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `df_mass` | `pd.DataFrame` | — | `get_mass_table()` 的返回值 |
| `min_points` | `int` | `6` | 拟合至少需要的 (scale, value) 点数；不足返回 NaN |
| `require_common_scales` | `bool` | `True` | 仅用所有 q 都覆盖到的 scales |
| `use_middle_scales` | `bool` | `True` | 仅用中间段 scales 拟合（避免两端饱和） |
| `fit_scale_frac` | `tuple(float, float)` | `(0.2, 0.8)` | use_middle_scales 时的中间段范围 |
| `if_auto_line_fit` | `bool` | `False` | 自动选最长线性段（最大化 R²） |
| `auto_fit_min_len_ratio` | `float` | `0.5` | 自动线段最少占总点数的比例 |
| `cap_d0_at_2` | `bool` | `False` | 若 D₀ > 2，逐步剔除最小 scale 直到 D₀ ≤ 2 |

#### 返回值

`pandas.DataFrame`，每个 q 一行，列：

| 列 | 类型 | 说明 |
|---|---|---|
| `q` | float | q 值（含强制添加的 q=0 和 q=1） |
| `tau` | float | τ(q) = log-log 斜率取负（q≠0,1）；q=0: τ=−D₀；q=1: τ=0 |
| `Dq` | float | 广义维度 D(q) |
| `D1` | float | 信息维度（仅 q=1 行非 NaN） |
| `intercept` | float | 拟合截距 |
| `r_value` | float | Pearson 相关系数 |
| `p_value` | float | p-值 |
| `std_err` | float | 斜率标准误 |
| `n_points` | int | 实际用于拟合的点数 |

---

### 4.3 `alpha_falpha_from_tau`

#### 功能
从 τ(q) 用样条求导得到 α(q)，再 Legendre 变换得到 f(α) = qα − τ。

#### 签名

```python
alpha_falpha_from_tau(self, df_fit, spline_k=3,
                      exclude_q1=True, spline_s=0)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `df_fit` | `pd.DataFrame` | — | `fit_tau_and_D1()` 的返回值 |
| `spline_k` | `int` | `3` | UnivariateSpline 阶数 |
| `exclude_q1` | `bool` | `True` | 排除 q=1（τ(1)=0 是奇点，避免数值问题） |
| `spline_s` | `float` | `0` | 样条平滑因子；0 表示完全插值 |

#### 返回值

`pandas.DataFrame`，列：

| 列 | 类型 | 说明 |
|---|---|---|
| `q`, `tau`, `Dq`, `D1`, `n_points` | — | 来自 df_fit |
| `alpha` | float | α(q) = dτ/dq |
| `f_alpha` | float | f(α) = qα − τ |

---

### 4.4 `get_mfs`（完整流水线）

#### 功能
一站式：计算 mass_table → 拟合 τ/Dq → 计算 (α, f(α))。**最常用入口**。

#### 签名

```python
get_mfs(self, max_size=None, max_scales=80, min_points=6, min_box=2,
        use_middle_scales=False, fit_scale_frac=(0.2, 0.8),
        if_auto_line_fit=False, auto_fit_min_len_ratio=0.5,
        spline_s=0, cap_d0_at_2=False)
```

参数语义同上述各方法（注意 `use_middle_scales` 默认 `False`）。

#### 返回值

3 元组 `(df_mass, df_fit, df_spec)`：

| 元素 | 类型 | 说明 |
|---|---|---|
| `df_mass` | `pd.DataFrame` | 同 `get_mass_table` |
| `df_fit` | `pd.DataFrame` | 同 `fit_tau_and_D1` |
| `df_spec` | `pd.DataFrame` | 同 `alpha_falpha_from_tau` |

---

### 4.5 `get_batch_mfs`（静态） — 多图批处理

#### 功能
对一组图像批量做 MFS。CPU 版串行循环；GPU 版**真正向量化**（一次性把整 batch 装上 GPU，按 scale 与 q 分块），通常 5×~50× 加速。

#### CPU 签名

```python
@staticmethod
CFA2DMFS.get_batch_mfs(
    img_list,                       # list[np.ndarray]
    q_list=np.linspace(-5, 5, 51),
    corp_type=-1,
    max_size=None,
    max_scales=80,
    min_box=2,
    min_points=6,
    use_middle_scales=False,
    fit_scale_frac=(0.2, 0.8),
    if_auto_line_fit=False,
    auto_fit_min_len_ratio=0.5,
    spline_s=0,
    cap_d0_at_2=False,
    bg_threshold=0.01,
    bg_reverse=False,
    bg_otsu=False,
    mu_floor=1e-12,
    with_progress=True,
)
```

#### GPU 签名（多 4 个参数）

```python
@staticmethod
CFA2DMFSGPU.get_batch_mfs(
    img_list, ..., # 上述全部参数
    device=None,                    # 'cuda' / 'cpu' / None
    dtype=torch.float32,            # fp64 时与 CPU 完全一致
    with_progress=True,
    img_chunk=None,                 # int / None；显存控制
    q_chunk=64,                     # int；q 维度分块
)
```

#### 参数差异说明

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `img_list` | `list[np.ndarray]` | — | 一组图像；GPU 版要求 batch 内**所有图同尺寸**，CPU 版可异形 |
| `device` | `str/torch.device/None` | `None` | 仅 GPU |
| `dtype` | `torch.dtype` | `torch.float32` | 仅 GPU；fp64 与 CPU 完全一致 |
| `img_chunk` | `int/None` | `None` | 仅 GPU；一次最多多少图同时驻留 GPU；OOM 时调小 |
| `q_chunk` | `int` | `64` | 仅 GPU；一次最多处理多少 q；OOM 时调小 |

#### 返回值

`list[tuple]`，每个元素是 `(df_mass, df_fit, df_spec)`，与 `get_mfs` 单图返回相同。**顺序与 `img_list` 顺序对应**。

---

### 4.6 `compute_alpha_map` — 逐像素奇异度

#### 功能
对每个像素计算局部奇异度 α(x, y)：用一组 scale 下该像素所在盒子的 μ 做 log-log 线性回归。

实现包含两项优化：
1. **Streaming OLS**：边算 μ 边累加 5 个 OLS 统计量，无需保留完整 (K, L, L) 张量
2. **嵌套网格优化**：当所有 scale 都是 s_min 整数倍时，α 在每个 s_min×s_min patch 内为常数；在粗网格上做 OLS 再上采样，内存与计算量缩小 s_min² 倍

#### 签名

CPU 与 GPU 完全相同：

```python
compute_alpha_map(self, scales=None, roi_mode='center',
                  empty_policy='nan')
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `scales` | `iterable[int] / None` | `None` | scale 列表；`None` 自动取 `[2, 4, 8, ..., L/4]` |
| `roi_mode` | `'center'` 或 `'topleft'` | `'center'` | ROI 裁剪方式 |
| `empty_policy` | `'nan'` 或 `'fill'` | `'nan'` | μ=0 时如何处理：`'nan'`= 跳过该 scale 的该像素；`'fill'`= 用该 scale 最小正 μ 替换 |

#### 返回值

2 元组 `(alpha_map, info)`：

| 元素 | 类型 | 说明 |
|---|---|---|
| `alpha_map` | `np.ndarray (L, L) float64` | 每个像素的 α；NaN 表示有效 scale < 2 |
| `info` | `dict` | 含 `'L'`（ROI 边长）、`'scales'`、`'log_eps'` |

---

### 4.7 `compute_alpha_map_batch`（静态）

#### 功能
批量版的 `compute_alpha_map`。

#### CPU 签名

```python
@staticmethod
CFA2DMFS.compute_alpha_map_batch(
    images,                         # list[np.ndarray]
    scales=None,
    roi_mode='center',
    bg_threshold=0.01,
    bg_reverse=False,
    bg_otsu=False,
    empty_policy='nan',
    with_progress=True,
    mu_floor=1e-12,
)
```

#### GPU 签名（多 3 个参数）

```python
@staticmethod
CFA2DMFSGPU.compute_alpha_map_batch(
    images, scales=None, roi_mode='center',
    bg_threshold=0.01, bg_reverse=False, bg_otsu=False,
    empty_policy='nan', with_progress=True,
    device=None,
    dtype=torch.float32,            # fp64 时与 CPU 完全一致
    img_chunk=None,
)
```

#### 返回值

2 元组 `(alpha_maps, info)`：

| 元素 | 类型 | 说明 |
|---|---|---|
| `alpha_maps` | `list[np.ndarray]` | 每个元素 `(L, L) float64`，与 `images` 顺序对应 |
| `info` | `dict` | 含公共 `'L'`、`'scales'`、`'log_eps'` |

---

### 4.8 `plot`

#### 功能
绘制 6 联子图（仅可视化，不返回数据）：
- 左上：log M(q, ε) 热力图（scale × q）
- 中上：f(α) vs α 曲线
- 右上：τ(q) 曲线
- 左下：D(q) 曲线
- 中下：每个 q 的 log-log 拟合点（含 D₀、D₁ 标记）
- 右下：拟合 R² vs q

#### 签名

```python
plot(self, df_mass, df_fit, df_spec)
```

参数为 `get_mfs()` 的三个返回值。返回 `None`（直接调用 plt.show 风格，函数末尾打印 fig）。

---

## 五、调用示例

### 示例 1：单图 MFS 完整流程（CPU）

```python
import cv2
import numpy as np
from FA2DMFS import CFA2DMFS

# 加载灰度图
img = cv2.imread("texture.png", cv2.IMREAD_GRAYSCALE)

# 构造 MFS 对象
mfs = CFA2DMFS(
    image=img,
    q_list=np.linspace(-5, 5, 51),
    bg_otsu=True,                  # 用 Otsu 自动二值化阈值
    with_progress=True,
)

# 一站式
df_mass, df_fit, df_spec = mfs.get_mfs(max_scales=40, min_points=5)

# 取关键维度
D0 = df_fit[df_fit['q'] == 0.0]['Dq'].iloc[0]
D1 = df_fit[df_fit['q'] == 1.0]['Dq'].iloc[0]
D2 = df_fit[df_fit['q'] == 2.0]['Dq'].iloc[0]

print(f"D₀ (容量) = {D0:.4f}")
print(f"D₁ (信息) = {D1:.4f}")
print(f"D₂ (关联) = {D2:.4f}")
print(f"奇异度宽度 Δα = {df_spec['alpha'].max() - df_spec['alpha'].min():.4f}")

# 可视化
mfs.plot(df_mass, df_fit, df_spec)
```

### 示例 2：单图 + 局部奇异度图（GPU，fp64）

```python
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from FA2DMFSGPU import CFA2DMFSGPU

img = cv2.imread("texture.png", cv2.IMREAD_GRAYSCALE)

# 用 fp64 与 CPU 完全一致
mfs = CFA2DMFSGPU(
    image=img,
    q_list=np.linspace(-5, 5, 51),
    bg_otsu=True,
    device='cuda',                  # 自动 fallback 到 cpu
    dtype=torch.float64,
)

# 多重分形谱
_, df_fit, df_spec = mfs.get_mfs(max_scales=40, min_points=5)

# 局部奇异度图
alpha_map, info = mfs.compute_alpha_map(
    scales=[2, 4, 8, 16, 32, 64],
    empty_policy='nan',
)

# 可视化 alpha 图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(img, cmap='gray')
ax1.set_title("Original")
im = ax2.imshow(alpha_map, cmap='inferno')
ax2.set_title(r"$\alpha(x, y)$")
plt.colorbar(im, ax=ax2)
plt.show()
```

### 示例 3：批处理 100 张图（GPU 加速）

```python
import cv2
import glob
import numpy as np
import torch
from FA2DMFSGPU import CFA2DMFSGPU

# 加载图像（必须同尺寸；不同尺寸用 CPU 版）
paths = sorted(glob.glob("dataset/*.png"))
imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

# 批处理 — 一次 GPU 计算 100 张
results = CFA2DMFSGPU.get_batch_mfs(
    img_list=imgs,
    q_list=np.linspace(-3, 3, 31),
    bg_threshold=0.01,
    device='cuda',
    dtype=torch.float64,
    img_chunk=16,                   # 显存控制
    q_chunk=32,
    with_progress=True,
)

# 收集每张图的 D₀、D₁、Δα
import pandas as pd
summary = []
for i, (df_mass, df_fit, df_spec) in enumerate(results):
    D0 = df_fit[df_fit['q'] == 0.0]['Dq'].iloc[0]
    D1 = df_fit[df_fit['q'] == 1.0]['Dq'].iloc[0]
    delta_alpha = (df_spec['alpha'].max() - df_spec['alpha'].min()
                   if not df_spec.empty else np.nan)
    summary.append({
        'path': paths[i], 'D0': D0, 'D1': D1, 'delta_alpha': delta_alpha
    })
df = pd.DataFrame(summary)
df.to_csv("mfs_summary.csv", index=False)
```

### 示例 4：批量计算局部 α 图

```python
import torch
import numpy as np
from FA2DMFSGPU import CFA2DMFSGPU

imgs = [...]  # list of np.ndarray, same shape

alpha_maps, info = CFA2DMFSGPU.compute_alpha_map_batch(
    images=imgs,
    scales=[2, 4, 8, 16, 32],
    bg_threshold=0.01,
    empty_policy='nan',
    device='cuda',
    dtype=torch.float64,
    img_chunk=8,
)

# 每张图的 α 平均值
mean_alphas = [np.nanmean(am) for am in alpha_maps]
```

### 示例 5：自定义 q 列表，仅看 D₀

```python
from FA2DMFS import CFA2DMFS
import numpy as np

# 即使只传一个 q，df_fit 仍会含强制的 q=0 和 q=1
mfs = CFA2DMFS(image=img, q_list=np.array([2.0]), bg_threshold=0.01)
_, df_fit, _ = mfs.get_mfs(max_scales=20)
print(df_fit[['q', 'Dq']])
# 输出 3 行：q=0.0, q=1.0, q=2.0
```

---

## 六、注意事项

### 6.1 关于 D₀ 与 D₁

**`df_fit` 总是包含 q=0 和 q=1 两行**，与用户的 `q_list` 是否含这两值无关。理由：
- D₀（容量维度）= box-counting 维度，是基础量
- D₁（信息维度）= 多重分形分析的标准量
- q=1 处 τ(1)=0 是 Legendre 变换的奇点，因此 `df_spec` 默认排除 q=1 行

### 6.2 CPU vs GPU 数值一致性

- **GPU MFS 默认 dtype = `torch.float64`**，与 CPU 完全一致（fp64 机器精度）
- **但 GPU 的 batch 方法 `get_batch_mfs` 与 `compute_alpha_map_batch` 默认 dtype = `torch.float32`**（性能优先）。需要 byte-exact 与 CPU 一致时，**显式传 `dtype=torch.float64`**

### 6.3 ROI 裁剪行为

所有方法把图裁成正方形 L×L，L = min(H, W)。`roi_mode='center'`（默认）取中心，`'topleft'` 取左上。**非正方形图会丢失部分像素**，如果需要全图分析请先 padding。

### 6.4 退化输入

- **均匀图**（所有像素相等）→ 经预处理后变全零图，mass_table 返回空 DataFrame，结果为空
- **全零图** → 同上，无信息可分析
- **单亮像素** → 数学上正确：α 在所有"被某 scale 的盒子覆盖到亮像素"的位置都是 0（不是只有亮像素那一点）

### 6.5 fit_range / fit_scale_frac 的选择

`use_middle_scales=True` + `fit_scale_frac=(0.2, 0.8)` 取中间 60% 的 scale 拟合。box-counting 的 log-log 曲线两端饱和，中间是真正幂律段。如果直接用全部 scales，FD 会有偏差。

### 6.6 GPU 显存控制

OOM 时调小：
- `img_chunk`（一次最多多少图）
- `q_chunk`（仅 `get_batch_mfs`，一次最多多少 q）

### 6.7 corp_type 含义

| 值 | 行为 |
|---|---|
| `-1` | 裁到 (H//s)·s 整数倍（最常用） |
| `0` | 严格要求 H, W 被 s 整除，否则抛 `ValueError` |
| `1` | 零填充到 ⌈H/s⌉·s |

### 6.8 数学正确性验证

实测：Sierpinski carpet（base-3 scales）→ D₀ = D₁ = D₂ = log(8)/log(3) = 1.892789（与理论值精确一致到 6 位小数）。
