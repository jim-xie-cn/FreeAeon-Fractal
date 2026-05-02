# 分形维度（FD: BC / DBC / SDBC）使用文档

本文档描述 `FAImageDimension.py`（CPU 版）和 `FAImageDimensionGPU.py`（GPU 版）的使用方法。两者数学上完全等价，在 fp64 下输出差异 ≤ 4.4e-16（机器精度）。

---

## 目录

- [一、数学背景](#一数学背景)
- [二、模块结构](#二模块结构)
- [三、CFAImageDimension / CFAImageDimensionGPU 类](#三cfaimagedimension--cfaimagedimensiongpu-类)
- [四、方法详解](#四方法详解)
  - [4.1 get_bc_fd](#41-get_bc_fd--二值-box-counting)
  - [4.2 get_dbc_fd](#42-get_dbc_fd--differential-box-counting)
  - [4.3 get_sdbc_fd](#43-get_sdbc_fd--shifted-dbc)
  - [4.4 get_fd](#44-get_fd--通用-log-log-拟合)
  - [4.5 get_batch_bc / dbc / sdbc（静态）](#45-get_batch_bc--dbc--sdbc静态)
  - [4.6 plot](#46-plot静态)
- [五、调用示例](#五调用示例)
- [六、注意事项](#六注意事项)

---

## 一、数学背景

box-counting 维度的一般形式：

$$D = -\lim_{s \to 0} \frac{\log N(s)}{\log s}$$

其中 N(s) 是边长 s 的盒子覆盖图像所需的"占用计数"。本工具实现三种盒子计数方式：

### BC（Box Counting，二值图）
对二值图，N(s) = 含至少 1 个非零像素的 s×s 盒数：

$$N_{BC}(s) = \#\{(\text{box}) : \exists\, p \in \text{box}, p > 0\}$$

适用于二值（轮廓、骨架、占据图等）。

### DBC（Differential Box Counting，Sarkar & Chaudhuri 1994）
对灰度图，每个 s×s 盒子覆盖的灰度区间内"高度盒"数：

$$n_r(i, j) = \left\lceil \frac{I_{\max}}{h} \right\rceil - \left\lceil \frac{I_{\min}}{h} \right\rceil + 1$$

其中 h = s · H / G_max，H = max(图边长)，G_max = 99 百分位。**总计**：

$$N_{DBC}(s) = \sum_{i, j} n_r(i, j)$$

### SDBC（Shifted DBC，Chen et al. 1995）
DBC 在边界跨越时会过计数。SDBC 把每个盒子的 z 坐标对齐到 I_min：

$$n_r(i, j) = \left\lfloor \frac{I_{\max} - I_{\min}}{h} \right\rfloor + 1$$

通常 SDBC ≤ DBC（对相同图）。

| 方法 | 输入 | 公式 |
|---|---|---|
| BC | 二值 (0/1 或非零=占用) | 仅计数占用盒 |
| DBC | 灰度 | $\lceil I_{\max}/h \rceil - \lceil I_{\min}/h \rceil + 1$ |
| SDBC | 灰度 | $\lfloor (I_{\max} - I_{\min})/h \rfloor + 1$ |

最终维度为 log-log 拟合斜率：

$$D = -\frac{d \log N(s)}{d \log s}$$

---

## 二、模块结构

| 模块 | 类名 | 默认设备 | 默认 dtype |
|---|---|---|---|
| `FAImageDimension.py` | `CFAImageDimension` | CPU（NumPy） | float64 |
| `FAImageDimensionGPU.py` | `CFAImageDimensionGPU` | 自动（CUDA→CPU） | torch.float32 |

GPU 类除多接受 `device`、`dtype`、`img_chunk` 外，与 CPU 类**完全相同的接口**。

---

## 三、CFAImageDimension / CFAImageDimensionGPU 类

### 构造函数

**CPU**：

```python
CFAImageDimension(
    image=None,                     # np.ndarray (H, W)；CPU 版必须传非 None
    max_size=None,                  # int / None
    max_scales=30,                  # int
    with_progress=True,             # bool
    min_size=2,                     # int
)
```

**GPU**（多 2 个参数）：

```python
CFAImageDimensionGPU(
    image=None, max_size=None, max_scales=30, with_progress=True,
    device=None,                    # str / torch.device / None
    dtype=torch.float32,            # torch.dtype
    min_size=2,
)
```

### 参数详解

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `np.ndarray (H, W)` 或 `None` | `None` | 2D 单通道；CPU 版**必填**，GPU 版可为 `None`（此时只能用 `get_fd()` 工具方法） |
| `max_size` | `int / None` | `None` | 最大 box 尺寸；`None` 取 `min(H, W)`；image=None 时必填 |
| `max_scales` | `int` | `30` | scales 数量上限；整数化去重后通常 ~25-28 个独立 scale |
| `with_progress` | `bool` | `True` | tqdm 进度条 |
| `min_size` | `int` | `2` | 最小 box 尺寸 |
| `device` | `str/torch.device/None` | `None` | GPU 专用；`None` 自动选 cuda |
| `dtype` | `torch.dtype` | `torch.float32` | GPU 专用；CPU/GPU byte-exact 一致请用 `torch.float64` |

### 实例属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `self.m_image` (CPU) / `self.img` (GPU) | `np.ndarray` / `torch.Tensor` | 原图像（CPU 不修改 dtype） |
| `self.m_scales` | `list[int]` | 实际使用的 scales（去重整数） |
| `self.device` | `torch.device` | 仅 GPU |
| `self.dtype` | `torch.dtype` | 仅 GPU |

---

## 四、方法详解

### 4.1 `get_bc_fd` — 二值 Box Counting

#### 功能
对**二值图**做 box-counting 维度估计。任意 >0 像素都算"占用"。

#### 签名

CPU 与 GPU 完全相同：

```python
get_bc_fd(self, corp_type=-1, fit_range=None)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `corp_type` | `int` | `-1` | ROI 裁剪策略：`-1`=裁到整数倍，`0`=严格要求整除否则报错，`1`=零填充 |
| `fit_range` | `tuple(float, float) / None` | `None` | `(s_min, s_max)` 限定拟合 box 范围；**强烈建议使用** |

#### 返回值

`dict`，键：

| 键 | 类型 | 说明 |
|---|---|---|
| `fd` | `float` | 拟合斜率（FD 估计）；数据不足或全零图时返回 NaN |
| `scales` | `list[float]` | 实际使用的 box 尺寸 |
| `counts` | `list[float]` | 每个尺度的占用盒数 N(s) |
| `log_scales` | `np.ndarray` | log(scales) |
| `log_counts` | `np.ndarray` | log(counts) |
| `intercept` | `float` | 拟合截距 |
| `r_value` | `float` | Pearson 相关系数 |
| `p_value` | `float` | p 值 |
| `std_err` | `float` | 斜率标准误 |

---

### 4.2 `get_dbc_fd` — Differential Box Counting

#### 功能
对**灰度图**做 DBC 维度估计（Sarkar & Chaudhuri 1994）。

#### 签名

```python
get_dbc_fd(self, corp_type=-1, fit_range=None)
```

参数同 `get_bc_fd`，返回字段同 `get_bc_fd`。

---

### 4.3 `get_sdbc_fd` — Shifted DBC

#### 功能
对**灰度图**做 SDBC 维度估计（Chen et al. 1995）。SDBC 修正了 DBC 在 box 边界处的过计数。

#### 签名

```python
get_sdbc_fd(self, corp_type=-1, fit_range=None)
```

参数同 `get_bc_fd`，返回字段同 `get_bc_fd`。

---

### 4.4 `get_fd` — 通用 log-log 拟合

#### 功能
工具函数：用户已有 `(scales, counts)` 对，直接拟合 log-log 斜率。可用于自定义 box-counting 实现的对接。

#### 签名

CPU：

```python
get_fd(self, scale_list, box_count_list)
```

GPU：

```python
get_fd(self, scale_list, box_count_list, fit_range=None)
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `scale_list` | array-like | box 尺寸序列 |
| `box_count_list` | array-like | 每个尺度的盒数 |
| `fit_range` | `tuple/None` | 拟合范围（仅 GPU 版接受） |

#### 返回值
`dict`，与 `get_bc_fd` 返回相同结构。

---

### 4.5 `get_batch_bc / dbc / sdbc`（静态）

#### 功能
对一组图像批量计算 BC/DBC/SDBC。CPU 版串行循环；GPU 版按 chunk 上 GPU 向量化（CUDA 上 5×~50× 加速）。

#### CPU 签名

```python
@staticmethod
CFAImageDimension.get_batch_bc(
    images,                         # Sequence[np.ndarray]
    max_size=None,                  # int / None
    max_scales=30,                  # int
    min_size=2,                     # int
    corp_type=-1,                   # int
    fit_range=None,                 # tuple / None
    with_progress=True,             # bool
) -> List[dict]

# get_batch_dbc, get_batch_sdbc 签名相同
```

#### GPU 签名（多 3 个参数）

```python
@staticmethod
CFAImageDimensionGPU.get_batch_bc(
    images, max_size=None, max_scales=30, min_size=2,
    corp_type=-1, fit_range=None,
    device=None,                    # str / torch.device / None
    dtype=torch.float32,            # fp64 时与 CPU byte-exact 一致
    img_chunk=None,                 # int / None；显存控制
    with_progress=True,
) -> List[dict]
```

#### 参数差异

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `images` | `Sequence[np.ndarray]` | — | 一组图像；GPU 版要求**所有图同尺寸**，CPU 版可异形 |
| `device` | `str/torch.device/None` | `None` | 仅 GPU |
| `dtype` | `torch.dtype` | `torch.float32` | 仅 GPU |
| `img_chunk` | `int/None` | `None` | 仅 GPU |

#### 返回值

`List[dict]`，每个元素是 `get_bc_fd()` 风格的 dict，**顺序与 `images` 顺序对应**。

---

### 4.6 `plot`（静态）

#### 功能
绘制 6 联子图：原图、二值图、BC/DBC/SDBC 三个 log-log 拟合图。

#### 签名

```python
plot(raw_img, gray_img, fd_bc, fd_dbc, fd_sdbc)
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `raw_img` | `np.ndarray` | 原始图（用于左上展示） |
| `gray_img` | `np.ndarray` | 二值图或预处理图（左中展示） |
| `fd_bc`, `fd_dbc`, `fd_sdbc` | `dict / None` | 各方法的返回结果；`None` 跳过对应子图 |

返回：`None`。

---

## 五、调用示例

### 示例 1：二值图 BC 维度（最简单）

```python
import cv2
import numpy as np
from FAImageDimension import CFAImageDimension

raw = cv2.imread("fractal.png", cv2.IMREAD_GRAYSCALE)
binary = (raw > 0).astype(np.uint8)             # 二值化

calc = CFAImageDimension(image=binary, max_scales=30)
result = calc.get_bc_fd(fit_range=(8, 200))

print(f"BC FD = {result['fd']:.4f}")
print(f"R² = {result['r_value']**2:.4f}")
print(f"使用的 scales: {result['scales']}")
print(f"对应的 counts: {result['counts']}")
```

### 示例 2：灰度图 DBC + SDBC

```python
import cv2
import numpy as np
from FAImageDimension import CFAImageDimension

img = cv2.imread("texture.png", cv2.IMREAD_GRAYSCALE)

calc = CFAImageDimension(image=img, max_scales=30)
fd_dbc  = calc.get_dbc_fd(fit_range=(4, 64))
fd_sdbc = calc.get_sdbc_fd(fit_range=(4, 64))

print(f"DBC  FD = {fd_dbc['fd']:.4f}")
print(f"SDBC FD = {fd_sdbc['fd']:.4f}")
```

### 示例 3：GPU 加速（必须用 fp64 与 CPU byte-exact 一致）

```python
import cv2
import numpy as np
import torch
from FAImageDimensionGPU import CFAImageDimensionGPU

img = cv2.imread("texture.png", cv2.IMREAD_GRAYSCALE)

# 默认 fp32，与 CPU 有 ~1e-4 量级差异（fp32 机器精度）
calc_fp32 = CFAImageDimensionGPU(image=img, max_scales=30, device='cuda')
fd_fp32 = calc_fp32.get_dbc_fd(fit_range=(4, 64))

# 用 fp64 与 CPU 完全一致
calc_fp64 = CFAImageDimensionGPU(
    image=img, max_scales=30, device='cuda',
    dtype=torch.float64,
)
fd_fp64 = calc_fp64.get_dbc_fd(fit_range=(4, 64))

print(f"fp32: {fd_fp32['fd']:.6f}")
print(f"fp64: {fd_fp64['fd']:.6f}")
```

### 示例 4：批处理 100 张图（GPU）

```python
import cv2, glob
import numpy as np
import torch
import pandas as pd
from FAImageDimensionGPU import CFAImageDimensionGPU

paths = sorted(glob.glob("dataset/*.png"))
imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

results = CFAImageDimensionGPU.get_batch_dbc(
    images=imgs,
    max_scales=30,
    fit_range=(4, 64),
    device='cuda',
    dtype=torch.float64,            # 与 CPU byte-exact
    img_chunk=16,                   # 一次最多 16 张图同时驻 GPU
    with_progress=True,
)

df = pd.DataFrame({
    'path': paths,
    'fd': [r['fd'] for r in results],
    'r2': [r['r_value']**2 for r in results],
})
df.to_csv("dbc_results.csv", index=False)
```

### 示例 5：异形图批处理（CPU 自动 fallback）

```python
from FAImageDimension import CFAImageDimension

# 不同尺寸的图
imgs = [
    np.random.randint(0, 256, (256, 256)).astype(np.uint8),
    np.random.randint(0, 256, (200, 320)).astype(np.uint8),
    np.random.randint(0, 256, (512, 384)).astype(np.uint8),
]

# CPU 版自动按图独立处理
results = CFAImageDimension.get_batch_dbc(
    images=imgs, max_scales=20, fit_range=(4, 64),
    with_progress=False,
)
fds = [r['fd'] for r in results]
print(f"FDs: {fds}")
```

### 示例 6：可视化三种方法

```python
import cv2
import numpy as np
from FAImageDimension import CFAImageDimension
import matplotlib.pyplot as plt

raw = cv2.imread("fractal.png", cv2.IMREAD_GRAYSCALE)
binary = (raw > 64).astype(np.uint8)

calc_bin  = CFAImageDimension(binary, max_scales=30)
calc_gray = CFAImageDimension(raw,    max_scales=30)

fd_bc   = calc_bin.get_bc_fd(fit_range=(4, 128))
fd_dbc  = calc_gray.get_dbc_fd(fit_range=(4, 64))
fd_sdbc = calc_gray.get_sdbc_fd(fit_range=(4, 64))

CFAImageDimension.plot(raw, binary, fd_bc, fd_dbc, fd_sdbc)
plt.show()
```

### 示例 7：仅有 (scales, counts) 数据

`get_fd` 是个工具方法，做单纯的 log-log 拟合。CPU 版构造时 `image` 是必填，**GPU 版** 才支持 `image=None`：

```python
import numpy as np
from FAImageDimensionGPU import CFAImageDimensionGPU

# 假设你已有自定义计数结果
my_scales = [2, 4, 8, 16, 32, 64]
my_counts = [1024, 256, 64, 16, 4, 1]

# 用 GPU 版可省略 image
calc = CFAImageDimensionGPU(image=None, max_size=64, with_progress=False)
fit = calc.get_fd(my_scales, my_counts)
print(f"slope = {fit['fd']:.4f}")
```

如果用 CPU 版，构造时随便传一张占位图就行（不影响 `get_fd` 的结果）：

```python
import numpy as np
from FAImageDimension import CFAImageDimension

placeholder = np.zeros((64, 64), dtype=np.uint8)
calc = CFAImageDimension(image=placeholder, max_size=64, with_progress=False)
fit = calc.get_fd([2, 4, 8, 16, 32, 64], [1024, 256, 64, 16, 4, 1])
print(f"slope = {fit['fd']:.4f}")
```

---

## 六、注意事项

### 6.1 BC 必须用二值图

**最常见错误**：把灰度图当二值图传给 `get_bc_fd`。结果 N(s) 几乎都接近最大值（任何非零像素都算占用），FD 趋近 2.0，无信息。

正确：
```python
binary = (raw > threshold).astype(np.uint8)
fd = CFAImageDimension(binary).get_bc_fd()
```

或者用 OpenCV：
```python
_, binary = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
fd = CFAImageDimension(binary).get_bc_fd()
```

### 6.2 DBC / SDBC 用灰度图

灰度图最好是 8-bit (0~255)。归一化常数 G_max 用 99 百分位避免离群值（hot pixel）。

### 6.3 fit_range 是改善精度的关键

box-counting 的 log-log 曲线**两端饱和**：
- 小 s 端：被像素离散化所限，N(s) 饱和到像素数
- 大 s 端：被图像尺寸所限，N(s) 饱和到 1

**只有中间是真正的幂律段**。**强烈建议**用 `fit_range=(s_min, s_max)` 选取中间段。

实测：Sierpinski carpet（理论 FD=1.8928）
- 不用 fit_range：BC ≈ 2.01（被两端污染）
- `fit_range=(8, 200)`：BC ≈ 1.94
- 用精确 base-3 scales：BC = 1.8928（精确）

### 6.4 GPU 默认 fp32 → 与 CPU 有 ~1e-4 差异

DBC/SDBC 涉及 `floor`/`ceil`，fp32 误差会让某些 box count 落在不同侧。**要 byte-exact 一致**时显式传 `dtype=torch.float64`。

| 默认 | CPU vs GPU 差异 |
|---|---|
| fp32（GPU 默认） | ~1e-4 ~ 1e-2（box 整数计数差异） |
| fp64 | ≤ 4.4e-16（机器精度） |

### 6.5 max_scales 是上限不是下限

logspace 整数化后会大量重复。`max_scales=30` 实际通常给 ~25-28 个独立 scale。设置过大没意义。

### 6.6 corp_type 选择

| 值 | 行为 | 何时用 |
|---|---|---|
| `-1`（默认） | 裁到 (H//s)·s 整数倍 | 最常用，损失少量边界像素 |
| `0` | 严格要求 H, W 被 s 整除，否则抛 ValueError | 调试或特定需求 |
| `1` | 零填充到 ⌈H/s⌉·s | 不想丢边界像素 |

### 6.7 退化输入

- **全零二值图** → BC FD = NaN（无占用盒）
- **全占用图** → BC FD ≈ 2（与 fit_range 有关）
- **均匀灰度图** → DBC/SDBC FD ≈ 2（每个盒只有 1 高度盒）

### 6.8 GPU 显存控制

OOM 时调小 `img_chunk`。BC 二值图占用最少（uint8），DBC/SDBC 灰度需要 `dtype` 大小（fp64=8B/像素）。

### 6.9 数学正确性验证

- Sierpinski carpet（base-3 scales）：BC = 1.8928 = log(8)/log(3)（精确）
- 平滑梯度图：DBC ≈ SDBC ≈ 2.03（接近 2，符合 2D 平滑信号）
- 全占用图：DBC FD ≈ 2.07（合理）
