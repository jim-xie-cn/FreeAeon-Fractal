# 分形维度分析 - CFAImageFD

## 应用场景

`CFAImageFD` 类用于计算2D图像的分形维度，是图像复杂度分析和纹理特征提取的重要工具。主要应用场景包括：

- **图像纹理分析**：量化图像的粗糙度和复杂性
- **医学图像分析**：分析组织结构的复杂程度
- **材料科学**：研究材料表面的分形特征
- **计算机视觉**：图像特征提取和分类
- **地质学**：地形和地貌复杂度分析

## 使用示例

### 基础用法

```python
import cv2
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImage import CFAImage

# 读取图像
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# BC方法需要二值图像
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# 计算三种分形维度
fd_bc = CFAImageFD(bin_image).get_bc_fd(corp_type=-1)
fd_dbc = CFAImageFD(gray_image).get_dbc_fd(corp_type=-1)
fd_sdbc = CFAImageFD(gray_image).get_sdbc_fd(corp_type=-1)

# 输出结果
print("BC 分形维度:", fd_bc['fd'])
print("DBC 分形维度:", fd_dbc['fd'])
print("SDBC 分形维度:", fd_sdbc['fd'])

# 可视化结果
CFAImageFD.plot(rgb_image, gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### GPU加速版本

```python
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU

fd_bc = CFAImageFDGPU(bin_image, device='cuda').get_bc_fd()
fd_dbc = CFAImageFDGPU(gray_image, device='cuda').get_dbc_fd()
fd_sdbc = CFAImageFDGPU(gray_image, device='cuda').get_sdbc_fd()
```

### 批量处理

```python
import cv2, glob
from FreeAeonFractal.FAImageFD import CFAImageFD

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]

results_bc = CFAImageFD.get_batch_bc(images)
results_dbc = CFAImageFD.get_batch_dbc(images)
results_sdbc = CFAImageFD.get_batch_sdbc(images)

for r in results_bc:
    print("BC FD:", r['fd'])
```

### 限制拟合范围

```python
# 只使用中间尺度进行拟合（避免对数-对数曲线末端的饱和效应）
fd_bc = CFAImageFD(bin_image, max_scales=30).get_bc_fd(fit_range=(4, 64))
```

### 自动拟合区间选择 (`auto_fit_range`)

盒计数 log-log 曲线呈 sigmoid 形 —— 小尺度受像素离散化饱和，大尺度受图像边界饱和，只有中间一段才是真正的幂律。把整条曲线当成一条直线去拟合会引入系统性 slope 偏置。`auto_fit_range=True` 用 F 统计量 `R²/(1−R²)·(n−2)` 自动挑选最线性的子窗口。

```python
# BC: 自动挑选线性 scaling region
result = CFAImageFD(img, max_scales=30).get_bc_fd(auto_fit_range=True)
print(f"D={result['fd']:.4f}, R²={result['r_value']**2:.4f}, "
      f"挑中的窗口={result['fit_range_used']}")

# DBC/SDBC: 默认 auto_slope_floor=2.0 防止选中小尺度的 BC-退化段
fd_dbc  = CFAImageFD(img, max_scales=30).get_dbc_fd(auto_fit_range=True)
fd_sdbc = CFAImageFD(img, max_scales=30).get_sdbc_fd(auto_fit_range=True)

# 关闭物理约束（不推荐，会让 D 跌破 2.0 物理下限）：
fd_old = CFAImageFD(img, max_scales=30).get_dbc_fd(
    auto_fit_range=True, auto_slope_floor=None)

# 显式 fit_range 永远优先级最高：
fd_manual = CFAImageFD(img, max_scales=30).get_bc_fd(
    fit_range=(4, 64), auto_fit_range=True)
# → fit_range_used == (4, 64)，auto 被忽略

# 批量：每张图独立挑窗口
batch = CFAImageFD.get_batch_dbc(imgs, max_scales=30,
                                  auto_fit_range=True,
                                  with_progress=False)
for r in batch:
    print(r['fd'], r['fit_range_used'])
```

#### `auto_fit_range` 适用范围

`auto_fit_range=True` 是一个**工具**，不是万能默认值。按图像类型选择：

**✅ 适合用 auto 的场景**：
- **纹理 / 表面粗糙度图像** —— scaling region 宽且清晰，auto 能稳定挑到中段
- **噪声场 / fBm 表面 / 合成分形** —— log-log 是典型 sigmoid 且中间段平坦，auto 比全段拟合好
- **大批量异构数据** —— 每张图的线性区不同，auto 逐图自动挑选，无需手工调参
- **高频细节丰富的医学影像** —— 例如肿瘤组织、视网膜纹理

**❌ 不要用 auto 的场景**（请用 default 或显式 `fit_range`）：
- **人脸图像（特别是经过 CLAHE / 归一化处理后）** —— log-log 曲线小尺度段会出现一段**很长的"BC 退化平台"**（slope ≈ 2.0，每块都 `n_r ≡ 1`）。即使默认 `auto_slope_floor=2.0`，picker 也会刚好擦着 floor 选中这段平台，返回 D 略高于 2（如 2.008）——**不是真实人脸的 DBC**（典型 2.2-2.5）。人脸研究请用 default 全段拟合，或手工 `fit_range=(8, 64)`；如果坚持用 auto，把 `auto_slope_floor` 提到 2.1 可以跳过平台。
- **极平滑 / 接近常数图** —— 根本没有真 scaling region，auto 要么选退化窗口要么直接回退到全曲线
- **严格复现文献数字** —— 发表的论文通常用全段或 dyadic 拟合。Auto 选窗逐图不同，破坏可比性

**⚠ 用 auto 时必查**：
- `result['fit_range_used']` —— auto 选了哪段？如果 `s_lo` 在 2 或 3，怀疑 BC 退化
- `result['r_value']**2` —— auto 本就偏好高 R²，**R² ≈ 1 不代表物理正确**，只代表数学上线性
- 数据集第一次跑前，建议至少画一次 log-log + local slope 图（`np.diff(log_N) / np.diff(log_s)`）确认真 scaling region 在哪

**诊断代码片段**：
```python
import matplotlib.pyplot as plt
r = CFAImageFD(img, max_scales=30, with_progress=False).get_dbc_fd()
scales = np.array(r['scales']); counts = np.array(r['counts'])
log_s = -np.log(scales); log_N = np.log(counts)
slope_local = np.diff(log_N) / np.diff(log_s)
mid_s = (scales[1:] + scales[:-1]) / 2
plt.plot(mid_s, slope_local, 'o-'); plt.xscale('log')
plt.axhline(2.0, ls='--', color='r')
plt.xlabel('scale s'); plt.ylabel('local slope (= local D)'); plt.show()
# slope 在 ≈ 2 处有长平台 → BC 退化，别用 auto
# slope 从 2 单调爬升到 3 → 用 auto 且 floor=2.0 没问题
```

## 类说明

### CFAImageFD

**描述**：用于计算2D图像分形维度的类，支持盒计数法（BC）、差分盒计数法（DBC）和移位差分盒计数法（SDBC）三种方法。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入图像（2D单通道数组） |
| `max_size` | int | None | 最大盒子尺寸（默认为图像最小维度） |
| `max_scales` | int | 30 | 目标不同尺度数量 |
| `with_progress` | bool | True | 是否显示进度条 |
| `min_size` | int | 2 | 最小盒子尺寸 |

#### 主要方法

##### 1. get_bc_fd(corp_type=-1, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=None, auto_slope_ceiling=None)

**描述**：使用盒计数法（BC）计算分形维度。任何正值像素计为已占用。

**参数**：
- `corp_type` (int)：图像处理方式
  - `-1`：自动裁剪为盒子尺寸的整数倍（默认）
  - `0`：不处理（要求图像尺寸恰好为整数倍）
  - `1`：补零填充
- `fit_range` (tuple 或 None)：可选的 `(最小尺度, 最大尺度)` 限制拟合范围
- `auto_fit_range` (bool, 默认 `False`)：开启后且 `fit_range` 为 `None` 时，自动用 F 统计量 `R²/(1−R²)·(n−2)` 挑选最线性的对数-对数子窗口。显式 `fit_range` 永远优先。
- `auto_fit_min_points` (int, 默认 `5`)：自动选窗口的最小点数
- `auto_slope_floor` (float 或 None, BC 默认 `None`)：拒绝 slope 低于此值的候选窗口。BC 理论范围 `D ∈ [0, 2]`，所以不设默认下限
- `auto_slope_ceiling` (float 或 None, BC 默认 `None`)：拒绝 slope 高于此值的候选窗口。传 `2.0` 可强制 BC 上限，避免 fitting artifact 给出 `D > 2`

**返回值** (dict)：
```python
{
    'fd': float,        # 分形维度值
    'scales': list,     # 使用的尺度列表（经过滤）
    'counts': list,     # 盒子计数列表（经过滤）
    'log_scales': list, # 拟合中的 log(1/r) 值
    'log_counts': list, # 拟合中的 log(N(r)) 值
    'intercept': float, # 拟合截距
    'r_value': float,   # 皮尔逊相关系数
    'p_value': float,   # p值
    'std_err': float,   # 标准误差
    'fit_range_used': tuple or None  # 实际应用的拟合区间。
                                     # None: 未做区间限制；
                                     # (s_lo, s_hi): 手动或 auto 挑出的窗口。
}
```

**适用场景**：适用于二值图像，计算覆盖前景所需的盒子数量。

##### 2. get_dbc_fd(corp_type=-1, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=2.0, auto_slope_ceiling=None)

**描述**：使用差分盒计数法（DBC）计算分形维度（Sarkar & Chaudhuri 1994）。

公式：`n_r = ceil(I_max / h) - ceil(I_min / h) + 1`，其中 `h = s × G_max / H`（`s` 为盒边长像素，`H` 为图像边长，`G_max` 为灰度范围，代码里取 `np.percentile(image, 99)` 以增强鲁棒性）。

**参数**：同 `get_bc_fd`，但 `auto_slope_floor` **默认为 `2.0`**（DBC 物理下限：2D 灰度 surface 对应 `D ∈ [2, 3]`）。

**为何默认 `auto_slope_floor=2.0`？** 在偏平滑图像（如人脸皮肤区）的小尺度上，每个块的 `I_max − I_min ≤ h`，导致 `n_r ≡ 1`，log-log 曲线退化为 BC 行为，slope ≈ 2。如果不设下限，`auto_fit_range=True` 经常会选中这段退化区间，给出略低于 2 的 D —— 物理上不可能。下限 2.0 强制 auto 落到真正的 DBC scaling region。如需关闭可传 `auto_slope_floor=None`。

**G_max 说明**：`G_max` 用 99 百分位（鲁棒），与 Sarkar & Chaudhuri 1994 原文的固定 `G=256` 不同。这是设计上的"内容自适应"取舍。跨图比较前请先做光照归一化。

**返回值**：同 `get_bc_fd`（包含 `fit_range_used`）

**适用场景**：适用于灰度图像，将图像视为3D曲面。

##### 3. get_sdbc_fd(corp_type=-1, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=2.0, auto_slope_ceiling=None)

**描述**：使用移位差分盒计数法（SDBC）计算分形维度（Chen et al. 1995）。

公式：`n_r = floor((I_max - I_min) / h) + 1`，h 计算同 DBC。这是 Chen 1995 "shift vertical origin" 几何构造的解析解，不是近似。

**参数**：同 `get_dbc_fd`（包括默认 `auto_slope_floor=2.0`）

**返回值**：同 `get_bc_fd`

**适用场景**：灰度图像的改进版 DBC，对加性亮度偏置免疫，在光照变化下更稳定 —— 人脸/纹理分类等场景推荐用 SDBC。

##### 4. get_fd(scale_list, box_count_list, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=None, auto_slope_ceiling=None)

**描述**：对自定义尺度和计数数据进行对数-对数线性回归的工具方法。支持上述全部 `auto_*` 参数。

##### 5. plot(raw_img, gray_img, bin_img, fd_bc, fd_dbc, fd_sdbc) [静态方法]

**描述**：以2×3子图网格可视化三种分形维度的结果。

##### 6. get_batch_bc / get_batch_dbc / get_batch_sdbc [静态方法]

**描述**：批量处理多张图像的静态方法，在批次中共享尺度生成。**全部支持 `auto_fit_range` / `auto_fit_min_points` / `auto_slope_floor` / `auto_slope_ceiling`**，每张图独立挑选自己的最优窗口；批量 `get_batch_dbc` / `get_batch_sdbc` 默认 `auto_slope_floor=2.0`，`get_batch_bc` 默认 `None`。

**返回值**：结果字典列表（每张图一个，含 `fit_range_used`）

## 算法说明

### BC（盒计数法）

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

### DBC（差分盒计数法）

```
h = s × G_max / H
n_r(i,j) = ceil(I_max / h) - ceil(I_min / h) + 1
```

### SDBC（移位差分盒计数法）

```
n_r(i,j) = floor((I_max - I_min) / h) + 1
```

## 重要说明

1. **图像预处理**：BC方法需要二值图像，DBC/SDBC直接接受灰度图像
2. **尺度选择**：`max_scales=30` 是合适的默认值；使用 `fit_range` 排除不可靠的极端尺度
3. **结果解读**：2D图像分形维度通常在1-2之间；`r_value` 接近1表示良好的幂律拟合
4. **性能优化**：大图像或批量处理时使用 `CFAImageFDGPU`

## 参考文献

- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.
- Sarkar, N., & Chaudhuri, B. B. (1994). *IEEE Transactions on Systems, Man, and Cybernetics*.
- Chen, W. S., et al. (1995). *Pattern Recognition*.
