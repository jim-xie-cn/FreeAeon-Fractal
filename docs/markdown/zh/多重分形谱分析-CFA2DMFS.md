# 多重分形谱分析 - CFA2DMFS

## 应用场景

`CFA2DMFS` 类用于计算2D灰度图像的多重分形谱（Multifractal Spectrum），是分析图像复杂度和异质性的高级工具。主要应用场景包括：

- **纹理分析**：量化图像纹理的多尺度特性
- **医学图像**：分析组织结构的异质性
- **材料科学**：研究材料表面的多重分形特征
- **金融分析**：分析价格波动的多重分形特性
- **地球科学**：研究地形地貌的多尺度特征

## 调用示例

### 基础用法

```python
import cv2
import numpy as np
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# 读取并转换为灰度图像
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 创建多重分形谱分析对象
q_list = np.linspace(-5, 5, 26)
MFS = CFA2DMFS(gray_image, q_list=q_list)

# 计算多重分形谱
df_mass, df_fit, df_spec = MFS.get_mfs()

# 查看结果
print(df_spec)

# 可视化结果
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU加速版本

```python
# 使用GPU加速版本
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS

# 其余代码相同
MFS = CFA2DMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFA2DMFS

**描述**：用于计算2D灰度图像多重分形谱的类，基于盒计数方法。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必需 | 输入灰度图像（2D数组） |
| `corp_type` | int | 0 | 图像裁剪类型（-1:裁剪, 0:严格匹配, 1:填充） |
| `q_list` | array-like | linspace(-5,5,51) | q值列表 |
| `with_progress` | bool | True | 是否显示进度条 |
| `bg_threshold` | float | 0.01 | 背景阈值 |
| `bg_reverse` | bool | False | 是否反转背景阈值 |
| `bg_otsu` | bool | False | 是否使用Otsu阈值化 |
| `mu_floor` | float | 1e-12 | 最小概率值（保留用于兼容） |

#### 主要方法

##### 1. get_mfs(max_size=None, max_scales=80, min_points=6, min_box=2, ...)

**描述**：完整的多重分形谱分析流程，返回质量表、拟合结果和谱参数。

**参数**：
- `max_size` (int): 最大盒子尺寸
- `max_scales` (int): 最大尺度数量
- `min_points` (int): 拟合的最小点数
- `min_box` (int): 最小盒子尺寸
- `use_middle_scales` (bool): 是否只使用中间尺度
- `fit_scale_frac` (tuple): 拟合尺度范围 (0.2, 0.8)
- `if_auto_line_fit` (bool): 是否自动线性拟合
- `auto_fit_min_len_ratio` (float): 自动拟合最小长度比例
- `spline_s` (float): 样条平滑参数
- `cap_d0_at_2` (bool): 是否限制D0≤2

**返回值** (tuple):
```python
(df_mass, df_fit, df_spec)
```

- `df_mass`: 质量表 DataFrame，包含列：
  - `scale`: 盒子尺寸
  - `eps`: 归一化尺度
  - `q`: q值
  - `value`: 根据kind不同含义不同
  - `kind`: 值类型（'logMq', 'N', 'S'）

- `df_fit`: 拟合结果 DataFrame，包含列：
  - `q`: q值
  - `tau`: τ(q) 质量指数
  - `Dq`: D(q) 广义维度
  - `D1`: D1 信息维度（仅q=1时）
  - `intercept`: 拟合截距
  - `r_value`: 相关系数
  - `p_value`: p值
  - `std_err`: 标准误差
  - `n_points`: 拟合点数

- `df_spec`: 多重分形谱 DataFrame，包含列：
  - `q`: q值
  - `tau`: τ(q)
  - `Dq`: D(q)
  - `alpha`: α(q) 奇异性强度
  - `f_alpha`: f(α) 多重分形谱

##### 2. get_mass_table(max_size=None, max_scales=80, min_box=2, roi_mode="center")

**描述**：计算每个尺度下的盒子质量表。

**返回值**：DataFrame，包含每个尺度和q值的质量信息。

##### 3. fit_tau_and_D1(df_mass, min_points=6, ...)

**描述**：从质量表拟合τ(q)和D1。

**返回值**：拟合结果DataFrame。

##### 4. alpha_falpha_from_tau(df_fit, spline_k=3, exclude_q1=True, spline_s=0)

**描述**：从τ(q)计算α(q)和f(α)。

**参数**：
- `df_fit`: 拟合结果DataFrame
- `spline_k` (int): 样条插值阶数
- `exclude_q1` (bool): 是否排除q=1
- `spline_s` (float): 样条平滑参数

**返回值**：多重分形谱DataFrame。

##### 5. plot(df_mass, df_fit, df_spec)

**描述**：可视化多重分形谱分析结果，包含6个子图：
1. log M(q, ε) 热图
2. f(α) vs α 多重分形谱
3. τ(q) vs q
4. D(q) vs q 广义维度
5. α vs q
6. f(α) vs q

## 理论背景

### 多重分形谱

多重分形谱描述了图像在不同尺度下的统计特性。核心参数包括：

#### 1. 配分函数 (Partition Function)
```
M(q, ε) = Σ μᵢ^q
```
其中 μᵢ 是第i个盒子的归一化质量。

#### 2. 质量指数 τ(q)
```
τ(q) = lim(ε→0) log M(q, ε) / log ε
```

#### 3. 广义维度 D(q)
```
D(q) = τ(q) / (q - 1), q ≠ 1
D(1) = lim(ε→0) Σ μᵢ log μᵢ / log ε  (信息维度)
```

特殊值：
- D(0): 容量维度
- D(1): 信息维度
- D(2): 关联维度

#### 4. 多重分形谱 f(α)
```
α(q) = dτ(q)/dq  (奇异性强度)
f(α) = qα - τ(q)  (多重分形谱)
```

### 计算方法

本实现使用改进的盒计数方法：

1. **对于 q ≠ 0, 1**：
   - 在对数空间计算 log M(q, ε)
   - 使用 logsumexp 避免数值溢出

2. **对于 q = 0**：
   - 计算非零盒子数 N
   - 拟合 log(N) vs log(1/ε)

3. **对于 q = 1**：
   - 计算 Shannon 熵 S = -Σ μᵢ log μᵢ
   - 拟合 S vs log(1/ε)

4. **固定ROI方案**：
   - 裁剪为固定的正方形ROI
   - 只使用ROI尺寸的约数作为盒子大小
   - 确保跨尺度的一致性

## 重要提示

1. **图像预处理**：
   - 输入必须是2D灰度图像
   - 图像会自动归一化到 [0, 1]
   - 可使用 `bg_threshold` 去除背景

2. **q值选择**：
   - 负q值对稀疏区域敏感
   - 正q值对密集区域敏感
   - 建议范围：-5 到 5
   - 点数建议：20-50个

3. **尺度参数**：
   - `max_scales` 影响采样密度，建议60-100
   - `min_box` 不要小于2
   - 使用固定ROI避免边界效应

4. **结果解释**：
   - f(α) 曲线越宽，多重分形性越强
   - Δα = α_max - α_min 量化异质性
   - D(q) 单调递减表示多重分形
   - r_value 接近1表示良好拟合

5. **性能优化**：
   - 使用GPU版本可显著加速
   - 大图像可先降采样
   - 减少q值数量可提高速度

## 常见问题

### Q: D(0) > 2 怎么办？
A: 可以设置 `cap_d0_at_2=True` 限制D(0)≤2，或检查图像预处理是否正确。

### Q: 如何判断是否为多重分形？
A: 检查 D(q) 是否随q单调变化，或者 f(α) 是否为凸函数且有一定宽度。

### Q: 计算结果不稳定？
A: 增加 `max_scales`，使用 `use_middle_scales=True`，或调整 `fit_scale_frac`。

## 参考文献

- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. Physical Review Letters.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. Chaos and Fractals.
