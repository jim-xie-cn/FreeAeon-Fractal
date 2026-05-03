# 多重分形谱分析 - CFAImageMFS

## 应用场景

`CFAImageMFS` 类用于计算2D灰度图像的多重分形谱，是分析图像复杂度和异质性的高级工具。主要应用场景包括：

- **纹理分析**：量化图像纹理的多尺度特性
- **医学图像**：分析组织结构的异质性
- **材料科学**：研究材料表面的多重分形特征
- **金融分析**：分析价格波动的多重分形特性
- **地球科学**：研究地形和地貌的多尺度特征

## 使用示例

### 基本用法

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

# 读取并转换为灰度图像
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 创建多重分形谱分析对象
q_list = np.linspace(-5, 5, 26)
MFS = CFAImageMFS(gray_image, q_list=q_list)

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
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

# 其余代码相同
MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

### 局部Hölder指数图（逐像素奇异度）

```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS

MFS = CFAImageMFS(gray_image)
alpha_map = MFS.compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

CFAImageMFS.plot_alpha_map(alpha_map)
```

### 批量处理

```python
import glob, cv2
from FreeAeonFractal.FAImageMFS import CFAImageMFS

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageMFS.get_batch_mfs(images, q_list=np.linspace(-5, 5, 26))
for df_mass, df_fit, df_spec in results:
    print(df_fit[['q','tau','Dq']].head())
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImageMFS

**描述**：基于box计数方法计算2D灰度图像多重分形谱的类。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入灰度图像（2D数组） |
| `corp_type` | int | -1 | 图像裁剪类型（-1:裁剪, 0:严格匹配, 1:填充） |
| `q_list` | array-like | linspace(-5,5,51) | q值列表 |
| `with_progress` | bool | True | 是否显示进度条 |
| `bg_threshold` | float | 0.01 | 背景阈值（低于此值的像素被屏蔽） |
| `bg_reverse` | bool | False | 是否反转背景阈值 |
| `bg_otsu` | bool | False | 是否使用Otsu阈值检测背景 |
| `mu_floor` | float | 1e-12 | 最小概率下限（向后兼容保留） |

#### 主要方法

##### 1. get_mass_table(max_size=None, max_scales=80, min_box=2, roi_mode="center")

**描述**：计算所有尺度和q值下的配分函数表。

**返回值**（DataFrame）：列 `scale, eps, q, value, kind`
- `kind` ∈ `{'logMq', 'N', 'S'}` — 对数配分函数、box计数、Shannon熵

##### 2. fit_tau_and_D1(df_mass, min_points=6, require_common_scales=True, use_middle_scales=False, fit_scale_frac=(0.2, 0.8), if_auto_line_fit=False, auto_fit_min_len_ratio=0.5, cap_d0_at_2=True)

**描述**：从质量表拟合 τ(q) 和 D(q)。

**返回值**（DataFrame）：列 `q, tau, Dq, D1, intercept, r_value, p_value, std_err, n_points`

**注意**：q=0行使用log(N)拟合；q=1行使用Shannon熵(S)拟合；其他q使用logMq拟合。df_fit中始终包含强制的q=0和q=1行。

##### 3. alpha_falpha_from_tau(df_fit, spline_k=3, exclude_q1=True, spline_s=0)

**描述**：通过对τ(q)的样条微分计算α(q)和f(α)。

**参数**：
- `spline_k`（int）：样条阶数（默认3）
- `exclude_q1`（bool）：是否从样条中排除q=1（避免不连续性）
- `spline_s`（float）：样条平滑因子（0=精确插值）

**返回值**（DataFrame）：列 `q, tau, Dq, alpha, f_alpha`

##### 4. get_mfs(max_size=None, max_scales=80, min_points=6, min_box=2, use_middle_scales=False, fit_scale_frac=(0.2, 0.8), if_auto_line_fit=False, auto_fit_min_len_ratio=0.5, spline_s=0, cap_d0_at_2=True)

**描述**：完整的多重分形谱分析流程。

**返回值**（tuple）：
```python
(df_mass, df_fit, df_spec)
```

- `df_mass`：质量表DataFrame，列：
  - `scale`：box大小
  - `eps`：归一化尺度
  - `q`：q值
  - `value`：根据kind不同含义不同
  - `kind`：值类型（'logMq', 'N', 'S'）

- `df_fit`：拟合结果DataFrame，列：
  - `q`：q值
  - `tau`：τ(q)质量指数
  - `Dq`：D(q)广义维度
  - `D1`：D1信息维度（仅q=1时）
  - `intercept`：拟合截距
  - `r_value`：相关系数
  - `p_value`：p值
  - `std_err`：标准误差
  - `n_points`：拟合点数

- `df_spec`：多重分形谱DataFrame，列：
  - `q`：q值
  - `tau`：τ(q)
  - `Dq`：D(q)
  - `alpha`：α(q)奇异性强度
  - `f_alpha`：f(α)多重分形谱

##### 5. compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

**描述**：通过流式OLS结合嵌套网格优化，计算逐像素的局部Hölder指数α。

**返回值**：局部α值的2D numpy数组

##### 6. compute_alpha_map_batch(images, ...) [静态方法]

**描述**：批量计算alpha图。

##### 7. plot(df_mass, df_fit, df_spec)

**描述**：可视化多重分形谱分析结果，包含6个子图：
1. log M(q, ε)热图
2. f(α) vs α 多重分形谱
3. τ(q) vs q
4. D(q) vs q 广义维度
5. α vs q
6. f(α) vs q

## 理论背景

### 多重分形谱

多重分形谱描述图像在不同尺度下的统计特性。核心参数包括：

#### 1. 配分函数
```
M(q, ε) = Σ μᵢ^q
```
其中 μᵢ 是第 i 个box的归一化质量。

#### 2. 质量指数 τ(q)
```
τ(q) = lim(ε→0) log M(q, ε) / log ε
```

#### 3. 广义维度 D(q)
```
D(q) = τ(q) / (q - 1), q ≠ 1
D(1) = lim(ε→0) Σ μᵢ log μᵢ / log ε  （信息维度）
```

特殊值：
- D(0)：容量维度
- D(1)：信息维度
- D(2)：关联维度

#### 4. 多重分形谱 f(α)
```
α(q) = dτ(q)/dq  （奇异性强度）
f(α) = qα - τ(q)  （多重分形谱）
```

## 注意事项

1. **图像预处理**：
   - 输入必须为2D灰度图像
   - 图像自动归一化到 [0, 1]
   - 使用 `bg_threshold` 屏蔽背景；`bg_otsu=True` 自动检测背景

2. **q值选择**：
   - 负q值对稀疏区域敏感
   - 正q值对密集区域敏感
   - 建议范围：-5 到 5
   - 建议点数：20-50

3. **尺度参数**：
   - `max_scales` 影响采样密度，建议60-80
   - `min_box` 不应小于2
   - `roi_mode="center"` 使用固定方形ROI以避免边缘效应

4. **结果解释**：
   - f(α)曲线越宽表示多重分形性越强
   - Δα = α_max - α_min 量化异质性
   - D(q)单调递减表示多重分形
   - r_value 接近1表示拟合良好

5. **性能优化**：
   - 使用GPU版本（`CFAImageMFSGPU`）显著加速
   - GPU版本使用 `q_chunk` 和 `img_chunk` 控制显存用量
   - 默认数据类型：单图像float64，批处理float32

## 参考文献

- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. *Chaos and Fractals*.
