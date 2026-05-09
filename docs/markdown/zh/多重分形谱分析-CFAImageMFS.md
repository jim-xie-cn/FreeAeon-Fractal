# 多重分形谱分析 - CFAImageMFS

## 应用场景

`CFAImageMFS` 类用于计算2D灰度图像的多重分形谱，是分析图像复杂度和异质性的高级工具。主要应用场景包括：

- **纹理分析**：量化图像纹理的多尺度特性
- **医学图像**：分析组织结构异质性
- **材料科学**：研究材料表面的多重分形特征
- **金融分析**：分析价格波动的多重分形特性
- **地球科学**：研究地形地貌的多尺度特征

## 使用示例

### 基础用法

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

# 读取并转换为灰度图像
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 创建多重分形谱分析对象
MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 51))

# 计算多重分形谱
df_mass, df_fit, df_spec = MFS.get_mfs()

# 查看结果
print(df_spec[['q', 'alpha', 'f_alpha']].head(10))

# 可视化
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU加速版本

```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 51))
df_mass, df_fit, df_spec = MFS.get_mfs()
MFS.plot(df_mass, df_fit, df_spec)
```

### 局部奇异度图

```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS

MFS = CFAImageMFS(gray_image)
alpha_map, info = MFS.compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")
CFAImageMFS.plot_alpha_map(alpha_map)
```

### 批量处理

```python
import glob, cv2, numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageMFS.get_batch_mfs(images, q_list=np.linspace(-5, 5, 26))

for df_mass, df_fit, df_spec in results:
    print(df_fit[['q', 'tau', 'Dq']].head())
```

## 类说明

### CFAImageMFS

**描述**：基于盒计数法的2D灰度图像多重分形分析类。使用固定正方形ROI（方案A）确保跨尺度的ε归一化一致性。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入2D灰度图像 |
| `corp_type` | int | -1 | 图像裁剪方式（-1:裁剪, 0:严格, 1:填充） |
| `q_list` | array-like | linspace(-5,5,51) | q值列表 |
| `with_progress` | bool | True | 是否显示进度条 |
| `bg_threshold` | float | 0.01 | 背景阈值（归一化后低于此值的像素置零） |
| `bg_reverse` | bool | False | 若为True则对高于阈值的像素置零 |
| `bg_otsu` | bool | False | 对原始图像使用Otsu法去除背景 |
| `mu_floor` | float | 1e-12 | 保留以兼容旧版API |

#### 主要方法

##### 1. get_mass_table(max_size=None, max_scales=80, min_box=2, roi_mode="center")

**描述**：计算各尺度的配分函数表。

**返回值** (DataFrame)：列 `scale, eps, q, value, kind`
- `kind='logMq'`：q≠0,1时的对数配分函数值
- `kind='N'`：q=0时的非零盒子数量
- `kind='S'`：q=1时的香农熵

##### 2. fit_tau_and_D1(df_mass, min_points=6, ...)

**描述**：通过线性回归从质量表拟合 τ(q) 和 D(q)。

**返回值** (DataFrame)：列 `q, tau, Dq, D1, intercept, r_value, p_value, std_err, n_points`

##### 3. alpha_falpha_from_tau(df_fit, spline_k=3, exclude_q1=True, spline_s=0)

**描述**：通过 τ(q) 的样条导数（Legendre变换）计算 α(q) 和 f(α)。

##### 4. get_mfs(max_size=None, max_scales=80, min_points=6, ...)

**描述**：完整多重分形谱分析流水线。

**返回值** (三个DataFrame的元组)：

**`df_mass`** — 原始配分函数表：scale, eps, q, value, kind

**`df_fit`** — 回归结果：q, tau, Dq, D1, intercept, r_value, p_value, std_err, n_points

**`df_spec`** — 多重分形谱：q, tau, Dq, alpha, f_alpha

##### 5. compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

**描述**：通过流式OLS计算逐像素局部Hölder指数 α(x,y)。

**返回值**：`(alpha_map, info)` — (L,L) float64数组和元信息字典

##### 6. plot(df_mass, df_fit, df_spec)

**描述**：以2×3子图网格可视化分析结果：log M(q,ε)热图、f(α) vs α、τ(q)、D(q)、α(q)、f(α) vs q。

##### 7. plot_alpha_map(alpha_map) [静态方法]

**描述**：使用jet色图可视化局部α分布图。

##### 8. get_batch_mfs(img_list, ...) [静态方法]

**描述**：批量CPU多重分形谱计算，与 `CFAImageMFSGPU.get_batch_mfs` API兼容。

## 理论背景

### 核心计算流程

1. **盒概率**：μᵢ(ε) = mass_i / total_mass
2. **配分函数**：M(q, ε) = Σᵢ μᵢ^q
3. **质量指数**：τ(q) = lim log M(q,ε) / log ε
4. **广义维度**：D(q) = τ(q)/(q-1)，q≠1；D(1)为信息维度
5. **多重分形谱**：α(q) = dτ/dq，f(α) = q·α − τ(q)

### 特殊维度含义

- **D(0)**：容量（盒计数）维度
- **D(1)**：信息维度
- **D(2)**：关联维度

## 重要说明

1. **图像预处理**：输入需为2D灰度数组；使用 `bg_threshold` 或 `bg_otsu=True` 去除背景
2. **q值选择**：推荐 −5到5，20-51个点；负q对稀疏区域敏感，正q对密集区域敏感
3. **结果解读**：f(α) 曲线越宽表示多重分形性越强；D(q)随q单调递减表示多重分形数据
4. **性能**：使用 `CFAImageMFSGPU` 可获得5-20×加速

## 参考文献

- Chhabra, A., & Jensen, R. V. (1989). *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). *Chaos and Fractals*.
