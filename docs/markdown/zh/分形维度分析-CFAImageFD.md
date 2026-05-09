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

##### 1. get_bc_fd(corp_type=-1, fit_range=None)

**描述**：使用盒计数法（BC）计算分形维度。任何正值像素计为已占用。

**参数**：
- `corp_type` (int)：图像处理方式
  - `-1`：自动裁剪为盒子尺寸的整数倍（默认）
  - `0`：不处理（要求图像尺寸恰好为整数倍）
  - `1`：补零填充
- `fit_range` (tuple 或 None)：可选的 `(最小尺度, 最大尺度)` 限制拟合范围

**返回值** (dict)：
```python
{
    'fd': float,        # 分形维度值
    'scales': list,     # 使用的尺度列表
    'counts': list,     # 盒子计数列表
    'log_scales': list, # 拟合中的 log(1/r) 值
    'log_counts': list, # 拟合中的 log(N(r)) 值
    'intercept': float, # 拟合截距
    'r_value': float,   # 皮尔逊相关系数
    'p_value': float,   # p值
    'std_err': float    # 标准误差
}
```

**适用场景**：适用于二值图像，计算覆盖前景所需的盒子数量。

##### 2. get_dbc_fd(corp_type=-1, fit_range=None)

**描述**：使用差分盒计数法（DBC）计算分形维度（Sarkar & Chaudhuri 1994）。

公式：`n_r = ceil(I_max / h) - ceil(I_min / h) + 1`，其中 `h = s × H / G_max`。

**适用场景**：适用于灰度图像，将图像视为3D曲面。

##### 3. get_sdbc_fd(corp_type=-1, fit_range=None)

**描述**：使用移位差分盒计数法（SDBC）计算分形维度（Chen et al. 1995）。

公式：`n_r = floor((I_max - I_min) / h) + 1`。

**适用场景**：灰度图像的改进版DBC，小尺度下精度更高。

##### 4. get_fd(scale_list, box_count_list)

**描述**：对自定义尺度和计数数据进行对数-对数线性回归的工具方法。

##### 5. plot(raw_img, gray_img, bin_img, fd_bc, fd_dbc, fd_sdbc) [静态方法]

**描述**：以2×3子图网格可视化三种分形维度的结果。

##### 6. get_batch_bc / get_batch_dbc / get_batch_sdbc [静态方法]

**描述**：批量处理多张图像的静态方法，在批次中共享尺度生成。

**返回值**：结果字典列表

## 算法说明

### BC（盒计数法）

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

### DBC（差分盒计数法）

```
h = s × H / G_max
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
