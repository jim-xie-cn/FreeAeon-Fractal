# 分形维度分析 - CFAImageFD

## 应用场景

`CFAImageFD` 类用于计算2D图像的分形维度，是图像复杂度分析和纹理特征提取的重要工具。主要应用场景包括：

- **图像纹理分析**：量化图像的粗糙度和复杂度
- **医学图像分析**：分析组织结构的复杂性
- **材料科学**：研究材料表面的分形特征
- **计算机视觉**：图像特征提取和分类
- **地质学**：地形和地貌复杂度分析

## 使用示例

### 基本用法

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
print("BC分形维度:", fd_bc['fd'])
print("DBC分形维度:", fd_dbc['fd'])
print("SDBC分形维度:", fd_sdbc['fd'])

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
import glob, cv2
from FreeAeonFractal.FAImageFD import CFAImageFD

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageFD.get_batch_bc(images)
for r in results:
    print("BC FD:", r['fd'])
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImageFD

**描述**：用于计算2D图像分形维度的类，支持Box-Counting（BC）、差分Box-Counting（DBC）和移位DBC（SDBC）三种计算方法。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入图像（单通道） |
| `max_size` | int | None | 最大box大小（默认：图像最小边长） |
| `max_scales` | int | 30 | 最大尺度数量 |
| `with_progress` | bool | True | 是否显示进度条 |
| `min_size` | int | 2 | 最小box大小 |

#### 主要方法

##### 1. get_bc_fd(corp_type=-1, fit_range=None)

**描述**：使用Box-Counting（BC）方法计算分形维度。

**参数**：
- `corp_type`（int）：图像裁剪方式
  - `-1`：自动裁剪为box大小的整数倍
  - `0`：不处理（要求图像尺寸已是box大小的整数倍）
  - `1`：填充
- `fit_range`（tuple或None）：可选的 `(min_scale, max_scale)` 限制拟合范围

**返回值**（dict）：
```python
{
    'fd': float,              # 分形维度值
    'scales': list,           # 尺度列表
    'counts': list,           # box计数列表
    'log_scales': list,       # 对数尺度
    'log_counts': list,       # 对数计数
    'intercept': float,       # 拟合截距
    'r_value': float,         # 相关系数
    'p_value': float,         # p值
    'std_err': float          # 标准误差
}
```

**适用场景**：适用于二值图像，计算占据空间的维度。

##### 2. get_dbc_fd(corp_type=-1, fit_range=None)

**描述**：使用差分Box-Counting（DBC）方法（Sarkar & Chaudhuri 1994）计算分形维度。使用 `n_r = ceil(I_max/h) - ceil(I_min/h) + 1`。

**参数**：同 `get_bc_fd`

**返回值**：同 `get_bc_fd`

**适用场景**：适用于灰度图像，在分形维度计算中考虑灰度信息。

##### 3. get_sdbc_fd(corp_type=-1, fit_range=None)

**描述**：使用移位DBC（SDBC）方法（Chen 1995）计算分形维度。使用 `n_r = floor((I_max - I_min)/h) + 1`。

**参数**：同 `get_bc_fd`

**返回值**：同 `get_bc_fd`

**适用场景**：SDBC是DBC的改进版本，使用更简化的计算方法，避免了原始DBC公式中的舍入误差。

##### 4. get_fd(scale_list, box_count_list)

**描述**：对自定义尺度和计数数据执行对数-对数拟合的工具方法。

**返回值**：与 `get_bc_fd` 相同的字典结构

##### 5. plot(raw_img, gray_img, bin_img, fd_bc, fd_dbc, fd_sdbc)

**描述**：静态方法，可视化原始图像、灰度图像、二值图像以及三种分形维度的拟合结果。

**参数**：
- `raw_img`：原始RGB图像
- `gray_img`：灰度图像
- `bin_img`：二值图像
- `fd_bc`：BC方法结果字典
- `fd_dbc`：DBC方法结果字典
- `fd_sdbc`：SDBC方法结果字典

##### 6. get_batch_bc / get_batch_dbc / get_batch_sdbc(images, ...)

**描述**：用于批量处理多个图像的静态方法。

**返回值**：结果字典列表

## 算法说明

### BC（Box-Counting）方法
通过统计在不同尺度下覆盖图像所需的非空box数量来计算分形维度：

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

其中 N(ε) 是尺度 ε 下的box数量。

### DBC（差分Box-Counting）方法
DBC考虑灰度高度信息，将图像视为3D曲面，每个box的高度为：

```
n_r = ceil(I_max / h) - ceil(I_min / h) + 1
```

其中 `h = max_val / (image_size / r)` 是灰度单元的高度。

### SDBC（移位DBC）方法
移位DBC使用简化的高度计算：

```
n_r = floor((I_max - I_min) / h) + 1
```

避免了原始DBC公式中的舍入误差。

## 注意事项

1. **图像预处理**：
   - BC方法需要二值图像，建议使用Otsu自动阈值化
   - DBC和SDBC方法适用于灰度图像

2. **尺度选择**：
   - `max_size` 影响分析的尺度范围
   - `max_scales` 影响采样密度，建议默认值 30
   - 使用 `fit_range` 排除不可靠的小/大尺度

3. **结果解释**：
   - 分形维度范围通常在1-2之间（2D图像）
   - 值越大表示图像越复杂
   - `r_value` 接近1表示幂律拟合良好

4. **性能优化**：
   - 大图像可先降采样
   - 设置 `with_progress=False` 提高计算速度
   - 批量处理使用GPU版本（`CFAImageFDGPU`）
   - 注意：GPU版本的 `p_value` 为 `None`（不计算）

## 参考文献

- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image. *IEEE Transactions on Systems, Man, and Cybernetics*.
- Chen, W. S., et al. (1995). Efficient fractal coding of images based on differential box counting. *Pattern Recognition*.
