# 分形维度分析 - CFAImageDimension

## 应用场景

`CFAImageDimension` 类用于计算2D图像的分形维度，是图像复杂度分析和纹理特征提取的重要工具。主要应用场景包括：

- **图像纹理分析**：量化图像的粗糙度和复杂度
- **医学图像分析**：分析组织结构的复杂性
- **材料科学**：研究材料表面的分形特征
- **计算机视觉**：图像特征提取和分类
- **地质学**：地形地貌的复杂度分析

## 调用示例

### 基础用法

```python
import cv2
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImage import CFAImage

# 读取图像
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 对于BC方法，需要二值化图像
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# 计算三种分形维度
fd_bc = CFAImageDimension(bin_image).get_bc_fd(corp_type=-1)
fd_dbc = CFAImageDimension(gray_image).get_dbc_fd(corp_type=-1)
fd_sdbc = CFAImageDimension(gray_image).get_sdbc_fd(corp_type=-1)

# 输出结果
print("BC分形维度:", fd_bc['fd'])
print("DBC分形维度:", fd_dbc['fd'])
print("SDBC分形维度:", fd_sdbc['fd'])

# 可视化结果
CFAImageDimension.plot(rgb_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImageDimension

**描述**：用于计算2D图像分形维度的类，支持三种不同的计算方法。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必需 | 输入图像（单通道） |
| `max_size` | int | None | 最大盒子尺寸（默认为图像最小边长） |
| `max_scales` | int | 100 | 最大尺度数量 |
| `with_progress` | bool | True | 是否显示进度条 |

#### 主要方法

##### 1. get_bc_fd(corp_type=-1)

**描述**：使用盒计数（Box-Counting, BC）方法计算分形维度。

**参数**：
- `corp_type` (int): 图像裁剪方法
  - `-1`: 自动裁剪到盒子尺寸的倍数
  - `0`: 不处理（需要图像尺寸已是盒子尺寸的倍数）
  - `1`: 填充

**返回值** (dict):
```python
{
    'fd': float,              # 分形维度值
    'scales': list,           # 尺度列表
    'counts': list,           # 盒子计数列表
    'log_scales': list,       # 对数尺度
    'log_counts': list,       # 对数计数
    'intercept': float,       # 拟合截距
    'r_value': float,         # 相关系数
    'p_value': float,         # p值
    'std_err': float          # 标准误差
}
```

**适用场景**：适用于二值图像，计算其占据空间的维度。

##### 2. get_dbc_fd(corp_type=-1)

**描述**：使用差分盒计数（Differential Box-Counting, DBC）方法计算分形维度。

**参数**：同 `get_bc_fd`

**返回值**：同 `get_bc_fd`

**适用场景**：适用于灰度图像，考虑灰度信息的分形维度计算。

##### 3. get_sdbc_fd(corp_type=-1)

**描述**：使用改进的差分盒计数（Simplified DBC, SDBC）方法计算分形维度。

**参数**：同 `get_bc_fd`

**返回值**：同 `get_bc_fd`

**适用场景**：SDBC是DBC的简化版本，计算更快，适用于灰度图像。

##### 4. plot(raw_img, gray_img, fd_bc, fd_dbc, fd_sdbc)

**描述**：静态方法，可视化原始图像、二值图像和三种分形维度的拟合结果。

**参数**：
- `raw_img`: 原始RGB图像
- `gray_img`: 二值化或灰度图像
- `fd_bc`: BC方法的结果字典
- `fd_dbc`: DBC方法的结果字典
- `fd_sdbc`: SDBC方法的结果字典

## 算法说明

### BC (Box-Counting) 方法
盒计数法通过在不同尺度下统计覆盖图像所需的非空盒子数量来计算分形维度。计算公式为：

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

其中 N(ε) 是尺度为 ε 时的盒子数量。

### DBC (Differential Box-Counting) 方法
差分盒计数法考虑灰度值的高度信息，将图像视为三维表面。每个盒子的高度由该区域的灰度范围决定。

### SDBC (Simplified DBC) 方法
简化差分盒计数法是DBC的改进版本，使用更简化的计算方式，在保持精度的同时提高计算效率。

## 重要提示

1. **图像预处理**：
   - BC方法需要二值图像，建议使用Otsu自动阈值化
   - DBC和SDBC方法适用于灰度图像

2. **尺度选择**：
   - `max_size` 影响分析的尺度范围
   - `max_scales` 影响采样密度，建议使用默认值100

3. **结果解释**：
   - 分形维度范围通常在1-2之间（2D图像）
   - 值越大表示图像越复杂
   - `r_value` 接近1表示良好的幂律拟合

4. **性能优化**：
   - 对于大图像，可以先降采样
   - 设置 `with_progress=False` 可以提高计算速度

## 参考文献

- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image.
