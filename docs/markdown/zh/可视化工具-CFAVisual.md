# 可视化工具 - CFAVisual

## 应用场景

`CFAVisual` 类提供分形点集和图像的可视化工具，用于展示分形数据的空间分布。主要应用场景包括：

- **分形点集展示**：可视化1D、2D、3D分形点集
- **分形图案验证**：验证生成的分形图案是否正确
- **教学演示**：展示分形几何的基本概念
- **数据可视化**：展示点云和图像数据

## 调用示例

### 1D点集可视化

```python
import numpy as np
from FreeAeonFractal.CFAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample
import matplotlib.pyplot as plt

# 生成康托集
points_1d = CFASample.get_Cantor_Set(iterations=1000)

# 可视化1D点集
plt.figure(figsize=(10, 2))
CFAVisual.plot_1d_points(points_1d)
plt.title('Cantor Set (1D)')
plt.show()
```

### 2D点集可视化

```python
# 生成谢尔宾斯基三角形
points_2d = CFASample.get_Sierpinski_Triangle(iterations=10000)

# 可视化2D点集
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_points(points_2d)
plt.title('Sierpinski Triangle (2D)')
plt.axis('equal')
plt.show()
```

### 3D点集可视化

```python
from mpl_toolkits.mplot3d import Axes3D

# 生成门格海绵
points_3d = CFASample.get_Menger_Sponge(iterations=50000)

# 可视化3D点集
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(points_3d, ax=ax)
ax.set_title('Menger Sponge (3D)')
plt.show()
```

### 图像可视化

```python
import cv2

# 读取图像
image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)

# 可视化图像
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_image(image, cmap='gray')
plt.title('Fractal Image')
plt.show()
```

### 综合示例

```python
# 综合展示多个分形图案
fig = plt.figure(figsize=(12, 8))

# 1D: 康托集
ax1 = fig.add_subplot(221)
points_1d = CFASample.get_Cantor_Set(iterations=1000)
CFAVisual.plot_1d_points(points_1d, ax1)
ax1.set_title('Cantor Set')

# 2D: 谢尔宾斯基三角形
ax2 = fig.add_subplot(222)
points_2d = CFASample.get_Sierpinski_Triangle(iterations=10000)
CFAVisual.plot_2d_points(points_2d, ax2)
ax2.set_title('Sierpinski Triangle')

# 2D: 巴恩斯利蕨
ax3 = fig.add_subplot(223)
fern = CFASample.get_Barnsley_Fern(iterations=20000)
CFAVisual.plot_2d_points(fern, ax3)
ax3.set_title('Barnsley Fern')

# 3D: 门格海绵
ax4 = fig.add_subplot(224, projection='3d')
sponge = CFASample.get_Menger_Sponge(iterations=50000)
CFAVisual.plot_3d_points(sponge, ax4)
ax4.set_title('Menger Sponge')

plt.tight_layout()
plt.show()
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAVisual

**描述**：提供分形点集和图像可视化的静态方法工具类。

所有方法都是静态方法，使用 `CFAVisual.方法名()` 调用。

#### 静态方法

##### 1. plot_1d_points(points, ax=plt)

**描述**：显示1D点集（如康托集）。

**参数**：
- `points` (numpy.ndarray): 1D点数组，形状为 (N,)
- `ax` (matplotlib axis): Matplotlib轴对象，默认为plt

**显示效果**：
- 点绘制在y=0的水平线上
- y轴隐藏
- 点的大小固定（s=1）

**示例**：
```python
points = np.array([0.1, 0.3, 0.7, 0.9])
CFAVisual.plot_1d_points(points)
plt.show()
```

##### 2. plot_2d_points(points, ax=plt)

**描述**：显示2D点集（如谢尔宾斯基三角形、巴恩斯利蕨）。

**参数**：
- `points` (numpy.ndarray): 2D点数组，形状为 (N, 2)
  - points[:, 0]: x坐标
  - points[:, 1]: y坐标
- `ax` (matplotlib axis): Matplotlib轴对象，默认为plt

**显示效果**：
- 红色点标记
- 点大小：s=1
- 散点图显示

**示例**：
```python
points = np.random.rand(1000, 2)
CFAVisual.plot_2d_points(points)
plt.axis('equal')
plt.show()
```

##### 3. plot_3d_points(points, ax=None)

**描述**：显示3D点集（如门格海绵）。

**参数**：
- `points` (numpy.ndarray): 3D点数组，形状为 (N, 3)
  - points[:, 0]: x坐标
  - points[:, 1]: y坐标
  - points[:, 2]: z坐标
- `ax` (matplotlib 3D axis): 3D轴对象。如果为None，自动创建

**显示效果**：
- 红色点标记
- 点大小：s=1
- 3D散点图

**注意**：需要导入 `from mpl_toolkits.mplot3d import Axes3D`

**示例**：
```python
from mpl_toolkits.mplot3d import Axes3D

points = np.random.rand(1000, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(points, ax)
plt.show()
```

##### 4. plot_2d_image(image, cmap='gray', ax=plt)

**描述**：显示2D图像。

**参数**：
- `image` (numpy.ndarray): 2D图像数组
- `cmap` (str): 色图，默认为'gray'
- `ax` (matplotlib axis): Matplotlib轴对象，默认为plt

**示例**：
```python
import cv2

image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
CFAVisual.plot_2d_image(image, cmap='viridis')
plt.show()
```

##### 5. plot_3d_image(img, ax=plt)

**描述**：显示3D图像（点云形式）。

**参数**：
- `img` (numpy.ndarray): 3D或4D点数组
  - 形状为 (N, 3): 只有坐标
  - 形状为 (N, 4): 坐标+值
- `ax` (matplotlib axis): Matplotlib轴对象

**显示效果**：
- 使用viridis色图
- 点大小根据值缩放
- 3D散点图

**示例**：
```python
# 带值的3D点云
points = np.random.rand(1000, 4)
CFAVisual.plot_3d_image(points)
plt.show()
```

## 与FASample配合使用

`CFAVisual` 通常与 `CFASample` 类配合使用，用于可视化生成的分形图案。

```python
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.CFAVisual import CFAVisual

# 生成并可视化康托集
cantor = CFASample.get_Cantor_Set(iterations=2000)
CFAVisual.plot_1d_points(cantor)
plt.title('Cantor Set (D ≈ 0.63)')
plt.show()

# 生成并可视化谢尔宾斯基三角形
triangle = CFASample.get_Sierpinski_Triangle(iterations=10000)
CFAVisual.plot_2d_points(triangle)
plt.title('Sierpinski Triangle (D ≈ 1.58)')
plt.axis('equal')
plt.show()

# 生成并可视化巴恩斯利蕨
fern = CFASample.get_Barnsley_Fern(iterations=50000)
CFAVisual.plot_2d_points(fern)
plt.title('Barnsley Fern (D ≈ 1.67)')
plt.axis('equal')
plt.show()
```

## 重要提示

1. **坐标轴**：
   - 1D点集：y轴自动隐藏
   - 2D点集：建议使用 `plt.axis('equal')` 保持比例
   - 3D点集：需要创建3D轴

2. **点的大小**：
   - 默认 s=1，适合大量点
   - 可以通过修改源码调整大小

3. **颜色**：
   - 1D/2D/3D点集默认红色
   - 图像可通过cmap参数调整

4. **性能**：
   - 大量点（>100万）可能较慢
   - 建议适当减少点数

## 依赖

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 3D可视化需要
```

## 参考文献

- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
