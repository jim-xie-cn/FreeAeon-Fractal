# 可视化工具 - CFAVisual

## 应用场景

`CFAVisual` 类提供分形点集和图像的可视化工具。主要应用场景包括：

- **分形教学**：可视化1D/2D/3D分形点集
- **研究验证**：展示生成的分形以验证IFS参数
- **图像显示**：并排展示2D分析结果
- **3D点云**：可视化3D分形结构

## 使用示例

### 1D点集可视化（康托集）

```python
import matplotlib.pyplot as plt
from FreeAeonFractal.FAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample

points_1d = CFASample.get_Cantor_Set(iterations=256)

fig, ax = plt.subplots(figsize=(10, 2))
CFAVisual.plot_1d_points(points_1d, ax=ax)
ax.set_title("康托集")
plt.show()
```

### 2D点集可视化（谢尔宾斯基三角形）

```python
import matplotlib.pyplot as plt
from FreeAeonFractal.FAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample

points_2d = CFASample.get_Sierpinski_Triangle(iterations=1024)

fig, ax = plt.subplots(figsize=(6, 6))
CFAVisual.plot_2d_points(points_2d, ax=ax)
ax.set_title("谢尔宾斯基三角形")
plt.show()
```

### 3D点集可视化（门格海绵）

```python
import matplotlib.pyplot as plt
from FreeAeonFractal.FAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample

points_3d = CFASample.get_Menger_Sponge(iterations=10240)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(points_3d, ax=ax)
ax.set_title("门格海绵")
plt.show()
```

### 2D图像显示

```python
import cv2
from FreeAeonFractal.FAVisual import CFAVisual

image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)
CFAVisual.plot_2d_image(image, cmap='gray')
```

## 类说明

### CFAVisual

**描述**：静态工具类，用于显示分形点集（1D、2D、3D）和图像。所有方法接受可选的 `ax` 参数以支持子图集成。

#### 方法

##### plot_1d_points(points, ax=plt)

**描述**：以水平散点图显示1D点（Y轴隐藏）。

**适用场景**：康托集等1D分形结构。

##### plot_2d_points(points, ax=plt)

**描述**：以散点图显示2D点。

**参数**：`points` — (N, 2) 坐标数组

**适用场景**：谢尔宾斯基三角形、巴恩斯利蕨等2D IFS分形。

##### plot_3d_points(points, ax=None)

**描述**：以3D散点图显示3D点。

**参数**：`points` — (N, 3) 坐标数组；`ax=None` 时自动创建新图形

**适用场景**：门格海绵等3D分形。

##### plot_2d_image(image, cmap='gray', ax=plt)

**描述**：使用 `imshow` 显示2D图像。

##### plot_3d_image(img, ax=plt)

**描述**：从结构化 (N, 4) 数组显示3D点云，第4列为颜色值。

## 重要说明

1. **ax参数**：所有方法接受 `ax=plt`（直接使用pyplot）或显式坐标轴对象
2. **点密度**：点数超过100K时建议先降采样再显示
3. **3D投影**：`plot_3d_points` 需要 `projection='3d'` 坐标轴；`ax=None` 时自动创建
