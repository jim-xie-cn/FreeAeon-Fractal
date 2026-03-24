# 分形样本生成 - CFASample

## 应用场景

`CFASample` 类用于生成经典的分形图案，基于迭代函数系统（IFS, Iterated Function System）。主要应用场景包括：

- **分形图案生成**：生成经典分形几何图案
- **教学演示**：展示分形几何的基本原理
- **测试数据**：为分形分析算法提供测试数据
- **艺术创作**：生成美观的分形艺术图案
- **算法验证**：验证分形维度计算的正确性

## 调用示例

### 康托集 (Cantor Set)

```python
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.CFAVisual import CFAVisual
import matplotlib.pyplot as plt

# 生成康托集（分形维度 ≈ 0.63）
cantor = CFASample.get_Cantor_Set(
    init_point=np.array([0.0]),
    iterations=2000
)

# 可视化
plt.figure(figsize=(12, 2))
CFAVisual.plot_1d_points(cantor)
plt.title('Cantor Set (Dimension ≈ 0.63)')
plt.show()
```

### 谢尔宾斯基三角形 (Sierpinski Triangle)

```python
# 生成谢尔宾斯基三角形（分形维度 ≈ 1.58）
triangle = CFASample.get_Sierpinski_Triangle(
    init_point=np.array([0.0, 0.0]),
    iterations=10000
)

# 可视化
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_points(triangle)
plt.title('Sierpinski Triangle (Dimension ≈ 1.58)')
plt.axis('equal')
plt.show()
```

### 巴恩斯利蕨 (Barnsley Fern)

```python
# 生成巴恩斯利蕨（分形维度 ≈ 1.67）
fern = CFASample.get_Barnsley_Fern(
    init_point=np.array([0.0, 0.0]),
    iterations=50000
)

# 可视化
plt.figure(figsize=(6, 10))
CFAVisual.plot_2d_points(fern)
plt.title('Barnsley Fern (Dimension ≈ 1.67)')
plt.axis('equal')
plt.show()
```

### 门格海绵 (Menger Sponge)

```python
from mpl_toolkits.mplot3d import Axes3D

# 生成门格海绵（3D分形）
sponge = CFASample.get_Menger_Sponge(
    init_point=np.array([0.0, 0.0, 0.0]),
    iterations=100000
)

# 可视化
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(sponge, ax)
ax.set_title('Menger Sponge (3D Fractal)')
plt.show()
```

### 点集转换为图像

```python
# 生成谢尔宾斯基三角形
points = CFASample.get_Sierpinski_Triangle(iterations=50000)

# 转换为图像
image = CFASample.get_image_from_points(
    points,
    img_size=(512, 512),
    margin=0.05
)

# 可视化
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_image(image, cmap='binary')
plt.title('Sierpinski Triangle as Image')
plt.show()
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFASample

**描述**：分形样本生成器，使用迭代函数系统（IFS）生成经典分形图案。

所有方法都是静态方法，使用 `CFASample.方法名()` 调用。

#### 静态方法

##### 1. get_Cantor_Set(init_point=np.array([0.0]), iterations=256)

**描述**：生成康托集（1D分形）。

**参数**：
- `init_point` (numpy.ndarray): 初始点，形状为 (1,)，默认 [0.0]
- `iterations` (int): 迭代次数，默认256

**返回值**：numpy.ndarray，形状为 (iterations, 1)

**分形维度**：约 0.6309

**特性**：
- 经典的1D分形集合
- 自相似结构
- 处处不连续

##### 2. get_Sierpinski_Triangle(init_point=np.array([0.0, 0.0]), iterations=256)

**描述**：生成谢尔宾斯基三角形（2D分形）。

**参数**：
- `init_point` (numpy.ndarray): 初始点，形状为 (2,)，默认 [0.0, 0.0]
- `iterations` (int): 迭代次数，默认256

**返回值**：numpy.ndarray，形状为 (iterations, 2)

**分形维度**：约 1.58

**特性**：
- 经典的2D分形图案
- 三角形自相似结构
- 迭代越多，图案越精细

**建议迭代次数**：
- 预览：1,000-5,000
- 详细：10,000-50,000
- 高精度：100,000+

##### 3. get_Barnsley_Fern(init_point=np.array([0.0, 0.0]), iterations=4096)

**描述**：生成巴恩斯利蕨（2D分形，模拟蕨类植物）。

**参数**：
- `init_point` (numpy.ndarray): 初始点，形状为 (2,)
- `iterations` (int): 迭代次数，默认4096

**返回值**：numpy.ndarray，形状为 (iterations, 2)

**分形维度**：约 1.67

**特性**：
- 模拟自然界蕨类植物
- 四个仿射变换
- 概率选择变换

**建议迭代次数**：
- 预览：5,000-10,000
- 详细：50,000-100,000
- 高精度：200,000+

##### 4. get_Menger_Sponge(init_point=np.array([0.0, 0.0, 0.0]), iterations=10240)

**描述**：生成门格海绵（3D分形）。

**参数**：
- `init_point` (numpy.ndarray): 初始点，形状为 (3,)
- `iterations` (int): 迭代次数，默认10240

**返回值**：numpy.ndarray，形状为 (iterations, 3)

**分形维度**：约 2.727

**特性**：
- 3D自相似结构
- 20个仿射变换
- 计算密集

**建议迭代次数**：
- 预览：10,000-50,000
- 详细：100,000-500,000
- 注意：迭代次数越多，计算时间越长

##### 5. get_4D_Points(init_point=np.array([0.0, 0.0, 0.0, 0.0]), iterations=4096)

**描述**：生成4D分形点集（用于测试和研究）。

**参数**：
- `init_point` (numpy.ndarray): 初始点，形状为 (4,)
- `iterations` (int): 迭代次数，默认4096

**返回值**：numpy.ndarray，形状为 (iterations, 4)

**用途**：高维分形研究和测试

##### 6. get_image_from_points(points, img_size=(512, 512), margin=0.05)

**描述**：将2D点集转换为图像。

**参数**：
- `points` (numpy.ndarray): 2D点数组，形状为 (N, 2)
- `img_size` (tuple): 输出图像尺寸 (height, width)
- `margin` (float): 边缘留白比例，默认0.05 (5%)

**返回值**：numpy.ndarray，uint8类型的二值图像
- 点的位置：255（白色）
- 背景：0（黑色）

**工作原理**：
1. 计算点的范围并添加边缘留白
2. 归一化点坐标到 [0, 1]
3. 映射到图像像素坐标
4. 在对应位置设置像素值

**示例**：
```python
# 生成分形点集
points = CFASample.get_Sierpinski_Triangle(iterations=50000)

# 转换为高分辨率图像
image = CFASample.get_image_from_points(
    points,
    img_size=(1024, 1024),
    margin=0.1
)

# 保存或分析
cv2.imwrite('sierpinski.png', image)

# 计算分形维度
from FreeAeonFractal.FAImageDimension import CFAImageDimension
fd = CFAImageDimension(image).get_bc_fd()
print(f"计算得到的分形维度: {fd['fd']:.4f}")
print(f"理论分形维度: 1.58")
```

##### 7. generate(init_point, iterations, trans_matrix, trans_probability)

**描述**：通用的IFS生成器（底层方法）。

**参数**：
- `init_point` (numpy.ndarray): 初始点
- `iterations` (int): 迭代次数
- `trans_matrix` (numpy.ndarray): 变换矩阵列表
- `trans_probability` (numpy.ndarray): 选择每个变换的概率

**返回值**：生成的点集

**用途**：创建自定义分形图案

**示例**：
```python
# 自定义简单的2D分形
custom_matrix = np.array([
    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],      # 缩小到左下
    [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]       # 缩小到右上
])
custom_prob = np.array([0.5, 0.5])

points = CFASample.generate(
    init_point=np.array([0.0, 0.0]),
    iterations=10000,
    trans_matrix=custom_matrix,
    trans_probability=custom_prob
)

CFAVisual.plot_2d_points(points)
plt.show()
```

## 理论背景

### 迭代函数系统（IFS）

IFS由一组仿射变换和对应的概率组成：

```
{(T₁, p₁), (T₂, p₂), ..., (Tₙ, pₙ)}
```

其中：
- Tᵢ: 仿射变换矩阵
- pᵢ: 选择该变换的概率
- Σpᵢ = 1

### 生成过程

1. 从初始点 x₀ 开始
2. 在每次迭代中：
   - 根据概率 pᵢ 随机选择一个变换 Tᵢ
   - 应用变换：xₙ₊₁ = Tᵢ(xₙ)
   - 记录新点
3. 重复指定次数

### 仿射变换

2D仿射变换形式：

```
[x']   [a  b] [x]   [e]
[y'] = [c  d] [y] + [f]
```

矩阵表示：
```
[x']   [a  b  e] [x]
[y'] = [c  d  f] [y]
[1 ]   [0  0  1] [1]
```

## 经典分形图案说明

### 1. 康托集 (Cantor Set)
- **维度**: log(2)/log(3) ≈ 0.6309
- **变换数**: 2个
- **特点**: 最简单的分形集合，处处不连续
- **应用**: 测试分形维度算法

### 2. 谢尔宾斯基三角形 (Sierpinski Triangle)
- **维度**: log(3)/log(2) ≈ 1.585
- **变换数**: 3个
- **特点**: 自相似三角形，边界无限长
- **应用**: 2D分形的经典示例

### 3. 巴恩斯利蕨 (Barnsley Fern)
- **维度**: 约 1.67
- **变换数**: 4个
- **特点**: 模拟自然蕨类植物
- **应用**: 展示分形在自然界的应用

### 4. 门格海绵 (Menger Sponge)
- **维度**: log(20)/log(3) ≈ 2.727
- **变换数**: 20个
- **特点**: 3D自相似结构，表面积无限
- **应用**: 3D分形示例

## 应用示例

### 验证分形维度算法

```python
import numpy as np
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.FAImageDimension import CFAImageDimension

# 生成谢尔宾斯基三角形
points = CFASample.get_Sierpinski_Triangle(iterations=100000)

# 转换为图像
image = CFASample.get_image_from_points(points, img_size=(1024, 1024))

# 计算分形维度
fd = CFAImageDimension(image).get_bc_fd()

print(f"计算得到的分形维度: {fd['fd']:.4f}")
print(f"理论分形维度: 1.585")
print(f"相对误差: {abs(fd['fd'] - 1.585) / 1.585 * 100:.2f}%")
```

### 批量生成测试图像

```python
import cv2

# 生成多个分形图案
fractals = {
    'sierpinski': CFASample.get_Sierpinski_Triangle(iterations=50000),
    'fern': CFASample.get_Barnsley_Fern(iterations=50000)
}

# 转换为图像并保存
for name, points in fractals.items():
    image = CFASample.get_image_from_points(points, img_size=(512, 512))
    cv2.imwrite(f'{name}.png', image)
    print(f"Saved {name}.png")
```

### 多重分形谱分析

```python
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# 生成分形图像
points = CFASample.get_Barnsley_Fern(iterations=100000)
image = CFASample.get_image_from_points(points, img_size=(512, 512))

# 多重分形谱分析
mfs = CFA2DMFS(image, q_list=np.linspace(-5, 5, 21))
df_mass, df_fit, df_spec = mfs.get_mfs()

# 可视化
mfs.plot(df_mass, df_fit, df_spec)
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFASample

**描述**：分形样本生成器，基于迭代函数系统（IFS）生成经典分形图案。

#### 静态方法列表

##### 主要生成方法

| 方法 | 维度 | 分形维度 | 变换数 | 默认迭代 |
|------|------|----------|--------|----------|
| `get_Cantor_Set()` | 1D | 0.63 | 2 | 256 |
| `get_Sierpinski_Triangle()` | 2D | 1.58 | 3 | 256 |
| `get_Barnsley_Fern()` | 2D | 1.67 | 4 | 4096 |
| `get_Menger_Sponge()` | 3D | 2.73 | 20 | 10240 |
| `get_4D_Points()` | 4D | - | 4 | 4096 |

##### 工具方法

- `generate()`: 通用IFS生成器
- `get_image_from_points()`: 点集转图像

## IFS变换矩阵

### 康托集变换
```python
g_m_Cantor_Set = np.array([
    [[1/3, 0]],           # 缩小到左1/3
    [[1/3, 2/3]]          # 缩小到右1/3
])
g_p_Cantor_Set = np.array([0.5, 0.5])
```

### 谢尔宾斯基三角形变换
三个变换分别将三角形缩小到左下、右下和顶部。

### 巴恩斯利蕨变换
四个变换模拟蕨类植物的茎、叶和分支结构。

## 重要提示

1. **迭代次数**：
   - 越多越精细，但计算时间线性增长
   - 1D: 1,000-10,000
   - 2D: 10,000-100,000
   - 3D: 50,000-500,000

2. **初始点**：
   - 通常使用原点
   - 不同初始点收敛到相同吸引子

3. **点集转图像**：
   - `img_size` 影响图像分辨率
   - `margin` 影响边缘留白
   - 点密度影响图像质量

4. **性能**：
   - 迭代计算较快（向量化）
   - 3D可视化可能较慢
   - 大迭代次数需要较多内存

5. **应用**：
   - 生成测试数据验证算法
   - 教学演示分形概念
   - 艺术创作

## 自定义分形

### 创建自定义IFS

```python
# 定义自己的分形变换
my_transforms = np.array([
    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],  # 缩小到左下
    [[0.5, 0.0, 0.5], [0.0, 0.5, 0.0]],  # 缩小到右下
    [[0.5, 0.0, 0.25], [0.0, 0.5, 0.5]]  # 缩小到上方中间
])
my_probabilities = np.array([0.33, 0.33, 0.34])

# 生成自定义分形
custom_fractal = CFASample.generate(
    init_point=np.array([0.0, 0.0]),
    iterations=20000,
    trans_matrix=my_transforms,
    trans_probability=my_probabilities
)

# 可视化
CFAVisual.plot_2d_points(custom_fractal)
plt.title('Custom Fractal')
plt.axis('equal')
plt.show()
```

## 常见问题

### Q: 为什么图案不清晰？
A: 增加迭代次数。2D分形建议至少10,000次，3D分形建议50,000次以上。

### Q: 如何计算生成图案的分形维度？
A: 使用 `get_image_from_points()` 转换为图像，然后用 `CFAImageDimension` 计算。

### Q: 3D可视化很慢？
A: 减少点数或使用更简单的标记样式。

### Q: 如何创建自己的分形？
A: 使用 `generate()` 方法，定义自己的变换矩阵和概率。

### Q: 点集能保存吗？
A: 可以用 `np.save()` 保存，`np.load()` 加载。

## 参考文献

- Barnsley, M. F. (1993). Fractals Everywhere (2nd ed.).
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
- Hutchinson, J. E. (1981). Fractals and self-similarity. Indiana University Mathematics Journal.
