# 分形样本生成 - CFASample

## 应用场景

`CFASample` 类基于迭代函数系统（IFS）生成经典分形图案。主要应用场景包括：

- **算法测试**：生成已知维度的标准分形
- **教学**：演示分形几何概念
- **基准测试**：在已知分形结构上比较分析方法
- **艺术创作**：生成分形艺术作品
- **数据增强**：创建合成分形纹理

## 使用示例

### 生成经典分形

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.FAVisual import CFAVisual

# 1D: 康托集（维度 ≈ 0.63）
points_1d = CFASample.get_Cantor_Set(iterations=256)

# 2D: 谢尔宾斯基三角形（维度 ≈ 1.58）
points_2d = CFASample.get_Sierpinski_Triangle(iterations=1024)

# 2D: 巴恩斯利蕨（维度 ≈ 1.67）
points_fern = CFASample.get_Barnsley_Fern(iterations=4096)

# 3D: 门格海绵（维度 ≈ 2.73）
points_3d = CFASample.get_Menger_Sponge(iterations=10240)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(141); CFAVisual.plot_1d_points(points_1d, ax1); ax1.set_title("康托集")
ax2 = fig.add_subplot(142); CFAVisual.plot_2d_points(points_2d, ax2); ax2.set_title("谢尔宾斯基三角形")
ax3 = fig.add_subplot(143); CFAVisual.plot_2d_points(points_fern, ax3); ax3.set_title("巴恩斯利蕨")
ax4 = fig.add_subplot(144, projection='3d'); CFAVisual.plot_3d_points(points_3d, ax4); ax4.set_title("门格海绵")
plt.tight_layout()
plt.show()
```

### 点集转图像

```python
from FreeAeonFractal.FASample import CFASample

points = CFASample.get_Sierpinski_Triangle(iterations=4096)
image = CFASample.get_image_from_points(points, img_size=(512, 512))

# 用于分形分析
from FreeAeonFractal.FAImageFD import CFAImageFD
fd_bc = CFAImageFD(image).get_bc_fd()
print("谢尔宾斯基FD (BC):", fd_bc['fd'])  # 应约为1.58
```

### 自定义IFS

```python
import numpy as np
from FreeAeonFractal.FASample import CFASample

trans_matrix = np.array([
    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
    [[0.5, 0.0, 1.0], [0.0, 0.5, 0.0]],
    [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
])
trans_probability = np.array([0.33, 0.33, 0.34])
init_point = np.array([0.0, 0.0])

points = CFASample.generate(init_point, iterations=2048, trans_matrix=trans_matrix, trans_probability=trans_probability)
```

## 类说明

### CFASample

**描述**：IFS（迭代函数系统）分形生成器。所有生成方法使用随机迭代仿射变换。

#### 方法

##### generate(init_point, iterations, trans_matrix, trans_probability) [静态方法]

**描述**：核心IFS生成器，迭代应用随机选择的仿射变换。

**参数**：
- `init_point`：在相应维度中的起始点
- `iterations`：IFS迭代次数
- `trans_matrix`：仿射变换矩阵数组，形状 `(n_transforms, ndim, ndim+1)`
- `trans_probability`：选择每个变换的概率（必须求和为1）

**返回值**：`(iterations, ndim)` 生成点数组

##### get_Cantor_Set(init_point=np.array([0.0]), iterations=256) [静态方法]

生成1D康托集点。**理论维度**：≈ 0.6309 (log 2 / log 3)

##### get_Sierpinski_Triangle(init_point=np.array([0.0, 0.0]), iterations=256) [静态方法]

生成2D谢尔宾斯基三角形点。**理论维度**：≈ 1.585 (log 3 / log 2)

##### get_Barnsley_Fern(init_point=np.array([0.0, 0.0]), iterations=4096)

生成2D巴恩斯利蕨点。**理论维度**：≈ 1.67

##### get_Menger_Sponge(init_point=np.array([0.0, 0.0, 0.0]), iterations=10240) [静态方法]

生成3D门格海绵点（20个压缩映射）。**理论维度**：≈ 2.727 (log 20 / log 3)

##### get_image_from_points(points, img_size=(512, 512), margin=0.05) [静态方法]

将2D IFS点云转换为二值uint8图像。

**参数**：
- `points`：(N, 2) 点数组
- `img_size`：输出图像尺寸 `(H, W)`
- `margin`：点边界周围的边距比例

**返回值**：(H, W) uint8图像，占用像素值为255

## 分形维度参考

| 分形 | 维度 | 方法 |
|------|------|------|
| 康托集 | ≈ 0.63 | 1D盒计数 |
| 谢尔宾斯基三角形 | ≈ 1.58 | 2D盒计数 |
| 巴恩斯利蕨 | ≈ 1.67 | 2D盒计数 |
| 门格海绵 | ≈ 2.73 | 3D盒计数 |

## 重要说明

1. **迭代次数**：迭代次数越多，分形近似越密集、越精确；分析时推荐1000+次
2. **图像尺寸**：分形维度分析时，256×256或更大图像能提供更可靠的尺度范围
3. **IFS收敛**：无论起始点如何，IFS混沌游戏都会收敛；如需精确结果可丢弃前几百个点

## 参考文献

- Barnsley, M. F. (1988). *Fractals Everywhere*. Academic Press.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.
