# 空隙度分析 - CFAImageLacunarity

## 应用场景

`CFAImageLacunarity` 类用于计算2D图像的空隙度（Lacunarity），是量化图像空间分布不均匀性的重要工具。主要应用场景包括：

- **纹理分析**：量化纹理的空隙和间隙特征
- **景观生态学**：分析植被覆盖的空间分布
- **材料科学**：研究材料孔隙结构
- **医学图像**：分析组织分布的均匀性
- **城市规划**：研究建筑物空间分布

## 调用示例

### 基础用法

```python
import cv2
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity
from FreeAeonFractal.FAImage import CFAImage

# 读取图像
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 创建空隙度分析对象
lacunarity = CFAImageLacunarity(
    gray_image,
    max_scales=256,
    with_progress=True,
    partition_mode="gliding"  # 或 "non-overlapping"
)

# 计算空隙度
lac_result = lacunarity.get_lacunarity(
    corp_type=-1,
    use_binary_mass=False,
    include_zero=True
)

# 拟合空隙度
fit_result = lacunarity.fit_lacunarity(lac_result)

# 输出结果
print("空隙度值:", lac_result["lacunarity"])
print("拟合斜率:", fit_result["slope"])
print("拟合R²:", fit_result["r_value"]**2)

# 可视化
lacunarity.plot(lac_result, fit_result)
```

### 二值图像分析

```python
# 使用Otsu自动二值化
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# 二值图像的空隙度
lacunarity_bin = CFAImageLacunarity(bin_image, partition_mode="gliding")
lac_bin = lacunarity_bin.get_lacunarity(use_binary_mass=True)
fit_bin = lacunarity_bin.fit_lacunarity(lac_bin)

print("二值图像空隙度:", lac_bin["lacunarity"])
```

### 对比不同分区模式

```python
# 滑动窗口模式（重叠）
lac_gliding = CFAImageLacunarity(gray_image, partition_mode="gliding")
result_gliding = lac_gliding.get_lacunarity()

# 非重叠模式
lac_nonoverlap = CFAImageLacunarity(gray_image, partition_mode="non-overlapping")
result_nonoverlap = lac_nonoverlap.get_lacunarity()

# 可视化对比
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
lac_gliding.plot(result_gliding, ax=ax, show=False, label="Gliding")
lac_nonoverlap.plot(result_nonoverlap, ax=ax, show=False, label="Non-overlapping")
plt.legend()
plt.show()
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImageLacunarity

**描述**：用于计算2D图像空隙度的类，支持两种盒子分区策略。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必需 | 输入图像（灰度或二值） |
| `max_size` | int | None | 最大盒子尺寸（默认为图像最小边长） |
| `max_scales` | int | 100 | 最大尺度数量 |
| `with_progress` | bool | True | 是否显示进度条 |
| `scales_mode` | str | "powers" | 尺度生成模式（"powers"或"logspace"） |
| `partition_mode` | str | "gliding" | 分区模式（"gliding"或"non-overlapping"） |

**scales_mode** 选项：
- `"powers"`: 2的幂次尺度 (2, 4, 8, 16, ...)
- `"logspace"`: 对数均匀分布的尺度

**partition_mode** 选项：
- `"gliding"`: 滑动窗口（重叠），使用积分图高效计算
- `"non-overlapping"`: 非重叠固定块

#### 主要方法

##### 1. get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)

**描述**：计算每个尺度下的空隙度。

**参数**：
- `corp_type` (int): 图像裁剪方式（仅非重叠模式）
  - `-1`: 自动裁剪
  - `0`: 不处理
  - `1`: 填充
- `use_binary_mass` (bool): 是否使用二值质量
  - `True`: 只计数非零像素
  - `False`: 使用实际像素值
- `include_zero` (bool): 是否包含零质量盒子

**返回值** (dict):
```python
{
    'scales': list,        # 尺度列表
    'lacunarity': list,    # 各尺度的空隙度值
    'mass_stats': list     # 质量统计信息列表
}
```

`mass_stats` 中每个元素包含：
```python
{
    'scale': int,          # 尺度
    'num_boxes': int,      # 盒子数量
    'mean_mass': float,    # 平均质量
    'var_mass': float,     # 质量方差
    'lambda': float        # 空隙度值
}
```

##### 2. fit_lacunarity(lac_result, min_valid_lambda=1.0+1e-12)

**描述**：对空隙度进行幂律拟合：log(Λ-1) vs log(r)

**参数**：
- `lac_result` (dict): `get_lacunarity()` 的返回结果
- `min_valid_lambda` (float): 最小有效空隙度值

**返回值** (dict):
```python
{
    'slope': float,                    # 拟合斜率
    'intercept': float,                # 拟合截距
    'r_value': float,                  # 相关系数
    'p_value': float,                  # p值
    'std_err': float,                  # 标准误差
    'log_scales': list,                # 对数尺度
    'log_lambda_minus_1': list         # log(Λ-1)
}
```

##### 3. plot(lac_result, fit_result=None, ax=None, show=True, title="Lacunarity", label=None)

**描述**：可视化空隙度曲线和拟合结果。

**参数**：
- `lac_result` (dict): 空隙度结果
- `fit_result` (dict): 拟合结果（可选）
- `ax`: Matplotlib轴对象（可选）
- `show` (bool): 是否立即显示
- `title` (str): 图标题
- `label` (str): 曲线标签

**图表**：
- 左图：Λ(r) vs r
- 右图（如果提供fit_result）：log(Λ-1) vs log(r) 及拟合线

## 理论背景

### 空隙度定义

空隙度 Λ(r) 定义为盒子质量的二阶矩与一阶矩平方的比值：

```
Λ(r) = E[M²(r)] / E[M(r)]²
```

其中：
- M(r): 尺度为r的盒子质量
- E[·]: 期望值

### 物理意义

- **Λ = 1**: 完全均匀分布
- **Λ > 1**: 存在空隙，分布不均匀
- **Λ越大**: 空隙越多，聚集性越强

### 尺度律

空隙度通常遵循幂律：

```
Λ(r) - 1 ∝ r^β
```

其中β为空隙度指数。

### 两种分区策略

#### 1. 滑动窗口（Gliding Box）
- 盒子重叠
- 使用积分图（Summed-Area Table）高效计算
- 采样更密集，结果更平滑
- 计算复杂度：O(N·S)，N为图像像素数，S为尺度数

#### 2. 非重叠盒子（Non-overlapping Box）
- 盒子不重叠，平铺整个图像
- 采样较稀疏
- 计算更快
- 计算复杂度：O(N·S/r²)

## 重要提示

1. **分区模式选择**：
   - `"gliding"`: 推荐用于一般分析，结果更平滑
   - `"non-overlapping"`: 更快，适合大图像

2. **质量模式**：
   - `use_binary_mass=True`: 二值图像或只关心占据情况
   - `use_binary_mass=False`: 灰度图像，考虑强度信息

3. **尺度模式**：
   - `"powers"`: 较少的尺度点，计算快
   - `"logspace"`: 更多尺度点，结果更精细

4. **零值处理**：
   - `include_zero=True`: 包含所有盒子
   - `include_zero=False`: 只考虑非零盒子

5. **结果解释**：
   - 空隙度值范围：[1, +∞)
   - 斜率β：量化尺度依赖性
   - R²接近1：良好的幂律拟合

6. **性能优化**：
   - 使用 `partition_mode="non-overlapping"` 加速
   - 减少 `max_scales` 值
   - 使用 `scales_mode="powers"`

## 应用示例

### 景观分析

```python
# 植被覆盖图
vegetation_map = cv2.imread('vegetation.png', cv2.IMREAD_GRAYSCALE)
bin_veg, _ = CFAImage.otsu_binarize(vegetation_map)

# 空隙度分析
lac_calc = CFAImageLacunarity(bin_veg, partition_mode="gliding")
lac = lac_calc.get_lacunarity(use_binary_mass=True)
fit = lac_calc.fit_lacunarity(lac)

# 解释结果
beta = fit['slope']
if beta > 0:
    print(f"植被呈聚集分布，聚集指数: {beta:.3f}")
else:
    print(f"植被呈均匀分布")
```

### 材料孔隙分析

```python
# 材料断面图
material_image = cv2.imread('material_surface.png', cv2.IMREAD_GRAYSCALE)

# 灰度空隙度
lac_gray = CFAImageLacunarity(material_image, max_scales=128)
result = lac_gray.get_lacunarity(use_binary_mass=False)

# 提取统计特征
mean_lac = np.mean(result['lacunarity'])
max_lac = np.max(result['lacunarity'])

print(f"平均空隙度: {mean_lac:.3f}")
print(f"最大空隙度: {max_lac:.3f}")
```

## 常见问题

### Q: 空隙度值 < 1 或 NaN？
A: 检查图像是否为空、是否有负值，或调整 `include_zero` 参数。

### Q: 如何选择分区模式？
A: 一般用 "gliding"，大图像或追求速度用 "non-overlapping"。

### Q: 拟合效果不好？
A: 尝试调整 `scales_mode` 或 `max_scales`，或使用对数尺度。

### Q: 灰度图用哪个质量模式？
A: `use_binary_mass=False` 保留灰度信息，`True` 只看分布。

## 静态方法

### _safe_lacunarity_from_masses(masses, eps=1e-12)

**描述**：从盒子质量数组安全地计算空隙度。

**参数**：
- `masses` (array-like): 盒子质量数组
- `eps` (float): 数值稳定性阈值

**返回值**：空隙度值（float）或 NaN

## 参考文献

- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. Physical Review E.
- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. Physical Review A.
