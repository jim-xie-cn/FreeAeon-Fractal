# 空隙度分析 - CFAImageLAC

## 应用场景

`CFAImageLAC` 类用于计算2D图像的空隙度，是量化空间分布异质性的重要工具。主要应用场景包括：

- **纹理分析**：量化纹理中的间隙和空洞特征
- **景观生态学**：分析植被覆盖的空间分布
- **材料科学**：研究材料的孔隙结构
- **医学图像**：分析组织分布均匀性
- **城市规划**：研究建筑物的空间分布

## 使用示例

### 基本用法

```python
import cv2
from FreeAeonFractal.FAImageLAC import CFAImageLAC
from FreeAeonFractal.FAImage import CFAImage

# 读取图像
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 创建空隙度分析对象
lacunarity = CFAImageLAC(
    gray_image,
    max_scales=100,
    with_progress=True,
    partition_mode="gliding"  # 或 "non-overlapping"
)

# 计算空隙度
lac_result = lacunarity.get_lacunarity(
    corp_type=-1,
    use_binary_mass=False,
    include_zero=True
)

# 拟合空隙度曲线
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
lacunarity_bin = CFAImageLAC(bin_image, partition_mode="gliding")
lac_bin = lacunarity_bin.get_lacunarity(use_binary_mass=True)
fit_bin = lacunarity_bin.fit_lacunarity(lac_bin)

print("二值图像空隙度:", lac_bin["lacunarity"])
```

### 批量处理

```python
import glob, cv2
from FreeAeonFractal.FAImageLAC import CFAImageLAC

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]

# 批量空隙度（CPU支持不同形状图像）
lac_results = CFAImageLAC.get_batch_lacunarity(images, max_scales=100)
fit_results = CFAImageLAC.fit_batch_lacunarity(lac_results)

for fit in fit_results:
    print("斜率:", fit["slope"], "R²:", fit["r_value"]**2)
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImageLAC

**描述**：用于计算2D图像空隙度的类，支持两种box分区策略。滑动box模式使用积分图像（累积面积表）进行高效计算。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入图像（灰度或二值） |
| `max_size` | int | None | 最大box大小（默认：图像最小边长） |
| `max_scales` | int | 100 | 最大尺度数量 |
| `with_progress` | bool | True | 是否显示进度条 |
| `scales_mode` | str | "powers" | 尺度生成模式（"powers"或"logspace"） |
| `partition_mode` | str | "gliding" | 分区模式（"gliding"或"non-overlapping"） |
| `min_size` | int | 2 | 最小box大小 |

**scales_mode** 选项：
- `"powers"`：2的幂次尺度（2, 4, 8, 16, ...）
- `"logspace"`：对数均匀分布尺度

**partition_mode** 选项：
- `"gliding"`：滑动窗口（重叠），使用积分图像高效计算
- `"non-overlapping"`：非重叠固定块

#### 主要方法

##### 1. get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)

**描述**：计算每个尺度下的空隙度。

**参数**：
- `corp_type`（int）：图像裁剪方式（仅非重叠模式）
  - `-1`：自动裁剪
  - `0`：不处理
  - `1`：填充
- `use_binary_mass`（bool）：是否使用二值质量
  - `True`：仅计数非零像素
  - `False`：使用实际像素值
- `include_zero`（bool）：是否包含零质量box

**返回值**（dict）：
```python
{
    'scales': list,        # 尺度列表
    'lacunarity': list,    # 各尺度下的空隙度值
    'mass_stats': list     # 质量统计信息列表
}
```

`mass_stats` 中每个元素包含：
```python
{
    'scale': int,          # 尺度
    'num_boxes': int,      # box数量
    'mean_mass': float,    # 平均质量
    'var_mass': float,     # 质量方差
    'lambda': float        # 空隙度值
}
```

##### 2. fit_lacunarity(lac_result, transform="log", fit_range=None)

**描述**：对空隙度进行幂律拟合。

**参数**：
- `lac_result`（dict）：`get_lacunarity()` 的返回结果
- `transform`（str）：变换模式
  - `"log"`：标准对数-对数拟合，斜率 = -β（Allain & Cloitre）
  - `"log_minus_1"`：拟合 log(Λ-1) vs log(r)
- `fit_range`（tuple或None）：可选的 `(min_scale, max_scale)` 限制拟合范围

**返回值**（dict）：
```python
{
    'slope': float,          # 拟合斜率
    'intercept': float,      # 拟合截距
    'r_value': float,        # 相关系数
    'p_value': float,        # p值
    'std_err': float,        # 标准误差
    'log_scales': list,      # 使用的对数尺度
    'log_lac': list          # 使用的对数空隙度值
}
```

##### 3. get_batch_lacunarity(images, ...) [静态方法]

**描述**：对多个图像进行批量空隙度计算。CPU版本支持不同形状的图像。

##### 4. fit_batch_lacunarity(lac_results, ...) [静态方法]

**描述**：批量拟合空隙度结果。

##### 5. plot(lac_result, fit_result=None, ax=None, show=True, title="Lacunarity", label=None)

**描述**：可视化空隙度曲线和拟合结果。

**参数**：
- `lac_result`（dict）：空隙度结果
- `fit_result`（dict）：拟合结果（可选）
- `ax`：Matplotlib轴对象（可选）
- `show`（bool）：是否立即显示
- `title`（str）：图表标题
- `label`（str）：曲线标签

## 理论背景

### 空隙度定义

空隙度 Λ(r) 定义为box质量的二阶矩与一阶矩平方之比：

```
Λ(r) = E[M²(r)] / E[M(r)]²
```

其中：
- M(r)：尺度r下的box质量
- E[·]：期望值

### 物理含义

- **Λ = 1**：完全均匀分布
- **Λ > 1**：存在间隙，分布不均匀
- **Λ越大**：间隙越多，聚集越强

### 标度律

空隙度通常遵循幂律：

```
Λ(r) ∝ r^(-β)     （对数变换，Allain & Cloitre）
```

其中 β 是空隙度指数。

## 注意事项

1. **分区模式选择**：
   - `"gliding"`：推荐用于一般分析，结果更平滑，使用积分图像实现O(1)逐像素计算
   - `"non-overlapping"`：速度更快，适合大图像

2. **质量模式**：
   - `use_binary_mass=True`：二值图像或只关心占用情况
   - `use_binary_mass=False`：灰度图像，考虑强度信息

3. **尺度模式**：
   - `"powers"`：尺度点更少，计算更快
   - `"logspace"`：尺度点更多，结果更精细

4. **零值处理**：
   - `include_zero=True`：包含所有box
   - `include_zero=False`：仅考虑非零box

5. **结果解释**：
   - 空隙度值范围：[1, +∞)
   - 斜率β：量化尺度依赖性
   - R² 接近1：幂律拟合良好

6. **性能优化**：
   - 使用 `partition_mode="non-overlapping"` 加速
   - 减少 `max_scales` 值
   - 使用 `scales_mode="powers"`
   - GPU版本（`CFAImageLACGPU`）适合大规模批处理（要求图像形状相同）

## 参考文献

- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. *Physical Review E*.
- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. *Physical Review A*.
