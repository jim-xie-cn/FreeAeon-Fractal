# 空隙度分析 - CFAImageLAC

## 应用场景

`CFAImageLAC` 类用于量化2D图像的空间异质性和间隙结构。主要应用场景包括：

- **生态学**：量化栖息地破碎化和间隙分布
- **材料科学**：表征多孔结构和内部几何形态
- **医学图像**：分析组织均匀性和病变分布
- **城市规划**：研究土地利用的空间分布模式
- **地质学**：量化岩石裂隙和孔隙分布

## 使用示例

### 基础用法

```python
import cv2
from FreeAeonFractal.FAImageLAC import CFAImageLAC
from FreeAeonFractal.FAImage import CFAImage

# 读取图像
gray_image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)

# 二值化
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# 滑动盒空隙度分析（默认）
calc = CFAImageLAC(bin_image, partition_mode="gliding")
lac_result = calc.get_lacunarity(use_binary_mass=True, include_zero=True)
fit_result = calc.fit_lacunarity(lac_result)

print("Lambda(r):", lac_result['lacunarity'])
print("斜率 (beta):", fit_result['slope'])
print("R²:", fit_result['r_value']**2)

# 可视化
calc.plot(lac_result, fit_result)
```

### 非重叠模式

```python
calc_nonoverlap = CFAImageLAC(gray_image, partition_mode="non-overlapping")
lac_nonoverlap = calc_nonoverlap.get_lacunarity()
fit_nonoverlap = calc_nonoverlap.fit_lacunarity(lac_nonoverlap)
```

### GPU加速版本

```python
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU

calc_gpu = CFAImageLACGPU(bin_image, device='cuda')
lac_result = calc_gpu.get_lacunarity()
```

### 批量处理

```python
import cv2, glob
from FreeAeonFractal.FAImageLAC import CFAImageLAC

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]

batch_results = CFAImageLAC.get_batch_lacunarity(
    images,
    partition_mode="gliding",
    use_binary_mass=True,
    with_progress=True
)

batch_fits = CFAImageLAC.fit_batch_lacunarity(batch_results)
for fit in batch_fits:
    print("斜率:", fit['slope'])
```

## 类说明

### CFAImageLAC

**描述**：单张2D图像的空隙度计算类。支持滑动盒（重叠）和非重叠盒两种分区模式。批量处理请使用静态方法 `get_batch_lacunarity` / `fit_batch_lacunarity`。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入2D单通道图像 |
| `max_size` | int | None | 最大盒子尺寸（默认为图像最小维度） |
| `max_scales` | int | 100 | 目标尺度数量 |
| `with_progress` | bool | True | 是否显示进度条 |
| `scales_mode` | str | `"powers"` | 尺度生成方式：`"powers"`（2,4,8,...）或 `"logspace"`（几何间距） |
| `partition_mode` | str | `"gliding"` | 盒子策略：`"gliding"` 或 `"non-overlapping"` |
| `min_size` | int | 2 | 最小盒子尺寸 |

#### 主要方法

##### 1. get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)

**描述**：计算 `self.scales` 中每个盒子尺寸的 Λ(r)。

**参数**：
- `corp_type` (int)：裁剪策略（用于非重叠模式）
- `use_binary_mass` (bool)：将图像视为二值（任何正值像素→1）再计算质量
- `include_zero` (bool)：若为False，则排除质量为0的空盒子

**返回值** (dict)：
```python
{
    'scales': list,       # 使用的盒子尺寸
    'lacunarity': list,   # Lambda(r) 值
    'mass_stats': list    # 每个尺度：scale, num_boxes, mean_mass, var_mass, lambda
}
```

**空隙度公式**：`Λ(r) = E[M²] / E[M]² = 1 + Var(M) / Mean(M)²`

##### 2. fit_lacunarity(lac_result, transform="log", fit_range=None)

**描述**：对空隙度曲线进行对数-对数线性回归拟合。

**参数**：
- `transform` (str)：
  - `"log"`：拟合 log(Λ) vs log(r)，标准分形分析方式，斜率 = −β
  - `"log_minus_1"`：拟合 log(Λ−1) vs log(r)，忽略均匀区域（遗留模式）
- `fit_range` (tuple 或 None)：`(r_min, r_max)` 限制回归范围

**返回值** (dict)：slope, intercept, r_value, p_value, std_err等

##### 3. plot(lac_result, fit_result=None, ...)

**描述**：以双对数坐标可视化 Λ(r) 曲线及可选的线性拟合面板。

##### 4. get_batch_lacunarity(images, ...) [静态方法]

**描述**：批量空隙度计算。当所有图像形状相同且为滑动模式时，积分图像在批次上向量化处理，效率最高。

##### 5. fit_batch_lacunarity(lac_results, ...) [静态方法]

**描述**：对 `get_batch_lacunarity` 返回的每个结果应用相同的拟合。

## 算法说明

### 空隙度定义

```
Λ(r) = E[M²] / E[M]² = 1 + Var(M) / Mean(M)²
```

下限为1（零方差 = 均匀分布），Λ越大表示空间异质性越强。

### 滑动盒法

每个 (r×r) 窗口在图像上滑动。积分图像（累积面积表）在尺度循环**外**计算一次，通过切片算术在每个尺度以 O(H×W) 时间提取所有盒质量：

```
M(y, x) = S[y+r, x+r] - S[y, x+r] - S[y+r, x] + S[y, x]
```

### 拟合

对于自相似分形：`Λ(r) ~ r^{-β}`，β = D − E（分形维度减嵌入维度）。

标准拟合（`transform="log"`）对 `log Λ(r)` vs `log r` 进行回归，报告斜率 = −β。

## 重要说明

1. **分区模式**：`"gliding"` 统计上更稳健（每个尺度更多样本），通过积分图像技巧保持高效；`"non-overlapping"` 适合需要样本独立性的场景

2. **二值 vs 灰度空隙度**：`use_binary_mass=True` 用于经典二值空隙度（Allain & Cloitre 1991）；默认（False）使用原始灰度强度作为质量

3. **结果解读**：Λ=1表示完全均匀；Λ>1表示存在空隙和聚集；负斜率β表示尺度相关的间隙闭合

## 参考文献

- Allain, C., & Cloitre, M. (1991). *Physical Review A*.
- Plotnick, R. E., et al. (1996). *Physical Review E*.
