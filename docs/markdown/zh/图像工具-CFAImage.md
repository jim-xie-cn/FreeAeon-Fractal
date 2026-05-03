# 图像处理工具 - CFAImage

## 应用场景

`CFAImage` 类提供图像处理的基础工具函数，包括图像分块、合并、掩码生成和ROI提取等。主要应用场景包括：

- **图像分块处理**：将图像分割为固定大小的块进行分析
- **图像预处理**：自动二值化、裁剪、填充
- **ROI提取**：基于多重分形特性提取感兴趣区域
- **数据增强**：随机采样图像块用于训练
- **图像分析**：支持分形维度和多重分形谱计算

## 调用示例

### Otsu自动二值化

```python
import cv2
from FreeAeonFractal.FAImage import CFAImage

# 读取灰度图像
gray_image = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# Otsu自动阈值化
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

print(f"自动阈值: {threshold}")

# 可视化
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(gray_image, cmap='gray')
ax1.set_title('原始图像')
ax2.imshow(bin_image, cmap='gray')
ax2.set_title(f'二值图像 (threshold={threshold:.1f})')
plt.show()
```

### 图像分块与合并

```python
import numpy as np

# 创建测试图像
image = np.zeros((256, 256), dtype=np.uint8)
cv2.circle(image, (128, 128), 80, 255, -1)

# 分块
block_size = (64, 64)
boxes, raw_blocks = CFAImage.get_boxes_from_image(
    image,
    block_size,
    corp_type=-1  # 自动裁剪
)

print(f"总块数: {boxes.shape[0]}")
print(f"原始块形状: {raw_blocks.shape}")

# 合并回图像
merged_image = CFAImage.get_image_from_boxes(raw_blocks)

# 验证
assert np.array_equal(merged_image, image[:256, :256])
```

### 创建掩码

```python
# 创建掩码（屏蔽特定块）
mask_positions = [(0, 0), (1, 1), (2, 2)]  # 屏蔽对角线块
mask_image = CFAImage.get_mask_from_boxes(raw_blocks, mask_positions)

# 应用掩码
masked_image = (merged_image * mask_image).astype(np.uint8)
```

### ROI提取

```python
# 基于多重分形特性提取ROI
rgb_image = cv2.imread('./images/face.png')

mask_union, masked_image = CFAImage.get_roi_by_q(
    image=rgb_image,
    q_range=(-5, 5),
    step=1.0,
    box_size=16,
    target_mass=0.90,
    combine_mode="or",
    use_grayscale_measure=True
)

# 可视化
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
ax1.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
ax1.set_title('原始图像')
ax2.imshow(mask_union, cmap='gray')
ax2.set_title('ROI掩码')
ax3.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
ax3.set_title('提取的ROI')
plt.show()
```

### 随机采样图像块

```python
# 随机采样用于数据增强
image = cv2.imread('./images/face.png')

# 采样100个25%大小的块
patches = CFAImage.get_random_patches(
    image,
    num_patches=100,
    ratio=0.25
)

print(f"采样了 {len(patches)} 个块")
print(f"每个块的大小: {patches[0].shape}")

# 可视化部分块
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImage

**描述**：图像处理工具类，提供静态方法用于图像的裁剪、填充、分块、合并等操作。

#### 静态方法列表

所有方法都是静态方法，使用 `CFAImage.方法名()` 调用。

##### 1. otsu_binarize(img)

**描述**：使用Otsu方法自动阈值化图像。

**参数**：
- `img` (ndarray): 输入图像（灰度或彩色）

**返回值** (tuple):
```python
(bin_img, threshold)
```
- `bin_img`: 二值图像（uint8, 值为0或255）
- `threshold`: 自动计算的阈值

**特性**：
- 自动转换彩色图像为灰度
- 自动处理float和uint8类型
- Otsu方法最小化类内方差

##### 2. crop_data(data, block_size)

**描述**：裁剪图像使其尺寸为block_size的倍数。

**参数**：
- `data` (ndarray): 输入图像（2D或3D）
- `block_size` (tuple): (bh, bw) 块大小

**返回值**：裁剪后的图像

**示例**：
```python
# 图像 256x256，裁剪为 240x240 (60的倍数)
cropped = CFAImage.crop_data(image, (60, 60))
```

##### 3. pad_data(data, block_size, mode="constant", constant_values=0)

**描述**：填充图像使其尺寸为block_size的倍数。

**参数**：
- `data` (ndarray): 输入图像
- `block_size` (tuple): 块大小
- `mode` (str): 填充模式（'constant', 'edge', 'reflect'等）
- `constant_values`: 常数填充值（mode='constant'时）

**返回值**：填充后的图像

**填充模式**：
- `'constant'`: 常数填充
- `'edge'`: 边缘值填充
- `'reflect'`: 镜像填充

##### 4. get_boxes_from_image(image, block_size, corp_type=-1)

**描述**：将图像分割为固定大小的块。

**参数**：
- `image` (ndarray): 输入图像
- `block_size` (tuple): (bh, bw) 块大小
- `corp_type` (int): 裁剪类型
  - `-1`: 自动裁剪
  - `0`: 不处理（需尺寸已匹配）
  - `1`: 填充

**返回值** (tuple):
```python
(blocks_reshaped, raw_blocks)
```

- **灰度图像**：
  - `blocks_reshaped`: (num_blocks, bh, bw)
  - `raw_blocks`: (nY, nX, bh, bw)

- **彩色图像**：
  - `blocks_reshaped`: (num_blocks, bh, bw, c)
  - `raw_blocks`: (nY, nX, bh, bw, c)

##### 5. get_image_from_boxes(raw_blocks)

**描述**：将块重新合并为图像。

**参数**：
- `raw_blocks` (ndarray): 块数组（由get_boxes_from_image返回的raw_blocks）

**返回值**：合并的图像

**支持形状**：
- 灰度：(nY, nX, bh, bw) → (H, W)
- 彩色：(nY, nX, bh, bw, c) → (H, W, c)

##### 6. get_mask_from_boxes(raw_blocks, mask_block_pos)

**描述**：根据块位置列表生成二值掩码图像。

**参数**：
- `raw_blocks` (ndarray): 块数组
- `mask_block_pos` (list): 需要屏蔽的块位置列表 [(y1,x1), (y2,x2), ...]

**返回值**：掩码图像（float32, 值为0或1）
- 指定位置的块：0
- 其他位置：1

##### 7. get_random_patches(image, num_patches=100, ratio=0.25)

**描述**：从图像中随机采样矩形块（可能部分重叠）。

**参数**：
- `image` (ndarray): 输入图像
- `num_patches` (int): 采样块数量
- `ratio` (float): 块大小比例（相对于图像尺寸）

**返回值**：图像块列表

**特性**：
- 采样无重复（不同左上角坐标）
- 可能有部分重叠
- 适用于数据增强

**限制**：
```python
max_possible = (W - patch_w) * (H - patch_h)
num_patches <= max_possible
```

##### 8. get_roi_by_q(image, q_range=(-5,5), step=1.0, box_size=16, target_mass=0.95, combine_mode="and", use_grayscale_measure=True, measure_mode=0)

**描述**：基于多重分形特性提取感兴趣区域（ROI）。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | ndarray | 必需 | 输入图像 |
| `q_range` | tuple | (-5, 5) | q值范围 |
| `step` | float | 1.0 | q值步长 |
| `box_size` | int | 16 | 盒子大小 |
| `target_mass` | float | 0.95 | 目标累积质量 |
| `combine_mode` | str | "and" | 通道合并模式（"and"或"or"） |
| `use_grayscale_measure` | bool | True | 是否用灰度计算质量 |
| `measure_mode` | int | 0 | 质量计算模式 |

**返回值** (tuple):
```python
(mask_union, masked_image)
```
- `mask_union`: 布尔掩码 (H, W)
- `masked_image`: 应用掩码后的图像

**工作原理**：
1. 将图像分块
2. 计算每个块的质量 μᵢ
3. 对于每个q值，计算权重 wᵢ = μᵢ^q
4. 选择累积权重达到target_mass的块
5. 合并所有q值的掩码（OR操作）
6. 多通道图像按combine_mode合并

**应用**：
- 提取图像的关键区域
- 基于强度分布的分割
- 多尺度特征提取

## 理论背景

### Otsu阈值法

Otsu方法通过最小化类内方差自动选择阈值：

```
σ²ω(t) = ω₀(t)σ²₀(t) + ω₁(t)σ²₁(t)
```

其中：
- ω₀, ω₁: 两类的权重
- σ²₀, σ²₁: 两类的方差
- t: 阈值

最优阈值：
```
t* = argmin σ²ω(t)
```

### ROI提取的多重分形基础

基于盒计数的多重分形分析：

1. **质量分布**：
```
μᵢ = mᵢ / Σmⱼ
```

2. **q阶权重**：
```
wᵢ(q) = μᵢ^q
```

3. **归一化概率**：
```
pᵢ(q) = wᵢ(q) / Σwⱼ(q)
```

4. **选择策略**：
   - q < 0: 强调低质量区域
   - q = 0: 均匀权重
   - q > 0: 强调高质量区域

## 重要提示

1. **图像类型**：
   - 支持灰度（2D）和彩色（3D）
   - 自动处理不同数据类型

2. **分块操作**：
   - 使用 `view_as_blocks` 避免复制
   - 支持灰度和彩色图像
   - 通道维度保持不变

3. **corp_type选择**：
   - `-1`: 适用于大多数情况，损失边界
   - `0`: 严格模式，需预先调整尺寸
   - `1`: 填充模式，保留所有像素

4. **ROI提取参数**：
   - `q_range`: 影响提取的区域类型
   - `box_size`: 影响空间分辨率
   - `target_mass`: 控制区域大小

5. **性能**：
   - 分块操作使用视图，内存高效
   - ROI提取计算密集，大图像较慢

## 应用示例

### 多尺度纹理分析

```python
# 提取不同尺度的纹理块
image = cv2.imread('texture.png', cv2.IMREAD_GRAYSCALE)

scales = [16, 32, 64]
for scale in scales:
    boxes, _ = CFAImage.get_boxes_from_image(image, (scale, scale))
    # 计算每个块的纹理特征
    features = [compute_texture_feature(box) for box in boxes]
```

### 自适应ROI分割

```python
# 基于多重分形特性的自适应分割
image = cv2.imread('image.png')

# 提取高强度区域 (q > 0)
_, roi_high = CFAImage.get_roi_by_q(
    image, q_range=(1, 5), target_mass=0.5
)

# 提取低强度区域 (q < 0)
_, roi_low = CFAImage.get_roi_by_q(
    image, q_range=(-5, -1), target_mass=0.5
)
```

## 常见问题

### Q: 为什么分块后无法完全恢复原图？
A: 使用 `corp_type=-1` 会裁剪边界，导致尺寸变化。使用 `corp_type=1` 填充可保留尺寸。

### Q: get_random_patches 报错？
A: 检查 `num_patches` 是否超过最大可能数量 `(W-patch_w) * (H-patch_h)`。

### Q: ROI提取结果为空？
A: 检查图像是否有有效内容，尝试调整 `q_range` 和 `target_mass`。

### Q: Otsu阈值不理想？
A: Otsu适用于双峰直方图，对于复杂图像可能需要其他方法。

## 依赖

```python
import numpy as np
import cv2
from skimage.util import view_as_blocks
```

## 参考文献

- Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics.
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
