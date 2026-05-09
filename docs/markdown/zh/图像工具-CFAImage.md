# 图像处理工具 - CFAImage

## 应用场景

`CFAImage` 类提供分形分析流水线所需的基础图像处理工具。主要应用场景包括：

- **预处理**：分形分析前的二值化和裁剪
- **分块操作**：将图像分割为多重分形计算所需的盒子
- **掩码生成**：基于块位置创建空间掩码
- **ROI提取**：基于多重分形度量提取感兴趣区域
- **数据增强**：深度学习工作流的随机块采样

## 使用示例

### Otsu二值化

```python
import cv2
from FreeAeonFractal.FAImage import CFAImage

gray = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# 自动Otsu阈值化
bin_image, threshold = CFAImage.otsu_binarize(gray)
print(f"阈值: {threshold}")
```

### 图像分块和合并

```python
import numpy as np
from FreeAeonFractal.FAImage import CFAImage

image = np.zeros((256, 256), dtype=np.uint8)

# 分割为64×64块
blocks, raw_blocks = CFAImage.get_boxes_from_image(image, block_size=(64, 64), corp_type=-1)
print("块数量:", blocks.shape[0])  # 16块（4×4网格）

# 合并回图像
merged = CFAImage.get_image_from_boxes(raw_blocks)
```

### 从块位置生成掩码

```python
# 将特定块置零
mask_pos = [(0, 0), (1, 1), (2, 2)]   # 块网格坐标（行, 列）
mask_image = CFAImage.get_mask_from_boxes(raw_blocks, mask_pos)
masked_image = (merged * mask_image).astype(np.uint8)
```

### 裁剪和填充

```python
# 裁剪为块尺寸的整数倍
cropped = CFAImage.crop_data(image, block_size=(64, 64))

# 填充为块尺寸的整数倍
padded = CFAImage.pad_data(image, block_size=(64, 64), mode="constant", constant_values=0)
```

### 随机块采样

```python
# 从图像中采样100个随机64×64块
patches = CFAImage.get_random_patches(image, num_patches=100, ratio=0.25)
```

### 基于多重分形q值的ROI提取

```python
import cv2
from FreeAeonFractal.FAImage import CFAImage

image = cv2.imread('./images/face.png')

# 基于多重分形度量提取ROI
mask_union, masked_image = CFAImage.get_roi_by_q(
    image=image,
    q_range=(-5, 5),
    step=1.0,
    box_size=16,
    target_mass=0.90,
    combine_mode="or",
    use_grayscale_measure=True
)
```

## 类说明

### CFAImage

**描述**：静态工具类，提供图像分块操作、二值化、掩码生成和基于多重分形的ROI提取功能。

#### 裁剪和填充

##### crop_data(data, block_size)

将空间维度裁剪为块尺寸的整数倍。

##### pad_data(data, block_size, mode="constant", constant_values=0)

将空间维度填充为块尺寸的整数倍。

#### 二值化

##### otsu_binarize(img)

**描述**：对图像应用Otsu自动阈值化生成二值图像。

**返回值**：`(bin_img, threshold)`
- `bin_img`：像素值为 {0, 255} 的二值uint8图像
- `threshold`：计算出的Otsu阈值

**说明**：彩色图像先转换为灰度；浮点图像自动缩放到 [0, 255]。

#### 分块操作

##### get_boxes_from_image(image, block_size, corp_type=-1)

**描述**：将图像分割为 (bh, bw) 大小的块，通道维度保持不变。

**参数**：
- `image`：2D灰度(H,W)或3D彩色(H,W,C)图像
- `block_size`：`(bh, bw)`
- `corp_type`：`-1` 裁剪（默认），`1` 填充，`0` 严格（不整除则报错）

**返回值**：`(blocks_reshaped, raw_blocks)`
- `blocks_reshaped`：灰度为 `(num_blocks, bh, bw)`；彩色为 `(num_blocks, bh, bw, C)`
- `raw_blocks`：网格布局 `(nY, nX, bh, bw)` 或 `(nY, nX, bh, bw, C)`

##### get_image_from_boxes(raw_blocks)

**描述**：将 raw_blocks 合并回图像。

##### get_mask_from_boxes(raw_blocks, mask_block_pos)

**描述**：构建二值掩码图像，`mask_block_pos` 中的块置0，其余置1。

**参数**：
- `mask_block_pos`：块网格坐标 `(y, x)` 的列表

**返回值**：`(H, W)` float32掩码，值为 {0, 1}

#### 随机采样

##### get_random_patches(image, num_patches=100, ratio=0.25)

**描述**：从图像中随机采样不重复的矩形块。

**参数**：
- `num_patches`：采样块数量
- `ratio`：块尺寸占图像维度的比例（如0.25 → H/4 × W/4）

**返回值**：块数组列表

#### ROI提取

##### get_roi_by_q(image, q_range=(-5,5), step=1.0, box_size=16, target_mass=0.95, combine_mode="and", use_grayscale_measure=True, measure_mode="intensity_sum")

**描述**：通过选择加权质量排名最高的块来提取ROI（多重分形重新加权）。

**参数**：
- `q_range`：矩阶数范围 `(q_min, q_max)`
- `step`：q迭代步长
- `box_size`：块尺寸（像素）
- `target_mass`：保留的累积质量分数（如0.90 = 保留前90%的加权质量）
- `combine_mode`：跨通道掩码组合方式：`"and"` 或 `"or"`
- `use_grayscale_measure`：彩色图像用灰度转换计算度量

**返回值**：`(mask_union, masked_image)`

## 重要说明

1. **corp_type**：`-1`（裁剪）最常用；`1`（填充）保留所有像素但添加零值
2. **ROI提取**：大正q值选择高强度/密集区域；大负q值选择低强度/稀疏区域
3. **块坐标系**：`raw_blocks[y, x, ...]` 对应像素 `[y*bh:(y+1)*bh, x*bw:(x+1)*bw]`
