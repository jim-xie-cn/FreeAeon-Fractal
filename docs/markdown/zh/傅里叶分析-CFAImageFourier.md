# 傅里叶分析 - CFAImageFourier

## 应用场景

`CFAImageFourier` 类提供图像的傅里叶频谱分析工具，用于分析图像的频率成分和进行频域处理。主要应用场景包括：

- **频域分析**：分析图像的频率分布特征
- **图像滤波**：通过频域掩码进行高通、低通、带通滤波
- **纹理分析**：提取图像的周期性和方向性特征
- **图像增强**：频域图像增强和复原
- **模式识别**：基于频谱特征的图像分类

## 调用示例

### 基础用法

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

# 读取图像（支持灰度或RGB）
rgb_image = cv2.imread('./images/face.png')

# 创建傅里叶分析对象
fourier = CFAImageFourier(rgb_image)

# 获取原始频谱
raw_mag, raw_phase = fourier.get_raw_spectrum()

# 获取可视化频谱
raw_mag_disp, raw_phase_disp = fourier.get_display_spectrum(alpha=1.5)

# 重构图像
reconstructed = fourier.get_reconstruct()

# 显示结果
fourier.plot(raw_mag_disp, raw_phase_disp,
             [], [], reconstructed, np.array([]))
```

### 频域滤波示例

```python
# 读取灰度图像
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
fourier = CFAImageFourier(gray_image)

# 获取原始频谱
raw_mag, raw_phase = fourier.get_raw_spectrum()

# 创建频率掩码（保留奇数频率）
h, w = raw_mag[0].shape
Y, X = np.ogrid[:h, :w]
mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

# 应用掩码到频谱
masked_mag = raw_mag * mask
masked_phase = raw_phase * mask

# 获取掩码后的可视化频谱
customized_mag_disp, customized_phase_disp = fourier.get_display_spectrum(
    alpha=1.5,
    magnitude=masked_mag,
    phase=masked_phase
)

# 重构滤波后的图像
masked_reconstructed = fourier.extract_by_freq_mask(mask)

# 可视化对比
fourier.plot(raw_mag_disp, raw_phase_disp,
             customized_mag_disp, customized_phase_disp,
             reconstructed, masked_reconstructed)
```

### 低通滤波器

```python
# 创建低通滤波器掩码（保留低频成分）
h, w = raw_mag[0].shape
center_y, center_x = h // 2, w // 2
radius = 30  # 截止半径

Y, X = np.ogrid[:h, :w]
distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
low_pass_mask = (distance <= radius).astype(np.uint8)

# 应用低通滤波
low_pass_result = fourier.extract_by_freq_mask(low_pass_mask)
```

### 高通滤波器

```python
# 创建高通滤波器掩码（去除低频成分）
high_pass_mask = (distance > radius).astype(np.uint8)

# 应用高通滤波
high_pass_result = fourier.extract_by_freq_mask(high_pass_mask)
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFAImageFourier

**描述**：提供图像傅里叶分析工具，支持灰度和RGB图像的频域分析和重构。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必需 | 输入图像（灰度或RGB） |

**注意**：初始化时自动计算傅里叶变换，无需手动调用。

#### 主要方法

##### 1. get_raw_spectrum()

**描述**：获取原始的幅度和相位频谱（用于重构）。

**参数**：无

**返回值** (tuple):
```python
(magnitude_list, phase_list)
```
- `magnitude_list`: 幅度频谱列表（每个通道一个）
- `phase_list`: 相位频谱列表（每个通道一个）

对于灰度图像，列表长度为1；RGB图像长度为3。

##### 2. get_display_spectrum(alpha=1.0, beta=0, magnitude=np.array([]), phase=np.array([]))

**描述**：生成增强的可视化频谱图像。

**参数**：
- `alpha` (float): 对比度增强因子（默认1.0）
- `beta` (float): 亮度偏移（默认0）
- `magnitude` (array): 自定义幅度频谱（可选）
- `phase` (array): 自定义相位频谱（可选）

**返回值** (tuple):
```python
(display_mag_list, display_phase_list)
```
返回归一化到0-255的8位图像列表，适合可视化。

**增强说明**：
- 幅度采用对数变换：log(1 + mag)
- 相位归一化到 [0, 1]

##### 3. get_reconstruct(magnitude=np.array([]), phase=np.array([]))

**描述**：从频域重构空间域图像。

**参数**：
- `magnitude` (array): 幅度频谱（可选，默认使用原始）
- `phase` (array): 相位频谱（可选，默认使用原始）

**返回值**：
- 灰度图像：(H, W) uint8数组
- RGB图像：(H, W, 3) uint8数组

##### 4. extract_by_freq_mask(mask_mag=np.array([]), mask_phase=np.array([]))

**描述**：使用二值掩码选择性保留频率成分，然后重构图像。

**参数**：
- `mask_mag` (array): 幅度掩码（与频谱同形状，1保留/0去除）
- `mask_phase` (array): 相位掩码（可选）

**返回值**：重构的滤波后图像

**应用**：
- 高通/低通滤波
- 方向性滤波
- 周期性噪声去除

##### 5. plot(raw_magnitude_disp, raw_phase_disp, customized_magnitude_disp, customized_phase_disp, full_reconstructed, mask_reconstructed)

**描述**：可视化分析结果，显示最多7个子图。

**参数**：
- `raw_magnitude_disp` (list): 原始幅度可视化
- `raw_phase_disp` (list): 原始相位可视化
- `customized_magnitude_disp` (list): 自定义幅度可视化
- `customized_phase_disp` (list): 自定义相位可视化
- `full_reconstructed` (array): 完整重构图像
- `mask_reconstructed` (array): 掩码重构图像

**子图布局**：
1. 原始图像
2. 原始幅度谱
3. 原始相位谱
4. 自定义幅度谱
5. 自定义相位谱
6. 完整重构
7. 掩码重构

## 静态方法

### get_image_components(image)

**描述**：计算单通道图像的幅度和相位。

**参数**：
- `image` (ndarray): 单通道图像

**返回值** (tuple):
```python
(magnitude, phase)
```

### normalize_and_enhance(array, alpha=1.0, beta=0)

**描述**：归一化数组到0-255并应用线性增强。

**参数**：
- `array` (ndarray): 输入数据
- `alpha` (float): 对比度缩放因子
- `beta` (float): 亮度偏移

**返回值**：uint8数组

## 理论背景

### 2D傅里叶变换

对于图像 I(x, y)，其2D傅里叶变换为：

```
F(u, v) = ∫∫ I(x, y) exp[-j2π(ux + vy)] dx dy
```

离散形式：

```
F(u, v) = Σₓ Σᵧ I(x, y) exp[-j2π(ux/M + vy/N)]
```

### 频谱组成

**幅度谱**：
```
|F(u, v)| = √[R²(u, v) + I²(u, v)]
```

**相位谱**：
```
φ(u, v) = arctan[I(u, v) / R(u, v)]
```

### 频移

使用 `np.fft.fftshift` 将零频率分量移到中心，便于可视化。

### 逆变换

```
I(x, y) = ∫∫ F(u, v) exp[j2π(ux + vy)] du dv
```

## 应用案例

### 案例1：周期性噪声去除

```python
# 读取带周期性噪声的图像
noisy_image = cv2.imread('noisy_image.png', cv2.IMREAD_GRAYSCALE)
fourier = CFAImageFourier(noisy_image)

# 获取频谱
mag, phase = fourier.get_raw_spectrum()

# 可视化找到噪声峰值位置
mag_disp, _ = fourier.get_display_spectrum(alpha=2.0)
# (手动识别噪声峰值的坐标)

# 创建带阻滤波器，去除特定频率
h, w = mag[0].shape
mask = np.ones((h, w), dtype=np.uint8)
# 假设噪声在 (y1, x1) 和 (y2, x2)
mask[y1-5:y1+5, x1-5:x1+5] = 0
mask[y2-5:y2+5, x2-5:x2+5] = 0

# 重构去噪图像
denoised = fourier.extract_by_freq_mask(mask)
```

### 案例2：边缘检测（高通滤波）

```python
# 高通滤波突出边缘
gray_image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
fourier = CFAImageFourier(gray_image)

# 创建高通滤波器
h, w = gray_image.shape
Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
center = (h // 2, w // 2)
dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)

# 保留高频（远离中心）
cutoff = 30
high_pass = (dist > cutoff).astype(np.uint8)

# 提取边缘
edges = fourier.extract_by_freq_mask(high_pass)
```

### 案例3：方向性滤波

```python
# 提取特定方向的特征
gray_image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
fourier = CFAImageFourier(gray_image)

h, w = gray_image.shape
Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
center = (h // 2, w // 2)

# 计算角度
angle = np.arctan2(Y - center[0], X - center[1])

# 保留水平方向（0° 和 180°）
angle_deg = np.degrees(angle)
horizontal_mask = ((np.abs(angle_deg) < 15) | (np.abs(angle_deg - 180) < 15)).astype(np.uint8)

# 提取水平纹理
horizontal_texture = fourier.extract_by_freq_mask(horizontal_mask)
```

## 重要提示

1. **输入图像**：
   - 支持灰度和RGB图像
   - 自动进行傅里叶变换

2. **频谱可视化**：
   - 使用对数变换增强可视化效果
   - `alpha` 参数控制对比度

3. **掩码设计**：
   - 掩码大小必须与频谱一致
   - 二值掩码：1保留，0去除
   - 对称掩码保持实值图像

4. **RGB处理**：
   - 每个通道独立处理
   - 最终合并重构

5. **性能考虑**：
   - FFT复杂度：O(N log N)
   - 大图像可能较慢
   - 使用灰度图像可加速

## 常见问题

### Q: 重构图像与原图不完全一致？
A: 浮点运算精度和归一化可能引入微小差异，但视觉上应一致。

### Q: 相位重要吗？
A: 相位包含图像结构信息，通常比幅度更重要。仅改变幅度影响较小。

### Q: 如何设计特定滤波器？
A: 了解频谱布局：中心=低频，边缘=高频；根据需求设计掩码形状。

### Q: 为什么用频移？
A: 将低频移到中心便于可视化和设计滤波器。

## 参考文献

- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.).
- Bracewell, R. N. (2000). The Fourier Transform and Its Applications (3rd ed.).
