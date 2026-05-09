# 傅里叶分析 - CFAImageFourier

## 应用场景

`CFAImageFourier` 类提供图像傅里叶频谱分析工具。主要应用场景包括：

- **周期噪声去除**：识别并滤除周期性干扰模式
- **纹理分析**：从频域提取方向性和周期性特征
- **图像增强**：频域锐化和平滑
- **模式识别**：基于频率的纹理分类
- **信号处理**：频率分量提取和重构

## 使用示例

### 基础用法

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

# 读取图像（支持灰度和RGB）
image = cv2.imread('./images/face.png')

# 创建傅里叶分析对象（__init__中完成FFT计算）
fourier = CFAImageFourier(image)

# 获取原始幅度和相位（用于重构）
raw_mag, raw_phase = fourier.get_raw_spectrum()

# 获取增强的频谱可视化
mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)

# 从完整频谱重构图像
full_reconstructed = fourier.get_reconstruct()

# 可视化
fourier.plot(
    raw_magnitude_disp=mag_disp,
    raw_phase_disp=phase_disp,
    full_reconstructed=full_reconstructed
)
```

### 自定义频率掩码滤波

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

image = cv2.imread('./images/face.png')
fourier = CFAImageFourier(image)

raw_mag, raw_phase = fourier.get_raw_spectrum()
mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)

# 创建自定义频率掩码（示例：保留奇数频率分量）
h, w = raw_mag[0].shape
Y, X = np.ogrid[:h, :w]
mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

# 应用掩码并可视化定制频谱
customized_mag = raw_mag * mask
customized_phase = raw_phase * mask
custom_mag_disp, custom_phase_disp = fourier.get_display_spectrum(
    alpha=1.5, magnitude=customized_mag, phase=customized_phase
)

# 从掩码重构图像
masked_reconstructed = fourier.extract_by_freq_mask(mask)

fourier.plot(
    raw_magnitude_disp=mag_disp,
    raw_phase_disp=phase_disp,
    customized_magnitude_disp=custom_mag_disp,
    customized_phase_disp=custom_phase_disp,
    full_reconstructed=fourier.get_reconstruct(),
    mask_reconstructed=masked_reconstructed
)
```

### 低通滤波（平滑/模糊）

```python
h, w = raw_mag[0].shape
cy, cx = h // 2, w // 2
radius = 30   # 保留距中心此距离内的频率
Y, X = np.ogrid[:h, :w]
mask = ((X - cx)**2 + (Y - cy)**2 <= radius**2).astype(np.uint8)
result = fourier.extract_by_freq_mask(mask)
```

## 类说明

### CFAImageFourier

**描述**：提供灰度或RGB图像的傅里叶分析功能。初始化时完成2D FFT计算。支持幅度/相位分解、增强可视化、图像重构和自定义频率掩码滤波。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | numpy.ndarray | 必填 | 输入灰度(H,W)或RGB(H,W,3)图像 |

**说明**：FFT在 `__init__` 中完成。灰度图像存储一对幅度/相位，RGB图像存储三对（每通道一对）。

#### 主要方法

##### 1. get_raw_spectrum()

**描述**：获取原始幅度和相位数据（用于重构）。

**返回值**：`(magnitude_list, phase_list)` — 原始（未对数缩放）数组

**注意**：仅将原始频谱用于重构；可视化频谱不应用于重构。

##### 2. get_display_spectrum(alpha=1.0, beta=0, magnitude=array([]), phase=array([]))

**描述**：生成增强的幅度和相位可视化图像。

**参数**：
- `alpha`：对比度缩放因子
- `beta`：亮度偏移
- `magnitude`：可选自定义幅度（默认使用存储的原始频谱）
- `phase`：可选自定义相位

**返回值**：`(display_mag_list, display_phase_list)` — 对数缩放后的8位可视化图像

##### 3. get_reconstruct(magnitude=array([]), phase=array([]))

**描述**：从幅度和相位重构空间域图像。

**算法**：`IFFT(IFFTSHIFT(mag * exp(1j * phase)))` 后归一化到 [0, 255]。

**返回值**：重构的uint8图像（灰度或BGR）

##### 4. extract_by_freq_mask(mask_mag=array([]), mask_phase=array([]))

**描述**：对频域应用二值掩码后重构。

**参数**：
- `mask_mag`：幅度掩码（1=保留，0=置零），形状与频率图相同
- `mask_phase`：相位掩码

**适用场景**：频域滤波——传入高通/低通/带通掩码选择性保留频率分量。

##### 5. plot(raw_magnitude_disp=[], raw_phase_disp=[], customized_magnitude_disp=[], customized_phase_disp=[], full_reconstructed=array([]), mask_reconstructed=array([]))

**描述**：并排展示原始图像、幅度、相位及重构图像（2行网格布局）。

##### 6. get_image_components(image) [静态方法]

**描述**：使用2D FFT计算单通道图像的幅度和相位。

##### 7. normalize_and_enhance(array, alpha=1.0, beta=0) [静态方法]

**描述**：将数组归一化到 [0, 255] 并应用线性对比度增强。

## 算法说明

### 2D FFT流程

```
1. FFT2:    F = np.fft.fft2(image)
2. 频移:    F_shift = np.fft.fftshift(F)   （DC分量移到中心）
3. 分解:    magnitude = |F_shift|, phase = angle(F_shift)
```

### 重构流程

```
1. 合成:    F_shift = magnitude * exp(1j * phase)
2. 逆频移:  F = np.fft.ifftshift(F_shift)
3. IFFT2:   img = Re(np.fft.ifft2(F))
4. 归一化到 [0, 255]
```

### 可视化增强

幅度在显示前进行对数缩放以压缩动态范围：`display = normalize(log(1 + |magnitude|))`

相位从 [−π, π] 偏移到 [0, 1] 后再归一化。

## 重要说明

1. **输入格式**：支持2D灰度(H,W)或3D RGB(H,W,3)；其他格式会抛出 `ValueError`
2. **原始 vs 显示频谱**：`get_raw_spectrum()` 用于重构；`get_display_spectrum()` 仅用于可视化
3. **通道处理**：RGB图像对每个通道独立执行FFT；重构时用 `cv2.merge` 合并通道
4. **掩码**：掩码形状必须与频率图 (H,W) 一致

## 参考文献

- Brigham, E. O. (1988). *The Fast Fourier Transform and Its Applications*. Prentice-Hall.
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
