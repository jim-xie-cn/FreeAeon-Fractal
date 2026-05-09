# FreeAeon-Fractal 使用文档

## 项目简介

**FreeAeon-Fractal** 是第一个支持GPU加速的图像多重分形分析Python工具包，用于计算图像和时间序列的**多重分形谱**、**分形维度**、**空隙度分析**、**局部奇异度**和**傅里叶频谱**。

### 主要特性

- 🎯 **多重分形谱分析**：支持2D图像和1D时间序列
- 📏 **分形维度计算**：BC、DBC、SDBC三种方法
- 🔍 **空隙度分析**：量化图像空间分布的不均匀性
- 🌊 **傅里叶分析**：频域分析和滤波
- ⚡ **GPU加速**：支持GPU加速计算（可选）
- 📊 **可视化**：内置丰富的可视化功能

### 安装

```bash
pip install FreeAeon-Fractal
```

**系统要求**：
- Python 3.6+
- OpenCV (`cv2`) 支持

## 应用领域

### 医学影像分析
- **组织复杂度**：通过分形维度量化组织结构
- **异质性分析**：多重分形谱揭示病变区域特征
- **纹理分类**：基于分形特征的图像分类

### 材料科学
- **表面形貌**：分形维度描述表面粗糙度
- **孔隙结构**：空隙度分析材料内部结构
- **断裂分析**：多重分形特征识别断裂模式

### 金融分析
- **价格波动**：多重分形谱分析股票价格
- **风险评估**：基于分形特征的风险量化
- **市场预测**：长程相关性分析

### 地球科学
- **地形分析**：分形维度描述地形复杂度
- **植被分布**：空隙度量化植被覆盖度
- **气候序列**：时间序列的多重分形分析

### 图像处理
- **纹理分类**：基于分形特征的纹理识别
- **图像分割**：基于多重分形的ROI提取
- **质量评估**：图像复杂度评价

## 快速开始

### 计算图像的多重分形谱

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
MFS.plot(df_mass, df_fit, df_spec)
```

### 计算图像的分形维度

```python
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImage import CFAImage

bin_image, threshold = CFAImage.otsu_binarize(gray_image)

fd_bc = CFAImageFD(bin_image).get_bc_fd()
fd_dbc = CFAImageFD(gray_image).get_dbc_fd()
fd_sdbc = CFAImageFD(gray_image).get_sdbc_fd()

CFAImageFD.plot(rgb_image, gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### 计算局部奇异度图（Alpha Map）

```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS

MFS = CFAImageMFS(gray_image)
alpha_map, info = MFS.compute_alpha_map()
CFAImageMFS.plot_alpha_map(alpha_map)
```

### 空隙度分析

```python
from FreeAeonFractal.FAImageLAC import CFAImageLAC

calc = CFAImageLAC(gray_image, partition_mode="gliding")
lac_result = calc.get_lacunarity()
fit_result = calc.fit_lacunarity(lac_result)
calc.plot(lac_result, fit_result)
```

### 傅里叶分析

```python
from FreeAeonFractal.FAImageFourier import CFAImageFourier

fourier = CFAImageFourier(rgb_image)
mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)
full_reconstructed = fourier.get_reconstruct()
fourier.plot(raw_magnitude_disp=mag_disp, raw_phase_disp=phase_disp, full_reconstructed=full_reconstructed)
```

### 计算时间序列的多重分形谱

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

x = np.cumsum(np.random.randn(5000))
mfs = CFASeriesMFS(x, q_list=np.linspace(-5, 5, 21))
df_mfs = mfs.get_mfs()
mfs.plot(df_mfs)
```

## 功能模块

### 1. 多重分形谱分析

| 类名 | 描述 | 应用场景 | 文档链接 |
|------|------|---------|----------|
| **CFAImageMFS** | 2D图像多重分形谱 | 纹理分析、医学图像、材料科学 | [查看详情](多重分形谱分析-CFAImageMFS.md) |
| **CFASeriesMFS** | 1D序列多重分形谱 | 金融时序、生理信号、气候数据 | [查看详情](序列多重分形谱-CFASeriesMFS.md) |

**核心概念**：
- **τ(q)** 质量指数：描述不同q阶下的标度行为
- **D(q)** 广义维度：量化数据在不同q下的维度特征
- **α(q)** 奇异性强度：描述数据的局部奇异性
- **f(α)** 多重分形谱：完整刻画数据的多重分形特性

**GPU加速**：
```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
```

### 2. 分形维度分析

| 类名 | 描述 | 方法 | 文档链接 |
|------|------|------|----------|
| **CFAImageFD** | 图像分形维度计算 | BC, DBC, SDBC | [查看详情](分形维度分析-CFAImageFD.md) |

**三种方法**：
- **BC** (Box-Counting)：适用于二值图像
- **DBC** (Differential Box-Counting)：适用于灰度图像（Sarkar & Chaudhuri 1994）
- **SDBC** (Shifted DBC)：改进版DBC，小尺度精度更高（Chen et al. 1995）

**维度范围**：
- 1D线条：D ≈ 1
- 2D平面：D ≈ 2
- 分形纹理：1 < D < 2

### 3. 空隙度分析

| 类名 | 描述 | 分区模式 | 文档链接 |
|------|------|---------|----------|
| **CFAImageLAC** | 图像空隙度分析 | Gliding, Non-overlapping | [查看详情](空隙度分析-CFAImageLAC.md) |

**分区模式**：
- **Gliding Box**：滑动窗口，使用积分图像高效计算
- **Non-overlapping Box**：非重叠块，适合独立性要求场景

**空隙度意义**：
- Λ = 1：完全均匀分布
- Λ > 1：存在空隙和聚集
- Λ越大：空间异质性越强

### 4. 傅里叶分析

| 类名 | 描述 | 功能 | 文档链接 |
|------|------|------|----------|
| **CFAImageFourier** | 图像傅里叶分析 | 频谱分析、滤波、重构 | [查看详情](傅里叶分析-CFAImageFourier.md) |

**主要功能**：
- 幅度和相位频谱计算
- 频域可视化（对数增强）
- 自定义频率掩码滤波
- 图像重构

### 5. 图像处理工具

| 类名 | 描述 | 主要方法 | 文档链接 |
|------|------|---------|----------|
| **CFAImage** | 图像处理工具类 | 分块、合并、二值化、ROI | [查看详情](图像工具-CFAImage.md) |

**核心功能**：
- **Otsu自动二值化**：自动阈值分割
- **图像分块/合并**：支持灰度和彩色图像
- **掩码生成**：基于块位置生成掩码
- **随机采样**：数据增强用途
- **ROI提取**：基于多重分形度量的区域提取

### 6. 可视化工具

| 类名 | 描述 | 支持维度 | 文档链接 |
|------|------|---------|----------|
| **CFAVisual** | 可视化工具类 | 1D, 2D, 3D点集和图像 | [查看详情](可视化工具-CFAVisual.md) |

### 7. 分形样本生成

| 类名 | 描述 | 支持图案 | 文档链接 |
|------|------|---------|----------|
| **CFASample** | 分形样本生成器 | 康托集、谢尔宾斯基三角形、巴恩斯利蕨、门格海绵 | [查看详情](分形样本生成-CFASample.md) |

### 8. GPU加速

| 功能 | CPU模块 | GPU模块 | 加速比 | 文档链接 |
|------|---------|---------|--------|----------|
| 2D多重分形谱 | CFAImageMFS | CFAImageMFSGPU | 5-20× | [查看详情](GPU加速版本.md) |
| 图像分形维度 | CFAImageFD | CFAImageFDGPU | 3-10× | [查看详情](GPU加速版本.md) |
| 图像空隙度 | CFAImageLAC | CFAImageLACGPU | 5-15× | [查看详情](GPU加速版本.md) |

```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
```

## 命令行使用

```bash
# 多重分形谱
python demo.py --mode mfs --image ./images/face.png

# 分形维度
python demo.py --mode fd --image ./images/fractal.png

# 局部奇异度图
python demo.py --mode alpha --image ./images/face.png

# 空隙度分析
python demo.py --mode lacunarity --image ./images/fractal.png

# 傅里叶分析
python demo.py --mode fourier --image ./images/face.png

# 序列多重分形
python demo.py --mode series
```

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--image` | 输入图像路径 | 图像文件路径 |
| `--mode` | 分析模式 | `fd`, `mfs`, `alpha`, `lacunarity`, `fourier`, `series` |

## demo.py 完整示例

以下为 `demo.py` 中各 `--mode` 对应的完整Python代码示例。

### mode=fd — 分形维度

```python
import cv2, time
import numpy as np
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImage import CFAImage

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

max_scales = 32

# --- 单张图像 ---
t0 = time.time()
fd_bc   = CFAImageFD(bin_image,  max_scales=max_scales).get_bc_fd(corp_type=-1)
fd_dbc  = CFAImageFD(gray_image, max_scales=max_scales).get_dbc_fd(corp_type=-1)
fd_sdbc = CFAImageFD(gray_image, max_scales=max_scales).get_sdbc_fd(corp_type=-1)
print(f"单张图像: {time.time()-t0:.3f}s")
print("  BC:",   fd_bc['fd'])
print("  DBC:",  fd_dbc['fd'])
print("  SDBC:", fd_sdbc['fd'])
CFAImageFD.plot(rgb_image, gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)

# --- 批量（100张图像）---
bin_imgs  = [bin_image]  * 100
gray_imgs = [gray_image] * 100
t0 = time.time()
bc_list   = CFAImageFD.get_batch_bc(bin_imgs,   max_scales=max_scales, with_progress=False)
dbc_list  = CFAImageFD.get_batch_dbc(gray_imgs,  max_scales=max_scales, with_progress=False)
sdbc_list = CFAImageFD.get_batch_sdbc(gray_imgs, max_scales=max_scales, with_progress=False)
print(f"批量（100张）: {time.time()-t0:.3f}s")
print(f"  batch BC FD[99]   = {bc_list[99]['fd']:.4f}")
print(f"  batch DBC FD[99]  = {dbc_list[99]['fd']:.4f}")
print(f"  batch SDBC FD[99] = {sdbc_list[99]['fd']:.4f}")
CFAImageFD.plot(rgb_image, gray_image, bin_image, bc_list[99], dbc_list[99], sdbc_list[99])
```

### mode=mfs — 多重分形谱

```python
import cv2, time
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

q_list = np.linspace(-10, 10, 101)

# --- 单张图像 ---
t0 = time.time()
MFS = CFAImageMFS(gray_image, q_list=q_list)
df_mass, df_fit, df_spec = MFS.get_mfs()
print(f"单张MFS: {time.time()-t0:.3f}s")
print(df_fit.head())
MFS.plot(df_mass, df_fit, df_spec)

# --- 批量（20张图像）---
t0 = time.time()
imgs = [gray_image] * 20
batch_results = CFAImageMFS.get_batch_mfs(
    imgs,
    with_progress=False, q_list=q_list, corp_type=-1,
    bg_reverse=False, bg_threshold=0.01, bg_otsu=False, max_scales=80,
    min_points=6, use_middle_scales=False, if_auto_line_fit=False,
    fit_scale_frac=(0.3, 0.7), auto_fit_min_len_ratio=0.6, cap_d0_at_2=False
)
df_mass1, df_fit1, df_spec1 = batch_results[0]
print(f"批量MFS（20张）: {time.time()-t0:.3f}s")
print(df_fit1.head())
MFS.plot(df_mass1, df_fit1, df_spec1)
```

### mode=alpha — 局部奇异度图

```python
import cv2, time
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

q_list = np.linspace(-5, 5, 51)
scales = list(range(1, 100))

# --- 单张图像 ---
t0 = time.time()
MFS = CFAImageMFS(gray_image, q_list=q_list)
alpha_map, info = MFS.compute_alpha_map(scales=scales)
print(f"单张alpha_map: {time.time()-t0:.3f}s")
print("  alpha map:", alpha_map)
print("  scale info:", info)
CFAImageMFS.plot_alpha_map(alpha_map)

# --- 批量（20张图像）---
t0 = time.time()
imgs = [gray_image] * 20
batch_alpha_map = CFAImageMFS.compute_alpha_map_batch(imgs, with_progress=False, scales=scales)
alpha_maps = batch_alpha_map[0]
infos      = batch_alpha_map[1]
print(f"批量alpha_map（20张）: {time.time()-t0:.3f}s")
print("  alpha map:", alpha_maps[0])
print("  scale info:", infos)
CFAImageMFS.plot_alpha_map(alpha_maps[0])
```

### mode=lacunarity — 空隙度分析

```python
import cv2, time
from FreeAeonFractal.FAImageLAC import CFAImageLAC

image_path = './images/fractal.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

lacunarity = CFAImageLAC(gray_image, max_scales=256, with_progress=True)

# --- 单张图像 ---
t0 = time.time()
lac_gray = lacunarity.get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)
fit_gray = lacunarity.fit_lacunarity(lac_gray)
print(f"单张空隙度: {time.time()-t0:.3f}s")
print("  Gray lacunarity:", lac_gray["lacunarity"])
print("  斜率:", fit_gray["slope"],
      "截距:", fit_gray["intercept"],
      "R:", fit_gray["r_value"],
      "P:", fit_gray["p_value"])
lacunarity.plot(lac_gray, fit_gray)

# --- 批量（100张图像）---
t0 = time.time()
imgs = [gray_image] * 100
batchs = CFAImageLAC.get_batch_lacunarity(
    imgs, scales_mode="powers", partition_mode="gliding",
    use_binary_mass=False, with_progress=False
)
fits = CFAImageLAC.fit_batch_lacunarity(batchs)
print(f"批量空隙度（100张）: {time.time()-t0:.3f}s")
print("  Gray lacunarity:", batchs[99]["lacunarity"])
print("  斜率:", fits[99]["slope"],
      "截距:", fits[99]["intercept"],
      "R:", fits[99]["r_value"],
      "P:", fits[99]["p_value"])
lacunarity.plot(batchs[99], fits[99])
```

### mode=fourier — 傅里叶分析

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# 支持灰度图和RGB图
fourier = CFAImageFourier(rgb_image)

# 获取原始频谱（用于重构）
raw_mag, raw_phase = fourier.get_raw_spectrum()

# 获取可视化频谱
raw_mag_disp, raw_phase_disp = fourier.get_display_spectrum(alpha=1.5)

# 创建自定义频率掩码（示例：保留奇数频率分量）
h, w = raw_mag[0].shape
Y, X = np.ogrid[:h, :w]
mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

# 获取掩码后的可视化频谱
customized_mag_list   = raw_mag   * mask
customized_phase_list = raw_phase * mask
customized_mag_disp, customized_phase_disp = fourier.get_display_spectrum(
    alpha=1.5,
    magnitude=customized_mag_list,
    phase=customized_phase_list
)

# 从原始频谱重构完整图像
full_reconstructed = fourier.get_reconstruct()

# 用频率掩码重构图像
masked_reconstructed = fourier.extract_by_freq_mask(mask)

# 显示所有结果
fourier.plot(
    raw_mag_disp,
    raw_phase_disp,
    customized_mag_disp,
    customized_phase_disp,
    full_reconstructed,
    masked_reconstructed
)
print(masked_reconstructed)
```

### mode=series — 序列多重分形谱

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

# 生成随机游走序列（示例数据）
x = np.cumsum(np.random.randn(5000))

q = np.linspace(-5, 5, 21)
mfs = CFASeriesMFS(x)
df_mfs = mfs.get_mfs()
mfs.plot(df_mfs)
print(df_mfs)
```

## 进阶用法

### 批量处理

```python
import glob, cv2, numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageMFS.get_batch_mfs(images, q_list=np.linspace(-5, 5, 26))
```

### 特征提取（机器学习）

```python
def extract_mf_features(image):
    mfs = CFAImageMFS(image, q_list=np.linspace(-5, 5, 11))
    _, df_fit, df_spec = mfs.get_mfs()
    return {
        'D0': df_fit.loc[df_fit['q'].round(2)==0.0, 'Dq'].values[0],
        'D1': df_fit.loc[df_fit['q'].round(2)==1.0, 'D1'].values[0],
        'delta_alpha': df_spec['alpha'].max() - df_spec['alpha'].min(),
    }
```

## 常见问题

### Q: 如何选择合适的q值范围？
A: 一般使用 `q ∈ [-5, 5]`，负q对稀疏区域敏感，正q对密集区域敏感。

### Q: 分形维度计算结果 > 2？
A: 检查图像预处理是否正确。对于2D图像，维度应在 [1, 2] 之间。

### Q: 如何提高计算速度？
A: (1) 使用GPU版本 (2) 减少q值数量 (3) 减少尺度数量 (4) 降采样大图像。

### Q: 多重分形谱不显著？
A: 检查 `Δh = h(-5) - h(5)` 或谱宽 `Δα`，如果值很小可能数据确实是单分形。

### Q: 如何判断是否为多重分形？
A: (1) D(q) 随q单调变化 (2) f(α) 为凸函数且有一定宽度 (3) Δh > 0.1。

## 项目结构

```
FreeAeon-Fractal/
├── FreeAeonFractal/          # 核心模块
│   ├── FAImageMFS.py         # 2D多重分形谱
│   ├── FAImageMFSGPU.py      # 2D多重分形谱(GPU)
│   ├── FASeriesMFS.py        # 1D多重分形谱
│   ├── FAImageFD.py          # 分形维度
│   ├── FAImageFDGPU.py       # 分形维度(GPU)
│   ├── FAImageLAC.py         # 空隙度
│   ├── FAImageLACGPU.py      # 空隙度(GPU)
│   ├── FAImageFourier.py     # 傅里叶分析
│   ├── FAImage.py            # 图像工具
│   ├── FASample.py           # 分形样本生成
│   ├── FAVisual.py           # 可视化工具
│   └── __init__.py
├── demo.py                   # 命令行接口
├── images/                   # 示例图像
├── requirements.txt
├── setup.py
└── README.md
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/LICENSE)。

## 作者

- **Jim Xie** - 📧 jim.xie.cn@outlook.com, xiewenwei@sina.com

## 引用

如果您在学术工作中使用本项目，请引用：

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal

## 相关链接

- 🔗 GitHub: https://github.com/jim-xie-cn/FreeAeon-Fractal
- 📦 PyPI: https://pypi.org/project/FreeAeon-Fractal/

## 参考文献

### 分形理论
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.

### 多重分形分析
- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. *Chaos and Fractals*.

### MFDFA方法
- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A*.

### 分形维度
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach. *IEEE Transactions on Systems, Man, and Cybernetics*.
- Chen, W. S., et al. (1995). Efficient fractal coding of images based on differential box counting. *Pattern Recognition*.

### 空隙度
- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. *Physical Review A*.
- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. *Physical Review E*.

---

© 2025 FreeAeon-Fractal. 保留所有权利。
