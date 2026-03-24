# FreeAeon-Fractal 使用文档

## 项目简介

**FreeAeon-Fractal** 是一个用于计算图像和时间序列的**多重分形谱**、**分形维度**、**分形空隙度**和**傅里叶频谱**的Python工具包。

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
- OpenCV (`cv2`)支持

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
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# 读取并转换为灰度图像
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 多重分形谱分析
MFS = CFA2DMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()

# 可视化结果
MFS.plot(df_mass, df_fit, df_spec)
```

### 计算图像的分形维度

```python
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImage import CFAImage

# 图像二值化
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# 计算分形维度
fd_bc = CFAImageDimension(bin_image).get_bc_fd()
fd_dbc = CFAImageDimension(gray_image).get_dbc_fd()
fd_sdbc = CFAImageDimension(gray_image).get_sdbc_fd()

# 可视化
CFAImageDimension.plot(rgb_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### 计算时间序列的多重分形谱

```python
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# 生成随机游走序列
x = np.cumsum(np.random.randn(5000))

# 多重分形谱分析
mfs = CFA1DMFS(x, q_list=np.linspace(-5, 5, 21))
df_mfs = mfs.get_mfs()

# 可视化
mfs.plot(df_mfs)
```

## 功能模块

### 1. 多重分形谱分析 (Multifractal Spectrum Analysis)

多重分形谱用于分析图像或序列在不同尺度下的统计特性，揭示数据的复杂度和异质性。

| 类名 | 描述 | 应用场景 | 文档链接 |
|------|------|---------|----------|
| **CFA2DMFS** | 2D图像多重分形谱 | 纹理分析、医学图像、材料科学 | [查看详情](多重分形谱分析-CFA2DMFS.md) |
| **CFA1DMFS** | 1D序列多重分形谱 | 金融时序、生理信号、气候数据 | [查看详情](序列多重分形谱-CFA1DMFS.md) |

**核心概念**：
- **τ(q)** 质量指数：描述不同q阶下的标度行为
- **D(q)** 广义维度：量化数据在不同q下的维度特征
- **α(q)** 奇异性强度：描述数据的局部奇异性
- **f(α)** 多重分形谱：完整刻画数据的多重分形特性

**GPU加速**：
```python
# 使用GPU版本
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
```

### 2. 分形维度分析 (Fractal Dimension Analysis)

分形维度量化图像的复杂度和自相似性，是图像纹理分析的重要工具。

| 类名 | 描述 | 方法 | 文档链接 |
|------|------|------|----------|
| **CFAImageDimension** | 图像分形维度计算 | BC, DBC, SDBC | [查看详情](分形维度分析-CFAImageDimension.md) |

**三种方法**：
- **BC** (Box-Counting)：适用于二值图像
- **DBC** (Differential Box-Counting)：适用于灰度图像
- **SDBC** (Simplified DBC)：DBC的简化版本，更快

**维度范围**：
- 1D线条：D ≈ 1
- 2D平面：D ≈ 2
- 分形纹理：1 < D < 2

### 3. 空隙度分析 (Lacunarity Analysis)

空隙度量化图像空间分布的不均匀性和间隙特征。

| 类名 | 描述 | 分区模式 | 文档链接 |
|------|------|---------|----------|
| **CFAImageLacunarity** | 图像空隙度分析 | Gliding, Non-overlapping | [查看详情](空隙度分析-CFAImageLacunarity.md) |

**分区模式**：
- **Gliding Box**：滑动窗口（重叠），结果更平滑
- **Non-overlapping Box**：非重叠块，计算更快

**空隙度意义**：
- Λ = 1：完全均匀分布
- Λ > 1：存在空隙和聚集
- Λ越大：空间异质性越强

### 4. 傅里叶分析 (Fourier Analysis)

傅里叶分析用于研究图像的频率成分和进行频域处理。

| 类名 | 描述 | 功能 | 文档链接 |
|------|------|------|----------|
| **CFAImageFourier** | 图像傅里叶分析 | 频谱分析、滤波、重构 | [查看详情](傅里叶分析-CFAImageFourier.md) |

**主要功能**：
- 幅度和相位频谱计算
- 频域可视化（对数增强）
- 频域滤波（高通、低通、带通）
- 图像重构

**应用案例**：
- 周期性噪声去除
- 边缘检测（高通滤波）
- 方向性滤波

### 5. 图像处理工具 (Image Processing Utilities)

提供图像分块、合并、掩码生成、ROI提取等基础工具。

| 类名 | 描述 | 主要方法 | 文档链接 |
|------|------|---------|----------|
| **CFAImage** | 图像处理工具类 | 分块、合并、二值化、ROI | [查看详情](图像工具-CFAImage.md) |

**核心功能**：
- **Otsu自动二值化**：自动阈值分割
- **图像分块/合并**：支持灰度和彩色图像
- **掩码生成**：基于块位置生成掩码
- **随机采样**：数据增强用途
- **ROI提取**：基于多重分形特性

### 6. 可视化工具 (Visualization Tools)

提供分形点集和图像的可视化工具。

| 类名 | 描述 | 支持维度 | 文档链接 |
|------|------|---------|----------|
| **CFAVisual** | 可视化工具类 | 1D, 2D, 3D点集和图像 | [查看详情](可视化工具-CFAVisual.md) |

**主要功能**：
- **1D点集显示**：康托集等1D分形
- **2D点集显示**：谢尔宾斯基三角形、巴恩斯利蕨等
- **3D点集显示**：门格海绵等3D分形
- **图像显示**：2D和3D图像可视化

### 7. 分形样本生成 (Fractal Sample Generator)

生成经典分形图案，用于测试、教学和艺术创作。

| 类名 | 描述 | 支持图案 | 文档链接 |
|------|------|---------|----------|
| **CFASample** | 分形样本生成器 | 康托集、谢尔宾斯基三角形、巴恩斯利蕨、门格海绵 | [查看详情](分形样本生成-CFASample.md) |

**支持的分形图案**：
- **康托集** (Cantor Set)：1D分形，维度 ≈ 0.63
- **谢尔宾斯基三角形** (Sierpinski Triangle)：2D分形，维度 ≈ 1.58
- **巴恩斯利蕨** (Barnsley Fern)：2D分形，维度 ≈ 1.67
- **门格海绵** (Menger Sponge)：3D分形，维度 ≈ 2.73

**应用场景**：
- 算法测试和验证
- 分形几何教学
- 艺术创作

### 8. GPU加速 (GPU Acceleration)

所有核心计算模块均提供GPU加速版本，显著提升性能。

| 功能 | CPU模块 | GPU模块 | 加速比 | 文档链接 |
|------|---------|---------|--------|----------|
| 2D多重分形谱 | CFA2DMFS | CFA2DMFSGPU | 5-20x | [查看详情](GPU加速版本.md) |
| 图像分形维度 | CFAImageDimension | CFAImageDimensionGPU | 3-10x | [查看详情](GPU加速版本.md) |
| 图像空隙度 | CFAImageLacunarity | CFAImageLacunarityGPU | 5-15x | [查看详情](GPU加速版本.md) |

**使用方式**：
```python
# 简单导入替换即可
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
# 其余代码完全相同！
```

**系统要求**：
- NVIDIA GPU（支持CUDA）
- PyTorch with CUDA

## 命令行使用

### 计算多重分形谱

```bash
python demo.py --mode mfs --image ./images/face.png
```

### 计算分形维度

```bash
python demo.py --mode fd --image ./images/fractal.png
```

### 空隙度分析

```bash
python demo.py --mode lacunarity --image ./images/fractal.png
```

### 傅里叶分析

```bash
python demo.py --mode fourier --image ./images/face.png
```

### 序列分析

```bash
python demo.py --mode series
```

### 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--image` | 输入图像路径 | 图像文件路径 |
| `--mode` | 分析模式 | `fd`, `mfs`, `lacunarity`, `fourier`, `series` |

## GPU加速

支持GPU加速的模块：

| 模块 | CPU版本 | GPU版本 |
|------|---------|---------|
| 2D多重分形谱 | `FA2DMFS.CFA2DMFS` | `FA2DMFSGPU.CFA2DMFSGPU` |
| 图像分形维度 | `FAImageDimension.CFAImageDimension` | `FAImageDimensionGPU.CFAImageDimensionGPU` |
| 图像空隙度 | `FAImageLacunarity.CFAImageLacunarity` | `FAImageLacunarityGPU.CFAImageLacunarityGPU` |

**使用方法**：

```python
# 导入GPU版本
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension

# 使用方式与CPU版本相同
MFS = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

**性能提升**：
- 大图像：5-10倍加速
- 多尺度分析：10-20倍加速
- 需要CUDA支持的GPU

## 常见问题

### Q: 如何选择合适的q值范围？
A: 一般使用 `q ∈ [-5, 5]`，负q对稀疏区域敏感，正q对密集区域敏感。可根据具体应用调整。

### Q: 分形维度计算结果 > 2？
A: 检查图像预处理是否正确。对于2D图像，维度应在 [1, 2] 之间。

### Q: 如何提高计算速度？
A: (1) 使用GPU版本 (2) 减少q值数量 (3) 减少尺度数量 (4) 降采样大图像

### Q: 多重分形谱不显著？
A: 检查 `Δh = h(-5) - h(5)` 或谱宽 `Δα`，如果值很小可能数据确实是单分形。

### Q: 如何判断是否为多重分形？
A: (1) D(q) 随q单调变化 (2) f(α) 为凸函数且有一定宽度 (3) Δh > 0.1

## 进阶用法

### 自定义尺度

```python
# 自定义尺度参数
scales = np.logspace(1, np.log2(256), 50, base=2).astype(int)
df_mass = mfs.get_mass_table(max_scales=50)
```

### 批量处理

```python
import glob

# 批量分析多个图像
image_files = glob.glob('./images/*.png')
results = []

for img_file in image_files:
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    mfs = CFA2DMFS(image)
    _, df_fit, _ = mfs.get_mfs()
    results.append({
        'file': img_file,
        'D0': df_fit.loc[df_fit['q'] == 0, 'Dq'].values[0],
        'D1': df_fit.loc[df_fit['q'] == 1, 'D1'].values[0]
    })

import pandas as pd
df_results = pd.DataFrame(results)
print(df_results)
```

### 特征提取

```python
# 提取多重分形特征用于机器学习
def extract_mf_features(image):
    mfs = CFA2DMFS(image, q_list=np.linspace(-5, 5, 11))
    _, df_fit, df_spec = mfs.get_mfs()

    features = {
        'D0': df_fit.loc[df_fit['q'] == 0, 'Dq'].values[0],
        'D1': df_fit.loc[df_fit['q'] == 1, 'D1'].values[0],
        'D2': df_fit.loc[df_fit['q'] == 2, 'Dq'].values[0],
        'alpha_max': df_spec['alpha'].max(),
        'alpha_min': df_spec['alpha'].min(),
        'delta_alpha': df_spec['alpha'].max() - df_spec['alpha'].min(),
        'f_alpha_max': df_spec['f_alpha'].max()
    }
    return features

# 提取特征
features = extract_mf_features(gray_image)
print(features)
```

## 项目结构

```
FreeAeon-Fractal/
├── FreeAeonFractal/          # 核心模块
│   ├── FA2DMFS.py            # 2D多重分形谱
│   ├── FA2DMFSGPU.py         # 2D多重分形谱(GPU)
│   ├── FA1DMFS.py            # 1D多重分形谱
│   ├── FAImageDimension.py   # 分形维度
│   ├── FAImageDimensionGPU.py# 分形维度(GPU)
│   ├── FAImageLacunarity.py  # 空隙度
│   ├── FAImageLacunarityGPU.py # 空隙度(GPU)
│   ├── FAImageFourier.py     # 傅里叶分析
│   ├── FAImage.py            # 图像工具
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
- **Yin Jie** - 📧 yinjiejspi@163.com
- **Cindy Ma** - 📧 453303661@qq.com
- **Wenjing Zhang** - 📧 634676988@qq.com
- **Danny Zhang** - 📧 zhyzxsw@126.com

## 引用

如果您在学术工作中使用本项目，请引用：

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal

## 相关链接

- 🔗 GitHub: https://github.com/jim-xie-cn/FreeAeon-Fractal
- 📦 PyPI: https://pypi.org/project/FreeAeon-Fractal/
- 📖 完整文档: https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/docs/markdown/en/index.md

## 参考文献

### 分形理论
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.

### 多重分形分析
- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. *Chaos and Fractals*.

### MFDFA方法
- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A*.

### 分形维度
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image. *IEEE Transactions on Systems, Man, and Cybernetics*.

### 空隙度
- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. *Physical Review E*.

---

© 2025 FreeAeon-Fractal. 保留所有权利。
