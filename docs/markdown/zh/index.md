# FreeAeon-Fractal 使用文档索引

## 概述

FreeAeon-Fractal 是一个强大的Python工具包,用于计算图像或时间序列的多重分形谱、分形维度、空隙度和傅里叶谱。该工具包提供了CPU和GPU两种实现方式,适用于各种规模的数据分析任务。

## 核心功能模块

### 1. 多重分形谱分析

- **[CFA1DMFS](./多重分形谱分析-CFA1DMFS.md)** - 一维时间序列的多重分形谱分析
  - 适用场景:金融时间序列分析、生理信号分析、地震数据分析
  - 主要功能:计算h(q)、τ(q)、α(q)、f(α)和D(q)

- **[CFA2DMFS](./多重分形谱分析-CFA2DMFS.md)** - 二维图像的多重分形谱分析(CPU版本)
  - 适用场景:医学影像分析、遥感图像分析、纹理分析
  - 主要功能:基于盒计数法的多重分形谱计算

- **[CFA2DMFSGPU](./多重分形谱分析-CFA2DMFSGPU.md)** - 二维图像的多重分形谱分析(GPU加速)
  - 适用场景:大规模图像批处理、实时分析场景
  - 主要功能:GPU加速的多重分形谱计算、批量处理支持

### 2. 分形维度计算

- **[CFAImageDimension](./分形维度计算-CFAImageDimension.md)** - 分形维度计算(CPU版本)
  - 适用场景:图像复杂度评估、纹理特征提取、形态学分析
  - 主要方法:BC(盒计数法)、DBC(差分盒计数法)、SDBC(简化差分盒计数法)

- **[CFAImageDimensionGPU](./分形维度计算-CFAImageDimensionGPU.md)** - 分形维度计算(GPU加速)
  - 适用场景:高分辨率图像分析、批量图像处理
  - 主要功能:GPU加速的BC/DBC/SDBC计算、批量处理支持

### 3. 空隙度分析

- **[CFAImageLacunarity](./空隙度分析-CFAImageLacunarity.md)** - 图像空隙度分析(CPU版本)
  - 适用场景:纹理异质性分析、空间分布特征提取、图像质量评估
  - 主要功能:滑动窗口法、非重叠窗口法空隙度计算

- **[CFAImageLacunarityGPU](./空隙度分析-CFAImageLacunarityGPU.md)** - 图像空隙度分析(GPU加速)
  - 适用场景:大规模图像数据集分析、实时空隙度计算
  - 主要功能:GPU加速的空隙度计算、批量处理支持

### 4. 傅里叶频域分析

- **[CFAImageFourier](./傅里叶分析-CFAImageFourier.md)** - 图像的傅里叶频域分析
  - 适用场景:频率成分分析、图像滤波、频域特征提取
  - 主要功能:幅度谱和相位谱计算、频域重建、频率掩码提取

### 5. 图像预处理工具

- **[CFAImage](./图像处理-CFAImage.md)** - 图像预处理和ROI提取工具
  - 适用场景:图像预处理、感兴趣区域提取、数据增强
  - 主要功能:图像裁剪/填充、分块操作、Otsu二值化、基于q值的ROI提取、随机采样

### 6. 分形样本生成

- **[CFASample](./样本生成-CFASample.md)** - 经典分形图案生成器
  - 适用场景:算法测试、教学演示、基准数据生成
  - 支持的分形:康托集、谢尔宾斯基三角、巴恩斯利蕨、门格海绵

### 7. 可视化工具

- **[CFAVisual](./可视化-CFAVisual.md)** - 点集和图像可视化工具
  - 适用场景:结果展示、数据探索、可视化分析
  - 主要功能:1D/2D/3D点集可视化、图像显示

## 快速开始

### 安装

```bash
pip install FreeAeon-Fractal
```

### 基本使用示例

#### 计算图像的多重分形谱

```python
import cv2
import numpy as np
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# 读取图像
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 创建多重分形谱分析对象
mfs = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))

# 计算多重分形谱
df_mass, df_fit, df_spec = mfs.get_mfs()

# 可视化结果
mfs.plot(df_mass, df_fit, df_spec)
```

#### 计算图像的分形维度

```python
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImage import CFAImage

# 图像二值化
bin_image, threshold = CFAImage.otsu_binarize(image)

# 计算分形维度
fd_bc = CFAImageDimension(bin_image).get_bc_fd()
fd_dbc = CFAImageDimension(image).get_dbc_fd()
fd_sdbc = CFAImageDimension(image).get_sdbc_fd()

print(f"BC分形维度: {fd_bc['fd']:.4f}")
print(f"DBC分形维度: {fd_dbc['fd']:.4f}")
print(f"SDBC分形维度: {fd_sdbc['fd']:.4f}")
```

## GPU加速

对于大规模数据处理,可以使用GPU加速版本:

```python
# 使用GPU版本的多重分形谱分析
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS

# 使用GPU版本的分形维度计算
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension
```

## 技术特点

- **多重分形理论**:基于最新的盒计数法多重分形谱计算算法
- **高性能计算**:支持CPU和GPU两种实现方式
- **批量处理**:支持多图像批量分析
- **数值稳定性**:使用logsumexp等数值稳定技术
- **易用性**:简洁的API设计,详细的文档说明

## 应用领域

- **医学影像**:肿瘤纹理分析、病理图像分类
- **遥感图像**:地表纹理分析、土地利用分类
- **材料科学**:材料表面形貌分析、微观结构表征
- **金融分析**:市场波动性分析、风险评估
- **信号处理**:生理信号分析、地震数据处理

## 参考文献

如果您在学术工作中使用本工具,请引用:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal

## 联系方式

- **作者**: Jim Xie
- **邮箱**: jim.xie.cn@outlook.com, xiewenwei@sina.com
- **GitHub**: https://github.com/jim-xie-cn/FreeAeon-Fractal

## 许可证

MIT License - 详见 [LICENSE](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/LICENSE)

---

**文档版本**: 1.0
**更新日期**: 2026-03-23
