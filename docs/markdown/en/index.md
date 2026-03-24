# FreeAeon-Fractal Documentation

## Project Overview

**FreeAeon-Fractal** is a Python toolkit for computing **Multifractal Spectra**, **Fractal Dimensions**, **Fractal Lacunarity**, and **Fourier Spectra** of images and time series.

### Key Features

- 🎯 **Multifractal Spectrum Analysis**: Supports 2D images and 1D time series
- 📏 **Fractal Dimension Calculation**: BC, DBC, and SDBC methods
- 🔍 **Lacunarity Analysis**: Quantify spatial heterogeneity
- 🌊 **Fourier Analysis**: Frequency domain analysis and filtering
- ⚡ **GPU Acceleration**: Optional GPU support for faster computation
- 📊 **Visualization**: Built-in rich visualization capabilities

### Installation

```bash
pip install FreeAeon-Fractal
```

**Requirements**:
- Python 3.6+
- OpenCV (`cv2`) support

## Application Domains

### Medical Imaging
- **Tissue Complexity**: Quantify tissue structure via fractal dimension
- **Heterogeneity Analysis**: Reveal lesion characteristics via multifractal spectrum
- **Texture Classification**: Image classification based on fractal features

### Materials Science
- **Surface Morphology**: Describe surface roughness via fractal dimension
- **Porous Structure**: Analyze internal structure via lacunarity
- **Fracture Analysis**: Identify fracture patterns via multifractal features

### Financial Analysis
- **Price Fluctuations**: Analyze stock prices via multifractal spectrum
- **Risk Assessment**: Quantify risk based on fractal features
- **Market Prediction**: Long-range correlation analysis

### Earth Sciences
- **Terrain Analysis**: Describe terrain complexity via fractal dimension
- **Vegetation Distribution**: Quantify vegetation coverage via lacunarity
- **Climate Series**: Multifractal analysis of time series

### Image Processing
- **Texture Classification**: Texture recognition based on fractal features
- **Image Segmentation**: ROI extraction based on multifractal analysis
- **Quality Assessment**: Image complexity evaluation

## Quick Start

### Multifractal Spectrum of an Image

```python
import cv2
import numpy as np
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# Load and convert to grayscale
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Multifractal spectrum analysis
MFS = CFA2DMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()

# Visualize results
MFS.plot(df_mass, df_fit, df_spec)
```

### Fractal Dimensions of an Image

```python
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImage import CFAImage

# Binarize image
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# Calculate fractal dimensions
fd_bc = CFAImageDimension(bin_image).get_bc_fd()
fd_dbc = CFAImageDimension(gray_image).get_dbc_fd()
fd_sdbc = CFAImageDimension(gray_image).get_sdbc_fd()

# Visualize
CFAImageDimension.plot(rgb_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### Multifractal Spectrum of a Time Series

```python
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# Generate random walk
x = np.cumsum(np.random.randn(5000))

# Multifractal analysis
mfs = CFA1DMFS(x, q_list=np.linspace(-5, 5, 21))
df_mfs = mfs.get_mfs()

# Visualize
mfs.plot(df_mfs)
```

## Feature Modules

### 1. Multifractal Spectrum Analysis

Multifractal spectrum reveals the complexity and heterogeneity of data across different scales.

| Class | Description | Applications | Documentation |
|-------|-------------|--------------|---------------|
| **CFA2DMFS** | 2D Image Multifractal Spectrum | Texture analysis, medical imaging, materials science | [Details](Multifractal-Spectrum-CFA2DMFS.md) |
| **CFA1DMFS** | 1D Series Multifractal Spectrum | Financial time series, physiological signals, climate data | [Details](Series-Multifractal-CFA1DMFS.md) |

**Core Concepts**:
- **τ(q)** Mass Exponent: Scaling behavior for different q orders
- **D(q)** Generalized Dimension: Dimension characteristics at different q
- **α(q)** Singularity Strength: Local singularity of data
- **f(α)** Multifractal Spectrum: Complete characterization of multifractal properties

**GPU Acceleration**:
```python
# Use GPU version
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
```

### 2. Fractal Dimension Analysis

Fractal dimension quantifies image complexity and self-similarity.

| Class | Description | Methods | Documentation |
|-------|-------------|---------|---------------|
| **CFAImageDimension** | Image Fractal Dimension | BC, DBC, SDBC | [Details](Fractal-Dimension-CFAImageDimension.md) |

**Three Methods**:
- **BC** (Box-Counting): For binary images
- **DBC** (Differential Box-Counting): For grayscale images
- **SDBC** (Simplified DBC): Faster simplified version

**Dimension Range**:
- 1D lines: D ≈ 1
- 2D planes: D ≈ 2
- Fractal textures: 1 < D < 2

### 3. Lacunarity Analysis

Lacunarity quantifies spatial heterogeneity and gap characteristics.

| Class | Description | Partition Modes | Documentation |
|-------|-------------|-----------------|---------------|
| **CFAImageLacunarity** | Image Lacunarity Analysis | Gliding, Non-overlapping | [Details](Lacunarity-Analysis-CFAImageLacunarity.md) |

**Partition Modes**:
- **Gliding Box**: Sliding windows (overlapping), smoother results
- **Non-overlapping Box**: Non-overlapping blocks, faster computation

**Lacunarity Meaning**:
- Λ = 1: Completely uniform distribution
- Λ > 1: Presence of gaps and clustering
- Larger Λ: Higher spatial heterogeneity

### 4. Fourier Analysis

Fourier analysis for studying image frequency components and frequency domain processing.

| Class | Description | Features | Documentation |
|-------|-------------|----------|---------------|
| **CFAImageFourier** | Image Fourier Analysis | Spectrum analysis, filtering, reconstruction | [Details](Fourier-Analysis-CFAImageFourier.md) |

**Main Features**:
- Magnitude and phase spectrum computation
- Frequency domain visualization (logarithmic enhancement)
- Frequency domain filtering (high-pass, low-pass, band-pass)
- Image reconstruction

**Applications**:
- Periodic noise removal
- Edge detection (high-pass filtering)
- Directional filtering

### 5. Image Processing Utilities

Basic tools for image blocking, merging, mask generation, and ROI extraction.

| Class | Description | Main Methods | Documentation |
|-------|-------------|--------------|---------------|
| **CFAImage** | Image Processing Utilities | Blocking, merging, binarization, ROI | [Details](Image-Utils-CFAImage.md) |

**Core Functions**:
- **Otsu Auto-Thresholding**: Automatic threshold segmentation
- **Image Blocking/Merging**: Support for grayscale and color images
- **Mask Generation**: Generate masks based on block positions
- **Random Sampling**: For data augmentation
- **ROI Extraction**: Based on multifractal properties

### 6. Visualization Tools

Visualization tools for fractal point sets and images.

| Class | Description | Supported Dimensions | Documentation |
|-------|-------------|---------------------|---------------|
| **CFAVisual** | Visualization Utilities | 1D, 2D, 3D point sets and images | [Details](Visualization-Tools-CFAVisual.md) |

**Main Features**:
- **1D Point Display**: Cantor Set and other 1D fractals
- **2D Point Display**: Sierpinski Triangle, Barnsley Fern, etc.
- **3D Point Display**: Menger Sponge and other 3D fractals
- **Image Display**: 2D and 3D image visualization

### 7. Fractal Sample Generator

Generate classic fractal patterns for testing, teaching, and artistic creation.

| Class | Description | Supported Patterns | Documentation |
|-------|-------------|-------------------|---------------|
| **CFASample** | Fractal Sample Generator | Cantor Set, Sierpinski Triangle, Barnsley Fern, Menger Sponge | [Details](Fractal-Sample-Generator-CFASample.md) |

**Supported Fractal Patterns**:
- **Cantor Set**: 1D fractal, dimension ≈ 0.63
- **Sierpinski Triangle**: 2D fractal, dimension ≈ 1.58
- **Barnsley Fern**: 2D fractal, dimension ≈ 1.67
- **Menger Sponge**: 3D fractal, dimension ≈ 2.73

**Applications**:
- Algorithm testing and validation
- Fractal geometry teaching
- Artistic creation

### 8. GPU Acceleration

All core computational modules provide GPU-accelerated versions for significant performance improvement.

| Feature | CPU Module | GPU Module | Speedup | Documentation |
|---------|------------|------------|---------|---------------|
| 2D Multifractal Spectrum | CFA2DMFS | CFA2DMFSGPU | 5-20x | [Details](GPU-Acceleration.md) |
| Image Fractal Dimension | CFAImageDimension | CFAImageDimensionGPU | 3-10x | [Details](GPU-Acceleration.md) |
| Image Lacunarity | CFAImageLacunarity | CFAImageLacunarityGPU | 5-15x | [Details](GPU-Acceleration.md) |

**Usage**:
```python
# Simple import replacement
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
# Rest of code remains identical!
```

**Requirements**:
- NVIDIA GPU (CUDA support)
- PyTorch with CUDA

## Command Line Usage

### Calculate Multifractal Spectrum

```bash
python demo.py --mode mfs --image ./images/face.png
```

### Calculate Fractal Dimension

```bash
python demo.py --mode fd --image ./images/fractal.png
```

### Lacunarity Analysis

```bash
python demo.py --mode lacunarity --image ./images/fractal.png
```

### Fourier Analysis

```bash
python demo.py --mode fourier --image ./images/face.png
```

### Series Analysis

```bash
python demo.py --mode series
```

### Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--image` | Input image path | Image file path |
| `--mode` | Analysis mode | `fd`, `mfs`, `lacunarity`, `fourier`, `series` |

## GPU Acceleration

Modules supporting GPU acceleration:

| Module | CPU Version | GPU Version |
|--------|-------------|-------------|
| 2D Multifractal Spectrum | `FA2DMFS.CFA2DMFS` | `FA2DMFSGPU.CFA2DMFSGPU` |
| Image Fractal Dimension | `FAImageDimension.CFAImageDimension` | `FAImageDimensionGPU.CFAImageDimensionGPU` |
| Image Lacunarity | `FAImageLacunarity.CFAImageLacunarity` | `FAImageLacunarityGPU.CFAImageLacunarityGPU` |

**Usage**:

```python
# Import GPU version
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension

# Use the same way as CPU version
MFS = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

**Performance Improvement**:
- Large images: 5-10x speedup
- Multi-scale analysis: 10-20x speedup
- Requires CUDA-enabled GPU

## FAQs

### Q: How to choose appropriate q range?
A: Generally use `q ∈ [-5, 5]`. Negative q is sensitive to sparse regions, positive q to dense regions.

### Q: Fractal dimension > 2?
A: Check image preprocessing. For 2D images, dimension should be in [1, 2].

### Q: How to improve computation speed?
A: (1) Use GPU version (2) Reduce number of q values (3) Reduce number of scales (4) Downsample large images

### Q: Multifractal spectrum not significant?
A: Check `Δh = h(-5) - h(5)` or spectrum width `Δα`. Small values may indicate monofractal data.

### Q: How to determine if data is multifractal?
A: (1) D(q) varies monotonically with q (2) f(α) is convex with certain width (3) Δh > 0.1

## Project Structure

```
FreeAeon-Fractal/
├── FreeAeonFractal/          # Core modules
│   ├── FA2DMFS.py            # 2D multifractal spectrum
│   ├── FA2DMFSGPU.py         # 2D multifractal spectrum (GPU)
│   ├── FA1DMFS.py            # 1D multifractal spectrum
│   ├── FAImageDimension.py   # Fractal dimension
│   ├── FAImageDimensionGPU.py# Fractal dimension (GPU)
│   ├── FAImageLacunarity.py  # Lacunarity
│   ├── FAImageLacunarityGPU.py # Lacunarity (GPU)
│   ├── FAImageFourier.py     # Fourier analysis
│   ├── FAImage.py            # Image utilities
│   └── __init__.py
├── demo.py                   # Command line interface
├── images/                   # Example images
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/LICENSE) for details.

## Authors

- **Jim Xie** - 📧 jim.xie.cn@outlook.com, xiewenwei@sina.com
- **Yin Jie** - 📧 yinjiejspi@163.com
- **Cindy Ma** - 📧 453303661@qq.com
- **Wenjing Zhang** - 📧 634676988@qq.com
- **Danny Zhang** - 📧 zhyzxsw@126.com

## Citation

If you use this project in academic work, please cite:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal

## Links

- 🔗 GitHub: https://github.com/jim-xie-cn/FreeAeon-Fractal
- 📦 PyPI: https://pypi.org/project/FreeAeon-Fractal/
- 📖 Documentation: https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/docs/markdown/en/index.md

## References

### Fractal Theory
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.

### Multifractal Analysis
- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. *Chaos and Fractals*.

### MFDFA Method
- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A*.

### Fractal Dimension
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image. *IEEE Transactions on Systems, Man, and Cybernetics*.

### Lacunarity
- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. *Physical Review E*.

---

© 2025 FreeAeon-Fractal. All rights reserved.
