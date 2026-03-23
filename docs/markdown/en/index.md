# FreeAeon-Fractal Documentation Index

## Overview

FreeAeon-Fractal is a powerful Python toolkit for computing **Multifractal Spectra**, **Fractal Dimensions**, **Lacunarity**, and **Fourier Spectra** of images or time series. This toolkit provides both CPU and GPU implementations, suitable for data analysis tasks of various scales.

## Core Functional Modules

### 1. Multifractal Spectrum Analysis

- **[CFA1DMFS](./MultifractalSpectrum-CFA1DMFS.md)** - 1D Time Series Multifractal Spectrum Analysis
  - Use Cases: Financial time series, physiological signal analysis, seismic data analysis
  - Key Features: Compute h(q), τ(q), α(q), f(α), and D(q)

- **[CFA2DMFS](./MultifractalSpectrum-CFA2DMFS.md)** - 2D Image Multifractal Spectrum (CPU Version)
  - Use Cases: Medical imaging, remote sensing, texture analysis
  - Key Features: Box-counting based multifractal spectrum computation

- **[CFA2DMFSGPU](./MultifractalSpectrum-CFA2DMFSGPU.md)** - 2D Image Multifractal Spectrum (GPU Accelerated)
  - Use Cases: Large-scale image batch processing, real-time analysis
  - Key Features: GPU-accelerated multifractal computation, batch processing support

### 2. Fractal Dimension Calculation

- **[CFAImageDimension](./FractalDimension-CFAImageDimension.md)** - Fractal Dimension Calculation (CPU Version)
  - Use Cases: Image complexity assessment, texture feature extraction, morphological analysis
  - Methods: BC (Box Counting), DBC (Differential Box Counting), SDBC (Simplified DBC)

- **[CFAImageDimensionGPU](./FractalDimension-CFAImageDimensionGPU.md)** - Fractal Dimension (GPU Accelerated)
  - Use Cases: High-resolution image analysis, batch image processing
  - Key Features: GPU-accelerated BC/DBC/SDBC, batch processing support

### 3. Lacunarity Analysis

- **[CFAImageLacunarity](./Lacunarity-CFAImageLacunarity.md)** - Image Lacunarity Analysis (CPU Version)
  - Use Cases: Texture heterogeneity analysis, spatial distribution features, image quality assessment
  - Methods: Gliding box and non-overlapping box partition

- **[CFAImageLacunarityGPU](./Lacunarity-CFAImageLacunarityGPU.md)** - Image Lacunarity (GPU Accelerated)
  - Use Cases: Large-scale image dataset analysis, real-time lacunarity computation
  - Key Features: GPU-accelerated lacunarity, batch processing support

### 4. Fourier Frequency Domain Analysis

- **[CFAImageFourier](./FourierAnalysis-CFAImageFourier.md)** - Fourier Frequency Domain Analysis
  - Use Cases: Frequency component analysis, image filtering, frequency feature extraction
  - Key Features: Magnitude and phase spectrum, frequency reconstruction, frequency masking

### 5. Image Preprocessing Tools

- **[CFAImage](./ImageProcessing-CFAImage.md)** - Image Preprocessing and ROI Extraction
  - Use Cases: Image preprocessing, ROI extraction, data augmentation
  - Key Features: Crop/pad, blocking, Otsu binarization, q-based ROI extraction, random sampling

### 6. Fractal Sample Generation

- **[CFASample](./SampleGeneration-CFASample.md)** - Classic Fractal Pattern Generator
  - Use Cases: Algorithm testing, teaching demonstrations, benchmark data generation
  - Supported Fractals: Cantor Set, Sierpinski Triangle, Barnsley Fern, Menger Sponge

### 7. Visualization Tools

- **[CFAVisual](./Visualization-CFAVisual.md)** - Point Set and Image Visualization
  - Use Cases: Result presentation, data exploration, visual analysis
  - Key Features: 1D/2D/3D point visualization, image display

## Quick Start

### Installation

```bash
pip install FreeAeon-Fractal
```

### Basic Usage Examples

#### Compute Multifractal Spectrum of an Image

```python
import cv2
import numpy as np
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# Load image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Create multifractal spectrum analyzer
mfs = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))

# Compute multifractal spectrum
df_mass, df_fit, df_spec = mfs.get_mfs()

# Visualize results
mfs.plot(df_mass, df_fit, df_spec)
```

#### Compute Fractal Dimension of an Image

```python
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImage import CFAImage

# Binarize image
bin_image, threshold = CFAImage.otsu_binarize(image)

# Compute fractal dimensions
fd_bc = CFAImageDimension(bin_image).get_bc_fd()
fd_dbc = CFAImageDimension(image).get_dbc_fd()
fd_sdbc = CFAImageDimension(image).get_sdbc_fd()

print(f"BC Fractal Dimension: {fd_bc['fd']:.4f}")
print(f"DBC Fractal Dimension: {fd_dbc['fd']:.4f}")
print(f"SDBC Fractal Dimension: {fd_sdbc['fd']:.4f}")
```

## GPU Acceleration

For large-scale data processing, use GPU-accelerated versions:

```python
# GPU version of multifractal spectrum analysis
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS

# GPU version of fractal dimension calculation
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension
```

## Technical Features

- **Multifractal Theory**: Based on state-of-the-art box-counting multifractal algorithms
- **High Performance**: Supports both CPU and GPU implementations
- **Batch Processing**: Supports multi-image batch analysis
- **Numerical Stability**: Uses logsumexp and other numerical stability techniques
- **Ease of Use**: Simple API design with comprehensive documentation

## Application Domains

- **Medical Imaging**: Tumor texture analysis, pathology image classification
- **Remote Sensing**: Surface texture analysis, land use classification
- **Materials Science**: Surface morphology analysis, microstructure characterization
- **Financial Analysis**: Market volatility analysis, risk assessment
- **Signal Processing**: Physiological signal analysis, seismic data processing

## Citation

If you use this toolkit in academic work, please cite:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal

## Contact

- **Author**: Jim Xie
- **Email**: jim.xie.cn@outlook.com, xiewenwei@sina.com
- **GitHub**: https://github.com/jim-xie-cn/FreeAeon-Fractal

## License

MIT License - See [LICENSE](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/LICENSE)

---

**Documentation Version**: 1.0
**Last Updated**: 2026-03-23
