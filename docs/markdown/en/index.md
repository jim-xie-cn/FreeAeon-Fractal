# FreeAeon-Fractal Documentation

## Project Overview

**FreeAeon-Fractal** is a Python toolkit for computing **Multifractal Spectra**, **Fractal Dimensions**, **Lacunarity**, and **Fourier Spectra** of images and time series. It is the first GPU-accelerated package of its kind.

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
from FreeAeonFractal.FAImageMFS import CFAImageMFS

rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
MFS.plot(df_mass, df_fit, df_spec)
```

### Fractal Dimensions of an Image

```python
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImage import CFAImage

bin_image, threshold = CFAImage.otsu_binarize(gray_image)

fd_bc = CFAImageFD(bin_image).get_bc_fd()
fd_dbc = CFAImageFD(gray_image).get_dbc_fd()
fd_sdbc = CFAImageFD(gray_image).get_sdbc_fd()

CFAImageFD.plot(rgb_image, gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### Local Alpha Map (per-pixel Hölder exponent)

```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS

MFS = CFAImageMFS(gray_image)
alpha_map, info = MFS.compute_alpha_map()
CFAImageMFS.plot_alpha_map(alpha_map)
```

### Lacunarity Analysis

```python
from FreeAeonFractal.FAImageLAC import CFAImageLAC

calc = CFAImageLAC(gray_image, partition_mode="gliding")
lac_result = calc.get_lacunarity()
fit_result = calc.fit_lacunarity(lac_result)
calc.plot(lac_result, fit_result)
```

### Fourier Analysis

```python
from FreeAeonFractal.FAImageFourier import CFAImageFourier

fourier = CFAImageFourier(rgb_image)
mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)
full_reconstructed = fourier.get_reconstruct()
fourier.plot(raw_magnitude_disp=mag_disp, raw_phase_disp=phase_disp, full_reconstructed=full_reconstructed)
```

### Multifractal Spectrum of a Time Series

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

x = np.cumsum(np.random.randn(5000))
mfs = CFASeriesMFS(x, q_list=np.linspace(-5, 5, 21))
df_mfs = mfs.get_mfs()
mfs.plot(df_mfs)
```

## Feature Modules

### 1. Multifractal Spectrum Analysis

| Class | Description | Applications | Documentation |
|-------|-------------|--------------|---------------|
| **CFAImageMFS** | 2D Image Multifractal Spectrum | Texture analysis, medical imaging, materials science | [Details](Multifractal-Spectrum-CFAImageMFS.md) |
| **CFASeriesMFS** | 1D Series Multifractal Spectrum | Financial time series, physiological signals, climate data | [Details](Series-Multifractal-CFASeriesMFS.md) |

**Core Concepts**:
- **τ(q)** Mass Exponent: Scaling behavior for different q orders
- **D(q)** Generalized Dimension: Dimension characteristics at different q
- **α(q)** Singularity Strength: Local singularity of data
- **f(α)** Multifractal Spectrum: Complete characterization of multifractal properties

**GPU Acceleration**:
```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
```

### 2. Fractal Dimension Analysis

| Class | Description | Methods | Documentation |
|-------|-------------|---------|---------------|
| **CFAImageFD** | Image Fractal Dimension | BC, DBC, SDBC | [Details](Fractal-Dimension-CFAImageFD.md) |

**Three Methods**:
- **BC** (Box-Counting): For binary images
- **DBC** (Differential Box-Counting): For grayscale images (Sarkar & Chaudhuri 1994)
- **SDBC** (Shifted DBC): Improved version with better accuracy at small scales (Chen et al. 1995)

**Dimension Range**:
- 1D lines: D ≈ 1
- 2D planes: D ≈ 2
- Fractal textures: 1 < D < 2

### 3. Lacunarity Analysis

| Class | Description | Partition Modes | Documentation |
|-------|-------------|-----------------|---------------|
| **CFAImageLAC** | Image Lacunarity Analysis | Gliding, Non-overlapping | [Details](Lacunarity-Analysis-CFAImageLAC.md) |

**Partition Modes**:
- **Gliding Box**: Sliding windows with integral image speedup — accurate and efficient
- **Non-overlapping Box**: Disjoint blocks — faster for large images

**Lacunarity Meaning**:
- Λ = 1: Completely uniform distribution
- Λ > 1: Presence of gaps and clustering
- Larger Λ: Higher spatial heterogeneity

### 4. Fourier Analysis

| Class | Description | Features | Documentation |
|-------|-------------|----------|---------------|
| **CFAImageFourier** | Image Fourier Analysis | Spectrum analysis, filtering, reconstruction | [Details](Fourier-Analysis-CFAImageFourier.md) |

**Main Features**:
- Magnitude and phase spectrum computation
- Frequency domain visualization (log-scale enhanced)
- Custom frequency mask filtering
- Image reconstruction from frequency domain

### 5. Image Processing Utilities

| Class | Description | Main Methods | Documentation |
|-------|-------------|--------------|---------------|
| **CFAImage** | Image Processing Utilities | Blocking, merging, binarization, ROI | [Details](Image-Utils-CFAImage.md) |

**Core Functions**:
- **Otsu Auto-Thresholding**: Automatic threshold segmentation
- **Image Blocking/Merging**: Support for grayscale and color images
- **Mask Generation**: Generate masks based on block positions
- **Random Patch Sampling**: For data augmentation
- **ROI Extraction**: Based on multifractal measure reweighting

### 6. Visualization Tools

| Class | Description | Supported Dimensions | Documentation |
|-------|-------------|---------------------|---------------|
| **CFAVisual** | Visualization Utilities | 1D, 2D, 3D point sets and images | [Details](Visualization-Tools-CFAVisual.md) |

**Main Features**:
- **1D Point Display**: Cantor Set and other 1D fractals
- **2D Point Display**: Sierpinski Triangle, Barnsley Fern, etc.
- **3D Point Display**: Menger Sponge and other 3D fractals
- **Image Display**: 2D and 3D image visualization

### 7. Fractal Sample Generator

| Class | Description | Supported Patterns | Documentation |
|-------|-------------|-------------------|---------------|
| **CFASample** | Fractal Sample Generator | Cantor Set, Sierpinski Triangle, Barnsley Fern, Menger Sponge | [Details](Fractal-Sample-Generator-CFASample.md) |

**Supported Fractal Patterns**:
- **Cantor Set**: 1D fractal, dimension ≈ 0.63
- **Sierpinski Triangle**: 2D fractal, dimension ≈ 1.58
- **Barnsley Fern**: 2D fractal, dimension ≈ 1.67
- **Menger Sponge**: 3D fractal, dimension ≈ 2.73

### 8. GPU Acceleration

| Feature | CPU Module | GPU Module | Speedup | Documentation |
|---------|------------|------------|---------|---------------|
| 2D Multifractal Spectrum | CFAImageMFS | CFAImageMFSGPU | 5-20× | [Details](GPU-Acceleration.md) |
| Image Fractal Dimension | CFAImageFD | CFAImageFDGPU | 3-10× | [Details](GPU-Acceleration.md) |
| Image Lacunarity | CFAImageLAC | CFAImageLACGPU | 5-15× | [Details](GPU-Acceleration.md) |

```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
```

## Command Line Usage

```bash
# Multifractal Spectrum
python demo.py --mode mfs --image ./images/face.png

# Fractal Dimension
python demo.py --mode fd --image ./images/fractal.png

# Lacunarity
python demo.py --mode lacunarity --image ./images/fractal.png

# Fourier Analysis
python demo.py --mode fourier --image ./images/face.png

# Local Alpha Map
python demo.py --mode alpha --image ./images/face.png

# Series Multifractal
python demo.py --mode series
```

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--image` | Input image path | Image file path |
| `--mode` | Analysis mode | `fd`, `mfs`, `alpha`, `lacunarity`, `fourier`, `series` |

## demo.py Examples

The following are the complete Python code examples from `demo.py`, corresponding to each `--mode`.

### mode=fd — Fractal Dimension

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

# --- Single image ---
t0 = time.time()
fd_bc   = CFAImageFD(bin_image,  max_scales=max_scales).get_bc_fd(corp_type=-1)
fd_dbc  = CFAImageFD(gray_image, max_scales=max_scales).get_dbc_fd(corp_type=-1)
fd_sdbc = CFAImageFD(gray_image, max_scales=max_scales).get_sdbc_fd(corp_type=-1)
print(f"Single (1) img: {time.time()-t0:.3f}s")
print("  BC:",   fd_bc['fd'])
print("  DBC:",  fd_dbc['fd'])
print("  SDBC:", fd_sdbc['fd'])
CFAImageFD.plot(rgb_image, gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)

# --- Batch (100 images) ---
bin_imgs  = [bin_image]  * 100
gray_imgs = [gray_image] * 100
t0 = time.time()
bc_list   = CFAImageFD.get_batch_bc(bin_imgs,   max_scales=max_scales, with_progress=False)
dbc_list  = CFAImageFD.get_batch_dbc(gray_imgs,  max_scales=max_scales, with_progress=False)
sdbc_list = CFAImageFD.get_batch_sdbc(gray_imgs, max_scales=max_scales, with_progress=False)
print(f"Batch (100 imgs): {time.time()-t0:.3f}s")
print(f"  batch BC FD[99]   = {bc_list[99]['fd']:.4f}")
print(f"  batch DBC FD[99]  = {dbc_list[99]['fd']:.4f}")
print(f"  batch SDBC FD[99] = {sdbc_list[99]['fd']:.4f}")
CFAImageFD.plot(rgb_image, gray_image, bin_image, bc_list[99], dbc_list[99], sdbc_list[99])
```

### mode=mfs — Multifractal Spectrum

```python
import cv2, time
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

q_list = np.linspace(-10, 10, 101)

# --- Single image ---
t0 = time.time()
MFS = CFAImageMFS(gray_image, q_list=q_list)
df_mass, df_fit, df_spec = MFS.get_mfs()
print(f"Single MFS (1) imgs: {time.time()-t0:.3f}s")
print(df_fit.head())
MFS.plot(df_mass, df_fit, df_spec)

# --- Batch (20 images) ---
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
print(f"Batch MFS (20) imgs: {time.time()-t0:.3f}s")
print(df_fit1.head())
MFS.plot(df_mass1, df_fit1, df_spec1)
```

### mode=alpha — Local Multifractal α-map

```python
import cv2, time
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

q_list = np.linspace(-5, 5, 51)
scales = list(range(1, 100))

# --- Single image ---
t0 = time.time()
MFS = CFAImageMFS(gray_image, q_list=q_list)
alpha_map, info = MFS.compute_alpha_map(scales=scales)
print(f"Single alpha_map (1) imgs: {time.time()-t0:.3f}s")
print("  alpha map:", alpha_map)
print("  scale info:", info)
CFAImageMFS.plot_alpha_map(alpha_map)

# --- Batch (20 images) ---
t0 = time.time()
imgs = [gray_image] * 20
batch_alpha_map = CFAImageMFS.compute_alpha_map_batch(imgs, with_progress=False, scales=scales)
alpha_maps = batch_alpha_map[0]
infos      = batch_alpha_map[1]
print(f"Batch alpha_map (20) imgs: {time.time()-t0:.3f}s")
print("  alpha map:", alpha_maps[0])
print("  scale info:", infos)
CFAImageMFS.plot_alpha_map(alpha_maps[0])
```

### mode=lacunarity — Lacunarity Analysis

```python
import cv2, time
from FreeAeonFractal.FAImageLAC import CFAImageLAC

image_path = './images/fractal.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

lacunarity = CFAImageLAC(gray_image, max_scales=256, with_progress=True)

# --- Single image ---
t0 = time.time()
lac_gray  = lacunarity.get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)
fit_gray  = lacunarity.fit_lacunarity(lac_gray)
print(f"Single lacunarity (1) imgs: {time.time()-t0:.3f}s")
print("  Gray lacunarity:", lac_gray["lacunarity"])
print("  Fit slope:", fit_gray["slope"],
      "intercept:", fit_gray["intercept"],
      "R:", fit_gray["r_value"],
      "P:", fit_gray["p_value"])
lacunarity.plot(lac_gray, fit_gray)

# --- Batch (100 images) ---
t0 = time.time()
imgs = [gray_image] * 100
batchs = CFAImageLAC.get_batch_lacunarity(
    imgs, scales_mode="powers", partition_mode="gliding",
    use_binary_mass=False, with_progress=False
)
fits = CFAImageLAC.fit_batch_lacunarity(batchs)
print(f"Batch lacunarity (100) imgs: {time.time()-t0:.3f}s")
print("  Gray lacunarity:", batchs[99]["lacunarity"])
print("  Fit slope:", fits[99]["slope"],
      "intercept:", fits[99]["intercept"],
      "R:", fits[99]["r_value"],
      "P:", fits[99]["p_value"])
lacunarity.plot(batchs[99], fits[99])
```

### mode=fourier — Fourier Analysis

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

image_path = './images/face.png'
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Supports both grayscale and RGB
fourier = CFAImageFourier(rgb_image)

# Get raw spectrum (for reconstruction)
raw_mag, raw_phase = fourier.get_raw_spectrum()

# Get display spectrum (for visualization)
raw_mag_disp, raw_phase_disp = fourier.get_display_spectrum(alpha=1.5)

# Create a custom frequency mask (keep odd-frequency components as example)
h, w = raw_mag[0].shape
Y, X = np.ogrid[:h, :w]
mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

# Get masked display spectrum
customized_mag_list   = raw_mag   * mask
customized_phase_list = raw_phase * mask
customized_mag_disp, customized_phase_disp = fourier.get_display_spectrum(
    alpha=1.5,
    magnitude=customized_mag_list,
    phase=customized_phase_list
)

# Reconstruct full image from raw spectrum
full_reconstructed = fourier.get_reconstruct()

# Reconstruct image using frequency mask
masked_reconstructed = fourier.extract_by_freq_mask(mask)

# Display all results
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

### mode=series — Series Multifractal Spectrum

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

# Generate a random walk series (example data)
x = np.cumsum(np.random.randn(5000))

q = np.linspace(-5, 5, 21)
mfs = CFASeriesMFS(x)
df_mfs = mfs.get_mfs()
mfs.plot(df_mfs)
print(df_mfs)
```

## Advanced Usage

### Batch Processing

```python
import glob, cv2, numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageMFS.get_batch_mfs(images, q_list=np.linspace(-5, 5, 26))
```

### Feature Extraction for Machine Learning

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

## FAQs

### Q: How to choose appropriate q range?
A: Generally use `q ∈ [-5, 5]`. Negative q is sensitive to sparse regions, positive q to dense regions.

### Q: Fractal dimension > 2?
A: Check image preprocessing. For 2D images, dimension should be in [1, 2].

### Q: How to improve computation speed?
A: (1) Use GPU version (2) Reduce number of q values (3) Reduce number of scales (4) Downsample large images.

### Q: Multifractal spectrum not significant?
A: Check `Δh = h(-5) - h(5)` or spectrum width `Δα`. Small values may indicate monofractal data.

### Q: How to determine if data is multifractal?
A: (1) D(q) varies monotonically with q (2) f(α) is convex with certain width (3) Δh > 0.1.

## Project Structure

```
FreeAeon-Fractal/
├── FreeAeonFractal/          # Core modules
│   ├── FAImageMFS.py         # 2D multifractal spectrum
│   ├── FAImageMFSGPU.py      # 2D multifractal spectrum (GPU)
│   ├── FASeriesMFS.py        # 1D multifractal spectrum
│   ├── FAImageFD.py          # Fractal dimension
│   ├── FAImageFDGPU.py       # Fractal dimension (GPU)
│   ├── FAImageLAC.py         # Lacunarity
│   ├── FAImageLACGPU.py      # Lacunarity (GPU)
│   ├── FAImageFourier.py     # Fourier analysis
│   ├── FAImage.py            # Image utilities
│   ├── FASample.py           # Fractal sample generator
│   ├── FAVisual.py           # Visualization tools
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

## Citation

If you use this project in academic work, please cite:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal

## Links

- 🔗 GitHub: https://github.com/jim-xie-cn/FreeAeon-Fractal
- 📦 PyPI: https://pypi.org/project/FreeAeon-Fractal/

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
- Chen, W. S., et al. (1995). Efficient fractal coding of images based on differential box counting. *Pattern Recognition*.

### Lacunarity
- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. *Physical Review A*.
- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. *Physical Review E*.

---

© 2025 FreeAeon-Fractal. All rights reserved.
