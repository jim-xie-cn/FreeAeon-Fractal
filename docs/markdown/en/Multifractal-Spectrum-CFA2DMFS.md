# Multifractal Spectrum Analysis - CFA2DMFS

## Application Scenarios

The `CFA2DMFS` class is used to calculate the Multifractal Spectrum of 2D grayscale images, serving as an advanced tool for analyzing image complexity and heterogeneity. Main application scenarios include:

- **Texture Analysis**: Quantify multi-scale properties of image textures
- **Medical Imaging**: Analyze tissue structure heterogeneity
- **Materials Science**: Study multifractal features of material surfaces
- **Financial Analysis**: Analyze multifractal properties of price fluctuations
- **Earth Sciences**: Study multi-scale features of terrain and landforms

## Usage Examples

### Basic Usage

```python
import cv2
import numpy as np
from FreeAeonFractal.FA2DMFS import CFA2DMFS

# Load and convert to grayscale
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Create multifractal spectrum analysis object
q_list = np.linspace(-5, 5, 26)
MFS = CFA2DMFS(gray_image, q_list=q_list)

# Calculate multifractal spectrum
df_mass, df_fit, df_spec = MFS.get_mfs()

# View results
print(df_spec)

# Visualize results
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU Accelerated Version

```python
# Use GPU accelerated version
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS

# Rest of code remains the same
MFS = CFA2DMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFA2DMFS

**Description**: Class for calculating 2D grayscale image multifractal spectrum based on box-counting method.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input grayscale image (2D array) |
| `corp_type` | int | 0 | Image crop type (-1:crop, 0:strict match, 1:pad) |
| `q_list` | array-like | linspace(-5,5,51) | q value list |
| `with_progress` | bool | True | Whether to show progress bar |
| `bg_threshold` | float | 0.01 | Background threshold |
| `bg_reverse` | bool | False | Whether to reverse background threshold |
| `bg_otsu` | bool | False | Whether to use Otsu thresholding |
| `mu_floor` | float | 1e-12 | Minimum probability value (kept for compatibility) |

#### Main Methods

##### 1. get_mfs(max_size=None, max_scales=80, min_points=6, min_box=2, ...)

**Description**: Complete multifractal spectrum analysis pipeline, returning mass table, fit results, and spectrum parameters.

**Parameters**:
- `max_size` (int): Maximum box size
- `max_scales` (int): Maximum number of scales
- `min_points` (int): Minimum number of points for fitting
- `min_box` (int): Minimum box size
- `use_middle_scales` (bool): Whether to use only middle scales
- `fit_scale_frac` (tuple): Fitting scale range (0.2, 0.8)
- `if_auto_line_fit` (bool): Whether to auto linear fit
- `auto_fit_min_len_ratio` (float): Auto fit minimum length ratio
- `spline_s` (float): Spline smoothing parameter
- `cap_d0_at_2` (bool): Whether to cap D0 ≤ 2

**Return Value** (tuple):
```python
(df_mass, df_fit, df_spec)
```

- `df_mass`: Mass table DataFrame with columns:
  - `scale`: Box size
  - `eps`: Normalized scale
  - `q`: q value
  - `value`: Different meanings based on kind
  - `kind`: Value type ('logMq', 'N', 'S')

- `df_fit`: Fit results DataFrame with columns:
  - `q`: q value
  - `tau`: τ(q) mass exponent
  - `Dq`: D(q) generalized dimension
  - `D1`: D1 information dimension (only for q=1)
  - `intercept`: Fit intercept
  - `r_value`: Correlation coefficient
  - `p_value`: p-value
  - `std_err`: Standard error
  - `n_points`: Number of fit points

- `df_spec`: Multifractal spectrum DataFrame with columns:
  - `q`: q value
  - `tau`: τ(q)
  - `Dq`: D(q)
  - `alpha`: α(q) singularity strength
  - `f_alpha`: f(α) multifractal spectrum

##### 2. plot(df_mass, df_fit, df_spec)

**Description**: Visualize multifractal spectrum analysis results with 6 subplots:
1. log M(q, ε) heatmap
2. f(α) vs α multifractal spectrum
3. τ(q) vs q
4. D(q) vs q generalized dimension
5. α vs q
6. f(α) vs q

## Theoretical Background

### Multifractal Spectrum

The multifractal spectrum describes the statistical properties of an image at different scales. Core parameters include:

#### 1. Partition Function
```
M(q, ε) = Σ μᵢ^q
```
Where μᵢ is the normalized mass of the i-th box.

#### 2. Mass Exponent τ(q)
```
τ(q) = lim(ε→0) log M(q, ε) / log ε
```

#### 3. Generalized Dimension D(q)
```
D(q) = τ(q) / (q - 1), q ≠ 1
D(1) = lim(ε→0) Σ μᵢ log μᵢ / log ε  (information dimension)
```

Special values:
- D(0): Capacity dimension
- D(1): Information dimension
- D(2): Correlation dimension

#### 4. Multifractal Spectrum f(α)
```
α(q) = dτ(q)/dq  (singularity strength)
f(α) = qα - τ(q)  (multifractal spectrum)
```

## Important Notes

1. **Image Preprocessing**:
   - Input must be 2D grayscale image
   - Image is automatically normalized to [0, 1]
   - Can use `bg_threshold` to remove background

2. **q Value Selection**:
   - Negative q values are sensitive to sparse regions
   - Positive q values are sensitive to dense regions
   - Recommended range: -5 to 5
   - Recommended points: 20-50

3. **Scale Parameters**:
   - `max_scales` affects sampling density, recommend 60-100
   - `min_box` should not be less than 2
   - Use fixed ROI to avoid edge effects

4. **Result Interpretation**:
   - Wider f(α) curve indicates stronger multifractality
   - Δα = α_max - α_min quantifies heterogeneity
   - D(q) monotonically decreasing indicates multifractality
   - r_value close to 1 indicates good fit

5. **Performance Optimization**:
   - Use GPU version for significant speedup
   - Downsample large images
   - Reduce number of q values to improve speed

## References

- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. Physical Review Letters.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. Chaos and Fractals.
