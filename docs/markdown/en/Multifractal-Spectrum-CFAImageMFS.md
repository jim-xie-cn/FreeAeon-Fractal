# Multifractal Spectrum Analysis - CFAImageMFS

## Application Scenarios

The `CFAImageMFS` class is used to calculate the Multifractal Spectrum of 2D grayscale images, serving as an advanced tool for analyzing image complexity and heterogeneity. Main application scenarios include:

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
from FreeAeonFractal.FAImageMFS import CFAImageMFS

# Load and convert to grayscale
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Create multifractal spectrum analysis object
q_list = np.linspace(-5, 5, 26)
MFS = CFAImageMFS(gray_image, q_list=q_list)

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
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

# Rest of code remains the same
MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

### Local Alpha Map (per-pixel Hölder exponent)

```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS

MFS = CFAImageMFS(gray_image)
alpha_map = MFS.compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

CFAImageMFS.plot_alpha_map(alpha_map)
```

### Batch Processing

```python
import glob, cv2
from FreeAeonFractal.FAImageMFS import CFAImageMFS

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageMFS.get_batch_mfs(images, q_list=np.linspace(-5, 5, 26))
for df_mass, df_fit, df_spec in results:
    print(df_fit[['q','tau','Dq']].head())
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageMFS

**Description**: Class for calculating 2D grayscale image multifractal spectrum based on box-counting method.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input grayscale image (2D array) |
| `corp_type` | int | -1 | Image crop type (-1:crop, 0:strict match, 1:pad) |
| `q_list` | array-like | linspace(-5,5,51) | q value list |
| `with_progress` | bool | True | Whether to show progress bar |
| `bg_threshold` | float | 0.01 | Background threshold (pixels below this are masked) |
| `bg_reverse` | bool | False | Whether to reverse background threshold |
| `bg_otsu` | bool | False | Whether to use Otsu thresholding for background |
| `mu_floor` | float | 1e-12 | Minimum probability floor (kept for compatibility) |

#### Main Methods

##### 1. get_mass_table(max_size=None, max_scales=80, min_box=2, roi_mode="center")

**Description**: Compute the partition function table across all scales and q values.

**Parameters**:
- `max_size` (int): Maximum box size
- `max_scales` (int): Maximum number of scales
- `min_box` (int): Minimum box size
- `roi_mode` (str): ROI crop mode, `"center"` uses fixed square ROI (Scheme A)

**Return Value** (DataFrame): Columns `scale, eps, q, value, kind`
- `kind` ∈ `{'logMq', 'N', 'S'}` — log partition function, box count, Shannon entropy

##### 2. fit_tau_and_D1(df_mass, min_points=6, require_common_scales=True, use_middle_scales=False, fit_scale_frac=(0.2, 0.8), if_auto_line_fit=False, auto_fit_min_len_ratio=0.5, cap_d0_at_2=True)

**Description**: Fit τ(q) and D(q) from the mass table.

**Return Value** (DataFrame): Columns `q, tau, Dq, D1, intercept, r_value, p_value, std_err, n_points`

**Notes**:
- q=0 row uses log(N) fit; q=1 row uses Shannon entropy (S) fit; other q use logMq fit
- Forced q=0 and q=1 rows are always included in `df_fit`

##### 3. alpha_falpha_from_tau(df_fit, spline_k=3, exclude_q1=True, spline_s=0)

**Description**: Compute α(q) and f(α) via spline derivative of τ(q).

**Parameters**:
- `df_fit`: Output from `fit_tau_and_D1`
- `spline_k` (int): Spline order (default 3)
- `exclude_q1` (bool): Whether to exclude q=1 from spline (avoids discontinuity)
- `spline_s` (float): Spline smoothing factor (0 = exact interpolation)

**Return Value** (DataFrame): Columns `q, tau, Dq, alpha, f_alpha`

##### 4. get_mfs(max_size=None, max_scales=80, min_points=6, min_box=2, use_middle_scales=False, fit_scale_frac=(0.2, 0.8), if_auto_line_fit=False, auto_fit_min_len_ratio=0.5, spline_s=0, cap_d0_at_2=True)

**Description**: Complete multifractal spectrum analysis pipeline.

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

##### 5. compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

**Description**: Compute per-pixel local Hölder exponent α via streaming OLS with nested-grid optimization.

**Parameters**:
- `scales`: Scale list (None = auto)
- `roi_mode` (str): ROI mode
- `empty_policy` (str): Policy for empty boxes — `"nan"` or `"zero"`

**Return Value**: 2D numpy array of local α values

##### 6. compute_alpha_map_batch(images, ...) [static]

**Description**: Batch computation of alpha maps.

##### 7. plot(df_mass, df_fit, df_spec)

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
   - Use `bg_threshold` to mask background; use `bg_otsu=True` for automatic background detection

2. **q Value Selection**:
   - Negative q values are sensitive to sparse regions
   - Positive q values are sensitive to dense regions
   - Recommended range: -5 to 5
   - Recommended points: 20-50

3. **Scale Parameters**:
   - `max_scales` affects sampling density, recommend 60-80
   - `min_box` should not be less than 2
   - `roi_mode="center"` uses a fixed square ROI to avoid edge effects

4. **Result Interpretation**:
   - Wider f(α) curve indicates stronger multifractality
   - Δα = α_max - α_min quantifies heterogeneity
   - D(q) monotonically decreasing indicates multifractality
   - r_value close to 1 indicates good fit

5. **Performance Optimization**:
   - Use GPU version (`CFAImageMFSGPU`) for significant speedup
   - GPU version uses `q_chunk` and `img_chunk` to control VRAM usage
   - Default dtype: float64 for single-image, float32 for batch

## References

- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. *Chaos and Fractals*.
