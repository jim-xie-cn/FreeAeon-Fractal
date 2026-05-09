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
q_list = np.linspace(-5, 5, 51)
MFS = CFAImageMFS(gray_image, q_list=q_list)

# Calculate multifractal spectrum (one call runs full pipeline)
df_mass, df_fit, df_spec = MFS.get_mfs()

# View results
print(df_spec[['q', 'alpha', 'f_alpha']].head(10))

# Visualize results
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU Accelerated Version

```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

MFS = CFAImageMFS(gray_image, q_list=np.linspace(-5, 5, 51))
df_mass, df_fit, df_spec = MFS.get_mfs()
MFS.plot(df_mass, df_fit, df_spec)
```

### Local Alpha Map (per-pixel Hölder exponent)

```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS

MFS = CFAImageMFS(gray_image)

# Compute per-pixel local singularity exponent
alpha_map, info = MFS.compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

# Visualize
CFAImageMFS.plot_alpha_map(alpha_map)
```

### Batch Processing

```python
import glob, cv2
import numpy as np
from FreeAeonFractal.FAImageMFS import CFAImageMFS

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]
results = CFAImageMFS.get_batch_mfs(images, q_list=np.linspace(-5, 5, 26))

for df_mass, df_fit, df_spec in results:
    print(df_fit[['q', 'tau', 'Dq']].head())
```

### Feature Extraction for Machine Learning

```python
def extract_mf_features(image):
    mfs = CFAImageMFS(image, q_list=np.linspace(-5, 5, 11))
    _, df_fit, df_spec = mfs.get_mfs()

    features = {
        'D0': df_fit.loc[df_fit['q'].round(2) == 0.0, 'Dq'].values[0],
        'D1': df_fit.loc[df_fit['q'].round(2) == 1.0, 'D1'].values[0],
        'D2': df_fit.loc[df_fit['q'].round(2) == 2.0, 'Dq'].values[0],
        'alpha_max': df_spec['alpha'].max(),
        'alpha_min': df_spec['alpha'].min(),
        'delta_alpha': df_spec['alpha'].max() - df_spec['alpha'].min(),
        'f_alpha_max': df_spec['f_alpha'].max()
    }
    return features
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageMFS

**Description**: Box-counting multifractal analysis on a 2D grayscale image. Uses a fixed square ROI (Scheme A) to ensure consistent ε normalization across all scales.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input 2D grayscale image |
| `corp_type` | int | -1 | Image cropping method (-1:crop, 0:strict, 1:pad) |
| `q_list` | array-like | linspace(-5,5,51) | q value list |
| `with_progress` | bool | True | Whether to show progress bar |
| `bg_threshold` | float | 0.01 | Background threshold (pixels below are zeroed after normalization) |
| `bg_reverse` | bool | False | If True, zero pixels *above* bg_threshold instead |
| `bg_otsu` | bool | False | Apply Otsu thresholding on raw image to remove background first |
| `mu_floor` | float | 1e-12 | Kept for API compatibility; not used for mu flooring |

**Notes on image preprocessing**: The image is normalized to [0, 1] before analysis. Background masking via `bg_threshold` or `bg_otsu` is applied after normalization.

#### Main Methods

##### 1. get_mass_table(max_size=None, max_scales=80, min_box=2, roi_mode="center")

**Description**: Compute the per-scale partition function table.

**Parameters**:
- `max_size` (int): Maximum box size
- `max_scales` (int): Maximum number of candidate scales
- `min_box` (int): Minimum box size
- `roi_mode` (str): `"center"` (default) or `"topleft"` — how to crop the fixed square ROI

**Return Value** (DataFrame): Columns `scale, eps, q, value, kind`
- `kind='logMq'`: log∑μᵢ^q for q≠0,1
- `kind='N'`: Count of non-zero boxes for q=0
- `kind='S'`: −∑μᵢ log μᵢ (Shannon entropy) for q=1

##### 2. fit_tau_and_D1(df_mass, min_points=6, ...)

**Description**: Fit τ(q) and D(q) from the mass table via linear regression.

**Key Parameters**:
- `min_points` (int): Minimum scale points required for a valid fit
- `require_common_scales` (bool): Restrict all q to the same scale set
- `use_middle_scales` (bool): Use only the middle fraction of scales (avoids saturation at extremes)
- `fit_scale_frac` (tuple): `(low, high)` fractions of the scale range to use when `use_middle_scales=True`
- `if_auto_line_fit` (bool): Automatically find the best linear segment
- `cap_d0_at_2` (bool): Iteratively remove extreme small-scale points until D0 ≤ 2

**Return Value** (DataFrame): Columns `q, tau, Dq, D1, intercept, r_value, p_value, std_err, n_points`

##### 3. alpha_falpha_from_tau(df_fit, spline_k=3, exclude_q1=True, spline_s=0)

**Description**: Compute α(q) and f(α) via spline derivative of τ(q) (Legendre transform).

**Parameters**:
- `spline_k` (int): Spline order (3 = cubic, default)
- `exclude_q1` (bool): Exclude q=1 from spline to avoid discontinuity
- `spline_s` (float): Smoothing factor (0 = exact interpolation)

**Return Value** (DataFrame): Columns `q, tau, Dq, alpha, f_alpha`

##### 4. get_mfs(max_size=None, max_scales=80, min_points=6, min_box=2, ...)

**Description**: Complete multifractal spectrum analysis pipeline (calls `get_mass_table` → `fit_tau_and_D1` → `alpha_falpha_from_tau`).

**Return Value** (tuple of three DataFrames):

**`df_mass`** — raw partition table:
| Column | Description |
|--------|-------------|
| `scale` | Box size (pixels) |
| `eps` | Normalized scale ε = scale / L |
| `q` | Moment order |
| `value` | logMq, N, or S depending on `kind` |
| `kind` | `'logMq'`, `'N'`, or `'S'` |

**`df_fit`** — regression results:
| Column | Description |
|--------|-------------|
| `q` | Moment order |
| `tau` | τ(q) mass exponent |
| `Dq` | D(q) generalized dimension |
| `D1` | D₁ information dimension (only filled for q=1) |
| `intercept` | Fit intercept |
| `r_value` | Pearson correlation coefficient |
| `p_value` | p-value |
| `std_err` | Standard error of slope |
| `n_points` | Number of scale points used in fit |

**`df_spec`** — multifractal spectrum:
| Column | Description |
|--------|-------------|
| `q` | Moment order |
| `tau` | τ(q) |
| `Dq` | D(q) |
| `alpha` | α(q) singularity strength |
| `f_alpha` | f(α) multifractal spectrum |

##### 5. compute_alpha_map(scales=None, roi_mode="center", empty_policy="nan")

**Description**: Compute the per-pixel local Hölder exponent α(x,y) via streaming OLS. For each pixel the slope of log μ(ε, x, y) vs log ε is estimated using a vectorized weighted least-squares pass (no per-pixel linregress call).

**Parameters**:
- `scales` (iterable or None): Box sizes to use. Default: powers of 2 up to L/4.
- `roi_mode` (str): `"center"` or `"topleft"`
- `empty_policy` (str): `"nan"` — pixels with zero measure at any scale get α=NaN; `"fill"` — replace zero measure with minimum positive at that scale

**Return Value**: `(alpha_map, info)` where `alpha_map` is a (L, L) float64 array and `info` contains `{L, scales, log_eps}`.

##### 6. compute_alpha_map_batch(images, ...) [static]

**Description**: Batch computation of alpha maps using the nested-grid streaming OLS optimization (runs on a coarse grid when all scales are multiples of the smallest scale, then upsamples — reducing memory and compute significantly).

##### 7. plot(df_mass, df_fit, df_spec)

**Description**: Visualize multifractal spectrum analysis results in a 2×3 subplot grid:
1. Heatmap of log M(q, ε) vs box size and q
2. f(α) vs α multifractal spectrum
3. τ(q) vs q mass exponent
4. D(q) vs q generalized dimension
5. α(q) vs q singularity strength
6. f(α) vs q

##### 8. plot_alpha_map(alpha_map) [static]

**Description**: Visualize the local α-map using a jet colormap.

##### 9. get_batch_mfs(img_list, ...) [static]

**Description**: Batch CPU multifractal spectrum; API-compatible with `CFAImageMFSGPU.get_batch_mfs`.

**Return Value**: List of `(df_mass, df_fit, df_spec)` tuples.

## Theoretical Background

### Multifractal Spectrum

#### 1. Box Probabilities
At scale ε, the measure in box i is normalized to get:
```
μᵢ(ε) = mass_i / total_mass
```

#### 2. Partition Function
```
M(q, ε) = Σᵢ μᵢ(ε)^q
```
For q=0 this counts non-empty boxes; for q=1 it gives e^(−Shannon entropy).

#### 3. Mass Exponent τ(q)
```
τ(q) = lim(ε→0) log M(q, ε) / log ε
```
Estimated by regressing log M(q,ε) on log(1/ε).

#### 4. Generalized Dimension D(q)
```
D(q) = τ(q) / (q − 1),  q ≠ 1
D(1) = lim(ε→0) Σᵢ μᵢ log μᵢ / log ε  (information dimension)
```

Special values:
- **D(0)** — Capacity (box-counting) dimension
- **D(1)** — Information dimension
- **D(2)** — Correlation dimension

#### 5. Multifractal Spectrum f(α)
```
α(q) = dτ(q)/dq    (singularity strength via spline derivative)
f(α) = q·α − τ(q)  (Legendre transform)
```

## Important Notes

1. **Image Preprocessing**:
   - Input must be a 2D single-channel grayscale array
   - Image is automatically normalized to [0, 1]
   - Use `bg_threshold` to mask near-zero background; use `bg_otsu=True` for automatic detection

2. **q Value Selection**:
   - Negative q: sensitive to sparse (low-density) regions
   - Positive q: sensitive to dense (high-density) regions
   - Recommended range: −5 to 5 with 20–51 points

3. **Scale Parameters**:
   - `max_scales=80` gives a good sampling density
   - `roi_mode="center"` crops a fixed square ROI to keep ε normalization constant across scales
   - `min_box≥2` avoids trivial single-pixel boxes

4. **Result Interpretation**:
   - Wider f(α) curve → stronger multifractality
   - Δα = α_max − α_min quantifies heterogeneity
   - D(q) monotonically decreasing with q → multifractal
   - `r_value` close to 1 indicates a good power-law fit

5. **Performance**:
   - Use `CFAImageMFSGPU` for 5–20× speedup on large images
   - Set `with_progress=False` for batch loops to reduce output clutter

## References

- Chhabra, A., & Jensen, R. V. (1989). Direct determination of the f(α) singularity spectrum. *Physical Review Letters*.
- Evertsz, C. J., & Mandelbrot, B. B. (1992). Multifractal measures. *Chaos and Fractals*.
- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis. *Physica A*.
