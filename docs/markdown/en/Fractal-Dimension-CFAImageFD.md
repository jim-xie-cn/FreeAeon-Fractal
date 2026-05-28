# Fractal Dimension Analysis - CFAImageFD

## Application Scenarios

The `CFAImageFD` class is used to calculate the fractal dimension of 2D images, serving as an important tool for image complexity analysis and texture feature extraction. Main application scenarios include:

- **Image Texture Analysis**: Quantify image roughness and complexity
- **Medical Image Analysis**: Analyze tissue structure complexity
- **Materials Science**: Study fractal features of material surfaces
- **Computer Vision**: Image feature extraction and classification
- **Geology**: Terrain and landform complexity analysis

## Usage Examples

### Basic Usage

```python
import cv2
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImage import CFAImage

# Read image
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
bin_image = (gray_image < 64).astype('uint8')

# For BC method, need binary image
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# Calculate three fractal dimensions
fd_bc = CFAImageFD(bin_image).get_bc_fd(corp_type=-1)
fd_dbc = CFAImageFD(gray_image).get_dbc_fd(corp_type=-1)
fd_sdbc = CFAImageFD(gray_image).get_sdbc_fd(corp_type=-1)

# Output results
print("BC Fractal Dimension:", fd_bc['fd'])
print("DBC Fractal Dimension:", fd_dbc['fd'])
print("SDBC Fractal Dimension:", fd_sdbc['fd'])

# Visualize results
CFAImageFD.plot(rgb_image, gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### GPU Accelerated Version

```python
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU

fd_bc = CFAImageFDGPU(bin_image, device='cuda').get_bc_fd()
fd_dbc = CFAImageFDGPU(gray_image, device='cuda').get_dbc_fd()
fd_sdbc = CFAImageFDGPU(gray_image, device='cuda').get_sdbc_fd()
```

### Batch Processing

```python
import cv2, glob
import numpy as np
from FreeAeonFractal.FAImageFD import CFAImageFD

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]

# Batch box-counting
results_bc = CFAImageFD.get_batch_bc(images)
results_dbc = CFAImageFD.get_batch_dbc(images)
results_sdbc = CFAImageFD.get_batch_sdbc(images)

for r in results_bc:
    print("BC FD:", r['fd'])
```

### Restrict Fit Range

```python
# Only use middle scales for fitting (avoids sigmoid tails)
fd_bc = CFAImageFD(bin_image, max_scales=30).get_bc_fd(fit_range=(4, 64))
```

### Automatic Fit-Range Selection (`auto_fit_range`)

Box-counting log-log curves are sigmoid-shaped — saturated at small scales (pixel discretization) and at large scales (image boundary). Fitting the full curve as a line introduces a systematic slope bias. `auto_fit_range=True` automatically picks the most-linear sub-window by maximising the regression F-statistic `R² / (1 - R²) · (n - 2)`.

```python
# BC: pick the linear scaling region automatically
result = CFAImageFD(img, max_scales=30).get_bc_fd(auto_fit_range=True)
print(f"D={result['fd']:.4f}, R²={result['r_value']**2:.4f}, "
      f"range_picked={result['fit_range_used']}")

# DBC/SDBC: default auto_slope_floor=2.0 prevents the picker from
# landing on the BC-degenerate small-scale segment.
fd_dbc  = CFAImageFD(img, max_scales=30).get_dbc_fd(auto_fit_range=True)
fd_sdbc = CFAImageFD(img, max_scales=30).get_sdbc_fd(auto_fit_range=True)

# Disable the safeguard if you have a reason to:
fd_old = CFAImageFD(img, max_scales=30).get_dbc_fd(
    auto_fit_range=True, auto_slope_floor=None)

# Explicit fit_range always overrides auto_fit_range:
fd_manual = CFAImageFD(img, max_scales=30).get_bc_fd(
    fit_range=(4, 64), auto_fit_range=True)
# → fd_manual['fit_range_used'] == (4, 64), auto ignored.

# Batch: each image gets its OWN auto-selected window
batch = CFAImageFD.get_batch_dbc(imgs, max_scales=30,
                                  auto_fit_range=True,
                                  with_progress=False)
for r in batch:
    print(r['fd'], r['fit_range_used'])
```

#### Application scope of `auto_fit_range`

`auto_fit_range=True` is a tool, NOT a one-size-fits-all default. Choose based on the image type:

**✅ Recommended for**:
- **Texture / surface-roughness images** — wide, well-defined linear scaling region; auto reliably picks the middle.
- **Noise fields, fBm surfaces, synthetic fractals** — clear sigmoid with a flat middle; auto > full-curve.
- **Large heterogeneous batches** — each image needs a different window; auto picks per image without manual tuning.
- **Medical imaging textures with rich high-frequency detail** — e.g. tumour-bearing tissue, retinal patterns.

**❌ NOT recommended for** (use `default` or explicit `fit_range` instead):
- **Face images (especially after CLAHE / normalization)** — the log-log curve has a long flat "BC-degenerate plateau" at slope ≈ 2.0 (small-scale segment where `n_r ≡ 1` everywhere). With default `auto_slope_floor=2.0` the picker often lands ON this plateau, returning D barely above 2 (e.g., 2.008) — not the true face DBC (typically 2.2–2.5). For face research use the default (full-curve) fit or `fit_range=(8, 64)` manually; alternatively raise `auto_slope_floor` to 2.1 to skip the plateau.
- **Very smooth or near-constant images** — no real scaling region to find; auto either picks a degenerate window or returns the full curve anyway.
- **Strict literature reproduction** — published papers usually report full-curve or dyadic-only fits. Auto-picked window varies image-to-image, breaking the comparison.

**⚠ Always check** when using auto:
- `result['fit_range_used']` — what window did auto pick? If it's at the small-scale boundary (`s_lo = 2 or 3`), suspect BC-degeneracy.
- `result['r_value']**2` — auto picks high-R² windows by design; R² near 1.0 doesn't guarantee physical correctness, only mathematical linearity.
- Plot the log-log curve + local slope (`np.diff(log_N) / np.diff(log_s)`) at least once per dataset to confirm where the true scaling region sits.

**Diagnostic snippet**:
```python
import matplotlib.pyplot as plt
r = CFAImageFD(img, max_scales=30, with_progress=False).get_dbc_fd()
scales = np.array(r['scales']); counts = np.array(r['counts'])
log_s = -np.log(scales); log_N = np.log(counts)
slope_local = np.diff(log_N) / np.diff(log_s)
mid_s = (scales[1:] + scales[:-1]) / 2
plt.plot(mid_s, slope_local, 'o-'); plt.xscale('log')
plt.axhline(2.0, ls='--', color='r')
plt.xlabel('scale s'); plt.ylabel('local slope (= local D)'); plt.show()
# Flat plateau at slope ≈ 2 → BC degenerate; don't use auto.
# Clear monotonic rise from 2 to 3 → use auto with floor=2.0 or default.
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageFD

**Description**: Class for calculating 2D image fractal dimensions, supporting three different calculation methods: Box-Counting (BC), Differential Box-Counting (DBC), and Shifted DBC (SDBC).

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input image (single channel 2D array) |
| `max_size` | int | None | Maximum box size (default: minimum image dimension) |
| `max_scales` | int | 30 | Target number of distinct scales |
| `with_progress` | bool | True | Whether to show progress bar |
| `min_size` | int | 2 | Minimum box size |

#### Main Methods

##### 1. get_bc_fd(corp_type=-1, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=None, auto_slope_ceiling=None)

**Description**: Calculate fractal dimension using Box-Counting (BC) method. Any positive pixel counts as occupied.

**Parameters**:
- `corp_type` (int): Image cropping method
  - `-1`: Auto crop to multiples of box size (default)
  - `0`: No processing (requires image dimensions to be exact multiples)
  - `1`: Zero-pad to multiples
- `fit_range` (tuple or None): Optional `(min_scale, max_scale)` to restrict log-log fit range
- `auto_fit_range` (bool, default `False`): When `True` AND `fit_range` is `None`, the most-linear sub-window of the log-log curve is selected automatically (F-statistic scoring: `R² / (1 - R²) · (n - 2)`). Explicit `fit_range` always overrides auto-selection.
- `auto_fit_min_points` (int, default `5`): Minimum number of points the auto-selected window must contain.
- `auto_slope_floor` (float or None, default `None` for BC): Reject auto-fit windows whose slope falls below this value. BC theory: `D ∈ [0, 2]`, so no default floor.
- `auto_slope_ceiling` (float or None, default `None` for BC): Reject auto-fit windows whose slope exceeds this value. Pass `2.0` to enforce the BC theoretical ceiling and reject fitting artifacts that would give `D > 2`.

**Return Value** (dict):
```python
{
    'fd': float,              # Fractal dimension value
    'scales': list,           # Used scale list (after filtering)
    'counts': list,           # Box count list (after filtering)
    'log_scales': list,       # log(1/r) values used in fit
    'log_counts': list,       # log(N(r)) values used in fit
    'intercept': float,       # Fit intercept
    'r_value': float,         # Correlation coefficient
    'p_value': float,         # p-value
    'std_err': float,         # Standard error
    'fit_range_used': tuple or None  # Range actually applied: None if
                                     # no range, (s_lo, s_hi) otherwise.
                                     # When auto_fit_range=True this
                                     # tells you which window was picked.
}
```

**Use Case**: Suitable for binary images; counts the boxes needed to cover the foreground.

##### 2. get_dbc_fd(corp_type=-1, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=2.0, auto_slope_ceiling=None)

**Description**: Calculate fractal dimension using Differential Box-Counting (DBC) method (Sarkar & Chaudhuri 1994).

Formula per box: `n_r = ceil(I_max / h) - ceil(I_min / h) + 1`, where `h = s * G_max / H` (`s` = box side in pixels, `H` = image side, `G_max` = gray-value range, here `np.percentile(image, 99)`).

**Parameters**: Same as `get_bc_fd`, except `auto_slope_floor` **defaults to `2.0`** (DBC theoretical floor: `D ∈ [2, 3]` for a 2D grayscale surface).

**Why the default `auto_slope_floor=2.0`?** At small scales on smooth-ish images, every block's `I_max − I_min ≤ h`, so `n_r ≡ 1` and the log-log curve degenerates to BC behavior with slope ≈ 2. Without this floor, `auto_fit_range=True` would often pick that degenerate segment, returning `D` slightly below 2 — physically impossible for DBC. The floor ensures the auto-picker lands on the real DBC scaling region instead. Pass `auto_slope_floor=None` to disable the safeguard.

**G_max caveat**: `G_max` is `np.percentile(image, 99)` for outlier robustness, NOT the fixed `G = 256` of the original paper. This makes the implementation content-adaptive: pre-normalize input to a fixed gray range before cross-image comparison.

**Return Value**: Same as `get_bc_fd` (includes `fit_range_used`).

**Use Case**: Suitable for grayscale images; treats the image as a 3D surface and measures its height variation.

##### 3. get_sdbc_fd(corp_type=-1, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=2.0, auto_slope_ceiling=None)

**Description**: Calculate fractal dimension using Shifted DBC (SDBC) method (Chen et al. 1995).

Formula per box: `n_r = floor((I_max - I_min) / h) + 1`, with the same `h` as DBC. This is the exact analytic result of Chen's "shift vertical origin" geometric construction, not an approximation.

**Parameters**: Same as `get_dbc_fd` (including `auto_slope_floor=2.0` default).

**Return Value**: Same as `get_bc_fd`.

**Use Case**: Improved version of DBC for grayscale images; intensity-invariant (immune to additive brightness shift), better stability under varying illumination — the preferred method for face / texture classification.

##### 4. get_fd(scale_list, box_count_list, fit_range=None, auto_fit_range=False, auto_fit_min_points=5, auto_slope_floor=None, auto_slope_ceiling=None)

**Description**: Utility method to perform log-log linear regression on custom scale and count data.

**Parameters**:
- `scale_list`: List of scale values (r)
- `box_count_list`: Corresponding box counts N(r)
- All `fit_range` / `auto_fit_range` / `auto_*` parameters as documented above.

**Return Value**: Same dict structure as `get_bc_fd` (includes `fit_range_used`).

##### 5. plot(raw_img, gray_img, bin_img, fd_bc, fd_dbc, fd_sdbc) [static]

**Description**: Visualize original image, grayscale image, binary image, and fitting results of three fractal dimensions in a 2×3 subplot grid.

**Parameters**:
- `raw_img`: Original RGB image
- `gray_img`: Grayscale image
- `bin_img`: Binary image
- `fd_bc`: BC method result dictionary
- `fd_dbc`: DBC method result dictionary
- `fd_sdbc`: SDBC method result dictionary

##### 6. get_batch_bc / get_batch_dbc / get_batch_sdbc(images, ...) [static]

**Description**: Static methods for batch processing of multiple images. Scale generation is shared across the batch to avoid redundant work.

**Parameters**:
- `images` (list of ndarray): List of 2D input images
- `max_size` (int): Maximum box size
- `max_scales` (int): Target number of scales
- `min_size` (int): Minimum box size
- `corp_type` (int): Cropping strategy
- `fit_range` (tuple or None): Fit range restriction
- `auto_fit_range` (bool, default `False`): Per-image auto-selection of the linear region. Each image gets its OWN best window picked independently; the chosen window is reported per image in `result['fit_range_used']`.
- `auto_fit_min_points` (int, default `5`): Minimum window size for auto-selection.
- `auto_slope_floor` / `auto_slope_ceiling` (float or None): Physical bounds on the auto-selected slope. Defaults:
  - `get_batch_bc`: both `None` (BC has no theoretical floor)
  - `get_batch_dbc` / `get_batch_sdbc`: `auto_slope_floor=2.0` (DBC/SDBC theoretical floor)
- `with_progress` (bool): Show progress bar

**Return Value**: List of result dictionaries (one per image), each including the new `fit_range_used` field.

## Algorithm Description

### BC (Box-Counting) Method

The box-counting method calculates fractal dimension by counting non-empty boxes at different scales. Formula:

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

Where N(ε) is the number of non-empty boxes at scale ε.

### DBC (Differential Box-Counting) Method

DBC treats the grayscale image as a 3D surface and counts the box height steps:

```
h = s × G_max / H
n_r(i,j) = ceil(I_max / h) - ceil(I_min / h) + 1
```

Where `H` is the image side (max dimension), `G_max` is the 99th percentile intensity, and `s` is the spatial box side in pixels. This direction (`s × G_max / H`, not `s × H / G_max`) matches Sarkar & Chaudhuri 1994's `[G/s'] = [M/s]` derivation.

### SDBC (Shifted DBC) Method

SDBC shifts each box's baseline to I_min before counting:

```
n_r(i,j) = floor((I_max - I_min) / h) + 1
```

This avoids the "+1" boundary offset in the DBC formula, reducing systematic overestimation at small scales.

### Scale Generation

Scales are geometrically spaced in `[min_size, max_size]` using `make_scales()`. Unlike `np.logspace(..., dtype=int)` which collapses many values to the same integer, the implementation generates floats, rounds, deduplicates, and sorts — ensuring the requested count of **distinct** integer scales.

### Log-Log Regression

Scales with N=0 are dropped from the regression instead of being replaced with epsilon (which would anchor the fit artificially). The `fit_range` parameter allows restricting to the power-law regime by excluding the sigmoid tails.

## Important Notes

1. **Image Preprocessing**:
   - BC method requires binary images; use `CFAImage.otsu_binarize()` or manual thresholding
   - DBC and SDBC methods accept grayscale images directly

2. **Scale Selection**:
   - `max_size` limits the scale range; set to image size for full range
   - `max_scales=30` is a good default; increase for smoother log-log curves
   - Use `fit_range` to exclude unreliable small/large scales

3. **Result Interpretation**:
   - Fractal dimension is typically 1–2 for 2D images
   - Larger values indicate more complex, space-filling images
   - `r_value` close to ±1 indicates good power-law fit

4. **Performance Optimization**:
   - Set `with_progress=False` to suppress the tqdm bar
   - Use GPU version (`CFAImageFDGPU`) for large images or batch processing
   - Note: GPU version sets `p_value=None` (not computed)

## References

- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image. *IEEE Transactions on Systems, Man, and Cybernetics*.
- Chen, W. S., et al. (1995). Efficient fractal coding of images based on differential box counting. *Pattern Recognition*.
