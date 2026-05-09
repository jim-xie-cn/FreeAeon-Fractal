# Lacunarity Analysis - CFAImageLAC

## Application Scenarios

The `CFAImageLAC` class is used to quantify the spatial heterogeneity and gap structure of 2D images. Main application scenarios include:

- **Ecology**: Quantify habitat fragmentation and gap distribution
- **Materials Science**: Characterize porous structure and internal geometry
- **Medical Imaging**: Analyze tissue uniformity and lesion distribution
- **Urban Planning**: Study spatial distribution of land use patterns
- **Geology**: Quantify rock fracture and void distribution

## Usage Examples

### Basic Usage

```python
import cv2
from FreeAeonFractal.FAImageLAC import CFAImageLAC
from FreeAeonFractal.FAImage import CFAImage

# Read image
gray_image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)

# Binarize for binary lacunarity
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# Gliding box lacunarity (default)
calc = CFAImageLAC(bin_image, partition_mode="gliding")
lac_result = calc.get_lacunarity(use_binary_mass=True, include_zero=True)
fit_result = calc.fit_lacunarity(lac_result)

print("Lambda(r):", lac_result['lacunarity'])
print("Slope (beta):", fit_result['slope'])
print("R²:", fit_result['r_value']**2)

# Visualize
calc.plot(lac_result, fit_result)
```

### Non-overlapping Mode

```python
calc_nonoverlap = CFAImageLAC(gray_image, partition_mode="non-overlapping")
lac_nonoverlap = calc_nonoverlap.get_lacunarity()
fit_nonoverlap = calc_nonoverlap.fit_lacunarity(lac_nonoverlap)
```

### GPU Accelerated Version

```python
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU

calc_gpu = CFAImageLACGPU(bin_image, device='cuda')
lac_result = calc_gpu.get_lacunarity()
```

### Batch Processing

```python
import cv2, glob
from FreeAeonFractal.FAImageLAC import CFAImageLAC

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]

# Batch gliding lacunarity
batch_results = CFAImageLAC.get_batch_lacunarity(
    images,
    partition_mode="gliding",
    use_binary_mass=True,
    with_progress=True
)

# Fit all results
batch_fits = CFAImageLAC.fit_batch_lacunarity(batch_results)
for fit in batch_fits:
    print("Slope:", fit['slope'])
```

### Log Transform Variants

```python
# Default: fit log(Lambda) vs log(r) — standard self-similar fractal slope
fit_log = calc.fit_lacunarity(lac_result, transform="log")

# Legacy: fit log(Lambda - 1) vs log(r) — useful when Lambda is close to 1
fit_log_m1 = calc.fit_lacunarity(lac_result, transform="log_minus_1")
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageLAC

**Description**: Lacunarity calculator for a single 2D image. Supports gliding box (overlapping) and non-overlapping box partition modes. For batch processing use the static methods `get_batch_lacunarity` / `fit_batch_lacunarity`.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input 2D single-channel image |
| `max_size` | int | None | Maximum box size (default: min image dimension) |
| `max_scales` | int | 100 | Target number of scales |
| `with_progress` | bool | True | Whether to show progress bar |
| `scales_mode` | str | `"powers"` | Scale generation: `"powers"` (2,4,8,...) or `"logspace"` (geometrically spaced) |
| `partition_mode` | str | `"gliding"` | Box strategy: `"gliding"` or `"non-overlapping"` |
| `min_size` | int | 2 | Minimum box size |

#### Main Methods

##### 1. get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)

**Description**: Compute Λ(r) for every box size in `self.scales`.

**Parameters**:
- `corp_type` (int): Cropping strategy (used for non-overlapping mode)
- `use_binary_mass` (bool): Treat image as binary (any positive pixel → 1) before summing
- `include_zero` (bool): If False, exclude empty boxes (mass=0) from statistics

**Return Value** (dict):
```python
{
    'scales': list,       # Box sizes used
    'lacunarity': list,   # Lambda(r) values
    'mass_stats': list    # Per-scale: scale, num_boxes, mean_mass, var_mass, lambda
}
```

**Lacunarity formula**: `Λ(r) = E[M²] / E[M]² = 1 + Var(M) / Mean(M)²`

##### 2. fit_lacunarity(lac_result, transform="log", fit_range=None)

**Description**: Fit the lacunarity curve with log-log regression.

**Parameters**:
- `lac_result` (dict): Output from `get_lacunarity`
- `transform` (str):
  - `"log"`: Fit log(Λ) vs log(r). Standard for self-similar fractals; slope = −β
  - `"log_minus_1"`: Fit log(Λ−1) vs log(r). Legacy mode, ignores uniform regions
- `fit_range` (tuple or None): `(r_min, r_max)` to restrict regression range

**Return Value** (dict):
```python
{
    'slope': float,
    'intercept': float,
    'r_value': float,
    'p_value': float,
    'std_err': float,
    'log_scales': list,
    'log_lambda_minus_1': list,
    'transform': str
}
```

##### 3. plot(lac_result, fit_result=None, ax=None, show=True, title="Lacunarity", label=None)

**Description**: Visualize Λ(r) on a log-log scale, and optionally the linear fit panel.

##### 4. get_batch_lacunarity(images, ...) [static]

**Description**: Batch lacunarity. When all images share the same shape and `partition_mode="gliding"`, integral images are vectorized across the batch for maximum efficiency.

**Parameters**:
- `images`: List of 2D arrays
- `max_size`, `max_scales`, `scales_mode`, `partition_mode`, `min_size`: Same as constructor
- `use_binary_mass`, `include_zero`: Same as `get_lacunarity`
- `with_progress` (bool): Show progress bar

**Return Value**: List of `get_lacunarity`-style dicts (one per image)

##### 5. fit_batch_lacunarity(lac_results, transform="log", fit_range=None) [static]

**Description**: Apply the same fit to every result returned by `get_batch_lacunarity`.

**Return Value**: List of fit result dicts

## Algorithm Description

### Lacunarity Definition

For a measure (image) and box of side r, let M be the box mass:

```
Λ(r) = E[M²] / E[M]² = 1 + Var(M) / Mean(M)²
```

Lower bound is 1 (zero variance = homogeneous). Larger Λ indicates stronger spatial heterogeneity.

### Gliding Box Method

Each (r×r) window slides over the image. The summed-area table (integral image) is computed **once** outside the scale loop. Per-scale box masses are extracted in O(H×W) time via:

```
M(y, x) = S[y+r, x+r] - S[y, x+r] - S[y+r, x] + S[y, x]
```

### Non-overlapping Box Method

Image is tiled with disjoint (r×r) blocks (edge remainder discarded). Box sums are vectorized via `reshape + transpose`.

### Lacunarity Fitting

For self-similar fractals: `Λ(r) ~ r^{−β}` with β = D − E.

The default `transform="log"` regresses `log Λ(r)` on `log r`, reporting slope = −β.

## Important Notes

1. **Partition Mode**:
   - `"gliding"` is more statistically robust (more samples per scale) and is fast via the integral image trick
   - `"non-overlapping"` is useful when sample independence is required

2. **Binary vs Gray Lacunarity**:
   - `use_binary_mass=True` for classic binary lacunarity (Allain & Cloitre 1991)
   - Default (False) uses raw grayscale intensity as mass

3. **Result Interpretation**:
   - Λ = 1: Perfectly uniform (homogeneous)
   - Λ > 1: Gaps and clustering present; larger = more heterogeneous
   - Negative slope β indicates scale-dependent gap closure

4. **Performance**:
   - Integral image computed once per image — efficient for many scales
   - Same-shape batch: batched integral image across N images simultaneously

## References

- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. *Physical Review A*.
- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. *Physical Review E*.
