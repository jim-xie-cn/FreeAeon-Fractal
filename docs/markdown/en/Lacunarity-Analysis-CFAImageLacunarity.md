# Lacunarity Analysis - CFAImageLacunarity

## Application Scenarios

The `CFAImageLacunarity` class is used to calculate the lacunarity of 2D images, serving as an important tool for quantifying spatial distribution heterogeneity. Main application scenarios include:

- **Texture Analysis**: Quantify gaps and void characteristics in textures
- **Landscape Ecology**: Analyze spatial distribution of vegetation coverage
- **Materials Science**: Study material pore structures
- **Medical Imaging**: Analyze tissue distribution uniformity
- **Urban Planning**: Study spatial distribution of buildings

## Usage Examples

### Basic Usage

```python
import cv2
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity
from FreeAeonFractal.FAImage import CFAImage

# Read image
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Create lacunarity analysis object
lacunarity = CFAImageLacunarity(
    gray_image,
    max_scales=256,
    with_progress=True,
    partition_mode="gliding"  # or "non-overlapping"
)

# Calculate lacunarity
lac_result = lacunarity.get_lacunarity(
    corp_type=-1,
    use_binary_mass=False,
    include_zero=True
)

# Fit lacunarity
fit_result = lacunarity.fit_lacunarity(lac_result)

# Output results
print("Lacunarity values:", lac_result["lacunarity"])
print("Fit slope:", fit_result["slope"])
print("Fit R²:", fit_result["r_value"]**2)

# Visualize
lacunarity.plot(lac_result, fit_result)
```

### Binary Image Analysis

```python
# Use Otsu auto-binarization
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# Lacunarity of binary image
lacunarity_bin = CFAImageLacunarity(bin_image, partition_mode="gliding")
lac_bin = lacunarity_bin.get_lacunarity(use_binary_mass=True)
fit_bin = lacunarity_bin.fit_lacunarity(lac_bin)

print("Binary image lacunarity:", lac_bin["lacunarity"])
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageLacunarity

**Description**: Class for calculating 2D image lacunarity, supporting two box partition strategies.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input image (grayscale or binary) |
| `max_size` | int | None | Maximum box size (default: minimum image dimension) |
| `max_scales` | int | 100 | Maximum number of scales |
| `with_progress` | bool | True | Whether to show progress bar |
| `scales_mode` | str | "powers" | Scale generation mode ("powers" or "logspace") |
| `partition_mode` | str | "gliding" | Partition mode ("gliding" or "non-overlapping") |

**scales_mode** Options:
- `"powers"`: Powers of 2 scales (2, 4, 8, 16, ...)
- `"logspace"`: Logarithmically uniform distributed scales

**partition_mode** Options:
- `"gliding"`: Sliding window (overlapping), efficient computation using integral images
- `"non-overlapping"`: Non-overlapping fixed blocks

#### Main Methods

##### 1. get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)

**Description**: Calculate lacunarity at each scale.

**Parameters**:
- `corp_type` (int): Image cropping method (non-overlapping mode only)
  - `-1`: Auto crop
  - `0`: No processing
  - `1`: Padding
- `use_binary_mass` (bool): Whether to use binary mass
  - `True`: Only count non-zero pixels
  - `False`: Use actual pixel values
- `include_zero` (bool): Whether to include zero-mass boxes

**Return Value** (dict):
```python
{
    'scales': list,        # Scale list
    'lacunarity': list,    # Lacunarity values at each scale
    'mass_stats': list     # Mass statistics info list
}
```

Each element in `mass_stats` contains:
```python
{
    'scale': int,          # Scale
    'num_boxes': int,      # Number of boxes
    'mean_mass': float,    # Mean mass
    'var_mass': float,     # Mass variance
    'lambda': float        # Lacunarity value
}
```

##### 2. fit_lacunarity(lac_result, min_valid_lambda=1.0+1e-12)

**Description**: Perform power-law fitting on lacunarity: log(Λ-1) vs log(r)

**Parameters**:
- `lac_result` (dict): Return result from `get_lacunarity()`
- `min_valid_lambda` (float): Minimum valid lacunarity value

**Return Value** (dict):
```python
{
    'slope': float,                    # Fit slope
    'intercept': float,                # Fit intercept
    'r_value': float,                  # Correlation coefficient
    'p_value': float,                  # p-value
    'std_err': float,                  # Standard error
    'log_scales': list,                # Logarithmic scales
    'log_lambda_minus_1': list         # log(Λ-1)
}
```

##### 3. plot(lac_result, fit_result=None, ax=None, show=True, title="Lacunarity", label=None)

**Description**: Visualize lacunarity curves and fitting results.

**Parameters**:
- `lac_result` (dict): Lacunarity results
- `fit_result` (dict): Fitting results (optional)
- `ax`: Matplotlib axis object (optional)
- `show` (bool): Whether to display immediately
- `title` (str): Plot title
- `label` (str): Curve label

**Plots**:
- Left plot: Λ(r) vs r
- Right plot (if fit_result provided): log(Λ-1) vs log(r) with fit line

## Theoretical Background

### Lacunarity Definition

Lacunarity Λ(r) is defined as the ratio of the second moment to the square of the first moment of box masses:

```
Λ(r) = E[M²(r)] / E[M(r)]²
```

Where:
- M(r): Box mass at scale r
- E[·]: Expectation value

### Physical Meaning

- **Λ = 1**: Completely uniform distribution
- **Λ > 1**: Presence of gaps, non-uniform distribution
- **Larger Λ**: More gaps, stronger clustering

### Scaling Law

Lacunarity typically follows a power law:

```
Λ(r) - 1 ∝ r^β
```

Where β is the lacunarity exponent.

## Important Notes

1. **Partition Mode Selection**:
   - `"gliding"`: Recommended for general analysis, smoother results
   - `"non-overlapping"`: Faster, suitable for large images

2. **Mass Mode**:
   - `use_binary_mass=True`: Binary images or only concerned with occupancy
   - `use_binary_mass=False`: Grayscale images, considering intensity information

3. **Scale Mode**:
   - `"powers"`: Fewer scale points, faster computation
   - `"logspace"`: More scale points, finer results

4. **Zero Value Processing**:
   - `include_zero=True`: Include all boxes
   - `include_zero=False`: Only consider non-zero boxes

5. **Result Interpretation**:
   - Lacunarity value range: [1, +∞)
   - Slope β: Quantifies scale dependency
   - R² close to 1: Good power-law fit

6. **Performance Optimization**:
   - Use `partition_mode="non-overlapping"` for speedup
   - Reduce `max_scales` value
   - Use `scales_mode="powers"`

## References

- Plotnick, R. E., et al. (1996). Lacunarity analysis: A general technique for the analysis of spatial patterns. Physical Review E.
- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. Physical Review A.
