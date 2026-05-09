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

##### 1. get_bc_fd(corp_type=-1, fit_range=None)

**Description**: Calculate fractal dimension using Box-Counting (BC) method. Any positive pixel counts as occupied.

**Parameters**:
- `corp_type` (int): Image cropping method
  - `-1`: Auto crop to multiples of box size (default)
  - `0`: No processing (requires image dimensions to be exact multiples)
  - `1`: Zero-pad to multiples
- `fit_range` (tuple or None): Optional `(min_scale, max_scale)` to restrict log-log fit range

**Return Value** (dict):
```python
{
    'fd': float,              # Fractal dimension value
    'scales': list,           # Used scale list
    'counts': list,           # Box count list
    'log_scales': list,       # log(1/r) values used in fit
    'log_counts': list,       # log(N(r)) values used in fit
    'intercept': float,       # Fit intercept
    'r_value': float,         # Correlation coefficient
    'p_value': float,         # p-value
    'std_err': float          # Standard error
}
```

**Use Case**: Suitable for binary images; counts the boxes needed to cover the foreground.

##### 2. get_dbc_fd(corp_type=-1, fit_range=None)

**Description**: Calculate fractal dimension using Differential Box-Counting (DBC) method (Sarkar & Chaudhuri 1994).

Formula per box: `n_r = ceil(I_max / h) - ceil(I_min / h) + 1`, where `h = s * H / G_max`.

**Parameters**: Same as `get_bc_fd`

**Return Value**: Same as `get_bc_fd`

**Use Case**: Suitable for grayscale images; treats the image as a 3D surface and measures its height variation.

##### 3. get_sdbc_fd(corp_type=-1, fit_range=None)

**Description**: Calculate fractal dimension using Shifted DBC (SDBC) method (Chen et al. 1995).

Formula per box: `n_r = floor((I_max - I_min) / h) + 1`. Aligns each box's baseline to I_min, avoiding the boundary-crossing overcount of plain DBC.

**Parameters**: Same as `get_bc_fd`

**Return Value**: Same as `get_bc_fd`

**Use Case**: Improved version of DBC for grayscale images with better accuracy at small scales.

##### 4. get_fd(scale_list, box_count_list)

**Description**: Utility method to perform log-log linear regression on custom scale and count data.

**Parameters**:
- `scale_list`: List of scale values (r)
- `box_count_list`: Corresponding box counts N(r)

**Return Value**: Same dict structure as `get_bc_fd`

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
- `with_progress` (bool): Show progress bar

**Return Value**: List of result dictionaries (one per image)

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
h = s × H / G_max
n_r(i,j) = ceil(I_max / h) - ceil(I_min / h) + 1
```

Where `H` is the image size (max dimension) and `G_max` is the 99th percentile intensity.

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
