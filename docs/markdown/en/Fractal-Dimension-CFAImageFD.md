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
results = CFAImageFD.get_batch_bc(images)
for r in results:
    print("BC FD:", r['fd'])
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
| `image` | numpy.ndarray | Required | Input image (single channel) |
| `max_size` | int | None | Maximum box size (default: minimum image dimension) |
| `max_scales` | int | 30 | Maximum number of scales |
| `with_progress` | bool | True | Whether to show progress bar |
| `min_size` | int | 2 | Minimum box size |

#### Main Methods

##### 1. get_bc_fd(corp_type=-1, fit_range=None)

**Description**: Calculate fractal dimension using Box-Counting (BC) method.

**Parameters**:
- `corp_type` (int): Image cropping method
  - `-1`: Auto crop to multiples of box size
  - `0`: No processing (requires image dimensions to be multiples of box size)
  - `1`: Padding
- `fit_range` (tuple or None): Optional `(min_scale, max_scale)` to restrict log-log fit range

**Return Value** (dict):
```python
{
    'fd': float,              # Fractal dimension value
    'scales': list,           # Scale list
    'counts': list,           # Box count list
    'log_scales': list,       # Logarithmic scales
    'log_counts': list,       # Logarithmic counts
    'intercept': float,       # Fit intercept
    'r_value': float,         # Correlation coefficient
    'p_value': float,         # p-value
    'std_err': float          # Standard error
}
```

**Use Case**: Suitable for binary images, calculating the dimension of occupied space.

##### 2. get_dbc_fd(corp_type=-1, fit_range=None)

**Description**: Calculate fractal dimension using Differential Box-Counting (DBC) method (Sarkar & Chaudhuri 1994). Uses `n_r = ceil(I_max/h) - ceil(I_min/h) + 1`.

**Parameters**: Same as `get_bc_fd`

**Return Value**: Same as `get_bc_fd`

**Use Case**: Suitable for grayscale images, considering grayscale information in fractal dimension calculation.

##### 3. get_sdbc_fd(corp_type=-1, fit_range=None)

**Description**: Calculate fractal dimension using Shifted DBC (SDBC) method (Chen 1995). Uses `n_r = floor((I_max - I_min)/h) + 1`.

**Parameters**: Same as `get_bc_fd`

**Return Value**: Same as `get_bc_fd`

**Use Case**: SDBC is a simplified version of DBC, faster computation, suitable for grayscale images.

##### 4. get_fd(scale_list, box_count_list)

**Description**: Utility method to perform log-log fit on custom scale and count data.

**Parameters**:
- `scale_list`: List of scale values
- `box_count_list`: Corresponding box counts

**Return Value**: Same dict structure as `get_bc_fd`

##### 5. plot(raw_img, gray_img, bin_img, fd_bc, fd_dbc, fd_sdbc)

**Description**: Static method to visualize original image, grayscale image, binary image, and fitting results of three fractal dimensions.

**Parameters**:
- `raw_img`: Original RGB image
- `gray_img`: Grayscale image
- `bin_img`: Binary image
- `fd_bc`: BC method result dictionary
- `fd_dbc`: DBC method result dictionary
- `fd_sdbc`: SDBC method result dictionary

##### 6. get_batch_bc / get_batch_dbc / get_batch_sdbc(images, ...)

**Description**: Static methods for batch processing of multiple images.

**Parameters**:
- `images` (list): List of input images
- Additional keyword arguments passed to the corresponding single-image method

**Return Value**: List of result dictionaries

## Algorithm Description

### BC (Box-Counting) Method
The box-counting method calculates fractal dimension by counting the number of non-empty boxes needed to cover the image at different scales. Formula:

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

Where N(ε) is the number of boxes at scale ε.

### DBC (Differential Box-Counting) Method
DBC considers grayscale height information, treating the image as a 3D surface. The height of each box is:

```
n_r = ceil(I_max / h) - ceil(I_min / h) + 1
```

Where `h = max_val / (image_size / r)` is the height of a grayscale unit.

### SDBC (Simplified DBC) Method
Shifted DBC uses a simplified height calculation:

```
n_r = floor((I_max - I_min) / h) + 1
```

This avoids the rounding artefact in the original DBC formula.

## Important Notes

1. **Image Preprocessing**:
   - BC method requires binary images, recommend using Otsu auto-thresholding
   - DBC and SDBC methods are suitable for grayscale images

2. **Scale Selection**:
   - `max_size` affects the scale range of analysis
   - `max_scales` affects sampling density, recommend default value 30
   - Use `fit_range` to exclude unreliable small/large scales

3. **Result Interpretation**:
   - Fractal dimension range typically 1-2 (for 2D images)
   - Larger values indicate more complex images
   - `r_value` close to 1 indicates good power-law fit

4. **Performance Optimization**:
   - For large images, can downsample first
   - Set `with_progress=False` to improve computation speed
   - Use GPU version (`CFAImageFDGPU`) for batch processing
   - Note: GPU version sets `p_value=None` (not computed)

## References

- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image. *IEEE Transactions on Systems, Man, and Cybernetics*.
- Chen, W. S., et al. (1995). Efficient fractal coding of images based on differential box counting. *Pattern Recognition*.
