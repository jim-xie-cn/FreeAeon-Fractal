# Fractal Dimension Analysis - CFAImageDimension

## Application Scenarios

The `CFAImageDimension` class is used to calculate the fractal dimension of 2D images, serving as an important tool for image complexity analysis and texture feature extraction. Main application scenarios include:

- **Image Texture Analysis**: Quantify image roughness and complexity
- **Medical Image Analysis**: Analyze tissue structure complexity
- **Materials Science**: Study fractal features of material surfaces
- **Computer Vision**: Image feature extraction and classification
- **Geology**: Terrain and landform complexity analysis

## Usage Examples

### Basic Usage

```python
import cv2
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImage import CFAImage

# Read image
rgb_image = cv2.imread('./images/fractal.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# For BC method, need binary image
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

# Calculate three fractal dimensions
fd_bc = CFAImageDimension(bin_image).get_bc_fd(corp_type=-1)
fd_dbc = CFAImageDimension(gray_image).get_dbc_fd(corp_type=-1)
fd_sdbc = CFAImageDimension(gray_image).get_sdbc_fd(corp_type=-1)

# Output results
print("BC Fractal Dimension:", fd_bc['fd'])
print("DBC Fractal Dimension:", fd_dbc['fd'])
print("SDBC Fractal Dimension:", fd_sdbc['fd'])

# Visualize results
CFAImageDimension.plot(rgb_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageDimension

**Description**: Class for calculating 2D image fractal dimensions, supporting three different calculation methods.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input image (single channel) |
| `max_size` | int | None | Maximum box size (default: minimum image dimension) |
| `max_scales` | int | 100 | Maximum number of scales |
| `with_progress` | bool | True | Whether to show progress bar |

#### Main Methods

##### 1. get_bc_fd(corp_type=-1)

**Description**: Calculate fractal dimension using Box-Counting (BC) method.

**Parameters**:
- `corp_type` (int): Image cropping method
  - `-1`: Auto crop to multiples of box size
  - `0`: No processing (requires image dimensions to be multiples of box size)
  - `1`: Padding

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

##### 2. get_dbc_fd(corp_type=-1)

**Description**: Calculate fractal dimension using Differential Box-Counting (DBC) method.

**Parameters**: Same as `get_bc_fd`

**Return Value**: Same as `get_bc_fd`

**Use Case**: Suitable for grayscale images, considering grayscale information in fractal dimension calculation.

##### 3. get_sdbc_fd(corp_type=-1)

**Description**: Calculate fractal dimension using Simplified DBC (SDBC) method.

**Parameters**: Same as `get_bc_fd`

**Return Value**: Same as `get_bc_fd`

**Use Case**: SDBC is a simplified version of DBC, faster computation, suitable for grayscale images.

##### 4. plot(raw_img, gray_img, fd_bc, fd_dbc, fd_sdbc)

**Description**: Static method to visualize original image, binary image, and fitting results of three fractal dimensions.

**Parameters**:
- `raw_img`: Original RGB image
- `gray_img`: Binary or grayscale image
- `fd_bc`: BC method result dictionary
- `fd_dbc`: DBC method result dictionary
- `fd_sdbc`: SDBC method result dictionary

## Algorithm Description

### BC (Box-Counting) Method
The box-counting method calculates fractal dimension by counting the number of non-empty boxes needed to cover the image at different scales. Formula:

```
D = lim(ε→0) log(N(ε)) / log(1/ε)
```

Where N(ε) is the number of boxes at scale ε.

### DBC (Differential Box-Counting) Method
DBC considers grayscale height information, treating the image as a 3D surface. The height of each box is determined by the grayscale range in that region.

### SDBC (Simplified DBC) Method
Simplified DBC is an improved version of DBC, using a more simplified calculation method while maintaining accuracy and improving computational efficiency.

## Important Notes

1. **Image Preprocessing**:
   - BC method requires binary images, recommend using Otsu auto-thresholding
   - DBC and SDBC methods are suitable for grayscale images

2. **Scale Selection**:
   - `max_size` affects the scale range of analysis
   - `max_scales` affects sampling density, recommend default value 100

3. **Result Interpretation**:
   - Fractal dimension range typically 1-2 (for 2D images)
   - Larger values indicate more complex images
   - `r_value` close to 1 indicates good power-law fit

4. **Performance Optimization**:
   - For large images, can downsample first
   - Set `with_progress=False` to improve computation speed

## References

- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
- Sarkar, N., & Chaudhuri, B. B. (1994). An efficient differential box-counting approach to compute fractal dimension of image.
