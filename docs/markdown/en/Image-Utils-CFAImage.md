# Image Processing Utilities - CFAImage

## Application Scenarios

The `CFAImage` class provides basic image processing utility functions, including image blocking, merging, mask generation, and ROI extraction. Main application scenarios include:

- **Image Block Processing**: Split images into fixed-size blocks for analysis
- **Image Preprocessing**: Auto-binarization, cropping, padding
- **ROI Extraction**: Extract regions of interest based on multifractal properties
- **Data Augmentation**: Random sampling of image patches for training
- **Image Analysis**: Support for fractal dimension and multifractal spectrum calculation

## Usage Examples

### Otsu Auto-Binarization

```python
import cv2
from FreeAeonFractal.FAImage import CFAImage

# Read grayscale image
gray_image = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# Otsu auto-thresholding
bin_image, threshold = CFAImage.otsu_binarize(gray_image)

print(f"Auto threshold: {threshold}")

# Visualize
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(gray_image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(bin_image, cmap='gray')
ax2.set_title(f'Binary Image (threshold={threshold:.1f})')
plt.show()
```

### Image Blocking and Merging

```python
import numpy as np

# Create test image
image = np.zeros((256, 256), dtype=np.uint8)
cv2.circle(image, (128, 128), 80, 255, -1)

# Block splitting
block_size = (64, 64)
boxes, raw_blocks = CFAImage.get_boxes_from_image(
    image,
    block_size,
    corp_type=-1  # Auto crop
)

print(f"Total blocks: {boxes.shape[0]}")
print(f"Raw blocks shape: {raw_blocks.shape}")

# Merge back to image
merged_image = CFAImage.get_image_from_boxes(raw_blocks)
```

### Create Mask

```python
# Create mask (mark specific blocks)
mask_positions = [(0, 0), (1, 1), (2, 2)]  # Block diagonal positions
mask_image = CFAImage.get_mask_from_boxes(raw_blocks, mask_positions)

# Apply mask
masked_image = (merged_image * mask_image).astype(np.uint8)
```

### ROI Extraction

```python
# Extract ROI based on multifractal properties
rgb_image = cv2.imread('./images/face.png')

mask_union, masked_image = CFAImage.get_roi_by_q(
    image=rgb_image,
    q_range=(-5, 5),
    step=1.0,
    box_size=16,
    target_mass=0.90,
    combine_mode="or",
    use_grayscale_measure=True,
    measure_mode=0
)

# Visualize
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
ax1.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax2.imshow(mask_union, cmap='gray')
ax2.set_title('ROI Mask')
ax3.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
ax3.set_title('Extracted ROI')
plt.show()
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImage

**Description**: Image processing utility class providing static methods for image cropping, padding, blocking, merging, and other operations.

#### Static Methods List

All methods are static, called using `CFAImage.method_name()`.

##### 1. otsu_binarize(img)

**Description**: Auto-threshold image using Otsu method.

**Parameters**:
- `img` (ndarray): Input image (grayscale or color)

**Return Value** (tuple):
```python
(bin_img, threshold)
```
- `bin_img`: Binary image (uint8, values 0 or 255)
- `threshold`: Auto-calculated threshold

**Features**:
- Automatically converts color images to grayscale
- Automatically handles float and uint8 types
- Otsu method minimizes intra-class variance

##### 2. crop_data(data, block_size)

**Description**: Crop image so dimensions are multiples of block_size.

**Parameters**:
- `data` (ndarray): Input image (2D or 3D)
- `block_size` (tuple): (bh, bw) block size

**Return Value**: Cropped image

##### 3. pad_data(data, block_size, mode="constant", constant_values=0)

**Description**: Pad image so dimensions are multiples of block_size.

**Parameters**:
- `data` (ndarray): Input image
- `block_size` (tuple): Block size
- `mode` (str): Padding mode ('constant', 'edge', 'reflect', etc.)
- `constant_values`: Constant padding value (when mode='constant')

**Return Value**: Padded image

##### 4. get_boxes_from_image(image, block_size, corp_type=-1)

**Description**: Split image into fixed-size blocks.

**Parameters**:
- `image` (ndarray): Input image
- `block_size` (tuple): (bh, bw) block size
- `corp_type` (int): Crop type
  - `-1`: Auto crop
  - `0`: No processing (requires size already matched)
  - `1`: Padding

**Return Value** (tuple):
```python
(blocks_reshaped, raw_blocks)
```

- **Grayscale image**:
  - `blocks_reshaped`: (num_blocks, bh, bw)
  - `raw_blocks`: (nY, nX, bh, bw)

- **Color image**:
  - `blocks_reshaped`: (num_blocks, bh, bw, c)
  - `raw_blocks`: (nY, nX, bh, bw, c)

##### 5. get_image_from_boxes(raw_blocks)

**Description**: Merge blocks back into image.

**Parameters**:
- `raw_blocks` (ndarray): Block array (raw_blocks returned by get_boxes_from_image)

**Return Value**: Merged image

**Supported Shapes**:
- Grayscale: (nY, nX, bh, bw) → (H, W)
- Color: (nY, nX, bh, bw, c) → (H, W, c)

##### 6. get_mask_from_boxes(raw_blocks, mask_block_pos)

**Description**: Generate binary mask from selected block positions.

**Parameters**:
- `raw_blocks` (ndarray): Block array from `get_boxes_from_image`
- `mask_block_pos` (list): List of (row, col) block positions to include in mask

**Return Value**: Binary mask image (same spatial size as original)

##### 7. get_random_patches(image, num_patches=100, ratio=0.25)

**Description**: Randomly sample rectangular patches from image (possibly partially overlapping).

**Parameters**:
- `image` (ndarray): Input image
- `num_patches` (int): Number of patches to sample
- `ratio` (float): Patch size ratio (relative to image dimensions)

**Return Value**: List of image patches

**Features**:
- Sampling without replacement (different top-left coordinates)
- May have partial overlap
- Suitable for data augmentation

##### 8. get_roi_by_q(image, q_range=(-5,5), step=1.0, box_size=16, target_mass=0.95, combine_mode="and", use_grayscale_measure=True, measure_mode=0)

**Description**: Extract region of interest (ROI) based on multifractal q-weighting.

**Parameters**:
- `image` (ndarray): Input image (grayscale or RGB)
- `q_range` (tuple): Range of q values
- `step` (float): Step size for q values
- `box_size` (int): Block size for mass computation
- `target_mass` (float): Fraction of total mass to include (0-1)
- `combine_mode` (str): How to combine masks across q values — `"and"` or `"or"`
- `use_grayscale_measure` (bool): Use grayscale intensity for mass
- `measure_mode` (int): Mass measure mode

**Return Value** (tuple):
```python
(mask_union, masked_image)
```
- `mask_union`: Boolean mask (H, W)
- `masked_image`: Image with mask applied

## Theoretical Background

### Otsu Thresholding

Otsu method automatically selects threshold by minimizing intra-class variance:

```
σ²ω(t) = ω₀(t)σ²₀(t) + ω₁(t)σ²₁(t)
```

Where:
- ω₀, ω₁: Weights of two classes
- σ²₀, σ²₁: Variances of two classes
- t: Threshold

Optimal threshold:
```
t* = argmin σ²ω(t)
```

## Important Notes

1. **Image Types**:
   - Supports grayscale (2D) and color (3D)
   - Automatically handles different data types

2. **Blocking Operations**:
   - Uses `view_as_blocks` to avoid copying
   - Supports grayscale and color images
   - Channel dimensions remain unchanged

3. **corp_type Selection**:
   - `-1`: Suitable for most cases, loses boundaries
   - `0`: Strict mode, requires pre-adjusted size
   - `1`: Padding mode, preserves all pixels

4. **Performance**:
   - Blocking operations use views, memory efficient
   - ROI extraction is compute-intensive, slow for large images

## References

- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*.
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
