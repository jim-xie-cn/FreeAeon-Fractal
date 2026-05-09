# Image Processing Utilities - CFAImage

## Application Scenarios

The `CFAImage` class provides basic image processing utility functions for use with the fractal analysis pipeline. Main application scenarios include:

- **Preprocessing**: Binarization and cropping before fractal analysis
- **Block Operations**: Splitting images into boxes for multifractal computation
- **Mask Generation**: Create spatial masks based on block positions
- **ROI Extraction**: Extract regions of interest based on multifractal measure
- **Data Augmentation**: Random patch sampling for deep learning workflows

## Usage Examples

### Otsu Binarization

```python
import cv2
from FreeAeonFractal.FAImage import CFAImage

gray = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# Automatic Otsu thresholding
bin_image, threshold = CFAImage.otsu_binarize(gray)
print(f"Threshold: {threshold}")
```

### Image Blocking and Merging

```python
import numpy as np
from FreeAeonFractal.FAImage import CFAImage

image = np.zeros((256, 256), dtype=np.uint8)

# Split into 64×64 blocks
blocks, raw_blocks = CFAImage.get_boxes_from_image(image, block_size=(64, 64), corp_type=-1)
print("Number of blocks:", blocks.shape[0])  # 16 blocks (4×4 grid)

# Merge back to image
merged = CFAImage.get_image_from_boxes(raw_blocks)
```

### Mask from Block Positions

```python
# Zero out specific blocks
mask_pos = [(0, 0), (1, 1), (2, 2)]   # (row, col) in block-grid coordinates
mask_image = CFAImage.get_mask_from_boxes(raw_blocks, mask_pos)
masked_image = (merged * mask_image).astype(np.uint8)
```

### Crop and Pad

```python
# Crop to multiples of block size
cropped = CFAImage.crop_data(image, block_size=(64, 64))

# Pad to multiples of block size
padded = CFAImage.pad_data(image, block_size=(64, 64), mode="constant", constant_values=0)
```

### Random Patch Sampling

```python
# Sample 100 random 64×64 patches from the image
patches = CFAImage.get_random_patches(image, num_patches=100, ratio=0.25)
```

### ROI Extraction by Multifractal q

```python
import cv2
from FreeAeonFractal.FAImage import CFAImage

image = cv2.imread('./images/face.png')

# Extract ROI based on multifractal measure
mask_union, masked_image = CFAImage.get_roi_by_q(
    image=image,
    q_range=(-5, 5),
    step=1.0,
    box_size=16,
    target_mass=0.90,
    combine_mode="or",
    use_grayscale_measure=True
)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImage

**Description**: Static utility class for image block operations, binarization, mask generation, and multifractal-based ROI extraction.

#### Crop and Pad

##### crop_data(data, block_size)

**Description**: Crop spatial dimensions so H and W are multiples of block_size.

**Parameters**:
- `data` (ndarray): 2D grayscale or 3D color image
- `block_size` (tuple): `(bh, bw)` block dimensions

**Return Value**: Cropped array

##### pad_data(data, block_size, mode="constant", constant_values=0)

**Description**: Pad spatial dimensions so H and W are multiples of block_size.

**Parameters**:
- `data` (ndarray): 2D or 3D image
- `block_size` (tuple): `(bh, bw)`
- `mode` (str): NumPy padding mode (e.g., `"constant"`, `"reflect"`)
- `constant_values` (int): Fill value when `mode="constant"`

**Return Value**: Padded array

#### Binarization

##### otsu_binarize(img)

**Description**: Apply Otsu automatic thresholding to produce a binary image.

**Parameters**:
- `img` (ndarray): 2D grayscale or 3D color image (float or uint8)

**Return Value**: `(bin_img, threshold)`
- `bin_img` (ndarray): Binary uint8 image with values {0, 255}
- `threshold` (float): Computed Otsu threshold

**Notes**: Color images are converted to grayscale first. Float images are scaled to [0, 255] before thresholding.

#### Block Operations

##### get_boxes_from_image(image, block_size, corp_type=-1)

**Description**: Split image into blocks of size (bh, bw) over spatial dimensions. Channels are preserved.

**Parameters**:
- `image` (ndarray): 2D grayscale (H, W) or 3D color (H, W, C)
- `block_size` (tuple): `(bh, bw)`
- `corp_type` (int): `-1` crop (default), `1` pad, `0` strict (error if not divisible)

**Return Value**: `(blocks_reshaped, raw_blocks)`
- `blocks_reshaped`: `(num_blocks, bh, bw)` for grayscale, `(num_blocks, bh, bw, C)` for color
- `raw_blocks`: `(nY, nX, bh, bw)` or `(nY, nX, bh, bw, C)` grid layout

##### get_image_from_boxes(raw_blocks)

**Description**: Merge raw_blocks back into an image.

**Parameters**:
- `raw_blocks`: `(nY, nX, bh, bw)` or `(nY, nX, bh, bw, C)`

**Return Value**: Reconstructed image `(H, W)` or `(H, W, C)`

##### get_mask_from_boxes(raw_blocks, mask_block_pos)

**Description**: Build a binary mask image where blocks in `mask_block_pos` are set to 0, others to 1.

**Parameters**:
- `raw_blocks`: Grid layout (as returned by `get_boxes_from_image`)
- `mask_block_pos` (list of tuples): List of `(y, x)` block-grid coordinates to zero out

**Return Value**: `(H, W)` float32 mask in {0, 1}

#### Random Sampling

##### get_random_patches(image, num_patches=100, ratio=0.25)

**Description**: Randomly sample non-duplicate rectangular patches from an image.

**Parameters**:
- `image` (ndarray): Input image
- `num_patches` (int): Number of patches to extract
- `ratio` (float): Patch size as fraction of image dimensions (e.g., 0.25 → H/4 × W/4 patches)

**Return Value**: List of patch arrays

**Raises**: `ValueError` if `num_patches` exceeds the maximum possible unique positions.

#### ROI Extraction

##### get_roi_by_q(image, q_range=(-5,5), step=1.0, box_size=16, target_mass=0.95, combine_mode="and", use_grayscale_measure=True, measure_mode="intensity_sum")

**Description**: Extract ROI by selecting boxes with top cumulative mass weighted by μᵢ^q (multifractal reweighting).

**Parameters**:
- `image` (ndarray): 2D grayscale or 3D color image
- `q_range` (tuple): `(q_min, q_max)` range of moment orders
- `step` (float): Step size for q iteration
- `box_size` (int): Box size in pixels
- `target_mass` (float): Cumulative mass fraction to retain (e.g., 0.90 = top 90% of weighted mass)
- `combine_mode` (str): How to combine masks across channels: `"and"` or `"or"`
- `use_grayscale_measure` (bool): For color images, use grayscale conversion for measure computation
- `measure_mode` (str): `"intensity_sum"` — box measure = sum of pixel values

**Return Value**: `(mask_union, masked_image)`
- `mask_union`: `(H, W)` bool array
- `masked_image`: Same shape as input; outside mask is zeroed

**Algorithm**:
1. Crop image to multiples of `box_size`
2. Compute box measure μᵢ = box_sum / total_sum
3. For each q, compute weights wᵢ ∝ μᵢ^q on active (μᵢ > 0) boxes
4. Select top boxes by cumulative weight until `target_mass` is reached
5. OR masks across all q values, then combine across channels

## Important Notes

1. **corp_type in get_boxes_from_image**:
   - `-1` (crop) is the most commonly used; `1` (pad) preserves all pixels but adds zeros

2. **ROI Extraction**:
   - Large `q` (positive) selects high-intensity/dense regions
   - Large `|q|` (negative) selects low-intensity/sparse regions
   - `target_mass` controls ROI coverage; lower values give tighter ROIs

3. **Block Coordinate System**:
   - `raw_blocks[y, x, ...]` corresponds to pixels `[y*bh:(y+1)*bh, x*bw:(x+1)*bw]`

4. **Color Image Handling**:
   - `get_boxes_from_image` preserves all channels; block shape includes C
   - `otsu_binarize` converts color to grayscale automatically
