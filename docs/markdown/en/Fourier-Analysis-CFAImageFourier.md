# Fourier Analysis - CFAImageFourier

## Application Scenarios

The `CFAImageFourier` class provides tools for Fourier spectrum analysis of images, used to analyze frequency components and perform frequency domain processing. Main application scenarios include:

- **Frequency Domain Analysis**: Analyze frequency distribution characteristics of images
- **Image Filtering**: High-pass, low-pass, and band-pass filtering through frequency domain masks
- **Texture Analysis**: Extract periodicity and directionality features
- **Image Enhancement**: Frequency domain image enhancement and restoration
- **Pattern Recognition**: Image classification based on spectral features

## Usage Examples

### Basic Usage

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

# Read image (supports grayscale or RGB)
rgb_image = cv2.imread('./images/face.png')

# Create Fourier analysis object
fourier = CFAImageFourier(rgb_image)

# Get raw spectrum
raw_mag, raw_phase = fourier.get_raw_spectrum()

# Get display spectrum
raw_mag_disp, raw_phase_disp = fourier.get_display_spectrum(alpha=1.5)

# Reconstruct image
reconstructed = fourier.get_reconstruct()

# Display results
fourier.plot(raw_mag_disp, raw_phase_disp,
             [], [], reconstructed, np.array([]))
```

### Frequency Domain Filtering Example

```python
# Read grayscale image
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
fourier = CFAImageFourier(gray_image)

# Get raw spectrum
raw_mag, raw_phase = fourier.get_raw_spectrum()

# Create frequency mask (preserve odd frequencies)
h, w = raw_mag[0].shape
Y, X = np.ogrid[:h, :w]
mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

# Apply mask to spectrum
masked_mag = raw_mag * mask
masked_phase = raw_phase * mask

# Get masked display spectrum
customized_mag_disp, customized_phase_disp = fourier.get_display_spectrum(
    alpha=1.5,
    magnitude=masked_mag,
    phase=masked_phase
)

# Reconstruct filtered image
masked_reconstructed = fourier.extract_by_freq_mask(mask)

# Visualize comparison
fourier.plot(raw_mag_disp, raw_phase_disp,
             customized_mag_disp, customized_phase_disp,
             reconstructed, masked_reconstructed)
```

### Low-Pass Filter

```python
# Create low-pass filter mask (preserve low-frequency components)
h, w = raw_mag[0].shape
center_y, center_x = h // 2, w // 2
radius = 30  # Cutoff radius

Y, X = np.ogrid[:h, :w]
distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
low_pass_mask = (distance <= radius).astype(np.uint8)

# Apply low-pass filter
low_pass_result = fourier.extract_by_freq_mask(low_pass_mask)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageFourier

**Description**: Provides image Fourier analysis tools, supporting frequency domain analysis and reconstruction of grayscale and RGB images.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input image (grayscale or RGB) |

**Note**: Fourier transform is automatically calculated during initialization, no manual call needed.

#### Main Methods

##### 1. get_raw_spectrum()

**Description**: Get raw magnitude and phase spectrum (for reconstruction).

**Parameters**: None

**Return Value** (tuple):
```python
(magnitude_list, phase_list)
```
- `magnitude_list`: Magnitude spectrum list (one per channel)
- `phase_list`: Phase spectrum list (one per channel)

For grayscale images, list length is 1; for RGB images, length is 3.

##### 2. get_display_spectrum(alpha=1.0, beta=0, magnitude=np.array([]), phase=np.array([]))

**Description**: Generate enhanced visualization of spectrum images.

**Parameters**:
- `alpha` (float): Contrast enhancement factor (default 1.0)
- `beta` (float): Brightness offset (default 0)
- `magnitude` (array): Custom magnitude spectrum (optional)
- `phase` (array): Custom phase spectrum (optional)

**Return Value** (tuple):
```python
(display_mag_list, display_phase_list)
```
Returns lists of 8-bit images normalized to 0-255, suitable for visualization.

**Enhancement Notes**:
- Magnitude uses logarithmic transform: log(1 + mag)
- Phase normalized to [0, 1]

##### 3. get_reconstruct(magnitude=np.array([]), phase=np.array([]))

**Description**: Reconstruct spatial domain image from frequency domain.

**Parameters**:
- `magnitude` (array): Magnitude spectrum (optional, default uses original)
- `phase` (array): Phase spectrum (optional, default uses original)

**Return Value**:
- Grayscale image: (H, W) uint8 array
- RGB image: (H, W, 3) uint8 array

##### 4. extract_by_freq_mask(mask_mag=np.array([]), mask_phase=np.array([]))

**Description**: Use binary mask to selectively preserve frequency components, then reconstruct image.

**Parameters**:
- `mask_mag` (array): Magnitude mask (same shape as spectrum, 1=keep/0=remove)
- `mask_phase` (array): Phase mask (optional)

**Return Value**: Reconstructed filtered image

**Applications**:
- High-pass/low-pass filtering
- Directional filtering
- Periodic noise removal

##### 5. plot(raw_magnitude_disp, raw_phase_disp, customized_magnitude_disp, customized_phase_disp, full_reconstructed, mask_reconstructed)

**Description**: Visualize analysis results, displaying up to 7 subplots.

**Subplot Layout**:
1. Original image
2. Raw magnitude spectrum
3. Raw phase spectrum
4. Customized magnitude spectrum
5. Customized phase spectrum
6. Full reconstruction
7. Masked reconstruction

## Theoretical Background

### 2D Fourier Transform

For image I(x, y), its 2D Fourier transform is:

```
F(u, v) = ∫∫ I(x, y) exp[-j2π(ux + vy)] dx dy
```

Discrete form:

```
F(u, v) = Σₓ Σᵧ I(x, y) exp[-j2π(ux/M + vy/N)]
```

### Spectrum Components

**Magnitude Spectrum**:
```
|F(u, v)| = √[R²(u, v) + I²(u, v)]
```

**Phase Spectrum**:
```
φ(u, v) = arctan[I(u, v) / R(u, v)]
```

### Frequency Shift

Uses `np.fft.fftshift` to move zero-frequency component to center for easier visualization.

### Inverse Transform

```
I(x, y) = ∫∫ F(u, v) exp[j2π(ux + vy)] du dv
```

## Important Notes

1. **Input Image**:
   - Supports grayscale and RGB images
   - Automatically performs Fourier transform

2. **Spectrum Visualization**:
   - Uses logarithmic transform for better visualization
   - `alpha` parameter controls contrast

3. **Mask Design**:
   - Mask size must match spectrum
   - Binary mask: 1=keep, 0=remove
   - Symmetric masks maintain real-valued images

4. **RGB Processing**:
   - Each channel processed independently
   - Finally merged for reconstruction

5. **Performance Considerations**:
   - FFT complexity: O(N log N)
   - Large images may be slow
   - Use grayscale for speedup

## References

- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.).
- Bracewell, R. N. (2000). The Fourier Transform and Its Applications (3rd ed.).
