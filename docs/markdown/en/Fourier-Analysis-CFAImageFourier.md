# Fourier Analysis - CFAImageFourier

## Application Scenarios

The `CFAImageFourier` class provides tools for Fourier spectrum analysis of images. Main application scenarios include:

- **Periodic Noise Removal**: Identify and filter periodic interference patterns
- **Texture Analysis**: Extract directional and periodic features from frequency domain
- **Image Enhancement**: Frequency-domain sharpening and smoothing
- **Pattern Recognition**: Frequency-based texture classification
- **Signal Processing**: Frequency component extraction and reconstruction

## Usage Examples

### Basic Usage

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

# Read image (supports grayscale and RGB)
image = cv2.imread('./images/face.png')

# Create Fourier analysis object (FFT computed in __init__)
fourier = CFAImageFourier(image)

# Get raw magnitude and phase (for reconstruction)
raw_mag, raw_phase = fourier.get_raw_spectrum()

# Get display-enhanced spectrum visualization
mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)

# Reconstruct image from full spectrum
full_reconstructed = fourier.get_reconstruct()

# Visualize
fourier.plot(
    raw_magnitude_disp=mag_disp,
    raw_phase_disp=phase_disp,
    full_reconstructed=full_reconstructed
)
```

### Frequency Domain Filtering with Custom Mask

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier

image = cv2.imread('./images/face.png')
fourier = CFAImageFourier(image)

raw_mag, raw_phase = fourier.get_raw_spectrum()
mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)

# Create a custom frequency mask (example: keep only odd-frequency components)
h, w = raw_mag[0].shape
Y, X = np.ogrid[:h, :w]
mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

# Apply mask and visualize customized spectrum
customized_mag = raw_mag * mask
customized_phase = raw_phase * mask
custom_mag_disp, custom_phase_disp = fourier.get_display_spectrum(
    alpha=1.5, magnitude=customized_mag, phase=customized_phase
)

# Reconstruct masked image
masked_reconstructed = fourier.extract_by_freq_mask(mask)

fourier.plot(
    raw_magnitude_disp=mag_disp,
    raw_phase_disp=phase_disp,
    customized_magnitude_disp=custom_mag_disp,
    customized_phase_disp=custom_phase_disp,
    full_reconstructed=full_reconstructed,
    mask_reconstructed=masked_reconstructed
)
```

### Grayscale Image

```python
image = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)
fourier = CFAImageFourier(image)
mag_disp, phase_disp = fourier.get_display_spectrum()
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAImageFourier

**Description**: Provides Fourier analysis for grayscale or RGB images. The 2D FFT is computed at initialization. Supports magnitude/phase decomposition, enhanced visualization, image reconstruction, and custom frequency masking.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | numpy.ndarray | Required | Input grayscale (H,W) or RGB (H,W,3) image |

**Note**: The FFT is computed in `__init__`. For grayscale images, one magnitude/phase pair is stored. For RGB images, three channel pairs are stored.

#### Main Methods

##### 1. get_raw_spectrum()

**Description**: Get raw magnitude and phase data (complex spectrum components).

**Return Value**:
- `magnitude` (list of ndarray): Per-channel magnitude arrays (not log-scaled)
- `phase` (list of ndarray): Per-channel phase arrays in radians [−π, π]

**Use Case**: Use raw spectrum for reconstruction; do NOT use display spectrum for reconstruction.

##### 2. get_display_spectrum(alpha=1.0, beta=0, magnitude=array([]), phase=array([]))

**Description**: Generate enhanced visualizations of magnitude and phase for display.

**Parameters**:
- `alpha` (float): Contrast scaling factor (default: 1.0)
- `beta` (float): Brightness offset (default: 0)
- `magnitude` (ndarray): Optional custom magnitude to visualize instead of stored one
- `phase` (ndarray): Optional custom phase to visualize instead of stored one

**Return Value**:
- `display_mag` (list): 8-bit normalized magnitude images (log-scaled: log(1 + |mag|))
- `display_phase` (list): 8-bit normalized phase images

##### 3. get_reconstruct(magnitude=array([]), phase=array([]))

**Description**: Reconstruct spatial-domain image from magnitude and phase.

**Parameters**:
- `magnitude` (ndarray): Optional custom magnitude (default: stored raw spectrum)
- `phase` (ndarray): Optional custom phase (default: stored raw spectrum)

**Return Value**: Reconstructed uint8 image (grayscale or BGR)

**Algorithm**: `IFFT(IFFTSHIFT(mag * exp(1j * phase)))`, then normalize to [0, 255].

##### 4. extract_by_freq_mask(mask_mag=array([]), mask_phase=array([]))

**Description**: Apply a binary mask to the frequency domain and reconstruct.

**Parameters**:
- `mask_mag` (ndarray): Binary mask for magnitude (1=keep, 0=zero out); same shape as frequency map
- `mask_phase` (ndarray): Binary mask for phase

**Return Value**: Reconstructed image after masking

**Use Case**: Frequency-domain filtering — pass high/low/band-pass masks to selectively retain frequency components.

##### 5. plot(raw_magnitude_disp=[], raw_phase_disp=[], customized_magnitude_disp=[], customized_phase_disp=[], full_reconstructed=array([]), mask_reconstructed=array([]))

**Description**: Display original, magnitude, phase, and reconstructed images side-by-side in a 2-row grid.

**Parameters**:
- `raw_magnitude_disp` (list): Visualized raw magnitude
- `raw_phase_disp` (list): Visualized raw phase
- `customized_magnitude_disp` (list): Visualized masked magnitude
- `customized_phase_disp` (list): Visualized masked phase
- `full_reconstructed` (ndarray): Full reconstruction result
- `mask_reconstructed` (ndarray): Masked reconstruction result

##### 6. get_image_components(image) [static]

**Description**: Compute magnitude and phase from a single-channel image using 2D FFT.

**Parameters**:
- `image` (ndarray): Single-channel image

**Return Value**: `(magnitude, phase)` — raw (not log-scaled) arrays

##### 7. normalize_and_enhance(array, alpha=1.0, beta=0) [static]

**Description**: Normalize an array to [0, 255] and apply linear contrast enhancement.

**Return Value**: 8-bit uint8 array suitable for visualization

## Algorithm Description

### 2D FFT Pipeline

```
1. FFT2: F = np.fft.fft2(image)
2. Shift:  F_shift = np.fft.fftshift(F)   (DC component moved to center)
3. Decompose: magnitude = |F_shift|, phase = angle(F_shift)
```

### Reconstruction Pipeline

```
1. Combine: F_shift = magnitude * exp(1j * phase)
2. Unshift: F = np.fft.ifftshift(F_shift)
3. IFFT2: img = Re(np.fft.ifft2(F))
4. Normalize to [0, 255]
```

### Display Enhancement

The magnitude is log-scaled before display to compress the dynamic range:
```
display = normalize(log(1 + |magnitude|))
```

Phase is shifted from [−π, π] to [0, 1] before normalization.

## Frequency Filtering Examples

### Low-pass Filter (smooth/blur)

```python
h, w = raw_mag[0].shape
cy, cx = h // 2, w // 2
radius = 30   # keep frequencies within this distance from center
Y, X = np.ogrid[:h, :w]
mask = ((X - cx)**2 + (Y - cy)**2 <= radius**2).astype(np.uint8)
result = fourier.extract_by_freq_mask(mask)
```

### High-pass Filter (sharpen/edge detection)

```python
mask = 1 - low_pass_mask   # invert the low-pass mask
result = fourier.extract_by_freq_mask(mask)
```

## Important Notes

1. **Input Format**:
   - Supports 2D grayscale (H, W) or 3D RGB (H, W, 3)
   - Raises `ValueError` for other formats

2. **Raw vs Display Spectrum**:
   - Use `get_raw_spectrum()` for reconstruction (non-log-scaled)
   - Use `get_display_spectrum()` only for visualization

3. **Channel Handling**:
   - RGB: three independent FFTs, one per channel
   - Reconstruction merges channels back with `cv2.merge`

4. **Mask Application**:
   - Mask must match the shape of the frequency map `(H, W)`
   - Use 0/1 binary masks; floating-point masks are also supported

## References

- Brigham, E. O. (1988). *The Fast Fourier Transform and Its Applications*. Prentice-Hall.
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
