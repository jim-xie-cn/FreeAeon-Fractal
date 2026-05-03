# GPU Acceleration

## Application Scenarios

FreeAeon-Fractal provides GPU-accelerated versions of core computational modules, significantly improving computation speed for large images and multi-scale analysis.

## Supported GPU Modules

| CPU Module | GPU Module | Speedup |
|------------|------------|---------|
| `FAImageMFS.CFAImageMFS` | `FAImageMFSGPU.CFAImageMFSGPU` | 5-20x |
| `FAImageFD.CFAImageFD` | `FAImageFDGPU.CFAImageFDGPU` | 3-10x |
| `FAImageLAC.CFAImageLAC` | `FAImageLACGPU.CFAImageLACGPU` | 5-15x |

## System Requirements

### Hardware
- NVIDIA GPU (CUDA support)
- Recommended VRAM ≥ 4GB
- Preferred VRAM ≥ 8GB (for large images)

### Software
```bash
# Install PyTorch (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FreeAeon-Fractal
pip install FreeAeon-Fractal
```

## Usage Examples

### GPU-Accelerated 2D Multifractal Spectrum

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU

# Load image
gray_image = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# Create GPU version (default dtype=torch.float64 for single image)
MFS = CFAImageMFSGPU(
    gray_image,
    q_list=np.linspace(-5, 5, 26),
    device='cuda',
    dtype=None  # auto: float64 for single, float32 for batch
)

# Compute (GPU accelerated)
df_mass, df_fit, df_spec = MFS.get_mfs()

# Visualize
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU-Accelerated Fractal Dimension

```python
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU
from FreeAeonFractal.FAImage import CFAImage

bin_image, _ = CFAImage.otsu_binarize(gray_image)

fd_gpu = CFAImageFDGPU(bin_image, device='cuda')
fd_bc = fd_gpu.get_bc_fd()
fd_dbc = fd_gpu.get_dbc_fd()
fd_sdbc = fd_gpu.get_sdbc_fd()

# Note: p_value is None in GPU version (not computed)
print("BC FD:", fd_bc['fd'])
```

### GPU-Accelerated Lacunarity

```python
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU

lac_gpu = CFAImageLACGPU(gray_image, device='cuda', dtype=None)
lac_result = lac_gpu.get_lacunarity()
```

### Easy Migration from CPU to GPU

**CPU version**:
```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS
```

**GPU version** (just change import):
```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
# Rest of code remains identical!
```

### Auto Device Selection

```python
import torch

# Automatically choose based on GPU availability
if torch.cuda.is_available():
    from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
    print("Using GPU acceleration")
else:
    from FreeAeonFractal.FAImageMFS import CFAImageMFS
    print("Using CPU")

MFS = CFAImageMFS(image, q_list=np.linspace(-5, 5, 26))
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Performance Comparison

### Test Environment
- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3080 (10GB)
- Image: 1024x1024 grayscale

### Performance Data

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| 2D Multifractal (q=51) | 45s | 4s | 11.3x |
| Fractal Dim BC | 8s | 2s | 4.0x |
| Fractal Dim DBC | 25s | 3s | 8.3x |
| Lacunarity | 35s | 4s | 8.8x |

## GPU-Specific Parameters

### CFAImageMFSGPU

Additional parameters:
- `device` (str): Device selection ('cuda', 'cpu', 'cuda:0', etc.)
- `dtype` (torch.dtype): Data type — `torch.float64` (default single-image) or `torch.float32` (default batch)
- `q_chunk` (int): Number of q values processed per GPU pass (controls VRAM)
- `img_chunk` (int): Number of images per GPU pass in batch mode

**Example**:
```python
MFS = CFAImageMFSGPU(
    image,
    device='cuda:0',           # Use first GPU
    dtype=torch.float32,       # Use single precision
    q_chunk=10,                # Process 10 q values at a time
)
```

### CFAImageFDGPU

Additional parameters:
- `device` (str): Device selection
- `dtype` (torch.dtype): Data type
- `img_chunk` (int): Batch chunk size

**Note**: `p_value` field is `None` in GPU results (not computed). Uses `torch.quantile` for 99th percentile normalization. Regression uses manual OLS (no SciPy).

### CFAImageLACGPU

Additional parameters:
- `device` (str): Device selection
- `dtype` (torch.dtype): Default `torch.float64`
- `img_chunk` (int): Batch chunk size

**Note**: GPU batch mode requires all images to have the same shape.

## Best Practices

1. **Development**: Use CPU version for quick testing
2. **Production**: Use GPU version for large-scale data
3. **Parameter Tuning**: Test on CPU, run on GPU
4. **Memory Management**: Use `q_chunk` and `img_chunk` to control VRAM usage
5. **Precision**: Use float64 for single-image accuracy, float32 for batch throughput
6. **Error Handling**: Gracefully fallback to CPU on errors

## Troubleshooting

### CUDA Out of Memory
```python
# Solutions:
# 1. Reduce image size
# 2. Use float32
# 3. Reduce q_chunk or img_chunk
# 4. Clear cache
import torch
torch.cuda.empty_cache()
```

## References

- **PyTorch**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
