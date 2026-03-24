# GPU Acceleration

## Application Scenarios

FreeAeon-Fractal provides GPU-accelerated versions of core computational modules, significantly improving computation speed for large images and multi-scale analysis.

## Supported GPU Modules

| CPU Module | GPU Module | Speedup |
|------------|------------|---------|
| `FA2DMFS.CFA2DMFS` | `FA2DMFSGPU.CFA2DMFSGPU` | 5-20x |
| `FAImageDimension.CFAImageDimension` | `FAImageDimensionGPU.CFAImageDimensionGPU` | 3-10x |
| `FAImageLacunarity.CFAImageLacunarity` | `FAImageLacunarityGPU.CFAImageLacunarityGPU` | 5-15x |

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
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU

# Load image
gray_image = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# Create GPU version
MFS = CFA2DMFSGPU(
    gray_image,
    q_list=np.linspace(-5, 5, 26),
    device='cuda'
)

# Compute (GPU accelerated)
df_mass, df_fit, df_spec = MFS.get_mfs()

# Visualize
MFS.plot(df_mass, df_fit, df_spec)
```

### Easy Migration from CPU to GPU

**CPU version**:
```python
from FreeAeonFractal.FA2DMFS import CFA2DMFS
```

**GPU version** (just change import):
```python
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
# Rest of code remains identical!
```

### Auto Device Selection

```python
import torch

# Automatically choose based on GPU availability
if torch.cuda.is_available():
    from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
    print("Using GPU acceleration")
else:
    from FreeAeonFractal.FA2DMFS import CFA2DMFS
    print("Using CPU")

MFS = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))
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

### CFA2DMFSGPU

Additional parameters:
- `device` (str): Device selection ('cuda', 'cpu', 'cuda:0', etc.)
- `dtype` (torch.dtype): Data type (torch.float32 or torch.float64)

**Example**:
```python
MFS = CFA2DMFSGPU(
    image,
    device='cuda:0',           # Use first GPU
    dtype=torch.float32        # Use single precision
)
```

## Best Practices

1. **Development**: Use CPU version for quick testing
2. **Production**: Use GPU version for large-scale data
3. **Parameter Tuning**: Test on CPU, run on GPU
4. **Memory Management**: Monitor VRAM usage regularly
5. **Error Handling**: Gracefully fallback to CPU on errors

## Troubleshooting

### CUDA Out of Memory
```python
# Solutions:
# 1. Reduce image size
# 2. Use float32
# 3. Reduce parameters
# 4. Clear cache
torch.cuda.empty_cache()
```

## References

- **PyTorch**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
