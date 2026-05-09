# GPU Acceleration

## Overview

FreeAeon-Fractal provides GPU-accelerated versions of core computational modules, significantly improving computation speed for large images and multi-scale analysis.

## Supported GPU Modules

| CPU Module | GPU Module | Typical Speedup |
|------------|------------|-----------------|
| `FAImageFD.CFAImageFD` | `FAImageFDGPU.CFAImageFDGPU` | 3–10× |
| `FAImageMFS.CFAImageMFS` | `FAImageMFSGPU.CFAImageMFSGPU` | 5–20× |
| `FAImageLAC.CFAImageLAC` | `FAImageLACGPU.CFAImageLACGPU` | 5–15× |

## Requirements

- NVIDIA GPU with CUDA support
- PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Usage

### Drop-in Import Replacement

The GPU classes share the same API as their CPU counterparts. Simply change the import:

```python
# CPU version
from FreeAeonFractal.FAImageMFS import CFAImageMFS

# GPU version (identical API)
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
```

### Fractal Dimension (GPU)

```python
import cv2
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU

gray = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)
bin_image = (gray < 64).astype('uint8')

fd_bc = CFAImageFDGPU(bin_image, device='cuda').get_bc_fd()
fd_dbc = CFAImageFDGPU(gray, device='cuda').get_dbc_fd()
fd_sdbc = CFAImageFDGPU(gray, device='cuda').get_sdbc_fd()

print("BC FD:", fd_bc['fd'])
print("DBC FD:", fd_dbc['fd'])
print("SDBC FD:", fd_sdbc['fd'])
```

### Multifractal Spectrum (GPU)

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

gray = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

MFS = CFAImageMFS(gray, q_list=np.linspace(-5, 5, 51))
df_mass, df_fit, df_spec = MFS.get_mfs()
MFS.plot(df_mass, df_fit, df_spec)
```

### Lacunarity (GPU)

```python
import cv2
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU

gray = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)

calc = CFAImageLACGPU(gray, device='cuda')
lac_result = calc.get_lacunarity()
fit_result = calc.fit_lacunarity(lac_result)
print("Slope:", fit_result['slope'])
```

### Batch Processing (GPU)

```python
import cv2, glob
import numpy as np
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob('./images/*.png')]

results = CFAImageMFSGPU.get_batch_mfs(
    images,
    q_list=np.linspace(-5, 5, 26),
    with_progress=True
)

for df_mass, df_fit, df_spec in results:
    print(df_fit[['q', 'Dq']].head(3))
```

## Performance Notes

| Scenario | Expected Speedup |
|----------|-----------------|
| Single large image (1024×1024) | 5–10× |
| Batch of 100+ images | 10–20× |
| Many q values (51+) | 5–15× |
| Many scales (80+) | 3–8× |

Speedup depends on GPU memory, image size, and number of q values. The GPU version uses float32 (vs float64 on CPU) for batch operations, which doubles throughput on tensor operations.

## API Differences from CPU

| Feature | CPU | GPU |
|---------|-----|-----|
| `p_value` | Computed | `None` (not computed) |
| Default dtype | float64 | float64 (single), float32 (batch) |
| `device` parameter | N/A | `'cuda'` or `'cpu'` |

## Fallback to CPU

If CUDA is unavailable, the GPU modules fall back to CPU computation. You can check availability:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```
