# GPU加速版本

## 概述

FreeAeon-Fractal 为核心计算模块提供GPU加速版本，显著提升大图像和多尺度分析的计算速度。

## 支持的GPU模块

| CPU模块 | GPU模块 | 典型加速比 |
|---------|---------|----------|
| `FAImageFD.CFAImageFD` | `FAImageFDGPU.CFAImageFDGPU` | 3–10× |
| `FAImageMFS.CFAImageMFS` | `FAImageMFSGPU.CFAImageMFSGPU` | 5–20× |
| `FAImageLAC.CFAImageLAC` | `FAImageLACGPU.CFAImageLACGPU` | 5–15× |

## 系统要求

- 支持CUDA的NVIDIA GPU
- 带CUDA的PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cu118`

## 使用方法

### 直接替换导入

GPU类与CPU版本共享相同API，只需更改导入语句：

```python
# CPU版本
from FreeAeonFractal.FAImageMFS import CFAImageMFS

# GPU版本（API完全相同）
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
```

### 分形维度（GPU）

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

### 多重分形谱（GPU）

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

gray = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

MFS = CFAImageMFS(gray, q_list=np.linspace(-5, 5, 51))
df_mass, df_fit, df_spec = MFS.get_mfs()
MFS.plot(df_mass, df_fit, df_spec)
```

### 空隙度（GPU）

```python
import cv2
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU

gray = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)

calc = CFAImageLACGPU(gray, device='cuda')
lac_result = calc.get_lacunarity()
fit_result = calc.fit_lacunarity(lac_result)
print("斜率:", fit_result['slope'])
```

### 批量处理（GPU）

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

## 性能说明

| 场景 | 预期加速比 |
|------|----------|
| 单张大图像（1024×1024） | 5–10× |
| 批量100+张图像 | 10–20× |
| 大量q值（51+） | 5–15× |
| 大量尺度（80+） | 3–8× |

加速效果取决于GPU显存、图像尺寸和q值数量。GPU版本批量操作使用float32（CPU使用float64），张量运算吞吐量翻倍。

## 与CPU版本的API差异

| 特性 | CPU | GPU |
|------|-----|-----|
| `p_value` | 已计算 | `None`（不计算） |
| 默认数据类型 | float64 | float64（单图），float32（批量） |
| `device` 参数 | 无 | `'cuda'` 或 `'cpu'` |

## CUDA可用性检查

```python
import torch
print("CUDA可用:", torch.cuda.is_available())
```

若CUDA不可用，GPU模块将回退到CPU计算。
