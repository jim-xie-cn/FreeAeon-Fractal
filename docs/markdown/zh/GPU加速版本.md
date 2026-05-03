# GPU加速版本

## 应用场景

FreeAeon-Fractal提供GPU加速版本的核心计算模块，显著提升大图像和多尺度分析的计算速度。主要应用场景包括：

- **大规模图像分析**：处理高分辨率图像（>1024x1024）
- **批量处理**：同时分析多个图像
- **实时分析**：需要快速响应的应用
- **多尺度分析**：大量尺度参数的计算

## 支持的GPU模块

| CPU模块 | GPU模块 | 加速比 |
|---------|---------|--------|
| `FAImageMFS.CFAImageMFS` | `FAImageMFSGPU.CFAImageMFSGPU` | 5-20x |
| `FAImageFD.CFAImageFD` | `FAImageFDGPU.CFAImageFDGPU` | 3-10x |
| `FAImageLAC.CFAImageLAC` | `FAImageLACGPU.CFAImageLACGPU` | 5-15x |

**注意**：实际加速比取决于图像大小、GPU型号和参数设置。

## 系统要求

### 硬件要求
- NVIDIA GPU（支持CUDA）
- 建议显存 ≥ 4GB
- 推荐显存 ≥ 8GB（处理大图像）

### 软件要求
```bash
# 安装PyTorch（CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装FreeAeon-Fractal
pip install FreeAeon-Fractal
```

### 验证CUDA可用性
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## 调用示例

### GPU加速的2D多重分形谱

```python
import cv2
import numpy as np
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU

# 读取图像
gray_image = cv2.imread('./images/face.png', cv2.IMREAD_GRAYSCALE)

# 创建GPU版本（单图像默认dtype=torch.float64）
MFS = CFAImageMFSGPU(
    gray_image,
    q_list=np.linspace(-5, 5, 26),
    device='cuda',
    dtype=None  # 自动：单图像float64，批处理float32
)

# 计算（GPU加速）
df_mass, df_fit, df_spec = MFS.get_mfs()

# 可视化
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU加速的分形维度

```python
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU
from FreeAeonFractal.FAImage import CFAImage

gray_image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)
bin_image, _ = CFAImage.otsu_binarize(gray_image)

fd_gpu = CFAImageFDGPU(bin_image, device='cuda')
fd_bc = fd_gpu.get_bc_fd()
fd_dbc = fd_gpu.get_dbc_fd()

# 注意：GPU版本的p_value为None（不计算）
print(f"BC (GPU): {fd_bc['fd']:.4f}")
print(f"DBC (GPU): {fd_dbc['fd']:.4f}")
```

### GPU加速的空隙度分析

```python
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU

lacunarity_gpu = CFAImageLACGPU(
    gray_image,
    max_scales=100,
    device='cuda'
)

lac_result = lacunarity_gpu.get_lacunarity()
fit_result = lacunarity_gpu.fit_lacunarity(lac_result)

lacunarity_gpu.plot(lac_result, fit_result)
```

### 使用别名简化代码

```python
# 在代码开头使用别名，后续代码无需修改
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD

# 后续代码与CPU版本完全相同
MFS = CFAImageMFS(image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

## GPU版本特有参数

### 1. CFAImageMFSGPU

额外参数：
- `device` (str): 设备选择（'cuda', 'cpu', 'cuda:0'等）
- `dtype` (torch.dtype): 数据类型 — 单图像默认`torch.float64`，批处理默认`torch.float32`
- `q_chunk` (int): 每次GPU计算的q值数量（控制显存）
- `img_chunk` (int): 批处理模式下每次GPU计算的图像数量

**示例**：
```python
MFS = CFAImageMFSGPU(
    image,
    device='cuda:0',           # 使用第一块GPU
    dtype=torch.float32,       # 使用单精度
    q_chunk=10,                # 每次处理10个q值
)
```

### 2. CFAImageFDGPU

额外参数：
- `device` (str): 设备选择
- `dtype` (torch.dtype): 数据类型
- `img_chunk` (int): 批处理分块大小

**注意**：GPU结果中 `p_value` 字段为 `None`（不计算）。使用 `torch.quantile` 进行99百分位归一化，回归使用手动OLS（不依赖SciPy）。

### 3. CFAImageLACGPU

额外参数：
- `device` (str): 设备选择
- `dtype` (torch.dtype): 默认 `torch.float64`
- `img_chunk` (int): 批处理分块大小

**注意**：GPU批处理模式要求所有图像形状相同。

## 性能对比

### 测试环境
- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3080 (10GB)
- 图像: 1024x1024 灰度图

### 性能数据

| 任务 | CPU时间 | GPU时间 | 加速比 |
|------|---------|---------|--------|
| 2D多重分形谱 (q=51) | 45s | 4s | 11.3x |
| 分形维度 BC | 8s | 2s | 4.0x |
| 分形维度 DBC | 25s | 3s | 8.3x |
| 空隙度分析 | 35s | 4s | 8.8x |

## 最佳实践

1. **开发阶段**：使用CPU版本快速测试
2. **生产环境**：使用GPU版本处理大批量数据
3. **参数调优**：在CPU上测试参数，在GPU上运行
4. **精度选择**：单图像使用float64保证精度，批处理使用float32提升吞吐量
5. **错误处理**：捕获CUDA异常，优雅降级到CPU

## 故障排除

### CUDA内存不足
```python
# 解决方案：
# 1. 减小图像尺寸
# 2. 使用float32
# 3. 减小q_chunk或img_chunk
# 4. 清除缓存
import torch
torch.cuda.empty_cache()
```

## 迁移指南

从CPU版本迁移到GPU版本只需修改导入语句：

**CPU版本**：
```python
from FreeAeonFractal.FAImageMFS import CFAImageMFS
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImageLAC import CFAImageLAC
```

**GPU版本**：
```python
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
# 其余代码完全相同！
```

### 条件导入（智能切换）

```python
import torch

if torch.cuda.is_available():
    from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
    print("使用GPU加速")
else:
    from FreeAeonFractal.FAImageMFS import CFAImageMFS
    print("使用CPU")

MFS = CFAImageMFS(image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

## 参考资源

- **PyTorch官网**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
