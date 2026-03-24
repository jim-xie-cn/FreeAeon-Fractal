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
| `FA2DMFS.CFA2DMFS` | `FA2DMFSGPU.CFA2DMFSGPU` | 5-20x |
| `FAImageDimension.CFAImageDimension` | `FAImageDimensionGPU.CFAImageDimensionGPU` | 3-10x |
| `FAImageLacunarity.CFAImageLacunarity` | `FAImageLacunarityGPU.CFAImageLacunarityGPU` | 5-15x |

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

# 或根据你的CUDA版本选择
# 访问 https://pytorch.org/ 获取安装命令

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

# 方式1：直接导入GPU版本
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU

# 读取图像
rgb_image = cv2.imread('./images/face.png')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 创建GPU版本的多重分形谱分析对象
MFS = CFA2DMFSGPU(
    gray_image,
    q_list=np.linspace(-5, 5, 26),
    device='cuda'  # 或 'cpu'
)

# 计算（GPU加速）
df_mass, df_fit, df_spec = MFS.get_mfs()

# 可视化
MFS.plot(df_mass, df_fit, df_spec)
```

### GPU加速的分形维度

```python
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU
from FreeAeonFractal.FAImage import CFAImage

# 图像预处理
gray_image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)
bin_image, _ = CFAImage.otsu_binarize(gray_image)

# GPU加速计算
fd_bc_gpu = CFAImageDimensionGPU(
    bin_image,
    device='cuda'
).get_bc_fd()

fd_dbc_gpu = CFAImageDimensionGPU(
    gray_image,
    device='cuda'
).get_dbc_fd()

print(f"BC (GPU): {fd_bc_gpu['fd']:.4f}")
print(f"DBC (GPU): {fd_dbc_gpu['fd']:.4f}")
```

### GPU加速的空隙度分析

```python
from FreeAeonFractal.FAImageLacunarityGPU import CFAImageLacunarityGPU

# 创建GPU版本的空隙度分析对象
lacunarity_gpu = CFAImageLacunarityGPU(
    gray_image,
    max_scales=256,
    device='cuda'
)

# 计算（GPU加速）
lac_result = lacunarity_gpu.get_lacunarity()
fit_result = lacunarity_gpu.fit_lacunarity(lac_result)

# 可视化
lacunarity_gpu.plot(lac_result, fit_result)
```

### 使用别名简化代码

```python
# 在代码开头使用别名，后续代码无需修改
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension

# 后续代码与CPU版本完全相同
MFS = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

## GPU版本特有参数

### 1. CFA2DMFSGPU

额外参数：
- `device` (str): 设备选择
  - `'cuda'`: 使用GPU（默认）
  - `'cpu'`: 使用CPU
  - `'cuda:0'`, `'cuda:1'`: 指定GPU编号
- `dtype` (torch.dtype): 数据类型
  - `torch.float64`: 双精度（默认，更准确）
  - `torch.float32`: 单精度（更快，显存更小）

**示例**：
```python
MFS = CFA2DMFSGPU(
    image,
    device='cuda:0',           # 使用第一块GPU
    dtype=torch.float32        # 使用单精度加速
)
```

### 2. CFAImageDimensionGPU

额外参数：
- `device` (str): 同上

**示例**：
```python
fd_calc = CFAImageDimensionGPU(
    image,
    device='cuda'
)
```

### 3. CFAImageLacunarityGPU

额外参数：
- `device` (str): 同上

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

**更大图像 (2048x2048)**：

| 任务 | CPU时间 | GPU时间 | 加速比 |
|------|---------|---------|--------|
| 2D多重分形谱 | 180s | 10s | 18.0x |
| 分形维度 DBC | 95s | 6s | 15.8x |

## 优化建议

### 1. 选择合适的数据类型

```python
# 对于一般分析，float32足够且更快
MFS = CFA2DMFSGPU(image, dtype=torch.float32)

# 对于高精度要求，使用float64
MFS = CFA2DMFSGPU(image, dtype=torch.float64)
```

### 2. 批量处理多个图像

```python
import glob

image_files = glob.glob('./images/*.png')

for img_file in image_files:
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # 使用GPU加速
    mfs = CFA2DMFSGPU(image, device='cuda')
    df_mass, df_fit, df_spec = mfs.get_mfs()

    # 保存结果
    df_spec.to_csv(f'{img_file}_mfs.csv')
```

### 3. 显存管理

```python
import torch

# 清理GPU缓存
torch.cuda.empty_cache()

# 检查显存使用
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 4. 自动设备选择

```python
# 自动选择可用设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

MFS = CFA2DMFSGPU(image, device=device)
```

## CPU vs GPU 使用场景

### 使用CPU版本：
- 小图像 (<512x512)
- 快速测试和原型开发
- 没有GPU的环境
- q值数量少 (<20)

### 使用GPU版本：
- 大图像 (≥1024x1024)
- 批量处理多个图像
- q值数量多 (≥30)
- 多尺度分析 (max_scales>80)
- 实时分析需求

## 常见问题

### Q: GPU版本报错 "CUDA out of memory"？
A:
1. 减小图像尺寸
2. 使用 `dtype=torch.float32`
3. 减少 `max_scales` 参数
4. 清理GPU缓存：`torch.cuda.empty_cache()`

### Q: GPU版本比CPU还慢？
A: 小图像或少量计算时，GPU初始化开销可能超过计算时间。建议图像 ≥512x512 时使用GPU。

### Q: 如何在多GPU环境选择GPU？
A: 使用 `device='cuda:0'` 或 `device='cuda:1'` 指定GPU编号。

### Q: GPU版本结果与CPU版本不同？
A: 微小差异是正常的（浮点运算顺序不同）。使用 `dtype=torch.float64` 可提高一致性。

### Q: 没有NVIDIA GPU怎么办？
A:
1. 使用CPU版本
2. 设置 `device='cpu'`（GPU版本会回退到CPU）
3. 考虑使用云GPU服务（Google Colab, AWS等）

## 实际应用案例

### 高分辨率医学图像分析

```python
# 读取高分辨率医学图像
medical_image = cv2.imread('high_res_tissue.png', cv2.IMREAD_GRAYSCALE)
print(f"Image size: {medical_image.shape}")  # 例如 (2048, 2048)

# 使用GPU加速
mfs = CFA2DMFSGPU(
    medical_image,
    q_list=np.linspace(-5, 5, 51),
    device='cuda',
    dtype=torch.float32
)

import time
start = time.time()
df_mass, df_fit, df_spec = mfs.get_mfs(max_scales=100)
elapsed = time.time() - start

print(f"GPU计算时间: {elapsed:.2f}秒")
```

### 批量材料表面分析

```python
import glob
import pandas as pd

results = []
image_files = glob.glob('./surfaces/*.png')

for img_file in image_files:
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # GPU加速
    fd_calc = CFAImageDimensionGPU(image, device='cuda')
    fd_bc = fd_calc.get_bc_fd()
    fd_dbc = fd_calc.get_dbc_fd()

    results.append({
        'file': img_file,
        'BC': fd_bc['fd'],
        'DBC': fd_dbc['fd']
    })

df_results = pd.DataFrame(results)
df_results.to_csv('batch_analysis_results.csv')
```

## 性能监控

```python
import time
import torch

def benchmark_gpu_vs_cpu(image, q_list):
    """对比GPU和CPU性能"""

    # CPU版本
    from FreeAeonFractal.FA2DMFS import CFA2DMFS
    start = time.time()
    mfs_cpu = CFA2DMFS(image, q_list=q_list, with_progress=False)
    df_mass, df_fit, df_spec = mfs_cpu.get_mfs()
    cpu_time = time.time() - start

    # GPU版本
    torch.cuda.synchronize()  # 确保之前的操作完成
    start = time.time()
    mfs_gpu = CFA2DMFSGPU(image, q_list=q_list, with_progress=False, device='cuda')
    df_mass, df_fit, df_spec = mfs_gpu.get_mfs()
    torch.cuda.synchronize()  # 等待GPU计算完成
    gpu_time = time.time() - start

    print(f"CPU时间: {cpu_time:.2f}秒")
    print(f"GPU时间: {gpu_time:.2f}秒")
    print(f"加速比: {cpu_time/gpu_time:.2f}x")

    return cpu_time, gpu_time

# 测试
image = cv2.imread('large_image.png', cv2.IMREAD_GRAYSCALE)
q_list = np.linspace(-5, 5, 51)
benchmark_gpu_vs_cpu(image, q_list)
```

## 重要提示

1. **首次运行**：
   - GPU版本首次运行可能较慢（CUDA初始化）
   - 后续运行会显著加快

2. **显存管理**：
   - 大图像可能占用大量显存
   - 及时清理：`torch.cuda.empty_cache()`
   - 监控显存：`torch.cuda.memory_allocated()`

3. **数据类型**：
   - float32：更快，显存更小，精度略低
   - float64：更准确，显存更大，速度略慢
   - 一般分析推荐float32

4. **设备选择**：
   - 自动选择：`device=None`（有GPU用GPU，否则用CPU）
   - 手动指定：`device='cuda'` 或 `device='cpu'`
   - 多GPU：`device='cuda:0'` 指定GPU编号

5. **API兼容性**：
   - GPU版本API与CPU版本完全兼容
   - 可以无缝切换，只需修改导入语句
   - 返回结果格式相同

## 迁移指南

### 从CPU迁移到GPU

只需修改导入语句：

**CPU版本**：
```python
from FreeAeonFractal.FA2DMFS import CFA2DMFS
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FAImageLacunarity import CFAImageLacunarity

# ... 其余代码不变
```

**GPU版本**：
```python
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension
from FreeAeonFractal.FAImageLacunarityGPU import CFAImageLacunarityGPU as CFAImageLacunarity

# ... 其余代码完全相同！
```

### 条件导入（智能切换）

```python
import torch

# 根据GPU可用性自动选择
if torch.cuda.is_available():
    from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
    print("Using GPU acceleration")
else:
    from FreeAeonFractal.FA2DMFS import CFA2DMFS
    print("Using CPU")

# 后续代码无需修改
MFS = CFA2DMFS(image, q_list=np.linspace(-5, 5, 26))
df_mass, df_fit, df_spec = MFS.get_mfs()
```

## GPU版本类说明

### 1. CFA2DMFSGPU

**继承自**: CFA2DMFS的GPU实现

**新增参数**：
- `device` (str): 'cuda', 'cpu', 'cuda:0'等
- `dtype` (torch.dtype): torch.float32 或 torch.float64

**核心优化**：
- GPU上计算盒子质量μ
- GPU上进行logsumexp计算
- CPU上进行回归和样条拟合

### 2. CFAImageDimensionGPU

**继承自**: CFAImageDimension的GPU实现

**新增参数**：
- `device` (str): 设备选择

**核心优化**：
- GPU上进行图像分块
- GPU上计算盒子统计
- 使用PyTorch的高效操作

### 3. CFAImageLacunarityGPU

**继承自**: CFAImageLacunarity的GPU实现

**新增参数**：
- `device` (str): 设备选择

**核心优化**：
- GPU上计算滑动窗口
- GPU上进行质量统计
- 高效的并行计算

## 性能优化技巧

### 1. 批处理

```python
# 使用GPU批量处理多个图像
def batch_analyze_gpu(image_list, q_list):
    results = []

    for i, image in enumerate(image_list):
        mfs = CFA2DMFSGPU(
            image,
            q_list=q_list,
            device='cuda',
            with_progress=(i == 0)  # 只显示第一个的进度
        )
        _, df_fit, _ = mfs.get_mfs()
        results.append(df_fit)

        # 定期清理显存
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return results
```

### 2. 混合精度

```python
# 使用半精度进行初步分析
mfs_fp16 = CFA2DMFSGPU(image, dtype=torch.float16, device='cuda')
# 注意：某些操作可能不支持float16

# 对重要样本使用双精度
mfs_fp64 = CFA2DMFSGPU(important_image, dtype=torch.float64, device='cuda')
```

### 3. 显存优化

```python
# 对于显存受限的情况
def analyze_with_limited_memory(image):
    # 使用更小的参数
    mfs = CFA2DMFSGPU(
        image,
        q_list=np.linspace(-5, 5, 21),  # 减少q值
        device='cuda',
        dtype=torch.float32  # 使用单精度
    )

    # 减少尺度数量
    df_mass, df_fit, df_spec = mfs.get_mfs(max_scales=50)

    return df_mass, df_fit, df_spec
```

## 故障排除

### 问题1: CUDA初始化失败
```python
# 检查CUDA是否可用
import torch
if not torch.cuda.is_available():
    print("CUDA not available. Install PyTorch with CUDA support.")
    # 回退到CPU版本
```

### 问题2: 显存不足
```python
# 解决方案：
# 1. 减小图像尺寸
image_small = cv2.resize(image, (512, 512))

# 2. 使用float32
mfs = CFA2DMFSGPU(image, dtype=torch.float32)

# 3. 减少参数
df_mass, df_fit, df_spec = mfs.get_mfs(max_scales=50)

# 4. 清理显存
torch.cuda.empty_cache()
```

### 问题3: 版本兼容性
```python
# 检查PyTorch版本
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# 推荐: PyTorch >= 1.9.0
```

## 最佳实践

1. **开发阶段**：使用CPU版本快速测试
2. **生产环境**：使用GPU版本处理大批量数据
3. **参数调优**：在CPU上测试参数，在GPU上运行
4. **显存监控**：定期检查显存使用情况
5. **错误处理**：捕获CUDA异常，优雅降级到CPU

```python
def safe_gpu_analysis(image, **kwargs):
    """带错误处理的GPU分析"""
    try:
        if torch.cuda.is_available():
            mfs = CFA2DMFSGPU(image, device='cuda', **kwargs)
            return mfs.get_mfs()
        else:
            raise RuntimeError("GPU not available")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"GPU failed: {e}, falling back to CPU")
        from FreeAeonFractal.FA2DMFS import CFA2DMFS
        mfs = CFA2DMFS(image, **kwargs)
        return mfs.get_mfs()
```

## 参考资源

- **PyTorch官网**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **GPU性能优化**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

## 总结

GPU加速版本能够显著提升大图像和多尺度分析的计算速度，特别适合：
- 高分辨率图像处理
- 批量数据分析
- 实时计算需求

通过简单的导入切换，即可享受GPU带来的性能提升！
