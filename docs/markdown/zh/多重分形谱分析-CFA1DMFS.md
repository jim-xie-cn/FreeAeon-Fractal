# CFA1DMFS - 一维时间序列多重分形谱分析

## 类概述

`CFA1DMFS` 类用于计算一维时间序列的多重分形谱(Multifractal Spectrum),包括广义Hurst指数h(q)、质量指数τ(q)、奇异性强度α(q)、多重分形谱f(α)以及广义维数D(q)。

## 应用场景

- **金融时间序列分析**:股票价格波动分析、市场风险评估
- **生理信号处理**:心电图(ECG)、脑电图(EEG)分析
- **地震数据分析**:地震波信号的复杂度分析
- **气象数据分析**:温度、降雨等时间序列的多尺度特征提取
- **网络流量分析**:互联网流量的多重分形特性研究

## 类初始化

### 构造函数

```python
CFA1DMFS(data, q_list=np.linspace(-5, 5, 51), with_progress=True)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | numpy.ndarray | 必需 | 输入的一维时间序列数据 |
| `q_list` | numpy.ndarray | `np.linspace(-5, 5, 51)` | q值范围,用于计算不同阶的分形指标 |
| `with_progress` | bool | `True` | 是否显示进度条 |

### 初始化示例

```python
import numpy as np
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# 生成随机游走时间序列
x = np.cumsum(np.random.randn(5000))

# 创建多重分形谱分析对象
mfs = CFA1DMFS(x, q_list=np.linspace(-5, 5, 21), with_progress=True)
```

## 主要方法

### 1. get_mfs() - 计算多重分形谱

计算完整的多重分形谱,包括h(q)、τ(q)、α(q)、f(α)和D(q)。

#### 方法签名

```python
get_mfs(lag_list=None, order=2)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lag_list` | numpy.ndarray | `None` | 自定义尺度窗口列表。如果为None,将自动生成推荐尺度 |
| `order` | int | `2` | DFA(去趋势波动分析)的多项式拟合阶数 |

#### 返回值

返回一个pandas DataFrame,包含以下列:

- `q`: q值
- `h(q)`: 广义Hurst指数
- `t(q)`: 质量指数 τ(q)
- `a(q)`: 奇异性强度 α(q)
- `f(a)`: 多重分形谱 f(α)
- `d(q)`: 广义维数 D(q)

#### 使用示例

```python
# 计算多重分形谱
df_mfs = mfs.get_mfs()

# 查看结果
print(df_mfs.head())
print(f"最大奇异性强度: {df_mfs['a(q)'].max():.4f}")
print(f"最小奇异性强度: {df_mfs['a(q)'].min():.4f}")
print(f"谱宽度: {df_mfs['a(q)'].max() - df_mfs['a(q)'].min():.4f}")
```

### 2. plot() - 可视化多重分形谱

绘制多重分形谱的六个关键图表。

#### 方法签名

```python
plot(df_mfs)
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `df_mfs` | pandas.DataFrame | 由`get_mfs()`返回的数据框 |

#### 绘制的图表

1. **H(q) vs q**: 广义Hurst指数随q的变化
2. **τ(q) vs q**: 质量指数随q的变化
3. **D(q) vs q**: 广义维数随q的变化
4. **α(q) vs q**: 奇异性强度随q的变化
5. **f(α) vs α**: 多重分形谱曲线(最重要的图表)
6. **Overview vs q**: 归一化的综合对比图

#### 使用示例

```python
# 可视化结果
mfs.plot(df_mfs)
```

## 完整使用示例

### 示例1: 基本使用

```python
import numpy as np
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# 生成布朗运动时间序列
np.random.seed(42)
x = np.cumsum(np.random.randn(5000))

# 创建分析对象
mfs = CFA1DMFS(x, q_list=np.linspace(-5, 5, 21))

# 计算多重分形谱
df_mfs = mfs.get_mfs()

# 输出关键指标
print("广义Hurst指数 H(2):", df_mfs[df_mfs['q']==2]['h(q)'].values[0])
print("信息维数 D(1):", df_mfs[df_mfs['q']==1]['d(q)'].values[0])
print("关联维数 D(2):", df_mfs[df_mfs['q']==2]['d(q)'].values[0])

# 可视化
mfs.plot(df_mfs)
```

### 示例2: 自定义尺度

```python
# 自定义尺度窗口
custom_lags = np.array([16, 32, 64, 128, 256, 512, 1024])

# 计算多重分形谱
df_mfs = mfs.get_mfs(lag_list=custom_lags, order=3)

# 可视化
mfs.plot(df_mfs)
```

### 示例3: 金融数据分析

```python
import pandas as pd
import numpy as np
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# 读取股票价格数据
prices = pd.read_csv('stock_prices.csv')['Close'].values

# 计算对数收益率
returns = np.diff(np.log(prices))

# 分析收益率的多重分形特性
mfs = CFA1DMFS(returns, q_list=np.linspace(-10, 10, 41))
df_mfs = mfs.get_mfs()

# 判断是否具有多重分形特性
h_min = df_mfs['h(q)'].min()
h_max = df_mfs['h(q)'].max()
is_multifractal = (h_max - h_min) > 0.1

print(f"Hurst指数范围: [{h_min:.4f}, {h_max:.4f}]")
print(f"是否具有多重分形特性: {is_multifractal}")

mfs.plot(df_mfs)
```

## 辅助函数

### recommended_lag() - 推荐尺度窗口

为中短序列生成密集的尺度窗口集,用于高精度多重分形分析。

#### 函数签名

```python
recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `x_len` | int | 必需 | 输入序列的长度 |
| `order` | int | `2` | DFA的多项式拟合阶数 |
| `num_scales` | int | `40` | 要生成的尺度数量(推荐30-40) |
| `s_min` | int | `None` | 最小尺度(默认: max(16, order+4)) |
| `s_max_ratio` | float | `0.25` | 最大尺度占序列长度的比例 |

#### 返回值

返回numpy数组,包含推荐的尺度窗口整数数组。

#### 使用示例

```python
from FreeAeonFractal.FA1DMFS import recommended_lag

# 为5000点序列生成推荐尺度
lags = recommended_lag(5000, order=2, num_scales=40)
print("推荐尺度:", lags)

# 使用推荐尺度进行分析
df_mfs = mfs.get_mfs(lag_list=lags)
```

## 理论背景

### 多重分形谱的物理意义

1. **广义Hurst指数 h(q)**
   - h(2) 即经典的Hurst指数
   - 反映时间序列的长程相关性
   - h > 0.5: 持续性(persistence)
   - h < 0.5: 反持续性(anti-persistence)
   - h = 0.5: 随机游走

2. **质量指数 τ(q)**
   - τ(q) = q·h(q) - 1
   - 非线性的τ(q)表明存在多重分形特性
   - 线性的τ(q)表明单分形特性

3. **奇异性强度 α(q)**
   - α = dτ/dq
   - 描述不同尺度下的局部Hölder指数
   - α的范围反映了信号的复杂度

4. **多重分形谱 f(α)**
   - f(α) = q·α - τ(q)
   - 描述具有相同奇异性强度α的子集的分形维数
   - 谱宽度 Δα = α_max - α_min 反映多重分形程度

5. **广义维数 D(q)**
   - D(q) = τ(q)/(q-1) for q≠1
   - D(0): 容量维数(Capacity dimension)
   - D(1): 信息维数(Information dimension)
   - D(2): 关联维数(Correlation dimension)

### 计算流程

1. 使用MF-DFA(多重分形去趋势波动分析)方法
2. 对每个尺度s和每个q值计算波动函数F_q(s)
3. 通过对数坐标线性拟合得到h(q)
4. 计算τ(q) = q·h(q) - 1
5. 通过数值微分计算α(q) = dτ/dq
6. 计算f(α) = q·α - τ(q)
7. 计算D(q) = τ(q)/(q-1)

## 注意事项

1. **数据长度**: 建议时间序列长度≥1000点,更长的序列能获得更可靠的结果
2. **q值范围**: 通常q∈[-10,10]已足够,过大的|q|可能导致数值不稳定
3. **尺度选择**: 默认尺度范围为[16, 0.25×序列长度],可根据需要调整
4. **多项式阶数**: order=1适用于趋势明显的数据,order=2适用于大多数情况
5. **D(1)近似**: 在q=1处,D(1)通过相邻点平均值近似计算

## 性能优化建议

1. 对于短序列(<1000点),减少num_scales可加快计算
2. 关闭进度条(`with_progress=False`)可略微提升批量处理速度
3. 对于多个序列的批量分析,考虑使用并行处理

## 相关文献

1. Kantelhardt, J. W., et al. (2002). "Multifractal detrended fluctuation analysis of nonstationary time series." Physica A, 316(1-4), 87-114.
2. Feder, J. (1988). "Fractals." Plenum Press, New York.

---

**文件位置**: `/Users/jim_xie/Desktop/FreeAeon-Fractal/FreeAeonFractal/FA1DMFS.py`
**类名**: `CFA1DMFS`
**功能类别**: 多重分形谱分析
