# 序列多重分形谱分析 - CFA1DMFS

## 应用场景

`CFA1DMFS` 类用于计算1D时间序列的多重分形谱，是分析序列复杂度和长程相关性的重要工具。主要应用场景包括：

- **金融时间序列**：分析股票价格、收益率的多重分形特性
- **生理信号分析**：心电图、脑电图的复杂度分析
- **气候数据**：温度、降水等气象序列的多尺度特征
- **网络流量**：互联网流量的自相似性分析
- **地震数据**：地震波形的多重分形特征

## 调用示例

### 基础用法

```python
import numpy as np
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# 生成随机游走序列
x = np.cumsum(np.random.randn(5000))

# 创建多重分形谱分析对象
q_list = np.linspace(-5, 5, 21)
mfs = CFA1DMFS(x, q_list=q_list)

# 计算多重分形谱
df_mfs = mfs.get_mfs()

# 查看结果
print(df_mfs)

# 可视化结果
mfs.plot(df_mfs)
```

### 自定义尺度参数

```python
# 自定义尺度窗口
lag_list = np.unique(np.logspace(np.log10(16), np.log10(1000), 40).astype(int))

# 计算多重分形谱
df_mfs = mfs.get_mfs(lag_list=lag_list, order=2)
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFA1DMFS

**描述**：用于计算1D时间序列多重分形谱的类，基于MFDFA（Multifractal Detrended Fluctuation Analysis）方法。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | numpy.ndarray | 必需 | 输入时间序列（1D数组） |
| `q_list` | array-like | linspace(-5,5,51) | q值列表 |
| `with_progress` | bool | True | 是否显示进度条 |

#### 主要方法

##### 1. get_mfs(lag_list=None, order=2)

**描述**：计算多重分形谱，包括广义Hurst指数、质量指数、奇异性强度和多重分形谱。

**参数**：
- `lag_list` (array-like): 自定义尺度窗口列表。如果为None，自动生成推荐尺度
- `order` (int): DFA多项式拟合阶数，默认为2

**返回值** (pandas.DataFrame):

包含以下列的DataFrame：

| 列名 | 说明 |
|------|------|
| `q` | q值 |
| `h(q)` | 广义Hurst指数 |
| `t(q)` | τ(q) 质量指数 |
| `a(q)` | α(q) 奇异性强度 |
| `f(a)` | f(α) 多重分形谱 |
| `d(q)` | D(q) 广义维度 |

**示例**：
```python
df_mfs = mfs.get_mfs()
print(df_mfs.head())
#    q      h(q)     t(q)     a(q)     f(a)     d(q)
# 0 -5.0   0.8234  -5.9170   0.9523   0.7445   1.9835
# 1 -4.0   0.7891  -4.1564   0.9012   0.7484   1.7854
# ...
```

##### 2. plot(df_mfs)

**描述**：可视化多重分形谱分析结果，包含6个子图：

1. **H(q) vs q**：广义Hurst指数
2. **τ(q) vs q**：质量指数
3. **D(q) vs q**：广义维度
4. **α(q) vs q**：奇异性强度
5. **f(α) vs α**：多重分形谱
6. **综合对比图**：归一化后的各指标对比

**参数**：
- `df_mfs` (DataFrame): 由 `get_mfs()` 返回的结果

## 理论背景

### MFDFA方法

多重分形去趋势波动分析（MFDFA）是DFA方法的推广，步骤如下：

#### 1. 累积偏离
```
Y(i) = Σ[x(k) - x̄]  (i = 1, 2, ..., N)
```

#### 2. 分段去趋势
将Y(i)分为不重叠的Ns = int(N/s)个窗口，在每个窗口内拟合多项式并计算方差。

#### 3. 波动函数
```
F²(s,v) = (1/s) Σ{Y[(v-1)s+i] - yᵥ(i)}²
```

#### 4. q阶波动函数
```
Fq(s) = {(1/Ns) Σ[F²(s,v)]^(q/2)}^(1/q)
```

#### 5. 尺度律
```
Fq(s) ~ s^h(q)
```
其中 h(q) 是广义Hurst指数。

### 多重分形参数

#### 1. 质量指数 τ(q)
```
τ(q) = q·h(q) - 1
```

#### 2. 奇异性强度 α(q)
```
α(q) = dτ(q)/dq = h(q) + q·dh(q)/dq
```

#### 3. 多重分形谱 f(α)
```
f(α) = q·α(q) - τ(q)
```

#### 4. 广义维度 D(q)
```
D(q) = τ(q) / (q - 1), q ≠ 1
```

### 参数解释

- **h(q)**：广义Hurst指数
  - h(2) = H (经典Hurst指数)
  - h(q) 单调递减表示多重分形

- **Δh = h(-5) - h(5)**：多重分形强度
  - Δh ≈ 0：单分形（均匀）
  - Δh > 0：多重分形（异质）

- **f(α)**：多重分形谱
  - 谱宽 Δα = α_max - α_min
  - 谱宽越大，多重分形性越强

## 辅助函数

### recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25)

**描述**：为中短序列生成推荐的尺度窗口集合。

**参数**：
- `x_len` (int): 序列长度
- `order` (int): DFA多项式阶数
- `num_scales` (int): 尺度数量
- `s_min` (int): 最小尺度（默认：max(16, order+4)）
- `s_max_ratio` (float): 最大尺度比例

**返回值**：推荐的尺度窗口数组

**示例**：
```python
from FreeAeonFractal.FA1DMFS import recommended_lag

lag = recommended_lag(5000, order=2, num_scales=40)
print(f"推荐尺度范围：{lag[0]} 到 {lag[-1]}")
```

## 重要提示

1. **序列长度**：
   - 最小长度建议 ≥ 1000
   - 长度越长，结果越稳定
   - 对于短序列，减少尺度数量

2. **q值选择**：
   - 负q值：对小波动敏感
   - 正q值：对大波动敏感
   - 建议范围：-5 到 5
   - 点数建议：20-50

3. **尺度参数**：
   - `order=1`：适用于无趋势序列
   - `order=2`：适用于线性趋势（推荐）
   - `order=3`：适用于二次趋势
   - 尺度数量：30-50个

4. **结果解释**：
   - h(q) 单调递减：多重分形
   - h(q) 恒定：单分形
   - Δh > 0.1：显著多重分形
   - f(α) 越宽：异质性越强

5. **性能考虑**：
   - 长序列计算较慢
   - 减少q值数量可提速
   - 使用推荐尺度函数优化

## 应用示例

### 金融时间序列分析

```python
import numpy as np
import pandas as pd
from FreeAeonFractal.FA1DMFS import CFA1DMFS

# 假设有股票收益率数据
returns = pd.read_csv('stock_returns.csv')['return'].values

# 多重分形分析
mfs = CFA1DMFS(returns, q_list=np.linspace(-5, 5, 21))
df_mfs = mfs.get_mfs()

# 检查多重分形性
delta_h = df_mfs.loc[df_mfs['q'] == -5, 'h(q)'].values[0] - \
          df_mfs.loc[df_mfs['q'] == 5, 'h(q)'].values[0]

if delta_h > 0.1:
    print(f"显著多重分形特性，Δh = {delta_h:.3f}")
else:
    print(f"单分形特性，Δh = {delta_h:.3f}")

# 可视化
mfs.plot(df_mfs)
```

### 生理信号分析

```python
# 心率变异性(HRV)数据
hrv_data = load_hrv_data()  # 假设函数

# 多重分形分析
mfs = CFA1DMFS(hrv_data)
df_mfs = mfs.get_mfs(order=2)

# 提取特征
h2 = df_mfs.loc[df_mfs['q'] == 2, 'h(q)'].values[0]  # Hurst指数
alpha_0 = df_mfs['a(q)'].max()  # 最大奇异性
width = df_mfs['a(q)'].max() - df_mfs['a(q)'].min()  # 谱宽

print(f"Hurst指数: {h2:.3f}")
print(f"谱宽: {width:.3f}")
```

## 常见问题

### Q: 序列太短怎么办？
A: 减少尺度数量（num_scales=20-30），或使用较小的 `s_max_ratio`。

### Q: 如何判断是否为多重分形？
A: 检查 h(q) 是否单调递减，或 Δh 是否 > 0.1。

### Q: order参数如何选择？
A: 对于平稳序列用1，有趋势用2，强趋势用3。一般情况推荐用2。

### Q: 计算很慢？
A: 减少q值数量，减少尺度数量，或使用较短的序列。

## 参数依赖

本类依赖 `MFDFA` 库：

```bash
pip install MFDFA
```

## 参考文献

- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. Physica A.
- Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. Physical Review E.
