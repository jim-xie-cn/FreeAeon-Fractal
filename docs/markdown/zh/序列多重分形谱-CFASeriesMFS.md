# 序列多重分形谱分析 - CFASeriesMFS

## 应用场景

`CFASeriesMFS` 类用于计算1D时间序列的多重分形谱，是分析序列复杂度和长程相关性的重要工具。主要应用场景包括：

- **金融时间序列**：分析股价和收益率的多重分形特性
- **生理信号分析**：心电图和脑电图的复杂度分析
- **气候数据**：温度和降水序列的多尺度特征
- **网络流量**：互联网流量的自相似性分析
- **地震数据**：地震波形的多重分形特征

## 使用示例

### 基本用法

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

# 生成随机游走序列
x = np.cumsum(np.random.randn(5000))

# 创建多重分形谱分析对象
q_list = np.linspace(-5, 5, 21)
mfs = CFASeriesMFS(x, q_list=q_list)

# 计算多重分形谱
df_mfs = mfs.get_mfs()

# 查看结果
print(df_mfs)

# 可视化结果
mfs.plot(df_mfs)
```

### 自定义尺度参数

```python
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS, recommended_lag

# 使用推荐尺度函数
lag_list = recommended_lag(len(x), order=2, num_scales=40)

# 计算多重分形谱
df_mfs = mfs.get_mfs(lag_list=lag_list, order=2)
```

### 安装

```bash
pip install FreeAeon-Fractal
```

## 类说明

### CFASeriesMFS

**描述**：基于MFDFA（多重分形去趋势波动分析）方法计算1D时间序列多重分形谱的类。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | numpy.ndarray | 必填 | 输入时间序列（1D数组） |
| `q_list` | array-like | linspace(-5,5,51) | q值列表 |
| `with_progress` | bool | True | 是否显示进度条 |

#### 模块级辅助函数

##### recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25)

**描述**：为MFDFA生成推荐的尺度（lag）列表。

**参数**：
- `x_len`（int）：输入序列长度
- `order`（int）：DFA多项式阶数
- `num_scales`（int）：尺度点数量
- `s_min`（int或None）：最小尺度（None时自动设置）
- `s_max_ratio`（float）：最大尺度占序列长度的比例（默认0.25）

**返回值**：整数lag值的numpy数组

#### 主要方法

##### 1. get_mfs(lag_list=None, order=2)

**描述**：计算多重分形谱，包括广义Hurst指数、质量指数、奇异性强度和多重分形谱。

**参数**：
- `lag_list`（array-like）：自定义尺度窗口列表。若为None，使用 `recommended_lag` 自动生成推荐尺度
- `order`（int）：DFA多项式拟合阶数，默认为2

**返回值**（pandas.DataFrame）：

包含以下列的DataFrame：

| 列名 | 说明 |
|------|------|
| `q` | q值 |
| `h(q)` | 广义Hurst指数 |
| `t(q)` | τ(q)质量指数 |
| `a(q)` | α(q)奇异性强度 |
| `f(a)` | f(α)多重分形谱 |
| `d(q)` | D(q)广义维度 |

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
6. **概览**：所有指标的归一化比较

**参数**：
- `df_mfs`（DataFrame）：`get_mfs()` 返回的结果

## 理论背景

### MFDFA方法

多重分形去趋势波动分析（MFDFA）是DFA方法的推广，步骤如下：

#### 1. 累积偏差
```
Y(i) = Σ[x(k) - x̄]  (i = 1, 2, ..., N)
```

#### 2. 分段去趋势
将Y(i)分成大小为s的非重叠窗口，在每个窗口内拟合多项式并计算方差。

#### 3. 波动函数
```
F²(s,v) = (1/s) Σ{Y[(v-1)s+i] - yᵥ(i)}²
```

#### 4. q阶波动函数
```
Fq(s) = {(1/Ns) Σ[F²(s,v)]^(q/2)}^(1/q)
```

#### 5. 标度律
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

## 注意事项

1. **序列长度**：
   - 建议最小长度 ≥ 1000
   - 序列越长，结果越稳定
   - 短序列应减少尺度数量

2. **q值选择**：
   - 负q：对小波动敏感
   - 正q：对大波动敏感
   - 建议范围：-5 到 5
   - 建议点数：20-50

3. **尺度参数**：
   - `order=1`：适合无趋势序列
   - `order=2`：适合线性趋势（推荐）
   - `order=3`：适合二次趋势
   - 使用 `recommended_lag()` 获取最优尺度

4. **结果解释**：
   - h(q)单调递减：多重分形
   - h(q)恒定：单分形
   - Δh > 0.1：显著多重分形性
   - f(α)越宽：异质性越强

5. **性能考虑**：
   - 长序列计算较慢
   - 减少q值数量可提速
   - 使用 `recommended_lag` 函数优化尺度选择

## 参考文献

- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A*.
- Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*.
