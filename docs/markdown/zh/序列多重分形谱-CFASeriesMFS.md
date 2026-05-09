# 序列多重分形谱分析 - CFASeriesMFS

## 应用场景

`CFASeriesMFS` 类使用MFDFA（多重分形去趋势波动分析）方法计算1D时间序列的多重分形谱。主要应用场景包括：

- **金融分析**：检测股票价格和收益率的多重分形结构
- **生理信号**：分析心跳间隔、脑电图等生物信号
- **气候数据**：研究温度和降水的长程相关性
- **地球物理序列**：地震、地震波和潮汐分析
- **网络流量**：网络流量模式分析

## 使用示例

### 基础用法

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

# 生成随机游走序列（示例数据）
x = np.cumsum(np.random.randn(5000))

# 创建MFS分析对象
q_list = np.linspace(-5, 5, 21)
mfs = CFASeriesMFS(x, q_list=q_list)

# 计算多重分形谱
df_mfs = mfs.get_mfs()

# 查看结果
print(df_mfs.head(10))

# 可视化
mfs.plot(df_mfs)
```

### 自定义尺度窗口

```python
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS, recommended_lag

x = np.cumsum(np.random.randn(10000))

# 使用推荐尺度
lag = recommended_lag(len(x), order=2, num_scales=40)
mfs = CFASeriesMFS(x)
df_mfs = mfs.get_mfs(lag_list=lag, order=2)
print(df_mfs)
```

### 高阶DFA

```python
# 使用3阶多项式去趋势（去除三次趋势）
df_mfs = mfs.get_mfs(order=3)
```

## 类说明

### CFASeriesMFS

**描述**：使用MFDFA（多重分形去趋势波动分析）方法分析1D时间序列的多重分形谱。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | 必填 | 输入时间序列（1D数组） |
| `q_list` | array-like | linspace(-5, 5, 51) | q矩值 |
| `with_progress` | bool | True | 是否显示进度条 |

#### 主要方法

##### 1. get_mfs(lag_list=None, order=2)

**描述**：使用MFDFA计算多重分形谱。

**参数**：
- `lag_list` (array-like 或 None)：尺度窗口（分段长度）。若为None，使用 `recommended_lag(len(data))`
- `order` (int)：DFA去趋势多项式阶数。默认2（去除线性趋势）
  - `order=1`：去除均值
  - `order=2`：去除线性趋势（默认）
  - `order=3`：去除二次趋势

**返回值** (DataFrame)：

| 列名 | 描述 |
|------|------|
| `q` | 矩阶数 |
| `h(q)` | 广义Hurst指数 |
| `t(q)` | 质量指数 τ(q) = q·h(q) − 1 |
| `a(q)` | 奇异性强度 α(q) = dτ/dq |
| `f(a)` | 多重分形谱 f(α) = q·α − τ(q) |
| `d(q)` | 广义维度 D(q) = τ(q)/(q−1) |

##### 2. plot(df_mfs)

**描述**：以2×3子图网格可视化多重分形谱：
1. H(q) vs q — 广义Hurst指数
2. τ(q) vs q — 质量指数
3. D(q) vs q — 广义维度
4. α(q) vs q — 奇异性强度
5. f(α) vs α — 多重分形谱（经典MFS图）
6. 概览：归一化的t(q), d(q), a(q), f(a) vs q

### 工具函数：recommended_lag

```python
from FreeAeonFractal.FASeriesMFS import recommended_lag

lag = recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25)
```

**描述**：生成适用于中短序列的几何间距尺度窗口集合。

**参数**：
- `x_len`：序列长度
- `order`：DFA多项式阶数（影响最小尺度）
- `num_scales`：生成的尺度点数（推荐30-40）
- `s_min`：最小尺度；默认 `max(16, order + 4)`
- `s_max_ratio`：最大尺度占序列长度的比例（默认0.25）

**返回值**：整数尺度窗口的 `numpy.ndarray`

## 理论背景

### MFDFA方法

MFDFA（Kantelhardt et al. 2002）将标准DFA扩展到检测非平稳时间序列中的多重分形特性。

#### 计算步骤

1. **累积曲线**：Y(i) = Σₜ x(t) − mean(x)
2. **分段**：划分为大小为s的不重叠窗口
3. **去趋势**：在每个窗口内拟合 `order` 阶多项式；计算方差 F²(s, v)
4. **波动函数**：对每个q，Fq(s) = (1/N_s Σᵥ [F²]^(q/2))^(1/q)
5. **标度律**：Fq(s) ~ s^h(q)，斜率 = 广义Hurst指数 h(q)

#### 关键指标

- **h(q)**：广义Hurst指数；h(2) = 标准Hurst指数
- **τ(q) = q·h(q) − 1**：质量指数
- **α(q) = dτ/dq**：奇异性强度（通过 `np.gradient` 计算）
- **f(α) = q·α − τ(q)**：多重分形谱
- **D(q) = τ(q)/(q−1)**：广义维度

#### 单分形 vs 多重分形

- **单分形**：h(q) 随q不变
- **多重分形**：h(q) 随q变化；Δh = h(−5) − h(5) > 0
- 谱宽 Δα = α_max − α_min 量化多重分形强度

## 重要说明

1. **序列长度**：建议最少1000个点；低于500点结果不可靠
2. **q值选择**：推荐 `np.linspace(-5, 5, 21)`；负q对低波动区域敏感，正q对高波动区域敏感
3. **DFA阶数**：`order=2` 是标准选择；高阶需要更多数据
4. **结果解读**：Δh > 0.1 表示存在多重分形特性；D(q) 随q单调递减表示多重分形信号
5. **依赖**：需要 `MFDFA` 包：`pip install MFDFA`

## 参考文献

- Kantelhardt, J. W., et al. (2002). *Physica A*.
- Peng, C.-K., et al. (1994). *Physical Review E*.
- Ihlen, E. A. F. (2012). *Frontiers in Physiology*.
