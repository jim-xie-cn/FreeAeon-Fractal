# Series Multifractal Spectrum Analysis - CFASeriesMFS

## Application Scenarios

The `CFASeriesMFS` class is used to calculate the multifractal spectrum of 1D time series, serving as an important tool for analyzing sequence complexity and long-range correlations. Main application scenarios include:

- **Financial Time Series**: Analyze multifractal properties of stock prices and returns
- **Physiological Signal Analysis**: Complexity analysis of ECG and EEG
- **Climate Data**: Multi-scale features of temperature and precipitation sequences
- **Network Traffic**: Self-similarity analysis of internet traffic
- **Seismic Data**: Multifractal features of seismic waveforms

## Usage Examples

### Basic Usage

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

# Generate random walk sequence
x = np.cumsum(np.random.randn(5000))

# Create multifractal spectrum analysis object
q_list = np.linspace(-5, 5, 21)
mfs = CFASeriesMFS(x, q_list=q_list)

# Calculate multifractal spectrum
df_mfs = mfs.get_mfs()

# View results
print(df_mfs)

# Visualize results
mfs.plot(df_mfs)
```

### Custom Scale Parameters

```python
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS, recommended_lag

# Use recommended scale function
lag_list = recommended_lag(len(x), order=2, num_scales=40)

# Calculate multifractal spectrum
df_mfs = mfs.get_mfs(lag_list=lag_list, order=2)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFASeriesMFS

**Description**: Class for calculating 1D time series multifractal spectrum based on MFDFA (Multifractal Detrended Fluctuation Analysis) method.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | numpy.ndarray | Required | Input time series (1D array) |
| `q_list` | array-like | linspace(-5,5,51) | q value list |
| `with_progress` | bool | True | Whether to show progress bar |

#### Module-level Helper

##### recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25)

**Description**: Generate a recommended scale (lag) list for MFDFA.

**Parameters**:
- `x_len` (int): Length of the input series
- `order` (int): DFA polynomial order
- `num_scales` (int): Number of scale points
- `s_min` (int or None): Minimum scale (auto if None)
- `s_max_ratio` (float): Maximum scale as fraction of series length (default 0.25)

**Return Value**: numpy array of integer lag values

#### Main Methods

##### 1. get_mfs(lag_list=None, order=2)

**Description**: Calculate multifractal spectrum, including generalized Hurst exponent, mass exponent, singularity strength, and multifractal spectrum.

**Parameters**:
- `lag_list` (array-like): Custom scale window list. If None, automatically generates recommended scales using `recommended_lag`
- `order` (int): DFA polynomial fitting order, default is 2

**Return Value** (pandas.DataFrame):

DataFrame containing the following columns:

| Column | Description |
|--------|-------------|
| `q` | q value |
| `h(q)` | Generalized Hurst exponent |
| `t(q)` | τ(q) mass exponent |
| `a(q)` | α(q) singularity strength |
| `f(a)` | f(α) multifractal spectrum |
| `d(q)` | D(q) generalized dimension |

**Example**:
```python
df_mfs = mfs.get_mfs()
print(df_mfs.head())
#    q      h(q)     t(q)     a(q)     f(a)     d(q)
# 0 -5.0   0.8234  -5.9170   0.9523   0.7445   1.9835
# 1 -4.0   0.7891  -4.1564   0.9012   0.7484   1.7854
# ...
```

##### 2. plot(df_mfs)

**Description**: Visualize multifractal spectrum analysis results with 6 subplots:

1. **H(q) vs q**: Generalized Hurst exponent
2. **τ(q) vs q**: Mass exponent
3. **D(q) vs q**: Generalized dimension
4. **α(q) vs q**: Singularity strength
5. **f(α) vs α**: Multifractal spectrum
6. **Overview**: Normalized comparison of all indices

**Parameters**:
- `df_mfs` (DataFrame): Results returned by `get_mfs()`

## Theoretical Background

### MFDFA Method

Multifractal Detrended Fluctuation Analysis (MFDFA) is a generalization of the DFA method with the following steps:

#### 1. Cumulative Deviation
```
Y(i) = Σ[x(k) - x̄]  (i = 1, 2, ..., N)
```

#### 2. Segment Detrending
Divide Y(i) into non-overlapping windows of size s, fit polynomials in each window, and calculate variance.

#### 3. Fluctuation Function
```
F²(s,v) = (1/s) Σ{Y[(v-1)s+i] - yᵥ(i)}²
```

#### 4. q-order Fluctuation Function
```
Fq(s) = {(1/Ns) Σ[F²(s,v)]^(q/2)}^(1/q)
```

#### 5. Scaling Law
```
Fq(s) ~ s^h(q)
```
Where h(q) is the generalized Hurst exponent.

### Multifractal Parameters

#### 1. Mass Exponent τ(q)
```
τ(q) = q·h(q) - 1
```

#### 2. Singularity Strength α(q)
```
α(q) = dτ(q)/dq = h(q) + q·dh(q)/dq
```

#### 3. Multifractal Spectrum f(α)
```
f(α) = q·α(q) - τ(q)
```

#### 4. Generalized Dimension D(q)
```
D(q) = τ(q) / (q - 1), q ≠ 1
```

## Important Notes

1. **Sequence Length**:
   - Minimum length recommended ≥ 1000
   - Longer sequences produce more stable results
   - For short sequences, reduce number of scales

2. **q Value Selection**:
   - Negative q: Sensitive to small fluctuations
   - Positive q: Sensitive to large fluctuations
   - Recommended range: -5 to 5
   - Recommended points: 20-50

3. **Scale Parameters**:
   - `order=1`: Suitable for trendless sequences
   - `order=2`: Suitable for linear trends (recommended)
   - `order=3`: Suitable for quadratic trends
   - Use `recommended_lag()` for optimal scale selection

4. **Result Interpretation**:
   - h(q) monotonically decreasing: Multifractal
   - h(q) constant: Monofractal
   - Δh > 0.1: Significant multifractality
   - Wider f(α): Stronger heterogeneity

5. **Performance Considerations**:
   - Long sequences compute slowly
   - Reduce number of q values to speed up
   - Use `recommended_lag` function for optimized scale selection

## References

- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A*.
- Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*.
