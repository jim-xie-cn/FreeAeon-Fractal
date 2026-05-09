# Series Multifractal Spectrum Analysis - CFASeriesMFS

## Application Scenarios

The `CFASeriesMFS` class is used to calculate the multifractal spectrum of 1D time series using the MFDFA (Multifractal Detrended Fluctuation Analysis) method. Main application scenarios include:

- **Financial Analysis**: Detect multifractal structure in stock prices and returns
- **Physiological Signals**: Analyze heartbeat intervals, EEG, and other biosignals
- **Climate Data**: Study long-range correlations in temperature and precipitation
- **Geophysical Series**: Earthquake, seismic, and tidal analysis
- **Internet Traffic**: Network traffic pattern analysis

## Usage Examples

### Basic Usage

```python
import numpy as np
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

# Generate a random walk (example series)
x = np.cumsum(np.random.randn(5000))

# Create MFS analysis object
q_list = np.linspace(-5, 5, 21)
mfs = CFASeriesMFS(x, q_list=q_list)

# Compute multifractal spectrum
df_mfs = mfs.get_mfs()

# View results
print(df_mfs.head(10))

# Visualize
mfs.plot(df_mfs)
```

### Custom Scale Windows

```python
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS, recommended_lag

x = np.cumsum(np.random.randn(10000))

# Use recommended lag
lag = recommended_lag(len(x), order=2, num_scales=40)
mfs = CFASeriesMFS(x)
df_mfs = mfs.get_mfs(lag_list=lag, order=2)
print(df_mfs)
```

### Higher-order DFA

```python
# Use polynomial order 3 for detrending (removes cubic trends)
df_mfs = mfs.get_mfs(order=3)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

**Additional requirement for series analysis**:
```bash
pip install MFDFA
```

## Class Description

### CFASeriesMFS

**Description**: Multifractal spectrum analysis for 1D time series using MFDFA (Multifractal Detrended Fluctuation Analysis).

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array-like | Required | Input time series (1D array) |
| `q_list` | array-like | linspace(-5, 5, 51) | q moment values |
| `with_progress` | bool | True | Whether to show progress bar |

#### Main Methods

##### 1. get_mfs(lag_list=None, order=2)

**Description**: Calculate the multifractal spectrum (MFS) using MFDFA.

**Parameters**:
- `lag_list` (array-like or None): Scale windows (segment lengths). If None, uses `recommended_lag(len(data))`
- `order` (int): Polynomial order for detrending in DFA. Default 2 (removes linear trend).
  - `order=1`: removes mean
  - `order=2`: removes linear trend (default)
  - `order=3`: removes quadratic trend

**Return Value** (DataFrame): Columns:

| Column | Description |
|--------|-------------|
| `q` | Moment order |
| `h(q)` | Generalized Hurst exponent |
| `t(q)` | Mass exponent τ(q) = q·h(q) − 1 |
| `a(q)` | Singularity strength α(q) = d τ(q)/dq |
| `f(a)` | Multifractal spectrum f(α) = q·α − τ(q) |
| `d(q)` | Generalized dimension D(q) = τ(q)/(q−1) |

##### 2. plot(df_mfs)

**Description**: Visualize multifractal spectrum in a 2×3 subplot grid:
1. H(q) vs q — generalized Hurst exponent
2. τ(q) vs q — mass exponent
3. D(q) vs q — generalized dimension
4. α(q) vs q — singularity strength
5. f(α) vs α — multifractal spectrum (the canonical MFS plot)
6. Overview: normalized t(q), d(q), a(q), f(a) vs q

### Utility Function: recommended_lag

```python
from FreeAeonFractal.FASeriesMFS import recommended_lag

lag = recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25)
```

**Description**: Generate a geometrically spaced set of scale windows suited for medium-to-short sequences.

**Parameters**:
- `x_len` (int): Length of the input sequence
- `order` (int): DFA polynomial order (affects minimum scale)
- `num_scales` (int): Number of scale points to generate (recommend 30–40)
- `s_min` (int or None): Minimum scale; default: `max(16, order + 4)`
- `s_max_ratio` (float): Maximum scale as fraction of sequence length (default: 0.25)

**Return Value**: `numpy.ndarray` of integer scale windows

## Theoretical Background

### MFDFA Method

MFDFA (Kantelhardt et al. 2002) extends standard DFA to detect multifractality in non-stationary time series.

#### Steps

1. **Profile**: Compute cumulative sum Y(i) = Σₜ x(t) − mean(x)

2. **Segmentation**: Divide into non-overlapping windows of size s (the `lag`)

3. **Detrending**: Fit a polynomial of order `order` in each window; compute variance F²(s, v)

4. **Fluctuation function**: For each q:
   ```
   Fq(s) = (1/N_s Σᵥ [F²(s,v)]^(q/2))^(1/q)    q ≠ 0
   ```

5. **Scaling**: `Fq(s) ~ s^h(q)` → slope = generalized Hurst exponent h(q)

#### Key Metrics

- **h(q)**: Generalized Hurst exponent; h(2) = standard Hurst exponent
- **τ(q) = q·h(q) − 1**: Mass exponent
- **α(q) = dτ/dq**: Singularity strength (via `np.gradient`)
- **f(α) = q·α − τ(q)**: Multifractal spectrum
- **D(q) = τ(q)/(q−1)**: Generalized dimension

#### Monofractal vs Multifractal

- **Monofractal**: h(q) is constant across q
- **Multifractal**: h(q) varies with q; Δh = h(−5) − h(5) > 0
- The width Δα = α_max − α_min quantifies multifractality strength

## Important Notes

1. **Sequence Length**:
   - Minimum recommended: 1000 points
   - Below 500 points: results become unreliable
   - `recommended_lag` raises `ValueError` if the sequence is too short

2. **q Value Selection**:
   - Avoid `q=0` (handled as limit); the MFDFA library returns values for the provided q array
   - Recommended: `np.linspace(-5, 5, 21)` or 51 points for finer resolution
   - Negative q: sensitive to low-fluctuation (quiet) regions
   - Positive q: sensitive to high-fluctuation (volatile) regions

3. **DFA Order**:
   - `order=2` is standard and sufficient for most cases
   - Higher orders remove lower-frequency polynomial trends but require more data

4. **Result Interpretation**:
   - Δh > 0.1 suggests multifractality
   - D(q) monotonically decreasing with q → multifractal signal
   - f(α) should form a convex arch; the peak is at D₀ (box-counting dimension)

5. **Dependencies**:
   - Requires the `MFDFA` package: `pip install MFDFA`

## References

- Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A*.
- Peng, C.-K., et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*.
- Ihlen, E. A. F. (2012). Introduction to multifractal detrended fluctuation analysis in Matlab. *Frontiers in Physiology*.
