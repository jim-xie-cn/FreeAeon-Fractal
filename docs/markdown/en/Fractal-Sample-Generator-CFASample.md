# Fractal Sample Generator - CFASample

## Application Scenarios

The `CFASample` class generates classic fractal patterns based on Iterated Function Systems (IFS). Main application scenarios include:

- **Fractal Pattern Generation**: Generate classic fractal geometric patterns
- **Teaching Demonstrations**: Demonstrate basic principles of fractal geometry
- **Test Data**: Provide test data for fractal analysis algorithms
- **Artistic Creation**: Generate beautiful fractal art patterns
- **Algorithm Validation**: Verify correctness of fractal dimension calculations

## Usage Examples

### Cantor Set

```python
import numpy as np
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.FAVisual import CFAVisual
import matplotlib.pyplot as plt

# Generate Cantor Set (fractal dimension ≈ 0.63)
cantor = CFASample.get_Cantor_Set(
    init_point=np.array([0.0]),
    iterations=256
)

# Visualize
plt.figure(figsize=(12, 2))
CFAVisual.plot_1d_points(cantor)
plt.title('Cantor Set (Dimension ≈ 0.63)')
plt.show()
```

### Sierpinski Triangle

```python
# Generate Sierpinski Triangle (fractal dimension ≈ 1.58)
triangle = CFASample.get_Sierpinski_Triangle(
    init_point=np.array([0.0, 0.0]),
    iterations=256
)

# Visualize
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_points(triangle)
plt.title('Sierpinski Triangle (Dimension ≈ 1.58)')
plt.axis('equal')
plt.show()
```

### Barnsley Fern

```python
# Generate Barnsley Fern (fractal dimension ≈ 1.67)
fern = CFASample.get_Barnsley_Fern(
    init_point=np.array([0.0, 0.0]),
    iterations=4096
)

# Visualize
plt.figure(figsize=(6, 10))
CFAVisual.plot_2d_points(fern)
plt.title('Barnsley Fern (Dimension ≈ 1.67)')
plt.axis('equal')
plt.show()
```

### Menger Sponge

```python
# Generate Menger Sponge (fractal dimension ≈ 2.73)
sponge = CFASample.get_Menger_Sponge(
    init_point=np.array([0.0, 0.0, 0.0]),
    iterations=10240
)

# Visualize 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(sponge, ax)
ax.set_title('Menger Sponge (Dimension ≈ 2.73)')
plt.show()
```

### Convert Points to Image

```python
# Generate fractal point set
points = CFASample.get_Sierpinski_Triangle(iterations=256)

# Convert to image
image = CFASample.get_image_from_points(
    points,
    img_size=(512, 512),
    margin=0.05
)

# Visualize
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_image(image, cmap='binary')
plt.title('Sierpinski Triangle as Image')
plt.show()
```

### Custom IFS

```python
# Generate custom fractal using IFS engine
trans_matrix = [...]       # List of affine transformation matrices
trans_probability = [...]  # Probability for each transformation

points = CFASample.generate(
    init_point=np.array([0.0, 0.0]),
    iterations=5000,
    trans_matrix=trans_matrix,
    trans_probability=trans_probability
)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFASample

**Description**: Fractal sample generator using Iterated Function Systems (IFS) to generate classic fractal patterns.

All methods are static, called using `CFASample.method_name()`.

#### Main Generation Methods

| Method | Dimension | Fractal Dim | Transforms | Default Iter |
|--------|-----------|-------------|------------|--------------|
| `get_Cantor_Set()` | 1D | 0.63 | 2 | 256 |
| `get_Sierpinski_Triangle()` | 2D | 1.58 | 3 | 256 |
| `get_Barnsley_Fern()` | 2D | 1.67 | 4 | 4096 |
| `get_Menger_Sponge()` | 3D | 2.727 | 20 | 10240 |
| `get_4D_Points()` | 4D | — | — | 4096 |

#### Methods

##### 1. generate(init_point, iterations, trans_matrix, trans_probability)

**Description**: Core IFS engine. Iterates a point through randomly chosen affine transformations.

**Parameters**:
- `init_point` (numpy.ndarray): Initial point
- `iterations` (int): Number of iterations
- `trans_matrix` (list): List of affine transformation matrices
- `trans_probability` (list): Probability for each transformation (must sum to 1)

**Return**: numpy.ndarray of generated points

##### 2. get_Cantor_Set(init_point, iterations=256)

**Description**: Generate Cantor Set (1D fractal).

**Parameters**:
- `init_point` (numpy.ndarray): Initial point, shape (1,), default [0.0]
- `iterations` (int): Number of iterations, default 256

**Return**: numpy.ndarray, shape (iterations, 1)

**Fractal Dimension**: ≈ 0.6309

##### 3. get_Sierpinski_Triangle(init_point, iterations=256)

**Description**: Generate Sierpinski Triangle (2D fractal).

**Parameters**:
- `init_point` (numpy.ndarray): Initial point, shape (2,)
- `iterations` (int): Number of iterations, default 256

**Return**: numpy.ndarray, shape (iterations, 2)

**Fractal Dimension**: ≈ 1.585

##### 4. get_Barnsley_Fern(init_point, iterations=4096)

**Description**: Generate Barnsley Fern (2D fractal).

**Parameters**:
- `init_point` (numpy.ndarray): Initial point, shape (2,)
- `iterations` (int): Number of iterations, default 4096

**Return**: numpy.ndarray, shape (iterations, 2)

**Fractal Dimension**: ≈ 1.67

##### 5. get_Menger_Sponge(init_point, iterations=10240)

**Description**: Generate Menger Sponge (3D fractal, 20 transforms).

**Parameters**:
- `init_point` (numpy.ndarray): Initial point, shape (3,)
- `iterations` (int): Number of iterations, default 10240

**Return**: numpy.ndarray, shape (iterations, 3)

**Fractal Dimension**: ≈ 2.727

##### 6. get_image_from_points(points, img_size=(512, 512), margin=0.05)

**Description**: Convert 2D point set to binary image.

**Parameters**:
- `points` (numpy.ndarray): 2D point array, shape (N, 2)
- `img_size` (tuple): Output image size (height, width)
- `margin` (float): Margin ratio, default 0.05 (5%)

**Return**: numpy.ndarray, uint8 binary image
- Point locations: 255 (white)
- Background: 0 (black)

## References

- Barnsley, M. F. (1993). Fractals Everywhere (2nd ed.).
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
