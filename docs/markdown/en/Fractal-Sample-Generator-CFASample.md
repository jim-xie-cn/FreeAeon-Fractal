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
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.CFAVisual import CFAVisual
import matplotlib.pyplot as plt

# Generate Cantor Set (fractal dimension ≈ 0.63)
cantor = CFASample.get_Cantor_Set(
    init_point=np.array([0.0]),
    iterations=2000
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
    iterations=10000
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
    iterations=50000
)

# Visualize
plt.figure(figsize=(6, 10))
CFAVisual.plot_2d_points(fern)
plt.title('Barnsley Fern (Dimension ≈ 1.67)')
plt.axis('equal')
plt.show()
```

### Convert Points to Image

```python
# Generate fractal point set
points = CFASample.get_Sierpinski_Triangle(iterations=50000)

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
| `get_Menger_Sponge()` | 3D | 2.73 | 20 | 10240 |

#### Methods

##### 1. get_Cantor_Set(init_point, iterations)

**Description**: Generate Cantor Set (1D fractal).

**Parameters**:
- `init_point` (numpy.ndarray): Initial point, shape (1,), default [0.0]
- `iterations` (int): Number of iterations, default 256

**Return**: numpy.ndarray, shape (iterations, 1)

**Fractal Dimension**: ≈ 0.6309

##### 2. get_image_from_points(points, img_size=(512, 512), margin=0.05)

**Description**: Convert 2D point set to image.

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
