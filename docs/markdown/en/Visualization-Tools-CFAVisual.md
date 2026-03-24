# Visualization Tools - CFAVisual

## Application Scenarios

The `CFAVisual` class provides visualization tools for fractal point sets and images. Main application scenarios include:

- **Fractal Point Set Display**: Visualize 1D, 2D, and 3D fractal point sets
- **Fractal Pattern Verification**: Verify generated fractal patterns
- **Teaching Demonstrations**: Demonstrate basic concepts of fractal geometry
- **Data Visualization**: Display point clouds and image data

## Usage Examples

### 1D Point Set Visualization

```python
import numpy as np
from FreeAeonFractal.CFAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample
import matplotlib.pyplot as plt

# Generate Cantor Set
points_1d = CFASample.get_Cantor_Set(iterations=1000)

# Visualize 1D point set
plt.figure(figsize=(10, 2))
CFAVisual.plot_1d_points(points_1d)
plt.title('Cantor Set (1D)')
plt.show()
```

### 2D Point Set Visualization

```python
# Generate Sierpinski Triangle
points_2d = CFASample.get_Sierpinski_Triangle(iterations=10000)

# Visualize 2D point set
plt.figure(figsize=(8, 8))
CFAVisual.plot_2d_points(points_2d)
plt.title('Sierpinski Triangle (2D)')
plt.axis('equal')
plt.show()
```

### 3D Point Set Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

# Generate Menger Sponge
points_3d = CFASample.get_Menger_Sponge(iterations=50000)

# Visualize 3D point set
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(points_3d, ax)
ax.set_title('Menger Sponge (3D)')
plt.show()
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAVisual

**Description**: Utility class providing static methods for visualizing fractal point sets and images.

All methods are static, called using `CFAVisual.method_name()`.

#### Static Methods

##### 1. plot_1d_points(points, ax=plt)

**Description**: Display 1D point set (e.g., Cantor Set).

**Parameters**:
- `points` (numpy.ndarray): 1D point array, shape (N,)
- `ax` (matplotlib axis): Matplotlib axis object, default is plt

**Display Features**:
- Points plotted on horizontal line at y=0
- y-axis hidden
- Fixed point size (s=1)

##### 2. plot_2d_points(points, ax=plt)

**Description**: Display 2D point set (e.g., Sierpinski Triangle, Barnsley Fern).

**Parameters**:
- `points` (numpy.ndarray): 2D point array, shape (N, 2)
- `ax` (matplotlib axis): Matplotlib axis object

**Display Features**:
- Red point markers
- Point size: s=1
- Scatter plot display

##### 3. plot_3d_points(points, ax=None)

**Description**: Display 3D point set (e.g., Menger Sponge).

**Parameters**:
- `points` (numpy.ndarray): 3D point array, shape (N, 3)
- `ax` (matplotlib 3D axis): 3D axis object. If None, automatically created

##### 4. plot_2d_image(image, cmap='gray', ax=plt)

**Description**: Display 2D image.

**Parameters**:
- `image` (numpy.ndarray): 2D image array
- `cmap` (str): Colormap, default 'gray'
- `ax` (matplotlib axis): Matplotlib axis object

## References

- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
